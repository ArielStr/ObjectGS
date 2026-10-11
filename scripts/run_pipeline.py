#!/usr/bin/env python3
"""End-to-end ObjectGS pipeline orchestrator."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def image_files(folder):
    if not folder.is_dir():
        return []
    return [p for p in folder.iterdir() if p.is_file()
            and p.suffix.lower() in IMAGE_EXTS]


def model_ok(folder):
    return folder.is_dir() and all(
        (folder / name).is_file() and (folder / name).stat().st_size > 0
        for name in MODEL_FILES
    )


def masks_ok(folder, images):
    """Validate binary object masks against source images."""
    from PIL import Image
    import numpy as np

    if not folder.is_dir() or not images:
        return False

    expected = {p.stem for p in images}
    actual = {p.stem for p in folder.glob("*.png")}

    if expected != actual:
        return False

    for image_path in images:
        mask_path = folder / f"{image_path.stem}.png"

        try:
            with Image.open(image_path) as image:
                image_size = image.size

            with Image.open(mask_path) as mask:
                if mask.mode != "L":
                    return False

                if mask.size != image_size:
                    return False

                pixels = np.asarray(mask)

                if not np.isin(pixels, [0, 255]).all():
                    return False

                if not np.any(pixels == 255):
                    return False

        except (OSError, ValueError):
            return False

    return True


def colmap_ok(folder):
    return (
        model_ok(folder / "sparse/0")
        and model_ok(folder / "undistorted/sparse")
        and bool(image_files(folder / "undistorted/images"))
    )


def scene_ok(folder):
    imgs = image_files(folder / "images")
    return (
        bool(imgs)
        and masks_ok(folder / "masks", imgs)
        and model_ok(folder / "sparse/0")
    )


def training_ok(folder, iterations):
    ply = folder / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    return ply.is_file() and ply.stat().st_size > 0


def run_stage(name, cmd, log_dir):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"

    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE: {name}", flush=True)
    print("Command:", " ".join(map(str, cmd)), flush=True)
    print("Log:", log_path, flush=True)

    started = time.monotonic()

    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1",
                 "QT_QPA_PLATFORM": "offscreen"},
        )

        for line in proc.stdout:
            log.write(line)
            log.flush()
            # Keep notebook output compact.
            if any(s in line for s in (
                "===== DONE", "Training complete",
                "Saving Gaussians", "Selected Points3D",
                "Registered images", "Points3D",
                "Traceback", "Error", "ERROR",
            )):
                print(line.rstrip(), flush=True)

        code = proc.wait()

    elapsed = round(time.monotonic() - started, 2)
    print(f"Stage {name}: exit={code}, elapsed={elapsed}s", flush=True)

    if code != 0:
        raise RuntimeError(
            f"Stage '{name}' failed. See {log_path}"
        )

    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Run ObjectGS end-to-end."
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--runtime", type=Path,
        default=ROOT / "third_party/gaussian-splatting",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", type=int, default=7000)
    parser.add_argument("--lambda-outside", type=float, default=1.0)
    parser.add_argument("--consistency", type=float, default=1.0)
    parser.add_argument("--camera-model", default="OPENCV")
    parser.add_argument(
        "--matcher", choices=["exhaustive", "sequential"],
        default="exhaustive",
    )
    parser.add_argument(
        "--sift-gpu", choices=["auto", "on", "off"],
        default="auto",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.images = args.images.expanduser().resolve()
    args.prompts = args.prompts.expanduser().resolve()
    args.sam_checkpoint = args.sam_checkpoint.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.runtime = args.runtime.expanduser().resolve()

    if not image_files(args.images):
        parser.error("No input images found.")
    if not args.prompts.is_file():
        parser.error(f"Prompts not found: {args.prompts}")
    if not args.sam_checkpoint.is_file():
        parser.error(f"SAM2 checkpoint not found: {args.sam_checkpoint}")
    if not args.runtime.is_dir():
        parser.error(f"Runtime not found: {args.runtime}")
    if args.iterations <= 0:
        parser.error("--iterations must be positive.")
    if not 0 <= args.consistency <= 1:
        parser.error("--consistency must be between 0 and 1.")

    out = args.output
    masks = out / "masks"
    colmap = out / "colmap"
    object_sfm = out / "object_sfm"
    scene = out / "gs_scene_object_init"
    training = out / f"objectgs_{args.iterations}"
    logs = out / "logs"

    # MVP safety policy: --force is allowed only with a fresh output.
    # This prevents overwriting or deleting existing reconstructions.
    if args.force and out.exists() and any(out.iterdir()):
        parser.error(
            "--force requires a new or empty --output directory. "
            "Existing outputs are never deleted by the pipeline."
        )

    out.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    timings = {}
    rerun_downstream = False

    def stage(name, valid, command, destructive=False):
        nonlocal rerun_downstream

        should_run = args.force or rerun_downstream or not valid()

        if not should_run:
            print(f"[SKIP] {name}: existing output validated", flush=True)
            return

        if destructive and not args.force:
            raise RuntimeError(
                f"{name} requires replacing existing output. "
                "Use a fresh --output directory or --force."
            )

        timings[name] = run_stage(name, command, logs)

        if not valid():
            raise RuntimeError(
                f"{name} returned success but expected outputs are invalid."
            )

        rerun_downstream = True

    stage(
        "sam2",
        lambda: masks_ok(masks, image_files(args.images)),
        [
            python, SCRIPTS / "generate_masks_sam2.py",
            "--images", args.images,
            "--output", masks,
            "--prompts", args.prompts,
            "--checkpoint", args.sam_checkpoint,
            "--device", args.device,
            *(["--overwrite"] if args.force else []),
        ],
        destructive=masks.exists(),
    )

    stage(
        "colmap",
        lambda: colmap_ok(colmap),
        [
            python, SCRIPTS / "run_colmap.py",
            "--images", args.images,
            "--workspace", colmap,
            "--camera-model", args.camera_model,
            "--matcher", args.matcher,
            "--sift-gpu", args.sift_gpu,
            *(["--overwrite"] if args.force else []),
        ],
        destructive=colmap.exists(),
    )

    stage(
        "object_sfm",
        lambda: model_ok(object_sfm),
        [
            python, SCRIPTS / "prepare_object_sfm.py",
            "--original-model", colmap / "sparse/0",
            "--undistorted-model", colmap / "undistorted/sparse",
            "--masks", masks,
            "--output", object_sfm,
            "--consistency", str(args.consistency),
        ],
        destructive=object_sfm.exists(),
    )

    stage(
        "prepare_scene",
        lambda: scene_ok(scene),
        [
            python, SCRIPTS / "prepare_scene.py",
            "--original-model", colmap / "sparse/0",
            "--undistorted-images", colmap / "undistorted/images",
            "--object-model", object_sfm,
            "--masks", masks,
            "--output", scene,
            *(["--overwrite"] if args.force else []),
        ],
        destructive=scene.exists(),
    )

    stage(
        "training",
        lambda: training_ok(training, args.iterations),
        [
            python, SCRIPTS / "train_objectgs.py",
            "--runtime", args.runtime,
            "--scene", scene,
            "--output", training,
            "--iterations", str(args.iterations),
            "--lambda-outside", str(args.lambda_outside),
        ],
        destructive=training.exists(),
    )

    summary = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "images": str(args.images),
        "output": str(out),
        "iterations": args.iterations,
        "timings_seconds": timings,
        "final_ply": str(
            training / "point_cloud" /
            f"iteration_{args.iterations}" / "point_cloud.ply"
        ),
    }

    (out / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n===== OBJECTGS PIPELINE COMPLETE =====")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
