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



def frames_ok(folder, video, fps, quality=2):
    """Validate extracted frames and their source video settings."""
    manifest_path = folder / "frames_manifest.json"

    if not manifest_path.is_file():
        return False

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        stat = video.stat()

        if data.get("video") != str(video):
            return False
        if data.get("video_size") != stat.st_size:
            return False
        if data.get("video_mtime_ns") != stat.st_mtime_ns:
            return False
        if data.get("fps") != fps:
            return False
        if data.get("quality") != quality:
            return False

        names = data.get("frames")
        if not isinstance(names, list) or not names:
            return False
        if len(names) != len(set(names)):
            return False
        if any(
            not isinstance(name, str)
            or not name.startswith("frame_")
            or not name.endswith(".jpg")
            for name in names
        ):
            return False

        actual = {p.name for p in image_files(folder)}
        if actual != set(names):
            return False

        return all(
            (folder / name).stat().st_size > 0
            for name in names
        )

    except (OSError, ValueError, TypeError, KeyError):
        return False


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


def registered_image_count(model_dir):
    """Read registered camera count from COLMAP model_analyzer."""
    if not model_ok(model_dir):
        return 0

    try:
        result = subprocess.run(
            [
                "colmap", "model_analyzer",
                "--path", str(model_dir),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        for line in (result.stdout + result.stderr).splitlines():
            if "Registered images:" in line:
                return int(line.split("Registered images:", 1)[1].strip())

    except (OSError, subprocess.CalledProcessError, ValueError):
        return 0

    return 0



def selected_colmap_model(folder):
    """Resolve the reconstruction selected by run_colmap.py."""
    marker = folder / "selected_model.txt"

    if not marker.is_file():
        return None

    relative = Path(marker.read_text(encoding="utf-8").strip())

    if relative.is_absolute() or ".." in relative.parts:
        return None

    model = folder / relative

    if not model.is_dir() or not model_ok(model):
        return None

    return model

def colmap_ok(folder, total_images, min_ratio=0.70, min_images=10):
    """Validate reconstruction files and registered camera coverage."""
    original = selected_colmap_model(folder)
    if original is None:
        return False
    undistorted = folder / "undistorted/sparse"

    if not (
        model_ok(original)
        and model_ok(undistorted)
        and bool(image_files(folder / "undistorted/images"))
    ):
        return False

    registered = registered_image_count(original)
    ratio = registered / total_images if total_images else 0.0

    print(
        f"[COLMAP QUALITY] Registered: {registered}/{total_images} "
        f"({ratio:.1%}), required: >= {min_images} and >= {min_ratio:.0%}",
        flush=True,
    )

    return registered >= min_images and ratio >= min_ratio


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
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--images", type=Path)
    inputs.add_argument("--video", type=Path)
    parser.add_argument("--point", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--prompts", type=Path)
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

    if args.video:
        args.video = args.video.expanduser().resolve()
        if not args.video.is_file():
            parser.error(f"Video not found: {args.video}")
        if args.point is None:
            parser.error("--point X Y is required with --video")
        if args.prompts is not None:
            parser.error("--prompts is only supported with --images")
        if args.fps <= 0:
            parser.error("--fps must be positive")
    else:
        args.images = args.images.expanduser().resolve()
        if args.prompts is None:
            parser.error("--prompts is required with --images")

    if args.prompts is not None:
        args.prompts = args.prompts.expanduser().resolve()
    args.sam_checkpoint = args.sam_checkpoint.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.runtime = args.runtime.expanduser().resolve()

    if args.images is not None and not image_files(args.images):
        parser.error("No input images found.")
    if args.prompts is not None and not args.prompts.is_file():
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
    if args.video:
        args.images = out / "images"

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

    if args.video:
        stage(
            "extract_frames",
            lambda: frames_ok(args.images, args.video, args.fps),
            [
                python, SCRIPTS / "extract_video_frames.py",
                "--video", args.video,
                "--output", args.images,
                "--fps", str(args.fps),
            ],
            destructive=args.images.exists(),
        )

    stage(
        "sam2",
        lambda: masks_ok(masks, image_files(args.images)),
        [
            python,
            SCRIPTS / (
                "generate_masks_sam2_video.py"
                if args.video else "generate_masks_sam2.py"
            ),
            "--images", args.images,
            "--output", masks,
            *(
                ["--point", *map(str, args.point)]
                if args.video else ["--prompts", args.prompts]
            ),
            "--checkpoint", args.sam_checkpoint,
            "--device", args.device,
            *(
                [] if args.video else
                (["--overwrite"] if args.force else [])
            ),
        ],
        destructive=(
            masks.exists() and any(masks.iterdir())
        ),
    )

    # Reject an existing low-quality reconstruction without rerunning
    # COLMAP or modifying any reconstruction files.
    existing_model = selected_colmap_model(colmap)
    if existing_model is not None:
        total_images = len(image_files(args.images))
        registered = registered_image_count(existing_model)
        ratio = registered / total_images if total_images else 0.0

        if registered < 10 or ratio < 0.70:
            raise RuntimeError(
                f"COLMAP reconstruction quality too low: "
                f"{registered}/{total_images} registered ({ratio:.1%}). "
                "Required: at least 10 images and 70%. "
                "Use a fresh --output directory to retry COLMAP."
            )

    stage(
        "colmap",
        lambda: colmap_ok(colmap, len(image_files(args.images))),
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

    original_model = selected_colmap_model(colmap)
    if original_model is None:
        raise RuntimeError(
            "Selected COLMAP reconstruction is missing or invalid."
        )

    stage(
        "object_sfm",
        lambda: model_ok(object_sfm),
        [
            python, SCRIPTS / "prepare_object_sfm.py",
            "--original-model", original_model,
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
            "--original-model", original_model,
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
