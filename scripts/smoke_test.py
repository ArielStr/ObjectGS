#!/usr/bin/env python3
"""Run a short ObjectGS training smoke test."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run a short ObjectGS training test."
    )
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--copy-scene", type=Path, default=None)
    parser.add_argument("--lambda-outside", type=float, default=1.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    gs = root / "third_party" / "gaussian-splatting"

    if args.iterations < 1:
        parser.error("--iterations must be positive")

    scene = args.scene.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not scene.is_dir():
        parser.error(f"Scene does not exist: {scene}")

    if args.copy_scene:
        local_scene = args.copy_scene.expanduser().resolve()

        if local_scene == scene:
            parser.error("--copy-scene must differ from --scene")

        if local_scene.exists():
            parser.error(f"Copy destination already exists: {local_scene}")

        print(f"Copying scene to {local_scene}", flush=True)
        shutil.copytree(scene, local_scene)
        scene = local_scene

    if not (gs / "train.py").is_file():
        parser.error(
            "Graphdeco not installed. Run setup.sh --mode gpu first."
        )

    images_dir = scene / "images"
    masks_dir = scene / "masks"
    sparse_dir = scene / "sparse" / "0"

    if not sparse_dir.is_dir() and (scene / "sparse").is_dir():
        sparse_dir = scene / "sparse"

    required = [
        sparse_dir / "cameras.bin",
        sparse_dir / "images.bin",
        sparse_dir / "points3D.bin",
    ]

    for path in required:
        if not path.is_file():
            parser.error(f"Missing COLMAP file: {path}")

    images = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ) if images_dir.is_dir() else []

    masks = sorted(masks_dir.glob("*.png")) if masks_dir.is_dir() else []

    if not images or not masks:
        parser.error("Scene must contain images and PNG masks")

    image_names = {p.stem for p in images}
    mask_names = {p.stem for p in masks}

    missing_masks = image_names - mask_names
    if missing_masks:
        parser.error(
            f"Missing masks for {len(missing_masks)} images: "
            + ", ".join(sorted(missing_masks)[:5])
        )

    if output.exists():
        parser.error(f"Output already exists: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.parent / (output.name + ".log")

    cmd = [
        sys.executable,
        "train.py",
        "-s", str(scene),
        "-m", str(output),
        "--iterations", str(args.iterations),
        "--lambda_outside", str(args.lambda_outside),
        "--mask_normalized_densification",
        "--disable_viewer",
        "--test_iterations", str(args.iterations),
        "--save_iterations", str(args.iterations),
    ]

    print("===== ObjectGS Smoke Test =====", flush=True)
    print("Scene:", scene, flush=True)
    print("Images:", len(images), flush=True)
    print("Masks:", len(masks), flush=True)
    print("Iterations:", args.iterations, flush=True)
    print("Output:", output, flush=True)
    print("Log:", log_path, flush=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            cmd,
            cwd=gs,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            errors="replace",
        )

        for line in process.stdout:
            log.write(line)
            log.flush()

            if any(token in line.lower() for token in [
                "[object mask]",
                "[maskloss]",
                "error",
                "traceback",
                "saving",
                "complete",
            ]):
                print(line.rstrip(), flush=True)

        return_code = process.wait()

    print("\n===== Result =====")
    print("Training return code:", return_code)

    if return_code != 0:
        print("FAILED: training process exited with an error")
        print("Last log lines:")
        print("\n".join(log_path.read_text(
            errors="replace"
        ).splitlines()[-40:]))
        return 1

    ply = (
        output / "point_cloud"
        / f"iteration_{args.iterations}"
        / "point_cloud.ply"
    )

    if not ply.is_file():
        print("FAILED: output point_cloud.ply not found")
        return 1

    log_text = log_path.read_text(errors="replace")

    if "[MASKLOSS]" not in log_text:
        print("FAILED: no MASKLOSS log entries found")
        return 1

    print("PASS: training completed")
    print("PASS: point cloud saved")
    print("PASS: ObjectGS mask loss executed")
    print("Smoke test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
