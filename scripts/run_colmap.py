#!/usr/bin/env python3

"""
Run the COLMAP reconstruction pipeline required by ObjectGS.

Pipeline:
    images/
      ↓
    feature extraction
      ↓
    exhaustive matching
      ↓
    sparse mapping
      ↓
    original sparse reconstruction
      ↓
    image undistortion
      ↓
    undistorted reconstruction

ObjectGS keeps both reconstructions:

1. Original reconstruction
   Used together with original-resolution object masks to compute
   multi-view mask consistency.

2. Undistorted reconstruction
   Used as the base reconstruction for Gaussian Splatting.
"""

import argparse
import shutil
import subprocess
from pathlib import Path


def run(cmd):
    print("\n" + "=" * 80)
    print(" ".join(str(x) for x in cmd))
    print("=" * 80)

    subprocess.run(
        [str(x) for x in cmd],
        check=True,
    )


def count_images(directory):
    extensions = {".jpg", ".jpeg", ".png"}
    return len([
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Run COLMAP reconstruction for ObjectGS."
    )

    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory containing input images.",
    )

    parser.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="Directory in which COLMAP outputs will be written.",
    )

    parser.add_argument(
        "--camera-model",
        default="OPENCV",
        help="COLMAP camera model used during feature extraction. Default: OPENCV",
    )

    parser.add_argument(
        "--single-camera",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Assume all images share one camera. Default: true.",
    )

    parser.add_argument(
        "--matcher",
        choices=["exhaustive", "sequential"],
        default="exhaustive",
        help="Feature matcher. Default: exhaustive.",
    )

    parser.add_argument(
        "--sift-gpu",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "Control GPU use for SIFT extraction and matching. "
            "'auto' preserves COLMAP defaults, 'on' forces GPU, "
            "and 'off' forces CPU. Default: auto."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing workspace before running.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the pipeline without executing COLMAP.",
    )

    args = parser.parse_args()

    images = args.images.resolve()
    workspace = args.workspace.resolve()

    if not images.is_dir():
        raise SystemExit(
            f"Images directory does not exist: {images}"
        )

    n_images = count_images(images)

    if n_images == 0:
        raise SystemExit(
            f"No supported images found in: {images}"
        )

    if shutil.which("colmap") is None and not args.dry_run:
        raise SystemExit(
            "COLMAP executable was not found in PATH."
        )

    if workspace.exists():
        if args.overwrite:
            print(f"Removing existing workspace: {workspace}")

            if not args.dry_run:
                shutil.rmtree(workspace)
        else:
            raise SystemExit(
                f"Workspace already exists: {workspace}\n"
                "Use --overwrite to replace it."
            )

    database = workspace / "database.db"
    sparse = workspace / "sparse"
    undistorted = workspace / "undistorted"

    if not args.dry_run:
        workspace.mkdir(parents=True, exist_ok=True)
        sparse.mkdir(parents=True, exist_ok=True)

    single_camera = "1" if args.single_camera else "0"

    sift_extraction_args = []
    sift_matching_args = []

    if args.sift_gpu == "on":
        sift_extraction_args = ["--SiftExtraction.use_gpu", "1"]
        sift_matching_args = ["--SiftMatching.use_gpu", "1"]
    elif args.sift_gpu == "off":
        sift_extraction_args = ["--SiftExtraction.use_gpu", "0"]
        sift_matching_args = ["--SiftMatching.use_gpu", "0"]

    commands = []

    # -----------------------------------------------------
    # 1. Feature extraction
    # -----------------------------------------------------

    commands.append([
        "colmap",
        "feature_extractor",
        "--database_path", database,
        "--image_path", images,
        "--ImageReader.camera_model", args.camera_model,
        "--ImageReader.single_camera", single_camera,
        *sift_extraction_args,
    ])

    # -----------------------------------------------------
    # 2. Feature matching
    # -----------------------------------------------------

    if args.matcher == "exhaustive":
        commands.append([
            "colmap",
            "exhaustive_matcher",
            "--database_path", database,
            *sift_matching_args,
        ])
    else:
        commands.append([
            "colmap",
            "sequential_matcher",
            "--database_path", database,
            *sift_matching_args,
        ])

    # -----------------------------------------------------
    # 3. Sparse mapping
    # -----------------------------------------------------

    commands.append([
        "colmap",
        "mapper",
        "--database_path", database,
        "--image_path", images,
        "--output_path", sparse,
    ])

    print("===== ObjectGS COLMAP Pipeline =====")
    print(f"Images        : {images}")
    print(f"Image count   : {n_images}")
    print(f"Workspace     : {workspace}")
    print(f"Camera model  : {args.camera_model}")
    print(f"Single camera : {args.single_camera}")
    print(f"Matcher       : {args.matcher}")
    print(f"SIFT GPU      : {args.sift_gpu}")
    print(f"Dry run       : {args.dry_run}")

    for cmd in commands:
        if args.dry_run:
            print("\n$", " ".join(str(x) for x in cmd))
        else:
            run(cmd)

    if args.dry_run:
        print(
            "\nNOTE: image_undistorter is not shown yet because its "
            "input reconstruction is selected only after mapping."
        )
        print("\n✓ Dry run complete.")
        return

    # -----------------------------------------------------
    # Select the largest sparse reconstruction
    # -----------------------------------------------------

    models = sorted([
        p for p in sparse.iterdir()
        if p.is_dir()
    ])

    if not models:
        raise RuntimeError(
            "COLMAP mapper produced no sparse reconstruction."
        )

    def registered_image_count(model_dir: Path) -> int:
        """
        Determine the number of registered images using
        COLMAP's model_analyzer output.

        This keeps pycolmap optional for this wrapper.
        """
        result = subprocess.run(
            [
                "colmap",
                "model_analyzer",
                "--path",
                str(model_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        output = result.stdout + result.stderr

        for line in output.splitlines():
            marker = "Registered images:"
            if marker in line:
                value = line.split(marker, 1)[1].strip()
                return int(value)

        raise RuntimeError(
            f"Could not determine registered image count for {model_dir}"
        )

    model_stats = []

    for candidate in models:
        count = registered_image_count(candidate)
        model_stats.append((count, candidate))

    # Prefer the reconstruction containing the largest number
    # of registered input images. The path is only a deterministic
    # tie-breaker when two components have the same size.
    model_stats.sort(
        key=lambda item: (-item[0], str(item[1]))
    )

    print("\nSparse reconstructions:")

    for count, candidate in model_stats:
        print(
            f"  {candidate.name}: "
            f"{count}/{n_images} registered images"
        )

    registered_count, model = model_stats[0]

    print(
        f"\nSelected sparse model: {model} "
        f"({registered_count}/{n_images} registered images)"
    )

    if registered_count < n_images:
        print(
            f"Warning: {n_images - registered_count} input image(s) "
            "were not registered by COLMAP."
        )

    # -----------------------------------------------------
    # 4. Undistortion
    # -----------------------------------------------------

    run([
        "colmap",
        "image_undistorter",
        "--image_path", images,
        "--input_path", model,
        "--output_path", undistorted,
        "--output_type", "COLMAP",
    ])

    undistorted_model = undistorted / "sparse"

    print("\n===== DONE =====")
    print(f"Original sparse model    : {model}")
    print(f"Undistorted sparse model : {undistorted_model}")
    print()
    print("Next ObjectGS stage:")
    print()
    print("python scripts/prepare_object_sfm.py \\")
    print(f"    --original-model {model} \\")
    print(f"    --undistorted-model {undistorted_model} \\")
    print("    --masks <scene>/masks \\")
    print("    --output <scene>/object_sfm")


if __name__ == "__main__":
    main()
