#!/usr/bin/env python3

"""
Prepare a complete ObjectGS training scene.

The script combines:

    1. COLMAP-undistorted RGB images
    2. Object-only filtered sparse reconstruction
    3. Object masks transformed into the same undistorted
       coordinate system

Expected output:

    scene/
        images/
        masks/
        sparse/
            0/

Mask transformation follows the ObjectGS reference pipeline:

    original binary mask
        -> COLMAP image_undistorter
        -> grayscale > 127
        -> binary {0, 255}

The resulting masks therefore share the exact geometry and
resolution of the undistorted training images.
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def collect_images(directory: Path):
    return {
        p.stem: p
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }


def copy_sparse_model(source: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)

    files = [
        p for p in source.iterdir()
        if p.is_file()
    ]

    if not files:
        raise RuntimeError(
            f"No COLMAP model files found in: {source}"
        )

    for src in files:
        shutil.copy2(src, destination / src.name)


def run(cmd):
    print("\n$", " ".join(str(x) for x in cmd))

    subprocess.run(
        [str(x) for x in cmd],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Prepare an ObjectGS training scene."
    )

    parser.add_argument(
        "--original-model",
        type=Path,
        required=True,
        help=(
            "Original distorted COLMAP sparse model. "
            "Used to geometrically undistort the masks."
        ),
    )

    parser.add_argument(
        "--undistorted-images",
        type=Path,
        required=True,
        help=(
            "COLMAP-undistorted RGB image directory "
            "(typically <workspace>/undistorted/images)."
        ),
    )

    parser.add_argument(
        "--object-model",
        type=Path,
        required=True,
        help=(
            "Filtered object-only COLMAP reconstruction "
            "produced by prepare_object_sfm.py."
        ),
    )

    parser.add_argument(
        "--masks",
        type=Path,
        required=True,
        help="Original-resolution binary object masks.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output ObjectGS training scene.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output scene.",
    )

    args = parser.parse_args()

    original_model = args.original_model.resolve()
    undistorted_images = args.undistorted_images.resolve()
    object_model = args.object_model.resolve()
    masks = args.masks.resolve()
    output = args.output.resolve()

    for path, name in [
        (original_model, "Original COLMAP model"),
        (undistorted_images, "Undistorted images"),
        (object_model, "Object COLMAP model"),
        (masks, "Masks"),
    ]:
        if not path.is_dir():
            raise SystemExit(
                f"{name} directory does not exist: {path}"
            )

    if shutil.which("colmap") is None:
        raise SystemExit(
            "COLMAP executable was not found in PATH."
        )

    rgb = collect_images(undistorted_images)
    mask_files = collect_images(masks)

    if not rgb:
        raise SystemExit("No undistorted RGB images found.")

    if not mask_files:
        raise SystemExit("No masks found.")

    # Only registered images appear in the undistorted scene.
    required_stems = set(rgb)
    available_masks = set(mask_files)

    missing_masks = sorted(
        required_stems - available_masks
    )

    if missing_masks:
        print("Missing masks for registered images:")
        for stem in missing_masks:
            print(" ", stem)

        raise SystemExit(
            f"Missing {len(missing_masks)} required mask(s)."
        )

    if output.exists():
        if args.overwrite:
            shutil.rmtree(output)
        else:
            raise SystemExit(
                f"Output already exists: {output}\n"
                "Use --overwrite to replace it."
            )

    output.mkdir(parents=True)

    out_images = output / "images"
    out_masks = output / "masks"
    out_sparse = output / "sparse" / "0"

    out_images.mkdir()
    out_masks.mkdir()
    out_sparse.mkdir(parents=True)

    print("===== ObjectGS Scene Preparation =====")
    print(f"Registered images : {len(rgb)}")
    print(f"Available masks   : {len(mask_files)}")
    print(f"Output            : {output}")

    # -----------------------------------------------------
    # 1. Copy undistorted RGB images
    # -----------------------------------------------------

    print("\n[1/3] Copying undistorted RGB images...")

    for stem, src in rgb.items():
        shutil.copy2(
            src,
            out_images / src.name,
        )

    # -----------------------------------------------------
    # 2. Copy object-only sparse reconstruction
    # -----------------------------------------------------

    print("\n[2/3] Copying object-only sparse reconstruction...")

    copy_sparse_model(
        object_model,
        out_sparse,
    )

    # -----------------------------------------------------
    # 3. Undistort object masks using COLMAP
    # -----------------------------------------------------

    print("\n[3/3] Undistorting object masks...")

    with tempfile.TemporaryDirectory(
        prefix="objectgs_masks_"
    ) as temp_dir:

        temp = Path(temp_dir)

        mask_input = temp / "mask_input"
        mask_undistorted = temp / "undistorted"

        mask_input.mkdir()

        # COLMAP resolves images by the names stored in the
        # reconstruction. Give each mask the exact filename
        # of its corresponding registered RGB image.
        for stem, rgb_path in rgb.items():

            mask_path = mask_files[stem]

            with Image.open(mask_path) as im:
                mask = np.asarray(
                    im.convert("L")
                )

            binary = (
                (mask > 127).astype(np.uint8) * 255
            )

            rgb_mask = np.repeat(
                binary[..., None],
                3,
                axis=2,
            )

            destination = (
                mask_input / rgb_path.name
            )

            # JPEG is intentionally used when the registered
            # COLMAP image uses JPEG. The final threshold after
            # geometric remapping restores a binary mask.
            Image.fromarray(rgb_mask).save(
                destination,
                quality=100,
                subsampling=0,
            )

        run([
            "colmap",
            "image_undistorter",
            "--image_path",
            mask_input,
            "--input_path",
            original_model,
            "--output_path",
            mask_undistorted,
            "--output_type",
            "COLMAP",
        ])

        transformed = collect_images(
            mask_undistorted / "images"
        )

        if set(transformed) != required_stems:
            raise RuntimeError(
                "Undistorted mask set does not match "
                "the registered RGB image set."
            )

        for stem, path in transformed.items():

            with Image.open(path) as im:
                remapped = np.asarray(
                    im.convert("L")
                )

            binary = (
                (remapped > 127).astype(np.uint8)
                * 255
            )

            rgb_path = rgb[stem]

            with Image.open(rgb_path) as im:
                expected_size = im.size

            if (
                binary.shape[1],
                binary.shape[0],
            ) != expected_size:
                raise RuntimeError(
                    f"{stem}: mask resolution "
                    f"{binary.shape[1]}x{binary.shape[0]} "
                    f"does not match RGB "
                    f"{expected_size[0]}x{expected_size[1]}."
                )

            Image.fromarray(
                binary,
                mode="L",
            ).save(
                out_masks / f"{stem}.png"
            )

    print("\n===== DONE =====")
    print(f"Images : {len(list(out_images.iterdir()))}")
    print(f"Masks  : {len(list(out_masks.iterdir()))}")
    print(f"Scene  : {output}")
    print()
    print("Ready for:")
    print("  scripts/train_objectgs.py")


if __name__ == "__main__":
    main()
