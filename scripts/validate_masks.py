#!/usr/bin/env python3

"""
Validate an image/mask dataset for ObjectGS.

Expected layout / contract:
    images/<stem>.<image_ext>
    masks/<stem>.png

Masks must:
    - have a matching source image
    - have the same spatial resolution as the image
    - be single-channel
    - be binary with values {0, 255}
    - contain at least one foreground pixel

The script does not require every input image to be registered by COLMAP.
It validates the segmentation input before downstream reconstruction.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def collect_files(directory: Path):
    return {
        p.stem: p
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix in IMAGE_EXTENSIONS
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate image/mask pairs for ObjectGS."
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory containing source images.",
    )
    parser.add_argument(
        "--masks",
        type=Path,
        required=True,
        help="Directory containing binary PNG object masks.",
    )
    args = parser.parse_args()

    images_dir = args.images
    masks_dir = args.masks

    if not images_dir.is_dir():
        raise SystemExit(f"Images directory does not exist: {images_dir}")

    if not masks_dir.is_dir():
        raise SystemExit(f"Masks directory does not exist: {masks_dir}")

    images = collect_files(images_dir)
    masks = collect_files(masks_dir)

    image_stems = set(images)
    mask_stems = set(masks)

    missing_masks = sorted(image_stems - mask_stems)
    extra_masks = sorted(mask_stems - image_stems)
    matched = sorted(image_stems & mask_stems)

    resolution_errors = []
    channel_errors = []
    value_errors = []
    empty_masks = []
    foreground_stats = []

    for stem in matched:
        image_path = images[stem]
        mask_path = masks[stem]

        with Image.open(image_path) as image:
            image_size = image.size  # (W, H)

        with Image.open(mask_path) as mask_image:
            mask_mode = mask_image.mode
            mask_size = mask_image.size
            mask_array = np.asarray(mask_image)

        # Masks are required to be stored as single-channel images.
        if mask_array.ndim != 2:
            channel_errors.append(
                (stem, mask_mode, tuple(mask_array.shape))
            )
            continue

        if mask_size != image_size:
            resolution_errors.append(
                (stem, image_size, mask_size)
            )

        unique_values = np.unique(mask_array)

        if not np.all(np.isin(unique_values, [0, 255])):
            value_errors.append(
                (stem, unique_values.tolist())
            )

        foreground_fraction = float(np.mean(mask_array == 255))
        foreground_stats.append((stem, foreground_fraction))

        if foreground_fraction == 0.0:
            empty_masks.append(stem)

    print("===== ObjectGS Mask Validation =====\n")

    print(f"Images               : {len(images)}")
    print(f"Masks                : {len(masks)}")
    print(f"Matched pairs        : {len(matched)}")
    print(f"Missing masks        : {len(missing_masks)}")
    print(f"Extra masks          : {len(extra_masks)}")
    print(f"Resolution errors    : {len(resolution_errors)}")
    print(f"Channel errors       : {len(channel_errors)}")
    print(f"Invalid binary masks : {len(value_errors)}")
    print(f"Empty masks          : {len(empty_masks)}")

    if foreground_stats:
        fractions = np.array(
            [fraction for _, fraction in foreground_stats],
            dtype=float,
        )

        print("\nForeground fraction:")
        print(f"  min    : {fractions.min():.4%}")
        print(f"  mean   : {fractions.mean():.4%}")
        print(f"  median : {np.median(fractions):.4%}")
        print(f"  max    : {fractions.max():.4%}")

    if missing_masks:
        print("\nMissing masks:")
        for stem in missing_masks:
            print(f"  {stem}")

    if extra_masks:
        print("\nMasks without source image:")
        for stem in extra_masks:
            print(f"  {stem}")

    if resolution_errors:
        print("\nResolution mismatches:")
        for stem, image_size, mask_size in resolution_errors:
            print(
                f"  {stem}: "
                f"image={image_size[0]}x{image_size[1]}, "
                f"mask={mask_size[0]}x{mask_size[1]}"
            )

    if channel_errors:
        print("\nNon-single-channel masks:")
        for stem, mode, shape in channel_errors:
            print(f"  {stem}: mode={mode}, shape={shape}")

    if value_errors:
        print("\nNon-binary masks:")
        for stem, values in value_errors:
            preview = values[:20]
            suffix = " ..." if len(values) > 20 else ""
            print(f"  {stem}: values={preview}{suffix}")

    if empty_masks:
        print("\nEmpty masks:")
        for stem in empty_masks:
            print(f"  {stem}")

    valid = not any([
        missing_masks,
        extra_masks,
        resolution_errors,
        channel_errors,
        value_errors,
        empty_masks,
    ])

    print("\n====================================")

    if valid:
        print("✓ Scene masks are valid for ObjectGS.")
        return 0

    print("✗ Scene masks are NOT valid for ObjectGS.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
