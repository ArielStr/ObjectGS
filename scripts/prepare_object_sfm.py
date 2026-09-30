#!/usr/bin/env python3

"""
Prepare an object-only COLMAP reconstruction for ObjectGS.

The object is defined by per-image binary masks.

Pipeline
--------
1. Load the original (distorted) COLMAP reconstruction.
2. For every Point3D, inspect all of its 2D observations.
3. Keep a Point3D only when every observation lies inside the
   corresponding object mask (mask consistency == 1.0).
4. Load the undistorted COLMAP reconstruction.
5. Keep the selected Point3D IDs and remove all other points.
6. Write a consistent object-only COLMAP reconstruction.

Important
---------
Mask consistency is evaluated in the ORIGINAL image coordinate system.
Therefore the masks must correspond to the original images used by
the original COLMAP reconstruction.

Pixel lookup follows the validated ObjectGS convention:

    px = round(x)
    py = round(y)
    foreground = mask[py, px] > 0

The Point3D IDs are expected to be preserved by COLMAP undistortion.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import pycolmap


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare an object-only SfM model for ObjectGS."
    )

    parser.add_argument(
        "--original-model",
        type=Path,
        required=True,
        help="Original COLMAP sparse model (before undistortion).",
    )

    parser.add_argument(
        "--undistorted-model",
        type=Path,
        required=True,
        help="Undistorted COLMAP sparse model.",
    )

    parser.add_argument(
        "--masks",
        type=Path,
        required=True,
        help="Directory containing object masks for the original images.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for the filtered COLMAP model.",
    )

    parser.add_argument(
        "--consistency",
        type=float,
        default=1.0,
        help=(
            "Minimum fraction of a Point3D's observations that must lie "
            "inside the object masks. Default: 1.0."
        ),
    )

    return parser.parse_args()


def load_masks(reconstruction, mask_dir):
    """Load one grayscale mask for every registered image."""

    masks = {}

    for image_id in reconstruction.reg_image_ids():

        image = reconstruction.image(image_id)

        mask_path = mask_dir / (Path(image.name).stem + ".png")

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Missing mask for image '{image.name}': {mask_path}"
            )

        mask = np.asarray(
            Image.open(mask_path).convert("L")
        )

        # The masks must live in the same coordinate system in which
        # the original COLMAP observations were measured.
        camera = reconstruction.camera(image.camera_id)

        if mask.shape != (camera.height, camera.width):
            raise ValueError(
                f"Mask resolution mismatch for '{image.name}'. "
                f"Mask={mask.shape[::-1]}, "
                f"COLMAP={(camera.width, camera.height)}."
            )

        masks[int(image_id)] = mask

    return masks


def select_object_points(reconstruction, masks, min_consistency=1.0):
    """
    Select Point3D IDs using multi-view mask consistency.

    For each Point3D:

        consistency =
            number of observations inside object masks
            -------------------------------------------
                    total number of observations

    A 2D observation is inside when the nearest integer pixel
    (Python round convention) has a non-zero mask value.
    """

    selected = set()

    total_observations = 0
    inside_observations = 0

    for point_id in reconstruction.point3D_ids():

        point_id = int(point_id)
        point = reconstruction.point3D(point_id)

        inside = 0
        total = 0

        for element in point.track.elements:

            image_id = int(element.image_id)
            image = reconstruction.image(image_id)

            point2D = image.points2D[element.point2D_idx]

            x, y = map(float, point2D.xy)

            px = int(round(x))
            py = int(round(y))

            mask = masks[image_id]
            height, width = mask.shape

            total += 1

            if (
                0 <= px < width
                and 0 <= py < height
                and mask[py, px] > 0
            ):
                inside += 1

        total_observations += total
        inside_observations += inside

        if total == 0:
            continue

        consistency = inside / total

        if consistency >= min_consistency:
            selected.add(point_id)

    return selected, total_observations, inside_observations


def filter_reconstruction(reconstruction, keep_ids):
    """Remove all Point3D objects whose IDs are not in keep_ids."""

    model_ids = set(
        int(point_id)
        for point_id in reconstruction.point3D_ids()
    )

    missing = keep_ids - model_ids

    if missing:
        examples = sorted(missing)[:10]

        raise RuntimeError(
            f"{len(missing)} selected Point3D IDs are missing from the "
            f"undistorted reconstruction. Examples: {examples}"
        )

    remove_ids = model_ids - keep_ids

    for point_id in remove_ids:
        reconstruction.delete_point3D(point_id)

    return len(remove_ids)


def main():
    args = parse_args()

    if not 0.0 <= args.consistency <= 1.0:
        raise ValueError("--consistency must be between 0 and 1.")

    print("===== ObjectGS: Object-SfM preparation =====")

    # ---------------------------------------------------------
    # Part A — multi-view mask consistency
    # ---------------------------------------------------------

    print("\n[1/4] Loading original COLMAP reconstruction...")

    original = pycolmap.Reconstruction(
        str(args.original_model)
    )

    print(f"Registered images : {original.num_reg_images():,}")
    print(f"Points3D          : {original.num_points3D():,}")
    print(
        f"Observations      : "
        f"{original.compute_num_observations():,}"
    )

    print("\n[2/4] Loading object masks...")

    masks = load_masks(
        original,
        args.masks,
    )

    print(f"Masks             : {len(masks):,}")

    print("\nComputing multi-view mask consistency...")

    selected_ids, total_obs, inside_obs = select_object_points(
        original,
        masks,
        min_consistency=args.consistency,
    )

    print(f"Observation checks: {total_obs:,}")
    print(f"Inside masks      : {inside_obs:,}")

    print(
        f"Selected Points3D : {len(selected_ids):,} / "
        f"{original.num_points3D():,} "
        f"({100.0 * len(selected_ids) / original.num_points3D():.1f}%)"
    )

    # ---------------------------------------------------------
    # Part B — filter undistorted reconstruction
    # ---------------------------------------------------------

    print("\n[3/4] Loading undistorted reconstruction...")

    undistorted = pycolmap.Reconstruction(
        str(args.undistorted_model)
    )

    print(f"Points before     : {undistorted.num_points3D():,}")
    print(
        f"Observations before: "
        f"{undistorted.compute_num_observations():,}"
    )

    removed = filter_reconstruction(
        undistorted,
        selected_ids,
    )

    print(f"Removed Points3D  : {removed:,}")
    print(f"Points after      : {undistorted.num_points3D():,}")
    print(
        f"Observations after: "
        f"{undistorted.compute_num_observations():,}"
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    print("\n[4/4] Writing object-only reconstruction...")

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    undistorted.write(
        str(args.output)
    )

    text_dir = args.output / "sparse_txt"
    text_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    undistorted.write_text(
        str(text_dir)
    )

    print(f"\nOutput: {args.output}")

    print("\n===== DONE =====")
    print(f"Registered images : {undistorted.num_reg_images():,}")
    print(f"Object Points3D   : {undistorted.num_points3D():,}")
    print(
        f"Observations      : "
        f"{undistorted.compute_num_observations():,}"
    )


if __name__ == "__main__":
    main()
