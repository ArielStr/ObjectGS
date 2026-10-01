#!/usr/bin/env python3
"""
Generate ObjectGS-compatible binary object masks with SAM2.

This utility is an optional frontend. ObjectGS itself only requires
binary object masks satisfying the mask contract checked by
scripts/validate_masks.py.

Input:
    images/<stem>.<ext>

Output:
    masks/<stem>.png

Output masks:
    - same spatial resolution as the source image
    - single-channel PNG
    - background = 0
    - foreground = 255

Prompts are supplied through a JSON file:

{
    "IMG_0001": {
        "box": [x0, y0, x1, y1]
    },
    "IMG_0002": {
        "points": [[x1, y1], [x2, y2]],
        "labels": [1, 0]
    }
}
"""

import argparse
import json
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


def load_prompts(path: Path):
    with path.open("r") as f:
        prompts = json.load(f)

    if not isinstance(prompts, dict):
        raise ValueError("Prompt JSON must contain an object keyed by image stem.")

    return prompts


def validate_prompt(stem, prompt):
    if not isinstance(prompt, dict):
        raise ValueError(f"{stem}: prompt must be a JSON object.")

    has_box = "box" in prompt
    has_points = "points" in prompt

    if not has_box and not has_points:
        raise ValueError(
            f"{stem}: prompt must contain 'box' and/or 'points'."
        )

    if has_box:
        box = prompt["box"]

        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(
                f"{stem}: box must be [x0, y0, x1, y1]."
            )

    if has_points:
        points = prompt["points"]
        labels = prompt.get("labels")

        if labels is None:
            raise ValueError(
                f"{stem}: point prompts require 'labels'."
            )

        if len(points) != len(labels):
            raise ValueError(
                f"{stem}: points and labels must have equal length."
            )

        if any(label not in (0, 1) for label in labels):
            raise ValueError(
                f"{stem}: point labels must be 0 or 1."
            )


def build_predictor(model_cfg, checkpoint, device):
    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        raise RuntimeError(
            "SAM2 is not installed. Install the official SAM2 package "
            "before running this utility."
        ) from exc

    sam2_model = build_sam2(
        model_cfg,
        checkpoint,
        device=device,
    )

    predictor = SAM2ImagePredictor(sam2_model)

    return predictor, torch


def predict_mask(predictor, image, prompt):
    predictor.set_image(image)

    point_coords = None
    point_labels = None
    box = None

    if "points" in prompt:
        point_coords = np.asarray(
            prompt["points"],
            dtype=np.float32,
        )

        point_labels = np.asarray(
            prompt["labels"],
            dtype=np.int32,
        )

    if "box" in prompt:
        box = np.asarray(
            prompt["box"],
            dtype=np.float32,
        )

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        multimask_output=True,
    )

    best = int(np.argmax(scores))

    return masks[best].astype(bool), float(scores[best])


def main():
    parser = argparse.ArgumentParser(
        description="Generate ObjectGS masks using SAM2."
    )

    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)

    parser.add_argument(
        "--model-cfg",
        default="configs/sam2.1/sam2.1_hiera_s.yaml",
        help=(
            "SAM2 Hydra model config name. "
            "Default: configs/sam2.1/sam2.1_hiera_s.yaml"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "Path to the SAM2 checkpoint "
            "(e.g. sam2.1_hiera_small.pt)."
        ),
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device. Default: cuda",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.images.is_dir():
        raise SystemExit(
            f"Images directory does not exist: {args.images}"
        )

    if not args.prompts.is_file():
        raise SystemExit(
            f"Prompt file does not exist: {args.prompts}"
        )

    images = collect_images(args.images)
    prompts = load_prompts(args.prompts)

    if not images:
        raise SystemExit("No images found.")

    missing_prompts = sorted(set(images) - set(prompts))
    unknown_prompts = sorted(set(prompts) - set(images))

    if missing_prompts:
        print("Missing prompts:")
        for stem in missing_prompts:
            print(" ", stem)
        raise SystemExit(
            f"Missing prompts for {len(missing_prompts)} image(s)."
        )

    if unknown_prompts:
        print("Warning: prompts without matching image:")
        for stem in unknown_prompts:
            print(" ", stem)

    for stem in images:
        validate_prompt(stem, prompts[stem])

    args.output.mkdir(parents=True, exist_ok=True)

    predictor, torch = build_predictor(
        args.model_cfg,
        args.checkpoint,
        args.device,
    )

    print("===== ObjectGS SAM2 Mask Generation =====")
    print(f"Images : {len(images)}")
    print(f"Output : {args.output}")
    print(f"Device : {args.device}")
    print()

    for index, (stem, image_path) in enumerate(images.items(), 1):

        output_path = args.output / f"{stem}.png"

        if output_path.exists() and not args.overwrite:
            print(
                f"[{index:03d}/{len(images):03d}] "
                f"{stem}: exists, skipping"
            )
            continue

        with Image.open(image_path) as pil:
            image = np.asarray(pil.convert("RGB"))

        mask, score = predict_mask(
            predictor,
            image,
            prompts[stem],
        )

        output = (mask.astype(np.uint8) * 255)

        Image.fromarray(
            output,
            mode="L",
        ).save(output_path)

        foreground = float(mask.mean())

        print(
            f"[{index:03d}/{len(images):03d}] "
            f"{stem}: "
            f"score={score:.4f} "
            f"foreground={foreground:.2%}"
        )

    print()
    print("✓ SAM2 mask generation complete.")
    print()
    print("Validate the generated masks with:")
    print(
        "python scripts/validate_masks.py "
        f"--images {args.images} "
        f"--masks {args.output}"
    )


if __name__ == "__main__":
    main()
