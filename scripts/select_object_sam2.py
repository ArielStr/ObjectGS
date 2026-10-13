#!/usr/bin/env python3
"""Generate numbered SAM2 object-mask candidates for manual selection."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def generate_candidates(image, checkpoint, config, device, points_per_side):
    model = build_sam2(
        config,
        str(checkpoint),
        device=device,
        apply_postprocessing=False,
    )

    generator = SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=points_per_side,
        points_per_batch=16,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.95,
        crop_n_layers=0,
        output_mode="binary_mask",
    )

    with torch.inference_mode():
        return generator.generate(image)


def create_contact_sheet(image, candidates, output, columns=4):
    height, width = image.shape[:2]

    preview_width = 320
    preview_height = max(1, round(height * preview_width / width))

    margin = 12
    label_height = 30
    rows = (len(candidates) + columns - 1) // columns

    sheet = Image.new(
        "RGB",
        (
            columns * (preview_width + margin) + margin,
            rows * (preview_height + label_height + margin) + margin,
        ),
        "white",
    )

    draw = ImageDraw.Draw(sheet)

    for index, candidate in enumerate(candidates):
        mask = candidate["segmentation"].astype(bool)

        overlay = image.copy()
        overlay[mask] = (
            0.45 * overlay[mask] +
            0.55 * np.array([0, 220, 80])
        ).astype(np.uint8)

        tile = Image.fromarray(overlay).resize(
            (preview_width, preview_height)
        )

        row, col = divmod(index, columns)
        x = margin + col * (preview_width + margin)
        y = margin + row * (preview_height + label_height + margin)

        sheet.paste(tile, (x, y))
        draw.text(
            (x + 5, y + preview_height + 5),
            f"Mask {index + 1} | area {candidate['area'] / (width * height):.1%}",
            fill="black",
        )

    sheet.save(output)



def refine_selected_mask(image, original_mask, checkpoint, config, device):
    """Refine one selected mask using an automatically derived box."""
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    original = original_mask.astype(bool)
    ys, xs = np.where(original)

    if len(xs) == 0:
        raise ValueError("Selected mask is empty.")

    height, width = original.shape
    pad = 30

    box = np.array([
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(width - 1, int(xs.max()) + pad),
        min(height - 1, int(ys.max()) + pad),
    ], dtype=np.float32)

    model = build_sam2(
        config,
        str(checkpoint),
        device=device,
    )

    predictor = SAM2ImagePredictor(model)

    with torch.inference_mode():
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            box=box,
            multimask_output=True,
        )

    original_area = int(original.sum())
    results = []

    for index, (candidate, score) in enumerate(zip(masks, scores), 1):
        candidate = candidate.astype(bool)

        intersection = np.logical_and(original, candidate).sum()
        union = np.logical_or(original, candidate).sum()

        iou = float(intersection / max(union, 1))
        area_ratio = float(candidate.sum() / original_area)

        valid = (
            iou >= 0.75
            and 0.80 <= area_ratio <= 1.25
        )

        combined = 0.60 * float(score) + 0.40 * iou

        results.append({
            "id": index,
            "sam_score": float(score),
            "mask_iou": iou,
            "area_ratio": area_ratio,
            "combined_score": combined,
            "valid": bool(valid),
        })

    valid_results = [r for r in results if r["valid"]]

    if valid_results:
        best = max(valid_results, key=lambda r: r["combined_score"])
        selected = masks[best["id"] - 1].astype(bool)
        selected_id = best["id"]
    else:
        selected = original.copy()
        selected_id = None

    del predictor, model

    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return selected, selected_id, results



def generate_box_candidates(image, checkpoint, config, device, box):
    """Generate SAM2 mask candidates from a user-defined bounding box."""
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    image = np.asarray(image)
    height, width = image.shape[:2]

    box = np.asarray(box, dtype=np.float32).reshape(4)
    x1, y1, x2, y2 = box.tolist()

    if not (0 <= x1 < x2 < width and 0 <= y1 < y2 < height):
        raise ValueError("Invalid bounding box coordinates.")

    model = build_sam2(
        config,
        str(checkpoint),
        device=device,
    )
    predictor = SAM2ImagePredictor(model)

    try:
        with torch.inference_mode():
            predictor.set_image(image)
            masks, scores, _ = predictor.predict(
                box=box,
                multimask_output=True,
            )

        masks = np.asarray(masks, dtype=bool).copy()
        scores = np.asarray(scores, dtype=np.float32).copy()

        return masks, scores

    finally:
        del predictor, model

        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

def run_selection(args):
    """Load a numbered candidate and save its automatically refined mask."""
    candidates_dir = args.output / "candidates"
    metadata_path = args.output / "candidates.json"

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Candidates metadata missing: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    entries = metadata["candidates"]

    selected_entry = next(
        (entry for entry in entries if entry["id"] == args.select_mask),
        None,
    )

    if selected_entry is None:
        raise ValueError(f"Mask ID {args.select_mask} not found.")

    mask_path = candidates_dir / selected_entry["file"]
    original = np.asarray(Image.open(mask_path).convert("L")) > 0
    image = np.asarray(Image.open(args.image).convert("RGB")).copy()

    if original.shape != image.shape[:2]:
        raise ValueError("Selected mask dimensions do not match image.")

    print(f"Refining Mask {args.select_mask}...", flush=True)

    final_mask, selected_id, results = refine_selected_mask(
        image,
        original,
        args.checkpoint,
        args.config,
        args.device,
    )

    final_path = args.output / "final_mask.png"

    Image.fromarray(
        final_mask.astype(np.uint8) * 255
    ).save(final_path)

    result = {
        "selected_mask_id": args.select_mask,
        "refinement_candidate": selected_id,
        "used_original": selected_id is None,
        "refinement_results": results,
        "final_mask": str(final_path),
    }

    (args.output / "selection.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("Refinement candidate:", selected_id)
    print("Final mask:", final_path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        default="configs/sam2.1/sam2.1_hiera_s.yaml",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--points-per-side", type=int, default=16)
    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument("--select-mask", type=int, default=None)
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if args.points_per_side <= 0 or args.max_candidates <= 0:
        parser.error("Candidate limits must be positive.")

    if args.select_mask is not None:
        run_selection(args)
        return

    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty.")

    args.output.mkdir(parents=True, exist_ok=True)
    masks_dir = args.output / "candidates"
    masks_dir.mkdir()

    image = np.asarray(
        Image.open(args.image).convert("RGB")
    ).copy()

    print("Generating SAM2 candidates...", flush=True)

    candidates = generate_candidates(
        image,
        args.checkpoint,
        args.config,
        args.device,
        args.points_per_side,
    )

    # Initial ranking: larger stable candidates first.
    candidates.sort(
        key=lambda c: (
            c.get("predicted_iou", 0),
            c.get("stability_score", 0),
            c.get("area", 0),
        ),
        reverse=True,
    )

    candidates = candidates[:args.max_candidates]

    if not candidates:
        raise RuntimeError("SAM2 produced no candidates.")

    metadata = []

    for index, candidate in enumerate(candidates, 1):
        mask = candidate["segmentation"].astype(np.uint8) * 255
        filename = f"mask_{index:03d}.png"

        Image.fromarray(mask).save(masks_dir / filename)

        metadata.append({
            "id": index,
            "file": filename,
            "area": int(candidate["area"]),
            "predicted_iou": float(candidate["predicted_iou"]),
            "stability_score": float(candidate["stability_score"]),
        })

    preview = args.output / "mask_candidates.png"
    create_contact_sheet(image, candidates, preview)

    (args.output / "candidates.json").write_text(
        json.dumps({
            "source_image": str(args.image.resolve()),
            "candidates": metadata,
        }, indent=2),
        encoding="utf-8",
    )

    print(f"Candidates: {len(candidates)}")
    print(f"Preview: {preview}")
    print(f"Masks: {masks_dir}")
    print("Select a numbered mask to continue.")


if __name__ == "__main__":
    main()
