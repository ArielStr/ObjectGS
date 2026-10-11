#!/usr/bin/env python3
"""Generate ObjectGS masks from ordered video frames using SAM2 tracking."""

import argparse
import re
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor


def frame_order(path):
    match = re.search(r"(\d+)$", path.stem)
    if match is None:
        raise ValueError(f"Frame filename must end with a number: {path.name}")
    return int(match.group(1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        default="configs/sam2.1/sam2.1_hiera_s.yaml",
    )
    parser.add_argument(
        "--point", nargs=2, type=float, required=True,
        metavar=("X", "Y"),
        help="Positive point in original first-frame pixel coordinates.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=["bf16", "fp32"],
        default="bf16",
        help="Inference precision (default: bf16).",
    )
    args = parser.parse_args()

    images_dir = args.images.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()

    if not images_dir.is_dir():
        parser.error(f"Image directory not found: {images_dir}")
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {output_dir}")

    frames = [
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg"}
    ]
    frames.sort(key=frame_order)

    if not frames:
        parser.error("No JPEG frames found.")

    indices = [frame_order(p) for p in frames]
    if len(indices) != len(set(indices)):
        parser.error("Duplicate numeric frame indices detected.")

    with Image.open(frames[0]) as image:
        width, height = image.size

    x, y = args.point
    if not (0 <= x < width and 0 <= y < height):
        parser.error(f"Point outside first frame ({width}x{height}).")

    print(f"Frames: {len(frames)}", flush=True)
    print(f"First frame: {frames[0].name}", flush=True)
    print(f"Positive point: ({x}, {y})", flush=True)

    predictor = build_sam2_video_predictor(
        args.config,
        str(checkpoint),
        device=args.device,
        vos_optimized=False,
    )

    # SAM2 requires numeric JPEG filenames.
    # Temporary symlinks preserve original images and avoid copying.
    with tempfile.TemporaryDirectory(prefix="objectgs_sam2_video_") as tmp:
        temp_dir = Path(tmp)

        for idx, frame in enumerate(frames):
            (temp_dir / f"{idx:06d}.jpg").symlink_to(frame)

        from contextlib import nullcontext

        if args.precision == "bf16":
            if not args.device.startswith("cuda"):
                parser.error("bf16 precision requires CUDA.")
            precision_context = torch.autocast(
                "cuda", dtype=torch.bfloat16
            )
        else:
            precision_context = nullcontext()

        with torch.inference_mode(), precision_context:
            state = predictor.init_state(
                video_path=str(temp_dir),
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
            )

            points = np.array([[x, y]], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)

            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=1,
                points=points,
                labels=labels,
            )

            output_dir.mkdir(parents=True, exist_ok=True)
            completed = set()

            for idx, obj_ids, logits in predictor.propagate_in_video(state):
                if 1 not in obj_ids:
                    raise RuntimeError(f"Object 1 missing in frame {idx}")

                obj_pos = list(obj_ids).index(1)
                mask = (
                    (logits[obj_pos] > 0)
                    .cpu()
                    .numpy()
                    .squeeze()
                    .astype(np.uint8) * 255
                )

                with Image.open(frames[idx]) as source:
                    expected_size = source.size

                if mask.shape != (expected_size[1], expected_size[0]):
                    raise RuntimeError(
                        f"Mask size mismatch for {frames[idx].name}: "
                        f"{mask.shape} vs {expected_size}"
                    )

                Image.fromarray(mask).save(
                    output_dir / f"{frames[idx].stem}.png"
                )
                completed.add(idx)

    if completed != set(range(len(frames))):
        raise RuntimeError(
            f"Missing tracked frames: "
            f"{sorted(set(range(len(frames))) - completed)}"
        )

    print(f"Saved {len(completed)} masks to {output_dir}")
    print("SAM2 video mask generation complete.")


if __name__ == "__main__":
    main()
