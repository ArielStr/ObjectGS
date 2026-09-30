#!/usr/bin/env python3

"""
Train ObjectGS using the patched Graphdeco 3D Gaussian Splatting runtime.

This script is a thin wrapper around the upstream train.py patched by
ObjectGS. It does not implement training itself.

Expected runtime:
    <runtime>/train.py

Expected scene:
    <scene>/
        images/
        masks/
        sparse/

The ObjectGS runtime must support:
    --lambda_outside
    --mask_normalized_densification
"""

import argparse
import subprocess
import sys
from pathlib import Path


def require_path(path: Path, description: str):
    if not path.exists():
        raise SystemExit(f"{description} does not exist: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train an ObjectGS reconstruction."
    )

    parser.add_argument(
        "--runtime",
        type=Path,
        required=True,
        help=(
            "Patched Graphdeco Gaussian Splatting runtime "
            "containing ObjectGS train.py."
        ),
    )

    parser.add_argument(
        "--scene",
        type=Path,
        required=True,
        help="Prepared ObjectGS scene containing images/, masks/, sparse/.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Training output directory.",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=7000,
        help="Number of training iterations. Default: 7000.",
    )

    parser.add_argument(
        "--lambda-outside",
        type=float,
        default=1.0,
        help="Weight of the outside-mask RGB penalty. Default: 1.0.",
    )

    parser.add_argument(
        "--save-iterations",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Iterations at which Gaussians are saved. "
            "Default: final iteration only."
        ),
    )

    parser.add_argument(
        "--checkpoint-iterations",
        type=int,
        nargs="*",
        default=None,
        help="Optional checkpoint iterations.",
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help=(
            "Optional upstream 3DGS resolution (-r). "
            "If omitted, upstream behavior is preserved."
        ),
    )

    parser.add_argument(
        "--no-mask-normalized-densification",
        action="store_true",
        help=(
            "Disable ObjectGS mask-normalized densification. "
            "Useful only for ablations."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the training command without running it.",
    )

    args, passthrough = parser.parse_known_args()

    runtime = args.runtime.resolve()
    scene = args.scene.resolve()
    output = args.output.resolve()

    train_py = runtime / "train.py"

    require_path(runtime, "Runtime")
    require_path(train_py, "Runtime train.py")
    require_path(scene, "Scene")
    require_path(scene / "images", "Scene images directory")
    require_path(scene / "masks", "Scene masks directory")
    require_path(scene / "sparse", "Scene sparse directory")

    if args.iterations <= 0:
        raise SystemExit("--iterations must be greater than zero.")

    if args.lambda_outside < 0:
        raise SystemExit("--lambda-outside must be non-negative.")

    save_iterations = (
        args.save_iterations
        if args.save_iterations
        else [args.iterations]
    )

    cmd = [
        sys.executable,
        str(train_py),
        "-s", str(scene),
        "-m", str(output),
        "--iterations", str(args.iterations),
        "--lambda_outside", str(args.lambda_outside),
        "--save_iterations",
        *[str(x) for x in save_iterations],
    ]

    if not args.no_mask_normalized_densification:
        cmd.append("--mask_normalized_densification")

    if args.checkpoint_iterations:
        cmd.extend([
            "--checkpoint_iterations",
            *[str(x) for x in args.checkpoint_iterations],
        ])

    if args.resolution is not None:
        cmd.extend([
            "-r",
            str(args.resolution),
        ])

    # Allow advanced users to pass additional upstream Graphdeco arguments.
    cmd.extend(passthrough)

    print("===== ObjectGS Training =====")
    print(f"Runtime                       : {runtime}")
    print(f"Scene                         : {scene}")
    print(f"Output                        : {output}")
    print(f"Iterations                    : {args.iterations}")
    print(f"Lambda outside                : {args.lambda_outside}")
    print(
        "Mask-normalized densification: "
        f"{not args.no_mask_normalized_densification}"
    )
    print(f"Save iterations               : {save_iterations}")

    if passthrough:
        print(
            "Additional upstream arguments : "
            + " ".join(passthrough)
        )

    print("\nCommand:")
    print(" ".join(cmd))

    if args.dry_run:
        print("\n✓ Dry run complete.")
        return

    output.mkdir(parents=True, exist_ok=True)

    print("\nStarting ObjectGS training...\n")

    subprocess.run(
        cmd,
        cwd=runtime,
        check=True,
    )

    print("\n===== DONE =====")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
