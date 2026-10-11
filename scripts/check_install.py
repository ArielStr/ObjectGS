#!/usr/bin/env python3
"""Verify an ObjectGS installation."""

import argparse
import importlib
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["cpu", "gpu"], default="gpu"
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    gs = root / "third_party" / "gaussian-splatting"
    patch = root / "objectgs.patch"

    failures = []

    def check(name, condition):
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {name}")
        if not condition:
            failures.append(name)

    print("===== ObjectGS Installation Check =====")

    for name in [
        "setup.sh",
        "requirements.txt",
        "objectgs.patch",
        "configs/objectgs_v1.yaml",
    ]:
        check(name, (root / name).is_file())

    check("Graphdeco train.py", (gs / "train.py").is_file())

    if gs.is_dir() and patch.is_file():
        result = subprocess.run(
            ["git", "apply", "--reverse", "--check", str(patch)],
            cwd=gs,
            capture_output=True,
            text=True,
        )
        check("ObjectGS patch applied", result.returncode == 0)
    else:
        check("ObjectGS patch applied", False)

    if args.mode == "gpu":
        try:
            import torch
            print("PyTorch:", torch.__version__)
            print("CUDA:", torch.version.cuda)
            check("CUDA available", torch.cuda.is_available())

            if torch.cuda.is_available():
                print("GPU:", torch.cuda.get_device_name(0))
        except Exception as exc:
            print("PyTorch error:", exc)
            check("PyTorch", False)

        for module in [
            "diff_gaussian_rasterization",
            "simple_knn",
            "fused_ssim",
        ]:
            try:
                importlib.import_module(module)
                check(module, True)
            except Exception as exc:
                print(f"{module}: {exc}")
                check(module, False)

    print("\n===== Result =====")
    if failures:
        print(f"FAILED: {len(failures)} check(s)")
        sys.exit(1)

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
