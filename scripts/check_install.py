#!/usr/bin/env python3
"""Validate the ObjectGS runtime without modifying it."""

import argparse
import importlib
import importlib.metadata
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GS_DIR = ROOT / "third_party" / "gaussian-splatting"
EXPECTED_COMMIT = "54c035f7834b564019656c3e3fcc3646292f727d"

failures = []
warnings = []


def report(name, ok, detail="", required=True):
    if ok:
        status = "PASS"
    elif required:
        status = "FAIL"
        failures.append(name)
    else:
        status = "WARN"
        warnings.append(name)

    print(f"[{status}] {name}: {detail}")


def check_package(name, import_name=None, required=True):
    try:
        version = importlib.metadata.version(name)
        module = import_name or name
        importlib.import_module(module)
        report(name, True, version)
    except Exception as exc:
        report(name, False, str(exc), required)


def check_colmap():
    executable = shutil.which("colmap")
    if executable is None:
        report(
            "COLMAP CLI",
            False,
            "Not found in PATH; required for run_colmap.py",
            required=False,
        )
        return

    result = subprocess.run(
        [executable, "-h"],
        capture_output=True,
        text=True,
    )
    report(
        "COLMAP CLI",
        result.returncode == 0,
        executable,
        required=False,
    )


def check_gpu():
    try:
        import torch
        report("PyTorch", True, torch.__version__)
    except ImportError:
        report("PyTorch", False, "Not installed")
        return

    cuda_available = torch.cuda.is_available()
    report(
        "PyTorch CUDA",
        cuda_available,
        f"torch CUDA={torch.version.cuda}",
    )

    nvcc = shutil.which("nvcc")
    report("CUDA compiler (nvcc)", nvcc is not None, nvcc or "Missing")

    if not GS_DIR.is_dir():
        report("Graphdeco checkout", False, "Missing")
        return

    result = subprocess.run(
        ["git", "-C", str(GS_DIR), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )

    commit = result.stdout.strip()
    report(
        "Graphdeco commit",
        result.returncode == 0 and commit == EXPECTED_COMMIT,
        commit or result.stderr.strip(),
    )

    patch = ROOT / "objectgs.patch"
    if not patch.exists():
        report("ObjectGS patch", False, "Patch file missing")
    else:
        result = subprocess.run(
            ["git", "-C", str(GS_DIR), "apply",
             "--reverse", "--check", str(patch)],
            capture_output=True,
            text=True,
        )
        report(
            "ObjectGS patch applied",
            result.returncode == 0,
            "Applied" if result.returncode == 0 else "Not detected",
        )

    for package, module in [
        ("diff-gaussian-rasterization", "diff_gaussian_rasterization"),
        ("simple-knn", "simple_knn"),
        ("fused-ssim", "fused_ssim"),
    ]:
        check_package(package, module)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["cpu", "gpu"], required=True
    )
    args = parser.parse_args()

    print(f"===== ObjectGS {args.mode.upper()} validation =====")
    print("Python:", sys.version.split()[0])

    for package, module in [
        ("numpy", "numpy"),
        ("Pillow", "PIL"),
        ("pycolmap", "pycolmap"),
        ("plyfile", "plyfile"),
        ("tqdm", "tqdm"),
    ]:
        check_package(package, module)

    check_colmap()

    if args.mode == "gpu":
        check_gpu()

    print("\n===== SUMMARY =====")
    print("Failures:", len(failures))
    print("Warnings:", len(warnings))

    if failures:
        print("Failed checks:", ", ".join(failures))
        sys.exit(1)

    print("ObjectGS validation passed.")


if __name__ == "__main__":
    main()
