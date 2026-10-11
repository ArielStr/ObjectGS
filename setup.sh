#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GS_DIR="$ROOT/third_party/gaussian-splatting"
PATCH="$ROOT/objectgs.patch"

UPSTREAM_URL="https://github.com/graphdeco-inria/gaussian-splatting.git"
UPSTREAM_COMMIT="54c035f7834b564019656c3e3fcc3646292f727d"

MODE=""
INSTALL_COLMAP=false
INSTALL_SAM2=false

usage() {
    echo "Usage: bash setup.sh --mode {cpu|gpu} [--install-colmap] [--install-sam2]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            [[ $# -ge 2 ]] || { usage; exit 1; }
            MODE="$2"
            shift 2
            ;;
        --install-sam2)
            INSTALL_SAM2=true
            shift
            ;;
        --install-colmap)
            INSTALL_COLMAP=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ "$MODE" != "cpu" && "$MODE" != "gpu" ]]; then
    usage
    exit 1
fi

command -v python >/dev/null || {
    echo "ERROR: Python not found"
    exit 1
}

echo "[ObjectGS] Installing core dependencies..."
python -m pip install -r "$ROOT/requirements.txt"

if [[ "$INSTALL_COLMAP" == true ]]; then
    if command -v colmap >/dev/null 2>&1; then
        echo "[ObjectGS] COLMAP already installed."
    elif command -v apt-get >/dev/null 2>&1; then
        echo "[ObjectGS] Installing COLMAP via apt..."
        if [[ "$(id -u)" -eq 0 ]]; then
            apt-get update -qq
            apt-get install -y colmap
        elif command -v sudo >/dev/null 2>&1; then
            sudo apt-get update -qq
            sudo apt-get install -y colmap
        else
            echo "ERROR: Root privileges or sudo required."
            exit 1
        fi
    else
        echo "ERROR: Automatic COLMAP installation requires apt-get."
        exit 1
    fi
fi

if [[ "$INSTALL_SAM2" == true ]]; then
    echo "[ObjectGS] Installing official SAM2..."
    python -m pip install "sam2"

    python - <<'PY_SAM2'
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
print("[ObjectGS] SAM2 imports OK")
PY_SAM2

    echo "[ObjectGS] Preparing SAM2 checkpoint..."
    mkdir -p "$ROOT/checkpoints"

    python - "$ROOT/checkpoints/sam2.1_hiera_small.pt" <<'PY_CHECKPOINT'
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

destination = Path(sys.argv[1])
url = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/"
    "092824/sam2.1_hiera_small.pt"
)

def valid(path):
    return path.is_file() and path.stat().st_size > 100_000_000

if valid(destination):
    print("[ObjectGS] SAM2 checkpoint already exists.")
else:
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=".sam2_download_",
        delete=False
    ) as tmp:
        temporary = Path(tmp.name)

    try:
        print("[ObjectGS] Downloading SAM2 checkpoint...")
        urllib.request.urlretrieve(url, temporary)
        if not valid(temporary):
            raise RuntimeError("Downloaded SAM2 checkpoint is invalid.")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

print("[ObjectGS] SAM2 checkpoint ready:", destination)
PY_CHECKPOINT
fi

if [[ "$MODE" == "cpu" ]]; then
    echo "[ObjectGS] CPU setup complete."
    echo "[ObjectGS] GPU training extensions were not installed."
    exit 0
fi

echo "[ObjectGS] Checking GPU build prerequisites..."

command -v nvcc >/dev/null || {
    echo "ERROR: nvcc not found. Install CUDA toolkit."
    exit 1
}

python - <<'PY_CHECK'
import torch
import sys

if not torch.cuda.is_available():
    sys.exit("ERROR: CUDA is not available in PyTorch.")

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
PY_CHECK

python -m pip install ninja

mkdir -p "$ROOT/third_party"

if [[ ! -d "$GS_DIR/.git" ]]; then
    if [[ -e "$GS_DIR" ]]; then
        echo "ERROR: Existing Graphdeco directory is not a Git checkout."
        exit 1
    fi

    echo "[ObjectGS] Cloning Graphdeco..."
    git clone "$UPSTREAM_URL" "$GS_DIR"

    echo "[ObjectGS] Checking out pinned commit..."
    git -C "$GS_DIR" checkout "$UPSTREAM_COMMIT"
fi

CURRENT_COMMIT="$(git -C "$GS_DIR" rev-parse HEAD)"

if [[ "$CURRENT_COMMIT" != "$UPSTREAM_COMMIT" ]]; then
    echo "ERROR: Existing Graphdeco checkout is at a different commit."
    echo "Expected: $UPSTREAM_COMMIT"
    echo "Found:    $CURRENT_COMMIT"
    exit 1
fi

git -C "$GS_DIR" submodule update --init --recursive

if git -C "$GS_DIR" apply --check "$PATCH"; then
    echo "[ObjectGS] Applying patch..."
    git -C "$GS_DIR" apply "$PATCH"
elif git -C "$GS_DIR" apply --reverse --check "$PATCH"; then
    echo "[ObjectGS] Patch already applied."
else
    echo "ERROR: ObjectGS patch cannot be applied cleanly."
    exit 1
fi

RASTERIZER_HEADER="$GS_DIR/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h"

if [[ ! -f "$RASTERIZER_HEADER" ]]; then
    echo "ERROR: Rasterizer header missing."
    exit 1
fi

if ! grep -Fq '#include <cstdint>' "$RASTERIZER_HEADER"; then
    echo "[ObjectGS] Adding missing cstdint header..."
    sed -i '1i #include <cstdint>' "$RASTERIZER_HEADER"
fi

echo "[ObjectGS] Installing CUDA extensions..."

for extension in \
    diff-gaussian-rasterization \
    simple-knn \
    fused-ssim
do
    python -m pip install \
        "$GS_DIR/submodules/$extension" \
        --no-build-isolation
done

python - <<'PY_VERIFY'
import importlib

for module in [
    "diff_gaussian_rasterization",
    "simple_knn",
    "fused_ssim",
]:
    importlib.import_module(module)
    print("[ObjectGS] OK:", module)
PY_VERIFY

echo "[ObjectGS] GPU setup complete."
