#!/usr/bin/env bash
set -e

OBJECTGS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

UPSTREAM_URL="https://github.com/graphdeco-inria/gaussian-splatting.git"
UPSTREAM_COMMIT="54c035f7834b564019656c3e3fcc3646292f727d"

THIRD_PARTY="$OBJECTGS_ROOT/third_party"
GS_DIR="$THIRD_PARTY/gaussian-splatting"
PATCH="$OBJECTGS_ROOT/objectgs.patch"

echo "========================================"
echo " ObjectGS setup"
echo "========================================"

mkdir -p "$THIRD_PARTY"

if [ -d "$GS_DIR" ]; then
    echo "[ObjectGS] Removing existing 3DGS checkout..."
    rm -rf "$GS_DIR"
fi

echo "[ObjectGS] Cloning official 3D Gaussian Splatting..."
git clone "$UPSTREAM_URL" "$GS_DIR"

echo "[ObjectGS] Checking out pinned upstream commit..."
git -C "$GS_DIR" checkout "$UPSTREAM_COMMIT"

echo "[ObjectGS] Initializing upstream submodules..."
git -C "$GS_DIR" submodule update --init --recursive

echo "[ObjectGS] Verifying ObjectGS patch..."
git -C "$GS_DIR" apply --check "$PATCH"

echo "[ObjectGS] Applying ObjectGS patch..."
git -C "$GS_DIR" apply "$PATCH"

echo
echo "========================================"
echo " ObjectGS core ready"
echo "========================================"
echo
echo "3DGS directory:"
echo "  $GS_DIR"
echo
echo "Pinned commit:"
echo "  $UPSTREAM_COMMIT"
echo
echo "ObjectGS patch:"
echo "  $PATCH"
