#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HXT1_MESH="${HXT1_MESH:-$ROOT/../hybrid_hxt1/work/HXT1_mesh.bin}"
OUT="${OUT:-$ROOT/work}"
GAMMA="${HXT2_GAMMA:-20}"
NU="${HXT2_NU:-1}"

command -v nvcc >/dev/null || { echo "ERROR: nvcc not found"; exit 2; }
[[ -f "$HXT1_MESH" ]] || { echo "ERROR: HXT1 mesh not found: $HXT1_MESH"; exit 2; }
mkdir -p "$OUT"

echo "=== HXT2 CUDA BUILD ==="
make -C "$ROOT" clean
make -C "$ROOT" -j"$(nproc)" ARCH="${ARCH:-sm_86}" 2>&1 | tee "$OUT/HXT2_BUILD.log"

echo
echo "=== HXT2 CONSTANT/LINEAR DIFFUSION + INTERNAL NITSCHE GATE ==="
"$ROOT/nodals_hxt2_gpu" --mesh "$HXT1_MESH" --nu "$NU" --gamma "$GAMMA" 2>&1 | tee "$OUT/HXT2_RUN.log"

grep -q '^HXT2_STATUS=PASS$' "$OUT/HXT2_RUN.log"
echo "HXT2_GATE_STATUS=PASS"
