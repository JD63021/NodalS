#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MESH="${MESH:-$ROOT/../hybrid_hxt1/work/HXT1_mesh.bin}"
OUT="${OUT:-$ROOT/work}"

command -v nvcc >/dev/null || { echo "ERROR: nvcc not found"; exit 2; }
[[ -f "$MESH" ]] || { echo "ERROR: HXT1 mesh not found: $MESH"; exit 2; }

mkdir -p "$OUT"
echo "=== HXT3A BUILD ==="
make -C "$ROOT" clean
make -C "$ROOT" -j"$(nproc)" ARCH="${ARCH:-sm_86}" 2>&1 | tee "$OUT/HXT3A_BUILD.log"

echo
echo "=== HXT3A PRESSURE / INTERFACE-FLUX ALGEBRA GATE ==="
"$ROOT/nodals_hxt3a_gpu" --mesh "$MESH" 2>&1 | tee "$OUT/HXT3A_RUN.log"

grep -q '^HXT3A_STATUS=PASS$' "$OUT/HXT3A_RUN.log"
echo "HXT3A_GATE_STATUS=PASS"
