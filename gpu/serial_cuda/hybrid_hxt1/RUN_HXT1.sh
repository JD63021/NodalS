#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HXT0="${HXT0:-$HOME/Downloads/NodalS_HXT0_HYBRID_MESH_20260911/work/HXT0_mesh.npz}"
OUT="${OUT:-$ROOT/work}"

command -v python3 >/dev/null || { echo "ERROR: python3 not found"; exit 2; }
command -v nvcc >/dev/null || { echo "ERROR: nvcc not found"; exit 2; }
[[ -f "$HXT0" ]] || { echo "ERROR: HXT0 mesh not found: $HXT0"; exit 2; }

mkdir -p "$OUT"
echo "=== HXT1 HOST PREPARE ==="
python3 "$ROOT/prepare_hxt1.py" --input "$HXT0" --outdir "$OUT" \
  2>&1 | tee "$OUT/HXT1_PREPARE.log"

echo
echo "=== HXT1 CUDA BUILD ==="
make -C "$ROOT" clean
make -C "$ROOT" -j"$(nproc)" ARCH="${ARCH:-sm_86}" \
  2>&1 | tee "$OUT/HXT1_BUILD.log"

echo
echo "=== HXT1 DEVICE-RESIDENT MESH/DOF SMOKE ==="
"$ROOT/nodals_hxt1_gpu" --mesh "$OUT/HXT1_mesh.bin" \
  2>&1 | tee "$OUT/HXT1_GPU.log"

echo
echo "=== HXT1 HOST AUDIT ==="
cat "$OUT/HXT1_AUDIT_HOST.txt"

grep -q '^HXT1_STATUS=PASS$' "$OUT/HXT1_GPU.log"
echo "HXT1_GATE_STATUS=PASS"
