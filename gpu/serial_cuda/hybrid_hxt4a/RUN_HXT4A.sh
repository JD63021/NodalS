#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MESH="${MESH:-$ROOT/../hybrid_hxt3b/work/hxt1/HXT1_mesh.bin}"
OUT="${OUT:-$ROOT/work}"
GAMMA="${HXT4A_GAMMA:-50}"
RAU="${HXT4A_RAU_SCALE:-24}"
AP="${HXT4A_ALPHA_P:-1}"
TOL="${HXT4A_SIMPLE_TOL:-1e-3}"
MAX="${HXT4A_MAX_OUTER:-300}"

[[ -f "$MESH" ]] || { echo "ERROR: HXT3B 1D gate mesh missing: $MESH"; exit 2; }
mkdir -p "$OUT"

echo "=== HXT4A BUILD FP64 + FP32 ==="
make -C "$ROOT" clean
make -C "$ROOT" -j"$(nproc)" ARCH="${ARCH:-sm_86}" 2>&1 | tee "$OUT/HXT4A_BUILD.log"

run_one () {
  local precision="$1" exe="$2" log="$3"
  echo
  echo "=== HXT4A $precision ==="
  local runner=("$exe")
  if command -v stdbuf >/dev/null 2>&1; then runner=(stdbuf -oL -eL "$exe"); fi
  "${runner[@]}" --mesh "$MESH" --gamma "$GAMMA" --rau-scale "$RAU" --alpha-p "$AP" --simple-tol "$TOL" --max-outer "$MAX" 2>&1 | tee "$log"
  grep -q '^HXT4A_PRECISION_STATUS=PASS$' "$log"
}

run_one fp64 "$ROOT/nodals_hxt4a_fp64" "$OUT/HXT4A_FP64.log"
run_one fp32 "$ROOT/nodals_hxt4a_fp32" "$OUT/HXT4A_FP32.log"

echo
echo "=== HXT4A FP32/FP64 PARITY ==="
python3 "$ROOT/compare_hxt4a.py" "$OUT/HXT4A_FP64.log" "$OUT/HXT4A_FP32.log" | tee "$OUT/HXT4A_PARITY.log"
grep -q '^HXT4A_PARITY_STATUS=PASS$' "$OUT/HXT4A_PARITY.log"

echo "HXT4A_GATE_STATUS=PASS"
