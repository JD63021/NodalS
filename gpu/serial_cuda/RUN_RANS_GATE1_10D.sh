#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MESH="${MESH:-$HOME/Desktop/meshes/nodals_vmfl003_10D_radial460k/foam_case/constant/polyMesh}"
PREC="${PREC:-fp64}"
CASE="${CASE:-active}"

case "$PREC" in
  fp64) EXE="$ROOT/nodals_gpu_h8_fp64" ;;
  fp32) EXE="$ROOT/nodals_gpu_h8_fp32" ;;
  *) echo "PREC must be fp64 or fp32" >&2; exit 2 ;;
esac

case "$CASE" in
  scale0)
    RE="${RE:-20}"
    BULK="${BULK:-1}"
    SCALE="0"
    TAG="VMFL003_10D_GATE1_SCALE0_${PREC^^}"
    ;;
  active)
    RE="${RE:-13691.7402481}"
    BULK="${BULK:-50}"
    SCALE="${MIXLEN_SCALE:-1.0}"
    TAG="VMFL003_10D_GATE1_ACTIVE_${PREC^^}"
    ;;
  *) echo "CASE must be scale0 or active" >&2; exit 3 ;;
esac

[[ -x "$EXE" ]] || { echo "missing executable: $EXE" >&2; exit 4; }
[[ -f "$MESH/owner" ]] || { echo "missing mesh: $MESH" >&2; exit 5; }

LOG="${LOG:-$ROOT/${TAG}.log}"

"$EXE" \
  --mesh "$MESH" \
  --tag "$TAG" \
  --wall wall \
  --inlet inlet \
  --outlet outlet \
  --re "$RE" \
  --bulk "$BULK" \
  --supg 0 \
  --mixing-length 1 \
  --mixlen-scale "$SCALE" \
  --run-mode fixed10 \
  --max-outer 10 \
  --momentum-work fgs1 \
  --alpha-u 0.50 \
  --alpha-p 0.50 \
  --p-rtol 0.5 \
  --p-max 20 \
  --amg-hierarchy cf \
  --cf-coarsening pmis \
  --cf-strength classical-negative \
  --cf-theta 0.25 \
  --cf-interp exti \
  --cf-pmax 8 \
  --cf-aggressive-first 0 \
  --amg-smoother jacobi \
  --amg-jacobi-omega 0.7 \
  --amg-spectrum-policy auto \
  --pressure-solver richardson \
  --pressure-richardson-omega 1.0 \
  2>&1 | tee "$LOG"

echo
echo "=== GATE1 SUMMARY ==="
grep -E 'NODALS_GPU_RANS_GATE1|NODALS_GPU_RANS_GATE1_MIXLEN|NODALS_GPU_H8_FIXED10|NODALS_GPU_H8_RESIDENCY|NODALS_GPU_RESULT' "$LOG" || true
