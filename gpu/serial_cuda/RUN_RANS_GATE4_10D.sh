#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MESH="${MESH:-$HOME/Desktop/meshes/nodals_vmfl003_10D_radial460k/foam_case/constant/polyMesh}"
PREC="${PREC:-fp64}"
EXE="$HERE/nodals_gpu_h8_${PREC}"
[[ -x "$EXE" ]] || { echo "ERROR: executable missing: $EXE"; exit 2; }
[[ -f "$MESH/owner" ]] || { echo "ERROR: mesh missing: $MESH"; exit 3; }
TAG="VMFL003_10D_GATE4_CONVERGED_${PREC^^}"
LOG="$HERE/${TAG}.log"

stdbuf -oL -eL "$EXE" \
  --mesh "$MESH" \
  --tag "$TAG" \
  --wall wall --inlet inlet --outlet outlet \
  --re 13691.7402481 --bulk 50 \
  --supg 0 \
  --mixing-length 1 --mixlen-scale 1.0 \
  --dg-inlet 1 \
  --weak-wall 1 \
  --run-mode converge --max-outer 1200 --simple-tol 1e-3 \
  --momentum-work fgs1 \
  --alpha-u 0.50 --alpha-p 0.50 \
  --p-rtol 0.90 --p-atol 1e-12 --p-max 50 \
  --amg-hierarchy cf \
  --cf-coarsening pmis --cf-strength classical-negative \
  --cf-theta 0.25 --cf-interp exti --cf-pmax 8 --cf-aggressive-first 0 \
  --amg-smoother jacobi --amg-jacobi-omega 0.7 \
  --amg-spectrum-policy auto \
  --pressure-solver pcg \
  2>&1 | tee "$LOG"

echo
echo "=== GATE4 SUMMARY ==="
grep -E 'NODALS_GPU_RANS_GATE1_MIXLEN|NODALS_GPU_RANS_GATE3_WALL|NODALS_GPU_H8_FULLCONV|NODALS_GPU_H8_RESIDENCY|NODALS_GPU_RESULT' "$LOG" || true
python3 "$HERE/CHECK_RANS_GATE4_10D.py" "$LOG"
