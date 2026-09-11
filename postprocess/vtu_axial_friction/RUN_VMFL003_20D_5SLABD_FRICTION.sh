#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

VTU="${1:-$ROOT/gpu/serial_cuda/results_rans_20d_5slabD_nosupg/VMFL003_20D_5SLABD_NOSUPG_GATE5H_FP32.vtu}"
OUTDIR="${OUTDIR:-${VTU%.vtu}_axial_friction}"

python3 "$HERE/analyze_vtu_axial_friction.py" "$VTU" \
  --outdir "$OUTDIR" \
  --diameter "${D:-0.004}" \
  --bulk "${BULK:-50}" \
  --rho "${RHO:-1}" \
  --re "${RE:-13691.7402481}" \
  --rel-roughness "${REL_ROUGHNESS:-0}" \
  --reference-f "${REFERENCE_F:-0.0284003}" \
  --xy-tol-D "${XY_TOL_D:-1e-6}" \
  --bin-width-D "${BIN_WIDTH_D:-0.25}" \
  --tail-start-D "${TAIL_START_D:-12}" \
  --window-D "${WINDOW_D:-2}"
