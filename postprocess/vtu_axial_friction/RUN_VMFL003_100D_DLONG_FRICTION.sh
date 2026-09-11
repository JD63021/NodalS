#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

VTU="${1:-$ROOT/gpu/serial_cuda/results_rans_100d_20dfine_dlong/VMFL003_100D_20DFINE_DLONG_GATE5D_FP32.vtu}"
OUTDIR="${OUTDIR:-${VTU%.vtu}_axial_friction}"

# D-long cells downstream, so 1D axial bins and 10D printed windows are more
# meaningful than the 0.25D/2D defaults used for the original 20D fine mesh.
python3 "$HERE/analyze_vtu_axial_friction.py" "$VTU" \
  --outdir "$OUTDIR" \
  --diameter "${D:-0.004}" \
  --bulk "${BULK:-50}" \
  --rho "${RHO:-1}" \
  --re "${RE:-13691.7402481}" \
  --rel-roughness "${REL_ROUGHNESS:-0}" \
  --reference-f "${REFERENCE_F:-0.0284003}" \
  --xy-tol-D "${XY_TOL_D:-1e-6}" \
  --bin-width-D "${BIN_WIDTH_D:-1.0}" \
  --tail-start-D "${TAIL_START_D:-20}" \
  --window-D "${WINDOW_D:-10}"
