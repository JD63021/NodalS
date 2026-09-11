#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$HOME/NodalS_GPU_RANS}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUNDIR="${RUNDIR:-$ROOT/gpu/serial_cuda/results_rans_20d_5slabD_gate5j_advective_diag}"
TAG="${TAG:-VMFL003_20D_5SLABD_GATE5J_ADVECTIVE_DIAG_FP32}"
VTU="${VTU:-$RUNDIR/${TAG}.vtu}"
WALL="${WALL:-$RUNDIR/${TAG}_wall_faces.csv}"
CONV="${CONV:-$RUNDIR/${TAG}_convection_form.csv}"
FRICDIR="${FRICDIR:-$RUNDIR/${TAG}_axial_friction}"

python3 "$ROOT/postprocess/vtu_axial_friction/analyze_vtu_axial_friction.py" "$VTU" \
  --outdir "$FRICDIR" \
  --diameter 0.004 --bulk 50 --rho 1 \
  --re 13691.7402481 --rel-roughness 0 \
  --reference-f 0.0284003 \
  --xy-tol-D 1e-6 --bin-width-D 0.25 \
  --tail-start-D 12 --window-D 2

python3 "$HERE/compare_advective_error.py" \
  --pair "$FRICDIR/cell_center_pairwise_dp_dz.csv" \
  --wall "$WALL" \
  --conv "$CONV" \
  --reference-f 0.0284003
