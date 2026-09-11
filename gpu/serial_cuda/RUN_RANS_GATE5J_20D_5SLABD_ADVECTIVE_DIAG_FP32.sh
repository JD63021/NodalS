#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUTDIR="${OUTDIR:-$HERE/results_rans_20d_5slabD_gate5j_advective_diag}"
TAG="${TAG:-VMFL003_20D_5SLABD_GATE5J_ADVECTIVE_DIAG_FP32}"
mkdir -p "$OUTDIR"

export NODALS_CONVECTION_FORM_DIAG_CSV="${CONV_DIAG_CSV:-$OUTDIR/${TAG}_convection_form.csv}"
export NODALS_CONVECTION_FORM_SLAB_D="${CONV_SLAB_D:-0.2}"

WALL_TRACTION_MODE=spalding \
CASE_NAME=gate5j_advective_diag \
TAG="$TAG" \
OUTDIR="$OUTDIR" \
VTU_OUT="$OUTDIR/${TAG}.vtu" \
WALL_DIAG_CSV="$OUTDIR/${TAG}_wall_faces.csv" \
exec "$HERE/RUN_RANS_GATE5I_20D_5SLABD_WALL_COMMON_FP32.sh"
