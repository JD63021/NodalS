#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUTDIR="${OUTDIR:-$HERE/results_rans_20d_5slabD_gate5k_discrete_budget}"
TAG="${TAG:-VMFL003_20D_5SLABD_GATE5K_DISCRETE_BUDGET_FP32}"
mkdir -p "$OUTDIR"

export NODALS_DISCRETE_BUDGET_CSV="${BUDGET_CSV:-$OUTDIR/${TAG}_discrete_budget.csv}"

# Same baseline physics as Gate5H/Gate5I/Gate5J:
# 5 slabs/D, no SUPG, weak Spalding, Cheb4 pressure AMG.
WALL_TRACTION_MODE=spalding \
CASE_NAME=gate5k_discrete_budget \
TAG="$TAG" \
OUTDIR="$OUTDIR" \
VTU_OUT="$OUTDIR/${TAG}.vtu" \
WALL_DIAG_CSV="$OUTDIR/${TAG}_wall_faces.csv" \
exec "$HERE/RUN_RANS_GATE5I_20D_5SLABD_WALL_COMMON_FP32.sh"
