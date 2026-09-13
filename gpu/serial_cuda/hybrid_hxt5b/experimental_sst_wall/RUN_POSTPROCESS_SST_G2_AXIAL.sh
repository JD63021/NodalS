#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="${RESULT_DIR:?Set RESULT_DIR=/path/to/results_sst_g2_*}"
DIAG_VTU="${DIAG_VTU:-$RESULT_DIR/HXT5B_full_rans_fp32_cell_diagnostics.vtu}"
SST_VTU="${SST_VTU:-$RESULT_DIR/HXT5B_full_rans_fp32_sst_g2.vtu}"
OUTDIR="${OUTDIR:-$RESULT_DIR/axial_postprocess}"
D="${D:-0.004}"
UB="${UB:-50}"
RE="${RE:-13691.7402481}"
AXIS="${AXIS:-z}"
P_BIN_D="${P_BIN_D:-0.25}"
P_WINDOW_D="${P_WINDOW_D:-2.0}"
P_STEP_D="${P_STEP_D:-0.25}"
WALL_ZONE_D="${WALL_ZONE_D:-0.5}"
PROFILE_HALF_D="${PROFILE_HALF_D:-0.25}"
RADIAL_BINS="${RADIAL_BINS:-30}"
DEV_START_FRAC="${DEV_START_FRAC:-0.70}"
DEV_END_FRAC="${DEV_END_FRAC:-0.90}"
PLOTS="${PLOTS:-auto}"

[[ -s "$DIAG_VTU" ]] || { echo "ERROR: missing diagnostic VTU: $DIAG_VTU"; exit 2; }
args=(
  --diag-vtu "$DIAG_VTU"
  --outdir "$OUTDIR"
  --D "$D" --ub "$UB" --re "$RE" --axis "$AXIS"
  --pressure-bin-D "$P_BIN_D"
  --pressure-window-D "$P_WINDOW_D"
  --pressure-step-D "$P_STEP_D"
  --wall-zone-D "$WALL_ZONE_D"
  --profile-halfwidth-D "$PROFILE_HALF_D"
  --radial-bins "$RADIAL_BINS"
  --developed-start-frac "$DEV_START_FRAC"
  --developed-end-frac "$DEV_END_FRAC"
  --plots "$PLOTS"
)
if [[ -s "$SST_VTU" ]]; then
  args+=(--sst-vtu "$SST_VTU")
fi

echo "=== SST G2 AXIAL POSTPROCESS ==="
echo "RESULT_DIR=$RESULT_DIR"
echo "DIAG_VTU=$DIAG_VTU"
echo "SST_VTU=$SST_VTU"
echo "OUTDIR=$OUTDIR"
python3 "$HERE/POSTPROCESS_SST_G2_AXIAL.py" "${args[@]}" | tee "$OUTDIR.console.log"

echo "SST_G2_AXIAL_POSTPROCESS_STATUS=PASS"
