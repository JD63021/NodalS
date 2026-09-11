#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/NodalS_GPU_RANS}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BROOT="${BROOT:-$ROOT/gpu/serial_cuda/results_rans_20d_5slabD_gate5i/baseline}"
FROOT="${FROOT:-$ROOT/gpu/serial_cuda/results_rans_20d_5slabD_gate5i/reference_shear}"

BVTU="${BVTU:-$BROOT/VMFL003_20D_5SLABD_GATE5I_BASELINE_FP32.vtu}"
FVTU="${FVTU:-$FROOT/VMFL003_20D_5SLABD_GATE5I_REFERENCE_SHEAR_FP32.vtu}"

BWALL="${BWALL:-$BROOT/VMFL003_20D_5SLABD_GATE5I_BASELINE_FP32_wall_faces.csv}"
FWALL="${FWALL:-$FROOT/VMFL003_20D_5SLABD_GATE5I_REFERENCE_SHEAR_FP32_wall_faces.csv}"

BPAIR="${BPAIR:-$BROOT/VMFL003_20D_5SLABD_GATE5I_BASELINE_FP32_axial_friction/cell_center_pairwise_dp_dz.csv}"
FPAIR="${FPAIR:-$FROOT/VMFL003_20D_5SLABD_GATE5I_REFERENCE_SHEAR_FP32_axial_friction/cell_center_pairwise_dp_dz.csv}"

OUTDIR="${OUTDIR:-$ROOT/gpu/serial_cuda/results_rans_20d_5slabD_gate5i/momentum_balance}"

for f in "$BVTU" "$FVTU" "$BWALL" "$FWALL"; do
  [[ -s "$f" ]] || { echo "ERROR: required input missing: $f"; exit 2; }
done

args=(
  --baseline-vtu "$BVTU"
  --forced-vtu "$FVTU"
  --baseline-wall "$BWALL"
  --forced-wall "$FWALL"
  --outdir "$OUTDIR"
  --diameter "${D:-0.004}"
  --bulk "${UBULK:-50}"
  --reference-f "${REFERENCE_F:-0.0284003}"
  --plot-z0-D "${PLOT_Z0_D:-10}"
  --plot-z1-D "${PLOT_Z1_D:-19.6}"
)

[[ -s "$BPAIR" ]] && args+=(--baseline-pairs "$BPAIR")
[[ -s "$FPAIR" ]] && args+=(--forced-pairs "$FPAIR")

python3 "$HERE/compare_gate5i_momentum_balance.py" "${args[@]}"

echo
echo "=== MOMENTUM BALANCE OUTPUT FILES ==="
find "$OUTDIR" -maxdepth 1 -type f -printf '%f\n' | sort
