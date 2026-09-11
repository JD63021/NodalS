#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$HOME/NodalS_GPU_RANS}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AN="$ROOT/postprocess/vtu_axial_friction/analyze_vtu_axial_friction.py"

BROOT="$ROOT/gpu/serial_cuda/results_rans_20d_5slabD_gate5i/baseline"
FROOT="$ROOT/gpu/serial_cuda/results_rans_20d_5slabD_gate5i/reference_shear"

BVTU="${BVTU:-$BROOT/VMFL003_20D_5SLABD_GATE5I_BASELINE_FP32.vtu}"
FVTU="${FVTU:-$FROOT/VMFL003_20D_5SLABD_GATE5I_REFERENCE_SHEAR_FP32.vtu}"
BWALL="${BWALL:-$BROOT/VMFL003_20D_5SLABD_GATE5I_BASELINE_FP32_wall_faces.csv}"
FWALL="${FWALL:-$FROOT/VMFL003_20D_5SLABD_GATE5I_REFERENCE_SHEAR_FP32_wall_faces.csv}"

for f in "$BVTU" "$FVTU" "$BWALL" "$FWALL"; do
  [[ -s "$f" ]] || { echo "ERROR: missing $f"; exit 2; }
done

BOUT="${BOUT:-${BVTU%.vtu}_axial_friction}"
FOUT="${FOUT:-${FVTU%.vtu}_axial_friction}"

python3 "$AN" "$BVTU" --outdir "$BOUT" \
  --diameter 0.004 --bulk 50 --rho 1 \
  --re 13691.7402481 --rel-roughness 0 \
  --reference-f 0.0284003 --xy-tol-D 1e-6 \
  --bin-width-D 0.25 --tail-start-D 12 --window-D 2

python3 "$AN" "$FVTU" --outdir "$FOUT" \
  --diameter 0.004 --bulk 50 --rho 1 \
  --re 13691.7402481 --rel-roughness 0 \
  --reference-f 0.0284003 --xy-tol-D 1e-6 \
  --bin-width-D 0.25 --tail-start-D 12 --window-D 2

python3 "$HERE/compare_wall_pressure.py" \
  --baseline-wall "$BWALL" \
  --forced-wall "$FWALL" \
  --baseline-pairs "$BOUT/cell_center_pairwise_dp_dz.csv" \
  --forced-pairs "$FOUT/cell_center_pairwise_dp_dz.csv" \
  --reference-f 0.0284003 \
  --out "$ROOT/gpu/serial_cuda/results_rans_20d_5slabD_gate5i/wall_causal_compare.csv"
