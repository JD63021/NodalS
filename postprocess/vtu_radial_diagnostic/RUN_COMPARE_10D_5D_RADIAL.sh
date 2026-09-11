#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/NodalS_GPU_RANS}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VTU10="${VTU10:-$ROOT/gpu/serial_cuda/results_rans_20d_10slabD_nosupg/VMFL003_20D_10SLABD_NOSUPG_GATE5G_FP32.vtu}"
VTU5="${VTU5:-$ROOT/gpu/serial_cuda/results_rans_20d_5slabD_nosupg/VMFL003_20D_5SLABD_NOSUPG_GATE5H_FP32.vtu}"
OUTDIR="${OUTDIR:-$ROOT/gpu/serial_cuda/results_rans_20d_axial_resolution_radial_compare}"

[[ -s "$VTU10" ]] || { echo "ERROR: missing 10-slabs/D VTU: $VTU10"; exit 2; }
[[ -s "$VTU5" ]] || { echo "ERROR: missing 5-slabs/D VTU: $VTU5"; exit 3; }

python3 "$HERE/compare_vtu_radial_error.py" \
  --vtu10 "$VTU10" \
  --vtu5 "$VTU5" \
  --outdir "$OUTDIR" \
  --diameter "${D:-0.004}" \
  --bulk "${UBULK:-50}" \
  --rho "${RHO:-1}" \
  --reference-f "${REFERENCE_F:-0.0284003}" \
  --z0-D "${Z0_D:-14}" \
  --z1-D "${Z1_D:-18}" \
  --radial-bins "${RADIAL_BINS:-20}" \
  --axial-bin-D "${AXIAL_BIN_D:-0.5}" \
  --xy-tol-D "${XY_TOL_D:-1e-6}"

echo
echo "=== OUTPUT FILES ==="
find "$OUTDIR" -maxdepth 1 -type f -printf '%f\n' | sort
