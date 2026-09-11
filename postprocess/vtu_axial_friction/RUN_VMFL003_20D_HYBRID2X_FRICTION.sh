#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VTU="${1:-$ROOT/gpu/serial_cuda/results_rans_20d_hybrid2x/VMFL003_20D_HYBRID2X_GATE5B_FP32.vtu}"
exec "$ROOT/postprocess/vtu_axial_friction/RUN_VMFL003_20D_FRICTION.sh" "$VTU"
