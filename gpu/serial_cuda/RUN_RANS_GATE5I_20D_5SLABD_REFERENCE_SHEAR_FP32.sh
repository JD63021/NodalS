#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WALL_TRACTION_MODE=reference_shear \
WALL_REFERENCE_F="${WALL_REFERENCE_F:-0.0284003}" \
CASE_NAME=reference_shear \
exec "$HERE/RUN_RANS_GATE5I_20D_5SLABD_WALL_COMMON_FP32.sh"
