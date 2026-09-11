#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WALL_TRACTION_MODE=spalding \
CASE_NAME=baseline \
exec "$HERE/RUN_RANS_GATE5I_20D_5SLABD_WALL_COMMON_FP32.sh"
