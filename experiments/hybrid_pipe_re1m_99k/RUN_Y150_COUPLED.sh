#!/usr/bin/env bash
# Hybrid HEX Q1+BF2/TET P1+BF3, y+150, OF_AUTO trace_log.
# Two selectable momentum solvers: M3 3x3 block FGS (2 sweeps) and
# 3Nv coupled FGMRES. Neither modifies the user's production SST source.
#
# Usage: bash RUN_Y150_COUPLED.sh prepare|dry|5
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="${NODALS_HYBRID_APP:-$HOME/NodalS_GPU_HYBRID_RE1M_99K/turbulence/sst_gpu_v2}"
WORK="${NODALS_HYBRID_M3_WORK:-$HOME/NODALS_HYBRID_Y150_M3_FGS2_WORK}"
BASE="${NODALS_Y150_WORK:-$HOME/NODALS_HYBRID_RE1M_Y150_TRACE_RUN}"
CASE="$WORK/cases/PIPE50_RE1M_HYBRID_Y150_M3_FGS2.case"
MESH="$WORK/mesh/HXT1_mesh.bin"
MODE="${1:-prepare}"

case "$MODE" in prepare|dry|5) ;; *) echo "Usage: $0 prepare|dry|5" >&2; exit 2;; esac
[[ -f "$APP/nodals.py" ]] || { echo "ERROR: missing isolated hybrid SST app: $APP" >&2;exit 2; }
[[ -f "$APP/source/puretet_profile5_eq5_hxt5b/modules/app/sst_application.inc" ]] || {
    echo "ERROR: no native source under $APP" >&2;exit 2;
}
mkdir -p "$WORK/mesh" "$WORK/cases"
if [[ ! -s "$BASE/mesh/HXT1_mesh.bin" ]];then
  echo "Missing source y+150 mesh; preparing old hybrid trace setup first."
  bash "$HERE/RUN_Y150_TRACE_LOG.sh" prepare
fi
[[ -s "$BASE/mesh/HXT1_mesh.bin" ]] || { echo "ERROR: no prepared y+150 HXT1 mesh";exit 2; }
if [[ ! -e "$MESH" ]];then
  ln -s "$BASE/mesh/HXT1_mesh.bin" "$MESH"
fi
if [[ ! -f "$CASE" ]];then
  cp "$HERE/PIPE50_RE1M_HYBRID_Y150_M3_FGS2.case" "$CASE"
fi
# The old source must already have the minimal hybrid and trace_log patches.
# This command adds only the new 3Nv solvers; it is idempotent.
python3 "$HERE/enable_hex_trace_log.py" --app "$APP"
python3 "$HERE/enable_hybrid_coupled.py" --app "$APP"
echo "HYBRID_COUPLED_APP=$APP"
echo "HYBRID_COUPLED_CASE=$CASE"
echo "HYBRID_Y150_MESH=$MESH"
echo "To edit: xdg-open '$CASE'"

if [[ "$MODE" == prepare ]];then
  echo "HYBRID_M3_PREPARED=YES (no CUDA build)"
  exit 0
fi
python3 "$APP/nodals.py" run --case "$CASE" --dry-run
if [[ "$MODE" == dry ]];then
  echo "HYBRID_M3_DRY=PASS (no CUDA build)"
  exit 0
fi

# One incremental NVCC build because the new native include is in the
# existing giant SST CUDA translation unit. Do NOT use make clean.
make -C "$APP" -j2 \
    NVCC="${NVCC:-/usr/local/cuda-12.2/bin/nvcc}" CUDA_ARCH=sm_86
echo "HYBRID_M3_CUDA_BUILT=YES; starting first 5 SIMPLE iterations"
python3 -u "$APP/nodals.py" run --case "$CASE" --steps 5 \
  --set output.print_every=5 --set output.metrics_every=5
