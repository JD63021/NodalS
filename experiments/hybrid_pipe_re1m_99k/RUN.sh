#!/usr/bin/env bash
# GitHub-native Re1M 99k hybrid experiment. Commands: prepare | dry | 5
# Keeps the November-14 production source untouched.  No chat downloads.
set -euo pipefail
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIG="${NODALS_ORIGINAL_APP:-$HOME/NodalS_GPU_GENROBIN_ADAPTIVE/turbulence/sst_gpu_v2}"
APP="${NODALS_HYBRID_APP:-$HOME/NodalS_GPU_HYBRID_RE1M_99K/turbulence/sst_gpu_v2}"
WORK="${NODALS_HYBRID_WORK:-$HOME/NODALS_HYBRID_RE1M_99K_RUN}"
CASE="$WORK/cases/PIPE50_RE1M_HYBRID_99K_ONEPOINT.case"
MODE="${1:-prepare}"
[[ "$MODE" == prepare || "$MODE" == dry || "$MODE" == 5 ]] || { echo "Usage: bash RUN.sh prepare|dry|5" >&2; exit 2; }
if [[ ! -f "$ORIG/nodals.py" ]]; then
    echo "Current-Nov14 source not found: $ORIG" >&2
    echo "Set NODALS_ORIGINAL_APP=/absolute/path/to/turbulence/sst_gpu_v2" >&2
    exit 1
fi
if [[ ! -f "$APP/nodals.py" ]]; then
    mkdir -p "$APP"
    echo "Copying source into isolated experimental app: $APP"
    rsync -a "$ORIG/" "$APP/"
fi
mkdir -p "$WORK/cases" "$WORK/mesh"
if [[ ! -f "$CASE" ]]; then
    cp "$EXP/cases/PIPE50_RE1M_HYBRID_99K_ONEPOINT.case" "$CASE"
fi
if [[ ! -f "$WORK/mesh/HXT1_mesh.bin" ]]; then
    python3 "$EXP/generate_hybrid_pipe.py" --outdir "$WORK/mesh"
    python3 "$APP/tools/prepare_hxt1.py" --input "$WORK/mesh/HXT0.npz" --outdir "$WORK/mesh"
fi
if [[ ! -s "$WORK/mesh/HXT1_mesh.bin" ]]; then
    echo "Missing or empty HXT1 mesh; aborting." >&2; exit 1
fi
python3 "$EXP/enable_hybrid_local.py" --app "$APP"
echo "EXPERIMENT=$EXP"
echo "ISOLATED_APP=$APP"
echo "EDITABLE_CASE=$CASE"
echo "MESH=$WORK/mesh/HXT1_mesh.bin"
if [[ "$MODE" == prepare ]]; then
    echo "PREPARED=YES. Edit the case, then run: bash '$EXP/RUN.sh' dry"
    exit 0
fi
python3 "$APP/nodals.py" run --case "$CASE" --dry-run
if [[ "$MODE" == dry ]]; then
    echo "DRY_RUN=PASS; CUDA NOT BUILT"
    exit 0
fi
# Only first CUDA-header modification rebuilds expensive CUDA translation unit.
# Preserve the copied build objects and .d dependency files to avoid extra work.
make -C "$APP" -j2 NVCC="${NVCC:-/usr/local/cuda-12.2/bin/nvcc}" CUDA_ARCH=sm_86
echo "Beginning 5-iteration GPU smoke; monitor terminal.log in output directory."
python3 -u "$APP/nodals.py" run --case "$CASE" --steps 5
