#!/usr/bin/env bash
# Prepare an independent ~99k hybrid mesh with y+150 at the first HEX midpoint,
# then run the existing OF_AUTO+Stage41 physical HEX trace_log kernel.
# Reuses the previously compiled isolated hybrid app; never touches production.
# Usage: bash RUN_Y150_TRACE_LOG.sh prepare|dry|5
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="${NODALS_HYBRID_APP:-$HOME/NodalS_GPU_HYBRID_RE1M_99K/turbulence/sst_gpu_v2}"
WORK="${NODALS_Y150_WORK:-$HOME/NODALS_HYBRID_RE1M_Y150_TRACE_RUN}"
CASE="$WORK/cases/PIPE50_RE1M_HYBRID_99K_Y150_TRACE_LOG.case"
MESH="$WORK/mesh/HXT1_mesh.bin"
MODE="${1:-prepare}"

case "$MODE" in
  prepare|dry|5) ;;
  *) echo "Usage: bash '$0' prepare|dry|5" >&2;exit 2;;
esac

if [[ ! -s "$APP/build/nodals_sst" || ! -f "$APP/nodals.py" ||
      ! -f "$APP/tools/prepare_hxt1.py" ]];then
    echo "The isolated hybrid app from the previous experiment is missing: $APP" >&2
    echo "Run the earlier 99k hybrid experiment setup first, or set NODALS_HYBRID_APP." >&2
    exit 2
fi
if ! grep -q 'legacy-segregated' "$APP/nodals.py" ||
   ! grep -q 'NODALS_HYBRID_INIT' "$APP/source/puretet_profile5_eq5_hxt5b/modules/app/sst_application.inc";then
    echo "This app has not been patched for native hybrid HEX/TET SST. Refusing." >&2
    exit 2
fi

mkdir -p "$WORK/mesh" "$WORK/cases"
if [[ ! -f "$CASE" ]];then
  cp "$HERE/PIPE50_RE1M_HYBRID_99K_Y150_TRACE_LOG.case" "$CASE"
fi

# This is the only mesh generation; case and VTUs live in an independent
# folder, preserving the divergent y+300 run for numerical comparison.
if [[ ! -s "$MESH" ]];then
  OLD_HXT0="${NODALS_OLD_HXT0:-$HOME/NODALS_HYBRID_RE1M_99K_RUN/mesh/HXT0.npz}"
  if [[ -f "$OLD_HXT0" && -f "$(dirname "$OLD_HXT0")/pipe_geometry.json" ]];then
    echo "Reusing exact y+300 99,216-cell topology: $OLD_HXT0"
    python3 "$HERE/rescale_hybrid_wall_layer.py" \
      --input "$OLD_HXT0" --outdir "$WORK/mesh" --target-yplus 150
  else
    echo "Previous HXT0 not found; generating fresh ~99k HEX/TET mesh"
    python3 "$HERE/generate_hybrid_pipe.py" \
      --outdir "$WORK/mesh" --sample-yplus 150 --target-cells 99216
  fi
  python3 "$APP/tools/prepare_hxt1.py" \
    --input "$WORK/mesh/HXT0.npz" --outdir "$WORK/mesh"
fi
test -s "$MESH" || { echo "HXT1 mesh generation failed" >&2;exit 1; }

# New code only reuses the already-implemented HEX symmetric trace kernel;
# the previously patched startup/pressure/momentum paths are unchanged.
python3 "$HERE/enable_hex_trace_log.py" --app "$APP"

echo "Y150_TRACE_LOG_WORK=$WORK"
echo "EDITABLE_CASE=$CASE"
echo "HYBRID_MESH=$MESH"
echo "NATIVE_APP=$APP"
echo "MESH_SUMMARY:"
cat "$WORK/mesh/pipe_geometry.json"

if [[ "$MODE" == prepare ]];then
  echo "PREPARED=PASS; no CUDA compilation"
  exit 0
fi
python3 "$APP/nodals.py" run --case "$CASE" --dry-run
if [[ "$MODE" == dry ]];then
  echo "DRY_RUN=PASS; no CUDA compilation"
  exit 0
fi

# This is one (unavoidable) incremental NVCC build because CUDA source
# dispatch and log-law kappa handling changed. Do not 'make clean'.
make -C "$APP" -j2 \
  NVCC="${NVCC:-/usr/local/cuda-12.2/bin/nvcc}" CUDA_ARCH=sm_86
echo "NATIVE_CUDA_BUILD=PASS; running 5 outer iterations only"
python3 -u "$APP/nodals.py" run --case "$CASE" --steps 5 \
  --set output.print_every=5 --set output.metrics_every=5
