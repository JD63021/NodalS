#!/usr/bin/env bash
set -euo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_MESH="${SOURCE_MESH:-$HOME/NodalS_GPU_HYBRID/gpu/serial_cuda/hybrid_hxt1/work/HXT1_mesh.bin}"
WALL_GAP_SCALE="${WALL_GAP_SCALE:-0.32}"
AXIAL_SCALE="${AXIAL_SCALE:-50}"
REF_YPLUS_MEAN="${REF_YPLUS_MEAN:-83.44770399615}"
REF_YPLUS_MAX="${REF_YPLUS_MAX:-101.7253636439}"
OUTDIR="${OUTDIR:-$SELF_DIR/mesh_500D_lowy}"
mkdir -p "$OUTDIR"
OUTMESH="$OUTDIR/HXT1_mesh_500D_lowy.bin"
MANIFEST="$OUTDIR/HXT1_mesh_500D_lowy.manifest.json"
[[ -f "$SOURCE_MESH" ]] || { echo "ERROR: source 10D HXT1 mesh not found: $SOURCE_MESH"; exit 2; }
python3 "$SELF_DIR/REMAP_HXT1_10D_TO_500D_LOWY.py" \
  --input "$SOURCE_MESH" --output "$OUTMESH" --manifest "$MANIFEST" \
  --axial-scale "$AXIAL_SCALE" --wall-gap-scale "$WALL_GAP_SCALE" \
  --ref-yplus-mean "$REF_YPLUS_MEAN" --ref-yplus-max "$REF_YPLUS_MAX"
echo "HXT1_500D_LOWY_MESH=$OUTMESH"
echo "HXT1_500D_LOWY_BUILD_STATUS=PASS"
