#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_MESH="${SOURCE_MESH:-$HOME/NodalS_GPU_HYBRID/gpu/serial_cuda/hybrid_hxt1/work/HXT1_mesh.bin}"
OUTDIR="${OUTDIR:-$ROOT/mesh_500D}"
OUTMESH="${OUTMESH:-$OUTDIR/HXT1_mesh_500D_stretched.bin}"
SCALE="${SCALE:-50}"
AXIS="${AXIS:-z}"
[[ -f "$SOURCE_MESH" ]] || { echo "ERROR: source HXT1 mesh missing: $SOURCE_MESH"; exit 2; }
mkdir -p "$OUTDIR"
python3 "$ROOT/STRETCH_HXT1_10D_TO_500D.py" \
  --input "$SOURCE_MESH" --output "$OUTMESH" --scale "$SCALE" --axis "$AXIS" \
  --manifest "$OUTDIR/HXT1_mesh_500D_stretched.manifest.json"
echo "HXT1_500D_MESH=$OUTMESH"
echo "HXT1_500D_BUILD_STATUS=PASS"
