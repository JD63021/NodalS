#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MESH_ROOT="${MESH_ROOT:-$HOME/Desktop/meshes/pipe}"
MESHES="${MESHES:-40k 111k 292k}"
RE_LIST="${RE_LIST:-${RE:-20}}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_ROOT="${SWEEP_ROOT:-$HOME/Downloads/P1BF3_GATE9N_PIPE_SWEEP_${STAMP}}"
mkdir -p "$SWEEP_ROOT"

echo "P1BF3_GATE9N_SWEEP_BEGIN meshes=[$MESHES] ReList=[$RE_LIST] root=$SWEEP_ROOT"
for re in $RE_LIST; do
  for mesh in $MESHES; do
    path="$MESH_ROOT/$mesh/constant/polyMesh"
    [[ -d "$path" ]] || { echo "P1BF3_GATE9N_SWEEP_SKIP mesh=$mesh reason=missing path=$path"; continue; }
    echo "P1BF3_GATE9N_SWEEP_CASE_BEGIN mesh=$mesh Re=$re"
    MESH_NAME="$mesh" MESH_PATH="$path" RE="$re" CASE_NAME="${mesh}_Re${re}_${INLET_PROFILE:-parabolic}" \
      OUTROOT="$SWEEP_ROOT/Re${re}/${mesh}" \
      bash "$ROOT/RUN_GATE9N_PIPE.sh"
  done
done
python3 "$ROOT/ANALYZE_PIPE_CONVERGENCE.py" "$SWEEP_ROOT" | tee "$SWEEP_ROOT/PIPE_CONVERGENCE_SUMMARY.tsv"
echo "P1BF3_GATE9N_SWEEP_RESULT=PASS root=$SWEEP_ROOT summary=$SWEEP_ROOT/PIPE_CONVERGENCE_SUMMARY.tsv"
