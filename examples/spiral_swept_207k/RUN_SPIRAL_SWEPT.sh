#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NODALS_ROOT:-$(cd "$HERE/../../turbulence/sst_gpu_v2" && pwd)}"
RUNNER="$HERE/run_adaptive_case.py"
MESH="$HERE/spiral_swept_207360_24tet.msh"
CASE="$HERE/SPIRAL_SWEPT_ADAPTIVE.case"
[[ -s "$MESH" ]] || { echo "ERROR: Mesh file not found: $MESH" >&2; exit 2; }
[[ -s "$CASE" ]] || { echo "ERROR: Case file not found: $CASE" >&2; exit 2; }
[[ -s "$ROOT/nodals.py" ]] || { echo "ERROR: NodalS V2 missing: $ROOT/nodals.py" >&2; exit 2; }
exec python3 -u "$RUNNER" --root "$ROOT" --case "$CASE" --mesh "$MESH" "$@"
