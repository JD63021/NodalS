#!/usr/bin/env bash
set -euo pipefail

# Repository-contained custom-AMG regression/example.
# The 40k OpenFOAM polyMesh is already tracked at:
#   meshes/pipe/40k/constant/polyMesh
#
# Usage:
#   ./examples/custom_amg_40k/run_40k_custom_amg_examples.sh pcg_smoothed
#   ./examples/custom_amg_40k/run_40k_custom_amg_examples.sh pcg_unsmoothed
#   ./examples/custom_amg_40k/run_40k_custom_amg_examples.sh richardson_smoothed
#   ./examples/custom_amg_40k/run_40k_custom_amg_examples.sh all

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-all}"
MESH="$ROOT/meshes/pipe/40k/constant/polyMesh"

for f in owner neighbour faces points boundary; do
  [[ -f "$MESH/$f" ]] || { echo "NODALS_40K_AMG_EXAMPLE status=MISSING_MESH_FILE file=$MESH/$f" >&2; exit 3; }
done
[[ -x "$ROOT/nodals_solver" ]] || { echo "NODALS_40K_AMG_EXAMPLE status=MISSING_SOLVER path=$ROOT/nodals_solver" >&2; exit 4; }

run_one() {
  local mode="$1" case_file
  case "$mode" in
    pcg_unsmoothed) case_file="$ROOT/cases/40k_Re20_pcg_unsmoothed_amg.case" ;;
    pcg_smoothed) case_file="$ROOT/cases/40k_Re20_pcg_smoothed_amg.case" ;;
    richardson_smoothed) case_file="$ROOT/cases/40k_Re20_richardson_smoothed_amg.case" ;;
    *) echo "unknown mode: $mode" >&2; return 2 ;;
  esac
  echo
  echo "================================================================"
  echo "NODALS_40K_AMG_EXAMPLE_START mode=$mode mesh=$MESH"
  echo "================================================================"
  cd "$ROOT"
  python3 scripts/nodals_case.py "$case_file" --set "solver.mesh=$MESH" --exe "$ROOT/nodals_solver"
  echo "NODALS_40K_AMG_EXAMPLE_END mode=$mode status=PASS"
}

case "$MODE" in
  all)
    run_one pcg_unsmoothed
    run_one pcg_smoothed
    run_one richardson_smoothed
    ;;
  pcg_unsmoothed|pcg_smoothed|richardson_smoothed) run_one "$MODE" ;;
  *) echo "Usage: $0 [pcg_unsmoothed|pcg_smoothed|richardson_smoothed|all]" >&2; exit 2 ;;
esac
