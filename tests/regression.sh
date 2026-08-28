#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-}"
python3 tests/check_source_freeze.py
python3 tests/test_case_translation.py
python3 scripts/nodals_case.py cases/test.case --dry-run >/tmp/nodals_case_dryrun.$$.txt
tail -3 /tmp/nodals_case_dryrun.$$.txt
rm -f /tmp/nodals_case_dryrun.$$.txt

echo "NODALS_REGRESSION static_status=PASS"
if [[ -z "$MODE" ]]; then
  echo "NODALS_REGRESSION status=PASS_STATIC_ONLY hint='--fast runs primary 40k fast path; --compact-parity compares compact path to frozen Gate9N'"
  exit 0
fi

# shellcheck source=../scripts/petsc_env.sh
source "$ROOT/scripts/petsc_env.sh"

if [[ "$MODE" == "--fast" ]]; then
  make -j"${JOBS:-$(nproc)}" nodals_solver
  MESH="${MESH_PATH:-$HOME/Desktop/meshes/pipe/40k/constant/polyMesh}"
  [[ -d "$MESH" ]] || { echo "NODALS_FAST_REGRESSION status=FAIL reason=missing_mesh path=$MESH"; exit 4; }
  OUT="${NODALS_FAST_ROOT:-$HOME/Downloads/NodalS_fast40k_$(date +%Y%m%d_%H%M%S)}"
  python3 scripts/nodals_case.py cases/test.case --set "solver.mesh=$MESH" --set "run.output_root=$OUT"
  LOG="$OUT/40k_Re20_fast_full_gamg.log"
  grep -q 'NODALS_FAST_GAMG_RICHARDSON' "$LOG" || { echo "NODALS_FAST_REGRESSION status=FAIL reason=fast_backend_marker_missing"; exit 5; }
  grep -q 'P1BF3_RESULT status=PASS' "$LOG" || { echo "NODALS_FAST_REGRESSION status=FAIL reason=solver_not_converged"; exit 6; }
  echo "NODALS_FAST_REGRESSION status=PASS backend=fast_full_gamg mesh=40k log=$LOG"
  exit 0
fi

if [[ "$MODE" == "--compact-parity" || "$MODE" == "--pipe" ]]; then
  make -j"${JOBS:-$(nproc)}" nodals_solver nodals_reference_solver
  MESH="${MESH_PATH:-$HOME/Desktop/meshes/pipe/111k/constant/polyMesh}"
  [[ -d "$MESH" ]] || { echo "NODALS_REGRESSION status=FAIL reason=missing_mesh path=$MESH"; exit 4; }
  TMP="${NODALS_REGRESSION_ROOT:-$HOME/Downloads/NodalS_v1.00_compact_parity_$(date +%Y%m%d_%H%M%S)}"
  REF="$TMP/reference"; NEW="$TMP/modular"; mkdir -p "$REF" "$NEW"
  CASE="cases/reference-111k-compact-cheb.case"
  python3 scripts/nodals_case.py "$CASE" --exe "$ROOT/nodals_reference_solver" --set "solver.mesh=$MESH" --set "run.output_root=$REF"
  python3 scripts/nodals_case.py "$CASE" --exe "$ROOT/nodals_solver" --set "solver.mesh=$MESH" --set "run.output_root=$NEW"
  python3 tests/compare_regression_logs.py "$REF/111k_Re20_parabolic.log" "$NEW/111k_Re20_parabolic.log"
  echo "NODALS_REGRESSION status=PASS_COMPACT_PARITY root=$TMP"
  exit 0
fi

echo "usage: $0 [--fast|--compact-parity]" >&2
exit 2
