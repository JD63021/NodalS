#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# shellcheck source=scripts/petsc_env.sh
source "$ROOT/scripts/petsc_env.sh"
JOBS="${JOBS:-$(nproc)}"

if [[ ! -f "$PETSC_DIR/lib/petsc/conf/variables" ]]; then
  echo "NODALS_BUILD status=FAIL reason=missing_petsc PETSC_DIR=$PETSC_DIR" >&2
  exit 2
fi

python3 tests/check_source_freeze.py
python3 tests/test_case_translation.py

echo "NODALS_BUILD PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH jobs=$JOBS"
if [[ -x "$PETSC_DIR/$PETSC_ARCH/bin/petsc-config" ]]; then
  "$PETSC_DIR/$PETSC_ARCH/bin/petsc-config" --configure-options 2>/dev/null || true
fi
make -j"$JOBS" nodals_solver
[[ -x ./nodals_solver ]] || { echo "NODALS_BUILD status=FAIL reason=missing_binary"; exit 3; }
echo "NODALS_BUILD status=PASS executable=$ROOT/nodals_solver"
