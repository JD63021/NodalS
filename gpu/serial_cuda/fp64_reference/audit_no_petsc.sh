#!/usr/bin/env bash
set -euo pipefail
EXE="${1:?exe}"
L="$(ldd "$EXE" 2>/dev/null || true)"
if grep -Eqi 'petsc|libmpi|open-rte|open-pal' <<<"$L"; then
  echo "G5D_DEPENDENCY_AUDIT status=FAIL"
  echo "$L"
  exit 1
fi
echo "G5D_DEPENDENCY_AUDIT status=PASS petsc=NONE mpi=NONE exe=$EXE"
