#!/usr/bin/env bash
set -euo pipefail
EXE="${1:-./nodals_gpu_h4_fp32}"
[[ -x "$EXE" ]] || { echo "H4_DEPENDENCY_AUDIT status=FAIL reason=exe_missing exe=$EXE"; exit 2; }
BAD="$(ldd "$EXE" 2>/dev/null | grep -Ei 'petsc|mpi|hypre' || true)"
if [[ -n "$BAD" ]]; then echo "$BAD"; echo "H4_DEPENDENCY_AUDIT status=FAIL petsc_or_mpi=FOUND"; exit 3; fi
echo "H4_DEPENDENCY_AUDIT status=PASS petsc=NONE mpi=NONE exe=$EXE"
