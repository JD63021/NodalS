#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Optional convenience: FIREDRAKE_VENV=/path/to/venv.  If omitted, use the
# currently active Firedrake environment.
if [[ -n "${FIREDRAKE_VENV:-}" ]]; then
  source "$FIREDRAKE_VENV/bin/activate"
fi

command -v python3 >/dev/null || { echo "python3 not found"; exit 2; }
command -v gmsh >/dev/null || { echo "gmsh not found"; exit 2; }
command -v mpiexec >/dev/null || { echo "mpiexec not found"; exit 2; }

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

NP="${NP:-16}"
export SIMPLE_RTOL="${SIMPLE_RTOL:-1e-4}"
export SIMPLE_MAX_ITS="${SIMPLE_MAX_ITS:-10000}"
export M_RTOL="${M_RTOL:-0.5}"
export M_PETSC_ATOL="${M_PETSC_ATOL:-1e-50}"
export M_MAX_ITS="${M_MAX_ITS:-20000}"
export P_RTOL="${P_RTOL:-0.5}"
export P_PETSC_ATOL="${P_PETSC_ATOL:-1e-50}"
export P_MAX_ITS="${P_MAX_ITS:-300}"
export P_MG_LEVEL_ITS="${P_MG_LEVEL_ITS:-6}"

mkdir -p mesh logs output/vmfl003

echo "=== Generate 192-quad VMFL003 cross-section ==="
gmsh mesh/pipe_ogrid_firedrake.geo \
  -2 -format msh2 \
  -setnumber R 0.002 \
  -setnumber CoreRatio 0.50 \
  -setnumber Nq 12 \
  -setnumber Nr 1 \
  -o mesh/pipe_ogrid_firedrake.msh \
  2>&1 | tee logs/gmsh.log

echo
echo "=== Exact BF2 smoke test (serial) ==="
python3 tools/smoke_bf2.py

echo
echo "=== VMFL003 benchmark (${NP} MPI ranks) ==="
mpiexec -n "$NP" python3 run_vmfl003.py \
  2>&1 | tee logs/vmfl003.log
