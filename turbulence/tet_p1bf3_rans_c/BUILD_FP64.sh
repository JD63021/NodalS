#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PETSC_DIR="${PETSC_DIR:-$HOME/src/petsc}"
export PETSC_ARCH="${PETSC_ARCH:-arch-linux-cuda-opt}"
cd "$HERE"
[[ -f Makefile ]] || { echo "ERROR: Makefile missing; branch installer should copy reference/Makefile.original here"; exit 2; }
make clean || true
make -j"${JOBS:-$(nproc)}" p1bf3_simple_foam_mpi
cp -f p1bf3_simple_foam_mpi p1bf3_simple_foam_mpi_fp64
sha256sum p1bf3_simple_foam_mpi.cpp p1bf3_simple_foam_mpi_fp64
echo "TET_RANS_C_BUILD status=PASS"
