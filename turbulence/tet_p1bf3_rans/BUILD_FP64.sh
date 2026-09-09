#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PETSC_DIR="${PETSC_DIR:-$HOME/src/petsc}"
export PETSC_ARCH="${PETSC_ARCH:-arch-linux-cuda-opt}"
cd "$HERE"

python3 tests/check_turbulence_source_freeze.py
make clean || true
make -j"${JOBS:-$(nproc)}"

sha256sum reference/p1bf3_rans_effectiveB_oracle.cpp \
          src/75_pressure_amg/custom_pressure_amg.inc \
          nodals_turbulence_cpu_fp64
echo "NODALS_TURBULENCE_BUILD status=PASS executable=$HERE/nodals_turbulence_cpu_fp64"
