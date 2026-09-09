# NodalS tetrahedral turbulence skeleton

This directory is the promoted modular form of the validated 2026-09-09
P1+BF3/P0 steady RANS pipe solver.

## Current validated physics

- tetrahedral P1+BF3/P0
- steady SIMPLEC
- central convection
- DG numerical-trace plug inlet
- natural pressure/open outlet
- weak Spalding wall function
- Nikuradse mixing-length eddy viscosity
- native FP64 momentum state and processor-block/local-SGS predictor
- native FP64 PCG pressure correction
- canonical compact effective-B plan:
  DG inlet trace + weak-wall z-only pressure coupling are interpreted once
- custom pressure AMG:
  - unsmoothed connected aggregation + processor-block SGS
  - smoothed aggregation + symmetric Chebyshev/Jacobi
- PETSc full-Schur GAMG path retained as fallback/oracle

The production build remains one C++ translation unit assembled from ordered
`.inc` fragments, matching the normal NodalS modular architecture.  The
pre-modular golden monolith is retained only under `reference/` as a numerical
oracle and is not built.

## Build

    ./BUILD_FP64.sh

## Run via .case

    python3 ../../scripts/nodals_turbulence_case.py \
      ../../cases/turbulence-vmfl003-10d-unsmoothed.case

or

    python3 ../../scripts/nodals_turbulence_case.py \
      ../../cases/turbulence-vmfl003-10d-smoothed.case

## Regression

    python3 tests/check_turbulence_source_freeze.py

After a run:

    python3 tests/check_vmfl003_regression.py /path/to/log \
      --mode unsmoothed_sgs

The next architectural extensions should preserve the existing seams:

1. keep boundary semantics inside the canonical effective-B layer;
2. make turbulence models produce lagged `nu_t` through a common model API;
3. let CPU/GPU backends consume the same model/boundary state;
4. add k-omega SST without rewriting SIMPLE/pressure coupling.
