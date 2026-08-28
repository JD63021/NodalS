# Checkpoint 2 — `.case` interface and preserved BC family

- Added `scripts/nodals_case.py` / `bin/nodals`.
- Added canonical `cases/test.case` reproducing Gate9N 111k Re=20 parabolic pipe options.
- Added `cases/generic-normal.case` for the existing generic normal-speed inlet path.
- No C++ numerical source was changed from Checkpoint 1.
- BC scope remains deliberately limited to wall, normal inlet, special parabolic pipe inlet and existing pressure-0/open outlet semantics.
- `[petsc]` passes expert options directly through; trailing CLI options after `--` override the case.
- `tests/test_case_translation.py` validates canonical translation and BC guards without requiring PETSc.
