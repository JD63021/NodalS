# Checkpoint 3 — build, install and parity regression

- `Makefile` builds `nodals_solver` with the latest documented PETSc arch default: `arch-cuda-debug`.
- `build.sh` records the PETSc configuration and runs static checkpoint tests before compiling.
- `install.sh` installs once under `${PREFIX:-$HOME/.local}`:
  - executable: `libexec/nodals/nodals_solver`
  - command: `bin/nodals`
  - example cases/docs: `share/nodals/`
- `tests/regression.sh` performs static checks.
- `tests/regression.sh --pipe` additionally builds the frozen monolithic reference with the same flags, runs both on the canonical 111k parabolic pipe case, and compares stable physics/convergence records.

### v1.00 Fix1 PETSc arch correction
The frozen uploaded Gate9N `Makefile` uses `PETSC_ARCH ?= arch-cuda-debug`; this is authoritative for the v1.00 build default. `scripts/petsc_env.sh` now honors an explicitly valid `PETSC_ARCH`, otherwise prefers the frozen default, otherwise discovers configured PETSc arches containing `lib/petsc/conf/petscvariables`.
