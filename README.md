# NodalS

**NodalS** is a finite-element **SIMPLE/SIMPLEC solver for steady incompressible fluid flow on tetrahedral meshes**.

The project is an attempt to make practical CFD workflows possible in a finite-element formulation while retaining the familiar segregated pressure–velocity solution strategy used by SIMPLE-family finite-volume solvers. The current code is a research/development solver rather than a general-purpose CFD package, but it already supports steady viscous flow, central convection, optional SUPG stabilization, MPI execution, and two pressure-solver architectures.

## What NodalS discretizes

The current mixed finite-element pair is inspired from Farrell et al,2019 (https://doi.org/10.1137/18M1219370).

\[
V_h = [P_1 + BF_3]^3, Q_h = P_0.
\]

For each tetrahedron, each scalar velocity component contains the four standard continuous linear \(P_1\) vertex basis functions together with four cubic face-bubble \(BF_3\) functions. Pressure is discontinuous piecewise constant \(P_0\), with one pressure degree of freedom per cell.

The face-bubble enrichment gives the velocity space additional divergence-coupling freedom while keeping the pressure field cell based. In three dimensions the velocity is vector valued, so the mixed problem retains the usual incompressible saddle-point structure.

At the algebraic level, the linearized incompressible problem has the familiar form

(A_u B^T)(u) = (f)
(B   0  )(p)   (0)

where \(A_u\) is the momentum operator and \(B\) is the discrete divergence operator.

## Pressure is part of the mixed saddle-point problem

NodalS does **not** make the pressure field usable by adding an artificial pressure-diffusion term to the physical equations. Pressure is coupled through the mixed FEM divergence/gradient blocks.

In the SIMPLE/SIMPLEC pressure-correction step, the current physical pressure action is the factored Schur-type operator

\[
S = B\,\mathrm{diag}(rAU)\,B^T.
\]

The matrix used by PETSc GAMG may be a full or compact **preconditioning matrix**, but that Pmat is not substituted for the physical pressure operator in the compact/factored formulation. In particular, the compact FE face-energy matrix is a preconditioner surrogate rather than an artificial pressure-dissipation term added to the governing equations.

This distinction is important: NodalS uses pressure preconditioning to accelerate a saddle-point pressure correction; it does not rely on a Rhie–Chow-style pressure filter to define the pressure physics.

## SIMPLE and SIMPLEC

The nonlinear/segregated outer algorithm is selectable between **SIMPLE** and **SIMPLEC**.

A typical outer iteration is:

1. assemble/update the momentum operator using the current velocity field,
2. solve the three scalar momentum predictors,
3. form the current momentum diagonal/relaxed diagonal and \(rAU\),
4. solve the pressure-correction equation,
5. correct pressure and velocity,
6. repeat until the outer residual gate is satisfied.

The current convergence gate checks the **initial momentum residuals of all three velocity components plus the continuity-equivalent pressure residual**. All four must satisfy the requested `simple.rtol`.

## Momentum solve

The production momentum path is a custom distributed scalar CSR operator for each velocity component with a native SGS/SOR-style iterative solve. The solver stores and updates the physical momentum diagonal because that information is also required by SIMPLE/SIMPLEC and by the pressure Schur operator.

For many current validation cases the momentum predictor requires only a small number of inner iterations. Under-relaxation, inner tolerance, relative residual drop, local sweep count, and SOR relaxation are exposed in the `.case` file.

## Pressure solve

NodalS v1.00 exposes two main pressure architectures.

### Fast full-GAMG backend

```ini
[pressure]
mode = fast_full_gamg
pmat = full
refresh = 100
ksp_type = richardson
rtol = 0.5
pc_type = gamg
```

The physical pressure operator remains the factored \(B\,\mathrm{diag}(rAU)\,B^T\) action. A full explicit Schur snapshot is used as the PETSc Pmat, and PETSc GAMG is used through a Richardson pressure solve.

`refresh` controls how often the full pressure Pmat/GAMG hierarchy is rebuilt. A value of `1` refreshes every outer iteration; larger values reuse the hierarchy and can be substantially faster when the pressure preconditioner evolves slowly.

### Compact low-memory backend

```ini
[pressure]
mode = compact_cheb
pmat = compact
refresh = 1
```

The same physical factored Schur action is retained, while the preconditioning matrix is replaced by a compact cell-neighbour FE face-energy surrogate. PETSc GAMG supplies the preconditioner application and a custom FP64 Chebyshev iteration advances the physical pressure correction.

This route reduces pressure-preconditioner storage at the cost of more pressure work in the current implementation.

A third `compact_pcg` mode is retained as a lower-level/legacy alternative.

## SUPG

For convection-dominated cases NodalS can add **streamline-upwind/Petrov–Galerkin (SUPG)** stabilization to the momentum equations.

SUPG is optional and is controlled independently from the pressure formulation. Current controls include the stabilization scale, implicit/explicit form, implementation kernel, and tetrahedral quadrature rule. The accepted high-Re development path has generally used the implicit fast kernel.

SUPG stabilizes the advective momentum operator; it is not a pressure-diffusion device.

## Boundary conditions currently exposed

NodalS v1.00 intentionally exposes a small BC set:

- no-slip wall,
- fixed normal-speed inlet,
- pressure-zero/open outlet using the existing natural-zero-traction/physical pressure-gauge treatment.

The generic inlet uses the average outward normal of the selected patch and interprets the case-file `speed` as a **positive inward magnitude**.

There is also a **special hard-coded Hagen–Poiseuille verification path** with a normalized parabolic inlet and Reynolds-number convenience input. That path is retained for verification and regression testing; it is not the generic boundary-condition interface.

For a generic `problem = flow` case, the currently exposed material property is the **kinematic viscosity `nu`**. Density, compressibility, thermal properties, turbulence models, and general material models are not yet part of the v1.00 case interface.

## Mesh format

The current mesh reader consumes tetrahedral meshes stored in the OpenFOAM `constant/polyMesh` format.

A typical case points directly to the `polyMesh` directory:

```ini
[solver]
mesh = ~/NodalS v1.00/meshes/shellsphere1/82ktet/constant/polyMesh
problem = flow
nu = 0.001
```

Patch roles are assigned in the `.case` file. New meshes should be checked with the built-in mesh/patch audit before assuming inlet/outlet orientation.

## MPI and parallel execution

NodalS uses **MPI** and PETSc.

The launcher controls rank count and Open MPI binding:

```ini
[run]
ranks = 16
map_by = core
bind_to = core
```

The current code uses a geometric RCB-style ownership partition for cells. Momentum rows are owned by MPI ranks and required off-rank velocity values are exchanged through explicit ghost/halo plans. Pressure PETSc vectors/matrices and GAMG operations are distributed over MPI.

The full-GAMG pressure path also closes the halo required by full-Schur `rAU` dependencies.

An important current limitation is that mesh loading/topology is still substantially **replicated per MPI rank** before/around the distributed solver structures. Reducing replicated topology and setup memory is future work; NodalS should not yet be described as a fully memory-distributed mesh implementation.

## Requirements and PETSc

NodalS is currently developed and tested on Linux with:

- a C++17 compiler,
- MPI (Open MPI or another PETSc-compatible MPI implementation),
- Python 3 for the `.case` launcher and regression scripts,
- PETSc 3.25.x,
- tetrahedral meshes in OpenFOAM `constant/polyMesh` format.

The current v1.00 development/validation workstation has used a PETSc 3.25.4
development snapshot (`v3.25.4-499-g9183a15b9d2`) with Open MPI and GNU
compilers. The recommended public baseline is **PETSc 3.25.4 / PETSc 3.25.x**.
Other PETSc versions may work, but they have not yet been declared part of the
v1.00 tested compatibility range.

NodalS does not currently require CUDA for its production solver path. A
CUDA-enabled PETSc build can be used (and was used during development), but the
present v1.00 MPI momentum and pressure paths should be regarded as CPU/MPI
paths. A CPU-optimized PETSc build is therefore the simplest recommended
configuration for new users.

### Recommended optimized PETSc build

PETSc defaults to a debugging build. For timing and production runs, create a
separate optimized `PETSC_ARCH` rather than replacing your debug installation.

From an existing PETSc source tree:

```bash
cd "$HOME/petsc-main"

export PETSC_DIR="$PWD"
export PETSC_ARCH=arch-linux-opt

./configure \
  PETSC_ARCH="$PETSC_ARCH" \
  --with-debugging=0 \
  --with-shared-libraries=1 \
  --with-mpi=1 \
  --with-fortran-bindings=0 \
  --download-f2cblaslapack=1 \
  --download-metis=1 \
  --download-parmetis=1 \
  COPTFLAGS="-O3 -march=native -mtune=native" \
  CXXOPTFLAGS="-O3 -march=native -mtune=native"

make PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH" all
make PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH" check
```

`-march=native -mtune=native` targets the machine on which PETSc is compiled.
If the resulting PETSc installation will be copied to machines with different
CPUs, omit those two flags and retain `-O3`.

METIS/ParMETIS are included in the recommended configuration because they are
useful PETSc/MPI graph-partitioning dependencies and match the development
environment. The current NodalS fast pressure path uses PETSc GAMG rather than
requiring HYPRE. HYPRE may be added to PETSc independently if desired.

### Optional CUDA-enabled PETSc

CUDA is **not required** by NodalS v1.00, but an existing CUDA-enabled PETSc
installation can also be used. PETSc's normal CUDA configure option is
`--with-cuda`.

For example, an optimized CUDA-enabled arch can be configured by adding:

```bash
--with-cuda
```

to the optimized PETSc configuration above, together with any site-specific
CUDA path/compiler settings required by the local PETSc installation.

### Debug PETSc build

For development, debugging, and PETSc diagnostic checks, use a separate debug
arch such as:

```bash
export PETSC_DIR="$HOME/petsc-main"
export PETSC_ARCH=arch-cuda-debug
```

The original v1.00 workstation validation has used such a debug PETSc arch.
It is useful for development but should **not** be used when publishing timing
or performance results.

## Building NodalS

The preferred build command is the project wrapper:

```bash
cd "$HOME/NodalS v1.00"

export PETSC_DIR="$HOME/petsc-main"
export PETSC_ARCH=arch-linux-opt

./build.sh
```

`build.sh`:

1. discovers and validates `PETSC_DIR` / `PETSC_ARCH`,
2. runs the frozen-reference and case-translation checks,
3. compiles and links `nodals_solver` against the selected PETSc arch.

For a development/debug build, simply select the debug PETSc arch instead:

```bash
export PETSC_DIR="$HOME/petsc-main"
export PETSC_ARCH=arch-cuda-debug

./build.sh
```

The direct Makefile route is also available:

```bash
make clean
make -j"$(nproc)" nodals_solver
```

but `./build.sh` is preferred because it also performs the NodalS pre-build
consistency checks.

The generated executable is:

```text
./nodals_solver
```

This is the low-level PETSc executable. Normal users should **not** pass `.case`
files directly to `nodals_solver`; use the case launcher instead:

```bash
./bin/nodals cases/test.case
```

The launcher reads the `.case`, constructs the MPI/PETSc command line, selects
the requested number of MPI ranks, and creates the requested output directory.

## Running a case

Dry-run first to inspect the generated MPI/PETSc command:

```bash
./bin/nodals cases/test.case --dry-run
```

Run:

```bash
./bin/nodals cases/test.case
```

Temporarily override a case value without editing the file:

```bash
./bin/nodals cases/test.case --set run.ranks=8
./bin/nodals cases/test.case --set pressure.refresh=1
```

Raw PETSc/source options may be appended after `--`:

```bash
./bin/nodals cases/test.case -- -ksp_view
```

The `[petsc]` section in a `.case` file provides the same kind of expert escape hatch.

## The annotated `test.case`

`cases/test.case` is intended to be the reference generic-flow template. It contains every normal case-parser control currently exposed by v1.00 and documents the remaining NodalS source-level switches under `[petsc]`.

The generic template intentionally does **not** activate the special pipe-only Reynolds-number/parabolic-inlet controls. Those are documented in comments and should remain in dedicated pipe verification cases.

## Current scope

NodalS v1.00 currently targets:

- steady incompressible flow,
- tetrahedral meshes,
- \([P_1+BF_3]^3/P_0\) mixed FEM,
- SIMPLE/SIMPLEC pressure–velocity segregation,
- central convection,
- optional SUPG momentum stabilization,
- MPI execution,
- PETSc GAMG-based pressure preconditioning.

Planned directions can include broader boundary conditions, heat transfer, turbulence modelling, additional material properties, lower-memory distributed mesh storage, and further solver/preconditioner development.

## Repository layout

```text
NodalS v1.00/
├── app/            # translation-unit entry point
├── bin/            # case launcher
├── cases/          # user-editable .case files
├── docs/           # case-format and design notes
├── meshes/         # optional bundled validation meshes
├── reference/      # frozen historical/reference source and notes
├── scripts/        # case parser and environment helpers
├── src/            # modularized solver source
├── tests/          # regression/static checks
├── build.sh
├── install.sh
├── Makefile
├── README.md
└── LICENSE
```

## Status

This is a research CFD codebase under active development. Validation cases and regression scripts are included, but users should independently verify numerical accuracy, boundary conditions, solver tolerances, and mesh suitability for their applications.

## License

NodalS is released under the **Apache License 2.0**. See [`LICENSE`](LICENSE).

Copyright 2026 NodalS contributors.
