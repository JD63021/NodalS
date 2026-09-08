# NodalS `.case` format — CPU/MPI v1.00

A CPU case file is an INI-style text file. `nodals` translates the friendly
case sections into the existing PETSc/source options and launches the MPI CPU
solver.

Precedence is:

1. solver compiled defaults,
2. `.case` values,
3. explicit PETSc/source arguments written after `--` on the `nodals` command line.

Example:

```bash
nodals cases/test.case -- -simple_rtol 1e-5
```

The `[petsc]` section remains the expert escape hatch: `foo = bar` becomes
`-foo bar`.

For the native CPU custom AMG, the reusable fully annotated example is:

```text
cases/cpu-custom-amg-annotated.case
```

For the independent H8 serial-CUDA case format, use:

```text
cases/gpu-h8-annotated.case
docs/GPU_CASE_FORMAT.md
bin/nodals-gpu
```

## Boundary conditions intentionally supported by the CPU interface

- wall: existing no-slip wall,
- inlet: existing fixed normal-speed inlet,
- inlet: existing special parabolic pipe inlet,
- outlet: existing open/natural-zero-traction pressure-gauge path, exposed as
  `outlet_type = pressure0`.

Default patch convention is wall `patch_0_0`, inlet `patch_2_0`, outlet
`patch_1_0`. For a normal inlet, inlet/outlet patch roles may be swapped in the
case. The special Hagen–Poiseuille parabolic verification path retains the
validated +z patch convention.

For `inlet_type = normal`, `speed` is a positive inward magnitude; the runner
converts it to the solver's average-outward-normal sign convention.

## CPU pressure backends

### Native custom AMG — unsmoothed aggregation

The native low-memory CPU hierarchy can use unsmoothed aggregate injection.
Select one of:

```ini
[pressure]
mode = pcg_unsmoothed
```

or the convenience fixed-smoother presets:

```text
pcg_unsmoothed_sgs
pcg_unsmoothed_jacobi
pcg_unsmoothed_chebyshev
```

With `mode = pcg_unsmoothed`, select the V-cycle smoother independently:

```ini
[pressure_amg]
smoother = sgs          # or jacobi or chebyshev
```

The outer pressure solver is native FP64 PCG. The hierarchy uses connected
rank-local aggregation; CPU PMIS is not implemented.

### Native custom AMG — smoothed aggregation

Use:

```ini
[pressure]
mode = pcg_smoothed

[pressure_amg]
smoother = chebyshev
```

for native FP64 PCG with smoothed aggregation, or:

```ini
[pressure]
mode = richardson_smoothed

[pressure_amg]
smoother = chebyshev
```

for the same smoothed hierarchy with native FP64 Richardson outside it.

The current smoothed-transfer construction requires the Chebyshev/spectral
path. SGS/Jacobi are valid unsmoothed-hierarchy smoothers, but are rejected for
smoothed aggregation.

### Native custom-AMG controls

`[pressure_amg]` exposes:

- `target_aggregate`
- `min_aggregate`
- `soft_max_aggregate`
- `smoother = sgs|jacobi|chebyshev`
- `chebyshev_degree`
- `power_iterations`
- `lambda_safety`
- `lambda_low_fraction`
- `coarse_target_rows`
- `interpolation_max_row_nnz`
- `sa_damping`
- `richardson_omega`

See `cases/cpu-custom-amg-annotated.case` for comments explaining which knobs
apply to unsmoothed transfer, smoothed transfer, Chebyshev smoothing, and the
Richardson outer solver.

### Fast full GAMG

```ini
[pressure]
mode = fast_full_gamg
pmat = full
refresh = 100
rtol = 0.5
atol = 1e-12
max_iterations = 20
pc_type = gamg
mg_level_ksp = richardson
mg_level_pc = jacobi
mg_level_iterations = 1
```

This retains the exact factored physical Schur action together with a lagged
full explicit Schur Pmat and PETSc GAMG-preconditioned Richardson solve.

### Full Pmat + native PCG

```ini
[pressure]
mode = full_pcg_gamg
pmat = full
```

This uses the full explicit Schur Pmat/PETSc GAMG as a preconditioner while the
outer pressure iteration is native FP64 PCG.

### Compact low-memory Chebyshev

```ini
[pressure]
mode = compact_cheb
pmat = compact
refresh = 1

[chebyshev]
mode = adaptive
power_iterations = 10
spectrum_refresh = 100
```

This keeps the exact factored physical Schur action, uses the compact FE
face-energy Pmat, PETSc GAMG as preconditioner, and the legacy/custom FP64
Chebyshev outer pressure iteration.

`compact_pcg` remains available as the corresponding legacy native-PCG compact
route.

## CPU SUPG

CPU SUPG is controlled through:

```ini
[supg]
enabled = false
tau_scale = 0.05
form = implicit
kernel = fast
quad_points = 64
```

GPU SUPG is not yet implemented and therefore is not advertised by the GPU case
format.

## Verification

Dry-run a CPU case to inspect the generated command:

```bash
./bin/nodals CASE --dry-run
```

Dry-run a CUDA case similarly:

```bash
./bin/nodals-gpu CASE --dry-run
```
