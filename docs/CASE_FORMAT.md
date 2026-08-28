# NodalS `.case` format — v1.00

A case file is an INI-style text file. The numerical executable is unchanged;
`nodals` translates the case into the existing validated PETSc options.

Precedence is:

1. solver compiled defaults,
2. `.case` values,
3. explicit PETSc arguments written after `--` on the `nodals` command line.

Example override:

```bash
nodals cases/test.case -- -simple_rtol 1e-5
```

The `[petsc]` section is a direct escape hatch: `foo = bar` becomes `-foo bar`, so
new/expert PETSc/source options remain tunable without changing the case parser.

## Boundary conditions intentionally supported in v1.00

- wall: existing no-slip wall,
- inlet: existing fixed normal-speed inlet,
- inlet: existing special parabolic pipe inlet,
- outlet: existing open/natural-zero-traction pressure-gauge path, exposed in the
  case file as `outlet_type = pressure0`.

Default patch convention is wall `patch_0_0`, inlet `patch_2_0`, outlet
`patch_1_0`. For a normal inlet, `patch_1_0` and `patch_2_0` may be swapped by
editing the case. A new industrial mesh should still be audited first.

The special parabolic Hagen–Poiseuille case intentionally keeps the validated
+z exact-diagnostic convention: inlet `patch_2_0`, outlet `patch_1_0`.

For `inlet_type = normal`, `speed` is a **positive inward magnitude**. The runner
converts it to the frozen solver's signed average-outward-normal convention.

## Pressure backends (v1.00)

NodalS exposes two production pressure architectures directly in the case file.

### Fast full GAMG (primary/default)

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

This selects the exact factored physical Schur operator together with a lagged full explicit Schur Pmat and PETSc GAMG-preconditioned Richardson solve. It corresponds to the historical fast pressure architecture.

### Compact low-memory Chebyshev

```ini
[pressure]
mode = compact_cheb
pmat = compact
refresh = 1
rtol = 0.1

[chebyshev]
mode = adaptive
power_iterations = 10
spectrum_refresh = 100
```

This keeps the exact factored physical Schur action but uses the compact FE-face-energy Pmat, PETSc GAMG only as the preconditioner, and the custom FP64 Chebyshev iteration.

Use `./bin/nodals CASE --dry-run` to verify which backend and PETSc options a case will launch.
