# Gate 9N — parabolic pipe inlet + convergence runner

This package exposes the existing validated `-problem pipe` implementation in the Gate 9M source.

## Parabolic inlet

Use `INLET_PROFILE=parabolic`. For the known `$HOME/Desktop/meshes/pipe/*` family the exact Hagen–Poiseuille path is oriented +z, therefore:

- wall = `patch_0_0`
- inlet = `patch_2_0` (z-min)
- outlet = `patch_1_0` (z-max)

The source evaluates the quadratic parabolic trace on each inlet triangle and rescales it so the discrete face-integrated inlet flux equals `BULK_VELOCITY * inletArea` exactly. It then reports exact pipe velocity and pressure errors.

Prefer specifying `RE` and `BULK_VELOCITY` and leaving `NU` blank. The code detects the actual mesh diameter and sets `nu = Ubulk*D/Re`. `NU` remains available as an explicit override.

## One case

```bash
MESH_NAME=111k \
RE=20 \
BULK_VELOCITY=1 \
INLET_PROFILE=parabolic \
PRESSURE_SOLVER=cheb_gamg \
PMAT=compact \
P_REFRESH=1 \
SPECTRUM_REFRESH=100 \
POWER_ITS=10 \
SIMPLE_VARIANT=simplec \
ALPHA_U=0.7 ALPHA_P=1 \
SIMPLE_RTOL=1e-3 \
SUPG=0 \
bash ./RUN_GATE9N_PIPE.sh
```

For Re=2000, if desired:

```bash
MESH_NAME=111k RE=2000 BULK_VELOCITY=1 INLET_PROFILE=parabolic \
P_REFRESH=1 SPECTRUM_REFRESH=100 POWER_ITS=10 \
SUPG=1 TAU_SCALE=0.05 \
bash ./RUN_GATE9N_PIPE.sh
```

## Pressure choices

- `PRESSURE_SOLVER=cheb_gamg` or `pcg_gamg`
- `PMAT=compact` or `full`
- `P_REFRESH`, `SPECTRUM_REFRESH`, `POWER_ITS`, `P_RTOL`, `P_ATOL`, `CHEB_*` all remain exposed.

## SIMPLE controls

`SIMPLE_VARIANT=simple|simplec`, `ALPHA_U`, `ALPHA_P`, `SIMPLE_RTOL`, `SIMPLE_MAX_ITS`, `U_RTOL`, `U_ATOL`, `U_REL_DROP`, etc. are exposed.

## Mesh sweep and convergence orders

Example, Re=20 on four meshes:

```bash
MESHES="40k 111k 292k 550k" RE=20 SUPG=0 bash ./RUN_PIPE_MESH_SWEEP.sh
```

High-Re example:

```bash
MESHES="40k 111k 292k 550k" RE=2000 SUPG=1 TAU_SCALE=0.05 \
P_REFRESH=1 SPECTRUM_REFRESH=100 bash ./RUN_PIPE_MESH_SWEEP.sh
```

Or run several Re values in one sweep:

```bash
MESHES="40k 111k 292k" RE_LIST="20 200 2000" SUPG=1 bash ./RUN_PIPE_MESH_SWEEP.sh
```

`ANALYZE_PIPE_CONVERGENCE.py` extracts `U_L2`, `P_shifted_L2`, fitted pressure-drop relative error, effective h=(V/N)^(1/3), pairwise orders, mass balance, outer iterations and timing.

For a clean spatial-order study, keep the physical and solver controls identical across meshes and tighten `SIMPLE_RTOL` enough that iteration error is comfortably below discretization error.

## Small Gate 9N source fix

Gate 9M printed `CHEB_REQUIRE_TARGET=0` but adaptive Chebyshev still aborted on a missed inner target because the acceptance condition applied the option only to fixed mode. Gate 9N changes that single condition so `CHEB_REQUIRE_TARGET=0` means the same thing in adaptive and fixed modes: keep a finite correction and allow SIMPLE to continue.
