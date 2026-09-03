# Firedrake Q1+BF2/Q0 VMFL003 benchmark

A compact Firedrake implementation of a segregated Q1+BF2/Q0 hexahedral SIMPLEC solver for the VMFL003 turbulent pipe benchmark.

The repository is intentionally small: it contains only the element, operators, turbulence/wall model, benchmark driver, mesh generator, and one smoke test.

## Discretization

Velocity uses 14 scalar modes per hex:

- 8 trilinear Q1 vertex functions;
- 6 shared face bubbles.

The face bubble on a face normal to `n` is

```text
Q1_endpoint(n) * Q2_midpoint(t1) * Q2_midpoint(t2)
```

where `t1,t2` are the two tangential coordinates.

This distinction matters. An ordinary tensor-Q2 face-centre nodal function is quadratic in the normal direction as well and is **not** this BF2 element.

Useful fingerprint:

```text
each BF2 mode at the hex centre = 0.5
```

while the ordinary tensor-Q2 face-node function is zero there.

Pressure is discontinuous Q0, one pressure unknown per hex.

## Physics

- steady incompressible SIMPLEC;
- central implicit convection with one-outer-iteration lagged advecting field;
- Nikuradse algebraic mixing length;
- Spalding weak wall law using an off-wall midpoint sample;
- DG prescribed inlet numerical trace;
- penalty-free nonsymmetric inlet Nitsche term using `nu + nu_t`;
- SUPG disabled for this benchmark.

The transverse velocity components are clamped to zero on the wall while the axial component remains a weak-wall unknown.

## SIMPLEC details

The production benchmark uses

```text
alpha_u = 0.7
alpha_p = 0.3
rAU scale = 2
```

and the pressure-only correction ordering used by this implementation: pressure is updated after the momentum predictor; there is no explicit `U += rAU B^T p'` correction.

Default linear solvers:

- momentum: FGMRES + MPI block Jacobi, local ILU(0), relative drop 0.5;
- pressure: CG + GAMG, Chebyshev/Jacobi levels, relative tolerance 0.5.

## Benchmark mesh

- radius: 0.002 m
- diameter: 0.004 m
- length: 2 m = 500D
- 192 quads per cross-section
- first 20D: 20 axial layers at 1D
- remaining 480D: 48 axial layers at 10D
- total: 13,056 hexes
- Q1+BF2 scalar velocity DOFs: 55,965
- Q0 pressure DOFs: 13,056

Fluid:

```text
rho = 1.225 kg/m^3
nu  = 1.4607346938775508e-5 m^2/s
U   = 50 m/s
Re  = 13691.7402481
```

## Run

Activate a Firedrake environment, then:

```bash
NP=16 bash run_benchmark.sh
```

Or point the runner at a venv:

```bash
FIREDRAKE_VENV="$HOME/firedrake-install/venv-firedrake" \
NP=16 \
bash run_benchmark.sh
```

The runner first checks the exact BF2 basis in serial, then launches the benchmark with MPI.

Expected smoke fingerprint:

```text
BF2_SMOKE_STATUS=PASS
localScalar=14
globalScalar=55965
pressure=13056
cellCenterFaceValues=[0.500000,0.500000,0.500000,0.500000,0.500000,0.500000]
```

## Verified regression result

A clean 16-rank rerun produced:

```text
SIMPLE iterations     469
developed f_D         0.0281961402   (300D--450D fit)
full 500D delta-p     21.745242 kPa
VMFL003 target error  +0.005712 %
```

The earlier verified implementation gave 458 iterations, developed `f_D=0.0281951649`, and full `delta-p=21.744872 kPa`; the clean refactor differs only at the expected level for a `1e-4` nonlinear stopping threshold with iterative MPI solves.

The public regression gate requires the full pressure drop to remain within 0.5% of the VMFL003 target.

## Output

```text
logs/vmfl003.log
output/vmfl003/axial_pressure_planes.csv
output/vmfl003/friction_zones_50D.csv
output/vmfl003/benchmark_summary.txt
```

## Tested environment

The development run used Firedrake 2025.10.2 with PETSc/petsc4py 3.24.0 and Gmsh 4.15.0. Firedrake is normally installed through its supported installer; this example does not provide a pip requirements file for Firedrake.

## Source layout

```text
nodals_firedrake/config.py      benchmark constants and solver defaults
nodals_firedrake/element.py     exact custom Q1+BF2 element
nodals_firedrake/operators.py   B operators, SIMPLEC algebra and PETSc KSP setup
nodals_firedrake/physics.py     mixing length, Spalding wall operator, momentum assembly
nodals_firedrake/benchmark.py   VMFL003 iteration loop and post-processing
tools/smoke_bf2.py              element/DOF/basis fingerprint
run_vmfl003.py                  thin Python entry point
run_benchmark.sh                mesh generation + smoke + MPI benchmark
```
