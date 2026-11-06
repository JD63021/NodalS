# NodalS

Research software for steady incompressible finite-element CFD. This repository
contains separately maintained CPU laminar, single-GPU laminar, algebraic
mixing-length RANS, and CUDA k-omega SST implementations. Mesh support and
boundary conditions differ between modules; do not assume arbitrary geometry
or mixed-element support.

## Solvers

| Solver | Path | Build / entry point |
| --- | --- | --- |
| CPU P1+BF3/P0 laminar | `src/`, `app/` | `./build.sh` (PETSc + MPI required) |
| CUDA laminar | `gpu/serial_cuda/` | `make -C gpu/serial_cuda fp64` (CUDA) |
| Nikuradse mixing-length | `turbulence/tet_p1bf3_rans/`, `gpu/serial_cuda/h9_rans_mixlen.cuh` | See module READMEs |
| CUDA k-omega SST (V2) | `turbulence/sst_gpu_v2/` | `make -C turbulence/sst_gpu_v2 -j4 NVCC=/usr/local/cuda-12.2/bin/nvcc CUDA_ARCH=sm_86` |

## GPU SST example (spiral, 207k tetrahedra)

Clone the repository, build the SST module as above, then run from the repo root:

```bash
bash examples/spiral_swept_207k/RUN_SPIRAL_SWEPT.sh --dry-run
bash examples/spiral_swept_207k/RUN_SPIRAL_SWEPT.sh
```

The example's case and mesh are under `examples/spiral_swept_207k/`. It uses
face-dependent adaptive Spalding/omega sampling and is intended for one inlet,
one outlet and one wall on tagged Gmsh MSH 2.2 **tetrahedral** meshes. To
change settings, edit `SPIRAL_SWEPT_ADAPTIVE.case` or pass `--set` flags.

This example uses kinematic pressure. For the spiral reference fluid density
of 1000 kg/m^3, multiply reported kinematic pressure drop by 1000 to get Pa.
The example now expresses the requested **20/20/20 FGS sweeps**, pressure
**rtol 1e-3** and target **y+ 30**; this specific version has **not** been
rerun as a full GPU numerical regression. The earlier converged it=220
experiment used different effective solver settings. Check `terminal.log`.

## Algebraic mixing-length / Firedrake comparison

- Native NodalS RANS: `turbulence/tet_p1bf3_rans/`
- Native C baseline: `turbulence/tet_p1bf3_rans_c/`
- GPU mixing-length: `gpu/serial_cuda/h9_rans_mixlen.cuh`
- Firedrake VMFL003 pipe: `examples/firedrake_vmfl003/`

## Scope

CUDA SST is currently a research/beta implementation with one inlet, one
outlet and wall patch, not a validated generic multi-patch CFD product.
The adaptive runner may report a missing scheduled checkpoint if a solver
stops early after convergence; that is an orchestration issue, not proof
of a numerical divergence. Review residuals and final native status.

`main` contains the supported current source and representative examples.
Historical development branches may be deleted after a full Git bundle
backup without rewriting the source history.
