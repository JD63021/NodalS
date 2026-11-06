# NodalS GPU k-omega SST V2

Single-GPU steady incompressible FEM RANS implementation on NodalS-compatible
pure tetrahedral meshes. Velocity uses P1+BF3, pressure P0 and the current
k/omega state is masked P1. This is a research/beta solver supporting the
existing one-inlet/one-outlet/one-wall boundary model.

## Compile (one CUDA build)

From the repository root:

```bash
make -C turbulence/sst_gpu_v2 -j4 NVCC=/usr/local/cuda-12.2/bin/nvcc CUDA_ARCH=sm_86
```

Requires CUDA and a C++17 compiler; `python3` with NumPy is used for Gmsh
MSH2 mesh preparation. Builds incrementally and does not need per-case
recompilation. Other CUDA devices require a suitable `CUDA_ARCH` value.

## Run example

```bash
bash examples/spiral_swept_207k/RUN_SPIRAL_SWEPT.sh --dry-run
bash examples/spiral_swept_207k/RUN_SPIRAL_SWEPT.sh
```

The launch script selects `turbulence/sst_gpu_v2` relative to the clone by
default. The `.case` and 207,360-tet `.msh` are in the example directory.
Use `--set section.key=value` to override case values. Run output and
checkpoints belong in `runs/`, not in the source checkout.

The U-bend file under `turbulence/sst_gpu_v2/examples/` is a development
reference that requires external U-bend mesh/profile inputs. It is not the
out-of-the-box example for a fresh clone.

## Limitations

- Only validated for properly tagged pure-tetrahedral meshes with a single
  inlet (tag 1), outlet (tag 2) and wall (tag 3).
- Physical pressure is recovered from kinematic pressure using fluid density.
- Staged adaptive sampling uses full-state checkpoints; a solver stopping
  early on convergence may currently trigger a runner checkpoint expectation
  error even when native convergence reports PASS.
- The spiral example's published 20-sweep, pressure `rtol=1e-3`, target
  `y+=30` controls require a fresh GPU numerical regression for exact parity
  with earlier runs that used differing effective settings.

Source directories may retain historical internal kernel/stage identifiers;
these are not separate user-installation stages or external setup gates.
