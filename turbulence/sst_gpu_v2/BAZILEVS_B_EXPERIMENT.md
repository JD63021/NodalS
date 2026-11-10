# GPU Bazilevs-B (additive tangential Nitsche consistency) — experimental

Branch: `exp/gpu-bazilevs-b-m3-20261110`. The original Spalding F2 face fit,
wall-function Robin coefficient, omega `of_auto/trace_log`, BF3 normal mean-flux
constraint, mesh and inlet are unchanged.

Select under `[wall_function]`:

```ini
tangential_consistency = symmetric_nitsche
consistency_viscosity = nu
```

Allowed viscosity values: `nu` or `nu_plus_nut` (frozen quadrature SST
effective viscosity). Set `tangential_consistency = off` to recover the
original momentum boundary operator.

The added bilinear form is the two symmetric tangential Nitsche consistency
terms, using `2 mu D(u)n` projected to the local face tangent plane.
The isotropic deviatoric correction has zero tangential projection.
The existing nonlinear Spalding wall term is deliberately **not** changed.
This is *Bazilevs-inspired variant B*, not the full 2007 slip-based formulation.

The additional 3x3 tensor blocks are assembled into the true M3 coupled
matrix before BF3 `T^t A T`. The same operator is included in the
matrix-free reference action used for diagnostics, so no uncoupled
rank-one shortcut is silently applied. Supported first-run configuration:
M3 FGS/SGS, pure tet, F1B, F2, BF3, implicit deviatoric stress,
and momentum SUPG off.

One-time CUDA FP64 build:

```bash
cd ~/NodalS
git fetch origin
git worktree add -b local/gpu-bazb ~/NodalS_GPU_BAZB origin/exp/gpu-bazilevs-b-m3-20261110
cd ~/NodalS_GPU_BAZB
make -C turbulence/sst_gpu_v2/source/puretet_profile5_eq5_hxt5b \
  NVCC=/usr/local/cuda-12.2/bin/nvcc ARCH=sm_86 -j1 nodals_hxt5b_fp64
```

Then run either case:

```bash
cd ~/NodalS_GPU_BAZB
BIN="$PWD/turbulence/sst_gpu_v2/source/puretet_profile5_eq5_hxt5b/nodals_hxt5b_fp64"
python3 turbulence/sst_gpu_v2/nodals.py run \
 --case examples/spiral_swept_207k/SPIRAL_SWEPT_BAZILEVS_B_NU.case \
 --binary "$BIN" --steps 5 --output "$HOME/NodalS_SST_LOCAL_RUNS/spiral_bazb_nu_smoke"
```

Replace `_NU.case` with `_NU_NUT.case` to test effective viscosity.
Each case uses a separate output directory. For a full adaptive sampling run,
use `examples/spiral_swept_207k/run_adaptive_case.py` and the chosen case.

## Validation priorities

1. C++/CUDA compilation, then fresh five-iteration smoke for each viscosity.
2. The baseline `off` path must give the existing result to roundoff.
3. Verify assembled M3 against the matrix-free reference operator before
   relying on the new path. Set the existing M3 operator parity diagnostic.
4. Examine sign/conditioning of the total new bilinear form: symmetric
   Nitsche consistency with only the physical Spalding penalty does **not**
   guarantee coercivity on arbitrary meshes.
5. Compare wall normal flux, pressure drop, wall (y^+), momentum residual,
   `k`/`omega`, and wall time.

Not compiled or executed in the ChatGPT environment, which has no nvcc/GPU.
No changes to the existing `main` branch.
