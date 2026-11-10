# Bazilevs (2007) numerical-slip penalty — GPU SST experiment

**Isolated branch:** `exp/gpu-bazilevs-slip-20261110` (the known-good
baseline and earlier Bazilevs-B branch are retained).

This experiment keeps the two tangential symmetric Nitsche consistency
terms already added in Bazilevs-B, but **replaces the tangential penalty
coefficient**, rather than summing penalties.

For every wall face and 12-point triangular quadrature point:
- Evaluate the *lagged numerical boundary velocity* in the existing
  P1+BF3 wall-trace basis `ca.bw`.
- Project it into the local face tangent plane `P=I-nn^T`.
- Set `h_n=3 V_tet/A_wallface`, `y_eff=h_n/C_b`, with `C_b=4`
  initially (Bazilevs et al., 2007).
- Evaluate the existing Spalding inversion
  `spalding(|P u_wall|,y_eff,nu,u_tau,y_plus,beta)`.
  For nonzero slip, `beta=u_tau*u_tau/|P u_wall|`.
  Its zero-slip molecular limit is `beta=nu/y_eff=C_b nu/h_n`.
- Freeze this `beta` for the SIMPLE/Picard step and assemble **the
  original single wall Robin term** with the new coefficient.
  Downstream F1B cross-blocks, M3 CSR and matrix-free coupled FGMRES all
  use the same frozen `coeff[q].beta`. The Nitsche consistency and
  normal BF3 flux constraint are unchanged.

The selected viscosity of Nitsche consistency is independently controlled
with `consistency_viscosity = nu` or `nu_plus_nut`; this first case
uses `nu`. The omega `of_auto` wall law and its existing independent
inward sampling are untouched. This differs from a pure 2007 paper
reproduction in turbulence closure and in BF3 normal-flux enforcement.

```ini
[wall_function]
tangential_consistency = symmetric_nitsche
consistency_viscosity = nu
penalty_source = numerical_slip
bazilevs_cb = 4.0
```

Switch to `penalty_source = offwall_spalding` for the old Bazilevs-B
control, or `tangential_consistency = off` for the unaugmented baseline.

The five-point off-wall momentum F2 fit is **skipped** in numerical-slip
mode and remains available unchanged in off-wall control mode. The
five-point omega sampler still operates and can fail if the solution blows up.

## Working-directory update and one-time FP64 rebuild

```bash
cd ~/NodalS_GPU_BAZB
git status --short
git fetch origin
git switch -c local/gpu-bazilevs-slip origin/exp/gpu-bazilevs-slip-20261110
make -C turbulence/sst_gpu_v2/source/puretet_profile5_eq5_hxt5b \
  NVCC=/usr/local/cuda-12.2/bin/nvcc ARCH=sm_86 -j1 nodals_hxt5b_fp64
```

The switch must be done with a clean worktree; the earlier `local/gpu-bazb`
branch remains available and untouched. A separate worktree is also possible.

Run eight iterations, **without momentum SUPG**, with true-residual
coupled FGMRES and strict failure on unmet tolerance:

```bash
cd ~/NodalS_GPU_BAZB
BIN="$PWD/turbulence/sst_gpu_v2/source/puretet_profile5_eq5_hxt5b/nodals_hxt5b_fp64"
CASE=examples/spiral_swept_207k/SPIRAL_SWEPT_BAZILEVS_SLIP_FGMRES.case
python3 turbulence/sst_gpu_v2/nodals.py run \
  --case "$CASE" --binary "$BIN" --steps 8 \
  --native-option mom-cap-policy=fail \
  --output "$HOME/NodalS_SST_LOCAL_RUNS/spiral_bazilevs_slip_smoke"
```

Inspect `terminal.log` for:
- `NODALS_BAZILEVS_SLIP enabled=1`;
- `STAGE22D4D_COUPLED_KRYLOV trueRel=... converged=1`;
- `STAGE22D4D_FROZEN_VECTOR_RESIDUAL`;
- `HXT5B_ERROR`, `NODALS_BAZILEVS_SLIP_INVALID`;
- pressure drop, update norms, wall y+, and normal mean flux.

## Validation limitation

The code was changed and inspected through GitHub, but **not compiled
or executed here**: this environment has no NVIDIA CUDA compiler/GPU.
Theoretical consistency and numerical stability are **not yet proven**.
Symmetric Nitsche with this wall-law-derived penalty may still be
insufficiently coercive on a P1+BF3 tetrahedral mesh. Stop and diagnose
true FGMRES convergence before longer simulations.

Reference: Bazilevs, Michler, Calo & Hughes (2007), *Weak Dirichlet
boundary conditions for wall-bounded turbulent flows*, CMAME 196,
4853–4862, equations for Spalding wall-law penalty.
