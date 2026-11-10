# GPU SST: generalized Robin–Nitsche wall experiment (2026-11-10)

Experimental branch: `exp/gpu-generalized-robin-nitsche-20261110`.
Build from experimental Bazilevs-B branch, preserving the known-good
Spalding solution as default; **do not merge into main before verification**.

## Mathematical definition

`beta_w` is the ORIGINAL frozen, F2 face-averaged, off-wall sampled
Spalding conductance of the current NodalS implementation.
`mu_c` is independently selectable `nu` or frozen `nu+nut` at
each wall quadrature point. `h_n=3V_tet/A_wallface`.
`beta_N=C_stab*mu_c/h_n`; `C_stab` does not affect wall-law sampling.

For tangential trial velocity `u_t`, test `w_t` and projected
deviatoric viscous traction `t_t(u)=P_t(2 mu_c D(u)n)`, define the
generalized Robin boundary form:

    aG(w,u) = - beta_w/(beta_w+beta_N) *
                 [<t_t(u),w_t>+<t_t(w),u_t>]
              + beta_w*beta_N/(beta_w+beta_N) * <u_t,w_t>
              - 1/(beta_w+beta_N) * <t_t(u),t_t(w)>.

NodalS ALREADY assembles `+beta_w <u_t,w_t>`. So the new kernel
adds exactly `aG-a0`, namely:
- `-alpha` times both symmetric tangential consistency terms,
- `-beta_w*alpha <u_t,w_t>`, and
- `-<t_t(u),t_t(w)>/(beta_w+beta_N)`,
with `alpha=beta_w/(beta_w+beta_N)`.

No additional physical friction term is added; coefficients are frozen
for the momentum Picard step. In the limit `C_stab -> infinity`,
`aG -> original Robin`; when `beta_w -> infinity` at fixed beta_N,
`aG -> symmetric Dirichlet Nitsche`. The latter is a formal limit:
avoid computing gigantic `beta_w` values just to approach it.

The derivation is adapted from Juntunen & Stenberg (2009),
*Nitsche's method for general boundary conditions*, for projected
Navier/Robin tangential traction. Formal stability for this particular
P1+BF3 SST/mixed SIMPLE implementation is **not yet proven**.
Using `nu` instead of the true `nu+nut` in boundary consistency is
a controlled approximation; test the effective-viscosity variant too.

The BF3 zero-integrated-normal-flux constraint, original normal penalty,
original F2 Spalding samples, and omega `of_auto` remain unchanged.
Coupled matrix-free FGMRES and M3 assembled nine-block CSR both include
the *same* new correction before BF3 T^t. Pressure Schur remains the
original approximation: review pressure correction behavior as a
separate validation task.

## Case knobs (implemented)

```ini
[wall_function]
boundary_form = generalized_robin_nitsche
tangential_consistency = off
stabilization_c = 15.0
consistency_viscosity = nu
spalding_sample_scale = 0.50
```

* `boundary_form = existing` restores original wall assembly (if
  `tangential_consistency = off`).
* `stabilization_c` is NUMERICAL stabilization only.
* `spalding_sample_scale` controls the original inward sampling, not
  `stabilization_c`. The ordinary `nodals.py run` does NOT run
  the external adaptive sampling controller.
* The second included case uses `consistency_viscosity=nu_plus_nut`.
* Do not enable momentum SUPG in the first experiment: the existing
  coupled-FGMRES momentum-SUPG branch is not wired for this study.
* Use advective momentum convection, consistent with prior stable spiral.

## Local build / test

```bash
cd ~/NodalS_GPU_BAZB
git status --short
git fetch origin
git switch -c local/genrobin-20261110 \
  origin/exp/gpu-generalized-robin-nitsche-20261110

python3 turbulence/sst_gpu_v2/tests/test_generalized_robin_algebra.py

make -C turbulence/sst_gpu_v2/source/puretet_profile5_eq5_hxt5b \
  NVCC=/usr/local/cuda-12.2/bin/nvcc ARCH=sm_86 -j1 nodals_hxt5b_fp64

BIN="$PWD/turbulence/sst_gpu_v2/source/puretet_profile5_eq5_hxt5b/nodals_hxt5b_fp64"
CASE=examples/spiral_swept_207k/SPIRAL_GENERALIZED_ROBIN_NITSCHE_FGMRES.case
python3 turbulence/sst_gpu_v2/nodals.py run \
  --case "$CASE" --binary "$BIN" --steps 8 \
  --native-option mom-cap-policy=fail \
  --output "$HOME/NodalS_SST_LOCAL_RUNS/spiral_genrobin_smoke"
```

Then compare *identical case* under `--set wall_function.boundary_form=existing`
and vary only `wall_function.spalding_sample_scale`, keeping `C_stab=15`.
A full 30–100 wall y+ investigation requires verified original off-wall
face-u_tau exports and the separate adaptive runner; these are deliberately
left unchanged in this commit.

Inspect `NODALS_GENROBIN_CONFIG`, `STAGE22D4D_COUPLED_KRYLOV`,
`STAGE22D4D_FROZEN_VECTOR_RESIDUAL`, pressure drop, F2 reported
wall y+, wall mean normal flux, and omega/k update norms.

## Current verification status

Static source reviews passed for both operator dispatches, including
BF3 transform ordering and exact old-F2 coefficient reuse.
Independent scalar algebra/symmetry/Robin/Dirichlet checks are included.
**No CUDA build or GPU runtime executed here**; do not claim validation,
parity or stability until workstation tests pass.
