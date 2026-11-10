# GPU generalized Robin–Nitsche with skew-symmetric full momentum SUPG

Experiment branch: `exp/gpu-genrobin-skew-supg-20261110`.
Base: `exp/gpu-generalized-robin-nitsche-20261110`.

## Scope

Enable **existing** full element-interior momentum SUPG with the generalized
Robin–Nitsche wall operator on pure tetrahedra, full P1+BF3 velocity,
P0 pressure, masked-Q1 k/omega, implicit deviatoric stress, and the original
three-component coupled FGMRES momentum solve.

- Physical off-wall F2 Spalding fitting stays unchanged.
- Generalized Robin correction stays unchanged, with independent `Cstab`.
- BF3 integrated normal-flux constraint (`T^t A T`) stays unchanged.
- No SUPG residual is added at the wall, inlet or outlet.
- Momentum SUPG's existing **element-interior** strong residual accounts for
  skew convection using `+0.5 (div a) u` and for the deviatoric stress
  with `nu_eff = nu + nut`, including finite-difference `grad(nu_eff)`.
  The element-interior gradient of P0 pressure is zero, and facet jumps are
  NOT included. The existing streamline-diffusion LHS and nonlinear RHS
  are paired by one `[supg] momentum` option.
- Matrix assembly: `stage33a_momentum_supg64` writes into the existing
  scalar `d_turb` and `d_diagRTurb` contribution BEFORE the
  momentum blocks are combined into CSR. `S22D4DOperator::apply()`
  starts with those same CSR blocks, then adds frozen F1B cross-wall terms,
  generalized Robin tangent corrections, deviatoric cross-component
  contributions, and finally applies the BF3 `T^t` transform.
- RHS: `momentum_full_deviatoric_supg_rhs64` operates on the momentum
  `d_rhs0..2` BEFORE `s22d4d_pack`. The off-component frozen wall
  lag `d_s22Rhs0..2` is subtracted by the pack exactly as before.
- Coupled FGMRES uses its actual residual and `mom-cap-policy=fail`.
- Original M3 FGS/SGS modes remain selectable. The new guard only opens
  the generalized-Robin plus coupled-FGMRES combination (other combinations
  remain prohibited), avoiding unnoticed changes to non-experimental paths.

## New case

`examples/spiral_swept_207k/SPIRAL_GENROBIN_SKEW_FULL_SUPG_FGMRES.case`

Key values:

```ini
[wall_function]
boundary_form = generalized_robin_nitsche
tangential_consistency = off
stabilization_c = 15.0
consistency_viscosity = nu
spalding_sample_scale = 0.50

[convection]
form = skew_symmetric

[supg]
momentum = true
k = false
omega = false
tau_scale = 0.05

[solver.u]
method = coupled-fgmres
rtol = 1e-4
max_iterations = 150
restart = 20

[stress]
form = deviatoric
```

`tau_scale` may be varied independently from `stabilization_c`.
Do NOT enable a second streamline diffusion or deprecated SUPG option.

## Recommended tests (workstation with CUDA 12.2)

1. `python3 turbulence/sst_gpu_v2/tests/test_generalized_robin_algebra.py`
2. `python3 turbulence/sst_gpu_v2/tests/test_genrobin_skew_supg_wiring.py`
3. Compile FP64 using the project Makefile.
4. First 8 SIMPLE steps on 207,360-tet spiral; then 20, then longer.
5. Inspect `NODALS_MOMENTUM_FULL_SUPG_CONFIG`,
   `NODALS_MOMENTUM_FULL_SUPG_LHS_ASSEMBLED`,
   `NODALS_MOMENTUM_FULL_SUPG_RHS_ASSEMBLED`,
   `NODALS_GENROBIN_SUPG`,
   `STAGE22D4D_COUPLED_KRYLOV`,
   `STAGE22D4D_FROZEN_VECTOR_RESIDUAL`.
6. Compare with *identical* generalized Robin case with
   `[convection] form=advective` and `[supg] momentum=false`;
   then skew/no SUPG and skew/SUPG to isolate stabilization.

CAUTION: No CUDA compiler/GPU is available in this environment.
This branch has not been built or simulated here. A source-contract test
does not establish full GPU matrix parity or physical stability.
