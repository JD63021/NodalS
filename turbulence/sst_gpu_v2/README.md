# NodalS GPU k–ω SST V2 — case and numerical options

**Status:** CUDA FP64 research/beta solver. This branch is a review/freeze candidate; the workstation's *currently tested* `~/Downloads/NodalS_SST_SOFTWARE_V2` remains authoritative until its source is synchronized by `FREEZE_FROM_LOCAL_V2.sh`. Do **not** merge solely on the basis of this documentation commit.

## Scope and discretization

Steady incompressible RANS with transported k–ω SST, SIMPLE pressure/velocity coupling and a CUDA finite-element backend. Pure tetrahedral meshes with a single inlet (physical tag 1), outlet (tag 2), wall (tag 3), tetrahedral volume (tag 4) are the exercised path. Velocity **P1+BF3**, pressure **P0 per tetrahedron**, k and ω **masked P1 by default** (`[sst] state_space=q1, compact_p1=false`). Experimental truly compact P1 (`compact_p1=true`) is **not equivalent** to the masked-P1 reference and requires L1 calibration; a diagnostic-only uncalibrated override is available in the native code. Do not assume arbitrary multi-patch geometries are supported.

### Build and run

From the repository root:

```bash
make -C turbulence/sst_gpu_v2 -j4 NVCC=/usr/local/cuda-12.2/bin/nvcc CUDA_ARCH=sm_86
cd turbulence/sst_gpu_v2
python3 -u nodals.py run --case examples/ubend_200k.case --dry-run
python3 -u nodals.py run --case /ABSOLUTE/PATH/TO/working.case --steps 200
```

The included U-bend reference case may require external geometry/inlet-profile files. Prefer a copied, edited case with the exact mesh paths and verified patch tags. Run data, VTU and checkpoints belong outside tracked source. No recompilation is needed to change a case.

## Case options: section-by-section reference

This table covers options exposed in the current V2 front-end and important newer experiments. **Review the live case and `nodals.py` for the exact supported keys before treating an experimental option as production-ready.**

| Section | Case keys / settings | Effect and cautions |
| --- | --- | --- |
| `[mesh]` | `format`, `path`, `diameter`, `patch_map` | Gmsh MSH2 or existing HXT1 mesh; hydraulic/characteristic diameter affects Re and wall heights. |
| `[physics]` | `density`, `reynolds`, `kinematic_viscosity` | Kinematic-pressure solver; case input consistency checks **Re = U D / ν** (tolerance approximately 0.1%). Fluid density=1 in current native contract; dimensional pressure is reconstructed externally. |
| `[inlet.main]` | `mode`, `velocity`, `speed`, `direction`, `profile_csv`, `profile_mapping`, `profile_flux`, `profile_corner_epsilon` | Single prescribed inlet vector / optional compatible developed profile CSV. Velocity must point *into* the inlet; changing Re alone requires corresponding speed change. |
| `[outlet.main]` / `[wall.main]` | `type` | Currently metadata for the implemented natural outlet and Spalding wall; editing these labels does not install a new boundary model. |
| `[initialization]` | `mode = plug / potential_flow` | Starting flow state only. |
| `[wall_distance]` | `method` | Geometry-based wall distance in exercised cases. |
| `[wall_sampling]` | `mode`, `max_crossings`, `bary_tolerance` | Wall-normal ray tracing and fallback behavior. The **thin-channel owner-tetrahedron-centroid fallback** avoids ray-sample failures on narrow cavities; it does *not* repair poor-quality elements or guarantee good turbulence solutions. Check the active native startup diagnostics for the fallback version and number of affected faces. |
| `[wall_function]` | `law`, `samples`, `spalding_sample_scale`, `zero_limit`, `nitsche_gamma`, `face_spalding`, `frozen_wall_cache`, `shared_wall_state`, `integral_utau`, `integral_nq`, optional `distances` | Spalding tangential traction, no-penetration and BF3 wall coupling. `samples` selects 1–5 fit points; the source may compute an underlying five-height profile. Sampling scale affects location, not mesh size. |
| `[omega_wall]` | `method=of_auto`, `sample_index`, `sample_scale`, `gamma`, `production`, `stage41_mode`, `trace_gamma` | ω wall treatment. `stage41_mode=trace_log` supports the near-wall log trace option; `sample_log` and `of_auto` are separate experiments. Distinct Spalding and ω sampling distances are intentional. |
| `[sst]` | `state_space`, `compact_p1`, `compact_quad`, `intensity`, `length_ratio`, `ur_k`, `ur_omega`, `k_solver`, `omega_solver`, `k_sweeps`, `omega_sweeps`, `check_every` | SST transport and scalar solver controls. Masked P1 is the stable reference; compact P1 is experimental. `fgs-fixed` with `check_every=0` may print NaN *linear residual diagnostics* because they are not evaluated; this is not automatically a nonfinite physical field. |
| `[convection]` | `form=advective / skew_symmetric / conservative / emac` | Momentum convection. Advective = `u*·∇u`; skew = `u*·∇u + 0.5(∇·u*)u`; conservative and EMAC are optional distinct operators with compatibility constraints. The scalar transport is independently assembled. Some industrial meshes were more stable with `advective`; this is not a universal accuracy finding. |
| `[stress]` | `form=vector_laplacian / deviatoric` | Momentum viscous-volume operator; deviatoric changes the operator. Check its native compatibility with momentum SUPG; do not assume both are simultaneously active just because requested in a case. |
| `[solver.u]` | `method=m3-fgs / m3-sgs / coupled-fgmres / m4-hybrid`, `sweeps`, `omega`, `rtol`, `atol`, `max_iterations`, `restart` | Momentum solver. M3 stores full 3×3 coupled block values: **two large FP64 block CSR work arrays** can exhaust 12-GB GPUs at 1M+ tets. Coupled FGMRES avoids the M3 allocations but uses Krylov workspace. `m4-hybrid` is experimental. |
| `[solver.p]` | `method`, `rtol`, `atol`, `max_iterations` | Pressure PCG–AMG / Jacobi choices, with one pressure unknown per tetrahedron. |
| `[amg]` | `hierarchy`, `coarsening`, `strength`, `interpolation`, `theta`, `terminal`, `smoother`, `fine_mode`, `fine_sweeps`, `coarse_sweeps` | CF–PMIS pressure AMG, `classical-negative` strength, `exti` interpolation. `smoother=mcgs` or `jacobi` changes preconditioning; Jacobi does **not** remove M3 momentum-block allocations. |
| `[simple]` | `max_iterations`, `physical_gate`, `ur_u`, `ur_p`, `simple_tol`, `continuity_atol`, `plateau_tol`, `plateau_samples` | SIMPLE outer iteration, under-relaxation and stopping criteria. Low tolerances are not proof of steady-state accuracy. |
| `[supg]` | `momentum`, `k`, `omega`, `tau_scale`, `k_tau_scale`, `omega_tau_scale` | Independently selected SUPG stabilization. The **local November 2026 unified full-residual momentum SUPG** supersedes older partial residual behavior in the active workstation source. The online source must be synced before this PR claims to contain that implementation. `tau_scale=0.05` has been exercised, but assess mesh/flow dependency. |
| `[adaptive_sampling]` | `enabled`, `first_refresh`, `refresh_every`, `max_refreshes`, target Spalding/ω `y+` controls (see current runner) | Optional checkpoint/restart face-based adaptation. The wrapper may expect a scheduled checkpoint even if the native solver stops early on convergence; inspect native output and wrapper separately. |
| `[restart]` | `path`, `input_fingerprint` | Native full-state `.ckpt` restart; **VTU is not a restart file**. |
| `[output]` | `console`, `print_every`, `metrics_every`, `diagnostics_every`, `vtu_every`, `checkpoint_every`, `directory`, `vtu_dir`, `checkpoint_dir`, `profile`, `profile_audit`, `legacy_final_diagnostics` | Output cadence, diagnostics and checkpoint management. |
| `[environment]` | Native environment variable passthrough | Experimental backend and legacy controls; check case/source exact spelling. |
| `[native]` | Native command-line option passthrough | Advanced native solver flags; use only after verifying CLI support. |

### Unified SUPG: what to verify in the local source

Recent development unified momentum SUPG with a **full strong momentum residual**. SUPG is still a *stabilization*, not a positivity limiter, and not a substitute for fixing highly distorted tetrahedra. When freezing the current local code, verify the actual functions and runtime log confirm the final residual rather than relying on the older online `stage33a_momentum_supg.cuh` filename alone. Scalar k and ω SUPG retain separate switches and separate τ scales; they need not be enabled with momentum.

**Compatibility:** the case front-end currently rejects momentum SUPG with some momentum convection/stress combinations. Preserve those checks and document the exact admitted combinations after local sync. `stress.form=deviatoric` and `supg.momentum=true` are specifically rejected together in the currently published V2 front-end; if the installed November 8 code supports this combination, that change must be included in the PR and verified.

### Wall sampling and high-surface-area meshes

On narrow cavities, ray tracing may not find a geometrically valid interior point. The local owner-tetrahedron-centroid fallback provides a finite geometric sampling location but does not change tetrahedral topology or repair tiny CAD edges. For shell/cavity geometries, memory should be budgeted using cell count, wall-face count, enriched velocity DOFs, momentum/pressure CSR nnz, and workspace—not *only* GB per million tets. M3's pair of nine-value-per-entry FP64 block buffers is especially important.

## Verification before merging

1. Run the local source-freeze helper from a clean checkout. It only updates the review branch; **never merge unverified code directly to main**.
2. Check the resulting diff for forbidden backups, `.vtu`, `.ckpt`, `.msh`, generated binary output, and local absolute file paths.
3. Confirm dry-run case parsing and generated native CLI flags for **advective**, the selected stress form, unified full-residual SUPG, split wall sampling, and the centroid fallback.
4. Run a short CUDA smoke check on the already working pure-tet Shell1 / Shell2 360k geometry, compare diagnostics with the workstation reference, and record peak GPU memory. This PR's initial online documentation commit alone **does not establish runtime parity**.
5. Inspect the **Files changed** tab and merge manually after the source commit and verification.

Source names can retain historical `stage` markers; they are internal kernel identifiers rather than installation requirements.
