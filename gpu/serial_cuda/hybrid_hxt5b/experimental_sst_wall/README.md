# Experimental SST wall-treatment branch — NOT production

**Branch intent:** preserve the 2026-09-12/13 NodalS HXT5B SST wall-treatment experiments exactly enough to reproduce and continue the y+ investigation without contaminating `main`.

This branch is deliberately **not finalized** and should **not be merged into `main` as a production wall model**.  It contains useful working machinery and several important negative/partial results, but the wall treatment is still strongly y+-dependent.

## Solver context

- NodalS serial CUDA hybrid HXT5B solver.
- Velocity space: HEX Q1+BF2 / TET P1+BF3.
- Pressure: one Q0/P0 unknown per cell.
- SIMPLE-like segregated loop, device-resident GPU path.
- Turbulence model under test: incompressible Menter SST-2003m.
- Momentum wall stress remains the existing off-wall Spalding treatment.
- `k` uses zero normal-gradient at the wall.
- Pressure is kinematic pressure, so physical pressure in Pa is `rho * p_kin`.

## Benchmark used for the wall study

ANSYS smooth-pipe benchmark geometry/conditions used in the 500D tests:

- pipe length `L = 2 m`
- radius `R = 0.002 m`, diameter `D = 0.004 m`
- bulk velocity `Ub = 50 m/s`
- density `rho = 1.225 kg/m^3`
- dynamic viscosity `mu = 1.7894e-5 Pa s`
- Reynolds number approximately `13691.74`
- ANSYS benchmark target pressure drop: `21744 Pa`
- equivalent NodalS kinematic pressure-drop target: `21744 / 1.225 = 17750.2041 m^2/s^2`

The long-pipe tests use the same topology/DOF count while stretching the original axial mesh.  The low-y+ sweeps preserve topology and DOF count but radially remap the HEX annulus; therefore they are not mathematically pure y-only changes because radial resolution distribution changes as well.

## Why this branch exists

The original high-Re SST wall implementation used a logarithmic omega target derived from the Spalding wall state and imposed it by symmetric Nitsche at the physical wall trace.  That looked reasonable at high y+, but a same-DOF y+ sweep exposed strong wall-resolution dependence.

The experiments then separated three wall modes:

### `trace_log`

Original behavior.  The logarithmic omega target is imposed at the physical FE wall trace.

### `sample_log`

Experimental correction.  The same log target is imposed at the existing off-wall sample point instead of at y=0.  The physical wall omega diffusion is left natural.  This markedly improved the mean-y+ ~27 case, but at mean-y+ ~16 the solution became strongly mesh sensitive and the pressure drop kept increasing.

### `of_auto`

Current experimental OpenFOAM-informed mode.  It keeps the Spalding momentum traction and off-wall omega constraint, but adds:

- viscous/log omega blending based on a wall Reynolds number,
- an OpenFOAM-inspired near-wall production correction applied to the wall HEX cell,
- consistent propagation of that production correction into the omega equation,
- diagnostics for the blend, production ratio and wall/sample eddy viscosity.

This is **not** byte-for-byte OpenFOAM behavior.  OpenFOAM operates on wall-adjacent FV cells; NodalS has continuous FE velocity/turbulence unknowns.  The current implementation maps the idea to the off-wall FE sample point and approximates a wall-cell volume.  That mapping is one of the things still under investigation.

## Important implementation details preserved here

The branch also preserves two fixes that were required during the low-y+ work:

1. **rAU positivity gate fix.**  The raw penalty-free momentum diagonal is allowed to become negative locally; the actual row-L1-relaxed denominator used by rAU/Schur must be finite and positive.  On the gap032 low-y+ mesh the raw no-penalty diagonal had 1600 negative entries, all interface-adjacent, while the actual relaxed denominator was positive.
2. **Separate momentum and Schur wall penalties.**  `MOM_NITSCHE_GAMMA` and `SCHUR_NITSCHE_GAMMA` are independent.  The wall-study baseline used momentum gamma 50 and Schur gamma 0.

## Current `of_auto` knobs

- `SST_OMEGA_WALL_MODE=trace_log|sample_log|of_auto`
- `SST_WALL_PENALTY_GAMMA=50`
- `SST_OF_KAPPA=0.41`
- `SST_OF_E=9.8`
- `SST_OF_BETA1=0.075`
- `SST_OF_RE_BLEND=11.0`
- `SST_OF_PRODUCTION=1`
- `SST_OF_PRODUCTION_SCALE=1.0`
- `SST_OF_WALL_VOLUME_SCALE=1.0`
- `SST_OF_PRODUCTION_LIMIT=10.0`

The principal new diagnostics are `NODALS_SST_OMEGA_WALL` and `NODALS_SST_OF_AUTO`.

## Results that motivated freezing the branch

These are engineering checkpoints, not final validation data.

| Wall treatment | mean y+ | kinematic total dp | physical total dp | error vs 21744 Pa | state/observation |
|---|---:|---:|---:|---:|---|
| older high-y+ SST treatment | ~83.5 | 16481.2 | 20189.5 Pa | -7.15% | developed interior, pressure/wall balance good |
| `sample_log` | ~27.3 | 17046.9 | 20882.4 Pa | -3.96% | physically stationary |
| `sample_log` | ~16 | 21658.9 at it=800 | 26532.1 Pa | +22.02% at it=800 | not settled; pathological upward drift; max nut/nu ~346 |
| `of_auto` | ~84.0 | ~16667.5 | ~20417.7 Pa | -6.10% | essentially stationary; viscous blend fraction ~2.8e-6 |
| `of_auto` | ~28.05 | ~17950.0 | ~21988.8 Pa | +1.13% | nearly stationary; best benchmark agreement so far |
| `of_auto` | ~16.27 | 20304.5 at it=800 | 24873.0 Pa | +14.39% at it=800 | still not settled; much smaller max nut/nu (~103) than `sample_log`, but still unacceptable y+ sensitivity |

The ~28 result being close to the benchmark is **not** sufficient to accept the model.  The y+ sweep is the crucial result: the same wall formulation does not yet approach a mesh-independent pressure drop over y+ ~16 to 84.

## Diagnostics learned from `of_auto`

At mean y+ ~84:

- `lamFracMean ~ 2.8e-6`, so the viscous omega branch is effectively off and the target is almost entirely logarithmic.
- `pkRatio ~ 1.78`, showing the production correction is active even though the viscous/log omega blend is not.

At mean y+ ~28:

- `lamFracMean ~ 1.26e-2`, so only a modest viscous contribution is present.
- `pkRatio ~ 1.288`.
- the solution settled near the ANSYS pressure drop, but this should be treated as a checkpoint rather than a tuned success.

At mean y+ ~16:

- V1 suppressed the extreme eddy-viscosity spike seen in `sample_log` (`maxNut/nu` fell from roughly 346 to roughly 103), which is a real improvement.
- nevertheless pressure drop continued to rise and the state was not adequately settled by iteration 800.

## What is considered understood

- The 500D pipe is long enough to obtain a developed interior; shorter 10D runs mixed wall friction with axial momentum-profile development.
- In a developed region, agreement between pressure-gradient friction and wall-shear friction is an internal momentum-balance check.  If those agree but both miss the benchmark, the remaining issue is primarily wall/turbulence modelling rather than axial development.
- Very loose pressure inner tolerance (`P_RTOL=0.9`) can still give a physically stationary SIMPLE outer state on these extreme stretched meshes; formal residual gates must not be confused with physical stationarity.
- The raw penalty-free rAU diagonal is not itself a valid positivity criterion after the row-L1 fixed-point relaxation used by the actual solver.
- `sample_log` demonstrated that treating the off-wall sample as the wall-function state is structurally better than forcing the same finite log-layer omega value onto the physical wall trace.
- `of_auto` V1 improves the low-y+ eddy-viscosity pathology but does not yet deliver all-y+ invariance.

## What remains unresolved / next work

1. Reproduce OpenFOAM's omega wall-function and production manipulation more literally at the wall-adjacent-state level and identify exactly what the FE analogue should be.
2. Audit whether the current production correction is being inserted with the correct measure/volume scaling and whether it double counts any part of the existing SST production.
3. Revisit the all-y+ viscous/log switching/blending formula and the definition of the wall Reynolds number used for the transition.
4. Reconcile constants used by the momentum Spalding law and the SST wall formulas (`kappa`, `E/B`) so the two wall models do not represent inconsistent log laws.
5. Examine the relation between the Spalding-derived effective wall viscosity and the SST sample-point `nut`; the current diagnostics show a substantial difference.
6. Repeat the same 500D y+ sweep only after each structural change; do not tune constants to make one y+ hit 21744 Pa.
7. Because the low-y+ meshes are same-DOF radial remaps, eventually repeat the study with genuinely refined wall-normal meshes to separate y+ sensitivity from radial-resolution redistribution.
8. Continue using pressure-gradient versus wall-shear momentum balance in the developed 350D-450D region as an internal consistency check.

## Reproduction baseline used in the study

Typical controls used for the comparison runs:

- `ALPHA_U=0.70`
- `ALPHA_P=1.0`
- `SST_ALPHA_K=0.70`
- `SST_ALPHA_OMEGA=0.70`
- `RAU_SCALE=20`
- momentum SGS1, `MOM_RTOL=0.01`, `MOM_MAX=50`
- pressure `pcg_amg`, `P_RTOL=0.9`, `P_MAX=50`
- PMIS / classical-negative strength / theta 0.25 / exti / pmax 8 / terminal 1000
- MCGS fine/coarse sweeps 2, symmetric
- SST scalar `RTOL=0.20`, `MAX=12`
- `WALL_SAMPLE_FRACTION=0.5`
- `MOM_NITSCHE_GAMMA=50`
- `SCHUR_NITSCHE_GAMMA=0`

The experiment runner and same-DOF mesh-remap/postprocessing helpers are stored alongside this README on the experimental branch.

## Branch policy

- Keep this branch as an experimental record.
- Do not fast-forward or merge it into `main` merely because the y+ ~28 checkpoint agrees with ANSYS.
- Future wall-function variants should be committed here (or on child experimental branches) with explicit y+ sweep results before any production integration is considered.
