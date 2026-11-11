# Experimental omega-only DG branch (2026-11-11)

This branch is deliberately isolated from the validated CG/Robin-Nitsche GPU SST solver.

## Frozen benchmark

- 199,584 tetrahedron mirror-symmetric U-bend, Re=72,600.
- Existing P1+BF3 velocity, P0 pressure, masked-Q1 k, and pressure coupling.
- Existing five-point shared-vector Spalding wall fit, physical-wall omega trace-log target at sample #3, wall-normal adaptive y+ refresh at iteration 50.
- Momentum advective, deviatoric stress, momentum/k/omega SUPG off in the supplied active case.
- No automatic substitution of a projected continuous omega for the DG field inside SST physical calculations.

## Target discretization

- Omega: tetra-local Bernstein P1, 4 DOF/tet (4*199584=798336).
- Diffusion: symmetric interior penalty DG with two consistency terms, eta_F=C_IP*(p+1)^2*max(mu-/h-,mu+/h+), initial C_IP=10.
- Convection: advective volume form plus face correction, upwind theta=1 to start; theta=0.5/0 later.
- Wall: preserve *physical* trace_log target and shared Spalding u_tau, reassemble symmetric Nitsche on local DG trace. Do not mistake omega_wall.gamma=1e6 for active trace_log gamma=1e5.
- Inlet: omega from current SST intensity/length. Outlet: natural diffusive flux; explicit backflow policy.
- Solver target: FGMRES with 4x4 element-block Jacobi; positive cell-mean-preserving limiter as independent option.
- Visualization only: averaged-to-vertex CG-like field permitted. For physics: evaluate true DG omega per element/point.

## Correctness checks

1. Portable reference: constant preservation, exact linear solution under SIPG/Nitsche, face conservation/sign reversal, positive/negative mesh orientation, row/column structure, and positivity limiter.
2. CUDA: identical local element/face operators, true residual and solver convergence, two- and multi-tetra manufactured solutions.
3. Full SST: use cell-local omega in omega and k equations, F1/F2, turbulent viscosity, wall target, momentum coefficients and diagnostics, retaining CG baseline unchanged.
4. U-bend: compare at 45/90/135/151 degrees and downstream; report physical and numerical error, wall condition, omega jumps, clipping and GPU timing.

## Development safety

The presence of a new DG module does not mean the G2 solver has been integrated, compiled or validated. DG must be opt-in and fail-fast until full coupling and checkpoint/restart are implemented. Never mark a result DG unless its native field accesses are verified.

Source snapshot used for planning: main as of 2026-11-11.

## Implementation status — 2026-11-11

- Portable two-tetra C++17 DG-P1 reference: **passed on user's Ubuntu workstation**.
- New isolated CUDA BSR4 kernels: `modules/transport/dg_omega_bsr4.cuh`.
- Native parity executable: `modules/transport/tests/dg_omega_gpu_parity.cu`.
- Run parity: `bash modules/transport/tests/RUN_GPU_PARITY.sh` from the tests directory, or invoke by absolute path.
- **CUDA kernel parity pending workstation test**. No compiler or RTX GPU is available in the remote code-writing environment.
- **Not G2-integrated**. Existing `[sst] state_space=q1` behavior remains unchanged.
- Next: verify GPU matrix/rhs/SpMV/diagonal-block parity, then construct mesh tagging and true DG omega physics coupling; do not run a U-bend as DG until the true field-access path is wired.

## GPU block structure contract

- `rowPtr` and `colInd` index tetrahedra (blocks), not scalar coefficients.
- `blocks[16 * slot + 4 * localRow + localCol]` stores a dense row-major 4x4 block.
- Each tetra is a diagonal block; each neighboring tetra contributes one off-diagonal block.
- Four face block slots `LL, LR, RL, RR` are computed on host and provided to face kernels.
- `vertexL/R[3]` map the same globally sorted face vertices to each tetra's local vertex numbering.
- Wall targets are values at three physical triangle quadrature points; for the actual U-bend these must be obtained from the established point-3 Spalding / trace-log machinery, not independently approximated.
- `Cell` coefficients are frozen-element placeholders for the current assembly parity test, **not** a completed quadrature-dependent SST transport model.

## Progress — native mesh and constitutive access

- User has confirmed `DG_OMEGA_GPU_PARITY_ALL_PASS` on RTX 3060, all four flux scenarios, worst tested absolute block-entry difference approximately `1.8e-12`.
- New `dg_omega_hxt1_topology.hpp` builds BSR4 connectivity from native HXT1 pure tetrahedral mesh and classifies exterior faces from native inlet/outlet tags, assigning remaining exterior faces to wall only in supported single-wall cases.
- `tests/RUN_TOPOLOGY.sh <HXT1_mesh.bin>` verifies topology, geometry and face incidence; unverified on full benchmark until workstation executes.
- `dg_omega_field_access.hpp` evaluates truly discontinuous DG-P1 omega and physical gradients from `omegaDG[4*cell+a]`. Its API is explicitly not a CG projection; `tests/RUN_FIELD_ACCESS.sh` verifies independent traces.
- These are **unintegrated modules**: native G2 SST, k transport, momentum and wall kernels still consume their old CG vectors. Do not turn on DG execution in G2 yet.
- Remaining steps: quadrature-dependent SST coefficient evaluation, full 5-point trace-log wall target on DG physical-face quadrature, GPU-resident FGMRES (4x4 block preconditioned), G2 constitutive field access across all kernels, restart and adaptive refresh, true residual monitoring, validation. Existing CG path must remain untouched.

## Full-mesh verified geometry / new device FGMRES — 2026-11-11

- User confirmed actual U-bend HXT1 topology: 199584 tets, 387792 interior faces, 22176 walls, 288 inlet, 288 outlet, 975168 BSR4 blocks, 212224900 bytes predicted (about 202.39 MiB, matrix+topology estimate).
- User confirmed GPU matrix parity and DG cell-local field-access tests pass.
- Newly committed `modules/transport/dg_omega_gpu_fgmres.cuh`: right-preconditioned GPU BSR4 FGMRES, CUDA SpMV, device Krylov vectors, cuBLAS norms/dots/axpys, second Gram-Schmidt pass and true residual per restart; 4x4 element-block Jacobi.
- Newly committed `modules/transport/tests/dg_omega_fullmesh_smoke.cu` and `RUN_FULLMESH.sh`: real HXT1 topology, assembled frozen-coefficient diffusion+reaction, constant omega=100 manufactured solution, zero velocity, same trace_gamma 1e5. Validates exact constant solution consistency, FGMRES true residual, finite omega field, error and memory/timing.
- This test is **pending compilation and execution on the user's workstation**. It is *not* a k-omega SST time/outer iteration.
- Example:
  ```bash
  cd ~/NodalS_DG_OMEGA
  git pull --ff-only
  ROOT=turbulence/sst_gpu_v2/source/puretet_profile5_eq5_hxt5b/modules/transport
  MESH=/home/jd/NodalS_GPU_GENROBIN_ADAPTIVE/turbulence/sst_gpu_v2/.cache/meshes/377e0b095dd01eed708b/HXT1_mesh.bin
  bash "$ROOT/tests/RUN_FULLMESH.sh" "$MESH" --rtol 1e-6 --restart 20 --max 250
  ```
- The large trace penalty may harm conditioning; test realistic gamma here before changing the active wall penalty. A successful constant field does not validate physical wall modeling or sharp omega transport.
- **No G2 CG-SST path has been modified.** Actual DG omega coupling to the physical wall/SST coefficient kernels remains required.

## Inexact DG linear solve comparison (RTX 3060 U-bend)

User's real full-mesh result with frozen uniform omega=100, nu=1.460734693878e-5, reaction 0.1, traceGamma=1e5:
- `DG_OMEGA_FULLMESH_CONSISTENCY relative=2.04e-16`: exact manufactured solution satisfies assembled BSR4 equation.
- GPU FGMRES(20), block Jacobi, rtol=1e-6: 74 iterations, 0.292 s, trueRel=9.846e-7, RMS omega error=23.97, max absolute error=42.32, min omega=57.68. This is *not* acceptable manufactured-solution accuracy despite algebraic stopping.
- The RHS is stiff-wall-penalty dominated. A tiny `||Ax-b||/||b||` alone does not certify physical accuracy; report the field error separately.
- User's *real* G2 scalar policy is usually fixed **two GS sweeps**, or FGMRES rtol approximately 0.1, not rtol=1e-6. Benchmark these practical choices before deep solver tuning.
- Added `dg_omega_gpu_block_gs.cuh`: graph-colored exact 4x4 element-block forward and symmetric GS, ≤5 colors for tetra face adjacency, independently selectable sweeps and relaxation.
- Fullmesh driver `RUN_FULLMESH.sh` now accepts `--solver block-fgs|block-sgs|fgmres --sweeps 2 --relaxation 1.0 --initial 0|100 --rtol 0.1`, reporting both relative true residual and manufactured field error. Two sweeps are intentionally an *inexact* solve, not required to approximate the exact solution to 0.1%.
- Need workstation run for CUDA compile and speed; no claim of successful SGS/FGS until verified.
- **No actual SST G2 integration yet.** This is algebraic operator conditioning and inexact solver benchmarking on real mesh.

## Native SST physics standalone increment (2026-11-11)

The user has executed three full-mesh algebraic comparisons:
| solver | solve time | true relative residual | RMS omega error | omega min |
|--|--:|--:|--:|--:|
| block-FGS 2 sweeps | 0.005 s | 1.070405e-4 | 97.711 | -92.784 |
| block-SGS 2 sweeps | 0.007 s | 7.292662e-5 | 97.302 | -83.082 |
| right-FGMRES rtol=0.1 | 0.023 s | 3.129882e-4 | 99.664 | -99.273 |

**These are cold-start synthetic linear solves from omega=0**, not characteristic of NodalS's warm-started Picard iteration. The relative residual is dominated by the large physical-wall Nitsche gamma=1e5 penalty. Negative field values are unacceptable without protection.

Latest experimental additions:
- `dg_omega_gpu_limiter.cuh`: underrelaxed update followed by cell-mean preserving Bernstein-P1 positivity scaling (with explicit separate audit of subfloor cell means), and vertex averaging exclusively for visualization. Never feed vertex-averaged omega back into SST.
- `dg_omega_sst_volume.cuh`: native HXT1 tetra element-local omega, native P1+BF3 velocity (including all four face bubble DOFs), existing vertex-Q1 k, explicit Stage20-style wall distance. Evaluate `nodals::sst2003m::evaluate_at_distance<double>` at four degree-two physical tetra quadrature points. Assemble omega volume diffusion, reaction and source. The result's cell-mean diffusivity/velocity feeds existing DG-P1 face kernel. The complete G2 code still requires 24-point volume quadrature/face coefficient correctness and exact trace-log wall target. **This is not yet G2-coupled**.
- `tests/dg_omega_sst_smoke.cu`, `tests/RUN_SST_STANDALONE.sh`: use real 199584-tet HXT1 mesh with deliberately synthetic zero velocity, constant k=0.01, initial omegaDG=100, synthetic constant wall distance=0.01. Assemble real SST coefficients on GPU, benchmark warm-started two block-FGS sweeps with omega UR=0.7 and positivity; print audits.
- This standalone SST test is pending CUDA compilation/execution on the user's workstation. The synthetic wall-distance field is **not** an acceptable substitute for G2 Stage20 wall distance in a U-bend simulation.
- The CG G2 path is untouched. The next important milestone is a complete G2 coupling that uses DG omega in **all** SST physics, momentum, k and wall sampling. A partial CG projection is not equivalent physics.

## First integrated native G2 SIMPLE dispatch — source committed, **not compiled** (2026-11-11)

The real-mesh warm SST pilot passed on RTX 3060:
- 199584 tets / 798336 DG unknowns, assembly 0.0149 s, warm-started block FGS(2) 0.0031 s.
- Under-relaxed DG omega range 93.7592..100.002; limitedCells=0, subfloorMean=0.
- True initial scalar residual 3.7745e-5, post-sweep 4.6033e-5 (slight *increase*); do not interpret 2 sweeps as algebraic convergence.

**The experimental SIMPLE-loop integration is now in source, awaiting first workstation CUDA compile and run:**
- `NODALS_DG_OMEGA=1` opt-in in native G2 application, all defaults remain CG.
- Persistent independent `omegaDG[4*tet+localVertex]`, 4x4 BSR4 SIPG/upwind, two warm block-FGS sweeps, omega UR and mean-preserving positivity.
- DG constitutive binding for actual tetrahedral k equation, standard/deviatoric momentum, implicit deviatoric stress, inlet momentum operator, wall function interior samples, physical momentum wall and SST statistics; no averaged CG field may be used in those patched providers.
- Physical 12-quadrature-point adaptive point3 omega target and effective diffusivity captured from **the original five-point Spalding** source; actual symmetric DG wall Nitsche replaces CG omega wall assembly. Interior face penalty C_IP=10, *distinct from physical wall trace gamma*.
- Stage20 actual geometric wall distance is mandatory, including DG omega volume and DG momentum constitutive feedback. G2 nonlinear relative omega update uses the true DG degrees of freedom.
- The existing CG omega buffer is projected from DG **only** for VTU/legacy output; Stage31 restart is not compatible and is **explicitly rejected** until DG checkpoint support is written.
- G2 gate currently requires pure-tet HXT1; `sst.state_space=q1` and `compact_p1=false`; `omega_wall.method=of_auto` and `stage41_mode=trace_log`, `omega_wall.production=false`; shared Spalding wall state; geometric wall distance.
- Runtime `STAGE31_CHECKPOINT_EVERY=0` is mandatory for an initial no-restart test. Do not run the adaptive checkpoint scheduler in this branch until DG checkpoint is implemented.
- The branch Makefiles now link cuBLAS for full 4*ntet DG omega update norms.
- **Important**: The new omega volume uses 4-point tetra quadrature for nonlinear SST coefficients and cell-mean mu on interior faces, whereas the reference CG volume uses 24-point quadrature; this is a distinct numerical approximation to be evaluated once an integrated run works.
- This is an experimental attempt; **GPU compilation, native CLI dispatch, wall launch, nonlinear convergence and physical U-bend validation have not been verified yet.**
- Build integrated binary:
  ```bash
  cd ~/NodalS_DG_OMEGA
  git pull --ff-only
  make -C turbulence/sst_gpu_v2 -j4 NVCC=/usr/local/cuda-12.2/bin/nvcc CUDA_ARCH=sm_86
  ```
- First native test should disable Stage31 restart and adaptive sampling, run 3–5 outer iterations, confirm `DG_OMEGA_G2_INIT`, `DG_OMEGA_G2_ACTIVE`, `DG_OMEGA_G2_STEP`, no field nonfinites, valid wall stats. Retain and compare the original CG baseline.

## Runtime omega DG convection flux (central/upwind blend)

Interior DG face flux in `dg_omega_bsr4.cuh` uses a Rusanov/Lax–Friedrichs dissipation coefficient `alpha=theta*abs(u_face dot n)`, where `u_face=(u_L+u_R)/2`.
- `theta=1`: current full upwind LF (default; positive dissipation).
- `theta=0`: central, no LF jump dissipation. Interior SIPG **diffusion** remains enabled.
- `theta=0.5`: half LF dissipation.
- Only omega transport changes. Momentum `[convection] form` and k scalar transport are unchanged.
- Inlet/outlet convective flux retains its established upwind/backflow handling for stability and prescribed boundary conditions.
- `[omega_dg] enabled=true` and `convection_theta=0` now mapped by `nodals.py` to native runtime environment variables without editing/rebuilding source for subsequent changes. Native G2 initialization prints `flux=LF_THETA_0+SIPG` and each step prints `convectionTheta=0.000`.
- NodalS executable must be **rebuilt once** after pulling the implementation that adds the runtime setting.
- Central convection can have significant oscillations or more positivity limiting, particularly because the current code uses 4-point SST volume quadrature, cell-averaged face coefficients, and only two block-FGS sweeps. Compare limiter/backtrack counts and wall/pressure outputs. A change in visual sharpness alone does not validate higher accuracy.
