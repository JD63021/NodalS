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
