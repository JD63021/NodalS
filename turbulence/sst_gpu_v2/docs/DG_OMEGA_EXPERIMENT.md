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
