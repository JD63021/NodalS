# Experimental Option A V3: P1+BF3 CUDA projected wall operator (2026-11-14)

Separate branch: \`exp/of12-v3-projected-wall-20261114\`. Original V2 remains available as \`option_a\`; original no-slip SST as \`noslip_sst\`. V3 is \`option_a_v3\`. No change to existing baseline default.

This is **a CUDA implementation, not yet a compiled/validated GPU result**. The earlier three numerical tests were local Python model tests only. The first CUDA smoke must establish real compilation, M3/CSR parity, valid frozen wall coefficients, and subsequent outer stability.

## Discrete operator

For each wall-owner tetrahedron, freeze Spalding \`u_tau\` from the existing F2 five-point face fit. The existing 12-point face cache \`bp\` defines the same first-tet normal-column average as in the CPU prototype. The tangential face-average velocity is \`Ubar_t = P_t (sum_q w_q Bp_q U / A)\`; with positive speed,

\`beta = u_tau^2 / |Ubar_t|\`.

The frozen additional **local** matrix is

\`Delta K = -(C C^T) / alpha + beta * sum_q w_q Bp_q^T P_t Bp_q\`,

where (component-major spatial index)

\`C_{ai,t} = integral_T nu_eff (n.grad phi_opp) [P_t((grad phi_a e_i + e_i grad phi_a)n)]_t dV\`,
\`alpha = integral_T nu_eff (n.grad phi_opp)^2 dV\`.

The two subtracted K_s-projected modes are the tangential velocity directions of the wall-opposite P1 vertex. The same 24-point quadrature and frozen local SST viscosity as \`implicit_dev_volume_tet.inc\` are used in both paths. C and alpha are built ONCE per SIMPLE iteration; the M3 CSR and matrix-free implementations consume the same cache. Projection is on physical P1+BF3 coefficients BEFORE the existing BF3 T/Tt transforms.

Strong no slip is unchanged. The old wall Robin and Nitsche penalty are zeroed, as in V2. The regular SST dev volume operator remains, but the two K_s modes are subtracted before the wall conductance is inserted. SST k/omega transport is unchanged.

This two-mode replacement is still a modeling hypothesis requiring validation against resolved wall stress and mesh variation; it is **not** an exact OF12 finite-volume stencil.

## Run on RTX 3060 workstation

From the \`~/NodalS_OF12_WALL\` worktree:

\`\`\`bash
git fetch origin
git switch --track origin/exp/of12-v3-projected-wall-20261114
cd ~/NodalS_OF12_WALL
NVCC=/usr/local/cuda-12.2/bin/nvcc MOMENTUM_SOLVER=m3-fgs \
  bash experiments/of12_nutwall_p1bf3/RUN_STR100K.sh option_a_v3 10
\`\`\`

The launcher builds by default. **Do not set \`SKIP_BUILD=1\` until after a successful V3 build.** Case \`[solver.u] sweeps\` is respected; default M3-FGS is used unless overridden. The same 99,552-tet input package is reused.

Early native log must contain:

- \`NODALS_OPTION_A enabled=1 ... NO_SLIP_PROJECTED_WALL_V3\`
- \`NODALS_OPTION_A_V3_WALL it=1 ... model=PROJECTED_MODE_REPLACEMENT status=PASS\`
- \`NODALS_M3_OPERATOR_PARITY it=1 relative=... tol=1e-9 status=PASS\`
- \`NODALS_OPTION_A_NOSLIP_AUDIT ... wallDofRMS=0 ... status=PASS\`

The first M3/CSR versus matrix-free parity check is mandatory for V3. A failure throws rather than silently continuing.

Once the 10-step smoke succeeds, run \`option_a_v3 200\` with \`SKIP_BUILD=1\`. Keep run comparisons at identical case settings and do not treat a completed 200-iteration run as converged.

## Unverified and scientific caveats

- No CUDA compiler/GPU is available in the assistant environment. Native build, startup parity and GPU wall-force audit must be run on the workstation.
- Need compare actual wall-traction resultant, streamwise pressure gradient, momentum/continuity residuals and segment-by-segment friction factors.
- CPU local positive-eigenvalue tests do not establish global nonlinear SIMPLE convergence.
- The effect of matching at the first-tet normal-column average versus a farther Spalding sampling plane remains a modeling choice.
- Full physical proof of no double-counting is only for the two removed projected modes; other unresolved near-wall modes still retain standard SST diffusion.
