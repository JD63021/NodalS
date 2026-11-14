# CPU M4 SST shared shifted-Spalding wall enrichment — EXPLICIT EXPERIMENT

Branch: exp/cpu-sst-spalding-enrichment-tiny-pipe-20261114
Base: exp/cpu-sst-komega-multicore

This is a **source-level implemented, but GPU/CPU-MPI-UNCOMPILED and numerically UNVERIFIED** SST/SIMPLE experiment. The assistant runtime lacks MPI, so it cannot run the real binary here. The 1,857-tet unenriched M4 control was compiled and successfully run for 20 iterations on the workstation by the user.

## Construction

Velocity:
  u(x) = sum_{a=0}^7 phi_a(x) U_a + sum_{i in selected vertices} lambda_i(x)[W(y_h)-I_h W] C_i

Selection: all wall vertices + first interior radial ring via y_i*u_tau/nu <= 80.
W(y)=u_tau*SpaldingInverse(y*u_tau/nu). The initial shape u_tau=0.5859 m/s is FROZEN throughout the current test, not dynamically updated. Its coefficients C_i are unknowns of the coupled velocity system.

- Continuous C0 across neighboring tets with common global enriched vertex coefficients.
- Strong zero wall trace; ordinary wall P1 and wall BF3 modes are eliminated.
- No tangential Robin wall model, no generalized wall viscosity replacement and no Nitsche penalty in enrichment mode.
- Every enriched tetrahedron's complete 12-mode PP/PE/EP/EE momentum volume block is integrated with one 6×6×N Duffy quadrature; no cross-block quadrature mismatch.
- The full implicit deviatoric stress and lagged advective convection enter the augmented operator.
- High-order triangular integration of enriched inlet DG terms.
- Pressure P0 B and B^T contain enriched flux integrals; resulting row-L1 diagonal rAU and assembled Schur contain the extra velocity modes.
- SST production / nut use complete grad u; scalar advection includes additional enriched carrier.
- Spalding face sampling and omega trace receive total enriched velocity.
- No enrichment in k or omega polynomial spaces: masked P1 scalars retained.

### Physical checks printed

NODALS_CPU_M4_ENRICH_CONFIG ...
NODALS_CPU_M4_ENRICH_WALL it=N physicalWallVelocityMax=... meanMolecularTau=...
NODALS_CPU_M4_ENRICH_UPDATE it=N velocityUpdateRel=...
SIMPLE it=N ...
SIMPLE_RESULT ...

The test fails if actual wall velocity exceeds 1e-10 m/s or if solver detects a nonfinite field. Do not infer global convergence from a pressure-drop plateau.

## Run on the workstation

You have already extracted Stage 1 in:
  ~/NodalS_CPU_ENRICH_20261114

Create a NEW worktree for this remote source. Avoid overwriting Stage1's existing patches/untracked files:

    cd ~/NodalS
    git fetch origin
    git worktree add "$HOME/NodalS_CPU_SST_ENRICH_M4" origin/exp/cpu-sst-spalding-enrichment-tiny-pipe-20261114
    cd "$HOME/NodalS_CPU_SST_ENRICH_M4"

Build-and-run 2-iteration experimental gate:

    bash experiments/cpu_sst_spalding_m4/RUN_TINY_PIPE.sh enriched 2

Then 20 iterations, with compiled binary reused:

    SKIP_BUILD=1 bash experiments/cpu_sst_spalding_m4/RUN_TINY_PIPE.sh enriched 20

For a same-branch 20-step Robin control:

    SKIP_BUILD=1 bash experiments/cpu_sst_spalding_m4/RUN_TINY_PIPE.sh control 20

The launcher copies the verified Stage1 .case to a unique RUN output folder,
changes simple.max_iterations, and switches enrichment only for the enriched run.
Inputs are drawn from ~/NodalS_CPU_ENRICH_20261114/experiments/cpu_sst_spalding_tiny_pipe.

## Required gating

1. M4 compiles on Ubuntu with mpicxx; compile diagnostics must be reviewed.
2. Nonzero number of added velocity DOFs, correct pressure dimensions.
3. 2-iteration gate: finite momenta, SST k/omega, positive pressure Schur diagonals, PCG convergence, exact wall velocity trace.
4. 20-step gate: monitor mean molecular wall tau, velocityUpdateRel, pRel, k/omega update, continuity, pressure drop, runtime.
5. If stable, increase to 100/200 iterations; never use 6D entrance pressure drop as Moody reference.
6. After evidence of stability, update frozen Spalding u_tau with a projected wall-shear field and reproject velocity between changes.
7. For long pipe accuracy checks, use 30D–50D with finer circumferential resolution.

Important unresolved physics: wall-shape u_tau is frozen uniform and SST wall model has not been fully validated after strong no-slip replacement; the independent 3D serial enriched mixing-length solver is not a proof of SST stability. A later robust implementation should update the enrichment shape with a consistent wall stress field.

## Safety

- Default .case enrichment.enabled=false preserves original M4 numerical route.
- Existing branch, GPU source and original experimental branch are not modified.
- Changes are committed only on this isolated CPU branch.
- No claim of a successful full CPU M4 enriched run before actual workstation build and run.
