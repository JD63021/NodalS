# Experimental CPU M4 enriched omega point3 modes (2026-11-14)

Source branch: exp/cpu-sst-enriched-omega-point-modes-20261114
Base: exp/cpu-sst-spalding-enrichment-tiny-pipe-20261114

## What changed (omega transport only)

The legacy \`omega_wall.method=of_auto\` +
\`omega_wall.stage41_mode=trace_log\` branch remains unchanged.

Two NEW stage41 modes are implemented for masked P1 SST omega:

1. \`of_auto\`: use the shared GPU \`evaluate_of_auto<double>\` viscous/log blend.
2. \`sample_log\`: use the same Spalding friction velocity but just the logarithmic omega target.

Both modes **insert a rank-one penalty on an interior ray sample** (third omega sample) in the tetrahedron actually hit by the ray. No physical-wall trace-Nitsche omega term is inserted in these modes. The P1 omega tests and P1+BF3 masked trial columns are exactly retained, consistent with the existing CPU masked-scalar transport architecture.

The GPU source
\`modules/wall/of_auto_point3_rank1_penalty.inc\` specifies
\`coefficient = gamma*(nu_eff/ell)*(wall_area*face_weight*ell)/ell\`,
where \`ell=y5-y1\` (the complete five-point fit span); the CPU code uses
the algebraically equivalent \`gamma*nu_eff*wall_area*face_weight/ell\`.

In the 1857-tet case, \`ell=0.0074687304-0.0001244788=0.0073442516 m\`.
The omega sample remains at \`y=0.0009797675557402 m\` (omega.sample_scale=.1).

The GPU portable OF-auto target blends
\`omegaVis=6 nu/(beta1*y^2)\` with
\`omegaLog=uTau/(sqrt(0.09)*0.4*y)\`.
The viscous blend weight is \`exp(-y sqrt(k)/(nu*ReBlend))\`,
where \`ReBlend=11\` by default.

At high sampled k the OF-auto target approaches sample_log.
This makes the **off-wall imposition** the important physical difference
from trace_log on the former solution.

No changes to velocity enrichment, pressure Schur, k zero-flux boundary
condition, SST volume physics, or wall drag have been made in this branch.

## How to run

Create a fresh worktree:

    cd ~/NodalS
    git fetch origin
    git worktree add -b exp/cpu-omega-point-local-20261114 \
      "$HOME/NodalS_CPU_SST_OMEGA_POINT" \
      origin/exp/cpu-sst-enriched-omega-point-modes-20261114
    cd ~/NodalS_CPU_SST_OMEGA_POINT

Preflight (no MPI required):

    bash experiments/cpu_sst_spalding_m4/tests/RUN_OMEGA_TARGET_TEST.sh

Compile and test 2 outer SIMPLE steps with gamma=10000:

    bash experiments/cpu_sst_spalding_m4/RUN_OMEGA_POINT_1000.sh of_auto 2 10000

After inspecting logs for positive Schur diagonal and stable PCG/FGMRES,
use the compiled binary for 20, then 1000 iterations:

    SKIP_BUILD=1 bash experiments/cpu_sst_spalding_m4/RUN_OMEGA_POINT_1000.sh of_auto 20 10000
    SKIP_BUILD=1 bash experiments/cpu_sst_spalding_m4/RUN_OMEGA_POINT_1000.sh of_auto 1000 10000

And the matched sample_log run:

    SKIP_BUILD=1 bash experiments/cpu_sst_spalding_m4/RUN_OMEGA_POINT_1000.sh sample_log 1000 10000

The runner generates a unique per-run editable .case. If desired, edit
that file and run the stored binary manually; the original Stage1 case
is not modified.

The runner switches solver.u.method from 3 fixed FGS sweeps, which
IGNORE momentum rtol, to true FGMRES rtol=1e-5, restart=40,
max_iterations=600; PCG rtol=1e-4 max=800; k and omega 6
fixed scalar sweeps each. It sets minimum pressure-drop iteration
to steps+1 to prevent the prior 1% window plateau (at it=245)
from prematurely stopping the requested 1000-step experiment.
Expected end status is MAX_ITERATIONS even if physically flat.

The 1e4 offwall penalty can cause numerical stiffness; if FGMRES/PCG
fails, report the exact first failing diagnostic rather than hiding it.
To isolate strength effects, compare gamma=1000 and gamma=10000 with
otherwise identical cases.

## What to check

\`NODALS_CPU_M4_OMEGA_POINT\` reports mode, gamma, effective length,
mean target, blending fraction, and Re_y every 20 iterations.
\`NODALS_CPU_M4_ENRICH_WALL\` continues to report molecular wall
shear and wall no-slip; \`SIMPLE it=...\` gives linear residuals,
continuity and pressure drop.

A stable run is NOT automatically a valid physical pipe-friction
calculation. Compare \`dp\`, molecular/effective wall shear, k/omega/nut,
and axial momentum budget; developing-pipe inlet/outlet fluxes matter.

IMPORTANT: The assistant runtime cannot compile or run this MPI code.
Implementation and static source checks are complete; compile and
numerical convergence are unverified until a workstation run.
