# Spalding equilibrium k at interior omega sample point: A/B/C (2026-11-14)

Branch: exp/cpu-sst-spalding-keq-point3-20261114
Base: exp/cpu-sst-enriched-budget-debug-20261114 (NOT the artificial momentum-nu_t isolation branch).

## Purpose

The user's prior Re=72600, 6D, 1857-tet pipe with shared Spalding enriched
P1+BF3 velocity and sample_log omega=10000 had approximate steady
kinematic pressure drop ~1067 and very large wall eddy viscosity.
A momentum-only νt isolation test confirmed this viscosity drives
most of the enormous modeled wall stress, but is intentionally not a
consistent SST solution.

This experiment retains **unscaled full SST** momentum and tests an
independently controlled k equilibrium wall constraint:
  A: [k_wall] method=no_flux (original natural zero-flux k wall).
  B: method=equilibrium_spalding; penalty_gamma=1000.
  C: method=equilibrium_spalding; penalty_gamma=10000.

All three: omega_wall.method=of_auto, stage41_mode=sample_log,
omega_wall.penalty_gamma=10000, point=3, fixed-k/omega sweeps=6 each;
FGMRES momentum rtol=1e-5, PCG pressure rtol=1e-4,
strong no-slip enriched momentum, frozen enrichment shape, same mesh,
same initialization, and same under-relaxation.

## Implementation

For every wall face quadrature ray, reuse the existing third omega
sampling location (sampleY=0.0009797675557402 m) and SAME dynamic
face-quadrature fitted friction velocity uTau from momentum wall fit.
The frozen shape uTau=0.5859 is **NOT** used as wall k target.

At the interior tetrahedral sample:
    k_eq = max(k_floor, uTau^2/sqrt(beta_star)), beta_star=0.09.

The rank-one point penalty is:
    K_ab += gamma_k*(nu+sigma_k*nu_t)*(A_face*QW_face/ell)
            *phi_a(x_sample)*phi_b(x_sample)
    RHS_a += gamma_k*(nu+sigma_k*nu_t)*(A_face*QW_face/ell)
             *phi_a(x_sample)*k_eq
where ell=y5-y1=0.0073442516 m (the existing five-point fit span).
Rows=4 masked P1 vertex tests; columns=8 P1+BF3 trials.
No extra k unknowns and NO wall-face penalty. The original natural
k zero flux remains when method=no_flux, and also coexists with this
interior sample constraint.

These k_eq and omega_log relations assume an equilibrium *log layer*.
They are not generally exact in the buffer layer; diagnostics report
actual sampled y+. The result is experimental and should NOT be
interpreted as validated high-Re SST wall modeling.

The case settings are:
    [k_wall]
    method = no_flux | equilibrium_spalding
    sample_index = 3
    penalty_gamma = 1000 or 10000
    # optional: penalty_length = ell (default matches omega fit span)

A mode: a physically exact no-op in scalar matrix/RHS, except for passive
post-sweep diagnostics. Default is no_flux and existing cases retain their
k wall treatment.

## Workstation run

Fresh isolated worktree:

    cd ~/NodalS
    git fetch origin
    git worktree add -b exp/cpu-keq-local-20261114 \
      "$HOME/NodalS_CPU_SST_KEQ_AB" \
      origin/exp/cpu-sst-spalding-keq-point3-20261114
    cd "$HOME/NodalS_CPU_SST_KEQ_AB"

Standalone C++ test, no MPI:

    bash experiments/cpu_sst_spalding_m4/tests/RUN_K_EQ_TARGET_TEST.sh

Compile and smoke test B for 2 SIMPLE iterations:

    bash experiments/cpu_sst_spalding_m4/RUN_K_EQ_AB.sh B 2

If it passes, run all A/B/C for 20 steps without recompiling:

    SKIP_BUILD=1 bash experiments/cpu_sst_spalding_m4/RUN_K_EQ_AB.sh A 20
    SKIP_BUILD=1 bash experiments/cpu_sst_spalding_m4/RUN_K_EQ_AB.sh B 20
    SKIP_BUILD=1 bash experiments/cpu_sst_spalding_m4/RUN_K_EQ_AB.sh C 20

To compare:

    python3 experiments/cpu_sst_spalding_m4/COMPARE_K_EQ_AB.py 20

If 20 steps behave safely, extend A/B/C to 50 or 100 to compare
the direction of pressure drop and stress; do not run 500 blindly.
The scripts always create unique run folders and keep the Stage1
mesh and case unchanged. Printed output every 20 steps and on it=1;
VTU on last iteration or every 50.

## Key output

    NODALS_CPU_M4_K_POINT mode=... gamma=...
        targetMean=... actualMean=... relativeMeanAbsError=...
        yPlusMean=... uTauMean=...
    NODALS_CPU_M4_OMEGA_POINT mode=sample_log ...
    NODALS_CPU_M4_RANS_WALL_BUDGET ...
        meanTauMol=... meanTauEff=... meanNutNu=... wallK=...
    NODALS_CPU_M4_AXIAL_MOMENTUM_BUDGET ...
    SIMPLE it=... dp=... contRel=...

Look for a reduction in actual sampled k toward dynamically fitted
equilibrium k, lower νt and effective wall shear, and a physically
sensible pressure drop *without* sacrificing continuity/velocity
updates. High penalty may increase scalar row stiffness. A gain at
20 steps is NOT a proof of steady-state accuracy.

**Compilation/MPI numerical tests have not been performed in the
assistant runtime;** run the standalone target test and 2-step
smoke before the A/B/C comparison.
