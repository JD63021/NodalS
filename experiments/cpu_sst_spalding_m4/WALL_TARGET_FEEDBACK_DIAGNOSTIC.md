# Diagnose dynamic Spalding uTau wall-target feedback versus momentum instability

Branch: exp/cpu-sst-keq-wall-feedback-isolation-20261114
Base: exp/cpu-sst-keq-diagonal-audit-20261114

## Evidence and hypothesis

The 1857-tet Re=72,600 pipe with full SST and off-wall equilibrium-k plus
sample-log omega gives physically more plausible quantities at it=20-60,
then diverges violently despite alphaU=0.4 and converged linear subsolves.

Variant C:
  it60: dp~17.68, velocityNorm~244, wall k~1.53
  it100: dp~314, velocityNorm~857, wall k~12.87
  it120: dp~7873, velocityNorm~5251, wall k~263
  it160: dp~610425, velocityNorm~102061, wall k~76326
The sampled omega and k targets both derive from dynamically fitted
Spalding uTau; as the velocity becomes unstable this creates positive
feedback. It is not proven whether feedback is the root instability or
only amplifies an unstable steady advective momentum discretization
without SUPG on an extremely coarse high-Re mesh.

## Controlled change

New optional knob:

    [sst]
    debug_fixed_target_utau = 0.5859

When supplied, only the **offwall k equilibrium** and **omega sample_log**
targets use this fixed friction velocity instead of fitted qUTau.
Everything else is unchanged, including:
- actual face Spalding fit (still computed for momentum);
- physical wall velocity field and enriched shape;
- momentum full nu + nuT; no artificial viscosity scaling;
- k/omega volume production, destruction, diffusion and advection;
- k/omega rank-one penalty strength and ray locations;
- SIMPLE, pressure coupling, ur_u, and linear tolerances.

This fixed target does NOT represent an acceptable final physical wall
function. It is intentionally a sensitivity diagnostic to isolate
feedback in dynamically fitted scalar wall targets.

Default with knob absent remains the original fully dynamic target.
Output NODALS_CPU_M4_K_POINT and OMEGA_POINT include
wallTargetUTau=FIXED|SPALDING_DYNAMIC.

## Workstation commands

  cd ~/NodalS
  git fetch origin
  git worktree add -b exp/cpu-wall-feedback-local-20261114 \
    "$HOME/NodalS_CPU_SST_WALL_FEEDBACK" \
    origin/exp/cpu-sst-keq-wall-feedback-isolation-20261114
  cd ~/NodalS_CPU_SST_WALL_FEEDBACK

Compile and compare:
  M4_DIAG_UR_U=0.4 M4_DIAG_FIXED_UTAU=0.5859 \
    bash experiments/cpu_sst_spalding_m4/RUN_K_EQ_AB.sh C 120

If the frozen-target branch still exhibits growing velocity updates,
inlet DG trace discrepancies, and pressure loss despite bounded
k/omega target, the primary instability is NOT the target feedback.
Investigate unstabilized advective momentum and actual inlet flux
consistency next (and then use fully consistent full-residual momentum
SUPG or energy-stable alternative).

If fixed targets prevent the runaway for 120 steps, that implicates
positive feedback of the dynamic Spalding uTau target; diagnose
target relaxation or a consistent coupled wall-law Jacobian, not
simply clipping physical uTau.

Compare dynamic baseline to frozen-target case in separate run folders.
NEVER use fixed target result to claim a validated SST solution.

A fast 60-step smoke is useful first if desired, followed by 120 steps.
The changed branch is code-reviewed but has not been built or run
with MPI on the assistant's machine; workstation execution required.
