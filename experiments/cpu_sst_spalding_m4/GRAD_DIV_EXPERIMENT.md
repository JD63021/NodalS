# CPU M4 SST — Full-Space Grad-Div Experiment (2026-11-14)

Branch: exp/cpu-sst-keq-grad-div-20261114
Parent: exp/cpu-sst-keq-mode-breakdown-20261114
Separate worktree recommended. Existing CPU/GPU branches unaffected.

## Motivation and finding

On the 1857-tet short pipe (D=0.0889 m, L=6D, Re=72600),
the original equilibrium-k and sample-log omega tests diverge at
outer iterations ~80-160 despite good SIMPLE integrated continuity.
At it40, x~2.17D: |u|/Ub=3.66, with contributions (P1,BF3,Spalding)
~(0.096, 0.047, 3.55) and large local div. At it60, x~1D,
local divergence normalized by Ub/D reaches ~-177.9.

Masked Q1 (P1+BF3 row-L1 with BF3 scalar coefficients held zero) is
active for k and omega. The velocity equation uses active P1+BF3+
shifted-Spalding enrichment, and P0 pressure constrains cell-mean
divergence, not every pointwise divergence mode.

## New optional momentum term

  [momentum]
  grad_div_gamma = 0.05

In kinematic, density-normalized NSE units:
 A^GD_{(a,i),(b,j)} = gamma ∫_Omega
                 (∂_i phi_a)(∂_j phi_b) dV
 resulting in gamma*(div u_h, div v_h), gamma [m2/s].

Default gamma=0.0. The parser requires gamma finite and nonnegative, with no arbitrary upper limit (including 1e6).
There is NO change to convection, original deviatoric stress,
turbulent viscosity, k/omega, pressure B/B^T, strong no-slip, DG inlet,
Spalding fit, SIMPLE relaxation, or physical inlet/outlet flux.

On fully Spalding-enriched cells, 12x12 scalar velocity shapes (36x36
vector block) are assembled on the SAME elevated 6x6xradial quadrature
used by enriched momentum, including full 8 PP + 4 E modes and ALL
P1/BF3/Spalding cross-blocks; grad-div added to Kextra only, not to
the original K. On cells without active Spalding modes, all 8 P1+BF3
velocity shapes receive the term via the same 5x5x5 Duffy quadrature
used by advective momentum.

The rank-one quadratic form on each point is symmetric PSD:
   x^T A^GD x = gamma*weight*(div u_h)^2 >= 0.
At zero gamma, assembly is skipped; the original assembled momentum
matrix is retained. Strong constrained DOFs undergo the pre-existing
elimination, row-L1 relaxation and Schur diag/rAU update with the new
matrix. The B and B^T pressure coupling remain the original operators.

This is a consistent Galerkin grad-div penalty for exactly solenoidal
flows, though large gamma changes the finite-dimensional
incompressibility approximation and may alter the discrete solution.
It is NOT the future algebraic B^T M_p^{-1} B stabilization;
the latter would only penalize the projection of divergence into P0
and may fail to suppress the unresolved pointwise divergence modes.
A richer divergence projection, local reconstruction, or full
quadrature stabilization would be needed for equivalence.

## Built-in diagnostics

  NODALS_CPU_M4_GRADDIV_CONFIG gamma=... enabled=0|1
  NODALS_CPU_M4_GRADDIV_AUDIT it=...
    divL2=... divRMS_D_over_U=...
    nearWallRMS_D_over_U=... coreRMS_D_over_U=...
    enrichedRMS_D_over_U=... maxAbsDiv_D_over_U=...
  NODALS_CPU_M4_VELOCITY_MODES ...
  NODALS_CPU_M4_AXIAL_SPIKE_BIN ...
  NODALS_CPU_M4_OUTLET_BACKFLOW ...
  NODALS_CPU_M4_K_POINT ...
  NODALS_CPU_M4_OMEGA_POINT ...
  SIMPLE it=... dp=... contRel=...

The L2 divergence is integrated on consistent 5^3 and elevated
6x6xradial quadrature with full physical velocity gradients.
The same five-point per-tet axial hotspot diagnostic is also retained.

## Workstation commands

Create isolated checkout:

  cd ~/NodalS
  git fetch origin
  git worktree add -b exp/cpu-gd-local-20261114 \
    "$HOME/NodalS_CPU_SST_GRADDIV" \
    origin/exp/cpu-sst-keq-grad-div-20261114
  cd "$HOME/NodalS_CPU_SST_GRADDIV"

Standalone matrix regression (requires g++):

  bash experiments/cpu_sst_spalding_m4/tests/RUN_GRAD_DIV_TEST.sh

One-rank M4 compile / 2-step smoke:

  M4_DIAG_UR_U=0.4 M4_GRAD_DIV_GAMMA=0.05 \
    bash experiments/cpu_sst_spalding_m4/RUN_K_EQ_AB.sh C 2

One-to-one 80-SIMPLE-step comparison:

  M4_DIAG_UR_U=0.4 M4_GRAD_DIV_GAMMA=0 \
    SKIP_BUILD=1 bash experiments/cpu_sst_spalding_m4/RUN_K_EQ_AB.sh C 80

  M4_DIAG_UR_U=0.4 M4_GRAD_DIV_GAMMA=0.05 \
    SKIP_BUILD=1 bash experiments/cpu_sst_spalding_m4/RUN_K_EQ_AB.sh C 80

  python3 experiments/cpu_sst_spalding_m4/COMPARE_GRAD_DIV.py C 80

If it reduces divRMS without destabilizing velocity, try gamma=0.5
and 80 or 100 steps. If gamma causes pressure PCG issues, inspect
matrix/Schur conditioning rather than silently changing pressure
tolerances. Small time histories at 20/40/60/80 are more informative
than a final 500-iteration value.

M4_GRAD_DIV_GAMMA is optional environment override in the experiment
launcher; it writes the [momentum] key into the generated .case.
One can also edit any normal .case manually, e.g. via xdg-open;
the solver parser handles the key without any wrapper.

The launcher creates separate directories with _gd<gamma>_ tags.
No source is edited or rebuilt when using SKIP_BUILD=1 after initial
compilation.

## Status

Code committed, statically checked. The assistant cannot access the
user's workstation or its MPI toolchain; full MPI compile and tests
must be executed there. The standalone unit test was added to the
branch but has likewise not yet been executed on that workstation.

## 2026-11-14 extension: exact DG-inlet RHS augmentation

See HIGH_GAMMA_AUGMENTED_RHS.md for selectable [momentum] grad_div_form=volume or constraint_p0, including the mandatory -gamma B^T Mp^-1 source_DG RHS term for the constraint_p0 mode.
