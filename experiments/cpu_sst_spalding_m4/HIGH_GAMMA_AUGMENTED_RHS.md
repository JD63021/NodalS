# Very-large grad-div gamma and matched DG-source augmentation

Experimental branch: exp/cpu-sst-keq-grad-div-20261114

## TWO DISTINCT, SELECTABLE formulations

The old full-volume form remains the default; it successfully lowered
the 6D-tet-pipe full-space divergence and nonlinear velocity updates at
gamma=0.05, but does not use the DG inlet RHS source:

  [momentum]
  grad_div_form = volume
  grad_div_gamma = 1000000

  Gvol(u,v) = gamma * integral_Omega (div u)(div v) dx

For the strong, homogeneous pointwise incompressibility target div(u)=0,
the correct grad-div RHS is zero. This is not the pressure-space
algebraic augmentation shown in the user's screenshot. With weak/DG
inlet enforcement, Gvol does NOT automatically vanish for every state
satisfying the *discrete* constraint B(u)+source=0; this is not a mistake
in its zero RHS, but it is a different stabilization and may perturb the
discrete constrained solution.

NEW optional exact P0 constraint augmentation:

  [momentum]
  grad_div_form = constraint_p0
  grad_div_gamma = 1000000

Schur's physical continuity is B*u + source_DG = 0, i.e., c=-source_DG.
P0 pressure mass M_p has diagonal tet volumes. Momentum adds BOTH:

  A <- A + gamma B^T M_p^-1 B
  rhs <- rhs - gamma B^T M_p^-1 source_DG
       = rhs + gamma B^T M_p^-1 c.

The augmented term is a sum of rank-one local cell contributions:
  gamma / |K| * B_K^T B_K,
  -gamma / |K| * B_K^T source_K.

The same shared routine builds (physical B, reduced B, source) for both
Schur and the augmented momentum, including all DG inlet face
subtractions, prescribed inlet flux, enriched pressure-divergence
coefficients, and strong wall constraints.

The code refuses a physical/reduced B mismatch on an *active* velocity
coefficient, rather than silently constructing different constraints.
Nonenriched/no-slip projection paths are NOT yet enabled for
constraint_p0; it requires the current experimental Spalding-enriched
strong-no-slip velocity space.

This pressure-space augmentation is algebraically consistent with
the original discrete pressure constraint, but only penalizes its P0
projection. Because the observed instability has severe pointwise
divergence while P0 continuity is already small, it is not guaranteed
to reproduce the stabilizing effect of full-volume grad-div. This
difference is the central purpose of the A/B comparison.

These two modes are mutually exclusive for each run.
Neither changes the original pressure Schur definition or changes k,
omega, Spalding, advective momentum, DG inlet, pressure tolerance, or
the SIMPLE under-relaxation parameters.

## Allowed gamma range

The former arbitrary 100 m2/s maximum was REMOVED from both the CPU
case parser and experiment runner. Any finite nonnegative double is
accepted, including 1e6. A value that causes nonfinite assembled
matrices, singular 3x3 blocks or PCG instability will still fail with
the existing diagnostic, and must not be silently clipped.

VERY IMPORTANT: gamma has kinematic viscosity units [m2/s], so
1e6 is extremely large relative to nu ~1.46e-5 m2/s.
Expect much worse matrix conditioning, pressure rAU scaling and slower
convergence; no physical accuracy improvement is guaranteed.
Changing gamma while running is not supported; start a fresh run.

## Workstation update

If an old run is still executing, wait until it finishes before
rebuilding the same worktree/binary (or use a NEW worktree).
Otherwise:

  cd ~/NodalS_CPU_SST_GRADDIV
  git status --short
  git fetch origin
  git merge --ff-only origin/exp/cpu-sst-keq-grad-div-20261114

Run the one-command standalone C++ kernel tests:

  bash experiments/cpu_sst_spalding_m4/tests/RUN_GRAD_DIV_TEST.sh

Compile the updated M4 solver once with a short run:

  M4_DIAG_UR_U=0.4 M4_GRAD_DIV_GAMMA=1e6 \
   M4_GRAD_DIV_FORM=volume \
   bash experiments/cpu_sst_spalding_m4/RUN_K_EQ_AB.sh C 2

Then use SKIP_BUILD=1 to compare for example 80 iterations:

  M4_DIAG_UR_U=0.4 M4_GRAD_DIV_GAMMA=1e6 \
   M4_GRAD_DIV_FORM=volume SKIP_BUILD=1 \
   bash experiments/cpu_sst_spalding_m4/RUN_K_EQ_AB.sh C 80

  M4_DIAG_UR_U=0.4 M4_GRAD_DIV_GAMMA=1e6 \
   M4_GRAD_DIV_FORM=constraint_p0 SKIP_BUILD=1 \
   bash experiments/cpu_sst_spalding_m4/RUN_K_EQ_AB.sh C 80

  python3 experiments/cpu_sst_spalding_m4/COMPARE_GRAD_DIV.py C 80

Runner writes separate ..._gd1e6_volume_... and
..._gd1e6_constraint_p0_... directories. To edit a generated case,
xdg-open its corresponding .../experiment.case, or insert keys in
any normal case manually, then invoke the M4 binary directly.

## Tests / caveats

Added:
  - full-volume 8-mode and 12-mode rank-one PSD symmetry tests;
  - algebraic augmentation test with gamma=1e6 and nonzero DG
    source, exact residual identity A_aug*u-b_aug=gamma*B^T Mp^-1
    (B*u+source), zero-gamma identity and positivity.

The code and test are committed but the assistant cannot compile the
full MPI build against the user's workstation toolchain. Run the
standalone test and 2-iteration smoke before a long or high-gamma run.
