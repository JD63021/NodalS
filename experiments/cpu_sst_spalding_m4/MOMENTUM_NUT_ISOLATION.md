# CPU M4 diagnostic: isolate large turbulent wall resistance

Branch: exp/cpu-sst-momentum-nut-isolation-20261114
Base: exp/cpu-sst-enriched-budget-debug-20261114

The user's 20-iteration sample_log gamma=10000 run at Re=72600 shows:
  it=1: mean molecular tau=0.36293, effective tau=0.61342, wall nuT/nu mean=0.756.
  it=20: mean molecular tau=0.07536, effective tau=9.64435, wall nuT/nu mean=265.9.
  it=20: pressure force=+1.04295, modeled wall stress z=-1.32314,
         approximate budget discrepancy=-0.37718 (28.7%).
  it=20: prescribed inlet flux=-0.0721571518, outlet=+0.0721570887;
         reconstructed inlet flux=-0.0707801286 (1.9% discrepancy
         because inlet DG condition imposes numerical flux, not a strong trace).
This makes large modeled near-wall SST viscosity the main candidate for excessive pressure drop.
The physical reference fully developed Darcy friction factor is only around 0.0193,
while observed apparent friction at it=20 is around 0.404.

New knob: [sst] momentum_nut_scale = 1.0 (default).
- 1.0: standard model, numerically identical to the parent diagnostic branch.
- 0.01: only 1% of νt acts in momentum viscous terms, while k/omega and
        SST nut remain fully calculated.
- 0: only molecular ν acts in momentum, while k/omega remain calculated.

IMPORTANT: these diagnostic values <1 are **not physically consistent SST**.
Do not publish friction factors or infer a validated RANS method from them.
This isolates the coupling responsible for the pressure loss.

Implementation consistency:
- Applies the factor to the non-enriched deviatoric PP viscosity increment.
- Applies it to the entire enriched PP/PE/EP/EE volume viscosity block.
- Applies it to both enriched and original inlet Nitsche viscosity coefficients.
- Leaves SST scalar production, destruction, transport, omega penalties,
  k zero-flux, and turbulence variables unchanged.
- Diagnostics log unscaled raw νt and physically active momentum-scaled τ_eff.

## Workstation commands

  cd ~/NodalS
  git fetch origin
  git worktree add -b exp/cpu-nut-isolation-local-20261114 \
    "$HOME/NodalS_CPU_SST_NUT_AB" \
    origin/exp/cpu-sst-momentum-nut-isolation-20261114
  cd "$HOME/NodalS_CPU_SST_NUT_AB"

Run 20-step scale=0 (compile first):
  bash experiments/cpu_sst_spalding_m4/RUN_OMEGA_POINT_1000.sh sample_log 20 10000 0

Then 20-step scale=0.01:
  SKIP_BUILD=1 bash experiments/cpu_sst_spalding_m4/RUN_OMEGA_POINT_1000.sh sample_log 20 10000 0.01

Optional: 20-step control with scale=1:
  SKIP_BUILD=1 bash experiments/cpu_sst_spalding_m4/RUN_OMEGA_POINT_1000.sh sample_log 20 10000 1

If stable, repeat to 100 steps for scales 0 and 0.01 (no 500-step run yet).
Compare:
  NODALS_CPU_M4_RANS_WALL_BUDGET
  NODALS_CPU_M4_AXIAL_MOMENTUM_BUDGET
  SIMPLE it=...
  pressure drop, k, omega, wall/raw nut, mass and update residuals.

If scale=0 causes pressure drop to collapse toward O(10) but scale=1
produces O(100) within 20 steps, the excessive turbulent viscosity
contribution to the momentum matrix is implicated. However the model
still must be repaired by a *consistent* SST wall treatment, not by
the arbitrary momentum-only scale.

If pressure remains huge at scale=0, investigate inlet DG
momentum consistency and pressure/divergence projection next.

The assistant could inspect and edit source via GitHub but cannot
compile this MPI branch on the workstation itself. Do not mistake
static checks for numerical verification.
