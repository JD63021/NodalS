# Why does the 1857-tet, 6D enriched SST pipe give Δp/ρ ≈1067? (diagnostic branch)

Branch: exp/cpu-sst-enriched-budget-debug-20261114
Base: exp/cpu-sst-enriched-omega-point-modes-20261114

The 500-iteration SAMPLE_LOG, gamma 10000, FGMRES/PCG run reported:
- kinematic Δp ≈1067.47548 m²/s² (relative window variation ~6.25e−4)
- molecular wall shear ~0.05223048 m²/s²
- zero physical wall slip
- pressure PCG and momentum FGMRES converged to requested linear tolerances.
- prior VTU showed cell-centred eddy viscosity near wall ~0.027 m²/s, ν=1.46e−5 m²/s.
This is too much friction (apparent Darcy ≈2.50) to interpret as normal entrance losses.

## Critical questions

1. Is the *total modeled* wall shear (ν + νT) ∂u/∂n, not molecular shear alone, of the same order as the pressure force?
2. Are near-wall k, omega and νT excessive?
3. Is there a large momentum-budget mismatch suggesting an assembly/discrete flux inconsistency instead of SST stress?
4. Is inlet DG prescribed mass flux different from reconstructed inlet flux?

The diagnostic does NOT alter momentum, turbulence, Schur, sampling, tolerances, or penalty.

Added log entries every 20 SIMPLE steps:

    NODALS_CPU_M4_RANS_WALL_BUDGET it=... meanTauMol=... meanTauEff=... tauEffToMol=... fMol=... fEff=... meanNutNu=... wallK=... wallOmega=...
    NODALS_CPU_M4_AXIAL_MOMENTUM_BUDGET it=... pressureForce=... wallStressZ=... wallStressMolZ=... endStressZ=... advOut=... advInPrescribed=... surfaceMismatch=... mismatchRel=...
    NODALS_CPU_M4_MASS_BOUNDARY it=... fluxInPrescribed=... fluxInComputed=... fluxOut=... fluxWall=...

Surface momentum budget integrates the **complete enriched gradient** and SST pointwise turbulent viscosity on the triangulated boundary. It uses adjacent P0 cell pressures on boundary faces, so it is an **approximate physical-boundary diagnostic**, not the exact FE weak residual. DG inlet numerical fluxes and nonlinear advective-form divergence terms may also contribute to discrepancy. A close balance is informative; a mismatch must be examined before declaring a discretization bug.

Offline tool reads the existing iteration-500 VTU, which lacks enriched DOF coefficients:

    python3 experiments/cpu_sst_spalding_m4/POSTPROCESS_VTU_PRESSURE_NUT.py \
      "$HOME/NodalS_SST_LOCAL_RUNS/CPU_M4_ENRICH_OMEGA_POINT_20261114/sample_log_1000_20261114_103900/flow_sst_cpu_it500.vtu" \
      0.05223047600546

It reports area-weighted inlet/outlet pressure and wall-adjacent *CELL-CENTRED* νT. That is a proxy, not an actual wall stress. The physically exact wall-gradient reconstruction is available only inside the solver.

## Workstation experiment

    cd ~/NodalS
    git fetch origin
    git worktree add -b exp/cpu-budget-debug-local-20261114 \
      "$HOME/NodalS_CPU_SST_BUDGET_DEBUG" \
      origin/exp/cpu-sst-enriched-budget-debug-20261114
    cd "$HOME/NodalS_CPU_SST_BUDGET_DEBUG"
    bash experiments/cpu_sst_spalding_m4/RUN_OMEGA_POINT_1000.sh sample_log 20 10000

This builds the modified solver, runs from the same initial conditions (no restart), and prints
budget at iterations 1 and 20. If safe and informative, extend to 100; there is no
need to repeat the 500-iteration run until the cause is understood.

## Interpretation

- If meanTauEff >> meanTauMol and net wallStressZ approximately balances pressureForce, excessive SST viscosity is the prime candidate. Investigate k and omega closure and near-wall nut; do not simply tune penalty stronger.
- If meanTauEff remains small relative to the pressure-force equivalent AND surfaceMismatch is large, audit pressure traction, DG inlet fluxes, discrete advective-form residual, and Schur/velocity coupling.
- If computed inlet flux is very different from prescribed DG inlet flux, inspect boundary momentum and kinetic-energy contributions.
- If meanNutNu is large but meanTauEff is not, the wall velocity-gradient geometry and vector stress may be inconsistent.
- If the result changes dramatically between sample_log and of_auto, the near-wall blending itself is relevant; at reported Rey~800 the blend should be mostly log.

## Limitations

The source changes have been committed and statically inspected, but *not* compiled with MPI in the assistant environment; the user must run the 20-step test on the workstation.
