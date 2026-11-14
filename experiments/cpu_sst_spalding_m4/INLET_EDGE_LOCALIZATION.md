# Does the 1857-tet enriched pipe instability START at its DG inlet/wall intersection?

Branch: exp/cpu-sst-keq-inlet-localization-20261114
Base: exp/cpu-sst-keq-diagonal-audit-20261114
This branch adds NO numerical-operator or boundary-condition changes.

## Why investigate this?

Re = 72600, D=0.0889 m, L/D=6, momentum P1+BF3+
Spalding enrichment, pressure P0, masked P1+BF3 k and omega,
DG inlet / strong physical wall no-slip.
Cases B/C with equilibrium k and omega point3 sample_log initially have
plausible wall effective stress, then their velocity grows catastrophically.

M3 inlet is **weak** DG/Nitsche prescribed velocity (momentum source and
inflow term) with NO strong facewise equality u_inlet=Uin.
The pressure continuity operator subtracts the velocity inlet face
contribution and uses prescribed Uin·n as boundary flux; therefore
the small SIMPLE "contRel" does NOT validate the independently
reconstructed velocity trace on the inlet.

At C it=80: inlet reconstructed flux approx -0.072474 vs
prescribed -0.072157, but velocity norm already ~312.
At C it=100: flux approx -0.072684 vs -0.072157,
velocity norm ~857, pressure drop ~314.
At C it=120: reconstructed inlet flux approx -0.037946;
at it140 sign of reconstructed inlet flux reverses.
This large flux mismatch may be a CONSEQUENCE or a DRIVER of runaway;
aggregate flux alone cannot localize where the first nonphysical
gradient appears.

The inlet-wall junction can exacerbate a plug-velocity inlet + strong
no-slip wall conflict at the edge. The wall law uses 1920 pointwise
ray/Spalding fits including wall faces adjacent to the inlet.
Those cells have very coarse resolution, and the traces at the
inlet-wall intersection are not forced to satisfy the plug profile.

## Passive native diagnostic (full enriched velocity, not vertex VTU)

At iteration 1 and each 20 iterations:
- NODALS_CPU_M4_INLET_TRACE_BAND: inlet-face **quadrature** RMS deviation
  of full enriched velocity from prescribed Uin, split into a
  wall-edge (distance <=0.1D) and core (>0.1D) band.
  Also min/max axial velocity and actual/prescribed inlet flux for
  each band; local errors may cancel in the global net flux.
- NODALS_CPU_M4_AXIAL_SPIKE_BIN: samples 5 interior locations per
  tetrahedron, including the enrichment. Axial bands:
  0-.25D, .25-.5D, .5-1D, 1-2D, 2-4D, and 4D to outlet.
  Reports max speed/U, minimum axial speed/U, reverse-flow sample
  fraction and cell/z/wall-distance of maximum.
- NODALS_CPU_M4_WALL_FIT_BIN: face-quadrature fitted Spalding uTau,
  binned by axial location. Since wall fit is constructed with
  *lagged* velocity, these values represent previous SIMPLE velocity.
- NODALS_CPU_M4_INLET_LOCALIZATION: location of worst wall uTau,
  and largest P1 k/omega sampled at interior tetrahedra.

No mesh or boundary modification, no extra stabilization,
no ad hoc clipping, and no changes to scalar or momentum equations.

## Workstation

    cd ~/NodalS
    git fetch origin
    git worktree add -b exp/cpu-inlet-local-local-20261114 \
      "$HOME/NodalS_CPU_SST_INLET_LOCAL" \
      origin/exp/cpu-sst-keq-inlet-localization-20261114
    cd "$HOME/NodalS_CPU_SST_INLET_LOCAL"

Compile and run C to 100 (avoids the late Spalding-fit crash):

    M4_DIAG_UR_U=0.4 bash experiments/cpu_sst_spalding_m4/RUN_K_EQ_AB.sh C 100

Inspect (latest run shown by path from RUN_K_EQ_AB):

    grep -E 'INLET_LOCALIZATION|INLET_TRACE_BAND|AXIAL_SPIKE_BIN|WALL_FIT_BIN|^SIMPLE it=' \
      "$HOME/NodalS_SST_LOCAL_RUNS/CPU_M4_K_EQ_AB_20261114"/C_100_uru0.4_*/console.log

If the near-inlet band spikes early while downstream bins remain near
U, the inlet/wall geometry + DG inlet is implicated. If first spikes
occur downstream, a pure inlet hypothesis is weakened. If core
inlet trace and wall-edge region are both bad, we should separate
the strength/consistency of the inlet DG terms from wall sampling.

Native diagnostic spatial samples are NOT exact global maxima; 5
tetrahedral points may miss narrow peaks. Inlet error uses 6x6
face integration, not an interpolation of vertex-only VTU.
No numerical convergence is claimed.

Assistant added and statically reviewed this branch but did not
compile/run MPI on the user's workstation. Inspect output before
making any changes to DG penalty, strong inlet, or SUPG.
