# Hybrid 99k SST: M3 FGS (2 sweeps) + coupled FGMRES

**Status: uncompiled experimental solver port.** The November 14 source has
been inspected and the exact source anchors checked; no CUDA compiler or
workstation run is available in the branch authoring environment.

## Target configuration

- Re=1,000,000, D=0.889 m, L=50D
- one HEX wall ring, nominal centroid y+=150; original 99,216-cell topology
- Q1+BF2 HEX velocity; P1+BF3 TET interior; P0 pressure
- HEX/TET interior DG/Nitsche penalty gamma=50
- one-point Spalding axial HEX wall + OF_AUTO physical HEX `trace_log` omega
- SST scalar Q1 masked, existing pressure Schur/AMG, advective vector Laplacian
- **M3 FGS exactly 2 sweeps**, optionally coupled 3Nv matrix-free FGMRES.

## Crucial numerical limitation

This implements the **hybrid 3Nv linear solver modes**, NOT the pure-tet
full-vector wall model.

The current hybrid **physical** momentum operator has:
- `A_x = A_y` (turbulent scalar volume + interface Nitsche + DG)
- `A_z` additionally includes axial HEX Spalding
- strongly clamped transverse velocity on cylindrical HEX wall
- **zero off-component momentum blocks** because this is vector-Laplacian
  with legacy axial wall; the true 9-block K is block-diagonal.

M3 therefore operates as a stacked 3x3-block FGS on the *same* current
frozen momentum operator, with zero offcomponent block entries. FGMRES is
the corresponding 3Nv reference action, not a different physical equation.
No tetrahedral wall BF3 projection is applied because the hybrid wall
has HEX Q1+BF2 faces; all existing interior BF3 degrees of freedom remain.

The first M3 outer step performs a **vector-level 1e-9 operator parity check**
between the 9-block assembled CSR and the matrix-free FGMRES operator.
Every M3 outer step monitors the full frozen 3Nv residual and restores the
previous momentum iterate if two sweeps increase its norm. This safeguard
does not alter a successful two-sweep result.

**This is not evidence of physical stability or validation.** If the old
hybrid wall physics is the source of divergence, switching to block FGS may
not fix it. A fully coupled *full-vector wall boundary formulation* would
require its own HEX Q1+BF2 3x3 boundary blocks and consistent pressure
correction/normal flux treatment, which are not delivered by this port.

## Run via GitHub on your workstation

```bash
cd "$HOME/NodalS"
BRANCH="exp/gpu-sst-hybrid-coupled-m3-fgmres-20261114"
TREE="$HOME/NODALS_HYBRID_COUPLED_GIT"
git fetch origin "$BRANCH"
git worktree add --detach "$TREE" "origin/$BRANCH"
EXP="$TREE/experiments/hybrid_pipe_re1m_99k"

# Keep the previously built isolated hybrid app unchanged until patching.
bash "$EXP/RUN_Y150_COUPLED.sh" prepare

CASE="$HOME/NODALS_HYBRID_Y150_M3_FGS2_WORK/cases/PIPE50_RE1M_HYBRID_Y150_M3_FGS2.case"
xdg-open "$CASE"

bash "$EXP/RUN_Y150_COUPLED.sh" dry
set -o pipefail
bash "$EXP/RUN_Y150_COUPLED.sh" 5 2>&1 | tee "$HOME/NODALS_HYBRID_Y150_M3_FGS2_WORK/build_and_5.log"
```

The runner reuses the **previous y+150 HXT1 mesh**, never the divergent
y+300 mesh; if necessary the earlier y+150 preparer creates it locally.
The first new native include requires **one incremental NVCC compile**, no
clean. Subsequent switches between M3 and FGMRES require no source change.

The reference FGMRES case is available at:
`PIPE50_RE1M_HYBRID_Y150_COUPLED_FGMRES.case`. Copy it next to the
generated M3 case (same mesh, separate output directory) to compare:
```bash
cp "$EXP/PIPE50_RE1M_HYBRID_Y150_COUPLED_FGMRES.case" \
  "$HOME/NODALS_HYBRID_Y150_M3_FGS2_WORK/cases/"
APP="$HOME/NodalS_GPU_HYBRID_RE1M_99K/turbulence/sst_gpu_v2"
python3 -u "$APP/nodals.py" run --case \
  "$HOME/NODALS_HYBRID_Y150_M3_FGS2_WORK/cases/PIPE50_RE1M_HYBRID_Y150_COUPLED_FGMRES.case" \
  --steps 5
```

### Expected startup and solver logs

`NODALS_HYBRID_COUPLED_CONFIG mode=m3-fgs sweeps=2` or
`mode=coupled-fgmres`.

At M3 iteration 1:
`NODALS_HYBRID_M3_OPERATOR_PARITY ... status=PASS`;
`NODALS_HYBRID_M3_FGS it=1 sweeps=2 ...`.
`NODALS_HYBRID_HEX_TRACE_LOG enabled=1` confirms the omega wall.

If CUDA compilation fails, capture the first actual `error:` lines from
`build_and_5.log` rather than repeating a full rebuild blindly. If the
operator parity fails, do not run further iterations: it means the coupled
solver does not represent the same frozen momentum equation.

## Moody pressure validation

Only after residuals and pressure correction become physically bounded:
```bash
RUN="$HOME/NodalS_SST_LOCAL_RUNS/PIPE50_RE1M_HYBRID_Y150_M3_FGS2"
bash "$EXP/POSTPROCESS_PIPE50_HYBRID.sh" "$RUN" latest
```
Use 20–30D, 30–40D, 40–50D, and overall 20–50D fitted-pressure errors.
