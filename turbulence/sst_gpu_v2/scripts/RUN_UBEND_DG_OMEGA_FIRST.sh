#!/usr/bin/env bash
# Run the FIRST 5 genuinely coupled SIMPLE iterations with DG omega enabled.
# Uses the proven mirror-symmetric 199584-tetra U-bend case from workstation.
# Git worktree-safe: no changes to original case, mesh or validated CG binary.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP="$(cd -- "$SCRIPT_DIR/.." && pwd)"
BIN="${BIN:-$APP/build/nodals_sst}"
CASE="${CASE:-$HOME/NodalS_GPU_GENROBIN_ADAPTIVE/examples/ubend_unstructured_200k/UBEND_MIRROR_GENROBIN_SKEW_SUPG_ADAPTIVE.case}"
STEPS="${STEPS:-5}"
OUT="${OUT:-$HOME/NodalS_SST_LOCAL_RUNS/ubend_dg_omega_g2_first_$(date +%Y%m%d_%H%M%S)}"

[[ -f "$CASE" ]] || { echo "DG_OMEGA_LAUNCH_ERROR case missing: $CASE";exit 2; }
[[ -x "$BIN" ]] || { echo "DG_OMEGA_LAUNCH_ERROR native binary missing: $BIN";exit 2; }
[[ "$STEPS" =~ ^[0-9]+$ ]] && (( STEPS>=1 && STEPS<=50 )) ||
  { echo "DG_OMEGA_LAUNCH_ERROR STEPS must be an integer 1..50";exit 2; }

echo "DG_OMEGA_G2_LAUNCH case=$CASE"
echo "DG_OMEGA_G2_LAUNCH binary=$BIN"
echo "DG_OMEGA_G2_LAUNCH iterations=$STEPS output=$OUT"
echo "DG_OMEGA_G2_LAUNCH note=This_branch_runner_does_not_dispatch_adaptive_run"
echo "DG_OMEGA_G2_LAUNCH note=DG_enabled_via_case_environment_not_shell_export"

# Important: nodals.py filters inherited STAGE*/NODALS*/SST* environment
# variables. Pass NODALS_DG_OMEGA through [environment] using --set.
# Stage31 CG-only checkpoints cannot represent DG omega, so disabled.
# Preserve all other tested momentum/pressure/wall/mesh case settings.
python3 -u "$APP/nodals.py" run \
  --case "$CASE" \
  --binary "$BIN" \
  --steps "$STEPS" \
  --output "$OUT" \
  --set environment.NODALS_DG_OMEGA=1 \
  --set restart.path= \
  --set output.checkpoint_every=0 \
  --set output.vtu_every=0 \
  --set output.console=full \
  --set output.diagnostics_every=0 \
  --set output.legacy_final_diagnostics=false \
  --set wall_distance.method=geometric \
  --set sst.state_space=q1 \
  --set sst.compact_p1=false \
  --set omega_wall.method=of_auto \
  --set omega_wall.stage41_mode=trace_log \
  --set omega_wall.trace_gamma=100000 \
  --set omega_wall.production=false \
  --set omega_wall.sample_index=3 \
  --set wall_function.samples=5 \
  --set wall_function.shared_wall_state=true \
  --set supg.k=false \
  --set supg.omega=false

echo "DG_OMEGA_G2_LAUNCH_FINISHED output=$OUT"
