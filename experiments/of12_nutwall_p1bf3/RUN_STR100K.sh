#!/usr/bin/env bash
# Isolated Re72600 pipe A/B experiment, 99,552 structured tetrahedra.
# Usage: bash experiments/of12_nutwall_p1bf3/RUN_STR100K.sh [baseline|option_a|noslip_sst] [STEPS]
set -euo pipefail
ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
MODE="${1:-baseline}"
STEPS="${2:-10}"
[[ "$MODE" == baseline || "$MODE" == option_a || "$MODE" == noslip_sst ]] ||
 { echo "MODE must be baseline, option_a, or noslip_sst" >&2; exit 2; }
[[ "$STEPS" =~ ^[0-9]+$ ]] && ((STEPS>0)) ||
 { echo "STEPS must be positive integer" >&2; exit 2; }
BRANCH="$(git -C "$ROOT" branch --show-current)"
[[ "$BRANCH" == "exp/of12-nutwall-p1bf3-20261113" ]] ||
 { echo "Wrong branch: $BRANCH" >&2; exit 2; }
PKG="${PKG:-$HOME/Downloads/PIPE50_STRUCTURED_RE72600_100K_20261113}"
[[ -d "$PKG" ]] || { echo "Missing extracted structured mesh package: $PKG" >&2; exit 3; }
CASE="$PKG/cases/PIPE50_STR100K_RE72600_5PT.case"
MESH="$PKG/mesh/PIPE50_STRUCTURED_100K_TET.msh"
[[ -f "$CASE" && -f "$MESH" ]] ||
 { echo "Missing case or mesh in $PKG" >&2; exit 3; }
APP="$ROOT/turbulence/sst_gpu_v2"
BIN="$APP/build/nodals_sst"
OUT_ROOT="${OUT_ROOT:-$HOME/NodalS_SST_LOCAL_RUNS/OF12_NUTWALL_STR100K_20261113}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$OUT_ROOT/${MODE}_${STEPS}steps_${STAMP}"
mkdir -p "$OUT"

if [[ "${SKIP_BUILD:-0}" != 1 ]]; then
 NVCC="${NVCC:-/usr/local/cuda-12.8/bin/nvcc}"
 [[ -x "$NVCC" ]] || NVCC="$(command -v nvcc)"
 echo "Compiling on isolated branch: $APP"
 make -C "$APP" -j2 NVCC="$NVCC" CUDA_ARCH=sm_86
fi
[[ -x "$BIN" ]] || { echo "Compiled binary missing: $BIN" >&2; exit 4; }
echo "MODE=$MODE STEPS=$STEPS"
echo "CASE=$CASE MESH=$MESH"
echo "BRANCH=$BRANCH COMMIT=$(git -C "$ROOT" rev-parse HEAD)"
sha256sum "$BIN" "$MESH" "$CASE"

WALL_BC=robin
[[ "$MODE" == option_a ]] && WALL_BC=of12_noslip
[[ "$MODE" == noslip_sst ]] && WALL_BC=strong_noslip_sst
echo "Using wall_function.momentum_bc=$WALL_BC"
# Diagnostics: FGMRES uses the same physics but insists on a true momentum
# residual, unlike M3 fixed sweeps. This overrides only solver.u.method.
MOMENTUM_SOLVER="${MOMENTUM_SOLVER:-}"
SOLVER_ARGS=()
if [[ -n "$MOMENTUM_SOLVER" ]]; then
 case "$MOMENTUM_SOLVER" in
  coupled-fgmres|m3-fgs|m3-sgs) ;;
  *) echo "Bad MOMENTUM_SOLVER=$MOMENTUM_SOLVER" >&2; exit 2 ;;
 esac
 SOLVER_ARGS=(--momentum-solver "$MOMENTUM_SOLVER")
fi
echo "MOMENTUM_SOLVER=${MOMENTUM_SOLVER:-case_default}"
# Keep A/B settings exactly the same. Only wall_function.momentum_bc differs.
# Prefix each run's log with a clear MODEL selection; native prints optionA flag.
set +e
python3 -u "$APP/nodals.py" run \
 --case "$CASE" --mesh "$MESH" --output "$OUT" \
 --binary "$BIN" --steps "$STEPS" \
 --set "wall_function.momentum_bc=$WALL_BC" \
 "${SOLVER_ARGS[@]}" \
 2>&1 | tee "$OUT/console.log"
NATIVE_RC=${PIPESTATUS[0]}
set -e

# A fixed 10-step smoke test normally finishes without nonlinear convergence.
# rc=3 is acceptable ONLY if the expected final iteration was completed.
if ((NATIVE_RC != 0)); then
 if ((NATIVE_RC==3)) && grep -q "NODALS_SST_G2_FINAL iterations=$STEPS " "$OUT/terminal.log"; then
   echo "SMOKE_EXECUTED mode=$MODE steps=$STEPS nativeExit=3 result=NOT_CONVERGED"
 else
   echo "ERROR: unexpected native exit $NATIVE_RC (not a completed $STEPS-step smoke)" >&2
   exit "$NATIVE_RC"
 fi
fi

if [[ "$MODE" == option_a || "$MODE" == noslip_sst ]]; then
 if ! grep -q 'NODALS_OPTION_A enabled=1' "$OUT/terminal.log"; then
   echo "FAIL: native solver did not confirm Option A wall model" >&2; exit 6
 fi
fi
echo "RUN_FINISHED mode=$MODE outcome=EXECUTED_NOT_NECESSARILY_CONVERGED path=$OUT"
