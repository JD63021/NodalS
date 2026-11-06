#!/usr/bin/env bash
set -euo pipefail
# Pure-tet 10D NodalS SST: all controls supplied on the launch command line.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$ROOT/RUN_PURETET_USER_CONTROLS_FP64.sh"
DEFAULTS="$ROOT/PURETET_CONTROLS.env"
usage() {
cat <<'HELP'
PURE-TET 10D SST | independent UR and linear solver tolerances

Usage:
  ./RUN_PURETET_CLI.sh [options]
  ./RUN_PURETET_CLI.sh --help

Four under-relaxation factors (0 < alpha <= 1):
  --u VALUE                alphaU
  --p VALUE                alphaP
  --k VALUE                alphaK
  --omega VALUE            alphaOmega

Four linear-solver tolerance pairs (momentum applies to Ux,Uy,Uz):
  --mom-rtol VALUE         momentum relative tolerance
  --mom-atol VALUE         momentum absolute tolerance
  --p-rtol VALUE           pressure relative tolerance
  --p-atol VALUE           pressure absolute tolerance
  --k-rtol VALUE           k relative tolerance
  --k-atol VALUE           k absolute tolerance
  --omega-rtol VALUE       omega relative tolerance
  --omega-atol VALUE       omega absolute tolerance

Limits and diagnostics:
  --outer N                maxOuter and SST_G2_MAX (both updated)
  --mom-max N              momentum FGMRES maximum iterations
  --mom-restart N          momentum FGMRES restart dimension
  --p-max N                pressure linear iteration cap
  --sst-max N              k/omega linear iteration cap
  --sst-restart N          k/omega FGMRES restart dimension
  --state-space MODE       bernstein_q2 (enriched TET), or q1 (vertex P1 diagnostic)
  --vtu-every N            SST checkpoint interval
  --omega-audit-every N    omega diagnostics at 1,2,3 and every N iterations (0 disables)
  --eq5-gamma VALUE        PROFILE5+EQ5 physical penalty (default from controls env)
  --eq5-points A|B|S1|S5  A = sample 3; B = 1,3,5; S1 = 1; S5 = 5
                           This controls k+omega EQ5 only, not velocity.
  --omega-wall-mode MODE    profile5_eq_sst (default) or of_auto (TET point3)
  --of-production 0|1      OF_AUTO wall-production adjustment (default OFF)
                           Both modes ALWAYS retain five Spalding samples.
                           frozen high-gamma factors set by PURETET_EXP3_GAMMA_FACTORS
  --physical-gate 0|1      require physical momentum gate (normally 1)
  --mom-physical-atol X    physical momentum absolute convergence threshold
  --cont-atol X            continuity absolute convergence threshold
  --plateau-tol X          relative plateau threshold
  --out PATH               new output directory (must not exist)
  --dry-run                print effective configuration, no solver launch
  --help                   show this help

Unspecified values come from your existing PURETET_CONTROLS.env.
This wrapper never edits that file, CUDA source, or the working HEX solver.
All runs start from the configured initial (plug) state; no restart is implied.

Example: bounded 100-outer test with loose momentum inner solve
  ./RUN_PURETET_CLI.sh --outer 100 --u .7 --p .3 --k .5 --omega .5 \
      --mom-rtol 1e-2 --mom-atol 0 --p-rtol .5 --p-atol 0 \
      --k-rtol 1e-2 --k-atol 0 --omega-rtol 1e-2 --omega-atol 0
HELP
}
if [[ $# -eq 0 ]]; then usage; exit 0; fi
[[ -f "$BASE" && -f "$DEFAULTS" ]] || {
  echo 'ERROR: install NODALS_PURETET_USER_CONTROLS_20261015 first.' >&2; exit 2;
}
OUT_ARG=""
DRY=0
while (($#)); do
  arg="$1"; shift
  case "$arg" in
    --help|-h) usage; exit 0;;
    --dry-run) DRY=1; continue;;
    --u|--alpha-u) var=ALPHA_U;;
    --p|--alpha-p) var=ALPHA_P;;
    --k|--alpha-k) var=SST_ALPHA_K;;
    --omega|--alpha-omega) var=SST_ALPHA_OMEGA;;
    --mom-rtol) var=MOM_RTOL;;
    --mom-atol) var=MOM_ATOL;;
    --p-rtol) var=P_RTOL;;
    --p-atol) var=P_ATOL;;
    --k-rtol) var=K_RTOL;;
    --k-atol) var=K_ATOL;;
    --omega-rtol) var=OMEGA_RTOL;;
    --omega-atol) var=OMEGA_ATOL;;
    --outer) var=MAX_OUTER;;
    --mom-max) var=MOM_MAX;;
    --mom-restart) var=MOM_FGMRES_RESTART;;
    --p-max) var=P_MAX;;
    --sst-max) var=SST_LINEAR_MAX;;
    --sst-restart) var=SST_FGMRES_RESTART;;
    --state-space) var=SST_STATE_SPACE;;
    --vtu-every) var=SST_G2_VTU_EVERY;;
    --omega-audit-every) var=PURETET_OMEGA_AUDIT_EVERY;;
    --eq5-gamma) var=SST_PROFILE5_EQ_GAMMA;;
    --eq5-points) var=PURETET_EQ5_SAMPLE_SET;;
    --omega-wall-mode) var=SST_OMEGA_WALL_MODE;;
    --of-production) var=SST_OF_PRODUCTION;;
    --physical-gate) var=SST_G2_PHYSICAL_GATE;;
    --mom-physical-atol) var=SST_G2_MOM_ATOL;;
    --cont-atol) var=SST_G2_CONT_ATOL;;
    --plateau-tol) var=SST_G2_PLATEAU_TOL;;
    --out) var=OUT_ARG;;
    *) echo "ERROR: unknown option: $arg" >&2; usage >&2; exit 2;;
  esac
  [[ $# -gt 0 && "$1" != --* ]] || { echo "ERROR: $arg requires a value" >&2; exit 2; }
  val="$1"; shift
  if [[ "$var" == MAX_OUTER ]]; then
    export MAX_OUTER="$val" SST_G2_MAX="$val"
  elif [[ "$var" == OUT_ARG ]]; then
    OUT_ARG="$val"
  else
    # var names all come from the case statement above, not user text.
    printf -v "$var" '%s' "$val"
    export "$var"
  fi
done
if [[ -n "$OUT_ARG" ]]; then
  export OUT="$OUT_ARG"
else
  export OUT="$ROOT/../results_PURETET_CLI_$(date +%Y%m%d_%H%M%S)_$$_fp64"
fi
export PURETET_OMEGA_AUDIT_EVERY="${PURETET_OMEGA_AUDIT_EVERY:-0}"
if [[ "${SST_OMEGA_WALL_MODE:-profile5_eq_sst}" != profile5_eq_sst &&
      "${SST_OMEGA_WALL_MODE:-profile5_eq_sst}" != of_auto ]]; then
  echo 'ERROR: pure-TET supports only profile5_eq_sst or of_auto' >&2; exit 2
fi
if [[ "${SST_OMEGA_WALL_MODE:-profile5_eq_sst}" == of_auto && -n "${PURETET_EQ5_SAMPLE_SET:-}" ]]; then
  echo 'ERROR: OF_AUTO cannot be combined with EQ5 sample masking' >&2; exit 2
fi
if [[ "${SST_OF_PRODUCTION:-0}" != 0 && "${SST_OF_PRODUCTION:-0}" != 1 ]]; then
  echo 'ERROR: --of-production must be 0 or 1' >&2; exit 2
fi
if [[ -n "${PURETET_EQ5_SAMPLE_SET:-}" ]]; then
  [[ "$PURETET_EQ5_SAMPLE_SET" == A || "$PURETET_EQ5_SAMPLE_SET" == B ||
     "$PURETET_EQ5_SAMPLE_SET" == S1 || "$PURETET_EQ5_SAMPLE_SET" == S5 ]] || {
    echo 'ERROR: --eq5-points must be A, B, S1, or S5' >&2;exit 2;
  }
fi
export CONTROLS_FILE="$DEFAULTS"
# Look at the exact effective variables (CLI overrides take priority via :- in env file).
(
  set -a
  # shellcheck disable=SC1090
  source "$DEFAULTS"
  set +a
  printf 'PURETET_CLI_EFFECTIVE UR U=%s P=%s K=%s OMEGA=%s\n' "$ALPHA_U" "$ALPHA_P" "$SST_ALPHA_K" "$SST_ALPHA_OMEGA"
  printf 'PURETET_CLI_EFFECTIVE MOM rtol=%s atol=%s max=%s restart=%s\n' "$MOM_RTOL" "$MOM_ATOL" "$MOM_MAX" "$MOM_FGMRES_RESTART"
  printf 'PURETET_CLI_EFFECTIVE P rtol=%s atol=%s max=%s\n' "$P_RTOL" "$P_ATOL" "$P_MAX"
  printf 'PURETET_CLI_EFFECTIVE K rtol=%s atol=%s\n' "$K_RTOL" "$K_ATOL"
  printf 'PURETET_CLI_EFFECTIVE OMEGA rtol=%s atol=%s\n' "$OMEGA_RTOL" "$OMEGA_ATOL"
  printf 'PURETET_CLI_EFFECTIVE outer=%s g2max=%s stateSpace=%s physicalGate=%s\n' "$MAX_OUTER" "$SST_G2_MAX" "$SST_STATE_SPACE" "$SST_G2_PHYSICAL_GATE"
  printf 'PURETET_OMEGA_AUDIT every=%s\n' "$PURETET_OMEGA_AUDIT_EVERY"
  printf 'PURETET_WALL_MODE mode=%s OFProduction=%s velocityFitSamples=1,2,3,4,5 status=PASS\n' "${SST_OMEGA_WALL_MODE:-profile5_eq_sst}" "${SST_OF_PRODUCTION:-0}"
  case "${PURETET_EQ5_SAMPLE_SET:-C}" in A) pp=3;;B) pp=1,3,5;;S1) pp=1;;S5) pp=5;;C) pp=1,2,3,4,5;;esac
  printf 'PURETET_AB5_SELECTION subset=%s samplePoints=%s velocityProfileSamples=5 gamma=%s\n' "${PURETET_EQ5_SAMPLE_SET:-C}" "$pp" "${SST_PROFILE5_EQ_GAMMA:-500}"
  printf 'PURETET_EXP3_GAMMA_ACTIVE gamma=%s factors=%s\n' "${SST_PROFILE5_EQ_GAMMA:-500}" "${PURETET_EXP3_GAMMA_FACTORS:-0.5,1,2,5,10,20,50,100}"
  printf 'PURETET_CLI_OUT=%s\n' "$OUT"
)
if ((DRY)); then echo 'PURETET_CLI_DRY_RUN status=PASS solverNotLaunched=1'; exit 0; fi
[[ "$SST_STATE_SPACE" == 'bernstein_q2' || "$SST_STATE_SPACE" == 'q1' ]] || { echo 'ERROR: --state-space must be bernstein_q2 or q1' >&2; exit 2; }
# Existing controlled runner performs validation and creates the new results directory.
set +e
bash "$BASE"
rc=$?
set -e
if [[ -f "$OUT/run.log" ]]; then
  python3 "$ROOT/audit_puretet_cli.py" "$OUT/run.log" || true
elif [[ -f "$OUT/driver.log" ]]; then
  python3 "$ROOT/audit_puretet_cli.py" "$OUT/driver.log" || true
fi
echo "PURETET_CLI_RESULT exitCode=$rc results=$OUT"
echo "PURETET_CLI_LOG_DRIVER=$OUT/driver.log"
echo "PURETET_CLI_LOG_SOLVER=$OUT/run.log"
exit "$rc"
