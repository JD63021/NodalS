#!/usr/bin/env bash
set -euo pipefail
# One editable run control file; paths can be overridden with CONTROLS_FILE.
CONTROLS_FILE="${CONTROLS_FILE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/PURETET_CONTROLS.env}"
[[ -f "$CONTROLS_FILE" ]] || { echo "ERROR: missing controls file: $CONTROLS_FILE" >&2; exit 2; }
set -a
# shellcheck disable=SC1090
source "$CONTROLS_FILE"
set +a
# Separate diagnostic run; does NOT edit CUDA sources or previous results.
# Momentum equations and wall laws are unchanged.  All nonlinear gates remain enforced.
# The case starts from a plug state; it does not continue a previous result.

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$ROOT"

MESH_DEFAULT="$ROOT/../mesh_puretet_10D/HXT1_mesh.bin"
export MESH="${Y10_10D_MESH:-${Y10_MESH:-$MESH_DEFAULT}}"
[[ -n "$MESH" && -f "$MESH" ]] || { echo "ERROR: 10D Y10 mesh not found"; exit 2; }

export PREC=fp64
export RE="${RE:-13691.740248127866}"
export BULK="${BULK:-50.0}"
export MIXLEN_SCALE=0.0
export SUPG=0

# Same accepted high-UR SIMPLE family; all four URs remain overrideable.
export SIMPLE_COUPLING=corrected
export ALPHA_U="${ALPHA_U:-0.70}"
export ALPHA_P="${ALPHA_P:-0.30}"
export RAU_SCALE="${RAU_SCALE:-"${RAU_SCALE:-2.00}"}"
export SIMPLE_TOL="${SIMPLE_TOL:-"${SIMPLE_TOL:-1e-3}"}"
export MAX_OUTER="${MAX_OUTER:-20}"
export OUTER_CONT_ATOL="${OUTER_CONT_ATOL:-1e-12}"

export MOM_WORK="${MOM_WORK:-fgmres-fgs}"
export MOM_OMEGA="${MOM_OMEGA:-1.0}"
export MOM_RTOL="${MOM_RTOL:-1e-4}"
export MOM_ATOL="${MOM_ATOL:-0}"
export MOM_MAX="${MOM_MAX:-1200}"
export MOM_CAP_POLICY="${MOM_CAP_POLICY:-fail}"
export MOM_FGMRES_RESTART="${MOM_FGMRES_RESTART:-80}"
export MOM_FGS_SWEEPS="${MOM_FGS_SWEEPS:-1}"

export PRESSURE_SOLVER="${PRESSURE_SOLVER:-pcg_amg}"
export PRESSURE_GAUGE="${PRESSURE_GAUGE:-natural}"
export P_RTOL="${P_RTOL:-0.5}"
export P_ATOL="${P_ATOL:-0}"
export P_MAX="${P_MAX:-600}"

export AMG_FINE_MODE="${AMG_FINE_MODE:-explicit}"
export AMG_HIERARCHY="${AMG_HIERARCHY:-cf}"
export CF_COARSENING="${CF_COARSENING:-pmis}"
export CF_STRENGTH="${CF_STRENGTH:-classical-negative}"
export CF_THETA="${CF_THETA:-0.25}"
export CF_INTERP="${CF_INTERP:-exti}"
export CF_PMAX="${CF_PMAX:-8}"
export CF_AGGRESSIVE_FIRST="${CF_AGGRESSIVE_FIRST:-0}"
export AMG_TERMINAL="${AMG_TERMINAL:-1000}"
export AMG_SMOOTHER="${AMG_SMOOTHER:-mcgs}"
export AMG_MCGS_FINE_SWEEPS="${AMG_MCGS_FINE_SWEEPS:-1}"
export AMG_MCGS_COARSE_SWEEPS="${AMG_MCGS_COARSE_SWEEPS:-1}"
export AMG_MCGS_OMEGA="${AMG_MCGS_OMEGA:-1.0}"
export AMG_MCGS_ORDER="${AMG_MCGS_ORDER:-symmetric}"
export AMG_CHEB_DEGREE="${AMG_CHEB_DEGREE:-4}"
export AMG_POWER_ITS="${AMG_POWER_ITS:-16}"
export AMG_SPECTRUM_POLICY="${AMG_SPECTRUM_POLICY:-auto}"

# Keep Robin wall-stress architecture.
export MOMENTUM_WALL_MODE=spalding_profile_integral
export SPALDING_MOLECULAR_CONSISTENCY=0
export WALL_SAMPLE_FRACTION=0.50
export MOM_NITSCHE_GAMMA=0
export SCHUR_NITSCHE_GAMMA=0

# Bernstein-Q2 + five-point local-equilibrium SST profile closure.
export SST_STATE_SPACE="${SST_STATE_SPACE:-bernstein_q2}"
export SST_OMEGA_WALL_MODE="${SST_OMEGA_WALL_MODE:-profile5_eq_sst}"
export SST_WALL_PENALTY_GAMMA="${SST_WALL_PENALTY_GAMMA:-500}"
export SST_PROFILE5_EQ_GAMMA="${SST_PROFILE5_EQ_GAMMA:-500}"
export SST_OF_KAPPA="${SST_OF_KAPPA:-0.40}"
export SST_OF_E="${SST_OF_E:-9.025013499434122}"
export SST_OF_BETA1="${SST_OF_BETA1:-0.075}"
export SST_OF_RE_BLEND="${SST_OF_RE_BLEND:-11.0}"
export SST_OF_PRODUCTION="${SST_OF_PRODUCTION:-0}"
export SST_OF_PRODUCTION_SCALE="${SST_OF_PRODUCTION_SCALE:-1.0}"
export SST_OF_WALL_VOLUME_SCALE="${SST_OF_WALL_VOLUME_SCALE:-1.0}"
export SST_OF_PRODUCTION_LIMIT="${SST_OF_PRODUCTION_LIMIT:-10.0}"

export SST_G1=0
export SST_G2=1
export SST_G2_START=plug
export SST_G2_MAX="${SST_G2_MAX:-20}"
export SST_G2_PHYSICAL_GATE="${SST_G2_PHYSICAL_GATE:-1}"
export SST_G2_METRICS_EVERY="${SST_G2_METRICS_EVERY:-1}"
export SST_G2_PRINT_EVERY="${SST_G2_PRINT_EVERY:-1}"
export SST_G2_PLATEAU_SAMPLES="${SST_G2_PLATEAU_SAMPLES:-5}"
export SST_G2_PLATEAU_TOL="${SST_G2_PLATEAU_TOL:-2.5e-3}"
export SST_G2_CONT_ATOL="${SST_G2_CONT_ATOL:-1e-6}"
export SST_G2_MOM_ATOL="${SST_G2_MOM_ATOL:-1e-5}"

export SST_ALPHA_K="${SST_ALPHA_K:-0.50}"
export SST_ALPHA_OMEGA="${SST_ALPHA_OMEGA:-0.50}"
export SST_LINEAR_SOLVER="${SST_LINEAR_SOLVER:-fgmres-fgs}"
export SST_LINEAR_RTOL="${SST_LINEAR_RTOL:-1e-2}"
export SST_LINEAR_ATOL="${SST_LINEAR_ATOL:-0}"
# Crucial: k and omega are independent and reach different CUDA solver options.
export SST_K_LINEAR_RTOL="${K_RTOL:-${SST_LINEAR_RTOL:-1e-2}}"
export SST_K_LINEAR_ATOL="${K_ATOL:-${SST_LINEAR_ATOL:-0}}"
export SST_OMEGA_LINEAR_RTOL="${OMEGA_RTOL:-${SST_LINEAR_RTOL:-1e-2}}"
export SST_OMEGA_LINEAR_ATOL="${OMEGA_ATOL:-${SST_LINEAR_ATOL:-0}}"
export SST_LINEAR_MAX="${SST_LINEAR_MAX:-200}"
export SST_FGMRES_RESTART="${SST_FGMRES_RESTART:-30}"
export SST_FGS_SWEEPS="${SST_FGS_SWEEPS:-1}"

export SST_DEVIATORIC_STRESS=0
export SST_DEVIATORIC_PRODUCTION=0
export SST_DEVIATORIC_MOMENTUM=0

# NQ12 remains for the local momentum Picard trial; SST wall targets use the SAME five remote profile points.
export SST_SHARED_SPALDING_WALL_STATE=1
export SST_PROFILE_INTEGRAL_UTAU=1
export SPALDING_PROFILE_NQ=12
export SST_PROFILE_INTEGRAL_NQ=12
export SST_PROFILE_INTEGRAL_CLOSURE=0
export SST_PROFILE_AVERAGE_CLOSURE=1

# PROFILE5: fit one u_tau to FIVE resolved velocity samples on each wall-normal line.
# The sample locations are fixed physical distances corresponding to reference
# y+ = 15, 23.75, 32.5, 41.25, 50 using the benchmark reference only to place fixed physical probes.
# The fit is equal-weight least squares of the implicit Spalding residual.
export SST_PROFILE_AVERAGE_WALL_LAW=spalding_profile5
export SST_PROFILE_MATCH_Y=0
export SST_PROFILE_FIT_Y1="${SST_PROFILE_FIT_Y1:-7.347338250770073e-05}"
export SST_PROFILE_FIT_Y2="${SST_PROFILE_FIT_Y2:-1.163328556371928e-04}"
export SST_PROFILE_FIT_Y3="${SST_PROFILE_FIT_Y3:-1.591923287666849e-04}"
export SST_PROFILE_FIT_Y4="${SST_PROFILE_FIT_Y4:-2.020518018961770e-04}"
export SST_PROFILE_FIT_Y5="${SST_PROFILE_FIT_Y5:-2.449112750256691e-04}"


# This is a 10D developing-pipe mesh from a plug inlet. Do not interpret any
# legacy Moody-error print as a fully developed accuracy metric. The intended
# comparison is 10D O-ring baseline versus 10D converted unstructured HEX8.

export PROFILE_AUDIT="${PROFILE_AUDIT:-0}"
export SST_G2_VTU_EVERY="${SST_G2_VTU_EVERY:-5}"

# Fail early for invalid settings instead of launching nvcc / allocating GPUs.
python3 - "$ALPHA_U" "$ALPHA_P" "$SST_ALPHA_K" "$SST_ALPHA_OMEGA" \
  "$MOM_RTOL" "$MOM_ATOL" "$P_RTOL" "$P_ATOL" \
  "$SST_K_LINEAR_RTOL" "$SST_K_LINEAR_ATOL" "$SST_OMEGA_LINEAR_RTOL" "$SST_OMEGA_LINEAR_ATOL" \
  "$MAX_OUTER" "$SST_G2_MAX" <<'PYVALIDATE'
import math,sys
keys=['ALPHA_U','ALPHA_P','SST_ALPHA_K','SST_ALPHA_OMEGA',
      'MOM_RTOL','MOM_ATOL','P_RTOL','P_ATOL',
      'K_RTOL','K_ATOL','OMEGA_RTOL','OMEGA_ATOL']
try:
    vals={k:float(v) for k,v in zip(keys,sys.argv[1:13])}
    for k in keys[:4]:
        assert math.isfinite(vals[k]) and 0<vals[k]<=1, f'{k}: 0 < UR <= 1 required'
    for eq in ['MOM','P','K','OMEGA']:
        rt,at=vals[eq+'_RTOL'],vals[eq+'_ATOL']
        assert math.isfinite(rt) and 0<=rt<1 and math.isfinite(at) and at>=0 and (rt>0 or at>0),f'{eq}: tolerances must be nonnegative and not both zero'
    for n in sys.argv[13:]:
        assert int(n)>0, 'outer iteration limits must be positive'
except (ValueError,AssertionError) as e:
    sys.exit('PURETET_USER_CONTROLS status=FAIL '+str(e))
print('PURETET_USER_CONTROLS status=PASS fourUR=VALID fourLinearPairs=VALID')
PYVALIDATE

TAG="PURETET_USER_CONTROLS"
OUT="${OUT:-$ROOT/../results_${TAG}_$(date +%Y%m%d_%H%M%S)_fp64}"
export OUT
export LOG="$OUT/run.log"
export SST_G2_VTU_DIR="$OUT"
export VTU="$OUT/flow.vtu"
# Never remove an earlier result directory without explicit action.
if [[ -e "$OUT" ]]; then echo "ERROR: output directory already exists: $OUT" >&2; exit 3; fi
mkdir -p "$OUT"

cat <<CFG
================================================================
  PURE-TET 10D / PROFILE5 + EQ5 / T6 CONTROLLED STABILITY PROBE
================================================================
PREC=$PREC
MESH=$MESH
SST_STATE_SPACE=$SST_STATE_SPACE
maxOuter=$MAX_OUTER sstG2Max=$SST_G2_MAX physicalGate=$SST_G2_PHYSICAL_GATE
momRtol=$MOM_RTOL momAtol=$MOM_ATOL momMax=$MOM_MAX momRestart=$MOM_FGMRES_RESTART
pressureRtol=$P_RTOL pressureAtol=$P_ATOL pressureMax=$P_MAX
kLinearRtol=$SST_K_LINEAR_RTOL kLinearAtol=$SST_K_LINEAR_ATOL
omegaLinearRtol=$SST_OMEGA_LINEAR_RTOL omegaLinearAtol=$SST_OMEGA_LINEAR_ATOL
sstLinearMax=$SST_LINEAR_MAX sstFgmresRestart=$SST_FGMRES_RESTART

MOMENTUM_WALL_MODE=$MOMENTUM_WALL_MODE
SST_PROFILE_AVERAGE_WALL_LAW=$SST_PROFILE_AVERAGE_WALL_LAW
SST_PROFILE_FIT_Y1=$SST_PROFILE_FIT_Y1   # reference y+=15
SST_PROFILE_FIT_Y2=$SST_PROFILE_FIT_Y2   # reference y+=23.75
SST_PROFILE_FIT_Y3=$SST_PROFILE_FIT_Y3   # reference y+=32.5
SST_PROFILE_FIT_Y4=$SST_PROFILE_FIT_Y4   # reference y+=41.25
SST_PROFILE_FIT_Y5=$SST_PROFILE_FIT_Y5   # reference y+=50
REFERENCE_UTAU=2.982171183676e+00
REFERENCE_MOODY_FD=2.845870390000e-02

wallVelocityReference=ABSOLUTE_RELATIVE_TO_STATIONARY_SOLID
wallVelocityLocation=FIVE_PRESCRIBED_PHYSICAL_NORMAL_HEIGHTS
profileFit=ONE_UTAU_EQUAL_WEIGHT_SPALDING_LEAST_SQUARES
reportedWallYPlus=MIDDLE_SAMPLE_Y3
momentumTrial=FIRST_TET_NORMAL_COLUMN_GAUSS4_90PCT_AVERAGE
momentumBeta=FITTED_UTAU_SQUARED_OVER_LOCAL_UBAR_WITH_FINITE_PROFILE_FALLBACK

SST_OMEGA_WALL_MODE=$SST_OMEGA_WALL_MODE
SST_SHARED_SPALDING_WALL_STATE=$SST_SHARED_SPALDING_WALL_STATE
SST_PROFILE_INTEGRAL_NQ=$SST_PROFILE_INTEGRAL_NQ
SST_PROFILE_AVERAGE_CLOSURE=$SST_PROFILE_AVERAGE_CLOSURE
SST_OF_PRODUCTION=$SST_OF_PRODUCTION
SST_PROFILE5_EQ_GAMMA=$SST_PROFILE5_EQ_GAMMA
sstWallModel=FIVE_POINT_LOCAL_EQUILIBRIUM_SST2003M
sstSampleLocations=SAME_FIVE_PHYSICAL_POINTS_AS_MOMENTUM
sstScalarFit=MULTIPOINT_WEIGHTED_LEAST_SQUARES
tableVelocityInversion=OFF
tableMomentum=OFF

ALPHA_U=$ALPHA_U
ALPHA_P=$ALPHA_P
SST_ALPHA_K=$SST_ALPHA_K
SST_ALPHA_OMEGA=$SST_ALPHA_OMEGA
RAU_SCALE=$RAU_SCALE
OUT=$OUT
================================================================
CFG

bash ./RUN_HXT5B.sh 2>&1 | tee "$OUT/driver.log"
