#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREC="${PREC:-fp32}"
EXE="$ROOT/nodals_hxt5b_${PREC}"
MESH="${MESH:-$ROOT/../hybrid_hxt1/work/HXT1_mesh.bin}"
OUT="${OUT:-$ROOT/results_full_rans_${PREC}}"
mkdir -p "$OUT"
LOG="${LOG:-$OUT/HXT5B_full_rans_${PREC}.log}"
VTU="${VTU:-$OUT/HXT5B_full_rans_${PREC}.vtu}"

RE="${RE:-13691.740248127866}"
BULK="${BULK:-50.0}"
MIXLEN_SCALE="${MIXLEN_SCALE:-1.0}"
WALL_SAMPLE_FRACTION="${WALL_SAMPLE_FRACTION:-0.5}"
MOMENTUM_WALL_MODE="${MOMENTUM_WALL_MODE:-spalding}"
SPALDING_MOLECULAR_CONSISTENCY="${SPALDING_MOLECULAR_CONSISTENCY:-0}" # SPALDING_MOLECULAR_CONSISTENCY_V1
WEAK_WALL_GAMMA="${WEAK_WALL_GAMMA:-4.0}"
WEAK_WALL_H_SCALE="${WEAK_WALL_H_SCALE:-1.0}"
WEAK_WALL_NUT_SCALE="${WEAK_WALL_NUT_SCALE:-1.0}"
MOM_NITSCHE_GAMMA="${MOM_NITSCHE_GAMMA:-${GAMMA:-50}}"
SCHUR_NITSCHE_GAMMA="${SCHUR_NITSCHE_GAMMA:-0}"
MAX_OUTER="${MAX_OUTER:-10000}"
SIMPLE_TOL="${SIMPLE_TOL:-1e-3}"
OUTER_CONT_ATOL="${OUTER_CONT_ATOL:-1e-12}"
ALPHA_U="${ALPHA_U:-0.70}"
ALPHA_P="${ALPHA_P:-1.00}"
RAU_SCALE="${RAU_SCALE:-2.0}"

MOM_WORK="${MOM_WORK:-sgs1}"
MOM_OMEGA="${MOM_OMEGA:-1.0}"
MOM_RTOL="${MOM_RTOL:-0.5}"
MOM_ATOL="${MOM_ATOL:-1e-8}"
MOM_MAX="${MOM_MAX:-50}"
MOM_CAP_POLICY="${MOM_CAP_POLICY:-warn}"
MOM_FGMRES_RESTART="${MOM_FGMRES_RESTART:-30}"
MOM_FGS_SWEEPS="${MOM_FGS_SWEEPS:-1}"

PRESSURE_SOLVER="${PRESSURE_SOLVER:-pcg_amg}"
PRESSURE_GAUGE="${PRESSURE_GAUGE:-pin}"
P_RTOL="${P_RTOL:-0.90}"
P_ATOL="${P_ATOL:-1e-12}"
P_MAX="${P_MAX:-50}"
POD_MODE="${POD_MODE:-off}"
POD_TRAIN_SNAPSHOTS="${POD_TRAIN_SNAPSHOTS:-10}"
POD_RANK="${POD_RANK:-2}"
POD_MAX_RANK="${POD_MAX_RANK:-64}"
POD_NOVELTY="${POD_NOVELTY:-1e-8}"
POD_TAIL_ITER="${POD_TAIL_ITER:-10}"
POD_HARD_ITS="${POD_HARD_ITS:-0}"
POD_ACCEPT_MIN_ABS="${POD_ACCEPT_MIN_ABS:-5}"
POD_ACCEPT_MIN_REL="${POD_ACCEPT_MIN_REL:-0.03}"
POD_REFRESH_ABS="${POD_REFRESH_ABS:-30}"
POD_REFRESH_REL="${POD_REFRESH_REL:-0.35}"
POD_RECOMPRESS_RANK="${POD_RECOMPRESS_RANK:-0}"
POD_RECOMPRESS_HISTORY="${POD_RECOMPRESS_HISTORY:-18}"
POD_RECOMPRESS_MAX_LOSS_ABS="${POD_RECOMPRESS_MAX_LOSS_ABS:-5}"
POD_RECOMPRESS_MAX_LOSS_REL="${POD_RECOMPRESS_MAX_LOSS_REL:-0.10}"
POD_RECOMPRESS_METHOD="${POD_RECOMPRESS_METHOD:-tail_pod}"
POD_RECOMPRESS_FORCE="${POD_RECOMPRESS_FORCE:-0}"
POD_NO_PROBE="${POD_NO_PROBE:-0}"

AMG_HIERARCHY="${AMG_HIERARCHY:-cf}"
CF_COARSENING="${CF_COARSENING:-pmis}"
CF_STRENGTH="${CF_STRENGTH:-classical-negative}"
CF_THETA="${CF_THETA:-0.25}"
CF_INTERP="${CF_INTERP:-exti}"
CF_PMAX="${CF_PMAX:-8}"
CF_AGGRESSIVE_FIRST="${CF_AGGRESSIVE_FIRST:-0}"
AMG_TERMINAL="${AMG_TERMINAL:-1000}"

# Turbulent pressure defaults.
AMG_FINE_MODE="${AMG_FINE_MODE:-explicit}"
AMG_SMOOTHER="${AMG_SMOOTHER:-cheb2}"
AMG_CHEB_DEGREE="${AMG_CHEB_DEGREE:-4}"
AMG_POWER_ITS="${AMG_POWER_ITS:-16}"
AMG_LAMBDA_SAFETY="${AMG_LAMBDA_SAFETY:-1.5}"
AMG_LAMBDA_LOW_FRACTION="${AMG_LAMBDA_LOW_FRACTION:-0.05}"
AMG_SPECTRUM_POLICY="${AMG_SPECTRUM_POLICY:-auto}"
AMG_JACOBI_OMEGA="${AMG_JACOBI_OMEGA:-0.7}"
AMG_JACOBI_FINE_SWEEPS="${AMG_JACOBI_FINE_SWEEPS:-1}"
AMG_JACOBI_COARSE_SWEEPS="${AMG_JACOBI_COARSE_SWEEPS:-1}"
AMG_MCGS_FINE_SWEEPS="${AMG_MCGS_FINE_SWEEPS:-1}"
AMG_MCGS_COARSE_SWEEPS="${AMG_MCGS_COARSE_SWEEPS:-1}"
AMG_MCGS_OMEGA="${AMG_MCGS_OMEGA:-1.0}"
AMG_MCGS_ORDER="${AMG_MCGS_ORDER:-symmetric}"

LAMINAR_PERF="${LAMINAR_PERF:-0}"
SST_G1="${SST_G1:-0}"
SST_G2="${SST_G2:-0}"
SST_G2_START="${SST_G2_START:-g0}"
SST_G2_PHYSICAL_GATE="${SST_G2_PHYSICAL_GATE:-1}"
SST_G2_METRICS_EVERY="${SST_G2_METRICS_EVERY:-1}"
SST_G2_MAX="${SST_G2_MAX:-800}"
SST_G2_PRINT_EVERY="${SST_G2_PRINT_EVERY:-20}"
SST_G2_PLATEAU_TOL="${SST_G2_PLATEAU_TOL:-2.5e-3}"
SST_G2_CONT_ATOL="${SST_G2_CONT_ATOL:-1e-6}"
SST_G2_PLATEAU_SAMPLES="${SST_G2_PLATEAU_SAMPLES:-5}"
SST_INTENSITY="${SST_INTENSITY:-0.05}"
SST_LENGTH_RATIO="${SST_LENGTH_RATIO:-0.07}"
SST_ALPHA_K="${SST_ALPHA_K:-0.5}"
SST_ALPHA_OMEGA="${SST_ALPHA_OMEGA:-0.5}"
SST_TOL="${SST_TOL:-1e-3}"
SST_MAX="${SST_MAX:-400}"
SST_LINEAR_RTOL="${SST_LINEAR_RTOL:-0.20}"
SST_LINEAR_ATOL="${SST_LINEAR_ATOL:-1e-7}"
# Split k/omega thresholds (default to legacy common scalar thresholds).
SST_K_LINEAR_RTOL="${SST_K_LINEAR_RTOL:-$SST_LINEAR_RTOL}"
SST_K_LINEAR_ATOL="${SST_K_LINEAR_ATOL:-$SST_LINEAR_ATOL}"
SST_OMEGA_LINEAR_RTOL="${SST_OMEGA_LINEAR_RTOL:-$SST_LINEAR_RTOL}"
SST_OMEGA_LINEAR_ATOL="${SST_OMEGA_LINEAR_ATOL:-$SST_LINEAR_ATOL}"
SST_LINEAR_MAX="${SST_LINEAR_MAX:-12}"
SST_GS_OMEGA="${SST_GS_OMEGA:-1.0}"
SST_LINEAR_SOLVER="${SST_LINEAR_SOLVER:-sgs1}"
SST_FGMRES_RESTART="${SST_FGMRES_RESTART:-30}"
SST_FGS_SWEEPS="${SST_FGS_SWEEPS:-1}"
SST_K_FLOOR="${SST_K_FLOOR:-1e-8}"
SST_OMEGA_FLOOR="${SST_OMEGA_FLOOR:-1e-3}"
# Legacy umbrella remains supported. New split switches default to it.
SST_DEVIATORIC_STRESS="${SST_DEVIATORIC_STRESS:-0}"
SST_DEVIATORIC_PRODUCTION="${SST_DEVIATORIC_PRODUCTION:-$SST_DEVIATORIC_STRESS}"
SST_DEVIATORIC_MOMENTUM="${SST_DEVIATORIC_MOMENTUM:-$SST_DEVIATORIC_STRESS}"
SST_SHARED_SPALDING_WALL_STATE="${SST_SHARED_SPALDING_WALL_STATE:-0}"
SST_PROFILE_INTEGRAL_UTAU="${SST_PROFILE_INTEGRAL_UTAU:-0}"
SST_PROFILE_INTEGRAL_NQ="${SST_PROFILE_INTEGRAL_NQ:-4}"
SST_PROFILE_INTEGRAL_CLOSURE="${SST_PROFILE_INTEGRAL_CLOSURE:-0}"
SST_PROFILE_AVERAGE_CLOSURE="${SST_PROFILE_AVERAGE_CLOSURE:-0}"
SST_PROFILE_AVERAGE_WALL_LAW="${SST_PROFILE_AVERAGE_WALL_LAW:-spalding}"
SST_PROFILE_MATCH_Y="${SST_PROFILE_MATCH_Y:-0}"
SST_PROFILE_FIT_Y1="${SST_PROFILE_FIT_Y1:-0}"
SST_PROFILE_FIT_Y2="${SST_PROFILE_FIT_Y2:-0}"
SST_PROFILE_FIT_Y3="${SST_PROFILE_FIT_Y3:-0}"
SST_PROFILE_FIT_Y4="${SST_PROFILE_FIT_Y4:-0}"
SST_PROFILE_FIT_Y5="${SST_PROFILE_FIT_Y5:-0}"
SST_PROFILE_CLOSURE_YPLUS_MIN="${SST_PROFILE_CLOSURE_YPLUS_MIN:-1.0}"
SST_WALL_PENALTY_GAMMA="${SST_WALL_PENALTY_GAMMA:-50}"
SST_PROFILE5_EQ_GAMMA="${SST_PROFILE5_EQ_GAMMA:-500}"
SST_OMEGA_WALL_MODE="${SST_OMEGA_WALL_MODE:-trace_log}"
SST_OF_KAPPA="${SST_OF_KAPPA:-0.41}"
SST_OF_E="${SST_OF_E:-9.8}"
SST_OF_BETA1="${SST_OF_BETA1:-0.075}"
SST_LOWRE_OMEGA_FACTOR="${SST_LOWRE_OMEGA_FACTOR:-60.0}"
SST_OF_RE_BLEND="${SST_OF_RE_BLEND:-11.0}"
SST_OF_PRODUCTION="${SST_OF_PRODUCTION:-1}"
SST_OF_PRODUCTION_SCALE="${SST_OF_PRODUCTION_SCALE:-1.0}"
SST_OF_WALL_VOLUME_SCALE="${SST_OF_WALL_VOLUME_SCALE:-1.0}"
SST_OF_PRODUCTION_LIMIT="${SST_OF_PRODUCTION_LIMIT:-10.0}"
SST_PRINT_EVERY="${SST_PRINT_EVERY:-10}"

# Exact enriched post-solve radial profile audit
PROFILE_AUDIT="${PROFILE_AUDIT:-1}"
PROFILE_Z_FRACS="${PROFILE_Z_FRACS:-0.50,0.75,0.90,0.98}"
PROFILE_RADIAL="${PROFILE_RADIAL:-65}"
PROFILE_ANGLES="${PROFILE_ANGLES:-16}"
PROFILE_RMAX="${PROFILE_RMAX:-0.98}"

cat <<CFG
=== HXT5B FULL TURBULENT-BOUNDARY GPU SETTINGS ===
precision=$PREC
mesh=$MESH
re=$RE
bulk=$BULK
mixlenScale=$MIXLEN_SCALE
inlet=DG numerical-trace plug
inletVelocityDOFs=free
dgMomentum=inflow + nonsymmetric Nitsche
wall=Firedrake Q1+BF2 off-wall Spalding
wallSampleFraction=$WALL_SAMPLE_FRACTION
momentumWallMode=$MOMENTUM_WALL_MODE
spaldingMolecularConsistency=$SPALDING_MOLECULAR_CONSISTENCY
weakWallGamma=$WEAK_WALL_GAMMA
weakWallHScale=$WEAK_WALL_H_SCALE
weakWallNutScale=$WEAK_WALL_NUT_SCALE
wallKappa=0.4
wallB=5.5
transverseWallClamp=Ux,Uy
axialWall=weak Uz
momentumWork=$MOM_WORK
momentumRelax=row_l1
momentumRtol=$MOM_RTOL
momentumAtol=$MOM_ATOL
momentumMax=$MOM_MAX
momentumCapPolicy=$MOM_CAP_POLICY
momentumFgmresRestart=$MOM_FGMRES_RESTART
momentumFgsSweeps=$MOM_FGS_SWEEPS
momentumRtol=$MOM_RTOL
momentumAtol=$MOM_ATOL
outerConvergence=continuity + x/y/z physical momentum initial-relative
simpleTol=$SIMPLE_TOL
pressureSolver=$PRESSURE_SOLVER
pressureGauge=$PRESSURE_GAUGE
pressureRtol=$P_RTOL
pressureAtol=$P_ATOL
pressureMax=$P_MAX
podMode=$POD_MODE
podTrainSnapshots=$POD_TRAIN_SNAPSHOTS
podRank=$POD_RANK
podMaxRank=$POD_MAX_RANK
podNovelty=$POD_NOVELTY
podTailIter=$POD_TAIL_ITER
podHardIts=$POD_HARD_ITS
podAcceptMinAbs=$POD_ACCEPT_MIN_ABS
podAcceptMinRel=$POD_ACCEPT_MIN_REL
podRefreshAbs=$POD_REFRESH_ABS
podRefreshRel=$POD_REFRESH_REL
podRecompressRank=$POD_RECOMPRESS_RANK
podRecompressHistory=$POD_RECOMPRESS_HISTORY
podRecompressMaxLossAbs=$POD_RECOMPRESS_MAX_LOSS_ABS
podRecompressMaxLossRel=$POD_RECOMPRESS_MAX_LOSS_REL
podRecompressMethod=$POD_RECOMPRESS_METHOD
podRecompressForce=$POD_RECOMPRESS_FORCE
podNoProbe=$POD_NO_PROBE
amgHierarchy=$AMG_HIERARCHY
cfCoarsening=$CF_COARSENING
cfStrength=$CF_STRENGTH
cfTheta=$CF_THETA
cfInterp=$CF_INTERP
cfPmax=$CF_PMAX
amgFineMode=$AMG_FINE_MODE
amgSmoother=$AMG_SMOOTHER
amgChebDegree=$AMG_CHEB_DEGREE
amgPowerIts=$AMG_POWER_ITS
amgLambdaSafety=$AMG_LAMBDA_SAFETY
amgLambdaLowFraction=$AMG_LAMBDA_LOW_FRACTION
amgSpectrumPolicy=$AMG_SPECTRUM_POLICY
alphaU=$ALPHA_U
alphaP=$ALPHA_P
rauScale=$RAU_SCALE
directionalRAU=1
rauPenaltyPolicy=SEPARATE_GAMMA
momentumNitscheGamma=$MOM_NITSCHE_GAMMA
schurNitscheGamma=$SCHUR_NITSCHE_GAMMA
coarseHierarchy=exact initial directional Schur snapshot
finePressureCSR=current directional Schur every outer
maxOuter=$MAX_OUTER
vtu=$VTU
laminarPerf=$LAMINAR_PERF
sstG1=$SST_G1
sstG2=$SST_G2
sstG2Start=$SST_G2_START
sstG2PhysicalGate=$SST_G2_PHYSICAL_GATE
sstG2MetricsEvery=$SST_G2_METRICS_EVERY
sstG2Max=$SST_G2_MAX
sstG2PrintEvery=$SST_G2_PRINT_EVERY
sstG2PlateauTol=$SST_G2_PLATEAU_TOL
sstG2ContAtol=$SST_G2_CONT_ATOL
sstG2PlateauSamples=$SST_G2_PLATEAU_SAMPLES
sstModel=SST-2003m
sstIntensity=$SST_INTENSITY
sstLengthScaleOverD=$SST_LENGTH_RATIO
sstAlphaK=$SST_ALPHA_K
sstAlphaOmega=$SST_ALPHA_OMEGA
sstTol=$SST_TOL
sstLinearSolver=$SST_LINEAR_SOLVER
sstLinearRtol=$SST_LINEAR_RTOL
sstLinearAtol=$SST_LINEAR_ATOL
sstKLinearRtol=$SST_K_LINEAR_RTOL
sstKLinearAtol=$SST_K_LINEAR_ATOL
sstOmegaLinearRtol=$SST_OMEGA_LINEAR_RTOL
sstOmegaLinearAtol=$SST_OMEGA_LINEAR_ATOL
sstLinearMax=$SST_LINEAR_MAX
sstFgmresRestart=$SST_FGMRES_RESTART
sstFgsSweeps=$SST_FGS_SWEEPS
sstOmegaWallMode=$SST_OMEGA_WALL_MODE
sstWallPenaltyGamma=$SST_WALL_PENALTY_GAMMA
sstOfKappa=$SST_OF_KAPPA
sstOfE=$SST_OF_E
sstOfBeta1=$SST_OF_BETA1
sstLowReOmegaFactor=$SST_LOWRE_OMEGA_FACTOR
sstOfReBlend=$SST_OF_RE_BLEND
sstOfProduction=$SST_OF_PRODUCTION
sstOfProductionScale=$SST_OF_PRODUCTION_SCALE
sstOfWallVolumeScale=$SST_OF_WALL_VOLUME_SCALE
sstOfProductionLimit=$SST_OF_PRODUCTION_LIMIT
sstDeviatoricStressLegacy=$SST_DEVIATORIC_STRESS
sstDeviatoricProduction=$SST_DEVIATORIC_PRODUCTION
sstDeviatoricMomentum=$SST_DEVIATORIC_MOMENTUM
sstSharedSpaldingWallState=$SST_SHARED_SPALDING_WALL_STATE
sstProfileIntegralUTau=$SST_PROFILE_INTEGRAL_UTAU
sstProfileIntegralNQ=$SST_PROFILE_INTEGRAL_NQ
sstProfile5EqGamma=$SST_PROFILE5_EQ_GAMMA
sstProfileIntegralClosure=$SST_PROFILE_INTEGRAL_CLOSURE
sstProfileAverageClosure=$SST_PROFILE_AVERAGE_CLOSURE
sstProfileClosureYPlusMin=$SST_PROFILE_CLOSURE_YPLUS_MIN
sstStateSpace=${SST_STATE_SPACE:-q1_carrier}
profileAudit=$PROFILE_AUDIT
profileZFractions=$PROFILE_Z_FRACS
profileRadial=$PROFILE_RADIAL
profileAngles=$PROFILE_ANGLES
profileRmax=$PROFILE_RMAX
CFG

# Stage17: RUN only. Compile once with build_stage17_solver.py.
if [[ ! -f "$EXE" || ! -x "$EXE" ]]; then
  echo "STAGE17_EXECUTABLE_MISSING: $EXE (build once with tools/build_stage17_solver.py)" >&2
  exit 2
fi
printf 'STAGE17_PRECOMPILED_EXECUTABLE=%s\n' "$EXE"

set +e
stdbuf -oL -eL "$EXE" \
  --mesh "$MESH" --vtu "$VTU" --re "$RE" --bulk "$BULK" --mixlen-scale "$MIXLEN_SCALE" --laminar-perf "$LAMINAR_PERF" \
  --profile-audit "$PROFILE_AUDIT" --profile-z-fracs "$PROFILE_Z_FRACS" --profile-radial "$PROFILE_RADIAL" --profile-angles "$PROFILE_ANGLES" --profile-rmax "$PROFILE_RMAX" \
  --wall-sample-fraction "$WALL_SAMPLE_FRACTION" --momentum-wall-mode "$MOMENTUM_WALL_MODE" --spalding-molecular-consistency "$SPALDING_MOLECULAR_CONSISTENCY" --weak-wall-gamma "$WEAK_WALL_GAMMA" --weak-wall-h-scale "$WEAK_WALL_H_SCALE" --weak-wall-nut-scale "$WEAK_WALL_NUT_SCALE" --gamma "$MOM_NITSCHE_GAMMA" --schur-nitsche-gamma "$SCHUR_NITSCHE_GAMMA" \
  --max-outer "$MAX_OUTER" --simple-tol "$SIMPLE_TOL" --outer-cont-atol "$OUTER_CONT_ATOL" \
  --alpha-u "$ALPHA_U" --alpha-p "$ALPHA_P" --rau-scale "$RAU_SCALE" \
  --momentum-work "$MOM_WORK" --mom-omega "$MOM_OMEGA" --mom-rtol "$MOM_RTOL" --mom-atol "$MOM_ATOL" --mom-max "$MOM_MAX" --mom-cap-policy "$MOM_CAP_POLICY" --mom-fgmres-restart "$MOM_FGMRES_RESTART" --mom-fgs-sweeps "$MOM_FGS_SWEEPS" \
  --pressure-solver "$PRESSURE_SOLVER" --pressure-gauge "$PRESSURE_GAUGE" --p-rtol "$P_RTOL" --p-atol "$P_ATOL" --p-max "$P_MAX" \
  --pod-mode "$POD_MODE" --pod-train-snapshots "$POD_TRAIN_SNAPSHOTS" --pod-rank "$POD_RANK" --pod-max-rank "$POD_MAX_RANK" --pod-novelty "$POD_NOVELTY" --pod-tail-iter "$POD_TAIL_ITER" --pod-hard-its "$POD_HARD_ITS" --pod-accept-min-abs "$POD_ACCEPT_MIN_ABS" --pod-accept-min-rel "$POD_ACCEPT_MIN_REL" --pod-refresh-abs "$POD_REFRESH_ABS" --pod-refresh-rel "$POD_REFRESH_REL" --pod-recompress-rank "$POD_RECOMPRESS_RANK" --pod-recompress-history "$POD_RECOMPRESS_HISTORY" --pod-recompress-max-loss-abs "$POD_RECOMPRESS_MAX_LOSS_ABS" --pod-recompress-max-loss-rel "$POD_RECOMPRESS_MAX_LOSS_REL" --pod-recompress-method "$POD_RECOMPRESS_METHOD" --pod-recompress-force "$POD_RECOMPRESS_FORCE" --pod-no-probe "$POD_NO_PROBE" \
  --amg-hierarchy "$AMG_HIERARCHY" --cf-coarsening "$CF_COARSENING" --cf-strength "$CF_STRENGTH" \
  --cf-theta "$CF_THETA" --cf-interp "$CF_INTERP" --cf-pmax "$CF_PMAX" --cf-aggressive-first "$CF_AGGRESSIVE_FIRST" \
  --amg-terminal "$AMG_TERMINAL" --amg-fine-mode "$AMG_FINE_MODE" --amg-smoother "$AMG_SMOOTHER" --amg-jacobi-omega "$AMG_JACOBI_OMEGA" \
  --amg-jacobi-fine-sweeps "$AMG_JACOBI_FINE_SWEEPS" --amg-jacobi-coarse-sweeps "$AMG_JACOBI_COARSE_SWEEPS" \
  --amg-cheb-degree "$AMG_CHEB_DEGREE" --amg-power-its "$AMG_POWER_ITS" \
  --amg-lambda-safety "$AMG_LAMBDA_SAFETY" --amg-lambda-low-fraction "$AMG_LAMBDA_LOW_FRACTION" \
  --amg-spectrum-policy "$AMG_SPECTRUM_POLICY" \
  --amg-mcgs-fine-sweeps "$AMG_MCGS_FINE_SWEEPS" --amg-mcgs-coarse-sweeps "$AMG_MCGS_COARSE_SWEEPS" \
  --amg-mcgs-omega "$AMG_MCGS_OMEGA" --amg-mcgs-order "$AMG_MCGS_ORDER" \
  --sst-g1 "$SST_G1" --sst-g2 "$SST_G2" --sst-g2-start "$SST_G2_START" --sst-g2-physical-gate "$SST_G2_PHYSICAL_GATE" --sst-g2-metrics-every "$SST_G2_METRICS_EVERY" --sst-g2-max "$SST_G2_MAX" --sst-g2-print-every "$SST_G2_PRINT_EVERY" --sst-g2-plateau-tol "$SST_G2_PLATEAU_TOL" --sst-g2-cont-atol "$SST_G2_CONT_ATOL" --sst-g2-plateau-samples "$SST_G2_PLATEAU_SAMPLES" --sst-intensity "$SST_INTENSITY" --sst-length-ratio "$SST_LENGTH_RATIO" \
  --sst-alpha-k "$SST_ALPHA_K" --sst-alpha-omega "$SST_ALPHA_OMEGA" --sst-tol "$SST_TOL" --sst-max "$SST_MAX" \
  --sst-linear-rtol "$SST_LINEAR_RTOL" --sst-linear-atol "$SST_LINEAR_ATOL" --sst-k-linear-rtol "$SST_K_LINEAR_RTOL" --sst-k-linear-atol "$SST_K_LINEAR_ATOL" --sst-omega-linear-rtol "$SST_OMEGA_LINEAR_RTOL" --sst-omega-linear-atol "$SST_OMEGA_LINEAR_ATOL" --sst-linear-max "$SST_LINEAR_MAX" --sst-gs-omega "$SST_GS_OMEGA" --sst-linear-solver "$SST_LINEAR_SOLVER" --sst-fgmres-restart "$SST_FGMRES_RESTART" --sst-fgs-sweeps "$SST_FGS_SWEEPS" \
  --sst-k-floor "$SST_K_FLOOR" --sst-omega-floor "$SST_OMEGA_FLOOR" --sst-deviatoric-stress "$SST_DEVIATORIC_STRESS" --sst-deviatoric-production "$SST_DEVIATORIC_PRODUCTION" --sst-deviatoric-momentum "$SST_DEVIATORIC_MOMENTUM" --sst-shared-spalding-wall-state "$SST_SHARED_SPALDING_WALL_STATE" --sst-profile-integral-utau "$SST_PROFILE_INTEGRAL_UTAU" --sst-profile-integral-nq "$SST_PROFILE_INTEGRAL_NQ" --sst-profile-integral-closure "$SST_PROFILE_INTEGRAL_CLOSURE" --sst-profile-average-closure "$SST_PROFILE_AVERAGE_CLOSURE" --sst-profile-average-wall-law "$SST_PROFILE_AVERAGE_WALL_LAW" --sst-profile-match-y "$SST_PROFILE_MATCH_Y" --sst-profile-fit-y1 "$SST_PROFILE_FIT_Y1" --sst-profile-fit-y2 "$SST_PROFILE_FIT_Y2" --sst-profile-fit-y3 "$SST_PROFILE_FIT_Y3" --sst-profile-fit-y4 "$SST_PROFILE_FIT_Y4" --sst-profile-fit-y5 "$SST_PROFILE_FIT_Y5" --sst-profile-closure-yplus-min "$SST_PROFILE_CLOSURE_YPLUS_MIN" --sst-wall-penalty-gamma "$SST_WALL_PENALTY_GAMMA" --sst-profile5-eq-gamma "$SST_PROFILE5_EQ_GAMMA" --sst-omega-wall-mode "$SST_OMEGA_WALL_MODE" \
  --sst-of-kappa "$SST_OF_KAPPA" --sst-of-E "$SST_OF_E" --sst-of-beta1 "$SST_OF_BETA1" --sst-lowre-omega-factor "$SST_LOWRE_OMEGA_FACTOR" --sst-of-re-blend "$SST_OF_RE_BLEND" --sst-of-production "$SST_OF_PRODUCTION" --sst-of-production-scale "$SST_OF_PRODUCTION_SCALE" --sst-of-wall-volume-scale "$SST_OF_WALL_VOLUME_SCALE" --sst-of-production-limit "$SST_OF_PRODUCTION_LIMIT" --sst-print-every "$SST_PRINT_EVERY" \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

echo
echo "=== HXT5B SUMMARY ==="
grep -E 'NODALS_BALANCED_POD|NODALS_PRESSURE_POD_MODE|NODALS_SST_G1|SST_G1_GATE_STATUS|NODALS_SST_G2|SST_G2_GATE_STATUS|NODALS_HXT5B_(DIRECTIONAL_SETUP|CONFIG|BOUNDARY_SETUP|MIXLEN|FINE_CSR_PARITY|SIMPLE|PRESSURE_FAIL|PHYSICS|OUTPUT)|NODALS_GPU_G8_CF_(SPECTRUM|HIERARCHY)|HXT5B_GATE_STATUS' "$LOG" || true
echo "HXT5B_LOG=$LOG"
echo "HXT5B_VTU=$VTU"
exit "$rc"
