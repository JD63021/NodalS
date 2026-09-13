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

PRESSURE_SOLVER="${PRESSURE_SOLVER:-pcg_amg}"
P_RTOL="${P_RTOL:-0.90}"
P_ATOL="${P_ATOL:-1e-12}"
P_MAX="${P_MAX:-50}"

AMG_HIERARCHY="${AMG_HIERARCHY:-cf}"
CF_COARSENING="${CF_COARSENING:-pmis}"
CF_STRENGTH="${CF_STRENGTH:-classical-negative}"
CF_THETA="${CF_THETA:-0.25}"
CF_INTERP="${CF_INTERP:-exti}"
CF_PMAX="${CF_PMAX:-8}"
CF_AGGRESSIVE_FIRST="${CF_AGGRESSIVE_FIRST:-0}"
AMG_TERMINAL="${AMG_TERMINAL:-1000}"

# Turbulent pressure defaults.
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

SST_G1="${SST_G1:-0}"
SST_G2="${SST_G2:-0}"
SST_G2_START="${SST_G2_START:-g0}"
SST_G2_MAX="${SST_G2_MAX:-800}"
SST_G2_PRINT_EVERY="${SST_G2_PRINT_EVERY:-20}"
SST_G2_PLATEAU_TOL="${SST_G2_PLATEAU_TOL:-2.5e-3}"
SST_G2_PLATEAU_SAMPLES="${SST_G2_PLATEAU_SAMPLES:-5}"
SST_INTENSITY="${SST_INTENSITY:-0.05}"
SST_LENGTH_RATIO="${SST_LENGTH_RATIO:-0.07}"
SST_ALPHA_K="${SST_ALPHA_K:-0.5}"
SST_ALPHA_OMEGA="${SST_ALPHA_OMEGA:-0.5}"
SST_TOL="${SST_TOL:-1e-3}"
SST_MAX="${SST_MAX:-400}"
SST_LINEAR_RTOL="${SST_LINEAR_RTOL:-0.20}"
SST_LINEAR_ATOL="${SST_LINEAR_ATOL:-1e-7}"
SST_LINEAR_MAX="${SST_LINEAR_MAX:-12}"
SST_GS_OMEGA="${SST_GS_OMEGA:-1.0}"
SST_K_FLOOR="${SST_K_FLOOR:-1e-8}"
SST_OMEGA_FLOOR="${SST_OMEGA_FLOOR:-1e-3}"
SST_WALL_PENALTY_GAMMA="${SST_WALL_PENALTY_GAMMA:-50}"
SST_OMEGA_WALL_MODE="${SST_OMEGA_WALL_MODE:-trace_log}"
SST_OF_KAPPA="${SST_OF_KAPPA:-0.41}"
SST_OF_E="${SST_OF_E:-9.8}"
SST_OF_BETA1="${SST_OF_BETA1:-0.075}"
SST_OF_RE_BLEND="${SST_OF_RE_BLEND:-11.0}"
SST_OF_PRODUCTION="${SST_OF_PRODUCTION:-1}"
SST_OF_PRODUCTION_SCALE="${SST_OF_PRODUCTION_SCALE:-1.0}"
SST_OF_WALL_VOLUME_SCALE="${SST_OF_WALL_VOLUME_SCALE:-1.0}"
SST_OF_PRODUCTION_LIMIT="${SST_OF_PRODUCTION_LIMIT:-10.0}"
SST_PRINT_EVERY="${SST_PRINT_EVERY:-10}"

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
momentumRtol=$MOM_RTOL
momentumAtol=$MOM_ATOL
outerConvergence=continuity + x/y/z physical momentum initial-relative
simpleTol=$SIMPLE_TOL
pressureSolver=$PRESSURE_SOLVER
pressureRtol=$P_RTOL
pressureAtol=$P_ATOL
pressureMax=$P_MAX
amgHierarchy=$AMG_HIERARCHY
cfCoarsening=$CF_COARSENING
cfStrength=$CF_STRENGTH
cfTheta=$CF_THETA
cfInterp=$CF_INTERP
cfPmax=$CF_PMAX
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
sstG1=$SST_G1
sstG2=$SST_G2
sstG2Start=$SST_G2_START
sstG2Max=$SST_G2_MAX
sstG2PrintEvery=$SST_G2_PRINT_EVERY
sstG2PlateauTol=$SST_G2_PLATEAU_TOL
sstG2PlateauSamples=$SST_G2_PLATEAU_SAMPLES
sstModel=SST-2003m
sstIntensity=$SST_INTENSITY
sstLengthScaleOverD=$SST_LENGTH_RATIO
sstAlphaK=$SST_ALPHA_K
sstAlphaOmega=$SST_ALPHA_OMEGA
sstTol=$SST_TOL
sstLinearRtol=$SST_LINEAR_RTOL
sstLinearMax=$SST_LINEAR_MAX
sstOmegaWallMode=$SST_OMEGA_WALL_MODE
sstWallPenaltyGamma=$SST_WALL_PENALTY_GAMMA
sstOfKappa=$SST_OF_KAPPA
sstOfE=$SST_OF_E
sstOfBeta1=$SST_OF_BETA1
sstOfReBlend=$SST_OF_RE_BLEND
sstOfProduction=$SST_OF_PRODUCTION
sstOfProductionScale=$SST_OF_PRODUCTION_SCALE
sstOfWallVolumeScale=$SST_OF_WALL_VOLUME_SCALE
sstOfProductionLimit=$SST_OF_PRODUCTION_LIMIT
CFG

make -C "$ROOT" -j"$(nproc)" ARCH="${ARCH:-sm_86}" "nodals_hxt5b_${PREC}"

set +e
stdbuf -oL -eL "$EXE" \
  --mesh "$MESH" --vtu "$VTU" --re "$RE" --bulk "$BULK" --mixlen-scale "$MIXLEN_SCALE" \
  --wall-sample-fraction "$WALL_SAMPLE_FRACTION" --gamma "$MOM_NITSCHE_GAMMA" --schur-nitsche-gamma "$SCHUR_NITSCHE_GAMMA" \
  --max-outer "$MAX_OUTER" --simple-tol "$SIMPLE_TOL" --outer-cont-atol "$OUTER_CONT_ATOL" \
  --alpha-u "$ALPHA_U" --alpha-p "$ALPHA_P" --rau-scale "$RAU_SCALE" \
  --momentum-work "$MOM_WORK" --mom-omega "$MOM_OMEGA" --mom-rtol "$MOM_RTOL" --mom-atol "$MOM_ATOL" --mom-max "$MOM_MAX" --mom-cap-policy "$MOM_CAP_POLICY" \
  --pressure-solver "$PRESSURE_SOLVER" --p-rtol "$P_RTOL" --p-atol "$P_ATOL" --p-max "$P_MAX" \
  --amg-hierarchy "$AMG_HIERARCHY" --cf-coarsening "$CF_COARSENING" --cf-strength "$CF_STRENGTH" \
  --cf-theta "$CF_THETA" --cf-interp "$CF_INTERP" --cf-pmax "$CF_PMAX" --cf-aggressive-first "$CF_AGGRESSIVE_FIRST" \
  --amg-terminal "$AMG_TERMINAL" --amg-smoother "$AMG_SMOOTHER" --amg-jacobi-omega "$AMG_JACOBI_OMEGA" \
  --amg-jacobi-fine-sweeps "$AMG_JACOBI_FINE_SWEEPS" --amg-jacobi-coarse-sweeps "$AMG_JACOBI_COARSE_SWEEPS" \
  --amg-cheb-degree "$AMG_CHEB_DEGREE" --amg-power-its "$AMG_POWER_ITS" \
  --amg-lambda-safety "$AMG_LAMBDA_SAFETY" --amg-lambda-low-fraction "$AMG_LAMBDA_LOW_FRACTION" \
  --amg-spectrum-policy "$AMG_SPECTRUM_POLICY" \
  --amg-mcgs-fine-sweeps "$AMG_MCGS_FINE_SWEEPS" --amg-mcgs-coarse-sweeps "$AMG_MCGS_COARSE_SWEEPS" \
  --amg-mcgs-omega "$AMG_MCGS_OMEGA" --amg-mcgs-order "$AMG_MCGS_ORDER" \
  --sst-g1 "$SST_G1" --sst-g2 "$SST_G2" --sst-g2-start "$SST_G2_START" --sst-g2-max "$SST_G2_MAX" --sst-g2-print-every "$SST_G2_PRINT_EVERY" --sst-g2-plateau-tol "$SST_G2_PLATEAU_TOL" --sst-g2-plateau-samples "$SST_G2_PLATEAU_SAMPLES" --sst-intensity "$SST_INTENSITY" --sst-length-ratio "$SST_LENGTH_RATIO" \
  --sst-alpha-k "$SST_ALPHA_K" --sst-alpha-omega "$SST_ALPHA_OMEGA" --sst-tol "$SST_TOL" --sst-max "$SST_MAX" \
  --sst-linear-rtol "$SST_LINEAR_RTOL" --sst-linear-atol "$SST_LINEAR_ATOL" --sst-linear-max "$SST_LINEAR_MAX" --sst-gs-omega "$SST_GS_OMEGA" \
  --sst-k-floor "$SST_K_FLOOR" --sst-omega-floor "$SST_OMEGA_FLOOR" --sst-wall-penalty-gamma "$SST_WALL_PENALTY_GAMMA" --sst-omega-wall-mode "$SST_OMEGA_WALL_MODE" \
  --sst-of-kappa "$SST_OF_KAPPA" --sst-of-E "$SST_OF_E" --sst-of-beta1 "$SST_OF_BETA1" --sst-of-re-blend "$SST_OF_RE_BLEND" --sst-of-production "$SST_OF_PRODUCTION" --sst-of-production-scale "$SST_OF_PRODUCTION_SCALE" --sst-of-wall-volume-scale "$SST_OF_WALL_VOLUME_SCALE" --sst-of-production-limit "$SST_OF_PRODUCTION_LIMIT" --sst-print-every "$SST_PRINT_EVERY" \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

echo
echo "=== HXT5B SUMMARY ==="
grep -E 'NODALS_SST_G1|SST_G1_GATE_STATUS|NODALS_SST_G2|SST_G2_GATE_STATUS|NODALS_HXT5B_(DIRECTIONAL_SETUP|CONFIG|BOUNDARY_SETUP|MIXLEN|FINE_CSR_PARITY|SIMPLE|PRESSURE_FAIL|PHYSICS|OUTPUT)|NODALS_GPU_G8_CF_(SPECTRUM|HIERARCHY)|HXT5B_GATE_STATUS' "$LOG" || true
echo "HXT5B_LOG=$LOG"
echo "HXT5B_VTU=$VTU"
exit "$rc"
