#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MESH="${MESH:-$HOME/Desktop/meshes/nodals_vmfl003_100D_20Dfine_Dlong_1125k/foam_case/constant/polyMesh}"
PREC="${PREC:-fp32}"
EXE="$HERE/nodals_gpu_h8_${PREC}"

RE="${RE:-13691.7402481}"
BULK="${BULK:-50}"
MIXLEN_SCALE="${MIXLEN_SCALE:-1.0}"

SUPG="${SUPG:-0}"
SUPG_TAU_SCALE="${SUPG_TAU_SCALE:-0.05}"
SUPG_MAGIC="${SUPG_MAGIC:-9.0}"
SUPG_FORM="${SUPG_FORM:-implicit}"
SUPG_QUAD_POINTS="${SUPG_QUAD_POINTS:-64}"
SUPG_KERNEL="${SUPG_KERNEL:-mixed64_8}"

MAX_OUTER="${MAX_OUTER:-3000}"
SIMPLE_TOL="${SIMPLE_TOL:-1e-3}"
ALPHA_U="${ALPHA_U:-0.70}"
ALPHA_P="${ALPHA_P:-1.00}"
RAU_SCALE="${RAU_SCALE:-2.0}"

MOM_WORK="${MOM_WORK:-fgs1}"
MOM_OMEGA="${MOM_OMEGA:-1.0}"
MOM_RTOL="${MOM_RTOL:-1e-6}"
MOM_ATOL="${MOM_ATOL:-1e-12}"
MOM_DROP="${MOM_DROP:-0.1}"
MOM_MAX="${MOM_MAX:-20000}"

PRESSURE_SOLVER="${PRESSURE_SOLVER:-pcg}"
P_RTOL="${P_RTOL:-0.90}"
P_ATOL="${P_ATOL:-1e-12}"
P_MAX="${P_MAX:-50}"
PRESSURE_RICHARDSON_OMEGA="${PRESSURE_RICHARDSON_OMEGA:-1.0}"
PRESSURE_CHEB_DEGREE="${PRESSURE_CHEB_DEGREE:-3}"
PRESSURE_POWER_ITS="${PRESSURE_POWER_ITS:-6}"
PRESSURE_CHEB_LOW_FRACTION="${PRESSURE_CHEB_LOW_FRACTION:-0.05}"
PRESSURE_POWER_SAFETY="${PRESSURE_POWER_SAFETY:-1.15}"

AMG_HIERARCHY="${AMG_HIERARCHY:-cf}"
CF_COARSENING="${CF_COARSENING:-pmis}"
CF_STRENGTH="${CF_STRENGTH:-classical-negative}"
CF_THETA="${CF_THETA:-0.25}"
CF_INTERP="${CF_INTERP:-exti}"
CF_PMAX="${CF_PMAX:-8}"
CF_AGGRESSIVE_FIRST="${CF_AGGRESSIVE_FIRST:-0}"
AMG_SMOOTHER="${AMG_SMOOTHER:-jacobi}"
AMG_JACOBI_OMEGA="${AMG_JACOBI_OMEGA:-0.7}"
AMG_JACOBI_FINE_SWEEPS="${AMG_JACOBI_FINE_SWEEPS:-1}"
AMG_JACOBI_COARSE_SWEEPS="${AMG_JACOBI_COARSE_SWEEPS:-1}"
AMG_MCGS_OMEGA="${AMG_MCGS_OMEGA:-1.0}"
AMG_MCGS_FINE_SWEEPS="${AMG_MCGS_FINE_SWEEPS:-1}"
AMG_MCGS_COARSE_SWEEPS="${AMG_MCGS_COARSE_SWEEPS:-1}"
AMG_MCGS_ORDER="${AMG_MCGS_ORDER:-symmetric}"
AMG_CHEB_DEGREE="${AMG_CHEB_DEGREE:-2}"
AMG_POWER_ITS="${AMG_POWER_ITS:-16}"
AMG_LAMBDA_SAFETY="${AMG_LAMBDA_SAFETY:-1.5}"
AMG_LAMBDA_LOW_FRACTION="${AMG_LAMBDA_LOW_FRACTION:-0.05}"
AMG_SPECTRUM_POLICY="${AMG_SPECTRUM_POLICY:-auto}"
AMG_TERMINAL="${AMG_TERMINAL:-1000}"

TAG="${TAG:-VMFL003_100D_20DFINE_DLONG_GATE5D_${PREC^^}}"
OUTDIR="${OUTDIR:-$HERE/results_rans_100d_20dfine_dlong}"
mkdir -p "$OUTDIR"
LOG="${LOG:-$OUTDIR/${TAG}.log}"
VTU_OUT="${VTU_OUT:-$OUTDIR/${TAG}.vtu}"

[[ -x "$EXE" ]] || { echo "ERROR: executable missing: $EXE"; exit 2; }
[[ -f "$MESH/owner" ]] || { echo "ERROR: 100D mesh missing: $MESH"; exit 3; }

cat <<CFG
=== GATE5D 100D: 20D FINE + D-LONG DOWNSTREAM RANS SETTINGS ===
precision=$PREC
mesh=$MESH
vtuOut=$VTU_OUT
axialPolicy=0-20D_original_Dover18__20-100D_Dlong
re=$RE
bulk=$BULK
mixlenScale=$MIXLEN_SCALE
supg=$SUPG
supgTauScale=$SUPG_TAU_SCALE
supgMagic=$SUPG_MAGIC
supgForm=$SUPG_FORM
supgQuadPoints=$SUPG_QUAD_POINTS
supgKernel=$SUPG_KERNEL
dgInlet=ON
weakWall=ON
maxOuter=$MAX_OUTER
simpleTol=$SIMPLE_TOL
alphaU=$ALPHA_U
alphaP=$ALPHA_P
rauScale=$RAU_SCALE
momentumWork=$MOM_WORK
momentumOmega=$MOM_OMEGA
pressureSolver=$PRESSURE_SOLVER
pressureRtol=$P_RTOL
pressureAtol=$P_ATOL
pressureMax=$P_MAX
pressureRichardsonOmega=$PRESSURE_RICHARDSON_OMEGA
pressureChebDegree=$PRESSURE_CHEB_DEGREE
pressurePowerIts=$PRESSURE_POWER_ITS
pressureChebLowFraction=$PRESSURE_CHEB_LOW_FRACTION
pressurePowerSafety=$PRESSURE_POWER_SAFETY
amgHierarchy=$AMG_HIERARCHY
amgSmoother=$AMG_SMOOTHER
amgJacobiOmega=$AMG_JACOBI_OMEGA
amgJacobiFineSweeps=$AMG_JACOBI_FINE_SWEEPS
amgJacobiCoarseSweeps=$AMG_JACOBI_COARSE_SWEEPS
amgMcgsOmega=$AMG_MCGS_OMEGA
amgMcgsFineSweeps=$AMG_MCGS_FINE_SWEEPS
amgMcgsCoarseSweeps=$AMG_MCGS_COARSE_SWEEPS
amgMcgsOrder=$AMG_MCGS_ORDER
amgChebDegree=$AMG_CHEB_DEGREE
amgPowerIts=$AMG_POWER_ITS
amgLambdaSafety=$AMG_LAMBDA_SAFETY
amgLambdaLowFraction=$AMG_LAMBDA_LOW_FRACTION
CFG

set +e
NODALS_GPU_SUPG_KERNEL="$SUPG_KERNEL" stdbuf -oL -eL "$EXE" \
  --mesh "$MESH" \
  --tag "$TAG" \
  --vtu-out "$VTU_OUT" \
  --wall wall --inlet inlet --outlet outlet \
  --re "$RE" --bulk "$BULK" \
  --mixing-length 1 --mixlen-scale "$MIXLEN_SCALE" \
  --dg-inlet 1 --weak-wall 1 \
  --supg "$SUPG" --supg-tau-scale "$SUPG_TAU_SCALE" --supg-magic "$SUPG_MAGIC" \
  --supg-form "$SUPG_FORM" --supg-quad-points "$SUPG_QUAD_POINTS" \
  --run-mode converge --max-outer "$MAX_OUTER" --simple-tol "$SIMPLE_TOL" \
  --alpha-u "$ALPHA_U" --alpha-p "$ALPHA_P" --rau-scale "$RAU_SCALE" \
  --momentum-work "$MOM_WORK" \
  --mom-omega "$MOM_OMEGA" --mom-rtol "$MOM_RTOL" --mom-atol "$MOM_ATOL" \
  --mom-drop "$MOM_DROP" --mom-max "$MOM_MAX" \
  --pressure-solver "$PRESSURE_SOLVER" \
  --p-rtol "$P_RTOL" --p-atol "$P_ATOL" --p-max "$P_MAX" \
  --pressure-richardson-omega "$PRESSURE_RICHARDSON_OMEGA" \
  --pressure-cheb-degree "$PRESSURE_CHEB_DEGREE" \
  --pressure-power-its "$PRESSURE_POWER_ITS" \
  --pressure-cheb-low-fraction "$PRESSURE_CHEB_LOW_FRACTION" \
  --pressure-power-safety "$PRESSURE_POWER_SAFETY" \
  --amg-hierarchy "$AMG_HIERARCHY" \
  --cf-coarsening "$CF_COARSENING" --cf-strength "$CF_STRENGTH" \
  --cf-theta "$CF_THETA" --cf-interp "$CF_INTERP" --cf-pmax "$CF_PMAX" \
  --cf-aggressive-first "$CF_AGGRESSIVE_FIRST" \
  --amg-terminal "$AMG_TERMINAL" \
  --amg-smoother "$AMG_SMOOTHER" --amg-jacobi-omega "$AMG_JACOBI_OMEGA" \
  --amg-jacobi-fine-sweeps "$AMG_JACOBI_FINE_SWEEPS" \
  --amg-jacobi-coarse-sweeps "$AMG_JACOBI_COARSE_SWEEPS" \
  --amg-mcgs-omega "$AMG_MCGS_OMEGA" \
  --amg-mcgs-fine-sweeps "$AMG_MCGS_FINE_SWEEPS" \
  --amg-mcgs-coarse-sweeps "$AMG_MCGS_COARSE_SWEEPS" \
  --amg-mcgs-order "$AMG_MCGS_ORDER" \
  --amg-cheb-degree "$AMG_CHEB_DEGREE" \
  --amg-power-its "$AMG_POWER_ITS" \
  --amg-lambda-safety "$AMG_LAMBDA_SAFETY" \
  --amg-lambda-low-fraction "$AMG_LAMBDA_LOW_FRACTION" \
  --amg-spectrum-policy "$AMG_SPECTRUM_POLICY" \
  2>&1 | tee "$LOG"
solver_rc=${PIPESTATUS[0]}
set -e

echo
echo "=== GATE5D 100D SUMMARY ==="
grep -E 'NODALS_GPU_RANS_SOLVER_CONTROLS|NODALS_GPU_RANS_SUPG|NODALS_GPU_H8_AMG_SMOOTHER|NODALS_GPU_AMG_MCGS_COLOR|NODALS_GPU_RANS_GATE1_MIXLEN|NODALS_GPU_RANS_GATE3_WALL|NODALS_GPU_RANS_VTU|NODALS_GPU_H8_FULLCONV|NODALS_GPU_H8_RESIDENCY|NODALS_GPU_RESULT' "$LOG" || true

vtu_ok=0
if [[ -s "$VTU_OUT" ]]; then
  vtu_ok=1
  printf 'NODALS_GPU_RANS_GATE5D_VTU_FILE path=%s bytes=%s status=PASS\n' \
    "$VTU_OUT" "$(stat -c %s "$VTU_OUT")"
else
  echo "NODALS_GPU_RANS_GATE5D_VTU_FILE path=$VTU_OUT status=FAIL"
fi

result_ok=0
grep -q '^NODALS_GPU_RESULT .*status=PASS' "$LOG" && result_ok=1
if (( solver_rc==0 && result_ok==1 && vtu_ok==1 )); then
  echo "NODALS_GPU_RANS_GATE5D_100D_RUNNER solverRC=0 resultPass=1 vtuPass=1 status=PASS log=$LOG vtu=$VTU_OUT"
  exit 0
fi

echo "NODALS_GPU_RANS_GATE5D_100D_RUNNER solverRC=$solver_rc resultPass=$result_ok vtuPass=$vtu_ok status=FAIL log=$LOG vtu=$VTU_OUT"
exit "$solver_rc"
