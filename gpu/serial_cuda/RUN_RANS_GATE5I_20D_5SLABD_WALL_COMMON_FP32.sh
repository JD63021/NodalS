#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="${WALL_TRACTION_MODE:-spalding}"
REF_F="${WALL_REFERENCE_F:-0.0284003}"

case "$MODE" in
  spalding) DEFAULT_CASE="baseline" ;;
  reference_shear) DEFAULT_CASE="reference_shear" ;;
  *) echo "ERROR: WALL_TRACTION_MODE must be spalding or reference_shear"; exit 2 ;;
esac
CASE_NAME="${CASE_NAME:-$DEFAULT_CASE}"

MESH="${MESH:-$HOME/Desktop/meshes/nodals_vmfl003_20D_5slabD_256k/foam_case/constant/polyMesh}"
PREC="${PREC:-fp32}"
EXE="$HERE/nodals_gpu_h8_${PREC}"

RE="${RE:-13691.7402481}"
BULK="${BULK:-50}"
MIXLEN_SCALE="${MIXLEN_SCALE:-1.0}"

MAX_OUTER="${MAX_OUTER:-3000}"
SIMPLE_TOL="${SIMPLE_TOL:-1e-3}"
ALPHA_U="${ALPHA_U:-0.70}"
ALPHA_P="${ALPHA_P:-1.00}"
RAU_SCALE="${RAU_SCALE:-2.0}"

MOM_WORK="${MOM_WORK:-fgs1}"
MOM_OMEGA="${MOM_OMEGA:-1.0}"

PRESSURE_SOLVER="${PRESSURE_SOLVER:-pcg}"
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
AMG_SMOOTHER="${AMG_SMOOTHER:-cheb2}"
AMG_CHEB_DEGREE="${AMG_CHEB_DEGREE:-4}"
AMG_POWER_ITS="${AMG_POWER_ITS:-16}"
AMG_JACOBI_OMEGA="${AMG_JACOBI_OMEGA:-0.7}"

SUPG="${SUPG:-0}"

TAG="${TAG:-VMFL003_20D_5SLABD_GATE5I_${CASE_NAME^^}_${PREC^^}}"
OUTDIR="${OUTDIR:-$HERE/results_rans_20d_5slabD_gate5i/${CASE_NAME}}"
mkdir -p "$OUTDIR"
LOG="${LOG:-$OUTDIR/${TAG}.log}"
VTU_OUT="${VTU_OUT:-$OUTDIR/${TAG}.vtu}"
WALL_DIAG_CSV="${WALL_DIAG_CSV:-$OUTDIR/${TAG}_wall_faces.csv}"

[[ -x "$EXE" ]] || { echo "ERROR: executable missing: $EXE"; exit 3; }
[[ -f "$MESH/owner" ]] || { echo "ERROR: mesh missing: $MESH"; exit 4; }

cat <<CFG
=== GATE5I 20D 5-SLAB/D WALL CAUSAL TEST ===
case=$CASE_NAME
wallTractionMode=$MODE
wallReferenceF=$REF_F
precision=$PREC
mesh=$MESH
vtuOut=$VTU_OUT
wallDiagCsv=$WALL_DIAG_CSV
slabsPerD=5
dzOverD=0.2
supg=$SUPG
amgSmoother=$AMG_SMOOTHER
amgChebDegree=$AMG_CHEB_DEGREE
amgPowerIts=$AMG_POWER_ITS
alphaU=$ALPHA_U
alphaP=$ALPHA_P
rauScale=$RAU_SCALE
simpleTol=$SIMPLE_TOL
maxOuter=$MAX_OUTER
CFG

set +e
stdbuf -oL -eL "$EXE" \
  --mesh "$MESH" \
  --tag "$TAG" \
  --vtu-out "$VTU_OUT" \
  --wall wall --inlet inlet --outlet outlet \
  --re "$RE" --bulk "$BULK" \
  --mixing-length 1 --mixlen-scale "$MIXLEN_SCALE" \
  --dg-inlet 1 --weak-wall 1 \
  --wall-traction-mode "$MODE" \
  --wall-reference-f "$REF_F" \
  --wall-diag-csv "$WALL_DIAG_CSV" \
  --supg "$SUPG" \
  --run-mode converge --max-outer "$MAX_OUTER" --simple-tol "$SIMPLE_TOL" \
  --alpha-u "$ALPHA_U" --alpha-p "$ALPHA_P" --rau-scale "$RAU_SCALE" \
  --momentum-work "$MOM_WORK" --mom-omega "$MOM_OMEGA" \
  --pressure-solver "$PRESSURE_SOLVER" \
  --p-rtol "$P_RTOL" --p-atol "$P_ATOL" --p-max "$P_MAX" \
  --amg-hierarchy "$AMG_HIERARCHY" \
  --cf-coarsening "$CF_COARSENING" --cf-strength "$CF_STRENGTH" \
  --cf-theta "$CF_THETA" --cf-interp "$CF_INTERP" --cf-pmax "$CF_PMAX" \
  --cf-aggressive-first "$CF_AGGRESSIVE_FIRST" \
  --amg-terminal "$AMG_TERMINAL" \
  --amg-smoother "$AMG_SMOOTHER" \
  --amg-cheb-degree "$AMG_CHEB_DEGREE" \
  --amg-power-its "$AMG_POWER_ITS" \
  --amg-jacobi-omega "$AMG_JACOBI_OMEGA" \
  2>&1 | tee "$LOG"
solver_rc=${PIPESTATUS[0]}
set -e

echo
echo "=== GATE5I SUMMARY ==="
grep -E 'NODALS_GPU_RANS_GATE5I_WALL_MODE|NODALS_GPU_RANS_GATE5I_WALL_DIAG|NODALS_GPU_RANS_GATE3_WALL|NODALS_GPU_RANS_VTU|NODALS_GPU_H8_FULLCONV|NODALS_GPU_RESULT' "$LOG" || true

vtu_ok=0; [[ -s "$VTU_OUT" ]] && vtu_ok=1
wall_ok=0; [[ -s "$WALL_DIAG_CSV" ]] && wall_ok=1
result_ok=0; grep -q '^NODALS_GPU_RESULT .*status=PASS' "$LOG" && result_ok=1

if (( solver_rc==0 && result_ok==1 && vtu_ok==1 && wall_ok==1 )); then
  echo "NODALS_GPU_RANS_GATE5I_RUNNER case=$CASE_NAME mode=$MODE solverRC=0 resultPass=1 vtuPass=1 wallCsvPass=1 status=PASS log=$LOG vtu=$VTU_OUT wallCsv=$WALL_DIAG_CSV"
  exit 0
fi
echo "NODALS_GPU_RANS_GATE5I_RUNNER case=$CASE_NAME mode=$MODE solverRC=$solver_rc resultPass=$result_ok vtuPass=$vtu_ok wallCsvPass=$wall_ok status=FAIL log=$LOG vtu=$VTU_OUT wallCsv=$WALL_DIAG_CSV"
exit "$solver_rc"
