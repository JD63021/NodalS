#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MESH="${MESH:-$HOME/Desktop/meshes/nodals_vmfl003_20D_10slabD_511k/foam_case/constant/polyMesh}"
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

TAG="${TAG:-VMFL003_20D_10SLABD_NOSUPG_GATE5G_${PREC^^}}"
OUTDIR="${OUTDIR:-$HERE/results_rans_20d_10slabD_nosupg}"
mkdir -p "$OUTDIR"
LOG="${LOG:-$OUTDIR/${TAG}.log}"
VTU_OUT="${VTU_OUT:-$OUTDIR/${TAG}.vtu}"

[[ -x "$EXE" ]] || { echo "ERROR: executable missing: $EXE"; exit 2; }
[[ -f "$MESH/owner" ]] || { echo "ERROR: mesh missing: $MESH"; exit 3; }

cat <<CFG
=== GATE5G 20D: 10 SLABS/D, NO SUPG ===
precision=$PREC
mesh=$MESH
vtuOut=$VTU_OUT
slabsPerD=10
dzOverD=0.1
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
echo "=== GATE5G 20D SUMMARY ==="
grep -E 'NODALS_GPU_RANS_SOLVER_CONTROLS|NODALS_GPU_RANS_SUPG|NODALS_GPU_H8_AMG_SMOOTHER|NODALS_GPU_RANS_GATE1_MIXLEN|NODALS_GPU_RANS_GATE3_WALL|NODALS_GPU_RANS_VTU|NODALS_GPU_H8_FULLCONV|NODALS_GPU_H8_RESIDENCY|NODALS_GPU_RESULT' "$LOG" || true

vtu_ok=0
[[ -s "$VTU_OUT" ]] && vtu_ok=1
result_ok=0
grep -q '^NODALS_GPU_RESULT .*status=PASS' "$LOG" && result_ok=1

if (( solver_rc==0 && result_ok==1 && vtu_ok==1 )); then
  printf 'NODALS_GPU_RANS_GATE5G_VTU_FILE path=%s bytes=%s status=PASS\n' \
    "$VTU_OUT" "$(stat -c %s "$VTU_OUT")"
  echo "NODALS_GPU_RANS_GATE5G_20D_RUNNER solverRC=0 resultPass=1 vtuPass=1 status=PASS log=$LOG vtu=$VTU_OUT"
  exit 0
fi
echo "NODALS_GPU_RANS_GATE5G_20D_RUNNER solverRC=$solver_rc resultPass=$result_ok vtuPass=$vtu_ok status=FAIL log=$LOG vtu=$VTU_OUT"
exit "$solver_rc"
