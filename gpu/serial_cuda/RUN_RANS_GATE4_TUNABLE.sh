#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MESH="${MESH:-$HOME/Desktop/meshes/nodals_vmfl003_10D_radial460k/foam_case/constant/polyMesh}"
PREC="${PREC:-fp32}"
EXE="$HERE/nodals_gpu_h8_${PREC}"

# Physics / case controls
RE="${RE:-13691.7402481}"
BULK="${BULK:-50}"
MIXLEN_SCALE="${MIXLEN_SCALE:-1.0}"

# Outer SIMPLE controls
MAX_OUTER="${MAX_OUTER:-3000}"
SIMPLE_TOL="${SIMPLE_TOL:-1e-3}"
ALPHA_U="${ALPHA_U:-0.50}"
ALPHA_P="${ALPHA_P:-0.50}"
RAU_SCALE="${RAU_SCALE:-1.0}"

# Momentum controls.  Current production path is fixed one-sweep FGS1.
MOM_WORK="${MOM_WORK:-fgs1}"
MOM_OMEGA="${MOM_OMEGA:-1.0}"
MOM_RTOL="${MOM_RTOL:-1e-6}"
MOM_ATOL="${MOM_ATOL:-1e-12}"
MOM_DROP="${MOM_DROP:-0.1}"
MOM_MAX="${MOM_MAX:-20000}"

# Pressure controls
PRESSURE_SOLVER="${PRESSURE_SOLVER:-pcg}"
P_RTOL="${P_RTOL:-0.90}"
P_ATOL="${P_ATOL:-1e-12}"
P_MAX="${P_MAX:-50}"

# AMG controls
AMG_HIERARCHY="${AMG_HIERARCHY:-cf}"
CF_COARSENING="${CF_COARSENING:-pmis}"
CF_STRENGTH="${CF_STRENGTH:-classical-negative}"
CF_THETA="${CF_THETA:-0.25}"
CF_INTERP="${CF_INTERP:-exti}"
CF_PMAX="${CF_PMAX:-8}"
CF_AGGRESSIVE_FIRST="${CF_AGGRESSIVE_FIRST:-0}"
AMG_SMOOTHER="${AMG_SMOOTHER:-jacobi}"
AMG_JACOBI_OMEGA="${AMG_JACOBI_OMEGA:-0.7}"
AMG_SPECTRUM_POLICY="${AMG_SPECTRUM_POLICY:-auto}"
AMG_TERMINAL="${AMG_TERMINAL:-1000}"

TAG="${TAG:-VMFL003_10D_GATE4_TUNABLE_${PREC^^}}"
LOG="${LOG:-$HERE/${TAG}.log}"

[[ -x "$EXE" ]] || { echo "ERROR: executable missing: $EXE"; exit 2; }
[[ -f "$MESH/owner" ]] || { echo "ERROR: mesh missing: $MESH"; exit 3; }

cat <<CFG
=== GATE4 TUNABLE SETTINGS ===
precision=$PREC
mesh=$MESH
re=$RE
bulk=$BULK
mixlenScale=$MIXLEN_SCALE
supg=OFF
dgInlet=ON
weakWall=ON
simpleVariant=SIMPLE             # current GPU algorithm; not silently changed
uRelaxMode=ROW_L1                # current GPU relaxation metric
rauMode=RELAXED_DIAG             # current GPU correction metric
maxOuter=$MAX_OUTER
simpleTol=$SIMPLE_TOL
alphaU=$ALPHA_U
alphaP=$ALPHA_P
rauScale=$RAU_SCALE
momentumWork=$MOM_WORK
momentumOmega=$MOM_OMEGA
momentumRtol=$MOM_RTOL           # parsed but inert for fixed one-sweep FGS1
momentumAtol=$MOM_ATOL           # parsed but inert for fixed one-sweep FGS1
momentumDrop=$MOM_DROP           # parsed but inert for fixed one-sweep FGS1
momentumMax=$MOM_MAX             # parsed but inert for fixed one-sweep FGS1
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
cfAggressiveFirst=$CF_AGGRESSIVE_FIRST
amgSmoother=$AMG_SMOOTHER
amgJacobiOmega=$AMG_JACOBI_OMEGA
amgSpectrumPolicy=$AMG_SPECTRUM_POLICY
amgTerminal=$AMG_TERMINAL
fineCsrRefreshEvery=1
CFG

set +e
stdbuf -oL -eL "$EXE" \
  --mesh "$MESH" \
  --tag "$TAG" \
  --wall wall --inlet inlet --outlet outlet \
  --re "$RE" --bulk "$BULK" \
  --supg 0 \
  --mixing-length 1 --mixlen-scale "$MIXLEN_SCALE" \
  --dg-inlet 1 \
  --weak-wall 1 \
  --run-mode converge --max-outer "$MAX_OUTER" --simple-tol "$SIMPLE_TOL" \
  --alpha-u "$ALPHA_U" --alpha-p "$ALPHA_P" --rau-scale "$RAU_SCALE" \
  --momentum-work "$MOM_WORK" \
  --mom-omega "$MOM_OMEGA" --mom-rtol "$MOM_RTOL" --mom-atol "$MOM_ATOL" \
  --mom-drop "$MOM_DROP" --mom-max "$MOM_MAX" \
  --pressure-solver "$PRESSURE_SOLVER" \
  --p-rtol "$P_RTOL" --p-atol "$P_ATOL" --p-max "$P_MAX" \
  --amg-hierarchy "$AMG_HIERARCHY" \
  --cf-coarsening "$CF_COARSENING" --cf-strength "$CF_STRENGTH" \
  --cf-theta "$CF_THETA" --cf-interp "$CF_INTERP" --cf-pmax "$CF_PMAX" \
  --cf-aggressive-first "$CF_AGGRESSIVE_FIRST" \
  --amg-terminal "$AMG_TERMINAL" \
  --amg-smoother "$AMG_SMOOTHER" --amg-jacobi-omega "$AMG_JACOBI_OMEGA" \
  --amg-spectrum-policy "$AMG_SPECTRUM_POLICY" \
  2>&1 | tee "$LOG"
solver_rc=${PIPESTATUS[0]}
set -e

echo
echo "=== GATE4 TUNABLE SUMMARY ==="
grep -E 'NODALS_GPU_RANS_SOLVER_CONTROLS|NODALS_GPU_RANS_GATE1_MIXLEN|NODALS_GPU_RANS_GATE3_WALL|NODALS_GPU_H8_FULLCONV|NODALS_GPU_H8_RESIDENCY|NODALS_GPU_RESULT' "$LOG" || true

if grep -q '^NODALS_GPU_H8_FULLCONV ' "$LOG" && [[ -x "$HERE/CHECK_RANS_GATE4_10D.py" || -f "$HERE/CHECK_RANS_GATE4_10D.py" ]]; then
  set +e
  python3 "$HERE/CHECK_RANS_GATE4_10D.py" "$LOG"
  check_rc=$?
  set -e
else
  check_rc=0
fi

echo "NODALS_GPU_RANS_GATE4_TUNABLE_RUNNER solverRC=$solver_rc checkerRC=$check_rc log=$LOG"
exit "$solver_rc"
