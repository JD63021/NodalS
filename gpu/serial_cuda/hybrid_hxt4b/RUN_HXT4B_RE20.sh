#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PREC="${PREC:-fp32}"
case "$PREC" in fp32|fp64) ;; *) echo "ERROR: PREC must be fp32 or fp64"; exit 2;; esac
EXE="$ROOT/nodals_hxt4b_${PREC}"
MESH="${MESH:-$ROOT/../hybrid_hxt1/work/HXT1_mesh.bin}"
OUTDIR="${OUTDIR:-$ROOT/results_re20_${PREC}}"
mkdir -p "$OUTDIR"
LOG="${LOG:-$OUTDIR/HXT4B_RE20_${PREC}.log}"
VTU="${VTU:-$OUTDIR/HXT4B_RE20_${PREC}.vtu}"

RE="${RE:-20}"
BULK="${BULK:-1.0}"
GAMMA="${GAMMA:-50}"

# Production SIMPLE defaults / requested HXT4B defaults.
MAX_OUTER="${MAX_OUTER:-3000}"
SIMPLE_TOL="${SIMPLE_TOL:-1e-3}"
ALPHA_U="${ALPHA_U:-0.70}"
ALPHA_P="${ALPHA_P:-1.00}"
RAU_SCALE="${RAU_SCALE:-2.0}"

# Fixed-work velocity smoother. SGS1 is intentionally the HXT4B default.
MOM_WORK="${MOM_WORK:-sgs1}"
MOM_OMEGA="${MOM_OMEGA:-1.0}"
MOM_RTOL="${MOM_RTOL:-0.5}"
MOM_ATOL="${MOM_ATOL:-1e-8}"

# HXT4B pressure remains PCG + Jacobi. Full GPU AMG arrives in HXT4C.
P_RTOL="${P_RTOL:-0.90}"
P_ATOL="${P_ATOL:-1e-12}"
P_MAX="${P_MAX:-50}"

[[ -f "$MESH" ]] || { echo "ERROR: full 10D HXT1 mesh missing: $MESH"; exit 2; }

cat <<CFG
=== HXT4B RE=20 HYBRID GPU SETTINGS ===
precision=$PREC
mesh=$MESH
vtu=$VTU
re=$RE
bulk=$BULK
gamma=$GAMMA
convection=central_advective_lagged
maxOuter=$MAX_OUTER
simpleTol=$SIMPLE_TOL
alphaU=$ALPHA_U
alphaP=$ALPHA_P
rauScale=$RAU_SCALE
rauPenaltyIncluded=0
rauIncludesConvectionDiagonal=1
momentumWork=$MOM_WORK
momentumOmega=$MOM_OMEGA
momentumRtol=$MOM_RTOL
momentumAtol=$MOM_ATOL
fixedWorkIgnoresMomentumTolerances=1
pressureSolver=pcg_jacobi
pressureRtol=$P_RTOL
pressureAtol=$P_ATOL
pressureMax=$P_MAX
CFG

echo
echo "=== BUILD HXT4B FP32 + FP64 ==="
make -C "$ROOT" -j"$(nproc)" ARCH="${ARCH:-sm_86}" 2>&1 | tee "$OUTDIR/BUILD.log"

echo
echo "=== LIVE HXT4B SOLVE ==="
runner=("$EXE")
if command -v stdbuf >/dev/null 2>&1; then runner=(stdbuf -oL -eL "$EXE"); fi
set +e
"${runner[@]}" \
  --mesh "$MESH" --vtu "$VTU" \
  --re "$RE" --bulk "$BULK" --gamma "$GAMMA" \
  --max-outer "$MAX_OUTER" --simple-tol "$SIMPLE_TOL" \
  --alpha-u "$ALPHA_U" --alpha-p "$ALPHA_P" --rau-scale "$RAU_SCALE" \
  --momentum-work "$MOM_WORK" --mom-omega "$MOM_OMEGA" \
  --mom-rtol "$MOM_RTOL" --mom-atol "$MOM_ATOL" \
  --p-rtol "$P_RTOL" --p-atol "$P_ATOL" --p-max "$P_MAX" \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

echo
echo "=== HXT4B SUMMARY ==="
grep -E '^NODALS_HXT4B_(CONFIG|HAGEN_POISEUILLE|SIMPLE|PHYSICS|OUTPUT)|^HXT4B_GATE_STATUS=' "$LOG" | tail -30 || true

echo "HXT4B_LOG=$LOG"
echo "HXT4B_VTU=$VTU"
exit "$rc"
