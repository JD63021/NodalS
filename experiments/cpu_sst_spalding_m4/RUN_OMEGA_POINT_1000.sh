#!/usr/bin/env bash
# Usage: bash experiments/cpu_sst_spalding_m4/RUN_OMEGA_POINT_1000.sh [of_auto|sample_log] [iterations=1000] [gamma=10000] [momentum_nut_scale=1.0 (diagnostic)]
# First run only 2 or 20 iterations as a numerical gate. Original Stage1 inputs are read-only.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODE="of_auto"; STEPS="1000"; GAMMA="10000"; MOM_NUT_SCALE="1"
if (( $#>=1 )); then MODE="$1"; fi
if (( $#>=2 )); then STEPS="$2"; fi
if (( $#>=3 )); then GAMMA="$3"; fi
if (( $#>=4 )); then MOM_NUT_SCALE="$4"; fi
[[ "$MODE" == of_auto || "$MODE" == sample_log ]] || { echo "Allowed modes: of_auto|sample_log" >&2; exit 2; }
[[ "$STEPS" =~ ^[0-9]+$ ]] && (( STEPS>0 )) || { echo "Iterations must be positive" >&2; exit 2; }
STAGE1_ROOT="$HOME/NodalS_CPU_ENRICH_20261114/experiments/cpu_sst_spalding_tiny_pipe"
BASE_CASE="$STAGE1_ROOT/cases/TINY_PIPE_6D_RE72600_CPU_M4_CONTROL.case"
MESH="$STAGE1_ROOT/mesh/TINY_PIPE_D0889_RE72600_6D_DELAUNAY_20261114.msh"
[[ -f "$BASE_CASE" && -f "$MESH" ]] || { echo "Missing Stage1 inputs: $BASE_CASE $MESH" >&2; exit 3; }
RUN_ROOT="$HOME/NodalS_SST_LOCAL_RUNS/CPU_M4_ENRICH_OMEGA_POINT_20261114"
RUN="$RUN_ROOT/$MODE"_"$STEPS"_nut"${MOM_NUT_SCALE}"_"$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"
CASE="$RUN/experiment.case"
python3 - "$BASE_CASE" "$CASE" "$MODE" "$STEPS" "$GAMMA" "$MOM_NUT_SCALE" <<'PY'
import configparser,sys,math
src,dst,mode,steps,gamma,nut_scale=sys.argv[1:]
n=int(steps);g=float(gamma);scale=float(nut_scale)
if not(math.isfinite(scale) and 0<=scale<=1):
    raise SystemExit("sst.momentum_nut_scale must be between 0 and 1")
if not(1<=g<=1e8 and math.isfinite(g)): raise SystemExit("omega penalty_gamma must be in [1,1e8]")
p=configparser.ConfigParser(interpolation=None)
p.read(src)
def setval(section,key,val):p.set(section,key,str(val))
setval("simple","max_iterations",n)
setval("solver.u","method","fgmres")          # Unlike fixed m3-fgs, honors rtol.
setval("solver.u","rtol","1e-5")
setval("solver.u","atol","1e-10")
setval("solver.u","restart",40)
setval("solver.u","max_iterations",600)
setval("solver.p","rtol","1e-4")
setval("solver.p","atol","1e-12")
setval("solver.p","max_iterations",800)
setval("sst","momentum_nut_scale",format(scale,'.12g'))
setval("sst","k_sweeps",6)
setval("sst","omega_sweeps",6)
setval("omega_wall","method","of_auto")
setval("omega_wall","stage41_mode",mode)
setval("omega_wall","penalty_gamma",format(g,'.12g'))
setval("omega_wall","production","false")
setval("wall_function","momentum_bc","strong_noslip_enriched")
# Set a stricter diagnostic but intentionally disable early plateau stopping.
setval("convergence","pressure_drop_window",20)
setval("convergence","min_iterations",max(n+1,21))
setval("convergence","pressure_drop_relative_tolerance","1e-4")
setval("output","print_every",min(20,n))
setval("output","vtu_every",100)
p.add_section("enrichment")
setval("enrichment","enabled","true")
setval("enrichment","utau_initial","0.5859")
setval("enrichment","yplus_cutoff","80")
setval("enrichment","quadrature_order",16)
with open(dst,"w") as f:p.write(f)
print("OMEGA_POINT_CASE mode=%s steps=%d gamma=%.8g momentumNutScale=%.8g DIAGNOSTIC_ONLY=1 momMethod=fgmres momRtol=1e-5 pRtol=1e-4 scalarSweeps=6 minIterations=%d"%(mode,n,g,scale,max(n+1,21)))
PY
echo "NODALS_CPU_M4_OMEGA_TEST mode=$MODE steps=$STEPS gamma=$GAMMA momentumNutScale=$MOM_NUT_SCALE run=$RUN"
if [[ "$(printenv SKIP_BUILD 2>/dev/null || echo 0)" != 1 ]]; then
 make -C "$ROOT/turbulence/sst_cpu_mc" -j2 m4-sst
fi
BIN="$ROOT/turbulence/sst_cpu_mc/build/nodals_sst_cpu_mc_m4"
[[ -x "$BIN" ]] || { echo "No executable $BIN" >&2; exit 4; }
sha256sum "$BIN" "$MESH" "$CASE"
mpirun -np 1 "$BIN" --mesh "$MESH" --case "$CASE" \
 --expect-ranks 1 --max-outer "$STEPS" --output-dir "$RUN" \
 2>&1 | tee "$RUN/console.log"
grep -q "NODALS_CPU_M4_OMEGA_POINT mode=$MODE" "$RUN/console.log" || { echo "Missing sampled omega mode diagnostic" >&2; exit 5; }
grep -q 'NODALS_CPU_M4_ENRICH_WALL' "$RUN/console.log" || { echo "Missing physical-wall diagnostics" >&2; exit 6; }
grep -q 'SIMPLE_RESULT' "$RUN/console.log" || { echo "Run stopped before complete SIMPLE_RESULT" >&2; exit 7; }
echo "NODALS_CPU_M4_OMEGA_TEST_COMPLETE output=$RUN"
