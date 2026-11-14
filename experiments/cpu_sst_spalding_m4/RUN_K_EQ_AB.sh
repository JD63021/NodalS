#!/usr/bin/env bash
# Three-way experiment: A=no_flux, B=k_eq gamma1000, C=k_eq gamma10000.
# Omega remains sample_log at offwall point3 gamma=10000 in ALL cases.
# Usage: bash experiments/cpu_sst_spalding_m4/RUN_K_EQ_AB.sh [A|B|C] [2|20|50|100]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODE="B"; STEPS="20"
if (( $#>=1 )); then MODE="$1"; fi
if (( $#>=2 )); then STEPS="$2"; fi
[[ "$MODE" =~ ^[ABC]$ ]] || { echo 'Mode must be A, B, or C' >&2; exit 2; }
[[ "$STEPS" =~ ^[0-9]+$ ]] && (( STEPS>=1 )) || { echo 'Positive iteration count required' >&2; exit 2; }
STAGE1="$HOME/NodalS_CPU_ENRICH_20261114/experiments/cpu_sst_spalding_tiny_pipe"
BASE_CASE="$STAGE1/cases/TINY_PIPE_6D_RE72600_CPU_M4_CONTROL.case"
MESH="$STAGE1/mesh/TINY_PIPE_D0889_RE72600_6D_DELAUNAY_20261114.msh"
[[ -f "$BASE_CASE" && -f "$MESH" ]] || { echo "Missing Stage1 base case or mesh" >&2; exit 3; }
RUN_ROOT="$HOME/NodalS_SST_LOCAL_RUNS/CPU_M4_K_EQ_AB_20261114"
# Optional diagnostic UR_U override. Unset preserves original case exactly.
UR_TAG="base"
if [[ -n "${M4_DIAG_UR_U:-}" ]]; then UR_TAG="uru${M4_DIAG_UR_U}"; fi
FIX_TAG="dynamic"
if [[ -n "${M4_DIAG_FIXED_UTAU:-}" ]]; then FIX_TAG="targetutau${M4_DIAG_FIXED_UTAU}"; fi
RUN="$RUN_ROOT/"$MODE"_"$STEPS"_"$UR_TAG"_"$FIX_TAG"_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"
CASE="$RUN/experiment.case"
python3 - "$BASE_CASE" "$CASE" "$MODE" "$STEPS" <<'PY'
import sys,configparser,os,math
src,dst,mode,steps=sys.argv[1:]
n=int(steps)
p=configparser.ConfigParser(interpolation=None,strict=True)
with open(src) as f:p.read_file(f)
def put(section,key,val):
    if not p.has_section(section):p.add_section(section)
    p.set(section,key,str(val))
put("simple","max_iterations",n)
if "M4_DIAG_UR_U" in os.environ:
    relax=float(os.environ["M4_DIAG_UR_U"])
    if not (math.isfinite(relax) and 0<relax<=1):
        raise SystemExit("M4_DIAG_UR_U must be in (0,1]")
    put("simple","ur_u",format(relax,".12g"))
    print("K_EQ_UR_AUDIT ur_u=%.9g override=1"%relax)
if "M4_DIAG_FIXED_UTAU" in os.environ:
    target_utau=float(os.environ["M4_DIAG_FIXED_UTAU"])
    if not(math.isfinite(target_utau) and target_utau>0):
        raise SystemExit("M4_DIAG_FIXED_UTAU must be positive")
    put("sst","debug_fixed_target_utau",format(target_utau,".12g"))
    print("K_EQ_WALL_TARGET_DIAGNOSTIC mode=FIXED targetUTau=%.9g"%target_utau)
put("solver.u","method","fgmres")
put("solver.u","rtol","1e-5")
put("solver.u","atol","1e-10")
put("solver.u","restart",40)
put("solver.u","max_iterations",600)
put("solver.p","rtol","1e-4")
put("solver.p","atol","1e-12")
put("solver.p","max_iterations",800)
put("sst","k_sweeps",6)
put("sst","omega_sweeps",6)
put("omega_wall","method","of_auto")
put("omega_wall","stage41_mode","sample_log")
put("omega_wall","sample_index",3)
put("omega_wall","penalty_gamma",10000)
put("omega_wall","production","false")
put("wall_function","momentum_bc","strong_noslip_enriched")
put("k_wall","method","no_flux" if mode=="A" else "equilibrium_spalding")
put("k_wall","sample_index",3)
put("k_wall","penalty_gamma",1000 if mode=="B" else 10000)
# Same full five-point ell = y5 - y1 as omega (default), not separate heuristic.
put("convergence","pressure_drop_window",20)
put("convergence","min_iterations",max(n+1,21))
put("convergence","pressure_drop_relative_tolerance","1e-4")
put("output","print_every",min(20,n))
put("output","vtu_every",min(50,n))
put("enrichment","enabled","true")
put("enrichment","utau_initial","0.5859")
put("enrichment","yplus_cutoff","80")
put("enrichment","quadrature_order",16)
with open(dst,"w") as f:p.write(f)
print("K_EQ_AB_SETUP variant=%s kMethod=%s kGamma=%s omegaMode=sample_log omegaGamma=10000 maxOuter=%d minIterations=%d"%(mode,p["k_wall"]["method"],p["k_wall"]["penalty_gamma"],n,max(n+1,21)))
PY
echo "NODALS_K_EQ_AB_RUN mode=$MODE steps=$STEPS run=$RUN"
if [[ "$(printenv SKIP_BUILD 2>/dev/null || echo 0)" != 1 ]]; then
 make -C "$ROOT/turbulence/sst_cpu_mc" -j2 m4-sst
fi
BIN="$ROOT/turbulence/sst_cpu_mc/build/nodals_sst_cpu_mc_m4"
[[ -x "$BIN" ]] || { echo "No executable at $BIN" >&2; exit 4; }
sha256sum "$BIN" "$MESH" "$CASE"
mpirun -np 1 "$BIN" --mesh "$MESH" --case "$CASE" \
 --expect-ranks 1 --max-outer "$STEPS" --output-dir "$RUN" \
 2>&1 | tee "$RUN/console.log"
k_mode="equilibrium_spalding"
if [[ "$MODE" == A ]]; then k_mode="no_flux"; fi
grep -q "NODALS_CPU_M4_K_POINT mode=$k_mode" "$RUN/console.log" || {
 echo 'Missing k wall diagnostic (wrong binary?)' >&2; exit 5; }
grep -q 'NODALS_CPU_M4_OMEGA_POINT mode=sample_log' "$RUN/console.log" || {
 echo 'Missing omega sample_log diagnostic' >&2; exit 6; }
grep -q 'NODALS_CPU_M4_RANS_WALL_BUDGET' "$RUN/console.log" || {
 echo 'Missing effective wall stress diagnostic' >&2; exit 7; }
grep -q 'SIMPLE_RESULT' "$RUN/console.log" || {
 echo 'Incomplete SIMPLE run' >&2; exit 8; }
echo "NODALS_K_EQ_AB_COMPLETE mode=$MODE output=$RUN"
