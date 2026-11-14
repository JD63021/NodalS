#!/usr/bin/env bash
# One-rank CPU M4 A/B on the exact 1857-tet 6D pipe.
# Usage: bash experiments/cpu_sst_spalding_m4/RUN_TINY_PIPE.sh [control|enriched] [20|100|200]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODE="enriched"; STEPS="20"
if (( $#>=1 )); then MODE="$1"; fi
if (( $#>=2 )); then STEPS="$2"; fi
[[ "$MODE" == enriched || "$MODE" == control ]] || { echo "mode: enriched|control" >&2; exit 2; }
[[ "$STEPS" =~ ^[0-9]+$ ]] && ((STEPS>=1)) || { echo "STEPS must be positive" >&2; exit 2; }
STAGE1_ROOT="$HOME/NodalS_CPU_ENRICH_20261114/experiments/cpu_sst_spalding_tiny_pipe"
BASE_CASE="$STAGE1_ROOT/cases/TINY_PIPE_6D_RE72600_CPU_M4_CONTROL.case"
MESH="$STAGE1_ROOT/mesh/TINY_PIPE_D0889_RE72600_6D_DELAUNAY_20261114.msh"
[[ -f "$BASE_CASE" && -f "$MESH" ]] || { echo "Missing Stage1 case or mesh: $BASE_CASE $MESH" >&2; exit 3; }
RUN_ROOT="$HOME/NodalS_SST_LOCAL_RUNS/CPU_M4_SHARED_SPALDING_20261114"
RUN="$RUN_ROOT/"$MODE"_"$STEPS"_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"
CASE="$RUN/experiment.case"
python3 - "$BASE_CASE" "$CASE" "$MODE" "$STEPS" <<'PY'
import pathlib,re,sys
src,dst,mode,steps=sys.argv[1:]
s=pathlib.Path(src).read_text()
bits=s.split('[simple]')
if len(bits)!=2: raise SystemExit("Expected one [simple] section")
head,rest=bits
pat=r'(?m)^max_iterations\s*=\s*\d+\s*$'
if not re.search(pat,rest): raise SystemExit("Missing simple.max_iterations")
rest=re.sub(pat,'max_iterations = '+steps,rest,count=1)
s=head+'[simple]'+rest
if mode=='enriched':
    if '[wall_function]' not in s or '[wall_sampling]' not in s:
        raise SystemExit("Missing wall_function or wall_sampling section")
    s=s.replace('[wall_sampling]','momentum_bc = strong_noslip_enriched\n\n[wall_sampling]',1)
    s+='\n[enrichment]\nenabled = true\nutau_initial = 0.5859\nyplus_cutoff = 80\nquadrature_order = 16\n'
else:
    s+='\n[enrichment]\nenabled = false\n'
pathlib.Path(dst).write_text(s)
PY
echo "NODALS_TINY_CPU_M4_SETUP mode=$MODE steps=$STEPS mesh=$MESH case=$CASE run=$RUN"
if [[ "${SKIP_BUILD:-0}" != 1 ]]; then
  make -C "$ROOT/turbulence/sst_cpu_mc" -j2 m4-sst
fi
BIN="$ROOT/turbulence/sst_cpu_mc/build/nodals_sst_cpu_mc_m4"
[[ -x "$BIN" ]] || { echo "No compiled executable: $BIN" >&2; exit 4; }
sha256sum "$BIN" "$MESH" "$CASE"
mpirun -np 1 "$BIN" --mesh "$MESH" --case "$CASE" \
 --expect-ranks 1 --max-outer "$STEPS" --output-dir "$RUN" \
 2>&1 | tee "$RUN/console.log"
if [[ "$MODE" == enriched ]]; then
 grep -q 'NODALS_CPU_M4_ENRICH_CONFIG enabled=1' "$RUN/console.log" ||
    { echo "ERROR: enrichment was not enabled" >&2; exit 5; }
 grep -q 'NODALS_CPU_M4_ENRICH_WALL' "$RUN/console.log" ||
    { echo "ERROR: no physical wall stress report" >&2; exit 6; }
fi
grep -q 'SIMPLE_RESULT' "$RUN/console.log" ||
 { echo "ERROR: run did not complete" >&2; exit 7; }
echo "NODALS_TINY_CPU_M4_COMPLETE mode=$MODE output=$RUN"
