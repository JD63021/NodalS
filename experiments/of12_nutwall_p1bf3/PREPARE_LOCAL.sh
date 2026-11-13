#!/usr/bin/env bash
# Workstation preflight for the independent no-slip wall-conductance experiment.
# Safe default: inspect + make editable Re=72,600 case; NO solver run.
set -euo pipefail
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$THIS_DIR" rev-parse --show-toplevel)"
OUT="${OUT:-$HOME/NodalS_SST_LOCAL_RUNS/OF12_NUTWALL_20261113}"
mkdir -p "$OUT"
LOG="$OUT/PREFLIGHT.log"
exec > >(tee "$LOG") 2>&1
echo "========== NODALS OF12 WALL-CONDUCTANCE EXPERIMENT / PREFLIGHT =========="
echo "Date: $(date -Is)"
echo "Experiment root: $ROOT"
echo "Branch: $(git -C "$ROOT" branch --show-current)"
echo "Commit: $(git -C "$ROOT" rev-parse HEAD)"
echo "Worktree changes: "
git -C "$ROOT" status --short | head -30
echo "NOTE: native Option-A wall operator is NOT yet implemented. Any current solver run is unchanged Robin baseline."
if [[ "$(git -C "$ROOT" branch --show-current)" != "exp/of12-nutwall-p1bf3-20261113" ]]; then
  echo "ERROR: select experimental branch worktree before running this file" >&2; exit 2
fi

MESH="${MESH_SRC:-$HOME/Downloads/NODALS_PIPE50_3D_DELAUNAY_20261111/PIPE50_3D_DELAUNAY.msh}"
if [[ ! -f "$MESH" ]]; then
 MESH="$(find "$HOME/Downloads" -maxdepth 6 -type f -name 'PIPE50_3D_DELAUNAY.msh' -print -quit)"
fi
if [[ ! -f "$MESH" ]]; then
 echo "ERROR: 333,779-tet Delaunay mesh missing. Supply MESH_SRC=/absolute/path/to/*.msh" >&2
 exit 3
fi
CASE="${CASE_SRC:-}"
if [[ -z "$CASE" ]]; then
 CASE="$(find "$HOME/Downloads" -maxdepth 7 -type f -name 'PIPE50_D3D_5PT_WIDE_5TO300.case' -print -quit)"
fi
if [[ -z "$CASE" ]]; then
 CASE="$(find "$HOME/Downloads" -maxdepth 7 -type f -iname '*PIPE50*D3D*.case' -print -quit)"
fi
if [[ -z "$CASE" ]]; then
 CASE="$HOME/Downloads/NODALS_PIPE_50D_MOODY_20261111/PIPE_50D_MOODY.case"
fi
if [[ ! -f "$CASE" ]]; then
 echo "ERROR: Delaunay or Moody editable case missing; use CASE_SRC=/absolute/path/to/reference.case" >&2
 exit 4
fi
echo "MESH=$MESH"
echo "CASE_SRC=$CASE"
ls -lh "$MESH" "$CASE"

CASE_OUT="$OUT/PIPE50_D3D_Re72600_OPTIONA_CONTROL.case"
python3 - "$CASE" "$CASE_OUT" "$MESH" <<'PY'
from pathlib import Path
import sys, configparser, re, math
src, dst, mesh = map(Path, sys.argv[1:])
cp=configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#",";"))
cp.read(src)
for sec in ("mesh","physics","inlet.main"):
    if not cp.has_section(sec):
        raise SystemExit(f"CASE_INVALID: missing section [{sec}]")
D=cp.getfloat("mesh","diameter")
nu=cp.getfloat("physics","kinematic_viscosity")
reynolds=72600.
U=reynolds*nu/D
mode=cp.get("inlet.main","mode",fallback="velocity").strip().lower()
if mode=="speed":
    vx=None
elif mode=="velocity":
    original=cp.get("inlet.main","velocity",fallback="")
    nums=[float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?",original)]
    if len(nums)!=3:
        raise SystemExit(f"CASE_INVALID: inlet.main.velocity expected 3 values, got {original!r}")
    mag=math.sqrt(sum(v*v for v in nums))
    if not mag>0: raise SystemExit("CASE_INVALID: zero inlet direction")
    vx=tuple(U*v/mag for v in nums)
else:
    raise SystemExit(f"CASE_INVALID: unexpected inlet mode {mode}")
updates={("mesh","path"):str(mesh.resolve()),
         ("physics","reynolds"):"72600"}
if mode=="speed":
    updates["inlet.main","speed"]=f"{U:.12g}"
else:
    updates["inlet.main","velocity"]="("+", ".join(f"{v:.12g}" for v in vx)+")"
if cp.has_option("inlet.main","profile_csv"):
    profile=Path(cp.get("inlet.main","profile_csv"))
    if not profile.is_absolute(): updates["inlet.main","profile_csv"]=str((src.parent/profile).resolve())
data=src.read_text().splitlines(keepends=True)
seen=set(); section=None; result=[]
for line in data:
    match=re.match(r'^\s*\[([^]]+)\]\s*(?:[#;].*)?$',line)
    if match: section=match.group(1).strip().lower()
    kv=re.match(r'^(\s*([A-Za-z0-9_.-]+)\s*=\s*)([^\r\n]*)(\r?\n?)$',line)
    key=(section,kv.group(2).lower()) if kv else None
    if key in updates:
        line=kv.group(1)+updates[key]+kv.group(4)
        seen.add(key)
    result.append(line)
missing=set(updates)-seen
if missing: raise SystemExit(f"CASE_INVALID: required entries absent {missing}; use an ordinary editable NodalS .case")
dst.write_text("".join(result))
print("PREPARED_CASE=",dst)
print(f"RE=72600 D={D:.10g} nu={nu:.12g} Ubulk={U:.10g} mode={mode}")
if vx: print("INLET_VECTOR=",vx)
print("SAMPLES=",cp.get("wall_function","samples",fallback="unset"))
print("STRESS=",cp.get("stress","form",fallback="unset"))
print("OMEGA_WALL=",cp.get("omega_wall","method",fallback="unset"))
PY

echo "Case section audit:"
grep -nE '^\[(mesh|physics|inlet.main|wall_function|omega_wall|stress|convection)\]|^(format|path|reynolds|velocity|speed|samples|method|form|distances|kinematic_viscosity|diameter)\s*=' "$CASE_OUT" | head -95 || true
echo
echo "Compilers:"
command -v nvcc || true
command -v python3 || true
echo
echo "Candidate solver binaries (only pre-existing binaries are supported by this preflight):"
BINARY="${BINARY:-}"
if [[ -z "$BINARY" ]]; then
 for f in "$HOME/NodalS_GPU_GENROBIN_ADAPTIVE/turbulence/sst_gpu_v2/build/nodals_sst"           "$HOME/NodalS/turbulence/sst_gpu_v2/build/nodals_sst"           "$HOME/NodalS_SST_MODULAR/turbulence/sst_gpu_v2/build/nodals_sst"; do
    if [[ -x "$f" ]]; then BINARY="$f"; break; fi
 done
fi
if [[ -n "$BINARY" && -x "$BINARY" ]]; then
 echo "BASELINE_BINARY=$BINARY"
 sha256sum "$BINARY"
else
 BINARY=""
 echo "BASELINE_BINARY=NOT_FOUND"
 echo "For an existing known-good compiled GPU SST solver, supply BINARY=/absolute/path/to/nodals_sst"
fi

echo "========== READY FOR CONTROL RUN =========="
echo "Editable CASE: $CASE_OUT"
echo "Log: $LOG"
echo "To open: xdg-open '$CASE_OUT'"
if [[ "${RUN_BASELINE:-0}" == "1" ]]; then
 if [[ -z "$BINARY" ]]; then echo "ERROR: RUN_BASELINE=1 but no known executable BINARY" >&2; exit 5; fi
 STEPS="${BASELINE_STEPS:-10}"
 echo "Executing $STEPS SIMPLE steps with original Robin wall method (NOT Option A)"
 echo "RUN: $OUT/baseline_robin"
 python3 -u "$ROOT/turbulence/sst_gpu_v2/nodals.py" run    --case "$CASE_OUT" --mesh "$MESH" --binary "$BINARY"    --steps "$STEPS" --output "$OUT/baseline_robin" 2>&1 | tee "$OUT/baseline_robin_console.log"
else
 echo "No solver run requested. To run a 10-iteration unchanged control: RUN_BASELINE=1 BINARY=/path/to/known_good/nodals_sst bash '$THIS_DIR/PREPARE_LOCAL.sh'"
fi
echo "========== PREFLIGHT DONE =========="
