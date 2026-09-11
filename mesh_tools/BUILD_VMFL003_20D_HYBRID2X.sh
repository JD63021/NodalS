#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTROOT="${OUTROOT:-$HOME/Desktop/meshes/nodals_vmfl003_20D_hybrid2x_690k}"
MSH_OUT="${MSH_OUT:-$OUTROOT/VMFL003_20D_HYBRID2X.msh}"
FOAM_CASE="${FOAM_CASE:-$OUTROOT/foam_case}"

find_source_msh() {
  local base="$HOME/Desktop/meshes/nodals_vmfl003_20D_radial920k"
  if [[ -n "${SRC_MSH:-}" ]]; then
    printf '%s\n' "$SRC_MSH"
    return
  fi
  mapfile -t cands < <(find "$base" -maxdepth 4 -type f -iname '*.msh' 2>/dev/null | sort)
  if (( ${#cands[@]} == 1 )); then
    printf '%s\n' "${cands[0]}"
    return
  fi
  if (( ${#cands[@]} > 1 )); then
    echo "ERROR: multiple source .msh files found. Set SRC_MSH explicitly:" >&2
    printf '  %s\n' "${cands[@]}" >&2
    exit 2
  fi
  mapfile -t cands < <(find "$HOME/Desktop/meshes" -maxdepth 4 -type f -iname '*.msh' 2>/dev/null | grep -Ei 'vmfl003.*20D|20D.*vmfl003' | sort)
  if (( ${#cands[@]} == 1 )); then
    printf '%s\n' "${cands[0]}"
    return
  fi
  echo "ERROR: could not identify the original 20D MSH2 source automatically." >&2
  echo "Set SRC_MSH=/full/path/to/original20D.msh and rerun." >&2
  if (( ${#cands[@]} > 1 )); then
    echo "Candidates:" >&2
    printf '  %s\n' "${cands[@]}" >&2
  fi
  exit 3
}

SRC="$(find_source_msh)"
[[ -f "$SRC" ]] || { echo "ERROR: source MSH not found: $SRC"; exit 4; }
command -v gmshToFoam >/dev/null || { echo "ERROR: gmshToFoam not found in PATH. Source your OpenFOAM environment first."; exit 5; }

mkdir -p "$OUTROOT" "$FOAM_CASE/system" "$FOAM_CASE/constant"
cat > "$FOAM_CASE/system/controlDict" <<'EOF'
FoamFile
{
    format      ascii;
    class       dictionary;
    object      controlDict;
}
application     foamRun;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
EOF

echo "=== BUILD VMFL003 20D HYBRID2X ==="
echo "sourceMsh=$SRC"
echo "outputRoot=$OUTROOT"
echo "policy=0-10D:D/18_identical 10-20D:D/9_2x_axial"

python3 "$HERE/GENERATE_VMFL003_20D_HYBRID2X.py" "$SRC" "$MSH_OUT" | tee "$OUTROOT/generate.log"

gmshToFoam -case "$FOAM_CASE" "$MSH_OUT" 2>&1 | tee "$OUTROOT/gmshToFoam.log"

BOUNDARY="$FOAM_CASE/constant/polyMesh/boundary"
OWNER="$FOAM_CASE/constant/polyMesh/owner"
[[ -s "$BOUNDARY" && -s "$OWNER" ]] || { echo "ERROR: polyMesh conversion failed"; exit 6; }
for p in inlet wall outlet; do
  grep -q "^[[:space:]]*$p[[:space:]]*$" "$BOUNDARY" || { echo "ERROR: patch '$p' missing from $BOUNDARY"; exit 7; }
done

printf 'NODALS_VMFL003_20D_HYBRID2X_BUILD mesh=%s source=%s expectedSlabs=270 expectedTets=690120 first10D_dzOverD=0.055555555556 second10D_dzOverD=0.111111111111 status=PASS\n' \
  "$FOAM_CASE/constant/polyMesh" "$SRC"
