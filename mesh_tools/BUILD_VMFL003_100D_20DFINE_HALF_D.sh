#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUTROOT="${OUTROOT:-$HOME/Desktop/meshes/nodals_vmfl003_100D_20Dfine_halfD_1329k}"
MSH_OUT="${MSH_OUT:-$OUTROOT/VMFL003_100D_20DFINE_HALFD.msh}"
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
  echo "ERROR: could not identify original 20D MSH2 source automatically." >&2
  exit 3
}

SRC="$(find_source_msh)"
[[ -f "$SRC" ]] || { echo "ERROR: source MSH not found: $SRC"; exit 4; }
command -v gmshToFoam >/dev/null || { echo "ERROR: gmshToFoam not found"; exit 5; }

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

echo "=== BUILD VMFL003 100D: ORIGINAL 20D + HALF-D DOWNSTREAM TETS ==="
echo "sourceMsh=$SRC"
echo "outputRoot=$OUTROOT"
echo "policy=0-20D:original_360_slabs 20-100D:160_slabs_each_0.5D"

python3 "$HERE/GENERATE_VMFL003_100D_20DFINE_HALF_D.py" "$SRC" "$MSH_OUT" \
  | tee "$OUTROOT/generate.log"

gmshToFoam -case "$FOAM_CASE" "$MSH_OUT" 2>&1 \
  | tee "$OUTROOT/gmshToFoam.log"

BOUNDARY="$FOAM_CASE/constant/polyMesh/boundary"
OWNER="$FOAM_CASE/constant/polyMesh/owner"
[[ -s "$BOUNDARY" && -s "$OWNER" ]] || { echo "ERROR: polyMesh conversion failed"; exit 6; }
for p in inlet wall outlet; do
  grep -q "^[[:space:]]*$p[[:space:]]*$" "$BOUNDARY" || {
    echo "ERROR: patch '$p' missing from $BOUNDARY"; exit 7;
  }
done

printf 'NODALS_VMFL003_100D_20DFINE_HALFD_BUILD mesh=%s source=%s source20DSlabs=360 downstreamSlabs=160 downstreamDzOverD=0.5 totalSlabs=520 expectedTets=1329120 status=PASS\n' \
  "$FOAM_CASE/constant/polyMesh" "$SRC"
