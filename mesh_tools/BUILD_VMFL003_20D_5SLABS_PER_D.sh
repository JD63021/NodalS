#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUTROOT="${OUTROOT:-$HOME/Desktop/meshes/nodals_vmfl003_20D_5slabD_256k}"
MSH_OUT="${MSH_OUT:-$OUTROOT/VMFL003_20D_5SLAB_PER_D.msh}"
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
  echo "ERROR: could not identify unique original 20D MSH source. Set SRC_MSH explicitly." >&2
  exit 2
}

SRC="$(find_source_msh)"
[[ -f "$SRC" ]] || { echo "ERROR: source MSH not found: $SRC"; exit 3; }
command -v gmshToFoam >/dev/null || { echo "ERROR: gmshToFoam not found"; exit 4; }

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

echo "=== BUILD VMFL003 20D: 5 AXIAL SLABS PER DIAMETER ==="
echo "sourceMsh=$SRC"
echo "outputRoot=$OUTROOT"
echo "policy=uniform_20D__5_slabs_per_D__dz=0.2D"

python3 "$HERE/GENERATE_VMFL003_20D_5SLABS_PER_D.py" "$SRC" "$MSH_OUT" \
  | tee "$OUTROOT/generate.log"

gmshToFoam -case "$FOAM_CASE" "$MSH_OUT" 2>&1 \
  | tee "$OUTROOT/gmshToFoam.log"

BOUNDARY="$FOAM_CASE/constant/polyMesh/boundary"
OWNER="$FOAM_CASE/constant/polyMesh/owner"
[[ -s "$BOUNDARY" && -s "$OWNER" ]] || { echo "ERROR: polyMesh conversion failed"; exit 5; }
for p in inlet wall outlet; do
  grep -q "^[[:space:]]*$p[[:space:]]*$" "$BOUNDARY" || {
    echo "ERROR: patch '$p' missing from $BOUNDARY"; exit 6;
  }
done

printf 'NODALS_VMFL003_20D_5SLABD_BUILD mesh=%s source=%s slabsPerD=5 totalSlabs=100 dzOverD=0.2 expectedTets=255600 status=PASS\n' \
  "$FOAM_CASE/constant/polyMesh" "$SRC"
