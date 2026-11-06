#!/usr/bin/env bash
# The handoff TAR does not include the original NodalS G8 AMG headers.
# Link the unchanged original CUDA support headers into this experiment only.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SERIAL_CUDA_ROOT:-$HOME/NodalS_SST_DEVIATORIC_Y84/gpu/serial_cuda}"
if [[ -f "$HERE/g8_cf_hierarchy_host.hpp" ]]; then
    echo "PURETET_DEPENDENCY_G8 found=$HERE/g8_cf_hierarchy_host.hpp"
    exit 0
fi
if [[ ! -f "$SRC/g8_cf_hierarchy_host.hpp" ]]; then
    echo "ERROR: required g8_cf_hierarchy_host.hpp not in this TAR or $SRC" >&2
    echo "Set SERIAL_CUDA_ROOT to the parent gpu/serial_cuda directory from the working NodalS checkout." >&2
    exit 2
fi
# Create only missing *symlinks*; don't copy or change working G8 source.
shopt -s nullglob
for file in "$SRC"/*.hpp "$SRC"/*.cuh "$SRC"/*.inc; do
    target="$HERE/$(basename "$file")"
    [[ -e "$target" || -L "$target" ]] || ln -s "$file" "$target"
done
[[ -e "$HERE/g8_cf_hierarchy_host.hpp" ]] || { echo "ERROR: symlink to G8 header missing" >&2; exit 3; }
echo "PURETET_DEPENDENCY_G8 source=$SRC status=PASS"
