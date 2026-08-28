#!/usr/bin/env bash
# Resolve the PETSc installation used by the frozen Gate9N lineage.
# May be sourced by build/regression scripts.

export PETSC_DIR="${PETSC_DIR:-$HOME/petsc-main}"

nodals_petsc_arch_valid() {
  local arch="$1"
  [[ -n "$arch" ]] && \
  [[ -d "$PETSC_DIR/$arch" ]] && \
  [[ -f "$PETSC_DIR/$arch/lib/petsc/conf/petscvariables" ]]
}

# The uploaded/frozen Gate9N Makefile is authoritative for v1.00.
NODALS_FROZEN_DEFAULT_PETSC_ARCH="arch-cuda-debug"

if [[ -n "${PETSC_ARCH:-}" ]] && nodals_petsc_arch_valid "$PETSC_ARCH"; then
  : # explicit valid user choice wins
elif nodals_petsc_arch_valid "$NODALS_FROZEN_DEFAULT_PETSC_ARCH"; then
  export PETSC_ARCH="$NODALS_FROZEN_DEFAULT_PETSC_ARCH"
else
  # Fall back to any configured arch below PETSC_DIR. Prefer optimized-looking
  # names if several exist, but never select a directory without petscvariables.
  mapfile -t _nodals_arches < <(
    find "$PETSC_DIR" -mindepth 4 -maxdepth 4 -type f \
      -path '*/lib/petsc/conf/petscvariables' -printf '%h\n' 2>/dev/null \
    | sed -E 's#/lib/petsc/conf$##; s#^.*/##' \
    | sort -u
  )

  if (( ${#_nodals_arches[@]} == 1 )); then
    export PETSC_ARCH="${_nodals_arches[0]}"
  elif (( ${#_nodals_arches[@]} > 1 )); then
    _chosen=""
    for _a in "${_nodals_arches[@]}"; do
      if [[ "$_a" == *opt* || "$_a" == *release* ]]; then _chosen="$_a"; break; fi
    done
    export PETSC_ARCH="${_chosen:-${_nodals_arches[0]}}"
    echo "NODALS_PETSC_DISCOVERY warning=multiple_arches selected=$PETSC_ARCH available=${_nodals_arches[*]}" >&2
  else
    echo "NODALS_PETSC_DISCOVERY status=FAIL reason=no_configured_arch PETSC_DIR=$PETSC_DIR" >&2
    return 2 2>/dev/null || exit 2
  fi
fi

if ! nodals_petsc_arch_valid "$PETSC_ARCH"; then
  echo "NODALS_PETSC_DISCOVERY status=FAIL reason=invalid_arch PETSC_DIR=$PETSC_DIR PETSC_ARCH=${PETSC_ARCH:-}" >&2
  return 2 2>/dev/null || exit 2
fi

echo "NODALS_PETSC_DISCOVERY status=PASS PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH" >&2
