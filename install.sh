#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${PREFIX:-$HOME/.local}"

[[ -x "$ROOT/nodals_solver" ]] || {
  echo "NODALS_INSTALL status=FAIL reason=solver_not_built hint='./build.sh first'" >&2
  exit 2
}

mkdir -p "$PREFIX/bin" "$PREFIX/libexec/nodals" "$PREFIX/share/nodals/cases" "$PREFIX/share/nodals/docs"
install -m 0755 "$ROOT/nodals_solver" "$PREFIX/libexec/nodals/nodals_solver"
install -m 0755 "$ROOT/scripts/nodals_case.py" "$PREFIX/bin/nodals"
for f in "$ROOT"/cases/*.case; do install -m 0644 "$f" "$PREFIX/share/nodals/cases/$(basename "$f")"; done
install -m 0644 "$ROOT/docs/CASE_FORMAT.md" "$PREFIX/share/nodals/docs/CASE_FORMAT.md"
install -m 0644 "$ROOT/VERSION" "$PREFIX/share/nodals/VERSION"

echo "NODALS_INSTALL status=PASS prefix=$PREFIX"
echo "NODALS_INSTALL command=$PREFIX/bin/nodals"
echo "NODALS_INSTALL example=$PREFIX/share/nodals/cases/test.case"
case ":$PATH:" in
  *":$PREFIX/bin:"*) ;;
  *) echo "NODALS_INSTALL note='add $PREFIX/bin to PATH to invoke nodals directly'" ;;
esac
