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
install -m 0755 "$ROOT/scripts/nodals_gpu_case.py" "$PREFIX/bin/nodals-gpu"
for f in "$ROOT"/cases/*.case; do install -m 0644 "$f" "$PREFIX/share/nodals/cases/$(basename "$f")"; done
install -m 0644 "$ROOT/docs/CASE_FORMAT.md" "$PREFIX/share/nodals/docs/CASE_FORMAT.md"
install -m 0644 "$ROOT/docs/CPU_CUSTOM_AMG_CASE.md" "$PREFIX/share/nodals/docs/CPU_CUSTOM_AMG_CASE.md"
install -m 0644 "$ROOT/docs/GPU_CASE_FORMAT.md" "$PREFIX/share/nodals/docs/GPU_CASE_FORMAT.md"
install -m 0644 "$ROOT/VERSION" "$PREFIX/share/nodals/VERSION"

GPU_INSTALLED=0
for precision in fp32 fp64; do
  src="$ROOT/gpu/serial_cuda/nodals_gpu_h8_$precision"
  if [[ -x "$src" ]]; then
    install -m 0755 "$src" "$PREFIX/libexec/nodals/nodals_gpu_h8_$precision"
    GPU_INSTALLED=1
  fi
done

echo "NODALS_INSTALL status=PASS prefix=$PREFIX"
echo "NODALS_INSTALL command=$PREFIX/bin/nodals"
echo "NODALS_INSTALL gpu_command=$PREFIX/bin/nodals-gpu"
echo "NODALS_INSTALL example=$PREFIX/share/nodals/cases/test.case"
echo "NODALS_INSTALL gpu_example=$PREFIX/share/nodals/cases/gpu-h8-annotated.case"
if [[ "$GPU_INSTALLED" -eq 0 ]]; then
  echo "NODALS_INSTALL note='GPU runner installed; build gpu/serial_cuda fp32 or fp64 before GPU execution'"
fi
case ":$PATH:" in
  *":$PREFIX/bin:"*) ;;
  *) echo "NODALS_INSTALL note='add $PREFIX/bin to PATH to invoke nodals/nodals-gpu directly'" ;;
esac
