#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
MAIN="$ROOT/h8_main.cu"

echo "=== H8 FP64 SOURCE AUDIT ==="

grep -q -- '-DNODALS_GPU_PRECISION_MODE=0' "$ROOT/Makefile" || {
  echo "FAIL: Makefile lacks full-FP64 target"; exit 51;
}
grep -q 'using StateReal = double;' "$ROOT/precision.hpp" || {
  echo "FAIL: precision.hpp lacks FP64 state mode"; exit 52;
}
grep -q 'using OperatorReal = double;' "$ROOT/precision.hpp" || {
  echo "FAIL: precision.hpp lacks FP64 operator mode"; exit 53;
}
grep -q 'using AMGReal = double;' "$ROOT/precision.hpp" || {
  echo "FAIL: precision.hpp lacks FP64 AMG mode"; exit 54;
}

if grep -n 'requires full FP32\|requires FP32 operator\|requires FP32 AMG' \
    "$ROOT/g5e_fp32_kernels.cuh"; then
  echo "FAIL: stale FP32-only kernel guard remains"; exit 55
fi

if grep -nE 'precision=fp32|deviceNumericStorage=FP32|state=FP32|operator=FP32|amg=FP32|explicit_FP32_CSR|24FP32' \
    "$MAIN"; then
  echo "FAIL: stale hard-coded FP32 runtime label remains"; exit 56
fi

if grep -n 'H8 is fixed10-only' "$MAIN"; then
  echo "FAIL: stale fixed10-only guard remains"; exit 57
fi

grep -q 'runMode=="fixed10"||runMode=="converge"' "$MAIN" || {
  echo "FAIL: convergence run mode not enabled"; exit 58;
}

if grep -R -nE 'fabsf\(av\[|sqrtf\(d\[|const float q=.*i\+1' \
    "$ROOT/g5e_fp32_kernels.cuh" \
    "$ROOT/h7_momentum_assembly.cuh" \
    "$ROOT/h8_precomputed_b.cuh"; then
  echo "FAIL: stale FP32-only math remains"; exit 59
fi

echo "H8_FP64_SOURCE_AUDIT status=PASS fp32=SUPPORTED fp64=SUPPORTED labels=DYNAMIC converge=ENABLED"
