#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
BIN="${TMPDIR:-/tmp}/nodals_dg_omega_gpu_parity"
NVCC="${NVCC:-/usr/local/cuda-12.2/bin/nvcc}"
if [[ ! -x "$NVCC" ]]; then NVCC="$(command -v nvcc || true)"; fi
if [[ -z "$NVCC" ]]; then echo "CUDA nvcc unavailable"; exit 2; fi
"$NVCC" -std=c++17 -O2 -arch=sm_86 -lineinfo \
  dg_omega_gpu_parity.cu -o "$BIN"
"$BIN"
