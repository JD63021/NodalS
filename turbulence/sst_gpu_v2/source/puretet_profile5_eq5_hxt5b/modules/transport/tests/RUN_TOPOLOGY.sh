#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
NVCC="${NVCC:-/usr/local/cuda-12.2/bin/nvcc}"
if [[ ! -x "$NVCC" ]]; then NVCC="$(command -v nvcc || true)"; fi
if [[ -z "$NVCC" ]]; then echo "CUDA compiler missing";exit 2;fi
BIN="${TMPDIR:-/tmp}/nodals_dg_omega_topology"
"$NVCC" -std=c++17 -O2 -arch=sm_86 dg_omega_hxt1_topology_test.cu -o "$BIN"
"$BIN" "$@"
