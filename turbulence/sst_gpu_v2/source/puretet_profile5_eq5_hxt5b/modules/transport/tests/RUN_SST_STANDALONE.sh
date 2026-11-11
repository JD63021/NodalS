#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ $# -lt 1 ]]; then echo "usage: $0 HXT1_mesh.bin [--sweeps 2] [--alpha 0.7]";exit 2;fi
NVCC="${NVCC:-/usr/local/cuda-12.2/bin/nvcc}"
if [[ ! -x "$NVCC" ]]; then NVCC="$(command -v nvcc || true)"; fi
if [[ -z "$NVCC" ]]; then echo "CUDA nvcc unavailable";exit 2;fi
BIN="${TMPDIR:-/tmp}/nodals_dg_omega_sst_standalone"
"$NVCC" -std=c++17 -O2 -arch=sm_86 -lineinfo dg_omega_sst_smoke.cu -lcublas -o "$BIN"
"$BIN" "$@"
