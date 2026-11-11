#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ $# -lt 1 ]]; then echo "usage: $0 /absolute/path/to/HXT1_mesh.bin [--rtol ...]";exit 2;fi
NVCC="${NVCC:-/usr/local/cuda-12.2/bin/nvcc}"
if [[ ! -x "$NVCC" ]]; then NVCC="$(command -v nvcc || true)"; fi
if [[ -z "$NVCC" ]]; then echo "DG_OMEGA_FULLMESH_FAIL nvcc not found";exit 2;fi
BIN="${TMPDIR:-/tmp}/nodals_dg_omega_fullmesh_smoke"
"$NVCC" -std=c++17 -O2 -arch=sm_86 -lineinfo \
   dg_omega_fullmesh_smoke.cu -lcublas -o "$BIN"
"$BIN" "$@"
