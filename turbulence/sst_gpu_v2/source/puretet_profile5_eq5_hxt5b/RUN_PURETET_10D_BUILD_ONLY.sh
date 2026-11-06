#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$ROOT/../setup_gpu_deps.sh"
make -C "$ROOT" -j"${JOBS:-2}" ARCH="${ARCH:-sm_86}" nodals_hxt5b_fp64
