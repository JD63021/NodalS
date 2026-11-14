#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
g++ -std=c++17 -O2 -Wall -Wextra -Werror \
 "$ROOT/experiments/cpu_sst_spalding_m4/tests/test_spalding_k_equilibrium.cpp" \
 -o "$TMP/test_spalding_k_equilibrium"
"$TMP/test_spalding_k_equilibrium"
