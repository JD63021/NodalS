#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
g++ -O2 -std=c++17 -Wall -Wextra \
 "$ROOT/experiments/cpu_sst_spalding_m4/tests/test_omega_targets.cpp" \
 -o "$TMP/test_omega_targets"
"$TMP/test_omega_targets"
