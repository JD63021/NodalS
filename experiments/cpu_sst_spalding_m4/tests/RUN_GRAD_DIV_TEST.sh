#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
g++ -std=c++17 -O2 -Wall -Wextra -Werror \
 "$ROOT/experiments/cpu_sst_spalding_m4/tests/test_grad_div_rank_one.cpp" \
 -o "$TMP/test_grad_div_rank_one"
"$TMP/test_grad_div_rank_one"
g++ -std=c++17 -O2 -Wall -Wextra -Werror \
 "$ROOT/experiments/cpu_sst_spalding_m4/tests/test_constraint_augmentation.cpp" \
 -o "$TMP/test_constraint_augmentation"
"$TMP/test_constraint_augmentation"
