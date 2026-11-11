#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
BIN="${TMPDIR:-/tmp}/nodals_dg_omega_field_access"
g++ -std=c++17 -O2 -Wall -Wextra -pedantic dg_omega_field_access_test.cpp -o "$BIN"
"$BIN"
