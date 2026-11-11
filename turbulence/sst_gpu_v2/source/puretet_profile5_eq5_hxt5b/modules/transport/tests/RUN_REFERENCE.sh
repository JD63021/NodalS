#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
g++ -std=c++17 -O2 -Wall -Wextra -pedantic test_dg_omega_p1_reference.cpp -o /tmp/nodals_dg_omega_p1_reference_test
/tmp/nodals_dg_omega_p1_reference_test
