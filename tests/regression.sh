#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 tests/test_case_translation.py
python3 tests/test_cpu_custom_amg_annotated.py
python3 tests/test_gpu_case_translation.py
python3 tests/test_gpu_supg_case_translation.py
printf 'NODALS_REGRESSION static_status=PASS\n'
if [[ "${1:-}" == "--fast" ]]; then
    bash "$ROOT/build.sh"
fi
