#!/usr/bin/env python3
from pathlib import Path
import hashlib
import sys

root = Path(__file__).resolve().parents[1]

reference = root / "reference" / "p1bf3_simple_foam_mpi.cpp"

if not reference.exists():
    print(f"NODALS_REFERENCE_FREEZE status=FAIL missing={reference}")
    sys.exit(1)

data = reference.read_bytes()
sha = hashlib.sha256(data).hexdigest()

EXPECTED = "d43884099ceb03b4871bacd50e961eb695d4518825e3910724b186786b67d39c"

print(
    f"NODALS_REFERENCE_FREEZE "
    f"reference_sha256={sha} "
    f"expected_sha256={EXPECTED}"
)

if sha != EXPECTED:
    print(
        "NODALS_REFERENCE_FREEZE status=FAIL "
        "reason=frozen_reference_monolith_changed"
    )
    sys.exit(2)

active_files = [
    root / "src" / "30_momentum" / "custom_momentum.inc",
    root / "src" / "80_app" / "main_solver.inc",
]

for p in active_files:
    if not p.exists():
        print(f"NODALS_ACTIVE_EXTENSION status=FAIL missing={p}")
        sys.exit(3)

active = "\n".join(
    p.read_text(errors="replace") for p in active_files
)

required = [
    "gamg_richardson",
    "NODALS_FAST_GAMG_RICHARDSON",
    "NODALS_FULL_PMAT_RAU_HALO",
]

missing = [x for x in required if x not in active]

if missing:
    print(
        "NODALS_ACTIVE_EXTENSION status=FAIL missing_markers="
        + ",".join(missing)
    )
    sys.exit(4)

print(
    "NODALS_REFERENCE_FREEZE status=PASS "
    "frozen_gate9n_monolith_preserved=1"
)

print(
    "NODALS_ACTIVE_EXTENSION status=PASS "
    "fast_full_gamg_backend=1 "
    "full_pmat_rau_halo_fix=1"
)
