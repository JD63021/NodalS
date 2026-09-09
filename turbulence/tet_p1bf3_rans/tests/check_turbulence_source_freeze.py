#!/usr/bin/env python3
from pathlib import Path
import hashlib, sys

root=Path(__file__).resolve().parents[1]
mods=[
 "src/00_core/preamble_mesh.inc",
 "src/10_problem/problem_config.inc",
 "src/20_discretization/discrete_boundary_fem.inc",
 "src/30_momentum/custom_momentum_core.inc",
 "src/40_pressure/effective_pressure.inc",
 "src/50_runtime/dynamic_runtime_plan.inc",
 "src/55_turbulence/mixing_length_spalding.inc",
 "src/60_momentum_dynamic/dynamic_operators.inc",
 "src/70_pressure_assembly/pressure_assembly.inc",
]
tail="src/80_app/main_solver.inc"
oracle=root/"reference/p1bf3_rans_effectiveB_oracle.cpp"
inc=root/"reference/original_amg_include.txt"

assembled="".join((root/m).read_text() for m in mods)+inc.read_text()+(root/tail).read_text()
ref=oracle.read_text()

sha=lambda s: hashlib.sha256(s.encode()).hexdigest()
print(f"TURB_SOURCE_FREEZE assembled_sha256={sha(assembled)} oracle_sha256={sha(ref)}")
if assembled != ref:
    print("TURB_SOURCE_FREEZE status=FAIL exact_text_match=0")
    sys.exit(1)
print("TURB_SOURCE_FREEZE status=PASS exact_text_match=1")
