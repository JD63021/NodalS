#!/usr/bin/env python3
from pathlib import Path
import hashlib
import sys

root = Path(__file__).resolve().parents[1]
reference = root / "reference" / "p1bf3_simple_foam_mpi.cpp"
EXPECTED = "0de1a33840d0f36fa84dd7a301bb2073c3dc344cb958defeb0bd7390334d4d01"

fragments = [
    root / "src" / "00_core" / "preamble.inc",
    root / "src" / "00_core" / "mesh_runtime.inc",
    root / "src" / "10_fem" / "basis_quadrature.inc",
    root / "src" / "20_boundary" / "problem_boundary.inc",
    root / "src" / "30_momentum" / "custom_momentum.inc",
    root / "src" / "35_stabilization" / "supg_dynamic_assembly.inc",
    root / "src" / "40_pressure" / "pressure_assembly.inc",
    root / "src" / "50_parallel" / "ownership_solve.inc",
    root / "src" / "60_output" / "diagnostics_vtu.inc",
    root / "src" / "70_simple" / "simple_support_experimental.inc",
    root / "src" / "80_app" / "main_solver.inc",
]

if not reference.exists():
    print(f"NODALS_REFERENCE_FREEZE status=FAIL missing={reference}")
    sys.exit(1)
for p in fragments:
    if not p.exists():
        print(f"NODALS_ACTIVE_ORACLE status=FAIL missing={p}")
        sys.exit(2)

ref = reference.read_bytes()
ref_sha = hashlib.sha256(ref).hexdigest()
active = b"".join(p.read_bytes() for p in fragments)
active_sha = hashlib.sha256(active).hexdigest()
exact = active == ref

print(f"NODALS_REFERENCE_FREEZE reference_sha256={ref_sha} expected_sha256={EXPECTED}")
print(f"NODALS_ACTIVE_ORACLE active_sha256={active_sha} expected_sha256={EXPECTED} exact_text_match={int(exact)}")

if ref_sha != EXPECTED:
    print("NODALS_REFERENCE_FREEZE status=FAIL reason=good_oracle_hash_mismatch")
    sys.exit(3)
if active_sha != EXPECTED or not exact:
    print("NODALS_ACTIVE_ORACLE status=FAIL reason=modular_fragments_do_not_reassemble_good_oracle")
    sys.exit(4)

text = active.decode("utf-8", errors="strict")
required = {
    "distributed_mesh": "P1BF3_DIST_ACTIVE",
    "root_only_mesh": "rootOnlyRead=1",
    "m1_colgid_release": "P1BF3_M1_RELEASE_COLGID",
    "m2_knu_eliminated": "A.kNu.clear(); A.kNu.shrink_to_fit()",
    "compact_dynamic_plan": "CustomDynamicRuntimeCellPlan",
    "compact_dynamic_marker": "removedFields=cell_gid_entity_gradLambda",
    "full_pressure_vertexcells_release": "vertexCells=RELEASED",
    "compact_fixed_values": "fixedDirValue",
    "factored_physical_schur": "exactOperator=factored_B_rAU_Bt",
    "richardson_gamg": "gamg_richardson",
    "local_sgs": "processorBlockJacobi+localSGS",
}
missing=[name for name,marker in required.items() if marker not in text]
if missing:
    print("NODALS_MEMORY_OPTIMIZATIONS status=FAIL missing=" + ",".join(missing))
    sys.exit(5)

print("NODALS_REFERENCE_FREEZE status=PASS oracle=FULLFAST_FP64_RICH_SCALE")
print("NODALS_ACTIVE_ORACLE status=PASS exact_text_match=1")
print("NODALS_MEMORY_OPTIMIZATIONS status=PASS " + " ".join(f"{k}=1" for k in required))
