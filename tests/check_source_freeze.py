#!/usr/bin/env python3
"""Freeze checks for the NodalS tree after custom-pressure-AMG integration.

The historical FULLFAST source remains the immutable numerical oracle.  The
custom AMG integration intentionally extends the final application driver and
adds src/45_pressure_amg, so the *entire* active translation unit can no longer
be byte-for-byte equal to that historical source.

This check therefore enforces three separate contracts:
  1. the historical reference file itself is unchanged;
  2. all numerical fragments preceding the application driver are still the
     exact prefix of that reference source;
  3. the reviewed AMG module and integrated application driver are exactly the
     versions validated by the large-mesh AMG campaign / local Git handoff.

Changing either integrated file is allowed in future development, but must be a
conscious change accompanied by updating the expected hash here after review.
"""
from pathlib import Path
import hashlib
import sys

root = Path(__file__).resolve().parents[1]
reference = root / "reference" / "p1bf3_simple_foam_mpi.cpp"
EXPECTED_REFERENCE = "0de1a33840d0f36fa84dd7a301bb2073c3dc344cb958defeb0bd7390334d4d01"
EXPECTED_FROZEN_PREFIX = "07dd1e4302f0dca6cf330a5a2cfd2e8380b6461984ea37b229f89f979b91fbb9"
EXPECTED_MAIN_SOLVER = "15715dcd06bfaa6ef40b56dd0eff4af2674bb3ccd3cdef2a07c7236d8f907ffb"
EXPECTED_CUSTOM_AMG = "1de234cf4e7edfac36874769bcc12f282615c621d1e3f826774af176a122385d"

frozen_prefix_fragments = [
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
]
main_solver = root / "src" / "80_app" / "main_solver.inc"
amg_module = root / "src" / "45_pressure_amg" / "custom_pressure_amg.inc"
app = root / "app" / "nodals_main.cpp"

for p in [reference, *frozen_prefix_fragments, main_solver, amg_module, app]:
    if not p.exists():
        print(f"NODALS_SOURCE_FREEZE status=FAIL missing={p}")
        sys.exit(1)

sha = lambda b: hashlib.sha256(b).hexdigest()
ref = reference.read_bytes()
ref_sha = sha(ref)
prefix = b"".join(p.read_bytes() for p in frozen_prefix_fragments)
prefix_sha = sha(prefix)
main_bytes = main_solver.read_bytes()
amg_bytes = amg_module.read_bytes()
main_sha = sha(main_bytes)
amg_sha = sha(amg_bytes)

print(f"NODALS_REFERENCE_FREEZE reference_sha256={ref_sha} expected_sha256={EXPECTED_REFERENCE}")
print(f"NODALS_FROZEN_PREFIX prefix_sha256={prefix_sha} expected_sha256={EXPECTED_FROZEN_PREFIX} exact_reference_prefix={int(ref.startswith(prefix))}")
print(f"NODALS_ACTIVE_AMG_INTEGRATION mainSolverSha256={main_sha} expectedMainSolverSha256={EXPECTED_MAIN_SOLVER} customAmgSha256={amg_sha} expectedCustomAmgSha256={EXPECTED_CUSTOM_AMG}")

if ref_sha != EXPECTED_REFERENCE:
    print("NODALS_REFERENCE_FREEZE status=FAIL reason=good_oracle_hash_mismatch")
    sys.exit(2)
if prefix_sha != EXPECTED_FROZEN_PREFIX or not ref.startswith(prefix):
    print("NODALS_FROZEN_PREFIX status=FAIL reason=protected_fragments_changed")
    sys.exit(3)
if main_sha != EXPECTED_MAIN_SOLVER:
    print("NODALS_ACTIVE_AMG_INTEGRATION status=FAIL reason=integrated_main_solver_changed")
    sys.exit(4)
if amg_sha != EXPECTED_CUSTOM_AMG:
    print("NODALS_ACTIVE_AMG_INTEGRATION status=FAIL reason=custom_pressure_amg_changed")
    sys.exit(5)

app_text = app.read_text(encoding="utf-8")
ordered_includes = [
    '#include "../src/40_pressure/pressure_assembly.inc"',
    '#include "../src/45_pressure_amg/custom_pressure_amg.inc"',
    '#include "../src/50_parallel/ownership_solve.inc"',
]
pos = [app_text.find(s) for s in ordered_includes]
if any(p < 0 for p in pos) or pos != sorted(pos):
    print("NODALS_ACTIVE_AMG_INTEGRATION status=FAIL reason=amg_include_order")
    sys.exit(6)

# Historical architecture markers must remain present in the integrated driver.
main_text = main_bytes.decode("utf-8", errors="strict")
legacy_required = {
    "distributed_mesh": "P1BF3_DIST_ACTIVE",
    "root_only_mesh": "rootOnlyRead=1",
    "m1_colgid_release": "P1BF3_M1_RELEASE_COLGID",
    "m2_knu_eliminated": "A.kNu.clear(); A.kNu.shrink_to_fit()",
    "compact_dynamic_plan": "CustomDynamicRuntimeCellPlan",
    "compact_dynamic_marker": "removedFields=cell_gid_entity_gradLambda",
    "full_pressure_vertexcells_release": "vertexCells=RELEASED",
    "compact_fixed_values": "fixedDirValue",
    "factored_physical_schur": "exactOperator=factored_B_rAU_Bt",
    "richardson_gamg_fallback": 'pressureSolveMode=="gamg_richardson"',
    "local_sgs": "processorBlockJacobi+localSGS",
}
missing_legacy = [k for k, marker in legacy_required.items() if marker not in main_text and marker not in prefix.decode("utf-8", errors="strict")]
if missing_legacy:
    print("NODALS_MEMORY_OPTIMIZATIONS status=FAIL missing=" + ",".join(missing_legacy))
    sys.exit(7)

amg_text = amg_bytes.decode("utf-8", errors="strict")
amg_required = {
    "unsmoothed_backend": "custom_agg_unsmoothed",
    "smoothed_backend": "custom_agg_smoothed",
    "matrix_free_fine": "matrix_free_B_rAU_Bt",
    "hierarchy_build": "customPressureAMGBuildHierarchy",
    "native_pcg": "customPressureAMGLivePCGSolve",
    "native_richardson": "customPressureAMGLiveRichardsonSolve",
    "no_explicit_fine_schur": "fineSchurCSR=0",
}
missing_amg = [k for k, marker in amg_required.items() if marker not in amg_text]
if missing_amg:
    print("NODALS_CUSTOM_AMG_CONTRACT status=FAIL missing=" + ",".join(missing_amg))
    sys.exit(8)

print("NODALS_REFERENCE_FREEZE status=PASS oracle=FULLFAST_FP64_RICH_SCALE")
print("NODALS_FROZEN_PREFIX status=PASS protectedFragments=10")
print("NODALS_ACTIVE_AMG_INTEGRATION status=PASS reviewedMainSolver=1 reviewedCustomAMG=1 includeOrder=PASS")
print("NODALS_MEMORY_OPTIMIZATIONS status=PASS " + " ".join(f"{k}=1" for k in legacy_required))
print("NODALS_CUSTOM_AMG_CONTRACT status=PASS " + " ".join(f"{k}=1" for k in amg_required))
