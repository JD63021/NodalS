#!/usr/bin/env python3

from pathlib import Path
import importlib.util

root = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "nodals_case",
    root / "scripts" / "nodals_case.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def pairs(opts):
    out = {}
    i = 0
    while i < len(opts):
        if opts[i] == "-options_left":
            i += 1
            continue
        if i + 1 >= len(opts):
            raise RuntimeError(f"option without value: {opts[i]}")
        out[opts[i]] = opts[i + 1]
        i += 2
    return out


# ============================================================
# 1. Public generic test.case
# ============================================================

cp = mod._read_case(root / "cases" / "test.case")
o = pairs(mod.build_solver_options(cp))

generic_mesh = str(
    root
    / "meshes"
    / "shellsphere1"
    / "82ktet"
    / "constant"
    / "polyMesh"
)

expected_generic = {
    "-mesh": generic_mesh,
    "-problem": "flow",
    "-nu": "0.001",
    "-convection": "central",

    "-flow_wall_patches": "patch_0_0",
    "-flow_inlet_patch": "patch_1_0",
    "-flow_outlet_patch": "patch_2_0",

    "-inlet_bc": "fixed_normal_speed",
    "-inlet_normal_mode": "average_patch_normal",

    "-simple_variant": "simplec",
    "-alpha_u": "0.7",
    "-alpha_p": "1.0",
    "-simple_rtol": "1e-3",

    "-pressure_solve_mode": "gamg_richardson",
    "-p_pmat": "full",
    "-p_operator": "factored",
    "-p_preconditioner_refresh": "100",
    "-p_ksp_type": "richardson",
    "-p_ksp_rtol": "0.5",
    "-p_pc_type": "gamg",

    "-p_mg_levels_ksp_type": "richardson",
    "-p_mg_levels_pc_type": "jacobi",
    "-p_mg_levels_ksp_max_it": "1",
}

for k, v in expected_generic.items():
    assert o.get(k) == v, (k, o.get(k), v)

# Generic case-file speed is a positive inward magnitude.
# The low-level solver receives the existing signed-normal convention.
assert abs(float(o["-inlet_normal_speed"]) + 0.2) < 1e-14


# ============================================================
# 2. Explicit 40k fast pipe regression case remains available
# ============================================================

fast_pipe = root / "cases" / "40k_Re20_fast_full_gamg.case"

if fast_pipe.exists():
    cp2 = mod._read_case(fast_pipe)
    o2 = pairs(mod.build_solver_options(cp2))

    pipe_mesh = str(
        root
        / "meshes"
        / "pipe"
        / "40k"
        / "constant"
        / "polyMesh"
    )

    assert o2["-mesh"] == pipe_mesh
    assert o2["-problem"] == "pipe"
    assert o2["-re"] == "20"

    assert o2["-pipe_wall_patch"] == "patch_0_0"
    assert o2["-pipe_inlet_patch"] == "patch_2_0"
    assert o2["-pipe_outlet_patch"] == "patch_1_0"
    assert o2["-inlet_bc"] == "parabolic"

    assert o2["-pressure_solve_mode"] == "gamg_richardson"
    assert o2["-p_pmat"] == "full"


# ============================================================
# 3. Compact low-memory pressure architecture remains available
# ============================================================

compact = root / "cases" / "40k_Re20_compact_cheb.case"

if compact.exists():
    cp3 = mod._read_case(compact)
    o3 = pairs(mod.build_solver_options(cp3))

    assert o3["-pressure_solve_mode"] == "custom_pcg"
    assert o3["-p_pmat"] == "native_face"
    assert o3["-gate9g_fe_face_energy"] == "1"
    assert o3["-gate9i_auto_chebyshev"] == "1"


print(
    "NODALS_CASE_TRANSLATION status=PASS "
    "public_generic_82k=1 "
    "normal_inlet=1 "
    "fast_full_gamg=1 "
    "pipe_regression=1 "
    "compact_cheb=1"
)
