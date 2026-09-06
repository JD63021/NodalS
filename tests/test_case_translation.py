#!/usr/bin/env python3

from pathlib import Path
import importlib.util

root = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location("nodals_case", root / "scripts" / "nodals_case.py")
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


def mesh_suffix_ok(value: str, suffix: str) -> bool:
    return Path(value).as_posix().endswith(suffix)


# 1. Explicit public generic case keeps the accepted PETSc full-GAMG translation.
cp = mod._read_case(root / "cases" / "test.case")
o = pairs(mod.build_solver_options(cp))
assert mesh_suffix_ok(o["-mesh"], "meshes/shellsphere1/82ktet/constant/polyMesh"), o["-mesh"]
expected_generic = {
    "-problem": "flow", "-nu": "0.001", "-convection": "central",
    "-flow_wall_patches": "patch_0_0", "-flow_inlet_patch": "patch_1_0", "-flow_outlet_patch": "patch_2_0",
    "-inlet_bc": "fixed_normal_speed", "-inlet_normal_mode": "average_patch_normal",
    "-simple_variant": "simplec", "-alpha_u": "0.7", "-alpha_p": "1.0", "-simple_rtol": "1e-3",
    "-pressure_solve_mode": "gamg_richardson", "-p_pmat": "full", "-p_operator": "factored",
    "-p_preconditioner_refresh": "100", "-p_ksp_type": "richardson", "-p_ksp_rtol": "0.5", "-p_pc_type": "gamg",
    "-p_mg_levels_ksp_type": "richardson", "-p_mg_levels_pc_type": "jacobi", "-p_mg_levels_ksp_max_it": "1",
}
for k, v in expected_generic.items():
    assert o.get(k) == v, (k, o.get(k), v)
assert abs(float(o["-inlet_normal_speed"]) + 0.2) < 1e-14

# 2. Existing explicit PETSc fallback remains available.
fast_pipe = root / "cases" / "40k_Re20_fast_full_gamg.case"
o2 = pairs(mod.build_solver_options(mod._read_case(fast_pipe)))
assert mesh_suffix_ok(o2["-mesh"], "meshes/pipe/40k/constant/polyMesh"), o2["-mesh"]
assert o2["-problem"] == "pipe" and o2["-re"] == "20"
assert o2["-pressure_solve_mode"] == "gamg_richardson" and o2["-p_pmat"] == "full"

# 3. Existing compact low-memory architecture remains available.
compact = root / "cases" / "40k_Re20_compact_cheb.case"
o3 = pairs(mod.build_solver_options(mod._read_case(compact)))
assert o3["-pressure_solve_mode"] == "custom_pcg"
assert o3["-p_pmat"] == "native_face"
assert o3["-gate9g_fe_face_energy"] == "1"
assert o3["-gate9i_auto_chebyshev"] == "1"

# 4. Native custom-AMG public case modes retain their outer/transfer pairing.
expected_amg = {
    "pcg_unsmoothed": ("custom_pcg", "custom_agg_unsmoothed", "sgs"),
    "pcg_smoothed": ("custom_pcg", "custom_agg_smoothed", "chebyshev"),
    "richardson_smoothed": ("custom_richardson", "custom_agg_smoothed", "chebyshev"),
}
for mode, (outer, backend, smoother) in expected_amg.items():
    case = root / "cases" / f"40k_Re20_{mode}_amg.case"
    ox = pairs(mod.build_solver_options(mod._read_case(case)))
    assert ox["-pressure_solve_mode"] == outer, (mode, ox.get("-pressure_solve_mode"), outer)
    assert ox["-pressure_pc_backend"] == backend, (mode, ox.get("-pressure_pc_backend"), backend)
    assert ox["-p_operator"] == "factored"
    assert ox["-p_preconditioner_refresh"] == "100"
    assert ox["-custom_amg_target_aggregate"] == "16"
    assert ox["-custom_amg_min_aggregate"] == "6"
    assert ox["-custom_amg_soft_max_aggregate"] == "18"
    assert ox["-custom_amg_smoother"] == smoother
    assert ox["-custom_amg_coarse_target"] == "1000"
    assert ox["-custom_amg_interp_max_nnz"] == "8"
    if smoother == "chebyshev":
        assert ox["-custom_amg_cheb_degree"] == "2"
        assert ox["-custom_amg_power_its"] == "16"
    else:
        assert "-custom_amg_cheb_degree" not in ox
        assert "-custom_amg_power_its" not in ox

# 5. Omitting pressure mode now chooses the low-memory native PCG/AMG path.
cp5 = mod._read_case(fast_pipe)
cp5.remove_option("pressure", "mode")
cp5.remove_option("pressure", "solver")
o5 = pairs(mod.build_solver_options(cp5))
assert o5["-pressure_solve_mode"] == "custom_pcg"
assert o5["-pressure_pc_backend"] == "custom_agg_unsmoothed"
assert o5["-custom_amg_target_aggregate"] == "16"
assert o5["-custom_amg_min_aggregate"] == "6"
assert o5["-custom_amg_soft_max_aggregate"] == "18"
assert o5["-custom_amg_smoother"] == "sgs"
assert "-custom_amg_power_its" not in o5
assert "-p_pc_type" not in o5

# 6. Coarsening remains mesh-dependent and overrideable from [pressure_amg].
cp6 = mod._read_case(fast_pipe)
cp6.set("pressure", "mode", "pcg_unsmoothed")
if not cp6.has_section("pressure_amg"):
    cp6.add_section("pressure_amg")
cp6.set("pressure_amg", "target_aggregate", "12")
cp6.set("pressure_amg", "min_aggregate", "6")
cp6.set("pressure_amg", "soft_max_aggregate", "14")
cp6.set("pressure_amg", "smoother", "jacobi")
o6 = pairs(mod.build_solver_options(cp6))
assert o6["-custom_amg_target_aggregate"] == "12"
assert o6["-custom_amg_min_aggregate"] == "6"
assert o6["-custom_amg_soft_max_aggregate"] == "14"
assert o6["-custom_amg_smoother"] == "jacobi"
assert "-custom_amg_power_its" not in o6

# 7. Chebyshev remains an explicit optional smoother on unsmoothed AMG.
cp7 = mod._read_case(fast_pipe)
cp7.set("pressure", "mode", "pcg_unsmoothed_chebyshev")
o7 = pairs(mod.build_solver_options(cp7))
assert o7["-custom_amg_smoother"] == "chebyshev"
assert o7["-custom_amg_power_its"] == "16"

# 8. Smoothed transfer rejects a no-spectrum smoother.
cp8 = mod._read_case(fast_pipe)
cp8.set("pressure", "mode", "pcg_smoothed")
if not cp8.has_section("pressure_amg"):
    cp8.add_section("pressure_amg")
cp8.set("pressure_amg", "smoother", "sgs")
try:
    mod.build_solver_options(cp8)
except ValueError as e:
    assert "requires [pressure_amg] smoother=chebyshev" in str(e)
else:
    raise AssertionError("smoothed AMG incorrectly accepted SGS smoother")

print("NODALS_CASE_TRANSLATION status=PASS explicit_petsc=1 compact=1 custom_amg_modes=3 default_low_memory=1 coarsening_knob=1 smoother_knobs=1")
