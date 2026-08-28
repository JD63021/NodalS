#!/usr/bin/env python3
"""NodalS v1.00 case runner.

The numerical executable remains the frozen Gate9N solver.  This program only
translates a human-readable .case file into the already validated PETSc command
line, launches MPI, captures a log, and leaves explicit trailing PETSc options
as the highest-priority overrides.
"""
from __future__ import annotations
import argparse
import configparser
import datetime as _dt
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Dict, Iterable, List, Tuple

TRUE = {"1", "true", "yes", "on"}
FALSE = {"0", "false", "no", "off"}


def _bool(v: str, where: str) -> bool:
    s = str(v).strip().lower()
    if s in TRUE:
        return True
    if s in FALSE:
        return False
    raise ValueError(f"{where}: expected boolean, got {v!r}")


def _expand(v: str) -> str:
    return os.path.expandvars(os.path.expanduser(v.strip()))


def _read_case(path: Path) -> configparser.ConfigParser:
    cp = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#", ";"))
    cp.optionxform = str.lower
    with path.open("r", encoding="utf-8") as f:
        cp.read_file(f)
    return cp


def _get(cp, section, key, default=None):
    if cp.has_option(section, key):
        return cp.get(section, key).strip()
    return default


def _require(cp, section, key):
    v = _get(cp, section, key)
    if v is None or v == "":
        raise ValueError(f"missing required [{section}] {key}")
    return v


def _opt(args: List[str], key: str, value) -> None:
    args.extend([f"-{key}", str(value)])


def build_solver_options(cp: configparser.ConfigParser) -> List[str]:
    """Translate the .case into the legacy solver's exact PETSc options."""
    a: List[str] = []

    problem = _get(cp, "solver", "problem", "pipe").lower()
    if problem not in {"pipe", "flow", "mms"}:
        raise ValueError("[solver] problem must be pipe, flow, or mms")
    mesh = _expand(_require(cp, "solver", "mesh"))
    _opt(a, "mesh", mesh)
    _opt(a, "problem", problem)

    re = _get(cp, "solver", "re", "20" if problem == "pipe" else "1")
    _opt(a, "re", re)
    nu = _get(cp, "solver", "nu", "").strip()
    if nu:
        _opt(a, "nu", nu)
    convection = _get(cp, "solver", "convection", "central").lower()
    if convection not in {"central", "none"}:
        raise ValueError("[solver] convection must be central or none")
    _opt(a, "convection", convection)

    # BC layer: intentionally only the BCs supported by the frozen solver.
    wall = _get(cp, "boundary", "wall_patch", "patch_0_0")
    inlet = _get(cp, "boundary", "inlet_patch", "patch_2_0")
    outlet = _get(cp, "boundary", "outlet_patch", "patch_1_0")
    wall_type = _get(cp, "boundary", "wall_type", "wall").lower()
    outlet_type = _get(cp, "boundary", "outlet_type", "pressure0").lower()
    if wall_type not in {"wall", "no_slip", "noslip"}:
        raise ValueError("NodalS v1.00 supports only no-slip wall BCs")
    if outlet_type not in {"pressure0", "pressure_0", "natural_zero_traction"}:
        raise ValueError("NodalS v1.00 supports only the existing pressure-0/open outlet semantics")

    inlet_type = _get(cp, "boundary", "inlet_type", "parabolic" if problem == "pipe" else "normal").lower()
    if inlet_type not in {"parabolic", "normal"}:
        raise ValueError("NodalS v1.00 inlet_type must be parabolic or normal")

    if problem == "pipe":
        _opt(a, "pipe_wall_patch", wall)
        _opt(a, "pipe_inlet_patch", inlet)
        _opt(a, "pipe_outlet_patch", outlet)
        bulk = _get(cp, "boundary", "bulk_velocity", "1.0")
        _opt(a, "pipe_bulk_velocity", bulk)
        if inlet_type == "parabolic":
            # The frozen exact HP diagnostic path is the +z convention from Gate9N.
            if inlet != "patch_2_0" or outlet != "patch_1_0":
                raise ValueError(
                    "parabolic pipe v1.00 uses the validated +z exact path: "
                    "inlet_patch=patch_2_0 and outlet_patch=patch_1_0"
                )
            _opt(a, "inlet_bc", "parabolic")
        else:
            _opt(a, "inlet_bc", "fixed_normal_speed")
            speed = abs(float(_get(cp, "boundary", "speed", bulk)))
            _opt(a, "inlet_normal_mode", "average_patch_normal")
            _opt(a, "inlet_normal_speed", f"{-speed:.17g}")
    elif problem == "flow":
        if inlet_type != "normal":
            raise ValueError("generic flow in v1.00 supports only the existing normal inlet")
        _opt(a, "flow_wall_patches", wall)
        _opt(a, "flow_inlet_patch", inlet)
        _opt(a, "flow_outlet_patch", outlet)
        speed = abs(float(_get(cp, "boundary", "speed", "0.2")))
        _opt(a, "inlet_bc", "fixed_normal_speed")
        _opt(a, "inlet_normal_mode", "average_patch_normal")
        _opt(a, "inlet_normal_speed", f"{-speed:.17g}")

    # SUPG.
    supg = _bool(_get(cp, "supg", "enabled", "false"), "[supg] enabled")
    _opt(a, "supg", 1 if supg else 0)
    _opt(a, "supg_tau_scale", _get(cp, "supg", "tau_scale", "0.05"))
    _opt(a, "supg_form", _get(cp, "supg", "form", "implicit"))
    _opt(a, "supg_kernel", _get(cp, "supg", "kernel", "fast"))
    _opt(a, "supg_quad_points", _get(cp, "supg", "quad_points", "64"))

    # SIMPLE/SIMPLEC.
    _opt(a, "simple_variant", _get(cp, "simple", "variant", "simplec"))
    _opt(a, "alpha_u", _get(cp, "simple", "alpha_u", "0.7"))
    _opt(a, "alpha_p", _get(cp, "simple", "alpha_p", "1.0"))
    _opt(a, "u_relax_mode", _get(cp, "simple", "u_relax_mode", "row_l1"))
    _opt(a, "rau_mode", _get(cp, "simple", "rau_mode", "diag"))
    _opt(a, "rau_scale", _get(cp, "simple", "rau_scale", "1"))
    _opt(a, "simplec_blend", _get(cp, "simple", "simplec_blend", "1"))
    _opt(a, "simplec_floor_fraction", _get(cp, "simple", "simplec_floor_fraction", "1e-6"))
    _opt(a, "simplec_fallback", _get(cp, "simple", "simplec_fallback", "diag"))
    _opt(a, "simple_rtol", _get(cp, "simple", "rtol", "1e-3"))
    _opt(a, "simple_max_it", _get(cp, "simple", "max_iterations", "10000"))

    # Momentum predictor.
    _opt(a, "u_rtol", _get(cp, "momentum", "rtol", "1e-8"))
    _opt(a, "u_atol", _get(cp, "momentum", "atol", "0"))
    _opt(a, "u_rel_drop", _get(cp, "momentum", "relative_drop", "0.5"))
    _opt(a, "u_max_it", _get(cp, "momentum", "max_iterations", "20000"))
    _opt(a, "u_check_every", _get(cp, "momentum", "check_every", "1"))
    _opt(a, "u_local_sweeps", _get(cp, "momentum", "local_sweeps", "1"))
    _opt(a, "u_sor_omega", _get(cp, "momentum", "sor_omega", "1"))

    # Pressure backend. NodalS exposes both accepted production architectures:
    #   fast_full_gamg  : full explicit Schur snapshot as Pmat + PETSc GAMG-preconditioned Richardson
    #   compact_cheb    : compact FE-face-energy Pmat + GAMG PC + custom FP64 Chebyshev
    # compact_pcg remains available as the legacy low-memory custom-PCG alternative.
    raw_psolver = _get(cp, "pressure", "mode", _get(cp, "pressure", "solver", "fast_full_gamg")).lower()
    aliases = {
        "fast_full_gamg": "fast_full_gamg",
        "full_gamg": "fast_full_gamg",
        "full_gamg_richardson": "fast_full_gamg",
        "gamg_richardson": "fast_full_gamg",
        "compact_cheb": "compact_cheb",
        "cheb_gamg": "compact_cheb",
        "low_memory": "compact_cheb",
        "low_memory_cheb": "compact_cheb",
        "compact_pcg": "compact_pcg",
        "pcg_gamg": "compact_pcg",
    }
    if raw_psolver not in aliases:
        raise ValueError("[pressure] mode must be fast_full_gamg, compact_cheb, or compact_pcg")
    psolver = aliases[raw_psolver]

    requested_pmat = _get(cp, "pressure", "pmat", "").lower().strip()
    if psolver == "fast_full_gamg":
        if requested_pmat and requested_pmat != "full":
            raise ValueError("fast_full_gamg requires [pressure] pmat = full")
        pmat = "full"
        solve_mode = "gamg_richardson"
        default_refresh, default_prtol, default_pmax = "100", "0.5", "20"
        default_pksp = "richardson"
    elif psolver == "compact_cheb":
        if requested_pmat and requested_pmat != "compact":
            raise ValueError("compact_cheb requires [pressure] pmat = compact")
        pmat = "compact"
        solve_mode = "custom_pcg"
        default_refresh, default_prtol, default_pmax = "1", "0.1", "300"
        default_pksp = "cg"
    else:
        pmat = requested_pmat or "compact"
        if pmat not in {"compact", "full"}:
            raise ValueError("[pressure] pmat must be compact or full")
        solve_mode = "custom_pcg"
        default_refresh, default_prtol, default_pmax = ("1", "0.1", "300") if pmat == "compact" else ("100", "0.1", "300")
        default_pksp = "cg"

    _opt(a, "pressure_solve_mode", solve_mode)
    _opt(a, "p_pmat", "native_face" if pmat == "compact" else "full")
    _opt(a, "p_operator", "factored")
    _opt(a, "p_preconditioner_refresh", _get(cp, "pressure", "refresh", default_refresh))
    _opt(a, "p_ksp_type", _get(cp, "pressure", "ksp_type", default_pksp))
    prtol = _get(cp, "pressure", "rtol", default_prtol)
    patol = _get(cp, "pressure", "atol", "1e-12")
    _opt(a, "p_ksp_rtol", prtol)
    _opt(a, "p_ksp_atol", patol)
    _opt(a, "p_ksp_divtol", _get(cp, "pressure", "divtol", "1e8"))
    _opt(a, "p_ksp_max_it", _get(cp, "pressure", "max_iterations", default_pmax))
    _opt(a, "p_pc_type", _get(cp, "pressure", "pc_type", "gamg"))
    _opt(a, "p_mg_levels_ksp_type", _get(cp, "pressure", "mg_level_ksp", "richardson"))
    _opt(a, "p_mg_levels_pc_type", _get(cp, "pressure", "mg_level_pc", "jacobi"))
    _opt(a, "p_mg_levels_ksp_max_it", _get(cp, "pressure", "mg_level_iterations", "1"))

    # Compact FE-face-energy and Chebyshev controls are enabled only for the low-memory route.
    compact_fe = (psolver in {"compact_cheb", "compact_pcg"} and pmat == "compact")
    _opt(a, "gate9g_fe_face_energy", 1 if compact_fe else 0)
    _opt(a, "gate9g_richardson", 0)
    _opt(a, "gate9h_chebyshev", 0)
    _opt(a, "gate9i_auto_chebyshev", 1 if psolver == "compact_cheb" else 0)
    _opt(a, "gate9i_power_its", _get(cp, "chebyshev", "power_iterations", "10"))
    _opt(a, "gate9i_safety", _get(cp, "chebyshev", "lambda_max_safety", "1.2"))
    _opt(a, "gate9i_lambda_min_fraction", _get(cp, "chebyshev", "lambda_min_fraction", "0.06"))
    _opt(a, "gate9i_rtol", _get(cp, "chebyshev", "rtol", prtol))
    _opt(a, "gate9i_atol", _get(cp, "chebyshev", "atol", patol))
    _opt(a, "gate9i_spectrum_refresh", _get(cp, "chebyshev", "spectrum_refresh", "100"))
    cmode = _get(cp, "chebyshev", "mode", "adaptive").lower()
    if cmode not in {"adaptive", "fixed"}:
        raise ValueError("[chebyshev] mode must be adaptive or fixed")
    fixed_steps = _get(cp, "chebyshev", "steps", "40") if cmode == "fixed" else "0"
    _opt(a, "gate9i_fixed_steps", fixed_steps)
    _opt(a, "gate9i_require_target", 1 if _bool(_get(cp, "chebyshev", "require_target", "false"), "[chebyshev] require_target") else 0)
    _opt(a, "gate9i_initial_block", _get(cp, "chebyshev", "initial_block", "8"))
    _opt(a, "gate9i_extend_block", _get(cp, "chebyshev", "extend_block", "4"))
    _opt(a, "gate9i_max_steps", _get(cp, "chebyshev", "max_steps", "40"))
    _opt(a, "gate9d_divergence_factor", _get(cp, "chebyshev", "divergence_factor", "1e8"))

    # Output/runtime instrumentation. vtu_output is injected by the launcher unless explicitly set.
    _opt(a, "write_vtu", 1 if _bool(_get(cp, "output", "write_vtu", "true"), "[output] write_vtu") else 0)
    explicit_vtu = _get(cp, "output", "vtu_output", "").strip()
    if explicit_vtu:
        _opt(a, "vtu_output", _expand(explicit_vtu))
    _opt(a, "vtu_velocity_mode", _get(cp, "output", "vtu_velocity_mode", "cell_average"))
    _opt(a, "resource_profile", 1 if _bool(_get(cp, "output", "resource_profile", "true"), "[output] resource_profile") else 0)
    _opt(a, "memory_audit", 1 if _bool(_get(cp, "output", "memory_audit", "false"), "[output] memory_audit") else 0)
    _opt(a, "mesh_audit_only", 1 if _bool(_get(cp, "output", "mesh_audit_only", "false"), "[output] mesh_audit_only") else 0)

    # Retain the known Gate9N production disable/strictness switches unless the case overrides them.
    defaults: Dict[str, str] = {
        "gate1_compare_pcg": "0", "gate3_mp_probe": "0", "gate5_kp_gamg_probe": "0",
        "gate6_diffusion_pcd_probe": "0", "gate7_cp_probe": "0", "gate8_esw_bc_probe": "0",
        "gate9e_ngfv": "0", "m3_static_reference": "0", "m4b_b_reference": "0",
        "m5b_pcg_reference": "0", "m6b_velocity_reference": "0",
        "custom_pressure_b_shadow_tol": "5e-12", "custom_pressure_pcg_reference_tol": "5e-10",
        "m10_pcg_profile": "0", "pressure_profile": "0",
    }
    if cp.has_section("petsc"):
        for k, v in cp.items("petsc"):
            defaults[k.lstrip("-").strip()] = v.strip()
    for k, v in defaults.items():
        _opt(a, k, v)
    a.append("-options_left")
    return a


def _case_overrides(cp, overrides: Iterable[str]) -> None:
    for item in overrides:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise ValueError(f"--set expects section.key=value, got {item!r}")
        lhs, value = item.split("=", 1)
        section, key = lhs.split(".", 1)
        section, key = section.strip().lower(), key.strip().lower()
        if not cp.has_section(section):
            cp.add_section(section)
        cp.set(section, key, value)


def _find_solver(script: Path, explicit: str | None) -> Path:
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(_expand(explicit)))
    if os.environ.get("NODALS_SOLVER_EXE"):
        candidates.append(Path(_expand(os.environ["NODALS_SOLVER_EXE"])))
    project = script.resolve().parents[1]
    candidates.append(project / "nodals_solver")
    # Installed layout: PREFIX/bin/nodals -> PREFIX/libexec/nodals/nodals_solver
    prefix = script.resolve().parent.parent
    candidates.append(prefix / "libexec" / "nodals" / "nodals_solver")
    for p in candidates:
        if p.is_file() and os.access(p, os.X_OK):
            return p
    return candidates[0] if candidates else project / "nodals_solver"


def _stream_run(cmd: List[str], log_path: Path, timeout: int) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            return proc.wait(timeout=None if timeout <= 0 else timeout)
        except KeyboardInterrupt:
            proc.terminate()
            raise
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                proc.kill()
            return 124


def main(argv: List[str]) -> int:
    passthrough: List[str] = []
    if "--" in argv:
        i = argv.index("--")
        passthrough = argv[i + 1:]
        argv = argv[:i]

    ap = argparse.ArgumentParser(prog="nodals", description="Run the frozen NodalS v1.00 FEM SIMPLE/SIMPLEC solver from a .case file")
    ap.add_argument("case", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="print the MPI command and do not execute")
    ap.add_argument("--dump-options-json", action="store_true", help="print translated PETSc options as JSON")
    ap.add_argument("--exe", help="solver executable override")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="temporary case override")
    ns = ap.parse_args(argv)

    case_path = ns.case.expanduser().resolve()
    cp = _read_case(case_path)
    _case_overrides(cp, ns.set)
    options = build_solver_options(cp)

    name = _get(cp, "run", "name", case_path.stem)
    ranks = int(_get(cp, "run", "ranks", "16"))
    timeout = int(_get(cp, "run", "timeout_seconds", "0"))
    map_by = _get(cp, "run", "map_by", "core")
    bind_to = _get(cp, "run", "bind_to", "core")
    report_bindings = _bool(_get(cp, "run", "report_bindings", "true"), "[run] report_bindings")
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_cfg = _get(cp, "run", "output_root", "auto")
    if out_cfg.lower() == "auto":
        outroot = Path.home() / "Downloads" / f"NodalS_{name}_{stamp}"
    else:
        outroot = Path(_expand(out_cfg))
    log_path = outroot / f"{name}.log"

    # Add an automatic VTU location only when the case did not explicitly supply one.
    if not _get(cp, "output", "vtu_output", "").strip():
        options += ["-vtu_output", str(outroot / f"{name}_solution.vtu")]

    solver = _find_solver(Path(__file__), ns.exe)
    mpirun = shutil.which("mpirun") or "mpirun"
    cmd = [mpirun, "-np", str(ranks)]
    if map_by:
        cmd += ["--map-by", map_by]
    if bind_to:
        cmd += ["--bind-to", bind_to]
    if report_bindings:
        cmd.append("--report-bindings")
    stdbuf = shutil.which("stdbuf")
    if stdbuf:
        cmd += [stdbuf, "-oL", "-eL"]
    cmd += [str(solver)] + options + passthrough

    print(f"NODALS_CASE name={name} file={case_path} ranks={ranks} output={outroot}")
    print("NODALS_COMMAND " + shlex.join(cmd))
    if ns.dump_options_json:
        print("NODALS_OPTIONS_JSON=" + json.dumps(options))
    if ns.dry_run:
        print("NODALS_RUN_RESULT status=DRY_RUN")
        return 0

    if not solver.is_file() or not os.access(solver, os.X_OK):
        print(f"NODALS_RUN_RESULT status=FAIL reason=missing_solver executable={solver}", file=sys.stderr)
        return 2
    outroot.mkdir(parents=True, exist_ok=True)
    rc = _stream_run(cmd, log_path, timeout)
    print(f"NODALS_ARTIFACTS log={log_path} outroot={outroot}")
    if rc != 0:
        print(f"NODALS_RUN_RESULT status=FAIL rc={rc}")
        return rc
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if "P1BF3_RESULT status=PASS" in text:
        print("NODALS_RUN_RESULT status=PASS")
        return 0
    if _bool(_get(cp, "output", "mesh_audit_only", "false"), "[output] mesh_audit_only") and "P1BF3_MESH_AUDIT_ONLY status=PASS" in text:
        print("NODALS_RUN_RESULT status=PASS_AUDIT_ONLY")
        return 0
    print("NODALS_RUN_RESULT status=NOT_CONVERGED reason=no_P1BF3_RESULT_PASS")
    return 21


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except (ValueError, configparser.Error, OSError) as e:
        print(f"NODALS_CASE_ERROR {e}", file=sys.stderr)
        raise SystemExit(4)
