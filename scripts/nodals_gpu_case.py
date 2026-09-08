#!/usr/bin/env python3
"""Run the NodalS H8 serial-CUDA solver from an annotated .case file.

This is intentionally a thin configuration/launch layer.  It does not change
H8 numerics: every friendly case key maps to an already existing H8 CLI option.
"""
from __future__ import annotations

import argparse
import configparser
import datetime as _dt
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Dict, Iterable, List

TRUE = {"1", "true", "yes", "on"}
FALSE = {"0", "false", "no", "off"}

KNOWN_KEYS: Dict[str, set[str]] = {
    "run": {"name", "precision", "run_mode", "output_root", "timeout_seconds"},
    "solver": {"mesh", "re", "bulk_velocity"},
    "boundary": {"wall_patch", "inlet_patch", "outlet_patch"},
    "simple": {"rtol", "max_iterations", "alpha_u", "alpha_p"},
    "momentum": {"work", "rtol", "atol", "relative_drop", "omega", "max_iterations"},
    "pressure": {
        "solver", "rtol", "atol", "max_iterations", "snapshot_tolerance",
        "fine_csr_refresh_every", "richardson_omega", "chebyshev_degree",
        "power_iterations", "chebyshev_low_fraction", "power_safety",
    },
    "pressure_amg": {
        "hierarchy", "smoother", "terminal_rows", "spectrum_policy",
        "power_iterations", "chebyshev_degree", "lambda_safety",
        "lambda_low_fraction", "jacobi_omega",
    },
    "pressure_amg_cf": {
        "coarsening", "strength", "theta", "interpolation", "pmax",
        "aggressive_first",
    },
}


def _expand(value: str) -> str:
    return os.path.expandvars(os.path.expanduser(value.strip()))


def _bool(value: str, where: str) -> bool:
    s = str(value).strip().lower()
    if s in TRUE:
        return True
    if s in FALSE:
        return False
    raise ValueError(f"{where}: expected boolean, got {value!r}")


def _read_case(path: Path) -> configparser.ConfigParser:
    cp = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#", ";"))
    cp.optionxform = str.lower
    with path.open("r", encoding="utf-8") as f:
        cp.read_file(f)
    _validate_case_keys(cp)
    return cp


def _validate_case_keys(cp: configparser.ConfigParser) -> None:
    for section in cp.sections():
        if section not in KNOWN_KEYS:
            raise ValueError(f"unknown GPU case section [{section}]")
        allowed = KNOWN_KEYS[section]
        for key, _ in cp.items(section):
            if key not in allowed:
                raise ValueError(f"unknown GPU case key [{section}] {key}")


def _get(cp: configparser.ConfigParser, section: str, key: str, default: str | None = None) -> str | None:
    if cp.has_option(section, key):
        return cp.get(section, key).strip()
    return default


def _require(cp: configparser.ConfigParser, section: str, key: str) -> str:
    value = _get(cp, section, key)
    if value is None or value == "":
        raise ValueError(f"missing required [{section}] {key}")
    return value


def _case_overrides(cp: configparser.ConfigParser, overrides: Iterable[str]) -> None:
    for item in overrides:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise ValueError(f"--set expects section.key=value, got {item!r}")
        lhs, value = item.split("=", 1)
        section, key = lhs.split(".", 1)
        section, key = section.strip().lower(), key.strip().lower()
        if section not in KNOWN_KEYS:
            raise ValueError(f"unknown GPU case section [{section}]")
        if key not in KNOWN_KEYS[section]:
            raise ValueError(f"unknown GPU case key [{section}] {key}")
        if not cp.has_section(section):
            cp.add_section(section)
        cp.set(section, key, value)


def _arg(args: List[str], name: str, value) -> None:
    args.extend([f"--{name}", str(value)])


def build_gpu_options(cp: configparser.ConfigParser) -> List[str]:
    """Translate a GPU .case into the current H8 command-line interface."""
    _validate_case_keys(cp)
    a: List[str] = []

    name = _get(cp, "run", "name", "nodals_gpu")
    run_mode = str(_get(cp, "run", "run_mode", "converge")).lower()
    if run_mode not in {"fixed10", "converge"}:
        raise ValueError("[run] run_mode must be fixed10 or converge")

    mesh = _expand(_require(cp, "solver", "mesh"))
    _arg(a, "mesh", mesh)
    _arg(a, "tag", name)
    _arg(a, "wall", _get(cp, "boundary", "wall_patch", "patch_0_0"))
    _arg(a, "inlet", _get(cp, "boundary", "inlet_patch", "patch_2_0"))
    _arg(a, "outlet", _get(cp, "boundary", "outlet_patch", "patch_1_0"))
    _arg(a, "re", _get(cp, "solver", "re", "20"))
    _arg(a, "bulk", _get(cp, "solver", "bulk_velocity", "1.0"))

    simple_rtol = _get(cp, "simple", "rtol", "1e-3")
    max_outer = int(str(_get(cp, "simple", "max_iterations", "2500")))
    if run_mode == "fixed10" and max_outer != 10:
        raise ValueError("[run] run_mode=fixed10 requires [simple] max_iterations=10")
    _arg(a, "simple-tol", simple_rtol)
    _arg(a, "max-outer", max_outer)
    _arg(a, "alpha-u", _get(cp, "simple", "alpha_u", "0.5"))
    _arg(a, "alpha-p", _get(cp, "simple", "alpha_p", "0.5"))

    work = str(_get(cp, "momentum", "work", "fgs1")).lower()
    if work != "fgs1":
        raise ValueError("current H8 supports only [momentum] work=fgs1")
    _arg(a, "mom-rtol", _get(cp, "momentum", "rtol", "1e-6"))
    _arg(a, "mom-atol", _get(cp, "momentum", "atol", "1e-12"))
    _arg(a, "mom-drop", _get(cp, "momentum", "relative_drop", "0.1"))
    _arg(a, "mom-omega", _get(cp, "momentum", "omega", "1.0"))
    _arg(a, "mom-max", _get(cp, "momentum", "max_iterations", "20000"))

    psolver = str(_get(cp, "pressure", "solver", "richardson")).lower()
    if psolver not in {"pcg", "richardson", "cheb"}:
        raise ValueError("[pressure] solver must be pcg, richardson, or cheb")
    _arg(a, "p-rtol", _get(cp, "pressure", "rtol", "0.5"))
    _arg(a, "p-atol", _get(cp, "pressure", "atol", "1e-12"))
    _arg(a, "p-max", _get(cp, "pressure", "max_iterations", "20"))
    _arg(a, "snapshot-tol", _get(cp, "pressure", "snapshot_tolerance", "5e-6"))
    refresh = int(str(_get(cp, "pressure", "fine_csr_refresh_every", "1")))
    if refresh != 1:
        raise ValueError("current H8 requires [pressure] fine_csr_refresh_every=1")
    _arg(a, "fine-csr-refresh-every", refresh)
    _arg(a, "pressure-solver", psolver)
    _arg(a, "pressure-richardson-omega", _get(cp, "pressure", "richardson_omega", "1.0"))
    _arg(a, "pressure-cheb-degree", _get(cp, "pressure", "chebyshev_degree", "3"))
    _arg(a, "pressure-power-its", _get(cp, "pressure", "power_iterations", "6"))
    _arg(a, "pressure-cheb-low-fraction", _get(cp, "pressure", "chebyshev_low_fraction", "0.05"))
    _arg(a, "pressure-power-safety", _get(cp, "pressure", "power_safety", "1.15"))

    hierarchy = str(_get(cp, "pressure_amg", "hierarchy", "cf")).lower()
    if hierarchy not in {"sa", "cf"}:
        raise ValueError("[pressure_amg] hierarchy must be sa or cf")
    smoother = str(_get(cp, "pressure_amg", "smoother", "jacobi")).lower()
    if smoother not in {"cheb2", "jacobi", "l1jacobi"}:
        raise ValueError("[pressure_amg] smoother must be cheb2, jacobi, or l1jacobi")
    spectrum = str(_get(cp, "pressure_amg", "spectrum_policy", "auto")).lower()
    if spectrum not in {"auto", "always", "off"}:
        raise ValueError("[pressure_amg] spectrum_policy must be auto, always, or off")
    if smoother == "cheb2" and spectrum == "off":
        raise ValueError("[pressure_amg] smoother=cheb2 requires spectrum_policy=auto or always")
    _arg(a, "amg-hierarchy", hierarchy)
    _arg(a, "amg-smoother", smoother)
    _arg(a, "amg-terminal", _get(cp, "pressure_amg", "terminal_rows", "1000"))
    _arg(a, "amg-spectrum-policy", spectrum)
    _arg(a, "amg-power-its", _get(cp, "pressure_amg", "power_iterations", "16"))
    _arg(a, "amg-cheb-degree", _get(cp, "pressure_amg", "chebyshev_degree", "2"))
    _arg(a, "amg-lambda-safety", _get(cp, "pressure_amg", "lambda_safety", "1.5"))
    _arg(a, "amg-lambda-low-fraction", _get(cp, "pressure_amg", "lambda_low_fraction", "0.05"))
    _arg(a, "amg-jacobi-omega", _get(cp, "pressure_amg", "jacobi_omega", "0.7"))

    coarsening = str(_get(cp, "pressure_amg_cf", "coarsening", "pmis")).lower()
    strength = str(_get(cp, "pressure_amg_cf", "strength", "classical-negative")).lower()
    interp = str(_get(cp, "pressure_amg_cf", "interpolation", "exti")).lower()
    if coarsening != "pmis":
        raise ValueError("current H8 CF hierarchy supports only [pressure_amg_cf] coarsening=pmis")
    if strength != "classical-negative":
        raise ValueError("current H8 CF hierarchy supports only [pressure_amg_cf] strength=classical-negative")
    if interp not in {"direct", "exti"}:
        raise ValueError("[pressure_amg_cf] interpolation must be direct or exti")
    aggressive = 1 if _bool(str(_get(cp, "pressure_amg_cf", "aggressive_first", "false")), "[pressure_amg_cf] aggressive_first") else 0
    _arg(a, "cf-coarsening", coarsening)
    _arg(a, "cf-strength", strength)
    _arg(a, "cf-theta", _get(cp, "pressure_amg_cf", "theta", "0.25"))
    _arg(a, "cf-interp", interp)
    _arg(a, "cf-pmax", _get(cp, "pressure_amg_cf", "pmax", "8"))
    _arg(a, "cf-aggressive-first", aggressive)

    _arg(a, "momentum-work", work)
    _arg(a, "run-mode", run_mode)
    return a


def _precision(cp: configparser.ConfigParser) -> str:
    precision = str(_get(cp, "run", "precision", "fp32")).lower()
    if precision not in {"fp32", "fp64"}:
        raise ValueError("[run] precision must be fp32 or fp64")
    return precision


def _find_solver(script: Path, precision: str, explicit: str | None) -> Path:
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(_expand(explicit)))
    env_name = f"NODALS_GPU_{precision.upper()}_EXE"
    if os.environ.get(env_name):
        candidates.append(Path(_expand(os.environ[env_name])))
    project = script.resolve().parents[1]
    binary = f"nodals_gpu_h8_{precision}"
    candidates.append(project / "gpu" / "serial_cuda" / binary)
    prefix = script.resolve().parent.parent
    candidates.append(prefix / "libexec" / "nodals" / binary)
    for p in candidates:
        if p.is_file() and os.access(p, os.X_OK):
            return p
    return candidates[0] if candidates else project / "gpu" / "serial_cuda" / binary


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

    ap = argparse.ArgumentParser(prog="nodals-gpu", description="Run NodalS H8 serial CUDA from a .case file")
    ap.add_argument("case", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dump-options-json", action="store_true")
    ap.add_argument("--exe", help="GPU executable override")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    ns = ap.parse_args(argv)

    case_path = ns.case.expanduser().resolve()
    cp = _read_case(case_path)
    _case_overrides(cp, ns.set)
    _validate_case_keys(cp)
    options = build_gpu_options(cp)
    precision = _precision(cp)
    solver = _find_solver(Path(__file__), precision, ns.exe)

    name = str(_get(cp, "run", "name", case_path.stem))
    timeout = int(str(_get(cp, "run", "timeout_seconds", "0")))
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_cfg = str(_get(cp, "run", "output_root", "auto"))
    if out_cfg.lower() == "auto":
        outroot = Path.home() / "Downloads" / f"NodalS_GPU_{name}_{stamp}"
    else:
        outroot = Path(_expand(out_cfg))
    log_path = outroot / f"{name}.log"

    cmd = [str(solver)] + options + passthrough
    print(f"NODALS_GPU_CASE name={name} file={case_path} precision={precision} output={outroot}")
    print("NODALS_GPU_COMMAND " + shlex.join(cmd))
    if ns.dump_options_json:
        print("NODALS_GPU_OPTIONS_JSON=" + json.dumps(options))
    if ns.dry_run:
        print("NODALS_GPU_CASE_RESULT status=DRY_RUN")
        return 0

    if not solver.is_file() or not os.access(solver, os.X_OK):
        print(f"NODALS_GPU_CASE_RESULT status=FAIL reason=missing_solver executable={solver}", file=sys.stderr)
        return 2
    rc = _stream_run(cmd, log_path, timeout)
    print(f"NODALS_GPU_ARTIFACTS log={log_path} outroot={outroot}")
    if rc != 0:
        print(f"NODALS_GPU_CASE_RESULT status=FAIL rc={rc}")
        return rc
    text = log_path.read_text(encoding="utf-8", errors="replace")
    passed = any(line.startswith("NODALS_GPU_RESULT gate=H8 ") and "status=PASS" in line for line in text.splitlines())
    if passed:
        print("NODALS_GPU_CASE_RESULT status=PASS")
        return 0
    print("NODALS_GPU_CASE_RESULT status=NOT_CONVERGED reason=no_NODALS_GPU_RESULT_PASS")
    return 21


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except (ValueError, configparser.Error, OSError) as e:
        print(f"NODALS_GPU_CASE_ERROR {e}", file=sys.stderr)
        raise SystemExit(4)
