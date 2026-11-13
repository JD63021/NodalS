#!/usr/bin/env python3
"""Compare EXISTING 99k-pipe native terminal logs, without modifying/re-running anything.

Run in the experimental worktree:
    python3 experiments/of12_nutwall_p1bf3/ANALYZE_LOCAL_LOGS.py

Optional: --runs-root /absolute/path/to/OF12_NUTWALL_STR100K_20261113
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

DEFAULT = Path.home() / "NodalS_SST_LOCAL_RUNS" / "OF12_NUTWALL_STR100K_20261113"
MARKERS = (
    "NODALS_OPTION_A enabled=1",
    "NODALS_OPTION_A_NOSLIP ",
    "NODALS_OPTION_A_WALL ",
    "NODALS_OPTION_A_NOSLIP_AUDIT ",
    "STAGE22D4I3A_IDENTITY ",
    "STAGE22D4I3A_PRESSURE ",
    "NODALS_TRUE_SIMPLE_CORRECTION ",
    "STAGE22D4B_WALL_NORMAL ",
    "NODALS_SST_PLOT_METRICS ",
    "NODALS_SST_RESIDUALS ",
    "NODALS_SST_G2_FIXED10_TIMING ",
    "NODALS_SST_G2_FINAL ",
    "STAGE27_S03A",
    "NODALS_SST_MOM",
)
KEYS = ("pressureDrop", "wallDarcy", "wallYPlusMean", "meanNutOverNu",
        "maxNutOverNu", "contAbs", "contRel", "UxAbs", "UyAbs", "UzAbs")
NUMBER = re.compile(r"(?<!\w)([A-Za-z]\w*)=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)")


def last_metric(lines: list[str], starts_with: str) -> dict[str, float]:
    for line in reversed(lines):
        if line.startswith(starts_with):
            return {k: float(v) for k, v in NUMBER.findall(line)}
    return {}


def selected(lines: list[str], marker: str, cap: int = 5) -> list[str]:
    matches = [line for line in lines if marker in line]
    if len(matches) <= cap:
        return matches
    return matches[:1] + [f"    ... {len(matches)-cap} intermediate lines omitted ..."] + matches[-(cap-1):]


def load(mode: str, root: Path) -> tuple[Path, list[str]]:
    candidates = sorted((x for x in root.glob(f"{mode}_*steps_*")
                         if (x / "terminal.log").is_file()),
                        key=lambda x: x.name)
    if not candidates:
        raise FileNotFoundError(f"No {mode} terminal.log under {root}")
    path = candidates[-1]
    return path, (path / "terminal.log").read_text(errors="replace").splitlines()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, default=DEFAULT)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    root: Path = args.runs_root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Missing directory: {root}")
    output = args.output or (root / "OPTION_A_DIAGNOSTIC_COMPARISON.txt")
    report: list[str] = []
    values: dict[str, dict[str, float]] = {}
    for mode in ("baseline", "option_a"):
        path, lines = load(mode, root)
        final = last_metric(lines, "NODALS_SST_G2_FINAL ")
        residual = last_metric(lines, "NODALS_SST_RESIDUALS ")
        plot = last_metric(lines, "NODALS_SST_PLOT_METRICS ")
        values[mode] = {**plot, **residual, **final}
        report += ["=" * 82, f"{mode.upper()} RUN: {path}",
                   f"Native log: {path/'terminal.log'} | lines={len(lines)}",
                   "VALIDATION MARKERS:"]
        for key in ("NODALS_OPTION_A enabled=1", "NODALS_OPTION_A_NOSLIP_AUDIT ",
                    "NODALS_OPTION_A_WALL ", "NODALS_SST_G2_FINAL "):
            report.append(f"  {key!r}: {sum(key in l for l in lines)}")
        if mode == "option_a" and not any("NODALS_OPTION_A enabled=1" in l for l in lines):
            report += ["WARNING: native Option A enable marker MISSING; do not trust model designation."]
        if mode == "option_a" and not any("NODALS_OPTION_A_NOSLIP_AUDIT " in l for l in lines):
            report += ["WARNING: post-pressure-correction no-slip audit MISSING."]
        report.append("KEY NATIVE SOLVER LINES:")
        for key in MARKERS:
            report += selected(lines, key)
        problems = [line for line in lines
                    if any(word in line.lower() for word in (
                        "cuda error", "runtime_error", "nonfinite",
                        "nan detected", "failed", "invalid", "exception",
                        "stage22d4i3a_identity ")) and "status=PASS" not in line]
        if problems:
            report += ["POSSIBLE ERRORS/IDENTITIES (last 20):"] + problems[-20:]
        else:
            report.append("POSSIBLE ERRORS/IDENTITIES: none from keyword scan")
    report += ["=" * 82, "AFTER FINAL PRINTED ITERATION: ROUGH A/B COMPARISON"]
    a, b = values["baseline"], values["option_a"]
    for key in KEYS:
        if key in a and key in b:
            base, opt = a[key], b[key]
            ratio = opt / base if base else float("nan")
            report.append(f"{key:18}  baseline={base: .10e}  option_a={opt: .10e}  ratio={ratio:.6g}")
    report += [
        "",
        "NOTE: These early 10-iteration values are UNCONVERGED, not validated friction factors.",
        "Legacy wallDarcy is a Spalding-based diagnostic; it is not proof",
        "that Option A transmits the same physical momentum flux.",
        "Review normal-velocity audit and Schur identity before longer simulations.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"\nSAVED_REPORT={output}")


if __name__ == "__main__":
    main()
