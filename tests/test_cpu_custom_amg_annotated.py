#!/usr/bin/env python3
from pathlib import Path
import configparser

root = Path(__file__).resolve().parents[1]
case = root / "cases" / "cpu-custom-amg-annotated.case"
cp = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#", ";"))
cp.optionxform = str.lower
cp.read(case)
assert cp.get("pressure", "mode") == "pcg_unsmoothed"
assert cp.get("pressure", "pmat") == "custom"
assert cp.get("pressure_amg", "smoother") == "sgs"
for key in (
    "target_aggregate", "min_aggregate", "soft_max_aggregate",
    "chebyshev_degree", "power_iterations", "lambda_safety",
    "lambda_low_fraction", "coarse_target_rows",
    "interpolation_max_row_nnz", "sa_damping", "richardson_omega",
):
    assert cp.has_option("pressure_amg", key), key
assert cp.get("simple", "u_relax_mode") == "row_l1"
assert not cp.has_option("simple", "nu_relax_mode")
print("NODALS_CPU_CUSTOM_AMG_ANNOTATED status=PASS unsmoothed=1 smoothed_documented=1 keys=11")
