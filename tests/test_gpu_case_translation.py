#!/usr/bin/env python3
from pathlib import Path
import importlib.util
import configparser

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("nodals_gpu_case", root / "scripts" / "nodals_gpu_case.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def pairs(opts):
    assert len(opts) % 2 == 0, opts
    return {opts[i]: opts[i + 1] for i in range(0, len(opts), 2)}


case = root / "cases" / "gpu-h8-annotated.case"
cp = mod._read_case(case)
o = pairs(mod.build_gpu_options(cp))

# Canonical annotated file selects the new CF/PMIS fast family.
assert o["--amg-hierarchy"] == "cf"
assert o["--cf-coarsening"] == "pmis"
assert o["--cf-strength"] == "classical-negative"
assert o["--cf-theta"] == "0.25"
assert o["--cf-interp"] == "exti"
assert o["--cf-pmax"] == "8"
assert o["--cf-aggressive-first"] == "0"
assert o["--amg-smoother"] == "jacobi"
assert o["--pressure-solver"] == "richardson"
assert o["--pressure-richardson-omega"] == "1.0"
assert o["--fine-csr-refresh-every"] == "1"
assert o["--momentum-work"] == "fgs1"
assert o["--run-mode"] == "converge"
assert mod._precision(cp) == "fp32"

# Original SA + Chebyshev + PCG remains one set of case overrides away.
cp2 = mod._read_case(case)
cp2.set("pressure_amg", "hierarchy", "sa")
cp2.set("pressure_amg", "smoother", "cheb2")
cp2.set("pressure", "solver", "pcg")
o2 = pairs(mod.build_gpu_options(cp2))
assert o2["--amg-hierarchy"] == "sa"
assert o2["--amg-smoother"] == "cheb2"
assert o2["--amg-spectrum-policy"] == "auto"
assert o2["--amg-power-its"] == "16"
assert o2["--amg-cheb-degree"] == "2"
assert o2["--pressure-solver"] == "pcg"

# L1-Jacobi and outer Chebyshev are both exposed.
cp3 = mod._read_case(case)
cp3.set("pressure_amg", "smoother", "l1jacobi")
cp3.set("pressure", "solver", "cheb")
o3 = pairs(mod.build_gpu_options(cp3))
assert o3["--amg-smoother"] == "l1jacobi"
assert o3["--pressure-solver"] == "cheb"
assert o3["--pressure-cheb-degree"] == "3"
assert o3["--pressure-power-its"] == "6"

# Fixed-10 is explicit and must be internally consistent.
cp4 = mod._read_case(case)
cp4.set("run", "run_mode", "fixed10")
cp4.set("simple", "max_iterations", "10")
o4 = pairs(mod.build_gpu_options(cp4))
assert o4["--run-mode"] == "fixed10"
assert o4["--max-outer"] == "10"

# Invalid Chebyshev-without-spectrum combination is rejected in the translator.
cp5 = mod._read_case(case)
cp5.set("pressure_amg", "smoother", "cheb2")
cp5.set("pressure_amg", "spectrum_policy", "off")
try:
    mod.build_gpu_options(cp5)
except ValueError as e:
    assert "requires spectrum_policy" in str(e)
else:
    raise AssertionError("cheb2 incorrectly accepted spectrum_policy=off")

# Unknown keys fail instead of being silently ignored.
cp6 = configparser.ConfigParser(interpolation=None)
cp6.optionxform = str.lower
cp6.read(case)
cp6.set("pressure_amg", "smother", "jacobi")
try:
    mod.build_gpu_options(cp6)
except ValueError as e:
    assert "unknown GPU case key" in str(e)
else:
    raise AssertionError("unknown GPU case key was silently ignored")

print("NODALS_GPU_CASE_TRANSLATION status=PASS cf_pmis=1 sa=1 smoothers=3 pressure_solvers=3 fixed10=1 strict_keys=1")
