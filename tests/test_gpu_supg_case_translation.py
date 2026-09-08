#!/usr/bin/env python3
from pathlib import Path
import configparser
import importlib.util

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("nodals_gpu_case",root/"scripts/nodals_gpu_case.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

cp=m._read_case(root/"cases/gpu-h8-annotated.case")
cp.set("supg","enabled","true")
cp.set("supg","tau_scale","0.05")
cp.set("supg","magic","9.0")
cp.set("supg","form","implicit")
cp.set("supg","quad_points","64")
opts=m.build_gpu_options(cp)
pairs={opts[i]:opts[i+1] for i in range(0,len(opts),2)}
assert pairs["--supg"]=="1"
assert pairs["--supg-tau-scale"]=="0.05"
assert pairs["--supg-magic"]=="9.0"
assert pairs["--supg-form"]=="implicit"
assert pairs["--supg-quad-points"]=="64"
assert pairs["--fine-csr-refresh-every"]=="1"

cp.set("supg","enabled","false")
opts=m.build_gpu_options(cp)
pairs={opts[i]:opts[i+1] for i in range(0,len(opts),2)}
assert pairs["--supg"]=="0"

cp.set("supg","enabled","true")
cp.set("supg","form","explicit")
try:
    m.build_gpu_options(cp)
except ValueError as e:
    assert "form=implicit" in str(e)
else:
    raise AssertionError("explicit GPU SUPG unexpectedly accepted")

cp.set("supg","form","implicit")
cp.set("supg","quad_points","27")
try:
    m.build_gpu_options(cp)
except ValueError as e:
    assert "quad_points=64" in str(e)
else:
    raise AssertionError("non-64 GPU SUPG unexpectedly accepted")

print("NODALS_GPU_SUPG_CASE_TRANSLATION status=PASS enabled=implicit64 disabled=fallback invalid_form=REJECT invalid_quad=REJECT")
