#!/usr/bin/env python3
"""Non-CUDA source-contract checks for generalized Robin + full momentum SUPG.

Checks routing and case compatibility; cannot establish GPU operator parity or
solver convergence. Designed to fail if the LHS/RHS SUPG pair is inadvertently
separated or the original Robin momentum/omega wall treatments are changed.
"""
from pathlib import Path
import configparser

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"source/puretet_profile5_eq5_hxt5b"
def read(path): return path.read_text(encoding="utf8")
app=read(SRC/"modules/app/sst_application.inc")
lhs=read(SRC/"modules/coupling/simple_momentum_assembly.inc")
rhs=read(SRC/"modules/coupling/simple_pressure_predictor.inc")
op=read(SRC/"modules/wall/stage22d4d_coupled_matrixfree.inc")
driver=read(ROOT/"nodals.py")
case=ROOT.parents[1]/"examples/spiral_swept_207k/SPIRAL_GENROBIN_SKEW_FULL_SUPG_FGMRES.case"
cp=configparser.ConfigParser(interpolation=None);cp.read(case)
assert cp["wall_function"]["boundary_form"]=="generalized_robin_nitsche"
assert cp["wall_function"]["tangential_consistency"]=="off"
assert cp["convection"]["form"]=="skew_symmetric"
assert cp["solver.u"]["method"]=="coupled-fgmres"
assert cp["stress"]["form"]=="deviatoric"
assert cp["supg"].getboolean("momentum")
assert not cp["supg"].getboolean("k")
assert not cp["supg"].getboolean("omega")
assert float(cp["supg"]["tau_scale"])==0.05
assert float(cp["wall_function"]["stabilization_c"])==15
assert "NODALS_GENROBIN_SUPG enabled=1" in app
assert "genrobinSupgFgmres" in app
assert "!genrobinSupgFgmres" in app
assert "stage33a_momentum_supg64<<<" in lhs
assert "d_turb.p,d_diagRTurb.p" in lhs
assert "momentum_full_deviatoric_supg_rhs64<<<" in rhs
assert "d_rhs0.p,d_rhs1.p,d_rhs2.p" in rhs
assert rhs.index("momentum_full_deviatoric_supg_rhs64<<<") < rhs.index("s22d4d_pack<<<")
assert "d4op.set_genrobin(genrobinViscosityMode,nu,genrobinC);" in rhs
assert "genrobin_tangent_action<<<" in op
assert "hxt5b::spmv<<<" in op
assert "'NODALS_MOMENTUM_FULL_SUPG':str(int(flag(cp,'supg','momentum',False)))" in driver
assert "generalized_robin_nitsche experiment requires momentum SUPG off" not in driver
print("GENROBIN_SKEW_SUPG_WIRING=PASS mode=COUPLED_FGMRES tauScale=0.05")
