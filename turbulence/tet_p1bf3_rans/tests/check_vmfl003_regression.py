#!/usr/bin/env python3
from pathlib import Path
import argparse,json,re,sys

ap=argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--mode",choices=["unsmoothed_sgs","smoothed_cheb_degree2"],required=True)
ns=ap.parse_args()

root=Path(__file__).resolve().parents[1]
ref=json.loads((root/"reference/vmfl003_10d_effectiveB_regression.json").read_text())
want=ref[ns.mode]
lines=Path(ns.log).read_text(errors="replace").splitlines()

def last(prefix):
    q=[x for x in lines if x.startswith(prefix)]
    return q[-1] if q else ""
def get(line,key,cast=float):
    m=re.search(r'(?:^|\s)'+re.escape(key)+r'=([^\s]+)',line)
    if not m: return None
    return cast(m.group(1))

result=last("P1BF3_RESULT ")
work=last("P1BF3_WORK ")
flow=last("P1BF3_FLOW_DIAGNOSTICS ")
press=last("P1BF3_FLOW_PRESSURE ")
nsf=last("P1BF3_NS_FINAL ")
hier=last("P1BF3_CUSTOM_AMG_HIERARCHY ")

got={
 "outerIts":get(result,"outerIts",int),
 "avgPCG":get(work,"avgPCG"),
 "dropPa":get(press,"dropInMinusOut"),
 "massRelative":get(flow,"massRelative"),
 "wallFSpalding":get(nsf,"wallFSpalding"),
 "amgRetainedMiB":get(hier,"retainedEstimateMiB"),
}
if "status=PASS" not in result:
    print("TURB_REGRESSION status=FAIL reason=solver_not_PASS")
    sys.exit(2)

# Exact iteration counts are diagnostic, not a physics requirement.
rtol=float(ref["physics_relative_tolerance"])
mass_tol=float(ref["mass_absolute_tolerance"])
checks={}
for k in ["dropPa","wallFSpalding"]:
    checks[k]=got[k] is not None and abs(got[k]-want[k])/max(abs(want[k]),1e-300) <= rtol
checks["massRelative"]=got["massRelative"] is not None and abs(got["massRelative"]-want["massRelative"]) <= mass_tol

# Algebra gates must remain present and pass.
checks["schurParity"]=any("P1BF3_M4B_LIVE_SCHUR_PARITY" in x and "status=PASS" in x for x in lines)
checks["diagParity"]=any("P1BF3_CUSTOM_AMG_GATE2_DIAG_PARITY" in x and "status=PASS" in x for x in lines)
checks["galerkinParity"]=any(
    (("P1BF3_CUSTOM_AMG_GATE3_GALERKIN" in x) or ("P1BF3_CUSTOM_AMG_SA_GALERKIN" in x))
    and "status=PASS" in x for x in lines)

print("TURB_REGRESSION mode="+ns.mode+" "+
      " ".join(f"{k}={v}" for k,v in got.items()))
print("TURB_REGRESSION_CHECKS "+" ".join(f"{k}={int(v)}" for k,v in checks.items()))
if not all(checks.values()):
    print("TURB_REGRESSION status=FAIL")
    sys.exit(1)
print("TURB_REGRESSION status=PASS")
