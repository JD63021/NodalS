#!/usr/bin/env python3
"""Compare latest C (or B) grad-div gamma runs at fixed SIMPLE iterations.
Usage: python3 experiments/cpu_sst_spalding_m4/COMPARE_GRAD_DIV.py C 80
Selects latest console.log per grad-div gamma and fixed alphaU tag.
"""
from pathlib import Path
from collections import defaultdict
import sys, re, math
mode=sys.argv[1] if len(sys.argv)>1 else "C"
steps=int(sys.argv[2]) if len(sys.argv)>2 else 80
root=Path.home()/"NodalS_SST_LOCAL_RUNS"/"CPU_M4_K_EQ_AB_20261114"
def last_prefix(s,key):
 return next((v for v in reversed(s.splitlines()) if v.startswith(key)),"")
def get(s,key):
 v=re.search(r"(?<!\w)"+re.escape(key)+r"=([^\s]+)",s)
 if not v:return float("nan")
 try:return float(v.group(1))
 except ValueError:return float("nan")
def fmt(x):return f"{x:.6g}" if math.isfinite(x) else "?"
def get_str(s,key):
 v=re.search(r"(?<!\w)"+re.escape(key)+r"=([^\s]+)",s)
 return v.group(1) if v else "?"
items={}
for logfile in root.glob(f"{mode}_{steps}_*_gd*/console.log"):
 s=logfile.read_text(errors="replace")
 config=last_prefix(s,"NODALS_CPU_M4_GRADDIV_CONFIG ")
 gd=get(config,"gamma")
 if not math.isfinite(gd):continue
 form=get_str(config,"form")
 key=(form,gd)
 if key not in items or logfile.stat().st_mtime>items[key][0].stat().st_mtime:
  items[key]=(logfile,s)
if not items:raise SystemExit(f"No grad-div runs for {mode} {steps} found under {root}")
print(f"variant={mode} max_outer={steps} (finite iterations, not validated convergence)")
print(" form             gamma[m2/s]  it    dp[m2/s2]  velUpdateRel   divRMS_D/U  nearWallDiv_D/U  coreDiv_D/U  maxDiv_D/U   pCG  status")
for (form,gd),(f,s) in sorted(items.items()):
 simple=last_prefix(s,"SIMPLE it=")
 audit=last_prefix(s,"NODALS_CPU_M4_GRADDIV_AUDIT ")
 update=last_prefix(s,"NODALS_CPU_M4_ENRICH_UPDATE ")
 fatal="FATAL" if "NODALS_CPU_SST_M4_FATAL" in s else ("DONE" if "SIMPLE_RESULT" in s else "RUNNING")
 vals=[gd,get(simple,"it"),get(simple,"dp"),get(update,"velocityUpdateRel"),
       get(audit,"divRMS_D_over_U"),get(audit,"nearWallRMS_D_over_U"),
       get(audit,"coreRMS_D_over_U"),get(audit,"maxAbsDiv_D_over_U"),
       get(simple,"pcg")]
 print(f"{form:<15} "+" ".join(f"{fmt(x):>12}" for x in vals)+f" {fatal}")
 print(f"  log: {f}")
print("Use equal alphaU, mesh and A/B/C mode for meaningful comparisons.")
