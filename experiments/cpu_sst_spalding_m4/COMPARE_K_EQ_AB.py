#!/usr/bin/env python3
"""Compare most recent finished A/B/C equilibrium-k runs for a given step count.
Usage: python3 experiments/cpu_sst_spalding_m4/COMPARE_K_EQ_AB.py [20]
The script does not guess convergence; prints the last logged iteration as-is.
"""
import sys, pathlib, re, os, math
steps=int(sys.argv[1]) if len(sys.argv)>1 else 20
root=pathlib.Path.home()/"NodalS_SST_LOCAL_RUNS/CPU_M4_K_EQ_AB_20261114"
if not root.exists():raise SystemExit("Run directory does not exist: "+str(root))
def get(line,key):
 m=re.search(r"(?<!\w)"+re.escape(key)+r"=([^\s]+)",line)
 if not m:return float("nan")
 try:return float(m.group(1))
 except ValueError:return float("nan")
print("variant  iteration  dp_kinematic     tauMol      tauEff    wall_nut/nu    wall_k    k_target    k_actual  k_rel_err   vel_update")
for mode in "ABC":
 paths=sorted(root.glob(f"{mode}_{steps}_*/console.log"), key=os.path.getmtime)
 if not paths:
  print(f"{mode}  MISSING");continue
 log=paths[-1].read_text(errors="replace")
 def latest(prefix):
  return next((x for x in reversed(log.splitlines()) if x.startswith(prefix)), "")
 simple=latest("SIMPLE it=")
 budget=latest("NODALS_CPU_M4_RANS_WALL_BUDGET")
 kline=latest("NODALS_CPU_M4_K_POINT")
 vline=latest("NODALS_CPU_M4_ENRICH_UPDATE")
 if not simple:
  print(f"{mode}  FAILED_OR_INCOMPLETE file={paths[-1]}");continue
 it=int(get(simple,"it"))
 fmt=lambda line,k,prec=5:(f"{get(line,k):.{prec}g}" if math.isfinite(get(line,k)) else "-")
 print(f"{mode:7} {it:7}  {fmt(simple,'dp'):>13}  {fmt(budget,'meanTauMol'):>10}  {fmt(budget,'meanTauEff'):>10}  {fmt(budget,'meanNutNu'):>13}  {fmt(budget,'wallK'):>9}  {fmt(kline,'targetMean'):>10}  {fmt(kline,'actualMean'):>10}  {fmt(kline,'relativeMeanAbsError'):>10}  {fmt(vline,'velocityUpdateRel'):>10}")
 print("    log:",paths[-1])
print("NOTE: A/B/C use identical full SST momentum and identical omega sample_log gamma=10000.")
print("NOTE: These are finite-iteration comparisons, not established physical convergence.")
