#!/usr/bin/env python3
from pathlib import Path
import re,sys

if len(sys.argv)!=2: raise SystemExit("usage: ANALYZE_H4.py <h4_results>")
root=Path(sys.argv[1])

pat=re.compile(
 r"NODALS_GPU_H4_FIXED10 tag=(\S+) cells=(\d+) fixedOuter=(\d+) momentumWork=(\S+) momentumPassesPerOuter=1 momentumResidualAudits=0 "
 r"diffusionPolicy=PERSISTENT_NUMERIC_CSR convectionPolicy=NUMERIC_ONLY_EACH_OUTER fineCsrRefreshEvery=1 fineCsrRefreshCount=(\d+) "
 r"finalRelCont=([0-9eE+.\-]+) avgPressureIts=([0-9eE+.\-]+) pressureDropFit=([0-9eE+.\-]+) exactPressureDrop=([0-9eE+.\-]+) "
 r"pressureDropRelErr=([0-9eE+.\-]+) loopMs=([0-9eE+.\-]+) avgSimpleMs=([0-9eE+.\-]+) assemblyAvgMs=([0-9eE+.\-]+) "
 r"momentumAvgMs=([0-9eE+.\-]+) continuityAvgMs=([0-9eE+.\-]+) pressureAvgMs=([0-9eE+.\-]+) pressureUpdateAvgMs=([0-9eE+.\-]+) status=(\S+)"
)
memre=re.compile(r"NODALS_GPU_H4_MEMORY tag=(\S+) point=after_upload cells=(\d+).*explicitMiB=([0-9eE+.\-]+)")

rows=[]
for lp in sorted(root.glob("*.log")):
    if lp.name in ("build.log","h4_master.log"): continue
    txt=lp.read_text(errors="replace")
    m=pat.search(txt)
    if not m: raise SystemExit(f"ERROR missing H4 fixed10 marker in {lp}")
    mm=memre.search(txt)
    if not mm: raise SystemExit(f"ERROR missing H4 memory marker in {lp}")
    r=dict(tag=m.group(1),cells=int(m.group(2)),outer=int(m.group(3)),mode=m.group(4),
           refreshCount=int(m.group(5)),relCont=float(m.group(6)),pIts=float(m.group(7)),
           dp=float(m.group(8)),dpErr=float(m.group(10)),loop=float(m.group(11)),
           simple=float(m.group(12)),assembly=float(m.group(13)),momentum=float(m.group(14)),
           continuity=float(m.group(15)),pressure=float(m.group(16)),pupdate=float(m.group(17)),
           explicit=float(mm.group(3)))
    r["nonsolver"]=r["assembly"]+r["continuity"]+r["pupdate"]
    r["miups"]=r["cells"]/r["simple"]/1000.0
    rows.append(r)

rows.sort(key=lambda r:(r["cells"],r["mode"]))
h3={
  768530:dict(simple=91.659,momentum=22.184,pressure=50.154,pIts=2.900,relCont=2.786),
  1143041:dict(simple=136.898,momentum=33.870,pressure=73.824,pIts=2.800,relCont=2.958),
}

lines=[
"NodalS H4 — persistent diffusion CSR + convection-only momentum update",
"========================================================================",
"Exactly 10 SIMPLE iterations; zero momentum residual audits.",
"Pressure backend unchanged from H3/H2B-r1.",
"FGS1 = forward every outer. ALTGS1 = odd forward, even backward.",
""
]
for r in rows:
    lines.append(
      f"{r['tag']}: cells={r['cells']:,} mode={r['mode']} SIMPLE={r['simple']:.3f} ms "
      f"({r['miups']:.3f} MIUPS) assembly={r['assembly']:.3f} momentum={r['momentum']:.3f} "
      f"continuity={r['continuity']:.3f} pressure={r['pressure']:.3f} pUpdate={r['pupdate']:.3f} "
      f"nonSolverSum={r['nonsolver']:.3f} ms pIts={r['pIts']:.3f} relCont10={r['relCont']:.3e} "
      f"explicit={r['explicit']:.1f} MiB"
    )
    if r["mode"]=="fgs1" and r["cells"] in h3:
        b=h3[r["cells"]]
        lines.append(
          f"  vs H3 FGS1 old diffusion reassembly: SIMPLE {b['simple']:.3f}->{r['simple']:.3f} ms "
          f"({b['simple']/r['simple']:.3f}x); momentum {b['momentum']:.3f}->{r['momentum']:.3f}; "
          f"pressure {b['pressure']:.3f}->{r['pressure']:.3f}; pIts {b['pIts']:.3f}->{r['pIts']:.3f}; "
          f"relCont10 {b['relCont']:.3e}->{r['relCont']:.3e}"
        )

lines += ["","ALTGS1 vs FGS1 with corrected assembly","--------------------------------------"]
for n in sorted(set(r["cells"] for r in rows)):
    d={r["mode"]:r for r in rows if r["cells"]==n}
    if "fgs1" in d and "altgs1" in d:
        f,a=d["fgs1"],d["altgs1"]
        lines.append(
          f"{n:,} cells: FGS {f['simple']:.3f} ms/{f['miups']:.3f} MIUPS -> "
          f"ALT {a['simple']:.3f} ms/{a['miups']:.3f} MIUPS; "
          f"ALT/FGS time={a['simple']/f['simple']:.4f}; pIts {f['pIts']:.3f}->{a['pIts']:.3f}; "
          f"relCont10 {f['relCont']:.3e}->{a['relCont']:.3e}"
        )

lines += [
"",
"Architecture",
"------------",
"Laminar diffusion numeric CSR is built once and uploaded once; each outer restores it device-to-device.",
"Only convection numeric coefficients and fixed-Dirichlet convection RHS are evaluated each outer.",
"The same scalar momentum matrix is shared by all three velocity components.",
"For variable/turbulent viscosity the diffusion numeric CSR is the refreshable object; topology/geometry remain persistent."
]
summary=root/"h4_summary.txt"
summary.write_text("\n".join(lines)+"\n")
print(summary.read_text(),end="")
print(f"SUMMARY={summary}")
