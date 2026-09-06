#!/usr/bin/env python3
from pathlib import Path
import re,sys

if len(sys.argv)!=2:
    raise SystemExit("usage: ANALYZE_H8_FP64.py <results_dir>")

root=Path(sys.argv[1])
order=["768k","1.1M","2M"]

# Accepted/measured H8 FP32 scaling reference from the immediately preceding gate.
fp32={
 "768k":{"cells":768530,"simple":59.699942,"miups":768530/59.699942/1000.0,"explicit":1288.975},
 "1.1M":{"cells":1143041,"simple":89.533596,"miups":1143041/89.533596/1000.0,"explicit":1932.058},
 "2M":{"cells":2104005,"simple":173.675152,"miups":2104005/173.675152/1000.0,"explicit":3592.318},
}

def kv(line,key,cast=float):
    m=re.search(r"(?:^| )"+re.escape(key)+r"=([^ ]+)",line)
    if not m:return None
    try:return cast(m.group(1))
    except:return None

rows=[]
for tag in order:
    p=root/f"{tag}.log"
    if not p.exists(): continue
    lines=p.read_text(errors="replace").splitlines()

    fixed=next((x for x in lines if x.startswith("NODALS_GPU_H8_FIXED10 ")),None)
    mem=next((x for x in lines if x.startswith("NODALS_GPU_H8_MEMORY ") and "point=after_upload" in x),None)
    btab=next((x for x in lines if x.startswith("NODALS_GPU_H8_BCOEFF ")),None)
    result=next((x for x in lines if x.startswith("NODALS_GPU_RESULT gate=H8 ")),None)

    if not fixed or not mem or not result:
        raise SystemExit(f"ERROR missing H8 FP64 markers for {tag}")
    if "precision=fp64" not in result:
        raise SystemExit(f"ERROR {tag} was not FP64")

    cells=kv(fixed,"cells",int)
    simple=kv(fixed,"avgSimpleMs")
    r={
      "tag":tag,
      "cells":cells,
      "simple":simple,
      "miups":cells/simple/1000.0,
      "assembly":kv(fixed,"assemblyAvgMs"),
      "momentum":kv(fixed,"momentumAvgMs"),
      "continuity":kv(fixed,"continuityAvgMs"),
      "pressure":kv(fixed,"pressureAvgMs"),
      "pits":kv(fixed,"avgPressureIts"),
      "rel":kv(fixed,"finalRelCont"),
      "explicit":kv(mem,"explicitMiB"),
      "bpc":kv(mem,"explicitBytesPerCell"),
    }
    if btab:
        r["btableMiB"]=kv(btab,"totalMiB")
        r["btableBpc"]=kv(btab,"bytesPerCell",int)

    tp=root/f"{tag}.time"
    if tp.exists():
        txt=tp.read_text(errors="replace")
        m=re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)",txt)
        if m:r["hostHwmMiB"]=int(m.group(1))/1024.0

    gp=root/f"{tag}.gpu_mem"
    if gp.exists():
        d={}
        for z in gp.read_text(errors="replace").splitlines():
            if "=" in z:
                a,b=z.split("=",1)
                try:d[a]=float(b)
                except:pass
        r["gpuPeakDelta"]=d.get("peakDeltaMiB")
        r["gpuPeak"]=d.get("peakMiB")
    rows.append(r)

print("NodalS cumulative H8 — full FP64 fixed-10 audit")
print("="*86)
print("StateReal=FP64, OperatorReal=FP64, AMGReal=FP64, reductions=FP64.")
print("H6 coarse SpMV + H7 assembly + H8 precomputed-B retained.")
print()
print(f"{'case':>6} {'cells':>10} {'ms/SIMPLE':>11} {'MIUPS':>8} {'explicitMiB':>12} {'B/cell':>9} {'GPUpeakΔ':>10}")
for r in rows:
    print(f"{r['tag']:>6} {r['cells']:10,d} {r['simple']:11.3f} {r['miups']:8.3f} "
          f"{r['explicit']:12.1f} {r['bpc']:9.1f} {r.get('gpuPeakDelta',float('nan')):10.1f}")

print()
print("FP64 solve-stage breakdown [ms/SIMPLE]:")
print(f"{'case':>6} {'assembly':>10} {'momentum':>10} {'continuity':>11} {'pressure':>10} {'pIts':>7} {'relCont10':>12}")
for r in rows:
    print(f"{r['tag']:>6} {r['assembly']:10.3f} {r['momentum']:10.3f} "
          f"{r['continuity']:11.3f} {r['pressure']:10.3f} {r['pits']:7.3f} {r['rel']:12.3e}")

print()
print("FP64 versus measured H8 FP32:")
print(f"{'case':>6} {'FP32 MIUPS':>11} {'FP64 MIUPS':>11} {'speed ratio':>12} {'FP32 MiB':>11} {'FP64 MiB':>11} {'mem ratio':>10}")
for r in rows:
    q=fp32[r["tag"]]
    print(f"{r['tag']:>6} {q['miups']:11.3f} {r['miups']:11.3f} "
          f"{r['miups']/q['miups']:12.3f} {q['explicit']:11.1f} {r['explicit']:11.1f} "
          f"{r['explicit']/q['explicit']:10.3f}")

print()
print("H8 B-table precision check:")
for r in rows:
    print(f"  {r['tag']}: {r.get('btableBpc',0)} bytes/cell, {r.get('btableMiB',float('nan')):.1f} MiB")
