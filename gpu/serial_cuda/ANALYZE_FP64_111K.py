#!/usr/bin/env python3
from pathlib import Path
import re,sys
if len(sys.argv)!=4:
    raise SystemExit("usage: ANALYZE_FP64_111K.py <log> <gpu_mem> <time>")
log=Path(sys.argv[1]).read_text(errors="replace").splitlines()

def kv(line,key,cast=float):
    m=re.search(r"(?:^| )"+re.escape(key)+r"=([^ ]+)",line)
    if not m:return None
    try:return cast(m.group(1))
    except:return None

cfg=next(x for x in log if x.startswith("NODALS_GPU_H8_CONFIG "))
full=next(x for x in log if x.startswith("NODALS_GPU_H8_FULLCONV "))
mem=next(x for x in log if x.startswith("NODALS_GPU_H8_MEMORY ") and "point=after_upload" in x)
res=next(x for x in log if x.startswith("NODALS_GPU_RESULT gate=H8 "))

if kv(cfg,"precision",str)!="fp64": raise SystemExit("ERROR not FP64")
if kv(res,"status",str)!="PASS": raise SystemExit("ERROR result not PASS")

gpu={}
for z in Path(sys.argv[2]).read_text(errors="replace").splitlines():
    if "=" in z:
        a,b=z.split("=",1)
        try:gpu[a]=float(b)
        except:pass

hosthwm=float("nan")
t=Path(sys.argv[3]).read_text(errors="replace")
m=re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)",t)
if m:hosthwm=int(m.group(1))/1024.0

print("NodalS cumulative H8 full-FP64 — 111k convergence gate")
print("="*72)
print(f"precision={kv(cfg,'precision',str)}")
print(f"outer iterations={kv(full,'outerIts',int)}")
print(f"final relCont={kv(full,'finalRelCont'):.6e}")
print(f"final momentum audit=[{kv(full,'finalAuditInitRel',str)}]")
print(f"avg pressure its={kv(full,'avgPressureIts'):.3f}")
print(f"avg SIMPLE={kv(full,'avgSimpleMs'):.3f} ms")
cells=kv(full,"cells",int); ms=kv(full,"avgSimpleMs")
print(f"throughput={cells/ms/1000.0:.3f} MIUPS")
print(f"pressure drop={kv(full,'pressureDropFit'):.9f}")
print(f"pressure-drop rel error={kv(full,'pressureDropRelErr'):.6e}")
print(f"explicit VRAM={kv(mem,'explicitMiB'):.1f} MiB")
print(f"explicit bytes/cell={kv(mem,'explicitBytesPerCell'):.1f}")
print(f"sampled GPU peak delta={gpu.get('peakDeltaMiB',float('nan')):.1f} MiB")
print(f"host HWM={hosthwm:.1f} MiB")
print("DECISION=FP64_111K_CONVERGENCE_PASS")
