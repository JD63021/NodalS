#!/usr/bin/env python3
from pathlib import Path
import re,sys,math

def read(path):
    text=Path(path).read_text()
    lines=[x for x in text.splitlines() if x.startswith("NODALS_HXT4A_RESULT ")]
    if not lines: raise SystemExit(f"missing NODALS_HXT4A_RESULT in {path}")
    line=lines[-1]
    d={}
    for k,v in re.findall(r'(\w+)=([^\s]+)',line):
        d[k]=v
    if d.get("status")!="PASS": raise SystemExit(f"precision run did not PASS: {line}")
    return d,line

a,la=read(sys.argv[1]);b,lb=read(sys.argv[2])
if a["precision"]!="fp64": a,b=b,a;la,lb=lb,la
def f(d,k): return float(d[k])
def rel(x,y): return abs(x-y)/max(abs(x),abs(y),1e-300)

checks={
 "A_action_norm_rel": rel(f(a,"Anorm"),f(b,"Anorm")),
 "B_action_norm_rel": rel(f(a,"Bnorm"),f(b,"Bnorm")),
 "S_action_norm_rel": rel(f(a,"Snorm"),f(b,"Snorm")),
 "pressure_slope_rel": rel(f(a,"slope"),f(b,"slope")),
 "pressure_delta_rel": rel(f(a,"delta"),f(b,"delta")),
}
limits={
 "A_action_norm_rel":5e-4,
 "B_action_norm_rel":2e-4,
 "S_action_norm_rel":8e-4,
 "pressure_slope_rel":2e-3,
 "pressure_delta_rel":2e-3,
}
ok=True
for k,v in checks.items():
    good=v<=limits[k];ok &= good
    print(f"NODALS_HXT4A_PARITY metric={k} value={v:.12e} limit={limits[k]:.12e} status={'PASS' if good else 'FAIL'}")
p_gap=abs(f(a,"pressureRelL2")-f(b,"pressureRelL2"))
good=p_gap<=5e-3;ok &= good
print(f"NODALS_HXT4A_PARITY metric=pressureRelL2_absGap value={p_gap:.12e} limit={5e-3:.12e} status={'PASS' if good else 'FAIL'}")
for d in (a,b):
    good=f(d,"finalRelCont")<=1.2e-3;ok &= good
    print(f"NODALS_HXT4A_PARITY metric={d['precision']}_continuity value={f(d,'finalRelCont'):.12e} limit={1.2e-3:.12e} status={'PASS' if good else 'FAIL'}")
print(f"HXT4A_PARITY_STATUS={'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 3)
