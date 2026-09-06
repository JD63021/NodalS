#!/usr/bin/env python3
from pathlib import Path
import re,sys

if len(sys.argv)!=2:
    raise SystemExit("usage: ANALYZE_H8.py <case.log>")
txt=Path(sys.argv[1]).read_text(errors="replace")

def line(prefix):
    x=next((z for z in txt.splitlines() if z.startswith(prefix)),None)
    if not x: raise SystemExit(f"ERROR missing {prefix}")
    return x

def kv(s,key,cast=float):
    m=re.search(r"(?:^| )"+re.escape(key)+r"=([^ ]+)",s)
    if not m: raise SystemExit(f"ERROR missing {key}")
    return cast(m.group(1))

bc=line("NODALS_GPU_H8_BCOEFF ")
rf=line("NODALS_GPU_H8_REFRESH_SELECT ")
bt=line("NODALS_GPU_H8_BT_SELECT ")
ct=line("NODALS_GPU_H8_CONTINUITY_SELECT ")
po=line("NODALS_GPU_H8_B_POLICY ")
fx=line("NODALS_GPU_H8_FIXED10 ")

cells=kv(fx,"cells",int)
simple=kv(fx,"avgSimpleMs")
assembly=kv(fx,"assemblyAvgMs")
momentum=kv(fx,"momentumAvgMs")
continuity=kv(fx,"continuityAvgMs")
pressure=kv(fx,"pressureAvgMs")
pits=kv(fx,"avgPressureIts")
rel=kv(fx,"finalRelCont")
miups=cells/simple/1000.0

h7=dict(simple=70.709,miups=10.869,assembly=6.236,momentum=22.042,pressure=39.666,pits=2.900,rel=2.786)

print("NodalS H8 — persistent precomputed B geometry")
print("="*58)
print("768k, FGS1, exactly 10 SIMPLE; H6 + H7 retained.")
print()
print(f"B table: {kv(bc,'valuesPerCell',int)} FP32 values/cell, "
      f"{kv(bc,'bytesPerCell',int)} B/cell, {kv(bc,'totalMiB'):.3f} MiB")
print()
print("Setup selections:")
print(f"  fine CSR refresh: {kv(rf,'selected',str)}; "
      f"{kv(rf,'dynamicMs'):.6f}->{kv(rf,'precomputedMs'):.6f} ms "
      f"(precompute speedup={kv(rf,'speedup'):.3f}x), "
      f"parity={kv(rf,'valParityRelL2'):.2e}")
print(f"  B^T p kernel: {kv(bt,'selected',str)}; "
      f"{kv(bt,'dynamicKernelMs'):.6f}->{kv(bt,'precomputedKernelMs'):.6f} ms "
      f"(precompute speedup={kv(bt,'speedup'):.3f}x), "
      f"parity={kv(bt,'parityRelL2'):.2e}")
print(f"  continuity B u: {kv(ct,'selected',str)}; "
      f"{kv(ct,'dynamicMs'):.6f}->{kv(ct,'precomputedMs'):.6f} ms "
      f"(precompute speedup={kv(ct,'speedup'):.3f}x), "
      f"parity={kv(ct,'parityRelL2'):.2e}")
print()
print("Integrated fixed-10:")
print(f"  SIMPLE={simple:.3f} ms = {miups:.3f} MIUPS")
print(f"  assembly={assembly:.3f} ms")
print(f"  momentum={momentum:.3f} ms")
print(f"  continuity={continuity:.3f} ms")
print(f"  pressure={pressure:.3f} ms")
print(f"  avg PCG its={pits:.3f}, relCont10={rel:.3e}")
print()
print("vs accepted H7:")
print(f"  SIMPLE: {h7['simple']:.3f}->{simple:.3f} ms ({h7['simple']/simple:.3f}x)")
print(f"  MIUPS : {h7['miups']:.3f}->{miups:.3f}")
print(f"  pressure: {h7['pressure']:.3f}->{pressure:.3f} ms "
      f"({h7['pressure']/pressure:.3f}x)")
print(f"  pIts: {h7['pits']:.3f}->{pits:.3f}")
print(f"  relCont10: {h7['rel']:.3e}->{rel:.3e}")
print()
if simple < h7["simple"]:
    print("DECISION=KEEP_H8_PRECOMPUTED_B")
else:
    print("DECISION=KEEP_H7_DYNAMIC_B")
