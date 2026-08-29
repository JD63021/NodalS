#!/usr/bin/env python3
from pathlib import Path
import math, re, sys

if len(sys.argv) != 2:
    raise SystemExit('usage: analyze_v100_fullfast_memory_regression.py OUTDIR')
root=Path(sys.argv[1])

TARGET={
 '40k':  dict(cells=40620,  outer=606,  rss=2715.3, hwm=2984.3),
 '111k': dict(cells=111183, outer=1025, rss=3251.4, hwm=3993.2),
 '292k': dict(cells=292236, outer=1792, rss=4169.0, hwm=6642.2),
}

def last(txt, pat, cast=float, default=float('nan')):
    ms=list(re.finditer(pat,txt,re.M))
    if not ms: return default
    try: return cast(ms[-1].group(1))
    except Exception: return default

def parse(mesh):
    p=root/f'{mesh}.log'
    if not p.exists(): return {'status':'NOLOG'}
    txt=p.read_text(errors='replace')
    result=list(re.finditer(r'^P1BF3_RESULT status=(\w+).*?outerIts=(\d+)',txt,re.M))
    status=result[-1].group(1) if result else 'UNKNOWN'
    outer=int(result[-1].group(2)) if result else last(txt,r'^P1BF3_WORK outerIts=(\d+)',int,0)
    rss=last(txt,r'^P1BF3_RESOURCE_MARK label=solve_end .*?rssSumMiB=([0-9.eE+-]+)')
    hwm=last(txt,r'^P1BF3_RESOURCE_MARK label=solve_end .*?hwmSumMiB=([0-9.eE+-]+)')
    simple=last(txt,r'^P1BF3_TIMING .*?simpleSolveSeconds=([0-9.eE+-]+)')
    total=last(txt,r'^P1BF3_TIMING .*?totalSeconds=([0-9.eE+-]+)')
    avg=last(txt,r'^P1BF3_WORK .*?avgPCG=([0-9.eE+-]+)')
    markers={
      'dist': bool(re.search(r'^P1BF3_DIST_ACTIVE .*rootOnlyRead=1 .*nonrootGlobalRetained=0',txt,re.M)),
      'm1': bool(re.search(r'^P1BF3_M1_RELEASE_COLGID .*retainedCapacity=0 .*status=PASS',txt,re.M)),
      'dyn': 'P1BF3_DYNPLAN_MEMORY_DETAIL mode=compact' in txt,
      'vertex': 'P1BF3_FULLFAST_PLAN_COMPACT vertexCells=RELEASED' in txt,
    }
    return dict(status=status,outer=outer,rss=rss,hwm=hwm,simple=simple,total=total,avg=avg,markers=markers)

def linfit(xs,ys):
    n=len(xs); sx=sum(xs); sy=sum(ys); sxx=sum(x*x for x in xs); sxy=sum(x*y for x,y in zip(xs,ys))
    b=(n*sxy-sx*sy)/(n*sxx-sx*sx); a=(sy-b*sx)/n
    return a,b

rows={m:parse(m) for m in TARGET}
print('='*108)
print(' NodalS v1.00 GOOD-ORACLE FULLFAST FP64 Richardson/GAMG regression')
print('='*108)
print(f"{'mesh':5s} {'outer':>7s} {'target':>7s} {'RSS MiB':>11s} {'target':>10s} {'HWM MiB':>11s} {'target':>10s} {'SIMPLE s':>10s} {'status':>8s}")
print('-'*108)
for m,t in TARGET.items():
    r=rows[m]
    print(f"{m:5s} {r.get('outer',0):7d} {t['outer']:7d} {r.get('rss',math.nan):11.1f} {t['rss']:10.1f} {r.get('hwm',math.nan):11.1f} {t['hwm']:10.1f} {r.get('simple',math.nan):10.4f} {r.get('status','?'):>8s}")

xs=[TARGET[m]['cells']/1e6 for m in TARGET]
rssg=[rows[m].get('rss',math.nan)/1024 for m in TARGET]
hwmg=[rows[m].get('hwm',math.nan)/1024 for m in TARGET]
valid=all(math.isfinite(x) for x in rssg+hwmg)
if valid:
    ri,rs=linfit(xs,rssg); hi,hs=linfit(xs,hwmg)
    print('\nMemory fits: GiB = intercept + slope * M cells')
    print(f'RSS intercept={ri:.4f} GiB slope={rs:.4f} GiB/Mcell historical=5.5049')
    print(f'HWM intercept={hi:.4f} GiB slope={hs:.4f} GiB/Mcell historical=14.2150')
else:
    rs=hs=math.nan

fail=[]
for m,t in TARGET.items():
    r=rows[m]
    if r.get('status')!='PASS': fail.append(f'{m}:solver_status')
    # Iteration counts should be extremely close for the same source/options, but allow 2% for runtime/version noise.
    if not r.get('outer') or abs(r['outer']-t['outer'])/t['outer']>0.02: fail.append(f'{m}:outer')
    # RSS/HWM are allocator/runtime-sensitive.  8% is tight enough to catch the 9-10 GiB/M regression while not failing on small drift.
    if not math.isfinite(r.get('rss',math.nan)) or abs(r['rss']-t['rss'])/t['rss']>0.08: fail.append(f'{m}:rss')
    if not math.isfinite(r.get('hwm',math.nan)) or abs(r['hwm']-t['hwm'])/t['hwm']>0.08: fail.append(f'{m}:hwm')
    for k,v in r.get('markers',{}).items():
        if not v: fail.append(f'{m}:marker_{k}')
if not math.isfinite(rs) or not (5.0 <= rs <= 6.1): fail.append('rss_slope')
if not math.isfinite(hs) or not (12.5 <= hs <= 16.0): fail.append('hwm_slope')

if fail:
    print('\nNODALS_V100_MEMORY_REGRESSION status=FAIL reasons=' + ','.join(fail))
    raise SystemExit(2)
print('\nNODALS_V100_MEMORY_REGRESSION status=PASS goodOracle=0de1a338 rssSlopeGate=PASS hwmSlopeGate=PASS')
