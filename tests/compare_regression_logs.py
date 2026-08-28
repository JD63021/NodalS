#!/usr/bin/env python3
from pathlib import Path
import math, re, sys
if len(sys.argv) != 3:
    raise SystemExit('usage: compare_regression_logs.py reference.log modular.log')
refp, newp = map(Path, sys.argv[1:])
TAGS = [
    'P1BF3_PIPE_GEOMETRY', 'P1BF3_PIPE_BC', 'P1BF3_GATE9I_SUMMARY',
    'P1BF3_PIPE_ERROR', 'P1BF3_PIPE_FLOW', 'P1BF3_PIPE_PRESSURE',
    'P1BF3_NS_FINAL', 'P1BF3_RESULT'
]
IGNORE_KEY_PARTS = ('second','time','epoch','rss','hwm','mib','bytes')
num_re = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$')
vec_re = re.compile(r'^\[([^]]+)\]$')

def last_records(path):
    records = {}
    for line in path.read_text(errors='replace').splitlines():
        for tag in TAGS:
            if line.startswith(tag):
                records[tag] = line
    return records

def kv(line):
    d={}
    for token in line.split()[1:]:
        if '=' in token:
            k,v=token.split('=',1); d[k]=v
    return d

R,N=last_records(refp),last_records(newp)
missing=[t for t in TAGS if t not in R or t not in N]
if missing:
    print('NODALS_REGRESSION_COMPARE status=FAIL missing=' + ','.join(missing)); raise SystemExit(2)
fail=[]
for tag in TAGS:
    if tag == 'P1BF3_PIPE_BC':
        if R[tag] != N[tag]: fail.append(f'{tag}:text')
        continue
    a,b=kv(R[tag]),kv(N[tag])
    for k in sorted(set(a)&set(b)):
        kl=k.lower()
        if any(x in kl for x in IGNORE_KEY_PARTS): continue
        va,vb=a[k],b[k]
        if num_re.match(va) and num_re.match(vb):
            xa,xb=float(va),float(vb)
            if math.isfinite(xa) and math.isfinite(xb):
                tol=1e-12 + 1e-10*max(abs(xa),abs(xb))
                if abs(xa-xb)>tol: fail.append(f'{tag}.{k}:{xa}!={xb}')
        else:
            ma,mb=vec_re.match(va),vec_re.match(vb)
            if ma and mb:
                try:
                    aa=[float(x) for x in ma.group(1).split(',')]; bb=[float(x) for x in mb.group(1).split(',')]
                except ValueError:
                    aa=bb=[]
                if len(aa)!=len(bb) or any(abs(x-y) > 1e-13 + 1e-10*max(abs(x),abs(y)) for x,y in zip(aa,bb)):
                    fail.append(f'{tag}.{k}:{va}!={vb}')
            elif va != vb:
                fail.append(f'{tag}.{k}:{va}!={vb}')
if fail:
    print('NODALS_REGRESSION_COMPARE status=FAIL differences=' + ';'.join(fail[:30]))
    raise SystemExit(3)
print('NODALS_REGRESSION_COMPARE status=PASS stable_records_match=1')
