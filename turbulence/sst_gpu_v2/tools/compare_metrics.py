#!/usr/bin/env python3
"""OPTIONAL numerical parity check between two CUDA log files (same initial state).
Compares common NODALS_SST_G2 iteration metrics; never blocks ordinary simulation.
"""
import argparse
import math
from pathlib import Path
import re
import sys

FIELDS=('absCont','kRelUpdate','omegaRelUpdate','nutMeanRelUpdate','meanNutOverNu',
        'wallDarcy','wallYPlusMean','pressureDrop','fluxImbalanceRel','pIn','pOut')
LINE=re.compile(r'^NODALS_SST_G2 it=(\d+) (.*)$',re.M)
NUMBER=re.compile(r'(\w+)=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)')
MOM=re.compile(r'absMom=\[([^\]]+)\]')

def read(path):
    result={}
    for it,data in LINE.findall(Path(path).read_text(errors='replace')):
        record={k:float(v) for k,v in NUMBER.findall(data) if k in FIELDS}
        mom=MOM.search(data)
        if mom:
            values=[float(s) for s in mom.group(1).split(',')]
            if len(values)==3:record.update(zip(('absMomX','absMomY','absMomZ'),values))
        result[int(it)]=record
    if not result:raise ValueError('No NODALS_SST_G2 iteration records found in '+str(path))
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference',type=Path)
    p.add_argument('candidate',type=Path)
    p.add_argument('--rtol',type=float,default=1e-5)
    p.add_argument('--atol',type=float,default=1e-10)
    p.add_argument('--final-only',action='store_true')
    args=p.parse_args()
    if args.rtol<0 or args.atol<0:raise ValueError('Tolerances must be nonnegative')
    ref=read(args.reference);new=read(args.candidate)
    common=sorted(set(ref)&set(new))
    if args.final_only:common=common[-1:]
    if not common:raise ValueError('No matching absolute iteration numbers; run with the same checkpoint and iteration range')
    fails=[];checks=0;worst=(0,None,None,None,None)
    for it in common:
        for key in set(ref[it])&set(new[it]):
            a,b=ref[it][key],new[it][key]
            threshold=args.atol+args.rtol*max(abs(a),abs(b))
            error=abs(a-b)
            ratio=error/threshold if threshold else float(error>0)
            checks+=1
            if ratio>worst[0]:worst=(ratio,it,key,a,b)
            if (not math.isfinite(a)) or (not math.isfinite(b)) or error>threshold:
                fails.append((it,key,a,b,error,threshold))
    print(f'ITERATIONS_COMPARED={len(common)} METRICS_COMPARED={checks} '
          f'FIRST={common[0]} LAST={common[-1]}')
    print('WORST_NORMALIZED_DIFFERENCE='+repr(worst))
    for row in fails[:30]:print('DIFFERENCE it=%s field=%s reference=%.12e new=%.12e absDiff=%.3e limit=%.3e'%row)
    if fails:
        print('NUMERICAL_PARITY=FAIL differing_metrics='+str(len(fails)))
        return 1
    print('NUMERICAL_PARITY=PASS (tolerance-based; not proof of physical convergence)')
    return 0

if __name__=='__main__':sys.exit(main())
