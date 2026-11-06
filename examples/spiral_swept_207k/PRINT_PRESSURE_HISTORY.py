#!/usr/bin/env python3
"""Print kinematic and physical pressure drops in all completed stages."""
from pathlib import Path
import re,argparse
ap=argparse.ArgumentParser()
ap.add_argument('--run',type=Path,default=Path.home()/'Downloads/NodalS_SST_SOFTWARE_V2/runs/SPIRAL_SWEPT_207K_YPLUS30_20SWEEPS')
ap.add_argument('--rho',type=float,default=1000.)
a=ap.parse_args()
rows=[]
for file in sorted(a.run.glob('stage_*/terminal.log')):
    text=file.read_text(errors='replace')
    for line in text.splitlines():
        if 'pressureDrop=' not in line:continue
        m=re.search(r'\b(?:it|iterations)=(\d+)\b',line)
        p=re.search(r'\bpressureDrop=([-+\d.eE]+)',line)
        if not p or not m:continue
        iteration=int(m.group(1)); dp=float(p.group(1));
        rows.append((iteration,dp,file.parent.name,'FINAL' if 'G2_FINAL' in line else 'PHYS'))
if not rows:
    print('No pressureDrop values found yet. Need metrics or final checkpoint logs.')
else:
    latest={}
    for row in rows:
        k=row[0]
        if k not in latest or row[3]=='FINAL':latest[k]=row
    print('iteration   NodalS_kinematic   physical_Pa   bar     stage')
    for it,dp,stage,typ in sorted(latest.values())[-40:]:
        print(f'{it:8d}   {dp:16.6f}   {dp*a.rho:11.1f}   {dp*a.rho/1e5:7.3f}   {stage} ({typ})')
    print('NOTE: unsteady or unconverged pressures are diagnostic, not validated predictions.')
