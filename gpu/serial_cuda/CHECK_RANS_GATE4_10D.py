#!/usr/bin/env python3
import math,re,sys
from pathlib import Path

if len(sys.argv) != 2:
    print('usage: CHECK_RANS_GATE4_10D.py LOG', file=sys.stderr)
    sys.exit(2)
text=Path(sys.argv[1]).read_text(errors='replace')

def last_line(prefix):
    lines=[x for x in text.splitlines() if x.startswith(prefix)]
    if not lines:
        raise SystemExit(f'GATE4_CHECK status=FAIL reason=missing_{prefix}')
    return lines[-1]

def val(line,key):
    m=re.search(r'(?:^|\s)'+re.escape(key)+r'=([^\s]+)',line)
    if not m:
        raise SystemExit(f'GATE4_CHECK status=FAIL reason=missing_{key}')
    return m.group(1)

full=last_line('NODALS_GPU_H8_FULLCONV ')
wall=last_line('NODALS_GPU_RANS_GATE3_WALL ')
res=last_line('NODALS_GPU_RESULT ')
resid=last_line('NODALS_GPU_H8_RESIDENCY ')
mix=last_line('NODALS_GPU_RANS_GATE1_MIXLEN ')

dp=float(val(full,'pressureDropFit'))
fwall=float(val(wall,'fWallSpalding'))
outer=int(val(full,'outerIts'))
avgp=float(val(full,'avgPressureIts'))
cont=float(val(full,'finalRelCont'))
conv=int(val(full,'converged'))
root=float(val(wall,'rootFailures'))
tan=float(val(wall,'tangentFailures'))

# Validated CPU 10D regression bounds from
# turbulence/tet_p1bf3_rans/reference/vmfl003_10d_effectiveB_regression.json
cpu_dp_unsm=418.503750127
cpu_dp_sm=418.510816661
cpu_f_unsm=0.0288076391
cpu_f_sm=0.0287982109
cpu_dp=0.5*(cpu_dp_unsm+cpu_dp_sm)
cpu_f=0.5*(cpu_f_unsm+cpu_f_sm)
physics_tol=0.002

dp_rel=abs(dp-cpu_dp)/cpu_dp
fw_rel=abs(fwall-cpu_f)/cpu_f

# VMFL003 geometry for this 10D mesh: R=0.002, D=0.004, L=10D=0.04,
# rho=1 in the incompressible formulation, Ubulk=50.
D=0.004; L=0.04; U=50.0; rho=1.0
f_dp=2.0*dp*D/(rho*L*U*U)
f_benchmark=0.0284003
f_dp_err=(f_dp/f_benchmark-1.0)
f_wall_err=(fwall/f_benchmark-1.0)

res_pass=val(res,'status')=='PASS'
resid_ok=('O_N_H2D_inside_SIMPLE=0' in resid and 'SCALAR_CONVERGENCE_AUDITS_ONLY' in resid)
wall_ok=(root==0.0 and tan==0.0)
physics_ok=(dp_rel <= physics_tol and fw_rel <= physics_tol)
ok=(conv==1 and res_pass and resid_ok and wall_ok and physics_ok and math.isfinite(dp) and math.isfinite(fwall))

print(f'NODALS_GPU_RANS_GATE4_CPU_PARITY outerIts={outer} finalRelCont={cont:.12e} avgPressureIts={avgp:.6f} '
      f'pressureDropFit={dp:.12e} cpuPressureDropRef={cpu_dp:.12e} pressureDropVsCpuRel={dp_rel:.12e} '
      f'wallFSpalding={fwall:.10f} cpuWallFRef={cpu_f:.10f} wallFVsCpuRel={fw_rel:.12e} '
      f'physicsTolerance={physics_tol:.3e} status={"PASS" if physics_ok else "FAIL"}')
print(f'NODALS_GPU_RANS_GATE4_FRICTION_FROM_DP fDarcyFromFull10DDrop={f_dp:.10f} '
      f'fullyDevelopedBenchmarkF={f_benchmark:.10f} relativeError={f_dp_err:.12e} percentError={100*f_dp_err:.6f} '
      f'wallFSpalding={fwall:.10f} wallVsBenchmarkPercentError={100*f_wall_err:.6f} '
      f'note=FULL_10D_DROP_INCLUDES_DEVELOPMENT_AND_TET_CONVECTION_EFFECT')
print(f'NODALS_GPU_RANS_GATE4_CHECK converged={conv} resultPass={int(res_pass)} residencyPass={int(resid_ok)} '
      f'wallFailures={int(root+tan)} physicsParity={int(physics_ok)} status={"PASS" if ok else "FAIL"}')
sys.exit(0 if ok else 1)
