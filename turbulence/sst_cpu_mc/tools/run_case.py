#!/usr/bin/env python3
"""Message-1-only CPU SST MPI probe using the *unchanged* GPU .case file.

No CFD steps are executed. Refuses to pretend a probe is a working SST solver.
All sections/keys are preserved in the input file; only stage-1 essentials
are interpreted and forwarded to the MPI probe.
"""
from __future__ import annotations
import argparse
import configparser
import os
from pathlib import Path
import shlex
import subprocess
import sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',required=True,type=Path)
    p.add_argument('--mesh',type=Path,help='stage-1 mesh-path override for probe only')
    p.add_argument('--ranks',type=int,default=16)
    p.add_argument('--mpi-exec',default='mpirun')
    p.add_argument('--probe',action='store_true',help='required: this is NOT a flow simulation')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if not a.probe:
        p.error('Message 1 implements ONLY --probe, not numerical SST or SIMPLE solves')
    if a.ranks<1:p.error('--ranks must be >=1')
    cp=configparser.ConfigParser(interpolation=None,strict=True)
    cp.optionxform=str
    case=a.case.expanduser().resolve(strict=True)
    if not cp.read(case):p.error('could not read .case')
    for section in ('mesh','physics','sst','simple','solver.u','solver.p'):
        if not cp.has_section(section):p.error(f'missing [{section}] in existing .case')
    case_mesh=Path(cp.get('mesh','path')).expanduser()
    if not case_mesh.is_absolute():case_mesh=case.parent/case_mesh
    mesh=(a.mesh.expanduser() if a.mesh else case_mesh).resolve(strict=True)
    if not (mesh.is_file() or (mesh.is_dir() and (mesh/'points').exists())):
        p.error('mesh must be a .msh file or OpenFOAM polyMesh directory')
    nu=cp.getfloat('physics','kinematic_viscosity')
    re=cp.getfloat('physics','reynolds')
    if not (nu>0 and re>0):p.error('invalid physical properties')
    root=Path(__file__).resolve().parents[1]
    exe=root/'build/nodals_sst_cpu_mc_probe'
    cmd=[a.mpi_exec,'-np',str(a.ranks),str(exe),'--case',str(case),
         '--mesh',str(mesh),'--nu',str(nu),'--re',str(re),
         '--expect-ranks',str(a.ranks)]
    print('NODALS_CPU_SST_M1_CASE status=PARSED_ONLY preservedCase='+str(case))
    print('NODALS_CPU_SST_M1_MESH path='+str(mesh)+' source='+('OVERRIDE' if a.mesh else 'CASE'))
    print('NODALS_CPU_SST_M1_COMMAND '+shlex.join(cmd))
    sys.stdout.flush()
    if a.dry_run:return
    if not exe.is_file():p.error('CPU probe not built; run make first')
    env=dict(os.environ)
    env['OMP_NUM_THREADS']='1'
    env['OPENBLAS_NUM_THREADS']='1'
    raise SystemExit(subprocess.run(cmd,env=env,check=False).returncode)
if __name__=='__main__':main()
