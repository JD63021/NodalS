#!/usr/bin/env python3
"""CPU-only regression for canonical NodalS autosampling dispatch and scheduling."""
from pathlib import Path
import argparse
import importlib.util
import subprocess
import sys
import tempfile
import textwrap
from unittest import mock

APP=Path(__file__).resolve().parents[1]
REPO=APP.parents[1]
sys.path.insert(0,str(APP))
import nodals
spec=importlib.util.spec_from_file_location('nodals_adaptive_run',APP/'adaptive_run.py')
adaptive=importlib.util.module_from_spec(spec)
spec.loader.exec_module(adaptive)

assert adaptive.schedule(500,True,50,500,1)==([50],[(0,50),(50,500)])
assert adaptive.schedule(500,False,50,500,1)==([],[(0,500)])
assert adaptive.schedule(500,True,50,50,2)==([50,100],[(0,50),(50,100),(100,500)])
assert adaptive.schedule(500,True,50,50,0)==([],[(0,500)])

case=REPO/'examples/ubend_unstructured_200k/UBEND_UNSTRUCTURED_GENROBIN_SKEW_SUPG_ADAPTIVE.case'
assert case.is_file(),case
cp=nodals.read_case(case,[])
assert nodals.flag(cp,'adaptive_sampling','enabled')
assert cp['wall_function']['boundary_form']=='generalized_robin_nitsche'
assert cp['convection']['form']=='skew_symmetric'
assert cp['supg'].getboolean('momentum')
assert cp['inlet.main']['profile_mapping']=='interpolate'

# Dispatch must precede mesh preparation; no local mesh/CUDA required.
args=argparse.Namespace(
    case=case,binary=Path('/tmp/nodals_unavailable_test_binary'),
    restart=None,mesh=None,output=Path('/tmp/ubend_adaptive_test_output'),
    cache=APP/'.cache/meshes',momentum_solver=None,
    native_option=['mom-cap-policy=fail'],set=['adaptive_sampling.enabled=true'],
    outer_iterations=None,steps=500,dry_run=True,prepare_only=False)
captured=[]
def fake_call(command,cwd=None):
    captured.extend(command)
    return 0
with mock.patch.object(nodals.subprocess,'call',side_effect=fake_call):
    assert nodals.run(args)==0
assert Path(captured[captured.index('-u')+1]).name=='adaptive_run.py'
assert captured[captured.index('--binary')+1]=='/tmp/nodals_unavailable_test_binary'
assert captured[captured.index('--final-iterations')+1]=='500'
assert captured[-1]=='--dry-run'
assert 'mom-cap-policy=fail' in captured

# Verify child stage commands suppress dispatcher recursion and preserve
# physics. No changes to the user editable original case.
with tempfile.TemporaryDirectory() as td:
    d=Path(td)
    (d/'mesh.msh').write_text('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n')
    cfg=d/'experiment.case'
    cfg.write_text(textwrap.dedent(f"""
    [mesh]
    path = {d/'mesh.msh'}
    [simple]
    max_iterations = 500
    [wall_function]
    samples = 1
    face_spalding = true
    [omega_wall]
    sample_index = 3
    [wall_sampling]
    mode = ray_trace
    [adaptive_sampling]
    enabled = true
    first_refresh = 50
    refresh_every = 500
    max_refreshes = 1
    target_spalding_yplus = 30
    target_omega_yplus = 30
    min_scale = 0.05
    max_scale = 0.90
    [output]
    checkpoint_every = 50
    vtu_every = 50
    """))
    proc=subprocess.run([sys.executable,str(APP/'adaptive_run.py'),
        '--root',str(APP),'--binary','/tmp/unavailable_binary',
        '--case',str(cfg),'--output',str(d/'result'),
        '--final-iterations','500','--dry-run',
        '--set','adaptive_sampling.enabled=true'],
        capture_output=True,text=True)
    assert proc.returncode==0,(proc.stdout,proc.stderr)
    assert 'refreshAt=[50]' in proc.stdout,proc.stdout
    lines=[v for v in proc.stdout.splitlines() if v.startswith('NODALS_ADAPTIVE_DRY_CMD')]
    assert len(lines)==2,proc.stdout
    assert '--steps 50' in lines[0]
    assert '--steps 450' in lines[1]
    for line in lines:
        assert line.index('adaptive_sampling.enabled=true') < line.index('adaptive_sampling.enabled=false'),line
        assert '--binary /tmp/unavailable_binary' in line,line
        assert 'NODALS_FACE_YPLUS_EXPORT=1' in line,line
    assert 'NODALS_FACE_YPLUS_UTAU_CSV=' in lines[1]
    assert not (d/'result').exists()

print('ADAPTIVE_DISPATCH_SCHEDULE_TEST=PASS: dry run, segment refresh and no recursion')
