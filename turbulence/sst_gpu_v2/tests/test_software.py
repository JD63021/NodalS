"""CPU-only tests: exercise the actual case launcher and geometry conversion.
CUDA compilation and numerical regression remain workstation tests.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'nodals.py'
CASE=ROOT/'examples/ubend_200k.case'

class TestSoftware(unittest.TestCase):
    def invoke(self,*rest,env=None):
        return subprocess.run([sys.executable,str(APP),'run','--case',str(CASE),*map(str,rest)],
                              capture_output=True,text=True,env=env)

    def test_baseline_dry_run(self):
        p=self.invoke('--dry-run','--steps','10')
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertIn('NODALS_SOFTWARE_MODE=m3-fgs scalar=q1 maxOuter=10',p.stdout)
        self.assertIn('--simple-coupling corrected',p.stdout)
        self.assertIn('--sst-g2-start plug',p.stdout)
        self.assertIn('--sst-k-fgs-sweeps 8',p.stdout)
        self.assertIn('--sst-omega-fgs-sweeps 8',p.stdout)
        # True native PROFILE5 closure is mandatory for the frozen pure-tet code.
        self.assertIn('--sst-profile-average-closure 1',p.stdout)
        self.assertIn('--sst-profile-average-wall-law spalding_profile5',p.stdout)
        self.assertIn('--sst-profile-fit-y1 0.001632945926233649',p.stdout)
        self.assertIn('--sst-profile-fit-y5 0.008164729631168245',p.stdout)
        self.assertIn('--sst-shared-spalding-wall-state 1',p.stdout)
        self.assertIn('--sst-of-production 0',p.stdout)

    def test_m4_selection_is_opt_in(self):
        p=self.invoke('--dry-run','--momentum-solver','m4-hybrid')
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertIn('NODALS_SOFTWARE_MODE=m4-hybrid',p.stdout)
        self.assertIn('M4 is experimental',p.stderr)

    def test_bad_density(self):
        p=self.invoke('--dry-run','--set','physics.density=1.3')
        self.assertNotEqual(p.returncode,0)
        self.assertIn('density must be 1',p.stderr)

    def test_case_overrides(self):
        p=self.invoke('--dry-run','--set','sst.k_sweeps=9')
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertIn('--sst-k-fgs-sweeps 9',p.stdout)

    def test_gpu_execution_path_without_any_stage_app(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            mesh=tmp/'mesh.bin'; mesh.write_bytes(b'TEST_HXT1')
            fake=tmp/'mock_cuda'
            fake.write_text('#!'+sys.executable+'\nimport json,os,sys\n'
                            'print("FAKE_CUDA_EXECUTION_OK")\n'
                            'with open("mock.json","w") as f:\n'
                            ' json.dump({"argv":sys.argv[1:],"env":{k:v for k,v in os.environ.items() if k.startswith(("SST","STAGE","NODALS"))}},f)\n')
            fake.chmod(0o755)
            out=tmp/'run'
            e=os.environ.copy(); e['STAGE26_INSTALL_HASH']='previous_stale_hash'
            e['NODALS_M3_MODE']='off'
            p=self.invoke('--mesh',mesh,'--binary',fake,'--output',out,'--steps','3',env=e)
            self.assertEqual(p.returncode,0,p.stdout+'\n'+p.stderr)
            self.assertIn('FAKE_CUDA_EXECUTION_OK',(out/'terminal.log').read_text())
            self.assertNotIn('FAKE_CUDA_EXECUTION_OK',p.stdout)
            record=json.loads((out/'mock.json').read_text())
            env=record['env']; argv=record['argv']
            self.assertEqual(env['NODALS_M3_MODE'],'fgs')
            self.assertEqual(env['SST22_TRACTION_MODEL'],'vector_projected')
            self.assertEqual(env['SST22D4D_MOM_MODE'],'coupled')
            self.assertEqual(env['STAGE27_INITIAL_VELOCITY'],'potential')
            self.assertEqual(env['NODALS_F1B_FROZEN_WALL'],'1')
            self.assertEqual(env['NODALS_F2_FACE_SPALDING'],'1')
            self.assertEqual(env['SST_WALL_SAMPLING_MODE'],'ray_trace')
            self.assertEqual(env['SST_WALL_RAY_MAX_CROSSINGS'],'4096')
            self.assertEqual(env['STAGE27_S02B_PACK'],'1')
            self.assertEqual(argv[argv.index('--sst-profile-average-closure')+1],'1')
            self.assertNotIn('STAGE26_INSTALL_HASH',env)
            self.assertEqual(argv[argv.index('--sst-g2-max')+1],'3')
            self.assertTrue((out/'invocation.json').exists())
            self.assertTrue((out/'terminal.log').exists())

    def test_explicit_sample_heights(self):
        sample='1e-4, 2e-4, 3e-4, 4e-4, 5e-4'
        p=self.invoke('--dry-run','--set','wall_function.distances='+sample)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertIn('--sst-profile-fit-y1 0.0001',p.stdout)
        self.assertIn('--sst-profile-fit-y5 0.00050000000000000001',p.stdout)
        p2=self.invoke('--dry-run','--set','wall_function.distances=0.4, 0.3, 0.2, 0.1, 0.05')
        self.assertNotEqual(p2.returncode,0)
        self.assertIn('strictly increasing',p2.stderr)

    def test_restart_steps_are_absolute_native_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            ckpt=tmp/'NODALS_STAGE31_STATE_it200.ckpt'
            ckpt.write_bytes(b'placeholder')
            p=self.invoke('--restart',ckpt,'--steps','10','--dry-run')
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertIn('--sst-g2-max 210',p.stdout)

    def test_generic_mesh_orientation(self):
        # Single tetrahedron: inlet normal is -x and outlet normal is NOT -x,
        # unlike the frozen U-bend converter's former hardcoded restriction.
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            mesh=tmp/'single_tet.msh'
            mesh.write_text('''$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n4\n1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n$EndNodes\n$Elements\n5\n1 2 2 1 1 1 3 4\n2 2 2 2 2 2 3 4\n3 2 2 3 3 1 2 3\n4 2 2 3 3 1 2 4\n5 4 2 4 4 1 2 3 4\n$EndElements\n''')
            from tools.msh2_to_hxt1 import convert
            d=convert(mesh,tmp/'hxt',ROOT/'tools/prepare_hxt1.py')
            self.assertEqual(d['tetrahedra'],1)
            self.assertTrue((tmp/'hxt/HXT1_mesh.bin').is_file())
            self.assertEqual(d['inlet_faces'],1)
            self.assertEqual(d['outlet_faces'],1)


    def test_lean_reporting_case_and_cadences(self):
        p=self.invoke('--dry-run')
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertIn('--sst-g2-print-every 10',p.stdout)
        self.assertIn('--sst-g2-metrics-every 10',p.stdout)
        self.assertIn('--sst-fgs-check-every 10',p.stdout)
        from importlib import util
        spec=util.spec_from_file_location('nodals_runner',APP)
        mod=util.module_from_spec(spec);spec.loader.exec_module(mod)
        self.assertTrue(mod.compact_line('NODALS_SST_RESIDUALS it=10 UxAbs=1\n'))
        self.assertFalse(mod.compact_line('NODALS_SST_G1_DIAG equation=k nonlinearIt=2\n'))
        self.assertFalse(mod.compact_line('STAGE27_AUDIT_SCHEDULE it=2\n'))
        bad=self.invoke('--dry-run','--set','output.metrics_every=7')
        self.assertNotEqual(bad.returncode,0)
        self.assertIn('multiple',bad.stderr)

    def test_case_restart_path_and_native_output_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            ckpt=tmp/'NODALS_STAGE31_STATE_it200.ckpt'
            ckpt.write_bytes(b'placeholder')
            mesh=tmp/'mesh.hxt1';mesh.write_bytes(b'HXT1')
            # A normal case path is enough; no --restart argument is required.
            p=self.invoke('--dry-run','--mesh',mesh,
                          '--set',f'restart.path={ckpt}',
                          '--set','restart.input_fingerprint='+('a'*64),
                          '--set','simple.max_iterations=300')
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertIn('--sst-g2-max 300',p.stdout)
            mock=tmp/'mock_cuda';mock.write_text(
                '#!'+sys.executable+'\nimport os,sys,json\n'
                'print("NODALS_SST_G1_DIAG equation=k nonlinearIt=1")\n'
                'print("NODALS_SST_RESIDUALS it=10 UxAbs=1")\n'
                'print("NODALS_SOFTWARE_COMPLETION=COMPLETE iterations=10 converged=0")\n'
                'with open("props.json","w") as f: json.dump({k:v for k,v in os.environ.items() if k in ["STAGE31_RESTART_FROM","STAGE31_CHECKPOINT_DIR","STAGE31_CHECKPOINT_EVERY","SST_G2_VTU_DIR","SST_G2_VTU_EVERY","NODALS_SST_DIAGNOSTIC_EVERY","NODALS_LEGACY_FINAL_DIAGNOSTICS"]},f)\n'
            );mock.chmod(0o755)
            out=tmp/'run'
            q=self.invoke('--mesh',mesh,'--binary',mock,'--output',out,
                          '--set',f'restart.path={ckpt}',
                          '--set','restart.input_fingerprint='+('a'*64))
            self.assertEqual(q.returncode,0,q.stderr)
            self.assertIn('NODALS_SST_RESIDUALS it=10',q.stdout)
            self.assertNotIn('NODALS_SST_G1_DIAG',q.stdout)
            self.assertIn('NODALS_SST_G1_DIAG',(out/'terminal.log').read_text())
            props=json.loads((out/'props.json').read_text())
            self.assertEqual(props['STAGE31_RESTART_FROM'],str(ckpt))
            self.assertEqual(props['STAGE31_CHECKPOINT_DIR'],str(out))
            self.assertEqual(props['STAGE31_CHECKPOINT_EVERY'],'100')
            self.assertEqual(props['SST_G2_VTU_EVERY'],'100')
            self.assertEqual(props['SST_G2_VTU_DIR'],str(out))
            self.assertEqual(props['NODALS_SST_DIAGNOSTIC_EVERY'],'0')
            self.assertEqual(props['NODALS_LEGACY_FINAL_DIAGNOSTICS'],'0')

if __name__=='__main__':unittest.main()
