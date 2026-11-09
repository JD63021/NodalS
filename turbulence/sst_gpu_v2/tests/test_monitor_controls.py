"""Self-contained non-CUDA smoke checks for November 9 SST monitoring controls.

These tests do not require U-bend meshes, external inlet profiles, or a GPU.
"""
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / 'examples' / 'shell2_360k_monitored_20261109.case'
RUNNER = ROOT / 'nodals.py'
MONITOR = ROOT / 'tools' / 'nodals_live_monitor.py'

class TestMonitorControls(unittest.TestCase):
    def call_case(self, *settings):
        with tempfile.TemporaryDirectory() as temp:
            # A file which exists is sufficient to test dry-run case parsing.
            # The test never invokes the actual mesh conversion or native solver.
            mesh = Path(temp) / 'placeholder.msh'
            mesh.write_text('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n')
            args = [sys.executable, str(RUNNER), 'run', '--case', str(CASE),
                    '--mesh', str(mesh), '--dry-run']
            for setting in settings:
                args += ['--set', setting]
            return subprocess.run(args, cwd=ROOT, capture_output=True, text=True)

    def test_monitor_parser(self):
        p = subprocess.run([sys.executable, str(MONITOR), '--self-test'],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout+p.stderr)
        self.assertIn('NODALS_MONITOR_SELFTEST=PASS', p.stdout)

    def test_legacy_case_dry_run(self):
        p = self.call_case()
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('NODALS_SOFTWARE_MODE=m3-fgs scalar=q1', p.stdout)

    def test_compact_q1_and_combined_metric_options(self):
        p = self.call_case('sst.compact_p1=true', 'convergence.metric=both',
                           'output.metrics_every=1')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('--sst-g2-metrics-every 1', p.stdout)

    def test_invalid_metric_cadence_rejected(self):
        p = self.call_case('convergence.metric=wall_traction')
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('metrics_every=1', p.stderr)

    def test_pressure_only_metric(self):
        p = self.call_case('convergence.metric=pressure_drop',
                           'output.metrics_every=1')
        self.assertEqual(p.returncode, 0, p.stderr)

if __name__ == '__main__':
    unittest.main()
