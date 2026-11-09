# Tested status at packaging, 2026-11-04

Source provenance: uploaded archive SHA256
`d00f30a89194e2d8eb7203cc1c045b31d6f4bf08746a4fbe101c5ce82ebf5e3f`.
All 185 files under `source/` are byte-identical to the uploaded source tree.

VERIFIED here:
- Host C++ compilation of all six existing `.cpp` modules into static runtime lib.
- Second incremental host `make` issued no compiler commands.
- Python and Bash syntax checks.
- Eight CPU-only integration tests, covering native command generation, M3/M4
  selection, source-case overrides, restart iteration mapping, fake-executable
  invocation without Stage26/Stage22C, and non-U-bend Gmsh conversion, plus explicit Spalding sample-height checks.
- Real 200245-tetrahedron Gmsh mesh converted to valid 30 MiB HXT1 binary;
  writer reports all geometric and DOF audit checks PASS.
- Direct launcher now explicitly supplies PROFILE5+OF_AUTO closure, five geometry-scaled Spalding sample heights, old ray-trace settings, F1B/F2, and OF wall-production controls required by the frozen native code. These settings were verified by source inspection but not by GPU execution.
- Original and generalized Gmsh converters produced byte-identical reference
  HXT1 binary (`d4af8b422553ca69c2bf97a0778fa69f8d5d02ad83e488b2807617683dae3d16`).
- Source files unchanged; SHA256 manifest included.

NOT TESTED here:
- The actual CUDA compile (`nvcc` unavailable in execution environment).
- Running RTX 3060 kernels or CUDA memory pressure.
- Ten native SIMPLE steps, matching Stage31 restart state, numerical/physical
  parity of pressure, mass, wall flux, turbulence and velocity, and performance.
- M4 GPU parity, M4 timing, 200-step secondary vortex comparisons.

This package is an integration candidate. Run `RUN_M3_10.sh` on the CUDA
workstation, examine `terminal.log`, and report any first native exception.
Do not accept new numerical defaults or alter public laminar code until same-
state M3 parity is verified.
