# NodalS pure-TET PROFILE5+EQ5 short-gate development port (2026-10-15)

This is an experimental first port, not a validated turbulent pipe computation.
The accepted HEX solver has not been modified. The pure-tet solver is a sibling source copy.

What was changed:
- Host HXT1 converter supports nhex=0 and has zero-count-safe audits.
- Pure-tet external triangular faces are reconstructed as TET walls, excluding inlet/outlet.
- Tet wall quadrature uses the source's 12-point triangular rule.
- Host setup locates five fixed PHYSICAL inward-normal sample heights by tetra adjacency walk;
  NOT_FOUND is fatal. The cache stores 8 P1+BF3 basis values + global DOFs.
- Reuses EXACT SAME five-point Spalding fit objective/root algorithm as HEX.
- G0 and SST G2 wall momentum uses fitted u_tau and local first-tet normal-column
  Gauss4 Picard trial average. The test function is exact P1+BF3 wall trace.
- TET EQ5 scalar penalty copies HEX five-point weighted least squares,
  with tet local 8x8 scalar CSR, triangular area rule, identical SST lookup.
- Original all-HEX branches are untouched in this copy.

Important numerical distinction:
- HEX run may select SST_STATE_SPACE=bernstein_q2, but existing TET volume kernels
  evaluate P1+BF3 (8 scalar modes). This is NOT a matched polynomial-space test.
- Local momentum Picard functional necessarily changes geometry: HEX normal-column
  average is not geometrically identical to TET first-element normal-column average.
- Pure-tet implementation is restricted to SST G2 PROFILE5+EQ5; other wall modes fail visibly.

Test statuses:
- Python source syntax: PASS.
- Small CPU HXT1 smoke test (5649 tet, 0 hex): PASS.
- All five sampling heights x twelve face quadrature points x 432 wall faces inside mesh (independent geometric smoke): PASS.
- NVCC compile: NOT TESTED (NVCC/CUDA toolchain unavailable in authoring environment).
- GPU cache execution & five nonlinear iterations: NOT TESTED.
- Full turbulent steady 10D comparison: NOT TESTED.

Important build dependency discovered after unpacking:
- The supplied user TAR omits `gpu/serial_cuda/g8_cf_hierarchy_host.hpp` and its parent
  support headers. They are required by `hybrid_amg.cuh`. No AMG has been recreated.
- `setup_gpu_deps.sh` creates missing read-only symlinks to the ORIGINAL source
  directory (`$HOME/NodalS_SST_DEVIATORIC_Y84/gpu/serial_cuda` by default).
- If that original checkout is elsewhere, run
  `SERIAL_CUDA_ROOT=/path/to/gpu/serial_cuda ./setup_gpu_deps.sh` before building.
- The port cannot compile from this TAR alone if those original headers are absent.

Prerequisites on Ubuntu GPU machine:
- NVIDIA CUDA nvcc, GPU supports ARCH=sm_86 default.
- Python numpy and gmsh package (pip install gmsh, or system Python with gmsh bindings).
- Python scipy only if running optional CPU host smoke test.

Recommended commands from directory containing this README:

    python3 tools/smoke_host_hxt1.py  # optional CPU smoke
    python3 tools/make_puretet_10d.py --outdir mesh_puretet_10D --edge 0.00035 --algorithm 1
    cd puretet_profile5_eq5_hxt5b
    ./RUN_PURETET_10D_BUILD_ONLY.sh
    ./RUN_PURETET_10D_5ITER_FP64.sh

First assess: PURETET_WALL_CLASSIFY; PROFILE5_TET_CACHE (all 5 heights);
PROFILE5_TET_PICARD_CACHE; NODALS_SST_PROFILE5_EQ5_TET_CONFIG;
then NODALS_SST_G2 and momentum/pressure residuals. A short run need not converge.
Do not call it a PASS before it compiles and actually runs on the GPU.

Runner defaults:
- Geometry D=0.004m L=0.04m, bulk=50m/s, Re=13691.7402481
- FIVE physical y distances from existing matched 10D Y10 runner:
  7.347338250770073e-05, 1.163328556371928e-04,
  1.591923287666849e-04, 2.020518018961770e-04,
  2.449112750256691e-04 metres.
- FP64 / SUPG=0 / 5 SST G2 iterations (plug start) / gamma=500.
- 10D is developing turbulent flow: total deltaP is not a Moody validation.

To change mesh on rerun, set Y10_10D_MESH=/path/to/HXT1_mesh.bin.
To change iterations, set SST_G2_MAX=10 MAX_OUTER=10.
To change results path, set OUT=/path/to/a/new_nonexistent_folder.
