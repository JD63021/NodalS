# GPU SST Re=1M: 99,216-cell hybrid experiment (2026-11-14)

**Status:** experimental, Python preflighted only. No NVCC compilation or CFD iterations have been verified on the workstation.

This branch deliberately contains an opt-in **small source patch** and an editable case, instead of replacing the larger local workstation source. Its precise baseline is the user's November 14 local SST tarball, which may differ materially from GitHub main.

### Mesh

The separate standalone bundle contains the generated geometry:

- 3,744 Q1+BF2 HEX wall cells (one radial ring)
- 95,472 P1+BF3 tetrahedra in the core
- 99,216 total cells; 3,744 interface quads, each matching two tet triangles
- D=0.889 m, L=44.45 m, Re=1e6; wall layer depth ~13.98065 mm; nominal midpoint y+=300
- HXT1 checks report all PASS: HEX/TET positive geometry, BF2 basis, correct matching face topology

This is *not* the user's original .msh. Binary `HXT1_mesh.bin` is supplied only in the user-downloadable experiment archive, not GitHub.

### Deliberately restricted physics

`legacy-segregated` keeps the previously available hybrid axial momentum / HEX Spalding wall; conservative HEX/TET interface and SST transport pieces are reused. For this first experiment momentum is **advective, vector-Laplacian and no SUPG**. We do not enable pure-tet M3, D4, generalized-Robin or adaptive centroid rays on a mixed mesh.

For omega the native `of_auto` HEX wall kernel is modified to obey the Stage41 `sample_log` switch, selecting the pure log target at the HEX midpoint. The same single-point Spalding friction velocity is used; OF production remains OFF.

A prior Re1M case used `trace_log`, deviatoric stress and SUPG. **Do not call this a numerical parity run**; these differing choices are explicit.

### How to run

Unpack the separately downloadable `NODALS_HYBRID_RE1M_99K_20261114.tar.gz`, which includes `mesh/HXT1_mesh.bin`, `cases/...`, `enable_hybrid_local.py` and `RUN.sh`. From that folder:

```bash
APP="$HOME/NodalS_GPU_GENROBIN_ADAPTIVE/turbulence/sst_gpu_v2"
NODALS_APP="$APP" DRY_RUN_ONLY=1 bash ./RUN.sh
NODALS_APP="$APP" STEPS=5 bash ./RUN.sh
```

Prefer running against an isolated copy/worktree of the November 14 app, not changing an active successful checkout. There is one unavoidable NVCC rebuild because a CUDA header changes. Subsequent mesh/case/runner changes require no further CUDA compile.

### Validation boundaries

The source patch has been tested for repeatable application to the **actual November 14 uploaded tar**, and the Python runner dry-run works. **CUDA compilation, running the mixed SST solver, first iteration and conservation, and final Moody friction comparison have NOT been tested**. Stop on native errors and inspect the CUDA logs before continuing.
