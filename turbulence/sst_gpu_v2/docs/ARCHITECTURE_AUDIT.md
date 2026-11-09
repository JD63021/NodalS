# NodalS SST architecture audit — initial integration freeze (2026-11-04)

## Scope and state

Frozen numerical source: the actual 2026-11-04 uploaded generalized SST tarball.
All **185 source files in `source/`** (not only the main CUDA TU) were copied
byte-for-byte from the freeze. The four original `.PRE_M4_BACKUP` files remain.
The new root `Makefile`, `nodals.py`, Gmsh generic converter, examples and tests
are *new software-layer files*. CUDA kernel mathematics, wall functions, SIMPLE,
M3, M4, k/omega masked P1, optional P2 remain unchanged.

The initial integration's objective is bypassing installation-only research
restrictions, not removing actual numerical sanity checks.

## Source and compilation dependency map

| Responsibility | Frozen source location | Workstation build behavior |
|---|---|---|
| CUDA entry point | `source/puretet_profile5_eq5_hxt5b/nodals_sst_main.cu` | One CUDA translation unit, rebuilt when any included dependency changes |
| App / SIMPLE orchestra | `modules/app/sst_application.inc`, `modules/coupling/simple_*.inc` | Included in CUDA TU |
| Geometry/meshes | `source/hybrid_hxt1/hxt1_mesh.hpp`, `modules/mesh/`, `modules/geometry/` | Header compiled through CUDA TU |
| Pressure/AMG | `hybrid_amg.cuh`, `modules/coupling/`, `modules/linear/` | Included in CUDA TU |
| Momentum / BF3 | `modules/coupling/g2_momentum_operator.inc`, `modules/wall/stage22d4d_coupled_matrixfree.inc`, `modules/wall/m4_hybrid_impl.inc` | Included in CUDA TU |
| Wall/Nitsche | `modules/wall/`, `modules/wall/portable/` | CUDA TU; separately compiled CPU wall math |
| SST k/omega | `hxt5c_sst_g1.cuh`, `hxt5d_sst_g2.cuh`, `modules/sst/`, `modules/transport/` | Included in CUDA TU |
| Native options | `modules/configuration/native_cli_parse.inc` | Included in CUDA TU |
| Host library | `modules/{physics,wall/portable,transport/portable,coupling/portable,runtime}/*.cpp` | 6 separately compiled C++ objects and static archive |
| Gmsh → HXT1 | `tools/msh2_to_hxt1.py`, `tools/prepare_hxt1.py` | Python/NumPy once per mesh content |
| Case and runtime | root `nodals.py` | Python per run; **no CUDA compilation** |
| Optional parity | `tools/compare_metrics.py`, `tests/test_software.py` | No impact on CUDA runs |

Old Makefile inside the frozen solver directory stays preserved for provenance;
**root Makefile is now the normal build entry**. `make` defaults to FP64 only,
no forced FP32 build. `-MMD` dependency generation tracks headers/includes.

## Mandatory historical launch barriers identified

1. Original **Stage26 Python runner** `runner/run_stage33bc_ubend.py` lines
   ~294–299: required installed `stage26_manifest.json` and compared the
   executable SHA256 with a frozen manifest. NOT present in the new launcher.
2. Original **Stage22C native `.case` installation validator** lived in the
   earlier installed `NodalS_SST_MODULAR` runtime, **not fully in this archive**.
   The KT records that it rejected an altered installed binary SHA256.
   New launcher does not import it or call `stage26_case_bridge.py`.
3. Prior wrappers installed a separate child app and generated frozen stage
   configuration and intermediate manifest metadata. New launcher uses native
   compiled executable directly and copies nothing.
4. Prior runner required a U-bend Gmsh mesh with both inlet/outlet normals
   approximately `(-1,0,0)` and equated inlet cap area to diameter. New
   converter requires only valid tagged exterior faces and geometric validity.
5. Prior driver had hardcoded `--samples=1`, `--omega-sample-index=3`, active
   omega distance and mandatory stage audit flags. New case exposes these
   controls, subject to the **unchanged** native solver's actual allowed modes.
6. Old `Stage31` checkpoint input fingerprints guard restart compatibility.
   These are retained **only for an explicit restart**; an ordinary fresh run
   never requires a checkpoint or installed executable identity.
7. There are still `STAGE*` diagnostic names, physical/numerical validation,
   and experimental P2 constraints **inside the frozen native CUDA code**.
   Those remain untouched. For example, current experimental P2 Stage5 has
   an exact reference-mesh eligibility branch; genuine P2 generalization is
   not asserted. Cleanup of those internals should be a later, independently
   validated source change, not bundled with the software integration.

## Direct native run requirements (verified by source inspection, not GPU test)

Unlike an arbitrary `--mesh` run using native CLI defaults, the frozen M3
operator requires an interlocking set of options formerly supplied by Stage22C:

- `--sst-g2 1 --sst-g2-start plug --simple-coupling corrected`.
- `SST22_TRACTION_MODEL=vector_projected`.
- `SST22D4_NITSCHE_ENABLE=1`, `SST22D4D_MOM_MODE=coupled`,
  `SST22D4I_BF3=1`, `SST22D4I3A_SCHUR=1`.
- `NODALS_M3_MODE=fgs`, `NODALS_M3_SWEEPS=5`, `NODALS_M3_OMEGA=0.7`.
- `SST_STATE_SPACE=q1`, `SST_COMPACT_P1=0`, k/omega fixed FGS8.
- Spalding PROFILE5 with `--sst-profile-average-closure 1`,
  `--sst-profile-average-wall-law spalding_profile5`, five strictly increasing
  `--sst-profile-fit-yN` physical distances (same reference formula as old runner),
  `--sst-shared-spalding-wall-state 1`, and OF_AUTO production selection.
- `SST_WALL_SAMPLING_MODE=ray_trace`, max crossings 4096, barycentric tol 1e-9,
  `STAGE27_S02B_PACK=1`, F1B/F2 cached wall, gamma, independent sampling scales,
  sample index, valid inlet vector and actual geometry diameter.
- `STAGE27_INITIAL_VELOCITY=potential` can be used alongside native G2 `plug`:
  the latter selects the required G2 initialization operator path; the former
  requests the later native potential-flow initialization stage.

New `nodals.py` explicitly sets these options rather than depending on side
effects in a chain of imported stage runners. The command/overrides are written
under the output directory as `invocation.json` for diagnostic transparency.

## Numerical parity plan — **NOT YET COMPLETED**

1. Compile on Ubuntu/CUDA12.2 sm_86; check `build/nodals_sst` exists, no external app.
2. Run fresh 10 SIMPLE iterations using `examples/ubend_200k.case`, check
   momentum/pressure/turbulence phase output, NaN and runtime errors.
3. For **meaningful regression parity**, restart the **same original M3** it200
   checkpoint, on the **same 200245-tet Gmsh mesh**, same effective physics
   and wall settings, run absolute iterations 201–210. Compare the original
   old M3 log with the new log using `tools/compare_metrics.py` (optional).
4. Check scalar update residuals, pressure drop, wall Darcy/y+, mass flux,
   wall-normal BF3 flux, and U-bend secondary flow values. Ten iterations are
   an integration smoke test, not physical convergence.
5. Only once M3 is parity-clean, test M4 operator/action/sweep parity on the
   same state, then compare 10-iteration GPU timings; do not infer gains from
   successful compilation.
6. Later: optional 200-step run + secondary vortices / Choi overlays, followed
   by separate source cleanup and public repository migration.

## Verification completed in this environment

- Uploaded source archive SHA256 matches its `.sha256` file.
- `source/` 185/185 files byte-identical to frozen original.
- `make build/libnodals_sst_runtime.a` built six CPU C++ objects successfully,
  and a subsequent incremental `make` did not rebuild them.
- `python3 -m unittest discover -s tests -v`: 8 tests passed, including a
  fake-executable launch proving direct native invocation and stale stage
  environment isolation. A fake executable is **not a GPU validation**.
- Reference 200245-tet Gmsh mesh converted to HXT1, full writer audited PASS.
  The new generalized converter produced a **byte-identical HXT1 binary** to
  the frozen U-bend-specific converter (SHA256
  `d4af8b422553ca69c2bf97a0778fa69f8d5d02ad83e488b2807617683dae3d16`).
  General non-U-bend one-tet surface orientation likewise converted and audited.
- Cannot run a CUDA compile/execute here: **`nvcc` and a CUDA GPU are absent**.
  No M3 numerical parity or CUDA performance claim is made.

## Suggested future public repository topology

Keep the current public laminar GPU implementation untouched through
verification. Once workstation tests pass, version and publish the cleaned SST
solver **as a separate subtree or separately maintained repository**, e.g.:

```text
NodalS/
  gpu/serial_cuda/              # existing laminar implementation, unchanged
  gpu/tet_sst/                  # verified standalone SST source package
  docs/                        # shared high-level numerical/project docs
```

First verify clone+compile+run from a clean checkout. Archive history/tags
instead of deleting benchmarks or old implementations before reproducing M3.

### Unresolved limitations after this integration

- CUDA TU remains monolithic; changes to included CUDA physics may still take
  several minutes of `nvcc` recompilation.
- Internal stage identifiers in CUDA symbols/logs remain (renaming would be a
  source change); optional P2 may still enforce development-specific bounds.
- Current Gmsh converter supports ASCII MSH2 with physical IDs 1/2/3/4, single
  inlet/outlet/wall classifications, not arbitrary named BC groups.
- Native input pressure uses density=1 kinematic formulation; independent
  density physics cannot be activated by this case layer.
- GPU CUDA build, direct execution, baseline numerical parity and M4 parity
  still require a real workstation run.
