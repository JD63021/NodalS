# NodalS SST MPI CPU — Message 1 foundation

**Scope: This is a compile-and-probe foundation, NOT yet a CFD solver.**
The following are **not** implemented here: mesh-dependent FE operators,
momentum/pressure solves, scalar transport systems, boundary assembly, wall
sampling, SIMPLE outer iterations, restart, output VTU. These are the next
implementation steps. The binary name explicitly includes `_probe`.

## Invariants

- Original GPU numerical files and `nodals.py` are **not modified or copied**.
- Reuse the GPU module's independently compiled, portable host mathematical
  APIs *by reference* in the CPU Makefile.
- Preserve existing `.case` format. Probe parser reads the real GPU `.case`
  without editing it or dropping sections; the full solver will implement
  interpretation of all applicable keys in later messages.
- MPI model is `16 ranks x 1 thread`, as requested. OpenMP is not introduced.
- Existing laminar solver and PETSc build are not touched in Message 1.
- Portable `evaluate_host` SST math currently includes a **pipe-distance**
  helper. A pointwise probe of it is NOT a generalized spiral solution; the
  future CPU FEM operator must use the geometry-general distance exactly as the
  existing GPU solver does.

## Build

From the cloned NodalS repository root:

```bash
make -C turbulence/sst_cpu_mc -j8 print-config
make -C turbulence/sst_cpu_mc -j8
```

The probe uses the same MPI toolchain as the laminar solver. `PETSC_DIR` and
`PETSC_ARCH` do not enter this Message-1 link because its executable does not
yet use the existing PETSc and custom AMG objects. They will be wired in when
the native momentum/pressure backend is integrated.

## Run the non-solver probe

```bash
python3 turbulence/sst_cpu_mc/tools/run_case.py \
  --case examples/spiral_swept_207k/SPIRAL_SWEPT_ADAPTIVE.case \
  --mesh /ABSOLUTE/PATH/TO/spiral13_135411.msh --ranks 16 --probe --dry-run
python3 turbulence/sst_cpu_mc/tools/run_case.py \
  --case examples/spiral_swept_207k/SPIRAL_SWEPT_ADAPTIVE.case \
  --mesh /ABSOLUTE/PATH/TO/spiral13_135411.msh --ranks 16 --probe
```

The `--mesh` override is only for the probe. The correct 135k `.case` will be
prepared from the GPU reference after the mesh geometry/inlet compatibility
check, not inferred from the directory name.

## Convert the provided OpenFOAM pure-tet mesh

```bash
python3 turbulence/sst_cpu_mc/tools/foam_tet_to_msh2.py \
  "$HOME/Desktop/meshes/spiral/13ktet" \
  -o "$HOME/Desktop/meshes/spiral/13ktet/spiral13_135411.msh"
```

The converter preserves the input point coordinates, original tetrahedral
cell topology, and boundary triangles. The original OpenFOAM mesh is never
changed. For the uploaded 2026-11-09 mesh, the verified IDs are inlet
`patch_2_0` -> tag 1, outlet `patch_1_0` -> tag 2, wall `patch_0_0` -> tag 3,
volume -> tag 4.

## Next-stage target interface

1. `src/mesh/`: CPU use of current laminar MPI ownership and tetrahedral FE
   geometry on the same mesh, with parity against GPU element operators.
2. `src/momentum/` + `src/pressure/`: preserve the current full coupled vector
   momentum, BF3 wall projection and Schur pressure correction; reuse native
   MPI SGS/PCG/AMG backends with no scalarized approximation.
3. `src/transport/` + `src/wall/`: add masked P1, k/omega sources and wall state.
4. `src/coupling/`: reproduce GPU SIMPLE order, adaptive wall sampling and
   checkpoint/output policies, then benchmark MPI rank counts.

The new source is deliberately split into build, MPI app, portable bridge,
mesh conversion, and case runner. It is not a new monolithic solver.
