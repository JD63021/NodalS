# NodalS v1.00 source layout

v1.00 uses **one C++ translation unit assembled from role-specific `.inc` files**.
This is intentional: the checkpoint-1 refactor is mechanical and preserves the exact
text, ordering, anonymous namespace boundaries, `static` linkage, PETSc call paths and
floating-point operation ordering of the frozen Gate9N source.

Modules:

- `00_core/` — PETSc/C++ preamble, OpenFOAM tet mesh reader, runtime/memory utilities.
- `10_fem/` — tetrahedral quadrature, P1+BF3 basis and MMS exact fields.
- `20_boundary/` — problem modes, patch geometry, wall/inlet/outlet preparation.
- `30_momentum/` — velocity halo/CSR machinery and central convection assembly.
- `35_stabilization/` — SUPG and dynamic momentum assembly plans.
- `40_pressure/` — exact/factored Schur action, compact/full Pmat, pressure assembly and profiling.
- `50_parallel/` — ownership, distributed solve support and gather helpers.
- `60_output/` — error/flow diagnostics and VTU output.
- `70_simple/` — SIMPLE/SIMPLEC support plus retained historical Gate probes/preconditioner experiments.
- `80_app/` — production `main()` and outer solver orchestration.

Do not independently reorder these fragments in v1.00. `tests/check_source_freeze.py`
proves that concatenating them reproduces the frozen monolithic source byte-for-byte.
A later release may turn stable modules into independently compiled `.cpp` files after
numerical parity has been established on the workstation.
