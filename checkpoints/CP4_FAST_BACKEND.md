# CP4 — Fast pressure backend exposed

The frozen Gate9N reference remains byte-preserved under `reference/`.
The active NodalS solver adds `-pressure_solve_mode gamg_richardson` by reusing the existing exact factored pressure operator, full Schur Pmat update/lag machinery, PETSc KSP container, and GAMG PC.

Case-level modes:
- `fast_full_gamg` — full Pmat + PETSc Richardson/GAMG (primary)
- `compact_cheb` — native-face FE energy Pmat + custom Chebyshev/GAMG (low memory)
- `compact_pcg` — compact/full Pmat + custom FP64 PCG/GAMG

No FE discretization, momentum operator, BC semantics, rAU construction, or pressure physical operator was changed.
