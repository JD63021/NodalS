# CPU native custom-AMG case interface

The CPU/MPI solver already contains two native low-memory custom-AMG hierarchy
families.  The public `.case` interface selects them through `[pressure] mode`
and tunes them through `[pressure_amg]`.

Use `cases/cpu-custom-amg-annotated.case` as the reusable, fully commented
reference.

## Unsmoothed aggregation

Use one of:

- `mode = pcg_unsmoothed` — native FP64 PCG, unsmoothed aggregate injection;
  `[pressure_amg] smoother` may be `sgs`, `jacobi`, or `chebyshev`.
- `mode = pcg_unsmoothed_sgs` — same hierarchy, fixed SGS smoother preset.
- `mode = pcg_unsmoothed_jacobi` — same hierarchy, fixed Jacobi preset.
- `mode = pcg_unsmoothed_chebyshev` — same hierarchy, fixed Chebyshev preset.

The CPU custom hierarchy is connected rank-local aggregation.  CPU PMIS is not
implemented and is intentionally not advertised by the CPU case format.

## Smoothed aggregation

Use:

- `mode = pcg_smoothed` — native FP64 PCG with smoothed aggregation transfer.
- `mode = richardson_smoothed` — native FP64 Richardson with the same smoothed
  hierarchy.

The current smoothed-transfer implementation requires
`[pressure_amg] smoother = chebyshev` because the transfer smoothing uses the
spectral estimate.

## Common custom-AMG controls

`[pressure_amg]` exposes aggregate-size controls, smoother selection,
Chebyshev/power-iteration controls, terminal coarse size, interpolation sparsity
cap, smoothed-aggregation damping, and Richardson omega.  The fully annotated
case documents which controls are active for each hierarchy.

Existing PETSc/full-GAMG and compact pressure modes remain available unchanged.
