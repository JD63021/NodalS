# NodalS custom pressure AMG

## Scope

The custom pressure AMG is the native low-memory preconditioner for the P1+BF3/P0 SIMPLE/SIMPLEC pressure-correction equation. The fine pressure operator remains the exact NodalS matrix-free Schur action

`A_p = B diag(rAU) B^T`.

The custom path does not build an explicit fine Schur CSR, the legacy flat pressure assembly plan, a PETSc pressure Pmat, PETSc GAMG, or a PETSc pressure KSP. PETSc full-Schur + GAMG remains a separate selectable fallback.

The AMG module is geometry-agnostic. It consumes distributed pressure ownership, pressure-cell adjacency, the exact Schur action/diagonal, and the pressure gauge policy. It does not contain pipe dimensions, patch names, inlet profiles, or other problem-specific logic.

## Public case modes

The normal `.case` interface exposes three validated combinations through `[pressure] mode`:

- `pcg_unsmoothed` — native FP64 PCG + unsmoothed aggregation AMG. Lowest memory.
- `pcg_smoothed` — native FP64 PCG + Jacobi-smoothed aggregation AMG. Stronger preconditioner.
- `richardson_smoothed` — native FP64 Richardson + smoothed aggregation AMG. Useful both as a production-capable stationary solve and as a clean AMG-vs-AMG comparison mode.

Richardson + unsmoothed aggregation is intentionally not a public case preset because the matched large-mesh stationary test showed that combination was not robust enough. Low-level command-line options remain available for developer diagnostics.

## Transfer and coarse operators

### Unsmoothed aggregation

Fine cells are grouped into deterministic connected rank-local aggregates. The tentative transfer injects one fine pressure unknown into one aggregate:

`P = P_t`, `R = P^T`.

The first coarse operator is built directly from the factored fine operator without materializing `A_0`:

`A_1 = P_t^T B diag(rAU) B^T P_t`.

### Smoothed aggregation

The tentative transfer is Jacobi-smoothed:

`P = (I - omega D^{-1} A) P_t`, `R = P^T`.

with

`omega = sa_damping / lambda_max(D^{-1} A)`.

The first smoothed transfer and first Galerkin coarse operator are streamed from the factored `B diag(rAU) B^T` representation, so a fine Schur CSR and generic fine-grid PtAP are still avoided. Subsequent levels use the custom explicit coarse CSR matrices already produced by the hierarchy.

## V-cycle

Each non-coarsest level applies a symmetric scaled Chebyshev/Jacobi correction before and after the coarse correction. Restriction is the transpose of prolongation. The final small coarse problem is gathered and solved with a rank-0 dense LU.

The fixed symmetric V-cycle is suitable for the native FP64 PCG outer iteration. The same V-cycle can also be used in stationary Richardson mode.

## Case-file controls

The hierarchy controls belong in a dedicated `[pressure_amg]` section. The defaults below are the values used in the validated 768k, 1.1M, 2M and 7.18M large-mesh campaigns.

| Key | Default | Meaning |
| --- | ---: | --- |
| `target_aggregate` | `8` | Preferred connected cells per rank-local aggregate |
| `min_aggregate` | `6` | Merge smaller aggregates through local face adjacency when possible |
| `soft_max_aggregate` | `10` | Diagnostic/preferred upper aggregate size |
| `chebyshev_degree` | `2` | Degree of each symmetric Chebyshev/Jacobi smoothing correction |
| `power_iterations` | `16` | Power iterations for the level spectral estimate |
| `lambda_safety` | `1.50` | Safety multiplier on estimated `lambda_max` |
| `lambda_low_fraction` | `0.05` | Lower spectral endpoint as a fraction of `lambda_max` |
| `coarse_target_rows` | `1000` | Global row target below which the final coarse LU is used |
| `interpolation_max_row_nnz` | `8` | Maximum retained interpolation entries per row after SA pruning |
| `sa_damping` | `1.3333333333333333` | Smoothed-aggregation Jacobi damping numerator |
| `richardson_omega` | `1.0` | Stationary Richardson correction scale |

The pressure outer solve controls remain in `[pressure]`:

- `refresh` — hierarchy/preconditioner refresh interval; validated default `100`.
- `rtol`, `atol`, `divtol`, `max_iterations` — native pressure outer stopping controls.

## Example

```ini
[pressure]
mode = pcg_smoothed
refresh = 100
rtol = 0.5
atol = 1e-12
divtol = 1e8
max_iterations = 20

[pressure_amg]
target_aggregate = 8
min_aggregate = 6
soft_max_aggregate = 10
chebyshev_degree = 2
power_iterations = 16
lambda_safety = 1.50
lambda_low_fraction = 0.05
coarse_target_rows = 1000
interpolation_max_row_nnz = 8
sa_damping = 1.3333333333333333
richardson_omega = 1.0
```

Change only `mode` to `pcg_unsmoothed` or `richardson_smoothed` to select the other validated combinations while keeping the hierarchy controls visible in the case.

## Runtime diagnostics

A custom run prints `P1BF3_CUSTOM_AMG_CONFIG`, which echoes the effective aggregation, smoother, spectral, interpolation and refresh controls. The existing hierarchy diagnostics also report level count, fine/final row counts, coarse/transfer nnz, retained hierarchy estimate and transfer family.

## Current scaling observations

The present implementation has been exercised through approximately 7.18 million tetrahedra on 16 MPI ranks. The large-mesh tests established that both custom variants remove the large PETSc full-Schur/GAMG setup-memory spike. Unsmoothed aggregation is the minimum-memory mode; smoothed aggregation retains a substantially larger hierarchy in exchange for roughly halving PCG iteration counts in the tested large-mesh regime.

These measurements are implementation observations, not hard-coded assumptions in the AMG module.
