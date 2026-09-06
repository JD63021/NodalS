# NodalS custom pressure AMG

## Scope

The native custom pressure AMG is the low-memory preconditioner for the P1+BF3/P0 SIMPLE/SIMPLEC pressure-correction equation. The exact fine pressure operator remains matrix-free:

`A_p = B diag(rAU) B^T`.

The custom path does not build an explicit fine Schur CSR, a PETSc pressure Pmat, PETSc GAMG, or a PETSc pressure KSP. Coarse AMG levels are explicit custom FP64 CSR matrices and the terminal coarse problem is solved by rank-0 dense LU. The PETSc pressure paths remain separately selectable.

The fused pressure and momentum kernels are structural execution optimizations and do not change the discrete operator. The pressure Schur action shares the `B^T` traversal across x/y/z, keeps the three scalar velocity halo exchanges, and shares the forward `B` traversal. The native momentum predictor shares CSR traversals across Ux/Uy/Uz while keeping the established scalar halo semantics.

## Default low-memory path

If a case omits `[pressure] mode`, the case translator selects native FP64 PCG with unsmoothed custom AMG. The default custom-AMG settings are:

- connected rank-local aggregation: target `16`, minimum `6`, soft maximum `18`;
- processor-block symmetric Gauss-Seidel smoothing;
- one SGS correction before and one after the coarse correction; each SGS correction is forward + backward, giving four triangular sweeps per V-cycle on every non-terminal level;
- exact matrix-free fine Schur and explicit custom coarse levels;
- terminal coarse target of `1000` global rows.

The aggregate sizes are tuning parameters, not discretization constants. They remain case-file options because mesh topology, partitioning and hierarchy depth can change the best value.

## Public modes and smoothers

`[pressure] mode = pcg_unsmoothed` is the default custom path. Its V-cycle smoother is selected independently by `[pressure_amg] smoother` and defaults to `sgs`.

For an unsmoothed hierarchy, `smoother` accepts:

- `sgs` — processor-block SGS; default;
- `jacobi` — one undamped Jacobi correction before and after coarse correction; no spectral power iterations;
- `chebyshev` — the previous symmetric scaled Chebyshev/Jacobi smoother.

Convenience presets remain available as `pcg_unsmoothed_sgs`, `pcg_unsmoothed_jacobi`, and `pcg_unsmoothed_chebyshev`, with aliases `custom_sgs`, `custom_jacobi`, and `custom_chebyshev`.

`pcg_smoothed` and `richardson_smoothed` remain available. Smoothed aggregation currently requires `smoother = chebyshev` because transfer smoothing uses the estimated spectral upper bound.

## Case-file controls

The custom hierarchy controls belong in `[pressure_amg]`:

| Key | Default | Meaning |
| --- | ---: | --- |
| `target_aggregate` | `16` | Preferred connected pressure cells per rank-local aggregate |
| `min_aggregate` | `6` | Merge smaller aggregates through local face adjacency when possible |
| `soft_max_aggregate` | `18` | Preferred/diagnostic upper aggregate size |
| `smoother` | `sgs` | `sgs`, `jacobi`, or `chebyshev` for unsmoothed AMG |
| `chebyshev_degree` | `2` | Degree of each Chebyshev/Jacobi correction |
| `power_iterations` | `16` | Power iterations for the spectral estimate |
| `lambda_safety` | `1.50` | Safety multiplier on estimated `lambda_max` |
| `lambda_low_fraction` | `0.05` | Lower spectral endpoint as a fraction of `lambda_max` |
| `coarse_target_rows` | `1000` | Global row target below which terminal LU is used |
| `interpolation_max_row_nnz` | `8` | Maximum retained interpolation entries after SA pruning |
| `sa_damping` | `1.3333333333333333` | Smoothed-aggregation Jacobi damping numerator |
| `richardson_omega` | `1.0` | Stationary Richardson correction scale |

For `sgs` and `jacobi`, the Chebyshev spectral controls are not used by the V-cycle. They may remain in a case so the smoother can be switched without rewriting the rest of the configuration.

## Example

```ini
[pressure]
mode = pcg_unsmoothed
refresh = 100
rtol = 0.5
atol = 1e-12
max_iterations = 20

[pressure_amg]
target_aggregate = 16
min_aggregate = 6
soft_max_aggregate = 18
smoother = sgs
coarse_target_rows = 1000
```

To test mesh-dependent coarsening, change `target_aggregate`, `min_aggregate`, and `soft_max_aggregate`. For example, target `12` with soft maximum `14` remains a valid runtime choice. To compare smoothers without changing the pressure operator or transfer family, keep `mode = pcg_unsmoothed` and change only `smoother`.

## Runtime diagnostics

Custom runs report `P1BF3_CUSTOM_AMG_CONFIG`, hierarchy sizes, retained hierarchy estimates, selected smoother, and whether spectral power iterations are enabled. The fine operator continues to report `fineSchurCSR=0`. Fused pressure and momentum execution paths are reported by `P1BF3_M4B_SCHUR_FUSION` and `P1BF3_MOMENTUM_FUSION`.
