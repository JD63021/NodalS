# Serial CUDA pressure solver / AMG options

This branch retains the original SA path and all experimentally useful pressure
infrastructure. Slower options are intentionally preserved.

## Backward-compatible defaults

Unless explicitly selected:

- `--amg-hierarchy sa`
- `--amg-smoother cheb2`
- `--pressure-solver pcg`
- `--fine-csr-refresh-every 1`

Thus old commands continue to use the original SA + Chebyshev + PCG path.

## Recommended RTX 3060 fast path

```bash
--amg-hierarchy cf \
--cf-coarsening pmis \
--cf-strength classical-negative \
--cf-theta 0.25 \
--cf-interp exti \
--cf-pmax 8 \
--cf-aggressive-first 0 \
--amg-smoother jacobi \
--amg-jacobi-omega 0.7 \
--amg-spectrum-policy auto \
--pressure-solver richardson \
--pressure-richardson-omega 1.0
```

## Hierarchy controls

- `--amg-hierarchy sa|cf`
- `--amg-terminal N`

CF-specific:
- `--cf-coarsening pmis`
- `--cf-strength classical-negative`
- `--cf-theta THETA`
- `--cf-interp direct|exti`
- `--cf-pmax N`
- `--cf-aggressive-first 0|1`

PMIS and classical-negative are currently the implemented CF choices. They are
named options so future HMIS/Falgout/alternate-strength implementations can
extend the CLI without changing its structure.

Aggressive-first exact transfer composition is retained but defaults OFF.

## AMG smoother and spectrum

- `--amg-smoother cheb2|jacobi|l1jacobi`
- `--amg-jacobi-omega OMEGA`
- `--amg-cheb-degree N`
- `--amg-spectrum-policy auto|always|off`
- `--amg-power-its N`
- `--amg-lambda-safety X`
- `--amg-lambda-low-fraction X`

Spectrum policy:
- `auto`: estimate the inner AMG spectrum only when Chebyshev needs it.
- `always`: keep power iteration even with Jacobi/L1-Jacobi.
- `off`: skip it. Invalid with the inner Chebyshev smoother.

L1-Jacobi denominator is `sum_j |A_ij|`. Fine L1 is refreshed with the fine
numeric pressure CSR; coarse L1 is the setup snapshot.

## Pressure outer solver

- `--pressure-solver pcg|richardson|cheb`
- `--pressure-richardson-omega OMEGA`

Outer Chebyshev:
- `--pressure-cheb-degree N`
- `--pressure-power-its N`
- `--pressure-cheb-low-fraction X`
- `--pressure-power-safety X`

Outer Chebyshev estimates the selected preconditioned operator `M^-1 A`. Its
power controls are separate from the inner AMG smoother power controls.

## Fine pressure numeric refresh

- `--fine-csr-refresh-every N`

This only refreshes the fine numeric `B rAU B^T` pressure operator and fine
smoother data. It does not rebuild PMIS, interpolation, Galerkin coarse matrices,
or the AMG hierarchy.

The AMG hierarchy is currently constructed once at startup.
