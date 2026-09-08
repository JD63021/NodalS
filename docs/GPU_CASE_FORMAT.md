# NodalS H8 serial-CUDA `.case` format

`bin/nodals-gpu` / `scripts/nodals_gpu_case.py` translate an INI-style `.case`
file to the existing H8 CUDA command line.  The runner does not change numerical
algorithms.

Use `cases/gpu-h8-annotated.case` as the canonical fully commented template.

## Precision and execution

`[run] precision = fp32|fp64` selects the already existing
`nodals_gpu_h8_fp32` or `nodals_gpu_h8_fp64` binary.  `run_mode` is `converge`
or `fixed10`; fixed10 requires `[simple] max_iterations = 10`.

The current H8 source is single-GPU and selects CUDA device 0.  Device selection
is therefore not exposed as a case option yet.

## Pressure hierarchy

`[pressure_amg] hierarchy = sa|cf` selects:

- `sa`: the original smoothed-aggregation hierarchy.
- `cf`: the C/F hierarchy.  Current C/F coarsening is PMIS with
  classical-negative strength.

The original historical pressure pairing is approximately:

- hierarchy `sa`
- smoother `cheb2`
- pressure solver `pcg`

The current fast CF/PMIS configuration is approximately:

- hierarchy `cf`
- coarsening `pmis`
- strength `classical-negative`
- interpolation `exti`
- smoother `jacobi`
- pressure solver `richardson`

## AMG smoothers

`[pressure_amg] smoother` accepts `cheb2`, `jacobi`, or `l1jacobi`.

The inner spectrum policy is `auto|always|off`.  `off` is rejected with
`cheb2`, because Chebyshev requires the spectral estimate.

## Pressure outer solver

`[pressure] solver` accepts `pcg`, `richardson`, or `cheb`.

Outer-Chebyshev power/degree/spectral controls are separate from the inner AMG
Chebyshev controls.

## CF/PMIS controls

`[pressure_amg_cf]` exposes all current C/F CLI controls:

- `coarsening = pmis`
- `strength = classical-negative`
- `theta`
- `interpolation = direct|exti`
- `pmax`
- `aggressive_first = true|false`

PMIS and classical-negative are the only currently implemented coarsening and
strength choices; the names are retained in the public format to make later
extensions non-breaking.

## Deliberately absent features

The GPU case interface does not pretend to expose features that H8 does not yet
implement: CUDA SUPG, DG inlet, algebraic mixing length, general GPU device
selection, or adaptive momentum inner stopping.
