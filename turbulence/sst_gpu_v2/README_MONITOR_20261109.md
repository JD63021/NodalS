# NodalS SST GPU V2 — 9 November 2026 monitor/convergence freeze

Source snapshot: `NodalS_SST_V2_DP_GATE_20261109`, supplied by the workstation owner.
This file describes the **actual source changes**, not a claimed CUDA validation.
The existing `README.md` from the November 8 GitHub freeze is retained.

## Added monitor

- `[output] live_monitor=true` launches `tools/nodals_live_monitor.py` beside the solver.
- `[output] monitor_refresh_seconds=3` chooses UI refresh interval.
- `[output] monitor_pressure_scale=1000` changes plotted pressure units only.
- The monitor reads `terminal.log` and plots available residual trends, pressure drop,
  and wallDarcy. It is explicitly fail-open: missing UI, matplotlib, or display
  cannot stop the numerical solve.
- The self-test is `python3 tools/nodals_live_monitor.py --self-test`.

## Optional outer convergence criterion

```ini
[convergence]
metric = legacy                 # legacy | pressure_drop | wall_traction | both
relative_tolerance = 0.01
window_iterations = 25
require_residuals = true

[output]
print_every = 10
metrics_every = 1               # REQUIRED for every non-legacy metric
```

A window is accepted when the relative span (max-min)/max(abs(mean),1e-12)
for the selected metric(s) is at most `relative_tolerance`. The implementation
uses the computed pressure drop and **wallDarcy friction factor** as its wall
stationarity measure; `wall_traction` is a configuration name, not an assertion
that the convergence code compares an independently integrated traction force.
When `require_residuals=true`, the existing mass, momentum and turbulence
checks must also pass. A health/finite-value check remains active.
`legacy` preserves the old decision logic.

## Q1 and numerical options

`[sst] state_space=q1, compact_p1=true` selects genuinely compact Q1 storage
for the transported scalar k and omega fields, whereas `compact_p1=false`
keeps the masked/enriched storage reference. **Velocity remains P1+BF3**;
pressure is P0. Compact scalar Q1 is not automatically numerically equivalent
to the masked reference. The snapshot retains momentum convection variants,
vector-Laplacian/deviatoric stress choices, and existing SUPG restrictions.

## Attached Shell2 case

`examples/shell2_360k_monitored_20261109.case` is a snapshot of the working
monitor configuration. Its mesh path is external and must point to the local
Shell2 mesh. It intentionally sets `convergence.metric=legacy` and
`sst.compact_p1=false`: running that case does **not** validate the new
non-legacy stopping gate or compact-Q1 physics. Adjust case values explicitly
for further experiments.

## Validation and limitations

- Pure-Python compile and monitor self-test verified in the handoff.
- New self-contained host-only tests in `tests/test_monitor_controls.py` test
  the parser, dry-run case options, compact-Q1 switch and metric cadence.
- Original `tests/test_software.py` depends on a missing U-bend reference mesh
  and external developed-profile CSV in this handoff; it cannot be used as an
  independent clean-environment pass/fail gate.
- CUDA compilation and numerical parity require the user's RTX 3060 workstation.
- Generated mesh caches, backups, run outputs and binaries are excluded.
