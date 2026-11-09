# V2 diagnostic / output control implementation (2026-11-04)

- Native `NODALS_SST_DIAGNOSTIC_EVERY` (0=off, N=every N steps) gates the
  coupled full-vector predictor residual before/after M3, startup M3 operator
  parity, Schur identity and wall-normal audit action. These operations are
  observation-only and should not feed the SIMPLE update. Strict parity **not
  yet GPU verified**.
- Reduced post-SIMPLE momentum residual and its pre-assembly old-velocity
  snapshots are evaluated at output.print_every (or when optional diagnostics
  or physical momentum tolerance demand them), not each step.
- `STAGE31_CHECKPOINT_EVERY=0` is now valid and suppresses binary checkpoints;
  N>0 checkpoint cadence and convergence-triggered checkpoint remain.
- `SST_G2_VTU_EVERY` and `SST_G2_VTU_DIR` set periodic flow and SST VTUs.
- `[restart].path` sets `STAGE31_RESTART_FROM`. The existing checksummed format
  and fingerprint protections are retained.
- Native `NODALS_SST_RESIDUALS it=...` is produced from available residual
  quantities on report steps (must coincide with metrics). For fixed scalar
  FGS, `check_every` no longer forces a residual audit at nonlinear iteration 1.
- Legacy hex-only cross-section diagnostics do not run on tet unless explicitly
  requested, and when requested on tet the hex audit is skipped. Native process
  can exit normally at a user-selected non-converged iteration limit.
- `nodals.py` compact console mode intentionally filters setup/stage chatter;
  raw terminal.log is saved. Checkpoint and VTU output still use CUDA code.

CPU-only tests: `python3 -m unittest discover -s tests -v`.
CUDA validation on RTX3060 still required before declaring numerical parity:
run 10 steps from fresh state with V1 and V2 identical physics settings, compare
wall/flux/k/omega/p/velocity and timing. Repeat same-state Stage31 restart when
checkpoint compatible. Inspect `terminal.log` and output VTU. No claim of
speedup is made until this is done.
