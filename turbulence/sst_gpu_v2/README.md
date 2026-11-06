# NodalS SST — Software V2: lean production output and case-based checkpoints

**Status (2026-11-04):** The Python launcher, host-only integration tests, and
200245-tet preprocessing work. **CUDA compilation, the modified native execution
path, M3 numerical parity, and performance changes still require an actual
RTX 3060 run.** The original physical discretization, assembled momentum and
SST operators, wall models, pressure method, and existing M3/M4 algorithms are
not deliberately altered. V2 modifies how optional diagnostic work is executed.

## Install and edit

```bash
cd ~/Downloads
tar -xzf NodalS_SST_SOFTWARE_V2.tar.gz
cd NodalS_SST_SOFTWARE_V2
python3 -m unittest discover -s tests -v
make -j4 NVCC=/usr/local/cuda-12.2/bin/nvcc CUDA_ARCH=sm_86
xdg-open examples/ubend_200k.case
```

**You must recompile ONCE for V2**, because this version changes CUDA
`.inc` files that control diagnostics. After that, editing the case does not
recompile anything. `make` is incremental; an unchanged CUDA source skips nvcc.

Run with only the case (no CLI settings override it):

```bash
python3 -u nodals.py run --case examples/ubend_200k.case
```

For an output-only dry run:

```bash
python3 nodals.py run --case examples/ubend_200k.case --dry-run
```

With `[output] console = compact`, your terminal shows residual reports only
at `print_every` (the default is 10), plus checkpoint notices, final completion,
and errors. **The full native output remains in each run's `terminal.log` for
support and debugging.** No mandatory Stage22C/Stage26 installed app or SHA
registration is used.

## Case controls

The annotated `examples/ubend_200k.case` includes all normal numerical options
plus the new controls:

```ini
[simple]
max_iterations = 200   # ABSOLUTE target including any restart iterations

[sst]
check_every = 10      # true k/omega linear residuals every 10, no first-step exception

[restart]
path =                # leave blank for fresh start; otherwise /absolute/..._it200.ckpt

[output]
console = compact     # compact | full
print_every = 10
metrics_every = 10    # must divide print_every
# 0 => no optional expensive full operator/wall-normal/identity audits
diagnostics_every = 0
legacy_final_diagnostics = false
vtu_every = 100       # flow+SST visualization snapshots; 0 => none
checkpoint_every = 100  # restartable FP64 U/P/k/omega .ckpt; 0 => none
# vtu_dir = vtu       # optional run-relative directory
# checkpoint_dir = checkpoints
```

Restart example: `path = /home/jd/NodalS_SST_LOCAL_RUNS/runX/NODALS_STAGE31_STATE_it200.ckpt`
with `[simple] max_iterations = 500` continues at iteration 201 and targets 500.
`--restart /path/to/file` overrides the case. `--steps 50` requests 50 **more**
steps from a checkpoint filename ending `_it200.ckpt` (native target 250).
A `.vtu` cannot be used for restart. Checkpoint reader validates FP64 vector
sizes, saved data checksum and case/mesh input fingerprint; it does **not**
validate the executable SHA. Legacy checkpoints with different fingerprints
require explicit compatibility review, not silent bypass.

### Reporting semantics

At iterations 10,20,... the one-line report is `NODALS_SST_RESIDUALS`:

- `UxAbs`, `UyAbs`, `UzAbs`: L2 norms of the reduced physical post-SIMPLE
  momentum residual. `UxRel`, etc. are divided by their **first reported**
  value. They are OUTER residual trend measures, not M3 FGS inner rtol.
- `pAbs`, `pRel`: the pressure linear equation solve residual and relative drop.
- `contAbs`, `contRel`: corrected SIMPLE continuity residual; relative to
  corrected continuity reference norm established near the beginning of the run.
- `kAbs`, `kRel`, `omegaAbs`, `omegaRel`: **true linear-system residual**
  after a fixed FGS sweep set, computed only at `sst.check_every`. If that
  setting is 0, these values are `nan` (not evaluated, NOT a solver failure).
- `kUpdateRel`, `omegaUpdateRel`: SST outer updates, continuously required for
  turbulence-state control (NOT true scalar linear solve residuals).

`print_every` must be a multiple of `metrics_every` and, for true scalar checks,
`sst.check_every` must divide `print_every`. You may use `metrics_every = 1`
if you want convergence/physical metrics computed each step but printed only at
10,20,... . With `metrics_every = 10`, the physical/plateau convergence decision
runs only at those measurement steps and at the last requested step. The
pressure correction/continuity and SST state updates **still run every step**.

`diagnostics_every = 0` skips the optional M3 before/after full-vector residual,
Schur identity, wall-normal and post-SIMPLE reduced momentum diagnostic **extra
operator actions**. The standard post-SIMPLE momentum residual needed for the
one-line report is still computed at print cadence. Positivity, nonfinite-state
checks, physically required SIMPLE operations, and PCG convergence remain active.
The raw native program still emits some legacy internal log messages (captured
in `terminal.log`); compact console mode hides these. Repeated diagnostic print
statements without extra GPU work have not all been removed in this iteration.

`legacy_final_diagnostics = false` skips the old hex-only axial cross-section
and extra diagnostic VTU on a tetrahedral mesh. Normal end-of-run flow/SST VTUs
are unaffected, and hex-only diagnostics may be requested on hex geometry.
Nonconvergence at the user-requested SIMPLE limit is reported as *not converged*,
not treated as a crashed GPU executable. Real invalid numerical states and
meaningful physics checks still fail.

## Scope and limitations

Native output and restart are tested through the Python runner and mock
executables here, not CUDA execution. The 200245-tet Gmsh reference is included;
other case geometry must currently use one inlet tag=1, one outlet tag=2,
wall tag=3 and volume tag=4, with Gmsh MSH 2.2 ASCII (or preconverted HXT1).
The reference fit distances are configured for this mesh; for different meshes
check and explicitly set the physical wall-sampling distances.

`make -j4` only builds modified objects; the large original single CUDA TU has
not been split. Binary hash gates do not affect standalone execution. M4 remains
experimental. Keep V1 for parity reference, and compare full `terminal.log`
physical measures after compiling V2 before making performance claims.
