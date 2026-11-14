# Augmented Lagrangian experiment — GPU SST, pure-tet, 2026-11-14

**Status: source implementation, NOT yet CUDA compiled or physically validated.**
This branch was based on the published GitHub commit `ca7d759b`, not the
possibly newer uncommitted workstation source. Do not overwrite existing
successful experiments.

## Exact discrete algebra

Original continuity in NodalS is `B*u + csrc = 0`.  For P0 pressure,
`Mp=diag(tet.volume)`. On the active continuity rows, the physical
momentum predictor is therefore

```
(A + gamma B^T Mp^-1 B) u = f - gamma B^T Mp^-1 csrc
```

where `B` is the SAME boundary-adjusted DG-inlet operator used in SIMPLE.
The augmentation is applied before BF3 transformation `T^T (...) T`.
Existing Spalding Robin/Nitsche tensors and masked velocity DOFs are untouched.

M3 FGS/SGS: add each cell's 9 velocity-block outer products to the
existing 9-block CSR before BF3 projection. No separate global CSR.
Coupled FGMRES: perform the same `B`, P0 diagonal scaling and `B^T`
operations within the frozen coupled momentum operator. Gamma=0 skips all
extra operations and uses the old code path.

The SIMPLE Schur is **not exact**: I3A uses an augmented diagonal
inverse-momentum surrogate `r/(1+r*diag(G))` followed by its usual
BF3-compatible `B C B^T` pressure operation. The *same* `C` is used
for velocity correction. This is a compatible SIMPLE approximation,
not `B A_gamma^-1 B^T`, and is NOT guaranteed good for huge gamma.
The pressure AMG preconditioner remains a preconditioner, not an exact
Schur replacement.

## Editable case settings

Append to the **existing successful pure-tet Re1M 99k .case** (do not
substitute the hybrid 95k tet + 3744 hex case):

```ini
[grad_div]
enabled = true
method = augmented_lagrangian
gamma = 0.1
```

No artificial upper bound on gamma. Gamma has kinematic viscosity-like
units in these equations. Increase only after the zero-gamma parity check.
Use the original compatible one-layer mesh, same omega trace settings,
same adaptive wall-sample stage, and all other successful Re1M settings.

```ini
[solver.u]
method = m3-fgs
sweeps = 2
```

The initial recommended experiment is `gamma=0` versus a small
nonzero gamma; do not assume 0.1 is optimal. If testing coupled
FGMRES, select `method=coupled-fgmres` and `rtol=0.1`.
The default production-style test is M3 FGS **2 sweeps**.

## Workstation commands (new worktree; preserves current checkout)

```bash
cd /home/jd/NodalS
git fetch origin exp/sst-augmented-lagrangian-20261114
git worktree add /home/jd/NodalS_AL_RE1M origin/exp/sst-augmented-lagrangian-20261114
APP=/home/jd/NodalS_AL_RE1M/turbulence/sst_gpu_v2
make -C "$APP" -j4 NVCC=/usr/local/cuda-12.2/bin/nvcc CUDA_ARCH=sm_86
# Set BASE_CASE to the existing pure-tet 99k Re1M .case on this workstation.
BASE_CASE=/absolute/path/to/existing/PIPE50_RE1M_99K_PURETET.case
mkdir -p /home/jd/NodalS_AL_RE1M_cases
cp "$BASE_CASE" /home/jd/NodalS_AL_RE1M_cases/RE1M_99K_AL.case
CASE=/home/jd/NodalS_AL_RE1M_cases/RE1M_99K_AL.case
# Add [grad_div] and set [solver.u] as above in your editor:
xdg-open "$CASE"
python3 "$APP/nodals.py" run --case "$CASE" --dry-run
python3 "$APP/nodals.py" run --case "$CASE" --steps 20 \
   --output /home/jd/NodalS_AL_RE1M_RUNS/gamma_trial
```

**Caution:** Copying a case can break relative mesh/profile paths.
Use absolute `mesh.path` and `inlet.main.profile_csv` or keep the
case next to the original until paths have been verified.

Before any long run, compare gamma=0 against the original on 1–5
outer steps. Then check M3 CSR / matrix-free AL action parity on the
first step with `[output] diagnostics_every=1`; check inlet mass,
wall-normal flux, full BF3 residuals, and pressure-correction identity.
Only then test 100+ steps and the Moody 20–30D, 30–40D, 40–49D fits.

## Limitations to investigate

- No CUDA compiler or RTX 3060 is available in the assistant's execution
  environment; CUDA build/run/production parity remain UNTESTED.
- The first parity gate is normally controlled by `diagnostics_every`;
  also inspect its exact log cadence.
- M3 FGS with 2 sweeps is intentionally inexact; AL convergence must be
  judged at the full SIMPLE level.
- For large gamma, the I3A diagonal inverse approximation may become a
  poor substitute for the exact augmented momentum inverse.
- The same relative tolerance should not be imposed on M3 2-sweep runs:
  it is only meaningful for coupled FGMRES.
- The pure-tet 99k mesh and its successful original adaptive-y+ inputs
  have NOT been uploaded here; use the workstation's existing case.
