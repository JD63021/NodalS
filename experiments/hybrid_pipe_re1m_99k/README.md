# 99,216-cell hybrid HEX/TET SST pipe — November 14, 2026
Status: **experimental**. Geometry/host conversion previously checked in uploaded snapshot; new GitHub-generated mesh must pass its own HXT1 audit. CUDA compile, first SIMPLE steps and numerical conservation are NOT verified.

## What is in GitHub

- `generate_hybrid_pipe.py`: creates mesh/HXT0.npz, exact previous 99,216-cell 3,744 HEX + 95,472 TET discretization near D=0.889m, L=44.45m with 96 angular and 39 axial segments.
- `enable_hybrid_local.py`: small opt-in patch for **the user's latest local November-14 SST app**. It does NOT replace the workstation's local source with the older GitHub main source.
- `PIPE50_RE1M_HYBRID_99K_ONEPOINT.case`: editable Re1M case, one-point HEX Spalding wall, OF_AUTO with sample_log target, volume k/omega SST, old segregated hybrid momentum path.
- `RUN.sh`: local generation + source isolation + preflight, then first five CUDA steps only.

`tools/prepare_hxt1.py` is already in the user's November-14 local app.  No mesh binary or chat artifact download is needed.

## Workstation commands

```bash
cd "$HOME/NodalS"
git fetch origin exp/gpu-sst-hybrid-pipe-re1m-99k-20261114
git worktree add --detach "$HOME/NODALS_HYBRID_RE1M_99K_GIT" origin/exp/gpu-sst-hybrid-pipe-re1m-99k-20261114

EXP="$HOME/NODALS_HYBRID_RE1M_99K_GIT/experiments/hybrid_pipe_re1m_99k"
bash "$EXP/RUN.sh" prepare
xdg-open "$HOME/NODALS_HYBRID_RE1M_99K_RUN/cases/PIPE50_RE1M_HYBRID_99K_ONEPOINT.case"
bash "$EXP/RUN.sh" dry
bash "$EXP/RUN.sh" 5
```

Runner defaults:
- source to copy once: `$HOME/NodalS_GPU_GENROBIN_ADAPTIVE/turbulence/sst_gpu_v2`
- isolated experimental app: `$HOME/NodalS_GPU_HYBRID_RE1M_99K/turbulence/sst_gpu_v2`
- generated HXT1 mesh and editable case: `$HOME/NODALS_HYBRID_RE1M_99K_RUN`

If source moved: `export NODALS_ORIGINAL_APP=/absolute/path/to/turbulence/sst_gpu_v2`.
If SciPy is missing, install `numpy` and `scipy` in your current Python environment. The local `tools/prepare_hxt1.py` does geometry and BF2 interface checks before CUDA compilation.

## Important validation limitations

- **Host preflight is not a numerical validation.** First five iterations may fail at compile/runtime; show the exact output rather than changing multiple solver parameters.
- This is an **independent formulation comparison**, not parity against the earlier Re1M one-layer trace_log/deviatoric/SUPG case. Initial choices are one-point `sample_log`, advective convection, vector Laplacian, no SUPG, original hybrid segregated momentum.
- First test the mixed-mesh pressure Schur, interface mass-flux consistency, wall impermeability, finite k/omega and initial residual trends before attempting a longer run.
- Only one necessary CUDA header patch is applied. `make` is incremental, but the changed header triggers the large CUDA TU compilation once. The isolated copy keeps build objects/dependency records where present; do not `make clean`.
- No compulsory hash gates, staging, or runtime package installation.

## Troubleshooting

`bash "$EXP/RUN.sh" prepare` validates mesh and patches isolated source without building CUDA. `bash "$EXP/RUN.sh" dry` checks the exact native invocation. `bash "$EXP/RUN.sh" 5` incrementally compiles and executes the first 5 steps. Read `$HOME/NodalS_SST_LOCAL_RUNS/PIPE50_RE1M_HYBRID_99K_ONEPOINT/terminal.log` after the run starts.

## November 14 hybrid native-startup repair

After the first successful NVCC build, a mixed mesh hit the unconditional
`STAGE25_ROTATION currently supports pure-tet` startup exception. The opt-in
`fix_hybrid_startup.py` (automatically called from `RUN.sh`) now:
- Keeps all existing pure-tet startup operations unchanged.
- For mixed HEX/TET only, requires zero rotation and native HXT1 patch tags,
  skips the tetra-only Stage19 registry and performs a boundary-plan audit.
- Removes the `SST22D4_NITSCHE_ENABLE` variable entirely in hybrid mode;
  the native parser treats its mere presence as a request to enable D4D.
- Requires the straight-pipe wall-distance choice `analytic_pipe`; this is
  also set in the GitHub case. **Already copied working cases must be changed
  manually from `method = geometric` to `method = analytic_pipe`.**

Re-fetch the experiment worktree, edit the existing local case, run `bash
"$EXP/RUN.sh" dry` and only then `bash "$EXP/RUN.sh" 5`.
This patch modifies an existing app `.inc` and needs one incremental NVCC
rebuild. It has not yet been verified by a successful GPU iteration.
