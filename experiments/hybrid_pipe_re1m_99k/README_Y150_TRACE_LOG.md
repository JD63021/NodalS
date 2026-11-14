# Re=1M hybrid pipe: y+150 wall-HEX centroid + OF_AUTO trace_log

Experimental branch: `exp/gpu-sst-hybrid-pipe-re1m-y150-tracelog-20261114`.

**Motivation.** The previous y+300/single-point-sample_log hybrid run was unstable.
Its iteration-1000 fitted local pressure factor over 20–50D was 0.036135885
(+210.31% relative to Moody 0.011645041). A linear pressure profile alone
does not imply a physically correct momentum solution.

## Deliberate, isolated changes

1. **Mesh:** preserve the original 99,216-cell HXT0 mesh exactly as to
   HEX/TET connectivity (3,744 wall HEX and 95,472 interior TET). Rescale
   *interior radial coordinates only*; outer wall coordinates and axial mesh
   remain unchanged. New wall HEX depth 6.990327 mm (was 13.980654 mm);
   midpoint sample distance 3.495164 mm => *nominal* y+ = 150 computed
   using the Moody-reference `u_tau = U_b sqrt(f/8) = 0.6268954 m/s`.
   This is not an imposed actual y+; measured y+ will respond to computed
   shear. `HXT1_mesh.bin` is generated locally and must pass the original
   positive Jacobian / interface / BF2 checks.

2. **SST omega:** keep `omega_wall.method=of_auto` and select
   `stage41_mode=trace_log`. Pure-tet Stage41 physical-wall trace applied
   only on tetrahedral boundary faces, but this mesh has HEX boundary faces.
   `enable_hex_trace_log.py` connects that option to the **already existing**
   `OMEGA_WALL_TRACE_LOG` HEX kernel. It uses the single midpoint Spalding
   friction velocity and `of_kappa=0.4` in
   `omega_log = u_tau/(sqrt(beta*)*kappa*y_sample)`, then enforces that
   target at the **physical HEX wall trace** through symmetric Nitsche:
   `-<mu dn omega,v> - <mu dn v,omega> + gamma*mu/h<omega,v>`.
   The physical HEX wall remains high-Re; k is natural zero gradient.
   `trace_gamma=50`, no trace5 profile and wall production remains OFF.

3. **Unchanged:** HEX/TET interface Nitsche `gamma=50`, original
   advective velocity form/vector-Laplacian and SIMPLE/Schur/AMG settings,
   single-point Spalding traction, masked-Q1 SST scalar transport,
   plug inlet, Re1M/D=.889 m, and no SUPG.

## GitHub-only workstation workflow

```bash
cd "$HOME/NodalS"
BRANCH="exp/gpu-sst-hybrid-pipe-re1m-y150-tracelog-20261114"
TREE="$HOME/NODALS_HYBRID_RE1M_Y150_GIT"
git fetch origin "$BRANCH"
git worktree add --detach "$TREE" "origin/$BRANCH"
EXP="$TREE/experiments/hybrid_pipe_re1m_99k"
bash "$EXP/RUN_Y150_TRACE_LOG.sh" prepare
xdg-open "$HOME/NODALS_HYBRID_RE1M_Y150_TRACE_RUN/cases/PIPE50_RE1M_HYBRID_99K_Y150_TRACE_LOG.case"
bash "$EXP/RUN_Y150_TRACE_LOG.sh" dry
bash "$EXP/RUN_Y150_TRACE_LOG.sh" 5
```

No ChatGPT archive download. The **previous isolated** source,
`$HOME/NodalS_GPU_HYBRID_RE1M_99K/turbulence/sst_gpu_v2`, is reused,
while mesh, case, output and checkpoints use **new directories**.
A single incremental NVCC rebuild is unavoidable because the HEX omega
host dispatch and constant-kappa line changed. Never `make clean`.
After the 5-iteration smoke, changes to case/mesh require no compile.

The runner prefers the previous mesh HXT0 at
`$HOME/NODALS_HYBRID_RE1M_99K_RUN/mesh/HXT0.npz`.
If unavailable, it uses the normal generator with `--sample-yplus 150`,
which may not preserve the exact previous interior topology. The standalone
`rescale_hybrid_wall_layer.py` is therefore the preferred controlled route.

Once five iterations execute, run a fresh 100-iteration check (no build):
```bash
APP="$HOME/NodalS_GPU_HYBRID_RE1M_99K/turbulence/sst_gpu_v2"
CASE="$HOME/NODALS_HYBRID_RE1M_Y150_TRACE_RUN/cases/PIPE50_RE1M_HYBRID_99K_Y150_TRACE_LOG.case"
python3 -u "$APP/nodals.py" run --case "$CASE" --steps 100 \
  --set output.print_every=10 --set output.metrics_every=10
```

Verify the *actual HEX kernel* logs:
`NODALS_HYBRID_HEX_TRACE_LOG enabled=1 ... PHYSICAL_HEX_WALL_TRACE`.
The generic native Stage41 startup message may still label the older
TET trace path; the HEX-specific dispatch diagnostic is decisive.

For local fitted Moody pressure errors:
```bash
RUN="$HOME/NodalS_SST_LOCAL_RUNS/PIPE50_RE1M_HYBRID_99K_Y150_TRACE_LOG"
bash "$EXP/POSTPROCESS_PIPE50_HYBRID.sh" "$RUN" latest
```

## Verification and limitations

The y+150 **radial-only rescale** was separately host-tested on the original
99,216-cell HXT0 mesh with `prepare_hxt1.py`: positive HEX/TET
Jacobians, interface matching, velocity basis and boundary mapping all PASS.
These are mesh/host checks, not a CUDA or numerical validation.

The opt-in CUDA patch is source-reviewed but **not yet compiled or executed
on the workstation**. Do not claim a solution exists before the GPU smoke,
and do not interpret convergence or friction from initial five iterations.
If residual growth persists even with trace_log, audit the mixed interface
mass/pressure Schur and the axial momentum budget rather than adjusting
relaxations or penalties indiscriminately.
