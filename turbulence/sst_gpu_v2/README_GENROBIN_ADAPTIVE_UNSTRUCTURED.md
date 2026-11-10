# NodalS canonical adaptive sampling for generalized Robin-Nitsche SST

Branch: exp/gpu-genrobin-adaptive-unstructured-20261110
Parent: exp/gpu-genrobin-skew-supg-20261110

## Failure resolved

Previously [adaptive_sampling] in a .case was metadata when executing
`python3 turbulence/sst_gpu_v2/nodals.py run` directly. Sampling scales
were frozen even when enabled=true. A separate example-specific script
had to be launched explicitly, and the original Python wrapper also
rejected coupled FGMRES + full momentum SUPG.

Now the normal `nodals.py run` dispatches to
`turbulence/sst_gpu_v2/adaptive_run.py` iff enabled=true. It schedules
the existing native F2 facewise fit/export and Stage31 checkpoint/restart,
without changing CUDA operators, physical wall fitting, or BF3 constraints.

For `enabled=false`, the previous direct native run path remains unchanged.
The adaptive child commands pass `adaptive_sampling.enabled=false` last,
thereby preventing infinite recursive dispatch. All other .case values,
including physical conditions, stress/convection, underrelaxation, Nitsche
stabilization and SUPG, remain editable.

## U-bend cases

- `examples/ubend_unstructured_200k/UBEND_MIRROR_GENROBIN_SKEW_SUPG_ADAPTIVE.case`
  for the 199,584-tet mirror-symmetric mesh.
- `examples/ubend_unstructured_200k/UBEND_UNSTRUCTURED_GENROBIN_SKEW_SUPG_ADAPTIVE.case`
  for original 200,245-tet unstructured mesh.

Both cases use Re=72600, D=.0889m, nu=1.460734693878e-5, bulk
11.9290594798 m/s, developed profile CSV (interpolated, no rescaling),
F2 one-point Spalding, OF-auto omega index 3, generalized Robin C=15
viscosity=nu, skew convection, full momentum SUPG tau=0.05, and
coupled FGMRES.

Initial scales are .10/.10. [adaptive_sampling] enabled=true,
first_refresh=50, refresh_every=500, max_refreshes=1,
target_spalding_yplus=30, target_omega_yplus=30, min_scale=.05,
max_scale=.90. Schedule: stages [0,50] and [50,500].
The facewise sample heights are frozen inside each SIMPLE segment;
the controller restarts at the segment transition. This is not a
within-one-kernel continuously refreshed sampling method.

The native facewise loader requires pure tetrahedra, F2 face Spalding,
one active Spalding point, omega sample index 3 and ray_trace. It computes
locally y_target = target_yplus * nu / u_tau, with face-dependent bounds,
and reports numbers of clamped and fallback samples. The resulting
area-mean y+ need not be exactly 30 as u_tau evolves.

## Recommended workstation verification

```bash
cd ~/NodalS_GPU_GENROBIN_SUPG
git fetch origin
git worktree add --detach ~/NodalS_GPU_GENROBIN_ADAPTIVE \
  origin/exp/gpu-genrobin-adaptive-unstructured-20261110

APP=$HOME/NodalS_GPU_GENROBIN_ADAPTIVE/turbulence/sst_gpu_v2
BIN=$HOME/NodalS_GPU_GENROBIN_SUPG/turbulence/sst_gpu_v2/source/puretet_profile5_eq5_hxt5b/nodals_hxt5b_fp64

python3 "$APP/tests/test_adaptive_dispatch.py"
python3 "$APP/tests/test_generalized_robin_algebra.py"
python3 "$APP/tests/test_genrobin_skew_supg_wiring.py"
```

No rebuild is required if the existing FP64 native executable from the
genrobin-skew-SUPG branch already works: this branch modifies only Python,
tests and .case files. To rebuild cleanly if desired, compile in the new
worktree and pass its binary explicitly instead.

```bash
ROOT=$HOME/NodalS_GPU_GENROBIN_ADAPTIVE
APP=$ROOT/turbulence/sst_gpu_v2
BIN=$HOME/NodalS_GPU_GENROBIN_SUPG/turbulence/sst_gpu_v2/source/puretet_profile5_eq5_hxt5b/nodals_hxt5b_fp64
CASE=$ROOT/examples/ubend_unstructured_200k/UBEND_UNSTRUCTURED_GENROBIN_SKEW_SUPG_ADAPTIVE.case
OUT=$HOME/NodalS_SST_LOCAL_RUNS/ubend_unstructured_genrobin_auto_$(date +%Y%m%d_%H%M%S)

xdg-open "$CASE"
python3 -u "$APP/nodals.py" run --case "$CASE" --binary "$BIN" \
 --steps 500 --output "$OUT" --native-option mom-cap-policy=fail
```

Expected diagnostics:
- `NODALS_ADAPTIVE_DISPATCH=ON`
- `NODALS_ADAPTIVE_PLAN ... refreshAt=[50]`
- `stage_0000_0050/NODALS_STAGE31_STATE_it50.ckpt`
- `stage_0000_0050/NODALS_FACE_UTAU_it50.csv`
- `stage_0050_0500/terminal.log` contains `STAGE31_RESTART_LOAD=PASS`
  and `NODALS_FACE_YPLUS_LOADED ... status=PASS`.

If a segment ends before its checkpoint is saved, tighten SIMPLE stopping
criteria (not the physical wall law). An output path is never overwritten
unless explicit --resume is used. This integration has not been CUDA-run
on the user's machine; confirm the above logs before trusting y+ metrics.
