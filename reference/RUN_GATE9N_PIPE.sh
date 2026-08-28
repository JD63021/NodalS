#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Gate 9N: straight-pipe runner exposing the source's validated parabolic inlet.
# Parabolic mode uses -problem pipe, so exact Hagen-Poiseuille diagnostics are printed.

MESH_ROOT="${MESH_ROOT:-$HOME/Desktop/meshes/pipe}"
MESH_NAME="${MESH_NAME:-111k}"
MESH_PATH="${MESH_PATH:-$MESH_ROOT/$MESH_NAME/constant/polyMesh}"
CASE_NAME="${CASE_NAME:-${MESH_NAME}_Re${RE:-20}_${INLET_PROFILE:-parabolic}}"
NP="${NP:-16}"
TIMEOUT_S="${TIMEOUT_S:-0}"

# Pipe-only geometry convention for the existing exact +z Hagen-Poiseuille path:
# patch_2_0 is z-min inlet (outward normal -z), patch_1_0 is z-max outlet (+z).
WALL_PATCH="${WALL_PATCH:-patch_0_0}"
INLET_PATCH="${INLET_PATCH:-patch_2_0}"
OUTLET_PATCH="${OUTLET_PATCH:-patch_1_0}"
INLET_PROFILE="${INLET_PROFILE:-parabolic}"          # parabolic | plug
BULK_VELOCITY="${BULK_VELOCITY:-1.0}"              # mean velocity for parabolic profile
INLET_SPEED="${INLET_SPEED:-$BULK_VELOCITY}"       # positive inward magnitude for plug profile
INLET_SIGNED_SPEED="${INLET_SIGNED_SPEED:--$INLET_SPEED}"
RE="${RE:-20}"                                      # diameter-based Re when NU is unset
NU="${NU:-}"                                        # optional override; blank => nu=Ubulk*D/Re from actual mesh

CONVECTION="${CONVECTION:-central}"
SUPG="${SUPG:-0}"
TAU_SCALE="${TAU_SCALE:-0.05}"
SUPG_FORM="${SUPG_FORM:-implicit}"
SUPG_KERNEL="${SUPG_KERNEL:-fast}"
SUPG_QUAD_POINTS="${SUPG_QUAD_POINTS:-64}"

# SIMPLE / SIMPLEC and under-relaxation.
SIMPLE_VARIANT="${SIMPLE_VARIANT:-simplec}"
SIMPLE_RTOL="${SIMPLE_RTOL:-1e-3}"
SIMPLE_MAX_ITS="${SIMPLE_MAX_ITS:-10000}"
ALPHA_U="${ALPHA_U:-0.7}"
ALPHA_P="${ALPHA_P:-1.0}"
U_RELAX_MODE="${U_RELAX_MODE:-row_l1}"
RAU_MODE="${RAU_MODE:-diag}"
RAU_SCALE="${RAU_SCALE:-1}"
SIMPLEC_BLEND="${SIMPLEC_BLEND:-1}"
SIMPLEC_FLOOR_FRACTION="${SIMPLEC_FLOOR_FRACTION:-1e-6}"
SIMPLEC_FALLBACK="${SIMPLEC_FALLBACK:-diag}"

# Momentum predictor controls.
U_RTOL="${U_RTOL:-1e-8}"
U_ATOL="${U_ATOL:-0}"
U_REL_DROP="${U_REL_DROP:-0.5}"
U_MAX_ITS="${U_MAX_ITS:-20000}"
U_CHECK_EVERY="${U_CHECK_EVERY:-1}"
U_LOCAL_SWEEPS="${U_LOCAL_SWEEPS:-1}"
U_SOR_OMEGA="${U_SOR_OMEGA:-1}"

# Pressure backend and Pmat selection.
PRESSURE_SOLVER="${PRESSURE_SOLVER:-cheb_gamg}"      # cheb_gamg | pcg_gamg
PMAT="${PMAT:-compact}"                              # compact | full
P_RTOL="${P_RTOL:-0.1}"
P_ATOL="${P_ATOL:-1e-12}"
P_DTOL="${P_DTOL:-1e8}"
P_MAX_ITS="${P_MAX_ITS:-300}"
P_REFRESH="${P_REFRESH:-1}"
P_PC_TYPE="${P_PC_TYPE:-gamg}"
P_MG_LEVEL_KSP="${P_MG_LEVEL_KSP:-richardson}"
P_MG_LEVEL_PC="${P_MG_LEVEL_PC:-jacobi}"
P_MG_LEVEL_ITS="${P_MG_LEVEL_ITS:-1}"

# Chebyshev controls.
SPECTRUM_REFRESH="${SPECTRUM_REFRESH:-100}"
POWER_ITS="${POWER_ITS:-10}"
LMAX_SAFETY="${LMAX_SAFETY:-1.2}"
LMIN_FRACTION="${LMIN_FRACTION:-0.06}"
CHEB_MODE="${CHEB_MODE:-adaptive}"                   # adaptive | fixed
CHEB_STEPS="${CHEB_STEPS:-40}"
CHEB_REQUIRE_TARGET="${CHEB_REQUIRE_TARGET:-0}"
CHEB_INITIAL_BLOCK="${CHEB_INITIAL_BLOCK:-8}"
CHEB_EXTEND_BLOCK="${CHEB_EXTEND_BLOCK:-4}"
CHEB_MAX_STEPS="${CHEB_MAX_STEPS:-40}"
CHEB_DIVERGENCE_FACTOR="${CHEB_DIVERGENCE_FACTOR:-1e8}"

WRITE_VTU="${WRITE_VTU:-1}"
VTU_VELOCITY_MODE="${VTU_VELOCITY_MODE:-cell_average}"
RESOURCE_PROFILE="${RESOURCE_PROFILE:-1}"
MEMORY_AUDIT="${MEMORY_AUDIT:-0}"
MESH_AUDIT_ONLY="${MESH_AUDIT_ONLY:-0}"

case "$INLET_PROFILE" in
  parabolic|plug) ;;
  *) echo "P1BF3_GATE9N_RUNNER_RESULT=FAIL reason=bad_inlet_profile value=$INLET_PROFILE expected=parabolic_or_plug"; exit 4 ;;
esac
case "$PRESSURE_SOLVER" in
  pcg_gamg|cheb_gamg) ;;
  *) echo "P1BF3_GATE9N_RUNNER_RESULT=FAIL reason=bad_pressure_solver value=$PRESSURE_SOLVER"; exit 4 ;;
esac
case "$PMAT" in
  compact) PMAT_ARG="native_face" ;;
  full) PMAT_ARG="full" ;;
  *) echo "P1BF3_GATE9N_RUNNER_RESULT=FAIL reason=bad_pmat value=$PMAT expected=compact_or_full"; exit 4 ;;
esac
case "$CHEB_MODE" in
  adaptive|fixed) ;;
  *) echo "P1BF3_GATE9N_RUNNER_RESULT=FAIL reason=bad_cheb_mode value=$CHEB_MODE"; exit 4 ;;
esac

[[ -d "$MESH_PATH" ]] || { echo "P1BF3_GATE9N_RUNNER_RESULT=FAIL reason=missing_mesh path=$MESH_PATH"; exit 3; }
if [[ "$INLET_PROFILE" == "parabolic" ]]; then
  # Existing exact pipe solution is +z. Protect the convergence diagnostics from a sign-reversed setup.
  if [[ "$INLET_PATCH" != "patch_2_0" || "$OUTLET_PATCH" != "patch_1_0" ]]; then
    echo "P1BF3_GATE9N_RUNNER_RESULT=FAIL reason=parabolic_exact_path_requires_plus_z inlet=$INLET_PATCH outlet=$OUTLET_PATCH expectedInlet=patch_2_0 expectedOutlet=patch_1_0"
    exit 5
  fi
  INLET_BC_ARG="parabolic"
else
  INLET_BC_ARG="fixed_normal_speed"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTROOT="${OUTROOT:-$HOME/Downloads/P1BF3_GATE9N_${CASE_NAME}_${PRESSURE_SOLVER}_${PMAT}_${STAMP}}"
mkdir -p "$OUTROOT"
LOG="$OUTROOT/${CASE_NAME}_gate9n.log"
VTU_OUTPUT="${VTU_OUTPUT:-$OUTROOT/${CASE_NAME}_gate9n_solution.vtu}"

bash "$ROOT/BUILD_FP64_OPT.sh" 2>&1 | tee "$OUTROOT/build.log"
EXE="$ROOT/p1bf3_simple_foam_mpi"
[[ -x "$EXE" ]] || { echo "P1BF3_GATE9N_RUNNER_RESULT=FAIL reason=missing_exe exe=$EXE"; exit 2; }

if [[ "$CHEB_MODE" == "fixed" ]]; then CHEB_FIXED_STEPS="$CHEB_STEPS"; else CHEB_FIXED_STEPS=0; fi
if [[ "$PRESSURE_SOLVER" == "cheb_gamg" ]]; then GATE9I=1; else GATE9I=0; fi

cat <<CFG
================ P1BF3 GATE9N PIPE CONFIG ================
case=$CASE_NAME mesh=$MESH_PATH np=$NP
wall=$WALL_PATCH inlet=$INLET_PATCH outlet=$OUTLET_PATCH inletProfile=$INLET_PROFILE
bulkVelocity=$BULK_VELOCITY plugSpeed=$INLET_SPEED ReDiameter=$RE nuOverride=${NU:-<derived Ubulk*D/Re>}
SIMPLE variant=$SIMPLE_VARIANT alphaU=$ALPHA_U alphaP=$ALPHA_P outerRtol=$SIMPLE_RTOL maxIts=$SIMPLE_MAX_ITS
velocity rtol=$U_RTOL atol=$U_ATOL relDrop=$U_REL_DROP localSweeps=$U_LOCAL_SWEEPS omega=$U_SOR_OMEGA
pressure solver=$PRESSURE_SOLVER pmat=$PMAT rtol=$P_RTOL atol=$P_ATOL pcRefresh=$P_REFRESH
cheb mode=$CHEB_MODE powerIts=$POWER_ITS spectrumRefresh=$SPECTRUM_REFRESH lmaxSafety=$LMAX_SAFETY lminFraction=$LMIN_FRACTION maxSteps=$CHEB_MAX_STEPS requireTarget=$CHEB_REQUIRE_TARGET
convection=$CONVECTION supg=$SUPG tauScale=$TAU_SCALE supgForm=$SUPG_FORM supgKernel=$SUPG_KERNEL supgQuad=$SUPG_QUAD_POINTS
===========================================================
CFG

ARGS=(
  -mesh "$MESH_PATH"
  -problem pipe
  -pipe_wall_patch "$WALL_PATCH" -pipe_inlet_patch "$INLET_PATCH" -pipe_outlet_patch "$OUTLET_PATCH"
  -inlet_bc "$INLET_BC_ARG" -pipe_bulk_velocity "$BULK_VELOCITY" -re "$RE"
  -convection "$CONVECTION"
  -supg "$SUPG" -supg_tau_scale "$TAU_SCALE" -supg_form "$SUPG_FORM" -supg_kernel "$SUPG_KERNEL" -supg_quad_points "$SUPG_QUAD_POINTS"

  -simple_variant "$SIMPLE_VARIANT" -alpha_u "$ALPHA_U" -alpha_p "$ALPHA_P"
  -u_relax_mode "$U_RELAX_MODE" -rau_mode "$RAU_MODE" -rau_scale "$RAU_SCALE"
  -simplec_blend "$SIMPLEC_BLEND" -simplec_floor_fraction "$SIMPLEC_FLOOR_FRACTION" -simplec_fallback "$SIMPLEC_FALLBACK"
  -simple_rtol "$SIMPLE_RTOL" -simple_max_it "$SIMPLE_MAX_ITS"

  -u_rtol "$U_RTOL" -u_atol "$U_ATOL" -u_rel_drop "$U_REL_DROP" -u_max_it "$U_MAX_ITS"
  -u_check_every "$U_CHECK_EVERY" -u_local_sweeps "$U_LOCAL_SWEEPS" -u_sor_omega "$U_SOR_OMEGA"

  -pressure_solve_mode custom_pcg
  -p_pmat "$PMAT_ARG" -p_operator factored -p_preconditioner_refresh "$P_REFRESH"
  -p_ksp_type cg -p_ksp_rtol "$P_RTOL" -p_ksp_atol "$P_ATOL" -p_ksp_divtol "$P_DTOL" -p_ksp_max_it "$P_MAX_ITS"
  -p_pc_type "$P_PC_TYPE"
  -p_mg_levels_ksp_type "$P_MG_LEVEL_KSP" -p_mg_levels_pc_type "$P_MG_LEVEL_PC" -p_mg_levels_ksp_max_it "$P_MG_LEVEL_ITS"

  -gate9g_fe_face_energy 1 -gate9g_richardson 0 -gate9h_chebyshev 0 -gate9i_auto_chebyshev "$GATE9I"
  -gate9i_power_its "$POWER_ITS" -gate9i_safety "$LMAX_SAFETY" -gate9i_lambda_min_fraction "$LMIN_FRACTION"
  -gate9i_rtol "$P_RTOL" -gate9i_atol "$P_ATOL" -gate9i_spectrum_refresh "$SPECTRUM_REFRESH"
  -gate9i_fixed_steps "$CHEB_FIXED_STEPS" -gate9i_require_target "$CHEB_REQUIRE_TARGET"
  -gate9i_initial_block "$CHEB_INITIAL_BLOCK" -gate9i_extend_block "$CHEB_EXTEND_BLOCK" -gate9i_max_steps "$CHEB_MAX_STEPS"
  -gate9d_divergence_factor "$CHEB_DIVERGENCE_FACTOR"

  -gate1_compare_pcg 0 -gate3_mp_probe 0 -gate5_kp_gamg_probe 0 -gate6_diffusion_pcd_probe 0 -gate7_cp_probe 0 -gate8_esw_bc_probe 0 -gate9e_ngfv 0
  -m3_static_reference 0 -m4b_b_reference 0 -m5b_pcg_reference 0 -m6b_velocity_reference 0
  -custom_pressure_b_shadow_tol 5e-12 -custom_pressure_pcg_reference_tol 5e-10
  -m10_pcg_profile 0 -pressure_profile 0
  -write_vtu "$WRITE_VTU" -vtu_output "$VTU_OUTPUT" -vtu_velocity_mode "$VTU_VELOCITY_MODE"
  -resource_profile "$RESOURCE_PROFILE" -memory_audit "$MEMORY_AUDIT" -mesh_audit_only "$MESH_AUDIT_ONLY"
  -options_left
)
if [[ "$INLET_PROFILE" == "plug" ]]; then ARGS+=( -inlet_normal_mode average_patch_normal -inlet_normal_speed "$INLET_SIGNED_SPEED" ); fi
if [[ -n "$NU" ]]; then ARGS+=( -nu "$NU" ); fi

CMD=(mpirun -np "$NP" --map-by core --bind-to core --report-bindings stdbuf -oL -eL "$EXE" "${ARGS[@]}")
printf 'P1BF3_GATE9N_COMMAND'; printf ' %q' "${CMD[@]}"; echo

set +e
if [[ "$TIMEOUT_S" != "0" ]]; then timeout --preserve-status "${TIMEOUT_S}s" "${CMD[@]}" 2>&1 | tee "$LOG"; else "${CMD[@]}" 2>&1 | tee "$LOG"; fi
RC=${PIPESTATUS[0]}
set -e

UNUSED="$(grep -Eic 'There are [1-9][0-9]* unused options|unused option[:=]' "$LOG" || true)"
echo
echo "================ GATE 9N PIPE SUMMARY ================"
grep -E 'P1BF3_PIPE_GEOMETRY|P1BF3_PIPE_BC|P1BF3_PRESSURE_SOLVER|P1BF3_GATE9I_SUMMARY|P1BF3_WORK |P1BF3_PIPE_ERROR|P1BF3_PIPE_FLOW|P1BF3_PIPE_PRESSURE|P1BF3_NS_FINAL|P1BF3_TIMING |P1BF3_RESULT |P1BF3_SOLVE_FAILURE' "$LOG" | tail -80 || true
echo "P1BF3_GATE9N_ARTIFACTS log=$LOG vtu=$VTU_OUTPUT outroot=$OUTROOT"

if [[ "$RC" -ne 0 ]]; then echo "P1BF3_GATE9N_RUNNER_RESULT=FAIL reason=nonzero_rc rc=$RC log=$LOG"; exit "$RC"; fi
if [[ "$UNUSED" -ne 0 ]]; then echo "P1BF3_GATE9N_RUNNER_RESULT=FAIL reason=unused_options count=$UNUSED log=$LOG"; exit 20; fi
if [[ "$MESH_AUDIT_ONLY" == "1" ]]; then echo "P1BF3_GATE9N_RUNNER_RESULT=PASS_AUDIT_ONLY case=$CASE_NAME log=$LOG"; exit 0; fi
if ! grep -q 'P1BF3_RESULT status=PASS' "$LOG"; then echo "P1BF3_GATE9N_RUNNER_RESULT=NOT_CONVERGED case=$CASE_NAME log=$LOG"; exit 21; fi

echo "P1BF3_GATE9N_RUNNER_RESULT=PASS case=$CASE_NAME inletProfile=$INLET_PROFILE Re=$RE pressureSolver=$PRESSURE_SOLVER pmat=$PMAT log=$LOG vtu=$VTU_OUTPUT"
