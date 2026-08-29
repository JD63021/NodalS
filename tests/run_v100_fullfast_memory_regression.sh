#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PETSC_DIR="${PETSC_DIR:-$HOME/src/petsc}"
PETSC_ARCH="${PETSC_ARCH:-arch-linux-cuda-opt}"
NP="${NP:-16}"
MESH_ROOT="${MESH_ROOT:-$HOME/Desktop/meshes/pipe}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUTROOT:-$HOME/Downloads/NodalS_v100_GOOD_ORACLE_REGRESSION_${STAMP}}"
mkdir -p "$OUT"

EXPECTED=0de1a33840d0f36fa84dd7a301bb2073c3dc344cb958defeb0bd7390334d4d01
python3 tests/check_source_freeze.py

[[ -f "$PETSC_DIR/$PETSC_ARCH/include/petscconf.h" ]] || {
  echo "NODALS_V100_MEMORY_REGRESSION status=FAIL reason=missing_petsc PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH" >&2; exit 2; }
for m in 40k 111k 292k; do
  [[ -d "$MESH_ROOT/$m/constant/polyMesh" ]] || {
    echo "NODALS_V100_MEMORY_REGRESSION status=FAIL reason=missing_mesh mesh=$m path=$MESH_ROOT/$m/constant/polyMesh" >&2; exit 3; }
done

echo "NODALS_V100_REGRESSION_CONFIG oracle=$EXPECTED PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH NP=$NP MESH_ROOT=$MESH_ROOT OUT=$OUT"

# Force rebuild because make does not know PETSC_DIR/PETSC_ARCH changed.
rm -f app/nodals_main.o nodals_solver
make -j"$(nproc)" PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH" nodals_solver
ldd ./nodals_solver | tee "$OUT/ldd.txt"
PETSC_SO="$(ldd ./nodals_solver | awk '/libpetsc\.so/{print $3; exit}')"
case "$PETSC_SO" in
  "$PETSC_DIR/$PETSC_ARCH/lib/"*) ;;
  *) echo "NODALS_V100_MEMORY_REGRESSION status=FAIL reason=wrong_petsc_link resolved=$PETSC_SO" >&2; exit 4;;
esac

run_case() {
  local m="$1" mesh="$MESH_ROOT/$m/constant/polyMesh" log="$OUT/$m.log"
  echo
  echo "================ NodalS v1.00 GOOD ORACLE $m ================"
  set +e
  mpirun -np "$NP" --map-by core --bind-to core --report-bindings stdbuf -oL -eL ./nodals_solver \
    -require_petsc_fp32 0 \
    -mesh "$mesh" -problem pipe \
    -pipe_wall_patch patch_0_0 -pipe_inlet_patch patch_2_0 -pipe_outlet_patch patch_1_0 \
    -inlet_bc parabolic -pipe_bulk_velocity 1.0 -re 20 \
    -convection central -supg 0 -supg_tau_scale 0.05 -supg_form implicit -supg_kernel fast -supg_quad_points 64 \
    -simple_variant simple -alpha_u 0.5 -alpha_p 0.5 \
    -u_relax_mode row_l1 -rau_mode diag -rau_scale 1 \
    -simplec_blend 1 -simplec_floor_fraction 1e-6 -simplec_fallback diag \
    -simple_rtol 1e-6 -simple_max_it 4000 \
    -u_rtol 1e-8 -u_atol 0 -u_rel_drop 0.1 -u_max_it 20000 -u_check_every 1 -u_local_sweeps 1 -u_sor_omega 1 \
    -pressure_solve_mode gamg_richardson \
    -p_pmat full -p_operator factored -p_preconditioner_refresh 100 \
    -p_ksp_type cg -p_ksp_rtol 0.5 -p_ksp_atol 1e-12 -p_ksp_divtol 1e8 -p_ksp_max_it 300 \
    -p_pc_type gamg -p_mg_levels_ksp_type richardson -p_mg_levels_pc_type jacobi -p_mg_levels_ksp_max_it 1 \
    -gate9g_fe_face_energy 0 -gate9g_richardson 0 -gate9h_chebyshev 0 -gate9i_auto_chebyshev 0 \
    -gate9i_power_its 10 -gate9i_safety 1.2 -gate9i_lambda_min_fraction 0.06 \
    -gate9i_rtol 0.5 -gate9i_atol 1e-12 -gate9i_spectrum_refresh 100 \
    -gate9i_fixed_steps 0 -gate9i_require_target 0 -gate9i_initial_block 8 -gate9i_extend_block 4 -gate9i_max_steps 80 \
    -gate9d_divergence_factor 1e8 \
    -gate1_compare_pcg 0 -gate3_mp_probe 0 -gate5_kp_gamg_probe 0 -gate6_diffusion_pcd_probe 0 \
    -gate7_cp_probe 0 -gate8_esw_bc_probe 0 -gate9e_ngfv 0 \
    -m3_static_reference 0 -m4b_b_reference 0 -m5b_pcg_reference 0 -m6b_velocity_reference 0 \
    -m1m2_legacy_reference 0 -dynamic_plan_compact 1 \
    -custom_pressure_b_shadow_tol 5e-12 -custom_pressure_pcg_reference_tol 5e-10 \
    -m10_pcg_profile 0 -pressure_profile 0 \
    -write_vtu 0 -resource_profile 1 -memory_audit 1 -mesh_audit_only 0 -distributed_mesh 1 \
    -options_left 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  set -e
  [[ $rc -eq 0 ]] || { echo "NODALS_V100_CASE status=FAIL mesh=$m rc=$rc log=$log"; exit "$rc"; }
  if grep -Eq 'There are [1-9][0-9]* unused options|unused option[:=]' "$log"; then
    echo "NODALS_V100_CASE status=FAIL mesh=$m reason=unused_options log=$log"; exit 20
  fi
  grep -q '^P1BF3_RESULT status=PASS' "$log" || { echo "NODALS_V100_CASE status=FAIL mesh=$m reason=solver_not_pass log=$log"; exit 21; }
  echo "NODALS_V100_CASE status=PASS mesh=$m log=$log"
}

for m in 40k 111k 292k; do run_case "$m"; done
python3 tests/analyze_v100_fullfast_memory_regression.py "$OUT" | tee "$OUT/summary.txt"
echo "NODALS_V100_MEMORY_REGRESSION_OUT=$OUT"
