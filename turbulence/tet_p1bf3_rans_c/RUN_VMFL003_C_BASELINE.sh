#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXE="${EXE:-$HERE/p1bf3_simple_foam_mpi_fp64}"
MESH="${MESH:-$HOME/Desktop/meshes/nodals_vmfl003_10D_radial460k/foam_case/constant/polyMesh}"
NP="${NP:-16}"

RE="${RE:-13691.7402481}"
NU="${NU:-1.460734693878e-5}"
BULK_VELOCITY="${BULK_VELOCITY:-50.0}"
INLET_NORMAL_SPEED="${INLET_NORMAL_SPEED:--50.0}"

ALPHA_U="${ALPHA_U:-0.70}"
ALPHA_P="${ALPHA_P:-1.00}"
SIMPLE_RTOL="${SIMPLE_RTOL:-1e-4}"
SIMPLE_MAX_IT="${SIMPLE_MAX_IT:-10000}"
P_KSP_RTOL="${P_KSP_RTOL:-0.9}"
P_PRECONDITIONER_REFRESH="${P_PRECONDITIONER_REFRESH:-1}"
WRITE_VTU="${WRITE_VTU:-1}"
MIXLEN_AUDIT="${MIXLEN_AUDIT:-1}"
MOMENTUM_BUDGET="${MOMENTUM_BUDGET:-0}"
OUT="${OUT:-$PWD/nodals_tet_rans_c_$(date +%Y%m%d_%H%M%S)}"
LOG="${LOG:-${OUT}.log}"
VTU="${VTU:-${OUT}.vtu}"

[[ -x "$EXE" ]] || { echo "ERROR: executable missing: $EXE"; exit 2; }
[[ -d "$MESH" ]] || { echo "ERROR: mesh missing: $MESH"; exit 3; }

export PETSC_OPTIONS="${PETSC_OPTIONS:-} -wall_sample_mode legacy_face_trace -wall_molecular_consistency 0 -momentum_budget ${MOMENTUM_BUDGET}"

mpirun -np "$NP" --map-by core --bind-to core "$EXE" \
  -mesh "$MESH" \
  -problem pipe \
  -re "$RE" \
  -nu "$NU" \
  -pipe_bulk_velocity "$BULK_VELOCITY" \
  -pipe_inlet_patch inlet \
  -pipe_outlet_patch outlet \
  -pipe_wall_patch wall \
  -initial_pipe_velocity plug \
  -inlet_bc fixed_normal_speed \
  -inlet_normal_speed "$INLET_NORMAL_SPEED" \
  -inlet_normal_mode average_patch_normal \
  -convection central \
  -mixing_length 1 \
  -mixlen_model nikuradse_pipe \
  -mixlen_scale 1.0 \
  -mixlen_strain_mode raw \
  -mixlen_audit "$MIXLEN_AUDIT" \
  -weak_wall_function 1 \
  -wall_law spalding \
  -wall_kappa 0.4 \
  -wall_B 5.5 \
  -wall_distance_factor 0.25 \
  -wall_beta_scale 1.0 \
  -wall_linearization consistent_tangent \
  -wall_tangent_blend 0.5 \
  -supg 0 \
  -simple_variant simple \
  -alpha_u "$ALPHA_U" \
  -alpha_p "$ALPHA_P" \
  -simple_rtol "$SIMPLE_RTOL" \
  -simple_max_it "$SIMPLE_MAX_IT" \
  -u_relax_mode row_l1 \
  -u_rel_drop 0.5 \
  -u_local_sweeps 1 \
  -u_sor_omega 1.0 \
  -rau_mode diag \
  -rau_scale 2.0 \
  -pressure_solve_mode petsc_fgmres \
  -p_operator factored \
  -p_pmat full \
  -p_preconditioner_refresh "$P_PRECONDITIONER_REFRESH" \
  -p_ksp_type fgmres \
  -p_ksp_rtol "$P_KSP_RTOL" \
  -p_ksp_atol 1e-12 \
  -p_ksp_divtol 1e8 \
  -p_ksp_max_it 1000 \
  -p_ksp_gmres_restart 30 \
  -p_pc_type gamg \
  -p_mg_levels_ksp_type chebyshev \
  -p_mg_levels_pc_type jacobi \
  -p_mg_levels_ksp_max_it 2 \
  -write_vtu "$WRITE_VTU" \
  -vtu_output "$VTU" \
  -vtu_velocity_mode both \
  -options_left \
  2>&1 | tee "$LOG"
