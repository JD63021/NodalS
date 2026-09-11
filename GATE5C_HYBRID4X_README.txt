NodalS GPU RANS Gate 5C: VMFL003 20D hybrid 4x axial aspect-ratio experiment

Purpose:
  Preserve the original 0-10D mesh exactly (D/18 axial spacing).
  From 10-20D, use 4x the original axial spacing (2D/9), i.e. 2x longer again than Gate 5B.

Expected topology:
  0-10D: 180 slabs
  10-20D: 45 slabs
  total: 225 slabs
  cross-section: 451 points
  tets/slab: 2556
  total tets: 575100
  total points: 101926

Workflow:
  mesh_tools/BUILD_VMFL003_20D_HYBRID4X.sh
  gpu/serial_cuda/RUN_RANS_GATE5C_20D_HYBRID4X_FP32.sh
  postprocess/vtu_axial_friction/RUN_VMFL003_20D_HYBRID4X_FRICTION.sh
