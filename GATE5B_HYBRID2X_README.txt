VMFL003 20D HYBRID2X mesh experiment

Purpose:
- preserve the existing 20D radial cross-section and exact first-slab tet template
- 0-10D: keep original D/18 axial spacing (180 slabs), identical to current mesh
- 10-20D: use 2x longer D/9 axial spacing (90 slabs)
- total: 270 slabs, 690120 tetrahedra, 122221 points

This isolates the downstream axial aspect-ratio effect while leaving the first 10D development region unchanged.

Workflow:
1) mesh_tools/BUILD_VMFL003_20D_HYBRID2X.sh
2) gpu/serial_cuda/RUN_RANS_GATE5B_20D_HYBRID2X_FP32.sh
3) postprocess/vtu_axial_friction/RUN_VMFL003_20D_HYBRID2X_FRICTION.sh

Solver defaults remain the successful Gate5 controls:
alphaU=0.7 alphaP=1 rauScale=2 simpleTol=1e-3 FP32.
