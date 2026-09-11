Gate 5G: VMFL003 20D uniform 10 axial slabs per diameter

Mesh:
  20D total length
  10 axial slabs per D
  200 slabs total
  dz = 0.1D
  2556 tets/slab
  511,200 tetrahedra total
  90,651 points

Physics/solver:
  FP32
  no SUPG
  Nikuradse mixing length
  DG plug inlet
  weak Spalding wall
  alphaU=0.7
  alphaP=1.0
  rAU scale=2
  SIMPLE tolerance=1e-3
  CF/PMIS pressure hierarchy
  Chebyshev AMG smoother degree 4, powerIts=16

Workflow:
  mesh_tools/BUILD_VMFL003_20D_10SLABS_PER_D.sh
  gpu/serial_cuda/RUN_RANS_GATE5G_20D_10SLABD_NOSUPG_FP32.sh
  postprocess/vtu_axial_friction/RUN_VMFL003_20D_10SLABD_FRICTION.sh
