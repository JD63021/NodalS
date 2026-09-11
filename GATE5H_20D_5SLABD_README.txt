Gate 5H: VMFL003 20D uniform 5 axial slabs per diameter

Mesh:
  20D total length
  5 axial slabs per D
  100 slabs total
  dz = 0.2D
  2556 tets/slab
  255,600 tetrahedra total
  45,551 points

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
  mesh_tools/BUILD_VMFL003_20D_5SLABS_PER_D.sh
  gpu/serial_cuda/RUN_RANS_GATE5H_20D_5SLABD_NOSUPG_FP32.sh
  postprocess/vtu_axial_friction/RUN_VMFL003_20D_5SLABD_FRICTION.sh
