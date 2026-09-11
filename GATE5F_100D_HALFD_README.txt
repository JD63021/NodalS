Gate 5F: 100D with original 20D fine region + 0.5D axial slabs downstream

Mesh:
  0-20D: original 360 slabs, preserved exactly from source20D.msh
  20-100D: 160 slabs, each dz=0.5D
  total slabs: 520
  total tets: 1,329,120
  total points: 234,971

Runner defaults deliberately reproduce the successful Gate5D stabilization:
  FP32
  SUPG=1, tauScale=0.05, magic=9
  pressure AMG smoother=cheb2
  AMG Chebyshev degree=4
  AMG power iterations=16
  alphaU=0.7
  alphaP=1.0
  rAU scale=2
  SIMPLE tolerance=1e-3

Workflow:
  mesh_tools/BUILD_VMFL003_100D_20DFINE_HALF_D.sh
  gpu/serial_cuda/RUN_RANS_GATE5F_100D_20DFINE_HALFD_FP32.sh
  postprocess/vtu_axial_friction/RUN_VMFL003_100D_HALFD_FRICTION.sh
