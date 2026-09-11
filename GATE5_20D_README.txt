Gate 5 adds no RANS physics. It promotes the successful FP32 10D controls to the
920,160-cell 20D mesh, adds an optional after-loop VTU writer, and adds a fully
separate VTU pressure-gradient/friction postprocessor.

Default 20D solver controls:
  precision=fp32
  alphaU=0.70
  alphaP=1.00
  rAU scale=2.0
  SIMPLE tolerance=1e-3
  max outer=3000
  pressure=PCG, rtol=0.9
  momentum work=fgs1
  mixing length=ON
  DG numerical-trace plug inlet=ON
  weak Spalding wall=ON
  SUPG=OFF

VTU fields:
  CellData p_P0: solved cell pressure
  CellData U0: exact P1+BF3 element-volume-average velocity
  PointData U0_stream: volume-weighted neighboring-cell U0 for visualization
  solve_converged, outer_iterations

The separate postprocessor under postprocess/vtu_axial_friction groups cell
centroids by aligned x-y rows and uses adjacent centers to evaluate local dp/dz
and local Darcy friction. It also computes the smooth Colebrook/Moody value.
