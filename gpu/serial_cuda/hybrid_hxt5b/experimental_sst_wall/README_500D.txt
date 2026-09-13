NodalS SST G2 — same-DOF 10D -> 500D axial stretch
=====================================================

Purpose
-------
Create a 500D pipe by multiplying only the axial coordinate of the validated
original 10D HXT1 mesh by 50. Connectivity, velocity/pressure DOF numbering,
and inlet/wall/outlet records are unchanged. HXT1 precomputed HEX/TET geometry
plans are rebuilt after the coordinate transformation.

This is intentionally NOT axial refinement. It has exactly the same algebraic
problem size as the 10D and 100D cases; the elements are 50x longer in z than
the original 10D mesh. Radial/wall-normal resolution is unchanged.

Step 1 — build stretched mesh
-----------------------------
SOURCE_MESH=/path/to/original/10D/HXT1_mesh.bin bash BUILD_HXT1_500D_STRETCHED.sh

Expected markers:
  HXT1_STRETCH_STATUS=PASS
  HXT1_500D_BUILD_STATUS=PASS

Audit checks include connectivity/DOF/boundary invariance, unchanged transverse
coordinates, length ratio=50, volume ratio=50, positive rebuilt geometry, and
unchanged binary size.

Step 2 — run SST G2
-------------------
Use RUN_SST_G2_500D_TUNABLE_FP32.sh with HXT5B_DIR pointing to the existing
NodalS_SST_G2_DIRECT_START_FIX1 source tree. The solver knobs are the same as
the successful 100D tunable run. The default startup remains direct plug flow
with zero precursor SIMPLE iterations.

Recommended first 500D settings
-------------------------------
Keep the successful 100D settings unchanged: alphaU=0.70, alphaP=1.0,
alphaK=alphaOmega=0.70, RAU_SCALE=20, momentum rtol=0.01, pressure rtol=1e-6,
pressure AMG MCGS with 2 fine and 2 coarse symmetric sweeps.

Postprocessing
--------------
The existing axial postprocessor infers L/D from coordinates, so it can be
used unchanged. For 500D its automatic 70%-90% developed candidate window is
350D-450D, leaving 50D before the outlet. Compare local pressure-gradient and
wall-shear Darcy factors there, and inspect the final outlet zone separately.
