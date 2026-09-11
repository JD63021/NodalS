Gate 5D: VMFL003 100D extreme downstream axial-coarsening test

Mesh policy
-----------
0-20D:
  Preserve all 361 axial planes from the original validated 20D MSH source.
  This is the original 360-slab D/18 mesh.

20-100D:
  Append 80 slabs, each exactly one diameter long (dz = D).

Counts
------
Total slabs: 440
Total tetrahedra: 440 * 2556 = 1,124,640
Total points: 441 * 451 = 198,891
Wall triangles: 440 * 96 = 42,240

The 18x jump in axial spacing at 20D is intentionally aggressive. This is a
stress test, not a production mesh recommendation.

Workflow
--------
1) mesh_tools/BUILD_VMFL003_100D_20DFINE_DLONG.sh
2) gpu/serial_cuda/RUN_RANS_GATE5D_100D_20DFINE_DLONG_FP32.sh
3) postprocess/vtu_axial_friction/RUN_VMFL003_100D_DLONG_FRICTION.sh
