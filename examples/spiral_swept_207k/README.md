# Spiral swept 207k-tetrahedron SST example

Mesh: swept curvilinear hex subdivided into 24 tets per hex, physical tags
inlet=1, outlet=2, wall=3. Volumetric flow=0.0005469756 m^3/s,
physical density=1000 kg/m^3, dynamic viscosity=0.01460735 Pa.s.
NodalS uses kinematic viscosity=1.460735e-5 m^2/s and kinematic pressure;
physical pressure difference in Pa is 1000 * NodalS pressureDrop.

Build first from repository root:

```bash
make -C turbulence/sst_gpu_v2 -j4 NVCC=/usr/local/cuda-12.2/bin/nvcc CUDA_ARCH=sm_86
bash examples/spiral_swept_207k/RUN_SPIRAL_SWEPT.sh --dry-run
bash examples/spiral_swept_207k/RUN_SPIRAL_SWEPT.sh
```

The launcher uses the in-repository `turbulence/sst_gpu_v2` solver by default;
`NODALS_ROOT` can select another built copy. The `.case` stores physical
properties and an editable adaptive-y+ schedule. The currently committed
settings reflect user-requested 20 sweeps for momentum/k/omega, PCG pressure
rtol=1e-3, target y+=30, and a 2-refresh schedule (it200/450).

**Numerical status:** A prior experiment reported internal converged=1 at
it220 and pressureDrop=1275.82 kinematic (~12.76 bar), but its effective
runtime settings differed from some earlier checked-in `.case` values.
This updated case is not claimed to reproduce that solution without a fresh
GPU run. The early-convergence checkpoint-scheduling issue also remains.
