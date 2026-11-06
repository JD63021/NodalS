# NodalS V2 GPU k-omega SST: spiral swept-207K experimental example

This example runs the CURRENT V2 CUDA SST source imported under
`turbulence/sst_gpu_v2/` (not the older standalone laminar CPU/MPI code).

- Geometry: 207,360 tetrahedra, face-centre/cell-centre split of a swept
  curvilinear grid. Boundary physical tags 1=inlet, 2=outlet, 3=wall.
- Physical inputs: Q=0.0005469756 m³/s, rho=1000 kg/m³,
  dynamic mu=0.01460735 Pa s, kinematic nu=1.460735e-5 m²/s.
- Pressure is kinematic (p/rho); delta-p in Pa = 1000 * NodalS output.
- Target y+=30, case includes adaptive face-based Spalding/omega sampling.
- Solver: M3-FGS P1+BF3 momentum, masked P1 k/omega,
  P0 pressure, PCG-AMG. Case requests 20 momentum and scalar sweeps.

## Reproduce on the existing Ubuntu V2 runtime

```bash
cd examples/spiral_swept_207k
NODALS_ROOT="$HOME/Downloads/NodalS_SST_SOFTWARE_V2" \
  bash RUN_SPIRAL_SWEPT.sh --dry-run
NODALS_ROOT="$HOME/Downloads/NodalS_SST_SOFTWARE_V2" \
  bash RUN_SPIRAL_SWEPT.sh
```

`RUN_SPIRAL_SWEPT.sh` requires an already built V2 native executable.
It uses the case's `[mesh]`, `[physics]`, `[inlet.main]`, `[wall.main]`,
`[outlet.main]`, `[sst]`, `[simple]`, and `[adaptive_sampling]` settings.
The mesh and case are experimental reference data, not a fully verified
multi-geometry regression suite.

## Known issues / validation boundaries

The reported spiral run reached `converged=1` at SIMPLE it=220 with
`NODALS_SST_G2_FINAL status=PASS`, wall y+ ~32.28 and pressure drop
~1275.82 (kinematic, ~12.76 bar). The *outer* wrapper nevertheless
reported `NODALS_SOFTWARE_COMPLETION=PHYSICAL_CHECK_FAILED` (exit 3),
and the adaptive runner then expected it=500 checkpoint and stopped.
Investigate this inconsistent completion classification and support
an early-stop checkpoint in the adaptive runner before declaring this
an official regression pass. Also verify native effective stress form
and pressure rtol in `terminal.log`: prior run output showed a different
stress form / tolerance than the intended `.case` settings.

The current SST software supports NodalS-compatible **pure tetrahedral**
meshes with correctly tagged inlet/outlet/wall patches, not arbitrary
mixed or hex meshes, arbitrary patch groupings, or multiple inlets/outlets.
Wall-normal sampling rays may need tuning/validation on new geometries.
