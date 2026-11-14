# CPU SST Spalding enrichment — experimental branch (2026-11-14)

This branch is forked from `exp/cpu-sst-komega-multicore` to protect existing CPU M4 and GPU solvers.

## Stage 1: available as an additive patch

The assistant has executed a dependency-free C++ basis/mesh audit using a **6D long, 1,857-tetrahedron Delaunay pipe**, at D=0.0889 m and Re=72,600. The tested shifted P1 Spalding wall-law basis is conforming and zero on the wall. The pressure-divergence coefficient of the shared enrichment is nonzero and must be included in the eventual coupled Schur operator.

The source, reproducible C++ tests, tagged .msh, baseline CPU M4 case, and an additive Git patch are supplied in the conversation as `NODALS_CPU_M4_SHIFTED_SPALDING_STAGE1_20261114.zip`. After downloading and extracting:

```bash
cd ~/NodalS_CPU_ENRICH_20261114
git apply --check ~/Downloads/NODALS_CPU_M4_SHIFTED_SPALDING_STAGE1_20261114.patch
git apply ~/Downloads/NODALS_CPU_M4_SHIFTED_SPALDING_STAGE1_20261114.patch
bash experiments/cpu_sst_spalding_tiny_pipe/tests/RUN_BASIS_TEST.sh
bash experiments/cpu_sst_spalding_tiny_pipe/tests/RUN_MESH_AUDIT.sh
```

The additive patch does not modify existing numerical code, so it can be reviewed before connecting the experimental enrichment to M4.

**Not yet implemented:** enlargement of the M4 velocity system, strong no-slip replacement for the legacy tangential Robin condition, enriched P0 pressure coupling/rAU/Schur, and SST-production evaluation from the total enriched gradient. The optional original M4 baseline run has not been compiled in the assistant's environment (no MPI).

Do **not** claim an enriched CPU SST calculation exists from this Stage 1 patch. See the packaged README for the complete validation protocol.
