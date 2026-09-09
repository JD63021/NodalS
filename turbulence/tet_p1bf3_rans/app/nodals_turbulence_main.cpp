// NodalS turbulence skeleton — modular P1+BF3/P0 steady RANS solver.
//
// Numerical oracle:
//   reference/p1bf3_rans_effectiveB_oracle.cpp
//
// This intentionally remains ONE C++ translation unit assembled from ordered
// .inc fragments, matching the production NodalS architecture.  The first
// modularization is text-preserving; later CPU/GPU/SST refactors must retain
// the regression gates before the oracle is replaced.

#include "../src/00_core/preamble_mesh.inc"
#include "../src/10_problem/problem_config.inc"
#include "../src/20_discretization/discrete_boundary_fem.inc"
#include "../src/30_momentum/custom_momentum_core.inc"
#include "../src/40_pressure/effective_pressure.inc"
#include "../src/50_runtime/dynamic_runtime_plan.inc"
#include "../src/55_turbulence/mixing_length_spalding.inc"
#include "../src/60_momentum_dynamic/dynamic_operators.inc"
#include "../src/70_pressure_assembly/pressure_assembly.inc"
#include "../src/75_pressure_amg/custom_pressure_amg.inc"
#include "../src/80_app/main_solver.inc"
