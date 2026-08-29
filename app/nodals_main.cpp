// NodalS v1.00
// Modular NodalS v1.00 solver.
// The authoritative FULLFAST-FP64-RICH-SCALE oracle remains under reference/.
// The ordered fragments below reassemble byte-for-byte to SHA256 0de1a338...
// and therefore preserve the accepted numerical, performance and memory architecture.
// Fragments intentionally remain one translation unit in v1.00.
#include "../src/00_core/preamble.inc"
#include "../src/00_core/mesh_runtime.inc"
#include "../src/10_fem/basis_quadrature.inc"
#include "../src/20_boundary/problem_boundary.inc"
#include "../src/30_momentum/custom_momentum.inc"
#include "../src/35_stabilization/supg_dynamic_assembly.inc"
#include "../src/40_pressure/pressure_assembly.inc"
#include "../src/50_parallel/ownership_solve.inc"
#include "../src/60_output/diagnostics_vtu.inc"
#include "../src/70_simple/simple_support_experimental.inc"
#include "../src/80_app/main_solver.inc"
