// NodalS v1.00
// Modular NodalS solver translation unit.
//
// The authoritative FULLFAST-FP64-RICH-SCALE oracle remains under reference/.
// Its protected numerical fragments stay unchanged.  The custom pressure AMG is
// inserted as an additional module between the pressure-assembly definitions it
// consumes and the later ownership/SIMPLE driver code that calls it.  Therefore
// the assembled translation unit is intentionally no longer byte-for-byte the
// frozen oracle, while the separately selectable PETSc fallback path preserves
// the frozen numerical implementation.
//
// Fragments intentionally remain one translation unit in v1.00 so the AMG can
// reuse the existing NodalS-native pressure/MPI data structures without a broad
// ABI refactor.
#include "../src/00_core/preamble.inc"
#include "../src/00_core/mesh_runtime.inc"
#include "../src/10_fem/basis_quadrature.inc"
#include "../src/20_boundary/problem_boundary.inc"
#include "../src/30_momentum/custom_momentum.inc"
#include "../src/35_stabilization/supg_dynamic_assembly.inc"
#include "../src/40_pressure/pressure_assembly.inc"
#include "../src/45_pressure_amg/custom_pressure_amg.inc"
#include "../src/50_parallel/ownership_solve.inc"
#include "../src/60_output/diagnostics_vtu.inc"
#include "../src/70_simple/simple_support_experimental.inc"
#include "../src/80_app/main_solver.inc"
