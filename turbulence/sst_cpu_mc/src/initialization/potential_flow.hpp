#pragma once
#include "../mesh/tet_mesh.hpp"
#include <array>
#include <vector>
namespace nodals::cpu_mc::m3 {
// Calls unchanged GPU host Stage27 potential-flow initialization on rank 0;
// distributes the physical P1+BF3 initial velocity to all MPI ranks.
std::vector<double> potential_velocity(const nodals::cpu_mc::m2::Mesh&mesh,
                                       std::array<double,3> inlet,int rank);
}
