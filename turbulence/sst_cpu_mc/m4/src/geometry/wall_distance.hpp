#pragma once
#include "../mesh/tet_mesh.hpp"
#include <vector>
namespace nodals::cpu_mc::m4 {
std::vector<double> geometric_wall_distance(const m2::Mesh&mesh,int rank);
}
