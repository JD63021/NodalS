#pragma once
#include "../mesh/tet_mesh.hpp"
#include <vector>
namespace nodals::cpu_mc::m2 {
struct Ownership { std::vector<int> cell,vertex,face;std::vector<int> countCell,countVertex,countFace;};
Ownership partition_rcb(const Mesh&m,int ranks);
}
