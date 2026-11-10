#pragma once
#include "../mesh/tet_mesh.hpp"
#include "../transport/sst_transport.hpp"
#include <vector>
#include <string>
namespace nodals::cpu_mc::m4 {
void write_solution_vtu(const m2::Mesh&mesh,const SSTTransport&sst,
  const std::vector<double>&velocity,const std::vector<double>&pressure,
  const std::string&filename);
}
