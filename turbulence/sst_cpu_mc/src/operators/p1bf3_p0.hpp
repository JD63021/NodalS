#pragma once
#include "../mesh/tet_mesh.hpp"
#include <array>
namespace nodals::cpu_mc::m2 {
// Exact integrals of grad(P1 + face BF3) over affine tetrahedra:
// P1 grad lambda_i: -S_i / 3, face bubble 27 product(lambda_j): (9/20) S_i.
// S_i is the outward-oriented face area vector, opposite vertex i.
struct Bcell {std::array<Vec3,8> b{};};
Bcell divergence_coefficients(const Mesh&m,int cell);
// Full weak deviatoric vector-Laplacian element bilinear form, 24x24.
// This is an algebraic kernel test: no momentum convection or wall law is assembled.
using Element24 = std::array<std::array<double,24>,24>;
Element24 deviatoric_element(const Mesh&m,int cell,double nu);
}
