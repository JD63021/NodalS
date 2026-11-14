#pragma once
// Consistent grad-div volume Galerkin rank-one quadrature contribution.
// For v_i = phi_a e_i and u_j = phi_b e_j,
//   A[(a,i),(b,j)] += gamma * w * d_i(phi_a) * d_j(phi_b).
// gamma has kinematic viscosity units [m^2/s]. No RHS or boundary term.
// Valid for 8-DOF P1+BF3 or 12-DOF P1+BF3+shifted-Spalding velocity.
// Standalone, header-only, independent of MPI and PETSc/CUDA.
#include <array>
#include <cstddef>
#include <cmath>
#include <stdexcept>
namespace nodals::cpu_mc::m4 {
template<std::size_t N,typename Matrix>
inline void add_grad_div_rank_one(Matrix&K,
 const std::array<std::array<double,3>,N>&g,
 double gamma_times_volume_quadrature_weight){
 const double coeff=gamma_times_volume_quadrature_weight;
 if(!(std::isfinite(coeff)&&coeff>=0))
  throw std::runtime_error("grad-div invalid coefficient or volume quadrature weight");
 if(coeff==0)return;
 for(std::size_t a=0;a<N;++a)for(std::size_t b=0;b<N;++b)
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)
    K[3*a+i][3*b+j]+=coeff*g[a][i]*g[b][j];
}
} // namespace nodals::cpu_mc::m4
