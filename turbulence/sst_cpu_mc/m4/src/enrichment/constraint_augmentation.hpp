#pragma once
// Algebraic augmentation of the pressure-continuity constraint
//   B u + source = 0    (equivalently B u = c, c=-source).
// M_p is diagonal with tetrahedron volumes.
//   A += gamma * B^T M_p^-1 B
//   b -= gamma * B^T M_p^-1 source  (= b + gamma * B^T M_p^-1 c)
// Critical: NEVER add only the matrix term for nonzero DG inlet source.
// The same B/source is constructed by discrete_constraint_row.hpp and Schur.
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
namespace nodals::cpu_mc::m4 {
template<std::size_t N,typename Matrix,typename Vector>
inline void add_constraint_augmentation(
 Matrix&K,Vector&rhs,const std::array<std::array<double,3>,N>&B,
 double source,double gamma_over_cell_volume){
 const double coeff=gamma_over_cell_volume;
 if(!(std::isfinite(coeff)&&coeff>=0&&std::isfinite(source)))
  throw std::runtime_error("constraint augmentation nonfinite gamma/volume or DG source");
 if(coeff==0)return;
 for(std::size_t a=0;a<N;++a)for(int i=0;i<3;++i){
  const double bi=B[a][i];
  rhs[3*a+i]-=coeff*bi*source;
  for(std::size_t b=0;b<N;++b)for(int j=0;j<3;++j)
   K[3*a+i][3*b+j]+=coeff*bi*B[b][j];
 }
}
} // namespace nodals::cpu_mc::m4
