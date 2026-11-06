#pragma once
// Stage22C: exact component-wise diagonal / lagged OFF-DIAGONAL split
// of beta*P_t + gamma_n*P_n, gamma_n=beta+C_n*nu/h_n.
// beta is the five-point Spalding coefficient per unit density, h_n the
// face-normal altitude of the wall-adjacent tet (3*volume/face area).
// The three scalar momentum solves use independent CSR VALUES, same CSR pattern.
#include "spalding_vector_core.hpp"
#include <cmath>
#if defined(__CUDACC__)
#define S22C_HD __host__ __device__ __forceinline__
#else
#define S22C_HD inline
#endif
namespace nodals { namespace wall { namespace stage22c {
using vector3::V3;
struct Split {
  double diag[3],lag[3],beta,gamma;
  bool valid;
};
S22C_HD Split split(V3 n,V3 u,double beta,double nu,double altitude,double Cn){
  Split s{};
  if(!vector3::unit_normal(n)||!vector3::finite(u)||
     !(beta>0.)||!isfinite(beta)||!(nu>0.)||!isfinite(nu)||
     !(altitude>0.)||!isfinite(altitude)||!(Cn>=0.)||!isfinite(Cn))return s;
  const double excess=Cn*nu/altitude;
  s.beta=beta;s.gamma=beta+excess;
  const double nn[3]={n.x,n.y,n.z},uu[3]={u.x,u.y,u.z};
  for(int i=0;i<3;i++){
    s.diag[i]=beta+excess*nn[i]*nn[i];
    double cross=0.;
    for(int j=0;j<3;j++)if(i!=j)cross+=nn[j]*uu[j];
    s.lag[i]=-excess*nn[i]*cross; // off-diagonal terms moved to RHS
    if(!(s.diag[i]>0.)||!isfinite(s.diag[i])||!isfinite(s.lag[i]))return Split{};
  }
  s.valid=true;
  return s;
}
S22C_HD V3 full_force(Split s,V3 n,V3 u){
  const double ndotu=vector3::dot(n,u);
  return vector3::minus(vector3::times(s.beta,u),
         vector3::times(-(s.gamma-s.beta)*ndotu,n));
}
}}} // nodals::wall::stage22c
#undef S22C_HD
