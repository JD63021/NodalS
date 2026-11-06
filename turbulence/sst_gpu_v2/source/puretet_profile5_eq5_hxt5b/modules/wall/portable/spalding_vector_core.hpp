#pragma once
// Stage22A. Portable, host/device vector Spalding data and operator algebra.
// No CUDA runtime, mesh, assembly, or pressure-coupling assumptions.
// Stage22B must provide a component-coupled momentum/Schur implementation
// before vector mode can be activated in the steady GPU solver.
#include "spalding_core.hpp"
#include <cmath>

#if defined(__CUDACC__)
#define NODALS_VEC_HD __host__ __device__ __forceinline__
#else
#define NODALS_VEC_HD inline
#endif

namespace nodals { namespace wall { namespace vector3 {
struct V3 { double x,y,z; };
struct Tensor3 { double a[3][3]; };
struct Result {
  double uTau,yPlus,beta,meanTangentialSpeed,fitRms,pointUTauRelSpan;
  V3 traction;              // -beta*(I-nn^T)*(uPicard-uWall), per unit density
  Tensor3 tangent;          // beta*(I-nn^T), positive semidefinite
  bool valid;
};
NODALS_VEC_HD V3 minus(V3 a,V3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
NODALS_VEC_HD V3 times(double s,V3 a){return {s*a.x,s*a.y,s*a.z};}
NODALS_VEC_HD double dot(V3 a,V3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
NODALS_VEC_HD double norm(V3 a){return sqrt(dot(a,a));}
NODALS_VEC_HD bool finite(V3 a){return isfinite(a.x)&&isfinite(a.y)&&isfinite(a.z);}
NODALS_VEC_HD bool unit_normal(V3 n){
  if(!finite(n))return false;
  const double n2=dot(n,n);
  return isfinite(n2)&&fabs(n2-1.0)<=1.e-9;
}
NODALS_VEC_HD V3 tangent_projection(V3 v,V3 n){
  return minus(v,times(dot(v,n),n));
}
NODALS_VEC_HD Tensor3 tangent_tensor(V3 n,double beta){
  Tensor3 M{};
  const double v[3]={n.x,n.y,n.z};
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)
    M.a[i][j]=beta*((i==j?1.0:0.0)-v[i]*v[j]);
  return M;
}
NODALS_VEC_HD V3 multiply(Tensor3 A,V3 v){
  return {A.a[0][0]*v.x+A.a[0][1]*v.y+A.a[0][2]*v.z,
          A.a[1][0]*v.x+A.a[1][1]*v.y+A.a[1][2]*v.z,
          A.a[2][0]*v.x+A.a[2][1]*v.y+A.a[2][2]*v.z};
}
// Tangential-speed magnitude is evaluated at each of FIVE sampling points,
// then sent to the UNCHANGED Spalding least-squares fit. We do NOT project
// onto an arbitrarily chosen single tangent vector.
// The physical linearization is a 3x3 projector, NOT three independent
// diagonal scalar Robin conditions. Stage22B is responsible for assembly.
NODALS_VEC_HD Result fit_five(const V3 sampled[5],const double y[5],
                             V3 inward_or_outward_unit_normal,
                             V3 uPicard,V3 uWall,double nu){
  Result out{};
  const V3 n=inward_or_outward_unit_normal;
  if(!sampled||!y||!unit_normal(n)||!finite(uPicard)||!finite(uWall)||
     !(nu>0.0)||!isfinite(nu))return out;
  double U[5],Usigned[5];
  for(int i=0;i<5;++i){
    if(!finite(sampled[i])||!isfinite(y[i])||!(y[i]>0.0)||
       (i>0 && !(y[i]>y[i-1])))return out;
    const V3 rel=tangent_projection(minus(sampled[i],uWall),n);
    U[i]=norm(rel);
    Usigned[i]=U[i];
    if(!isfinite(U[i]))return out;
  }
  double uMid=0.0,betaFit=0.0;
  if(!fit_profile5_values(U,Usigned,y,nu,out.uTau,out.yPlus,betaFit,
                          uMid,out.fitRms,out.pointUTauRelSpan))return out;
  const V3 vbar=tangent_projection(minus(uPicard,uWall),n);
  const double speed=norm(vbar);
  out.meanTangentialSpeed=speed;
  const double floor=1.0e-6*fmax(fabs(uMid),1.0);
  // Same finite local Robin-beta fallback as the original tetrahedral kernel.
  out.beta=(speed>floor)?out.uTau*out.uTau/speed:betaFit;
  if(!isfinite(out.beta)||!(out.beta>0.0))return out;
  out.tangent=tangent_tensor(n,out.beta);
  out.traction=times(-1.0,multiply(out.tangent,minus(uPicard,uWall)));
  out.valid=finite(out.traction)&&isfinite(out.fitRms)&&isfinite(out.yPlus);
  return out;
}
}}} // namespace nodals::wall::vector3
#undef NODALS_VEC_HD
