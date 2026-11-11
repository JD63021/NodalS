#pragma once
// DG omega constitutive field access for NodalS SST-2003m.
//
// This helper is explicitly cell-local: omega[4*tet+a]. No vertex sharing.
// It may be called from the omega transport kernel, the k transport kernel,
// the momentum viscosity/production kernels and physical wall reconstruction.
// A postprocessing CG projection is NEVER a constitutive source here.
//
// For a P1 tetra the physical gradients of barycentric basis functions are
// independent of the reference quadrature point. Caller supplies the exact
// geometric gradient components, reconstructed from the HXT1 geometry.
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>
#if defined(__CUDACC__)
#define NODALS_OMEGA_DG_HD __host__ __device__ __forceinline__
#else
#define NODALS_OMEGA_DG_HD inline
#endif

namespace nodals {namespace omega_dg_field {
struct ValueGradient {
  double value;
  double grad[3];
};
struct Gradient {
  double g[4][3];
};
NODALS_OMEGA_DG_HD ValueGradient evaluate(
    const double*omega4,const Gradient&geo,const double lambda[4]){
  ValueGradient z{0.,{0.,0.,0.}};
  for(int a=0;a<4;++a){
    z.value+=omega4[a]*lambda[a];
    for(int d=0;d<3;++d)z.grad[d]+=omega4[a]*geo.g[a][d];
  }
  return z;
}
NODALS_OMEGA_DG_HD ValueGradient evaluate_cell(
    const double*omegaDG,int tet,const Gradient&geo,const double lambda[4]){
  return evaluate(omegaDG+4*(std::size_t)tet,geo,lambda);
}
NODALS_OMEGA_DG_HD Gradient from_inverse_jacobian(const double invJ[9]){
  // HXT1 convention follows hxt2::tet_phys_grad:
  // grad lambda_a = grad lambda_ref_a * inverse Jacobian.
  Gradient out{};
  constexpr double ref[4][3]={
    {-1.,-1.,-1.},{1.,0.,0.},{0.,1.,0.},{0.,0.,1.}
  };
  for(int a=0;a<4;++a)
    for(int d=0;d<3;++d)
      out.g[a][d]=ref[a][0]*invJ[d]+ref[a][1]*invJ[3+d]+ref[a][2]*invJ[6+d];
  return out;
}
NODALS_OMEGA_DG_HD void write_cell(
    double*omegaDG,int cell,const double w[4]){
  for(int a=0;a<4;++a)omegaDG[4*(std::size_t)cell+a]=w[a];
}
NODALS_OMEGA_DG_HD ValueGradient trace(
    const double*omegaDG,int tet,const Gradient&geo,
    const int faceLocalVertices[3],const double faceBary[3]){
  double l[4]={0,0,0,0};
  for(int j=0;j<3;++j)l[faceLocalVertices[j]]=faceBary[j];
  return evaluate_cell(omegaDG,tet,geo,l);
}
}} // namespace nodals::omega_dg_field
#undef NODALS_OMEGA_DG_HD
