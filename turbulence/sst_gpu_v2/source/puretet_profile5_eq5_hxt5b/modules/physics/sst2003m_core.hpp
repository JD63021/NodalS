#pragma once
// NodalS Stage12: first independently reusable physics component.
// No CUDA runtime, HXT geometry, global state or SIMPLE dependencies.
// The pipe-specific distance remains explicit and untouched until geometry generalization.
#include <cmath>
#if defined(__CUDACC__)
  #define NODALS_SST12_HD __host__ __device__ __forceinline__
#else
  #define NODALS_SST12_HD inline
#endif
namespace nodals { namespace sst2003m {
#include "../sst/sst_constants.inc"
template<class Real>
struct LocalState {
  Real nut=0,F1=0,F2=0,sigmaK=0,sigmaW=0,beta=0,gamma=0,strain=0;
  Real pk=0,srcOmega=0,reactK=0,reactOmega=0; Real omegaProductionAudit=0,omegaCrossAudit=0; // PURETET_EXP1_SOURCE
  bool kFloored=false,omegaFloored=false;
};


// All arithmetic statements below are the exact accepted Stage7 fragments.
// Explicit deviatoric flag replaces the former CUDA constant-memory access;
// the adapter passes that original constant unchanged.
template<class Real>
NODALS_SST12_HD LocalState<Real> evaluate_pipe(
    Real x,Real y,Real radius,Real nu,
    const Real gu[9],Real kRaw,Real omegaRaw,
    const Real gk[3],const Real gw[3],Real kFloor,Real omegaFloor,
    int c_sst_deviatoric_production)
{
  using LocalSST = LocalState<Real>;
#include "../sst/sst_kinematics.inc"
#include "../sst/sst_pipe_distance.inc"
#include "../sst/sst_blending.inc"
#include "../sst/sst_eddy_viscosity.inc"
#include "../sst/sst_sources.inc"
 }
}} // nodals::sst2003m
#undef NODALS_SST12_HD
