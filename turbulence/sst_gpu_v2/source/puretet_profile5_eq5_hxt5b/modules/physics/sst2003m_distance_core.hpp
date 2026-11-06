#pragma once
// Numerical physics is sourced from the original SST fragments. Only d changes.
#include "sst2003m_core.hpp"
#if defined(__CUDACC__)
#define NODALS_SST20_HD __host__ __device__ __forceinline__
#else
#define NODALS_SST20_HD inline
#endif
namespace nodals { namespace sst2003m {
template<class Real>
NODALS_SST20_HD LocalState<Real> evaluate_at_distance(
    Real distance,Real radius,Real nu,
    const Real gu[9],Real kRaw,Real omegaRaw,
    const Real gk[3],const Real gw[3],Real kFloor,Real omegaFloor,
    int c_sst_deviatoric_production)
{
  using LocalSST=LocalState<Real>;
#include "../sst/sst_kinematics.inc"
  const Real d=fmax(distance,fmax(Real(1e-10)*radius,Real(1e-12)));
#include "../sst/sst_blending.inc"
#include "../sst/sst_eddy_viscosity.inc"
#include "../sst/sst_sources.inc"
}
}} // nodals::sst2003m
#undef NODALS_SST20_HD
