#pragma once
// NodalS Stage14: mathematically portable scalar transport policies.
// No HXT mesh, pipe geometry, CUDA allocation, global matrix or SIMPLE dependence.
// Caller owns all fields. The numerical state/memory remains with the caller.
#if defined(__CUDACC__)
#define NODALS_TRANSPORT_HD __host__ __device__ __forceinline__
#else
#define NODALS_TRANSPORT_HD inline
#endif
namespace nodals { namespace transport {
template<class Real> struct VolumeCoefficients {
  Real diffusion,reaction,source,omegaProductionAudit,omegaCrossAudit;
};
template<class Real,class State>
NODALS_TRANSPORT_HD VolumeCoefficients<Real> volume_coefficients(
    int equation,int kEquation,Real wt,const State&ss)
{
  VolumeCoefficients<Real> o;
  o.diffusion=wt*(equation==kEquation?ss.sigmaK:ss.sigmaW)*ss.nut;
  o.reaction=wt*(equation==kEquation?ss.reactK:ss.reactOmega);
  o.source=wt*(equation==kEquation?ss.pk:ss.srcOmega);
  o.omegaProductionAudit=wt*ss.omegaProductionAudit;
  o.omegaCrossAudit=wt*ss.omegaCrossAudit;
  return o;
}
template<class Real> struct RowL1Plan {Real delta,diagonal;};
template<class Real>
NODALS_TRANSPORT_HD RowL1Plan<Real> row_l1_plan(Real unrelaxedDiagonal,Real rowAbsSum,Real alpha)
{
  const Real fac=Real(1)/alpha-Real(1);
  const Real delta=fac*rowAbsSum;const Real dn=unrelaxedDiagonal+delta;
  return {delta,dn};
}
// Non-owning field binding for future CPU/GPU transport implementations.
// This intentionally introduces NO allocation, copy, synchronization or layout conversion.
template<class Real> struct ReadOnlyFields {
  const Real *u0=nullptr,*u1=nullptr,*u2=nullptr,*k=nullptr,*omega=nullptr;
  int velocityDofs=0,scalarDofs=0;
};
}} // nodals::transport
#undef NODALS_TRANSPORT_HD
