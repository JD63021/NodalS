#pragma once
// Independent OpenFOAM-style omega wall target, shared Spalding wall state.
// NO matrix assembly, interpolation, CUDA runtime, or HXT state is present.
#include <cmath>
#if defined(__CUDACC__)
#define NODALS_WALL_HD __host__ __device__ __forceinline__
#else
#define NODALS_WALL_HD inline
#endif
namespace nodals { namespace wall {
template<class Real> struct OfAutoState {
  Real omegaTarget=0,omegaVis=0,omegaLog=0,lamFrac=0,rey=0,yPlusK=0,pkAuto=0,nutSpalding=0;
};
// SST input requires the fields pk/nut only; accepts either host or device SST state.
template<class Real,class SST>
NODALS_WALL_HD OfAutoState<Real> evaluate_of_auto(
    Real yd,Real nu,Real Ut,Real uTau,Real kRaw,Real omegaRaw,const SST&ss,
    Real kFloor,Real omegaFloor,Real kappa,Real E,Real beta1,Real reBlend,Real prodLimit,
    int sharedSpaldingWallState)
{

  OfAutoState<Real> q;
  const Real kval=fmax(kRaw,kFloor),om=fmax(omegaRaw,omegaFloor);
  const Real Cmu25=sqrt(sqrt(Real(0.09))),Cmu5=sqrt(Real(0.09));
  q.rey=yd*sqrt(kval)/nu;
  q.yPlusK=Cmu25*q.rey;
  q.lamFrac=exp(-q.rey/fmax(reBlend,Real(1e-6)));
  q.lamFrac=fmin(fmax(q.lamFrac,Real(0)),Real(1));
  const Real turbFrac=Real(1)-q.lamFrac;
  const Real magGrad=Ut/fmax(yd,Real(1e-20));

  // Historical OF-auto branch formed its own k-based uStar and used kappa=0.41/E=9.8.
  // In shared mode, use the exact same u_tau returned by the momentum Spalding root
  // and the exact momentum-wall constants kappa=0.4, B=5.5.  The automatic
  // viscous/log blending itself is intentionally left unchanged so this is a
  // controlled wall-state consistency A/B rather than a new wall model.
  const bool shared=(sharedSpaldingWallState!=0);
  const Real kap=shared?Real(0.4):fmax(kappa,Real(1e-6));
  const Real Espg=shared?Real(9.025013499434122):E; // exp(0.4*5.5)
  const Real uStar=shared
    ? fmax(uTau,Real(0))
    : sqrt(fmax(Real(0),q.lamFrac*nu*magGrad+turbFrac*Cmu5*kval));

  q.omegaVis=Real(6)*nu/(fmax(beta1,Real(1e-8))*yd*yd);
  q.omegaLog=uStar/(Cmu5*kap*yd);
  q.omegaTarget=q.lamFrac*q.omegaVis+turbFrac*q.omegaLog;

  Real gLog=ss.pk;
  if(shared){
    // Same Spalding wall state as momentum: G_log = u_tau^3/(kappa*y).
    if(uTau>Real(1e-12) && yd>Real(1e-20))
      gLog=fmax(Real(0),uTau*uTau*uTau/(kap*yd));
  }else{
    const Real arg=fmax(Espg*q.yPlusK,Real(1.000001));
    const Real uPlus=log(arg)/kap;
    if(q.yPlusK>Real(1e-8)&&uPlus>Real(1e-8)){
      const Real t=uStar*magGrad*yd/uPlus;
      gLog=t*t/(nu*kap*q.yPlusK);
    }
  }
  q.pkAuto=q.lamFrac*ss.pk+turbFrac*gLog;
  const Real lim=fmax(prodLimit,Real(0))*Real(0.09)*kval*om;
  q.pkAuto=fmin(fmax(q.pkAuto,Real(0)),lim);
  q.nutSpalding=(Ut>Real(1e-12))?fmax(Real(0),uTau*uTau/(Ut/yd)-nu):Real(0);
  return q;
}
}} // nodals::wall
#undef NODALS_WALL_HD
