// Stage13: OF_AUTO target state is computed in the shared host/device core.
using OFWallState = nodals::wall::OfAutoState<Real>;
__device__ __forceinline__ OFWallState of_auto_wall_state(
    Real yd,Real nu,Real Ut,Real uTau,Real kRaw,Real omegaRaw,const LocalSST&ss,
    Real kFloor,Real omegaFloor,Real kappa,Real E,Real beta1,Real reBlend,Real prodLimit,
    int sharedSpaldingWallState)
{
  return nodals::wall::evaluate_of_auto<Real>(
      yd,nu,Ut,uTau,kRaw,omegaRaw,ss,kFloor,omegaFloor,kappa,E,beta1,reBlend,
      prodLimit,sharedSpaldingWallState);
}
