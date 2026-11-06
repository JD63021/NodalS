// PURETET_S1_S5_OFAUTO_V1
// Adapt HEX of_auto_wall_state to the PURE TET wall samples. The five-point
// Spalding u_tau remains the only velocity wall-law inversion; sample #3 is
// the scalar evaluation point. Not the HEX normal-column average-first scheme.
// Both k and omega transport volume equations are untouched. A single omega
// off-wall rank-one penalty is applied. Optional OF production correction can
// be enabled explicitly (OFF by default for the controlled experiment).
// STAGE41: target selection for the existing point omega penalty.
__device__ __constant__ int stage41_sample_log=0;
// The independent physical trace Nitsche treatment; all existing modes OFF.
__device__ __constant__ int stage41_trace_log=0;
__device__ __constant__ double stage41_trace_gamma=50.;
// Experimental selection, not a change to the original log/OF_AUTO modes.
__device__ __constant__ int stage41_couette_point3=0;
// STAGE32: extra omega-only geometry. Momentum/Spalding continue to use mp.
__device__ __constant__ int stage32_omega_active=0;
__device__ __constant__ const hxt5b::TetWallPackedPoint* stage32_omega_ray=nullptr;
__device__ __constant__ double stage32_omega_penalty_ell=0.;
__global__ void sst_tet_of_auto_spalding5_point3(
    const hxt5b::TetFacePlan*F,int nf,
    const nodals_hxt1::Point*pts,const nodals_hxt1::TetConn*tc,
    const nodals_hxt1::TetGeom*tg,const nodals_hxt1::TetVel*tv,
    const std::int32_t*slot,
    const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,
    Real*mat,Real*rhs,double radius,double nu,double penaltyGamma,
    double kFloor,double omegaFloor,double kap,double ee,double beta1,
    double reBlend,double prodLimit,int ofProduction,double prodScale,
    double wallVolumeScale,int equation,
    const hxt5b::TetWallPackedPoint*matchPoints,
    double*wallStats,unsigned long long*bad,
    ::nodals::wall::shared27::Record*stage27Saved)
{
  const int fi=(int)blockIdx.x;
  if(fi>=nf)return;
  const int q=(int)threadIdx.x;
  if(q>=12)return;
  if(equation==K_EQ && !ofProduction)return;
  const hxt5b::TetFacePlan f=F[fi];
  // OF_AUTO uses the independent omega sample, plus the geometric
  // end-to-end thickness y5-y1 (no need to store endpoints as ray records).
  hxt5b::TetWallMatchPoint pm=hxt5b::stage27_s02b_at(
    matchPoints,fi,q,::stage24b_omega_slot);
  if(stage32_omega_active){
    if(!stage32_omega_ray){atomicAdd(bad,1ULL);return;}
    const auto &r=stage32_omega_ray[(std::size_t)fi*12u+(std::size_t)q];
    pm.cell=r.cell;pm.y=r.y;
    #pragma unroll
    for(int j=0;j<8;++j)pm.g[j]=r.g[j];
    #pragma unroll
    for(int j=0;j<4;++j)pm.b[j]=r.l[j];
    #pragma unroll
    for(int i=0;i<4;++i){
      double v=27.;
      #pragma unroll
      for(int j=0;j<4;++j)if(j!=i)v*=pm.b[j];
      pm.b[4+i]=v;
    }
  }
  const double y1=hxt5b::stage27_s02b_y(0),y5=hxt5b::stage27_s02b_y(4);
  const double ell=stage32_omega_active?stage32_omega_penalty_ell:(y5-y1);
  if(!(f.area>0.0 && pm.cell>=0 && pm.y>0.0 && ell>0.0 &&
       isfinite(f.area) && isfinite(ell) && penaltyGamma>0.0 && nu>0.0)){
    atomicAdd(bad,1ULL);return;
  }
  double ut=0.0,ypMid=0.0,beta=0.0,um=0.0,fitRms=0.0,fitSpan=0.0;
  // Stage25A Fix2: OF_AUTO u_tau must use the SAME tangential vector
  // sampling as the BF3 momentum wall; the old Z-only fit is not covariant.
  if(!hxt5b::spalding_match_profile5_tet_vector(
       matchPoints,fi,q,u0,u1,u2,f.n,nu,ut,ypMid,beta,um,fitRms,fitSpan,
       stage27Saved?stage27Saved+(std::size_t)fi*12u+(std::size_t)q:nullptr)){
    atomicAdd(bad,1ULL);return;
  }
#include "of_auto_point3_reconstruction.inc"
#include "of_auto_point3_target.inc"
#include "of_auto_point3_rank1_penalty.inc"
}
