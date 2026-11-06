#pragma once
// Stage2.C.3: cache the *pure nonlinear Spalding fit* from post-correction
// OF_AUTO. Reuse ONLY when every selected tangential speed matches exactly,
// sampling count and molecular nu match. No reuse of Picard Robin beta.
// Strict equality is intentional: no tolerance-based staleness is allowed.
namespace nodals { namespace wall { namespace shared27 {
struct Record {
  double magnitude[5];          // exact FP64 speed arguments of the fit
  double ut,yp,betaFit,uReference,rms,span,nu;
  int count,valid;
};
static_assert(sizeof(Record)==104,"Stage2.C.3 record layout changed");
#if defined(__CUDACC__)
#define S27C3_HD __host__ __device__ __forceinline__
#else
#define S27C3_HD inline
#endif
S27C3_HD vector3::Result fit_or_reuse(
  const vector3::V3 sampled[5],const double y[5],vector3::V3 normal,
  vector3::V3 pic,vector3::V3 wall,double nu,int n,
  const Record* saved,bool* reused=nullptr){
  using namespace vector3;
  if(saved && saved->valid==1 && saved->count==n && saved->nu==nu &&
     n>=1 && n<=5 && sampled && y && unit_normal(normal) &&
     finite(pic) && finite(wall) && nu>0. && isfinite(nu)){
    bool ok=true;
    double prevActiveY=0.;
    const unsigned mask=::nodals::wall::active::active_slot_mask(n);
    for(int i=0;i<5;++i){
      if(!(mask&(1u<<i)))continue;
      if(!finite(sampled[i]) || !isfinite(y[i]) || !(y[i]>prevActiveY)){
        ok=false;break;
      }
      prevActiveY=y[i];
      {
        const V3 tv=tangent_projection(minus(sampled[i],wall),normal);
        const double mag=norm(tv);
        if(!isfinite(mag) || mag!=saved->magnitude[i]){ok=false;break;}
      }
    }
    if(ok){
      Result out{};
      out.uTau=saved->ut;out.yPlus=saved->yp;
      out.fitRms=saved->rms;out.pointUTauRelSpan=saved->span;
      const V3 vbar=tangent_projection(minus(pic,wall),normal);
      const double speed=norm(vbar);
      out.meanTangentialSpeed=speed;
      const double floor=1.e-6*fmax(fabs(saved->uReference),1.);
      // Match the fresh active fit at exactly uTau=0. The frozen
      // five-point reuse path is unchanged, including its original rule.
      out.beta=active::select_picard_beta(out.uTau,speed,floor,
        saved->betaFit,n,active::viscous_limit_enabled());
      if(isfinite(out.beta) && out.beta>0.){
        out.tangent=tangent_tensor(normal,out.beta);
        out.traction=times(-1.,multiply(out.tangent,minus(pic,wall)));
        out.valid=finite(out.traction)&&isfinite(out.fitRms)&&isfinite(out.yPlus);
        if(out.valid){if(reused)*reused=true;return out;}
      }
    }
  }
  if(reused)*reused=false;
  return vector3::fit_active(sampled,y,normal,pic,wall,nu,n);
}
#undef S27C3_HD
}}} // nodals::wall::shared27
