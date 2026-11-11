  // Keep the original sample_log and OF_AUTO branches intact.
  Real stage41OmegaTarget=stage41_sample_log ? oa.omegaLog : oa.omegaTarget;
  if(stage41_couette_point3){
    // u_tau is from the SAME five-point full tangential-vector Spalding fit
    // as momentum. No velocity inversion or additional k wall constraint.
    // pm.y is the physical interior omega-point-3 ray distance (also works
    // with the existing independent Stage32 omega ray).
    const double yp=(double)ut*pm.y/nu;
    if(!(isfinite(yp) && yp>=0.0)){
      atomicAdd(bad,1ULL);return;
    }
    double omegaEq;
    if(yp==0.0){
      // Exact viscous asymptote; avoids 0 * infinity at stagnant points.
      omegaEq=6.0*nu/(0.075*pm.y*pm.y);
    }else{
      // Existing generated 1D SST-2003m Couette table (log-y+ interpolated).
      // lookup_yplus uses the viscous asymptote below y+=0.05 and clamps
      // beyond the table maximum y+=3000. This is the original table policy.
      const auto tableState=hxt5c_table::lookup_yplus(yp);
      if(!tableState.ok){atomicAdd(bad,1ULL);return;}
      omegaEq=tableState.omegaPlus*(double)ut*(double)ut/nu;
    }
    if(!(isfinite(omegaEq) && omegaEq>0.0)){
      atomicAdd(bad,1ULL);return;
    }
    stage41OmegaTarget=(Real)omegaEq;
  }
  const double nuEff=(double)((Real)nu+ss.sigmaW*ss.nut);
  // DG omega only: harvest the EXACT existing sample-3 log target and
  // sigmaW*nu_t diffusivity (from the same five-point Spalding inversion).
  // No assembly into the old CG omega matrix while this collector is bound.
  if(equation==OMEGA_EQ && hxt5c::dg_omega_wall_target_q12){
    if(!hxt5c::dg_omega_wall_mu_q12||
       !(isfinite((double)stage41OmegaTarget) && stage41OmegaTarget>Real(0)) ||
       !(isfinite(nuEff) && nuEff>0.)){
      atomicAdd(bad,1ULL);return;
    }
    const std::size_t qi=(std::size_t)fi*12u+(std::size_t)q;
    hxt5c::dg_omega_wall_target_q12[qi]=(double)stage41OmegaTarget;
    hxt5c::dg_omega_wall_mu_q12[qi]=nuEff;
    if(wallStats){
      const double ww=(double)hxt5b::QW12[q]*f.area*ell;
      atomicAdd(wallStats+0,ww);
      atomicAdd(wallStats+1,(double)stage41OmegaTarget*ww);
      atomicAdd(wallStats+2,(double)(pm.y*ut/nu)*ww);
      atomicAdd(wallStats+3,(double)wv*ww);
      atomicAdd(wallStats+5,(double)oa.rey*ww);
      atomicAdd(wallStats+6,(double)oa.lamFrac*ww);
      atomicAdd(wallStats+7,(double)oa.omegaVis*ww);
      atomicAdd(wallStats+8,(double)oa.omegaLog*ww);
      atomicAdd(wallStats+9,(double)ss.pk*ww);
      atomicAdd(wallStats+10,(double)oa.pkAuto*ww);
      atomicAdd(wallStats+11,(double)(oa.nutSpalding/(Real)nu)*ww);
      atomicAdd(wallStats+12,(double)(ss.nut/(Real)nu)*ww);
    }
    return;
  }
  const double vweight=(double)hxt5b::QW12[q]*f.area*ell;
  const double coeff=penaltyGamma*nuEff/ell*vweight/ell;
  if(!(isfinite(nuEff)&&nuEff>0.0 && isfinite(coeff)&&coeff>0.0 &&
       isfinite((double)stage41OmegaTarget) && stage41OmegaTarget>Real(0))){
    atomicAdd(bad,1ULL);return;
  }
  if(stage41_trace_log){
#include "of_auto_trace_log_nitsche.inc"
    return;
  }
  if(equation==OMEGA_EQ){
    // One sample, one rank-one positive-semidefinite penalty block.
    for(int a=0;a<8;++a)for(int b=0;b<8;++b){
      const Real v=(Real)coeff*(Real)ph[a]*(Real)ph[b];
      atomicAdd(mat+slot[(std::size_t)pm.cell*64ull+(std::size_t)(a*8+b)],v);
    }
    for(int a=0;a<8;++a)
      atomicAdd(rhs+pm.g[a],(Real)coeff*(Real)ph[a]*stage41OmegaTarget);
    if(wallStats){
      const double vw=vweight;
      atomicAdd(wallStats+0,vw);
      atomicAdd(wallStats+1,(double)stage41OmegaTarget*vw);
      atomicAdd(wallStats+2,pm.y*ut/nu*vw);
      atomicAdd(wallStats+3,(double)wv*vw);
      atomicAdd(wallStats+5,(double)oa.rey*vw);
      atomicAdd(wallStats+6,(double)oa.lamFrac*vw);
      atomicAdd(wallStats+7,(double)oa.omegaVis*vw);
      atomicAdd(wallStats+8,(double)oa.omegaLog*vw);
      atomicAdd(wallStats+9,(double)ss.pk*vw);
      atomicAdd(wallStats+10,(double)oa.pkAuto*vw);
      atomicAdd(wallStats+11,(double)(oa.nutSpalding/(Real)nu)*vw);
      atomicAdd(wallStats+12,(double)(ss.nut/(Real)nu)*vw);
    }
  }
  if(ofProduction){
    const Real dpk=(Real)prodScale*(oa.pkAuto-ss.pk);
    const Real nt=fmax(ss.nut,fmax(Real(1e-12)*(Real)nu,Real(1e-20)));
    const Real src=(equation==K_EQ)?dpk:(ss.gamma*dpk/nt);
    const Real vw=(Real)(vweight*wallVolumeScale);
    for(int a=0;a<8;++a)atomicAdd(rhs+pm.g[a],vw*src*(Real)ph[a]);
  }
