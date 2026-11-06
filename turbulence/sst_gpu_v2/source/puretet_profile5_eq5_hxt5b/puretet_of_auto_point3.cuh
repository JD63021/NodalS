// PURETET_S1_S5_OFAUTO_V1
// Adapt HEX of_auto_wall_state to the PURE TET wall samples. The five-point
// Spalding u_tau remains the only velocity wall-law inversion; sample #3 is
// the scalar evaluation point. Not the HEX normal-column average-first scheme.
// Both k and omega transport volume equations are untouched. A single omega
// off-wall rank-one penalty is applied. Optional OF production correction can
// be enabled explicitly (OFF by default for the controlled experiment).
__global__ void sst_tet_of_auto_spalding5_point3(
    const hxt5b::TetFacePlan*F,int nf,
    const nodals_hxt1::Point*pts,const nodals_hxt1::TetConn*tc,
    const nodals_hxt1::TetGeom*tg,const std::int32_t*slot,
    const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,
    Real*mat,Real*rhs,double radius,double nu,double penaltyGamma,
    double kFloor,double omegaFloor,double kap,double ee,double beta1,
    double reBlend,double prodLimit,int ofProduction,double prodScale,
    double wallVolumeScale,int equation,
    const hxt5b::TetWallMatchPoint*matchPoints,
    double*wallStats,unsigned long long*bad)
{
  const int fi=(int)blockIdx.x;
  if(fi>=nf)return;
  const int q=(int)threadIdx.x;
  if(q>=12)return;
  if(equation==K_EQ && !ofProduction)return;
  const hxt5b::TetFacePlan f=F[fi];
  const std::size_t base=((std::size_t)fi*12u+(std::size_t)q)*5u;
  const hxt5b::TetWallMatchPoint &pm=matchPoints[base+2u];
  const double y1=matchPoints[base+0u].y,y5=matchPoints[base+4u].y;
  const double ell=y5-y1;
  if(!(f.area>0.0 && pm.cell>=0 && pm.y>0.0 && ell>0.0 &&
       isfinite(f.area) && isfinite(ell) && penaltyGamma>0.0 && nu>0.0)){
    atomicAdd(bad,1ULL);return;
  }
  double ut=0.0,ypMid=0.0,beta=0.0,um=0.0,fitRms=0.0,fitSpan=0.0;
  if(!hxt5b::spalding_match_profile5_tet(
       matchPoints,fi,q,u2,nu,ut,ypMid,beta,um,fitRms,fitSpan)){
    atomicAdd(bad,1ULL);return;
  }
  // Reconstruct the P1+BF3 state and physical gradient at the SAME point
  // used in the wall-normal sampling cache. In state-space q1, the enriched
  // coefficients are explicitly zeroed by the outer solver.
  const double l[4]={pm.b[0],pm.b[1],pm.b[2],pm.b[3]};
  double ph[8],gr[8][3],gp[8][3];
  hxt2::tet_basis(l[0],l[1],l[2],l[3],ph,gr);
  hxt2::tet_phys_grad(gr,tg[pm.cell],gp);
  Real gu[9]={},gk[3]={},gw[3]={};
  Real kv=0,wv=0,us[3]={};
  for(int a=0;a<8;++a){
    const int g=pm.g[a];
    if(g<0){atomicAdd(bad,1ULL);return;}
    const Real ka=k[g],wa=omega[g],ua=u0[g],va=u1[g],za=u2[g];
    const Real pa=(Real)ph[a];
    kv+=ka*pa;wv+=wa*pa;
    us[0]+=ua*pa;us[1]+=va*pa;us[2]+=za*pa;
    for(int d=0;d<3;++d){
      const Real gd=(Real)gp[a][d];
      gu[d]+=ua*gd;gu[3+d]+=va*gd;gu[6+d]+=za*gd;
      gk[d]+=ka*gd;gw[d]+=wa*gd;
    }
  }
  double x=0.0,y=0.0;
  for(int a=0;a<4;++a){
    const auto X=pts[tc[pm.cell].v[a]];
    x+=l[a]*X.x;y+=l[a]*X.y;
  }
  const LocalSST ss=local_sst((Real)x,(Real)y,(Real)radius,(Real)nu,
      gu,kv,wv,gk,gw,(Real)kFloor,(Real)omegaFloor);
  const Real normal=us[0]*(Real)f.n[0]+us[1]*(Real)f.n[1]+us[2]*(Real)f.n[2];
  const Real Ut=sqrt(fmax(Real(0),us[0]*us[0]+us[1]*us[1]+us[2]*us[2]-normal*normal));
  // Reuse the original HEX OF_AUTO viscous/log blend EXACTLY, including the
  // shared five-point Spalding u_tau. No independent k-based u_tau inversion.
  const OFWallState oa=of_auto_wall_state((Real)pm.y,(Real)nu,Ut,(Real)ut,
     kv,wv,ss,(Real)kFloor,(Real)omegaFloor,
     (Real)kap,(Real)ee,(Real)beta1,(Real)reBlend,
     (Real)prodLimit,1);
  const double nuEff=(double)((Real)nu+ss.sigmaW*ss.nut);
  const double vweight=(double)hxt5b::QW12[q]*f.area*ell;
  const double coeff=penaltyGamma*nuEff/ell*vweight/ell;
  if(!(isfinite(nuEff)&&nuEff>0.0 && isfinite(coeff)&&coeff>0.0 &&
       isfinite((double)oa.omegaTarget) && oa.omegaTarget>Real(0))){
    atomicAdd(bad,1ULL);return;
  }
  if(equation==OMEGA_EQ){
    // One sample, one rank-one positive-semidefinite penalty block.
    for(int a=0;a<8;++a)for(int b=0;b<8;++b){
      const Real v=(Real)coeff*(Real)ph[a]*(Real)ph[b];
      atomicAdd(mat+slot[(std::size_t)pm.cell*64ull+(std::size_t)(a*8+b)],v);
    }
    for(int a=0;a<8;++a)
      atomicAdd(rhs+pm.g[a],(Real)coeff*(Real)ph[a]*oa.omegaTarget);
    if(wallStats){
      const double vw=vweight;
      atomicAdd(wallStats+0,vw);
      atomicAdd(wallStats+1,(double)oa.omegaTarget*vw);
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
}
