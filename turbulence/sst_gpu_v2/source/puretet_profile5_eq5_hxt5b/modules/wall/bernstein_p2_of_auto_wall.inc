// Pure-tet five-point Spalding / OF-auto OMEGA wall, true Bernstein P2 scalar
// test/trial basis (10 entries). Momentum wall law is the unchanged BF3 one.
// This preserves the ORIGINAL sample location, u_tau inversion, target and
// rank-one structure, but assembles it onto 100 true P2 scalar slots.
// Include inside namespace hxt5c after OFWallState and other wall helpers.
// Not yet wired into the complete G2 application.
__global__ void sst_p2_tet_of_auto_spalding5_point3(
  const hxt5b::TetFacePlan*F,int nf,
  const nodals_hxt1::Point*pts,const nodals_hxt1::TetConn*tc,
  const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,
  const nodals::p2::Tet10Device*p2cells,const std::int32_t*slot100,
  const Real*u0,const Real*u1,const Real*u2,
  const Real*kP2,const Real*omegaP2,Real*mat,Real*rhs,
  double radius,double nu,double penaltyGamma,
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
  if(q>=12 || (equation==K_EQ && !ofProduction))return;
  const hxt5b::TetFacePlan f=F[fi];
  hxt5b::TetWallMatchPoint pm=hxt5b::stage27_s02b_at(
      matchPoints,fi,q,::stage24b_omega_slot);
  if(stage32_omega_active){
    if(!stage32_omega_ray){atomicAdd(bad,1ULL);return;}
    const auto &r=stage32_omega_ray[(std::size_t)fi*12u+(std::size_t)q];
    pm.cell=r.cell;pm.y=r.y;
    for(int j=0;j<8;++j)pm.g[j]=r.g[j];
    for(int j=0;j<4;++j)pm.b[j]=r.l[j];
  }
  const double ell=stage32_omega_active?stage32_omega_penalty_ell:
       (hxt5b::stage27_s02b_y(4)-hxt5b::stage27_s02b_y(0));
  if(!(f.area>0 && pm.cell>=0 && pm.y>0 && ell>0 &&
       isfinite(f.area) && isfinite(ell) && penaltyGamma>0 && nu>0)){
    atomicAdd(bad,1ULL);return;
  }
  double ut=0,ypMid=0,beta=0,um=0,fitRms=0,fitSpan=0;
  if(!hxt5b::spalding_match_profile5_tet_vector(
       matchPoints,fi,q,u0,u1,u2,f.n,nu,ut,ypMid,beta,um,fitRms,fitSpan,
       stage27Saved?stage27Saved+(std::size_t)fi*12u+(std::size_t)q:nullptr)){
    atomicAdd(bad,1ULL);return;
  }
  if(pm.cell<0){atomicAdd(bad,1ULL);return;}
  const double l[4]={pm.b[0],pm.b[1],pm.b[2],pm.b[3]};
  double ph[10],gr[10][3],gp[10][3];
  nodals::p2::basis(l,ph,gr);
  nodals::p2::physical_gradient(gr,tg[pm.cell].invJ,gp);
  double velPh[8],velGr[8][3],velGp[8][3];
  hxt2::tet_basis(l[0],l[1],l[2],l[3],velPh,velGr);
  hxt2::tet_phys_grad(velGr,tg[pm.cell],velGp);
  Real gu[9]={},gk[3]={},gw[3]={},kv=0,wv=0,us[3]={};
  for(int a=0;a<10;++a){const int id=p2cells[pm.cell].g[a];
    const Real ka=kP2[id],wa=omegaP2[id],pa=(Real)ph[a];
    kv+=ka*pa;wv+=wa*pa;
    for(int d=0;d<3;++d){gk[d]+=ka*(Real)gp[a][d];gw[d]+=wa*(Real)gp[a][d];}
  }
  for(int a=0;a<8;++a){const int id=pm.g[a];
    if(id<0){atomicAdd(bad,1ULL);return;}
    const Real pa=(Real)velPh[a];
    const Real ua[3]={u0[id],u1[id],u2[id]};
    for(int d=0;d<3;++d){us[d]+=ua[d]*pa;
      for(int j=0;j<3;++j)gu[3*d+j]+=ua[d]*(Real)velGp[a][j];}
  }
  Real x=0,y=0;for(int a=0;a<4;++a){const auto X=pts[tc[pm.cell].v[a]];
    x+=(Real)(l[a]*X.x);y+=(Real)(l[a]*X.y);}
  const LocalSST ss=local_sst(x,y,(Real)radius,(Real)nu,gu,kv,wv,gk,gw,
       (Real)kFloor,(Real)omegaFloor,stage20_distance(tc,pm.cell,l));
  const Real normal=us[0]*(Real)f.n[0]+us[1]*(Real)f.n[1]+us[2]*(Real)f.n[2];
  const Real Ut=sqrt(fmax(Real(0),us[0]*us[0]+us[1]*us[1]+us[2]*us[2]-normal*normal));
  const OFWallState oa=of_auto_wall_state((Real)pm.y,(Real)nu,Ut,(Real)ut,
     kv,wv,ss,(Real)kFloor,(Real)omegaFloor,
     (Real)kap,(Real)ee,(Real)beta1,(Real)reBlend,(Real)prodLimit,1);
  const Real omegaTarget=stage41_sample_log?oa.omegaLog:oa.omegaTarget;
  const double nuEff=(double)((Real)nu+ss.sigmaW*ss.nut);
  const double vw=(double)hxt5b::QW12[q]*f.area*ell;
  const double coeff=penaltyGamma*nuEff/ell*vw/ell;
  if(!(isfinite(nuEff)&&nuEff>0 && isfinite(coeff)&&coeff>0 &&
       isfinite((double)omegaTarget) && omegaTarget>Real(0))){
    atomicAdd(bad,1ULL);return;
  }
  if(equation==OMEGA_EQ){
    for(int a=0;a<10;++a)for(int b=0;b<10;++b)
      atomicAdd(mat+slot100[(std::size_t)pm.cell*100ull+a*10+b],
        (Real)coeff*(Real)ph[a]*(Real)ph[b]);
    for(int a=0;a<10;++a)
      atomicAdd(rhs+p2cells[pm.cell].g[a],(Real)coeff*(Real)ph[a]*omegaTarget);
    if(wallStats){
      atomicAdd(wallStats+0,vw);
      atomicAdd(wallStats+1,(double)omegaTarget*vw);
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
    const Real src=equation==K_EQ?dpk:ss.gamma*dpk/nt;
    const Real weighted=(Real)(vw*wallVolumeScale);
    for(int a=0;a<10;++a)
      atomicAdd(rhs+p2cells[pm.cell].g[a],weighted*src*(Real)ph[a]);
  }
}
