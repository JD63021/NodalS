// Stage22D.4a: production-shape Nitsche + Spalding GPU face assembly.
// COMPILE ONLY in D4a: no call site until D4b pressure and momentum wiring.
// This independent path returns separate scalar blocks and the lagged
// off-component action. Full normal Nitsche uses PHYSICAL wall trace + dn;
// tangential Robin uses physical wall traces, and beta is derived from
// the five normal-ray Spalding samples (as in Stage22D2).
//
// FULL local operator: A_ij(a,b) = n_i n_j [
//    -nu phi_a dn_b - nu dn_a phi_b + gamma nu/h phi_a phi_b]
//    + beta(delta_ij-n_i n_j) phi_a phi_b.
// Entries i==j are assembled into three scalar CSRs; i!=j are applied
// explicitly to lagged velocity in RHS. No block matrix is allocated.
__global__ void s22d4_nitsche_tet_component(
    const TetFacePlan*F,int nf,const TetWallQuadCache*qc,
    const TetWallPackedPoint*mp,const nodals_hxt1::TetVel*tv,
    const nodals_hxt1::TetGeom*tg,const std::int32_t*slot,
    const Real*u0,const Real*u1,const Real*u2,
    Real*Ax,Real*Ay,Real*Az,Real*rx,Real*ry,Real*rz,
    double nu,double gammaN,unsigned long long*bad)
{
  const int fi=(int)blockIdx.x;
  if(fi>=nf)return;
  const auto f=F[fi];const auto tvv=tv[f.cell];const auto g=tg[f.cell];
  if(!(nu>0&&gammaN>0&&f.area>0&&g.volume>0)){
    if(threadIdx.x==0)atomicAdd(bad,1ULL);return;
  }
  const double h=3.*g.volume/f.area;
  const double n[3]={f.n[0],f.n[1],f.n[2]};
  const double dref[3]={
    g.invJ[0]*n[0]+g.invJ[1]*n[1]+g.invJ[2]*n[2],
    g.invJ[3]*n[0]+g.invJ[4]*n[1]+g.invJ[5]*n[2],
    g.invJ[6]*n[0]+g.invJ[7]*n[1]+g.invJ[8]*n[2]};
  const Real* us[3]={u0,u1,u2};Real* mats[3]={Ax,Ay,Az};Real* rhs[3]={rx,ry,rz};
  for(int q=(int)threadIdx.x;q<12;q+=(int)blockDim.x){
    const auto &ca=qc[(std::size_t)fi*12u+(std::size_t)q];
    if(!(ca.w>0.&&mp)){atomicAdd(bad,1ULL);continue;}
    using V=::nodals::wall::vector3::V3;
    const V vn{n[0],n[1],n[2]};
    V samples[5];double yy[5];
    for(int m=0;m<5;++m){
      const auto &a=stage27_s02b_at(mp,fi,q,m);
      V s{0,0,0};
      for(int b=0;b<8;++b){
        const int ib=a.g[b];const double weight=a.b[b];
        s.x+=weight*(double)u0[ib];s.y+=weight*(double)u1[ib];s.z+=weight*(double)u2[ib];
      }
      samples[m]=s;yy[m]=a.y;
    }
    V pic{0,0,0};
    for(int b=0;b<8;++b){
      const int ib=tvv.g[b];const double weight=ca.bp[b];
      pic.x+=weight*(double)u0[ib];pic.y+=weight*(double)u1[ib];pic.z+=weight*(double)u2[ib];
    }
    const auto fit=::nodals::wall::vector3::fit_active(samples,yy,vn,pic,V{0.,0.,0.},nu,::stage24b_active_points);
    if(!fit.valid || !(fit.beta>=0) || !isfinite(fit.beta)){
      atomicAdd(bad,1ULL);continue;
    }
    // Fill affine barycentrics from the constant device face topology.
    double lam[4]={0.,0.,0.};
    lam[hxt2::c_tet_face[(int)f.lf][0]]=QL0[q];
    lam[hxt2::c_tet_face[(int)f.lf][1]]=QL1[q];
    lam[hxt2::c_tet_face[(int)f.lf][2]]=QL2[q];
    const double dl[4]={-dref[0]-dref[1]-dref[2],dref[0],dref[1],dref[2]};
    double dn[8];
    for(int a=0;a<4;++a)dn[a]=dl[a];
    for(int a=0;a<4;++a){
      double da=0.;
      for(int j=0;j<4;++j)if(j!=a){
        double product=27.*dl[j];
        for(int k=0;k<4;++k)if(k!=a&&k!=j)product*=lam[k];
        da+=product;
      }
      dn[4+a]=da;
    }
    for(int a=0;a<8;++a){
      const int ia=tvv.g[a];const double pa=ca.bw[a],dna=dn[a];
      for(int b=0;b<8;++b){
        const int ib=tvv.g[b];
        const double pb=ca.bw[b],dnb=dn[b];
        const double normalQ=ca.w*(-nu*pa*dnb-nu*dna*pb+gammaN*nu/h*pa*pb);
        const double robin=ca.w*fit.beta*pa*pb;
        const int position=slot[(std::size_t)f.cell*64u+(std::size_t)(a*8+b)];
        for(int i=0;i<3;++i){
          const double nii=n[i];
          for(int j=0;j<3;++j){
            const double nij=nii*n[j];
            const double entry=nij*normalQ + ((i==j?1.:0.)-nij)*robin;
            if(i==j)atomicAdd(mats[i]+position,(Real)entry);
            else atomicAdd(rhs[i]+ia,(Real)(-entry*(double)us[j][ib]));
          }
        }
      }
    }
  }
}
