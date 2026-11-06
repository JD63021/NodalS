// Stage22B segregated tangential Spalding with a stronger normal Robin penalty.
// Included INSIDE hxt5b after TetWallMatchPoint, TetWallQuadCache, TetFacePlan.
// For eta >= 0, the physical wall tensor is beta*(P+(1+eta)*nn^T)
// = beta*(1+eta)*I - eta*beta*P.
// Put beta*(1+eta)*I in all THREE scalar momentum matrices and lag
// eta*beta*P*u_picard on the RHS. This fixed-point splitting is stable
// on the wall subspace: tangent amplification eta/(1+eta), normal 0.
__global__ void s22b_wall_tet_segregated(
    const TetFacePlan*F,int nf,const TetWallQuadCache*cache,
    const TetWallPackedPoint*mp,const nodals_hxt1::TetVel*tv,
    const std::int32_t*slot,const Real*u0,const Real*u1,const Real*u2,
    Real*wall,Real*diag,Real*rhs0,Real*rhs1,Real*rhs2,
    double nu,double eta,double*stats,unsigned long long*bad)
{
  const int fi=(int)blockIdx.x;
  if(fi>=nf)return;
  const TetFacePlan f=F[fi];
  const auto T=tv[f.cell];
  using V=::nodals::wall::vector3::V3;
  const V n{f.n[0],f.n[1],f.n[2]};
  for(int q=(int)threadIdx.x;q<12;q+=(int)blockDim.x){
    const auto &ca=cache[(std::size_t)fi*12u+(std::size_t)q];
    V samples[5];double yy[5];
    bool ok=mp && ca.w>0. && ca.ybar>0. && nu>0. && eta>=0. && isfinite(eta);
    if(!ok){atomicAdd(bad,1ULL);continue;}
    for(int m=0;m<5;++m){
      const auto &a=stage27_s02b_at(mp,fi,q,m);
      V s{0.,0.,0.};
      for(int j=0;j<8;++j){
        const int g=a.g[j];const double b=a.b[j];
        s.x+=(double)u0[g]*b;s.y+=(double)u1[g]*b;s.z+=(double)u2[g]*b;
      }
      samples[m]=s;yy[m]=a.y;
    }
    V pic{0.,0.,0.},wallTrace{0.,0.,0.};
    for(int j=0;j<8;++j){
      const int g=T.g[j];
      pic.x+=(double)u0[g]*ca.bp[j];
      pic.y+=(double)u1[g]*ca.bp[j];
      pic.z+=(double)u2[g]*ca.bp[j];
      wallTrace.x+=(double)u0[g]*ca.bw[j];
      wallTrace.y+=(double)u1[g]*ca.bw[j];
      wallTrace.z+=(double)u2[g]*ca.bw[j];
    }
    const auto fit=::nodals::wall::vector3::fit_active(samples,yy,n,pic,V{0.,0.,0.},nu,::stage24b_active_points);
    if(!fit.valid){atomicAdd(bad,1ULL);continue;}
    const double coeff=ca.w*fit.beta*(1.+eta);
    const V tangent=::nodals::wall::vector3::tangent_projection(pic,n);
    const V lag=::nodals::wall::vector3::times(ca.w*fit.beta*eta,tangent);
    if(!isfinite(coeff)||!::nodals::wall::vector3::finite(lag)){
      atomicAdd(bad,1ULL);continue;
    }
    for(int a=0;a<8;++a){
      const int ia=T.g[a];const Real left=(Real)ca.bw[a];
      atomicAdd(rhs0+ia,left*(Real)lag.x);
      atomicAdd(rhs1+ia,left*(Real)lag.y);
      atomicAdd(rhs2+ia,left*(Real)lag.z);
      for(int b=0;b<8;++b){
        const Real entry=left*(Real)ca.bp[b]*(Real)coeff;
        atomicAdd(wall+slot[(std::size_t)f.cell*64u+(std::size_t)(a*8+b)],entry);
        if(a==b)atomicAdd(diag+ia,entry);
      }
    }
    atomicAdd(stats+0,ca.w);
    atomicAdd(stats+1,ca.w*fit.uTau*fit.uTau);
    atomicAdd(stats+2,ca.w*fit.yPlus);
    atomicAdd(stats+3,ca.w*fit.meanTangentialSpeed);
    amax(stats+4,fit.yPlus);amax(stats+5,fit.uTau);
    atomicAdd(stats+7,ca.w*fit.fitRms);
    atomicAdd(stats+8,ca.w*fit.pointUTauRelSpan);
    atomicAdd(stats+9,ca.w*yy[2]);
    // Separate, non-canceling wall-normal velocity audit.
    const double normalU=::nodals::wall::vector3::dot(wallTrace,n);
    atomicAdd(stats+16,ca.w*fabs(normalU));
    atomicAdd(stats+17,ca.w*normalU);
  }
}

__global__ void s22b_combine(std::size_t n,const Real*b,const Real*c,
  const Real*t,const Real*dg,const Real*wall,Real*xy,Real*z){
  std::size_t i=(std::size_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n){const Real a=b[i]+c[i]+t[i]+dg[i]+wall[i];xy[i]=a;z[i]=a;}
}

__global__ void s22b_add_rhs(int n,const Real*lag,Real*rhs){
  const int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i<n)rhs[i]+=lag[i];
}
__global__ void s22b_add_extra(int n,const Real*lag,const Real*dev,Real*out){
  const int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i<n)out[i]=lag[i]+dev[i];
}
