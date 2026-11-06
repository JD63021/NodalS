// Stage22C physical segregated wall: separate X/Y/Z CSR VALUES and rAU,
// exact full wall traction at the Picard fixed point; cross-components lagged.
// No isotropic (1+eta)*beta drag is added to the tangential direction.
__global__ void s22c_wall_tet_component(
    const TetFacePlan*F,int nf,const TetWallQuadCache*cache,
    const TetWallPackedPoint*mp,const nodals_hxt1::TetVel*tv,
    const nodals_hxt1::TetGeom*tg,const std::int32_t*slot,
    const Real*u0,const Real*u1,const Real*u2,
    Real*wallX,Real*wallY,Real*wallZ,
    Real*diagX,Real*diagY,Real*diagZ,
    Real*rhsX,Real*rhsY,Real*rhsZ,
    double nu,double Cn,double*stats,unsigned long long*bad)
{
  const int fi=(int)blockIdx.x;
  if(fi>=nf)return;
  const TetFacePlan f=F[fi];const auto T=tv[f.cell];
  using V=::nodals::wall::vector3::V3;
  const V n{f.n[0],f.n[1],f.n[2]};
  const double hn=3.*tg[f.cell].volume/f.area;
  for(int q=(int)threadIdx.x;q<12;q+=(int)blockDim.x){
    const auto &ca=cache[(std::size_t)fi*12u+(std::size_t)q];
    V samples[5];double yy[5];
    const bool ok=mp&&ca.w>0.&&ca.ybar>0.&&nu>0.&&Cn>=0.&&
      f.area>0.&&isfinite(hn)&&hn>0.&&isfinite(Cn);
    if(!ok){atomicAdd(bad,1ULL);continue;}
    for(int m=0;m<5;++m){
      const auto &a=stage27_s02b_at(mp,fi,q,m);
      V s{0.,0.,0.};
      for(int j=0;j<8;++j){
        const int g=a.g[j];const double b=a.b[j];
        s.x+=(double)u0[g]*b;s.y+=(double)u1[g]*b;s.z+=(double)u2[g]*b;
      }
      samples[m]=s;yy[m]=stage27_s02b_y(m);
    }
    V pic{0.,0.,0.};
    for(int j=0;j<8;++j){
      const int g=T.g[j];
      pic.x+=(double)u0[g]*ca.bp[j];
      pic.y+=(double)u1[g]*ca.bp[j];
      pic.z+=(double)u2[g]*ca.bp[j];
    }
    const auto fit=::nodals::wall::vector3::fit_active(samples,yy,n,pic,V{0.,0.,0.},nu,::stage24b_active_points);
    if(!fit.valid){atomicAdd(bad,1ULL);continue;}
    const auto s=::nodals::wall::stage22c::split(n,pic,fit.beta,nu,hn,Cn);
    if(!s.valid){atomicAdd(bad,1ULL);continue;}
    for(int a=0;a<8;++a){
      const int ia=T.g[a];const Real left=(Real)ca.bw[a];
      atomicAdd(rhsX+ia,left*(Real)(ca.w*s.lag[0]));
      atomicAdd(rhsY+ia,left*(Real)(ca.w*s.lag[1]));
      atomicAdd(rhsZ+ia,left*(Real)(ca.w*s.lag[2]));
      for(int b=0;b<8;++b){
        const std::size_t ix=(std::size_t)f.cell*64u+(std::size_t)(a*8+b);
        const int position=slot[ix];
        const Real wt=left*(Real)ca.bp[b]*(Real)ca.w;
        const Real xx=wt*(Real)s.diag[0], yy=wt*(Real)s.diag[1], zz=wt*(Real)s.diag[2];
        atomicAdd(wallX+position,xx);
        atomicAdd(wallY+position,yy);
        atomicAdd(wallZ+position,zz);
        if(a==b){atomicAdd(diagX+ia,xx);atomicAdd(diagY+ia,yy);atomicAdd(diagZ+ia,zz);}
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
  }
}
__global__ void s22c_combine(std::size_t n,const Real*b,const Real*c,
  const Real*t,const Real*dg,const Real*wx,const Real*wy,const Real*wz,
  Real*x,Real*y,Real*z){
  const std::size_t i=(std::size_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n){const Real a=b[i]+c[i]+t[i]+dg[i];
    x[i]=a+wx[i];y[i]=a+wy[i];z[i]=a+wz[i];}
}
// Exactly the same row-L1 equation relaxation used by the legacy solver,
// applied independently for each component and its own diagonal Schur rAU.
__global__ void s22c_finalize_scalar(int n,const unsigned char*fixed,
   const std::int64_t*row,const std::int32_t*diagPos,
   Real*values,Real*rAU,Real*diagOriginal,Real*relaxDelta,
   double alphaU,double rauScale,unsigned long long*bad){
  const int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i>=n)return;
  const int d=diagPos[i];const Real original=values[d];
  diagOriginal[i]=original;
  Real l1=Real(0);
  for(std::int64_t k=row[i];k<row[i+1];++k)l1+=(Real)fabs((double)values[k]);
  const Real delta=(Real)(1./alphaU-1.)*l1;
  relaxDelta[i]=delta;
  if(fixed[i]){rAU[i]=Real(0);return;}
  const Real denominator=original+delta;
  if(!(l1>Real(0))||!(denominator>Real(0))||
     !isfinite((double)original)||!isfinite((double)delta)||
     !isfinite((double)denominator)){
    atomicAdd(bad,1ULL);rAU[i]=Real(0);return;
  }
  values[d]=denominator;
  rAU[i]=(Real)(rauScale/(double)denominator);
}
// Post-SIMPLE correction wall-normal and ACTUAL vector traction audit.
// Does not use cached values from the pre-correction wall assembly.
__global__ void s22c_wall_post_audit(const TetFacePlan*F,int nf,
    const TetWallQuadCache*cache,const TetWallPackedPoint*mp,
    const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,
    const Real*u0,const Real*u1,const Real*u2,
    double nu,double Cn,double*stats,unsigned long long*bad){
  const int fi=(int)blockIdx.x;if(fi>=nf)return;
  const auto f=F[fi];const auto T=tv[f.cell];
  using V=::nodals::wall::vector3::V3;
  const V n{f.n[0],f.n[1],f.n[2]};
  const double hn=3.*tg[f.cell].volume/f.area;
  for(int q=(int)threadIdx.x;q<12;q+=(int)blockDim.x){
    const auto&ca=cache[(std::size_t)fi*12u+(std::size_t)q];
    V pic{0,0,0},trace{0,0,0},samples[5];double yy[5];
    for(int j=0;j<8;++j){const int g=T.g[j];
      pic.x+=(double)u0[g]*ca.bp[j];pic.y+=(double)u1[g]*ca.bp[j];pic.z+=(double)u2[g]*ca.bp[j];
      trace.x+=(double)u0[g]*ca.bw[j];trace.y+=(double)u1[g]*ca.bw[j];trace.z+=(double)u2[g]*ca.bw[j];}
    for(int m=0;m<5;++m){const auto&a=stage27_s02b_at(mp,fi,q,m);
      V v{0,0,0};for(int j=0;j<8;++j){const int g=a.g[j];
        v.x+=(double)u0[g]*a.b[j];v.y+=(double)u1[g]*a.b[j];v.z+=(double)u2[g]*a.b[j];}
      samples[m]=v;yy[m]=stage27_s02b_y(m);}
    const auto fit=::nodals::wall::vector3::fit_active(samples,yy,n,pic,V{0,0,0},nu,::stage24b_active_points);
    const auto split=::nodals::wall::stage22c::split(n,pic,fit.beta,nu,hn,Cn);
    if(!fit.valid||!split.valid){atomicAdd(bad,1ULL);continue;}
    const V traction=::nodals::wall::stage22c::full_force(split,n,pic);
    const double un=::nodals::wall::vector3::dot(trace,n);
    atomicAdd(stats+0,ca.w);
    atomicAdd(stats+1,ca.w*fabs(un));
    atomicAdd(stats+2,ca.w*un);
    atomicAdd(stats+3,ca.w*traction.x);
    atomicAdd(stats+4,ca.w*traction.y);
    atomicAdd(stats+5,ca.w*traction.z);
    atomicAdd(stats+6,ca.w*(split.gamma-split.beta));
  }
}
