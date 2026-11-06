// Stage22D.2; diagnostic-only face-normal Spalding fit. 
// Axis-Z traction and strong transverse constraints are unchanged.
// Same five-point Spalding objective and traction as the HEX PROFILE5 mode.
// Local first-tetra normal-column quadrature supplies the Picard trial
// functional, while the physical wall trace supplies the test basis.
__global__ void s22d_face_fit_axial(
    const TetFacePlan*F,int nf,const TetWallQuadCache*cache,
    const TetWallPackedPoint*mp,const nodals_hxt1::TetVel*tv,
    const std::int32_t*slot,const Real*u0,const Real*u1,const Real*u2,Real*mat,Real*diag,
    double nu,double*stats,unsigned long long*bad)
{
  const int fi=(int)blockIdx.x;
  if(fi>=nf)return;
  const TetFacePlan f=F[fi];
  const auto T=tv[f.cell];
  __shared__ Real BW[12][8],BP[12][8],BETAW[12];
  __shared__ double AR[12],UT2[12],YPW[12],SLW[12],YPMAX[12],UTMAX[12];
  __shared__ double RMSW[12],SPANW[12],YBARW[12];
  if(threadIdx.x<12){
    const int q=(int)threadIdx.x;
    const TetWallQuadCache ca=cache[(std::size_t)fi*12u+(std::size_t)q];
    double ut=0.0,yp=0.0,beta=0.0,um=0.0,rms=0.0,span=0.0;
    bool ok=mp!=nullptr && ca.w>0.0 && ca.ybar>0.0 && nu>0.0;
    // Stage22D.2: ONLY the five-point speed and Picard beta use the
    // triangle's tangential projector. THE AXIAL MATRIX/TEST/CONSTRAINTS,
    // SST, and pressure correction stay bitwise in their original branch.
    if(ok){
      using Vec=::nodals::wall::vector3::V3;
      const Vec n{f.n[0],f.n[1],f.n[2]};
      Vec sampled[5];double y[5];
      for(int m=0;m<5;++m){
        const auto &p=stage27_s02b_at(mp,fi,q,m);
        Vec U{0.,0.,0.};
        for(int a=0;a<8;++a){
          const int g=p.g[a];const double w=p.b[a];
          U.x+=(double)u0[g]*w;U.y+=(double)u1[g]*w;U.z+=(double)u2[g]*w;
        }
        sampled[m]=U;y[m]=stage27_s02b_y(m);
      }
      Vec pic{0.,0.,0.};
      for(int a=0;a<8;++a){
        const int g=T.g[a];const double w=ca.bp[a];
        pic.x+=(double)u0[g]*w;pic.y+=(double)u1[g]*w;pic.z+=(double)u2[g]*w;
      }
      const auto fit=::nodals::wall::vector3::fit_active(sampled,y,n,pic,Vec{0.,0.,0.},nu,::stage24b_active_points);
      ok=fit.valid;
      if(ok){
        ut=fit.uTau;yp=fit.yPlus;beta=fit.beta;um=fit.meanTangentialSpeed;
        rms=fit.fitRms;span=fit.pointUTauRelSpan;
        if(um<=1.e-6*fmax(fabs(um),1.))atomicAdd(stats+10,ca.w);
        ok=isfinite(beta)&&beta>0.&&isfinite(ut);
      }
    }
    if(!ok){atomicAdd(bad,1ULL);beta=0.0;ut=yp=um=rms=span=0.0;}
    for(int a=0;a<8;++a){BW[q][a]=ok?(Real)ca.bw[a]:Real(0);BP[q][a]=ok?(Real)ca.bp[a]:Real(0);}
    BETAW[q]=(Real)(ca.w*beta);
    AR[q]=ca.w;UT2[q]=ca.w*ut*ut;YPW[q]=ca.w*yp;
    SLW[q]=ca.w*fabs(um);YPMAX[q]=yp;UTMAX[q]=ut;
    RMSW[q]=ca.w*rms;SPANW[q]=ca.w*span;YBARW[q]=(ok&&mp)?ca.w*stage27_s02b_y(2):0.0;
  }
  __syncthreads();
  for(int p=(int)threadIdx.x;p<64;p+=(int)blockDim.x){
    const int a=p/8,b=p%8;Real z=0;
    for(int q=0;q<12;++q)z+=BETAW[q]*BW[q][a]*BP[q][b];
    atomicAdd(mat+slot[(std::size_t)f.cell*64u+(std::size_t)p],z);
    if(a==b)atomicAdd(diag+T.g[a],z);
  }
  if(threadIdx.x==0){
    double A=0,Tau=0,Y=0,S=0,MY=0,MU=0,R=0,SP=0,YB=0;
    for(int q=0;q<12;++q){
      A+=AR[q];Tau+=UT2[q];Y+=YPW[q];S+=SLW[q];
      MY=fmax(MY,YPMAX[q]);MU=fmax(MU,UTMAX[q]);
      R+=RMSW[q];SP+=SPANW[q];YB+=YBARW[q];
    }
    atomicAdd(stats+0,A);atomicAdd(stats+1,Tau);
    atomicAdd(stats+2,Y);atomicAdd(stats+3,S);
    amax(stats+4,MY);amax(stats+5,MU);
    atomicAdd(stats+7,R);atomicAdd(stats+8,SP);atomicAdd(stats+9,YB);
  }
}

