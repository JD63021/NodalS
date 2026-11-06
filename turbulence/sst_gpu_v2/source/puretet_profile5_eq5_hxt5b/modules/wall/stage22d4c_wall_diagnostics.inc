// Stage22D.4C independent post-predictor/post-correction diagnostics.
// CG velocity traces integrated over ACTUAL affine faces, not CAD normals.
// Slots (double): 0 area, 1 int|un|, 2 int un, 3 int un^2,
// 4 int utau^2, 5 int utau, 6 int yplus, 7 int tractionZ,
// 8 int |ut|, 9 sum local FE cross-component Picard defect L1,
// 10 sum local FE wall weak action L1 (NOT full momentum residual),
// 11 sum local FE cross defect^2, 12 positive flux, 13 negative flux,
// 14 bitwise atomicMax |un|, 15 tractionX, 16 tractionY, 17 penalty energy.
// All quantities are per unit density when expressed as forces.
__global__ void s22d4c_wall_diagnostic(
    const hxt5b::TetFacePlan*F,int nf,const hxt5b::TetWallQuadCache*qc,
    const hxt5b::TetWallPackedPoint*mp,const nodals_hxt1::TetVel*tv,
    const nodals_hxt1::TetGeom*tg,const nodals_hxt1::Point*pts,
    const nodals_hxt1::TetConn*tc,
    const Real*u0,const Real*u1,const Real*u2,
    const Real*old0,const Real*old1,const Real*old2,
    const Real*k,const Real*omega,
    double nu,double radius,double kFloor,double omegaFloor,double gammaN,
    double*stats,unsigned long long*bad,bool sampledConductance,bool bf3FluxMode){
  int fi=(int)blockIdx.x;if(fi>=nf)return;
  const auto f=F[fi];const auto tvv=tv[f.cell];const auto g=tg[f.cell];
  if(!(nu>0.&&gammaN>0.&&f.area>0.&&g.volume>0.)){
    if(threadIdx.x==0)atomicAdd(bad,1ULL);return;
  }
  const double h=3.*g.volume/f.area;
  const double n[3]={f.n[0],f.n[1],f.n[2]};
  const Real* u[3]={u0,u1,u2};
  const Real* o[3]={old0,old1,old2};
  using V=::nodals::wall::vector3::V3;
  const V normal{n[0],n[1],n[2]};
  for(int q=(int)threadIdx.x;q<12;q+=(int)blockDim.x){
    const auto &ca=qc[(std::size_t)fi*12u+(std::size_t)q];
    if(!(ca.w>0.)||!mp){atomicAdd(bad,1ULL);continue;}
    V samples[5],oldSamples[5];double yy[5];
    for(int m=0;m<5;++m){
      const auto&a=stage27_s02b_at(mp,fi,q,m);
      V v{0.,0.,0.},vold{0.,0.,0.};
      for(int b=0;b<8;++b){int ib=a.g[b];double w=a.b[b];
        v.x+=w*(double)u0[ib];v.y+=w*(double)u1[ib];v.z+=w*(double)u2[ib];
        vold.x+=w*(double)old0[ib];vold.y+=w*(double)old1[ib];vold.z+=w*(double)old2[ib];
      }samples[m]=v;oldSamples[m]=vold;yy[m]=hxt5b::stage27_s02b_y(m);
    }
    double faceU[3]={},lagU[3]={};
    V pic{0.,0.,0.},oldPic{0.,0.,0.};
    for(int b=0;b<8;++b){int ib=tvv.g[b];double w=ca.bp[b];
      pic.x+=w*(double)u0[ib];pic.y+=w*(double)u1[ib];pic.z+=w*(double)u2[ib];
      oldPic.x+=w*(double)old0[ib];oldPic.y+=w*(double)old1[ib];oldPic.z+=w*(double)old2[ib];
    }
    for(int i=0;i<3;++i)for(int b=0;b<8;++b){
      int ib=tvv.g[b];faceU[i]+=ca.bw[b]*(double)u[i][ib];
      lagU[i]+=ca.bw[b]*(double)o[i][ib];
    }
    const auto fit=::nodals::wall::vector3::fit_active(samples,yy,normal,pic,V{0.,0.,0.},nu,::stage24b_active_points);
    const auto fitOld=::nodals::wall::vector3::fit_active(oldSamples,yy,normal,oldPic,V{0.,0.,0.},nu,::stage24b_active_points);
    if(!fit.valid||!fitOld.valid||!isfinite(fit.beta)||!isfinite(fitOld.beta)){atomicAdd(bad,1ULL);continue;}
    const double un=n[0]*faceU[0]+n[1]*faceU[1]+n[2]*faceU[2];
    const double ut[3]={faceU[0]-n[0]*un,faceU[1]-n[1]*un,faceU[2]-n[2]*un};
    const double utmag=sqrt(ut[0]*ut[0]+ut[1]*ut[1]+ut[2]*ut[2]);
    const double sampledNormal=pic.x*n[0]+pic.y*n[1]+pic.z*n[2];
    const double sampledTangent[3]={pic.x-sampledNormal*n[0],
      pic.y-sampledNormal*n[1],pic.z-sampledNormal*n[2]};
    atomicAdd(stats+0,ca.w);atomicAdd(stats+1,ca.w*fabs(un));
    atomicAdd(stats+2,ca.w*un);atomicAdd(stats+3,ca.w*un*un);
    atomicAdd(stats+4,ca.w*fit.uTau*fit.uTau);
    atomicAdd(stats+5,ca.w*fit.uTau);atomicAdd(stats+6,ca.w*fit.yPlus);
    atomicAdd(stats+7,-ca.w*fit.beta*(sampledConductance?sampledTangent[2]:ut[2]));
    atomicAdd(stats+8,ca.w*utmag);
    atomicAdd(stats+15,-ca.w*fit.beta*(sampledConductance?sampledTangent[0]:ut[0]));
    atomicAdd(stats+16,-ca.w*fit.beta*(sampledConductance?sampledTangent[1]:ut[1]));
    if(un>=0.)atomicAdd(stats+12,ca.w*un);
    else atomicAdd(stats+13,ca.w*un);
    // The double value is nonnegative; positive IEEE bits compare monotonically.
    atomicMax((unsigned long long*)(stats+14),(unsigned long long)__double_as_longlong(fabs(un)));
    double lam[4]={0.,0.,0.,0.};
    lam[hxt2::c_tet_face[(int)f.lf][0]]=hxt5b::QL0[q];
    lam[hxt2::c_tet_face[(int)f.lf][1]]=hxt5b::QL1[q];
    lam[hxt2::c_tet_face[(int)f.lf][2]]=hxt5b::QL2[q];
    double ph[8],gr[8][3],gp[8][3],dn[8];
    hxt2::tet_basis(lam[0],lam[1],lam[2],lam[3],ph,gr);
    hxt2::tet_phys_grad(gr,g,gp);
    Real gu[9]={Real(0)},gk[3]={Real(0)},gw[3]={Real(0)},kv=Real(0),wv=Real(0);
    double xx=0.,yyc=0.;
    for(int a=0;a<4;++a){auto P=pts[tc[f.cell].v[a]];
      xx+=lam[a]*P.x;yyc+=lam[a]*P.y;
    }
    double du[3]={},dlagU[3]={};
    double derivU[3]={},derivOld[3]={};
    for(int a=0;a<8;++a){int ia=tvv.g[a];
      kv+=(Real)ph[a]*k[ia];wv+=(Real)ph[a]*omega[ia];
      dn[a]=gp[a][0]*n[0]+gp[a][1]*n[1]+gp[a][2]*n[2];
      for(int j=0;j<3;++j){
        du[j]+=(double)u[j][ia]*dn[a];
        dlagU[j]+=(double)o[j][ia]*dn[a];
      }
      for(int d=0;d<3;++d){
        const Real gd=(Real)gp[a][d];
        gu[d]+=o[0][ia]*gd;gu[3+d]+=o[1][ia]*gd;gu[6+d]+=o[2][ia]*gd;
        gk[d]+=k[ia]*gd;gw[d]+=omega[ia]*gd;
      }
    }
    const auto ss=hxt5c::local_sst((Real)xx,(Real)yyc,(Real)radius,(Real)nu,
        gu,kv,wv,gk,gw,(Real)kFloor,(Real)omegaFloor);
    const double nueff=nu+(double)ss.nut;
    if(!(nueff>0.)||!isfinite(nueff)){atomicAdd(bad,1ULL);continue;}
    atomicAdd(stats+17,ca.w*gammaN*nueff/h*un*un);
    double ndU=0.;for(int j=0;j<3;++j)ndU+=n[j]*du[j];
    const double fullNormalFlux=un;
    // FE-test-space full wall action; NOT an equilibrium residual by itself.
    // Include the pressure test/gradient terms only in the global solve.
    double defect1=0.,defect2=0.,weak1=0.;
    const double diffPic[3]={pic.x-oldPic.x,pic.y-oldPic.y,pic.z-oldPic.z};
    const double diffPicNormal=n[0]*diffPic[0]+n[1]*diffPic[1]+n[2]*diffPic[2];
    for(int a=0;a<8;++a){
      const double pa=ca.bw[a],spa=ca.bp[a],dna=dn[a];
      for(int i=0;i<3;++i){
        double crossUn=0.,crossDUn=0.;
        for(int j=0;j<3;++j)if(j!=i){
          crossUn+=n[j]*(faceU[j]-lagU[j]);
          crossDUn+=n[j]*(du[j]-dlagU[j]);
        }
        // Keep archived diagnostic formulas bit-for-bit in the original mode.
        double off=0.,full=0.;
        if(!sampledConductance){
          off=ca.w*n[i]*(-nueff*pa*crossDUn
             -nueff*dna*crossUn +(gammaN*nueff/h-fitOld.beta)*pa*crossUn);
          full=ca.w*(n[i]*(-nueff*pa*ndU -nueff*dna*fullNormalFlux
             +gammaN*nueff/h*pa*fullNormalFlux) +fit.beta*pa*ut[i]);
        }else{
          // The sampled tangential friction has its own test and trial
          // functional; the unchanged BF3 normal term stays at wall trace.
          const double tangentOff=-spa*n[i]*(diffPicNormal-n[i]*diffPic[i]);
          off=ca.w*(n[i]*(bf3FluxMode ?
             gammaN*nueff/h*pa*crossUn :
             -nueff*pa*crossDUn-nueff*dna*crossUn+gammaN*nueff/h*pa*crossUn)
             +fitOld.beta*tangentOff);
          full=ca.w*(n[i]*(bf3FluxMode ?
             gammaN*nueff/h*pa*fullNormalFlux :
             -nueff*pa*ndU-nueff*dna*fullNormalFlux+
               gammaN*nueff/h*pa*fullNormalFlux)
             +fit.beta*spa*sampledTangent[i]);
        }
        defect1+=fabs(off);defect2+=off*off;weak1+=fabs(full);
      }
    }
    atomicAdd(stats+9,defect1);atomicAdd(stats+10,weak1);
    atomicAdd(stats+11,defect2);
  }
}

// Independent FE flux on planar inlet/outlet faces; same 12-point Gauss
// triangle rule, including P1+BF3 trace functions.
__global__ void s22d4c_patch_flux(
    const hxt5b::TetFacePlan*F,int nf,const nodals_hxt1::TetVel*tv,
    const Real*u0,const Real*u1,const Real*u2,double*total){
  const int fi=(int)blockIdx.x;if(fi>=nf)return;
  const auto f=F[fi];const auto tvv=tv[f.cell];
  for(int q=(int)threadIdx.x;q<12;q+=(int)blockDim.x){
    double l[4]={0.,0.,0.,0.};
    l[hxt2::c_tet_face[(int)f.lf][0]]=hxt5b::QL0[q];
    l[hxt2::c_tet_face[(int)f.lf][1]]=hxt5b::QL1[q];
    l[hxt2::c_tet_face[(int)f.lf][2]]=hxt5b::QL2[q];
    double ph[8],gr[8][3];hxt2::tet_basis(l[0],l[1],l[2],l[3],ph,gr);
    double vn=0.;for(int a=0;a<8;++a){int ia=tvv.g[a];
      vn+=ph[a]*(f.n[0]*(double)u0[ia]+f.n[1]*(double)u1[ia]+f.n[2]*(double)u2[ia]);
    }
    atomicAdd(total,hxt5b::QW12[q]*f.area*vn);
  }
}

// Independent volume divergence audit. Four-point degree-2 tetrahedral rule
// integrates the derivatives of cubic BF3 enrichment basis exactly.
__global__ void s22d4c_volume_divergence(
    const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,
    int ntet,const Real*u0,const Real*u1,const Real*u2,double*total){
  int t=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(t>=ntet)return;
  const auto v = tv[t];
  const auto g = tg[t];
  constexpr double a=0.5854101966249685,b=0.1381966011250105;
  double divInt=0.;
  for(int q=0;q<4;++q){
    double l[4]={b,b,b,b};l[q]=a;
    double ph[8],gr[8][3],gp[8][3];
    hxt2::tet_basis(l[0],l[1],l[2],l[3],ph,gr);
    hxt2::tet_phys_grad(gr,g,gp);
    double d=0.;
    for(int j=0;j<8;++j){int ij=v.g[j];
      d+=(double)u0[ij]*gp[j][0]+(double)u1[ij]*gp[j][1]+(double)u2[ij]*gp[j][2];
    }
    divInt+=d;
  }
  atomicAdd(total,divInt*g.volume/4.);
}
