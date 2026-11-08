// NODALS_CONSERVATIVE_FULL_ELEMENT_SUPG_V1
// Conservative physical convection: div(a tensor u) = a.grad(u) + div(a)*u.
// This kernel supplies the Picard-lagged right-hand SUPG correction
//   -sum_K int tau (a.grad v) [div(a)*u_lag
//                      - div(2 nu_eff D_dev(u_lag))] dx.
// The independent, positive conservative SD LHS already contributes
//   sum_K int tau (a.grad v)*(a.grad u_new) dx.
// At a converged fixed point, they combine to the complete element-interior
// conservative/deviatoric strong residual, with P0 pressure grad=0 inside K.
// Momentum source f=0 in the spiral case. Face traction and pressure jumps
// are NOT part of this volume-only SUPG implementation.
//
// Requires skew_full_supg_rhs.cuh included FIRST for the exact P1+BF3 Hessian
// and the sfull_nueff local-SST centered Cartesian derivative helper.
// Using a distinct kernel preserves the working skew-SUPG implementation.
// NOTE: the viscosity derivative is a finite-difference approximation.

__global__ void cfull_conservative_dev_rhs64(
    const nodals_hxt1::Point* __restrict__ pts,
    const nodals_hxt1::TetConn* __restrict__ tc,
    const nodals_hxt1::TetVel* __restrict__ tv,
    const nodals_hxt1::TetGeom* __restrict__ tg,
    const Real* __restrict__ ux,const Real* __restrict__ uy,
    const Real* __restrict__ uz,const Real* __restrict__ k,
    const Real* __restrict__ omega,
    Real* __restrict__ rhsx,Real* __restrict__ rhsy,
    Real* __restrict__ rhsz,
    std::uint64_t nc,double radius,double nu,double kFloor,
    double omegaFloor,double tauScale,double magic,double fdRelative,
    unsigned long long* __restrict__ bad){
  const std::uint64_t c=(std::uint64_t)blockIdx.x;
  if(c>=nc)return;
  const int tid=(int)threadIdx.x;
  const auto&G=tg[c];
  __shared__ double U[3][8],K[8],W[8],nueffCentroid;
  __shared__ double stream[64][8],weight[64],corr[64][3];
  if(tid<8){
    const int a=tv[c].g[tid];
    U[0][tid]=(double)ux[a];U[1][tid]=(double)uy[a];
    U[2][tid]=(double)uz[a];
    K[tid]=(double)k[a];W[tid]=(double)omega[a];
  }
  __syncthreads();
  if(tid==0){
    const double cent[4]={0.25,0.25,0.25,0.25};
    nueffCentroid=sfull_nueff(pts,tc,G,c,cent,U,K,W,
        radius,nu,kFloor,omegaFloor);
    if(!(nueffCentroid>0 && isfinite(nueffCentroid)))atomicAdd(bad,1ULL);
  }
  __syncthreads();
  if(tid<64){
    const int ir=tid>>4,is=(tid>>2)&3,it=tid&3;
    const double r=s33_rn[ir],s=s33_sn[is],t=s33_tn[it];
    const double lam[4]={(1-r)*(1-s)*(1-t),r,(1-r)*s,(1-r)*(1-s)*t};
    double ph[8],gr[8][3],gp[8][3];
    hxt2::tet_basis(lam[0],lam[1],lam[2],lam[3],ph,gr);
    hxt2::tet_phys_grad(gr,G,gp);
    double vv[3]={},grad[3][3]={},H[3][3][3]={};
    for(int a=0;a<8;++a){
      double bh[3][3];sfull_basis_hessian(G,lam,a,bh);
      for(int i=0;i<3;++i){
        vv[i]+=U[i][a]*ph[a];
        for(int d=0;d<3;++d){
          grad[i][d]+=U[i][a]*gp[a][d];
          for(int e=0;e<3;++e)H[i][d][e]+=U[i][a]*bh[d][e];
        }
      }
    }
    const double divergence=grad[0][0]+grad[1][1]+grad[2][2];
    const double speed2=vv[0]*vv[0]+vv[1]*vv[1]+vv[2]*vv[2];
    const double ne=sfull_nueff(pts,tc,G,c,lam,U,K,W,
        radius,nu,kFloor,omegaFloor);
    // Directional perturbations in Cartesian coordinates are equivalent
    // to lambda_a(x +/- eps e_d)=lambda_a(x) +/- eps grad(lambda_a)_d.
    // Keep both samples strictly inside the tetrahedron.
    double maxGradient=1e-300,minLam=1.;
    for(int a=0;a<4;++a){
      minLam=fmin(minLam,lam[a]);
      for(int d=0;d<3;++d)
        maxGradient=fmax(maxGradient,fabs(sfull_grad_lam(G,a,d)));
    }
    const double h=sqrt(fmax((double)G.h2,1e-30));
    const double eps=fmin(fdRelative*h,0.25*minLam/maxGradient);
    double dne[3]={};bool valid=(G.det>0 && G.h2>0 && nueffCentroid>0 && isfinite(nueffCentroid) && ne>0 &&
        isfinite(ne)&&eps>1e-16*h&&isfinite(eps));
    if(valid){
      for(int d=0;d<3;++d){
        double lp[4],lm[4];
        for(int a=0;a<4;++a){const double delta=eps*sfull_grad_lam(G,a,d);
          lp[a]=lam[a]+delta;lm[a]=lam[a]-delta;}
        const double np=sfull_nueff(pts,tc,G,c,lp,U,K,W,
          radius,nu,kFloor,omegaFloor);
        const double nm=sfull_nueff(pts,tc,G,c,lm,U,K,W,
          radius,nu,kFloor,omegaFloor);
        if(!(np>0&&nm>0&&isfinite(np)&&isfinite(nm)))valid=false;
        dne[d]=(np-nm)/(2.0*eps);
        if(!isfinite(dne[d]))valid=false;
      }
    }
    if(!valid){atomicAdd(bad,1ULL);weight[tid]=0.;
      for(int i=0;i<3;++i)corr[tid][i]=0.;
      for(int a=0;a<8;++a)stream[tid][a]=0.;
    }else{
      // Same tau as the pre-existing D LHS: nu_eff frozen at centroid.
      const double diff=4.0*nueffCentroid/G.h2;
      const double den=fmax(4.0*speed2/G.h2+magic*diff*diff,1e-30);
      const double tau=tauScale/sqrt(den);
      weight[tid]=tau*G.det*s33_rw[ir]*s33_sw[is]*s33_tw[it];
      for(int a=0;a<8;++a)
        stream[tid][a]=vv[0]*gp[a][0]+vv[1]*gp[a][1]+vv[2]*gp[a][2];
      for(int i=0;i<3;++i){
        double lap=0.,graddiv=0.,viscGradient=0.;
        for(int j=0;j<3;++j){
          lap+=H[i][j][j];graddiv+=H[j][i][j];
          const double dij=(i==j?1.0:0.0);
          viscGradient+=dne[j]*(grad[i][j]+grad[j][i]
              -(2.0/3.0)*dij*divergence);
        }
        const double divStress=ne*(lap+graddiv/3.0)+viscGradient;
        corr[tid][i]=divergence*vv[i]-divStress;
      }
    }
  }
  __syncthreads();
  if(tid<24){
    const int i=tid/8,a=tid%8;
    double val=0.;
    for(int q=0;q<64;++q)val-=weight[q]*stream[q][a]*corr[q][i];
    const int id=tv[c].g[a];
    if(i==0)atomicAdd(rhsx+id,(Real)val);
    else if(i==1)atomicAdd(rhsy+id,(Real)val);
    else atomicAdd(rhsz+id,(Real)val);
  }
}
