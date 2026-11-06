#pragma once
// Stage24B: real N-point least-squares wall-law inference, N=1..5, with a
// fixed five-slot ray cache. N=5 deliberately calls the frozen Stage22 fit.
#include "spalding_vector_core.hpp"
#if defined(__CUDACC__)
__device__ __constant__ int stage27_spalding_zero_limit=0;
#endif
#if defined(__CUDACC__)
#define S24B_HD __host__ __device__ __forceinline__
#else
#define S24B_HD inline
#endif
namespace nodals {namespace wall {namespace active {
// Small pure function also serves as the CPU regression target.
S24B_HD double select_picard_beta(double ut,double speed,double floor,
    double betaFit,int n,bool viscousLimit){
  if(viscousLimit && n<5 && !(ut>0.))return betaFit;
  return speed>floor?ut*ut/speed:betaFit;
}
S24B_HD bool viscous_limit_enabled(){
#if defined(__CUDA_ARCH__)
  return ::stage27_spalding_zero_limit!=0;
#else
  return false; // CPU host calls use explicit select_picard_beta opt-in.
#endif
}

S24B_HD int slot(int n,int j){
  // Always use valid, distinct ray positions and preserve the old 5-sample
  // fit exactly. For N=1 the physical center sample is used.
  const int ids[5][5]={{2,-1,-1,-1,-1},{0,4,-1,-1,-1},
                       {0,2,4,-1,-1},{0,1,3,4,-1},{0,1,2,3,4}};
  return (n>=1&&n<=5&&j>=0&&j<n)?ids[n-1][j]:-1;
}
// Stage 2.A: static mask of physical ray slots used by each N-point fit.
// The legacy five-slot geometry remains intact and always validated.
// [1]->slot2; [2]->slots0,4; [3]->slots0,2,4; [4]->0,1,3,4;
// [5]->0,1,2,3,4. Only interpolation work is skipped, not fitting.
S24B_HD unsigned active_slot_mask(int n){
  switch(n){
    case 1:return 0x04u;
    case 2:return 0x11u;
    case 3:return 0x15u;
    case 4:return 0x1bu;
    case 5:return 0x1fu;
    default:return 0u;
  }
}
S24B_HD double objective(const double U[5],const double y[5],int n,double nu,double ut){
  if(!(n>=1&&n<=5&&ut>0&&nu>0)||!isfinite(ut))return 1.0e300;
  double J=0.;
  for(int m=0;m<n;++m){
    const double r=spy(U[m]/ut,.4,5.5)-y[m]*ut/nu;
    if(!isfinite(r))return 1.0e300;
    J+=r*r;
  }
  return J/(double)n;
}
S24B_HD bool fit_values(const double Uall[5],const double Sall[5],
  const double yall[5],int n,double nu,double&ut,double&yp,double&beta,
  double&uReference,double&rms,double&relSpan){
  if(n==5)return fit_profile5_values(Uall,Sall,yall,nu,ut,yp,beta,uReference,rms,relSpan);
  if(n<1||n>4||!(nu>0.))return false;
  double U[5]={},S[5]={},y[5]={};
  double rlo=1.e300,rhi=0.,rsum=0.,Umax=0.;int nr=0;
  for(int m=0;m<n;++m){
    const int id=slot(n,m);U[m]=Uall[id];S[m]=Sall[id];y[m]=yall[id];
    if(!(y[m]>0.)||!isfinite(y[m])||!isfinite(U[m])||!isfinite(S[m])||
       (m>0&&!(y[m]>y[m-1])))return false;
    Umax=fmax(Umax,U[m]);
    double rr=0.,yr=0.,bb=0.;
    if(spalding(U[m],y[m],nu,rr,yr,bb)&&rr>0.&&isfinite(rr)){
      rlo=fmin(rlo,rr);rhi=fmax(rhi,rr);rsum+=rr;++nr;
    }
  }
  const int mid=n/2;uReference=S[mid];
  if(Umax<=1.e-14){ut=0.;yp=0.;beta=nu/y[mid];rms=0.;relSpan=0.;
    return isfinite(beta)&&beta>0.;}
  double scale=nr?rsum/nr:sqrt(nu*Umax/fmax(y[mid],1.e-30));
  if(!(scale>0.)||!isfinite(scale))scale=1.;
  double lo=fmax(.05*scale,1.e-10),hi=fmax(20.*scale,100.*lo);
  if(nr){lo=fmax(fmin(lo,.20*rlo),1.e-10);hi=fmax(hi,5.*rhi);}
  ut=fmin(hi,fmax(lo,scale));bool gn=true;
  for(int it=0;it<12;++it){
    double J0=0.,g=0.,h=0.;
    if(!spalding_gn_accumulate(U,y,n,nu,ut,J0,g,h) ||
       J0>=1.e299 || !(h>0.)){gn=false;break;}
    double step=g/h,trial=fmin(hi,fmax(lo,ut-step));
    double Jt=objective(U,y,n,nu,trial);
    for(int ls=0;ls<10&&(!isfinite(Jt)||Jt>=1.e299||Jt>J0);++ls){
      step*=.5;trial=fmin(hi,fmax(lo,ut-step));Jt=objective(U,y,n,nu,trial);
    }
    if(!isfinite(Jt)||Jt>=1.e299||Jt>J0){gn=false;break;}
    const double rel=fabs(trial-ut)/fmax(ut,1.e-30);
    ut=trial;if(rel<1.e-10)break;
  }
  double J=objective(U,y,n,nu,ut);
  if(!gn||!(ut>0.)||!isfinite(ut)||!isfinite(J)||J>=1.e299){
    double xa=log(lo),xb=log(hi),phi=.6180339887498948482;
    double x1=xb-phi*(xb-xa),x2=xa+phi*(xb-xa);
    double f1=objective(U,y,n,nu,exp(x1)),f2=objective(U,y,n,nu,exp(x2));
    for(int it=0;it<28;++it){
      if(f1<=f2){xb=x2;x2=x1;f2=f1;x1=xb-phi*(xb-xa);
        f1=objective(U,y,n,nu,exp(x1));}
      else{xa=x1;x1=x2;f1=f2;x2=xa+phi*(xb-xa);
        f2=objective(U,y,n,nu,exp(x2));}
    }
    ut=exp(f1<=f2?x1:x2);J=fmin(f1,f2);
  }
  if(!(ut>0.)||!isfinite(ut)||!isfinite(J)||J>=1.e299)return false;
  yp=y[mid]*ut/nu;
  beta=fabs(uReference)>1.e-14?ut*ut/fabs(uReference):nu/y[mid];
  rms=sqrt(fmax(J,0.));
  relSpan=nr>=2&&rhi>=rlo?(rhi-rlo)/fmax(ut,1.e-30):0.;
  return isfinite(yp)&&isfinite(beta)&&beta>0.&&isfinite(rms)&&isfinite(relSpan);
}
}}} // nodals::wall::active
namespace nodals {namespace wall {namespace vector3 {
S24B_HD Result fit_active(const V3 sampled[5],const double y[5],V3 n,
  V3 uPicard,V3 uWall,double nu,int count){
  if(count==5)return fit_five(sampled,y,n,uPicard,uWall,nu);
  Result out{};
  if(count<1||count>4||!sampled||!y||!unit_normal(n)||!finite(uPicard)||
     !finite(uWall)||!(nu>0.)||!isfinite(nu))return out;
  double U[5]={},Y[5]={};
  double prevActiveY=0.;
  const unsigned mask=active::active_slot_mask(count);
  for(int i=0;i<5;++i){
    if(!(mask&(1u<<i))){U[i]=0.;Y[i]=y[i];continue;}
    if(!finite(sampled[i])||!isfinite(y[i])||!(y[i]>prevActiveY))return out;
    prevActiveY=y[i];
    U[i]=norm(tangent_projection(minus(sampled[i],uWall),n));Y[i]=y[i];
  }
  double uMid=0.,betaFit=0.;
  if(!active::fit_values(U,U,Y,count,nu,out.uTau,out.yPlus,betaFit,
                          uMid,out.fitRms,out.pointUTauRelSpan))return out;
  const V3 vbar=tangent_projection(minus(uPicard,uWall),n);
  const double speed=norm(vbar);
  out.meanTangentialSpeed=speed;
  const double floor=1.e-6*fmax(fabs(uMid),1.);
  // For a stationary wall-parallel sample the Spalding fit deliberately
  // returns uTau=0 and the molecular limiting Robin coefficient nu/y.
  // Do not overwrite that positive limit with 0/speed when the Picard
  // trace has a small tangential component. All positive-uTau fits retain
  // the unchanged Stage24B linearization.
  out.beta=active::select_picard_beta(out.uTau,speed,floor,betaFit,
                                      count,active::viscous_limit_enabled());
  if(!isfinite(out.beta)||!(out.beta>0.))return out;
  out.tangent=tangent_tensor(n,out.beta);
  out.traction=times(-1.,multiply(out.tangent,minus(uPicard,uWall)));
  out.valid=finite(out.traction)&&isfinite(out.fitRms)&&isfinite(out.yPlus);
  return out;
}
}}}
#include "stage27_s02c3_shared_fit.hpp"
#undef S24B_HD
