#pragma once
// Independent five-point Spalding mathematics; no mesh, CUDA runtime or HXT state.
#include <cmath>
#include <cstddef>
#if defined(__CUDACC__)
#define NODALS_WALL_HD __host__ __device__ __forceinline__
#else
#define NODALS_WALL_HD inline
#endif
namespace nodals { namespace wall {
using std::isfinite;
NODALS_WALL_HD double spy(double up,double k,double B){double x=k*up,rem;if(fabs(x)<1e-3){double x2=x*x,x4=x2*x2;rem=x4*(1.0/24+x/120+x2/720+x2*x/5040);}else rem=expm1(x)-x-.5*x*x-x*x*x/6;return up+exp(-k*B)*rem;}
NODALS_WALL_HD double spg(double up,double reY){return up*spy(up,.4,5.5)-reY;}
// ============================================================
// HXT5B_SPALDING_NEWTON_V1
//
// Same Spalding wall law/root as before.
//
// Old:
//   bracket + 70 fixed bisection iterations.
//
// New:
//   same bracket
//   physics-informed initial guess
//   safeguarded Newton inside bracket
//
// If Newton ever proposes a point outside the valid bracket,
// fall back to the bracket midpoint.
//
// No change to U_tau, y+, beta definitions.
// ============================================================

NODALS_WALL_HD
double spyd_newton(double up,double k,double B)
{
  const double x=k*up;
  double drem;

  if(fabs(x)<1e-3){
    const double x2=x*x;
    const double x3=x2*x;

    // d/dup [
    // exp(x)-1-x-x^2/2-x^3/6
    // ]
    //
    // = k [x^3/6+x^4/24+x^5/120+...]
    drem =
      k*x3*(
        1.0/6.0 +
        x/24.0 +
        x2/120.0 +
        x2*x/720.0 +
        x2*x2/5040.0
      );
  }
  else {
    drem =
      k*(
        exp(x) -
        1.0 -
        x -
        0.5*x*x
      );
  }

  return
    1.0 +
    exp(-k*B)*drem;
}


NODALS_WALL_HD
double spgd_newton(double up,double reY)
{
  (void)reY;

  const double S=spy(up,.4,5.5);

  return
    S +
    up*spyd_newton(up,.4,5.5);
}


NODALS_WALL_HD
bool spalding(
    double slip,
    double y,
    double nu,
    double&ut,
    double&yp,
    double&beta)
{
  const double U=fabs(slip);
  const double reY=y*U/nu;

  if(!(y>0) ||
     !(nu>0) ||
     !isfinite(U))
    return false;

  if(reY<=1e-14){
    ut=0;
    yp=0;
    beta=nu/y;
    return true;
  }

  // ----------------------------------------------------------
  // Preserve original root bracket.
  // ----------------------------------------------------------
  double lo=0.0;
  double hi=fmax(
    1.0,
    sqrt(reY)+1.0);

  for(int k=0;
      k<40 && spg(hi,reY)<0.0;
      ++k)
    hi*=2.0;

  const double gh=spg(hi,reY);

  if(!isfinite(gh) || gh<0.0)
    return false;

  // ----------------------------------------------------------
  // Initial guess.
  //
  // Viscous region:
  //     U+ ~= sqrt(Re_y)
  //
  // Log region:
  // fixed-point form of
  //     U+ ~= (1/kappa) ln(E Re_y / U+)
  // ----------------------------------------------------------
  double up;

  if(reY<50.0){
    up=sqrt(reY);
  }
  else {
    constexpr double invK=2.5;
    constexpr double E=
      9.025013499434122; // exp(0.4*5.5)

    up=
      invK*
      log(
        fmax(
          E*reY,
          1.0000001));

#pragma unroll
    for(int k=0;k<4;++k){
      const double arg=
        E*reY/fmax(up,1e-12);

      if(arg>1.0)
        up=invK*log(arg);
    }
  }

  if(!(up>lo && up<hi) ||
     !isfinite(up))
    up=0.5*(lo+hi);

  // ----------------------------------------------------------
  // Safeguarded Newton.
  //
  // Six steps were enough to reproduce the old bisection root
  // to essentially FP64 accuracy over a broad Re_y range.
  // ----------------------------------------------------------
#pragma unroll
  for(int k=0;k<6;++k){

    const double g=spg(up,reY);

    if(!isfinite(g))
      return false;

    if(fabs(g) <=
       1e-13*fmax(reY,1e-300))
      break;

    if(g>0.0)
      hi=up;
    else
      lo=up;

    const double dg=
      spgd_newton(up,reY);

    double candidate;

    if(isfinite(dg) && dg>0.0)
      candidate=up-g/dg;
    else
      candidate=0.5*(lo+hi);

    // Maintain the root bracket unconditionally.
    if(!(candidate>lo &&
         candidate<hi) ||
       !isfinite(candidate))
      candidate=0.5*(lo+hi);

    up=candidate;
  }

  // Final validity check.
  const double gf=spg(up,reY);

  if(!isfinite(gf) ||
     !(up>0.0))
    return false;

  ut=U/up;
  yp=y*ut/nu;
  beta=ut*ut/U;

  return
    isfinite(beta) &&
    beta>0.0 &&
    isfinite(yp);
}



NODALS_WALL_HD
double spalding_profile5_objective(
    const double U[5],const double y[5],double nu,double ut)
{
  if(!(ut>0.0)||!(nu>0.0)||!isfinite(ut))return 1.0e300;
  double J=0.0;
  #pragma unroll
  for(int m=0;m<5;++m){
    const double r=spy(U[m]/ut,.4,5.5)-y[m]*ut/nu;
    if(!isfinite(r))return 1.0e300;
    J+=r*r;
  }
  return J/5.0;
}


// Stage 2.C.1: one evaluation of each Spalding residual per
// Gauss-Newton iteration rather than separate objective and derivative loops.
// Exact legacy objective J=sum(r^2)/n and GN g=sum(r dr), h=sum(dr^2).
// Same FP64 operations and summation order within each accumulator.
NODALS_WALL_HD bool spalding_gn_accumulate(
    const double U[5],const double y[5],int n,double nu,double ut,
    double&J,double&g,double&h)
{
  if(!(n>=1 && n<=5 && nu>0.0 && ut>0.0) || !isfinite(ut))return false;
  double sum=0.0;
  g=0.0;h=0.0;
  #pragma unroll
  for(int m=0;m<n;++m){
    const double up=U[m]/ut;
    const double r=spy(up,.4,5.5)-y[m]*ut/nu;
    const double dr=-(up/ut)*spyd_newton(up,.4,5.5)-y[m]/nu;
    if(!isfinite(r)||!isfinite(dr))return false;
    sum+=r*r;
    g+=r*dr;
    h+=dr*dr;
  }
  J=sum/(double)n;
  return isfinite(J) && isfinite(g) && isfinite(h);
}

// Pure mathematical profile fit; callers provide velocities and physical heights.
// Values may be preinterpolated by a CUDA tetrahedron cache or by a CPU mesh backend.
NODALS_WALL_HD bool fit_profile5_values(
    const double U[5],const double Usigned[5],const double y[5],double nu,
    double&ut,double&ypMid,double&beta,double&uMid,
    double&fitRms,double&pointUTauRelSpan)
{
  if(!(nu>0.0))return false;
  double root[5]={0,0,0,0,0};
  double minRoot=1.0e300,maxRoot=0.0,sumRoot=0.0,Umax=0.0;
  int nRoot=0;
  #pragma unroll
  for(int m=0;m<5;++m){
    Umax=fmax(Umax,U[m]);
    double r=0.0,yp=0.0,bt=0.0;
    const bool rok=spalding(U[m],y[m],nu,r,yp,bt);
    if(rok && r>0.0 && isfinite(r)){
      root[m]=r; minRoot=fmin(minRoot,r); maxRoot=fmax(maxRoot,r);
      sumRoot+=r; ++nRoot;
    }
  }
  #pragma unroll
  for(int m=1;m<5;++m)if(!(y[m-1]<y[m]))return false;

  if(Umax<=1.0e-14){
    ut=0.0; ypMid=0.0; uMid=Usigned[2]; beta=nu/y[2];
    fitRms=0.0; pointUTauRelSpan=0.0;
    return isfinite(beta)&&beta>0.0;
  }

  double scale=0.0;
  if(nRoot>0)scale=sumRoot/(double)nRoot;
  else scale=sqrt(nu*Umax/fmax(y[2],1.0e-30));
  if(!(scale>0.0)||!isfinite(scale))scale=1.0;

  double lo=fmax(0.05*scale,1.0e-10);
  double hi=fmax(20.0*scale,100.0*lo);
  if(nRoot>0){
    lo=fmax(fmin(lo,0.20*minRoot),1.0e-10);
    hi=fmax(hi,5.0*maxRoot);
  }
  ut=fmin(hi,fmax(lo,scale));

  bool gnOK=true;
  #pragma unroll
  for(int it=0;it<12;++it){
    double J0=0.0,g=0.0,h=0.0;
    if(!spalding_gn_accumulate(U,y,5,nu,ut,J0,g,h) ||
       J0>=1.0e299 || !(h>0.0)){gnOK=false;break;}
    double step=g/h;
    double trial=fmin(hi,fmax(lo,ut-step));
    double Jt=spalding_profile5_objective(U,y,nu,trial);
    #pragma unroll
    for(int ls=0;ls<10 && (!isfinite(Jt)||Jt>=1.0e299||Jt>J0);++ls){
      step*=0.5; trial=fmin(hi,fmax(lo,ut-step));
      Jt=spalding_profile5_objective(U,y,nu,trial);
    }
    if(!isfinite(Jt)||Jt>=1.0e299||Jt>J0){gnOK=false;break;}
    const double rel=fabs(trial-ut)/fmax(ut,1.0e-30);
    ut=trial; if(rel<1.0e-10)break;
  }

  double J=spalding_profile5_objective(U,y,nu,ut);
  if(!gnOK || !(ut>0.0) || !isfinite(ut) || !isfinite(J) || J>=1.0e299){
    double xa=log(lo),xb=log(hi);
    const double phi=0.6180339887498948482;
    double x1=xb-phi*(xb-xa),x2=xa+phi*(xb-xa);
    double f1=spalding_profile5_objective(U,y,nu,exp(x1));
    double f2=spalding_profile5_objective(U,y,nu,exp(x2));
    #pragma unroll
    for(int it=0;it<28;++it){
      if(f1<=f2){
        xb=x2; x2=x1; f2=f1; x1=xb-phi*(xb-xa);
        f1=spalding_profile5_objective(U,y,nu,exp(x1));
      }else{
        xa=x1; x1=x2; f1=f2; x2=xa+phi*(xb-xa);
        f2=spalding_profile5_objective(U,y,nu,exp(x2));
      }
    }
    const double xbest=(f1<=f2)?x1:x2;
    ut=exp(xbest); J=(f1<=f2)?f1:f2;
  }

  if(!(ut>0.0)||!isfinite(ut)||!isfinite(J)||J>=1.0e299)return false;
  ypMid=y[2]*ut/nu; uMid=Usigned[2];
  const double Ua=fabs(uMid);
  beta=(Ua>1.0e-14)?ut*ut/Ua:nu/y[2];
  fitRms=sqrt(fmax(J,0.0));
  pointUTauRelSpan=(nRoot>=2 && maxRoot>=minRoot && minRoot<1.0e299)
    ?(maxRoot-minRoot)/fmax(ut,1.0e-30):0.0;
  return isfinite(ypMid)&&isfinite(beta)&&beta>0.0&&
         isfinite(fitRms)&&isfinite(pointUTauRelSpan);
}
}} // nodals::wall
#undef NODALS_WALL_HD
