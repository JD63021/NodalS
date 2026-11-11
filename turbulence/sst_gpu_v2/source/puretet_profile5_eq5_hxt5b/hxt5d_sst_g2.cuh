#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>
#include <chrono> // M3 coupled CSR fixed-sweep timing
#include "puretet_exp2_core.hpp"
#include "modules/coupling/dg_omega_g2_adapter.cuh"

// TRUE_Q1_TRANSPORT_V1
// G2: fully coupled SST-2003m feedback for the existing HXT5B SIMPLE loop.
// k/omega use the G1 scalar operators.  Momentum keeps the existing DG inlet,
// Spalding wall and pressure coupling; only the turbulent viscosity provider is
// switched from Nikuradse mixing length to SST(k,omega,U).
namespace hxt5d {
using Real=hxt4b::Real;
using nodals_hxt1::Dev;

struct BF3BernsteinFace {int v0,v1,v2;};
#include "modules/coupling/g2_state.inc"
#include "modules/coupling/bernstein_bf3_tet.inc"
#include "modules/wall/stage22d4b_nitsche_live.inc"
#include "modules/wall/stage22d4c_wall_diagnostics.inc"
#include "modules/wall/stage22d4i_constraint.inc"
#include "modules/wall/stage22d4i2_residual.cuh"
#include "modules/coupling/stage22d4i3a_schur.cuh"
#include "modules/coupling/implicit_dev_volume_tet.inc"
// NODALS_EMAC_PICARD_V1
#include "modules/coupling/emac_picard_tet.inc"
#include "modules/wall/stage22d4d_coupled_matrixfree.inc"
struct StepResult {
  double kRel=0,omegaRel=0,nutRel=0,wallOmegaMean=0,wallOmegaSampleMean=0,wallOmegaTraceMean=0,wallYPlusMean=0;
  hxt5c::Stats stats;
  hxt5c::LinearResult kLinear,omegaLinear;
  unsigned long long kClips=0,omegaClips=0;
};

#include "puretet_omega_diag_v1.hpp"
#include "puretet_omega_matrix_audit_exp1.hpp"
#include "puretet_omega_frozen_exp2.hpp"
#include "puretet_exp3_gamma.hpp"

// ---------------------------------------------------------------------------
// EXPERIMENTAL MOMENTUM WALL MODES
//
// spalding:
//   Existing off-wall Spalding wall-stress operator.
//
// weak_molecular:
//   Homogeneous tangential no-slip imposed weakly with the symmetric,
//   adjoint-consistent Nitsche form, using molecular nu in all wall terms.
//
// weak_effective:
//   Same weak BC, but use nu + nut(sample) consistently in both consistency
//   terms and the penalty term.  nut(sample) is evaluated from the current SST
//   state at the same off-wall FE sample location.
//
// The current HXT5B pipe topology strongly clamps Ux/Uy on the cylindrical
// wall.  The operator below therefore acts on the free axial/tangential Uz.
// ---------------------------------------------------------------------------
enum MomentumWallMode {
  MOM_WALL_SPALDING=0,
  MOM_WALL_WEAK_MOLECULAR=1,
  MOM_WALL_WEAK_EFFECTIVE=2,
  MOM_WALL_SPALDING_TRACE=3,
  MOM_WALL_SPALDING_ALIGNED=4,
  MOM_WALL_SPALDING_COMPATIBLE=5,
  MOM_WALL_SST_CONSISTENT_TABLE=6,
  MOM_WALL_SPALDING_PROFILE_INTEGRAL=7
};

inline const char* momentum_wall_mode_name(int m){
  return m==MOM_WALL_WEAK_MOLECULAR ? "weak_molecular" :
         m==MOM_WALL_WEAK_EFFECTIVE ? "weak_effective" :
         m==MOM_WALL_SPALDING_TRACE ? "spalding_trace" :
         m==MOM_WALL_SPALDING_ALIGNED ? "spalding_aligned" :
         m==MOM_WALL_SPALDING_COMPATIBLE ? "spalding_compatible" :
         m==MOM_WALL_SST_CONSISTENT_TABLE ? "sst_consistent_table" :
         m==MOM_WALL_SPALDING_PROFILE_INTEGRAL ? "spalding_profile_integral" :
                                        "spalding";
}

// ---------------------------------------------------------------------------
// Published-style trace Spalding weak wall diagnostic mode.
//
// This is deliberately separate from the accepted off-wall Spalding operator.
// It follows the older weak-Dirichlet / wall-law idea:
//
//   tau_B = u_tau^2 / |u_t,Gamma|
//
// with u_tau obtained by inverting Spalding from the physical-wall FE trace
// and a wall-normal mesh length h_n.  In the present pipe branch Ux/Uy are
// strongly clamped at the cylindrical wall, so this scalar kernel acts on the
// remaining axial/tangential component Uz.
//
// Picard linearization: tau_B is evaluated from the current outer iterate and
// frozen in the next momentum matrix.
//
// If molecularConsistency != 0, add ONLY the symmetric molecular consistency
// terms
//
//   -<v, nu d_n u> - <nu d_n v, u>
//
// to the same physical-wall trace operator.  No extra gamma*nu/h penalty is
// added: the Spalding tau_B term itself is the nonlinear penalty/Robin term.
// ---------------------------------------------------------------------------
__global__ void sst_wall_spalding_trace_hex(
    const hxt5b::HexFacePlan*F,int nf,
    const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,
    const nodals_hxt1::HexVel*hv,const std::int32_t*slot,
    const Real*u2,Real*mat,Real*diag,
    double nu,double sampleFraction,int molecularConsistency,
    double*stats,unsigned long long*bad)
{
  int fi=(int)blockIdx.x;
  if(fi>=nf)return;
  auto f=F[fi];
  auto H=hv[f.cell];

  __shared__ Real U[14];
  __shared__ Real PH[16][14],DN[16][14],BETAW[16],QW[16];
  __shared__ double AR[16],UT2[16],YPW[16],SLW[16],YPMAX[16],UTMAX[16];

  if(threadIdx.x<14)U[threadIdx.x]=u2[H.g[threadIdx.x]];
  __syncthreads();

  if(threadIdx.x<16){
    int q=(int)threadIdx.x,i=q>>2,j=q&3;

    double cw[3]={0,0,0};
    cw[f.fd]=f.fv;
    cw[f.d0]=hxt5a::h5_hex_x[i];
    cw[f.d1]=hxt5a::h5_hex_x[j];

    double cs[3]={cw[0],cw[1],cw[2]};
    cs[f.fd]=f.fv+sampleFraction*(-2.0*f.fv);

    double pw[14],grw[14][3],gpw[14][3],Iw[9],detw;
    hxt2::hex_basis_ref(cw[0],cw[1],cw[2],pw,grw);

    bool ok=
      sampleFraction>0.0 &&
      hxt2::hex_metric(pts,hc[f.cell],cw[0],cw[1],cw[2],Iw,detw);

    if(ok)hxt2::phys_grad14(grw,Iw,gpw);

    double Xw[3]={0,0,0},dXw[3][3]={{0}};
    double Xs[3]={0,0,0},dXs[3][3]={{0}};
    if(ok){
      hxt5b::q1dev(pts,hc[f.cell],cw[0],cw[1],cw[2],Xw,dXw);
      hxt5b::q1dev(pts,hc[f.cell],cs[0],cs[1],cs[2],Xs,dXs);
    }

    double jac=0.0,ydist=0.0,hn=0.0,n[3]={0,0,0};
    if(ok){
      double cr[3]={
        dXw[f.d0][1]*dXw[f.d1][2]-dXw[f.d0][2]*dXw[f.d1][1],
        dXw[f.d0][2]*dXw[f.d1][0]-dXw[f.d0][0]*dXw[f.d1][2],
        dXw[f.d0][0]*dXw[f.d1][1]-dXw[f.d0][1]*dXw[f.d1][0]
      };
      if(cr[0]*f.n[0]+cr[1]*f.n[1]+cr[2]*f.n[2]<0.0){
        cr[0]=-cr[0];cr[1]=-cr[1];cr[2]=-cr[2];
      }
      jac=sqrt(cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]);
      if(jac>0.0){
        n[0]=cr[0]/jac;n[1]=cr[1]/jac;n[2]=cr[2]/jac;
        ydist=(Xw[0]-Xs[0])*n[0]+
              (Xw[1]-Xs[1])*n[1]+
              (Xw[2]-Xs[2])*n[2];
        // Full wall-normal element length.  The auxiliary off-wall sample is
        // used only to reconstruct h_n; its velocity does not enter this mode.
        hn=ydist/sampleFraction;
      }
      ok=jac>0.0&&ydist>0.0&&hn>0.0&&isfinite(hn);
    }

    double slip=0.0,ut=0.0,yp=0.0,beta=0.0;
    if(ok){
      for(int a=0;a<14;++a){
        PH[q][a]=(Real)pw[a];
        DN[q][a]=(Real)(
          gpw[a][0]*n[0]+gpw[a][1]*n[1]+gpw[a][2]*n[2]);
        slip+=(double)U[a]*pw[a];
      }

      // Bazilevs et al. weak-wall construction:
      // wall-law distance y = h_b / C_I^b, with C_I^b = 4.
      const double CIb = 4.0;
      const double yWall = hn/CIb;

      if(!hxt5b::spalding(slip,yWall,nu,ut,yp,beta)){
        beta=nu/yWall;
        ut=0.0;yp=0.0;
        atomicAdd(stats+6,1.0);
      }
    }

    if(!ok){
      atomicAdd(bad,1ULL);
      jac=0.0;slip=ut=yp=beta=0.0;
      for(int a=0;a<14;++a){PH[q][a]=Real(0);DN[q][a]=Real(0);}
    }

    const double w=hxt5a::h5_hex_w[i]*hxt5a::h5_hex_w[j]*jac;
    BETAW[q]=(Real)(beta*w);
    QW[q]=(Real)w;
    AR[q]=w;
    UT2[q]=ut*ut*w;
    YPW[q]=yp*w;
    SLW[q]=fabs(slip)*w;
    YPMAX[q]=yp;
    UTMAX[q]=ut;
  }
  __syncthreads();

  for(int p=(int)threadIdx.x;p<196;p+=(int)blockDim.x){
    int a=p/14,b=p%14;
    Real z=0;
    for(int q=0;q<16;++q){
      z += BETAW[q]*PH[q][a]*PH[q][b];
      if(molecularConsistency){
        z += QW[q]*(Real)nu*(
          -PH[q][a]*DN[q][b]
          -DN[q][a]*PH[q][b]);
      }
    }
    atomicAdd(mat+slot[(size_t)f.cell*196+p],z);
    if(a==b)atomicAdd(diag+H.g[a],z);
  }

  if(threadIdx.x==0){
    double A=0,U2A=0,Y=0,S=0,MY=0,MU=0;
    for(int q=0;q<16;++q){
      A+=AR[q];U2A+=UT2[q];Y+=YPW[q];S+=SLW[q];
      MY=fmax(MY,YPMAX[q]);MU=fmax(MU,UTMAX[q]);
    }
    atomicAdd(stats+0,A);
    atomicAdd(stats+1,U2A);
    atomicAdd(stats+2,Y);
    atomicAdd(stats+3,S);
    hxt5b::amax(stats+4,MY);
    hxt5b::amax(stats+5,MU);
  }
}


// ---------------------------------------------------------------------------
// FE profile-integral Spalding momentum wall mode.
//
// Instead of inverting Spalding at a single off-wall point, use the complete
// wall-normal FE profile in the wall-adjacent HEX.  A 4-point Gauss rule in
// the normal reference direction determines one u_tau from the mean signed
// Spalding y+ residual.  The Picard traction is then linearized against the
// profile-averaged FE velocity functional, not against an arbitrary point.
// ---------------------------------------------------------------------------
__global__ void sst_wall_spalding_profile_integral_hex(
    const hxt5b::HexFacePlan*F,int nf,
    const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,
    const nodals_hxt1::HexVel*hv,const std::int32_t*slot,
    const Real*u2,Real*mat,Real*diag,double nu,int profileIntegralNQ,
    int profileAverageClosure,int profileAverageWallLaw,
    const hxt5b::WallMatchPoint*matchPoints,
    double*stats,unsigned long long*bad)
{
  int fi=(int)blockIdx.x;if(fi>=nf)return;auto f=F[fi];auto H=hv[f.cell];
  __shared__ Real U[14],BW[16][14],BP[16][14],BETAW[16];
  __shared__ double AR[16],UT2[16],YPW[16],SLW[16],YPMAX[16],UTMAX[16];
  __shared__ double RMSW[16],SPANW[16],YBARW[16];
  if(threadIdx.x<14)U[threadIdx.x]=u2[H.g[threadIdx.x]];
  __syncthreads();

  if(threadIdx.x<16){
    const int q=(int)threadIdx.x,i=q>>2,j=q&3;
    double cw[3]={0,0,0};cw[f.fd]=f.fv;cw[f.d0]=hxt5a::h5_hex_x[i];cw[f.d1]=hxt5a::h5_hex_x[j];
    double pw[14],gw[14][3];hxt2::hex_basis_ref(cw[0],cw[1],cw[2],pw,gw);
    double Xw[3],dXw[3][3];hxt5b::q1dev(pts,hc[f.cell],cw[0],cw[1],cw[2],Xw,dXw);
    double cr[3]={dXw[f.d0][1]*dXw[f.d1][2]-dXw[f.d0][2]*dXw[f.d1][1],dXw[f.d0][2]*dXw[f.d1][0]-dXw[f.d0][0]*dXw[f.d1][2],dXw[f.d0][0]*dXw[f.d1][1]-dXw[f.d0][1]*dXw[f.d1][0]};
    if(cr[0]*f.n[0]+cr[1]*f.n[1]+cr[2]*f.n[2]<0.0){cr[0]=-cr[0];cr[1]=-cr[1];cr[2]=-cr[2];}
    double jac=sqrt(cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]);
    double ut=0.0,yp=0.0,beta=0.0,ubar=0.0,ybar=0.0,rms=0.0,span=0.0,bp[14];
    bool ok=jac>0.0&&isfinite(jac);
    bool betaFallback=false;
    if(ok){
      if(profileAverageClosure){
        if(profileAverageWallLaw==5){
          // PROFILE5 owns the wall stress. First-cell NQ integration is
          // kinematic only and supplies the local Picard trial functional.
          double uLocalSigned=0.0;
          ok=hxt5b::profile_average_velocity_basis_hex(
            pts,hc[f.cell],f,cw,U,profileIntegralNQ,uLocalSigned,bp);
          if(ok){
            double betaFit=0.0,um=0.0,fitRms=0.0,fitSpan=0.0;
            ok=hxt5b::spalding_match_profile5(
              matchPoints,fi,q,u2,nu,ut,yp,betaFit,um,fitRms,fitSpan);
            if(ok){
              const double uLocal=fabs(uLocalSigned);
              const double localFloor=1.0e-6*fmax(fabs(um),1.0);
              if(uLocal>localFloor)beta=ut*ut/uLocal;
              else{beta=betaFit;betaFallback=true;}
              ubar=um;
              ybar=matchPoints[((std::size_t)fi*16u+(std::size_t)q)*5u+2u].y;
              rms=fitRms;span=fitSpan;
              ok=isfinite(beta)&&beta>0.0&&isfinite(ut)&&ut>=0.0&&isfinite(yp)&&yp>=0.0;
            }
          }
        }else if(profileAverageWallLaw==4){
          // PROFILE3 owns the wall stress.  The first-cell normal-column
          // integration is retained ONLY to provide the local Picard trial
          // functional Bbar and its current value Ubar_local.  There is no
          // separate first-cell Spalding inversion in this mode.
          double uLocalSigned=0.0;
          ok=hxt5b::profile_average_velocity_basis_hex(
            pts,hc[f.cell],f,cw,U,profileIntegralNQ,
            uLocalSigned,bp);

          if(ok){
            double betaFit=0.0,um=0.0,fitRms=0.0,fitSpan=0.0;
            ok=hxt5b::spalding_match_profile3(
              matchPoints,fi,q,u2,nu,ut,yp,betaFit,um,fitRms,fitSpan);

            if(ok){
              const double uLocal=fabs(uLocalSigned);
              const double localFloor=1.0e-6*fmax(fabs(um),1.0);

              if(uLocal>localFloor){
                // Normal path: current Picard state reproduces tau_w=u_tau^2.
                beta=ut*ut/uLocal;
              }else{
                // Degenerate local FE profile at one quadrature location:
                // do not create an unbounded tau/Ubar coefficient and do not
                // kill the whole solve.  Use the finite profile-fit coefficient.
                beta=betaFit;
                betaFallback=true;
              }

              ubar=um;
              ybar=matchPoints[
                ((std::size_t)fi*16u+(std::size_t)q)*3u+1u].y;
              rms=fitRms;
              span=fitSpan;

              ok=isfinite(beta)&&beta>0.0&&
                 isfinite(ut)&&ut>=0.0&&
                 isfinite(yp)&&yp>=0.0;
            }
          }
        }else if(profileAverageWallLaw==3){
          // Preserve the single-match-height experiment exactly as before.
          double utLocal=0.0,ypLocal=0.0,betaLocal=0.0;
          ok=hxt5b::spalding_profile_average_hex(
            pts,hc[f.cell],f,cw,U,nu,profileIntegralNQ,
            utLocal,ypLocal,betaLocal,ubar,ybar,bp,rms,span);
          if(ok){
            double betaMatch=0.0,um=0.0;
            ok=hxt5b::spalding_match_height_point(
              matchPoints,fi,q,u2,nu,ut,yp,betaMatch,um);
            if(ok){
              const double uLocal=fabs(ubar);
              if(!(uLocal>1e-14))ok=false;
              else{
                beta=ut*ut/uLocal;
                ubar=um;
                ybar=matchPoints[(std::size_t)fi*16u+(std::size_t)q].y;
              }
            }
          }
        }else if(profileAverageWallLaw==2)
          ok=hxt5b::spalding_profile_average_relative_wall_hex(pts,hc[f.cell],f,cw,U,nu,profileIntegralNQ,ut,yp,beta,ubar,ybar,bp,rms,span);
        else if(profileAverageWallLaw==1)
          ok=hxt5b::loglaw_profile_average_hex(pts,hc[f.cell],f,cw,U,nu,profileIntegralNQ,ut,yp,beta,ubar,ybar,bp,rms,span);
        else
          ok=hxt5b::spalding_profile_average_hex(pts,hc[f.cell],f,cw,U,nu,profileIntegralNQ,ut,yp,beta,ubar,ybar,bp,rms,span);
      }else
        ok=hxt5b::spalding_profile_hex(pts,hc[f.cell],f,cw,U,nu,profileIntegralNQ,ut,yp,beta,ubar,ybar,bp,rms,span);
    }
    if(!ok){
      atomicAdd(bad,1ULL);atomicAdd(stats+6,1.0);
      jac=ut=yp=beta=ubar=ybar=rms=span=0.0;
      for(int a=0;a<14;++a)bp[a]=0.0;
    }
    for(int a=0;a<14;++a){BW[q][a]=ok?(Real)pw[a]:Real(0);BP[q][a]=ok?(Real)bp[a]:Real(0);}
    const double w=hxt5a::h5_hex_w[i]*hxt5a::h5_hex_w[j]*jac;
    BETAW[q]=(Real)(beta*w);AR[q]=w;UT2[q]=ut*ut*w;YPW[q]=yp*w;SLW[q]=fabs(ubar)*w;YPMAX[q]=yp;UTMAX[q]=ut;
    RMSW[q]=rms*w;SPANW[q]=span*w;YBARW[q]=ybar*w;
    if(betaFallback && w>0.0)atomicAdd(stats+10,w);
  }
  __syncthreads();

  for(int p=(int)threadIdx.x;p<196;p+=(int)blockDim.x){
    const int a=p/14,b=p%14;Real z=0;
    for(int q=0;q<16;++q)z+=BETAW[q]*BW[q][a]*BP[q][b];
    atomicAdd(mat+slot[(size_t)f.cell*196+p],z);
    if(a==b)atomicAdd(diag+H.g[a],z);
  }

  if(threadIdx.x==0){
    double A=0,T=0,Y=0,S=0,MY=0,MU=0,R=0,SP=0,YB=0;
    for(int q=0;q<16;++q){A+=AR[q];T+=UT2[q];Y+=YPW[q];S+=SLW[q];MY=fmax(MY,YPMAX[q]);MU=fmax(MU,UTMAX[q]);R+=RMSW[q];SP+=SPANW[q];YB+=YBARW[q];}
    atomicAdd(stats+0,A);atomicAdd(stats+1,T);atomicAdd(stats+2,Y);atomicAdd(stats+3,S);hxt5b::amax(stats+4,MY);hxt5b::amax(stats+5,MU);
    atomicAdd(stats+7,R);atomicAdd(stats+8,SP);atomicAdd(stats+9,YB);
  }
}

// ---------------------------------------------------------------------------
// Carrier/Lacombe-compatible Spalding momentum wall mode.
//
// Keep the accepted NodalS off-wall Spalding/Picard operator and sampling
// architecture, but replace tau_w=u_tau^2 by the compatible SST target
// tau_w=u_tau*uStar.  u_tau is still obtained by the existing Spalding
// inversion at sample A.  No molecular Nitsche terms are added in this mode.
// ---------------------------------------------------------------------------
__global__ void sst_wall_spalding_compatible_hex(
    const hxt5b::HexFacePlan*F,int nf,
    const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,
    const nodals_hxt1::HexVel*hv,const std::int32_t*slot,
    const Real*u2,const Real*k,Real*mat,Real*diag,
    double nu,double sampleFraction,double kFloor,
    double*stats,unsigned long long*bad)
{
  int fi=(int)blockIdx.x;if(fi>=nf)return;auto f=F[fi];auto H=hv[f.cell];
  __shared__ Real U[14],K[14],BW[16][14],BS[16][14],BETAW[16];
  __shared__ double AR[16],TAUW[16],YPW[16],SLW[16],YPMAX[16],UTMAX[16];
  if(threadIdx.x<14){int g=H.g[threadIdx.x];U[threadIdx.x]=u2[g];K[threadIdx.x]=k[g];}
  __syncthreads();
  if(threadIdx.x<16){
    int q=(int)threadIdx.x,i=q>>2,j=q&3;
    double cw[3]={0,0,0};cw[f.fd]=f.fv;cw[f.d0]=hxt5a::h5_hex_x[i];cw[f.d1]=hxt5a::h5_hex_x[j];
    double cs[3]={cw[0],cw[1],cw[2]};cs[f.fd]=f.fv+sampleFraction*(-2*f.fv);
    double pw[14],gw[14][3],ps[14],gs[14][3];hxt2::hex_basis_ref(cw[0],cw[1],cw[2],pw,gw);hxt2::hex_basis_ref(cs[0],cs[1],cs[2],ps,gs);
    double Xw[3],dXw[3][3],Xs[3],dXs[3][3];hxt5b::q1dev(pts,hc[f.cell],cw[0],cw[1],cw[2],Xw,dXw);hxt5b::q1dev(pts,hc[f.cell],cs[0],cs[1],cs[2],Xs,dXs);
    double rr=hypot(Xw[0],Xw[1]),jac=0,y=0,sl=0,kA=0,ut=0,yp=0,betaOld=0,beta=0,tau=0;
    bool ok=rr>0&&sampleFraction>0.0&&sampleFraction<=1.0;
    if(ok){double nr[3]={Xw[0]/rr,Xw[1]/rr,0};double cr[3]={dXw[f.d0][1]*dXw[f.d1][2]-dXw[f.d0][2]*dXw[f.d1][1],dXw[f.d0][2]*dXw[f.d1][0]-dXw[f.d0][0]*dXw[f.d1][2],dXw[f.d0][0]*dXw[f.d1][1]-dXw[f.d0][1]*dXw[f.d1][0]};if(cr[0]*nr[0]+cr[1]*nr[1]+cr[2]*nr[2]<0){cr[0]*=-1;cr[1]*=-1;cr[2]*=-1;}jac=sqrt(cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]);y=(Xw[0]-Xs[0])*nr[0]+(Xw[1]-Xs[1])*nr[1];ok=y>0&&jac>0;}
    for(int a=0;a<14;++a){BW[q][a]=ok?(Real)pw[a]:Real(0);BS[q][a]=ok?(Real)ps[a]:Real(0);if(ok){sl+=(double)U[a]*ps[a];kA+=(double)K[a]*ps[a];}}
    if(ok&&!hxt5b::spalding(sl,y,nu,ut,yp,betaOld)){atomicAdd(stats+6,1.0);ok=false;}
    if(ok){const double Uabs=fabs(sl),kval=fmax(kA,kFloor);const double uTauVisc=sqrt(fmax(0.0,nu*Uabs/y));const double uStarLog=pow((double)hxt5c::BETA_STAR,0.25)*sqrt(kval);const double a2=uTauVisc*uTauVisc,b2=uStarLog*uStarLog;const double uStar=sqrt(sqrt(a2*a2+b2*b2));tau=ut*uStar;beta=(Uabs>1e-14)?tau/Uabs:nu/y;ok=isfinite(beta)&&beta>0.0&&isfinite(tau);}
    if(!ok){atomicAdd(bad,1ULL);beta=ut=yp=sl=tau=jac=0;for(int a=0;a<14;++a){BW[q][a]=BS[q][a]=Real(0);}}
    const double w=hxt5a::h5_hex_w[i]*hxt5a::h5_hex_w[j]*jac;BETAW[q]=(Real)(beta*w);AR[q]=w;TAUW[q]=tau*w;YPW[q]=yp*w;SLW[q]=fabs(sl)*w;YPMAX[q]=yp;UTMAX[q]=ut;
  }
  __syncthreads();
  for(int p=(int)threadIdx.x;p<196;p+=(int)blockDim.x){int a=p/14,b=p%14;Real z=0;for(int q=0;q<16;++q)z+=BETAW[q]*BW[q][a]*BS[q][b];atomicAdd(mat+slot[(size_t)f.cell*196+p],z);if(a==b)atomicAdd(diag+H.g[a],z);}
  if(threadIdx.x==0){double A=0,T=0,Y=0,S=0,MY=0,MU=0;for(int q=0;q<16;++q){A+=AR[q];T+=TAUW[q];Y+=YPW[q];S+=SLW[q];MY=fmax(MY,YPMAX[q]);MU=fmax(MU,UTMAX[q]);}atomicAdd(stats+0,A);atomicAdd(stats+1,T);atomicAdd(stats+2,Y);atomicAdd(stats+3,S);hxt5b::amax(stats+4,MY);hxt5b::amax(stats+5,MU);}
}

// ---------------------------------------------------------------------------
// Model-consistent SST table momentum wall mode.
//
// Legacy path: invert the table at the existing off-wall point sample.
// Average-first path: use NQ physical-volume quadrature through the complete
// wall-adjacent normal column to obtain Ubar, ybar and the averaged FE trial
// functional.  Perform exactly ONE table inversion at (|Ubar|,ybar), then use
// tau_w=u_tau^2.  The wall trace remains the test functional, exactly as in the
// profile-averaged Spalding traction.  This avoids interpreting every point in
// a Robin/slip-trace wall cell as an independent wall-law state.
// ---------------------------------------------------------------------------
__global__ void sst_wall_consistent_table_hex(
    const hxt5b::HexFacePlan*F,int nf,
    const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,
    const nodals_hxt1::HexVel*hv,const std::int32_t*slot,
    const Real*u2,Real*mat,Real*diag,
    double nu,double sampleFraction,int profileIntegralNQ,int profileAverageClosure,
    double*stats,unsigned long long*bad)
{
  int fi=(int)blockIdx.x;if(fi>=nf)return;auto f=F[fi];auto H=hv[f.cell];
  __shared__ Real U[14],BW[16][14],BT[16][14],BETAW[16];
  __shared__ double AR[16],TAUW[16],YPW[16],SLW[16],YPMAX[16],UTMAX[16];
  if(threadIdx.x<14){int g=H.g[threadIdx.x];U[threadIdx.x]=u2[g];}
  __syncthreads();
  if(threadIdx.x<16){
    const int q=(int)threadIdx.x,i=q>>2,j=q&3;
    double cw[3]={0,0,0};cw[f.fd]=f.fv;cw[f.d0]=hxt5a::h5_hex_x[i];cw[f.d1]=hxt5a::h5_hex_x[j];
    double pw[14],gw[14][3];hxt2::hex_basis_ref(cw[0],cw[1],cw[2],pw,gw);
    double Xw[3],dXw[3][3];hxt5b::q1dev(pts,hc[f.cell],cw[0],cw[1],cw[2],Xw,dXw);
    double cr[3]={dXw[f.d0][1]*dXw[f.d1][2]-dXw[f.d0][2]*dXw[f.d1][1],dXw[f.d0][2]*dXw[f.d1][0]-dXw[f.d0][0]*dXw[f.d1][2],dXw[f.d0][0]*dXw[f.d1][1]-dXw[f.d0][1]*dXw[f.d1][0]};
    if(cr[0]*f.n[0]+cr[1]*f.n[1]+cr[2]*f.n[2]<0.0){cr[0]=-cr[0];cr[1]=-cr[1];cr[2]=-cr[2];}
    const double jac=sqrt(cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]);
    bool ok=jac>0.0&&isfinite(jac);
    double urep=0.0,yrep=0.0,ut=0.0,yp=0.0,beta=0.0,tau=0.0,btrial[14]={0};

    if(ok&&profileAverageClosure){
      ok=(profileIntegralNQ==4||profileIntegralNQ==8||profileIntegralNQ==12);
      double n[3]={cr[0]/jac,cr[1]/jac,cr[2]/jac};
      double sw=0.0,su=0.0,sy=0.0;
      for(int m=0;ok&&m<profileIntegralNQ;++m){
        double xi=0.0,wref=0.0;
        if(!hxt5b::spalding_profile_gauss_point(profileIntegralNQ,m,xi,wref)){ok=false;break;}
        double c[3]={cw[0],cw[1],cw[2]};c[f.fd]=-f.fv*xi;
        double ph[14],gr[14][3],I[9],det=0.0;
        hxt2::hex_basis_ref(c[0],c[1],c[2],ph,gr);
        if(!hxt2::hex_metric(pts,hc[f.cell],c[0],c[1],c[2],I,det)||!(det>0.0)||!isfinite(det)){ok=false;break;}
        double X[3],dX[3][3];hxt5b::q1dev(pts,hc[f.cell],c[0],c[1],c[2],X,dX);
        const double y=(Xw[0]-X[0])*n[0]+(Xw[1]-X[1])*n[1]+(Xw[2]-X[2])*n[2];
        const double ww=wref*det;
        if(!(y>0.0)||!(ww>0.0)||!isfinite(y)||!isfinite(ww)){ok=false;break;}
        double uq=0.0;for(int a=0;a<14;++a){uq+=(double)U[a]*ph[a];btrial[a]+=ww*ph[a];}
        sw+=ww;su+=ww*uq;sy+=ww*y;
      }
      if(ok)ok=sw>0.0&&isfinite(sw);
      if(ok){urep=su/sw;yrep=sy/sw;for(int a=0;a<14;++a)btrial[a]/=sw;}
    }else if(ok){
      double cs[3]={cw[0],cw[1],cw[2]};cs[f.fd]=f.fv+sampleFraction*(-2*f.fv);
      double ps[14],gs[14][3];hxt2::hex_basis_ref(cs[0],cs[1],cs[2],ps,gs);
      double Xs[3],dXs[3][3];hxt5b::q1dev(pts,hc[f.cell],cs[0],cs[1],cs[2],Xs,dXs);
      double n[3]={cr[0]/jac,cr[1]/jac,cr[2]/jac};
      yrep=(Xw[0]-Xs[0])*n[0]+(Xw[1]-Xs[1])*n[1]+(Xw[2]-Xs[2])*n[2];
      ok=yrep>0.0&&sampleFraction>0.0&&sampleFraction<=1.0;
      for(int a=0;a<14;++a){btrial[a]=ps[a];if(ok)urep+=(double)U[a]*ps[a];}
    }

    hxt5c_table::WallTableState ta;
    if(ok){ta=hxt5c_table::invert_velocity(fabs(urep),yrep,nu);ok=ta.ok;}
    if(ok){ut=ta.uTau;yp=ta.yPlus;tau=ut*ut;beta=(fabs(urep)>1e-14)?tau/fabs(urep):nu/yrep;ok=isfinite(beta)&&beta>0.0&&isfinite(tau);}
    if(!ok){atomicAdd(bad,1ULL);atomicAdd(stats+6,1.0);ut=yp=beta=tau=urep=0.0;for(int a=0;a<14;++a)btrial[a]=0.0;}
    for(int a=0;a<14;++a){BW[q][a]=ok?(Real)pw[a]:Real(0);BT[q][a]=ok?(Real)btrial[a]:Real(0);}
    const double w=hxt5a::h5_hex_w[i]*hxt5a::h5_hex_w[j]*(ok?jac:0.0);
    BETAW[q]=(Real)(beta*w);AR[q]=w;TAUW[q]=tau*w;YPW[q]=yp*w;SLW[q]=fabs(urep)*w;YPMAX[q]=yp;UTMAX[q]=ut;
  }
  __syncthreads();
  for(int p=(int)threadIdx.x;p<196;p+=(int)blockDim.x){int a=p/14,b=p%14;Real z=0;for(int q=0;q<16;++q)z+=BETAW[q]*BW[q][a]*BT[q][b];atomicAdd(mat+slot[(size_t)f.cell*196+p],z);if(a==b)atomicAdd(diag+H.g[a],z);}
  if(threadIdx.x==0){double A=0,T=0,Y=0,S=0,MY=0,MU=0;for(int q=0;q<16;++q){A+=AR[q];T+=TAUW[q];Y+=YPW[q];S+=SLW[q];MY=fmax(MY,YPMAX[q]);MU=fmax(MU,UTMAX[q]);}atomicAdd(stats+0,A);atomicAdd(stats+1,T);atomicAdd(stats+2,Y);atomicAdd(stats+3,S);hxt5b::amax(stats+4,MY);hxt5b::amax(stats+5,MU);}
}

// ---------------------------------------------------------------------------
// WD3-aligned weak Spalding wall mode (Golshan et al. 2015 architecture).
//
// The law of the wall is evaluated at the existing off-wall sample A:
//
//   U_A, y_A  --Spalding-->  u_tau
//
// The physical wall trace supplies the weak Dirichlet penalty direction/state.
// For the scalar axial/tangential component in this all-HEX pipe branch,
//
//   tau_B^wall = [u_tau^2 - tau_nu,h] / |u_Gamma|
//   tau_nu,h    = -nu * (du_h/dn)|_Gamma projected into the flow direction
//   tau_B^WD    = C_b * nu / h_b
//   tau_B       = max(tau_B^wall, tau_B^WD)
//
// with h_b reconstructed from y_A/sampleFraction.  The wall-law target u_tau
// still comes from the off-wall sample A.  The molecular stress subtraction
// now uses the actual FE wall-normal derivative, while nu*|U_A|/y_A is kept
// only as a diagnostic.  Here Spalding is retained as the chosen law of the
// wall; the 2015 paper used a logarithmic law for u_tau in its WD3 tests.
//
// Picard treatment: tau_B is evaluated from the current outer iterate and is
// frozen in the next momentum matrix.
//
// If molecularConsistency==1, also add the two molecular Nitsche consistency
// terms.  In that case the molecular shear estimate is subtracted in
// tau_B^wall.  If molecularConsistency==0, no molecular shear is subtracted,
// because that resolved wall-stress contribution is not present in the weak
// boundary operator.
// ---------------------------------------------------------------------------
__global__ void sst_wall_spalding_aligned_hex(
    const hxt5b::HexFacePlan*F,int nf,
    const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,
    const nodals_hxt1::HexVel*hv,const std::int32_t*slot,
    const Real*u2,Real*mat,Real*diag,
    double nu,double sampleFraction,int molecularConsistency,double Cb,
    double*stats,unsigned long long*bad)
{
  int fi=(int)blockIdx.x;
  if(fi>=nf)return;
  auto f=F[fi];
  auto H=hv[f.cell];

  __shared__ Real U[14];
  __shared__ Real PH[16][14],DN[16][14],BETAW[16],QW[16];
  __shared__ double AR[16],UT2[16],YPW[16],SLW[16],YPMAX[16],UTMAX[16];
  __shared__ double FLOORAREA[16],NEGAREA[16],MOLW[16],MOLSECW[16],PENW[16];

  if(threadIdx.x<14)U[threadIdx.x]=u2[H.g[threadIdx.x]];
  __syncthreads();

  if(threadIdx.x<16){
    int q=(int)threadIdx.x,i=q>>2,j=q&3;

    double cw[3]={0,0,0};
    cw[f.fd]=f.fv;
    cw[f.d0]=hxt5a::h5_hex_x[i];
    cw[f.d1]=hxt5a::h5_hex_x[j];

    double cs[3]={cw[0],cw[1],cw[2]};
    cs[f.fd]=f.fv+sampleFraction*(-2.0*f.fv);

    double pw[14],grw[14][3],gpw[14][3],Iw[9],detw;
    double ps[14],grs[14][3];
    hxt2::hex_basis_ref(cw[0],cw[1],cw[2],pw,grw);
    hxt2::hex_basis_ref(cs[0],cs[1],cs[2],ps,grs);

    bool ok=
      sampleFraction>0.0 && sampleFraction<=1.0 && Cb>0.0 &&
      hxt2::hex_metric(pts,hc[f.cell],cw[0],cw[1],cw[2],Iw,detw);
    if(ok)hxt2::phys_grad14(grw,Iw,gpw);

    double Xw[3]={0,0,0},dXw[3][3]={{0}};
    double Xs[3]={0,0,0},dXs[3][3]={{0}};
    if(ok){
      hxt5b::q1dev(pts,hc[f.cell],cw[0],cw[1],cw[2],Xw,dXw);
      hxt5b::q1dev(pts,hc[f.cell],cs[0],cs[1],cs[2],Xs,dXs);
    }

    double jac=0.0,yA=0.0,hb=0.0,n[3]={0,0,0};
    if(ok){
      double cr[3]={
        dXw[f.d0][1]*dXw[f.d1][2]-dXw[f.d0][2]*dXw[f.d1][1],
        dXw[f.d0][2]*dXw[f.d1][0]-dXw[f.d0][0]*dXw[f.d1][2],
        dXw[f.d0][0]*dXw[f.d1][1]-dXw[f.d0][1]*dXw[f.d1][0]
      };
      if(cr[0]*f.n[0]+cr[1]*f.n[1]+cr[2]*f.n[2]<0.0){
        cr[0]=-cr[0];cr[1]=-cr[1];cr[2]=-cr[2];
      }
      jac=sqrt(cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]);
      if(jac>0.0){
        n[0]=cr[0]/jac;n[1]=cr[1]/jac;n[2]=cr[2]/jac;
        yA=(Xw[0]-Xs[0])*n[0]+
           (Xw[1]-Xs[1])*n[1]+
           (Xw[2]-Xs[2])*n[2];
        hb=yA/sampleFraction;
      }
      ok=jac>0.0&&yA>0.0&&hb>0.0&&isfinite(yA)&&isfinite(hb);
    }

    double uWall=0.0,uA=0.0,dnUWall=0.0,ut=0.0,yp=0.0,betaSample=0.0;
    double tauMol=0.0,tauMolSecant=0.0,betaWall=0.0,betaWD=0.0,beta=0.0;
    bool floorActive=false,negativeDeficit=false;

    if(ok){
      for(int a=0;a<14;++a){
        PH[q][a]=(Real)pw[a];
        DN[q][a]=(Real)(
          gpw[a][0]*n[0]+gpw[a][1]*n[1]+gpw[a][2]*n[2]);
        uWall+=(double)U[a]*pw[a];
        uA+=(double)U[a]*ps[a];
        dnUWall+=(double)U[a]*(double)DN[q][a];
      }

      if(!hxt5b::spalding(uA,yA,nu,ut,yp,betaSample)){
        // Singular/root-failure fallback: use the viscous stress supported by
        // the off-wall sample, while preserving the positive WD floor below.
        const double UA=fabs(uA);
        ut=sqrt(fmax(nu*UA/yA,0.0));
        yp=yA*ut/nu;
        atomicAdd(stats+6,1.0);
      }

      const double tauTarget=ut*ut;
      const double UA=fabs(uA);
      const double UW=fabs(uWall);

      // Resolved molecular wall shear from the actual FE wall-normal
      // derivative. n is outward; project into the local flow direction so
      // positive tauMol means resisting drag. Keep nu*|U_A|/y_A only as a
      // diagnostic of the old secant approximation.
      const double flowSign =
        (fabs(uA)>1e-14) ? (uA>0.0 ? 1.0 : -1.0)
                         : (uWall>=0.0 ? 1.0 : -1.0);
      tauMolSecant=molecularConsistency ? nu*UA/yA : 0.0;
      tauMol=molecularConsistency ? (-nu*dnUWall*flowSign) : 0.0;
      const double deficit=tauTarget-tauMol;
      negativeDeficit=(deficit<=0.0);

      // Avoid a numerical singularity at exactly zero wall slip.  In that
      // limit fall back to the ordinary stable weak-Dirichlet coefficient.
      if(UW>1e-14 && isfinite(UW)) betaWall=deficit/UW;
      else betaWall=-1.0e300;

      betaWD=Cb*nu/hb;
      beta=fmax(betaWall,betaWD);
      floorActive=(betaWD>=betaWall);

      ok=isfinite(beta)&&beta>0.0&&isfinite(tauMol)&&isfinite(tauMolSecant);
    }

    if(!ok){
      atomicAdd(bad,1ULL);
      jac=0.0;uWall=uA=dnUWall=ut=yp=tauMol=tauMolSecant=betaWall=betaWD=beta=0.0;
      floorActive=false;negativeDeficit=false;
      for(int a=0;a<14;++a){PH[q][a]=Real(0);DN[q][a]=Real(0);}
    }

    const double w=hxt5a::h5_hex_w[i]*hxt5a::h5_hex_w[j]*jac;
    BETAW[q]=(Real)(beta*w);
    QW[q]=(Real)w;
    AR[q]=w;
    UT2[q]=ut*ut*w;                // target total wall stress
    YPW[q]=yp*w;                   // y+ at off-wall sample A
    SLW[q]=fabs(uWall)*w;           // actual weak wall trace/slip
    YPMAX[q]=yp;
    UTMAX[q]=ut;
    FLOORAREA[q]=floorActive?w:0.0;
    NEGAREA[q]=negativeDeficit?w:0.0;
    MOLW[q]=tauMol*w;
    MOLSECW[q]=tauMolSecant*w;
    PENW[q]=beta*fabs(uWall)*w;     // actual penalty-stress magnitude
  }
  __syncthreads();

  for(int p=(int)threadIdx.x;p<196;p+=(int)blockDim.x){
    int a=p/14,b=p%14;
    Real z=0;
    for(int q=0;q<16;++q){
      z += BETAW[q]*PH[q][a]*PH[q][b];
      if(molecularConsistency){
        z += QW[q]*(Real)nu*(
          -PH[q][a]*DN[q][b]
          -DN[q][a]*PH[q][b]);
      }
    }
    atomicAdd(mat+slot[(size_t)f.cell*196+p],z);
    if(a==b)atomicAdd(diag+H.g[a],z);
  }

  if(threadIdx.x==0){
    double A=0,U2A=0,Y=0,S=0,MY=0,MU=0,FA=0,NA=0,TM=0,TMS=0,TP=0;
    for(int q=0;q<16;++q){
      A+=AR[q];U2A+=UT2[q];Y+=YPW[q];S+=SLW[q];
      MY=fmax(MY,YPMAX[q]);MU=fmax(MU,UTMAX[q]);
      FA+=FLOORAREA[q];NA+=NEGAREA[q];TM+=MOLW[q];TMS+=MOLSECW[q];TP+=PENW[q];
    }
    atomicAdd(stats+0,A);
    atomicAdd(stats+1,U2A);
    atomicAdd(stats+2,Y);
    atomicAdd(stats+3,S);
    hxt5b::amax(stats+4,MY);
    hxt5b::amax(stats+5,MU);
    atomicAdd(stats+7,FA);
    atomicAdd(stats+8,NA);
    atomicAdd(stats+9,TM);
    atomicAdd(stats+10,TP);
    atomicAdd(stats+11,TMS);
  }
}

__global__ void sst_wall_nitsche_hex(
    const hxt5b::HexFacePlan*F,int nf,
    const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,
    const nodals_hxt1::HexVel*hv,const std::int32_t*slot,
    const Real*u0,const Real*u1,const Real*u2,
    const Real*k,const Real*omega,
    Real*mat,Real*diag,
    double radius,double nu,double sampleFraction,
    double weakGamma,double hScale,double nutScale,int useEffective,
    double kFloor,double omegaFloor,
    double*stats,unsigned long long*bad)
{
  int fi=(int)blockIdx.x;
  if(fi>=nf)return;
  auto f=F[fi];
  auto H=hv[f.cell];

  __shared__ Real U0[14],U1[14],U2[14],K[14],W[14];
  __shared__ Real PH[16][14],DN[16][14],MU[16],PEN[16],QW[16];
  __shared__ double AR[16],UT2[16],YPW[16],SLW[16],YPMAX[16],UTMAX[16],MUR[16];

  if(threadIdx.x<14){
    int g=H.g[threadIdx.x];
    U0[threadIdx.x]=u0[g];
    U1[threadIdx.x]=u1[g];
    U2[threadIdx.x]=u2[g];
    K[threadIdx.x]=k[g];
    W[threadIdx.x]=omega[g];
  }
  __syncthreads();

  if(threadIdx.x<16){
    int q=(int)threadIdx.x,i=q>>2,j=q&3;

    double cw[3]={0,0,0};
    cw[f.fd]=f.fv;
    cw[f.d0]=hxt5a::h5_hex_x[i];
    cw[f.d1]=hxt5a::h5_hex_x[j];

    double cs[3]={cw[0],cw[1],cw[2]};
    cs[f.fd]=f.fv+sampleFraction*(-2.0*f.fv);

    double pw[14],grw[14][3],gpw[14][3],Iw[9],detw;
    double ps[14],grs[14][3],gps[14][3],Is[9],dets;
    hxt2::hex_basis_ref(cw[0],cw[1],cw[2],pw,grw);
    hxt2::hex_basis_ref(cs[0],cs[1],cs[2],ps,grs);

    bool ok=
      hxt2::hex_metric(pts,hc[f.cell],cw[0],cw[1],cw[2],Iw,detw) &&
      hxt2::hex_metric(pts,hc[f.cell],cs[0],cs[1],cs[2],Is,dets);

    if(ok){
      hxt2::phys_grad14(grw,Iw,gpw);
      hxt2::phys_grad14(grs,Is,gps);
    }

    double Xw[3]={0,0,0},dXw[3][3]={{0}};
    double Xs[3]={0,0,0},dXs[3][3]={{0}};
    if(ok){
      hxt5b::q1dev(pts,hc[f.cell],cw[0],cw[1],cw[2],Xw,dXw);
      hxt5b::q1dev(pts,hc[f.cell],cs[0],cs[1],cs[2],Xs,dXs);
    }

    double jac=0.0,ydist=0.0,hn=0.0,n[3]={0,0,0};
    if(ok){
      double cr[3]={
        dXw[f.d0][1]*dXw[f.d1][2]-dXw[f.d0][2]*dXw[f.d1][1],
        dXw[f.d0][2]*dXw[f.d1][0]-dXw[f.d0][0]*dXw[f.d1][2],
        dXw[f.d0][0]*dXw[f.d1][1]-dXw[f.d0][1]*dXw[f.d1][0]
      };
      if(cr[0]*f.n[0]+cr[1]*f.n[1]+cr[2]*f.n[2]<0.0){
        cr[0]=-cr[0];cr[1]=-cr[1];cr[2]=-cr[2];
      }
      jac=sqrt(cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]);
      if(jac>0.0){
        n[0]=cr[0]/jac;n[1]=cr[1]/jac;n[2]=cr[2]/jac;
        ydist=(Xw[0]-Xs[0])*n[0]+(Xw[1]-Xs[1])*n[1]+(Xw[2]-Xs[2])*n[2];
        // For Q1 geometry at fixed tangential reference coordinates, the map
        // is linear in the wall-normal reference coordinate.  Therefore this
        // reconstructs the complete local wall-normal cell thickness and does
        // not make the penalty depend on sampleFraction.
        hn=ydist/sampleFraction;
      }
      ok=jac>0.0&&ydist>0.0&&hn>0.0&&hScale>0.0;
    }

    Real mu=(Real)nu;
    double slip=0.0,ut=0.0,yp=0.0,beta=0.0;

    if(ok){
      Real gu[9]={0},gk[3]={0},gw[3]={0},kv=0,wv=0;

      for(int a=0;a<14;++a){
        PH[q][a]=(Real)pw[a];
        DN[q][a]=(Real)(
          gpw[a][0]*n[0]+gpw[a][1]*n[1]+gpw[a][2]*n[2]);

        Real pa=(Real)ps[a];
        kv+=K[a]*pa;
        wv+=W[a]*pa;
        slip+=(double)U2[a]*ps[a];

        for(int d=0;d<3;++d){
          Real gd=(Real)gps[a][d];
          gk[d]+=K[a]*gd;
          gw[d]+=W[a]*gd;
          gu[d]+=U0[a]*gd;
          gu[3+d]+=U1[a]*gd;
          gu[6+d]+=U2[a]*gd;
        }
      }

      hxt5c::hex_sst_carrier((Real)cs[0],(Real)cs[1],(Real)cs[2],Is,K,W,(Real)kFloor,(Real)omegaFloor,kv,wv,gk,gw);

      if(useEffective){
        auto ss=hxt5c::local_sst(
          (Real)Xs[0],(Real)Xs[1],(Real)radius,(Real)nu,
          gu,kv,wv,gk,gw,(Real)kFloor,(Real)omegaFloor);
        mu=(Real)nu+(Real)nutScale*ss.nut;
      }

      if(!(mu>Real(0))||!isfinite((double)mu))ok=false;

      // Retain the old Spalding u_tau/y+ quantities as diagnostics only.
      // In weak_* modes they DO NOT contribute to the momentum matrix.
      if(ok&&!hxt5b::spalding(slip,ydist,nu,ut,yp,beta)){
        ut=0.0;yp=0.0;
        atomicAdd(stats+6,1.0);
      }
    }

    if(!ok){
      atomicAdd(bad,1ULL);
      mu=Real(0);jac=0.0;slip=ut=yp=0.0;
      for(int a=0;a<14;++a){
        PH[q][a]=Real(0);
        DN[q][a]=Real(0);
      }
    }

    double w=hxt5a::h5_hex_w[i]*hxt5a::h5_hex_w[j]*jac;
    MU[q]=mu;
    PEN[q]=ok?(Real)(weakGamma*(double)mu/(hScale*hn)):Real(0);
    QW[q]=(Real)w;

    AR[q]=w;
    UT2[q]=ut*ut*w;
    YPW[q]=yp*w;
    SLW[q]=fabs(slip)*w;
    YPMAX[q]=yp;
    UTMAX[q]=ut;
    MUR[q]=(nu>0.0)?((double)mu/nu)*w:0.0;
  }
  __syncthreads();

  for(int p=(int)threadIdx.x;p<196;p+=(int)blockDim.x){
    int a=p/14,b=p%14;
    Real z=0;
    for(int q=0;q<16;++q){
      // Homogeneous symmetric / adjoint-consistent scalar Nitsche:
      //
      //  -v mu d_n u - mu d_n v u + gamma mu/h v u
      //
      // This matches the current component-wise grad-grad momentum diffusion.
      z+=QW[q]*(
          MU[q]*(-PH[q][a]*DN[q][b]-DN[q][a]*PH[q][b])
        + PEN[q]*PH[q][a]*PH[q][b]);
    }
    atomicAdd(mat+slot[(size_t)f.cell*196+p],z);
    if(a==b)atomicAdd(diag+H.g[a],z);
  }

  if(threadIdx.x==0){
    double A=0,U2A=0,Y=0,S=0,MY=0,MUmax=0,MURsum=0;
    for(int q=0;q<16;++q){
      A+=AR[q];U2A+=UT2[q];Y+=YPW[q];S+=SLW[q];
      MY=fmax(MY,YPMAX[q]);MUmax=fmax(MUmax,UTMAX[q]);MURsum+=MUR[q];
    }
    atomicAdd(stats+0,A);
    atomicAdd(stats+1,U2A);
    atomicAdd(stats+2,Y);
    atomicAdd(stats+3,S);
    hxt5b::amax(stats+4,MY);
    hxt5b::amax(stats+5,MUmax);
    // Existing wall stats allocate an eighth slot.  In weak modes use it as
    // area-weighted mu_wall/nu diagnostic; existing wallstats() ignores it.
    atomicAdd(stats+7,MURsum);
  }
}


// ---------------------------------------------------------------------------
// Diagnostic only: physical molecular axial traction on the HEX wall.
//
// For constant axial test v_z = 1, the first molecular Nitsche consistency
// term contributes
//
//     - integral_wall nu * d_n(u_z) dA
//
// while the adjoint term containing d_n(v_z) vanishes.
//
// This kernel does NOT modify the numerical operator.
// ---------------------------------------------------------------------------
__global__ void spalding_molecular_force_audit_hex(
    const hxt5b::HexFacePlan* F,
    int nf,
    const nodals_hxt1::Point* pts,
    const nodals_hxt1::HexConn* hc,
    const nodals_hxt1::HexVel* hv,
    const Real* u2,
    double nu,
    double z0,
    double z1,
    double* stats,
    unsigned long long* bad)
{
  const int fi=(int)blockIdx.x;
  if(fi>=nf)return;

  const auto f=F[fi];
  const auto H=hv[f.cell];

  __shared__ Real U2[14];
  __shared__ double QA[16];
  __shared__ double QF[16];

  if(threadIdx.x<14){
    U2[threadIdx.x]=u2[H.g[threadIdx.x]];
  }
  __syncthreads();

  if(threadIdx.x<16){
    const int q=(int)threadIdx.x;
    const int i=q>>2;
    const int j=q&3;

    QA[q]=0.0;
    QF[q]=0.0;

    double cw[3]={0,0,0};
    cw[f.fd]=f.fv;
    cw[f.d0]=hxt5a::h5_hex_x[i];
    cw[f.d1]=hxt5a::h5_hex_x[j];

    double pw[14],grw[14][3],gpw[14][3],Iw[9],detw;

    hxt2::hex_basis_ref(
        cw[0],cw[1],cw[2],
        pw,grw);

    bool ok=hxt2::hex_metric(
        pts,hc[f.cell],
        cw[0],cw[1],cw[2],
        Iw,detw);

    if(ok){
      hxt2::phys_grad14(grw,Iw,gpw);
    }

    double Xw[3]={0,0,0};
    double dXw[3][3]={{0,0,0},{0,0,0},{0,0,0}};

    if(ok){
      hxt5b::q1dev(
          pts,hc[f.cell],
          cw[0],cw[1],cw[2],
          Xw,dXw);
    }

    double jac=0.0;
    double n[3]={0,0,0};

    if(ok){
      double cr[3]={
        dXw[f.d0][1]*dXw[f.d1][2]
          - dXw[f.d0][2]*dXw[f.d1][1],

        dXw[f.d0][2]*dXw[f.d1][0]
          - dXw[f.d0][0]*dXw[f.d1][2],

        dXw[f.d0][0]*dXw[f.d1][1]
          - dXw[f.d0][1]*dXw[f.d1][0]
      };

      if(cr[0]*f.n[0]+cr[1]*f.n[1]+cr[2]*f.n[2]<0.0){
        cr[0]=-cr[0];
        cr[1]=-cr[1];
        cr[2]=-cr[2];
      }

      jac=sqrt(
          cr[0]*cr[0]
        + cr[1]*cr[1]
        + cr[2]*cr[2]);

      if(jac>0.0){
        n[0]=cr[0]/jac;
        n[1]=cr[1]/jac;
        n[2]=cr[2]/jac;
      }else{
        ok=false;
      }
    }

    if(!ok){
      atomicAdd(bad,1ULL);
    }else if(Xw[2]>=z0-1e-12 && Xw[2]<=z1+1e-12){

      double dnuz=0.0;

      for(int a=0;a<14;++a){
        const double dna=
            gpw[a][0]*n[0]
          + gpw[a][1]*n[1]
          + gpw[a][2]*n[2];

        dnuz+=(double)U2[a]*dna;
      }

      const double w=
          hxt5a::h5_hex_w[i]
        * hxt5a::h5_hex_w[j]
        * jac;

      QA[q]=w;

      // Positive value means additional resistance to +z flow.
      QF[q]=-nu*dnuz*w;
    }
  }

  __syncthreads();

  if(threadIdx.x==0){
    double a=0.0;
    double fmol=0.0;

    for(int q=0;q<16;++q){
      a+=QA[q];
      fmol+=QF[q];
    }

    atomicAdd(stats+0,a);
    atomicAdd(stats+1,fmol);
  }
}

// ---------- SST turbulent diffusion for momentum ----------
//
// The existing implicit matrix represents the scalar part
//     int nu_eff grad(v_i).grad(U_i) dV.
// For a Boussinesq/deviatoric stress the complete weak volume term is
//     int grad(v_i)_j nu_eff [d_j U_i + d_i U_j - 2/3 div(U) delta_ij] dV.
// We retain the first term implicitly (so the segregated scalar momentum
// matrices and SIMPLE preconditioner are unchanged) and lag the remaining
// transpose/deviatoric correction on the RHS.  At the nonlinear fixed point
// this is the full deviatoric volume stress without introducing a 3x3 block
// velocity solve.
//
// This first implementation is exact for the all-HEX y+ regression path.
// Hybrid HEX/TET interface consistency terms remain the existing scalar form;
// assemble_momentum_sst therefore rejects deviatoric mode on hybrid meshes
// rather than silently claiming full equivalence.
__global__ void sst_mom_dev_rhs_hex(
    const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,const nodals_hxt1::HexVel*hv,
    const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,
    Real*r0,Real*r1,Real*r2,std::uint64_t n,double radius,double nu,double kFloor,double omegaFloor,
    unsigned long long*bad)
{
  std::uint64_t c=(std::uint64_t)blockIdx.x;if(c>=n)return;
  __shared__ Real U0[14],U1[14],U2[14],K[14],W[14];
  __shared__ Real GP[64][14][3],GU[64][9],NUEW[64];
  if(threadIdx.x<14){int g=hv[c].g[threadIdx.x];U0[threadIdx.x]=u0[g];U1[threadIdx.x]=u1[g];U2[threadIdx.x]=u2[g];K[threadIdx.x]=k[g];W[threadIdx.x]=omega[g];}
  __syncthreads();
  const int iq=(int)threadIdx.x;
  if(iq<64){
    int ix=iq>>4,iy=(iq>>2)&3,iz=iq&3;
    double xr=hxt5a::h5_hex_x[ix],yr=hxt5a::h5_hex_x[iy],zr=hxt5a::h5_hex_x[iz];
    double qw=hxt5a::h5_hex_w[ix]*hxt5a::h5_hex_w[iy]*hxt5a::h5_hex_w[iz];
    double ph[14],gr[14][3],gp[14][3],I[9],det;
    hxt2::hex_basis_ref(xr,yr,zr,ph,gr);
    if(!hxt2::hex_metric(pts,hc[c],xr,yr,zr,I,det)){
      atomicAdd(bad,1ULL);NUEW[iq]=Real(0);
      for(int m=0;m<9;++m)GU[iq][m]=Real(0);
      for(int a=0;a<14;++a)for(int d=0;d<3;++d)GP[iq][a][d]=Real(0);
    }else{
      hxt2::phys_grad14(gr,I,gp);
      Real gu[9]={0},gk[3]={0},gw[3]={0},kv=0,wv=0;
      for(int a=0;a<14;++a){
        for(int d=0;d<3;++d){
          Real gd=(Real)gp[a][d];GP[iq][a][d]=gd;
          gu[d]+=U0[a]*gd;gu[3+d]+=U1[a]*gd;gu[6+d]+=U2[a]*gd;
        }
      }
      hxt5c::hex_sst_carrier((Real)xr,(Real)yr,(Real)zr,I,K,W,(Real)kFloor,(Real)omegaFloor,kv,wv,gk,gw);
      double x=0,y=0;for(int a=0;a<8;++a){double N=.125*(1+hxt4a::h4_hex_sign[a][0]*xr)*(1+hxt4a::h4_hex_sign[a][1]*yr)*(1+hxt4a::h4_hex_sign[a][2]*zr);auto X=pts[hc[c].v[a]];x+=N*X.x;y+=N*X.y;}
      auto s=hxt5c::local_sst((Real)x,(Real)y,(Real)radius,(Real)nu,gu,kv,wv,gk,gw,(Real)kFloor,(Real)omegaFloor);
      for(int m=0;m<9;++m)GU[iq][m]=gu[m];
      NUEW[iq]=((Real)nu+s.nut)*(Real)(det*qw);
    }
  }
  __syncthreads();
  for(int a=(int)threadIdx.x;a<14;a+=(int)blockDim.x){
    Real z[3]={0,0,0};
    for(int q=0;q<64;++q){
      const Real dx=GP[q][a][0],dy=GP[q][a][1],dz=GP[q][a][2];
      const Real dphi[3]={dx,dy,dz};
      const Real divU=GU[q][0]+GU[q][4]+GU[q][8];
      #pragma unroll
      for(int i=0;i<3;++i){
        Real cross=Real(0);
        #pragma unroll
        for(int j=0;j<3;++j)cross+=dphi[j]*GU[q][3*j+i];
        cross-=Real(2.0/3.0)*dphi[i]*divU;
        z[i]-=NUEW[q]*cross; // move lagged correction from LHS to RHS
      }
    }
    const int g=hv[c].g[a];atomicAdd(r0+g,z[0]);atomicAdd(r1+g,z[1]);atomicAdd(r2+g,z[2]);
  }
}

__global__ void sst_mom_dev_rhs_tet(
    const nodals_hxt1::Point*pts,const nodals_hxt1::TetConn*tc,const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,
    const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,
    Real*r0,Real*r1,Real*r2,std::uint64_t n,double radius,double nu,double kFloor,double omegaFloor)
{
  std::uint64_t c=(std::uint64_t)blockIdx.x;if(c>=n)return;
  __shared__ Real U0[8],U1[8],U2[8],K[8],W[8],GP[24][8][3],GU[24][9],NUEW[24];
  if(threadIdx.x<8){int g=tv[c].g[threadIdx.x];U0[threadIdx.x]=u0[g];U1[threadIdx.x]=u1[g];U2[threadIdx.x]=u2[g];K[threadIdx.x]=k[g];W[threadIdx.x]=omega[g];}__syncthreads();
  int iq=(int)threadIdx.x;
  if(iq<24){
    double l[4]={hxt5a::h5_tet24_l0[iq],hxt5a::h5_tet24_l1[iq],hxt5a::h5_tet24_l2[iq],hxt5a::h5_tet24_l3[iq]};double qw=hxt5a::h5_tet24_w[iq];
    double ph[8],gr[8][3],gp[8][3];hxt2::tet_basis(l[0],l[1],l[2],l[3],ph,gr);hxt2::tet_phys_grad(gr,tg[c],gp);
    Real gu[9]={0},gk[3]={0},gw[3]={0},kv=0,wv=0;
    for(int a=0;a<8;++a){Real pa=(Real)ph[a];kv+=K[a]*pa;wv+=W[a]*pa;for(int d=0;d<3;++d){Real gd=(Real)gp[a][d];GP[iq][a][d]=gd;gk[d]+=K[a]*gd;gw[d]+=W[a]*gd;gu[d]+=U0[a]*gd;gu[3+d]+=U1[a]*gd;gu[6+d]+=U2[a]*gd;}}
    hxt5c::eval_dg_omega_p1((int)c,l,tg[c],wv,gw);double x=0,y=0;for(int a=0;a<4;++a){auto X=pts[tc[c].v[a]];x+=l[a]*X.x;y+=l[a]*X.y;}
    auto s=hxt5c::local_sst((Real)x,(Real)y,(Real)radius,(Real)nu,gu,kv,wv,gk,gw,(Real)kFloor,(Real)omegaFloor);
    for(int m=0;m<9;++m)GU[iq][m]=gu[m];NUEW[iq]=((Real)nu+s.nut)*(Real)(tg[c].det*qw);
  }
  __syncthreads();
  for(int a=(int)threadIdx.x;a<8;a+=(int)blockDim.x){
    Real z[3]={0,0,0};
    for(int q=0;q<24;++q){
      const Real dphi[3]={GP[q][a][0],GP[q][a][1],GP[q][a][2]};
      const Real divU=GU[q][0]+GU[q][4]+GU[q][8];
      #pragma unroll
      for(int i=0;i<3;++i){
        Real cross=Real(0);
        #pragma unroll
        for(int j=0;j<3;++j)cross+=dphi[j]*GU[q][3*j+i];
        cross-=Real(2.0/3.0)*dphi[i]*divU;z[i]-=NUEW[q]*cross;}
    }
    const int g=tv[c].g[a];atomicAdd(r0+g,z[0]);atomicAdd(r1+g,z[1]);atomicAdd(r2+g,z[2]);
  }
}

__global__ void sst_mom_hex(
    const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,const nodals_hxt1::HexVel*hv,
    const std::int32_t*slot,const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,
    Real*turb,Real*diagRTurb,std::uint64_t n,double radius,double nu,double kFloor,double omegaFloor,
    unsigned long long*bad)
{
  std::uint64_t c=(std::uint64_t)blockIdx.x;if(c>=n)return;
  __shared__ Real U0[14],U1[14],U2[14],K[14],W[14];
  __shared__ Real GP[64][14][3],NW[64];
  if(threadIdx.x<14){int g=hv[c].g[threadIdx.x];U0[threadIdx.x]=u0[g];U1[threadIdx.x]=u1[g];U2[threadIdx.x]=u2[g];K[threadIdx.x]=k[g];W[threadIdx.x]=omega[g];}
  __syncthreads();
  int iq=(int)threadIdx.x;
  if(iq<64){
    int ix=iq>>4,iy=(iq>>2)&3,iz=iq&3;double xr=hxt5a::h5_hex_x[ix],yr=hxt5a::h5_hex_x[iy],zr=hxt5a::h5_hex_x[iz];double qw=hxt5a::h5_hex_w[ix]*hxt5a::h5_hex_w[iy]*hxt5a::h5_hex_w[iz];
    double ph[14],gr[14][3],gp[14][3],I[9],det;hxt2::hex_basis_ref(xr,yr,zr,ph,gr);
    if(!hxt2::hex_metric(pts,hc[c],xr,yr,zr,I,det)){atomicAdd(bad,1ULL);NW[iq]=Real(0);for(int a=0;a<14;++a)for(int d=0;d<3;++d)GP[iq][a][d]=Real(0);}else{
      hxt2::phys_grad14(gr,I,gp);Real gu[9]={0},gk[3]={0},gw[3]={0},kv=0,wv=0;
      for(int a=0;a<14;++a){Real pa=(Real)ph[a];for(int d=0;d<3;++d){Real gd=(Real)gp[a][d];GP[iq][a][d]=gd;gu[d]+=U0[a]*gd;gu[3+d]+=U1[a]*gd;gu[6+d]+=U2[a]*gd;}}
      hxt5c::hex_sst_carrier((Real)xr,(Real)yr,(Real)zr,I,K,W,(Real)kFloor,(Real)omegaFloor,kv,wv,gk,gw);
      double x=0,y=0;for(int a=0;a<8;++a){double N=.125*(1+hxt4a::h4_hex_sign[a][0]*xr)*(1+hxt4a::h4_hex_sign[a][1]*yr)*(1+hxt4a::h4_hex_sign[a][2]*zr);auto X=pts[hc[c].v[a]];x+=N*X.x;y+=N*X.y;}
      auto s=hxt5c::local_sst((Real)x,(Real)y,(Real)radius,(Real)nu,gu,kv,wv,gk,gw,(Real)kFloor,(Real)omegaFloor);NW[iq]=s.nut*(Real)(det*qw);
    }
  }
  __syncthreads();
  for(int p=(int)threadIdx.x;p<196;p+=(int)blockDim.x){int a=p/14,b=p%14;Real v=0;for(int q=0;q<64;++q){Real gd=GP[q][a][0]*GP[q][b][0]+GP[q][a][1]*GP[q][b][1]+GP[q][a][2]*GP[q][b][2];v+=NW[q]*gd;}atomicAdd(turb+slot[c*196ull+(std::size_t)p],v);if(a==b)atomicAdd(diagRTurb+hv[c].g[a],v);}
}

__global__ void sst_mom_tet(
    const nodals_hxt1::Point*pts,const nodals_hxt1::TetConn*tc,const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,
    const std::int32_t*slot,const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,
    Real*turb,Real*diagRTurb,std::uint64_t n,double radius,double nu,double kFloor,double omegaFloor)
{
  std::uint64_t c=(std::uint64_t)blockIdx.x;if(c>=n)return;
  // PERF_V4_TET24_MOMENTUM_VOLUME
  __shared__ Real U0[8],U1[8],U2[8],K[8],W[8],GP[24][8][3],NW[24];
  if(threadIdx.x<8){int g=tv[c].g[threadIdx.x];U0[threadIdx.x]=u0[g];U1[threadIdx.x]=u1[g];U2[threadIdx.x]=u2[g];K[threadIdx.x]=k[g];W[threadIdx.x]=omega[g];}__syncthreads();
  int iq=(int)threadIdx.x;if(iq<24){double l[4]={hxt5a::h5_tet24_l0[iq],hxt5a::h5_tet24_l1[iq],hxt5a::h5_tet24_l2[iq],hxt5a::h5_tet24_l3[iq]};double qw=hxt5a::h5_tet24_w[iq];double ph[8],gr[8][3],gp[8][3];hxt2::tet_basis(l[0],l[1],l[2],l[3],ph,gr);hxt2::tet_phys_grad(gr,tg[c],gp);Real gu[9]={0},gk[3]={0},gw[3]={0},kv=0,wv=0;for(int a=0;a<8;++a){Real pa=(Real)ph[a];kv+=K[a]*pa;wv+=W[a]*pa;for(int d=0;d<3;++d){Real gd=(Real)gp[a][d];GP[iq][a][d]=gd;gk[d]+=K[a]*gd;gw[d]+=W[a]*gd;gu[d]+=U0[a]*gd;gu[3+d]+=U1[a]*gd;gu[6+d]+=U2[a]*gd;}}hxt5c::eval_dg_omega_p1((int)c,l,tg[c],wv,gw);double x=0,y=0;for(int a=0;a<4;++a){auto X=pts[tc[c].v[a]];x+=l[a]*X.x;y+=l[a]*X.y;}auto ss=hxt5c::local_sst((Real)x,(Real)y,(Real)radius,(Real)nu,gu,kv,wv,gk,gw,(Real)kFloor,(Real)omegaFloor);NW[iq]=ss.nut*(Real)(tg[c].det*qw);}__syncthreads();
  for(int p=(int)threadIdx.x;p<64;p+=(int)blockDim.x){int a=p/8,b=p%8;Real v=0;for(int q=0;q<24;++q){Real gd=GP[q][a][0]*GP[q][b][0]+GP[q][a][1]*GP[q][b][1]+GP[q][a][2]*GP[q][b][2];v+=NW[q]*gd;}atomicAdd(turb+slot[c*64ull+(std::size_t)p],v);if(a==b)atomicAdd(diagRTurb+tv[c].g[a],v);}
}


#if defined(NODALS_P2_COMPILE_SMOKE)
#include "modules/coupling/bernstein_p2_momentum_nut.inc"
#endif
// ============================================================
// STAGE33A_MOM_SUPG: optional GPU warp-per-tet operator
#include "modules/stabilization/stage33a_momentum_supg.cuh"
#include "modules/stabilization/momentum_full_residual_supg.cuh" // NODALS_UNIFIED_FULL_MOMENTUM_SUPG_V1
// HXT5D_SST_MOM_INTERFACE_CACHED_V1
//
// Same turbulent hybrid momentum-interface contribution as
// sst_mom_interface.
//
// Geometry/topology/CSR positions come from the persistent
// hxt5c::SSTInterfaceCache.
// ============================================================
__global__ void sst_mom_interface_cached(
    const nodals_hxt1::HexVel*hv,
    const nodals_hxt1::TetVel*tv,
    const hxt2::InterfacePlan*P,
    const hxt5c::SSTInterfaceQuadCache*Q,
    const hxt5c::SSTInterfaceSideCache*SC,
    const Real*u0,const Real*u1,const Real*u2,
    const Real*k,const Real*omega,
    Real*turb,
    Real*diagRTurb,
    std::uint64_t n,
    double radius,double nu,double gamma,
    double kFloor,double omegaFloor)
{
  const std::uint64_t iside=
    (std::uint64_t)blockIdx.x;

  if(iside>=2*n)
    return;

  const int ir=(int)(iside>>1);
  const int side=(int)(iside&1);

  const auto&p=P[ir];
  const auto&HG=hv[p.hexCell];

  const int tcell=p.tetCell[side];
  const auto&TG=tv[tcell];

  const auto&S=SC[iside];
  const int un=S.un;

  __shared__ Real HU0[14],HU1[14],HU2[14];
  __shared__ Real HK[14],HW[14];

  __shared__ Real TU0[8],TU1[8],TU2[8];
  __shared__ Real TK[8],TW[8];

  __shared__ Real NH[12],NT[12];

  if(threadIdx.x<14){
    const int g=HG.g[threadIdx.x];

    HU0[threadIdx.x]=u0[g];
    HU1[threadIdx.x]=u1[g];
    HU2[threadIdx.x]=u2[g];

    HK[threadIdx.x]=k[g];
    HW[threadIdx.x]=omega[g];
  }

  if(threadIdx.x<8){
    const int g=TG.g[threadIdx.x];

    TU0[threadIdx.x]=u0[g];
    TU1[threadIdx.x]=u1[g];
    TU2[threadIdx.x]=u2[g];

    TK[threadIdx.x]=k[g];
    TW[threadIdx.x]=omega[g];
  }

  __syncthreads();

  const int q=(int)threadIdx.x;

  if(q<12){

    const auto&C=
      Q[iside*12ull+(std::size_t)q];

    Real guh[9]={0,0,0,0,0,0,0,0,0};
    Real gut[9]={0,0,0,0,0,0,0,0,0};

    Real gkh[3]={0,0,0};
    Real gwh[3]={0,0,0};

    Real gkt[3]={0,0,0};
    Real gwt[3]={0,0,0};

    Real kh=0,wh=0;
    Real kt=0,wt=0;

    for(int a=0;a<14;++a){

      const Real pa=C.ph[a];

      kh+=HK[a]*pa;
      wh+=HW[a]*pa;

      for(int d=0;d<3;++d){

        const Real gd=C.gph[a][d];

        gkh[d]+=HK[a]*gd;
        gwh[d]+=HW[a]*gd;

        guh[d]+=HU0[a]*gd;
        guh[3+d]+=HU1[a]*gd;
        guh[6+d]+=HU2[a]*gd;
      }
    }

    for(int a=0;a<8;++a){

      const Real pa=C.pt[a];

      kt+=TK[a]*pa;
      wt+=TW[a]*pa;

      for(int d=0;d<3;++d){

        const Real gd=C.gpt[a][d];

        gkt[d]+=TK[a]*gd;
        gwt[d]+=TW[a]*gd;

        gut[d]+=TU0[a]*gd;
        gut[3+d]+=TU1[a]*gd;
        gut[6+d]+=TU2[a]*gd;
      }
    }

    NH[q]=
      hxt5c::local_sst(
        C.xh,C.yh,
        (Real)radius,(Real)nu,
        guh,
        kh,wh,
        gkh,gwh,
        (Real)kFloor,
        (Real)omegaFloor).nut;

    NT[q]=
      hxt5c::local_sst(
        C.xt,C.yt,
        (Real)radius,(Real)nu,
        gut,
        kt,wt,
        gkt,gwt,
        (Real)kFloor,
        (Real)omegaFloor).nut;
  }

  __syncthreads();

  const Real tauBase=
    (Real)(
      gamma*nu*
      fmax(
        p.invHHex,
        p.invHTet[side]));

  const int npair=un*un;

  for(int pair=(int)threadIdx.x;
      pair<npair;
      pair+=(int)blockDim.x)
  {
    const int ii=pair/un;
    const int jj=pair%un;

    const int gi=S.ug[ii];

    Real kij=0;
    Real kiiCons=0;

    for(int q2=0;q2<12;++q2){

      const auto&C=
        Q[iside*12ull+(std::size_t)q2];

      const Real ji=C.jump[ii];
      const Real jjv=C.jump[jj];

      const Real di=
        Real(.5)*
        (NH[q2]*C.hn[ii]
        +NT[q2]*C.tn[ii]);

      const Real dj=
        Real(.5)*
        (NH[q2]*C.hn[jj]
        +NT[q2]*C.tn[jj]);

      const Real cons=
        -(dj*ji+di*jjv);

      const Real tauEff=
        (Real)gamma*
        fmax(
          ((Real)nu+NH[q2])*
            (Real)p.invHHex,
          ((Real)nu+NT[q2])*
            (Real)p.invHTet[side]);

      const Real deltaTau=
        fmax(
          Real(0),
          tauEff-tauBase);

      kij+=
        C.wq*
        (cons+
         deltaTau*ji*jjv);

      if(ii==jj)
        kiiCons+=
          C.wq*cons;
    }

    const int sl=
      S.slot[ii*22+jj];

    atomicAdd(
      turb+sl,
      kij);

    if(ii==jj)
      atomicAdd(
        diagRTurb+gi,
        kiiCons);
  }
}


__global__ void sst_mom_interface(
    const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,const nodals_hxt1::TetConn*tc,
    const nodals_hxt1::HexVel*hv,const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,const hxt2::InterfacePlan*P,
    const std::int64_t*row,const std::int32_t*col,const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,
    Real*turb,Real*diagRTurb,std::uint64_t n,double radius,double nu,double gamma,double kFloor,double omegaFloor,unsigned long long*bad)
{
  std::uint64_t iside=(std::uint64_t)blockIdx.x;if(iside>=2*n)return;int ir=(int)(iside>>1),side=(int)(iside&1);const auto&p=P[ir];const auto&HG=hv[p.hexCell];int tcell=p.tetCell[side],tf=p.tetFace[side];const auto&TG=tv[tcell];
  __shared__ int ug[22],un;__shared__ Real HU0[14],HU1[14],HU2[14],HK[14],HW[14],TU0[8],TU1[8],TU2[8],TK[8],TW[8];
  // PERF_V3_Q12_MOMENTUM_INTERFACE: positive degree-6 triangle rule
  __shared__ Real PH[12][14],GPH[12][14][3],PT[12][8],GPT[12][8][3],NH[12],NT[12],QW[12];
  if(threadIdx.x==0){int m=0;for(int a=0;a<14;++a)ug[m++]=HG.g[a];for(int b=0;b<8;++b){int g=TG.g[b],found=0;for(int q=0;q<m;++q)if(ug[q]==g)found=1;if(!found)ug[m++]=g;}un=m;}
  if(threadIdx.x<14){int g=HG.g[threadIdx.x];HU0[threadIdx.x]=u0[g];HU1[threadIdx.x]=u1[g];HU2[threadIdx.x]=u2[g];HK[threadIdx.x]=k[g];HW[threadIdx.x]=omega[g];}
  if(threadIdx.x<8){int g=TG.g[threadIdx.x];TU0[threadIdx.x]=u0[g];TU1[threadIdx.x]=u1[g];TU2[threadIdx.x]=u2[g];TK[threadIdx.x]=k[g];TW[threadIdx.x]=omega[g];}__syncthreads();
  for(int q=(int)threadIdx.x;q<12;q+=(int)blockDim.x){double L[3]={hxt5b::QL0[q],hxt5b::QL1[q],hxt5b::QL2[q]};QW[q]=(Real)(hxt5b::QW12[q]*p.areaTri[side]);double xr=0,yr=0,zr=0;for(int j=0;j<3;++j){int hl=p.hexTriLocal[side][j];xr+=L[j]*hxt4a::h4_hex_sign[hl][0];yr+=L[j]*hxt4a::h4_hex_sign[hl][1];zr+=L[j]*hxt4a::h4_hex_sign[hl][2];}double ph[14],grh[14][3],gph[14][3],I[9],det;hxt2::hex_basis_ref(xr,yr,zr,ph,grh);if(!hxt2::hex_metric(pts,hc[p.hexCell],xr,yr,zr,I,det)){atomicAdd(bad,1ULL);NH[q]=NT[q]=Real(0);continue;}hxt2::phys_grad14(grh,I,gph);double lam[4]={0,0,0,0};for(int j=0;j<3;++j)lam[hxt4a::h4_tet_face[tf][j]]=L[j];double pt[8],grt[8][3],gpt[8][3];hxt2::tet_basis(lam[0],lam[1],lam[2],lam[3],pt,grt);hxt2::tet_phys_grad(grt,tg[tcell],gpt);
    Real guh[9]={0},gut[9]={0},gkh[3]={0},gwh[3]={0},gkt[3]={0},gwt[3]={0},kh=0,wh=0,kt=0,wt=0;
    for(int a=0;a<14;++a){PH[q][a]=(Real)ph[a];kh+=HK[a]*(Real)ph[a];wh+=HW[a]*(Real)ph[a];for(int d=0;d<3;++d){Real gd=(Real)gph[a][d];GPH[q][a][d]=gd;gkh[d]+=HK[a]*gd;gwh[d]+=HW[a]*gd;guh[d]+=HU0[a]*gd;guh[3+d]+=HU1[a]*gd;guh[6+d]+=HU2[a]*gd;}}
    for(int a=0;a<8;++a){PT[q][a]=(Real)pt[a];kt+=TK[a]*(Real)pt[a];wt+=TW[a]*(Real)pt[a];for(int d=0;d<3;++d){Real gd=(Real)gpt[a][d];GPT[q][a][d]=gd;gkt[d]+=TK[a]*gd;gwt[d]+=TW[a]*gd;gut[d]+=TU0[a]*gd;gut[3+d]+=TU1[a]*gd;gut[6+d]+=TU2[a]*gd;}}
    double xh=0,yh=0;for(int a=0;a<8;++a){double N=.125*(1+hxt4a::h4_hex_sign[a][0]*xr)*(1+hxt4a::h4_hex_sign[a][1]*yr)*(1+hxt4a::h4_hex_sign[a][2]*zr);auto X=pts[hc[p.hexCell].v[a]];xh+=N*X.x;yh+=N*X.y;}double xt=0,yt=0;for(int a=0;a<4;++a){auto X=pts[tc[tcell].v[a]];xt+=lam[a]*X.x;yt+=lam[a]*X.y;}
    NH[q]=hxt5c::local_sst((Real)xh,(Real)yh,(Real)radius,(Real)nu,guh,kh,wh,gkh,gwh,(Real)kFloor,(Real)omegaFloor).nut;NT[q]=hxt5c::local_sst((Real)xt,(Real)yt,(Real)radius,(Real)nu,gut,kt,wt,gkt,gwt,(Real)kFloor,(Real)omegaFloor).nut;
  }
  __syncthreads();Real nx=(Real)p.normal[0],ny=(Real)p.normal[1],nz=(Real)p.normal[2];Real tauBase=(Real)(gamma*nu*fmax(p.invHHex,p.invHTet[side]));int npair=un*un;
  for(int pair=(int)threadIdx.x;pair<npair;pair+=(int)blockDim.x){int ii=pair/un,jj=pair%un,gi=ug[ii],gj=ug[jj];Real kij=0,kiiCons=0;for(int q=0;q<12;++q){Real ji=0,jjv=0,di=0,dj=0;for(int a=0;a<14;++a){if(HG.g[a]==gi){ji+=PH[q][a];di+=Real(.5)*NH[q]*(GPH[q][a][0]*nx+GPH[q][a][1]*ny+GPH[q][a][2]*nz);}if(HG.g[a]==gj){jjv+=PH[q][a];dj+=Real(.5)*NH[q]*(GPH[q][a][0]*nx+GPH[q][a][1]*ny+GPH[q][a][2]*nz);}}for(int a=0;a<8;++a){if(TG.g[a]==gi){ji-=PT[q][a];di+=Real(.5)*NT[q]*(GPT[q][a][0]*nx+GPT[q][a][1]*ny+GPT[q][a][2]*nz);}if(TG.g[a]==gj){jjv-=PT[q][a];dj+=Real(.5)*NT[q]*(GPT[q][a][0]*nx+GPT[q][a][1]*ny+GPT[q][a][2]*nz);}}Real cons=-(dj*ji+di*jjv);Real tauEff=(Real)gamma*fmax(((Real)nu+NH[q])*(Real)p.invHHex,((Real)nu+NT[q])*(Real)p.invHTet[side]);Real deltaTau=fmax(Real(0),tauEff-tauBase);kij+=QW[q]*(cons+deltaTau*ji*jjv);if(ii==jj)kiiCons+=QW[q]*cons;}int sl=hxt4b::csr_find(row,col,gi,gj);atomicAdd(turb+sl,kij);if(ii==jj)atomicAdd(diagRTurb+gi,kiiCons);}
}

#include "modules/coupling/g2_momentum_operator.inc"
// ---------- SST-aware DG velocity inlet; Spalding wall stays unchanged ----------
__global__ void sst_dg_hex(const hxt5b::HexFacePlan*F,int nf,const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,const nodals_hxt1::HexVel*hv,const std::int32_t*slot,const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,Real*mat,Real*diag,Real*rhsx,Real*rhsy,Real*rhsz,double inletX,double inletY,double inletZ,double radius,double nu,double kFloor,double omegaFloor,unsigned long long*bad){
  int fi=blockIdx.x;if(fi>=nf)return;auto f=F[fi];auto H=hv[f.cell];__shared__ Real U0[14],U1[14],U2[14],K[14],W[14],ph[16][14],gp[16][14][3],nue[16],wq[16];if(threadIdx.x<14){int g=H.g[threadIdx.x];U0[threadIdx.x]=u0[g];U1[threadIdx.x]=u1[g];U2[threadIdx.x]=u2[g];K[threadIdx.x]=k[g];W[threadIdx.x]=omega[g];}__syncthreads();
  if(threadIdx.x<16){int q=threadIdx.x,i=q>>2,j=q&3;double c[3]={0,0,0};c[f.fd]=f.fv;c[f.d0]=hxt5a::h5_hex_x[i];c[f.d1]=hxt5a::h5_hex_x[j];double pp[14],gr[14][3],gg[14][3],I[9],det;hxt2::hex_basis_ref(c[0],c[1],c[2],pp,gr);bool ok=hxt2::hex_metric(pts,hc[f.cell],c[0],c[1],c[2],I,det);double X[3]={0},dX[3][3]={{0}};if(ok){hxt2::phys_grad14(gr,I,gg);hxt5b::q1dev(pts,hc[f.cell],c[0],c[1],c[2],X,dX);}double jac=0;if(ok){double cr[3]={dX[f.d0][1]*dX[f.d1][2]-dX[f.d0][2]*dX[f.d1][1],dX[f.d0][2]*dX[f.d1][0]-dX[f.d0][0]*dX[f.d1][2],dX[f.d0][0]*dX[f.d1][1]-dX[f.d0][1]*dX[f.d1][0]};if(cr[0]*f.n[0]+cr[1]*f.n[1]+cr[2]*f.n[2]<0){cr[0]*=-1;cr[1]*=-1;cr[2]*=-1;}jac=sqrt(cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]);ok=jac>0;}Real gu[9]={0},gk[3]={0},gw[3]={0},kv=0,wv=0;for(int a=0;a<14;++a){ph[q][a]=ok?(Real)pp[a]:Real(0);for(int d=0;d<3;++d){Real gd=ok?(Real)gg[a][d]:Real(0);gp[q][a][d]=gd;gu[d]+=U0[a]*gd;gu[3+d]+=U1[a]*gd;gu[6+d]+=U2[a]*gd;}}if(ok)hxt5c::hex_sst_carrier((Real)c[0],(Real)c[1],(Real)c[2],I,K,W,(Real)kFloor,(Real)omegaFloor,kv,wv,gk,gw);if(!ok){atomicAdd(bad,1ULL);nue[q]=wq[q]=Real(0);}else{auto ss=hxt5c::local_sst((Real)X[0],(Real)X[1],(Real)radius,(Real)nu,gu,kv,wv,gk,gw,(Real)kFloor,(Real)omegaFloor);nue[q]=(Real)nu+ss.nut;wq[q]=(Real)(hxt5a::h5_hex_w[i]*hxt5a::h5_hex_w[j]*jac);}}__syncthreads();Real inflow=fmax(Real(0),Real(-(inletX*f.n[0]+inletY*f.n[1]+inletZ*f.n[2])));
  for(int p=threadIdx.x;p<196;p+=blockDim.x){int a=p/14,b=p%14;Real z=0;for(int q=0;q<16;++q){Real dna=gp[q][a][0]*f.n[0]+gp[q][a][1]*f.n[1]+gp[q][a][2]*f.n[2],dnb=gp[q][b][0]*f.n[0]+gp[q][b][1]*f.n[1]+gp[q][b][2]*f.n[2];z+=wq[q]*(inflow*ph[q][a]*ph[q][b]+nue[q]*(-ph[q][a]*dnb+dna*ph[q][b]));}atomicAdd(mat+slot[(size_t)f.cell*196+p],z);if(a==b)atomicAdd(diag+H.g[a],z);}for(int a=threadIdx.x;a<14;a+=blockDim.x){Real z=0;for(int q=0;q<16;++q){Real dna=gp[q][a][0]*f.n[0]+gp[q][a][1]*f.n[1]+gp[q][a][2]*f.n[2];z+=wq[q]*((inflow*ph[q][a]+nue[q]*dna));}atomicAdd(rhsx+H.g[a],z*(Real)inletX);atomicAdd(rhsy+H.g[a],z*(Real)inletY);atomicAdd(rhsz+H.g[a],z*(Real)inletZ);}
}

__global__ void sst_dg_tet(const hxt5b::TetFacePlan*F,int nf,const nodals_hxt1::Point*pts,const nodals_hxt1::TetConn*tc,const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,const std::int32_t*slot,const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,Real*mat,Real*diag,Real*rhsx,Real*rhsy,Real*rhsz,double inletX,double inletY,double inletZ,double radius,double nu,double kFloor,double omegaFloor
#if defined(NODALS_P2_STAGE4_COUPLED)
  ,const nodals::p2::Tet10Device *p2cells,const Real *kP2,const Real *wP2
#endif
  ){
  int fi=blockIdx.x;if(fi>=nf)return;auto f=F[fi];auto T=tv[f.cell];__shared__ Real U0[8],U1[8],U2[8],K[8],W[8],ph[12][8],gp[12][8][3],nue[12],wq[12],ivx[12],ivy[12],ivz[12],qin[12];if(threadIdx.x<8){int g=T.g[threadIdx.x];U0[threadIdx.x]=u0[g];U1[threadIdx.x]=u1[g];U2[threadIdx.x]=u2[g];K[threadIdx.x]=k[g];W[threadIdx.x]=omega[g];}__syncthreads();if(threadIdx.x<12){int q=threadIdx.x,a0=hxt4a::h4_tet_face[f.lf][0],a1=hxt4a::h4_tet_face[f.lf][1],a2=hxt4a::h4_tet_face[f.lf][2];double l[4]={0,0,0,0};l[a0]=hxt5b::QL0[q];l[a1]=hxt5b::QL1[q];l[a2]=hxt5b::QL2[q];double pp[8],gr[8][3],gg[8][3];hxt2::tet_basis(l[0],l[1],l[2],l[3],pp,gr);hxt2::tet_phys_grad(gr,tg[f.cell],gg);Real gu[9]={0},gk[3]={0},gw[3]={0},kv=0,wv=0;double x=0,y=0;for(int a=0;a<8;++a){ph[q][a]=(Real)pp[a];
#if defined(NODALS_P2_STAGE4_COUPLED)
  if(!p2cells){
#endif
  kv+=K[a]*(Real)pp[a];wv+=W[a]*(Real)pp[a];
#if defined(NODALS_P2_STAGE4_COUPLED)
  }
#endif
for(int d=0;d<3;++d){Real gd=(Real)gg[a][d];gp[q][a][d]=gd;
#if defined(NODALS_P2_STAGE4_COUPLED)
  if(!p2cells){
#endif
  gk[d]+=K[a]*gd;gw[d]+=W[a]*gd;
#if defined(NODALS_P2_STAGE4_COUPLED)
  }
#endif
  gu[d]+=U0[a]*gd;gu[3+d]+=U1[a]*gd;gu[6+d]+=U2[a]*gd;}}
#if defined(NODALS_P2_STAGE4_COUPLED)
  if(p2cells){
    double bp2[10],br2[10][3],bg2[10][3];
    nodals::p2::basis(l,bp2,br2);
    nodals::p2::physical_gradient(br2,tg[f.cell].invJ,bg2);
    for(int a=0;a<10;++a){
      const int ia=p2cells[f.cell].g[a];
      const Real kc=kP2[ia],wc=wP2[ia];
      kv+=kc*(Real)bp2[a];wv+=wc*(Real)bp2[a];
      for(int d=0;d<3;++d){gk[d]+=kc*(Real)bg2[a][d];gw[d]+=wc*(Real)bg2[a][d];}
    }
  }
#endif
  for(int a=0;a<4;++a){auto P=pts[tc[f.cell].v[a]];x+=l[a]*P.x;y+=l[a]*P.y;}auto ss=hxt5c::local_sst((Real)x,(Real)y,(Real)radius,(Real)nu,gu,kv,wv,gk,gw,(Real)kFloor,(Real)omegaFloor);nue[q]=(Real)nu+ss.nut;wq[q]=(Real)(hxt5b::QW12[q]*f.area);
    Real vx=(Real)inletX,vy=(Real)inletY,vz=(Real)inletZ;
    if(stage27_inlet::profileActive){
      auto conn=tc[f.cell];vx=vy=vz=Real(0);
      for(int a=0;a<4;++a){std::size_t id=(std::size_t)conn.v[a];
        vx+=(Real)l[a]*stage27_inlet::velocityX[id];
        vy+=(Real)l[a]*stage27_inlet::velocityY[id];
        vz+=(Real)l[a]*stage27_inlet::velocityZ[id];}
    }
    ivx[q]=vx;ivy[q]=vy;ivz[q]=vz;
    qin[q]=fmax(Real(0),Real(-(vx*f.n[0]+vy*f.n[1]+vz*f.n[2])));}__syncthreads();Real inflow=fmax(Real(0),Real(-(inletX*f.n[0]+inletY*f.n[1]+inletZ*f.n[2])));for(int p=threadIdx.x;p<64;p+=blockDim.x){int a=p/8,b=p%8;Real z=0;for(int q=0;q<12;++q){Real dna=gp[q][a][0]*f.n[0]+gp[q][a][1]*f.n[1]+gp[q][a][2]*f.n[2],dnb=gp[q][b][0]*f.n[0]+gp[q][b][1]*f.n[1]+gp[q][b][2]*f.n[2];z+=wq[q]*((stage27_inlet::profileActive?qin[q]:inflow)*ph[q][a]*ph[q][b]+nue[q]*(-ph[q][a]*dnb+dna*ph[q][b]));}atomicAdd(mat+slot[(size_t)f.cell*64+p],z);if(a==b)atomicAdd(diag+T.g[a],z);}for(int a=threadIdx.x;a<8;a+=blockDim.x){Real z=0;for(int q=0;q<12;++q){Real dna=gp[q][a][0]*f.n[0]+gp[q][a][1]*f.n[1]+gp[q][a][2]*f.n[2];Real weight=wq[q]*((stage27_inlet::profileActive?qin[q]:inflow)*ph[q][a]+nue[q]*dna);
      z+=weight;
    }
    if(!stage27_inlet::profileActive){
      atomicAdd(rhsx+T.g[a],z*(Real)inletX);
      atomicAdd(rhsy+T.g[a],z*(Real)inletY);
      atomicAdd(rhsz+T.g[a],z*(Real)inletZ);
    }else{
      Real zx=0,zy=0,zz=0;
      for(int q=0;q<12;++q){
        Real dna=gp[q][a][0]*f.n[0]+gp[q][a][1]*f.n[1]+gp[q][a][2]*f.n[2];
        Real weight=wq[q]*(qin[q]*ph[q][a]+nue[q]*dna);
        zx+=weight*ivx[q];zy+=weight*ivy[q];zz+=weight*ivz[q];
      }
      atomicAdd(rhsx+T.g[a],zx);atomicAdd(rhsy+T.g[a],zy);atomicAdd(rhsz+T.g[a],zz);
    }}
}

#include "modules/coupling/g2_boundary_operator.inc"
inline void print_spalding_profile_integral_stats(const Dev<double>&ws,int outer){
  if(ws.n<10)return;
  std::vector<double>h(ws.n,0.0);
  HXT1_CUDA(cudaMemcpy(h.data(),ws.p,ws.n*sizeof(double),cudaMemcpyDeviceToHost));
  const double A=h[0];if(!(A>0.0))return;
  std::printf(
    "NODALS_SST_SPALDING_PROFILE_INTEGRAL it=%d "
    "rmsUTau=%.12e meanYPlus=%.12e meanProfileVelocity=%.12e "
    "meanResidualRMS=%.12e meanPointUTauRelSpan=%.12e meanProfileY=%.12e "
    "localBetaFallbackAreaFrac=%.12e status=PASS\n",
    outer,sqrt(fmax(h[1]/A,0.0)),h[2]/A,h[3]/A,h[7]/A,h[8]/A,h[9]/A,
    h.size()>10?h[10]/A:0.0);
}

inline void print_spalding_aligned_stats(const Dev<double>&ws,int outer){
  if(ws.n<11)return;
  std::vector<double>h(ws.n,0.0);
  HXT1_CUDA(cudaMemcpy(h.data(),ws.p,ws.n*sizeof(double),cudaMemcpyDeviceToHost));
  const double A=h[0];
  if(!(A>0.0))return;
  std::printf(
    "NODALS_SST_SPALDING_ALIGNED it=%d "
    "wdFloorAreaFrac=%.12e negativeDeficitAreaFrac=%.12e "
    "meanTargetTau=%.12e meanMolTauFE=%.12e meanMolTauSecant=%.12e "
    "meanPenaltyTau=%.12e targetMinusMolFEMinusPenalty=%.12e status=PASS\n",
    outer,h[7]/A,h[8]/A,h[1]/A,h[9]/A,h[11]/A,h[10]/A,
    h[1]/A-h[9]/A-h[10]/A);
}

#include "modules/coupling/g2_step_initialization.inc"
#include "modules/coupling/g2_step_convection.inc"
#include "modules/coupling/g2_step_k.inc"
#include "modules/coupling/g2_step_omega.inc"
#include "modules/coupling/g2_step_diagnostics.inc"
  return R;
}

} // namespace hxt5d
