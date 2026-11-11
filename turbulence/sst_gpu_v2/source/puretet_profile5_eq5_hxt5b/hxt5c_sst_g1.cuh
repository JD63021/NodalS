#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "sst_consistent_table_data.cuh"
#include "modules/physics/sst2003m_core.hpp"
#include "modules/physics/sst2003m_distance_core.hpp"

// G1: frozen-flow Menter SST-2003m transport equations.
// Momentum remains the converged HXT5B Nikuradse/Spalding solution; this file
// only advances k and omega and diagnoses the resulting SST eddy viscosity.
#include "modules/wall/portable/of_auto_core.hpp"
#include "modules/transport/portable/transport_core.hpp"
// Pure CG Bernstein P2 alternative scalar basis/connectivity, compile-gated.
#include "modules/transport/bernstein_p2_tet.hpp"
#include "modules/transport/bernstein_p2_device_connectivity.hpp"
namespace hxt5c {
using Real=hxt4b::Real;
using nodals_hxt1::Dev;
// Omega-only DG-P1 constitutive binding. Null preserves the validated CG path.
// A DG-enabled G2 iteration MUST bind the current 4*ntet field before momentum.
// Each call overwrites CG omega value and gradient at the physical quadrature
// point; vertex-averaged omega is never a constitutive input when this is set.
__device__ const Real* dg_omega_p1_constitutive=nullptr;
// Optional DG wall-target collection from the existing five-point Spalding
// omega target calculation. The arrays follow BH.wallT order and Q12 order.
// The original CG wall Nitsche assembly remains unchanged when unbound.
__device__ double* dg_omega_wall_target_q12=nullptr;
__device__ double* dg_omega_wall_mu_q12=nullptr;
inline void bind_dg_omega_wall_q12(double*target,double*mu){
  HXT1_CUDA(cudaMemcpyToSymbol(dg_omega_wall_target_q12,&target,sizeof(target)));
  HXT1_CUDA(cudaMemcpyToSymbol(dg_omega_wall_mu_q12,&mu,sizeof(mu)));
}

inline void bind_dg_omega_p1(const Real*deviceField){
  HXT1_CUDA(cudaMemcpyToSymbol(dg_omega_p1_constitutive,
                              &deviceField,sizeof(deviceField)));
}
__device__ __forceinline__ void eval_dg_omega_p1(
  int cell,const double lambda[4],const nodals_hxt1::TetGeom&tet,
  Real&value,Real grad[3]){
  if(!dg_omega_p1_constitutive)return;
  value=Real(0);grad[0]=grad[1]=grad[2]=Real(0);
  const double r[4][3]={{-1.,-1.,-1.},{1.,0.,0.},{0.,1.,0.},{0.,0.,1.}};
  for(int a=0;a<4;++a){
    const Real w=dg_omega_p1_constitutive[(std::size_t)4*cell+a];
    value+=Real(lambda[a])*w;
    for(int d=0;d<3;++d){
      const double g=r[a][0]*tet.invJ[d]+
                      r[a][1]*tet.invJ[3+d]+r[a][2]*tet.invJ[6+d];
      grad[d]+=Real(g)*w;
    }
  }
}
#include "modules/geometry/wall_distance_device.cuh"
#include "puretet_omega_source_audit_exp1.hpp"

#include "modules/sst/sst_constants.inc"
enum Equation : int { K_EQ=0, OMEGA_EQ=1 };
enum OmegaWallMode : int { OMEGA_WALL_TRACE_LOG=0, OMEGA_WALL_SAMPLE_LOG=1, OMEGA_WALL_OF_AUTO=2, OMEGA_WALL_COMPATIBLE=3, OMEGA_WALL_CONSISTENT_TABLE=4,
  OMEGA_WALL_LOWRE_RESOLVED=5,
  OMEGA_WALL_PROFILE5_EQ_SST
};
inline const char* omega_wall_mode_name(int m){if(m==OMEGA_WALL_PROFILE5_EQ_SST)return "profile5_eq_sst";
  switch(m){
    case OMEGA_WALL_TRACE_LOG: return "trace_log";
    case OMEGA_WALL_SAMPLE_LOG: return "sample_log";
    case OMEGA_WALL_OF_AUTO: return "of_auto";
    case OMEGA_WALL_COMPATIBLE: return "compatible";
    case OMEGA_WALL_CONSISTENT_TABLE: return "consistent_table";
    case OMEGA_WALL_LOWRE_RESOLVED: return "lowre_resolved";
    default: return "unknown";
  }
}

struct Options {
  bool enabled=false;
  double intensity=0.05;
  double lengthRatio=0.07;
  double alphaK=0.5;
  double alphaOmega=0.5;
  double nonlinearTol=1.0e-3;
  int nonlinearMax=400;
  double linearRtol=0.20;
  double linearAtol=1.0e-7;
  // Per-equation linear convergence thresholds. -1 means inherit the
  // legacy common SST_LINEAR_RTOL/ATOL (backward compatible).
  double kLinearRtol=-1.0;
  double kLinearAtol=-1.0;
  double omegaLinearRtol=-1.0;
  double omegaLinearAtol=-1.0;
  int linearMax=12;
  double gsOmega=1.0;
  std::string linearSolver="sgs1";
  int fgmresRestart=30;
  int fgsSweeps=1;
  // Stage5.A: per-equation scalar linear-method overrides, both default to
  // inherit (identical Stage3.B solver dispatch). Direct CLI or native env.
  std::string s05aKMode="inherit",s05aOmegaMode="inherit";
  int s05aKSweeps=8,s05aOmegaSweeps=8;
  int s05aCheckEvery=0;  // zero: no inner true-residual reductions

  int printEvery=10;
  double kFloor=1.0e-8;
  double omegaFloor=1.0e-3;
  double wallPenaltyGamma=50.0;
  // Five-point local-equilibrium SST profile least-squares strength.
  double profile5EqGamma=500.0;
  // PURETET_AB5_MODE_MASK_V1: A/B tests restrict *only* k/omega EQ5 samples.
  // Spalding u_tau continues to fit all five velocity samples.
  // Defaults to the untouched five-point experiment if no selector is supplied.
  // Mask is read on the host in assemble_equation. No persistent solver state is changed.
  int omegaWallMode=OMEGA_WALL_TRACE_LOG;
  // OpenFOAM-informed automatic wall treatment (experimental FE adaptation).
  double ofKappa=0.41;
  double ofE=9.8;
  double ofBeta1=0.075;
  double ofReBlend=11.0;
  bool ofProduction=true;
  double ofProductionScale=1.0;
  double ofWallVolumeScale=1.0;
  double ofProductionLimit=10.0;
  // Split the two deviations from the historical divergence-free shortcut:
  //  - deviatoricProduction: use dev(2S):grad(U) in SST production.
  //  - deviatoricMomentum: add the matching cross/deviatoric momentum stress.
  // They are independent so production-only and momentum-only A/B tests are possible.
  bool deviatoricProduction=false;
  bool deviatoricMomentum=false;
  // Experimental consistency switch: make OF-auto SST wall log branch use
  // the exact same Spalding u_tau and wall-law constants as momentum.
  bool sharedSpaldingWallState=false;
  // FE-native Spalding inversion from the whole wall-normal velocity profile.
  // This changes only how u_tau is inferred; the existing SST target/penalty
  // placement remains controlled by wallSampleFraction.
  bool profileIntegralUTau=false;
  // Normal Gauss-Legendre order used by the nonlinear profile-integral wall law.
  // Allowed values: 4, 8, 12.  NQ=4 exactly preserves the original experiment.
  int profileIntegralNQ=4;
  // Experimental full-profile OF-auto closure.  When enabled, the same
  // profile-integrated u_tau is used to evaluate the OF-auto omega target and
  // production correction throughout the wall-adjacent HEX rather than only at
  // wallSampleFraction.  The singular omega target is applied only for y+ at or
  // above profileClosureYPlusMin.
  bool profileIntegralClosure=false;
  double profileClosureYPlusMin=1.0;
  // Average-first NQ wall closure.  NQ is used to volume-average U, k, omega,
  // gradients and y through the wall-adjacent column.  One OF-auto state is
  // then evaluated from those averages and imposed through a rank-one
  // cell-average omega penalty plus a volume-distributed production source.
  bool profileAverageClosure=false;
  // 0=Spalding(Ubar), 1=log(Ubar), 2=Spalding(Ubar-Uwall).
  int profileAverageWallLaw=0;
  // Prescribed physical matching height for single-point wall-law sampling.
  double profileMatchY=0.0;
  // Three prescribed physical heights for equal-weight Spalding profile fit.
  double profileFitY1=0.0,profileFitY2=0.0,profileFitY3=0.0,profileFitY4=0.0,profileFitY5=0.0;
  const hxt5b::WallMatchPoint* profileMatchPoints=nullptr;
  const hxt5b::TetWallPackedPoint* profileTetMatchPoints=nullptr;
  ::nodals::wall::shared27::Record* stage27SavedWallFit=nullptr;
  const hxt5b::TetFacePlan* tetWallPlans=nullptr;
  const hxt5b::TetWallQuadCache* tetWallQuads=nullptr;
  // Wall-resolved low-Re SST asymptote: omega_w=C*nu/(beta1*y1^2).
  double lowReOmegaFactor=60.0;
};

enum SSTStateSpace : int {
  SST_STATE_Q1_CARRIER=0,
  SST_STATE_Q1=1,
  SST_STATE_BERNSTEIN_Q2=2,
  SST_STATE_Q1BF2_UNLIMITED=3
};

inline int sst_state_space_mode_host()
{
  const char* e=std::getenv("SST_STATE_SPACE");
  if(!e || !*e) return SST_STATE_Q1_CARRIER;
  const std::string v(e);
  if(v=="q1_carrier") return SST_STATE_Q1_CARRIER;
  if(v=="q1") return SST_STATE_Q1;
  if(v=="bernstein_q2") return SST_STATE_BERNSTEIN_Q2;
  if(v=="q1bf2" || v=="q1bf2_unlimited" || v=="unlimited14") return SST_STATE_Q1BF2_UNLIMITED;
  throw std::runtime_error("SST_STATE_SPACE must be q1, q1_carrier, bernstein_q2, or q1bf2_unlimited");
}

inline const char* sst_state_space_name(int m)
{
  return m==SST_STATE_Q1 ? "Q1" :
         (m==SST_STATE_BERNSTEIN_Q2 ? "BERNSTEIN_Q2" :
          (m==SST_STATE_Q1BF2_UNLIMITED ? "Q1_BF2_UNLIMITED" : "Q1_CARRIER"));
}

inline bool q1_transport_enabled()
{
  return sst_state_space_mode_host()==SST_STATE_Q1;
}

__device__ __constant__ int c_sst_state_space_mode;
__device__ __constant__ int c_sst_deviatoric_production;

inline void sync_sst_state_space_mode()
{
  const int m=sst_state_space_mode_host();
  HXT1_CUDA(cudaMemcpyToSymbol(c_sst_state_space_mode,&m,sizeof(int)));
}

inline void sync_sst_deviatoric_production(bool enabled)
{
  const int m=enabled?1:0;
  HXT1_CUDA(cudaMemcpyToSymbol(c_sst_deviatoric_production,&m,sizeof(int)));
}

// HXT compatibility adapter: no retained constitutive arithmetic.
using LocalSST = nodals::sst2003m::LocalState<Real>;
__device__ __forceinline__ LocalSST local_sst(
    Real x,Real y,Real radius,Real nu,
    const Real gu[9],Real kRaw,Real omegaRaw,
    const Real gk[3],const Real gw[3],Real kFloor,Real omegaFloor,
    Real geometricDistance=Real(-1))
{
  if(geometricDistance>=Real(0))return nodals::sst2003m::evaluate_at_distance<Real>(
      geometricDistance,radius,nu,gu,kRaw,omegaRaw,gk,gw,kFloor,omegaFloor,
      c_sst_deviatoric_production);
  return nodals::sst2003m::evaluate_pipe<Real>(
      x,y,radius,nu,gu,kRaw,omegaRaw,gk,gw,kFloor,omegaFloor,
      c_sst_deviatoric_production);
}


// ---------------------------------------------------------------------------
// TRUE_Q1_TRANSPORT_V1
// Q1_VERTEX_CARRIER_V1
//
// The HEX SST transport space has 14 local coefficients: 8 vertex modes plus
// 6 enrichment/face modes.  Only the global vertex coefficients are subjected
// to the positivity floor after each scalar solve.  The enrichment coefficients
// are modal corrections and may have either sign.  Feeding the raw 14-mode
// interpolant into nonlinear SST constitutive functions can therefore create
// nonphysical negative/huge k or omega between DOFs even when all vertices are
// positive.
//
// The Q1 carrier below evaluates the nonlinear SST state from the eight
// positivity-limited vertex coefficients only.  The full 14-DOF field remains
// the unknown of the transport equation; only constitutive-state evaluation is
// filtered.  This is intentionally analogous to a positivity-preserving
// subcell/carrier state for nonlinear coefficients.
// ---------------------------------------------------------------------------
__device__ __forceinline__ Real hex_q1_shape(int a,Real xr,Real yr,Real zr)
{
  const Real c[3]={xr,yr,zr};
  Real q=Real(1);
  #pragma unroll
  for(int d=0;d<3;++d)
    q*=Real(0.5)*(Real(1)+(Real)hxt4a::h4_hex_sign[a][d]*c[d]);
  return q;
}

__device__ __forceinline__ Real hex_q1_value(
    Real xr,Real yr,Real zr,const Real*q,Real floorValue)
{
  Real v=Real(0);
  #pragma unroll
  for(int a=0;a<8;++a)
    v+=fmax(q[a],floorValue)*hex_q1_shape(a,xr,yr,zr);
  return v;
}

__device__ __forceinline__ void hex_q1_carrier(
    Real xr,Real yr,Real zr,const double I[9],
    const Real*K,const Real*W,Real kFloor,Real omegaFloor,
    Real&kv,Real&wv,Real gk[3],Real gw[3])
{
  double gr[14][3],gp[14][3];
  for(int a=0;a<14;++a)
    for(int d=0;d<3;++d)
      gr[a][d]=0.0;

  const double c[3]={(double)xr,(double)yr,(double)zr};
  Real kc[8],wc[8];
  kv=wv=Real(0);
  for(int d=0;d<3;++d)gk[d]=gw[d]=Real(0);

  #pragma unroll
  for(int a=0;a<8;++a){
    double f[3],df[3];
    #pragma unroll
    for(int d=0;d<3;++d){
      const double sg=(double)hxt4a::h4_hex_sign[a][d];
      f[d]=0.5*(1.0+sg*c[d]);
      df[d]=0.5*sg;
    }
    const double N=f[0]*f[1]*f[2];
    gr[a][0]=df[0]*f[1]*f[2];
    gr[a][1]=f[0]*df[1]*f[2];
    gr[a][2]=f[0]*f[1]*df[2];

    kc[a]=fmax(K[a],kFloor);
    wc[a]=fmax(W[a],omegaFloor);
    kv+=kc[a]*(Real)N;
    wv+=wc[a]*(Real)N;
  }

  hxt2::phys_grad14(gr,I,gp);

  #pragma unroll
  for(int a=0;a<8;++a)
    #pragma unroll
    for(int d=0;d<3;++d){
      gk[d]+=kc[a]*(Real)gp[a][d];
      gw[d]+=wc[a]*(Real)gp[a][d];
    }
}


// ---------------------------------------------------------------------------
// BERNSTEIN_Q2_POSITIVE_CARRIER_V1
//
// The existing 14-mode HEX scalar polynomial is Q1 plus six tensor-Q2 face
// bubbles.  It is therefore a 14-dimensional subspace of tensor-product Q2.
// Convert that polynomial to its 27 quadratic Bernstein controls.  Because
// Bernstein basis functions are nonnegative and form a partition of unity,
// nonnegative controls imply a positive field everywhere in the element.
//
// We keep the positive, degree-elevated Q1 control net fixed and multiply all
// six face-bubble coefficients by the largest theta in [0,1] that keeps every
// one of the 27 Bernstein controls >= the requested floor.  This is a minimal
// positivity limiter: if all raw Q2 controls are already admissible theta=1
// and the full quadratic state is unchanged.  No arbitrary upper cap is used.
//
// IMPORTANT: transport remains the existing FULL_14DOF Q1+face-bubble space.
// This is a positivity-preserving Bernstein-Q2 constitutive carrier, not a
// 27-independent-DOF Q2 transport discretization.
// ---------------------------------------------------------------------------
__device__ __forceinline__ Real bern_q1_control(
    int ix,int iy,int iz,const Real*q,Real floorValue)
{
  const Real xr=(Real)(ix-1),yr=(Real)(iy-1),zr=(Real)(iz-1);
  return hex_q1_value(xr,yr,zr,q,floorValue);
}

__device__ __forceinline__ Real bern_face_delta(
    int ix,int iy,int iz,const Real*q)
{
  Real d=Real(0);
  // face 0 z-, face 1 z+, face 2 y-, face 3 x+, face 4 y+, face 5 x-
  if(ix==1 && iy==1){
    if(iz==0)d+=Real(4)*q[8];
    if(iz==1)d+=Real(2)*(q[8]+q[9]);
    if(iz==2)d+=Real(4)*q[9];
  }
  if(ix==1 && iz==1){
    if(iy==0)d+=Real(4)*q[10];
    if(iy==1)d+=Real(2)*(q[10]+q[12]);
    if(iy==2)d+=Real(4)*q[12];
  }
  if(iy==1 && iz==1){
    if(ix==0)d+=Real(4)*q[13];
    if(ix==1)d+=Real(2)*(q[13]+q[11]);
    if(ix==2)d+=Real(4)*q[11];
  }
  return d;
}

__device__ __forceinline__ Real bern_theta_positive(
    const Real*q,Real floorValue)
{
  Real theta=Real(1);
  #pragma unroll
  for(int ix=0;ix<3;++ix)
    #pragma unroll
    for(int iy=0;iy<3;++iy)
      #pragma unroll
      for(int iz=0;iz<3;++iz){
        const Real base=bern_q1_control(ix,iy,iz,q,floorValue);
        const Real delta=bern_face_delta(ix,iy,iz,q);
        if(delta<Real(0) && base+delta<floorValue){
          const Real den=-delta;
          Real t=(base-floorValue)/den;
          t=fmin(Real(1),fmax(Real(0),t));
          theta=fmin(theta,t);
        }
      }
  return theta;
}

__device__ __forceinline__ void hex_bernstein_q2_carrier(
    Real xr,Real yr,Real zr,const double I[9],
    const Real*K,const Real*W,Real kFloor,Real omegaFloor,
    Real&kv,Real&wv,Real gk[3],Real gw[3])
{
  const Real tk=bern_theta_positive(K,kFloor);
  const Real tw=bern_theta_positive(W,omegaFloor);

  double ph[14],gr[14][3],gp[14][3];
  hxt2::hex_basis_ref((double)xr,(double)yr,(double)zr,ph,gr);
  hxt2::phys_grad14(gr,I,gp);

  kv=wv=Real(0);
  for(int d=0;d<3;++d)gk[d]=gw[d]=Real(0);

  #pragma unroll
  for(int a=0;a<14;++a){
    const Real kc=(a<8)?fmax(K[a],kFloor):tk*K[a];
    const Real wc=(a<8)?fmax(W[a],omegaFloor):tw*W[a];
    const Real pa=(Real)ph[a];
    kv+=kc*pa;
    wv+=wc*pa;
    #pragma unroll
    for(int d=0;d<3;++d){
      const Real gd=(Real)gp[a][d];
      gk[d]+=kc*gd;
      gw[d]+=wc*gd;
    }
  }

  // Roundoff-only guard.  The Bernstein control proof should already keep
  // these above the floor everywhere.
  kv=fmax(kv,kFloor);
  wv=fmax(wv,omegaFloor);
}

// Historical unrestricted Q1+BF2 constitutive state.  All 14 transport
// coefficients are evaluated directly at the quadrature/sample point.
// There is no Bernstein control-net limiter and no Q1 carrier projection.
// local_sst() still applies the existing pointwise safety floors to k/omega.
__device__ __forceinline__ void hex_q1bf2_unlimited_carrier(
    Real xr,Real yr,Real zr,const double I[9],
    const Real*K,const Real*W,
    Real&kv,Real&wv,Real gk[3],Real gw[3])
{
  double ph[14],gr[14][3],gp[14][3];
  hxt2::hex_basis_ref((double)xr,(double)yr,(double)zr,ph,gr);
  hxt2::phys_grad14(gr,I,gp);
  kv=wv=Real(0);
  for(int d=0;d<3;++d)gk[d]=gw[d]=Real(0);
  #pragma unroll
  for(int a=0;a<14;++a){
    const Real pa=(Real)ph[a];
    kv+=K[a]*pa;
    wv+=W[a]*pa;
    #pragma unroll
    for(int d=0;d<3;++d){
      const Real gd=(Real)gp[a][d];
      gk[d]+=K[a]*gd;
      gw[d]+=W[a]*gd;
    }
  }
}

__device__ __forceinline__ void hex_sst_carrier(
    Real xr,Real yr,Real zr,const double I[9],
    const Real*K,const Real*W,Real kFloor,Real omegaFloor,
    Real&kv,Real&wv,Real gk[3],Real gw[3])
{
  if(c_sst_state_space_mode==SST_STATE_BERNSTEIN_Q2)
    hex_bernstein_q2_carrier(xr,yr,zr,I,K,W,kFloor,omegaFloor,kv,wv,gk,gw);
  else if(c_sst_state_space_mode==SST_STATE_Q1BF2_UNLIMITED)
    hex_q1bf2_unlimited_carrier(xr,yr,zr,I,K,W,kv,wv,gk,gw);
  else
    hex_q1_carrier(xr,yr,zr,I,K,W,kFloor,omegaFloor,kv,wv,gk,gw);
}

__global__ void zero_real(std::size_t n,Real*x){std::size_t i=(std::size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]=Real(0);}
__global__ void combine_scalar(std::size_t n,const Real*base,const Real*conv,const Real*vol,const Real*dg,const Real*wall,Real*val){std::size_t i=(std::size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)val[i]=base[i]+conv[i]+vol[i]+dg[i]+wall[i];}
__global__ void relax_field(int n,const Real*old,Real alpha,Real*x){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)x[i]=old[i]+alpha*(x[i]-old[i]);}
__global__ void zero_enrichment_kernel(int first,int n,Real*x){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x)+first;if(i<n)x[i]=Real(0);}

__global__ void sst_hex_volume(
    const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,const nodals_hxt1::HexVel*hv,
    const std::int32_t*slot,const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,
    Real*mat,Real*rhs,std::uint64_t n,double radius,double nu,int equation,double kFloor,double omegaFloor,
    unsigned long long*bad)
{
  std::uint64_t c=(std::uint64_t)blockIdx.x;if(c>=n)return;
  __shared__ Real U0[14],U1[14],U2[14],K[14],W[14];
  __shared__ Real PH[64][14],GP[64][14][3],DW[64],RW[64],SW[64];
  if(threadIdx.x<14){int g=hv[c].g[threadIdx.x];U0[threadIdx.x]=u0[g];U1[threadIdx.x]=u1[g];U2[threadIdx.x]=u2[g];K[threadIdx.x]=k[g];W[threadIdx.x]=omega[g];}
  __syncthreads();
  int iq=(int)threadIdx.x;
  if(iq<64){
    int ix=iq>>4,iy=(iq>>2)&3,iz=iq&3;
    double xr=hxt5a::h5_hex_x[ix],yr=hxt5a::h5_hex_x[iy],zr=hxt5a::h5_hex_x[iz];
    double qw=hxt5a::h5_hex_w[ix]*hxt5a::h5_hex_w[iy]*hxt5a::h5_hex_w[iz];
    double ph[14],gr[14][3],gp[14][3],I[9],det;
    hxt2::hex_basis_ref(xr,yr,zr,ph,gr);
    const bool geomOk=hxt2::hex_metric(pts,hc[c],xr,yr,zr,I,det);
    if(!geomOk){
      atomicAdd(bad,1ULL);DW[iq]=RW[iq]=SW[iq]=Real(0);
      for(int a=0;a<14;++a){PH[iq][a]=0;for(int d=0;d<3;++d)GP[iq][a][d]=0;}
    }else{
      hxt2::phys_grad14(gr,I,gp);
      Real gu[9]={0,0,0,0,0,0,0,0,0},gk[3]={0,0,0},gw[3]={0,0,0};Real kv=0,wv=0;
      for(int a=0;a<14;++a){
        Real pa=(Real)ph[a];PH[iq][a]=pa;
        for(int d=0;d<3;++d){Real gd=(Real)gp[a][d];GP[iq][a][d]=gd;gu[d]+=U0[a]*gd;gu[3+d]+=U1[a]*gd;gu[6+d]+=U2[a]*gd;}
      }
      hex_sst_carrier((Real)xr,(Real)yr,(Real)zr,I,K,W,(Real)kFloor,(Real)omegaFloor,kv,wv,gk,gw);
      double xq=0,yq=0;for(int a=0;a<8;++a){double N=.125*(1+hxt4a::h4_hex_sign[a][0]*xr)*(1+hxt4a::h4_hex_sign[a][1]*yr)*(1+hxt4a::h4_hex_sign[a][2]*zr);auto X=pts[hc[c].v[a]];xq+=N*X.x;yq+=N*X.y;}
      LocalSST s=local_sst((Real)xq,(Real)yq,(Real)radius,(Real)nu,gu,kv,wv,gk,gw,(Real)kFloor,(Real)omegaFloor);
      const Real wt=(Real)(det*qw);
      DW[iq]=wt*(equation==K_EQ?s.sigmaK:s.sigmaW)*s.nut;
      RW[iq]=wt*(equation==K_EQ?s.reactK:s.reactOmega);
      SW[iq]=wt*(equation==K_EQ?s.pk:s.srcOmega);
    }
  }
  __syncthreads();
  for(int p=(int)threadIdx.x;p<196;p+=(int)blockDim.x){int a=p/14,b=p%14;Real z=0;for(int q=0;q<64;++q){Real gd=GP[q][a][0]*GP[q][b][0]+GP[q][a][1]*GP[q][b][1]+GP[q][a][2]*GP[q][b][2];z+=DW[q]*gd+RW[q]*PH[q][a]*PH[q][b];}atomicAdd(mat+slot[c*196ull+(std::size_t)p],z);}
  for(int a=(int)threadIdx.x;a<14;a+=(int)blockDim.x){Real z=0;for(int q=0;q<64;++q)z+=SW[q]*PH[q][a];atomicAdd(rhs+hv[c].g[a],z);}
}

#include "modules/transport/tet_volume_kernel.inc"
#if defined(NODALS_P2_COMPILE_SMOKE)
#include "modules/transport/bernstein_p2_sst_volume.inc"
#endif
#include "modules/transport/p1_compact.cuh"
#include "modules/transport/interface_cache.inc"
#include "modules/transport/interface_cached_flux.inc"
#include "modules/transport/interface_uncached_flux.inc"
__global__ void sst_inlet_hex(
    const hxt5b::HexFacePlan*F,int nf,const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,const nodals_hxt1::HexVel*hv,const std::int32_t*slot,
    const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,Real*mat,Real*rhs,double qIn,double inletX,double inletY,double inletZ,double radius,double nu,int equation,double kFloor,double omegaFloor,unsigned long long*bad)
{
  int fi=(int)blockIdx.x;if(fi>=nf)return;auto f=F[fi];auto H=hv[f.cell];
  __shared__ Real U0[14],U1[14],U2[14],K[14],W[14],PH[16][14],GP[16][14][3],NUE[16],WQ[16];
  if(threadIdx.x<14){int g=H.g[threadIdx.x];U0[threadIdx.x]=u0[g];U1[threadIdx.x]=u1[g];U2[threadIdx.x]=u2[g];K[threadIdx.x]=k[g];W[threadIdx.x]=omega[g];}__syncthreads();
  if(threadIdx.x<16){int q=(int)threadIdx.x,i=q>>2,j=q&3;double c[3]={0,0,0};c[f.fd]=f.fv;c[f.d0]=hxt5a::h5_hex_x[i];c[f.d1]=hxt5a::h5_hex_x[j];double ph[14],gr[14][3],gp[14][3],I[9],det;
    hxt2::hex_basis_ref(c[0],c[1],c[2],ph,gr);const bool geomOk=hxt2::hex_metric(pts,hc[f.cell],c[0],c[1],c[2],I,det);
    if(!geomOk){
      atomicAdd(bad,1ULL);WQ[q]=NUE[q]=Real(0);for(int a=0;a<14;++a){PH[q][a]=0;for(int d=0;d<3;++d)GP[q][a][d]=0;}
    }else{
      hxt2::phys_grad14(gr,I,gp);
      double X[3],dX[3][3];hxt5b::q1dev(pts,hc[f.cell],c[0],c[1],c[2],X,dX);double cr[3]={dX[f.d0][1]*dX[f.d1][2]-dX[f.d0][2]*dX[f.d1][1],dX[f.d0][2]*dX[f.d1][0]-dX[f.d0][0]*dX[f.d1][2],dX[f.d0][0]*dX[f.d1][1]-dX[f.d0][1]*dX[f.d1][0]};if(cr[0]*f.n[0]+cr[1]*f.n[1]+cr[2]*f.n[2]<0){cr[0]*=-1;cr[1]*=-1;cr[2]*=-1;}double jac=sqrt(cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]);
      Real gu[9]={0,0,0,0,0,0,0,0,0},gk[3]={0,0,0},gw[3]={0,0,0},kv=0,wv=0;for(int a=0;a<14;++a){Real pa=(Real)ph[a];PH[q][a]=pa;for(int d=0;d<3;++d){Real gd=(Real)gp[a][d];GP[q][a][d]=gd;gu[d]+=U0[a]*gd;gu[3+d]+=U1[a]*gd;gu[6+d]+=U2[a]*gd;}}
      hex_sst_carrier((Real)c[0],(Real)c[1],(Real)c[2],I,K,W,(Real)kFloor,(Real)omegaFloor,kv,wv,gk,gw);
      LocalSST ss=local_sst((Real)X[0],(Real)X[1],(Real)radius,(Real)nu,gu,kv,wv,gk,gw,(Real)kFloor,(Real)omegaFloor);Real extra=(equation==K_EQ?ss.sigmaK:ss.sigmaW)*ss.nut;NUE[q]=(Real)nu+extra;WQ[q]=(Real)(hxt5a::h5_hex_w[i]*hxt5a::h5_hex_w[j]*jac);
    }
  }__syncthreads();
  Real inflow=fmax(Real(0),Real(-(inletX*f.n[0]+inletY*f.n[1]+inletZ*f.n[2])));
  for(int p=(int)threadIdx.x;p<196;p+=(int)blockDim.x){int a=p/14,b=p%14;Real z=0;for(int q=0;q<16;++q){Real dna=GP[q][a][0]*(Real)f.n[0]+GP[q][a][1]*(Real)f.n[1]+GP[q][a][2]*(Real)f.n[2],dnb=GP[q][b][0]*(Real)f.n[0]+GP[q][b][1]*(Real)f.n[1]+GP[q][b][2]*(Real)f.n[2];z+=WQ[q]*(inflow*PH[q][a]*PH[q][b]+NUE[q]*(-PH[q][a]*dnb+dna*PH[q][b]));}atomicAdd(mat+slot[(std::size_t)f.cell*196ull+(std::size_t)p],z);}
  for(int a=(int)threadIdx.x;a<14;a+=(int)blockDim.x){Real z=0;for(int q=0;q<16;++q){Real dna=GP[q][a][0]*(Real)f.n[0]+GP[q][a][1]*(Real)f.n[1]+GP[q][a][2]*(Real)f.n[2];z+=WQ[q]*(inflow*PH[q][a]*(Real)qIn+NUE[q]*dna*(Real)qIn);}atomicAdd(rhs+H.g[a],z);}
}

#if defined(NODALS_P2_COMPILE_SMOKE)
#include "modules/transport/bernstein_p2_inlet.inc"
#endif
#include "modules/transport/tet_inlet_flux.inc"
#include "modules/wall/of_auto_wall_state.inc"

// Carrier/Lacombe compatible SST wall state, adapted to the existing NodalS
// off-wall Spalding sampling.  The compatible coupling is published; using
// Spalding for u_tau and the current off-wall sample/penalty placement is the
// NodalS adaptation being tested here.
struct CompatibleWallState {
  Real uTauVisc=0,uStarLog=0,uStar=0,tauTarget=0,yPlus=0;
  Real omegaVis=0,omegaLog=0,omegaTarget=0,pkCompat=0;
};

__device__ __forceinline__ CompatibleWallState compatible_wall_state(
    Real yd,Real nu,Real Ut,Real uTau,Real kRaw,const LocalSST&ss,
    Real kFloor,Real kappa)
{
  CompatibleWallState q;
  const Real tiny=Real(1e-20);
  const Real y=fmax(yd,tiny),kap=fmax(kappa,Real(1e-6));
  const Real kval=fmax(kRaw,kFloor);
  const Real betaStar=Real(BETA_STAR);
  const Real betaStar25=sqrt(sqrt(betaStar));
  const Real sqrtBetaStar=sqrt(betaStar);
  q.uTauVisc=sqrt(fmax(Real(0),nu*fmax(Ut,Real(0))/y));
  q.uStarLog=betaStar25*sqrt(kval);
  const Real uv2=q.uTauVisc*q.uTauVisc,ul2=q.uStarLog*q.uStarLog;
  q.uStar=sqrt(sqrt(uv2*uv2+ul2*ul2));
  q.tauTarget=fmax(Real(0),uTau*q.uStar);
  q.yPlus=y*fmax(uTau,Real(0))/fmax(nu,tiny);
  const Real S=fmax(ss.strain,Real(0));
  const Real omega0=Real(1600)*S;
  const Real root=sqrt(fmax(Real(0),betaStar*omega0/(Real(6)*fmax(nu,tiny))));
  const Real denom=Real(1)+y*root;
  q.omegaVis=omega0/(denom*denom);
  q.omegaLog=fmax(uTau,Real(0))/(kap*y*sqrtBetaStar);
  q.omegaTarget=sqrt(q.omegaVis*q.omegaVis+q.omegaLog*q.omegaLog);
  q.pkCompat=ss.pk;
  if(q.yPlus>Real(10))q.pkCompat=fmax(Real(0),uTau*uTau*q.uStar/(kap*y));
  return q;
}

__global__ void sst_omega_wall_hex(
    const hxt5b::HexFacePlan*F,int nf,const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,const nodals_hxt1::HexVel*hv,const std::int32_t*slot,
    const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,Real*mat,Real*rhs,double radius,double nu,double sampleFraction,double penaltyGamma,double kFloor,double omegaFloor,
    int equation,int wallMode,double ofKappa,double ofE,double ofBeta1,double ofReBlend,double ofProdLimit,int sharedSpaldingWallState,int profileIntegralUTau,int profileIntegralNQ,double lowReOmegaFactor,double*wallStats,unsigned long long*bad)
{
  int fi=(int)blockIdx.x;if(fi>=nf)return;auto f=F[fi];auto H=hv[f.cell];
  __shared__ Real U0[14],U1[14],U2[14],K[14],W[14],PH[16][14],GP[16][14][3],NUE[16],TAU[16],GVAL[16],WQ[16],WSAMPLE[16],WTRACE[16];
  if(threadIdx.x<14){int g=H.g[threadIdx.x];U0[threadIdx.x]=u0[g];U1[threadIdx.x]=u1[g];U2[threadIdx.x]=u2[g];K[threadIdx.x]=k[g];W[threadIdx.x]=omega[g];}__syncthreads();
  if(threadIdx.x<16){
    int q=(int)threadIdx.x,i=q>>2,j=q&3;
    for(int a=0;a<14;++a){PH[q][a]=Real(0);for(int d=0;d<3;++d)GP[q][a][d]=Real(0);}NUE[q]=TAU[q]=GVAL[q]=WQ[q]=WSAMPLE[q]=WTRACE[q]=Real(0);
    double cw[3]={0,0,0};cw[f.fd]=f.fv;cw[f.d0]=hxt5a::h5_hex_x[i];cw[f.d1]=hxt5a::h5_hex_x[j];double cs[3]={cw[0],cw[1],cw[2]};cs[f.fd]=f.fv+sampleFraction*(-2*f.fv);
    double phw[14],grw[14][3],gpw[14][3],Iw[9],detw;double phs[14],grs[14][3],gps[14][3],Is[9],dets;
    hxt2::hex_basis_ref(cw[0],cw[1],cw[2],phw,grw);hxt2::hex_basis_ref(cs[0],cs[1],cs[2],phs,grs);
    bool ok=hxt2::hex_metric(pts,hc[f.cell],cw[0],cw[1],cw[2],Iw,detw)&&hxt2::hex_metric(pts,hc[f.cell],cs[0],cs[1],cs[2],Is,dets);
    if(ok){
      hxt2::phys_grad14(grw,Iw,gpw);hxt2::phys_grad14(grs,Is,gps);
      double Xw[3],dXw[3][3],Xs[3],dXs[3][3];hxt5b::q1dev(pts,hc[f.cell],cw[0],cw[1],cw[2],Xw,dXw);hxt5b::q1dev(pts,hc[f.cell],cs[0],cs[1],cs[2],Xs,dXs);
      double rr=hypot(Xw[0],Xw[1]),jac=0,yd=0,ut=0,yp=0,betaWall=0,slip=0,nr0=0,nr1=0;
      ok=rr>0;
      if(ok){nr0=Xw[0]/rr;nr1=Xw[1]/rr;double nr[3]={nr0,nr1,0};double cr[3]={dXw[f.d0][1]*dXw[f.d1][2]-dXw[f.d0][2]*dXw[f.d1][1],dXw[f.d0][2]*dXw[f.d1][0]-dXw[f.d0][0]*dXw[f.d1][2],dXw[f.d0][0]*dXw[f.d1][1]-dXw[f.d0][1]*dXw[f.d1][0]};if(cr[0]*nr[0]+cr[1]*nr[1]+cr[2]*nr[2]<0){cr[0]*=-1;cr[1]*=-1;cr[2]*=-1;}jac=sqrt(cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]);yd=(Xw[0]-Xs[0])*nr[0]+(Xw[1]-Xs[1])*nr[1];ok=jac>0&&yd>0;}
      Real gu[9]={0,0,0,0,0,0,0,0,0},gk[3]={0,0,0},gw[3]={0,0,0},kv=0,wv=0,ww=0,us0=0,us1=0,us2=0;
      const bool samplePlacement=(wallMode==OMEGA_WALL_SAMPLE_LOG||wallMode==OMEGA_WALL_OF_AUTO||wallMode==OMEGA_WALL_COMPATIBLE||wallMode==OMEGA_WALL_CONSISTENT_TABLE);
      for(int a=0;a<14;++a){
        PH[q][a]=samplePlacement ? (a<8 ? hex_q1_shape(a,(Real)cs[0],(Real)cs[1],(Real)cs[2]) : Real(0)) : (Real)phw[a];
        for(int d=0;d<3;++d)GP[q][a][d]=(Real)gpw[a][d];
        if(ok){Real ps=(Real)phs[a];us0+=U0[a]*ps;us1+=U1[a]*ps;us2+=U2[a]*ps;slip+=(double)U2[a]*phs[a];for(int d=0;d<3;++d){Real gd=(Real)gps[a][d];gu[d]+=U0[a]*gd;gu[3+d]+=U1[a]*gd;gu[6+d]+=U2[a]*gd;}}
      }
      if(ok){
        hex_sst_carrier((Real)cs[0],(Real)cs[1],(Real)cs[2],Is,K,W,(Real)kFloor,(Real)omegaFloor,kv,wv,gk,gw);
        ww=hex_q1_value((Real)cw[0],(Real)cw[1],(Real)cw[2],W,(Real)omegaFloor);
      }
      if(ok){
        if(profileIntegralUTau){
          double ypProf=0.0,ubarProf=0.0,ybarProf=0.0,rmsProf=0.0,spanProf=0.0,bavgProf[14];
          if(!hxt5b::spalding_profile_hex(pts,hc[f.cell],f,cw,U2,nu,profileIntegralNQ,ut,ypProf,betaWall,ubarProf,ybarProf,bavgProf,rmsProf,spanProf))ok=false;
          else yp=yd*ut/nu; // keep reported/target y+ tied to the existing SST sample location
        }else if(!hxt5b::spalding(slip,yd,nu,ut,yp,betaWall))ok=false;
      }
      if(ok){
        LocalSST ss=local_sst((Real)Xs[0],(Real)Xs[1],(Real)radius,(Real)nu,gu,kv,wv,gk,gw,(Real)kFloor,(Real)omegaFloor);Real nue=(Real)nu+(equation==K_EQ?ss.sigmaK:ss.sigmaW)*ss.nut;NUE[q]=nue;TAU[q]=(Real)(penaltyGamma/yd)*nue;
        Real target=(equation==OMEGA_EQ)?(Real)(ut/(sqrt(BETA_STAR)*KAPPA_OMEGA_WALL*yd)):kv;
        // Pure wall-resolved SST mode:
        //   k_w = 0
        //   omega_w = C_omega * nu / (beta1 * y1^2)
        // Here yd is the existing physical off-wall reference distance; with
        // WALL_SAMPLE_FRACTION=0.5 it is the first-cell midplane distance.
        // Spalding quantities computed above are DIAGNOSTIC ONLY in this mode.
        if(wallMode==OMEGA_WALL_LOWRE_RESOLVED){
          target=(equation==K_EQ)
            ? Real(0)
            : (Real)(lowReOmegaFactor*nu/(fmax(ofBeta1,1.0e-12)*yd*yd));
        }
        OFWallState oa;CompatibleWallState ca;hxt5c_table::WallTableState ta;Real Ut=0;
        if(wallMode==OMEGA_WALL_OF_AUTO||wallMode==OMEGA_WALL_COMPATIBLE||wallMode==OMEGA_WALL_CONSISTENT_TABLE){Real un=us0*(Real)nr0+us1*(Real)nr1;Ut=sqrt(fmax(Real(0),us0*us0+us1*us1+us2*us2-un*un));}
        if(wallMode==OMEGA_WALL_OF_AUTO){oa=of_auto_wall_state((Real)yd,(Real)nu,Ut,(Real)ut,kv,wv,ss,(Real)kFloor,(Real)omegaFloor,(Real)ofKappa,(Real)ofE,(Real)ofBeta1,(Real)ofReBlend,(Real)ofProdLimit,sharedSpaldingWallState);target=oa.omegaTarget;}
        else if(wallMode==OMEGA_WALL_COMPATIBLE){ca=compatible_wall_state((Real)yd,(Real)nu,Ut,(Real)ut,kv,ss,(Real)kFloor,(Real)ofKappa);target=ca.omegaTarget;}
        else if(wallMode==OMEGA_WALL_CONSISTENT_TABLE){ta=hxt5c_table::invert_velocity((double)Ut,yd,nu);if(!ta.ok)ok=false;else target=(equation==K_EQ)?(Real)(ta.kPlus*ta.uTau*ta.uTau):(Real)(ta.omegaPlus*ta.uTau*ta.uTau/nu);}
        if(ok){GVAL[q]=target;WQ[q]=(Real)(hxt5a::h5_hex_w[i]*hxt5a::h5_hex_w[j]*jac);WSAMPLE[q]=(equation==K_EQ)?kv:wv;WTRACE[q]=(equation==K_EQ)?kv:ww;}
        else {NUE[q]=TAU[q]=GVAL[q]=WQ[q]=WSAMPLE[q]=WTRACE[q]=Real(0);}
        if(ok&&wallStats){
          atomicAdd(wallStats+0,(double)WQ[q]);atomicAdd(wallStats+1,(double)GVAL[q]*(double)WQ[q]);atomicAdd(wallStats+2,yp*(double)WQ[q]);atomicAdd(wallStats+3,(double)WSAMPLE[q]*(double)WQ[q]);atomicAdd(wallStats+4,(double)WTRACE[q]*(double)WQ[q]);
          if(wallMode==OMEGA_WALL_OF_AUTO){atomicAdd(wallStats+5,(double)oa.rey*(double)WQ[q]);atomicAdd(wallStats+6,(double)oa.lamFrac*(double)WQ[q]);atomicAdd(wallStats+7,(double)oa.omegaVis*(double)WQ[q]);atomicAdd(wallStats+8,(double)oa.omegaLog*(double)WQ[q]);atomicAdd(wallStats+9,(double)ss.pk*(double)WQ[q]);atomicAdd(wallStats+10,(double)oa.pkAuto*(double)WQ[q]);atomicAdd(wallStats+11,(double)(oa.nutSpalding/(Real)nu)*(double)WQ[q]);atomicAdd(wallStats+12,(double)(ss.nut/(Real)nu)*(double)WQ[q]);}
          else if(wallMode==OMEGA_WALL_COMPATIBLE){atomicAdd(wallStats+5,(double)ut*(double)WQ[q]);atomicAdd(wallStats+6,(double)ca.uTauVisc*(double)WQ[q]);atomicAdd(wallStats+7,(double)ca.uStarLog*(double)WQ[q]);atomicAdd(wallStats+8,(double)ca.uStar*(double)WQ[q]);atomicAdd(wallStats+9,(double)ss.pk*(double)WQ[q]);atomicAdd(wallStats+10,(double)ca.pkCompat*(double)WQ[q]);atomicAdd(wallStats+11,(double)ca.tauTarget*(double)WQ[q]);atomicAdd(wallStats+12,(double)(ss.nut/(Real)nu)*(double)WQ[q]);}
          else if(wallMode==OMEGA_WALL_CONSISTENT_TABLE&&equation==OMEGA_EQ&&ta.ok){const double kt=ta.kPlus*ta.uTau*ta.uTau,ot=ta.omegaPlus*ta.uTau*ta.uTau/nu;atomicAdd(wallStats+5,ta.uTau*(double)WQ[q]);atomicAdd(wallStats+6,ta.uPlus*(double)WQ[q]);atomicAdd(wallStats+7,ta.kPlus*(double)WQ[q]);atomicAdd(wallStats+8,ta.omegaPlus*(double)WQ[q]);atomicAdd(wallStats+9,kt*(double)WQ[q]);atomicAdd(wallStats+10,ot*(double)WQ[q]);atomicAdd(wallStats+11,(double)kv*(double)WQ[q]);atomicAdd(wallStats+12,(double)(ss.nut/(Real)nu)*(double)WQ[q]);}
        }
      }
    }
    if(!ok)atomicAdd(bad,1ULL);
  }
  __syncthreads();
  for(int p=(int)threadIdx.x;p<196;p+=(int)blockDim.x){int a=p/14,b=p%14;Real z=0;for(int q=0;q<16;++q){
    if(wallMode==OMEGA_WALL_SAMPLE_LOG||wallMode==OMEGA_WALL_OF_AUTO||wallMode==OMEGA_WALL_COMPATIBLE||wallMode==OMEGA_WALL_CONSISTENT_TABLE){
      z+=WQ[q]*(TAU[q]*PH[q][a]*PH[q][b]);
    }else{
      Real dna=GP[q][a][0]*(Real)f.n[0]+GP[q][a][1]*(Real)f.n[1]+GP[q][a][2]*(Real)f.n[2],dnb=GP[q][b][0]*(Real)f.n[0]+GP[q][b][1]*(Real)f.n[1]+GP[q][b][2]*(Real)f.n[2];
      z+=WQ[q]*(-NUE[q]*PH[q][a]*dnb-NUE[q]*dna*PH[q][b]+TAU[q]*PH[q][a]*PH[q][b]);
    }}atomicAdd(mat+slot[(std::size_t)f.cell*196ull+(std::size_t)p],z);}
  for(int a=(int)threadIdx.x;a<14;a+=(int)blockDim.x){Real z=0;for(int q=0;q<16;++q){
    if(wallMode==OMEGA_WALL_SAMPLE_LOG||wallMode==OMEGA_WALL_OF_AUTO||wallMode==OMEGA_WALL_COMPATIBLE||wallMode==OMEGA_WALL_CONSISTENT_TABLE){
      z+=WQ[q]*(TAU[q]*PH[q][a]*GVAL[q]);
    }else{
      Real dna=GP[q][a][0]*(Real)f.n[0]+GP[q][a][1]*(Real)f.n[1]+GP[q][a][2]*(Real)f.n[2];
      z+=WQ[q]*(-NUE[q]*dna*GVAL[q]+TAU[q]*PH[q][a]*GVAL[q]);
    }}atomicAdd(rhs+H.g[a],z);}
}

__global__ void sst_of_wall_production_hex(
    const hxt5b::HexFacePlan*F,int nf,const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,const nodals_hxt1::HexVel*hv,
    const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,Real*rhs,double radius,double nu,double sampleFraction,
    double kFloor,double omegaFloor,double ofKappa,double ofE,double ofBeta1,double ofReBlend,double ofProdLimit,double prodScale,double wallVolumeScale,int wallMode,int equation,int sharedSpaldingWallState,int profileIntegralUTau,int profileIntegralNQ,unsigned long long*bad)
{
  int fi=(int)blockIdx.x;if(fi>=nf)return;auto f=F[fi];auto H=hv[f.cell];
  __shared__ Real U0[14],U1[14],U2[14],K[14],W[14],PH[16][14],SRC[16],VW[16];
  if(threadIdx.x<14){int g=H.g[threadIdx.x];U0[threadIdx.x]=u0[g];U1[threadIdx.x]=u1[g];U2[threadIdx.x]=u2[g];K[threadIdx.x]=k[g];W[threadIdx.x]=omega[g];}__syncthreads();
  if(threadIdx.x<16){int q=(int)threadIdx.x,i=q>>2,j=q&3;SRC[q]=VW[q]=Real(0);for(int a=0;a<14;++a)PH[q][a]=Real(0);double cw[3]={0,0,0};cw[f.fd]=f.fv;cw[f.d0]=hxt5a::h5_hex_x[i];cw[f.d1]=hxt5a::h5_hex_x[j];double cs[3]={cw[0],cw[1],cw[2]};cs[f.fd]=f.fv+sampleFraction*(-2*f.fv);double phs[14],grs[14][3],gps[14][3],Is[9],dets;hxt2::hex_basis_ref(cs[0],cs[1],cs[2],phs,grs);bool ok=hxt2::hex_metric(pts,hc[f.cell],cs[0],cs[1],cs[2],Is,dets);double Xw[3],dXw[3][3],Xs[3],dXs[3][3];if(ok){hxt2::phys_grad14(grs,Is,gps);hxt5b::q1dev(pts,hc[f.cell],cw[0],cw[1],cw[2],Xw,dXw);hxt5b::q1dev(pts,hc[f.cell],cs[0],cs[1],cs[2],Xs,dXs);}double rr=ok?hypot(Xw[0],Xw[1]):0,jac=0,yd=0,ut=0,yp=0,betaWall=0,slip=0,nr0=0,nr1=0;if(ok&&rr>0){nr0=Xw[0]/rr;nr1=Xw[1]/rr;double nr[3]={nr0,nr1,0};double cr[3]={dXw[f.d0][1]*dXw[f.d1][2]-dXw[f.d0][2]*dXw[f.d1][1],dXw[f.d0][2]*dXw[f.d1][0]-dXw[f.d0][0]*dXw[f.d1][2],dXw[f.d0][0]*dXw[f.d1][1]-dXw[f.d0][1]*dXw[f.d1][0]};if(cr[0]*nr[0]+cr[1]*nr[1]+cr[2]*nr[2]<0){cr[0]*=-1;cr[1]*=-1;cr[2]*=-1;}jac=sqrt(cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]);yd=(Xw[0]-Xs[0])*nr[0]+(Xw[1]-Xs[1])*nr[1];ok=jac>0&&yd>0;}else ok=false;Real gu[9]={0},gk[3]={0},gw[3]={0},kv=0,wv=0,us0=0,us1=0,us2=0;if(ok){for(int a=0;a<14;++a){Real ps=(Real)phs[a];PH[q][a]=ps;kv+=K[a]*ps;wv+=W[a]*ps;us0+=U0[a]*ps;us1+=U1[a]*ps;us2+=U2[a]*ps;slip+=(double)U2[a]*phs[a];for(int d=0;d<3;++d){Real gd=(Real)gps[a][d];gk[d]+=K[a]*gd;gw[d]+=W[a]*gd;gu[d]+=U0[a]*gd;gu[3+d]+=U1[a]*gd;gu[6+d]+=U2[a]*gd;}}if(profileIntegralUTau){double ypProf=0.0,ubarProf=0.0,ybarProf=0.0,rmsProf=0.0,spanProf=0.0,bavgProf[14];if(!hxt5b::spalding_profile_hex(pts,hc[f.cell],f,cw,U2,nu,profileIntegralNQ,ut,ypProf,betaWall,ubarProf,ybarProf,bavgProf,rmsProf,spanProf))ok=false;else yp=yd*ut/nu;}else if(!hxt5b::spalding(slip,yd,nu,ut,yp,betaWall))ok=false;}if(ok){hex_sst_carrier((Real)cs[0],(Real)cs[1],(Real)cs[2],Is,K,W,(Real)kFloor,(Real)omegaFloor,kv,wv,gk,gw);LocalSST ss=local_sst((Real)Xs[0],(Real)Xs[1],(Real)radius,(Real)nu,gu,kv,wv,gk,gw,(Real)kFloor,(Real)omegaFloor);Real un=us0*(Real)nr0+us1*(Real)nr1;Real Ut=sqrt(fmax(Real(0),us0*us0+us1*us1+us2*us2-un*un));Real pkWall=ss.pk;if(wallMode==OMEGA_WALL_COMPATIBLE){CompatibleWallState ca=compatible_wall_state((Real)yd,(Real)nu,Ut,(Real)ut,kv,ss,(Real)kFloor,(Real)ofKappa);pkWall=ca.pkCompat;}else{OFWallState oa=of_auto_wall_state((Real)yd,(Real)nu,Ut,(Real)ut,kv,wv,ss,(Real)kFloor,(Real)omegaFloor,(Real)ofKappa,(Real)ofE,(Real)ofBeta1,(Real)ofReBlend,(Real)ofProdLimit,sharedSpaldingWallState);pkWall=oa.pkAuto;}Real dpk=(Real)prodScale*(pkWall-ss.pk);Real src=dpk;if(equation==OMEGA_EQ){Real nt=fmax(ss.nut,fmax(Real(1e-12)*(Real)nu,Real(1e-20)));src=ss.gamma*dpk/nt;}SRC[q]=src;Real h=(Real)(yd/fmax(sampleFraction,1e-6));VW[q]=(Real)(hxt5a::h5_hex_w[i]*hxt5a::h5_hex_w[j]*jac)*h*(Real)wallVolumeScale;}if(!ok)atomicAdd(bad,1ULL);}
  __syncthreads();for(int a=(int)threadIdx.x;a<14;a+=(int)blockDim.x){Real z=0;for(int q=0;q<16;++q)z+=VW[q]*SRC[q]*PH[q][a];atomicAdd(rhs+H.g[a],z);}
}


// ---------------------------------------------------------------------------
// Average-first OF-auto wall closure.
//
// NQ is used only as an accurate volume-integration rule through each
// wall-adjacent normal column.  The primitive/constitutive state is first
// reduced to one representative wall-model state:
//
//   Ubar, grad(U)bar, kbar, omegabar, grad(k)bar, grad(omega)bar, Xbar, ybar.
//
// kbar/omegabar and their gradients are averages of the ACTIVE SST carrier
// (Bernstein-Q2, Q1, or Q1+BF2-unlimited), not averages of raw modal
// coefficients.  One Spalding u_tau is obtained from (Ubar_z,ybar), and one
// OF-auto state is evaluated at the averaged state.  This avoids demanding
// that a Robin/slip-like FE wall trace itself lie on a pointwise wall-law
// profile and avoids integrating the singular omega_vis~1/y^2 target down to
// the physical wall.
//
// The omega condition constrains the SAME NQ cell-average functional via the
// rank-one penalty
//
//   lambda V (Bbar omega - omega_target)^2,
//
// while the k/omega production correction is treated as one averaged source
// distributed with the integrated FE basis.  wallSampleFraction is not used
// by the live closure.
// ---------------------------------------------------------------------------
__global__ void sst_of_profile_average_closure_hex(
    const hxt5b::HexFacePlan*F,int nf,
    const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,
    const nodals_hxt1::HexVel*hv,const std::int32_t*slot,
    const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,
    Real*mat,Real*rhs,
    double radius,double nu,double penaltyGamma,
    double kFloor,double omegaFloor,
    double ofKappa,double ofE,double ofBeta1,double ofReBlend,double ofProdLimit,
    double prodScale,double wallVolumeScale,
    int equation,int sharedSpaldingWallState,int profileIntegralNQ,
    int profileAverageWallLaw,const hxt5b::WallMatchPoint*matchPoints,
    int ofProduction,double*wallStats,unsigned long long*bad)
{
  constexpr int NS=16;
  const int fi=(int)blockIdx.x;
  if(fi>=nf)return;
  const auto f=F[fi];
  const auto H=hv[f.cell];

  __shared__ Real U0[14],U1[14],U2[14],K[14],W[14];
  __shared__ Real BAVG[NS][14],LAM[NS],TARGET[NS],SRC[NS],VCOL[NS];
  __shared__ double WSTAT[13][NS];

  if(threadIdx.x<14){
    const int g=H.g[threadIdx.x];
    U0[threadIdx.x]=u0[g];U1[threadIdx.x]=u1[g];U2[threadIdx.x]=u2[g];
    K[threadIdx.x]=k[g];W[threadIdx.x]=omega[g];
  }
  __syncthreads();

  if(threadIdx.x<NS){
    const int qs=(int)threadIdx.x,i=qs>>2,j=qs&3;
    for(int a=0;a<14;++a)BAVG[qs][a]=Real(0);
    LAM[qs]=TARGET[qs]=SRC[qs]=VCOL[qs]=Real(0);
    for(int r=0;r<13;++r)WSTAT[r][qs]=0.0;

    bool ok=(profileIntegralNQ==4||profileIntegralNQ==8||profileIntegralNQ==12);
    double cw[3]={0,0,0};
    cw[f.fd]=f.fv;cw[f.d0]=hxt5a::h5_hex_x[i];cw[f.d1]=hxt5a::h5_hex_x[j];

    double pw[14],gwWall[14][3];
    hxt2::hex_basis_ref(cw[0],cw[1],cw[2],pw,gwWall);
    double Xw[3]={0,0,0},dXw[3][3]={{0}},Xo[3]={0,0,0},dXo[3][3]={{0}};
    hxt5b::q1dev(pts,hc[f.cell],cw[0],cw[1],cw[2],Xw,dXw);
    double cr[3]={
      dXw[f.d0][1]*dXw[f.d1][2]-dXw[f.d0][2]*dXw[f.d1][1],
      dXw[f.d0][2]*dXw[f.d1][0]-dXw[f.d0][0]*dXw[f.d1][2],
      dXw[f.d0][0]*dXw[f.d1][1]-dXw[f.d0][1]*dXw[f.d1][0]
    };
    if(cr[0]*f.n[0]+cr[1]*f.n[1]+cr[2]*f.n[2]<0.0){
      cr[0]=-cr[0];cr[1]=-cr[1];cr[2]=-cr[2];
    }
    const double aj=sqrt(cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]);
    double n[3]={0,0,0},hn=0.0;
    if(ok)ok=aj>0.0&&isfinite(aj);
    if(ok){
      n[0]=cr[0]/aj;n[1]=cr[1]/aj;n[2]=cr[2]/aj;
      double co[3]={cw[0],cw[1],cw[2]};co[f.fd]=-f.fv;
      hxt5b::q1dev(pts,hc[f.cell],co[0],co[1],co[2],Xo,dXo);
      hn=(Xw[0]-Xo[0])*n[0]+(Xw[1]-Xo[1])*n[1]+(Xw[2]-Xo[2])*n[2];
      ok=hn>0.0&&isfinite(hn);
    }

    // Exact same average-first wall state used by the momentum traction.
    // profileAverageWallLaw: 0=Spalding, 1=pure high-Re log law.
    double ut=0.0,ypSp=0.0,beta=0.0,ubarZ=0.0,ybarSp=0.0,rms=0.0,span=0.0,bmom[14];
    if(ok){
      if(profileAverageWallLaw==5){
        // PROFILE5 owns the wall-stress inversion completely.
        double um=0.0,fitRms=0.0,fitSpan=0.0;
        ok=hxt5b::spalding_match_profile5(
          matchPoints,fi,qs,u2,nu,ut,ypSp,beta,um,fitRms,fitSpan);
        if(ok){
          ubarZ=um;
          ybarSp=matchPoints[((std::size_t)fi*16u+(std::size_t)qs)*5u+2u].y;
          rms=fitRms;span=fitSpan;
        }
      }else if(profileAverageWallLaw==4){
        // PROFILE3 owns the wall-stress inversion completely.
        // Do NOT gate it on a separate first-cell Spalding inversion.
        // The first-cell NQ12 carrier averages for k, omega, gradU, etc.
        // are assembled independently below and remain unchanged.
        double um=0.0,fitRms=0.0,fitSpan=0.0;
        ok=hxt5b::spalding_match_profile3(
          matchPoints,fi,qs,u2,nu,ut,ypSp,beta,um,fitRms,fitSpan);
        if(ok){
          ubarZ=um;
          ybarSp=matchPoints[((std::size_t)fi*16u+(std::size_t)qs)*3u+1u].y;
          rms=fitRms;span=fitSpan;
        }
      }else if(profileAverageWallLaw==3){
        // Single prescribed match-height mode still retains the legacy
        // first-cell average used by that experiment.
        double utLocal=0.0,ypLocal=0.0,betaLocal=0.0;
        ok=hxt5b::spalding_profile_average_hex(
          pts,hc[f.cell],f,cw,U2,nu,profileIntegralNQ,
          utLocal,ypLocal,betaLocal,ubarZ,ybarSp,bmom,rms,span);
        if(ok){
          double um=0.0;
          ok=hxt5b::spalding_match_height_point(
            matchPoints,fi,qs,u2,nu,ut,ypSp,beta,um);
          if(ok){
            ubarZ=um;
            ybarSp=matchPoints[(std::size_t)fi*16u+(std::size_t)qs].y;
          }
        }
      }else if(profileAverageWallLaw==2)
        ok=hxt5b::spalding_profile_average_relative_wall_hex(
          pts,hc[f.cell],f,cw,U2,nu,profileIntegralNQ,
          ut,ypSp,beta,ubarZ,ybarSp,bmom,rms,span);
      else if(profileAverageWallLaw==1)
        ok=hxt5b::loglaw_profile_average_hex(
          pts,hc[f.cell],f,cw,U2,nu,profileIntegralNQ,
          ut,ypSp,beta,ubarZ,ybarSp,bmom,rms,span);
      else
        ok=hxt5b::spalding_profile_average_hex(
          pts,hc[f.cell],f,cw,U2,nu,profileIntegralNQ,
          ut,ypSp,beta,ubarZ,ybarSp,bmom,rms,span);
    }

    double sw=0.0,sy=0.0,bint[14]={0};
    double xav[3]={0,0,0};
    Real uav[3]={0,0,0},guav[9]={0,0,0,0,0,0,0,0,0};
    Real kav=0,wav=0,gkav[3]={0,0,0},gwav[3]={0,0,0};

    for(int m=0;ok&&m<profileIntegralNQ;++m){
      double xi=0.0,wref=0.0;
      if(!hxt5b::spalding_profile_gauss_point(profileIntegralNQ,m,xi,wref)){ok=false;break;}
      double c[3]={cw[0],cw[1],cw[2]};c[f.fd]=-f.fv*xi;
      double ph[14],gr[14][3],gp[14][3],I[9],det=0.0;
      hxt2::hex_basis_ref(c[0],c[1],c[2],ph,gr);
      if(!hxt2::hex_metric(pts,hc[f.cell],c[0],c[1],c[2],I,det)){ok=false;break;}
      if(!(det>0.0)||!isfinite(det)){ok=false;break;}
      hxt2::phys_grad14(gr,I,gp);

      double X[3],dX[3][3];hxt5b::q1dev(pts,hc[f.cell],c[0],c[1],c[2],X,dX);
      const double yd=(Xw[0]-X[0])*n[0]+(Xw[1]-X[1])*n[1]+(Xw[2]-X[2])*n[2];
      const double ww=wref*det;
      if(!(yd>0.0)||!(ww>0.0)||!isfinite(yd)||!isfinite(ww)){ok=false;break;}

      Real uq[3]={0,0,0},guq[9]={0,0,0,0,0,0,0,0,0};
      for(int a=0;a<14;++a){
        const Real pa=(Real)ph[a];
        uq[0]+=U0[a]*pa;uq[1]+=U1[a]*pa;uq[2]+=U2[a]*pa;
        bint[a]+=ww*ph[a];
        for(int d=0;d<3;++d){
          const Real gd=(Real)gp[a][d];
          guq[d]+=U0[a]*gd;guq[3+d]+=U1[a]*gd;guq[6+d]+=U2[a]*gd;
        }
      }
      Real kq=0,wq=0,gkq[3]={0,0,0},gwq[3]={0,0,0};
      hex_sst_carrier((Real)c[0],(Real)c[1],(Real)c[2],I,K,W,
        (Real)kFloor,(Real)omegaFloor,kq,wq,gkq,gwq);

      sw+=ww;sy+=ww*yd;
      for(int d=0;d<3;++d){xav[d]+=ww*X[d];uav[d]+=(Real)ww*uq[d];gkav[d]+=(Real)ww*gkq[d];gwav[d]+=(Real)ww*gwq[d];}
      for(int d=0;d<9;++d)guav[d]+=(Real)ww*guq[d];
      kav+=(Real)ww*kq;wav+=(Real)ww*wq;
    }

    if(ok)ok=sw>0.0&&isfinite(sw);
    if(ok){
      const Real inv=(Real)(1.0/sw);
      for(int a=0;a<14;++a)BAVG[qs][a]=(Real)(bint[a]/sw);
      for(int d=0;d<3;++d){xav[d]/=sw;uav[d]*=inv;gkav[d]*=inv;gwav[d]*=inv;}
      for(int d=0;d<9;++d)guav[d]*=inv;
      kav*=inv;wav*=inv;
      const double ybar=sy/sw;
      ok=ybar>0.0&&isfinite(ybar)&&isfinite(ut);

      if(ok){
        const LocalSST ss=local_sst((Real)xav[0],(Real)xav[1],(Real)radius,(Real)nu,
          guav,kav,wav,gkav,gwav,(Real)kFloor,(Real)omegaFloor);
        const Real un=uav[0]*(Real)n[0]+uav[1]*(Real)n[1]+uav[2]*(Real)n[2];
        const Real Ut=sqrt(fmax(Real(0),uav[0]*uav[0]+uav[1]*uav[1]+uav[2]*uav[2]-un*un));
        const OFWallState oa=of_auto_wall_state(
          (Real)ybar,(Real)nu,Ut,(Real)ut,kav,wav,ss,
          (Real)kFloor,(Real)omegaFloor,(Real)ofKappa,(Real)ofE,
          (Real)ofBeta1,(Real)ofReBlend,(Real)ofProdLimit,
          sharedSpaldingWallState);

        const double V=hxt5a::h5_hex_w[i]*hxt5a::h5_hex_w[j]*sw;
        const Real nue=(Real)nu+ss.sigmaW*ss.nut;
        const Real lambda=(Real)penaltyGamma*nue/(Real)(hn*hn);
        VCOL[qs]=(Real)V;LAM[qs]=lambda;TARGET[qs]=oa.omegaTarget;

        if(ofProduction){
          const Real dpk=(Real)prodScale*(oa.pkAuto-ss.pk);
          Real src=dpk;
          if(equation==OMEGA_EQ){
            const Real nt=fmax(ss.nut,fmax(Real(1e-12)*(Real)nu,Real(1e-20)));
            src=ss.gamma*dpk/nt;
          }
          SRC[qs]=src;
        }

        const double yp=ybar*ut/nu;
        double wtrace=0.0;for(int a=0;a<14;++a)wtrace+=(double)W[a]*pw[a];
        WSTAT[0][qs]=V;
        WSTAT[1][qs]=(double)oa.omegaTarget*V;
        WSTAT[2][qs]=yp*V;
        WSTAT[3][qs]=(double)wav*V;
        WSTAT[4][qs]=wtrace*V;
        WSTAT[5][qs]=(double)oa.rey*V;
        WSTAT[6][qs]=(double)oa.lamFrac*V;
        WSTAT[7][qs]=(double)oa.omegaVis*V;
        WSTAT[8][qs]=(double)oa.omegaLog*V;
        WSTAT[9][qs]=(double)ss.pk*V;
        WSTAT[10][qs]=(double)oa.pkAuto*V;
        WSTAT[11][qs]=(double)(oa.nutSpalding/(Real)nu)*V;
        WSTAT[12][qs]=(double)(ss.nut/(Real)nu)*V;
      }
    }
    if(!ok)atomicAdd(bad,1ULL);
  }
  __syncthreads();

  // Rank-one penalty on the NQ volume-average omega functional.
  if(equation==OMEGA_EQ){
    for(int p=(int)threadIdx.x;p<196;p+=(int)blockDim.x){
      const int a=p/14,b=p%14;Real z=0;
      for(int qs=0;qs<NS;++qs)
        z+=VCOL[qs]*LAM[qs]*BAVG[qs][a]*BAVG[qs][b];
      atomicAdd(mat+slot[(std::size_t)f.cell*196ull+(std::size_t)p],z);
    }
  }

  for(int a=(int)threadIdx.x;a<14;a+=(int)blockDim.x){
    Real z=0;
    for(int qs=0;qs<NS;++qs){
      z+=VCOL[qs]*(Real)wallVolumeScale*SRC[qs]*BAVG[qs][a];
      if(equation==OMEGA_EQ)
        z+=VCOL[qs]*LAM[qs]*BAVG[qs][a]*TARGET[qs];
    }
    atomicAdd(rhs+H.g[a],z);
  }

  if(wallStats){
    for(int r=(int)threadIdx.x;r<13;r+=(int)blockDim.x){
      double z=0.0;for(int qs=0;qs<NS;++qs)z+=WSTAT[r][qs];
      atomicAdd(wallStats+r,z);
    }
  }
}

// ---------------------------------------------------------------------------
// Average-first log-law omega closures (sample_log / trace_log).
//
// Both variants obtain ONE Spalding wall state from the NQ physical-volume
// averages (Ubar,ybar).  The log omega target is therefore
//
//   omega_log = u_tau/(sqrt(beta*)*kappa*ybar).
//
// sample_log: constrain the NQ cell-average omega functional with the same
// rank-one volume penalty used by the stable average-first OF-auto closure.
//
// trace_log: use the SAME average-derived target, but impose it on the physical
// wall trace with the historical symmetric Nitsche form.  This isolates target
// placement while keeping the representative wall state identical.
// ---------------------------------------------------------------------------
__global__ void sst_log_profile_average_closure_hex(
    const hxt5b::HexFacePlan*F,int nf,
    const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,
    const nodals_hxt1::HexVel*hv,const std::int32_t*slot,
    const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,
    Real*mat,Real*rhs,
    double radius,double nu,double penaltyGamma,
    double kFloor,double omegaFloor,
    int wallMode,int profileIntegralNQ,
    double*wallStats,unsigned long long*bad)
{
  constexpr int NS=16;
  const int fi=(int)blockIdx.x;
  if(fi>=nf)return;
  const auto f=F[fi];
  const auto H=hv[f.cell];

  __shared__ Real U0[14],U1[14],U2[14],K[14],W[14];
  __shared__ Real BAVG[NS][14],PHW[NS][14],DNW[NS][14];
  __shared__ Real NUE[NS],TAU[NS],TARGET[NS],VCOL[NS],WFACE[NS];
  __shared__ double WSTAT[5][NS];

  if(threadIdx.x<14){
    const int g=H.g[threadIdx.x];
    U0[threadIdx.x]=u0[g];U1[threadIdx.x]=u1[g];U2[threadIdx.x]=u2[g];
    K[threadIdx.x]=k[g];W[threadIdx.x]=omega[g];
  }
  __syncthreads();

  if(threadIdx.x<NS){
    const int qs=(int)threadIdx.x,i=qs>>2,j=qs&3;
    for(int a=0;a<14;++a){BAVG[qs][a]=PHW[qs][a]=DNW[qs][a]=Real(0);}
    NUE[qs]=TAU[qs]=TARGET[qs]=VCOL[qs]=WFACE[qs]=Real(0);
    for(int r=0;r<5;++r)WSTAT[r][qs]=0.0;

    bool ok=(profileIntegralNQ==4||profileIntegralNQ==8||profileIntegralNQ==12) &&
            (wallMode==OMEGA_WALL_SAMPLE_LOG||wallMode==OMEGA_WALL_TRACE_LOG);
    double cw[3]={0,0,0};
    cw[f.fd]=f.fv;cw[f.d0]=hxt5a::h5_hex_x[i];cw[f.d1]=hxt5a::h5_hex_x[j];

    double pw[14],grw[14][3],gpw[14][3],Iw[9],detw=0.0;
    hxt2::hex_basis_ref(cw[0],cw[1],cw[2],pw,grw);
    if(ok)ok=hxt2::hex_metric(pts,hc[f.cell],cw[0],cw[1],cw[2],Iw,detw);
    if(ok)hxt2::phys_grad14(grw,Iw,gpw);

    double Xw[3]={0,0,0},dXw[3][3]={{0}},Xo[3]={0,0,0},dXo[3][3]={{0}};
    hxt5b::q1dev(pts,hc[f.cell],cw[0],cw[1],cw[2],Xw,dXw);
    double cr[3]={
      dXw[f.d0][1]*dXw[f.d1][2]-dXw[f.d0][2]*dXw[f.d1][1],
      dXw[f.d0][2]*dXw[f.d1][0]-dXw[f.d0][0]*dXw[f.d1][2],
      dXw[f.d0][0]*dXw[f.d1][1]-dXw[f.d0][1]*dXw[f.d1][0]
    };
    if(cr[0]*f.n[0]+cr[1]*f.n[1]+cr[2]*f.n[2]<0.0){cr[0]=-cr[0];cr[1]=-cr[1];cr[2]=-cr[2];}
    const double aj=sqrt(cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]);
    double n[3]={0,0,0},hn=0.0;
    if(ok)ok=aj>0.0&&isfinite(aj);
    if(ok){
      n[0]=cr[0]/aj;n[1]=cr[1]/aj;n[2]=cr[2]/aj;
      double co[3]={cw[0],cw[1],cw[2]};co[f.fd]=-f.fv;
      hxt5b::q1dev(pts,hc[f.cell],co[0],co[1],co[2],Xo,dXo);
      hn=(Xw[0]-Xo[0])*n[0]+(Xw[1]-Xo[1])*n[1]+(Xw[2]-Xo[2])*n[2];
      ok=hn>0.0&&isfinite(hn);
    }

    // One representative Spalding state, identical to average-first momentum.
    double ut=0.0,ypSp=0.0,beta=0.0,ubarZ=0.0,ybarSp=0.0,rms=0.0,span=0.0,bmom[14];
    if(ok)ok=hxt5b::spalding_profile_average_hex(
      pts,hc[f.cell],f,cw,U2,nu,profileIntegralNQ,
      ut,ypSp,beta,ubarZ,ybarSp,bmom,rms,span);

    double sw=0.0,sy=0.0,bint[14]={0};
    double xav[3]={0,0,0};
    Real uav[3]={0,0,0},guav[9]={0,0,0,0,0,0,0,0,0};
    Real kav=0,wav=0,gkav[3]={0,0,0},gwav[3]={0,0,0};

    for(int m=0;ok&&m<profileIntegralNQ;++m){
      double xi=0.0,wref=0.0;
      if(!hxt5b::spalding_profile_gauss_point(profileIntegralNQ,m,xi,wref)){ok=false;break;}
      double c[3]={cw[0],cw[1],cw[2]};c[f.fd]=-f.fv*xi;
      double ph[14],gr[14][3],gp[14][3],I[9],det=0.0;
      hxt2::hex_basis_ref(c[0],c[1],c[2],ph,gr);
      if(!hxt2::hex_metric(pts,hc[f.cell],c[0],c[1],c[2],I,det)){ok=false;break;}
      if(!(det>0.0)||!isfinite(det)){ok=false;break;}
      hxt2::phys_grad14(gr,I,gp);

      double X[3],dX[3][3];hxt5b::q1dev(pts,hc[f.cell],c[0],c[1],c[2],X,dX);
      const double yd=(Xw[0]-X[0])*n[0]+(Xw[1]-X[1])*n[1]+(Xw[2]-X[2])*n[2];
      const double ww=wref*det;
      if(!(yd>0.0)||!(ww>0.0)||!isfinite(yd)||!isfinite(ww)){ok=false;break;}

      Real uq[3]={0,0,0},guq[9]={0,0,0,0,0,0,0,0,0};
      for(int a=0;a<14;++a){
        const Real pa=(Real)ph[a];
        uq[0]+=U0[a]*pa;uq[1]+=U1[a]*pa;uq[2]+=U2[a]*pa;
        bint[a]+=ww*ph[a];
        for(int d=0;d<3;++d){
          const Real gd=(Real)gp[a][d];
          guq[d]+=U0[a]*gd;guq[3+d]+=U1[a]*gd;guq[6+d]+=U2[a]*gd;
        }
      }
      Real kq=0,wq=0,gkq[3]={0,0,0},gwq[3]={0,0,0};
      hex_sst_carrier((Real)c[0],(Real)c[1],(Real)c[2],I,K,W,
        (Real)kFloor,(Real)omegaFloor,kq,wq,gkq,gwq);

      sw+=ww;sy+=ww*yd;
      for(int d=0;d<3;++d){xav[d]+=ww*X[d];uav[d]+=(Real)ww*uq[d];gkav[d]+=(Real)ww*gkq[d];gwav[d]+=(Real)ww*gwq[d];}
      for(int d=0;d<9;++d)guav[d]+=(Real)ww*guq[d];
      kav+=(Real)ww*kq;wav+=(Real)ww*wq;
    }

    if(ok)ok=sw>0.0&&isfinite(sw);
    if(ok){
      const Real inv=(Real)(1.0/sw);
      for(int a=0;a<14;++a)BAVG[qs][a]=(Real)(bint[a]/sw);
      for(int d=0;d<3;++d){xav[d]/=sw;uav[d]*=inv;gkav[d]*=inv;gwav[d]*=inv;}
      for(int d=0;d<9;++d)guav[d]*=inv;
      kav*=inv;wav*=inv;
      const double ybar=sy/sw;
      ok=ybar>0.0&&isfinite(ybar)&&ut>0.0&&isfinite(ut);

      if(ok){
        const LocalSST ss=local_sst((Real)xav[0],(Real)xav[1],(Real)radius,(Real)nu,
          guav,kav,wav,gkav,gwav,(Real)kFloor,(Real)omegaFloor);
        const Real nue=(Real)nu+ss.sigmaW*ss.nut;
        const Real target=(Real)(ut/(sqrt(BETA_STAR)*KAPPA_OMEGA_WALL*ybar));
        const double V=hxt5a::h5_hex_w[i]*hxt5a::h5_hex_w[j]*sw;
        const double wf=hxt5a::h5_hex_w[i]*hxt5a::h5_hex_w[j]*aj;

        NUE[qs]=nue;TARGET[qs]=target;VCOL[qs]=(Real)V;WFACE[qs]=(Real)wf;
        if(wallMode==OMEGA_WALL_SAMPLE_LOG){
          TAU[qs]=(Real)penaltyGamma*nue/(Real)(hn*hn);
        }else{
          // Preserve the historical trace-log Nitsche scaling, but use the
          // representative average distance rather than an arbitrary sample.
          TAU[qs]=(Real)(penaltyGamma/ybar)*nue;
          for(int a=0;a<14;++a){
            PHW[qs][a]=(Real)pw[a];
            DNW[qs][a]=(Real)(gpw[a][0]*n[0]+gpw[a][1]*n[1]+gpw[a][2]*n[2]);
          }
        }

        double wtrace=0.0;for(int a=0;a<14;++a)wtrace+=(double)W[a]*pw[a];
        const double yp=ybar*ut/nu;
        WSTAT[0][qs]=V;
        WSTAT[1][qs]=(double)target*V;
        WSTAT[2][qs]=yp*V;
        WSTAT[3][qs]=(double)wav*V;
        WSTAT[4][qs]=wtrace*V;
      }
    }
    if(!ok)atomicAdd(bad,1ULL);
  }
  __syncthreads();

  if(wallMode==OMEGA_WALL_SAMPLE_LOG){
    for(int p=(int)threadIdx.x;p<196;p+=(int)blockDim.x){
      const int a=p/14,b=p%14;Real z=0;
      for(int qs=0;qs<NS;++qs)z+=VCOL[qs]*TAU[qs]*BAVG[qs][a]*BAVG[qs][b];
      atomicAdd(mat+slot[(std::size_t)f.cell*196ull+(std::size_t)p],z);
    }
    for(int a=(int)threadIdx.x;a<14;a+=(int)blockDim.x){
      Real z=0;for(int qs=0;qs<NS;++qs)z+=VCOL[qs]*TAU[qs]*BAVG[qs][a]*TARGET[qs];
      atomicAdd(rhs+H.g[a],z);
    }
  }else{
    for(int p=(int)threadIdx.x;p<196;p+=(int)blockDim.x){
      const int a=p/14,b=p%14;Real z=0;
      for(int qs=0;qs<NS;++qs)
        z+=WFACE[qs]*(-NUE[qs]*PHW[qs][a]*DNW[qs][b]-NUE[qs]*DNW[qs][a]*PHW[qs][b]+TAU[qs]*PHW[qs][a]*PHW[qs][b]);
      atomicAdd(mat+slot[(std::size_t)f.cell*196ull+(std::size_t)p],z);
    }
    for(int a=(int)threadIdx.x;a<14;a+=(int)blockDim.x){
      Real z=0;
      for(int qs=0;qs<NS;++qs)z+=WFACE[qs]*(-NUE[qs]*DNW[qs][a]*TARGET[qs]+TAU[qs]*PHW[qs][a]*TARGET[qs]);
      atomicAdd(rhs+H.g[a],z);
    }
  }

  if(wallStats){
    for(int r=(int)threadIdx.x;r<5;r+=(int)blockDim.x){
      double z=0.0;for(int qs=0;qs<NS;++qs)z+=WSTAT[r][qs];
      atomicAdd(wallStats+r,z);
    }
  }
}


// ---------------------------------------------------------------------------
// PROFILE5 MULTI-POINT LOCAL-EQUILIBRIUM SST WALL MODEL
//
// General wall-model statement at each wall surface quadrature point q:
//   1) use the existing five physical wall-normal velocity samples;
//   2) fit ONE Spalding friction velocity u_tau (unchanged momentum model);
//   3) at those SAME five physical locations form y+ = y*u_tau/nu;
//   4) evaluate the universal 1-D SST-2003m local-equilibrium profile
//      [k+(y+), omega+(y+), nut+(y+)] from lookup_yplus();
//   5) weakly fit the transported k and omega FE fields to those five targets
//      by a weighted least-squares/rank-one point functional.
//
// No pipe radius, Moody friction factor, analytic pipe solution, or table
// velocity inversion enters this closure.  The table supplies only the local
// equilibrium SST profile once u_tau has been obtained from the numerical
// velocity field.  Native SST volume production/diffusion remain active;
// OF-auto's extra wall production is intentionally disabled for this mode.
//
// Functional for phi in {k,omega}:
//   J_phi = 1/2 sum_{q,m} C_qm [ phi_h(x_qm) - phi_eq(y_qm;u_tau_q) ]^2
//
// where C_qm uses wall-area quadrature, a trapezoidal normal-profile weight,
// and a diffusion-scaled penalty.  Every sample acts only on the containing
// HEX's 14 scalar DOFs, so matrix sparsity remains the ordinary element-local
// scalar sparsity even when a sample lies beyond the first wall cell.
// ---------------------------------------------------------------------------
__global__ void sst_profile5_equilibrium_closure_hex(
    const hxt5b::HexFacePlan*F,int nf,
    const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,
    const nodals_hxt1::HexVel*hv,const std::int32_t*slot,
    const Real*u2,const Real*k,const Real*omega,
    Real*mat,Real*rhs,double nu,double profileGamma,int equation,
    const hxt5b::WallMatchPoint*matchPoints,
    double*wallStats,unsigned long long*bad)
{
  constexpr int NQ=16,NM=5,NS=NQ*NM;
  const int fi=(int)blockIdx.x;
  if(fi>=nf)return;
  const auto f=F[fi];

  __shared__ double UTAU[NQ],WFACE[NQ];
  __shared__ int QOK[NQ];
  __shared__ Real TARGET[NS],COEFF[NS],VOLW[NS],YPLUS[NS];
  __shared__ int CELL[NS];

  if(threadIdx.x<NQ){
    const int q=(int)threadIdx.x,i=q>>2,j=q&3;
    QOK[q]=0;UTAU[q]=0.0;WFACE[q]=0.0;

    double cw[3]={0,0,0};
    cw[f.fd]=f.fv;
    cw[f.d0]=hxt5a::h5_hex_x[i];
    cw[f.d1]=hxt5a::h5_hex_x[j];

    double Xw[3]={0,0,0},dXw[3][3]={{0}};
    hxt5b::q1dev(pts,hc[f.cell],cw[0],cw[1],cw[2],Xw,dXw);
    double cr[3]={
      dXw[f.d0][1]*dXw[f.d1][2]-dXw[f.d0][2]*dXw[f.d1][1],
      dXw[f.d0][2]*dXw[f.d1][0]-dXw[f.d0][0]*dXw[f.d1][2],
      dXw[f.d0][0]*dXw[f.d1][1]-dXw[f.d0][1]*dXw[f.d1][0]
    };
    if(cr[0]*f.n[0]+cr[1]*f.n[1]+cr[2]*f.n[2]<0.0){cr[0]=-cr[0];cr[1]=-cr[1];cr[2]=-cr[2];}
    const double aj=sqrt(cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]);

    bool ok=aj>0.0&&isfinite(aj)&&matchPoints!=nullptr&&nu>0.0&&profileGamma>0.0;
    double ut=0.0,ypMid=0.0,beta=0.0,uMid=0.0,rms=0.0,span=0.0;
    if(ok)ok=hxt5b::spalding_match_profile5(
      matchPoints,fi,q,u2,nu,ut,ypMid,beta,uMid,rms,span);
    if(ok)ok=ut>=0.0&&isfinite(ut);

    if(ok){
      UTAU[q]=ut;
      WFACE[q]=hxt5a::h5_hex_w[i]*hxt5a::h5_hex_w[j]*aj;
      QOK[q]=1;
    }else{
      atomicAdd(bad,1ULL);
    }
  }
  __syncthreads();

  for(int sidx=(int)threadIdx.x;sidx<NS;sidx+=(int)blockDim.x){
    TARGET[sidx]=COEFF[sidx]=VOLW[sidx]=YPLUS[sidx]=Real(0);
    CELL[sidx]=-1;

    const int q=sidx/NM,m=sidx%NM;
    if(!QOK[q])continue;

    const std::size_t base=((std::size_t)fi*16u+(std::size_t)q)*5u;
    const auto &pm=matchPoints[base+(std::size_t)m];
    if(pm.cell<0||!(pm.y>0.0)||!isfinite(pm.y)){
      atomicAdd(bad,1ULL);continue;
    }

    double yy[NM];
    #pragma unroll
    for(int r=0;r<NM;++r)yy[r]=matchPoints[base+(std::size_t)r].y;
    bool yok=true;
    #pragma unroll
    for(int r=1;r<NM;++r)yok=yok&&(yy[r]>yy[r-1])&&isfinite(yy[r]);
    if(!yok){atomicAdd(bad,1ULL);continue;}

    const double ut=UTAU[q];
    const double yp=pm.y*ut/nu;
    double target=0.0,nutPlus=0.0;
    if(yp>1.0e-12){
      const hxt5c_table::WallTableState ta=hxt5c_table::lookup_yplus(yp);
      if(!ta.ok){atomicAdd(bad,1ULL);continue;}
      target=(equation==K_EQ)
        ? ta.kPlus*ut*ut
        : ta.omegaPlus*ut*ut/nu;
      nutPlus=ta.nutPlus;
    }else{
      // Correct u_tau -> 0 limit of the same SST wall profile:
      // k -> 0 and omega+ ~ 6/(beta1*y+^2), hence
      // omega -> 6*nu/(beta1*y^2), finite even at stagnation.
      target=(equation==K_EQ)?0.0:6.0*nu/(0.075*pm.y*pm.y);
      nutPlus=0.0;
    }

    // Five-point trapezoidal profile weight over [y1,y5].
    double wy=0.0,ell=0.0;
    if(m==0){wy=0.5*(yy[1]-yy[0]);ell=yy[1]-yy[0];}
    else if(m==NM-1){wy=0.5*(yy[NM-1]-yy[NM-2]);ell=yy[NM-1]-yy[NM-2];}
    else{wy=0.5*(yy[m+1]-yy[m-1]);ell=wy;}

    // Penalty scaling only: use the equilibrium-profile total diffusivity
    // scale nu*(1+nut+) and divide gamma over five points so the aggregate
    // strength remains comparable to a single wall closure.
    const double nue=nu*(1.0+fmax(0.0,nutPlus));
    const double vw=WFACE[q]*wy;
    const double lam=(profileGamma/5.0)*nue/(ell*ell);
    if(!(target>=0.0)||!isfinite(target)||!(vw>0.0)||!(lam>0.0)||!isfinite(lam)){
      atomicAdd(bad,1ULL);continue;
    }

    TARGET[sidx]=(Real)target;
    COEFF[sidx]=(Real)(vw*lam);
    VOLW[sidx]=(Real)vw;
    YPLUS[sidx]=(Real)yp;
    CELL[sidx]=pm.cell;
  }
  __syncthreads();

  // Rank-one least-squares Hessian on each sample's containing HEX.
  for(int z=(int)threadIdx.x;z<NS*196;z+=(int)blockDim.x){
    const int sidx=z/196,p=z%196;
    const int c=CELL[sidx];
    if(c<0)continue;
    const int a=p/14,b=p%14;
    const int q=sidx/NM,m=sidx%NM;
    const auto &pm=matchPoints[((std::size_t)fi*16u+(std::size_t)q)*5u+(std::size_t)m];
    const Real v=COEFF[sidx]*(Real)pm.b[a]*(Real)pm.b[b];
    atomicAdd(mat+slot[(std::size_t)c*196ull+(std::size_t)p],v);
  }

  // Rank-one least-squares target RHS.
  for(int z=(int)threadIdx.x;z<NS*14;z+=(int)blockDim.x){
    const int sidx=z/14,a=z%14;
    const int c=CELL[sidx];
    if(c<0)continue;
    const int q=sidx/NM,m=sidx%NM;
    const auto &pm=matchPoints[((std::size_t)fi*16u+(std::size_t)q)*5u+(std::size_t)m];
    atomicAdd(rhs+pm.g[a],COEFF[sidx]*(Real)pm.b[a]*TARGET[sidx]);
  }

  // Diagnostics use the physical profile volume weights, not penalty weights.
  if(wallStats){
    for(int sidx=(int)threadIdx.x;sidx<NS;sidx+=(int)blockDim.x){
      if(CELL[sidx]<0)continue;
      const int q=sidx/NM,m=sidx%NM;
      const auto &pm=matchPoints[((std::size_t)fi*16u+(std::size_t)q)*5u+(std::size_t)m];
      double cur=0.0;
      #pragma unroll
      for(int a=0;a<14;++a){
        const int g=pm.g[a];
        if(g>=0)cur+=(double)(equation==K_EQ?k[g]:omega[g])*pm.b[a];
      }
      const double vw=(double)VOLW[sidx];
      atomicAdd(wallStats+0,vw);
      atomicAdd(wallStats+1,(double)TARGET[sidx]*vw);
      atomicAdd(wallStats+2,(double)YPLUS[sidx]*vw);
      atomicAdd(wallStats+3,cur*vw);
      if(equation==OMEGA_EQ)atomicAdd(wallStats+5,UTAU[q]*vw);
    }
  }
}

#include "modules/wall/of_auto_point3_kernel.inc"
#if defined(NODALS_P2_COMPILE_SMOKE)
#include "modules/wall/bernstein_p2_of_auto_wall.inc"
#endif // PURETET_S1_S5_OFAUTO_V1

// Exact five-point equilibrium target and trapezoidal penalty, tetra carrier.
__global__ void sst_profile5_equilibrium_closure_tet(
    const hxt5b::TetFacePlan*F,int nf,
    const nodals_hxt1::Point*pts,const nodals_hxt1::TetConn*tc,
    const nodals_hxt1::TetVel*tv,const std::int32_t*slot,
    const Real*u2,const Real*k,const Real*omega,
    Real*mat,Real*rhs,double nu,double profileGamma,int equation,int pointMask,
    const hxt5b::TetWallPackedPoint*matchPoints,
    double*wallStats,unsigned long long*bad)
{
  constexpr int NQ=12,NM=5,NS=NQ*NM;
  const int fi=(int)blockIdx.x;
  if(fi>=nf)return;
  const auto f=F[fi];

  __shared__ double UTAU[NQ],WFACE[NQ];
  __shared__ int QOK[NQ];
  __shared__ Real TARGET[NS],COEFF[NS],BASE_COEFF[NS],VOLW[NS],YPLUS[NS];
  __shared__ int CELL[NS];

  if(threadIdx.x<NQ){
    const int q=(int)threadIdx.x;
    QOK[q]=0;UTAU[q]=0.0;WFACE[q]=0.0;
    bool ok=f.area>0.0&&isfinite(f.area)&&matchPoints!=nullptr&&nu>0.0&&profileGamma>0.0;
    double ut=0.0,ypMid=0.0,beta=0.0,uMid=0.0,rms=0.0,span=0.0;
    if(ok)ok=hxt5b::spalding_match_profile5_tet(
      matchPoints,fi,q,u2,nu,ut,ypMid,beta,uMid,rms,span);
    if(ok)ok=ut>=0.0&&isfinite(ut);

    if(ok){
      UTAU[q]=ut;
      WFACE[q]=hxt5b::QW12[q]*f.area;
      QOK[q]=1;
    }else{
      atomicAdd(bad,1ULL);
    }
  }
  __syncthreads();

  for(int sidx=(int)threadIdx.x;sidx<NS;sidx+=(int)blockDim.x){
    TARGET[sidx]=COEFF[sidx]=BASE_COEFF[sidx]=VOLW[sidx]=YPLUS[sidx]=Real(0);
    CELL[sidx]=-1;

    const int q=sidx/NM,m=sidx%NM;
    if(!QOK[q])continue;

    const std::size_t base=((std::size_t)fi*12u+(std::size_t)q)*5u;
    const auto &pm=hxt5b::stage27_s02b_at(matchPoints,fi,q,m);
    if(pm.cell<0||!(pm.y>0.0)||!isfinite(pm.y)){
      atomicAdd(bad,1ULL);continue;
    }

    double yy[NM];
    #pragma unroll
    for(int r=0;r<NM;++r)yy[r]=hxt5b::stage27_s02b_y(r);
    bool yok=true;
    #pragma unroll
    for(int r=1;r<NM;++r)yok=yok&&(yy[r]>yy[r-1])&&isfinite(yy[r]);
    if(!yok){atomicAdd(bad,1ULL);continue;}

    const double ut=UTAU[q];
    const double yp=pm.y*ut/nu;
    double target=0.0,nutPlus=0.0;
    if(yp>1.0e-12){
      const hxt5c_table::WallTableState ta=hxt5c_table::lookup_yplus(yp);
      if(!ta.ok){atomicAdd(bad,1ULL);continue;}
      target=(equation==K_EQ)
        ? ta.kPlus*ut*ut
        : ta.omegaPlus*ut*ut/nu;
      nutPlus=ta.nutPlus;
    }else{
      // Correct u_tau -> 0 limit of the same SST wall profile:
      // k -> 0 and omega+ ~ 6/(beta1*y+^2), hence
      // omega -> 6*nu/(beta1*y^2), finite even at stagnation.
      target=(equation==K_EQ)?0.0:6.0*nu/(0.075*pm.y*pm.y);
      nutPlus=0.0;
    }

    // Five-point trapezoidal profile weight over [y1,y5].
    double wy=0.0,ell=0.0;
    if(m==0){wy=0.5*(yy[1]-yy[0]);ell=yy[1]-yy[0];}
    else if(m==NM-1){wy=0.5*(yy[NM-1]-yy[NM-2]);ell=yy[NM-1]-yy[NM-2];}
    else{wy=0.5*(yy[m+1]-yy[m-1]);ell=wy;}

    // Penalty scaling only: use the equilibrium-profile total diffusivity
    // scale nu*(1+nut+) and divide gamma over five points so the aggregate
    // strength remains comparable to a single wall closure.
    const double nue=nu*(1.0+fmax(0.0,nutPlus));
    const double vw=WFACE[q]*wy;
    const double lam=(profileGamma/5.0)*nue/(ell*ell);
    if(!(target>=0.0)||!isfinite(target)||!(vw>0.0)||!(lam>0.0)||!isfinite(lam)){
      atomicAdd(bad,1ULL);continue;
    }

    TARGET[sidx]=(Real)target;
    COEFF[sidx]=(Real)(vw*lam);
    BASE_COEFF[sidx]=COEFF[sidx];
    VOLW[sidx]=(Real)vw;
    YPLUS[sidx]=(Real)yp;
    CELL[sidx]=pm.cell;
  }
  __syncthreads();

  // PURETET_AB5_MODE_MASK_V1: A=central sample #3 only; B=#1,#3,#5.
  // Normalize each wall quadrature point by the *actual* sum of EQ5 C_m
  // coefficients (not just sample count) to preserve aggregate strength.
  // Keep unselected sample TARGET/CELL/BASE_COEFF for ALL-FIVE diagnostics.
  if(threadIdx.x<NQ){
    const int q=(int)threadIdx.x;
    if(QOK[q]){
      double sumAll=0.0,sumSelected=0.0;
      for(int m=0;m<NM;++m){
        const double weight=(double)BASE_COEFF[q*NM+m];
        sumAll+=weight;
        if((pointMask&(1<<m))!=0)sumSelected+=weight;
      }
      if(!(sumAll>0.0 && sumSelected>0.0) || !isfinite(sumAll) || !isfinite(sumSelected)){
        atomicAdd(bad,1ULL);
      }else{
        const double scale=sumAll/sumSelected;
        for(int m=0;m<NM;++m){
          const int s=q*NM+m;
          COEFF[s]=((pointMask&(1<<m))!=0)
            ?(Real)((double)BASE_COEFF[s]*scale):Real(0);
        }
        if(wallStats && equation==OMEGA_EQ){
          atomicAdd(wallStats+7,sumAll);
          atomicAdd(wallStats+8,sumSelected*scale);
          atomicAdd(wallStats+9,(double)__popc((unsigned int)pointMask));
          atomicAdd(wallStats+10,sumSelected);
          atomicAdd(wallStats+12,(double)NM);
        }
      }
    }
  }
  __syncthreads();

  // Rank-one least-squares Hessian on each sample's containing HEX.
  for(int z=(int)threadIdx.x;z<NS*64;z+=(int)blockDim.x){
    const int sidx=z/64,p=z%64;
    const int c=CELL[sidx];
    if(c<0)continue;
    const int a=p/8,b=p%8;
    const int q=sidx/NM,m=sidx%NM;
    const auto &pm=hxt5b::stage27_s02b_at(matchPoints,fi,q,m);
    const Real v=COEFF[sidx]*(Real)pm.b[a]*(Real)pm.b[b];
    atomicAdd(mat+slot[(std::size_t)c*64ull+(std::size_t)p],v);
  }

  // Rank-one least-squares target RHS.
  for(int z=(int)threadIdx.x;z<NS*8;z+=(int)blockDim.x){
    const int sidx=z/8,a=z%8;
    const int c=CELL[sidx];
    if(c<0)continue;
    const int q=sidx/NM,m=sidx%NM;
    const auto &pm=hxt5b::stage27_s02b_at(matchPoints,fi,q,m);
    atomicAdd(rhs+pm.g[a],COEFF[sidx]*(Real)pm.b[a]*TARGET[sidx]);
  }

  // Diagnostics use the physical profile volume weights, not penalty weights.
  if(wallStats){
    for(int sidx=(int)threadIdx.x;sidx<NS;sidx+=(int)blockDim.x){
      if(CELL[sidx]<0)continue;
      const int q=sidx/NM,m=sidx%NM;
      const auto &pm=hxt5b::stage27_s02b_at(matchPoints,fi,q,m);
      double cur=0.0;
      #pragma unroll
      for(int a=0;a<8;++a){
        const int g=pm.g[a];
        if(g>=0)cur+=(double)(equation==K_EQ?k[g]:omega[g])*pm.b[a];
      }
      const double vw=(double)VOLW[sidx];
      atomicAdd(wallStats+0,vw);
      atomicAdd(wallStats+1,(double)TARGET[sidx]*vw);
      atomicAdd(wallStats+2,(double)YPLUS[sidx]*vw);
      atomicAdd(wallStats+3,cur*vw);
      if(equation==OMEGA_EQ){
        atomicAdd(wallStats+5,UTAU[q]*vw);
        // EXP3: gamma-weighted target^2 sum, used to recover the exact
        // five-point least-squares RMS without changing the wall equations.
        atomicAdd(wallStats+6,(double)COEFF[sidx]*(double)TARGET[sidx]*(double)TARGET[sidx]);
        // Unmodified 5-point target evaluation on transported OLD omega.
        // Compare A and B against the same physical sample set (no cherry-pick).
        const double cc=(double)BASE_COEFF[sidx];
        const double target=(double)TARGET[sidx];
        const double err=cur-target;
        atomicAdd(wallStats+13+m,cc);
        atomicAdd(wallStats+18+m,cc*target*target);
        atomicAdd(wallStats+23+m,cc*err*err);
        atomicAdd(wallStats+28+m,cc*err);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Average-first SST-consistent-table wall closure.
//
// NQ is used only to construct one physical-volume-weighted representative
// wall-column state.  The table is inverted ONCE at (|Ubar_t|, ybar).  That
// single self-consistent table state supplies both k and omega targets:
//
//   k_target     = kPlus     * u_tau^2,
//   omega_target = omegaPlus * u_tau^2 / nu.
//
// Both scalar equations constrain the same NQ cell-average FE functional via
// a rank-one volume penalty.  There is deliberately no OF-auto production
// source in this mode; this preserves the original consistent-table model.
// ---------------------------------------------------------------------------
__global__ void sst_table_profile_average_closure_hex(
    const hxt5b::HexFacePlan*F,int nf,
    const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,
    const nodals_hxt1::HexVel*hv,const std::int32_t*slot,
    const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,
    Real*mat,Real*rhs,
    double radius,double nu,double penaltyGamma,
    double kFloor,double omegaFloor,
    int equation,int profileIntegralNQ,
    double*wallStats,unsigned long long*bad)
{
  constexpr int NS=16;
  const int fi=(int)blockIdx.x;
  if(fi>=nf)return;
  const auto f=F[fi];
  const auto H=hv[f.cell];

  __shared__ Real U0[14],U1[14],U2[14],K[14],W[14];
  __shared__ Real BAVG[NS][14],LAM[NS],TARGET[NS],VCOL[NS];
  __shared__ double WSTAT[13][NS];

  if(threadIdx.x<14){
    const int g=H.g[threadIdx.x];
    U0[threadIdx.x]=u0[g];U1[threadIdx.x]=u1[g];U2[threadIdx.x]=u2[g];
    K[threadIdx.x]=k[g];W[threadIdx.x]=omega[g];
  }
  __syncthreads();

  if(threadIdx.x<NS){
    const int qs=(int)threadIdx.x,i=qs>>2,j=qs&3;
    for(int a=0;a<14;++a)BAVG[qs][a]=Real(0);
    LAM[qs]=TARGET[qs]=VCOL[qs]=Real(0);
    for(int r=0;r<13;++r)WSTAT[r][qs]=0.0;

    bool ok=(profileIntegralNQ==4||profileIntegralNQ==8||profileIntegralNQ==12);
    double cw[3]={0,0,0};
    cw[f.fd]=f.fv;cw[f.d0]=hxt5a::h5_hex_x[i];cw[f.d1]=hxt5a::h5_hex_x[j];

    double pw[14],gwWall[14][3];
    hxt2::hex_basis_ref(cw[0],cw[1],cw[2],pw,gwWall);
    double Xw[3]={0,0,0},dXw[3][3]={{0}},Xo[3]={0,0,0},dXo[3][3]={{0}};
    hxt5b::q1dev(pts,hc[f.cell],cw[0],cw[1],cw[2],Xw,dXw);
    double cr[3]={
      dXw[f.d0][1]*dXw[f.d1][2]-dXw[f.d0][2]*dXw[f.d1][1],
      dXw[f.d0][2]*dXw[f.d1][0]-dXw[f.d0][0]*dXw[f.d1][2],
      dXw[f.d0][0]*dXw[f.d1][1]-dXw[f.d0][1]*dXw[f.d1][0]
    };
    if(cr[0]*f.n[0]+cr[1]*f.n[1]+cr[2]*f.n[2]<0.0){cr[0]=-cr[0];cr[1]=-cr[1];cr[2]=-cr[2];}
    const double aj=sqrt(cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]);
    double n[3]={0,0,0},hn=0.0;
    if(ok)ok=aj>0.0&&isfinite(aj);
    if(ok){
      n[0]=cr[0]/aj;n[1]=cr[1]/aj;n[2]=cr[2]/aj;
      double co[3]={cw[0],cw[1],cw[2]};co[f.fd]=-f.fv;
      hxt5b::q1dev(pts,hc[f.cell],co[0],co[1],co[2],Xo,dXo);
      hn=(Xw[0]-Xo[0])*n[0]+(Xw[1]-Xo[1])*n[1]+(Xw[2]-Xo[2])*n[2];
      ok=hn>0.0&&isfinite(hn);
    }

    double sw=0.0,sy=0.0,bint[14]={0},xav[3]={0,0,0};
    Real uav[3]={0,0,0},guav[9]={0,0,0,0,0,0,0,0,0};
    Real kav=0,wav=0,gkav[3]={0,0,0},gwav[3]={0,0,0};

    for(int m=0;ok&&m<profileIntegralNQ;++m){
      double xi=0.0,wref=0.0;
      if(!hxt5b::spalding_profile_gauss_point(profileIntegralNQ,m,xi,wref)){ok=false;break;}
      double c[3]={cw[0],cw[1],cw[2]};c[f.fd]=-f.fv*xi;
      double ph[14],gr[14][3],gp[14][3],I[9],det=0.0;
      hxt2::hex_basis_ref(c[0],c[1],c[2],ph,gr);
      if(!hxt2::hex_metric(pts,hc[f.cell],c[0],c[1],c[2],I,det)){ok=false;break;}
      if(!(det>0.0)||!isfinite(det)){ok=false;break;}
      hxt2::phys_grad14(gr,I,gp);

      double X[3],dX[3][3];hxt5b::q1dev(pts,hc[f.cell],c[0],c[1],c[2],X,dX);
      const double yd=(Xw[0]-X[0])*n[0]+(Xw[1]-X[1])*n[1]+(Xw[2]-X[2])*n[2];
      const double ww=wref*det;
      if(!(yd>0.0)||!(ww>0.0)||!isfinite(yd)||!isfinite(ww)){ok=false;break;}

      Real uq[3]={0,0,0},guq[9]={0,0,0,0,0,0,0,0,0};
      for(int a=0;a<14;++a){
        const Real pa=(Real)ph[a];
        uq[0]+=U0[a]*pa;uq[1]+=U1[a]*pa;uq[2]+=U2[a]*pa;
        bint[a]+=ww*ph[a];
        for(int d=0;d<3;++d){
          const Real gd=(Real)gp[a][d];
          guq[d]+=U0[a]*gd;guq[3+d]+=U1[a]*gd;guq[6+d]+=U2[a]*gd;
        }
      }
      Real kq=0,wq=0,gkq[3]={0,0,0},gwq[3]={0,0,0};
      hex_sst_carrier((Real)c[0],(Real)c[1],(Real)c[2],I,K,W,
        (Real)kFloor,(Real)omegaFloor,kq,wq,gkq,gwq);

      sw+=ww;sy+=ww*yd;
      for(int d=0;d<3;++d){xav[d]+=ww*X[d];uav[d]+=(Real)ww*uq[d];gkav[d]+=(Real)ww*gkq[d];gwav[d]+=(Real)ww*gwq[d];}
      for(int d=0;d<9;++d)guav[d]+=(Real)ww*guq[d];
      kav+=(Real)ww*kq;wav+=(Real)ww*wq;
    }

    if(ok)ok=sw>0.0&&isfinite(sw);
    if(ok){
      const Real inv=(Real)(1.0/sw);
      for(int a=0;a<14;++a)BAVG[qs][a]=(Real)(bint[a]/sw);
      for(int d=0;d<3;++d){xav[d]/=sw;uav[d]*=inv;gkav[d]*=inv;gwav[d]*=inv;}
      for(int d=0;d<9;++d)guav[d]*=inv;
      kav*=inv;wav*=inv;
      const double ybar=sy/sw;
      ok=ybar>0.0&&isfinite(ybar);

      if(ok){
        const LocalSST ss=local_sst((Real)xav[0],(Real)xav[1],(Real)radius,(Real)nu,
          guav,kav,wav,gkav,gwav,(Real)kFloor,(Real)omegaFloor);
        const Real un=uav[0]*(Real)n[0]+uav[1]*(Real)n[1]+uav[2]*(Real)n[2];
        const Real Ut=sqrt(fmax(Real(0),uav[0]*uav[0]+uav[1]*uav[1]+uav[2]*uav[2]-un*un));
        const hxt5c_table::WallTableState ta=hxt5c_table::invert_velocity((double)Ut,ybar,nu);
        ok=ta.ok;
        if(ok){
          const Real kTarget=(Real)(ta.kPlus*ta.uTau*ta.uTau);
          const Real wTarget=(Real)(ta.omegaPlus*ta.uTau*ta.uTau/nu);
          const Real target=(equation==K_EQ)?kTarget:wTarget;
          const double V=hxt5a::h5_hex_w[i]*hxt5a::h5_hex_w[j]*sw;
          const Real nue=(Real)nu+(equation==K_EQ?ss.sigmaK:ss.sigmaW)*ss.nut;
          const Real lambda=(Real)penaltyGamma*nue/(Real)(hn*hn);
          VCOL[qs]=(Real)V;LAM[qs]=lambda;TARGET[qs]=target;

          double wtrace=0.0;for(int a=0;a<14;++a)wtrace+=(double)W[a]*pw[a];
          WSTAT[0][qs]=V;
          WSTAT[1][qs]=(double)target*V;
          WSTAT[2][qs]=ta.yPlus*V;
          WSTAT[3][qs]=(double)(equation==K_EQ?kav:wav)*V;
          WSTAT[4][qs]=(equation==K_EQ?(double)kav:wtrace)*V;
          if(equation==OMEGA_EQ){
            WSTAT[5][qs]=ta.uTau*V;
            WSTAT[6][qs]=ta.uPlus*V;
            WSTAT[7][qs]=ta.kPlus*V;
            WSTAT[8][qs]=ta.omegaPlus*V;
            WSTAT[9][qs]=(double)kTarget*V;
            WSTAT[10][qs]=(double)wTarget*V;
            WSTAT[11][qs]=(double)kav*V;
            WSTAT[12][qs]=(double)(ss.nut/(Real)nu)*V;
          }
        }
      }
    }
    if(!ok)atomicAdd(bad,1ULL);
  }
  __syncthreads();

  // Rank-one penalty on the same NQ cell-average functional for BOTH k and omega.
  for(int p=(int)threadIdx.x;p<196;p+=(int)blockDim.x){
    const int a=p/14,b=p%14;Real z=0;
    for(int qs=0;qs<NS;++qs)z+=VCOL[qs]*LAM[qs]*BAVG[qs][a]*BAVG[qs][b];
    atomicAdd(mat+slot[(std::size_t)f.cell*196ull+(std::size_t)p],z);
  }
  for(int a=(int)threadIdx.x;a<14;a+=(int)blockDim.x){
    Real z=0;for(int qs=0;qs<NS;++qs)z+=VCOL[qs]*LAM[qs]*BAVG[qs][a]*TARGET[qs];
    atomicAdd(rhs+H.g[a],z);
  }
  if(wallStats){
    for(int r=(int)threadIdx.x;r<13;r+=(int)blockDim.x){
      double z=0.0;for(int qs=0;qs<NS;++qs)z+=WSTAT[r][qs];
      atomicAdd(wallStats+r,z);
    }
  }
}

// ---------------------------------------------------------------------------
// Full-profile OF-auto wall closure.
//
// One profile-integrated Spalding u_tau is computed per surface quadrature
// column.  The OF-auto SST state is then evaluated at NQ normal Gauss points
// through the wall-adjacent HEX.  The k/omega production correction is volume
// integrated with the actual FE basis.  The omega target is imposed as a
// volumetric least-squares/penalty projection using
//
//   lambda = gamma * nu_eff / h_n^2,
//
// where h_n is the full projected wall-normal HEX thickness.  Because
// omega_vis ~ 1/y^2, the omega target is activated only for y+ >= ypMin.
// This avoids a non-integrable endpoint singularity while removing the
// arbitrary wallSampleFraction from the live OF-auto closure.
// ---------------------------------------------------------------------------
__global__ void sst_of_profile_closure_hex(
    const hxt5b::HexFacePlan*F,int nf,
    const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,
    const nodals_hxt1::HexVel*hv,const std::int32_t*slot,
    const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,
    Real*mat,Real*rhs,
    double radius,double nu,double penaltyGamma,
    double kFloor,double omegaFloor,
    double ofKappa,double ofE,double ofBeta1,double ofReBlend,double ofProdLimit,
    double prodScale,double wallVolumeScale,
    int equation,int sharedSpaldingWallState,int profileIntegralNQ,
    double ypMin,int ofProduction,
    double*wallStats,unsigned long long*bad)
{
  constexpr int NS=16,MAXNQ=12,MAXQ=NS*MAXNQ;
  const int fi=(int)blockIdx.x;
  if(fi>=nf)return;
  const auto f=F[fi];
  const auto H=hv[f.cell];

  __shared__ Real U0[14],U1[14],U2[14],K[14],W[14];
  __shared__ double UTAU[NS],HN[NS];
  __shared__ Real PH[MAXQ][14],LAM[MAXQ],GVAL[MAXQ],SRC[MAXQ],VW[MAXQ];

  if(threadIdx.x<14){
    const int g=H.g[threadIdx.x];
    U0[threadIdx.x]=u0[g];
    U1[threadIdx.x]=u1[g];
    U2[threadIdx.x]=u2[g];
    K[threadIdx.x]=k[g];
    W[threadIdx.x]=omega[g];
  }
  __syncthreads();

  // One common u_tau for each tangential surface quadrature column.
  if(threadIdx.x<NS){
    const int qs=(int)threadIdx.x,i=qs>>2,j=qs&3;
    UTAU[qs]=0.0;HN[qs]=0.0;
    double cw[3]={0,0,0};
    cw[f.fd]=f.fv;
    cw[f.d0]=hxt5a::h5_hex_x[i];
    cw[f.d1]=hxt5a::h5_hex_x[j];

    double ut=0.0,yp=0.0,beta=0.0,ubar=0.0,ybar=0.0,rms=0.0,span=0.0,bavg[14];
    bool ok=hxt5b::spalding_profile_hex(
      pts,hc[f.cell],f,cw,U2,nu,profileIntegralNQ,
      ut,yp,beta,ubar,ybar,bavg,rms,span);

    double Xw[3],dXw[3][3],Xo[3],dXo[3][3];
    if(ok){
      hxt5b::q1dev(pts,hc[f.cell],cw[0],cw[1],cw[2],Xw,dXw);
      double cr[3]={
        dXw[f.d0][1]*dXw[f.d1][2]-dXw[f.d0][2]*dXw[f.d1][1],
        dXw[f.d0][2]*dXw[f.d1][0]-dXw[f.d0][0]*dXw[f.d1][2],
        dXw[f.d0][0]*dXw[f.d1][1]-dXw[f.d0][1]*dXw[f.d1][0]
      };
      if(cr[0]*f.n[0]+cr[1]*f.n[1]+cr[2]*f.n[2]<0.0){
        cr[0]=-cr[0];cr[1]=-cr[1];cr[2]=-cr[2];
      }
      const double aj=sqrt(cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]);
      ok=aj>0.0&&isfinite(aj);
      if(ok){
        const double n[3]={cr[0]/aj,cr[1]/aj,cr[2]/aj};
        double co[3]={cw[0],cw[1],cw[2]};co[f.fd]=-f.fv;
        hxt5b::q1dev(pts,hc[f.cell],co[0],co[1],co[2],Xo,dXo);
        const double hn=(Xw[0]-Xo[0])*n[0]+(Xw[1]-Xo[1])*n[1]+(Xw[2]-Xo[2])*n[2];
        ok=hn>0.0&&isfinite(hn);
        if(ok){UTAU[qs]=ut;HN[qs]=hn;}
      }
    }
    if(!ok)atomicAdd(bad,1ULL);
  }
  __syncthreads();

  const int nq=profileIntegralNQ;
  const int nqp=NS*nq;
  const int q=(int)threadIdx.x;
  if(q<nqp){
    for(int a=0;a<14;++a)PH[q][a]=Real(0);
    LAM[q]=GVAL[q]=SRC[q]=VW[q]=Real(0);

    const int qs=q/nq,m=q-qs*nq,i=qs>>2,j=qs&3;
    const double ut=UTAU[qs],hn=HN[qs];
    bool ok=hn>0.0&&isfinite(ut);

    double xi=0.0,wref=0.0;
    if(ok)ok=hxt5b::spalding_profile_gauss_point(nq,m,xi,wref);

    double cw[3]={0,0,0};
    cw[f.fd]=f.fv;
    cw[f.d0]=hxt5a::h5_hex_x[i];
    cw[f.d1]=hxt5a::h5_hex_x[j];

    double Xw[3]={0,0,0},dXw[3][3]={{0}};
    double n[3]={0,0,0};
    if(ok){
      hxt5b::q1dev(pts,hc[f.cell],cw[0],cw[1],cw[2],Xw,dXw);
      double cr[3]={
        dXw[f.d0][1]*dXw[f.d1][2]-dXw[f.d0][2]*dXw[f.d1][1],
        dXw[f.d0][2]*dXw[f.d1][0]-dXw[f.d0][0]*dXw[f.d1][2],
        dXw[f.d0][0]*dXw[f.d1][1]-dXw[f.d0][1]*dXw[f.d1][0]
      };
      if(cr[0]*f.n[0]+cr[1]*f.n[1]+cr[2]*f.n[2]<0.0){
        cr[0]=-cr[0];cr[1]=-cr[1];cr[2]=-cr[2];
      }
      const double aj=sqrt(cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]);
      ok=aj>0.0&&isfinite(aj);
      if(ok){n[0]=cr[0]/aj;n[1]=cr[1]/aj;n[2]=cr[2]/aj;}
    }

    double c[3]={cw[0],cw[1],cw[2]};
    c[f.fd]=-f.fv*xi;
    double ph[14],gr[14][3],gp[14][3],I[9],det=0.0;
    if(ok){
      hxt2::hex_basis_ref(c[0],c[1],c[2],ph,gr);
      ok=hxt2::hex_metric(pts,hc[f.cell],c[0],c[1],c[2],I,det);
    }
    if(ok)hxt2::phys_grad14(gr,I,gp);

    double X[3]={0,0,0},dX[3][3]={{0}};
    if(ok)hxt5b::q1dev(pts,hc[f.cell],c[0],c[1],c[2],X,dX);
    const double yd=ok?((Xw[0]-X[0])*n[0]+(Xw[1]-X[1])*n[1]+(Xw[2]-X[2])*n[2]):0.0;
    if(ok)ok=yd>0.0&&det>0.0&&isfinite(yd)&&isfinite(det);

    Real gu[9]={0},gk[3]={0},gw[3]={0},kv=0,wv=0,us0=0,us1=0,us2=0;
    if(ok){
      for(int a=0;a<14;++a){
        const Real pa=(Real)ph[a];
        PH[q][a]=pa;
        us0+=U0[a]*pa;us1+=U1[a]*pa;us2+=U2[a]*pa;
        for(int d=0;d<3;++d){
          const Real gd=(Real)gp[a][d];
          gu[d]+=U0[a]*gd;gu[3+d]+=U1[a]*gd;gu[6+d]+=U2[a]*gd;
        }
      }
      hex_sst_carrier((Real)c[0],(Real)c[1],(Real)c[2],I,K,W,
        (Real)kFloor,(Real)omegaFloor,kv,wv,gk,gw);

      const LocalSST ss=local_sst((Real)X[0],(Real)X[1],(Real)radius,(Real)nu,
        gu,kv,wv,gk,gw,(Real)kFloor,(Real)omegaFloor);

      const Real un=us0*(Real)n[0]+us1*(Real)n[1]+us2*(Real)n[2];
      const Real Ut=sqrt(fmax(Real(0),us0*us0+us1*us1+us2*us2-un*un));
      const OFWallState oa=of_auto_wall_state(
        (Real)yd,(Real)nu,Ut,(Real)ut,kv,wv,ss,
        (Real)kFloor,(Real)omegaFloor,(Real)ofKappa,(Real)ofE,
        (Real)ofBeta1,(Real)ofReBlend,(Real)ofProdLimit,
        sharedSpaldingWallState);

      const double volw=
        hxt5a::h5_hex_w[i]*hxt5a::h5_hex_w[j]*wref*det;
      VW[q]=(Real)volw;

      if(ofProduction){
        const Real dpk=(Real)prodScale*(oa.pkAuto-ss.pk);
        Real src=dpk;
        if(equation==OMEGA_EQ){
          const Real nt=fmax(ss.nut,fmax(Real(1e-12)*(Real)nu,Real(1e-20)));
          src=ss.gamma*dpk/nt;
        }
        SRC[q]=src;
      }

      const double yp=yd*ut/nu;
      if(equation==OMEGA_EQ && yp>=ypMin){
        const Real nue=(Real)nu+ss.sigmaW*ss.nut;
        LAM[q]=(Real)(penaltyGamma)*nue/(Real)(hn*hn);
        GVAL[q]=oa.omegaTarget;

        if(wallStats){
          const double sw=volw;
          const Real wtrace=hex_q1_value(
            (Real)cw[0],(Real)cw[1],(Real)cw[2],W,(Real)omegaFloor);
          atomicAdd(wallStats+0,sw);
          atomicAdd(wallStats+1,(double)oa.omegaTarget*sw);
          atomicAdd(wallStats+2,yp*sw);
          atomicAdd(wallStats+3,(double)wv*sw);
          atomicAdd(wallStats+4,(double)wtrace*sw);
          atomicAdd(wallStats+5,(double)oa.rey*sw);
          atomicAdd(wallStats+6,(double)oa.lamFrac*sw);
          atomicAdd(wallStats+7,(double)oa.omegaVis*sw);
          atomicAdd(wallStats+8,(double)oa.omegaLog*sw);
          atomicAdd(wallStats+9,(double)ss.pk*sw);
          atomicAdd(wallStats+10,(double)oa.pkAuto*sw);
          atomicAdd(wallStats+11,(double)(oa.nutSpalding/(Real)nu)*sw);
          atomicAdd(wallStats+12,(double)(ss.nut/(Real)nu)*sw);
        }
      }
    }
    if(!ok)atomicAdd(bad,1ULL);
  }
  __syncthreads();

  // Omega profile target: volumetric least-squares/penalty projection.
  if(equation==OMEGA_EQ){
    for(int p=(int)threadIdx.x;p<196;p+=(int)blockDim.x){
      const int a=p/14,b=p%14;
      Real z=0;
      for(int qq=0;qq<nqp;++qq)
        z+=VW[qq]*LAM[qq]*PH[qq][a]*PH[qq][b];
      atomicAdd(mat+slot[(std::size_t)f.cell*196ull+(std::size_t)p],z);
    }
  }

  // Profile-integrated production correction for both k and omega, plus the
  // omega profile-target RHS when solving omega.
  for(int a=(int)threadIdx.x;a<14;a+=(int)blockDim.x){
    Real z=0;
    for(int qq=0;qq<nqp;++qq){
      z+=VW[qq]*((Real)wallVolumeScale*SRC[qq]*PH[qq][a]);
      if(equation==OMEGA_EQ)
        z+=VW[qq]*LAM[qq]*PH[qq][a]*GVAL[qq];
    }
    atomicAdd(rhs+H.g[a],z);
  }
}

__device__ __forceinline__ void atomic_max_double(double*addr,double v){auto*p=(unsigned long long*)addr;unsigned long long old=*p,ass;do{ass=old;if(__longlong_as_double((long long)ass)>=v)break;old=atomicCAS(p,ass,(unsigned long long)__double_as_longlong(v));}while(ass!=old);}

__global__ void sst_hex_stats(const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,const nodals_hxt1::HexVel*hv,const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,std::uint64_t n,double radius,double nu,double kFloor,double omegaFloor,double*stats,unsigned long long*bad){
  std::uint64_t c=(std::uint64_t)blockIdx.x;if(c>=n)return;__shared__ Real U0[14],U1[14],U2[14],K[14],W[14];if(threadIdx.x<14){int g=hv[c].g[threadIdx.x];U0[threadIdx.x]=u0[g];U1[threadIdx.x]=u1[g];U2[threadIdx.x]=u2[g];K[threadIdx.x]=k[g];W[threadIdx.x]=omega[g];}__syncthreads();int iq=(int)threadIdx.x;if(iq>=64)return;int ix=iq>>4,iy=(iq>>2)&3,iz=iq&3;double xr=hxt5a::h5_hex_x[ix],yr=hxt5a::h5_hex_x[iy],zr=hxt5a::h5_hex_x[iz],qw=hxt5a::h5_hex_w[ix]*hxt5a::h5_hex_w[iy]*hxt5a::h5_hex_w[iz];double ph[14],gr[14][3],gp[14][3],I[9],det;hxt2::hex_basis_ref(xr,yr,zr,ph,gr);if(!hxt2::hex_metric(pts,hc[c],xr,yr,zr,I,det)){atomicAdd(bad,1ULL);return;}hxt2::phys_grad14(gr,I,gp);Real gu[9]={0,0,0,0,0,0,0,0,0},gk[3]={0,0,0},gw[3]={0,0,0},kv=0,wv=0;for(int a=0;a<14;++a){Real pa=(Real)ph[a];kv+=K[a]*pa;wv+=W[a]*pa;for(int d=0;d<3;++d){Real gd=(Real)gp[a][d];gk[d]+=K[a]*gd;gw[d]+=W[a]*gd;gu[d]+=U0[a]*gd;gu[3+d]+=U1[a]*gd;gu[6+d]+=U2[a]*gd;}}hex_sst_carrier((Real)xr,(Real)yr,(Real)zr,I,K,W,(Real)kFloor,(Real)omegaFloor,kv,wv,gk,gw);double x=0,y=0;for(int a=0;a<8;++a){double N=.125*(1+hxt4a::h4_hex_sign[a][0]*xr)*(1+hxt4a::h4_hex_sign[a][1]*yr)*(1+hxt4a::h4_hex_sign[a][2]*zr);auto X=pts[hc[c].v[a]];x+=N*X.x;y+=N*X.y;}LocalSST s=local_sst((Real)x,(Real)y,(Real)radius,(Real)nu,gu,kv,wv,gk,gw,(Real)kFloor,(Real)omegaFloor);double wt=det*qw,rat=(double)s.nut/nu;atomicAdd(stats+0,rat*wt);atomicAdd(stats+1,wt);atomic_max_double(stats+2,rat);atomic_max_double(stats+3,(double)s.F1);atomic_max_double(stats+4,(double)s.F2);if(s.kFloored)atomicAdd(stats+5,wt);if(s.omegaFloored)atomicAdd(stats+6,wt);
}

__global__ void sst_tet_stats(const nodals_hxt1::Point*pts,const nodals_hxt1::TetConn*tc,const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,std::uint64_t n,double radius,double nu,double kFloor,double omegaFloor,double*stats){
  std::uint64_t c=(std::uint64_t)blockIdx.x;if(c>=n)return;__shared__ Real U0[8],U1[8],U2[8],K[8],W[8];if(threadIdx.x<8){int g=tv[c].g[threadIdx.x];U0[threadIdx.x]=u0[g];U1[threadIdx.x]=u1[g];U2[threadIdx.x]=u2[g];K[threadIdx.x]=k[g];W[threadIdx.x]=omega[g];}__syncthreads();int iq=(int)threadIdx.x;if(iq>=64)return;int ir=iq>>4,is=(iq>>2)&3,it=iq&3;double r=hxt5a::h5_tet_rn[ir],s0=hxt5a::h5_tet_sn[is],t=hxt5a::h5_tet_tn[it],omr=1-r,oms=1-s0,l[4]={omr*oms*(1-t),r,omr*s0,omr*oms*t},qw=hxt5a::h5_tet_rw[ir]*hxt5a::h5_tet_sw[is]*hxt5a::h5_tet_tw[it];double ph[8],gr[8][3],gp[8][3];hxt2::tet_basis(l[0],l[1],l[2],l[3],ph,gr);hxt2::tet_phys_grad(gr,tg[c],gp);Real gu[9]={0,0,0,0,0,0,0,0,0},gk[3]={0,0,0},gw[3]={0,0,0},kv=0,wv=0;for(int a=0;a<8;++a){Real pa=(Real)ph[a];kv+=K[a]*pa;wv+=W[a]*pa;for(int d=0;d<3;++d){Real gd=(Real)gp[a][d];gk[d]+=K[a]*gd;gw[d]+=W[a]*gd;gu[d]+=U0[a]*gd;gu[3+d]+=U1[a]*gd;gu[6+d]+=U2[a]*gd;}}hxt5c::eval_dg_omega_p1((int)c,l,tg[c],wv,gw);double x=0,y=0;for(int a=0;a<4;++a){auto X=pts[tc[c].v[a]];x+=l[a]*X.x;y+=l[a]*X.y;}LocalSST s=local_sst((Real)x,(Real)y,(Real)radius,(Real)nu,gu,kv,wv,gk,gw,(Real)kFloor,(Real)omegaFloor,hxt5c::dg_omega_p1_constitutive?stage20_distance(tc,(int)c,l):Real(-1));double wt=tg[c].det*qw,rat=(double)s.nut/nu;atomicAdd(stats+0,rat*wt);atomicAdd(stats+1,wt);atomic_max_double(stats+2,rat);atomic_max_double(stats+3,(double)s.F1);atomic_max_double(stats+4,(double)s.F2);if(s.kFloored)atomicAdd(stats+5,wt);if(s.omegaFloored)atomicAdd(stats+6,wt);
}

#include "modules/transport/scalar_relaxation.inc"
#include "modules/transport/scalar_linear_solver.inc"
struct Stats{double meanNutOverNu=0,maxNutOverNu=0,maxF1=0,maxF2=0,kFloorFrac=0,omegaFloorFrac=0;};
inline Stats collect_stats(std::uint64_t nhex,std::uint64_t ntet,const Dev<nodals_hxt1::Point>&pts,const Dev<nodals_hxt1::HexConn>&hc,const Dev<nodals_hxt1::TetConn>&tc,const Dev<nodals_hxt1::HexVel>&hv,const Dev<nodals_hxt1::TetVel>&tv,const Dev<nodals_hxt1::TetGeom>&tg,const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,double radius,double nu,const Options&O,Dev<double>&stats,Dev<unsigned long long>&bad){HXT1_CUDA(cudaMemset(stats.p,0,stats.n*sizeof(double)));HXT1_CUDA(cudaMemset(bad.p,0,bad.n*sizeof(unsigned long long)));if(nhex)sst_hex_stats<<<nhex,64>>>(pts.p,hc.p,hv.p,u0,u1,u2,k,omega,nhex,radius,nu,O.kFloor,O.omegaFloor,stats.p,bad.p);if(ntet)sst_tet_stats<<<ntet,64>>>(pts.p,tc.p,tv.p,tg.p,u0,u1,u2,k,omega,ntet,radius,nu,O.kFloor,O.omegaFloor,stats.p);HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());unsigned long long nb=0;HXT1_CUDA(cudaMemcpy(&nb,bad.p,sizeof(nb),cudaMemcpyDeviceToHost));if(nb)throw std::runtime_error("SST G1 stats geometry failure count="+std::to_string(nb));std::vector<double>h(7,0);HXT1_CUDA(cudaMemcpy(h.data(),stats.p,7*sizeof(double),cudaMemcpyDeviceToHost));Stats s;if(h[1]>0){s.meanNutOverNu=h[0]/h[1];s.kFloorFrac=h[5]/h[1];s.omegaFloorFrac=h[6]/h[1];}s.maxNutOverNu=h[2];s.maxF1=h[3];s.maxF2=h[4];return s;}

#include "modules/stabilization/stage33bc_scalar_supg.cuh"
#include "modules/transport/transport_preflight.inc"
#include "modules/transport/transport_volume_launch.inc"
#include "modules/transport/transport_face_wall.inc"
#include "modules/transport/transport_finalize.inc"
inline void write_sst_vtu(const std::string&path,const nodals_hxt1::HostMesh&M,const std::vector<Real>&k,const std::vector<Real>&omega){std::ofstream f(path);if(!f)throw std::runtime_error("cannot write "+path);f<<std::setprecision(16)<<"<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n<UnstructuredGrid><Piece NumberOfPoints=\""<<M.points.size()<<"\" NumberOfCells=\""<<(M.hexes.size()+M.tets.size())<<"\">\n<Points><DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";for(auto&q:M.points)f<<q.x<<" "<<q.y<<" "<<q.z<<"\n";f<<"</DataArray></Points><Cells><DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";for(auto&h:M.hexes){for(int a=0;a<8;++a)f<<h.v[a]<<" ";f<<"\n";}for(auto&t:M.tets){for(int a=0;a<4;++a)f<<t.v[a]<<" ";f<<"\n";}f<<"</DataArray><DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";long long off=0;for(std::size_t i=0;i<M.hexes.size();++i){off+=8;f<<off<<"\n";}for(std::size_t i=0;i<M.tets.size();++i){off+=4;f<<off<<"\n";}f<<"</DataArray><DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";for(std::size_t i=0;i<M.hexes.size();++i)f<<"12\n";for(std::size_t i=0;i<M.tets.size();++i)f<<"10\n";f<<"</DataArray></Cells><PointData><DataArray type=\"Float64\" Name=\"k_vertex\" format=\"ascii\">\n";for(std::size_t i=0;i<M.points.size();++i)f<<(double)k[i]<<"\n";f<<"</DataArray><DataArray type=\"Float64\" Name=\"omega_vertex\" format=\"ascii\">\n";for(std::size_t i=0;i<M.points.size();++i)f<<(double)omega[i]<<"\n";f<<"</DataArray></PointData></Piece></UnstructuredGrid></VTKFile>\n";}

struct Result{bool converged=false,gate=false;int iterations=0;double kIn=0,omegaIn=0;Stats stats;double wallOmegaMean=0,wallYPlusMean=0,kRel=0,omegaRel=0,nutRel=0;unsigned long long vertexKClips=0,vertexOmegaClips=0;};

inline Result run_frozen(
    const nodals_hxt1::HostMesh&M,const hxt5b::BoundaryHost&BH,
    const Dev<hxt5b::HexFacePlan>&inH,const Dev<hxt5b::TetFacePlan>&inT,const Dev<hxt5b::HexFacePlan>&wallH,
    const Dev<nodals_hxt1::Point>&pts,const Dev<nodals_hxt1::HexConn>&hc,const Dev<nodals_hxt1::TetConn>&tc,const Dev<nodals_hxt1::HexVel>&hv,const Dev<nodals_hxt1::TetVel>&tv,const Dev<nodals_hxt1::TetGeom>&tg,const Dev<hxt2::InterfacePlan>&ip,
    hxt4b::GpuCSR&A,const hxt4b::ColoringHost&C,const Dev<unsigned char>&fixed,const Real*u0,const Real*u1,const Real*u2,
    Dev<Real>&vol,Dev<Real>&dg,Dev<Real>&wall,Dev<unsigned long long>&bad,
    double bulk,double radius,double nu,double interfaceGamma,double wallSampleFraction,const Options&O,const std::string&vtuPath)
{
  if(!O.enabled)return {};
  if(!(O.intensity>0&&O.lengthRatio>0&&O.alphaK>0&&O.alphaK<=1&&O.alphaOmega>0&&O.alphaOmega<=1&&O.nonlinearTol>0&&O.nonlinearMax>0&&O.linearRtol>0&&O.linearRtol<1&&O.linearMax>0))throw std::runtime_error("bad SST G1 option");
  if(!(O.linearSolver=="sgs1"||O.linearSolver=="fgs1"||O.linearSolver=="fgmres-fgs")||O.fgmresRestart<=0||O.fgsSweeps<=0)throw std::runtime_error("bad SST G1 linear solver option");
  const int n=A.n,nvert=(int)M.h.nv;const double D=2*radius,Lt=O.lengthRatio*D,kIn=1.5*std::pow(O.intensity*bulk,2),omegaIn=std::sqrt(kIn)/(std::pow(C_MU,0.25)*Lt);
  std::vector<Real>hk0((std::size_t)n,0),hw0((std::size_t)n,0);for(int i=0;i<nvert;++i){hk0[(std::size_t)i]=(Real)kIn;hw0[(std::size_t)i]=(Real)omegaIn;}
  Dev<Real>k(hk0),omega(hw0),oldK(hk0),oldW(hw0),rhs(std::vector<Real>((std::size_t)n,0));Dev<double>stats(std::vector<double>(7,0)),wallStats(std::vector<double>(13,0));Dev<unsigned long long>clipK(std::vector<unsigned long long>(1,0)),clipW(std::vector<unsigned long long>(1,0)),diagAudit(std::vector<unsigned long long>(2,0));hxt4a::Reducer<Real>red(n);
  sync_sst_state_space_mode();
  sync_sst_deviatoric_production(O.deviatoricProduction);
  const int stateMode=sst_state_space_mode_host();
  const bool q1Transport=(stateMode==SST_STATE_Q1);
  std::vector<unsigned char> q1MaskHost((std::size_t)n,0);
  if(q1Transport) for(int i=nvert;i<n;++i) q1MaskHost[(std::size_t)i]=1;
  Dev<unsigned char> q1Mask(q1MaskHost);
  const Dev<unsigned char>& scalarFixed=q1Transport?q1Mask:fixed;
  std::printf("NODALS_SST_STATE_SPACE mode=%s transport=%s constitutive=%s activeDofs=%d constrainedEnrichmentDofs=%d positivity=BERNSTEIN_CONTROL_NET_IF_Q2 status=PASS\n",
    sst_state_space_name(stateMode),
    q1Transport?"VERTEX_Q1":"FULL_14DOF",
    stateMode==SST_STATE_BERNSTEIN_Q2?"BERNSTEIN_Q2_POSITIVE_LIMITED14":
      (stateMode==SST_STATE_Q1BF2_UNLIMITED?"Q1_BF2_DIRECT_14DOF_UNLIMITED":"VERTEX_Q1"),
    q1Transport?nvert:n,
    q1Transport?(n-nvert):0);
  // Frozen velocity: rebuild convection once from the converged G0 field.
  HXT1_CUDA(cudaMemset(A.conv.p,0,A.nnz*sizeof(Real)));HXT1_CUDA(cudaMemset(bad.p,0,bad.n*sizeof(unsigned long long)));
  if(M.h.nhex)hxt4b::hex_conv_assemble<<<M.h.nhex,128>>>(pts.p,hc.p,hv.p,A.hexSlot.p,u0,u1,u2,A.conv.p,M.h.nhex,bad.p);if(M.h.ntet)hxt4b::tet_conv_assemble<<<M.h.ntet,64>>>(tv.p,tg.p,A.tetSlot.p,u0,u1,u2,A.conv.p,M.h.ntet);HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());unsigned long long nb=0;HXT1_CUDA(cudaMemcpy(&nb,bad.p,sizeof(nb),cudaMemcpyDeviceToHost));if(nb)throw std::runtime_error("SST G1 frozen convection geometry failure");
  std::printf("NODALS_SST_G1_CONFIG model=SST-2003m precision=%s coupling=FROZEN_G0_VELOCITY momentumFeedback=OFF wallDistance=ANALYTIC_PIPE_R_MINUS_R turbulenceInlet=DG_TRACE intensity=%.6g lengthScaleOverD=%.6g kIn=%.12e omegaIn=%.12e outlet=NATURAL_ZERO_GRADIENT kWall=%s omegaWallMode=%s alphaK=%.6g alphaOmega=%.6g nonlinearTol=%.3e nonlinearMax=%d linear=%s nonlinearRelax=ROW_L1_FIXED_POINT postFieldRelax=OFF linearRtol=%.3e linearMax=%d status=PASS\n",hxt4b::PRECISION,O.intensity,O.lengthRatio,kIn,omegaIn,O.omegaWallMode==OMEGA_WALL_CONSISTENT_TABLE?"CONSISTENT_TABLE_SAMPLE":"HIGH_RE_ZERO_GRADIENT",omega_wall_mode_name(O.omegaWallMode),O.alphaK,O.alphaOmega,O.nonlinearTol,O.nonlinearMax,O.linearSolver.c_str(),O.linearRtol,O.linearMax);
  Result R;R.kIn=kIn;R.omegaIn=omegaIn;double prevNut=-1;
  for(int it=1;it<=O.nonlinearMax;++it){
    HXT1_CUDA(cudaMemcpy(oldK.p,k.p,(std::size_t)n*sizeof(Real),cudaMemcpyDeviceToDevice));HXT1_CUDA(cudaMemcpy(oldW.p,omega.p,(std::size_t)n*sizeof(Real),cudaMemcpyDeviceToDevice));
    assemble_equation(K_EQ,M,BH,inH,inT,wallH,pts,hc,tc,hv,tv,tg,ip,A,u0,u1,u2,oldK.p,oldW.p,vol,dg,wall,rhs,wallStats,bad,diagAudit,kIn,bulk,radius,nu,interfaceGamma,wallSampleFraction,O,it);
    auto lk=scalar_solve(A,C,scalarFixed,rhs.p,k.p,red,O,it,"k");if(q1Transport&&n>nvert)zero_enrichment_kernel<<<(n-nvert+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(nvert,n,k.p);HXT1_CUDA(cudaMemset(clipK.p,0,sizeof(unsigned long long)));floor_vertex_kernel<<<(nvert+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(nvert,(Real)O.kFloor,k.p,clipK.p);
    assemble_equation(OMEGA_EQ,M,BH,inH,inT,wallH,pts,hc,tc,hv,tv,tg,ip,A,u0,u1,u2,k.p,oldW.p,vol,dg,wall,rhs,wallStats,bad,diagAudit,omegaIn,bulk,radius,nu,interfaceGamma,wallSampleFraction,O,it);
    auto lw=scalar_solve(A,C,scalarFixed,rhs.p,omega.p,red,O,it,"omega");if(q1Transport&&n>nvert)zero_enrichment_kernel<<<(n-nvert+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(nvert,n,omega.p);HXT1_CUDA(cudaMemset(clipW.p,0,sizeof(unsigned long long)));floor_vertex_kernel<<<(nvert+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(nvert,(Real)O.omegaFloor,omega.p,clipW.p);HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
    field_diff_kernel<<<(n+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(n,k.p,oldK.p,A.tmp.p);double dk=red.norm(A.tmp.p),nk=red.norm(k.p);field_diff_kernel<<<(n+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(n,omega.p,oldW.p,A.tmp.p);double dw=red.norm(A.tmp.p),nw=red.norm(omega.p);double rk=dk/std::max(nk,1e-300),rw=dw/std::max(nw,1e-300);
    auto ss=collect_stats(M.h.nhex,M.h.ntet,pts,hc,tc,hv,tv,tg,u0,u1,u2,k.p,omega.p,radius,nu,O,stats,bad);double rn=prevNut>0?std::abs(ss.meanNutOverNu-prevNut)/std::max(ss.meanNutOverNu,1e-300):std::numeric_limits<double>::infinity();prevNut=ss.meanNutOverNu;
    std::vector<double>hws(13,0);HXT1_CUDA(cudaMemcpy(hws.data(),wallStats.p,13*sizeof(double),cudaMemcpyDeviceToHost));double wom=hws[0]>0?hws[1]/hws[0]:0,wyp=hws[0]>0?hws[2]/hws[0]:0,wsamp=hws[0]>0?hws[3]/hws[0]:0,wtrace=hws[0]>0?hws[4]/hws[0]:0;
    if(it<=5||it%O.printEvery==0)std::printf("NODALS_SST_OMEGA_WALL mode=%s nonlinearIt=%d targetMean=%.6e sampleMean=%.6e traceMean=%.6e sampleOverTarget=%.6e traceOverTarget=%.6e yPlusMean=%.6e penaltyGamma=%.6e status=PASS\n",omega_wall_mode_name(O.omegaWallMode),it,wom,wsamp,wtrace,wom>0?wsamp/wom:0,wom>0?wtrace/wom:0,wyp,O.wallPenaltyGamma);if((it<=5||it%O.printEvery==0)&&O.omegaWallMode==OMEGA_WALL_OF_AUTO&&hws[0]>0)std::printf("NODALS_SST_OF_AUTO nonlinearIt=%d ReYMean=%.6e lamFracMean=%.6e omegaVisMean=%.6e omegaLogMean=%.6e pkBulkMean=%.6e pkAutoMean=%.6e pkRatio=%.6e nutSpaldingOverNuMean=%.6e nutSSTOverNuSampleMean=%.6e production=%d prodScale=%.6e wallVolumeScale=%.6e sharedSpaldingState=%d status=PASS\n",it,hws[5]/hws[0],hws[6]/hws[0],hws[7]/hws[0],hws[8]/hws[0],hws[9]/hws[0],hws[10]/hws[0],hws[9]!=0?hws[10]/hws[9]:0,hws[11]/hws[0],hws[12]/hws[0],(int)O.ofProduction,O.ofProductionScale,O.ofWallVolumeScale,(int)O.sharedSpaldingWallState);if((it<=5||it%O.printEvery==0)&&O.omegaWallMode==OMEGA_WALL_COMPATIBLE&&hws[0]>0)std::printf("NODALS_SST_COMPATIBLE nonlinearIt=%d uTauMean=%.6e uTauViscMean=%.6e uStarLogMean=%.6e uStarMean=%.6e uStarOverUTau=%.6e tauCompatMean=%.6e pkBulkMean=%.6e pkCompatMean=%.6e pkRatio=%.6e production=%d prodScale=%.6e wallVolumeScale=%.6e status=PASS\n",it,hws[5]/hws[0],hws[6]/hws[0],hws[7]/hws[0],hws[8]/hws[0],hws[5]!=0?hws[8]/hws[5]:0,hws[11]/hws[0],hws[9]/hws[0],hws[10]/hws[0],hws[9]!=0?hws[10]/hws[9]:0,(int)O.ofProduction,O.ofProductionScale,O.ofWallVolumeScale);if((it<=5||it%O.printEvery==0)&&O.omegaWallMode==OMEGA_WALL_CONSISTENT_TABLE&&hws[0]>0)std::printf("NODALS_SST_CONSISTENT_TABLE nonlinearIt=%d uTauMean=%.6e UPlusMean=%.6e kPlusMean=%.6e omegaPlusMean=%.6e kTargetMean=%.6e omegaTargetMean=%.6e kSampleMean=%.6e nutSSTOverNuSampleMean=%.6e status=PASS\n",it,hws[5]/hws[0],hws[6]/hws[0],hws[7]/hws[0],hws[8]/hws[0],hws[9]/hws[0],hws[10]/hws[0],hws[11]/hws[0],hws[12]/hws[0]);unsigned long long ck=0,cw=0;HXT1_CUDA(cudaMemcpy(&ck,clipK.p,sizeof(ck),cudaMemcpyDeviceToHost));HXT1_CUDA(cudaMemcpy(&cw,clipW.p,sizeof(cw),cudaMemcpyDeviceToHost));R.vertexKClips+=ck;R.vertexOmegaClips+=cw;
    bool finite=std::isfinite(rk)&&std::isfinite(rw)&&std::isfinite(ss.meanNutOverNu)&&std::isfinite(ss.maxNutOverNu)&&std::isfinite(wom);if(!finite)throw std::runtime_error("SST G1 nonfinite nonlinear diagnostic");R.iterations=it;R.kRel=rk;R.omegaRel=rw;R.nutRel=rn;R.stats=ss;R.wallOmegaMean=wom;R.wallYPlusMean=wyp;
    R.converged=(it>=5&&rk<=O.nonlinearTol&&rw<=O.nonlinearTol&&rn<=O.nonlinearTol);
    if(it<=5||it%O.printEvery==0||R.converged)std::printf("NODALS_SST_G1 it=%d kRelUpdate=%.6e omegaRelUpdate=%.6e nutMeanRelUpdate=%.6e kLinearWorks=%d kLinearRel=%.6e omegaLinearWorks=%d omegaLinearRel=%.6e meanNutOverNu=%.6e maxNutOverNu=%.6e maxF1=%.6e maxF2=%.6e kFloorVolumeFrac=%.6e omegaFloorVolumeFrac=%.6e wallOmegaMean=%.6e wallYPlusMean=%.6e vertexClipsK=%llu vertexClipsOmega=%llu converged=%d status=PASS\n",it,rk,rw,rn,lk.works,lk.rel,lw.works,lw.rel,ss.meanNutOverNu,ss.maxNutOverNu,ss.maxF1,ss.maxF2,ss.kFloorFrac,ss.omegaFloorFrac,wom,wyp,ck,cw,(int)R.converged);
    if(R.converged)break;
    if(it>=10&&(ss.meanNutOverNu>1e8||ss.maxNutOverNu>1e10))throw std::runtime_error("SST G1 eddy-viscosity divergence guard");
  }
  std::vector<Real>hk((std::size_t)n),hw((std::size_t)n);HXT1_CUDA(cudaMemcpy(hk.data(),k.p,(std::size_t)n*sizeof(Real),cudaMemcpyDeviceToHost));HXT1_CUDA(cudaMemcpy(hw.data(),omega.p,(std::size_t)n*sizeof(Real),cudaMemcpyDeviceToHost));write_sst_vtu(vtuPath,M,hk,hw);
  R.gate=R.converged&&R.stats.meanNutOverNu>0&&R.stats.maxNutOverNu>0&&R.stats.kFloorFrac<0.05&&R.stats.omegaFloorFrac<0.05&&std::isfinite(R.wallOmegaMean)&&R.wallOmegaMean>0;
  std::printf("NODALS_SST_G1_FINAL iterations=%d kRelUpdate=%.12e omegaRelUpdate=%.12e nutMeanRelUpdate=%.12e meanNutOverNu=%.12e maxNutOverNu=%.12e kFloorVolumeFrac=%.12e omegaFloorVolumeFrac=%.12e wallOmegaMean=%.12e wallYPlusMean=%.12e vertexClipsK=%llu vertexClipsOmega=%llu vtu=%s status=%s\n",R.iterations,R.kRel,R.omegaRel,R.nutRel,R.stats.meanNutOverNu,R.stats.maxNutOverNu,R.stats.kFloorFrac,R.stats.omegaFloorFrac,R.wallOmegaMean,R.wallYPlusMean,R.vertexKClips,R.vertexOmegaClips,vtuPath.c_str(),R.gate?"PASS":"FAIL");std::printf("SST_G1_GATE_STATUS=%s\n",R.gate?"PASS":"FAIL");return R;
}

} // namespace hxt5c
