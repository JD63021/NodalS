#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// G1: frozen-flow Menter SST-2003m transport equations.
// Momentum remains the converged HXT5B Nikuradse/Spalding solution; this file
// only advances k and omega and diagnoses the resulting SST eddy viscosity.
namespace hxt5c {
using Real=hxt4b::Real;
using nodals_hxt1::Dev;

static constexpr double BETA_STAR=0.09;
static constexpr double A1=0.31;
static constexpr double SIGMA_K1=0.85;
static constexpr double SIGMA_K2=1.00;
static constexpr double SIGMA_W1=0.50;
static constexpr double SIGMA_W2=0.856;
static constexpr double BETA1=0.075;
static constexpr double BETA2=0.0828;
static constexpr double GAMMA1=5.0/9.0;
static constexpr double GAMMA2=0.44;
static constexpr double C_MU=0.09;
static constexpr double KAPPA_OMEGA_WALL=0.41;

enum Equation : int { K_EQ=0, OMEGA_EQ=1 };
enum OmegaWallMode : int { OMEGA_WALL_TRACE_LOG=0, OMEGA_WALL_SAMPLE_LOG=1, OMEGA_WALL_OF_AUTO=2 };
inline const char* omega_wall_mode_name(int m){return m==OMEGA_WALL_OF_AUTO?"of_auto":(m==OMEGA_WALL_SAMPLE_LOG?"sample_log":"trace_log");}

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
  int linearMax=12;
  double gsOmega=1.0;
  int printEvery=10;
  double kFloor=1.0e-8;
  double omegaFloor=1.0e-3;
  double wallPenaltyGamma=50.0;
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
};

struct LocalSST {
  Real nut=0,F1=0,F2=0,sigmaK=0,sigmaW=0,beta=0,gamma=0;
  Real pk=0,srcOmega=0,reactK=0,reactOmega=0;
  bool kFloored=false,omegaFloored=false;
};

__device__ __forceinline__ LocalSST local_sst(
    Real x,Real y,Real radius,Real nu,
    const Real gu[9],Real kRaw,Real omegaRaw,
    const Real gk[3],const Real gw[3],Real kFloor,Real omegaFloor)
{
  LocalSST q;
  q.kFloored=!(kRaw>kFloor);q.omegaFloored=!(omegaRaw>omegaFloor);
  const Real k=fmax(kRaw,kFloor),om=fmax(omegaRaw,omegaFloor);
  Real ss=0;
  #pragma unroll
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){Real sij=Real(0.5)*(gu[3*i+j]+gu[3*j+i]);ss+=sij*sij;}
  const Real S2=fmax(Real(0),Real(2)*ss),S=sqrt(S2);
  const Real rr=sqrt(x*x+y*y);
  const Real d=fmax(radius-rr,fmax(Real(1e-10)*radius,Real(1e-12)));
  const Real dotkw=gk[0]*gw[0]+gk[1]*gw[1]+gk[2]*gw[2];
  const Real CD=fmax(Real(2*SIGMA_W2)*dotkw/om,Real(1e-10));
  const Real sqk=sqrt(k),d2=d*d;
  const Real a11=sqk/(Real(BETA_STAR)*om*d);
  const Real a12=Real(500)*nu/(d2*om);
  const Real a13=Real(4*SIGMA_W2)*k/(CD*d2);
  const Real arg1=fmin(fmax(a11,a12),a13);
  const Real arg2=fmax(Real(2)*a11,a12);
  const Real a1sq=arg1*arg1;
  q.F1=tanh(a1sq*a1sq);q.F2=tanh(arg2*arg2);
  q.sigmaK=q.F1*Real(SIGMA_K1)+(Real(1)-q.F1)*Real(SIGMA_K2);
  q.sigmaW=q.F1*Real(SIGMA_W1)+(Real(1)-q.F1)*Real(SIGMA_W2);
  q.beta=q.F1*Real(BETA1)+(Real(1)-q.F1)*Real(BETA2);
  q.gamma=q.F1*Real(GAMMA1)+(Real(1)-q.F1)*Real(GAMMA2);
  q.nut=Real(A1)*k/fmax(Real(A1)*om,S*q.F2);
  q.nut=fmax(q.nut,Real(0));
  const Real prodRaw=q.nut*S2;
  const Real prodLim=Real(10*BETA_STAR)*k*om;
  q.pk=fmin(prodRaw,prodLim);
  q.reactK=Real(BETA_STAR)*om;
  q.reactOmega=q.beta*om; // beta*omega_old * omega_new
  Real omegaProd;
  const Real nutTiny=fmax(Real(1e-12)*nu,Real(1e-20));
  if(q.nut>nutTiny)omegaProd=q.gamma*q.pk/q.nut;
  else omegaProd=q.gamma*S2;
  const Real cross=Real(2*SIGMA_W2)*(Real(1)-q.F1)*dotkw/om;
  q.srcOmega=omegaProd+cross;
  return q;
}

__global__ void zero_real(std::size_t n,Real*x){std::size_t i=(std::size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]=Real(0);}
__global__ void combine_scalar(std::size_t n,const Real*base,const Real*conv,const Real*vol,const Real*dg,const Real*wall,Real*val){std::size_t i=(std::size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)val[i]=base[i]+conv[i]+vol[i]+dg[i]+wall[i];}
__global__ void relax_field(int n,const Real*old,Real alpha,Real*x){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)x[i]=old[i]+alpha*(x[i]-old[i]);}

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
        Real pa=(Real)ph[a];PH[iq][a]=pa;kv+=K[a]*pa;wv+=W[a]*pa;
        for(int d=0;d<3;++d){Real gd=(Real)gp[a][d];GP[iq][a][d]=gd;gk[d]+=K[a]*gd;gw[d]+=W[a]*gd;gu[d]+=U0[a]*gd;gu[3+d]+=U1[a]*gd;gu[6+d]+=U2[a]*gd;}
      }
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

__global__ void sst_tet_volume(
    const nodals_hxt1::Point*pts,const nodals_hxt1::TetConn*tc,const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,
    const std::int32_t*slot,const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,
    Real*mat,Real*rhs,std::uint64_t n,double radius,double nu,int equation,double kFloor,double omegaFloor)
{
  std::uint64_t c=(std::uint64_t)blockIdx.x;if(c>=n)return;
  __shared__ Real U0[8],U1[8],U2[8],K[8],W[8];
  __shared__ Real PH[64][8],GP[64][8][3],DW[64],RW[64],SW[64];
  if(threadIdx.x<8){int g=tv[c].g[threadIdx.x];U0[threadIdx.x]=u0[g];U1[threadIdx.x]=u1[g];U2[threadIdx.x]=u2[g];K[threadIdx.x]=k[g];W[threadIdx.x]=omega[g];}
  __syncthreads();
  int iq=(int)threadIdx.x;
  if(iq<64){
    int ir=iq>>4,is=(iq>>2)&3,it=iq&3;double r=hxt5a::h5_tet_rn[ir],s0=hxt5a::h5_tet_sn[is],t=hxt5a::h5_tet_tn[it],omr=1-r,oms=1-s0;
    double l[4]={omr*oms*(1-t),r,omr*s0,omr*oms*t};double qw=hxt5a::h5_tet_rw[ir]*hxt5a::h5_tet_sw[is]*hxt5a::h5_tet_tw[it];
    double ph[8],gr[8][3],gp[8][3];hxt2::tet_basis(l[0],l[1],l[2],l[3],ph,gr);hxt2::tet_phys_grad(gr,tg[c],gp);
    Real gu[9]={0,0,0,0,0,0,0,0,0},gk[3]={0,0,0},gw[3]={0,0,0},kv=0,wv=0;
    for(int a=0;a<8;++a){Real pa=(Real)ph[a];PH[iq][a]=pa;kv+=K[a]*pa;wv+=W[a]*pa;for(int d=0;d<3;++d){Real gd=(Real)gp[a][d];GP[iq][a][d]=gd;gk[d]+=K[a]*gd;gw[d]+=W[a]*gd;gu[d]+=U0[a]*gd;gu[3+d]+=U1[a]*gd;gu[6+d]+=U2[a]*gd;}}
    double xq=0,yq=0;for(int a=0;a<4;++a){auto X=pts[tc[c].v[a]];xq+=l[a]*X.x;yq+=l[a]*X.y;}
    LocalSST ss=local_sst((Real)xq,(Real)yq,(Real)radius,(Real)nu,gu,kv,wv,gk,gw,(Real)kFloor,(Real)omegaFloor);Real wt=(Real)(tg[c].det*qw);
    DW[iq]=wt*(equation==K_EQ?ss.sigmaK:ss.sigmaW)*ss.nut;RW[iq]=wt*(equation==K_EQ?ss.reactK:ss.reactOmega);SW[iq]=wt*(equation==K_EQ?ss.pk:ss.srcOmega);
  }
  __syncthreads();
  for(int p=(int)threadIdx.x;p<64;p+=(int)blockDim.x){int a=p/8,b=p%8;Real z=0;for(int q=0;q<64;++q){Real gd=GP[q][a][0]*GP[q][b][0]+GP[q][a][1]*GP[q][b][1]+GP[q][a][2]*GP[q][b][2];z+=DW[q]*gd+RW[q]*PH[q][a]*PH[q][b];}atomicAdd(mat+slot[c*64ull+(std::size_t)p],z);}
  for(int a=(int)threadIdx.x;a<8;a+=(int)blockDim.x){Real z=0;for(int q=0;q<64;++q)z+=SW[q]*PH[q][a];atomicAdd(rhs+tv[c].g[a],z);}
}

__global__ void sst_interface_diffusion(
    const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,const nodals_hxt1::TetConn*tc,
    const nodals_hxt1::HexVel*hv,const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,
    const hxt2::InterfacePlan*P,const std::int64_t*row,const std::int32_t*col,
    const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,Real*mat,std::uint64_t n,
    double radius,double nu,double gammaPenalty,int equation,double kFloor,double omegaFloor,unsigned long long*bad)
{
  std::uint64_t iside=(std::uint64_t)blockIdx.x;if(iside>=2*n)return;
  int ir=(int)(iside>>1),side=(int)(iside&1);const auto&p=P[ir];const auto&HG=hv[p.hexCell];int tcell=p.tetCell[side],tf=p.tetFace[side];const auto&TG=tv[tcell];
  __shared__ int ug[22];__shared__ int un;
  __shared__ Real hu0[14],hu1[14],hu2[14],hk[14],hw[14],tu0[8],tu1[8],tu2[8],tk[8],tw[8];
  __shared__ Real sph[25][14],sgph[25][14][3],spt[25][8],sgpt[25][8][3],sdh[25],sdt[25],swq[25];
  if(threadIdx.x==0){int m=0;for(int a=0;a<14;++a)ug[m++]=HG.g[a];for(int b=0;b<8;++b){int g=TG.g[b],found=0;for(int q=0;q<m;++q)if(ug[q]==g)found=1;if(!found)ug[m++]=g;}un=m;}
  if(threadIdx.x<14){int g=HG.g[threadIdx.x];hu0[threadIdx.x]=u0[g];hu1[threadIdx.x]=u1[g];hu2[threadIdx.x]=u2[g];hk[threadIdx.x]=k[g];hw[threadIdx.x]=omega[g];}
  if(threadIdx.x<8){int g=TG.g[threadIdx.x];tu0[threadIdx.x]=u0[g];tu1[threadIdx.x]=u1[g];tu2[threadIdx.x]=u2[g];tk[threadIdx.x]=k[g];tw[threadIdx.x]=omega[g];}
  __syncthreads();
  for(int q=(int)threadIdx.x;q<25;q+=(int)blockDim.x){
    int ia=q/5,ib=q%5;double rr=hxt2::c_q5_x[ia],ss=hxt2::c_q5_x[ib],om=1-rr,L[3]={om*(1-ss),rr,om*ss};swq[q]=(Real)(hxt2::c_q5_w[ia]*hxt2::c_q5_w[ib]*om*2*p.areaTri[side]);
    double xr=0,yr=0,zr=0;for(int m=0;m<3;++m){int hl=p.hexTriLocal[side][m];xr+=L[m]*hxt4a::h4_hex_sign[hl][0];yr+=L[m]*hxt4a::h4_hex_sign[hl][1];zr+=L[m]*hxt4a::h4_hex_sign[hl][2];}
    double ph[14],grh[14][3],gph[14][3],I[9],det;hxt2::hex_basis_ref(xr,yr,zr,ph,grh);
    if(!hxt2::hex_metric(pts,hc[p.hexCell],xr,yr,zr,I,det)){atomicAdd(bad,1ULL);sdh[q]=sdt[q]=0;continue;}hxt2::phys_grad14(grh,I,gph);
    double lam[4]={0,0,0,0};for(int m=0;m<3;++m)lam[hxt4a::h4_tet_face[tf][m]]=L[m];double pt[8],grt[8][3],gpt[8][3];hxt2::tet_basis(lam[0],lam[1],lam[2],lam[3],pt,grt);hxt2::tet_phys_grad(grt,tg[tcell],gpt);
    Real guh[9]={0,0,0,0,0,0,0,0,0},gut[9]={0,0,0,0,0,0,0,0,0},gkh[3]={0,0,0},gwh[3]={0,0,0},gkt[3]={0,0,0},gwt[3]={0,0,0};Real kvh=0,wvh=0,kvt=0,wvt=0;
    for(int a=0;a<14;++a){sph[q][a]=(Real)ph[a];kvh+=hk[a]*(Real)ph[a];wvh+=hw[a]*(Real)ph[a];for(int d=0;d<3;++d){Real gd=(Real)gph[a][d];sgph[q][a][d]=gd;gkh[d]+=hk[a]*gd;gwh[d]+=hw[a]*gd;guh[d]+=hu0[a]*gd;guh[3+d]+=hu1[a]*gd;guh[6+d]+=hu2[a]*gd;}}
    for(int a=0;a<8;++a){spt[q][a]=(Real)pt[a];kvt+=tk[a]*(Real)pt[a];wvt+=tw[a]*(Real)pt[a];for(int d=0;d<3;++d){Real gd=(Real)gpt[a][d];sgpt[q][a][d]=gd;gkt[d]+=tk[a]*gd;gwt[d]+=tw[a]*gd;gut[d]+=tu0[a]*gd;gut[3+d]+=tu1[a]*gd;gut[6+d]+=tu2[a]*gd;}}
    double xh=0,yh=0;for(int a=0;a<8;++a){double N=.125*(1+hxt4a::h4_hex_sign[a][0]*xr)*(1+hxt4a::h4_hex_sign[a][1]*yr)*(1+hxt4a::h4_hex_sign[a][2]*zr);auto X=pts[hc[p.hexCell].v[a]];xh+=N*X.x;yh+=N*X.y;}
    double xt=0,yt=0;for(int a=0;a<4;++a){auto X=pts[tc[tcell].v[a]];xt+=lam[a]*X.x;yt+=lam[a]*X.y;}
    LocalSST sh=local_sst((Real)xh,(Real)yh,(Real)radius,(Real)nu,guh,kvh,wvh,gkh,gwh,(Real)kFloor,(Real)omegaFloor);LocalSST st=local_sst((Real)xt,(Real)yt,(Real)radius,(Real)nu,gut,kvt,wvt,gkt,gwt,(Real)kFloor,(Real)omegaFloor);
    sdh[q]=(equation==K_EQ?sh.sigmaK:sh.sigmaW)*sh.nut;sdt[q]=(equation==K_EQ?st.sigmaK:st.sigmaW)*st.nut;
  }
  __syncthreads();
  const Real nx=(Real)p.normal[0],ny=(Real)p.normal[1],nz=(Real)p.normal[2];const Real tauBase=(Real)(gammaPenalty*nu*fmax(p.invHHex,p.invHTet[side]));int npair=un*un;
  for(int pair=(int)threadIdx.x;pair<npair;pair+=(int)blockDim.x){int ii=pair/un,jj=pair%un,gi=ug[ii],gj=ug[jj];Real kij=0;
    for(int q=0;q<25;++q){Real ji=0,jjv=0,di=0,dj=0;for(int a=0;a<14;++a){if(HG.g[a]==gi){ji+=sph[q][a];di+=Real(.5)*sdh[q]*(sgph[q][a][0]*nx+sgph[q][a][1]*ny+sgph[q][a][2]*nz);}if(HG.g[a]==gj){jjv+=sph[q][a];dj+=Real(.5)*sdh[q]*(sgph[q][a][0]*nx+sgph[q][a][1]*ny+sgph[q][a][2]*nz);}}
      for(int a=0;a<8;++a){if(TG.g[a]==gi){ji-=spt[q][a];di+=Real(.5)*sdt[q]*(sgpt[q][a][0]*nx+sgpt[q][a][1]*ny+sgpt[q][a][2]*nz);}if(TG.g[a]==gj){jjv-=spt[q][a];dj+=Real(.5)*sdt[q]*(sgpt[q][a][0]*nx+sgpt[q][a][1]*ny+sgpt[q][a][2]*nz);}}
      Real cons=-(dj*ji+di*jjv);Real tauFull=(Real)gammaPenalty*fmax(((Real)nu+sdh[q])*(Real)p.invHHex,((Real)nu+sdt[q])*(Real)p.invHTet[side]);Real deltaTau=fmax(Real(0),tauFull-tauBase);kij+=swq[q]*(cons+deltaTau*ji*jjv);
    }int sl=hxt4b::csr_find(row,col,gi,gj);atomicAdd(mat+sl,kij);
  }
}

__global__ void sst_inlet_hex(
    const hxt5b::HexFacePlan*F,int nf,const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,const nodals_hxt1::HexVel*hv,const std::int32_t*slot,
    const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,Real*mat,Real*rhs,double qIn,double bulk,double radius,double nu,int equation,double kFloor,double omegaFloor,unsigned long long*bad)
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
      Real gu[9]={0,0,0,0,0,0,0,0,0},gk[3]={0,0,0},gw[3]={0,0,0},kv=0,wv=0;for(int a=0;a<14;++a){Real pa=(Real)ph[a];PH[q][a]=pa;kv+=K[a]*pa;wv+=W[a]*pa;for(int d=0;d<3;++d){Real gd=(Real)gp[a][d];GP[q][a][d]=gd;gk[d]+=K[a]*gd;gw[d]+=W[a]*gd;gu[d]+=U0[a]*gd;gu[3+d]+=U1[a]*gd;gu[6+d]+=U2[a]*gd;}}
      LocalSST ss=local_sst((Real)X[0],(Real)X[1],(Real)radius,(Real)nu,gu,kv,wv,gk,gw,(Real)kFloor,(Real)omegaFloor);Real extra=(equation==K_EQ?ss.sigmaK:ss.sigmaW)*ss.nut;NUE[q]=(Real)nu+extra;WQ[q]=(Real)(hxt5a::h5_hex_w[i]*hxt5a::h5_hex_w[j]*jac);
    }
  }__syncthreads();
  Real inflow=fmax(Real(0),Real(-bulk*f.n[2]));
  for(int p=(int)threadIdx.x;p<196;p+=(int)blockDim.x){int a=p/14,b=p%14;Real z=0;for(int q=0;q<16;++q){Real dna=GP[q][a][0]*(Real)f.n[0]+GP[q][a][1]*(Real)f.n[1]+GP[q][a][2]*(Real)f.n[2],dnb=GP[q][b][0]*(Real)f.n[0]+GP[q][b][1]*(Real)f.n[1]+GP[q][b][2]*(Real)f.n[2];z+=WQ[q]*(inflow*PH[q][a]*PH[q][b]+NUE[q]*(-PH[q][a]*dnb+dna*PH[q][b]));}atomicAdd(mat+slot[(std::size_t)f.cell*196ull+(std::size_t)p],z);}
  for(int a=(int)threadIdx.x;a<14;a+=(int)blockDim.x){Real z=0;for(int q=0;q<16;++q){Real dna=GP[q][a][0]*(Real)f.n[0]+GP[q][a][1]*(Real)f.n[1]+GP[q][a][2]*(Real)f.n[2];z+=WQ[q]*(inflow*PH[q][a]*(Real)qIn+NUE[q]*dna*(Real)qIn);}atomicAdd(rhs+H.g[a],z);}
}

__global__ void sst_inlet_tet(
    const hxt5b::TetFacePlan*F,int nf,const nodals_hxt1::Point*pts,const nodals_hxt1::TetConn*tc,const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,const std::int32_t*slot,
    const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,Real*mat,Real*rhs,double qIn,double bulk,double radius,double nu,int equation,double kFloor,double omegaFloor)
{
  int fi=(int)blockIdx.x;if(fi>=nf)return;auto f=F[fi];auto T=tv[f.cell];
  __shared__ Real U0[8],U1[8],U2[8],K[8],W[8],PH[12][8],GP[12][8][3],NUE[12],WQ[12];if(threadIdx.x<8){int g=T.g[threadIdx.x];U0[threadIdx.x]=u0[g];U1[threadIdx.x]=u1[g];U2[threadIdx.x]=u2[g];K[threadIdx.x]=k[g];W[threadIdx.x]=omega[g];}__syncthreads();
  if(threadIdx.x<12){int q=(int)threadIdx.x,a0=hxt4a::h4_tet_face[f.lf][0],a1=hxt4a::h4_tet_face[f.lf][1],a2=hxt4a::h4_tet_face[f.lf][2];double l[4]={0,0,0,0};l[a0]=hxt5b::QL0[q];l[a1]=hxt5b::QL1[q];l[a2]=hxt5b::QL2[q];double ph[8],gr[8][3],gp[8][3];hxt2::tet_basis(l[0],l[1],l[2],l[3],ph,gr);hxt2::tet_phys_grad(gr,tg[f.cell],gp);
    Real gu[9]={0,0,0,0,0,0,0,0,0},gk[3]={0,0,0},gw[3]={0,0,0},kv=0,wv=0;for(int a=0;a<8;++a){Real pa=(Real)ph[a];PH[q][a]=pa;kv+=K[a]*pa;wv+=W[a]*pa;for(int d=0;d<3;++d){Real gd=(Real)gp[a][d];GP[q][a][d]=gd;gk[d]+=K[a]*gd;gw[d]+=W[a]*gd;gu[d]+=U0[a]*gd;gu[3+d]+=U1[a]*gd;gu[6+d]+=U2[a]*gd;}}
    double x=0,y=0;for(int a=0;a<4;++a){auto X=pts[tc[f.cell].v[a]];x+=l[a]*X.x;y+=l[a]*X.y;}LocalSST ss=local_sst((Real)x,(Real)y,(Real)radius,(Real)nu,gu,kv,wv,gk,gw,(Real)kFloor,(Real)omegaFloor);Real extra=(equation==K_EQ?ss.sigmaK:ss.sigmaW)*ss.nut;NUE[q]=(Real)nu+extra;WQ[q]=(Real)(hxt5b::QW12[q]*f.area);
  }__syncthreads();Real inflow=fmax(Real(0),Real(-bulk*f.n[2]));
  for(int p=(int)threadIdx.x;p<64;p+=(int)blockDim.x){int a=p/8,b=p%8;Real z=0;for(int q=0;q<12;++q){Real dna=GP[q][a][0]*(Real)f.n[0]+GP[q][a][1]*(Real)f.n[1]+GP[q][a][2]*(Real)f.n[2],dnb=GP[q][b][0]*(Real)f.n[0]+GP[q][b][1]*(Real)f.n[1]+GP[q][b][2]*(Real)f.n[2];z+=WQ[q]*(inflow*PH[q][a]*PH[q][b]+NUE[q]*(-PH[q][a]*dnb+dna*PH[q][b]));}atomicAdd(mat+slot[(std::size_t)f.cell*64ull+(std::size_t)p],z);}
  for(int a=(int)threadIdx.x;a<8;a+=(int)blockDim.x){Real z=0;for(int q=0;q<12;++q){Real dna=GP[q][a][0]*(Real)f.n[0]+GP[q][a][1]*(Real)f.n[1]+GP[q][a][2]*(Real)f.n[2];z+=WQ[q]*(inflow*PH[q][a]*(Real)qIn+NUE[q]*dna*(Real)qIn);}atomicAdd(rhs+T.g[a],z);}
}

struct OFWallState {
  Real omegaTarget=0,omegaVis=0,omegaLog=0,lamFrac=0,rey=0,yPlusK=0,pkAuto=0,nutSpalding=0;
};

__device__ __forceinline__ OFWallState of_auto_wall_state(
    Real yd,Real nu,Real Ut,Real uTau,Real kRaw,Real omegaRaw,const LocalSST&ss,
    Real kFloor,Real omegaFloor,Real kappa,Real E,Real beta1,Real reBlend,Real prodLimit)
{
  OFWallState q;
  const Real kval=fmax(kRaw,kFloor),om=fmax(omegaRaw,omegaFloor);
  const Real Cmu25=sqrt(sqrt(Real(C_MU))),Cmu5=sqrt(Real(C_MU));
  q.rey=yd*sqrt(kval)/nu;
  q.yPlusK=Cmu25*q.rey;
  q.lamFrac=exp(-q.rey/fmax(reBlend,Real(1e-6)));
  q.lamFrac=fmin(fmax(q.lamFrac,Real(0)),Real(1));
  const Real turbFrac=Real(1)-q.lamFrac;
  const Real magGrad=Ut/fmax(yd,Real(1e-20));
  const Real uStar=sqrt(fmax(Real(0),q.lamFrac*nu*magGrad+turbFrac*Cmu5*kval));
  q.omegaVis=Real(6)*nu/(fmax(beta1,Real(1e-8))*yd*yd);
  q.omegaLog=uStar/(Cmu5*fmax(kappa,Real(1e-6))*yd);
  q.omegaTarget=q.lamFrac*q.omegaVis+turbFrac*q.omegaLog;
  const Real arg=fmax(E*q.yPlusK,Real(1.000001));
  const Real uPlus=log(arg)/fmax(kappa,Real(1e-6));
  Real gLog=ss.pk;
  if(q.yPlusK>Real(1e-8)&&uPlus>Real(1e-8)){
    const Real t=uStar*magGrad*yd/uPlus;
    gLog=t*t/(nu*fmax(kappa,Real(1e-6))*q.yPlusK);
  }
  q.pkAuto=q.lamFrac*ss.pk+turbFrac*gLog;
  const Real lim=fmax(prodLimit,Real(0))*Real(BETA_STAR)*kval*om;
  q.pkAuto=fmin(fmax(q.pkAuto,Real(0)),lim);
  q.nutSpalding=(Ut>Real(1e-12))?fmax(Real(0),uTau*uTau/(Ut/yd)-nu):Real(0);
  return q;
}

__global__ void sst_omega_wall_hex(
    const hxt5b::HexFacePlan*F,int nf,const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,const nodals_hxt1::HexVel*hv,const std::int32_t*slot,
    const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,Real*mat,Real*rhs,double radius,double nu,double sampleFraction,double penaltyGamma,double kFloor,double omegaFloor,
    int wallMode,double ofKappa,double ofE,double ofBeta1,double ofReBlend,double ofProdLimit,double*wallStats,unsigned long long*bad)
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
      for(int a=0;a<14;++a){
        const bool samplePlacement=(wallMode==OMEGA_WALL_SAMPLE_LOG||wallMode==OMEGA_WALL_OF_AUTO);
        PH[q][a]=samplePlacement?(Real)phs[a]:(Real)phw[a];
        for(int d=0;d<3;++d)GP[q][a][d]=(Real)gpw[a][d];
        if(ok){Real ps=(Real)phs[a],pw=(Real)phw[a];kv+=K[a]*ps;wv+=W[a]*ps;ww+=W[a]*pw;us0+=U0[a]*ps;us1+=U1[a]*ps;us2+=U2[a]*ps;slip+=(double)U2[a]*phs[a];for(int d=0;d<3;++d){Real gd=(Real)gps[a][d];gk[d]+=K[a]*gd;gw[d]+=W[a]*gd;gu[d]+=U0[a]*gd;gu[3+d]+=U1[a]*gd;gu[6+d]+=U2[a]*gd;}}
      }
      if(ok&&!hxt5b::spalding(slip,yd,nu,ut,yp,betaWall))ok=false;
      if(ok){
        LocalSST ss=local_sst((Real)Xs[0],(Real)Xs[1],(Real)radius,(Real)nu,gu,kv,wv,gk,gw,(Real)kFloor,(Real)omegaFloor);Real nue=(Real)nu+ss.sigmaW*ss.nut;NUE[q]=nue;TAU[q]=(Real)(penaltyGamma/yd)*nue;
        Real target=(Real)(ut/(sqrt(BETA_STAR)*KAPPA_OMEGA_WALL*yd));
        OFWallState oa;Real Ut=0;
        if(wallMode==OMEGA_WALL_OF_AUTO){Real un=us0*(Real)nr0+us1*(Real)nr1;Ut=sqrt(fmax(Real(0),us0*us0+us1*us1+us2*us2-un*un));oa=of_auto_wall_state((Real)yd,(Real)nu,Ut,(Real)ut,kv,wv,ss,(Real)kFloor,(Real)omegaFloor,(Real)ofKappa,(Real)ofE,(Real)ofBeta1,(Real)ofReBlend,(Real)ofProdLimit);target=oa.omegaTarget;}
        GVAL[q]=target;WQ[q]=(Real)(hxt5a::h5_hex_w[i]*hxt5a::h5_hex_w[j]*jac);WSAMPLE[q]=wv;WTRACE[q]=ww;
        if(wallStats){atomicAdd(wallStats+0,(double)WQ[q]);atomicAdd(wallStats+1,(double)GVAL[q]*(double)WQ[q]);atomicAdd(wallStats+2,yp*(double)WQ[q]);atomicAdd(wallStats+3,(double)WSAMPLE[q]*(double)WQ[q]);atomicAdd(wallStats+4,(double)WTRACE[q]*(double)WQ[q]);if(wallMode==OMEGA_WALL_OF_AUTO){atomicAdd(wallStats+5,(double)oa.rey*(double)WQ[q]);atomicAdd(wallStats+6,(double)oa.lamFrac*(double)WQ[q]);atomicAdd(wallStats+7,(double)oa.omegaVis*(double)WQ[q]);atomicAdd(wallStats+8,(double)oa.omegaLog*(double)WQ[q]);atomicAdd(wallStats+9,(double)ss.pk*(double)WQ[q]);atomicAdd(wallStats+10,(double)oa.pkAuto*(double)WQ[q]);atomicAdd(wallStats+11,(double)(oa.nutSpalding/(Real)nu)*(double)WQ[q]);atomicAdd(wallStats+12,(double)(ss.nut/(Real)nu)*(double)WQ[q]);}}
      }
    }
    if(!ok)atomicAdd(bad,1ULL);
  }
  __syncthreads();
  for(int p=(int)threadIdx.x;p<196;p+=(int)blockDim.x){int a=p/14,b=p%14;Real z=0;for(int q=0;q<16;++q){
    if(wallMode==OMEGA_WALL_SAMPLE_LOG||wallMode==OMEGA_WALL_OF_AUTO){
      z+=WQ[q]*(TAU[q]*PH[q][a]*PH[q][b]);
    }else{
      Real dna=GP[q][a][0]*(Real)f.n[0]+GP[q][a][1]*(Real)f.n[1]+GP[q][a][2]*(Real)f.n[2],dnb=GP[q][b][0]*(Real)f.n[0]+GP[q][b][1]*(Real)f.n[1]+GP[q][b][2]*(Real)f.n[2];
      z+=WQ[q]*(-NUE[q]*PH[q][a]*dnb-NUE[q]*dna*PH[q][b]+TAU[q]*PH[q][a]*PH[q][b]);
    }}atomicAdd(mat+slot[(std::size_t)f.cell*196ull+(std::size_t)p],z);}
  for(int a=(int)threadIdx.x;a<14;a+=(int)blockDim.x){Real z=0;for(int q=0;q<16;++q){
    if(wallMode==OMEGA_WALL_SAMPLE_LOG||wallMode==OMEGA_WALL_OF_AUTO){
      z+=WQ[q]*(TAU[q]*PH[q][a]*GVAL[q]);
    }else{
      Real dna=GP[q][a][0]*(Real)f.n[0]+GP[q][a][1]*(Real)f.n[1]+GP[q][a][2]*(Real)f.n[2];
      z+=WQ[q]*(-NUE[q]*dna*GVAL[q]+TAU[q]*PH[q][a]*GVAL[q]);
    }}atomicAdd(rhs+H.g[a],z);}
}

__global__ void sst_of_wall_production_hex(
    const hxt5b::HexFacePlan*F,int nf,const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,const nodals_hxt1::HexVel*hv,
    const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,Real*rhs,double radius,double nu,double sampleFraction,
    double kFloor,double omegaFloor,double ofKappa,double ofE,double ofBeta1,double ofReBlend,double ofProdLimit,double prodScale,double wallVolumeScale,int equation,unsigned long long*bad)
{
  int fi=(int)blockIdx.x;if(fi>=nf)return;auto f=F[fi];auto H=hv[f.cell];
  __shared__ Real U0[14],U1[14],U2[14],K[14],W[14],PH[16][14],SRC[16],VW[16];
  if(threadIdx.x<14){int g=H.g[threadIdx.x];U0[threadIdx.x]=u0[g];U1[threadIdx.x]=u1[g];U2[threadIdx.x]=u2[g];K[threadIdx.x]=k[g];W[threadIdx.x]=omega[g];}__syncthreads();
  if(threadIdx.x<16){int q=(int)threadIdx.x,i=q>>2,j=q&3;SRC[q]=VW[q]=Real(0);for(int a=0;a<14;++a)PH[q][a]=Real(0);double cw[3]={0,0,0};cw[f.fd]=f.fv;cw[f.d0]=hxt5a::h5_hex_x[i];cw[f.d1]=hxt5a::h5_hex_x[j];double cs[3]={cw[0],cw[1],cw[2]};cs[f.fd]=f.fv+sampleFraction*(-2*f.fv);double phs[14],grs[14][3],gps[14][3],Is[9],dets;hxt2::hex_basis_ref(cs[0],cs[1],cs[2],phs,grs);bool ok=hxt2::hex_metric(pts,hc[f.cell],cs[0],cs[1],cs[2],Is,dets);double Xw[3],dXw[3][3],Xs[3],dXs[3][3];if(ok){hxt2::phys_grad14(grs,Is,gps);hxt5b::q1dev(pts,hc[f.cell],cw[0],cw[1],cw[2],Xw,dXw);hxt5b::q1dev(pts,hc[f.cell],cs[0],cs[1],cs[2],Xs,dXs);}double rr=ok?hypot(Xw[0],Xw[1]):0,jac=0,yd=0,ut=0,yp=0,betaWall=0,slip=0,nr0=0,nr1=0;if(ok&&rr>0){nr0=Xw[0]/rr;nr1=Xw[1]/rr;double nr[3]={nr0,nr1,0};double cr[3]={dXw[f.d0][1]*dXw[f.d1][2]-dXw[f.d0][2]*dXw[f.d1][1],dXw[f.d0][2]*dXw[f.d1][0]-dXw[f.d0][0]*dXw[f.d1][2],dXw[f.d0][0]*dXw[f.d1][1]-dXw[f.d0][1]*dXw[f.d1][0]};if(cr[0]*nr[0]+cr[1]*nr[1]+cr[2]*nr[2]<0){cr[0]*=-1;cr[1]*=-1;cr[2]*=-1;}jac=sqrt(cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]);yd=(Xw[0]-Xs[0])*nr[0]+(Xw[1]-Xs[1])*nr[1];ok=jac>0&&yd>0;}else ok=false;Real gu[9]={0},gk[3]={0},gw[3]={0},kv=0,wv=0,us0=0,us1=0,us2=0;if(ok){for(int a=0;a<14;++a){Real ps=(Real)phs[a];PH[q][a]=ps;kv+=K[a]*ps;wv+=W[a]*ps;us0+=U0[a]*ps;us1+=U1[a]*ps;us2+=U2[a]*ps;slip+=(double)U2[a]*phs[a];for(int d=0;d<3;++d){Real gd=(Real)gps[a][d];gk[d]+=K[a]*gd;gw[d]+=W[a]*gd;gu[d]+=U0[a]*gd;gu[3+d]+=U1[a]*gd;gu[6+d]+=U2[a]*gd;}}if(!hxt5b::spalding(slip,yd,nu,ut,yp,betaWall))ok=false;}if(ok){LocalSST ss=local_sst((Real)Xs[0],(Real)Xs[1],(Real)radius,(Real)nu,gu,kv,wv,gk,gw,(Real)kFloor,(Real)omegaFloor);Real un=us0*(Real)nr0+us1*(Real)nr1;Real Ut=sqrt(fmax(Real(0),us0*us0+us1*us1+us2*us2-un*un));OFWallState oa=of_auto_wall_state((Real)yd,(Real)nu,Ut,(Real)ut,kv,wv,ss,(Real)kFloor,(Real)omegaFloor,(Real)ofKappa,(Real)ofE,(Real)ofBeta1,(Real)ofReBlend,(Real)ofProdLimit);Real dpk=(Real)prodScale*(oa.pkAuto-ss.pk);Real src=dpk;if(equation==OMEGA_EQ){Real nt=fmax(ss.nut,fmax(Real(1e-12)*(Real)nu,Real(1e-20)));src=ss.gamma*dpk/nt;}SRC[q]=src;Real h=(Real)(yd/fmax(sampleFraction,1e-6));VW[q]=(Real)(hxt5a::h5_hex_w[i]*hxt5a::h5_hex_w[j]*jac)*h*(Real)wallVolumeScale;}if(!ok)atomicAdd(bad,1ULL);}
  __syncthreads();for(int a=(int)threadIdx.x;a<14;a+=(int)blockDim.x){Real z=0;for(int q=0;q<16;++q)z+=VW[q]*SRC[q]*PH[q][a];atomicAdd(rhs+H.g[a],z);}
}

__device__ __forceinline__ void atomic_max_double(double*addr,double v){auto*p=(unsigned long long*)addr;unsigned long long old=*p,ass;do{ass=old;if(__longlong_as_double((long long)ass)>=v)break;old=atomicCAS(p,ass,(unsigned long long)__double_as_longlong(v));}while(ass!=old);}

__global__ void sst_hex_stats(const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,const nodals_hxt1::HexVel*hv,const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,std::uint64_t n,double radius,double nu,double kFloor,double omegaFloor,double*stats,unsigned long long*bad){
  std::uint64_t c=(std::uint64_t)blockIdx.x;if(c>=n)return;__shared__ Real U0[14],U1[14],U2[14],K[14],W[14];if(threadIdx.x<14){int g=hv[c].g[threadIdx.x];U0[threadIdx.x]=u0[g];U1[threadIdx.x]=u1[g];U2[threadIdx.x]=u2[g];K[threadIdx.x]=k[g];W[threadIdx.x]=omega[g];}__syncthreads();int iq=(int)threadIdx.x;if(iq>=64)return;int ix=iq>>4,iy=(iq>>2)&3,iz=iq&3;double xr=hxt5a::h5_hex_x[ix],yr=hxt5a::h5_hex_x[iy],zr=hxt5a::h5_hex_x[iz],qw=hxt5a::h5_hex_w[ix]*hxt5a::h5_hex_w[iy]*hxt5a::h5_hex_w[iz];double ph[14],gr[14][3],gp[14][3],I[9],det;hxt2::hex_basis_ref(xr,yr,zr,ph,gr);if(!hxt2::hex_metric(pts,hc[c],xr,yr,zr,I,det)){atomicAdd(bad,1ULL);return;}hxt2::phys_grad14(gr,I,gp);Real gu[9]={0,0,0,0,0,0,0,0,0},gk[3]={0,0,0},gw[3]={0,0,0},kv=0,wv=0;for(int a=0;a<14;++a){Real pa=(Real)ph[a];kv+=K[a]*pa;wv+=W[a]*pa;for(int d=0;d<3;++d){Real gd=(Real)gp[a][d];gk[d]+=K[a]*gd;gw[d]+=W[a]*gd;gu[d]+=U0[a]*gd;gu[3+d]+=U1[a]*gd;gu[6+d]+=U2[a]*gd;}}double x=0,y=0;for(int a=0;a<8;++a){double N=.125*(1+hxt4a::h4_hex_sign[a][0]*xr)*(1+hxt4a::h4_hex_sign[a][1]*yr)*(1+hxt4a::h4_hex_sign[a][2]*zr);auto X=pts[hc[c].v[a]];x+=N*X.x;y+=N*X.y;}LocalSST s=local_sst((Real)x,(Real)y,(Real)radius,(Real)nu,gu,kv,wv,gk,gw,(Real)kFloor,(Real)omegaFloor);double wt=det*qw,rat=(double)s.nut/nu;atomicAdd(stats+0,rat*wt);atomicAdd(stats+1,wt);atomic_max_double(stats+2,rat);atomic_max_double(stats+3,(double)s.F1);atomic_max_double(stats+4,(double)s.F2);if(s.kFloored)atomicAdd(stats+5,wt);if(s.omegaFloored)atomicAdd(stats+6,wt);
}

__global__ void sst_tet_stats(const nodals_hxt1::Point*pts,const nodals_hxt1::TetConn*tc,const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,std::uint64_t n,double radius,double nu,double kFloor,double omegaFloor,double*stats){
  std::uint64_t c=(std::uint64_t)blockIdx.x;if(c>=n)return;__shared__ Real U0[8],U1[8],U2[8],K[8],W[8];if(threadIdx.x<8){int g=tv[c].g[threadIdx.x];U0[threadIdx.x]=u0[g];U1[threadIdx.x]=u1[g];U2[threadIdx.x]=u2[g];K[threadIdx.x]=k[g];W[threadIdx.x]=omega[g];}__syncthreads();int iq=(int)threadIdx.x;if(iq>=64)return;int ir=iq>>4,is=(iq>>2)&3,it=iq&3;double r=hxt5a::h5_tet_rn[ir],s0=hxt5a::h5_tet_sn[is],t=hxt5a::h5_tet_tn[it],omr=1-r,oms=1-s0,l[4]={omr*oms*(1-t),r,omr*s0,omr*oms*t},qw=hxt5a::h5_tet_rw[ir]*hxt5a::h5_tet_sw[is]*hxt5a::h5_tet_tw[it];double ph[8],gr[8][3],gp[8][3];hxt2::tet_basis(l[0],l[1],l[2],l[3],ph,gr);hxt2::tet_phys_grad(gr,tg[c],gp);Real gu[9]={0,0,0,0,0,0,0,0,0},gk[3]={0,0,0},gw[3]={0,0,0},kv=0,wv=0;for(int a=0;a<8;++a){Real pa=(Real)ph[a];kv+=K[a]*pa;wv+=W[a]*pa;for(int d=0;d<3;++d){Real gd=(Real)gp[a][d];gk[d]+=K[a]*gd;gw[d]+=W[a]*gd;gu[d]+=U0[a]*gd;gu[3+d]+=U1[a]*gd;gu[6+d]+=U2[a]*gd;}}double x=0,y=0;for(int a=0;a<4;++a){auto X=pts[tc[c].v[a]];x+=l[a]*X.x;y+=l[a]*X.y;}LocalSST s=local_sst((Real)x,(Real)y,(Real)radius,(Real)nu,gu,kv,wv,gk,gw,(Real)kFloor,(Real)omegaFloor);double wt=tg[c].det*qw,rat=(double)s.nut/nu;atomicAdd(stats+0,rat*wt);atomicAdd(stats+1,wt);atomic_max_double(stats+2,rat);atomic_max_double(stats+3,(double)s.F1);atomic_max_double(stats+4,(double)s.F2);if(s.kFloored)atomicAdd(stats+5,wt);if(s.omegaFloored)atomicAdd(stats+6,wt);
}

// Algebraic row-L1 under-relaxation for advection-dominated SST scalars.
// This mirrors the validated HXT5B momentum policy while preserving the
// fixed point of the physical scalar equation:
//   (A + Delta) q_new = b + Delta q_old,
//   Delta_i = (1/alpha - 1) * sum_j |A_ij|.
// For alpha <= 0.5 this also regularizes a finite raw central-advection
// diagonal for SGS without adding a term to the converged PDE.
__global__ void scalar_row_l1_relax_kernel(
    int n,const std::int64_t*row,const std::int32_t*dp,Real*val,Real*rhs,
    const Real*old,Real alpha,unsigned long long*audit)
{
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i>=n)return;
  const int d=dp[i];const Real d0=val[d];
  if(!(d0>Real(0))||!isfinite((double)d0))atomicAdd(audit+0,1ULL);
  Real l1=Real(0);bool finite=true;
  for(std::int64_t q=row[i];q<row[i+1];++q){Real a=val[q];if(!isfinite((double)a))finite=false;l1+=(Real)fabs((double)a);}
  const Real fac=Real(1)/alpha-Real(1);const Real delta=fac*l1;const Real dn=d0+delta;
  if(!finite||!(l1>Real(0))||!isfinite((double)delta)||!(dn>Real(0))||!isfinite((double)dn)){atomicAdd(audit+1,1ULL);return;}
  val[d]=dn;rhs[i]+=delta*old[i];
}
__global__ void field_diff_kernel(int n,const Real*a,const Real*b,Real*d){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)d[i]=a[i]-b[i];}
__global__ void floor_vertex_kernel(int n,Real floorv,Real*x,unsigned long long*count){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n&&(!(x[i]>=floorv)||!isfinite((double)x[i]))){x[i]=floorv;atomicAdd(count,1ULL);}}

struct LinearResult{int works=0;double r0=0,rn=0,rel=0;bool ok=false;};
inline double scalar_residual(hxt4b::GpuCSR&A,const Dev<unsigned char>&fixed,const Real*rhs,const Real*x,hxt4a::Reducer<Real>&red){A.spmv(x,A.tmp.p);hxt4b::residual_free_kernel<<<(A.n+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(A.n,fixed.p,rhs,A.tmp.p,A.res.p);HXT1_CUDA(cudaGetLastError());return red.norm(A.res.p);}
inline LinearResult scalar_solve(hxt4b::GpuCSR&A,const hxt4b::ColoringHost&C,const Dev<unsigned char>&fixed,const Real*rhs,Real*x,hxt4a::Reducer<Real>&red,const Options&O,int nonlinearIt){LinearResult R;R.r0=scalar_residual(A,fixed,rhs,x,red);R.rn=R.r0;if(R.r0==0){R.ok=true;return R;}double target=std::max(O.linearAtol,O.linearRtol*R.r0);for(int j=0;j<O.linearMax;++j){hxt5b::work("sgs1",nonlinearIt+j,A,C,A.val.p,rhs,x,O.gsOmega,fixed.p);R.works=j+1;R.rn=scalar_residual(A,fixed,rhs,x,red);if(!std::isfinite(R.rn))break;if(R.rn<=target){R.ok=true;break;}}R.rel=R.rn/std::max(R.r0,1e-300);return R;}

struct Stats{double meanNutOverNu=0,maxNutOverNu=0,maxF1=0,maxF2=0,kFloorFrac=0,omegaFloorFrac=0;};
inline Stats collect_stats(std::uint64_t nhex,std::uint64_t ntet,const Dev<nodals_hxt1::Point>&pts,const Dev<nodals_hxt1::HexConn>&hc,const Dev<nodals_hxt1::TetConn>&tc,const Dev<nodals_hxt1::HexVel>&hv,const Dev<nodals_hxt1::TetVel>&tv,const Dev<nodals_hxt1::TetGeom>&tg,const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,double radius,double nu,const Options&O,Dev<double>&stats,Dev<unsigned long long>&bad){HXT1_CUDA(cudaMemset(stats.p,0,stats.n*sizeof(double)));HXT1_CUDA(cudaMemset(bad.p,0,bad.n*sizeof(unsigned long long)));if(nhex)sst_hex_stats<<<nhex,64>>>(pts.p,hc.p,hv.p,u0,u1,u2,k,omega,nhex,radius,nu,O.kFloor,O.omegaFloor,stats.p,bad.p);if(ntet)sst_tet_stats<<<ntet,64>>>(pts.p,tc.p,tv.p,tg.p,u0,u1,u2,k,omega,ntet,radius,nu,O.kFloor,O.omegaFloor,stats.p);HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());unsigned long long nb=0;HXT1_CUDA(cudaMemcpy(&nb,bad.p,sizeof(nb),cudaMemcpyDeviceToHost));if(nb)throw std::runtime_error("SST G1 stats geometry failure count="+std::to_string(nb));std::vector<double>h(7,0);HXT1_CUDA(cudaMemcpy(h.data(),stats.p,7*sizeof(double),cudaMemcpyDeviceToHost));Stats s;if(h[1]>0){s.meanNutOverNu=h[0]/h[1];s.kFloorFrac=h[5]/h[1];s.omegaFloorFrac=h[6]/h[1];}s.maxNutOverNu=h[2];s.maxF1=h[3];s.maxF2=h[4];return s;}

inline void assemble_equation(
    Equation eq,const nodals_hxt1::HostMesh&M,const hxt5b::BoundaryHost&BH,
    const Dev<hxt5b::HexFacePlan>&inH,const Dev<hxt5b::TetFacePlan>&inT,const Dev<hxt5b::HexFacePlan>&wallH,
    const Dev<nodals_hxt1::Point>&pts,const Dev<nodals_hxt1::HexConn>&hc,const Dev<nodals_hxt1::TetConn>&tc,const Dev<nodals_hxt1::HexVel>&hv,const Dev<nodals_hxt1::TetVel>&tv,const Dev<nodals_hxt1::TetGeom>&tg,const Dev<hxt2::InterfacePlan>&ip,
    hxt4b::GpuCSR&A,const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,
    Dev<Real>&vol,Dev<Real>&dg,Dev<Real>&wall,Dev<Real>&rhs,Dev<double>&wallStats,Dev<unsigned long long>&bad,Dev<unsigned long long>&diagAudit,
    double qIn,double bulk,double radius,double nu,double interfaceGamma,double wallSampleFraction,const Options&O,int nonlinearIt)
{
  HXT1_CUDA(cudaMemset(vol.p,0,vol.n*sizeof(Real)));HXT1_CUDA(cudaMemset(dg.p,0,dg.n*sizeof(Real)));HXT1_CUDA(cudaMemset(wall.p,0,wall.n*sizeof(Real)));HXT1_CUDA(cudaMemset(rhs.p,0,rhs.n*sizeof(Real)));HXT1_CUDA(cudaMemset(wallStats.p,0,wallStats.n*sizeof(double)));HXT1_CUDA(cudaMemset(bad.p,0,bad.n*sizeof(unsigned long long)));
  if(M.h.nhex)sst_hex_volume<<<M.h.nhex,128>>>(pts.p,hc.p,hv.p,A.hexSlot.p,u0,u1,u2,k,omega,vol.p,rhs.p,M.h.nhex,radius,nu,(int)eq,O.kFloor,O.omegaFloor,bad.p);
  if(M.h.ntet)sst_tet_volume<<<M.h.ntet,64>>>(pts.p,tc.p,tv.p,tg.p,A.tetSlot.p,u0,u1,u2,k,omega,vol.p,rhs.p,M.h.ntet,radius,nu,(int)eq,O.kFloor,O.omegaFloor);
  if(M.h.ninterface)sst_interface_diffusion<<<2*M.h.ninterface,64>>>(pts.p,hc.p,tc.p,hv.p,tv.p,tg.p,ip.p,A.row.p,A.col.p,u0,u1,u2,k,omega,vol.p,M.h.ninterface,radius,nu,interfaceGamma,(int)eq,O.kFloor,O.omegaFloor,bad.p);
  if(!BH.inH.empty())sst_inlet_hex<<<BH.inH.size(),64>>>(inH.p,(int)BH.inH.size(),pts.p,hc.p,hv.p,A.hexSlot.p,u0,u1,u2,k,omega,dg.p,rhs.p,qIn,bulk,radius,nu,(int)eq,O.kFloor,O.omegaFloor,bad.p);
  if(!BH.inT.empty())sst_inlet_tet<<<BH.inT.size(),64>>>(inT.p,(int)BH.inT.size(),pts.p,tc.p,tv.p,tg.p,A.tetSlot.p,u0,u1,u2,k,omega,dg.p,rhs.p,qIn,bulk,radius,nu,(int)eq,O.kFloor,O.omegaFloor);
  if(eq==OMEGA_EQ&&!BH.wallH.empty())sst_omega_wall_hex<<<BH.wallH.size(),64>>>(wallH.p,(int)BH.wallH.size(),pts.p,hc.p,hv.p,A.hexSlot.p,u0,u1,u2,k,omega,wall.p,rhs.p,radius,nu,wallSampleFraction,O.wallPenaltyGamma,O.kFloor,O.omegaFloor,O.omegaWallMode,O.ofKappa,O.ofE,O.ofBeta1,O.ofReBlend,O.ofProductionLimit,wallStats.p,bad.p);
  if(O.omegaWallMode==OMEGA_WALL_OF_AUTO&&O.ofProduction&&!BH.wallH.empty())sst_of_wall_production_hex<<<BH.wallH.size(),64>>>(wallH.p,(int)BH.wallH.size(),pts.p,hc.p,hv.p,u0,u1,u2,k,omega,rhs.p,radius,nu,wallSampleFraction,O.kFloor,O.omegaFloor,O.ofKappa,O.ofE,O.ofBeta1,O.ofReBlend,O.ofProductionLimit,O.ofProductionScale,O.ofWallVolumeScale,(int)eq,bad.p);
  combine_scalar<<<(A.nnz+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(A.nnz,A.base.p,A.conv.p,vol.p,dg.p,wall.p,A.val.p);
  HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());unsigned long long nb=0;HXT1_CUDA(cudaMemcpy(&nb,bad.p,sizeof(nb),cudaMemcpyDeviceToHost));if(nb)throw std::runtime_error("SST G1 assembly geometry/root failure count="+std::to_string(nb));
  HXT1_CUDA(cudaMemset(diagAudit.p,0,diagAudit.n*sizeof(unsigned long long)));
  const Real alpha=(Real)(eq==K_EQ?O.alphaK:O.alphaOmega);
  scalar_row_l1_relax_kernel<<<(A.n+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(A.n,A.row.p,A.diagPos.p,A.val.p,rhs.p,eq==K_EQ?k:omega,alpha,diagAudit.p);
  HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
  unsigned long long ha[2]={0,0};HXT1_CUDA(cudaMemcpy(ha,diagAudit.p,2*sizeof(unsigned long long),cudaMemcpyDeviceToHost));
  if(nonlinearIt<=5||nonlinearIt%O.printEvery==0||ha[1])std::printf("NODALS_SST_G1_DIAG equation=%s nonlinearIt=%d rawNonpositive=%llu relaxedNonpositive=%llu alpha=%.6g relaxation=ROW_L1_FIXED_POINT status=%s\n",eq==K_EQ?"k":"omega",nonlinearIt,ha[0],ha[1],(double)alpha,ha[1]?"FAIL":"PASS");
  if(ha[1])throw std::runtime_error("SST G1 relaxed scalar diagonal invalid count="+std::to_string(ha[1]));
}

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
  const int n=A.n,nvert=(int)M.h.nv;const double D=2*radius,Lt=O.lengthRatio*D,kIn=1.5*std::pow(O.intensity*bulk,2),omegaIn=std::sqrt(kIn)/(std::pow(C_MU,0.25)*Lt);
  std::vector<Real>hk0((std::size_t)n,0),hw0((std::size_t)n,0);for(int i=0;i<nvert;++i){hk0[(std::size_t)i]=(Real)kIn;hw0[(std::size_t)i]=(Real)omegaIn;}
  Dev<Real>k(hk0),omega(hw0),oldK(hk0),oldW(hw0),rhs(std::vector<Real>((std::size_t)n,0));Dev<double>stats(std::vector<double>(7,0)),wallStats(std::vector<double>(13,0));Dev<unsigned long long>clipK(std::vector<unsigned long long>(1,0)),clipW(std::vector<unsigned long long>(1,0)),diagAudit(std::vector<unsigned long long>(2,0));hxt4a::Reducer<Real>red(n);
  // Frozen velocity: rebuild convection once from the converged G0 field.
  HXT1_CUDA(cudaMemset(A.conv.p,0,A.nnz*sizeof(Real)));HXT1_CUDA(cudaMemset(bad.p,0,bad.n*sizeof(unsigned long long)));
  if(M.h.nhex)hxt4b::hex_conv_assemble<<<M.h.nhex,128>>>(pts.p,hc.p,hv.p,A.hexSlot.p,u0,u1,u2,A.conv.p,M.h.nhex,bad.p);if(M.h.ntet)hxt4b::tet_conv_assemble<<<M.h.ntet,64>>>(tv.p,tg.p,A.tetSlot.p,u0,u1,u2,A.conv.p,M.h.ntet);HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());unsigned long long nb=0;HXT1_CUDA(cudaMemcpy(&nb,bad.p,sizeof(nb),cudaMemcpyDeviceToHost));if(nb)throw std::runtime_error("SST G1 frozen convection geometry failure");
  std::printf("NODALS_SST_G1_CONFIG model=SST-2003m precision=%s coupling=FROZEN_G0_VELOCITY momentumFeedback=OFF wallDistance=ANALYTIC_PIPE_R_MINUS_R turbulenceInlet=DG_TRACE intensity=%.6g lengthScaleOverD=%.6g kIn=%.12e omegaIn=%.12e outlet=NATURAL_ZERO_GRADIENT kWall=HIGH_RE_ZERO_GRADIENT omegaWallMode=%s alphaK=%.6g alphaOmega=%.6g nonlinearTol=%.3e nonlinearMax=%d linear=SGS1 nonlinearRelax=ROW_L1_FIXED_POINT postFieldRelax=OFF linearRtol=%.3e linearMax=%d status=PASS\n",hxt4b::PRECISION,O.intensity,O.lengthRatio,kIn,omegaIn,omega_wall_mode_name(O.omegaWallMode),O.alphaK,O.alphaOmega,O.nonlinearTol,O.nonlinearMax,O.linearRtol,O.linearMax);
  Result R;R.kIn=kIn;R.omegaIn=omegaIn;double prevNut=-1;
  for(int it=1;it<=O.nonlinearMax;++it){
    HXT1_CUDA(cudaMemcpy(oldK.p,k.p,(std::size_t)n*sizeof(Real),cudaMemcpyDeviceToDevice));HXT1_CUDA(cudaMemcpy(oldW.p,omega.p,(std::size_t)n*sizeof(Real),cudaMemcpyDeviceToDevice));
    assemble_equation(K_EQ,M,BH,inH,inT,wallH,pts,hc,tc,hv,tv,tg,ip,A,u0,u1,u2,oldK.p,oldW.p,vol,dg,wall,rhs,wallStats,bad,diagAudit,kIn,bulk,radius,nu,interfaceGamma,wallSampleFraction,O,it);
    auto lk=scalar_solve(A,C,fixed,rhs.p,k.p,red,O,it);HXT1_CUDA(cudaMemset(clipK.p,0,sizeof(unsigned long long)));floor_vertex_kernel<<<(nvert+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(nvert,(Real)O.kFloor,k.p,clipK.p);
    assemble_equation(OMEGA_EQ,M,BH,inH,inT,wallH,pts,hc,tc,hv,tv,tg,ip,A,u0,u1,u2,k.p,oldW.p,vol,dg,wall,rhs,wallStats,bad,diagAudit,omegaIn,bulk,radius,nu,interfaceGamma,wallSampleFraction,O,it);
    auto lw=scalar_solve(A,C,fixed,rhs.p,omega.p,red,O,it);HXT1_CUDA(cudaMemset(clipW.p,0,sizeof(unsigned long long)));floor_vertex_kernel<<<(nvert+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(nvert,(Real)O.omegaFloor,omega.p,clipW.p);HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
    field_diff_kernel<<<(n+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(n,k.p,oldK.p,A.tmp.p);double dk=red.norm(A.tmp.p),nk=red.norm(k.p);field_diff_kernel<<<(n+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(n,omega.p,oldW.p,A.tmp.p);double dw=red.norm(A.tmp.p),nw=red.norm(omega.p);double rk=dk/std::max(nk,1e-300),rw=dw/std::max(nw,1e-300);
    auto ss=collect_stats(M.h.nhex,M.h.ntet,pts,hc,tc,hv,tv,tg,u0,u1,u2,k.p,omega.p,radius,nu,O,stats,bad);double rn=prevNut>0?std::abs(ss.meanNutOverNu-prevNut)/std::max(ss.meanNutOverNu,1e-300):std::numeric_limits<double>::infinity();prevNut=ss.meanNutOverNu;
    std::vector<double>hws(13,0);HXT1_CUDA(cudaMemcpy(hws.data(),wallStats.p,13*sizeof(double),cudaMemcpyDeviceToHost));double wom=hws[0]>0?hws[1]/hws[0]:0,wyp=hws[0]>0?hws[2]/hws[0]:0,wsamp=hws[0]>0?hws[3]/hws[0]:0,wtrace=hws[0]>0?hws[4]/hws[0]:0;
    if(it<=5||it%O.printEvery==0)std::printf("NODALS_SST_OMEGA_WALL mode=%s nonlinearIt=%d targetMean=%.6e sampleMean=%.6e traceMean=%.6e sampleOverTarget=%.6e traceOverTarget=%.6e yPlusMean=%.6e penaltyGamma=%.6e status=PASS\n",omega_wall_mode_name(O.omegaWallMode),it,wom,wsamp,wtrace,wom>0?wsamp/wom:0,wom>0?wtrace/wom:0,wyp,O.wallPenaltyGamma);if((it<=5||it%O.printEvery==0)&&O.omegaWallMode==OMEGA_WALL_OF_AUTO&&hws[0]>0)std::printf("NODALS_SST_OF_AUTO nonlinearIt=%d ReYMean=%.6e lamFracMean=%.6e omegaVisMean=%.6e omegaLogMean=%.6e pkBulkMean=%.6e pkAutoMean=%.6e pkRatio=%.6e nutSpaldingOverNuMean=%.6e nutSSTOverNuSampleMean=%.6e production=%d prodScale=%.6e wallVolumeScale=%.6e status=PASS\n",it,hws[5]/hws[0],hws[6]/hws[0],hws[7]/hws[0],hws[8]/hws[0],hws[9]/hws[0],hws[10]/hws[0],hws[9]!=0?hws[10]/hws[9]:0,hws[11]/hws[0],hws[12]/hws[0],(int)O.ofProduction,O.ofProductionScale,O.ofWallVolumeScale);unsigned long long ck=0,cw=0;HXT1_CUDA(cudaMemcpy(&ck,clipK.p,sizeof(ck),cudaMemcpyDeviceToHost));HXT1_CUDA(cudaMemcpy(&cw,clipW.p,sizeof(cw),cudaMemcpyDeviceToHost));R.vertexKClips+=ck;R.vertexOmegaClips+=cw;
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
