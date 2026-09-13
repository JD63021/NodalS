#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

// G2: fully coupled SST-2003m feedback for the existing HXT5B SIMPLE loop.
// k/omega use the G1 scalar operators.  Momentum keeps the existing DG inlet,
// Spalding wall and pressure coupling; only the turbulent viscosity provider is
// switched from Nikuradse mixing length to SST(k,omega,U).
namespace hxt5d {
using Real=hxt4b::Real;
using nodals_hxt1::Dev;

struct State {
  Dev<Real> k,omega,oldK,oldW,rhs;
  Dev<double> stats,wallStats;
  Dev<unsigned long long> clipK,clipW,diagAudit;
  double kIn=0,omegaIn=0,prevMeanNut=-1;
  unsigned long long totalKClips=0,totalOmegaClips=0;
  State(int n,int nvert,double bulk,double radius,const hxt5c::Options&O){
    const double D=2*radius,Lt=O.lengthRatio*D;
    kIn=1.5*std::pow(O.intensity*bulk,2);
    omegaIn=std::sqrt(kIn)/(std::pow(hxt5c::C_MU,0.25)*Lt);
    std::vector<Real> hk((std::size_t)n,Real(0)),hw((std::size_t)n,Real(0));
    for(int i=0;i<nvert;++i){hk[(std::size_t)i]=(Real)kIn;hw[(std::size_t)i]=(Real)omegaIn;}
    k.upload(hk);omega.upload(hw);oldK.upload(hk);oldW.upload(hw);
    rhs.upload(std::vector<Real>((std::size_t)n,Real(0)));
    stats.upload(std::vector<double>(7,0));wallStats.upload(std::vector<double>(13,0));
    clipK.upload(std::vector<unsigned long long>(1,0));clipW.upload(std::vector<unsigned long long>(1,0));diagAudit.upload(std::vector<unsigned long long>(2,0));
  }
};

struct StepResult {
  double kRel=0,omegaRel=0,nutRel=0,wallOmegaMean=0,wallOmegaSampleMean=0,wallOmegaTraceMean=0,wallYPlusMean=0;
  hxt5c::Stats stats;
  hxt5c::LinearResult kLinear,omegaLinear;
  unsigned long long kClips=0,omegaClips=0;
};

// ---------- SST turbulent diffusion for momentum ----------
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
      for(int a=0;a<14;++a){Real pa=(Real)ph[a];kv+=K[a]*pa;wv+=W[a]*pa;for(int d=0;d<3;++d){Real gd=(Real)gp[a][d];GP[iq][a][d]=gd;gk[d]+=K[a]*gd;gw[d]+=W[a]*gd;gu[d]+=U0[a]*gd;gu[3+d]+=U1[a]*gd;gu[6+d]+=U2[a]*gd;}}
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
  __shared__ Real U0[8],U1[8],U2[8],K[8],W[8],GP[64][8][3],NW[64];
  if(threadIdx.x<8){int g=tv[c].g[threadIdx.x];U0[threadIdx.x]=u0[g];U1[threadIdx.x]=u1[g];U2[threadIdx.x]=u2[g];K[threadIdx.x]=k[g];W[threadIdx.x]=omega[g];}__syncthreads();
  int iq=(int)threadIdx.x;if(iq<64){int ir=iq>>4,is=(iq>>2)&3,it=iq&3;double r=hxt5a::h5_tet_rn[ir],s0=hxt5a::h5_tet_sn[is],t=hxt5a::h5_tet_tn[it],omr=1-r,oms=1-s0,l[4]={omr*oms*(1-t),r,omr*s0,omr*oms*t};double qw=hxt5a::h5_tet_rw[ir]*hxt5a::h5_tet_sw[is]*hxt5a::h5_tet_tw[it];double ph[8],gr[8][3],gp[8][3];hxt2::tet_basis(l[0],l[1],l[2],l[3],ph,gr);hxt2::tet_phys_grad(gr,tg[c],gp);Real gu[9]={0},gk[3]={0},gw[3]={0},kv=0,wv=0;for(int a=0;a<8;++a){Real pa=(Real)ph[a];kv+=K[a]*pa;wv+=W[a]*pa;for(int d=0;d<3;++d){Real gd=(Real)gp[a][d];GP[iq][a][d]=gd;gk[d]+=K[a]*gd;gw[d]+=W[a]*gd;gu[d]+=U0[a]*gd;gu[3+d]+=U1[a]*gd;gu[6+d]+=U2[a]*gd;}}double x=0,y=0;for(int a=0;a<4;++a){auto X=pts[tc[c].v[a]];x+=l[a]*X.x;y+=l[a]*X.y;}auto ss=hxt5c::local_sst((Real)x,(Real)y,(Real)radius,(Real)nu,gu,kv,wv,gk,gw,(Real)kFloor,(Real)omegaFloor);NW[iq]=ss.nut*(Real)(tg[c].det*qw);}__syncthreads();
  for(int p=(int)threadIdx.x;p<64;p+=(int)blockDim.x){int a=p/8,b=p%8;Real v=0;for(int q=0;q<64;++q){Real gd=GP[q][a][0]*GP[q][b][0]+GP[q][a][1]*GP[q][b][1]+GP[q][a][2]*GP[q][b][2];v+=NW[q]*gd;}atomicAdd(turb+slot[c*64ull+(std::size_t)p],v);if(a==b)atomicAdd(diagRTurb+tv[c].g[a],v);}
}

__global__ void sst_mom_interface(
    const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,const nodals_hxt1::TetConn*tc,
    const nodals_hxt1::HexVel*hv,const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,const hxt2::InterfacePlan*P,
    const std::int64_t*row,const std::int32_t*col,const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,
    Real*turb,Real*diagRTurb,std::uint64_t n,double radius,double nu,double gamma,double kFloor,double omegaFloor,unsigned long long*bad)
{
  std::uint64_t iside=(std::uint64_t)blockIdx.x;if(iside>=2*n)return;int ir=(int)(iside>>1),side=(int)(iside&1);const auto&p=P[ir];const auto&HG=hv[p.hexCell];int tcell=p.tetCell[side],tf=p.tetFace[side];const auto&TG=tv[tcell];
  __shared__ int ug[22],un;__shared__ Real HU0[14],HU1[14],HU2[14],HK[14],HW[14],TU0[8],TU1[8],TU2[8],TK[8],TW[8];
  __shared__ Real PH[25][14],GPH[25][14][3],PT[25][8],GPT[25][8][3],NH[25],NT[25],QW[25];
  if(threadIdx.x==0){int m=0;for(int a=0;a<14;++a)ug[m++]=HG.g[a];for(int b=0;b<8;++b){int g=TG.g[b],found=0;for(int q=0;q<m;++q)if(ug[q]==g)found=1;if(!found)ug[m++]=g;}un=m;}
  if(threadIdx.x<14){int g=HG.g[threadIdx.x];HU0[threadIdx.x]=u0[g];HU1[threadIdx.x]=u1[g];HU2[threadIdx.x]=u2[g];HK[threadIdx.x]=k[g];HW[threadIdx.x]=omega[g];}
  if(threadIdx.x<8){int g=TG.g[threadIdx.x];TU0[threadIdx.x]=u0[g];TU1[threadIdx.x]=u1[g];TU2[threadIdx.x]=u2[g];TK[threadIdx.x]=k[g];TW[threadIdx.x]=omega[g];}__syncthreads();
  for(int q=(int)threadIdx.x;q<25;q+=(int)blockDim.x){int ia=q/5,ib=q%5;double rr=hxt2::c_q5_x[ia],ss0=hxt2::c_q5_x[ib],om=1-rr,L[3]={om*(1-ss0),rr,om*ss0};QW[q]=(Real)(hxt2::c_q5_w[ia]*hxt2::c_q5_w[ib]*om*2*p.areaTri[side]);double xr=0,yr=0,zr=0;for(int j=0;j<3;++j){int hl=p.hexTriLocal[side][j];xr+=L[j]*hxt4a::h4_hex_sign[hl][0];yr+=L[j]*hxt4a::h4_hex_sign[hl][1];zr+=L[j]*hxt4a::h4_hex_sign[hl][2];}double ph[14],grh[14][3],gph[14][3],I[9],det;hxt2::hex_basis_ref(xr,yr,zr,ph,grh);if(!hxt2::hex_metric(pts,hc[p.hexCell],xr,yr,zr,I,det)){atomicAdd(bad,1ULL);NH[q]=NT[q]=Real(0);continue;}hxt2::phys_grad14(grh,I,gph);double lam[4]={0,0,0,0};for(int j=0;j<3;++j)lam[hxt4a::h4_tet_face[tf][j]]=L[j];double pt[8],grt[8][3],gpt[8][3];hxt2::tet_basis(lam[0],lam[1],lam[2],lam[3],pt,grt);hxt2::tet_phys_grad(grt,tg[tcell],gpt);
    Real guh[9]={0},gut[9]={0},gkh[3]={0},gwh[3]={0},gkt[3]={0},gwt[3]={0},kh=0,wh=0,kt=0,wt=0;
    for(int a=0;a<14;++a){PH[q][a]=(Real)ph[a];kh+=HK[a]*(Real)ph[a];wh+=HW[a]*(Real)ph[a];for(int d=0;d<3;++d){Real gd=(Real)gph[a][d];GPH[q][a][d]=gd;gkh[d]+=HK[a]*gd;gwh[d]+=HW[a]*gd;guh[d]+=HU0[a]*gd;guh[3+d]+=HU1[a]*gd;guh[6+d]+=HU2[a]*gd;}}
    for(int a=0;a<8;++a){PT[q][a]=(Real)pt[a];kt+=TK[a]*(Real)pt[a];wt+=TW[a]*(Real)pt[a];for(int d=0;d<3;++d){Real gd=(Real)gpt[a][d];GPT[q][a][d]=gd;gkt[d]+=TK[a]*gd;gwt[d]+=TW[a]*gd;gut[d]+=TU0[a]*gd;gut[3+d]+=TU1[a]*gd;gut[6+d]+=TU2[a]*gd;}}
    double xh=0,yh=0;for(int a=0;a<8;++a){double N=.125*(1+hxt4a::h4_hex_sign[a][0]*xr)*(1+hxt4a::h4_hex_sign[a][1]*yr)*(1+hxt4a::h4_hex_sign[a][2]*zr);auto X=pts[hc[p.hexCell].v[a]];xh+=N*X.x;yh+=N*X.y;}double xt=0,yt=0;for(int a=0;a<4;++a){auto X=pts[tc[tcell].v[a]];xt+=lam[a]*X.x;yt+=lam[a]*X.y;}
    NH[q]=hxt5c::local_sst((Real)xh,(Real)yh,(Real)radius,(Real)nu,guh,kh,wh,gkh,gwh,(Real)kFloor,(Real)omegaFloor).nut;NT[q]=hxt5c::local_sst((Real)xt,(Real)yt,(Real)radius,(Real)nu,gut,kt,wt,gkt,gwt,(Real)kFloor,(Real)omegaFloor).nut;
  }
  __syncthreads();Real nx=(Real)p.normal[0],ny=(Real)p.normal[1],nz=(Real)p.normal[2];Real tauBase=(Real)(gamma*nu*fmax(p.invHHex,p.invHTet[side]));int npair=un*un;
  for(int pair=(int)threadIdx.x;pair<npair;pair+=(int)blockDim.x){int ii=pair/un,jj=pair%un,gi=ug[ii],gj=ug[jj];Real kij=0,kiiCons=0;for(int q=0;q<25;++q){Real ji=0,jjv=0,di=0,dj=0;for(int a=0;a<14;++a){if(HG.g[a]==gi){ji+=PH[q][a];di+=Real(.5)*NH[q]*(GPH[q][a][0]*nx+GPH[q][a][1]*ny+GPH[q][a][2]*nz);}if(HG.g[a]==gj){jjv+=PH[q][a];dj+=Real(.5)*NH[q]*(GPH[q][a][0]*nx+GPH[q][a][1]*ny+GPH[q][a][2]*nz);}}for(int a=0;a<8;++a){if(TG.g[a]==gi){ji-=PT[q][a];di+=Real(.5)*NT[q]*(GPT[q][a][0]*nx+GPT[q][a][1]*ny+GPT[q][a][2]*nz);}if(TG.g[a]==gj){jjv-=PT[q][a];dj+=Real(.5)*NT[q]*(GPT[q][a][0]*nx+GPT[q][a][1]*ny+GPT[q][a][2]*nz);}}Real cons=-(dj*ji+di*jjv);Real tauEff=(Real)gamma*fmax(((Real)nu+NH[q])*(Real)p.invHHex,((Real)nu+NT[q])*(Real)p.invHTet[side]);Real deltaTau=fmax(Real(0),tauEff-tauBase);kij+=QW[q]*(cons+deltaTau*ji*jjv);if(ii==jj)kiiCons+=QW[q]*cons;}int sl=hxt4b::csr_find(row,col,gi,gj);atomicAdd(turb+sl,kij);if(ii==jj)atomicAdd(diagRTurb+gi,kiiCons);}
}

inline hxt5c::Stats assemble_momentum_sst(
    const nodals_hxt1::HostMesh&M,const Dev<nodals_hxt1::Point>&pts,const Dev<nodals_hxt1::HexConn>&hc,const Dev<nodals_hxt1::TetConn>&tc,
    const Dev<nodals_hxt1::HexVel>&hv,const Dev<nodals_hxt1::TetVel>&tv,const Dev<nodals_hxt1::TetGeom>&tg,const Dev<hxt2::InterfacePlan>&ip,
    hxt4b::GpuCSR&A,const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,Dev<Real>&turb,Dev<Real>&diagRTurb,
    double radius,double nu,double gamma,const hxt5c::Options&O,Dev<double>&stats,Dev<unsigned long long>&bad)
{
  HXT1_CUDA(cudaMemset(turb.p,0,turb.n*sizeof(Real)));HXT1_CUDA(cudaMemset(diagRTurb.p,0,diagRTurb.n*sizeof(Real)));HXT1_CUDA(cudaMemset(bad.p,0,bad.n*sizeof(unsigned long long)));
  if(M.h.nhex)sst_mom_hex<<<M.h.nhex,128>>>(pts.p,hc.p,hv.p,A.hexSlot.p,u0,u1,u2,k,omega,turb.p,diagRTurb.p,M.h.nhex,radius,nu,O.kFloor,O.omegaFloor,bad.p);
  if(M.h.ntet)sst_mom_tet<<<M.h.ntet,64>>>(pts.p,tc.p,tv.p,tg.p,A.tetSlot.p,u0,u1,u2,k,omega,turb.p,diagRTurb.p,M.h.ntet,radius,nu,O.kFloor,O.omegaFloor);
  if(M.h.ninterface)sst_mom_interface<<<2*M.h.ninterface,64>>>(pts.p,hc.p,tc.p,hv.p,tv.p,tg.p,ip.p,A.row.p,A.col.p,u0,u1,u2,k,omega,turb.p,diagRTurb.p,M.h.ninterface,radius,nu,gamma,O.kFloor,O.omegaFloor,bad.p);
  HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());unsigned long long nb=0;HXT1_CUDA(cudaMemcpy(&nb,bad.p,sizeof(nb),cudaMemcpyDeviceToHost));if(nb)throw std::runtime_error("SST G2 momentum diffusion geometry/interface failure count="+std::to_string(nb));
  return hxt5c::collect_stats(M.h.nhex,M.h.ntet,pts,hc,tc,hv,tv,tg,u0,u1,u2,k,omega,radius,nu,O,stats,bad);
}

// ---------- SST-aware DG velocity inlet; Spalding wall stays unchanged ----------
__global__ void sst_dg_hex(const hxt5b::HexFacePlan*F,int nf,const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,const nodals_hxt1::HexVel*hv,const std::int32_t*slot,const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,Real*mat,Real*diag,Real*rhsz,double bulk,double radius,double nu,double kFloor,double omegaFloor,unsigned long long*bad){
  int fi=blockIdx.x;if(fi>=nf)return;auto f=F[fi];auto H=hv[f.cell];__shared__ Real U0[14],U1[14],U2[14],K[14],W[14],ph[16][14],gp[16][14][3],nue[16],wq[16];if(threadIdx.x<14){int g=H.g[threadIdx.x];U0[threadIdx.x]=u0[g];U1[threadIdx.x]=u1[g];U2[threadIdx.x]=u2[g];K[threadIdx.x]=k[g];W[threadIdx.x]=omega[g];}__syncthreads();
  if(threadIdx.x<16){int q=threadIdx.x,i=q>>2,j=q&3;double c[3]={0,0,0};c[f.fd]=f.fv;c[f.d0]=hxt5a::h5_hex_x[i];c[f.d1]=hxt5a::h5_hex_x[j];double pp[14],gr[14][3],gg[14][3],I[9],det;hxt2::hex_basis_ref(c[0],c[1],c[2],pp,gr);bool ok=hxt2::hex_metric(pts,hc[f.cell],c[0],c[1],c[2],I,det);double X[3]={0},dX[3][3]={{0}};if(ok){hxt2::phys_grad14(gr,I,gg);hxt5b::q1dev(pts,hc[f.cell],c[0],c[1],c[2],X,dX);}double jac=0;if(ok){double cr[3]={dX[f.d0][1]*dX[f.d1][2]-dX[f.d0][2]*dX[f.d1][1],dX[f.d0][2]*dX[f.d1][0]-dX[f.d0][0]*dX[f.d1][2],dX[f.d0][0]*dX[f.d1][1]-dX[f.d0][1]*dX[f.d1][0]};if(cr[0]*f.n[0]+cr[1]*f.n[1]+cr[2]*f.n[2]<0){cr[0]*=-1;cr[1]*=-1;cr[2]*=-1;}jac=sqrt(cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]);ok=jac>0;}Real gu[9]={0},gk[3]={0},gw[3]={0},kv=0,wv=0;for(int a=0;a<14;++a){ph[q][a]=ok?(Real)pp[a]:Real(0);for(int d=0;d<3;++d){Real gd=ok?(Real)gg[a][d]:Real(0);gp[q][a][d]=gd;gk[d]+=K[a]*gd;gw[d]+=W[a]*gd;gu[d]+=U0[a]*gd;gu[3+d]+=U1[a]*gd;gu[6+d]+=U2[a]*gd;}if(ok){kv+=K[a]*(Real)pp[a];wv+=W[a]*(Real)pp[a];}}if(!ok){atomicAdd(bad,1ULL);nue[q]=wq[q]=Real(0);}else{auto ss=hxt5c::local_sst((Real)X[0],(Real)X[1],(Real)radius,(Real)nu,gu,kv,wv,gk,gw,(Real)kFloor,(Real)omegaFloor);nue[q]=(Real)nu+ss.nut;wq[q]=(Real)(hxt5a::h5_hex_w[i]*hxt5a::h5_hex_w[j]*jac);}}__syncthreads();Real inflow=fmax(Real(0),Real(-bulk*f.n[2]));
  for(int p=threadIdx.x;p<196;p+=blockDim.x){int a=p/14,b=p%14;Real z=0;for(int q=0;q<16;++q){Real dna=gp[q][a][0]*f.n[0]+gp[q][a][1]*f.n[1]+gp[q][a][2]*f.n[2],dnb=gp[q][b][0]*f.n[0]+gp[q][b][1]*f.n[1]+gp[q][b][2]*f.n[2];z+=wq[q]*(inflow*ph[q][a]*ph[q][b]+nue[q]*(-ph[q][a]*dnb+dna*ph[q][b]));}atomicAdd(mat+slot[(size_t)f.cell*196+p],z);if(a==b)atomicAdd(diag+H.g[a],z);}for(int a=threadIdx.x;a<14;a+=blockDim.x){Real z=0;for(int q=0;q<16;++q){Real dna=gp[q][a][0]*f.n[0]+gp[q][a][1]*f.n[1]+gp[q][a][2]*f.n[2];z+=wq[q]*(inflow*ph[q][a]*bulk+nue[q]*dna*bulk);}atomicAdd(rhsz+H.g[a],z);}
}

__global__ void sst_dg_tet(const hxt5b::TetFacePlan*F,int nf,const nodals_hxt1::Point*pts,const nodals_hxt1::TetConn*tc,const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,const std::int32_t*slot,const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,Real*mat,Real*diag,Real*rhsz,double bulk,double radius,double nu,double kFloor,double omegaFloor){
  int fi=blockIdx.x;if(fi>=nf)return;auto f=F[fi];auto T=tv[f.cell];__shared__ Real U0[8],U1[8],U2[8],K[8],W[8],ph[12][8],gp[12][8][3],nue[12],wq[12];if(threadIdx.x<8){int g=T.g[threadIdx.x];U0[threadIdx.x]=u0[g];U1[threadIdx.x]=u1[g];U2[threadIdx.x]=u2[g];K[threadIdx.x]=k[g];W[threadIdx.x]=omega[g];}__syncthreads();if(threadIdx.x<12){int q=threadIdx.x,a0=hxt4a::h4_tet_face[f.lf][0],a1=hxt4a::h4_tet_face[f.lf][1],a2=hxt4a::h4_tet_face[f.lf][2];double l[4]={0,0,0,0};l[a0]=hxt5b::QL0[q];l[a1]=hxt5b::QL1[q];l[a2]=hxt5b::QL2[q];double pp[8],gr[8][3],gg[8][3];hxt2::tet_basis(l[0],l[1],l[2],l[3],pp,gr);hxt2::tet_phys_grad(gr,tg[f.cell],gg);Real gu[9]={0},gk[3]={0},gw[3]={0},kv=0,wv=0;double x=0,y=0;for(int a=0;a<8;++a){ph[q][a]=(Real)pp[a];kv+=K[a]*(Real)pp[a];wv+=W[a]*(Real)pp[a];for(int d=0;d<3;++d){Real gd=(Real)gg[a][d];gp[q][a][d]=gd;gk[d]+=K[a]*gd;gw[d]+=W[a]*gd;gu[d]+=U0[a]*gd;gu[3+d]+=U1[a]*gd;gu[6+d]+=U2[a]*gd;}}for(int a=0;a<4;++a){auto P=pts[tc[f.cell].v[a]];x+=l[a]*P.x;y+=l[a]*P.y;}auto ss=hxt5c::local_sst((Real)x,(Real)y,(Real)radius,(Real)nu,gu,kv,wv,gk,gw,(Real)kFloor,(Real)omegaFloor);nue[q]=(Real)nu+ss.nut;wq[q]=(Real)(hxt5b::QW12[q]*f.area);}__syncthreads();Real inflow=fmax(Real(0),Real(-bulk*f.n[2]));for(int p=threadIdx.x;p<64;p+=blockDim.x){int a=p/8,b=p%8;Real z=0;for(int q=0;q<12;++q){Real dna=gp[q][a][0]*f.n[0]+gp[q][a][1]*f.n[1]+gp[q][a][2]*f.n[2],dnb=gp[q][b][0]*f.n[0]+gp[q][b][1]*f.n[1]+gp[q][b][2]*f.n[2];z+=wq[q]*(inflow*ph[q][a]*ph[q][b]+nue[q]*(-ph[q][a]*dnb+dna*ph[q][b]));}atomicAdd(mat+slot[(size_t)f.cell*64+p],z);if(a==b)atomicAdd(diag+T.g[a],z);}for(int a=threadIdx.x;a<8;a+=blockDim.x){Real z=0;for(int q=0;q<12;++q){Real dna=gp[q][a][0]*f.n[0]+gp[q][a][1]*f.n[1]+gp[q][a][2]*f.n[2];z+=wq[q]*(inflow*ph[q][a]*bulk+nue[q]*dna*bulk);}atomicAdd(rhsz+T.g[a],z);}
}

inline void assemble_boundary_sst(const hxt5b::BoundaryHost&H,const Dev<hxt5b::HexFacePlan>&ih,const Dev<hxt5b::TetFacePlan>&it,const Dev<hxt5b::HexFacePlan>&wh,const Dev<nodals_hxt1::Point>&pts,const Dev<nodals_hxt1::HexConn>&hc,const Dev<nodals_hxt1::TetConn>&tc,const Dev<nodals_hxt1::HexVel>&hv,const Dev<nodals_hxt1::TetVel>&tv,const Dev<nodals_hxt1::TetGeom>&tg,hxt4b::GpuCSR&A,const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,Dev<Real>&dg,Dev<Real>&ddg,Dev<Real>&rz,Dev<Real>&wall,Dev<Real>&dwall,Dev<double>&ws,Dev<unsigned long long>&bad,double bulk,double radius,double nu,double sf,const hxt5c::Options&O){
  HXT1_CUDA(cudaMemset(dg.p,0,dg.n*sizeof(Real)));HXT1_CUDA(cudaMemset(ddg.p,0,ddg.n*sizeof(Real)));HXT1_CUDA(cudaMemset(rz.p,0,rz.n*sizeof(Real)));HXT1_CUDA(cudaMemset(wall.p,0,wall.n*sizeof(Real)));HXT1_CUDA(cudaMemset(dwall.p,0,dwall.n*sizeof(Real)));HXT1_CUDA(cudaMemset(ws.p,0,ws.n*sizeof(double)));HXT1_CUDA(cudaMemset(bad.p,0,bad.n*sizeof(unsigned long long)));
  if(!H.inH.empty())sst_dg_hex<<<H.inH.size(),64>>>(ih.p,(int)H.inH.size(),pts.p,hc.p,hv.p,A.hexSlot.p,u0,u1,u2,k,omega,dg.p,ddg.p,rz.p,bulk,radius,nu,O.kFloor,O.omegaFloor,bad.p);
  if(!H.inT.empty())sst_dg_tet<<<H.inT.size(),64>>>(it.p,(int)H.inT.size(),pts.p,tc.p,tv.p,tg.p,A.tetSlot.p,u0,u1,u2,k,omega,dg.p,ddg.p,rz.p,bulk,radius,nu,O.kFloor,O.omegaFloor);
  if(!H.wallH.empty())hxt5b::wall_hex<<<H.wallH.size(),64>>>(wh.p,(int)H.wallH.size(),pts.p,hc.p,hv.p,A.hexSlot.p,u2,wall.p,dwall.p,nu,sf,ws.p,bad.p);
  HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());unsigned long long q=0;HXT1_CUDA(cudaMemcpy(&q,bad.p,sizeof(q),cudaMemcpyDeviceToHost));if(q)throw std::runtime_error("SST G2 momentum boundary failure count="+std::to_string(q));
}

inline StepResult step_turbulence(
    State&S,int outer,const nodals_hxt1::HostMesh&M,const hxt5b::BoundaryHost&BH,
    const Dev<hxt5b::HexFacePlan>&inH,const Dev<hxt5b::TetFacePlan>&inT,const Dev<hxt5b::HexFacePlan>&wallH,
    const Dev<nodals_hxt1::Point>&pts,const Dev<nodals_hxt1::HexConn>&hc,const Dev<nodals_hxt1::TetConn>&tc,const Dev<nodals_hxt1::HexVel>&hv,const Dev<nodals_hxt1::TetVel>&tv,const Dev<nodals_hxt1::TetGeom>&tg,const Dev<hxt2::InterfacePlan>&ip,
    hxt4b::GpuCSR&A,const hxt4b::ColoringHost&C,const Dev<unsigned char>&fixed,const Real*u0,const Real*u1,const Real*u2,
    Dev<Real>&vol,Dev<Real>&dg,Dev<Real>&wall,Dev<unsigned long long>&bad,hxt4a::Reducer<Real>&red,
    double bulk,double radius,double nu,double gamma,double wallSampleFraction,const hxt5c::Options&O)
{
  HXT1_CUDA(cudaMemcpy(S.oldK.p,S.k.p,(std::size_t)A.n*sizeof(Real),cudaMemcpyDeviceToDevice));HXT1_CUDA(cudaMemcpy(S.oldW.p,S.omega.p,(std::size_t)A.n*sizeof(Real),cudaMemcpyDeviceToDevice));
  // scalar advection must follow the corrected current velocity
  HXT1_CUDA(cudaMemset(A.conv.p,0,A.nnz*sizeof(Real)));HXT1_CUDA(cudaMemset(bad.p,0,bad.n*sizeof(unsigned long long)));
  if(M.h.nhex)hxt4b::hex_conv_assemble<<<M.h.nhex,128>>>(pts.p,hc.p,hv.p,A.hexSlot.p,u0,u1,u2,A.conv.p,M.h.nhex,bad.p);if(M.h.ntet)hxt4b::tet_conv_assemble<<<M.h.ntet,64>>>(tv.p,tg.p,A.tetSlot.p,u0,u1,u2,A.conv.p,M.h.ntet);HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());unsigned long long nb=0;HXT1_CUDA(cudaMemcpy(&nb,bad.p,sizeof(nb),cudaMemcpyDeviceToHost));if(nb)throw std::runtime_error("SST G2 scalar convection geometry failure");
  hxt5c::assemble_equation(hxt5c::K_EQ,M,BH,inH,inT,wallH,pts,hc,tc,hv,tv,tg,ip,A,u0,u1,u2,S.oldK.p,S.oldW.p,vol,dg,wall,S.rhs,S.wallStats,bad,S.diagAudit,S.kIn,bulk,radius,nu,gamma,wallSampleFraction,O,outer);
  auto lk=hxt5c::scalar_solve(A,C,fixed,S.rhs.p,S.k.p,red,O,outer);HXT1_CUDA(cudaMemset(S.clipK.p,0,sizeof(unsigned long long)));hxt5c::floor_vertex_kernel<<<(M.h.nv+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>((int)M.h.nv,(Real)O.kFloor,S.k.p,S.clipK.p);
  hxt5c::assemble_equation(hxt5c::OMEGA_EQ,M,BH,inH,inT,wallH,pts,hc,tc,hv,tv,tg,ip,A,u0,u1,u2,S.k.p,S.oldW.p,vol,dg,wall,S.rhs,S.wallStats,bad,S.diagAudit,S.omegaIn,bulk,radius,nu,gamma,wallSampleFraction,O,outer);
  auto lw=hxt5c::scalar_solve(A,C,fixed,S.rhs.p,S.omega.p,red,O,outer);HXT1_CUDA(cudaMemset(S.clipW.p,0,sizeof(unsigned long long)));hxt5c::floor_vertex_kernel<<<(M.h.nv+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>((int)M.h.nv,(Real)O.omegaFloor,S.omega.p,S.clipW.p);HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
  hxt5c::field_diff_kernel<<<(A.n+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(A.n,S.k.p,S.oldK.p,A.tmp.p);double dk=red.norm(A.tmp.p),nk=red.norm(S.k.p);hxt5c::field_diff_kernel<<<(A.n+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(A.n,S.omega.p,S.oldW.p,A.tmp.p);double dw=red.norm(A.tmp.p),nw=red.norm(S.omega.p);
  StepResult R;R.kRel=dk/std::max(nk,1e-300);R.omegaRel=dw/std::max(nw,1e-300);R.stats=hxt5c::collect_stats(M.h.nhex,M.h.ntet,pts,hc,tc,hv,tv,tg,u0,u1,u2,S.k.p,S.omega.p,radius,nu,O,S.stats,bad);R.nutRel=S.prevMeanNut>0?std::abs(R.stats.meanNutOverNu-S.prevMeanNut)/std::max(R.stats.meanNutOverNu,1e-300):std::numeric_limits<double>::infinity();S.prevMeanNut=R.stats.meanNutOverNu;R.kLinear=lk;R.omegaLinear=lw;
  std::vector<double>hws(13,0);HXT1_CUDA(cudaMemcpy(hws.data(),S.wallStats.p,13*sizeof(double),cudaMemcpyDeviceToHost));R.wallOmegaMean=hws[0]>0?hws[1]/hws[0]:0;R.wallYPlusMean=hws[0]>0?hws[2]/hws[0]:0;R.wallOmegaSampleMean=hws[0]>0?hws[3]/hws[0]:0;R.wallOmegaTraceMean=hws[0]>0?hws[4]/hws[0]:0;
  if(outer<=5||outer%O.printEvery==0)std::printf("NODALS_SST_OMEGA_WALL mode=%s outer=%d targetMean=%.6e sampleMean=%.6e traceMean=%.6e sampleOverTarget=%.6e traceOverTarget=%.6e yPlusMean=%.6e penaltyGamma=%.6e status=PASS\n",hxt5c::omega_wall_mode_name(O.omegaWallMode),outer,R.wallOmegaMean,R.wallOmegaSampleMean,R.wallOmegaTraceMean,R.wallOmegaMean>0?R.wallOmegaSampleMean/R.wallOmegaMean:0,R.wallOmegaMean>0?R.wallOmegaTraceMean/R.wallOmegaMean:0,R.wallYPlusMean,O.wallPenaltyGamma);if((outer<=5||outer%O.printEvery==0)&&O.omegaWallMode==hxt5c::OMEGA_WALL_OF_AUTO&&hws[0]>0)std::printf("NODALS_SST_OF_AUTO outer=%d ReYMean=%.6e lamFracMean=%.6e omegaVisMean=%.6e omegaLogMean=%.6e pkBulkMean=%.6e pkAutoMean=%.6e pkRatio=%.6e nutSpaldingOverNuMean=%.6e nutSSTOverNuSampleMean=%.6e production=%d prodScale=%.6e wallVolumeScale=%.6e status=PASS\n",outer,hws[5]/hws[0],hws[6]/hws[0],hws[7]/hws[0],hws[8]/hws[0],hws[9]/hws[0],hws[10]/hws[0],hws[9]!=0?hws[10]/hws[9]:0,hws[11]/hws[0],hws[12]/hws[0],(int)O.ofProduction,O.ofProductionScale,O.ofWallVolumeScale);
  HXT1_CUDA(cudaMemcpy(&R.kClips,S.clipK.p,sizeof(R.kClips),cudaMemcpyDeviceToHost));HXT1_CUDA(cudaMemcpy(&R.omegaClips,S.clipW.p,sizeof(R.omegaClips),cudaMemcpyDeviceToHost));S.totalKClips+=R.kClips;S.totalOmegaClips+=R.omegaClips;
  if(!std::isfinite(R.kRel)||!std::isfinite(R.omegaRel)||!std::isfinite(R.stats.meanNutOverNu)||!std::isfinite(R.stats.maxNutOverNu))throw std::runtime_error("SST G2 nonfinite turbulence update");if(R.stats.meanNutOverNu>1e8||R.stats.maxNutOverNu>1e10)throw std::runtime_error("SST G2 eddy viscosity divergence guard");
  return R;
}

} // namespace hxt5d
