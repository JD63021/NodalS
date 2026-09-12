// NodalS HXT4B
// Mixed HEX(Q1+BF2)/TET(P1+BF3) GPU SIMPLE, real 10D Re=20 pipe.
// Adds lagged central advective convection and production-style fixed-work
// multicolor GS velocity predictor. Pressure remains PCG+Jacobi until HXT4C.
// Compile-time FP32/FP64 state/operator arithmetic, FP64 geometry/reductions.

#define HXT4A_EMBED_MAIN hxt4a_embedded_gate_main
#include "../hybrid_hxt4a/hxt4a_main.cu"
#undef HXT4A_EMBED_MAIN

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace hxt4b {
using nodals_hxt1::Dev;
using nodals_hxt1::HostMesh;
using Real=hxt4a::Real;
static constexpr int TPB=256;
static constexpr const char* PRECISION=hxt4a::PRECISION;

struct CSRHost {
  int n=0;
  std::vector<std::int64_t> row;
  std::vector<std::int32_t> col,diagPos;
  std::vector<std::int32_t> hexSlot,tetSlot;
};

static inline std::uint64_t key32(std::uint32_t i,std::uint32_t j){return (std::uint64_t(i)<<32)|std::uint64_t(j);}

static CSRHost build_velocity_csr(const HostMesh&M,const std::vector<hxt2::InterfacePlan>&IP){
  const int n=(int)M.h.nvel;
  std::vector<std::uint64_t> k;
  const std::size_t estimate=(std::size_t)M.h.nhex*196ull+(std::size_t)M.h.ntet*64ull+(std::size_t)IP.size()*2ull*361ull;
  k.reserve(estimate);
  auto clique=[&](const std::vector<int>&g){for(int a:g)for(int b:g)k.push_back(key32((std::uint32_t)a,(std::uint32_t)b));};
  for(const auto&v:M.hexVel){std::vector<int>g(14);for(int a=0;a<14;++a)g[a]=v.g[a];clique(g);}
  for(const auto&v:M.tetVel){std::vector<int>g(8);for(int a=0;a<8;++a)g[a]=v.g[a];clique(g);}
  for(const auto&p:IP)for(int s=0;s<2;++s){
    std::vector<int>g;g.reserve(22);
    const auto&h=M.hexVel[(std::size_t)p.hexCell];const auto&t=M.tetVel[(std::size_t)p.tetCell[s]];
    for(int a=0;a<14;++a)g.push_back(h.g[a]);
    for(int b=0;b<8;++b)if(std::find(g.begin(),g.end(),t.g[b])==g.end())g.push_back(t.g[b]);
    clique(g);
  }
  std::sort(k.begin(),k.end());k.erase(std::unique(k.begin(),k.end()),k.end());
  CSRHost A;A.n=n;A.row.assign((std::size_t)n+1,0);A.col.resize(k.size());
  for(auto q:k)++A.row[(std::size_t)(std::uint32_t)(q>>32)+1];
  for(int i=0;i<n;++i)A.row[(std::size_t)i+1]+=A.row[(std::size_t)i];
  std::vector<std::int64_t>next=A.row;
  for(auto q:k){int i=(int)(std::uint32_t)(q>>32),j=(int)(std::uint32_t)q;A.col[(std::size_t)next[(std::size_t)i]++]=j;}
  A.diagPos.assign((std::size_t)n,-1);
  for(int i=0;i<n;++i){auto a=A.row[(std::size_t)i],e=A.row[(std::size_t)i+1];auto it=std::lower_bound(A.col.begin()+a,A.col.begin()+e,i);if(it==A.col.begin()+e||*it!=i)throw std::runtime_error("HXT4B CSR missing diagonal");A.diagPos[(std::size_t)i]=(std::int32_t)(it-A.col.begin());}
  auto slot=[&](int i,int j)->std::int32_t{auto a=A.row[(std::size_t)i],e=A.row[(std::size_t)i+1];auto it=std::lower_bound(A.col.begin()+a,A.col.begin()+e,j);if(it==A.col.begin()+e||*it!=j)throw std::runtime_error("HXT4B CSR slot missing");return (std::int32_t)(it-A.col.begin());};
  A.hexSlot.resize((std::size_t)M.h.nhex*196ull);
  for(std::size_t c=0;c<M.hexVel.size();++c)for(int a=0;a<14;++a)for(int b=0;b<14;++b)A.hexSlot[c*196ull+(std::size_t)a*14+b]=slot(M.hexVel[c].g[a],M.hexVel[c].g[b]);
  A.tetSlot.resize((std::size_t)M.h.ntet*64ull);
  for(std::size_t c=0;c<M.tetVel.size();++c)for(int a=0;a<8;++a)for(int b=0;b<8;++b)A.tetSlot[c*64ull+(std::size_t)a*8+b]=slot(M.tetVel[c].g[a],M.tetVel[c].g[b]);
  return A;
}

struct ColoringHost {int ncolors=0,maxColor=0;std::vector<std::int32_t>rows,off;};
static ColoringHost color_free_rows(const CSRHost&A,const std::vector<unsigned char>&fixed){
  std::vector<int>color((std::size_t)A.n,-1),mark(128,-1);int nc=0,freeN=0;
  for(int i=0;i<A.n;++i){
    if(fixed[(std::size_t)i])continue;
    ++freeN;
    for(auto q=A.row[(std::size_t)i];q<A.row[(std::size_t)i+1];++q){int j=A.col[(std::size_t)q];if(j==i||j>i||fixed[(std::size_t)j])continue;int c=color[(std::size_t)j];if(c>=0){if(c>=(int)mark.size())mark.resize((std::size_t)c+32,-1);mark[(std::size_t)c]=i;}}
    int c=0;while(c<(int)mark.size()&&mark[(std::size_t)c]==i)++c;if(c>=(int)mark.size())mark.resize((std::size_t)c+32,-1);color[(std::size_t)i]=c;nc=std::max(nc,c+1);
  }
  ColoringHost C;C.ncolors=nc;C.off.assign((std::size_t)nc+1,0);
  for(int i=0;i<A.n;++i)if(color[(std::size_t)i]>=0)++C.off[(std::size_t)color[(std::size_t)i]+1];
  for(int c=0;c<nc;++c)C.off[(std::size_t)c+1]+=C.off[(std::size_t)c];
  C.rows.resize((std::size_t)freeN);auto next=C.off;
  for(int i=0;i<A.n;++i)if(color[(std::size_t)i]>=0)C.rows[(std::size_t)next[(std::size_t)color[(std::size_t)i]]++]=i;
  for(int c=0;c<nc;++c)C.maxColor=std::max(C.maxColor,(int)(C.off[(std::size_t)c+1]-C.off[(std::size_t)c]));
  // audit symmetric conflict
  for(int i=0;i<A.n;++i)if(color[(std::size_t)i]>=0)for(auto q=A.row[(std::size_t)i];q<A.row[(std::size_t)i+1];++q){int j=A.col[(std::size_t)q];if(j!=i&&!fixed[(std::size_t)j]&&color[(std::size_t)j]==color[(std::size_t)i])throw std::runtime_error("HXT4B coloring conflict");}
  return C;
}

__device__ __forceinline__ int csr_find(const std::int64_t*row,const std::int32_t*col,int i,int j){
  std::int64_t a=row[i],b=row[i+1];while(a<b){auto m=(a+b)>>1;int v=col[m];if(v<j)a=m+1;else b=m;}return (int)a;
}

__global__ void hex_diff_assemble(const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,const nodals_hxt1::HexVel*hv,const std::int32_t*slot,Real*base,Real*diagR,std::uint64_t n,double nu,unsigned long long*bad){
  std::uint64_t c=blockIdx.x;if(c>=n)return;int p=threadIdx.x;if(p>=196)return;int a=p/14,b=p%14;double kab=0;
  for(int ix=0;ix<3;++ix)for(int iy=0;iy<3;++iy)for(int iz=0;iz<3;++iz){double x=hxt2::c_g3_x[ix],y=hxt2::c_g3_x[iy],z=hxt2::c_g3_x[iz],w=hxt2::c_g3_w[ix]*hxt2::c_g3_w[iy]*hxt2::c_g3_w[iz];double ph[14],gr[14][3],gp[14][3],I[9],det;hxt2::hex_basis_ref(x,y,z,ph,gr);if(!hxt2::hex_metric(pts,hc[c],x,y,z,I,det)){atomicAdd(bad,1ULL);return;}hxt2::phys_grad14(gr,I,gp);kab+=nu*det*w*(gp[a][0]*gp[b][0]+gp[a][1]*gp[b][1]+gp[a][2]*gp[b][2]);}
  atomicAdd(base+slot[c*196ull+p],(Real)kab);if(a==b)atomicAdd(diagR+hv[c].g[a],(Real)kab);
}

__global__ void tet_diff_assemble(const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,const std::int32_t*slot,Real*base,Real*diagR,std::uint64_t n,double nu){
  std::uint64_t c=blockIdx.x;if(c>=n)return;int p=threadIdx.x;if(p>=64)return;int a=p/8,b=p%8;const auto&g=tg[c];double metric[3][3]={{0}};for(int j=0;j<3;++j)for(int k=0;k<3;++k)for(int d=0;d<3;++d)metric[j][k]+=g.invJ[3*j+d]*g.invJ[3*k+d];double q=0;for(int j=0;j<3;++j)for(int k=0;k<3;++k)q+=hxt2::c_tet_T[(((a*8+b)*3+j)*3+k)]*metric[j][k];q*=nu*g.det;atomicAdd(base+slot[c*64ull+p],(Real)q);if(a==b)atomicAdd(diagR+tv[c].g[a],(Real)q);
}

__global__ void interface_assemble(const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,const nodals_hxt1::HexVel*hv,const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,const hxt2::InterfacePlan*P,const std::int64_t*row,const std::int32_t*col,Real*base,Real*diagR,std::uint64_t n,double nu,double gamma,unsigned long long*bad){
  std::uint64_t iside=blockIdx.x;if(iside>=2*n)return;int ir=(int)(iside>>1),side=(int)(iside&1);const auto&p=P[ir];const auto&HG=hv[p.hexCell];const auto&TG=tv[p.tetCell[side]];
  __shared__ int ug[22];__shared__ int un;
  if(threadIdx.x==0){int m=0;for(int a=0;a<14;++a)ug[m++]=HG.g[a];for(int b=0;b<8;++b){int g=TG.g[b],found=0;for(int q=0;q<m;++q)if(ug[q]==g)found=1;if(!found)ug[m++]=g;}un=m;}__syncthreads();
  const double nx=p.normal[0],ny=p.normal[1],nz=p.normal[2],tau=gamma*nu*fmax(p.invHHex,p.invHTet[side]);int tcell=p.tetCell[side],tf=p.tetFace[side];
  const int npair=un*un;
  for(int pair=(int)threadIdx.x;pair<npair;pair+=(int)blockDim.x){
    int ii=pair/un,jj=pair%un,gi=ug[ii],gj=ug[jj];double kij=0,kiiCons=0;
    for(int ia=0;ia<5;++ia)for(int ib=0;ib<5;++ib){double rr=hxt2::c_q5_x[ia],ss=hxt2::c_q5_x[ib],om=1-rr,L[3]={om*(1-ss),rr,om*ss};double w=hxt2::c_q5_w[ia]*hxt2::c_q5_w[ib]*om*2*p.areaTri[side];double xr=0,yr=0,zr=0;for(int q=0;q<3;++q){int hl=p.hexTriLocal[side][q];xr+=L[q]*hxt4a::h4_hex_sign[hl][0];yr+=L[q]*hxt4a::h4_hex_sign[hl][1];zr+=L[q]*hxt4a::h4_hex_sign[hl][2];}
      double ph[14],grh[14][3],gph[14][3],I[9],det;hxt2::hex_basis_ref(xr,yr,zr,ph,grh);if(!hxt2::hex_metric(pts,hc[p.hexCell],xr,yr,zr,I,det)){atomicAdd(bad,1ULL);continue;}hxt2::phys_grad14(grh,I,gph);double lam[4]={0,0,0,0};for(int q=0;q<3;++q)lam[hxt4a::h4_tet_face[tf][q]]=L[q];double pt[8],grt[8][3],gpt[8][3];hxt2::tet_basis(lam[0],lam[1],lam[2],lam[3],pt,grt);hxt2::tet_phys_grad(grt,tg[tcell],gpt);
      double ji=0,di=0,jjv=0,dj=0;
      for(int aa=0;aa<14;++aa){
        if(HG.g[aa]==gi){ji+=ph[aa];di+=0.5*nu*(gph[aa][0]*nx+gph[aa][1]*ny+gph[aa][2]*nz);}
        if(HG.g[aa]==gj){jjv+=ph[aa];dj+=0.5*nu*(gph[aa][0]*nx+gph[aa][1]*ny+gph[aa][2]*nz);}
      }
      for(int bb=0;bb<8;++bb){
        if(TG.g[bb]==gi){ji-=pt[bb];di+=0.5*nu*(gpt[bb][0]*nx+gpt[bb][1]*ny+gpt[bb][2]*nz);}
        if(TG.g[bb]==gj){jjv-=pt[bb];dj+=0.5*nu*(gpt[bb][0]*nx+gpt[bb][1]*ny+gpt[bb][2]*nz);}
      }
      double cons=-dj*ji-di*jjv;kij+=w*(cons+tau*ji*jjv);if(ii==jj)kiiCons+=w*cons;
    }
    int sl=csr_find(row,col,gi,gj);atomicAdd(base+sl,(Real)kij);if(ii==jj)atomicAdd(diagR+gi,(Real)kiiCons);
  }
}

__global__ void hex_conv_assemble(const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,const nodals_hxt1::HexVel*hv,const std::int32_t*slot,const Real*u0,const Real*u1,const Real*u2,Real*conv,std::uint64_t n,unsigned long long*bad){
  std::uint64_t c=blockIdx.x;if(c>=n)return;__shared__ Real s0[14],s1[14],s2[14];if(threadIdx.x<14){int g=hv[c].g[threadIdx.x];s0[threadIdx.x]=u0[g];s1[threadIdx.x]=u1[g];s2[threadIdx.x]=u2[g];}__syncthreads();
  for(int pp=(int)threadIdx.x;pp<196;pp+=(int)blockDim.x){int a=pp/14,b=pp%14;double cab=0;
    for(int ix=0;ix<3;++ix)for(int iy=0;iy<3;++iy)for(int iz=0;iz<3;++iz){double x=hxt2::c_g3_x[ix],y=hxt2::c_g3_x[iy],z=hxt2::c_g3_x[iz],w=hxt2::c_g3_w[ix]*hxt2::c_g3_w[iy]*hxt2::c_g3_w[iz];double ph[14],gr[14][3],gp[14][3],I[9],det;hxt2::hex_basis_ref(x,y,z,ph,gr);if(!hxt2::hex_metric(pts,hc[c],x,y,z,I,det)){atomicAdd(bad,1ULL);continue;}hxt2::phys_grad14(gr,I,gp);double uq[3]={0};for(int m=0;m<14;++m){uq[0]+=(double)s0[m]*ph[m];uq[1]+=(double)s1[m]*ph[m];uq[2]+=(double)s2[m]*ph[m];}cab+=det*w*ph[a]*(uq[0]*gp[b][0]+uq[1]*gp[b][1]+uq[2]*gp[b][2]);}
    atomicAdd(conv+slot[c*196ull+pp],(Real)cab);
  }
}

// Reference triple tensor T[a,m,b,j] = int phi_a phi_m dphi_b/dxi_j dxi.
__device__ __constant__ double h4b_tet_convT[8*8*8*3];

__global__ void tet_conv_assemble(const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,const std::int32_t*slot,const Real*u0,const Real*u1,const Real*u2,Real*conv,std::uint64_t n){
  std::uint64_t c=blockIdx.x;if(c>=n)return;__shared__ Real s0[8],s1[8],s2[8];if(threadIdx.x<8){int g=tv[c].g[threadIdx.x];s0[threadIdx.x]=u0[g];s1[threadIdx.x]=u1[g];s2[threadIdx.x]=u2[g];}__syncthreads();int p=threadIdx.x;if(p>=64)return;int a=p/8,b=p%8;const auto&g=tg[c];double q=0;for(int m=0;m<8;++m)for(int j=0;j<3;++j){double ur=(double)s0[m]*g.invJ[3*j+0]+(double)s1[m]*g.invJ[3*j+1]+(double)s2[m]*g.invJ[3*j+2];q+=ur*h4b_tet_convT[(((a*8+m)*8+b)*3+j)];}q*=g.det;atomicAdd(conv+slot[c*64ull+p],(Real)q);
}

__global__ void combine_values_kernel(std::size_t n,const Real*base,const Real*conv,Real*val){std::size_t i=(std::size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)val[i]=base[i]+conv[i];}
__global__ void finalize_rows_kernel(int n,const unsigned char*fixed,const std::int32_t*diagPos,Real*val,const Real*diagRBase,const Real*conv,Real*rau,Real*diagOriginal,double alphaU,double rauScale,unsigned long long*bad){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i>=n)return;int d=diagPos[i];Real orig=val[d];diagOriginal[i]=orig;if(fixed[i]){rau[i]=(Real)0;return;}Real den=diagRBase[i]+conv[d];if(!(orig>(Real)0)||!(den>(Real)0)||!isfinite((double)orig)||!isfinite((double)den)){atomicAdd(bad,1ULL);rau[i]=(Real)0;return;}val[d]=(Real)((double)orig/alphaU);rau[i]=(Real)(rauScale*alphaU/(double)den);}

__global__ void build_momentum_rhs_kernel(int n,const unsigned char*fixed,const Real*bt,const Real*uold,const Real*diagOriginal,double alphaU,Real*rhs){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n){if(fixed[i])rhs[i]=(Real)0;else rhs[i]=bt[i]+(Real)(((1.0-alphaU)/alphaU)*(double)diagOriginal[i]*(double)uold[i]);}}
__global__ void pressure_update_minus_kernel(int n,int pin,Real alpha,const Real*dp,Real*p){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n){if(i==pin)p[i]=(Real)0;else p[i]-=alpha*dp[i];}}

__global__ void mcgs_color_kernel(int begin,int count,const std::int32_t*rows,const std::int64_t*row,const std::int32_t*col,const Real*val,const std::int32_t*diagPos,const Real*rhs,Real*x,double omega){int q=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(q>=count)return;int i=rows[begin+q];Real off=0;for(auto k=row[i];k<row[i+1];++k){int j=col[k];if(j!=i)off+=val[k]*x[j];}Real d=val[diagPos[i]],gs=(rhs[i]-off)/d;x[i]=x[i]+(Real)omega*(gs-x[i]);}

__global__ void csr_spmv_kernel(int n,const std::int64_t*row,const std::int32_t*col,const Real*val,const Real*x,Real*y){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n){Real s=0;for(auto k=row[i];k<row[i+1];++k)s+=val[k]*x[col[k]];y[i]=s;}}
__global__ void residual_free_kernel(int n,const unsigned char*fixed,const Real*b,const Real*Ax,Real*r){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)r[i]=fixed[i]?(Real)0:b[i]-Ax[i];}

struct GpuCSR {
  int n=0;std::size_t nnz=0;Dev<std::int64_t>row;Dev<std::int32_t>col,diagPos,hexSlot,tetSlot,colorRows;std::vector<std::int32_t>colorOff;Dev<Real>base,conv,val,diagRBase,diagOriginal,rau,tmp,res;
  GpuCSR(const CSRHost&H,const ColoringHost&C):n(H.n),nnz(H.col.size()),row(H.row),col(H.col),diagPos(H.diagPos),hexSlot(H.hexSlot),tetSlot(H.tetSlot),colorRows(C.rows),colorOff(C.off),base(std::vector<Real>(H.col.size(),0)),conv(std::vector<Real>(H.col.size(),0)),val(std::vector<Real>(H.col.size(),0)),diagRBase(std::vector<Real>((std::size_t)H.n,0)),diagOriginal(std::vector<Real>((std::size_t)H.n,0)),rau(std::vector<Real>((std::size_t)H.n,0)),tmp(std::vector<Real>((std::size_t)H.n,0)),res(std::vector<Real>((std::size_t)H.n,0)){}
  void spmv(const Real*x,Real*y){csr_spmv_kernel<<<(n+TPB-1)/TPB,TPB>>>(n,row.p,col.p,val.p,x,y);HXT1_CUDA(cudaGetLastError());}
};

static std::array<double,8*8*8*3> build_tet_conv_tensor(){
  const double rn[5]={0.034578939918215090,0.17348032077169567,0.38988638706551931,0.63433347263088680,0.85105421294701644};
  const double rw[5]={0.081764784285771011,0.12619896189991137,0.089200161221590066,0.032055600722961895,0.0041138252030990035};
  const double sn[5]={0.039809857051468722,0.19801341787360821,0.43797481024738616,0.69546427335363614,0.90146491420117358};
  const double sw[5]={0.096781590226651476,0.16717463809436969,0.14638698708466985,0.073908870072616678,0.015747914521692299};
  const double tn[5]={0.046910077030668018,0.23076534494715845,0.5,0.76923465505284150,0.95308992296933198};
  const double tw[5]={0.11846344252809449,0.23931433524968326,0.28444444444444450,0.23931433524968326,0.11846344252809449};
  std::array<double,8*8*8*3>T{};const double gl[4][3]={{-1,-1,-1},{1,0,0},{0,1,0},{0,0,1}};
  for(int ir=0;ir<5;++ir)for(int is=0;is<5;++is)for(int it=0;it<5;++it){double r=rn[ir],s=sn[is],t=tn[it],omr=1-r,oms=1-s,l[4]={omr*oms*(1-t),r,omr*s,omr*oms*t},w=rw[ir]*sw[is]*tw[it];double ph[8],gr[8][3];for(int i=0;i<4;++i){ph[i]=l[i];for(int d=0;d<3;++d)gr[i][d]=gl[i][d];}for(int i=0;i<4;++i){int js[3],kk=0;for(int j=0;j<4;++j)if(j!=i)js[kk++]=j;ph[4+i]=27*l[js[0]]*l[js[1]]*l[js[2]];for(int d=0;d<3;++d)gr[4+i][d]=27*(gl[js[0]][d]*l[js[1]]*l[js[2]]+l[js[0]]*gl[js[1]][d]*l[js[2]]+l[js[0]]*l[js[1]]*gl[js[2]][d]);}for(int a=0;a<8;++a)for(int m=0;m<8;++m)for(int b=0;b<8;++b)for(int j=0;j<3;++j)T[(((a*8+m)*8+b)*3+j)]+=ph[a]*ph[m]*gr[b][j]*w;}
  return T;
}

struct FixedData {std::vector<unsigned char>mask;std::vector<Real>u0,u1,u2;};
static inline double profile(double x,double y,double R,double bulk){double r2=x*x+y*y;return std::max(0.0,2.0*bulk*(1.0-r2/(R*R)));}
static FixedData build_pipe_initial(const HostMesh&M,double R,double bulk){
  const int n=(int)M.h.nvel;FixedData F;F.mask.assign((std::size_t)n,0);F.u0.assign((std::size_t)n,0);F.u1.assign((std::size_t)n,0);F.u2.assign((std::size_t)n,0);
  // Useful initial guess: analytic parabolic values on every shared vertex; enrichments start at zero.
  for(std::size_t v=0;v<M.points.size();++v)F.u2[v]=(Real)profile(M.points[v].x,M.points[v].y,R,bulk);
  auto wall=[&](const nodals_hxt1::BoundaryRec&r){const auto&h=M.hexes[(std::size_t)r.cell];F.mask[(std::size_t)r.bubble]=1;F.u2[(std::size_t)r.bubble]=0;for(int k=0;k<4;++k){int g=h.v[hxt2::HEX_FACE[r.localFace][k]];F.mask[(std::size_t)g]=1;F.u0[(std::size_t)g]=F.u1[(std::size_t)g]=F.u2[(std::size_t)g]=0;}};
  for(const auto&r:M.wallHex)wall(r);
  auto inletHex=[&](const nodals_hxt1::BoundaryRec&r){const auto&h=M.hexes[(std::size_t)r.cell];double av=0,cx=0,cy=0;for(int k=0;k<4;++k){int g=h.v[hxt2::HEX_FACE[r.localFace][k]];double q=profile(M.points[(std::size_t)g].x,M.points[(std::size_t)g].y,R,bulk);F.mask[(std::size_t)g]=1;F.u2[(std::size_t)g]=(Real)q;av+=.25*q;cx+=.25*M.points[(std::size_t)g].x;cy+=.25*M.points[(std::size_t)g].y;}F.mask[(std::size_t)r.bubble]=1;F.u2[(std::size_t)r.bubble]=(Real)(profile(cx,cy,R,bulk)-av);};
  auto inletTet=[&](const nodals_hxt1::BoundaryRec&r){const auto&t=M.tets[(std::size_t)r.cell];double av=0,cx=0,cy=0;for(int k=0;k<3;++k){int g=t.v[hxt2::TET_FACE[r.localFace][k]];double q=profile(M.points[(std::size_t)g].x,M.points[(std::size_t)g].y,R,bulk);F.mask[(std::size_t)g]=1;F.u2[(std::size_t)g]=(Real)q;av+=q/3;cx+=M.points[(std::size_t)g].x/3;cy+=M.points[(std::size_t)g].y/3;}F.mask[(std::size_t)r.bubble]=1;F.u2[(std::size_t)r.bubble]=(Real)(profile(cx,cy,R,bulk)-av);};
  for(const auto&r:M.inHex)inletHex(r);for(const auto&r:M.inTet)inletTet(r);
  return F;
}

static void do_sweep(GpuCSR&G,const ColoringHost&C,const Real*rhs,Real*x,double omega,bool forward){
  if(forward){for(int c=0;c<C.ncolors;++c){int b=C.off[(std::size_t)c],n=C.off[(std::size_t)c+1]-b;mcgs_color_kernel<<<(n+TPB-1)/TPB,TPB>>>(b,n,G.colorRows.p,G.row.p,G.col.p,G.val.p,G.diagPos.p,rhs,x,omega);}}
  else {for(int c=C.ncolors-1;c>=0;--c){int b=C.off[(std::size_t)c],n=C.off[(std::size_t)c+1]-b;mcgs_color_kernel<<<(n+TPB-1)/TPB,TPB>>>(b,n,G.colorRows.p,G.row.p,G.col.p,G.val.p,G.diagPos.p,rhs,x,omega);}}
  HXT1_CUDA(cudaGetLastError());
}
static void momentum_work(const std::string&mode,int outer,GpuCSR&G,const ColoringHost&C,const Real*rhs,Real*x,double omega){if(mode=="fgs1")do_sweep(G,C,rhs,x,omega,true);else if(mode=="altgs1")do_sweep(G,C,rhs,x,omega,(outer&1)!=0);else if(mode=="sgs1"){do_sweep(G,C,rhs,x,omega,true);do_sweep(G,C,rhs,x,omega,false);}else throw std::runtime_error("HXT4B momentum-work must be fgs1, altgs1, or sgs1");}

template<class Action>
hxt4a::CGResult pcg_prod(Action&Act,const Real*rhs,Real*x,const Real*invdiag,int n,double rtol,double atol,int maxit,hxt4a::CGWork<Real>&W,int pin){
  Act.apply(x,W.q.p);hxt4a::residual_kernel<Real><<<(n+TPB-1)/TPB,TPB>>>(n,rhs,W.q.p,W.r.p);hxt4a::pin_zero_kernel<Real><<<1,1>>>(pin,W.r.p);double r0=W.red.norm(W.r.p);if(r0==0)return {0,0,true};double target=std::max(atol,rtol*r0);hxt4a::precond_kernel<Real><<<(n+TPB-1)/TPB,TPB>>>(n,invdiag,W.r.p,W.z.p);HXT1_CUDA(cudaMemcpy(W.p.p,W.z.p,n*sizeof(Real),cudaMemcpyDeviceToDevice));double rho=W.red.dot(W.r.p,W.z.p);hxt4a::CGResult R;
  for(int k=0;k<maxit;++k){Act.apply(W.p.p,W.q.p);double pq=W.red.dot(W.p.p,W.q.p);if(!(pq>0))throw std::runtime_error("HXT4B pressure PCG pAp nonpositive");Real a=(Real)(rho/pq);hxt4a::axpy2_kernel<Real><<<(n+TPB-1)/TPB,TPB>>>(n,a,W.p.p,W.q.p,x,W.r.p);hxt4a::pin_zero_kernel<Real><<<1,1>>>(pin,x);hxt4a::pin_zero_kernel<Real><<<1,1>>>(pin,W.r.p);double rn=W.red.norm(W.r.p);R.its=k+1;R.rel=rn/r0;if(rn<=target){R.ok=true;break;}hxt4a::precond_kernel<Real><<<(n+TPB-1)/TPB,TPB>>>(n,invdiag,W.r.p,W.z.p);double nr=W.red.dot(W.r.p,W.z.p);hxt4a::pupdate_kernel<Real><<<(n+TPB-1)/TPB,TPB>>>(n,(Real)(nr/rho),W.z.p,W.p.p);hxt4a::pin_zero_kernel<Real><<<1,1>>>(pin,W.p.p);rho=nr;}
  return R;
}

static void write_vtu(const std::string&path,const HostMesh&M,const std::vector<Real>&u0,const std::vector<Real>&u1,const std::vector<Real>&u2,const std::vector<Real>&p){
  const int nh=(int)M.h.nhex,np=nh+(int)M.h.ntet;std::vector<long double>w(M.points.size(),0),pa(M.points.size(),0);for(int c=0;c<nh;++c){double V=M.hexGeom[(std::size_t)c].volume/8;for(int a=0;a<8;++a){int v=M.hexes[(std::size_t)c].v[a];w[v]+=V;pa[v]+=V*(double)p[(std::size_t)c];}}for(int c=0;c<(int)M.h.ntet;++c){double V=M.tetGeom[(std::size_t)c].volume/4;for(int a=0;a<4;++a){int v=M.tets[(std::size_t)c].v[a];w[v]+=V;pa[v]+=V*(double)p[(std::size_t)(nh+c)];}}
  std::ofstream f(path);if(!f)throw std::runtime_error("cannot write "+path);f<<std::setprecision(16)<<"<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n<UnstructuredGrid><Piece NumberOfPoints=\""<<M.points.size()<<"\" NumberOfCells=\""<<np<<"\">\n<Points><DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";for(const auto&q:M.points)f<<q.x<<" "<<q.y<<" "<<q.z<<"\n";f<<"</DataArray></Points><Cells><DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";for(const auto&h:M.hexes){for(int a=0;a<8;++a)f<<h.v[a]<<" ";f<<"\n";}for(const auto&t:M.tets){for(int a=0;a<4;++a)f<<t.v[a]<<" ";f<<"\n";}f<<"</DataArray><DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";long long o=0;for(int c=0;c<nh;++c){o+=8;f<<o<<"\n";}for(int c=0;c<(int)M.h.ntet;++c){o+=4;f<<o<<"\n";}f<<"</DataArray><DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";for(int c=0;c<nh;++c)f<<"12\n";for(int c=0;c<(int)M.h.ntet;++c)f<<"10\n";f<<"</DataArray></Cells><PointData><DataArray type=\"Float64\" Name=\"velocity_Q1P1_vertex\" NumberOfComponents=\"3\" format=\"ascii\">\n";for(std::size_t i=0;i<M.points.size();++i)f<<(double)u0[i]<<" "<<(double)u1[i]<<" "<<(double)u2[i]<<"\n";f<<"</DataArray><DataArray type=\"Float64\" Name=\"pressure_Q0P0_vertex_volume_average\" format=\"ascii\">\n";for(std::size_t i=0;i<M.points.size();++i)f<<(w[i]>0?(double)(pa[i]/w[i]):0)<<"\n";f<<"</DataArray></PointData><CellData><DataArray type=\"Float64\" Name=\"pressure_Q0P0\" format=\"ascii\">\n";for(auto q:p)f<<(double)q<<"\n";f<<"</DataArray><DataArray type=\"Int32\" Name=\"region\" format=\"ascii\">\n";for(int c=0;c<nh;++c)f<<"1\n";for(int c=0;c<(int)M.h.ntet;++c)f<<"2\n";f<<"</DataArray></CellData></Piece></UnstructuredGrid></VTKFile>\n";
}

} // namespace hxt4b

#ifdef HXT4B_EMBED_MAIN
int HXT4B_EMBED_MAIN(int argc,char**argv){
#else
int main(int argc,char**argv){
#endif
  try{
    using namespace hxt4b;
    std::string mesh,vtu="HXT4B_Re20.vtu",momWork="sgs1";double Re=20,bulk=1,gamma=50,alphaU=.7,alphaP=1,rauScale=2,simpleTol=1e-3,momOmega=1,momRtol=.5,momAtol=1e-8,pRtol=.9,pAtol=1e-12;int maxOuter=3000,pMax=50;
    for(int i=1;i<argc;++i){std::string a=argv[i];auto nx=[&](){if(i+1>=argc)throw std::runtime_error("missing value for "+a);return std::string(argv[++i]);};if(a=="--mesh")mesh=nx();else if(a=="--vtu")vtu=nx();else if(a=="--re")Re=std::stod(nx());else if(a=="--bulk")bulk=std::stod(nx());else if(a=="--gamma")gamma=std::stod(nx());else if(a=="--alpha-u")alphaU=std::stod(nx());else if(a=="--alpha-p")alphaP=std::stod(nx());else if(a=="--rau-scale")rauScale=std::stod(nx());else if(a=="--simple-tol")simpleTol=std::stod(nx());else if(a=="--max-outer")maxOuter=std::stoi(nx());else if(a=="--momentum-work")momWork=nx();else if(a=="--mom-omega")momOmega=std::stod(nx());else if(a=="--mom-rtol")momRtol=std::stod(nx());else if(a=="--mom-atol")momAtol=std::stod(nx());else if(a=="--p-rtol")pRtol=std::stod(nx());else if(a=="--p-atol")pAtol=std::stod(nx());else if(a=="--p-max")pMax=std::stoi(nx());else throw std::runtime_error("unknown arg "+a);}
    if(mesh.empty())throw std::runtime_error("--mesh required");if(!(alphaU>0&&alphaU<=1&&alphaP>0))throw std::runtime_error("bad relaxation");
    HostMesh M=nodals_hxt1::load(mesh);auto IP=hxt2::build_interface_plans(M);auto I3=hxt3a::build_if(M);auto Bh=hxt3a::build_b(M,I3);const int nv=Bh.nv,np=Bh.np,nh=(int)M.h.nhex;
    double zmin=1e300,zmax=-1e300,R=0;for(const auto&q:M.points){zmin=std::min(zmin,q.z);zmax=std::max(zmax,q.z);R=std::max(R,std::sqrt(q.x*q.x+q.y*q.y));}double L=zmax-zmin,D=2*R,nu=bulk*D/Re,dpdzExact=-32*nu*bulk/(D*D),dropExact=-dpdzExact*L;
    if(std::abs(L/D-10)>1e-6)throw std::runtime_error("HXT4B expects 10D HXT1 mesh");

    auto fixed=build_pipe_initial(M,R,bulk);Dev<unsigned char>d_fixed(fixed.mask);Dev<Real>d_u0(fixed.u0),d_u1(fixed.u1),d_u2(fixed.u2);
    auto H=build_velocity_csr(M,IP);auto C=color_free_rows(H,fixed.mask);GpuCSR A(H,C);
    Dev<nodals_hxt1::Point>d_pts(M.points);Dev<nodals_hxt1::HexConn>d_hc(M.hexes);Dev<nodals_hxt1::HexVel>d_hv(M.hexVel);Dev<nodals_hxt1::TetVel>d_tv(M.tetVel);Dev<nodals_hxt1::TetGeom>d_tg(M.tetGeom);Dev<hxt2::InterfacePlan>d_ip(IP);Dev<unsigned long long>d_bad(std::vector<unsigned long long>(1,0));
    auto TT=hxt2::build_tet_tensor();HXT1_CUDA(cudaMemcpyToSymbol(hxt2::c_tet_T,TT.data(),TT.size()*sizeof(double)));auto CT=build_tet_conv_tensor();HXT1_CUDA(cudaMemcpyToSymbol(h4b_tet_convT,CT.data(),CT.size()*sizeof(double)));

    std::printf("NODALS_HXT4B_KERNEL_POLICY interfaceThreads=128 interfacePairTraversal=STRIDED hexConvectionThreads=128 hexPairTraversal=STRIDED status=PASS\n");
    HXT1_CUDA(cudaMemset(A.base.p,0,A.nnz*sizeof(Real)));HXT1_CUDA(cudaMemset(A.diagRBase.p,0,nv*sizeof(Real)));HXT1_CUDA(cudaMemset(d_bad.p,0,sizeof(unsigned long long)));
    if(M.h.nhex)hex_diff_assemble<<<M.h.nhex,256>>>(d_pts.p,d_hc.p,d_hv.p,A.hexSlot.p,A.base.p,A.diagRBase.p,M.h.nhex,nu,d_bad.p);
    if(M.h.ntet)tet_diff_assemble<<<M.h.ntet,64>>>(d_tv.p,d_tg.p,A.tetSlot.p,A.base.p,A.diagRBase.p,M.h.ntet,nu);
    if(M.h.ninterface)interface_assemble<<<2*M.h.ninterface,128>>>(d_pts.p,d_hc.p,d_hv.p,d_tv.p,d_tg.p,d_ip.p,A.row.p,A.col.p,A.base.p,A.diagRBase.p,M.h.ninterface,nu,gamma,d_bad.p);
    HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());unsigned long long bad=0;HXT1_CUDA(cudaMemcpy(&bad,d_bad.p,sizeof(bad),cudaMemcpyDeviceToHost));if(bad)throw std::runtime_error("baseline assembly geometry failures");

    hxt4a::BDevice<Real>B(Bh);int pin=0;if(!M.outHex.empty())pin=M.outHex[0].cell;else if(!M.outTet.empty())pin=nh+M.outTet[0].cell;
    Dev<Real>d_p(std::vector<Real>((std::size_t)np,0)),d_dp(std::vector<Real>((std::size_t)np,0)),d_cont(std::vector<Real>((std::size_t)np,0)),d_sd(std::vector<Real>((std::size_t)np,0)),d_isd(std::vector<Real>((std::size_t)np,0));
    Dev<Real>d_bt0(std::vector<Real>((std::size_t)nv,0)),d_bt1(std::vector<Real>((std::size_t)nv,0)),d_bt2(std::vector<Real>((std::size_t)nv,0)),d_rhs0(std::vector<Real>((std::size_t)nv,0)),d_rhs1(std::vector<Real>((std::size_t)nv,0)),d_rhs2(std::vector<Real>((std::size_t)nv,0));
    hxt4a::Reducer<Real>pred(np),ured(nv);hxt4a::CGWork<Real>WP(np);

    std::printf("NODALS_HXT4B_CONFIG precision=%s geometry=fp64 reductions=fp64 cellsHex=%llu cellsTet=%llu velocityDofs=%d pressureDofs=%d csrNnz=%zu colors=%d maxColor=%d Re=%.12g bulk=%.12g nu=%.12e D=%.12e L=%.12e gamma=%.6g convection=CENTRAL_ADVECTIVE_LAGGED interfaceConvection=VOLUME_ONLY_TANGENTIAL_RE20_GATE momentumWork=%s momOmega=%.6g momRtol=%.3e momAtol=%.3e fixedWorkIgnoresMomTolerances=1 pressureSolver=PCG_JACOBI momentumSign=A_u_MINUS_BT_p pRtol=%.3e pAtol=%.3e pMax=%d alphaU=%.6g alphaP=%.6g rauScale=%.6g rauPenaltyIncluded=0 simpleTol=%.3e status=PASS\n",PRECISION,(unsigned long long)M.h.nhex,(unsigned long long)M.h.ntet,nv,np,A.nnz,C.ncolors,C.maxColor,Re,bulk,nu,D,L,gamma,momWork.c_str(),momOmega,momRtol,momAtol,pRtol,pAtol,pMax,alphaU,alphaP,rauScale,simpleTol);
    std::printf("NODALS_HXT4B_HAGEN_POISEUILLE expectedDpDz=%.12e expectedDropPinMinusPout=%.12e inlet=STRONG_PARABOLIC wall=STRONG_NOSLIP outlet=NATURAL_TRACTION pressureGaugePin=%d status=PASS\n",dpdzExact,dropExact,pin);

    double c0=-1;bool converged=false;int outer=0;long long pIts=0;
    for(int it=1;it<=maxOuter;++it){
      HXT1_CUDA(cudaMemset(A.conv.p,0,A.nnz*sizeof(Real)));HXT1_CUDA(cudaMemset(d_bad.p,0,sizeof(unsigned long long)));
      if(M.h.nhex)hex_conv_assemble<<<M.h.nhex,128>>>(d_pts.p,d_hc.p,d_hv.p,A.hexSlot.p,d_u0.p,d_u1.p,d_u2.p,A.conv.p,M.h.nhex,d_bad.p);
      if(M.h.ntet)tet_conv_assemble<<<M.h.ntet,64>>>(d_tv.p,d_tg.p,A.tetSlot.p,d_u0.p,d_u1.p,d_u2.p,A.conv.p,M.h.ntet);
      combine_values_kernel<<<(A.nnz+TPB-1)/TPB,TPB>>>(A.nnz,A.base.p,A.conv.p,A.val.p);HXT1_CUDA(cudaMemset(d_bad.p,0,sizeof(unsigned long long)));finalize_rows_kernel<<<(nv+TPB-1)/TPB,TPB>>>(nv,d_fixed.p,A.diagPos.p,A.val.p,A.diagRBase.p,A.conv.p,A.rau.p,A.diagOriginal.p,alphaU,rauScale,d_bad.p);HXT1_CUDA(cudaDeviceSynchronize());HXT1_CUDA(cudaMemcpy(&bad,d_bad.p,sizeof(bad),cudaMemcpyDeviceToHost));if(bad)throw std::runtime_error("nonpositive relaxed/rAU diagonal count="+std::to_string(bad));
      hxt4a::schur_diag_kernel<Real><<<(np+TPB-1)/TPB,TPB>>>(np,B.row.p,B.col.p,B.bx.p,B.by.p,B.bz.p,A.rau.p,d_sd.p,pin);hxt4a::invert_diag_kernel<Real><<<(np+TPB-1)/TPB,TPB>>>(np,d_sd.p,d_isd.p,pin);hxt4a::SAction<Real>S(B,A.rau.p,pin);
      hxt4a::BT_apply(B,d_p.p,d_bt0.p,d_bt1.p,d_bt2.p);build_momentum_rhs_kernel<<<(nv+TPB-1)/TPB,TPB>>>(nv,d_fixed.p,d_bt0.p,d_u0.p,A.diagOriginal.p,alphaU,d_rhs0.p);build_momentum_rhs_kernel<<<(nv+TPB-1)/TPB,TPB>>>(nv,d_fixed.p,d_bt1.p,d_u1.p,A.diagOriginal.p,alphaU,d_rhs1.p);build_momentum_rhs_kernel<<<(nv+TPB-1)/TPB,TPB>>>(nv,d_fixed.p,d_bt2.p,d_u2.p,A.diagOriginal.p,alphaU,d_rhs2.p);
      momentum_work(momWork,it,A,C,d_rhs0.p,d_u0.p,momOmega);momentum_work(momWork,it,A,C,d_rhs1.p,d_u1.p,momOmega);momentum_work(momWork,it,A,C,d_rhs2.p,d_u2.p,momOmega);
      hxt4a::B_apply(B,d_u0.p,d_u1.p,d_u2.p,d_cont.p);hxt4a::pin_zero_kernel<Real><<<1,1>>>(pin,d_cont.p);double cn=pred.norm(d_cont.p);if(it==1)c0=cn;double rel=cn/std::max(c0,1e-300);
      HXT1_CUDA(cudaMemset(d_dp.p,0,np*sizeof(Real)));auto pr=pcg_prod(S,d_cont.p,d_dp.p,d_isd.p,np,pRtol,pAtol,pMax,WP,pin);if(!pr.ok)throw std::runtime_error("pressure PCG failed production tolerance");pIts+=pr.its;pressure_update_minus_kernel<<<(np+TPB-1)/TPB,TPB>>>(np,pin,(Real)alphaP,d_dp.p,d_p.p);
      outer=it;converged=(it>1&&rel<=simpleTol);if(it<=5||it%20==0||converged){A.spmv(d_u2.p,A.tmp.p);residual_free_kernel<<<(nv+TPB-1)/TPB,TPB>>>(nv,d_fixed.p,d_rhs2.p,A.tmp.p,A.res.p);double mr=ured.norm(A.res.p)/std::max(ured.norm(d_rhs2.p),1e-300);std::printf("NODALS_HXT4B_SIMPLE it=%d relCont=%.12e absCont=%.12e zMomentumResidualRel=%.12e pCG=%d pRel=%.3e converged=%d status=PASS\n",it,rel,cn,mr,pr.its,pr.rel,(int)converged);}if(converged)break;
      if(it>=5&&(!std::isfinite(rel)||rel>1e8))throw std::runtime_error("SIMPLE divergence guard");
    }
    if(!converged)throw std::runtime_error("HXT4B SIMPLE maxOuter without continuity convergence");

    std::vector<Real>hp((std::size_t)np),hu0((std::size_t)nv),hu1((std::size_t)nv),hu2((std::size_t)nv);HXT1_CUDA(cudaMemcpy(hp.data(),d_p.p,np*sizeof(Real),cudaMemcpyDeviceToHost));HXT1_CUDA(cudaMemcpy(hu0.data(),d_u0.p,nv*sizeof(Real),cudaMemcpyDeviceToHost));HXT1_CUDA(cudaMemcpy(hu1.data(),d_u1.p,nv*sizeof(Real),cudaMemcpyDeviceToHost));HXT1_CUDA(cudaMemcpy(hu2.data(),d_u2.p,nv*sizeof(Real),cudaMemcpyDeviceToHost));
    auto pd=hxt4a::to_double(hp);double slope=hxt4a::pressure_slope(M,pd),pinP=hxt4a::boundary_p(M,pd,true),poutP=hxt4a::boundary_p(M,pd,false),drop=pinP-poutP,slopeErr=std::abs(slope-dpdzExact)/std::abs(dpdzExact),dropErr=std::abs(drop-dropExact)/std::abs(dropExact);long double en=0,ed=0,et=0;for(std::size_t v=0;v<M.points.size();++v){double ex=profile(M.points[v].x,M.points[v].y,R,bulk),e=(double)hu2[v]-ex;en+=e*e;ed+=ex*ex;et+=(double)hu0[v]*(double)hu0[v]+(double)hu1[v]*(double)hu1[v];}double prof=std::sqrt((double)(en/std::max((long double)1e-300,ed))),trans=std::sqrt((double)(et/std::max((long double)1e-300,ed)));
    // Final convection action norm on z velocity (unrelaxed conv matrix only).
    combine_values_kernel<<<(A.nnz+TPB-1)/TPB,TPB>>>(A.nnz,A.conv.p,A.conv.p,A.val.p); // 2*conv, corrected in norm ratio below
    A.spmv(d_u2.p,A.tmp.p);double convNorm=.5*ured.norm(A.tmp.p),uNorm=ured.norm(d_u2.p);
    write_vtu(vtu,M,hu0,hu1,hu2,hp);
    bool physics=dropErr<0.05&&slopeErr<0.05&&prof<0.15&&std::isfinite(convNorm);
    std::printf("NODALS_HXT4B_PHYSICS pressureSlope=%.12e exactSlope=%.12e slopeRelError=%.12e pIn=%.12e pOut=%.12e pressureDrop=%.12e exactDrop=%.12e dropRelError=%.12e vertexVelocityProfileRelL2=%.12e transverseVertexRelL2=%.12e convectionActionNorm=%.12e velocityNorm=%.12e avgPressureCG=%.6f status=%s\n",slope,dpdzExact,slopeErr,pinP,poutP,drop,dropExact,dropErr,prof,trans,convNorm,uNorm,outer?(double)pIts/outer:0.0,physics?"PASS":"FAIL");
    std::printf("NODALS_HXT4B_OUTPUT vtu=%s status=PASS\n",vtu.c_str());std::printf("HXT4B_GATE_STATUS=%s\n",physics?"PASS":"FAIL");return physics?0:3;
  }catch(const std::exception&e){std::fprintf(stderr,"HXT4B_ERROR precision=%s: %s\n",hxt4b::PRECISION,e.what());std::fprintf(stderr,"HXT4B_GATE_STATUS=FAIL\n");return 2;}
}
