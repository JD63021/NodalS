#pragma once
// HXT5A: mixed HEX(Q1+BF2)/TET(P1+BF3) Nikuradse algebraic eddy viscosity.
// Matches the validated NodalS tet GPU model:
//   ell/R = max(0, 0.14 - 0.08 eta^2 - 0.06 eta^4)
//   nu_t  = scale * ell^2 * sqrt(2 S:S)
// No turbulence unknowns are introduced.

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace hxt5a {

using Real = hxt4b::Real;
using nodals_hxt1::Dev;
static constexpr int MIX_TPB=64;

__device__ __constant__ double h5_hex_x[4]={
  -0.8611363115940525752,-0.3399810435848562648,
   0.3399810435848562648, 0.8611363115940525752};
__device__ __constant__ double h5_hex_w[4]={
  0.3478548451374538574,0.6521451548625461426,
  0.6521451548625461426,0.3478548451374538574};

// Exact positive collapsed 4x4x4 rule used by the existing GPU RANS path.
__device__ __constant__ double h5_tet_rn[4]={0.0485005494469972764,0.238600737551862341,0.517047295104367421,0.795851417896772828};
__device__ __constant__ double h5_tet_rw[4]={0.110888415611277741,0.143458789799214448,0.0686338871729230970,0.0103522407499180812};
__device__ __constant__ double h5_tet_sn[4]={0.0571041961145177246,0.276843013638123803,0.583590432368916834,0.860240135656219485};
__device__ __constant__ double h5_tet_sw[4]={0.135506913431488518,0.203464568010271102,0.129847547608232333,0.0311809709500080849};
__device__ __constant__ double h5_tet_tn[4]={0.0694318442029737137,0.330009478207571871,0.669990521792428129,0.930568155797026231};
__device__ __constant__ double h5_tet_tw[4]={0.173927422568726897,0.326072577431273103,0.326072577431273103,0.173927422568726897};

__device__ __forceinline__ void h5_atomic_max_nonnegative(double*addr,double v){
  auto*p=reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old=*p,assumed;
  do{
    assumed=old;
    if(__longlong_as_double((long long)assumed)>=v)break;
    old=atomicCAS(p,assumed,(unsigned long long)__double_as_longlong(v));
  }while(assumed!=old);
}

__device__ __forceinline__ Real h5_nikuradse_nut(
    Real x,Real y,Real radius,Real scale,const Real gu[9],Real*strainOut=nullptr)
{
  Real ss=Real(0);
  #pragma unroll
  for(int i=0;i<3;++i){
    #pragma unroll
    for(int j=0;j<3;++j){
      Real sij=Real(0.5)*(gu[3*i+j]+gu[3*j+i]);
      ss+=sij*sij;
    }
  }
  Real strain=sqrt(fmax(Real(0),Real(2)*ss));
  if(strainOut)*strainOut=strain;
  Real eta=sqrt(x*x+y*y)/radius;
  eta=fmin(Real(1),fmax(Real(0),eta));
  Real e2=eta*eta,e4=e2*e2;
  Real ell=radius*fmax(Real(0),Real(0.14)-Real(0.08)*e2-Real(0.06)*e4);
  return scale*ell*ell*strain;
}

__global__ void h5_combine3_kernel(
    std::size_t n,const Real*base,const Real*conv,const Real*turb,Real*val)
{
  std::size_t i=(std::size_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n)val[i]=base[i]+conv[i]+turb[i];
}

__global__ void h5_finalize_rows_turb_kernel(
    int n,const unsigned char*fixed,const std::int32_t*diagPos,
    Real*val,const Real*diagRBase,const Real*conv,const Real*diagRTurb,
    Real*rau,Real*diagOriginal,double alphaU,double rauScale,
    unsigned long long*bad)
{
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i>=n)return;
  int d=diagPos[i];
  Real orig=val[d];
  diagOriginal[i]=orig;
  if(fixed[i]){rau[i]=Real(0);return;}
  // Physical response includes eddy-viscosity volume + interface consistency.
  // Nitsche penalty (molecular and turbulent increment) is excluded from rAU.
  Real den=diagRBase[i]+conv[d]+diagRTurb[i];
  if(!(orig>Real(0))||!(den>Real(0))||!isfinite((double)orig)||!isfinite((double)den)){
    atomicAdd(bad,1ULL);rau[i]=Real(0);return;
  }
  val[d]=(Real)((double)orig/alphaU);
  rau[i]=(Real)(rauScale*alphaU/(double)den);
}

__global__ void h5_physical_momentum_residual_kernel(
    int n,const unsigned char*fixed,const Real*bt,const Real*u,
    const Real*diagOriginal,double alphaU,const Real*ArelaxedU,Real*r)
{
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i>=n)return;
  if(fixed[i]){r[i]=Real(0);return;}
  const Real relaxExtra=(Real)(((1.0-alphaU)/alphaU)*(double)diagOriginal[i]*(double)u[i]);
  const Real Aphysical=ArelaxedU[i]-relaxExtra;
  r[i]=bt[i]-Aphysical; // A(u_lag)u - B^T p = 0
}

__global__ void h5_hex_mixlen_kernel(
    const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,
    const nodals_hxt1::HexVel*hv,const std::int32_t*slot,
    const Real*u0,const Real*u1,const Real*u2,
    Real*turb,Real*diagRTurb,std::uint64_t n,
    double radius,double nu,double scale,double*stats,unsigned long long*bad)
{
  std::uint64_t c=(std::uint64_t)blockIdx.x;
  if(c>=n)return;
  __shared__ Real su0[14],su1[14],su2[14];
  __shared__ Real sgp[64][14][3];
  __shared__ Real snutw[64];

  if(threadIdx.x<14){
    int g=hv[c].g[threadIdx.x];
    su0[threadIdx.x]=u0[g];su1[threadIdx.x]=u1[g];su2[threadIdx.x]=u2[g];
  }
  __syncthreads();

  const int iq=(int)threadIdx.x;
  if(iq<64){
    const int ix=iq>>4,iy=(iq>>2)&3,iz=iq&3;
    const double xr=h5_hex_x[ix],yr=h5_hex_x[iy],zr=h5_hex_x[iz];
    const double qw=h5_hex_w[ix]*h5_hex_w[iy]*h5_hex_w[iz];
    double ph[14],gr[14][3],gp[14][3],I[9],det;
    hxt2::hex_basis_ref(xr,yr,zr,ph,gr);
    if(!hxt2::hex_metric(pts,hc[c],xr,yr,zr,I,det)){
      atomicAdd(bad,1ULL);snutw[iq]=Real(0);
      for(int a=0;a<14;++a)for(int d=0;d<3;++d)sgp[iq][a][d]=Real(0);
    }else{
      hxt2::phys_grad14(gr,I,gp);
      Real gu[9]={0,0,0,0,0,0,0,0,0};
      for(int a=0;a<14;++a){
        for(int d=0;d<3;++d){
          Real gd=(Real)gp[a][d];sgp[iq][a][d]=gd;
          gu[d]+=su0[a]*gd;gu[3+d]+=su1[a]*gd;gu[6+d]+=su2[a]*gd;
        }
      }
      double xq=0,yq=0;
      for(int a=0;a<8;++a){
        const double N=0.125*(1.0+hxt4a::h4_hex_sign[a][0]*xr)*(1.0+hxt4a::h4_hex_sign[a][1]*yr)*(1.0+hxt4a::h4_hex_sign[a][2]*zr);
        const auto&X=pts[hc[c].v[a]];xq+=N*X.x;yq+=N*X.y;
      }
      Real strain=0;
      Real nut=h5_nikuradse_nut((Real)xq,(Real)yq,(Real)radius,(Real)scale,gu,&strain);
      const Real w=(Real)(det*qw);snutw[iq]=nut*w;
      if(stats){
        const double wd=(double)w,ratio=(double)nut/nu;
        atomicAdd(stats+0,ratio*wd);atomicAdd(stats+1,wd);
        h5_atomic_max_nonnegative(stats+2,ratio);h5_atomic_max_nonnegative(stats+3,(double)strain);
      }
    }
  }
  __syncthreads();

  for(int p=(int)threadIdx.x;p<196;p+=(int)blockDim.x){
    int a=p/14,b=p%14;Real kab=0;
    #pragma unroll
    for(int q=0;q<64;++q){
      Real gd=sgp[q][a][0]*sgp[q][b][0]+sgp[q][a][1]*sgp[q][b][1]+sgp[q][a][2]*sgp[q][b][2];
      kab+=snutw[q]*gd;
    }
    atomicAdd(turb+slot[c*196ull+(std::size_t)p],kab);
    if(a==b)atomicAdd(diagRTurb+hv[c].g[a],kab);
  }
}

__global__ void h5_tet_mixlen_kernel(
    const nodals_hxt1::Point*pts,const nodals_hxt1::TetConn*tc,
    const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,
    const std::int32_t*slot,const Real*u0,const Real*u1,const Real*u2,
    Real*turb,Real*diagRTurb,std::uint64_t n,
    double radius,double nu,double scale,double*stats)
{
  std::uint64_t c=(std::uint64_t)blockIdx.x;
  if(c>=n)return;
  __shared__ Real su0[8],su1[8],su2[8];
  __shared__ Real sgp[64][8][3];
  __shared__ Real snutw[64];
  if(threadIdx.x<8){int g=tv[c].g[threadIdx.x];su0[threadIdx.x]=u0[g];su1[threadIdx.x]=u1[g];su2[threadIdx.x]=u2[g];}
  __syncthreads();

  const int iq=(int)threadIdx.x;
  if(iq<64){
    const int ir=iq>>4,is=(iq>>2)&3,it=iq&3;
    const double r=h5_tet_rn[ir],s=h5_tet_sn[is],t=h5_tet_tn[it],omr=1.0-r,oms=1.0-s;
    const double lam[4]={omr*oms*(1.0-t),r,omr*s,omr*oms*t};
    const double qw=h5_tet_rw[ir]*h5_tet_sw[is]*h5_tet_tw[it];
    double ph[8],gr[8][3],gp[8][3];
    hxt2::tet_basis(lam[0],lam[1],lam[2],lam[3],ph,gr);hxt2::tet_phys_grad(gr,tg[c],gp);
    Real gu[9]={0,0,0,0,0,0,0,0,0};
    for(int a=0;a<8;++a){
      for(int d=0;d<3;++d){Real gd=(Real)gp[a][d];sgp[iq][a][d]=gd;gu[d]+=su0[a]*gd;gu[3+d]+=su1[a]*gd;gu[6+d]+=su2[a]*gd;}
    }
    double xq=0,yq=0;
    for(int a=0;a<4;++a){const auto&X=pts[tc[c].v[a]];xq+=lam[a]*X.x;yq+=lam[a]*X.y;}
    Real strain=0;Real nut=h5_nikuradse_nut((Real)xq,(Real)yq,(Real)radius,(Real)scale,gu,&strain);
    const Real w=(Real)(tg[c].det*qw);snutw[iq]=nut*w;
    if(stats){const double wd=(double)w,ratio=(double)nut/nu;atomicAdd(stats+0,ratio*wd);atomicAdd(stats+1,wd);h5_atomic_max_nonnegative(stats+2,ratio);h5_atomic_max_nonnegative(stats+3,(double)strain);}
  }
  __syncthreads();

  for(int p=(int)threadIdx.x;p<64;p+=(int)blockDim.x){
    int a=p/8,b=p%8;Real kab=0;
    #pragma unroll
    for(int q=0;q<64;++q){Real gd=sgp[q][a][0]*sgp[q][b][0]+sgp[q][a][1]*sgp[q][b][1]+sgp[q][a][2]*sgp[q][b][2];kab+=snutw[q]*gd;}
    atomicAdd(turb+slot[c*64ull+(std::size_t)p],kab);
    if(a==b)atomicAdd(diagRTurb+tv[c].g[a],kab);
  }
}

__global__ void h5_interface_mixlen_kernel(
    const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,const nodals_hxt1::TetConn*tc,
    const nodals_hxt1::HexVel*hv,const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,
    const hxt2::InterfacePlan*P,const std::int64_t*row,const std::int32_t*col,
    const Real*u0,const Real*u1,const Real*u2,Real*turb,Real*diagRTurb,std::uint64_t n,
    double radius,double nu,double scale,double gamma,unsigned long long*bad)
{
  std::uint64_t iside=(std::uint64_t)blockIdx.x;if(iside>=2*n)return;
  int ir=(int)(iside>>1),side=(int)(iside&1);const auto&p=P[ir];const auto&HG=hv[p.hexCell];int tcell=p.tetCell[side],tf=p.tetFace[side];const auto&TG=tv[tcell];
  __shared__ int ug[22];__shared__ int un;
  __shared__ Real uh0[14],uh1[14],uh2[14],ut0[8],ut1[8],ut2[8];
  __shared__ Real sph[25][14],sgph[25][14][3],spt[25][8],sgpt[25][8][3],snuth[25],snutt[25],sw[25];
  if(threadIdx.x==0){int m=0;for(int a=0;a<14;++a)ug[m++]=HG.g[a];for(int b=0;b<8;++b){int g=TG.g[b],found=0;for(int q=0;q<m;++q)if(ug[q]==g)found=1;if(!found)ug[m++]=g;}un=m;}
  if(threadIdx.x<14){int g=HG.g[threadIdx.x];uh0[threadIdx.x]=u0[g];uh1[threadIdx.x]=u1[g];uh2[threadIdx.x]=u2[g];}
  if(threadIdx.x<8){int g=TG.g[threadIdx.x];ut0[threadIdx.x]=u0[g];ut1[threadIdx.x]=u1[g];ut2[threadIdx.x]=u2[g];}
  __syncthreads();

  for(int q=(int)threadIdx.x;q<25;q+=(int)blockDim.x){
    int ia=q/5,ib=q%5;double rr=hxt2::c_q5_x[ia],ss=hxt2::c_q5_x[ib],om=1.0-rr,L[3]={om*(1.0-ss),rr,om*ss};
    sw[q]=(Real)(hxt2::c_q5_w[ia]*hxt2::c_q5_w[ib]*om*2.0*p.areaTri[side]);
    double xr=0,yr=0,zr=0;for(int k=0;k<3;++k){int hl=p.hexTriLocal[side][k];xr+=L[k]*hxt4a::h4_hex_sign[hl][0];yr+=L[k]*hxt4a::h4_hex_sign[hl][1];zr+=L[k]*hxt4a::h4_hex_sign[hl][2];}
    double ph[14],grh[14][3],gph[14][3],I[9],det;hxt2::hex_basis_ref(xr,yr,zr,ph,grh);
    if(!hxt2::hex_metric(pts,hc[p.hexCell],xr,yr,zr,I,det)){atomicAdd(bad,1ULL);snuth[q]=snutt[q]=Real(0);continue;}
    hxt2::phys_grad14(grh,I,gph);
    double lam[4]={0,0,0,0};for(int k=0;k<3;++k)lam[hxt4a::h4_tet_face[tf][k]]=L[k];
    double pt[8],grt[8][3],gpt[8][3];hxt2::tet_basis(lam[0],lam[1],lam[2],lam[3],pt,grt);hxt2::tet_phys_grad(grt,tg[tcell],gpt);
    Real guh[9]={0,0,0,0,0,0,0,0,0},gut[9]={0,0,0,0,0,0,0,0,0};
    for(int a=0;a<14;++a){sph[q][a]=(Real)ph[a];for(int d=0;d<3;++d){Real gd=(Real)gph[a][d];sgph[q][a][d]=gd;guh[d]+=uh0[a]*gd;guh[3+d]+=uh1[a]*gd;guh[6+d]+=uh2[a]*gd;}}
    for(int a=0;a<8;++a){spt[q][a]=(Real)pt[a];for(int d=0;d<3;++d){Real gd=(Real)gpt[a][d];sgpt[q][a][d]=gd;gut[d]+=ut0[a]*gd;gut[3+d]+=ut1[a]*gd;gut[6+d]+=ut2[a]*gd;}}
    double xq=0,yq=0;for(int a=0;a<8;++a){double N=0.125*(1.0+hxt4a::h4_hex_sign[a][0]*xr)*(1.0+hxt4a::h4_hex_sign[a][1]*yr)*(1.0+hxt4a::h4_hex_sign[a][2]*zr);const auto&X=pts[hc[p.hexCell].v[a]];xq+=N*X.x;yq+=N*X.y;}
    snuth[q]=h5_nikuradse_nut((Real)xq,(Real)yq,(Real)radius,(Real)scale,guh,nullptr);
    double xt=0,yt=0;for(int a=0;a<4;++a){const auto&X=pts[tc[tcell].v[a]];xt+=lam[a]*X.x;yt+=lam[a]*X.y;}
    snutt[q]=h5_nikuradse_nut((Real)xt,(Real)yt,(Real)radius,(Real)scale,gut,nullptr);
  }
  __syncthreads();

  const Real nx=(Real)p.normal[0],ny=(Real)p.normal[1],nz=(Real)p.normal[2];
  const Real tauBase=(Real)(gamma*nu*fmax(p.invHHex,p.invHTet[side]));
  int npair=un*un;
  for(int pair=(int)threadIdx.x;pair<npair;pair+=(int)blockDim.x){
    int ii=pair/un,jj=pair%un,gi=ug[ii],gj=ug[jj];Real kij=0,kiiCons=0;
    for(int q=0;q<25;++q){
      Real ji=0,jjv=0,di=0,dj=0;
      for(int a=0;a<14;++a){
        if(HG.g[a]==gi){ji+=sph[q][a];di+=Real(0.5)*snuth[q]*(sgph[q][a][0]*nx+sgph[q][a][1]*ny+sgph[q][a][2]*nz);}
        if(HG.g[a]==gj){jjv+=sph[q][a];dj+=Real(0.5)*snuth[q]*(sgph[q][a][0]*nx+sgph[q][a][1]*ny+sgph[q][a][2]*nz);}
      }
      for(int a=0;a<8;++a){
        if(TG.g[a]==gi){ji-=spt[q][a];di+=Real(0.5)*snutt[q]*(sgpt[q][a][0]*nx+sgpt[q][a][1]*ny+sgpt[q][a][2]*nz);}
        if(TG.g[a]==gj){jjv-=spt[q][a];dj+=Real(0.5)*snutt[q]*(sgpt[q][a][0]*nx+sgpt[q][a][1]*ny+sgpt[q][a][2]*nz);}
      }
      Real cons=-(dj*ji+di*jjv);
      Real tauEff=(Real)gamma*fmax(((Real)nu+snuth[q])*(Real)p.invHHex,((Real)nu+snutt[q])*(Real)p.invHTet[side]);
      Real deltaTau=fmax(Real(0),tauEff-tauBase);
      kij+=sw[q]*(cons+deltaTau*ji*jjv);if(ii==jj)kiiCons+=sw[q]*cons;
    }
    int sl=hxt4b::csr_find(row,col,gi,gj);atomicAdd(turb+sl,kij);if(ii==jj)atomicAdd(diagRTurb+gi,kiiCons);
  }
}

struct MixStats {double meanNutOverNu=0,maxNutOverNu=0,maxStrain=0,volumeWeight=0;};

inline MixStats h5_assemble_mixlen(
    std::uint64_t nhex,std::uint64_t ntet,std::uint64_t nif,
    const Dev<nodals_hxt1::Point>&d_pts,const Dev<nodals_hxt1::HexConn>&d_hc,const Dev<nodals_hxt1::TetConn>&d_tc,
    const Dev<nodals_hxt1::HexVel>&d_hv,const Dev<nodals_hxt1::TetVel>&d_tv,const Dev<nodals_hxt1::TetGeom>&d_tg,
    const Dev<hxt2::InterfacePlan>&d_ip,hxt4b::GpuCSR&A,
    const Real*u0,const Real*u1,const Real*u2,Dev<Real>&turb,Dev<Real>&diagRTurb,Dev<double>&stats,
    double radius,double nu,double scale,double gamma,Dev<unsigned long long>&bad)
{
  HXT1_CUDA(cudaMemset(turb.p,0,turb.n*sizeof(Real)));HXT1_CUDA(cudaMemset(diagRTurb.p,0,diagRTurb.n*sizeof(Real)));
  HXT1_CUDA(cudaMemset(stats.p,0,stats.n*sizeof(double)));HXT1_CUDA(cudaMemset(bad.p,0,bad.n*sizeof(unsigned long long)));
  if(scale>0.0){
    if(nhex)h5_hex_mixlen_kernel<<<nhex,MIX_TPB>>>(d_pts.p,d_hc.p,d_hv.p,A.hexSlot.p,u0,u1,u2,turb.p,diagRTurb.p,nhex,radius,nu,scale,stats.p,bad.p);
    if(ntet)h5_tet_mixlen_kernel<<<ntet,MIX_TPB>>>(d_pts.p,d_tc.p,d_tv.p,d_tg.p,A.tetSlot.p,u0,u1,u2,turb.p,diagRTurb.p,ntet,radius,nu,scale,stats.p);
    if(nif)h5_interface_mixlen_kernel<<<2*nif,64>>>(d_pts.p,d_hc.p,d_tc.p,d_hv.p,d_tv.p,d_tg.p,d_ip.p,A.row.p,A.col.p,u0,u1,u2,turb.p,diagRTurb.p,nif,radius,nu,scale,gamma,bad.p);
  }
  HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
  unsigned long long nb=0;HXT1_CUDA(cudaMemcpy(&nb,bad.p,sizeof(nb),cudaMemcpyDeviceToHost));
  if(nb)throw std::runtime_error("HXT5A mixing-length geometry/interface failures="+std::to_string(nb));
  std::vector<double>h(4,0.0);HXT1_CUDA(cudaMemcpy(h.data(),stats.p,4*sizeof(double),cudaMemcpyDeviceToHost));
  MixStats q;q.volumeWeight=h[1];q.meanNutOverNu=h[1]>0?h[0]/h[1]:0;q.maxNutOverNu=h[2];q.maxStrain=h[3];
  if(!std::isfinite(q.meanNutOverNu)||!std::isfinite(q.maxNutOverNu)||!std::isfinite(q.maxStrain))throw std::runtime_error("HXT5A nonfinite mixing-length diagnostic");
  return q;
}

inline double h5_physical_residual_norm(hxt4b::GpuCSR&A,const unsigned char*d_fixed,const Real*bt,const Real*u,double alphaU,hxt4a::Reducer<Real>&red){
  A.spmv(u,A.tmp.p);
  h5_physical_momentum_residual_kernel<<<(A.n+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(A.n,d_fixed,bt,u,A.diagOriginal.p,alphaU,A.tmp.p,A.res.p);
  HXT1_CUDA(cudaGetLastError());return red.norm(A.res.p);
}

} // namespace hxt5a
