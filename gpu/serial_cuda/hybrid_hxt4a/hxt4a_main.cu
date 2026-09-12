// NodalS HXT4A
// Compile-time FP32/FP64 hybrid HEX/TET GPU operator + manufactured Stokes parity gate.
//
// Geometry, Jacobians, quadrature tables, host assembly, diagnostics and
// reductions remain FP64. Device state/operator arithmetic is Real=float/double.

#define main hxt2_embedded_gate_main
#include "../hybrid_hxt2/hxt2_main.cu"
#undef main
#define main hxt3a_embedded_gate_main
#include "../hybrid_hxt3a/hxt3a_main.cu"
#undef main

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace hxt4a {

using nodals_hxt1::Dev;
using nodals_hxt1::HostMesh;

#ifdef HXT4A_USE_FLOAT
using Real=float;
static constexpr const char* PRECISION="fp32";
#else
using Real=double;
static constexpr const char* PRECISION="fp64";
#endif

static constexpr int TPB=256;

__device__ __constant__ int h4_hex_sign[8][3]={
  {-1,-1,-1},{1,-1,-1},{1,1,-1},{-1,1,-1},
  {-1,-1, 1},{1,-1, 1},{1,1, 1},{-1,1, 1}
};
__device__ __constant__ int h4_tet_face[4][3]={
  {1,2,3},{0,3,2},{0,1,3},{0,2,1}
};

template<class T> std::vector<T> cast_vec(const std::vector<double>&a){
  std::vector<T>b(a.size());for(std::size_t i=0;i<a.size();++i)b[i]=(T)a[i];return b;
}
template<class T> std::vector<double> to_double(const std::vector<T>&a){
  std::vector<double>b(a.size());for(std::size_t i=0;i<a.size();++i)b[i]=(double)a[i];return b;
}

template<class T>
__global__ void fixed_override_kernel(int n,const unsigned char*fixed,const T*x,T*y){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n&&fixed[i])y[i]=x[i];
}
template<class T>
__global__ void zero_fixed_kernel(int n,const unsigned char*fixed,T*x){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n&&fixed[i])x[i]=(T)0;
}
template<class T> __global__ void pin_zero_kernel(int pin,T*x){if(blockIdx.x==0&&threadIdx.x==0)x[pin]=(T)0;}
template<class T> __global__ void pin_output_zero_kernel(int pin,T*y){if(blockIdx.x==0&&threadIdx.x==0)y[pin]=(T)0;}

template<class T>
__global__ void residual_kernel(int n,const T*b,const T*Ax,T*r){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)r[i]=b[i]-Ax[i];
}
template<class T>
__global__ void precond_kernel(int n,const T*invdiag,const T*r,T*z){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)z[i]=invdiag[i]*r[i];
}
template<class T>
__global__ void axpy2_kernel(int n,T a,const T*p,const T*q,T*x,T*r){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n){x[i]+=a*p[i];r[i]-=a*q[i];}
}
template<class T>
__global__ void pupdate_kernel(int n,T beta,const T*z,T*p){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)p[i]=z[i]+beta*p[i];
}
template<class T>
__global__ void rhs_force_minus_bt_kernel(int n,const unsigned char*fixed,const T*f,const T*bt,T*rhs){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)rhs[i]=fixed[i]?(T)0:(f[i]-bt[i]);
}
template<class T>
__global__ void pressure_update_kernel(int n,int pin,T alpha,const T*dp,T*p){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n){if(i==pin)p[i]=(T)0;else p[i]+=alpha*dp[i];}
}
template<class T>
__global__ void scale3_kernel(int n,const T*r,T*x,T*y,T*z){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n){T a=r[i];x[i]*=a;y[i]*=a;z[i]*=a;}
}
template<class T>
__global__ void copy_fixed_zero_kernel(int n,const unsigned char*fixed,const T*src,T*dst){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)dst[i]=fixed[i]?(T)0:src[i];
}

template<class T>
__global__ void dot_blocks_kernel(const T*x,const T*y,int n,double*out){
  __shared__ double s[TPB];
  int tid=threadIdx.x,i=blockIdx.x*blockDim.x+tid;double q=0.0;
  for(int k=i;k<n;k+=gridDim.x*blockDim.x)q+=(double)x[k]*(double)y[k];
  s[tid]=q;__syncthreads();
  for(int d=TPB/2;d>0;d>>=1){if(tid<d)s[tid]+=s[tid+d];__syncthreads();}
  if(tid==0)out[blockIdx.x]=s[0];
}

template<class T>
struct Reducer {
  int n=0,blocks=0;Dev<double>partial;std::vector<double>host;
  explicit Reducer(int n_):n(n_),blocks(std::max(1,std::min(256,(n_+TPB-1)/TPB))),
    partial(std::vector<double>((std::size_t)std::max(1,std::min(256,(n_+TPB-1)/TPB)),0.0)),
    host((std::size_t)std::max(1,std::min(256,(n_+TPB-1)/TPB)),0.0){}
  double dot(const T*x,const T*y){
    dot_blocks_kernel<T><<<blocks,TPB>>>(x,y,n,partial.p);HXT1_CUDA(cudaGetLastError());
    HXT1_CUDA(cudaMemcpy(host.data(),partial.p,host.size()*sizeof(double),cudaMemcpyDeviceToHost));
    long double q=0;for(double v:host)q+=(long double)v;return (double)q;
  }
  double norm(const T*x){
    const double q=dot(x,x);
    if(!std::isfinite(q)) return std::numeric_limits<double>::quiet_NaN();
    return std::sqrt(std::max(0.0,q));
  }
};

template<class T>
__global__ void hex_volume_kernel(const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,
                                  const nodals_hxt1::HexVel*hv,const T*u,T*r,
                                  std::uint64_t n,double nu,unsigned long long*bad){
  std::uint64_t c=(std::uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
  T ul[14];for(int a=0;a<14;++a)ul[a]=u[hv[c].g[a]];T rl[14]={0};
  for(int ix=0;ix<3;++ix)for(int iy=0;iy<3;++iy)for(int iz=0;iz<3;++iz){
    const double x=hxt2::c_g3_x[ix],y=hxt2::c_g3_x[iy],z=hxt2::c_g3_x[iz];
    const double w=hxt2::c_g3_w[ix]*hxt2::c_g3_w[iy]*hxt2::c_g3_w[iz];
    double ph[14],gr[14][3],gp[14][3],I[9],det;
    hxt2::hex_basis_ref(x,y,z,ph,gr);
    if(!hxt2::hex_metric(pts,hc[c],x,y,z,I,det)){atomicAdd(bad,1ULL);continue;}
    hxt2::phys_grad14(gr,I,gp);
    T gu[3]={(T)0,(T)0,(T)0};
    for(int b=0;b<14;++b)for(int d=0;d<3;++d)gu[d]+=ul[b]*(T)gp[b][d];
    const T q=(T)(nu*det*w);
    for(int a=0;a<14;++a)
      rl[a]+=q*((T)gp[a][0]*gu[0]+(T)gp[a][1]*gu[1]+(T)gp[a][2]*gu[2]);
  }
  for(int a=0;a<14;++a)atomicAdd(r+hv[c].g[a],rl[a]);
}

template<class T>
__global__ void tet_volume_kernel(const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,
                                  const T*u,T*r,std::uint64_t n,double nu){
  std::uint64_t c=(std::uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
  const auto&g=tg[c];double metric[3][3]={{0}};
  for(int j=0;j<3;++j)for(int k=0;k<3;++k)for(int d=0;d<3;++d)metric[j][k]+=g.invJ[3*j+d]*g.invJ[3*k+d];
  T ul[8];for(int a=0;a<8;++a)ul[a]=u[tv[c].g[a]];T rl[8]={0};
  for(int a=0;a<8;++a)for(int b=0;b<8;++b){
    double kd=0;for(int j=0;j<3;++j)for(int k=0;k<3;++k)kd+=hxt2::c_tet_T[(((a*8+b)*3+j)*3+k)]*metric[j][k];
    rl[a]+=(T)(nu*g.det*kd)*ul[b];
  }
  for(int a=0;a<8;++a)atomicAdd(r+tv[c].g[a],rl[a]);
}

template<class T>
__global__ void interface_kernel(const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,
                                 const nodals_hxt1::TetConn*tc,const nodals_hxt1::HexVel*hv,
                                 const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,
                                 const hxt2::InterfacePlan*P,const T*u,T*r,std::uint64_t n,
                                 double nu,double gamma,unsigned long long*bad){
  (void)tc;
  std::uint64_t irc=(std::uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if(irc>=n)return;
  const auto&p=P[irc];T urh[14];for(int a=0;a<14;++a)urh[a]=u[hv[p.hexCell].g[a]];T rrh[14]={0};
  const T nx=(T)p.normal[0],ny=(T)p.normal[1],nz=(T)p.normal[2];
  for(int side=0;side<2;++side){
    const int tcell=p.tetCell[side],tf=p.tetFace[side];
    T urt[8];for(int a=0;a<8;++a)urt[a]=u[tv[tcell].g[a]];T rrt[8]={0};
    const T tau=(T)(gamma*nu*fmax(p.invHHex,p.invHTet[side]));
    for(int ia=0;ia<5;++ia)for(int ib=0;ib<5;++ib){
      const double rr=hxt2::c_q5_x[ia],ss=hxt2::c_q5_x[ib],om=1.0-rr;
      const double L[3]={om*(1.0-ss),rr,om*ss};
      const T w=(T)(hxt2::c_q5_w[ia]*hxt2::c_q5_w[ib]*om*2.0*p.areaTri[side]);
      double xr=0,yr=0,zr=0;
      for(int k=0;k<3;++k){const int hl=p.hexTriLocal[side][k];xr+=L[k]*h4_hex_sign[hl][0];yr+=L[k]*h4_hex_sign[hl][1];zr+=L[k]*h4_hex_sign[hl][2];}
      double phD[14],grh[14][3],gphD[14][3],Ih[9],deth;
      hxt2::hex_basis_ref(xr,yr,zr,phD,grh);
      if(!hxt2::hex_metric(pts,hc[p.hexCell],xr,yr,zr,Ih,deth)){atomicAdd(bad,1ULL);continue;}
      hxt2::phys_grad14(grh,Ih,gphD);
      double lam[4]={0,0,0,0};for(int k=0;k<3;++k)lam[h4_tet_face[tf][k]]=L[k];
      double ptD[8],grt[8][3],gptD[8][3];hxt2::tet_basis(lam[0],lam[1],lam[2],lam[3],ptD,grt);hxt2::tet_phys_grad(grt,tg[tcell],gptD);

      T uh=0,ut=0,gh[3]={0,0,0},gt[3]={0,0,0};
      for(int a=0;a<14;++a){T ph=(T)phD[a];uh+=urh[a]*ph;for(int d=0;d<3;++d)gh[d]+=urh[a]*(T)gphD[a][d];}
      for(int a=0;a<8;++a){T pt=(T)ptD[a];ut+=urt[a]*pt;for(int d=0;d<3;++d)gt[d]+=urt[a]*(T)gptD[a][d];}
      const T jump=uh-ut;
      const T fh=(T)nu*(gh[0]*nx+gh[1]*ny+gh[2]*nz);
      const T ft=(T)nu*(gt[0]*nx+gt[1]*ny+gt[2]*nz);
      const T avg=(T)0.5*(fh+ft);
      for(int a=0;a<14;++a){
        const T ph=(T)phD[a],dn=(T)gphD[a][0]*nx+(T)gphD[a][1]*ny+(T)gphD[a][2]*nz;
        rrh[a]+=w*(-avg*ph-(T)(0.5*nu)*dn*jump+tau*jump*ph);
      }
      for(int a=0;a<8;++a){
        const T pt=(T)ptD[a],dn=(T)gptD[a][0]*nx+(T)gptD[a][1]*ny+(T)gptD[a][2]*nz;
        rrt[a]+=w*( avg*pt-(T)(0.5*nu)*dn*jump-tau*jump*pt);
      }
    }
    for(int a=0;a<8;++a)atomicAdd(r+tv[tcell].g[a],rrt[a]);
  }
  for(int a=0;a<14;++a)atomicAdd(r+hv[p.hexCell].g[a],rrh[a]);
}

template<class T>
__global__ void hex_diag_pair_kernel(const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,
                                     const nodals_hxt1::HexVel*hv,T*df,T*dr,
                                     std::uint64_t n,double nu,unsigned long long*bad){
  std::uint64_t c=(std::uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;T dl[14]={0};
  for(int ix=0;ix<3;++ix)for(int iy=0;iy<3;++iy)for(int iz=0;iz<3;++iz){
    const double x=hxt2::c_g3_x[ix],y=hxt2::c_g3_x[iy],z=hxt2::c_g3_x[iz];
    const double w=hxt2::c_g3_w[ix]*hxt2::c_g3_w[iy]*hxt2::c_g3_w[iz];
    double ph[14],gr[14][3],gp[14][3],I[9],det;hxt2::hex_basis_ref(x,y,z,ph,gr);
    if(!hxt2::hex_metric(pts,hc[c],x,y,z,I,det)){atomicAdd(bad,1ULL);continue;}hxt2::phys_grad14(gr,I,gp);
    for(int a=0;a<14;++a)dl[a]+=(T)(nu*det*w*(gp[a][0]*gp[a][0]+gp[a][1]*gp[a][1]+gp[a][2]*gp[a][2]));
  }
  for(int a=0;a<14;++a){atomicAdd(df+hv[c].g[a],dl[a]);atomicAdd(dr+hv[c].g[a],dl[a]);}
}
template<class T>
__global__ void tet_diag_pair_kernel(const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,
                                     T*df,T*dr,std::uint64_t n,double nu){
  std::uint64_t c=(std::uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;const auto&g=tg[c];
  double metric[3][3]={{0}};for(int j=0;j<3;++j)for(int k=0;k<3;++k)for(int d=0;d<3;++d)metric[j][k]+=g.invJ[3*j+d]*g.invJ[3*k+d];
  for(int a=0;a<8;++a){double kd=0;for(int j=0;j<3;++j)for(int k=0;k<3;++k)kd+=hxt2::c_tet_T[(((a*8+a)*3+j)*3+k)]*metric[j][k];T q=(T)(nu*g.det*kd);atomicAdd(df+tv[c].g[a],q);atomicAdd(dr+tv[c].g[a],q);}
}
template<class T>
__global__ void interface_diag_pair_kernel(const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,
                                           const nodals_hxt1::TetVel*tv,const nodals_hxt1::HexVel*hv,
                                           const nodals_hxt1::TetGeom*tg,const hxt2::InterfacePlan*P,
                                           T*df,T*dr,std::uint64_t n,double nu,double gamma,
                                           unsigned long long*bad){
  std::uint64_t ir=(std::uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if(ir>=n)return;const auto&p=P[ir];
  const auto&HG=hv[p.hexCell];const T nx=(T)p.normal[0],ny=(T)p.normal[1],nz=(T)p.normal[2];
  for(int side=0;side<2;++side){
    const int tcell=p.tetCell[side],tf=p.tetFace[side];const auto&TG=tv[tcell];const T tau=(T)(gamma*nu*fmax(p.invHHex,p.invHTet[side]));
    for(int ia=0;ia<5;++ia)for(int ib=0;ib<5;++ib){
      const double rr=hxt2::c_q5_x[ia],ss=hxt2::c_q5_x[ib],om=1.0-rr;
      const double L[3]={om*(1.0-ss),rr,om*ss};const T w=(T)(hxt2::c_q5_w[ia]*hxt2::c_q5_w[ib]*om*2.0*p.areaTri[side]);
      double xr=0,yr=0,zr=0;for(int k=0;k<3;++k){int hl=p.hexTriLocal[side][k];xr+=L[k]*h4_hex_sign[hl][0];yr+=L[k]*h4_hex_sign[hl][1];zr+=L[k]*h4_hex_sign[hl][2];}
      double ph[14],grh[14][3],gph[14][3],Ih[9],deth;hxt2::hex_basis_ref(xr,yr,zr,ph,grh);
      if(!hxt2::hex_metric(pts,hc[p.hexCell],xr,yr,zr,Ih,deth)){atomicAdd(bad,1ULL);continue;}hxt2::phys_grad14(grh,Ih,gph);
      double lam[4]={0,0,0,0};for(int k=0;k<3;++k)lam[h4_tet_face[tf][k]]=L[k];
      double pt[8],grt[8][3],gpt[8][3];hxt2::tet_basis(lam[0],lam[1],lam[2],lam[3],pt,grt);hxt2::tet_phys_grad(grt,tg[tcell],gpt);

      bool matched[8]={false,false,false,false,false,false,false,false};
      for(int a=0;a<14;++a){
        int mb=-1;for(int b=0;b<8;++b)if(HG.g[a]==TG.g[b]){mb=b;break;}
        T jump=(T)ph[a];
        T avgdn=(T)(0.5*nu)*((T)gph[a][0]*nx+(T)gph[a][1]*ny+(T)gph[a][2]*nz);
        if(mb>=0){matched[mb]=true;jump-=(T)pt[mb];avgdn+=(T)(0.5*nu)*((T)gpt[mb][0]*nx+(T)gpt[mb][1]*ny+(T)gpt[mb][2]*nz);}
        T cons=w*((T)-2*avgdn*jump),pen=w*tau*jump*jump;
        atomicAdd(df+HG.g[a],cons+pen);atomicAdd(dr+HG.g[a],cons);
      }
      for(int b=0;b<8;++b)if(!matched[b]){
        T jump=-(T)pt[b];T avgdn=(T)(0.5*nu)*((T)gpt[b][0]*nx+(T)gpt[b][1]*ny+(T)gpt[b][2]*nz);
        T cons=w*((T)-2*avgdn*jump),pen=w*tau*jump*jump;
        atomicAdd(df+TG.g[b],cons+pen);atomicAdd(dr+TG.g[b],cons);
      }
    }
  }
}

template<class T>
__global__ void finalize_diag_kernel(int n,const unsigned char*fixed,T*df,T*dr,T*idf,T*rau,T scale,unsigned long long*bad){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i>=n)return;
  if(fixed[i]){df[i]=(T)1;dr[i]=(T)1;idf[i]=(T)1;rau[i]=(T)0;return;}
  T a=df[i],b=dr[i];
  if(!(a>(T)0)||!(b>(T)0)||!isfinite((double)a)||!isfinite((double)b)){atomicAdd(bad,1ULL);idf[i]=rau[i]=(T)0;}
  else {idf[i]=(T)1/a;rau[i]=scale/b;}
}

inline std::vector<unsigned char> build_fixed_mask(const HostMesh&M){
  std::vector<unsigned char>f((std::size_t)M.h.nvel,0);
  auto hf=[&](const nodals_hxt1::BoundaryRec&r){f[(std::size_t)r.bubble]=1;const auto&h=M.hexes[(std::size_t)r.cell];for(int k=0;k<4;++k)f[(std::size_t)h.v[hxt2::HEX_FACE[r.localFace][k]]]=1;};
  auto tf=[&](const nodals_hxt1::BoundaryRec&r){f[(std::size_t)r.bubble]=1;const auto&t=M.tets[(std::size_t)r.cell];for(int k=0;k<3;++k)f[(std::size_t)t.v[hxt2::TET_FACE[r.localFace][k]]]=1;};
  for(const auto&r:M.wallHex)hf(r);for(const auto&r:M.inHex)hf(r);for(const auto&r:M.outHex)hf(r);for(const auto&r:M.inTet)tf(r);for(const auto&r:M.outTet)tf(r);
  return f;
}

template<class T>
struct AAction {
  const HostMesh&M;const std::vector<hxt2::InterfacePlan>&P;
  Dev<nodals_hxt1::Point>d_points;Dev<nodals_hxt1::HexConn>d_hex;Dev<nodals_hxt1::TetConn>d_tet;
  Dev<nodals_hxt1::HexVel>d_hv;Dev<nodals_hxt1::TetVel>d_tv;Dev<nodals_hxt1::TetGeom>d_tg;Dev<hxt2::InterfacePlan>d_if;
  Dev<unsigned char>&fixed;double nu,gamma;unsigned long long*bad=nullptr;
  AAction(const HostMesh&m,const std::vector<hxt2::InterfacePlan>&p,Dev<unsigned char>&f,double n,double g):
    M(m),P(p),d_points(m.points),d_hex(m.hexes),d_tet(m.tets),d_hv(m.hexVel),d_tv(m.tetVel),d_tg(m.tetGeom),d_if(p),fixed(f),nu(n),gamma(g){
    HXT1_CUDA(cudaMalloc((void**)&bad,sizeof(*bad)));
  }
  ~AAction(){if(bad)cudaFree(bad);}
  void apply(const T*x,T*y){
    HXT1_CUDA(cudaMemset(y,0,M.h.nvel*sizeof(T)));HXT1_CUDA(cudaMemset(bad,0,sizeof(*bad)));const int B=128;
    if(M.h.nhex)hex_volume_kernel<T><<<(M.h.nhex+B-1)/B,B>>>(d_points.p,d_hex.p,d_hv.p,x,y,M.h.nhex,nu,bad);
    if(M.h.ntet)tet_volume_kernel<T><<<(M.h.ntet+B-1)/B,B>>>(d_tv.p,d_tg.p,x,y,M.h.ntet,nu);
    if(M.h.ninterface)interface_kernel<T><<<(M.h.ninterface+B-1)/B,B>>>(d_points.p,d_hex.p,d_tet.p,d_hv.p,d_tv.p,d_tg.p,d_if.p,x,y,M.h.ninterface,nu,gamma,bad);
    fixed_override_kernel<T><<<(M.h.nvel+TPB-1)/TPB,TPB>>>((int)M.h.nvel,fixed.p,x,y);
    HXT1_CUDA(cudaGetLastError());
  }
};

template<class T>
struct BDevice {
  int np=0,nv=0;Dev<std::int64_t>row;Dev<std::int32_t>col;Dev<T>bx,by,bz;
  BDevice(const hxt3a::BCSR&A):np(A.np),nv(A.nv),row(A.row),col(A.col),bx(cast_vec<T>(A.bx)),by(cast_vec<T>(A.by)),bz(cast_vec<T>(A.bz)){}
};

template<class T>
__global__ void b_kernel(int np,const std::int64_t*row,const std::int32_t*col,const T*bx,const T*by,const T*bz,const T*u0,const T*u1,const T*u2,T*y){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i>=np)return;T s=0;
  for(std::int64_t k=row[i];k<row[i+1];++k){int g=col[k];s+=bx[k]*u0[g]+by[k]*u1[g]+bz[k]*u2[g];}y[i]=s;
}
template<class T>
__global__ void bt_kernel(int np,const std::int64_t*row,const std::int32_t*col,const T*bx,const T*by,const T*bz,const T*p,T*v0,T*v1,T*v2){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i>=np)return;T q=p[i];
  for(std::int64_t k=row[i];k<row[i+1];++k){int g=col[k];atomicAdd(v0+g,bx[k]*q);atomicAdd(v1+g,by[k]*q);atomicAdd(v2+g,bz[k]*q);}
}
template<class T>
inline void B_apply(BDevice<T>&B,const T*u0,const T*u1,const T*u2,T*y){
  b_kernel<T><<<(B.np+TPB-1)/TPB,TPB>>>(B.np,B.row.p,B.col.p,B.bx.p,B.by.p,B.bz.p,u0,u1,u2,y);HXT1_CUDA(cudaGetLastError());
}
template<class T>
inline void BT_apply(BDevice<T>&B,const T*p,T*v0,T*v1,T*v2){
  HXT1_CUDA(cudaMemset(v0,0,B.nv*sizeof(T)));HXT1_CUDA(cudaMemset(v1,0,B.nv*sizeof(T)));HXT1_CUDA(cudaMemset(v2,0,B.nv*sizeof(T)));
  bt_kernel<T><<<(B.np+TPB-1)/TPB,TPB>>>(B.np,B.row.p,B.col.p,B.bx.p,B.by.p,B.bz.p,p,v0,v1,v2);HXT1_CUDA(cudaGetLastError());
}

template<class T>
__global__ void schur_diag_kernel(int np,const std::int64_t*row,const std::int32_t*col,const T*bx,const T*by,const T*bz,const T*rau,T*d,int pin){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i>=np)return;if(i==pin){d[i]=(T)1;return;}T s=0;
  for(std::int64_t k=row[i];k<row[i+1];++k){int g=col[k];s+=rau[g]*(bx[k]*bx[k]+by[k]*by[k]+bz[k]*bz[k]);}d[i]=s;
}
template<class T>
__global__ void invert_diag_kernel(int n,const T*d,T*id,int pin){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)id[i]=(i==pin)?(T)1:(T)1/d[i];
}

template<class T>
struct SAction {
  BDevice<T>&B;const T*rau;int pin;Dev<T>v0,v1,v2;
  SAction(BDevice<T>&b,const T*r,int p):B(b),rau(r),pin(p),v0(std::vector<T>((std::size_t)b.nv,0)),v1(std::vector<T>((std::size_t)b.nv,0)),v2(std::vector<T>((std::size_t)b.nv,0)){}
  void apply(const T*x,T*y){
    BT_apply(B,x,v0.p,v1.p,v2.p);scale3_kernel<T><<<(B.nv+TPB-1)/TPB,TPB>>>(B.nv,rau,v0.p,v1.p,v2.p);
    B_apply(B,v0.p,v1.p,v2.p,y);pin_output_zero_kernel<T><<<1,1>>>(pin,y);HXT1_CUDA(cudaGetLastError());
  }
};

struct CGResult{int its=0;double rel=1.0;bool ok=false;};

template<class T>
struct CGWork {
  int n;Dev<T>r,z,p,q;Reducer<T>red;
  explicit CGWork(int n_):n(n_),r(std::vector<T>((std::size_t)n_,0)),z(std::vector<T>((std::size_t)n_,0)),p(std::vector<T>((std::size_t)n_,0)),q(std::vector<T>((std::size_t)n_,0)),red(n_){}
};

template<class T,class Action>
CGResult cg(Action&Act,const T*rhs,T*x,const T*invdiag,int n,double rtol,int maxit,CGWork<T>&W,int pin=-1){
  Act.apply(x,W.q.p);residual_kernel<T><<<(n+TPB-1)/TPB,TPB>>>(n,rhs,W.q.p,W.r.p);if(pin>=0)pin_zero_kernel<T><<<1,1>>>(pin,W.r.p);
  double r0=W.red.norm(W.r.p);if(!std::isfinite(r0))throw std::runtime_error("HXT4A CG initial residual nonfinite");
  if(r0==0)return {0,0,true};
  precond_kernel<T><<<(n+TPB-1)/TPB,TPB>>>(n,invdiag,W.r.p,W.z.p);HXT1_CUDA(cudaMemcpy(W.p.p,W.z.p,n*sizeof(T),cudaMemcpyDeviceToDevice));
  double rho=W.red.dot(W.r.p,W.z.p);if(!(rho>0))throw std::runtime_error("HXT4A CG initial rho nonpositive");
  CGResult R;const double target=rtol*r0;
  for(int k=0;k<maxit;++k){
    Act.apply(W.p.p,W.q.p);double pq=W.red.dot(W.p.p,W.q.p);if(!(pq>0)||!std::isfinite(pq))throw std::runtime_error("HXT4A CG pAp nonpositive");
    T a=(T)(rho/pq);axpy2_kernel<T><<<(n+TPB-1)/TPB,TPB>>>(n,a,W.p.p,W.q.p,x,W.r.p);
    if(pin>=0){pin_zero_kernel<T><<<1,1>>>(pin,x);pin_zero_kernel<T><<<1,1>>>(pin,W.r.p);}
    double rn=W.red.norm(W.r.p);R.its=k+1;R.rel=rn/r0;if(rn<=target){R.ok=true;break;}
    precond_kernel<T><<<(n+TPB-1)/TPB,TPB>>>(n,invdiag,W.r.p,W.z.p);double nr=W.red.dot(W.r.p,W.z.p);if(!(nr>0))throw std::runtime_error("HXT4A CG rho nonpositive");
    pupdate_kernel<T><<<(n+TPB-1)/TPB,TPB>>>(n,(T)(nr/rho),W.z.p,W.p.p);if(pin>=0)pin_zero_kernel<T><<<1,1>>>(pin,W.p.p);rho=nr;
  }
  return R;
}

inline double centroid_z_hex(const HostMesh&M,int c){double z=0;for(int a=0;a<8;++a)z+=M.points[(std::size_t)M.hexes[(std::size_t)c].v[a]].z;return z/8.0;}
inline double centroid_z_tet(const HostMesh&M,int c){double z=0;for(int a=0;a<4;++a)z+=M.points[(std::size_t)M.tets[(std::size_t)c].v[a]].z;return z/4.0;}

inline double pressure_rel(const std::vector<double>&a,const std::vector<double>&b){
  long double n=0,d=0;for(std::size_t i=0;i<a.size();++i){long double q=a[i]-b[i];n+=q*q;d+=(long double)b[i]*b[i];}return std::sqrt((double)n/std::max((double)d,1e-300));
}
inline double pressure_slope(const HostMesh&M,const std::vector<double>&p){
  int nh=(int)M.h.nhex,np=nh+(int)M.h.ntet;long double W=0,Wz=0,Wp=0,Wzz=0,Wzp=0;
  for(int c=0;c<np;++c){bool h=c<nh;int lc=h?c:c-nh;double z=h?centroid_z_hex(M,lc):centroid_z_tet(M,lc);double v=h?M.hexGeom[(std::size_t)lc].volume:M.tetGeom[(std::size_t)lc].volume;W+=v;Wz+=v*z;Wp+=v*p[(std::size_t)c];Wzz+=v*z*z;Wzp+=v*z*p[(std::size_t)c];}
  return (double)((W*Wzp-Wz*Wp)/(W*Wzz-Wz*Wz));
}
inline double boundary_p(const HostMesh&M,const std::vector<double>&p,bool inlet){
  int nh=(int)M.h.nhex;long double num=0,den=0;const auto&hq=inlet?M.inHex:M.outHex;const auto&tq=inlet?M.inTet:M.outTet;
  for(const auto&r:hq){const auto&h=M.hexes[(std::size_t)r.cell];hxt3a::V3 q[4];for(int k=0;k<4;++k)q[k]=hxt3a::pv(M.points[(std::size_t)h.v[hxt2::HEX_FACE[r.localFace][k]]]);double A=hxt3a::tri_area(q[0],q[1],q[2])+hxt3a::tri_area(q[0],q[2],q[3]);num+=(long double)A*p[(std::size_t)r.cell];den+=A;}
  for(const auto&r:tq){const auto&t=M.tets[(std::size_t)r.cell];hxt3a::V3 q[3];for(int k=0;k<3;++k)q[k]=hxt3a::pv(M.points[(std::size_t)t.v[hxt2::TET_FACE[r.localFace][k]]]);double A=hxt3a::tri_area(q[0],q[1],q[2]);num+=(long double)A*p[(std::size_t)(nh+r.cell)];den+=A;}
  return (double)(num/den);
}

} // namespace hxt4a

#ifdef HXT4A_EMBED_MAIN
int HXT4A_EMBED_MAIN(int argc,char**argv){
#else
int main(int argc,char**argv){
#endif
  try{
    using namespace hxt4a;
    std::string mesh;double nu=1.0,gamma=50.0,rauScale=24.0,alphaP=1.0,simpleTol=1e-3;int maxOuter=300;
    for(int i=1;i<argc;++i){
      std::string a=argv[i];auto next=[&](){if(i+1>=argc)throw std::runtime_error("missing value for "+a);return std::string(argv[++i]);};
      if(a=="--mesh")mesh=next();else if(a=="--nu")nu=std::stod(next());else if(a=="--gamma")gamma=std::stod(next());else if(a=="--rau-scale")rauScale=std::stod(next());else if(a=="--alpha-p")alphaP=std::stod(next());else if(a=="--simple-tol")simpleTol=std::stod(next());else if(a=="--max-outer")maxOuter=std::stoi(next());else throw std::runtime_error("unknown argument "+a);
    }
    if(mesh.empty())throw std::runtime_error("--mesh required");
    HostMesh M=nodals_hxt1::load(mesh);auto P=hxt2::build_interface_plans(M);auto P3=hxt3a::build_if(M);auto Bh=hxt3a::build_b(M,P3);
    const int nv=Bh.nv,np=Bh.np,nh=(int)M.h.nhex,pin=0;
    auto tetT=hxt2::build_tet_tensor();HXT1_CUDA(cudaMemcpyToSymbol(hxt2::c_tet_T,tetT.data(),tetT.size()*sizeof(double)));

    int dev=0;cudaDeviceProp prop{};HXT1_CUDA(cudaGetDevice(&dev));HXT1_CUDA(cudaGetDeviceProperties(&prop,dev));
    auto fixedH=build_fixed_mask(M);Dev<unsigned char>d_fixed(fixedH);

    AAction<Real>A(M,P,d_fixed,nu,gamma);BDevice<Real>B(Bh);

    // Full A diagonal for momentum Jacobi; penalty-free Nitsche diagonal for rAU.
    Dev<Real>d_df(std::vector<Real>((std::size_t)nv,0)),d_dr(std::vector<Real>((std::size_t)nv,0)),d_idf(std::vector<Real>((std::size_t)nv,0)),d_rau(std::vector<Real>((std::size_t)nv,0));
    Dev<unsigned long long>d_bad(std::vector<unsigned long long>(1,0));
    const int C=128;
    if(M.h.nhex)hex_diag_pair_kernel<Real><<<(M.h.nhex+C-1)/C,C>>>(A.d_points.p,A.d_hex.p,A.d_hv.p,d_df.p,d_dr.p,M.h.nhex,nu,A.bad);
    if(M.h.ntet)tet_diag_pair_kernel<Real><<<(M.h.ntet+C-1)/C,C>>>(A.d_tv.p,A.d_tg.p,d_df.p,d_dr.p,M.h.ntet,nu);
    if(M.h.ninterface)interface_diag_pair_kernel<Real><<<(M.h.ninterface+C-1)/C,C>>>(A.d_points.p,A.d_hex.p,A.d_tv.p,A.d_hv.p,A.d_tg.p,A.d_if.p,d_df.p,d_dr.p,M.h.ninterface,nu,gamma,A.bad);
    finalize_diag_kernel<Real><<<(nv+TPB-1)/TPB,TPB>>>(nv,d_fixed.p,d_df.p,d_dr.p,d_idf.p,d_rau.p,(Real)rauScale,d_bad.p);HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
    unsigned long long bad=0;HXT1_CUDA(cudaMemcpy(&bad,d_bad.p,sizeof(bad),cudaMemcpyDeviceToHost));if(bad)throw std::runtime_error("HXT4A nonpositive diagonal count "+std::to_string(bad));

    Dev<Real>d_sd(std::vector<Real>((std::size_t)np,0)),d_isd(std::vector<Real>((std::size_t)np,0));
    schur_diag_kernel<Real><<<(np+TPB-1)/TPB,TPB>>>(np,B.row.p,B.col.p,B.bx.p,B.by.p,B.bz.p,d_rau.p,d_sd.p,pin);
    invert_diag_kernel<Real><<<(np+TPB-1)/TPB,TPB>>>(np,d_sd.p,d_isd.p,pin);HXT1_CUDA(cudaGetLastError());

    SAction<Real>S(B,d_rau.p,pin);
    Reducer<Real>ured(nv),pred(np);

    // Deterministic operator-action audit.
    std::vector<Real>ha((std::size_t)nv),hb0((std::size_t)nv),hb1((std::size_t)nv),hb2((std::size_t)nv),hp((std::size_t)np);
    for(int g=0;g<nv;++g){double q=g+1;ha[(std::size_t)g]=(Real)std::sin(0.00113*q);hb0[(std::size_t)g]=(Real)std::cos(.00131*q);hb1[(std::size_t)g]=(Real)std::sin(.00091*q+.2);hb2[(std::size_t)g]=(Real)std::cos(.00157*q-.1);}
    for(int i=0;i<np;++i)hp[(std::size_t)i]=(Real)std::sin(.00107*(i+1));
    Dev<Real>da(ha),dAr(std::vector<Real>((std::size_t)nv,0)),db0(hb0),db1(hb1),db2(hb2),dBr(std::vector<Real>((std::size_t)np,0)),dpAudit(hp),dSr(std::vector<Real>((std::size_t)np,0));
    A.apply(da.p,dAr.p);B_apply(B,db0.p,db1.p,db2.p,dBr.p);S.apply(dpAudit.p,dSr.p);
    HXT1_CUDA(cudaDeviceSynchronize());
    const double aAudit=ured.norm(dAr.p),bAudit=pred.norm(dBr.p),sAudit=pred.norm(dSr.p);

    // Manufactured p*: axial linear Q0/P0, shifted to pin.
    double zmin=1e300,zmax=-1e300;for(const auto&q:M.points){zmin=std::min(zmin,q.z);zmax=std::max(zmax,q.z);}double L=zmax-zmin;
    std::vector<double>pexD((std::size_t)np);
    for(int c=0;c<nh;++c)pexD[(std::size_t)c]=(centroid_z_hex(M,c)-zmin)/L;
    for(int c=0;c<(int)M.h.ntet;++c)pexD[(std::size_t)(nh+c)]=(centroid_z_tet(M,c)-zmin)/L;
    double sh=pexD[0];for(double&q:pexD)q-=sh;
    auto pexR=cast_vec<Real>(pexD);Dev<Real>d_pex(pexR);

    // f = B^T p* in selected arithmetic, then strong-zero fixed rows.
    Dev<Real>d_f0(std::vector<Real>((std::size_t)nv,0)),d_f1(std::vector<Real>((std::size_t)nv,0)),d_f2(std::vector<Real>((std::size_t)nv,0));
    BT_apply(B,d_pex.p,d_f0.p,d_f1.p,d_f2.p);
    zero_fixed_kernel<Real><<<(nv+TPB-1)/TPB,TPB>>>(nv,d_fixed.p,d_f0.p);zero_fixed_kernel<Real><<<(nv+TPB-1)/TPB,TPB>>>(nv,d_fixed.p,d_f1.p);zero_fixed_kernel<Real><<<(nv+TPB-1)/TPB,TPB>>>(nv,d_fixed.p,d_f2.p);

    Dev<Real>d_u0(std::vector<Real>((std::size_t)nv,0)),d_u1(std::vector<Real>((std::size_t)nv,0)),d_u2(std::vector<Real>((std::size_t)nv,0));
    Dev<Real>d_p(std::vector<Real>((std::size_t)np,0)),d_dp(std::vector<Real>((std::size_t)np,0)),d_cont(std::vector<Real>((std::size_t)np,0));
    Dev<Real>d_bt0(std::vector<Real>((std::size_t)nv,0)),d_bt1(std::vector<Real>((std::size_t)nv,0)),d_bt2(std::vector<Real>((std::size_t)nv,0));
    Dev<Real>d_rhs0(std::vector<Real>((std::size_t)nv,0)),d_rhs1(std::vector<Real>((std::size_t)nv,0)),d_rhs2(std::vector<Real>((std::size_t)nv,0));
    CGWork<Real>W0(nv),W1(nv),W2(nv),WP(np);

    const double mRtol=std::is_same<Real,float>::value?5e-5:1e-10;
    const double pRtol=std::is_same<Real,float>::value?1e-4:1e-10;
    std::printf("NODALS_HXT4A_CONFIG precision=%s scalarBytes=%zu geometry=fp64 reductions=fp64 stateOperator=%s gamma=%.6f rauScale=%.6f alphaP=%.6f simpleTol=%.3e pressureSolver=PCG_JACOBI status=PASS\n",PRECISION,sizeof(Real),PRECISION,gamma,rauScale,alphaP,simpleTol);
    std::printf("NODALS_HXT4A_OPERATOR_AUDIT precision=%s Anorm=%.12e Bnorm=%.12e Snorm=%.12e status=PASS\n",PRECISION,aAudit,bAudit,sAudit);

    double cont0=-1,firstU=-1,finalRel=1;int outer=0;long long mcg=0,pcg=0;bool conv=false;
    for(int it=1;it<=maxOuter;++it){
      BT_apply(B,d_p.p,d_bt0.p,d_bt1.p,d_bt2.p);
      rhs_force_minus_bt_kernel<Real><<<(nv+TPB-1)/TPB,TPB>>>(nv,d_fixed.p,d_f0.p,d_bt0.p,d_rhs0.p);
      rhs_force_minus_bt_kernel<Real><<<(nv+TPB-1)/TPB,TPB>>>(nv,d_fixed.p,d_f1.p,d_bt1.p,d_rhs1.p);
      rhs_force_minus_bt_kernel<Real><<<(nv+TPB-1)/TPB,TPB>>>(nv,d_fixed.p,d_f2.p,d_bt2.p,d_rhs2.p);
      auto m0=cg<Real>(A,d_rhs0.p,d_u0.p,d_idf.p,nv,mRtol,2500,W0);
      auto m1=cg<Real>(A,d_rhs1.p,d_u1.p,d_idf.p,nv,mRtol,2500,W1);
      auto m2=cg<Real>(A,d_rhs2.p,d_u2.p,d_idf.p,nv,mRtol,2500,W2);
      if(!(m0.ok&&m1.ok&&m2.ok))throw std::runtime_error("HXT4A momentum CG failed");mcg+=m0.its+m1.its+m2.its;
      B_apply(B,d_u0.p,d_u1.p,d_u2.p,d_cont.p);pin_zero_kernel<Real><<<1,1>>>(pin,d_cont.p);
      double cn=pred.norm(d_cont.p);if(it==1)cont0=cn;finalRel=cn/std::max(cont0,1e-300);
      double un=std::sqrt(ured.dot(d_u0.p,d_u0.p)+ured.dot(d_u1.p,d_u1.p)+ured.dot(d_u2.p,d_u2.p));if(it==1)firstU=un;
      HXT1_CUDA(cudaMemset(d_dp.p,0,np*sizeof(Real)));auto pr=cg<Real>(S,d_cont.p,d_dp.p,d_isd.p,np,pRtol,2500,WP,pin);if(!pr.ok)throw std::runtime_error("HXT4A pressure CG failed");pcg+=pr.its;
      pressure_update_kernel<Real><<<(np+TPB-1)/TPB,TPB>>>(np,pin,(Real)alphaP,d_dp.p,d_p.p);
      outer=it;conv=(it>1&&finalRel<=simpleTol);
      if(it<=5||it%20==0||conv)std::printf("NODALS_HXT4A_SIMPLE precision=%s it=%d relCont=%.12e velocityNorm=%.12e mCG=[%d,%d,%d] pCG=%d converged=%d status=PASS\n",PRECISION,it,finalRel,un,m0.its,m1.its,m2.its,pr.its,(int)conv);
      if(conv)break;
    }
    if(!conv)throw std::runtime_error("HXT4A SIMPLE maxOuter");

    // Final predictor at final p.
    BT_apply(B,d_p.p,d_bt0.p,d_bt1.p,d_bt2.p);
    rhs_force_minus_bt_kernel<Real><<<(nv+TPB-1)/TPB,TPB>>>(nv,d_fixed.p,d_f0.p,d_bt0.p,d_rhs0.p);
    rhs_force_minus_bt_kernel<Real><<<(nv+TPB-1)/TPB,TPB>>>(nv,d_fixed.p,d_f1.p,d_bt1.p,d_rhs1.p);
    rhs_force_minus_bt_kernel<Real><<<(nv+TPB-1)/TPB,TPB>>>(nv,d_fixed.p,d_f2.p,d_bt2.p,d_rhs2.p);
    auto fm0=cg<Real>(A,d_rhs0.p,d_u0.p,d_idf.p,nv,mRtol,2500,W0);auto fm1=cg<Real>(A,d_rhs1.p,d_u1.p,d_idf.p,nv,mRtol,2500,W1);auto fm2=cg<Real>(A,d_rhs2.p,d_u2.p,d_idf.p,nv,mRtol,2500,W2);
    if(!(fm0.ok&&fm1.ok&&fm2.ok))throw std::runtime_error("HXT4A final momentum failed");
    B_apply(B,d_u0.p,d_u1.p,d_u2.p,d_cont.p);pin_zero_kernel<Real><<<1,1>>>(pin,d_cont.p);
    double finalC=pred.norm(d_cont.p),finalCR=finalC/std::max(cont0,1e-300);
    double finalU=std::sqrt(ured.dot(d_u0.p,d_u0.p)+ured.dot(d_u1.p,d_u1.p)+ured.dot(d_u2.p,d_u2.p)),uRel=finalU/std::max(firstU,1e-300);

    std::vector<Real>pfR((std::size_t)np);HXT1_CUDA(cudaMemcpy(pfR.data(),d_p.p,np*sizeof(Real),cudaMemcpyDeviceToHost));auto pf=to_double(pfR);
    double pRel=pressure_rel(pf,pexD),sl=pressure_slope(M,pf),slEx=pressure_slope(M,pexD);
    double pinR=boundary_p(M,pf,true),poutR=boundary_p(M,pf,false),pinE=boundary_p(M,pexD,true),poutE=boundary_p(M,pexD,false);
    double delta=poutR-pinR,deltaEx=poutE-pinE;

    bool pass=finalCR<=1.2e-3 && uRel<5e-4 && std::abs(sl-slEx)/std::abs(slEx)<5e-3 && std::abs(delta-deltaEx)/std::abs(deltaEx)<5e-3;
    std::printf("NODALS_HXT4A_RESULT precision=%s outer=%d finalRelCont=%.12e pressureRelL2=%.12e slope=%.12e slopeExact=%.12e delta=%.12e deltaExact=%.12e velocityRel=%.12e avgMomentumCG=%.6f avgPressureCG=%.6f Anorm=%.12e Bnorm=%.12e Snorm=%.12e status=%s\n",
      PRECISION,outer,finalCR,pRel,sl,slEx,delta,deltaEx,uRel,outer?(double)mcg/(3.0*outer):0.0,outer?(double)pcg/outer:0.0,aAudit,bAudit,sAudit,pass?"PASS":"FAIL");
    std::printf("HXT4A_PRECISION_STATUS=%s\n",pass?"PASS":"FAIL");
    return pass?0:3;
  }catch(const std::exception&e){
    std::fprintf(stderr,"HXT4A_ERROR precision=%s: %s\n",hxt4a::PRECISION,e.what());
    std::fprintf(stderr,"HXT4A_PRECISION_STATUS=FAIL\n");return 2;
  }
}
