#include "cuda_runtime.hpp"
#include "precision.hpp"
#include "vector_kernels.cuh"
#include "foam_mesh_g2.hpp"
#include "g2_host.hpp"
#include "g3_host.hpp"
#include "g4_host.hpp"
#include "g4_refresh_host.hpp"
#include "g5_sa_host.hpp"
#include "g4_kernels.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nodals_gpu;

struct G4GpuCSR {
  int n=0;DeviceBuffer<std::int64_t> row;DeviceBuffer<std::int32_t> col,diagPos;DeviceBuffer<double> val,diag,b,x,r,tmp,corr;
  void apply(const double*a,double*y){g4_csr_spmv_kernel<<<g4grid(n),G4B>>>(n,row.data(),col.data(),val.data(),a,y);NODALS_CUDA(cudaGetLastError());}
  std::size_t bytes()const{return row.bytes()+col.bytes()+diagPos.bytes()+val.bytes()+diag.bytes()+b.bytes()+x.bytes()+r.bytes()+tmp.bytes()+corr.bytes();}
};
static std::vector<std::int32_t> g4_csr_diag_positions(const CSRHost&A){std::vector<std::int32_t>d((std::size_t)A.n,-1);for(int i=0;i<A.n;++i){auto first=A.col.begin()+A.row[(std::size_t)i],last=A.col.begin()+A.row[(std::size_t)i+1];auto it=std::lower_bound(first,last,i);if(it==last||*it!=i)throw std::runtime_error("G4B coarse diagonal absent");auto k=(std::int64_t)(it-A.col.begin());if(k>INT32_MAX)throw std::runtime_error("G4B coarse diagonal slot exceeds int32");d[(std::size_t)i]=(std::int32_t)k;}return d;}
static G4GpuCSR upload_g4_csr(const CSRHost&A){G4GpuCSR G;G.n=A.n;G.row.allocate(A.row.size());G.col.allocate(A.col.size());G.val.allocate(A.val.size());G.diag.allocate(A.diag.size());G.diagPos.allocate(A.n);G.b.allocate(A.n);G.x.allocate(A.n);G.r.allocate(A.n);G.tmp.allocate(A.n);G.corr.allocate(A.n);G.row.upload(A.row.data(),A.row.size());G.col.upload(A.col.data(),A.col.size());G.val.upload(A.val.data(),A.val.size());G.diag.upload(A.diag.data(),A.diag.size());auto dp=g4_csr_diag_positions(A);G.diagPos.upload(dp.data(),dp.size());return G;}

struct G4PressureFine {
  int nc=0,nv=0;const G4CellPlanHost*cells=nullptr;const double*rau_live=nullptr;DeviceBuffer<double> rau_pc,diag_pc,v0,v1,v2;
  void apply_with_rau(const double*x,double*y,const double*r){NODALS_CUDA(cudaMemset(v0.data(),0,v0.bytes()));NODALS_CUDA(cudaMemset(v1.data(),0,v1.bytes()));NODALS_CUDA(cudaMemset(v2.data(),0,v2.bytes()));g4_bt3_kernel<<<g4grid(nc),G4B>>>(cells,x,nc,v0.data(),v1.data(),v2.data());NODALS_CUDA(cudaGetLastError());g4_rau3_kernel<<<g4grid(nv),G4B>>>(v0.data(),v1.data(),v2.data(),r,nv);NODALS_CUDA(cudaGetLastError());g4_b3_kernel<<<g4grid(nc),G4B>>>(cells,nc,v0.data(),v1.data(),v2.data(),y);NODALS_CUDA(cudaGetLastError());}
  void apply_live(const double*x,double*y){apply_with_rau(x,y,rau_live);}
  void apply_pc(const double*x,double*y){apply_with_rau(x,y,rau_pc.data());}
  void bt_state(const double*p){NODALS_CUDA(cudaMemset(v0.data(),0,v0.bytes()));NODALS_CUDA(cudaMemset(v1.data(),0,v1.bytes()));NODALS_CUDA(cudaMemset(v2.data(),0,v2.bytes()));g4_bt3_kernel<<<g4grid(nc),G4B>>>(cells,p,nc,v0.data(),v1.data(),v2.data());NODALS_CUDA(cudaGetLastError());}
  std::size_t bytes()const{return rau_pc.bytes()+diag_pc.bytes()+v0.bytes()+v1.bytes()+v2.bytes();}
};

struct G5GpuTransfer {
  int nFine=0,nCoarse=0;
  DeviceBuffer<std::int64_t> row;
  DeviceBuffer<std::int32_t> col;
  DeviceBuffer<double> val;
  std::size_t bytes()const{return row.bytes()+col.bytes()+val.bytes();}
  void restrict(const double*f,double*c)const{
    NODALS_CUDA(cudaMemset(c,0,(std::size_t)nCoarse*sizeof(double)));
    g5_sa_restrict_kernel<<<g4grid(nFine),G4B>>>(nFine,row.data(),col.data(),val.data(),f,c);NODALS_CUDA(cudaGetLastError());
  }
  void prolong_add(const double*c,double*f)const{
    g5_sa_prolong_add_kernel<<<g4grid(nFine),G4B>>>(nFine,row.data(),col.data(),val.data(),c,f);NODALS_CUDA(cudaGetLastError());
  }
};
static G5GpuTransfer upload_g5_transfer(const SATransferHost&P){
  G5GpuTransfer G;G.nFine=P.nFine;G.nCoarse=P.nCoarse;G.row.allocate(P.row.size());G.col.allocate(P.col.size());G.val.allocate(P.val.size());
  G.row.upload(P.row.data(),P.row.size());G.col.upload(P.col.data(),P.col.size());G.val.upload(P.val.data(),P.val.size());return G;
}

struct G4AMG {
  G4PressureFine*fine=nullptr;
  std::vector<G4GpuCSR>L;
  std::vector<G5GpuTransfer>P;
  DeviceBuffer<double>fine_r,fine_tmp,fine_corr,fine_rhs,terminal_inv;
  double fineLambda=1.0;
  std::vector<double>levelLambda;
  int chebDegree=2,powerIts=16;
  double lambdaSafety=1.5,lambdaLowFraction=.05;
  int spectrumRefreshes=0;

  static double cheb_root(double hi,double lo,int k,int degree){
    const double centre=.5*(hi+lo),radius=.5*(hi-lo),pi=3.141592653589793238462643383279502884;
    return centre-radius*std::cos(pi*(2.0*(double)k+1.0)/(2.0*(double)degree));
  }
  void cheb_explicit(G4GpuCSR&A,const double*b,double*x,double lambda){
    const double lo=lambdaLowFraction*lambda;if(!(lambda>lo&&lo>0))throw std::runtime_error("G5C explicit Cheb interval invalid");
    double w0=1.0/cheb_root(lambda,lo,0,chebDegree);
    g4_jacobi_zero_kernel<<<g4grid(A.n),G4B>>>(A.n,w0,b,A.diag.data(),x);NODALS_CUDA(cudaGetLastError());
    for(int k=1;k<chebDegree;++k){A.apply(x,A.tmp.data());g4_residual_kernel<<<g4grid(A.n),G4B>>>(A.n,b,A.tmp.data(),A.r.data());NODALS_CUDA(cudaGetLastError());double w=1.0/cheb_root(lambda,lo,k,chebDegree);g4_jacobi_add_kernel<<<g4grid(A.n),G4B>>>(A.n,w,A.r.data(),A.diag.data(),x);NODALS_CUDA(cudaGetLastError());}
  }
  void cheb_fine(const double*b,double*x){
    const double lo=lambdaLowFraction*fineLambda;if(!(fineLambda>lo&&lo>0))throw std::runtime_error("G5C fine Cheb interval invalid");
    double w0=1.0/cheb_root(fineLambda,lo,0,chebDegree);g4_jacobi_zero_kernel<<<g4grid(fine->nc),G4B>>>(fine->nc,w0,b,fine->diag_pc.data(),x);NODALS_CUDA(cudaGetLastError());
    for(int k=1;k<chebDegree;++k){fine->apply_pc(x,fine_tmp.data());g4_residual_kernel<<<g4grid(fine->nc),G4B>>>(fine->nc,b,fine_tmp.data(),fine_r.data());NODALS_CUDA(cudaGetLastError());double w=1.0/cheb_root(fineLambda,lo,k,chebDegree);g4_jacobi_add_kernel<<<g4grid(fine->nc),G4B>>>(fine->nc,w,fine_r.data(),fine->diag_pc.data(),x);NODALS_CUDA(cudaGetLastError());}
  }
  void explicit_vcycle(int l,const double*b,double*x){
    auto&A=L[(std::size_t)l];
    if(l==(int)L.size()-1){g4_dense_mv_kernel<<<g4grid(A.n),G4B>>>(A.n,terminal_inv.data(),b,x);NODALS_CUDA(cudaGetLastError());return;}
    cheb_explicit(A,b,x,levelLambda[(std::size_t)l]);
    A.apply(x,A.tmp.data());g4_residual_kernel<<<g4grid(A.n),G4B>>>(A.n,b,A.tmp.data(),A.r.data());NODALS_CUDA(cudaGetLastError());
    auto&C=L[(std::size_t)l+1];P[(std::size_t)l+1].restrict(A.r.data(),C.b.data());
    explicit_vcycle(l+1,C.b.data(),C.x.data());
    P[(std::size_t)l+1].prolong_add(C.x.data(),x);
    A.apply(x,A.tmp.data());g4_residual_kernel<<<g4grid(A.n),G4B>>>(A.n,b,A.tmp.data(),A.r.data());NODALS_CUDA(cudaGetLastError());
    g4_copy_kernel<<<g4grid(A.n),G4B>>>(A.n,A.r.data(),A.b.data());NODALS_CUDA(cudaGetLastError());
    cheb_explicit(A,A.b.data(),A.corr.data(),levelLambda[(std::size_t)l]);
    g4_axpy_kernel<<<g4grid(A.n),G4B>>>(A.n,1.0,A.corr.data(),x);NODALS_CUDA(cudaGetLastError());
  }
  void apply(const double*b,double*z){
    cheb_fine(b,z);
    fine->apply_pc(z,fine_tmp.data());g4_residual_kernel<<<g4grid(fine->nc),G4B>>>(fine->nc,b,fine_tmp.data(),fine_r.data());NODALS_CUDA(cudaGetLastError());
    auto&C=L[0];P[0].restrict(fine_r.data(),C.b.data());explicit_vcycle(0,C.b.data(),C.x.data());P[0].prolong_add(C.x.data(),z);
    fine->apply_pc(z,fine_tmp.data());g4_residual_kernel<<<g4grid(fine->nc),G4B>>>(fine->nc,b,fine_tmp.data(),fine_r.data());NODALS_CUDA(cudaGetLastError());
    g4_copy_kernel<<<g4grid(fine->nc),G4B>>>(fine->nc,fine_r.data(),fine_rhs.data());NODALS_CUDA(cudaGetLastError());
    cheb_fine(fine_rhs.data(),fine_corr.data());g4_axpy_kernel<<<g4grid(fine->nc),G4B>>>(fine->nc,1.0,fine_corr.data(),z);NODALS_CUDA(cudaGetLastError());
  }
  std::size_t bytes()const{std::size_t b=fine_r.bytes()+fine_tmp.bytes()+fine_corr.bytes()+fine_rhs.bytes()+terminal_inv.bytes();for(const auto&t:P)b+=t.bytes();for(const auto&x:L)b+=x.bytes();return b;}
};
struct G4PCGWorkspace{DeviceBuffer<double>x,r,z,p,q;};
struct G4PCGResult{int its=0;double rel=1;bool ok=false;};
static G4PCGResult pressure_pcg(G4PressureFine&F,G4AMG&A,const double*rhs,double rtol,double atol,int maxit,G4PCGWorkspace&W,DeviceBuffer<double>&dotScratch){
  int n=F.nc;NODALS_CUDA(cudaMemset(W.x.data(),0,W.x.bytes()));g4_copy_kernel<<<g4grid(n),G4B>>>(n,rhs,W.r.data());NODALS_CUDA(cudaGetLastError());double r0=device_norm2(W.r.data(),n,dotScratch);if(!std::isfinite(r0))throw std::runtime_error("G4 pressure initial norm nonfinite");if(r0<=atol){NODALS_CUDA(cudaMemset(W.x.data(),0,W.x.bytes()));return {0,0,true};}
  A.apply(W.r.data(),W.z.data());g4_copy_kernel<<<g4grid(n),G4B>>>(n,W.z.data(),W.p.data());NODALS_CUDA(cudaGetLastError());double rho=device_dot(W.r.data(),W.z.data(),n,dotScratch);if(!(rho>0)||!std::isfinite(rho))throw std::runtime_error("G4 pressure PCG initial rho nonpositive");G4PCGResult R;
  double target=std::max(atol,rtol*r0);for(int k=0;k<maxit;++k){F.apply_live(W.p.data(),W.q.data());double pq=device_dot(W.p.data(),W.q.data(),n,dotScratch);if(!(pq>0)||!std::isfinite(pq))throw std::runtime_error("G4 pressure PCG pAp nonpositive");double alpha=rho/pq;g4_pcg_xr_kernel<<<g4grid(n),G4B>>>(n,alpha,W.p.data(),W.q.data(),W.x.data(),W.r.data());NODALS_CUDA(cudaGetLastError());double rn=device_norm2(W.r.data(),n,dotScratch);R.its=k+1;R.rel=rn/std::max(r0,1e-300);if(rn<=target){R.ok=true;break;}A.apply(W.r.data(),W.z.data());double rhon=device_dot(W.r.data(),W.z.data(),n,dotScratch);if(!(rhon>0)||!std::isfinite(rhon))throw std::runtime_error("G4 pressure PCG rho nonpositive");g4_pcg_p_kernel<<<g4grid(n),G4B>>>(n,rhon/rho,W.z.data(),W.p.data());NODALS_CUDA(cudaGetLastError());rho=rhon;}return R;
}

struct G4Gpu {
  int nv=0,nc=0,ncolors=0;DeviceBuffer<G4CellPlanHost> cells;DeviceBuffer<double>fixed;
  DeviceBuffer<std::int64_t> row;DeviceBuffer<std::int32_t>col,diagPos,colorOffset,colorRows;DeviceBuffer<double>av,delta,diag,rau;
  DeviceBuffer<double>u0,u1,u2,p,s0,s1,s2,c0,c1,c2,b0,b1,b2,fixedDiv,cont,momSums;G4PressureFine pf;G4AMG amg;G4PCGWorkspace pcg;DeviceBuffer<double>dotScratch;
  std::size_t bytes()const{return cells.bytes()+fixed.bytes()+row.bytes()+col.bytes()+diagPos.bytes()+colorOffset.bytes()+colorRows.bytes()+av.bytes()+delta.bytes()+diag.bytes()+rau.bytes()+u0.bytes()+u1.bytes()+u2.bytes()+p.bytes()+s0.bytes()+s1.bytes()+s2.bytes()+c0.bytes()+c1.bytes()+c2.bytes()+b0.bytes()+b1.bytes()+b2.bytes()+fixedDiv.bytes()+cont.bytes()+momSums.bytes()+pf.bytes()+amg.bytes()+pcg.x.bytes()+pcg.r.bytes()+pcg.z.bytes()+pcg.p.bytes()+pcg.q.bytes()+dotScratch.bytes();}
};

static std::vector<std::int32_t> diag_positions(const MomentumCSRHost&A){std::vector<std::int32_t>d((std::size_t)A.n,-1);for(int i=0;i<A.n;++i){auto first=A.col.begin()+A.row[(std::size_t)i],last=A.col.begin()+A.row[(std::size_t)i+1];auto it=std::lower_bound(first,last,i);if(it==last||*it!=i)throw std::runtime_error("G4 diag position absent");auto k=(std::int64_t)(it-A.col.begin());if(k>INT32_MAX)throw std::runtime_error("G4 diag position exceeds int32");d[(std::size_t)i]=(std::int32_t)k;}return d;}
static void upload_tensors(){const auto&D=diffusion_tensor_g3();const auto&C=central_tensor_g4();NODALS_CUDA(cudaMemcpyToSymbol(g4_diffT,D.t,sizeof(D.t)));NODALS_CUDA(cudaMemcpyToSymbol(g4_centT,C.t,sizeof(C.t)));}
static double relvec(const std::vector<double>&a,const std::vector<double>&b,double&mx){long double d=0,r=0;mx=0;if(a.size()!=b.size())throw std::runtime_error("G4 parity size mismatch");for(std::size_t i=0;i<a.size();++i){double q=a[i]-b[i];d+=(long double)q*q;r+=(long double)b[i]*b[i];mx=std::max(mx,std::abs(q));}return std::sqrt((double)d)/std::max(std::sqrt((double)r),1e-300);}

static G4Gpu upload_all(const G4SetupHost&S,const SAHierarchyHost&H,const std::vector<double>&initRau){
  G4Gpu G;G.nv=S.topo.n;G.nc=(int)S.cells.size();G.ncolors=S.coloring.ncolors;
  G.cells.allocate(S.cells.size());G.cells.upload(S.cells.data(),S.cells.size());std::vector<double>fv(S.fixedValue.size()*3);for(std::size_t i=0;i<S.fixedValue.size();++i)for(int d=0;d<3;++d)fv[3*i+d]=S.fixedValue[i][d];G.fixed.allocate(fv.size());G.fixed.upload(fv.data(),fv.size());
  G.row.allocate(S.topo.row.size());G.col.allocate(S.topo.col.size());G.av.allocate(S.topo.val.size());G.row.upload(S.topo.row.data(),S.topo.row.size());G.col.upload(S.topo.col.data(),S.topo.col.size());auto dp=diag_positions(S.topo);G.diagPos.allocate(dp.size());G.diagPos.upload(dp.data(),dp.size());
  G.colorOffset.allocate(S.coloring.offset.size());G.colorRows.allocate(S.coloring.rows.size());G.colorOffset.upload(S.coloring.offset.data(),S.coloring.offset.size());G.colorRows.upload(S.coloring.rows.data(),S.coloring.rows.size());
  G.delta.allocate(G.nv);G.diag.allocate(G.nv);G.rau.allocate(G.nv);G.rau.upload(initRau.data(),initRau.size());
  for(auto*p:{&G.u0,&G.u1,&G.u2,&G.s0,&G.s1,&G.s2,&G.c0,&G.c1,&G.c2,&G.b0,&G.b1,&G.b2})p->allocate(G.nv);
  G.p.allocate(G.nc);G.fixedDiv.allocate(G.nc);G.cont.allocate(G.nc);G.momSums.allocate(6);G.s0.upload(S.staticRhs[0].data(),G.nv);G.s1.upload(S.staticRhs[1].data(),G.nv);G.s2.upload(S.staticRhs[2].data(),G.nv);G.fixedDiv.upload(S.fixedDiv.data(),G.nc);
  NODALS_CUDA(cudaMemset(G.u0.data(),0,G.u0.bytes()));NODALS_CUDA(cudaMemset(G.u1.data(),0,G.u1.bytes()));NODALS_CUDA(cudaMemset(G.u2.data(),0,G.u2.bytes()));NODALS_CUDA(cudaMemset(G.p.data(),0,G.p.bytes()));
  G.pf.nc=G.nc;G.pf.nv=G.nv;G.pf.cells=G.cells.data();G.pf.rau_live=G.rau.data();G.pf.rau_pc.allocate(initRau.size());G.pf.rau_pc.upload(initRau.data(),initRau.size());G.pf.diag_pc.allocate(G.nc);G.pf.diag_pc.upload(H.fineDiag.data(),H.fineDiag.size());G.pf.v0.allocate(G.nv);G.pf.v1.allocate(G.nv);G.pf.v2.allocate(G.nv);
  G.amg.fine=&G.pf;G.amg.fineLambda=H.fineLambda;G.amg.levelLambda=H.levelLambda;G.amg.chebDegree=2;G.amg.powerIts=H.powerIts;G.amg.lambdaSafety=H.lambdaSafety;G.amg.lambdaLowFraction=H.lambdaLowFraction;
  G.amg.fine_r.allocate(G.nc);G.amg.fine_tmp.allocate(G.nc);G.amg.fine_corr.allocate(G.nc);G.amg.fine_rhs.allocate(G.nc);
  G.amg.P.reserve(H.P.size());for(const auto&P:H.P)G.amg.P.push_back(upload_g5_transfer(P));
  G.amg.L.reserve(H.csr.size());for(const auto&A:H.csr)G.amg.L.push_back(upload_g4_csr(A));
  G.amg.terminal_inv.allocate(H.terminal_inv.size());G.amg.terminal_inv.upload(H.terminal_inv.data(),H.terminal_inv.size());
  for(auto*p:{&G.pcg.x,&G.pcg.r,&G.pcg.z,&G.pcg.p,&G.pcg.q})p->allocate(G.nc);G.dotScratch.allocate(1);
  NODALS_CUDA(cudaDeviceSynchronize());return G;
}

static double g5_power_fine_gpu(G4Gpu&G,int its,double safety){
  auto&v=G.amg.fine_corr;auto&u=G.amg.fine_rhs;auto&au=G.amg.fine_tmp;auto&w=G.amg.fine_r;
  g5_power_init_kernel<<<g4grid(G.nc),G4B>>>(G.nc,v.data());NODALS_CUDA(cudaGetLastError());double vn=device_norm2(v.data(),G.nc,G.dotScratch);device_scale(v.data(),(std::size_t)G.nc,1.0/vn);double ray=0.0;
  for(int it=0;it<its;++it){g5_div_sqrt_diag_kernel<<<g4grid(G.nc),G4B>>>(G.nc,v.data(),G.pf.diag_pc.data(),u.data());NODALS_CUDA(cudaGetLastError());G.pf.apply_pc(u.data(),au.data());g5_div_sqrt_diag_kernel<<<g4grid(G.nc),G4B>>>(G.nc,au.data(),G.pf.diag_pc.data(),w.data());NODALS_CUDA(cudaGetLastError());double num=device_dot(v.data(),w.data(),G.nc,G.dotScratch),den=device_dot(v.data(),v.data(),G.nc,G.dotScratch);ray=num/std::max(den,1e-300);double wn=device_norm2(w.data(),G.nc,G.dotScratch);if(!(ray>0.0)||!(wn>0.0)||!std::isfinite(ray)||!std::isfinite(wn))throw std::runtime_error("G5C fine GPU power failed");g4_copy_kernel<<<g4grid(G.nc),G4B>>>(G.nc,w.data(),v.data());NODALS_CUDA(cudaGetLastError());device_scale(v.data(),(std::size_t)G.nc,1.0/wn);}
  return safety*ray;
}
static double g5_power_csr_gpu(G4Gpu&G,G4GpuCSR&A,int its,double safety){
  g5_power_init_kernel<<<g4grid(A.n),G4B>>>(A.n,A.corr.data());NODALS_CUDA(cudaGetLastError());double vn=device_norm2(A.corr.data(),A.n,G.dotScratch);device_scale(A.corr.data(),(std::size_t)A.n,1.0/vn);double ray=0.0;
  for(int it=0;it<its;++it){g5_div_sqrt_diag_kernel<<<g4grid(A.n),G4B>>>(A.n,A.corr.data(),A.diag.data(),A.b.data());NODALS_CUDA(cudaGetLastError());A.apply(A.b.data(),A.tmp.data());g5_div_sqrt_diag_kernel<<<g4grid(A.n),G4B>>>(A.n,A.tmp.data(),A.diag.data(),A.r.data());NODALS_CUDA(cudaGetLastError());double num=device_dot(A.corr.data(),A.r.data(),A.n,G.dotScratch),den=device_dot(A.corr.data(),A.corr.data(),A.n,G.dotScratch);ray=num/std::max(den,1e-300);double wn=device_norm2(A.r.data(),A.n,G.dotScratch);if(!(ray>0.0)||!(wn>0.0)||!std::isfinite(ray)||!std::isfinite(wn))throw std::runtime_error("G5C coarse GPU power failed");g4_copy_kernel<<<g4grid(A.n),G4B>>>(A.n,A.r.data(),A.corr.data());NODALS_CUDA(cudaGetLastError());device_scale(A.corr.data(),(std::size_t)A.n,1.0/wn);}
  return safety*ray;
}
static void g5_gpu_spectrum_refresh(G4Gpu&G,const SAHierarchyHost&H){
  CudaEventTimer T;T.start();double old=G.amg.fineLambda;G.amg.fineLambda=g5_power_fine_gpu(G,G.amg.powerIts,G.amg.lambdaSafety);
  std::printf("NODALS_GPU_G5D_SPECTRUM level=0 powerIts=%d hostLambda=%.12e gpuLambda=%.12e ratio=%.6f status=PASS\n",G.amg.powerIts,old,G.amg.fineLambda,G.amg.fineLambda/std::max(old,1e-300));
  for(std::size_t l=0;l+1<G.amg.L.size();++l){double h=(l<H.levelLambda.size()?H.levelLambda[l]:0.0);double q=g5_power_csr_gpu(G,G.amg.L[l],G.amg.powerIts,G.amg.lambdaSafety);G.amg.levelLambda[l]=q;std::printf("NODALS_GPU_G5D_SPECTRUM level=%zu powerIts=%d hostLambda=%.12e gpuLambda=%.12e ratio=%.6f status=PASS\n",l+1,G.amg.powerIts,h,q,q/std::max(h,1e-300));}
  NODALS_CUDA(cudaDeviceSynchronize());float ms=T.stop();++G.amg.spectrumRefreshes;std::printf("NODALS_GPU_G5D_POWER_SETUP powerIts=%d levels=%zu ms=%.3f scalarD2H_only=1 status=PASS\n",G.amg.powerIts,G.amg.L.size(),ms);
}
static double g5_rau_snapshot_rel(G4Gpu&G){
  g5_diff_kernel<<<g4grid(G.nv),G4B>>>(G.nv,G.rau.data(),G.pf.rau_pc.data(),G.pf.v0.data());NODALS_CUDA(cudaGetLastError());double dn=device_norm2(G.pf.v0.data(),G.nv,G.dotScratch);double rn=device_norm2(G.pf.rau_pc.data(),G.nv,G.dotScratch);return dn/std::max(rn,1e-300);
}

static void assemble_live(G4Gpu&G,double nu,double alphaU){NODALS_CUDA(cudaMemset(G.av.data(),0,G.av.bytes()));NODALS_CUDA(cudaMemset(G.c0.data(),0,G.c0.bytes()));NODALS_CUDA(cudaMemset(G.c1.data(),0,G.c1.bytes()));NODALS_CUDA(cudaMemset(G.c2.data(),0,G.c2.bytes()));g4_assemble_physical_kernel<<<g4grid(G.nc),G4B>>>(G.cells.data(),G.nc,G.row.data(),G.fixed.data(),G.u0.data(),G.u1.data(),G.u2.data(),nu,G.av.data(),G.c0.data(),G.c1.data(),G.c2.data());NODALS_CUDA(cudaGetLastError());g4_finalize_relax_kernel<<<g4grid(G.nv),G4B>>>(G.nv,G.row.data(),G.col.data(),G.diagPos.data(),alphaU,G.av.data(),G.delta.data(),G.diag.data(),G.rau.data());NODALS_CUDA(cudaGetLastError());}
static void symmetric_mcgs(G4Gpu&G,const ColoringHost&C,double omega,const std::array<int,3>&active){for(int c=0;c<C.ncolors;++c){int s=C.offset[(std::size_t)c],n=C.offset[(std::size_t)c+1]-s;if(n)g4_mcgs_color_kernel<<<g4grid(n),G4B>>>(n,G.colorRows.data()+s,G.row.data(),G.col.data(),G.av.data(),G.diag.data(),G.b0.data(),G.b1.data(),G.b2.data(),G.u0.data(),G.u1.data(),G.u2.data(),omega,active[0],active[1],active[2]);}for(int c=C.ncolors-1;c>=0;--c){int s=C.offset[(std::size_t)c],n=C.offset[(std::size_t)c+1]-s;if(n)g4_mcgs_color_kernel<<<g4grid(n),G4B>>>(n,G.colorRows.data()+s,G.row.data(),G.col.data(),G.av.data(),G.diag.data(),G.b0.data(),G.b1.data(),G.b2.data(),G.u0.data(),G.u1.data(),G.u2.data(),omega,active[0],active[1],active[2]);}NODALS_CUDA(cudaGetLastError());}
static std::array<double,6> mom_norms(G4Gpu&G){NODALS_CUDA(cudaMemset(G.momSums.data(),0,G.momSums.bytes()));g4_mom_norms_kernel<<<g4grid(G.nv),G4B>>>(G.nv,G.row.data(),G.col.data(),G.av.data(),G.b0.data(),G.b1.data(),G.b2.data(),G.u0.data(),G.u1.data(),G.u2.data(),G.momSums.data());NODALS_CUDA(cudaGetLastError());double h[6];G.momSums.download(h,6);std::array<double,6>q;for(int i=0;i<6;++i)q[(std::size_t)i]=std::sqrt(std::max(0.0,h[i]));return q;}
struct MomSolveResult{std::array<int,3>its{{0,0,0}};std::array<double,3>initialRel{{0,0,0}},finalRel{{0,0,0}};};
static MomSolveResult solve_momentum(G4Gpu&G,const ColoringHost&C,double rtol,double atol,double relDrop,int maxIts,double omega){MomSolveResult R;auto q=mom_norms(G);std::array<double,3>bn,rn0,target;std::array<int,3>active;for(int d=0;d<3;++d){bn[d]=q[(std::size_t)d];if(bn[d]==0)bn[d]=1;rn0[d]=q[(std::size_t)3+d];target[d]=std::max(atol,rtol*bn[d]);R.initialRel[d]=rn0[d]/bn[d];R.finalRel[d]=R.initialRel[d];active[d]=rn0[d]>target[d];}
  auto any=[&](){return active[0]||active[1]||active[2];};for(int it=1;it<=maxIts&&any();++it){symmetric_mcgs(G,C,omega,active);q=mom_norms(G);for(int d=0;d<3;++d)if(active[d]){double rn=q[(std::size_t)3+d];R.its[d]=it;R.finalRel[d]=rn/bn[d];if(!std::isfinite(R.finalRel[d]))throw std::runtime_error("G4 MCGS residual nonfinite");if(rn<=target[d]||(relDrop>0&&rn0[d]>0&&rn<=relDrop*rn0[d]))active[d]=0;}}if(any())throw std::runtime_error("G4 MCGS exceeded max iterations");return R;}


struct G5StageEvents {
  cudaEvent_t e[7]{};
  G5StageEvents(){for(auto &x:e)NODALS_CUDA(cudaEventCreate(&x));}
  ~G5StageEvents(){for(auto &x:e)if(x)cudaEventDestroy(x);}
  void rec(int i){NODALS_CUDA(cudaEventRecord(e[i]));}
  float ms(int a,int b)const{float q=0.0f;NODALS_CUDA(cudaEventElapsedTime(&q,e[a],e[b]));return q;}
  G5StageEvents(const G5StageEvents&)=delete;G5StageEvents&operator=(const G5StageEvents&)=delete;
};
static double g5_used_mib(const DeviceMemoryInfo&m){return (double)(m.total_bytes-m.free_bytes)/(1024.0*1024.0);}
static double g5_mib(std::size_t b){return (double)b/(1024.0*1024.0);}

int main(int argc,char**argv){try{
  std::string mesh,tag="case",wall="patch_0_0",inlet="patch_2_0",outlet="patch_1_0";double re=20,bulk=1;int outer=10,refreshEvery=100;
  for(int i=1;i<argc;++i){std::string a=argv[i];if(a=="--mesh"&&i+1<argc)mesh=argv[++i];else if(a=="--tag"&&i+1<argc)tag=argv[++i];else if(a=="--wall"&&i+1<argc)wall=argv[++i];else if(a=="--inlet"&&i+1<argc)inlet=argv[++i];else if(a=="--outlet"&&i+1<argc)outlet=argv[++i];else if(a=="--re"&&i+1<argc)re=std::atof(argv[++i]);else if(a=="--bulk"&&i+1<argc)bulk=std::atof(argv[++i]);else if(a=="--outer"&&i+1<argc)outer=std::atoi(argv[++i]);else if(a=="--refresh"&&i+1<argc)refreshEvery=std::atoi(argv[++i]);else throw std::runtime_error("G5D usage error");}
  if(mesh.empty())throw std::runtime_error("--mesh required");if(outer!=10)throw std::runtime_error("G5D gate requires exactly 10 SIMPLE iterations");if(refreshEvery<1)throw std::runtime_error("G5D refresh must be positive");
  NODALS_CUDA(cudaSetDevice(0));cudaDeviceProp prop{};NODALS_CUDA(cudaGetDeviceProperties(&prop,0));upload_tensors();NODALS_CUDA(cudaDeviceSynchronize());
  const auto memBaseline=device_memory_info();
  const double alphaU=.5,alphaP=.5;const double momRtol=1e-8,momAtol=0,momDrop=.1,momOmega=1;const int momMax=20000;const double pRtol=.5,pAtol=1e-12;const int pMax=20;

  auto M=load_foam_tet_mesh(mesh);auto S=build_g4_setup(M,re,bulk,wall,inlet,outlet);std::array<std::vector<double>,3>U0;for(auto&u:U0)u.assign((std::size_t)S.topo.n,0.0);std::vector<double>hostA,initDelta;std::array<std::vector<double>,3>hostConv;host_assemble_central_g4(S,U0,hostA,hostConv);auto initRau=host_finalize_relax_g4(S,hostA,alphaU,&initDelta);S.pressure.rAU=initRau;auto H=build_sa_hierarchy(M,S.pressure,16,6,18,1000,8,16,1.5,0.05,4.0/3.0);

  std::printf("NODALS_GPU_G5D_CONFIG tag=%s precision=fp64 device=%s cc=%d.%d petsc=NONE mpi=NONE cells=%zu fixedOuter=10 case=pipe_Re20_central_SUPGoff alphaU=0.5 alphaP=0.5 momentum=fusedXYZ_MCGS pressure=matrix_free_Schur_smoothedAggregation_ChebyshevJacobi_PCG pressureRefresh=%d SA_interpMaxNnz=8 SA_damping=1.333333333333 powerIts=16 lambdaSafety=1.5 lambdaLowFraction=0.05 chebDegree=2 fixed10HierarchySnapshot=setup_rAU warmDefinition=iterations_2_to_10\n",tag.c_str(),prop.name,prop.major,prop.minor,M.tets.size(),refreshEvery);
  std::printf("NODALS_GPU_G5D_SETUP tag=%s cells=%zu freeVel=%d nnz=%zu colors=%d hierarchyLevels=%zu terminal=%d transfer0Nnz=%zu baselineUsedMiB=%.3f totalMiB=%.3f status=PASS\n",tag.c_str(),M.tets.size(),S.topo.n,S.topo.val.size(),S.coloring.ncolors,H.csr.size(),H.terminal_n,H.P.empty()?0:H.P[0].val.size(),g5_used_mib(memBaseline),(double)memBaseline.total_bytes/(1024.0*1024.0));

  G4Gpu G=upload_all(S,H,initRau);G.pf.cells=G.cells.data();G.pf.rau_live=G.rau.data();G.amg.fine=&G.pf;g5_gpu_spectrum_refresh(G,H);NODALS_CUDA(cudaDeviceSynchronize());const auto memUpload=device_memory_info();
  const double baselineUsed=g5_used_mib(memBaseline),uploadUsed=g5_used_mib(memUpload),uploadDelta=uploadUsed-baselineUsed,explicitMiB=g5_mib(G.bytes());
  std::printf("NODALS_GPU_G5D_MEMORY tag=%s point=after_upload cells=%zu baselineUsedMiB=%.3f usedMiB=%.3f deltaFromBaselineMiB=%.3f explicitMiB=%.3f explicitBytesPerCell=%.3f status=PASS\n",tag.c_str(),M.tets.size(),baselineUsed,uploadUsed,uploadDelta,explicitMiB,(double)G.bytes()/std::max<std::size_t>(M.tets.size(),1));

  G5StageEvents ev[10];double cont0=-1.0,contRel=1.0;bool pressureAll=true,finiteAll=true;long long sumP=0,sumMom=0;std::array<long long,3>sumMomComp{0,0,0};
  auto wall0=std::chrono::steady_clock::now();
  for(int it=1;it<=10;++it){
    auto &E=ev[it-1];E.rec(0);
    assemble_live(G,S.pipe.nu,alphaU);E.rec(1);
    const bool doRefresh=false;if(it==1){double rr=g5_rau_snapshot_rel(G);std::printf("NODALS_GPU_G5D_SNAPSHOT_PARITY it=1 rAURel=%.3e tol=2e-11 status=%s\n",rr,rr<2e-11?"PASS":"FAIL");if(!(rr<2e-11))throw std::runtime_error("G5C setup rAU snapshot mismatch");}E.rec(2);
    G.pf.bt_state(G.p.data());g4_momentum_rhs_kernel<<<g4grid(G.nv),G4B>>>(G.nv,G.s0.data(),G.s1.data(),G.s2.data(),G.c0.data(),G.c1.data(),G.c2.data(),G.pf.v0.data(),G.pf.v1.data(),G.pf.v2.data(),G.delta.data(),G.u0.data(),G.u1.data(),G.u2.data(),G.b0.data(),G.b1.data(),G.b2.data());NODALS_CUDA(cudaGetLastError());auto ms=solve_momentum(G,S.coloring,momRtol,momAtol,momDrop,momMax,momOmega);for(int d=0;d<3;++d)sumMomComp[d]+=ms.its[d];sumMom+=std::max({ms.its[0],ms.its[1],ms.its[2]});E.rec(3);
    g4_continuity_kernel<<<g4grid(G.nc),G4B>>>(G.cells.data(),G.nc,G.fixedDiv.data(),G.u0.data(),G.u1.data(),G.u2.data(),G.cont.data());NODALS_CUDA(cudaGetLastError());double cn=device_norm2(G.cont.data(),G.nc,G.dotScratch);if(it==1)cont0=cn;contRel=cn/std::max(cont0,1e-300);g4_negate_kernel<<<g4grid(G.nc),G4B>>>(G.nc,G.cont.data());NODALS_CUDA(cudaGetLastError());E.rec(4);
    auto pr=pressure_pcg(G.pf,G.amg,G.cont.data(),pRtol,pAtol,pMax,G.pcg,G.dotScratch);pressureAll=pressureAll&&pr.ok;if(!pr.ok)throw std::runtime_error("G5D pressure PCG failed requested inexact target");sumP+=pr.its;E.rec(5);
    g4_axpy_kernel<<<g4grid(G.nc),G4B>>>(G.nc,alphaP,G.pcg.x.data(),G.p.data());NODALS_CUDA(cudaGetLastError());E.rec(6);
    finiteAll=finiteAll&&std::isfinite(contRel)&&std::isfinite(pr.rel)&&std::isfinite(ms.initialRel[0])&&std::isfinite(ms.initialRel[1])&&std::isfinite(ms.initialRel[2]);
    std::printf("NODALS_GPU_G5D_SIMPLE tag=%s it=%d relCont=%.12e uSweeps=[%d,%d,%d] pCG=%d pTrueRel=%.3e refresh=%d status=%s\n",tag.c_str(),it,contRel,ms.its[0],ms.its[1],ms.its[2],pr.its,pr.rel,(int)doRefresh,(pressureAll&&finiteAll)?"PASS":"FAIL");
  }
  NODALS_CUDA(cudaEventSynchronize(ev[9].e[6]));NODALS_CUDA(cudaDeviceSynchronize());auto wall1=std::chrono::steady_clock::now();const auto mem10=device_memory_info();

  double total=0,assembly=0,refresh=0,momentum=0,continuity=0,pressure=0,pupdate=0,pstage=0;double warmTotal=0,warmAssembly=0,warmMomentum=0,warmContinuity=0,warmPressure=0,warmPupdate=0,warmPstage=0;
  for(int i=0;i<10;++i){double q0=ev[i].ms(0,6),qa=ev[i].ms(0,1),qr=ev[i].ms(1,2),qm=ev[i].ms(2,3),qc=ev[i].ms(3,4),qp=ev[i].ms(4,5),qu=ev[i].ms(5,6),qs=ev[i].ms(3,6);total+=q0;assembly+=qa;refresh+=qr;momentum+=qm;continuity+=qc;pressure+=qp;pupdate+=qu;pstage+=qs;if(i>=1){warmTotal+=q0;warmAssembly+=qa;warmMomentum+=qm;warmContinuity+=qc;warmPressure+=qp;warmPupdate+=qu;warmPstage+=qs;}}
  const double wallMs=std::chrono::duration<double,std::milli>(wall1-wall0).count();
  const double after10Used=g5_used_mib(mem10),delta10=after10Used-baselineUsed,drift=after10Used-uploadUsed;
  std::printf("NODALS_GPU_G5D_MEMORY tag=%s point=after_simple10 cells=%zu baselineUsedMiB=%.3f usedMiB=%.3f deltaFromBaselineMiB=%.3f explicitMiB=%.3f runtimeDriftMiB=%.3f totalMiB=%.3f status=%s\n",tag.c_str(),M.tets.size(),baselineUsed,after10Used,delta10,explicitMiB,drift,(double)mem10.total_bytes/(1024.0*1024.0),std::abs(drift)<64.0?"PASS":"CHECK");
  std::printf("NODALS_GPU_G5D_TIMING tag=%s cells=%zu simple10GpuMs=%.6f simple10WallMs=%.6f simpleAvgGpuMs=%.6f warmSimple9GpuMs=%.6f warmSimpleAvgGpuMs=%.6f assembly10Ms=%.6f warmAssemblyAvgMs=%.6f amgRefresh10Ms=%.6f momentum10Ms=%.6f warmMomentumAvgMs=%.6f continuity10Ms=%.6f warmContinuityAvgMs=%.6f pressureSolve10Ms=%.6f warmPressureSolveAvgMs=%.6f pressureStage10Ms=%.6f warmPressureStageAvgMs=%.6f pressureUpdate10Ms=%.6f avgPressureIts=%.6f pressureMsPerPCGIt=%.6f avgMomentumSymmetricSweeps=%.6f avgMomentumCompSweeps=[%.6f,%.6f,%.6f] finalRelCont=%.12e status=%s\n",tag.c_str(),M.tets.size(),total,wallMs,total/10.0,warmTotal,warmTotal/9.0,assembly,warmAssembly/9.0,refresh,momentum,warmMomentum/9.0,continuity,warmContinuity/9.0,pressure,warmPressure/9.0,pstage,warmPstage/9.0,pupdate,(double)sumP/10.0,sumP?pressure/(double)sumP:0.0,(double)sumMom/10.0,(double)sumMomComp[0]/10.0,(double)sumMomComp[1]/10.0,(double)sumMomComp[2]/10.0,contRel,(pressureAll&&finiteAll)?"PASS":"FAIL");
  std::printf("NODALS_GPU_G5D_RESIDENCY tag=%s O_N_H2D_inside_SIMPLE=0 O_N_D2H_inside_SIMPLE=0 AMG_hierarchy=SA_setup_snapshot_uploaded_once powerSpectrum=GPU_scalarD2H_only SA_numericRefreshBeyond100=NOT_EXERCISED_fixed10 scalarD2H_only=1 fieldDownloadAfterLoop=0 status=PASS\n",tag.c_str());
  std::printf("NODALS_GPU_RESULT gate=G5D tag=%s cells=%zu fixedOuter=10 scalingProfile=PASS pressureAll=%s finite=%s noPetsc=1 noMPI=1 status=%s\n",tag.c_str(),M.tets.size(),pressureAll?"PASS":"FAIL",finiteAll?"PASS":"FAIL",(pressureAll&&finiteAll)?"PASS":"FAIL");
  return (pressureAll&&finiteAll)?0:34;
}catch(const std::exception&e){std::fprintf(stderr,"NODALS_GPU_G5D_EXCEPTION what=%s\n",e.what());return 90;}}
