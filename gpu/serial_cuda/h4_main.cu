#include "cuda_runtime.hpp"
#include "precision.hpp"
#include "vector_kernels.cuh"
#include "foam_mesh_g2.hpp"
#include "g2_host.hpp"
#include "g3_host.hpp"
#include "g4_host.hpp"
#include "g4_refresh_host.hpp"
#include "g5_sa_host.hpp"
#include "g5e_fp32_kernels.cuh"
#include "h2b_fine_csr.cuh"
#include "h4_kernels.cuh"
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
#include <type_traits>
#include <vector>

using namespace nodals_gpu;

enum H0Cat {
  H0_FINE_LIVE_TOTAL=0,H0_FINE_LIVE_ZERO,H0_FINE_LIVE_BT,H0_FINE_LIVE_RAU,H0_FINE_LIVE_B,
  H0_FINE_PC_TOTAL,H0_FINE_PC_ZERO,H0_FINE_PC_BT,H0_FINE_PC_RAU,H0_FINE_PC_B,
  H0_FINE_CSR_REFRESH,
  H0_AMG_TOTAL,H0_RESTRICT,H0_PROLONG,H0_COARSE_SPMV,H0_TERMINAL_DENSE,H0_PRESS_REDUCTION,
  H0_MOM_BT_TOTAL,H0_MOM_BT_ZERO,H0_MOM_BT_KERNEL,H0_MOM_RHS,
  H0_MOM_NORM_INITIAL,H0_MOM_NORM_AFTER,H0_MOM_FORWARD,H0_MOM_BACKWARD,
  H0_CAT_COUNT
};
static const char* h0_cat_name(int c){
  static const char* n[H0_CAT_COUNT]={
    "fine_live_total","fine_live_zero","fine_live_bt","fine_live_rau","fine_live_b",
    "fine_pc_total","fine_pc_zero","fine_pc_bt","fine_pc_rau","fine_pc_b",
    "fine_csr_refresh",
    "amg_total","restrict","prolong","coarse_spmv","terminal_dense","pressure_reduction",
    "mom_bt_total","mom_bt_zero","mom_bt_kernel","mom_rhs",
    "mom_norm_initial","mom_norm_after","mom_forward","mom_backward"};
  return (c>=0&&c<H0_CAT_COUNT)?n[c]:"unknown";
}
struct H0Profiler {
  static constexpr int MAX_LEVEL=8;
  static constexpr int CAP=2048;
  struct Rec {cudaEvent_t a{},b{};int cat=0,level=-1;};
  std::vector<Rec> rec;
  int used=0,currentOuter=0;
  double total[H0_CAT_COUNT][MAX_LEVEL+1]{};
  double warm[H0_CAT_COUNT][MAX_LEVEL+1]{};
  long long calls[H0_CAT_COUNT][MAX_LEVEL+1]{};
  long long warmCalls[H0_CAT_COUNT][MAX_LEVEL+1]{};
  long long recordCount=0;
  H0Profiler(){rec.resize(CAP);for(auto&r:rec){NODALS_CUDA(cudaEventCreate(&r.a));NODALS_CUDA(cudaEventCreate(&r.b));}}
  ~H0Profiler(){for(auto&r:rec){if(r.a)cudaEventDestroy(r.a);if(r.b)cudaEventDestroy(r.b);}}
  int slot(int level)const{return level<0?0:std::min(level+1,MAX_LEVEL);}
  int begin(int cat,int level=-1){if(used>=CAP)throw std::runtime_error("H2 profiler record capacity exceeded");int i=used++;rec[(std::size_t)i].cat=cat;rec[(std::size_t)i].level=level;NODALS_CUDA(cudaEventRecord(rec[(std::size_t)i].a));return i;}
  void end(int id){NODALS_CUDA(cudaEventRecord(rec[(std::size_t)id].b));}
  void collect(){if(!used)return;NODALS_CUDA(cudaEventSynchronize(rec[(std::size_t)used-1].b));for(int i=0;i<used;++i){float ms=0;NODALS_CUDA(cudaEventElapsedTime(&ms,rec[(std::size_t)i].a,rec[(std::size_t)i].b));auto&r=rec[(std::size_t)i];int sl=slot(r.level);total[r.cat][sl]+=ms;calls[r.cat][sl]++;if(currentOuter>=2){warm[r.cat][sl]+=ms;warmCalls[r.cat][sl]++;}recordCount++;}used=0;}
};
static H0Profiler* g_h0=nullptr;
static inline int h0_begin(int cat,int level=-1){return g_h0?g_h0->begin(cat,level):-1;}
static inline void h0_end(int id){if(g_h0&&id>=0)g_h0->end(id);}
template<class T> static double h0_pressure_dot(const T*x,const T*y,std::size_t n,DeviceBuffer<double>&scratch){int q=h0_begin(H0_PRESS_REDUCTION);double v=device_dot(x,y,n,scratch);h0_end(q);return v;}
template<class T> static double h0_pressure_norm2(const T*x,std::size_t n,DeviceBuffer<double>&scratch){return std::sqrt(h0_pressure_dot(x,x,n,scratch));}

template<class T>
static std::vector<T> g5e_cast_vec(const std::vector<double>&v){
  std::vector<T> q(v.size());for(std::size_t i=0;i<v.size();++i)q[i]=(T)v[i];return q;
}
static std::vector<G4CellPlanDevice> g5e_cast_cells(const std::vector<G4CellPlanHost>&h){
  std::vector<G4CellPlanDevice>d(h.size());
  for(std::size_t i=0;i<h.size();++i){for(int a=0;a<8;++a)d[i].ref[a]=h[i].ref[a];for(int a=0;a<64;++a)d[i].rowSlot[a]=h[i].rowSlot[a];d[i].det=(OperatorReal)h[i].det;for(int q=0;q<9;++q)d[i].invJ[q]=(OperatorReal)h[i].invJ[q];}
  return d;
}

struct G4GpuCSR {
  int n=0;DeviceBuffer<std::int64_t> row;DeviceBuffer<std::int32_t> col,diagPos;DeviceBuffer<AMGReal> val,diag,b,x,r,tmp,corr;
  void apply(const AMGReal*a,AMGReal*y,int level=-1){int q=h0_begin(H0_COARSE_SPMV,level);g4_csr_spmv_kernel<<<g4grid(n),G4B>>>(n,row.data(),col.data(),val.data(),a,y);NODALS_CUDA(cudaGetLastError());h0_end(q);}
  std::size_t bytes()const{return row.bytes()+col.bytes()+diagPos.bytes()+val.bytes()+diag.bytes()+b.bytes()+x.bytes()+r.bytes()+tmp.bytes()+corr.bytes();}
};
static std::vector<std::int32_t> g4_csr_diag_positions(const CSRHost&A){std::vector<std::int32_t>d((std::size_t)A.n,-1);for(int i=0;i<A.n;++i){auto first=A.col.begin()+A.row[(std::size_t)i],last=A.col.begin()+A.row[(std::size_t)i+1];auto it=std::lower_bound(first,last,i);if(it==last||*it!=i)throw std::runtime_error("G4B coarse diagonal absent");auto k=(std::int64_t)(it-A.col.begin());if(k>INT32_MAX)throw std::runtime_error("G4B coarse diagonal slot exceeds int32");d[(std::size_t)i]=(std::int32_t)k;}return d;}
static G4GpuCSR upload_g4_csr(const CSRHost&A){G4GpuCSR G;G.n=A.n;G.row.allocate(A.row.size());G.col.allocate(A.col.size());G.val.allocate(A.val.size());G.diag.allocate(A.diag.size());G.diagPos.allocate(A.n);G.b.allocate(A.n);G.x.allocate(A.n);G.r.allocate(A.n);G.tmp.allocate(A.n);G.corr.allocate(A.n);G.row.upload(A.row.data(),A.row.size());G.col.upload(A.col.data(),A.col.size());auto vv=g5e_cast_vec<AMGReal>(A.val),dd=g5e_cast_vec<AMGReal>(A.diag);G.val.upload(vv.data(),vv.size());G.diag.upload(dd.data(),dd.size());auto dp=g4_csr_diag_positions(A);G.diagPos.upload(dp.data(),dp.size());return G;}


struct H2FineCSRDevice {
  int n=0;
  DeviceBuffer<std::int32_t> row,col,diagPos,incOff,contribOff;
  DeviceBuffer<std::uint32_t> packed;
  DeviceBuffer<std::uint8_t> slot;
  DeviceBuffer<AMGReal> val;

  void refresh(const G4CellPlanDevice*cells,const OperatorReal*rau,AMGReal*diag){
    int q=h0_begin(H0_FINE_CSR_REFRESH);
    constexpr int B=256;
    h2b_fine_csr_refresh_warp_kernel<<<h2_warp_grid(n),B>>>(
      n,cells,rau,row.data(),diagPos.data(),incOff.data(),packed.data(),
      contribOff.data(),slot.data(),val.data(),diag);
    NODALS_CUDA(cudaGetLastError());
    h0_end(q);
  }
  void refresh_serial_reference(const G4CellPlanDevice*cells,const OperatorReal*rau,AMGReal*diag){
    h2_fine_csr_refresh_kernel<<<g4grid(n),G4B>>>(
      n,cells,rau,row.data(),diagPos.data(),incOff.data(),packed.data(),
      contribOff.data(),slot.data(),val.data(),diag);
    NODALS_CUDA(cudaGetLastError());
  }
  void apply(const AMGReal*x,AMGReal*y)const{
    constexpr int B=256;
    h2_fine_csr_spmv_warp_kernel<<<h2_warp_grid(n),B>>>(
      n,row.data(),col.data(),val.data(),x,y);
    NODALS_CUDA(cudaGetLastError());
  }
  std::size_t bytes()const{
    return row.bytes()+col.bytes()+diagPos.bytes()+incOff.bytes()+packed.bytes()+
           contribOff.bytes()+slot.bytes()+val.bytes();
  }
};
static H2FineCSRDevice upload_h2_fine_csr(const H2FineCSRHost&H){
  H2FineCSRDevice G;G.n=H.n;
  G.row.allocate(H.row.size());G.col.allocate(H.col.size());G.diagPos.allocate(H.diagPos.size());
  G.incOff.allocate(H.incOff.size());G.packed.allocate(H.packed.size());
  G.contribOff.allocate(H.contribOff.size());G.slot.allocate(H.slot.size());G.val.allocate(H.col.size());
  G.row.upload(H.row.data(),H.row.size());G.col.upload(H.col.data(),H.col.size());G.diagPos.upload(H.diagPos.data(),H.diagPos.size());
  G.incOff.upload(H.incOff.data(),H.incOff.size());G.packed.upload(H.packed.data(),H.packed.size());
  G.contribOff.upload(H.contribOff.data(),H.contribOff.size());G.slot.upload(H.slot.data(),H.slot.size());
  return G;
}

struct G4PressureFine {
  int nc=0,nv=0;const G4CellPlanDevice*cells=nullptr;const OperatorReal*rau_live=nullptr;H2FineCSRDevice*csr_pc=nullptr;bool physicalUseCurrentCSR=false;DeviceBuffer<AMGReal> rau_pc,diag_pc,v0,v1,v2;
  void apply_with_rau(const StateReal*x,StateReal*y,const OperatorReal*r,bool live){
    int qt=h0_begin(live?H0_FINE_LIVE_TOTAL:H0_FINE_PC_TOTAL);
    int q=h0_begin(live?H0_FINE_LIVE_ZERO:H0_FINE_PC_ZERO);NODALS_CUDA(cudaMemset(v0.data(),0,v0.bytes()));NODALS_CUDA(cudaMemset(v1.data(),0,v1.bytes()));NODALS_CUDA(cudaMemset(v2.data(),0,v2.bytes()));h0_end(q);
    q=h0_begin(live?H0_FINE_LIVE_BT:H0_FINE_PC_BT);g4_bt3_kernel<<<g4grid(nc),G4B>>>(cells,x,nc,v0.data(),v1.data(),v2.data());NODALS_CUDA(cudaGetLastError());h0_end(q);
    q=h0_begin(live?H0_FINE_LIVE_RAU:H0_FINE_PC_RAU);g4_rau3_kernel<<<g4grid(nv),G4B>>>(v0.data(),v1.data(),v2.data(),r,nv);NODALS_CUDA(cudaGetLastError());h0_end(q);
    q=h0_begin(live?H0_FINE_LIVE_B:H0_FINE_PC_B);g4_b3_kernel<<<g4grid(nc),G4B>>>(cells,nc,v0.data(),v1.data(),v2.data(),y);NODALS_CUDA(cudaGetLastError());h0_end(q);h0_end(qt);
  }
  void apply_live(const StateReal*x,StateReal*y){
    if(physicalUseCurrentCSR){
      static_assert(std::is_same<StateReal,AMGReal>::value,"H2B physical CSR requires matching FP32 state/AMG types");
      if(!csr_pc)throw std::runtime_error("H2B physical CSR pointer null");
      int qt=h0_begin(H0_FINE_LIVE_TOTAL);csr_pc->apply((const AMGReal*)x,(AMGReal*)y);h0_end(qt);
    } else apply_with_rau(x,y,rau_live,true);
  }
  void apply_pc_mf(const AMGReal*x,AMGReal*y){apply_with_rau(x,y,rau_pc.data(),false);}
  void apply_pc(const AMGReal*x,AMGReal*y){
    if(!csr_pc)throw std::runtime_error("H2B fine CSR pointer null");
    int qt=h0_begin(H0_FINE_PC_TOTAL);csr_pc->apply(x,y);h0_end(qt);
  }
  void bt_state(const StateReal*p){int qt=h0_begin(H0_MOM_BT_TOTAL);int q=h0_begin(H0_MOM_BT_ZERO);NODALS_CUDA(cudaMemset(v0.data(),0,v0.bytes()));NODALS_CUDA(cudaMemset(v1.data(),0,v1.bytes()));NODALS_CUDA(cudaMemset(v2.data(),0,v2.bytes()));h0_end(q);q=h0_begin(H0_MOM_BT_KERNEL);g4_bt3_kernel<<<g4grid(nc),G4B>>>(cells,p,nc,v0.data(),v1.data(),v2.data());NODALS_CUDA(cudaGetLastError());h0_end(q);h0_end(qt);}
  std::size_t bytes()const{return rau_pc.bytes()+diag_pc.bytes()+v0.bytes()+v1.bytes()+v2.bytes();}
};

struct G5GpuTransfer {
  int nFine=0,nCoarse=0;
  DeviceBuffer<std::int64_t> row;
  DeviceBuffer<std::int32_t> col;
  DeviceBuffer<AMGReal> val;
  std::size_t bytes()const{return row.bytes()+col.bytes()+val.bytes();}
  void restrict(const AMGReal*f,AMGReal*c,int level)const{int q=h0_begin(H0_RESTRICT,level);
    NODALS_CUDA(cudaMemset(c,0,(std::size_t)nCoarse*sizeof(AMGReal)));
    g5_sa_restrict_kernel<<<g4grid(nFine),G4B>>>(nFine,row.data(),col.data(),val.data(),f,c);NODALS_CUDA(cudaGetLastError());h0_end(q);
  }
  void prolong_add(const AMGReal*c,AMGReal*f,int level)const{int q=h0_begin(H0_PROLONG,level);
    g5_sa_prolong_add_kernel<<<g4grid(nFine),G4B>>>(nFine,row.data(),col.data(),val.data(),c,f);NODALS_CUDA(cudaGetLastError());h0_end(q);
  }
};
static G5GpuTransfer upload_g5_transfer(const SATransferHost&P){
  G5GpuTransfer G;G.nFine=P.nFine;G.nCoarse=P.nCoarse;G.row.allocate(P.row.size());G.col.allocate(P.col.size());G.val.allocate(P.val.size());
  G.row.upload(P.row.data(),P.row.size());G.col.upload(P.col.data(),P.col.size());auto pv=g5e_cast_vec<AMGReal>(P.val);G.val.upload(pv.data(),pv.size());return G;
}

struct G4AMG {
  G4PressureFine*fine=nullptr;
  std::vector<G4GpuCSR>L;
  std::vector<G5GpuTransfer>P;
  DeviceBuffer<AMGReal>fine_r,fine_tmp,fine_corr,fine_rhs,terminal_inv;
  double fineLambda=1.0;
  std::vector<double>levelLambda;
  int chebDegree=2,powerIts=16;
  double lambdaSafety=1.5,lambdaLowFraction=.05;
  int spectrumRefreshes=0;

  static double cheb_root(double hi,double lo,int k,int degree){
    const double centre=.5*(hi+lo),radius=.5*(hi-lo),pi=3.141592653589793238462643383279502884;
    return centre-radius*std::cos(pi*(2.0*(double)k+1.0)/(2.0*(double)degree));
  }
  void cheb_explicit(G4GpuCSR&A,const AMGReal*b,AMGReal*x,double lambda,int level){
    const double lo=lambdaLowFraction*lambda;if(!(lambda>lo&&lo>0))throw std::runtime_error("G5C explicit Cheb interval invalid");
    double w0=1.0/cheb_root(lambda,lo,0,chebDegree);
    g4_jacobi_zero_kernel<<<g4grid(A.n),G4B>>>(A.n,w0,b,A.diag.data(),x);NODALS_CUDA(cudaGetLastError());
    for(int k=1;k<chebDegree;++k){A.apply(x,A.tmp.data(),level);g4_residual_kernel<<<g4grid(A.n),G4B>>>(A.n,b,A.tmp.data(),A.r.data());NODALS_CUDA(cudaGetLastError());double w=1.0/cheb_root(lambda,lo,k,chebDegree);g4_jacobi_add_kernel<<<g4grid(A.n),G4B>>>(A.n,w,A.r.data(),A.diag.data(),x);NODALS_CUDA(cudaGetLastError());}
  }
  void cheb_fine(const AMGReal*b,AMGReal*x){
    const double lo=lambdaLowFraction*fineLambda;if(!(fineLambda>lo&&lo>0))throw std::runtime_error("G5C fine Cheb interval invalid");
    double w0=1.0/cheb_root(fineLambda,lo,0,chebDegree);g4_jacobi_zero_kernel<<<g4grid(fine->nc),G4B>>>(fine->nc,w0,b,fine->diag_pc.data(),x);NODALS_CUDA(cudaGetLastError());
    for(int k=1;k<chebDegree;++k){fine->apply_pc(x,fine_tmp.data());g4_residual_kernel<<<g4grid(fine->nc),G4B>>>(fine->nc,b,fine_tmp.data(),fine_r.data());NODALS_CUDA(cudaGetLastError());double w=1.0/cheb_root(fineLambda,lo,k,chebDegree);g4_jacobi_add_kernel<<<g4grid(fine->nc),G4B>>>(fine->nc,w,fine_r.data(),fine->diag_pc.data(),x);NODALS_CUDA(cudaGetLastError());}
  }
  void explicit_vcycle(int l,const AMGReal*b,AMGReal*x){
    auto&A=L[(std::size_t)l];
    if(l==(int)L.size()-1){int q=h0_begin(H0_TERMINAL_DENSE,l);g4_dense_mv_kernel<<<g4grid(A.n),G4B>>>(A.n,terminal_inv.data(),b,x);NODALS_CUDA(cudaGetLastError());h0_end(q);return;}
    cheb_explicit(A,b,x,levelLambda[(std::size_t)l],l);
    A.apply(x,A.tmp.data(),l);g4_residual_kernel<<<g4grid(A.n),G4B>>>(A.n,b,A.tmp.data(),A.r.data());NODALS_CUDA(cudaGetLastError());
    auto&C=L[(std::size_t)l+1];P[(std::size_t)l+1].restrict(A.r.data(),C.b.data(),l+1);
    explicit_vcycle(l+1,C.b.data(),C.x.data());
    P[(std::size_t)l+1].prolong_add(C.x.data(),x,l+1);
    A.apply(x,A.tmp.data(),l);g4_residual_kernel<<<g4grid(A.n),G4B>>>(A.n,b,A.tmp.data(),A.r.data());NODALS_CUDA(cudaGetLastError());
    g4_copy_kernel<<<g4grid(A.n),G4B>>>(A.n,A.r.data(),A.b.data());NODALS_CUDA(cudaGetLastError());
    cheb_explicit(A,A.b.data(),A.corr.data(),levelLambda[(std::size_t)l],l);
    g4_axpy_kernel<<<g4grid(A.n),G4B>>>(A.n,1.0,A.corr.data(),x);NODALS_CUDA(cudaGetLastError());
  }
  void apply(const AMGReal*b,AMGReal*z){
    cheb_fine(b,z);
    fine->apply_pc(z,fine_tmp.data());g4_residual_kernel<<<g4grid(fine->nc),G4B>>>(fine->nc,b,fine_tmp.data(),fine_r.data());NODALS_CUDA(cudaGetLastError());
    auto&C=L[0];P[0].restrict(fine_r.data(),C.b.data(),0);explicit_vcycle(0,C.b.data(),C.x.data());P[0].prolong_add(C.x.data(),z,0);
    fine->apply_pc(z,fine_tmp.data());g4_residual_kernel<<<g4grid(fine->nc),G4B>>>(fine->nc,b,fine_tmp.data(),fine_r.data());NODALS_CUDA(cudaGetLastError());
    g4_copy_kernel<<<g4grid(fine->nc),G4B>>>(fine->nc,fine_r.data(),fine_rhs.data());NODALS_CUDA(cudaGetLastError());
    cheb_fine(fine_rhs.data(),fine_corr.data());g4_axpy_kernel<<<g4grid(fine->nc),G4B>>>(fine->nc,1.0,fine_corr.data(),z);NODALS_CUDA(cudaGetLastError());
  }
  std::size_t bytes()const{std::size_t b=fine_r.bytes()+fine_tmp.bytes()+fine_corr.bytes()+fine_rhs.bytes()+terminal_inv.bytes();for(const auto&t:P)b+=t.bytes();for(const auto&x:L)b+=x.bytes();return b;}
};
struct G4PCGWorkspace{DeviceBuffer<StateReal>x,r,z,p,q;};
struct G4PCGResult{int its=0;double rel=1;bool ok=false;};
static G4PCGResult pressure_pcg(G4PressureFine&F,G4AMG&A,const StateReal*rhs,double rtol,double atol,int maxit,G4PCGWorkspace&W,DeviceBuffer<double>&dotScratch){
  int n=F.nc;NODALS_CUDA(cudaMemset(W.x.data(),0,W.x.bytes()));g4_copy_kernel<<<g4grid(n),G4B>>>(n,rhs,W.r.data());NODALS_CUDA(cudaGetLastError());double r0=h0_pressure_norm2(W.r.data(),n,dotScratch);if(!std::isfinite(r0))throw std::runtime_error("G4 pressure initial norm nonfinite");if(r0<=atol){NODALS_CUDA(cudaMemset(W.x.data(),0,W.x.bytes()));return {0,0,true};}
  int qamg=h0_begin(H0_AMG_TOTAL);A.apply(W.r.data(),W.z.data());h0_end(qamg);g4_copy_kernel<<<g4grid(n),G4B>>>(n,W.z.data(),W.p.data());NODALS_CUDA(cudaGetLastError());double rho=h0_pressure_dot(W.r.data(),W.z.data(),n,dotScratch);if(!(rho>0)||!std::isfinite(rho))throw std::runtime_error("G4 pressure PCG initial rho nonpositive");G4PCGResult R;
  double target=std::max(atol,rtol*r0);for(int k=0;k<maxit;++k){F.apply_live(W.p.data(),W.q.data());double pq=h0_pressure_dot(W.p.data(),W.q.data(),n,dotScratch);if(!(pq>0)||!std::isfinite(pq))throw std::runtime_error("G4 pressure PCG pAp nonpositive");double alpha=rho/pq;g4_pcg_xr_kernel<<<g4grid(n),G4B>>>(n,alpha,W.p.data(),W.q.data(),W.x.data(),W.r.data());NODALS_CUDA(cudaGetLastError());double rn=h0_pressure_norm2(W.r.data(),n,dotScratch);R.its=k+1;R.rel=rn/std::max(r0,1e-300);if(rn<=target){R.ok=true;break;}qamg=h0_begin(H0_AMG_TOTAL);A.apply(W.r.data(),W.z.data());h0_end(qamg);double rhon=h0_pressure_dot(W.r.data(),W.z.data(),n,dotScratch);if(!(rhon>0)||!std::isfinite(rhon))throw std::runtime_error("G4 pressure PCG rho nonpositive");g4_pcg_p_kernel<<<g4grid(n),G4B>>>(n,rhon/rho,W.z.data(),W.p.data());NODALS_CUDA(cudaGetLastError());rho=rhon;}return R;
}

struct G4Gpu {
  int nv=0,nc=0,ncolors=0;DeviceBuffer<G4CellPlanDevice> cells;DeviceBuffer<StateReal>fixed;
  DeviceBuffer<std::int64_t> row;DeviceBuffer<std::int32_t>col,diagPos,colorOffset,colorRows;DeviceBuffer<OperatorReal>diffusion,av,delta,diag,rau;
  DeviceBuffer<StateReal>u0,u1,u2,p,s0,s1,s2,c0,c1,c2,b0,b1,b2,fixedDiv,cont;
  DeviceBuffer<double>momSums;H2FineCSRDevice fineCsr;G4PressureFine pf;G4AMG amg;G4PCGWorkspace pcg;DeviceBuffer<double>dotScratch;
  std::size_t bytes()const{return cells.bytes()+fixed.bytes()+row.bytes()+col.bytes()+diagPos.bytes()+colorOffset.bytes()+colorRows.bytes()+diffusion.bytes()+av.bytes()+delta.bytes()+diag.bytes()+rau.bytes()+u0.bytes()+u1.bytes()+u2.bytes()+p.bytes()+s0.bytes()+s1.bytes()+s2.bytes()+c0.bytes()+c1.bytes()+c2.bytes()+b0.bytes()+b1.bytes()+b2.bytes()+fixedDiv.bytes()+cont.bytes()+momSums.bytes()+fineCsr.bytes()+pf.bytes()+amg.bytes()+pcg.x.bytes()+pcg.r.bytes()+pcg.z.bytes()+pcg.p.bytes()+pcg.q.bytes()+dotScratch.bytes();}
};

static std::vector<std::int32_t> diag_positions(const MomentumCSRHost&A){std::vector<std::int32_t>d((std::size_t)A.n,-1);for(int i=0;i<A.n;++i){auto first=A.col.begin()+A.row[(std::size_t)i],last=A.col.begin()+A.row[(std::size_t)i+1];auto it=std::lower_bound(first,last,i);if(it==last||*it!=i)throw std::runtime_error("G4 diag position absent");auto k=(std::int64_t)(it-A.col.begin());if(k>INT32_MAX)throw std::runtime_error("G4 diag position exceeds int32");d[(std::size_t)i]=(std::int32_t)k;}return d;}
static void upload_tensors(){
  const auto&D=diffusion_tensor_g3();const auto&C=central_tensor_g4();
  OperatorReal df[8*8*3*3],ct[8*8*8*3];
  for(std::size_t i=0;i<sizeof(df)/sizeof(df[0]);++i)df[i]=(OperatorReal)reinterpret_cast<const double*>(D.t)[i];
  for(std::size_t i=0;i<sizeof(ct)/sizeof(ct[0]);++i)ct[i]=(OperatorReal)reinterpret_cast<const double*>(C.t)[i];
  NODALS_CUDA(cudaMemcpyToSymbol(g4_diffT,df,sizeof(df)));NODALS_CUDA(cudaMemcpyToSymbol(g4_centT,ct,sizeof(ct)));
}
static double relvec(const std::vector<double>&a,const std::vector<double>&b,double&mx){long double d=0,r=0;mx=0;if(a.size()!=b.size())throw std::runtime_error("G4 parity size mismatch");for(std::size_t i=0;i<a.size();++i){double q=a[i]-b[i];d+=(long double)q*q;r+=(long double)b[i]*b[i];mx=std::max(mx,std::abs(q));}return std::sqrt((double)d)/std::max(std::sqrt((double)r),1e-300);}

static G4Gpu upload_all(const G4SetupHost&S,const SAHierarchyHost&H,const std::vector<double>&initRau,const H2FineCSRHost&FH){
  G4Gpu G;G.nv=S.topo.n;G.nc=(int)S.cells.size();G.ncolors=S.coloring.ncolors;
  auto dc=g5e_cast_cells(S.cells);G.cells.allocate(dc.size());G.cells.upload(dc.data(),dc.size());
  std::vector<StateReal>fv(S.fixedValue.size()*3);for(std::size_t i=0;i<S.fixedValue.size();++i)for(int d=0;d<3;++d)fv[3*i+d]=(StateReal)S.fixedValue[i][d];
  G.fixed.allocate(fv.size());G.fixed.upload(fv.data(),fv.size());
  G.row.allocate(S.topo.row.size());G.col.allocate(S.topo.col.size());G.diffusion.allocate(S.topo.val.size());G.av.allocate(S.topo.val.size());G.row.upload(S.topo.row.data(),S.topo.row.size());G.col.upload(S.topo.col.data(),S.topo.col.size());auto diff0=g5e_cast_vec<OperatorReal>(S.topo.val);G.diffusion.upload(diff0.data(),diff0.size());auto dp=diag_positions(S.topo);G.diagPos.allocate(dp.size());G.diagPos.upload(dp.data(),dp.size());
  G.colorOffset.allocate(S.coloring.offset.size());G.colorRows.allocate(S.coloring.rows.size());G.colorOffset.upload(S.coloring.offset.data(),S.coloring.offset.size());G.colorRows.upload(S.coloring.rows.data(),S.coloring.rows.size());
  G.delta.allocate(G.nv);G.diag.allocate(G.nv);G.rau.allocate(G.nv);auto ir=g5e_cast_vec<OperatorReal>(initRau);G.rau.upload(ir.data(),ir.size());
  for(auto*p:{&G.u0,&G.u1,&G.u2,&G.s0,&G.s1,&G.s2,&G.c0,&G.c1,&G.c2,&G.b0,&G.b1,&G.b2})p->allocate(G.nv);
  G.p.allocate(G.nc);G.fixedDiv.allocate(G.nc);G.cont.allocate(G.nc);G.momSums.allocate(6);
  auto s0=g5e_cast_vec<StateReal>(S.staticRhs[0]),s1=g5e_cast_vec<StateReal>(S.staticRhs[1]),s2=g5e_cast_vec<StateReal>(S.staticRhs[2]),fd=g5e_cast_vec<StateReal>(S.fixedDiv);
  G.s0.upload(s0.data(),G.nv);G.s1.upload(s1.data(),G.nv);G.s2.upload(s2.data(),G.nv);G.fixedDiv.upload(fd.data(),G.nc);
  NODALS_CUDA(cudaMemset(G.u0.data(),0,G.u0.bytes()));NODALS_CUDA(cudaMemset(G.u1.data(),0,G.u1.bytes()));NODALS_CUDA(cudaMemset(G.u2.data(),0,G.u2.bytes()));NODALS_CUDA(cudaMemset(G.p.data(),0,G.p.bytes()));
  G.fineCsr=upload_h2_fine_csr(FH);
  G.pf.nc=G.nc;G.pf.nv=G.nv;G.pf.cells=G.cells.data();G.pf.rau_live=G.rau.data();G.pf.csr_pc=&G.fineCsr;G.pf.rau_pc.allocate(ir.size());G.pf.rau_pc.upload(ir.data(),ir.size());auto fdiag=g5e_cast_vec<AMGReal>(H.fineDiag);G.pf.diag_pc.allocate(G.nc);G.pf.diag_pc.upload(fdiag.data(),fdiag.size());G.pf.v0.allocate(G.nv);G.pf.v1.allocate(G.nv);G.pf.v2.allocate(G.nv);
  G.amg.fine=&G.pf;G.amg.fineLambda=H.fineLambda;G.amg.levelLambda=H.levelLambda;G.amg.chebDegree=2;G.amg.powerIts=H.powerIts;G.amg.lambdaSafety=H.lambdaSafety;G.amg.lambdaLowFraction=H.lambdaLowFraction;
  G.amg.fine_r.allocate(G.nc);G.amg.fine_tmp.allocate(G.nc);G.amg.fine_corr.allocate(G.nc);G.amg.fine_rhs.allocate(G.nc);
  G.amg.P.reserve(H.P.size());for(const auto&P:H.P)G.amg.P.push_back(upload_g5_transfer(P));
  G.amg.L.reserve(H.csr.size());for(const auto&A:H.csr)G.amg.L.push_back(upload_g4_csr(A));
  auto tinv=g5e_cast_vec<AMGReal>(H.terminal_inv);G.amg.terminal_inv.allocate(tinv.size());G.amg.terminal_inv.upload(tinv.data(),tinv.size());
  for(auto*p:{&G.pcg.x,&G.pcg.r,&G.pcg.z,&G.pcg.p,&G.pcg.q})p->allocate(G.nc);G.dotScratch.allocate(1);
  NODALS_CUDA(cudaDeviceSynchronize());return G;
}

static double g5_power_fine_gpu(G4Gpu&G,int its,double safety){
  auto&v=G.amg.fine_corr;auto&u=G.amg.fine_rhs;auto&au=G.amg.fine_tmp;auto&w=G.amg.fine_r;
  g5_power_init_kernel<<<g4grid(G.nc),G4B>>>(G.nc,v.data());NODALS_CUDA(cudaGetLastError());double vn=device_norm2(v.data(),G.nc,G.dotScratch);device_scale(v.data(),(std::size_t)G.nc,AMGReal(1.0/vn));double ray=0.0;
  for(int it=0;it<its;++it){g5_div_sqrt_diag_kernel<<<g4grid(G.nc),G4B>>>(G.nc,v.data(),G.pf.diag_pc.data(),u.data());NODALS_CUDA(cudaGetLastError());G.pf.apply_pc(u.data(),au.data());g5_div_sqrt_diag_kernel<<<g4grid(G.nc),G4B>>>(G.nc,au.data(),G.pf.diag_pc.data(),w.data());NODALS_CUDA(cudaGetLastError());double num=device_dot(v.data(),w.data(),G.nc,G.dotScratch),den=device_dot(v.data(),v.data(),G.nc,G.dotScratch);ray=num/std::max(den,1e-300);double wn=device_norm2(w.data(),G.nc,G.dotScratch);if(!(ray>0.0)||!(wn>0.0)||!std::isfinite(ray)||!std::isfinite(wn))throw std::runtime_error("G5C fine GPU power failed");g4_copy_kernel<<<g4grid(G.nc),G4B>>>(G.nc,w.data(),v.data());NODALS_CUDA(cudaGetLastError());device_scale(v.data(),(std::size_t)G.nc,AMGReal(1.0/wn));}
  return safety*ray;
}
static double g5_power_csr_gpu(G4Gpu&G,G4GpuCSR&A,int its,double safety){
  g5_power_init_kernel<<<g4grid(A.n),G4B>>>(A.n,A.corr.data());NODALS_CUDA(cudaGetLastError());double vn=device_norm2(A.corr.data(),A.n,G.dotScratch);device_scale(A.corr.data(),(std::size_t)A.n,AMGReal(1.0/vn));double ray=0.0;
  for(int it=0;it<its;++it){g5_div_sqrt_diag_kernel<<<g4grid(A.n),G4B>>>(A.n,A.corr.data(),A.diag.data(),A.b.data());NODALS_CUDA(cudaGetLastError());A.apply(A.b.data(),A.tmp.data());g5_div_sqrt_diag_kernel<<<g4grid(A.n),G4B>>>(A.n,A.tmp.data(),A.diag.data(),A.r.data());NODALS_CUDA(cudaGetLastError());double num=device_dot(A.corr.data(),A.r.data(),A.n,G.dotScratch),den=device_dot(A.corr.data(),A.corr.data(),A.n,G.dotScratch);ray=num/std::max(den,1e-300);double wn=device_norm2(A.r.data(),A.n,G.dotScratch);if(!(ray>0.0)||!(wn>0.0)||!std::isfinite(ray)||!std::isfinite(wn))throw std::runtime_error("G5C coarse GPU power failed");g4_copy_kernel<<<g4grid(A.n),G4B>>>(A.n,A.r.data(),A.corr.data());NODALS_CUDA(cudaGetLastError());device_scale(A.corr.data(),(std::size_t)A.n,AMGReal(1.0/wn));}
  return safety*ray;
}
static void g5_gpu_spectrum_refresh(G4Gpu&G,const SAHierarchyHost&H){
  CudaEventTimer T;T.start();double old=G.amg.fineLambda;G.amg.fineLambda=g5_power_fine_gpu(G,G.amg.powerIts,G.amg.lambdaSafety);
  std::printf("NODALS_GPU_H4_SPECTRUM level=0 powerIts=%d hostLambda=%.12e gpuLambda=%.12e ratio=%.6f status=PASS\n",G.amg.powerIts,old,G.amg.fineLambda,G.amg.fineLambda/std::max(old,1e-300));
  for(std::size_t l=0;l+1<G.amg.L.size();++l){double h=(l<H.levelLambda.size()?H.levelLambda[l]:0.0);double q=g5_power_csr_gpu(G,G.amg.L[l],G.amg.powerIts,G.amg.lambdaSafety);G.amg.levelLambda[l]=q;std::printf("NODALS_GPU_H4_SPECTRUM level=%zu powerIts=%d hostLambda=%.12e gpuLambda=%.12e ratio=%.6f status=PASS\n",l+1,G.amg.powerIts,h,q,q/std::max(h,1e-300));}
  NODALS_CUDA(cudaDeviceSynchronize());float ms=T.stop();++G.amg.spectrumRefreshes;std::printf("NODALS_GPU_H4_POWER_SETUP powerIts=%d levels=%zu ms=%.3f scalarD2H_only=1 status=PASS\n",G.amg.powerIts,G.amg.L.size(),ms);
}
static double g5_rau_snapshot_rel(G4Gpu&G){
  g5_diff_kernel<<<g4grid(G.nv),G4B>>>(G.nv,G.rau.data(),G.pf.rau_pc.data(),G.pf.v0.data());NODALS_CUDA(cudaGetLastError());double dn=device_norm2(G.pf.v0.data(),G.nv,G.dotScratch);double rn=device_norm2(G.pf.rau_pc.data(),G.nv,G.dotScratch);return dn/std::max(rn,1e-300);
}


static void h2_action_parity_and_bench(G4Gpu&G,const char*tag,int reps=24){
  auto&x=G.amg.fine_rhs;auto&ymf=G.amg.fine_tmp;auto&ycsr=G.amg.fine_r;auto&diff=G.amg.fine_corr;
  g5_power_init_kernel<<<g4grid(G.nc),G4B>>>(G.nc,x.data());NODALS_CUDA(cudaGetLastError());
  G.pf.apply_pc_mf(x.data(),ymf.data());G.fineCsr.apply(x.data(),ycsr.data());
  g5_diff_kernel<<<g4grid(G.nc),G4B>>>(G.nc,ycsr.data(),ymf.data(),diff.data());NODALS_CUDA(cudaGetLastError());
  double dn=device_norm2(diff.data(),G.nc,G.dotScratch),rn=device_norm2(ymf.data(),G.nc,G.dotScratch);
  double rel=dn/std::max(rn,1e-300);
  std::printf("NODALS_GPU_H4_CSR_PARITY tag=%s relL2=%.12e tol=5e-5 status=%s\n",tag,rel,rel<=5e-5?"PASS":"FAIL");
  if(!(rel<=5e-5))throw std::runtime_error("H2B fine CSR action parity failed");

  for(int k=0;k<3;++k){G.pf.apply_pc_mf(x.data(),ymf.data());G.fineCsr.apply(x.data(),ycsr.data());}
  NODALS_CUDA(cudaDeviceSynchronize());
  CudaEventTimer T1;T1.start();for(int k=0;k<reps;++k)G.pf.apply_pc_mf(x.data(),ymf.data());double mf=(double)T1.stop()/reps;
  CudaEventTimer T2;T2.start();for(int k=0;k<reps;++k)G.fineCsr.apply(x.data(),ycsr.data());double cs=(double)T2.stop()/reps;
  std::printf("NODALS_GPU_H4_ACTION_BENCH tag=%s reps=%d matrixFreeMs=%.6f csrWarpMs=%.6f speedup=%.6f savedMsPerAction=%.6f status=PASS\n",
    tag,reps,mf,cs,mf/cs,mf-cs);
}

static void h2b_refresh_bench(G4Gpu&G,const char*tag,int reps=12){
  // Compare H2 serial-per-row refresh against H2B warp-per-row using identical
  // setup rAU.  The final operation is warp refresh, leaving CSR numerics valid.
  for(int k=0;k<2;++k)G.fineCsr.refresh_serial_reference(G.cells.data(),G.rau.data(),G.pf.diag_pc.data());
  NODALS_CUDA(cudaDeviceSynchronize());
  CudaEventTimer T0;T0.start();
  for(int k=0;k<reps;++k)G.fineCsr.refresh_serial_reference(G.cells.data(),G.rau.data(),G.pf.diag_pc.data());
  double serialMs=(double)T0.stop()/reps;

  for(int k=0;k<2;++k)G.fineCsr.refresh(G.cells.data(),G.rau.data(),G.pf.diag_pc.data());
  NODALS_CUDA(cudaDeviceSynchronize());
  CudaEventTimer T1;T1.start();
  for(int k=0;k<reps;++k)G.fineCsr.refresh(G.cells.data(),G.rau.data(),G.pf.diag_pc.data());
  double warpMs=(double)T1.stop()/reps;
  // One final current warp refresh for deterministic post-benchmark state.
  G.fineCsr.refresh(G.cells.data(),G.rau.data(),G.pf.diag_pc.data());
  NODALS_CUDA(cudaDeviceSynchronize());
  std::printf("NODALS_GPU_H4_REFRESH_BENCH tag=%s reps=%d serialRowMs=%.6f warpRowMs=%.6f speedup=%.6f savedMsPerRefresh=%.6f status=PASS\\n",
    tag,reps,serialMs,warpMs,serialMs/warpMs,serialMs-warpMs);
}

static void assemble_live(G4Gpu&G,double alphaU){
  NODALS_CUDA(cudaMemcpyAsync(G.av.data(),G.diffusion.data(),G.av.bytes(),cudaMemcpyDeviceToDevice));
  NODALS_CUDA(cudaMemsetAsync(G.c0.data(),0,G.c0.bytes()));
  NODALS_CUDA(cudaMemsetAsync(G.c1.data(),0,G.c1.bytes()));
  NODALS_CUDA(cudaMemsetAsync(G.c2.data(),0,G.c2.bytes()));
  h4_assemble_convection_only_kernel<<<g4grid(G.nc),G4B>>>(
    G.cells.data(),G.nc,G.row.data(),G.fixed.data(),
    G.u0.data(),G.u1.data(),G.u2.data(),
    G.av.data(),G.c0.data(),G.c1.data(),G.c2.data());
  NODALS_CUDA(cudaGetLastError());
  g4_finalize_relax_kernel<<<g4grid(G.nv),G4B>>>(
    G.nv,G.row.data(),G.col.data(),G.diagPos.data(),OperatorReal(alphaU),
    G.av.data(),G.delta.data(),G.diag.data(),G.rau.data());
  NODALS_CUDA(cudaGetLastError());
}
static void forward_mcgs_fixed(G4Gpu&G,const ColoringHost&C,double omega){
  int q=h0_begin(H0_MOM_FORWARD);
  for(int c=0;c<C.ncolors;++c){
    int s=C.offset[(std::size_t)c],n=C.offset[(std::size_t)c+1]-s;
    if(n)g4_mcgs_color_kernel<<<g4grid(n),G4B>>>(n,G.colorRows.data()+s,G.row.data(),G.col.data(),G.av.data(),G.diag.data(),G.b0.data(),G.b1.data(),G.b2.data(),G.u0.data(),G.u1.data(),G.u2.data(),OperatorReal(omega),1,1,1);
  }
  NODALS_CUDA(cudaGetLastError());h0_end(q);
}
static void backward_mcgs_fixed(G4Gpu&G,const ColoringHost&C,double omega){
  int q=h0_begin(H0_MOM_BACKWARD);
  for(int c=C.ncolors-1;c>=0;--c){
    int s=C.offset[(std::size_t)c],n=C.offset[(std::size_t)c+1]-s;
    if(n)g4_mcgs_color_kernel<<<g4grid(n),G4B>>>(n,G.colorRows.data()+s,G.row.data(),G.col.data(),G.av.data(),G.diag.data(),G.b0.data(),G.b1.data(),G.b2.data(),G.u0.data(),G.u1.data(),G.u2.data(),OperatorReal(omega),1,1,1);
  }
  NODALS_CUDA(cudaGetLastError());h0_end(q);
}
static const char* fixed_momentum_work(G4Gpu&G,const ColoringHost&C,double omega,const std::string&mode,int outer){
  if(mode=="fgs1"){
    forward_mcgs_fixed(G,C,omega);
    return "FWD";
  }
  if(mode=="altgs1"){
    if(outer&1){forward_mcgs_fixed(G,C,omega);return "FWD";}
    backward_mcgs_fixed(G,C,omega);return "BWD";
  }
  throw std::runtime_error("H4 unknown fixed momentum work mode");
}
static std::array<double,3> momentum_initial_rel_audit(G4Gpu&G){
  int qp=h0_begin(H0_MOM_NORM_INITIAL);
  NODALS_CUDA(cudaMemset(G.momSums.data(),0,G.momSums.bytes()));
  g4_mom_norms_kernel<<<g4grid(G.nv),G4B>>>(G.nv,G.row.data(),G.col.data(),G.av.data(),G.b0.data(),G.b1.data(),G.b2.data(),G.u0.data(),G.u1.data(),G.u2.data(),G.momSums.data());
  NODALS_CUDA(cudaGetLastError());
  double h[6];G.momSums.download(h,6);h0_end(qp);
  std::array<double,3>q{};
  for(int d=0;d<3;++d){double bn=std::sqrt(std::max(0.0,h[d]));if(bn==0.0)bn=1.0;double rn=std::sqrt(std::max(0.0,h[3+d]));q[(std::size_t)d]=rn/bn;}
  return q;
}


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

static void h0_print_profile(const char*tag,int outer,double pressureStageMs,double momentumStageMs,const H0Profiler&P){
  std::printf("NODALS_GPU_H4_PROFILE_SUMMARY tag=%s outer=%d pressureStageTotalMs=%.6f momentumStageTotalMs=%.6f records=%lld eventTiming=NO_EXTRA_PER_SPAN_SYNC status=PASS\n",tag,outer,pressureStageMs,momentumStageMs,P.recordCount);
  for(int c=0;c<H0_CAT_COUNT;++c){for(int sl=0;sl<=H0Profiler::MAX_LEVEL;++sl){if(P.calls[c][sl]==0)continue;int lev=sl==0?-1:sl-1;double t=P.total[c][sl],w=P.warm[c][sl];long long n=P.calls[c][sl],nw=P.warmCalls[c][sl];
    const bool mom=c>=H0_MOM_BT_TOTAL;double denom=mom?momentumStageMs:pressureStageMs;double pct=denom>0?100.0*t/denom:0.0;
    std::printf("NODALS_GPU_H4_PROFILE tag=%s domain=%s category=%s level=%d calls=%lld totalMs=%.6f perCallMs=%.6f perSimpleMs=%.6f pctStage=%.3f warmCalls=%lld warmTotalMs=%.6f warmPerSimpleMs=%.6f\n",tag,mom?"momentum":"pressure",h0_cat_name(c),lev,n,t,t/std::max<long long>(n,1),t/std::max(outer,1),pct,nw,w,w/std::max(outer-1,1));
  }}
}

int main(int argc,char**argv){try{
  std::string mesh,tag="40k_fp32_tune",wall="patch_0_0",inlet="patch_2_0",outlet="patch_1_0";
  double re=20,bulk=1,simpleTol=1e-6;
  double alphaU=.5,alphaP=.5;
  double momRtol=1e-6,momAtol=1e-12,momDrop=.1,momOmega=1.0;
  double pRtol=.5,pAtol=1e-12;
  double snapshotTol=5e-6;
  int maxOuter=2500,momMax=20000,pMax=20;
  int fineCsrRefreshEvery=1;
  std::string momentumWork="fgs1",runMode="fixed10";

  for(int i=1;i<argc;++i){std::string a=argv[i];
    if(a=="--mesh"&&i+1<argc)mesh=argv[++i];else if(a=="--tag"&&i+1<argc)tag=argv[++i];
    else if(a=="--wall"&&i+1<argc)wall=argv[++i];else if(a=="--inlet"&&i+1<argc)inlet=argv[++i];
    else if(a=="--outlet"&&i+1<argc)outlet=argv[++i];else if(a=="--re"&&i+1<argc)re=std::atof(argv[++i]);
    else if(a=="--bulk"&&i+1<argc)bulk=std::atof(argv[++i]);
    else if(a=="--simple-tol"&&i+1<argc)simpleTol=std::atof(argv[++i]);
    else if(a=="--max-outer"&&i+1<argc)maxOuter=std::atoi(argv[++i]);
    else if(a=="--alpha-u"&&i+1<argc)alphaU=std::atof(argv[++i]);
    else if(a=="--alpha-p"&&i+1<argc)alphaP=std::atof(argv[++i]);
    else if(a=="--mom-rtol"&&i+1<argc)momRtol=std::atof(argv[++i]);
    else if(a=="--mom-atol"&&i+1<argc)momAtol=std::atof(argv[++i]);
    else if(a=="--mom-drop"&&i+1<argc)momDrop=std::atof(argv[++i]);
    else if(a=="--mom-omega"&&i+1<argc)momOmega=std::atof(argv[++i]);
    else if(a=="--mom-max"&&i+1<argc)momMax=std::atoi(argv[++i]);
    else if(a=="--p-rtol"&&i+1<argc)pRtol=std::atof(argv[++i]);
    else if(a=="--p-atol"&&i+1<argc)pAtol=std::atof(argv[++i]);
    else if(a=="--p-max"&&i+1<argc)pMax=std::atoi(argv[++i]);
    else if(a=="--snapshot-tol"&&i+1<argc)snapshotTol=std::atof(argv[++i]);
    else if(a=="--fine-csr-refresh-every"&&i+1<argc)fineCsrRefreshEvery=std::atoi(argv[++i]);
    else if(a=="--momentum-work"&&i+1<argc)momentumWork=argv[++i];
    else if(a=="--run-mode"&&i+1<argc)runMode=argv[++i];
    else throw std::runtime_error("H4 usage error");
  }
  if(mesh.empty())throw std::runtime_error("--mesh required");
  if(!(simpleTol>0.0) || maxOuter<1)throw std::runtime_error("H4 invalid SIMPLE controls");
  if(!(alphaU>0.0&&alphaU<=1.0) || !(alphaP>0.0&&alphaP<=1.0))throw std::runtime_error("H4 alpha-u/alpha-p must be in (0,1]");
  if(momRtol<0.0 || momAtol<0.0 || momDrop<0.0 || momMax<1 || !(momOmega>0.0))throw std::runtime_error("H4 invalid momentum controls");
  if(pRtol<0.0 || pAtol<0.0 || pMax<1)throw std::runtime_error("H4 invalid pressure controls");
  if(!(snapshotTol>0.0))throw std::runtime_error("H4 snapshot tolerance must be positive");
  if(fineCsrRefreshEvery!=1)throw std::runtime_error("H4 requires fine CSR refreshEvery=1");
  if(!(momentumWork=="fgs1"||momentumWork=="altgs1"))throw std::runtime_error("H4 momentum-work must be fgs1 or altgs1");
  if(runMode!="fixed10")throw std::runtime_error("H4 is fixed10-only");
  if(!(runMode=="fixed10"||runMode=="converge"))throw std::runtime_error("H4 run-mode must be fixed10 or converge");
  if(runMode=="fixed10"&&maxOuter!=10)throw std::runtime_error("H4 fixed10 requires maxOuter=10");
  const bool physicalUseCurrentCSR=true;

  NODALS_CUDA(cudaSetDevice(0));cudaDeviceProp prop{};NODALS_CUDA(cudaGetDeviceProperties(&prop,0));upload_tensors();NODALS_CUDA(cudaDeviceSynchronize());
  const auto memBaseline=device_memory_info();
  auto M=load_foam_tet_mesh(mesh);auto S=build_g4_setup(M,re,bulk,wall,inlet,outlet);
  std::array<std::vector<double>,3>U0;for(auto&u:U0)u.assign((std::size_t)S.topo.n,0.0);
  std::vector<double>hostA,initDelta;std::array<std::vector<double>,3>hostConv;
  host_assemble_central_g4(S,U0,hostA,hostConv);auto initRau=host_finalize_relax_g4(S,hostA,alphaU,&initDelta);S.pressure.rAU=initRau;
  auto H=build_sa_hierarchy(M,S.pressure,16,6,18,1000,8,16,1.5,0.05,4.0/3.0);
  auto FH=build_h2_fine_csr_host(S);

  std::printf("NODALS_GPU_H4_CONFIG tag=%s precision=%s device=%s cc=%d.%d petsc=NONE mpi=NONE cells=%zu runMode=%s momentumWork=%s momentumResidualPolicy=%s physicalOperator=exact_current_CSR fineAMG=explicit_FP32_CSR_warp fineCsrNumericRefreshEvery=1 refreshKernel=warp_per_row spectrumRefresh=setup_only coarseHierarchyNumeric=setup_snapshot momentumDiffusion=PERSISTENT_NUMERIC_CSR momentumConvection=NUMERIC_ONLY_SHARED_XYZ alphaU=%.8g alphaP=%.8g simpleTol=%.3e maxOuter=%d momentumOmega=%.8g pressureRtol=%.3e pressureAtol=%.3e pressureMaxIts=%d reductions=FP64 state=FP32 operator=FP32 amg=FP32\n",
    tag.c_str(),kPrecisionName,prop.name,prop.major,prop.minor,M.tets.size(),runMode.c_str(),momentumWork.c_str(),runMode=="fixed10"?"NONE":"CONVERGENCE_ONLY_PRE_SWEEP",alphaU,alphaP,simpleTol,maxOuter,momOmega,pRtol,pAtol,pMax);
  std::printf("NODALS_GPU_H4_TUNING alphaU=%.8g alphaP=%.8g momentumWork=%s momentumOmega=%.8g momentumAdaptiveTol=DISABLED momentumResidualPolicy=%s pRtol=%.3e pAtol=%.3e pMax=%d simpleTol=%.3e maxOuter=%d snapshotTol=%.3e fineCsrRefreshEvery=1 status=PASS\n",
    alphaU,alphaP,momentumWork.c_str(),momOmega,runMode=="fixed10"?"NONE":"CONVERGENCE_ONLY_PRE_SWEEP",pRtol,pAtol,pMax,simpleTol,maxOuter,snapshotTol);
  std::printf("NODALS_GPU_H4_FINE_CSR_TOPOLOGY tag=%s cells=%zu nnz=%zu rowMean=%.6f rowMax=%u directedContrib=%llu csrMiB=%.3f compactRefreshMetadataMiB=%.3f totalFineCsrMiB=%.3f bytesPerCell=%.3f status=PASS\n",
    tag.c_str(),M.tets.size(),FH.col.size(),FH.rowMean,FH.rowMax,(unsigned long long)FH.directedContrib,
    g5_mib(FH.csr_bytes()),g5_mib(FH.metadata_bytes()),g5_mib(FH.bytes()),(double)FH.bytes()/std::max<std::size_t>(M.tets.size(),1));
  std::printf("NODALS_GPU_H4_SETUP tag=%s cells=%zu freeVel=%d momentumNnz=%zu colors=%d hierarchyLevels=%zu terminal=%d transfer0Nnz=%zu cellPlanHostBytes=%zu cellPlanDeviceBytes=%zu baselineUsedMiB=%.3f totalMiB=%.3f status=PASS\n",
    tag.c_str(),M.tets.size(),S.topo.n,S.topo.val.size(),S.coloring.ncolors,H.csr.size(),H.terminal_n,H.P.empty()?0:H.P[0].val.size(),sizeof(G4CellPlanHost),sizeof(G4CellPlanDevice),g5_used_mib(memBaseline),(double)memBaseline.total_bytes/(1024.0*1024.0));
  std::printf("NODALS_GPU_H4_MOMENTUM_ASSEMBLY_DESIGN tag=%s diffusionCSR=BUILT_ONCE_HOST_UPLOADED_ONCE_DEVICE_PERSISTENT convectionCSRTopology=STATIC convectionNumeric=REFRESH_EACH_OUTER commonOperatorXYZ=YES pressureGradientBTopology=STATIC pressureGradientAction=APPLY_CURRENT_P fixedDiffusionDirichletRHS=PERSISTENT variableViscosityDesign=REFRESH_DIFFUSION_NUMERICS_ONLY_NO_TOPOLOGY_REBUILD status=PASS\n",tag.c_str());

  G4Gpu G=upload_all(S,H,initRau,FH);G.pf.cells=G.cells.data();G.pf.rau_live=G.rau.data();G.pf.csr_pc=&G.fineCsr;G.pf.physicalUseCurrentCSR=physicalUseCurrentCSR;G.amg.fine=&G.pf;
  G.fineCsr.refresh(G.cells.data(),G.rau.data(),G.pf.diag_pc.data());
  h2b_refresh_bench(G,tag.c_str(),12);
  g5_gpu_spectrum_refresh(G,H);
  h2_action_parity_and_bench(G,tag.c_str(),24);
  H0Profiler H0P;g_h0=&H0P;
  NODALS_CUDA(cudaDeviceSynchronize());const auto memUpload=device_memory_info();
  const double baselineUsed=g5_used_mib(memBaseline),uploadUsed=g5_used_mib(memUpload),explicitMiB=g5_mib(G.bytes());
  std::printf("NODALS_GPU_H4_MEMORY tag=%s point=after_upload cells=%zu baselineUsedMiB=%.3f usedMiB=%.3f deltaFromBaselineMiB=%.3f explicitMiB=%.3f explicitBytesPerCell=%.3f status=PASS\n",
    tag.c_str(),M.tets.size(),baselineUsed,uploadUsed,uploadUsed-baselineUsed,explicitMiB,(double)G.bytes()/std::max<std::size_t>(M.tets.size(),1));

  bool converged=false,pressureAll=true,finiteAll=true;double cont0=-1.0,contRel=1.0;
  long long sumP=0;int convergenceAuditCalls=0;std::array<double,3>lastAuditRel{{NAN,NAN,NAN}};
  double tAssembly=0,tMomentum=0,tContinuity=0,tPressure=0,tPupdate=0;
  auto wall0=std::chrono::steady_clock::now();
  int finalIt=0,fineCsrRefreshCount=0;
  for(int it=1;it<=maxOuter;++it){
    H0P.currentOuter=it;if(H0P.used)throw std::runtime_error("H4 profiler nonempty at outer start");
    G5StageEvents E;E.rec(0);
    assemble_live(G,alphaU);E.rec(1);
    if(it==1){
      double rr=g5_rau_snapshot_rel(G);
      std::printf("NODALS_GPU_H4_SNAPSHOT_PARITY it=1 rAURel=%.3e tol=%.3e status=%s\n",rr,snapshotTol,rr<snapshotTol?"PASS":"FAIL");
      if(!(rr<snapshotTol))throw std::runtime_error("H4 setup rAU snapshot mismatch");
    }
    G.pf.bt_state(G.p.data());
    {int q=h0_begin(H0_MOM_RHS);g4_momentum_rhs_kernel<<<g4grid(G.nv),G4B>>>(G.nv,G.s0.data(),G.s1.data(),G.s2.data(),G.c0.data(),G.c1.data(),G.c2.data(),G.pf.v0.data(),G.pf.v1.data(),G.pf.v2.data(),G.delta.data(),G.u0.data(),G.u1.data(),G.u2.data(),G.b0.data(),G.b1.data(),G.b2.data());NODALS_CUDA(cudaGetLastError());h0_end(q);}
    bool auditedThisOuter=false;
    std::array<double,3>auditRel{{NAN,NAN,NAN}};
    if(runMode=="converge" && it>1 && contRel<=simpleTol){
      auditRel=momentum_initial_rel_audit(G);lastAuditRel=auditRel;++convergenceAuditCalls;auditedThisOuter=true;
    }
    const char* momentumDirection=fixed_momentum_work(G,S.coloring,momOmega,momentumWork,it);E.rec(2);

    g4_continuity_kernel<<<g4grid(G.nc),G4B>>>(G.cells.data(),G.nc,G.fixedDiv.data(),G.u0.data(),G.u1.data(),G.u2.data(),G.cont.data());NODALS_CUDA(cudaGetLastError());
    double cn=device_norm2(G.cont.data(),G.nc,G.dotScratch);if(it==1)cont0=cn;contRel=cn/std::max(cont0,1e-300);
    g4_negate_kernel<<<g4grid(G.nc),G4B>>>(G.nc,G.cont.data());NODALS_CUDA(cudaGetLastError());E.rec(3);

    if(it==1 || ((it-1)%fineCsrRefreshEvery)==0){
      G.fineCsr.refresh(G.cells.data(),G.rau.data(),G.pf.diag_pc.data());
      ++fineCsrRefreshCount;
    }
    auto pr=pressure_pcg(G.pf,G.amg,G.cont.data(),pRtol,pAtol,pMax,G.pcg,G.dotScratch);
    pressureAll=pressureAll&&pr.ok;if(!pr.ok)throw std::runtime_error("H4 pressure PCG failed requested FP32 inexact target");sumP+=pr.its;E.rec(4);
    g4_axpy_kernel<<<g4grid(G.nc),G4B>>>(G.nc,alphaP,G.pcg.x.data(),G.p.data());NODALS_CUDA(cudaGetLastError());E.rec(5);
    NODALS_CUDA(cudaEventSynchronize(E.e[5]));H0P.collect();

    tAssembly+=E.ms(0,1);tMomentum+=E.ms(1,2);tContinuity+=E.ms(2,3);tPressure+=E.ms(3,4);tPupdate+=E.ms(4,5);
    finiteAll=finiteAll&&std::isfinite(contRel)&&std::isfinite(pr.rel);
    bool allMomentumMet=false;
    if(auditedThisOuter){
      allMomentumMet=std::isfinite(auditRel[0])&&std::isfinite(auditRel[1])&&std::isfinite(auditRel[2])&&
        auditRel[0]<=simpleTol&&auditRel[1]<=simpleTol&&auditRel[2]<=simpleTol;
      finiteAll=finiteAll&&std::isfinite(auditRel[0])&&std::isfinite(auditRel[1])&&std::isfinite(auditRel[2]);
    }
    converged=(runMode=="converge")&&finiteAll&&pressureAll&&contRel<=simpleTol&&auditedThisOuter&&allMomentumMet;finalIt=it;

    if(it<=10 || (it%25)==0 || auditedThisOuter || converged)
      std::printf("NODALS_GPU_H4_SIMPLE tag=%s it=%d relCont=%.12e momentumWork=%s momentumDirection=%s residualAudit=%s auditInitRel=[%.3e,%.3e,%.3e] pCG=%d pTrueRel=%.3e allMomentumMet=%d converged=%d status=%s\n",
        tag.c_str(),it,contRel,momentumWork.c_str(),momentumDirection,auditedThisOuter?"DONE":"SKIPPED",auditRel[0],auditRel[1],auditRel[2],pr.its,pr.rel,(int)allMomentumMet,(int)converged,(pressureAll&&finiteAll)?"PASS":"FAIL");
    if(runMode=="converge"&&converged)break;
  }
  NODALS_CUDA(cudaDeviceSynchronize());auto wall1=std::chrono::steady_clock::now();const auto memEnd=device_memory_info();

  std::vector<StateReal>pf((std::size_t)G.nc);G.p.download(pf.data(),pf.size());std::vector<double>pd(pf.size());for(std::size_t i=0;i<pf.size();++i)pd[i]=(double)pf[i];
  double dp=pressure_drop_fit_g4(M,pd),exact=S.pipe.hpDrop,dpErr=std::abs(dp-exact)/std::max(std::abs(exact),1e-300);
  double wallMs=std::chrono::duration<double,std::milli>(wall1-wall0).count();
  double usedEnd=g5_used_mib(memEnd);

  if(runMode=="fixed10"){
    std::printf("NODALS_GPU_H4_FIXED10 tag=%s cells=%zu fixedOuter=%d momentumWork=%s momentumPassesPerOuter=1 momentumResidualAudits=0 diffusionPolicy=PERSISTENT_NUMERIC_CSR convectionPolicy=NUMERIC_ONLY_EACH_OUTER fineCsrRefreshEvery=1 fineCsrRefreshCount=%d finalRelCont=%.12e avgPressureIts=%.6f pressureDropFit=%.12e exactPressureDrop=%.12e pressureDropRelErr=%.12e loopMs=%.3f avgSimpleMs=%.6f assemblyAvgMs=%.6f momentumAvgMs=%.6f continuityAvgMs=%.6f pressureAvgMs=%.6f pressureUpdateAvgMs=%.6f status=%s\n",
      tag.c_str(),M.tets.size(),finalIt,momentumWork.c_str(),fineCsrRefreshCount,contRel,finalIt?(double)sumP/finalIt:0.0,dp,exact,dpErr,wallMs,finalIt?wallMs/finalIt:0.0,finalIt?tAssembly/finalIt:0.0,finalIt?tMomentum/finalIt:0.0,finalIt?tContinuity/finalIt:0.0,finalIt?tPressure/finalIt:0.0,finalIt?tPupdate/finalIt:0.0,(pressureAll&&finiteAll&&finalIt==10)?"PASS":"FAIL");
  }else{
    std::printf("NODALS_GPU_H4_FULLCONV tag=%s cells=%zu momentumWork=%s simpleTol=%.3e outerIts=%d converged=%d finalRelCont=%.12e convergenceAuditCalls=%d finalAuditInitRel=[%.3e,%.3e,%.3e] avgPressureIts=%.6f pressureDropFit=%.12e exactPressureDrop=%.12e pressureDropRelErr=%.12e loopMs=%.3f avgSimpleMs=%.6f assemblyAvgMs=%.6f momentumAvgMs=%.6f continuityAvgMs=%.6f pressureAvgMs=%.6f pressureUpdateAvgMs=%.6f status=%s\n",
      tag.c_str(),M.tets.size(),momentumWork.c_str(),simpleTol,finalIt,(int)converged,contRel,convergenceAuditCalls,lastAuditRel[0],lastAuditRel[1],lastAuditRel[2],finalIt?(double)sumP/finalIt:0.0,dp,exact,dpErr,wallMs,finalIt?wallMs/finalIt:0.0,finalIt?tAssembly/finalIt:0.0,finalIt?tMomentum/finalIt:0.0,finalIt?tContinuity/finalIt:0.0,finalIt?tPressure/finalIt:0.0,finalIt?tPupdate/finalIt:0.0,converged?"PASS":"FAIL");
  }
  std::printf("NODALS_GPU_H4_MOMENTUM_AUDIT tag=%s runMode=%s policy=%s calls=%d postSweepAudits=0 status=PASS\n",tag.c_str(),runMode.c_str(),runMode=="fixed10"?"NONE":"CONVERGENCE_ONLY_PRE_SWEEP",convergenceAuditCalls);
  h0_print_profile(tag.c_str(),finalIt,tPressure,tMomentum,H0P);
  std::printf("NODALS_GPU_H4_MEMORY tag=%s point=after_convergence cells=%zu baselineUsedMiB=%.3f usedMiB=%.3f deltaFromBaselineMiB=%.3f explicitMiB=%.3f runtimeDriftMiB=%.3f totalMiB=%.3f status=PASS\n",
    tag.c_str(),M.tets.size(),baselineUsed,usedEnd,usedEnd-baselineUsed,explicitMiB,usedEnd-uploadUsed,(double)memEnd.total_bytes/(1024.0*1024.0));
  std::printf("NODALS_GPU_H4_RESIDENCY tag=%s O_N_H2D_inside_SIMPLE=0 O_N_D2H_inside_SIMPLE=%s finalPressureD2H=AFTER_LOOP reductions=FP64 deviceNumericStorage=FP32 physicalOperator=CURRENT_EXACT_CSR fineCsrNumericRefreshEvery=1 refreshKernel=WARP_PER_ROW momentumWork=%s momentumResidualPolicy=%s momentumDiffusion=PERSISTENT_NUMERIC_CSR momentumConvection=NUMERIC_ONLY_EACH_OUTER spectrumRefresh=SETUP_ONLY coarseSANumeric=SETUP_SNAPSHOT H4_scope=FIXED_MOMENTUM_WORK status=PASS\n",tag.c_str(),runMode=="fixed10"?"0":"SCALAR_CONVERGENCE_AUDITS_ONLY",momentumWork.c_str(),runMode=="fixed10"?"NONE":"CONVERGENCE_ONLY_PRE_SWEEP");
  const bool finalPass=(runMode=="fixed10")?(finalIt==10&&pressureAll&&finiteAll):converged;
  std::printf("NODALS_GPU_RESULT gate=H4 tag=%s cells=%zu precision=fp32 runMode=%s momentumWork=%s outer=%d simpleTol=%.3e fineCsrRefreshEvery=1 physicalOperator=current_exact_CSR pressureAll=%s finite=%s noPetsc=1 noMPI=1 status=%s\n",
    tag.c_str(),M.tets.size(),runMode.c_str(),momentumWork.c_str(),finalIt,simpleTol,pressureAll?"PASS":"FAIL",finiteAll?"PASS":"FAIL",finalPass?"PASS":"FAIL");
  return finalPass?0:35;
}catch(const std::exception&e){std::fprintf(stderr,"NODALS_GPU_H4_EXCEPTION what=%s\n",e.what());return 90;}}
