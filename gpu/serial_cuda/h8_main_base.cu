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
#include "h6_coarse_spmv.cuh"
#include "h7_momentum_assembly.cuh"
#include "h8_precomputed_b.cuh"
#include "g6_cf_host.hpp"
#include "g6_pmis_host.hpp"
#include "g8_cf_hierarchy_host.hpp"
#include "g9_l1_jacobi.cuh"
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

#include "h8_hp_diagnostics.inc"

// PM2: production uses previously validated setup policies.
// Set NODALS_SETUP_AUTOTUNE=1 to execute the legacy benchmark/parity selectors.
static bool gate6_setup_autotune_enabled()
{
  const char*e=std::getenv("NODALS_SETUP_AUTOTUNE");
  return e && *e &&
         std::strcmp(e,"0")!=0 &&
         std::strcmp(e,"false")!=0 &&
         std::strcmp(e,"FALSE")!=0 &&
         std::strcmp(e,"off")!=0 &&
         std::strcmp(e,"OFF")!=0;
}


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


// P7D_MCGS_IMPLEMENTATION
struct G4ColoringHost {
  int ncolors=0;
  int maxColorSize=0;
  std::vector<std::int32_t> rows;
  std::vector<std::int32_t> off;
};

struct G4ColoringDevice {
  int ncolors=0;
  int maxColorSize=0;
  DeviceBuffer<std::int32_t> rows;
  std::vector<std::int32_t> offHost;
  std::size_t bytes()const{return rows.bytes();}
};

template<class RowVec,class ColVec>
static G4ColoringHost g4_build_greedy_coloring(
    int n,const RowVec&row,const ColVec&col,const char*label)
{
  if(n<1 || row.size()!=(std::size_t)n+1)
    throw std::runtime_error("P7D MCGS coloring invalid CSR dimensions");

  std::size_t maxWidth=0;
  for(int i=0;i<n;++i){
    const auto a=(std::int64_t)row[(std::size_t)i];
    const auto b=(std::int64_t)row[(std::size_t)i+1];
    if(a<0 || b<a || (std::size_t)b>col.size())
      throw std::runtime_error("P7D MCGS coloring invalid CSR row");
    maxWidth=std::max(maxWidth,(std::size_t)(b-a));
  }

  std::vector<std::int32_t> color((std::size_t)n,-1);
  std::vector<int> mark(maxWidth+2,-1);
  int ncolors=0;

  for(int i=0;i<n;++i){
    const auto a=(std::int64_t)row[(std::size_t)i];
    const auto b=(std::int64_t)row[(std::size_t)i+1];
    for(std::int64_t k=a;k<b;++k){
      const int j=(int)col[(std::size_t)k];
      if(j<0 || j>=n)throw std::runtime_error("P7D MCGS column out of range");
      if(j==i)continue;
      const int cj=color[(std::size_t)j];
      if(cj>=0){
        if((std::size_t)cj>=mark.size())mark.resize((std::size_t)cj+2,-1);
        mark[(std::size_t)cj]=i;
      }
    }
    int c=0;
    while((std::size_t)c<mark.size() && mark[(std::size_t)c]==i)++c;
    if((std::size_t)c>=mark.size())mark.resize((std::size_t)c+2,-1);
    color[(std::size_t)i]=(std::int32_t)c;
    ncolors=std::max(ncolors,c+1);
  }

  for(int i=0;i<n;++i){
    const auto a=(std::int64_t)row[(std::size_t)i];
    const auto b=(std::int64_t)row[(std::size_t)i+1];
    for(std::int64_t k=a;k<b;++k){
      const int j=(int)col[(std::size_t)k];
      if(j!=i && color[(std::size_t)j]==color[(std::size_t)i]){
        std::fprintf(stderr,
          "P7D_MCGS_COLOR_CONFLICT label=%s row=%d col=%d color=%d\n",
          label?label:"unknown",i,j,(int)color[(std::size_t)i]);
        throw std::runtime_error("P7D MCGS coloring conflict");
      }
    }
  }

  G4ColoringHost C;
  C.ncolors=ncolors;
  C.off.assign((std::size_t)ncolors+1,0);
  for(int i=0;i<n;++i)++C.off[(std::size_t)color[(std::size_t)i]+1];
  for(int c=0;c<ncolors;++c)C.off[(std::size_t)c+1]+=C.off[(std::size_t)c];
  C.rows.resize((std::size_t)n);
  auto next=C.off;
  for(int i=0;i<n;++i)
    C.rows[(std::size_t)next[(std::size_t)color[(std::size_t)i]]++]=(std::int32_t)i;
  for(int c=0;c<ncolors;++c)
    C.maxColorSize=std::max(C.maxColorSize,
      (int)(C.off[(std::size_t)c+1]-C.off[(std::size_t)c]));
  return C;
}

static void g4_upload_coloring(G4ColoringDevice&D,const G4ColoringHost&H){
  D.ncolors=H.ncolors;
  D.maxColorSize=H.maxColorSize;
  D.offHost=H.off;
  D.rows.allocate(H.rows.size());
  D.rows.upload(H.rows.data(),H.rows.size());
}

template<class RowT>
__global__ void g4_mcgs_color_kernel(
    int begin,int count,const std::int32_t*colorRows,
    const RowT*row,const std::int32_t*col,const AMGReal*val,
    const AMGReal*diag,const AMGReal*b,AMGReal*x,double omega)
{
  const int q=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(q>=count)return;
  const int i=(int)colorRows[(std::size_t)begin+(std::size_t)q];
  AMGReal off=AMGReal(0);
  const RowT a=row[(std::size_t)i],e=row[(std::size_t)i+1];
  for(RowT k=a;k<e;++k){
    const int j=(int)col[(std::size_t)k];
    if(j!=i)off+=val[(std::size_t)k]*x[(std::size_t)j];
  }
  const AMGReal old=x[(std::size_t)i];
  const AMGReal gs=(b[(std::size_t)i]-off)/diag[(std::size_t)i];
  x[(std::size_t)i]=old+AMGReal(omega)*(gs-old);
}

struct G4GpuCSR {
  int n=0;
  bool useWarp=false;
  DeviceBuffer<std::int64_t> row;
  DeviceBuffer<std::int32_t> col,diagPos;
  DeviceBuffer<AMGReal> val,diag,l1,b,x,r,tmp,corr;
  G4ColoringDevice mcgs;

  void apply_scalar_raw(const AMGReal*a,AMGReal*y)const{
    g4_csr_spmv_kernel<<<g4grid(n),G4B>>>(
      n,row.data(),col.data(),val.data(),a,y);
    NODALS_CUDA(cudaGetLastError());
  }
  void apply_warp_raw(const AMGReal*a,AMGReal*y)const{
    constexpr int B=256;
    h6_coarse_csr_spmv_warp_kernel<<<h6_warp_grid(n),B>>>(
      n,row.data(),col.data(),val.data(),a,y);
    NODALS_CUDA(cudaGetLastError());
  }
  void apply(const AMGReal*a,AMGReal*y,int level=-1){
    int q=h0_begin(H0_COARSE_SPMV,level);
    if(useWarp)apply_warp_raw(a,y); else apply_scalar_raw(a,y);
    h0_end(q);
  }
  std::size_t bytes()const{
    return row.bytes()+col.bytes()+diagPos.bytes()+val.bytes()+diag.bytes()+l1.bytes()+
           b.bytes()+x.bytes()+r.bytes()+tmp.bytes()+corr.bytes()+mcgs.bytes();
  }
};
static std::vector<std::int32_t> g4_csr_diag_positions(const CSRHost&A){std::vector<std::int32_t>d((std::size_t)A.n,-1);for(int i=0;i<A.n;++i){auto first=A.col.begin()+A.row[(std::size_t)i],last=A.col.begin()+A.row[(std::size_t)i+1];auto it=std::lower_bound(first,last,i);if(it==last||*it!=i)throw std::runtime_error("G4B coarse diagonal absent");auto k=(std::int64_t)(it-A.col.begin());if(k>INT32_MAX)throw std::runtime_error("G4B coarse diagonal slot exceeds int32");d[(std::size_t)i]=(std::int32_t)k;}return d;}
static G4GpuCSR upload_g4_csr(const CSRHost&A){G4GpuCSR G;G.n=A.n;G.row.allocate(A.row.size());G.col.allocate(A.col.size());G.val.allocate(A.val.size());G.diag.allocate(A.diag.size());G.l1.allocate(A.n);G.diagPos.allocate(A.n);G.b.allocate(A.n);G.x.allocate(A.n);G.r.allocate(A.n);G.tmp.allocate(A.n);G.corr.allocate(A.n);G.row.upload(A.row.data(),A.row.size());G.col.upload(A.col.data(),A.col.size());auto vv=g5e_cast_vec<AMGReal>(A.val),dd=g5e_cast_vec<AMGReal>(A.diag);G.val.upload(vv.data(),vv.size());G.diag.upload(dd.data(),dd.size());std::vector<AMGReal> l1h((std::size_t)A.n);for(int i=0;i<A.n;++i){double ss=0.0;for(std::int64_t k=A.row[(std::size_t)i];k<A.row[(std::size_t)i+1];++k)ss+=std::abs(A.val[(std::size_t)k]);if(!(ss>0.0)||!std::isfinite(ss))throw std::runtime_error("AMG coarse L1 invalid");l1h[(std::size_t)i]=(AMGReal)ss;}G.l1.upload(l1h.data(),l1h.size());auto dp=g4_csr_diag_positions(A);G.diagPos.upload(dp.data(),dp.size());return G;}


struct H2FineCSRDevice {
  int n=0;
  bool usePrecomputedB=false;
  const OperatorReal* bcoeff=nullptr;
  DeviceBuffer<std::int32_t> row,col,diagPos,incOff,contribOff;
  DeviceBuffer<std::uint32_t> packed;
  DeviceBuffer<std::uint8_t> slot;
  DeviceBuffer<AMGReal> val;
  G4ColoringDevice mcgs;

  void refresh_dynamic_raw(
      const G4CellPlanDevice*cells,const OperatorReal*rau,AMGReal*diag){
    constexpr int B=256;
    h2b_fine_csr_refresh_warp_kernel<<<h2_warp_grid(n),B>>>(
      n,cells,rau,row.data(),diagPos.data(),incOff.data(),packed.data(),
      contribOff.data(),slot.data(),val.data(),diag);
    NODALS_CUDA(cudaGetLastError());
  }
  void refresh_precomputed_raw(
      const G4CellPlanDevice*cells,const OperatorReal*rau,AMGReal*diag){
    if(!bcoeff)throw std::runtime_error("H8 fine CSR bcoeff pointer null");
    constexpr int B=256;
    h8_fine_csr_refresh_warp_kernel<<<h2_warp_grid(n),B>>>(
      n,cells,bcoeff,rau,row.data(),diagPos.data(),incOff.data(),packed.data(),
      contribOff.data(),slot.data(),val.data(),diag);
    NODALS_CUDA(cudaGetLastError());
  }
  void refresh(const G4CellPlanDevice*cells,const OperatorReal*rau,AMGReal*diag,AMGReal*l1Out=nullptr){
    int q=h0_begin(H0_FINE_CSR_REFRESH);
    if(usePrecomputedB)refresh_precomputed_raw(cells,rau,diag);
    else refresh_dynamic_raw(cells,rau,diag);
    if(l1Out){constexpr int B=256;g9_fine_l1_warp_kernel<<<h2_warp_grid(n),B>>>(n,row.data(),val.data(),l1Out);NODALS_CUDA(cudaGetLastError());}
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
           contribOff.bytes()+slot.bytes()+val.bytes()+mcgs.bytes();
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
  int nc=0,nv=0;
  const G4CellPlanDevice*cells=nullptr;
  const OperatorReal*bcoeff=nullptr;
  const OperatorReal*rau_live=nullptr;
  H2FineCSRDevice*csr_pc=nullptr;
  bool physicalUseCurrentCSR=false;
  bool usePrecomputedBT=false;
  bool usePrecomputedBAction=false;
  DeviceBuffer<AMGReal> rau_pc,diag_pc,v0,v1,v2;
  void apply_with_rau(const StateReal*x,StateReal*y,const OperatorReal*r,bool live){
    int qt=h0_begin(live?H0_FINE_LIVE_TOTAL:H0_FINE_PC_TOTAL);
    int q=h0_begin(live?H0_FINE_LIVE_ZERO:H0_FINE_PC_ZERO);NODALS_CUDA(cudaMemset(v0.data(),0,v0.bytes()));NODALS_CUDA(cudaMemset(v1.data(),0,v1.bytes()));NODALS_CUDA(cudaMemset(v2.data(),0,v2.bytes()));h0_end(q);
    q=h0_begin(live?H0_FINE_LIVE_BT:H0_FINE_PC_BT);
    if(usePrecomputedBT){
      if(!bcoeff)throw std::runtime_error("H8 pressure BT bcoeff pointer null");
      h8_bt3_kernel<<<g4grid(nc),G4B>>>(
        cells,bcoeff,x,nc,v0.data(),v1.data(),v2.data());
    }else{
      g4_bt3_kernel<<<g4grid(nc),G4B>>>(
        cells,x,nc,v0.data(),v1.data(),v2.data());
    }
    NODALS_CUDA(cudaGetLastError());h0_end(q);
    q=h0_begin(live?H0_FINE_LIVE_RAU:H0_FINE_PC_RAU);g4_rau3_kernel<<<g4grid(nv),G4B>>>(v0.data(),v1.data(),v2.data(),r,nv);NODALS_CUDA(cudaGetLastError());h0_end(q);
    q=h0_begin(live?H0_FINE_LIVE_B:H0_FINE_PC_B);
    if(usePrecomputedBAction){
      if(!bcoeff)throw std::runtime_error("H8 pressure B bcoeff pointer null");
      h8_b3_kernel<<<g4grid(nc),G4B>>>(
        cells,bcoeff,nc,v0.data(),v1.data(),v2.data(),y);
    }else{
      g4_b3_kernel<<<g4grid(nc),G4B>>>(
        cells,nc,v0.data(),v1.data(),v2.data(),y);
    }
    NODALS_CUDA(cudaGetLastError());h0_end(q);h0_end(qt);
  }
  void apply_live(const StateReal*x,StateReal*y){
    if(physicalUseCurrentCSR){
      static_assert(std::is_same<StateReal,AMGReal>::value,"H2B physical CSR requires matching StateReal/AMGReal types");
      if(!csr_pc)throw std::runtime_error("H2B physical CSR pointer null");
      int qt=h0_begin(H0_FINE_LIVE_TOTAL);csr_pc->apply((const AMGReal*)x,(AMGReal*)y);h0_end(qt);
    } else apply_with_rau(x,y,rau_live,true);
  }
  void apply_pc_mf(const AMGReal*x,AMGReal*y){apply_with_rau(x,y,rau_pc.data(),false);}
  void apply_pc(const AMGReal*x,AMGReal*y){
    if(!csr_pc)throw std::runtime_error("H2B fine CSR pointer null");
    int qt=h0_begin(H0_FINE_PC_TOTAL);csr_pc->apply(x,y);h0_end(qt);
  }
  void bt_state(const StateReal*p){
    int qt=h0_begin(H0_MOM_BT_TOTAL);
    int q=h0_begin(H0_MOM_BT_ZERO);
    NODALS_CUDA(cudaMemset(v0.data(),0,v0.bytes()));
    NODALS_CUDA(cudaMemset(v1.data(),0,v1.bytes()));
    NODALS_CUDA(cudaMemset(v2.data(),0,v2.bytes()));
    h0_end(q);
    q=h0_begin(H0_MOM_BT_KERNEL);
    if(usePrecomputedBT){
      if(!bcoeff)throw std::runtime_error("H8 momentum BT bcoeff pointer null");
      h8_bt3_kernel<<<g4grid(nc),G4B>>>(
        cells,bcoeff,p,nc,v0.data(),v1.data(),v2.data());
    }else{
      g4_bt3_kernel<<<g4grid(nc),G4B>>>(
        cells,p,nc,v0.data(),v1.data(),v2.data());
    }
    NODALS_CUDA(cudaGetLastError());
    h0_end(q);h0_end(qt);
  }
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
  DeviceBuffer<AMGReal>fine_r,fine_tmp,fine_corr,fine_rhs,fine_l1,terminal_inv;
  double fineLambda=1.0;
  std::vector<double>levelLambda;
  int chebDegree=2,powerIts=16;
  double lambdaSafety=1.5,lambdaLowFraction=.05;
  std::string smoother="cheb2";
  double jacobiOmega=.7;
  int mcgsFineSweeps=1,mcgsCoarseSweeps=1;
  double mcgsOmega=1.0;
  std::string mcgsOrder="symmetric";
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
  void jacobi_explicit(G4GpuCSR&A,const AMGReal*b,AMGReal*x){
    g4_jacobi_zero_kernel<<<g4grid(A.n),G4B>>>(
      A.n,jacobiOmega,b,(smoother=="l1jacobi"?A.l1.data():A.diag.data()),x);
    NODALS_CUDA(cudaGetLastError());
  }
  void jacobi_fine(const AMGReal*b,AMGReal*x){
    g4_jacobi_zero_kernel<<<g4grid(fine->nc),G4B>>>(
      fine->nc,jacobiOmega,b,(smoother=="l1jacobi"?fine_l1.data():fine->diag_pc.data()),x);
    NODALS_CUDA(cudaGetLastError());
  }

  template<class RowT>
  void mcgs_color_pass(
      const RowT*row,const std::int32_t*col,const AMGReal*val,
      const AMGReal*diag,const AMGReal*b,AMGReal*x,
      const G4ColoringDevice&C,bool reverse)
  {
    if(C.ncolors<1 || C.offHost.size()!=(std::size_t)C.ncolors+1)
      throw std::runtime_error("P7D MCGS coloring unavailable");
    constexpr int B=256;
    for(int qq=0;qq<C.ncolors;++qq){
      const int c=reverse?(C.ncolors-1-qq):qq;
      const int begin=(int)C.offHost[(std::size_t)c];
      const int end=(int)C.offHost[(std::size_t)c+1];
      const int count=end-begin;
      if(count<=0)continue;
      const int grid=(count+B-1)/B;
      g4_mcgs_color_kernel<RowT><<<grid,B>>>(
        begin,count,C.rows.data(),row,col,val,diag,b,x,mcgsOmega);
      NODALS_CUDA(cudaGetLastError());
    }
  }

  template<class RowT>
  void mcgs_apply(
      int n,const RowT*row,const std::int32_t*col,const AMGReal*val,
      const AMGReal*diag,const AMGReal*b,AMGReal*x,
      const G4ColoringDevice&C,int sweeps)
  {
    if(sweeps<1)throw std::runtime_error("P7D MCGS sweeps must be positive");
    NODALS_CUDA(cudaMemset(x,0,(std::size_t)n*sizeof(AMGReal)));
    for(int s=0;s<sweeps;++s){
      if(mcgsOrder=="forward"){
        mcgs_color_pass(row,col,val,diag,b,x,C,false);
      }else if(mcgsOrder=="backward"){
        mcgs_color_pass(row,col,val,diag,b,x,C,true);
      }else if(mcgsOrder=="symmetric"){
        mcgs_color_pass(row,col,val,diag,b,x,C,false);
        mcgs_color_pass(row,col,val,diag,b,x,C,true);
      }else{
        throw std::runtime_error("P7D unknown MCGS order");
      }
    }
  }

  void mcgs_explicit(G4GpuCSR&A,const AMGReal*b,AMGReal*x){
    mcgs_apply(A.n,A.row.data(),A.col.data(),A.val.data(),A.diag.data(),
               b,x,A.mcgs,mcgsCoarseSweeps);
  }

  void mcgs_fine(const AMGReal*b,AMGReal*x){
    if(!fine || !fine->csr_pc)throw std::runtime_error("P7D fine MCGS CSR unavailable");
    auto&C=*fine->csr_pc;
    mcgs_apply(fine->nc,C.row.data(),C.col.data(),C.val.data(),fine->diag_pc.data(),
               b,x,C.mcgs,mcgsFineSweeps);
  }

  void smooth_explicit(G4GpuCSR&A,const AMGReal*b,AMGReal*x,double lambda,int level){
    if(smoother=="cheb2"){
      cheb_explicit(A,b,x,lambda,level);
    }else if((smoother=="jacobi"||smoother=="l1jacobi")){
      jacobi_explicit(A,b,x);
    }else if(smoother=="mcgs"){
      mcgs_explicit(A,b,x);
    }else{
      throw std::runtime_error("H8 unknown AMG smoother");
    }
  }
  void smooth_fine(const AMGReal*b,AMGReal*x){
    if(smoother=="cheb2"){
      cheb_fine(b,x);
    }else if((smoother=="jacobi"||smoother=="l1jacobi")){
      jacobi_fine(b,x);
    }else if(smoother=="mcgs"){
      mcgs_fine(b,x);
    }else{
      throw std::runtime_error("H8 unknown AMG smoother");
    }
  }

  void explicit_vcycle(int l,const AMGReal*b,AMGReal*x){
    auto&A=L[(std::size_t)l];
    if(l==(int)L.size()-1){int q=h0_begin(H0_TERMINAL_DENSE,l);g4_dense_mv_kernel<<<g4grid(A.n),G4B>>>(A.n,terminal_inv.data(),b,x);NODALS_CUDA(cudaGetLastError());h0_end(q);return;}
    smooth_explicit(A,b,x,levelLambda[(std::size_t)l],l);
    A.apply(x,A.tmp.data(),l);g4_residual_kernel<<<g4grid(A.n),G4B>>>(A.n,b,A.tmp.data(),A.r.data());NODALS_CUDA(cudaGetLastError());
    auto&C=L[(std::size_t)l+1];P[(std::size_t)l+1].restrict(A.r.data(),C.b.data(),l+1);
    explicit_vcycle(l+1,C.b.data(),C.x.data());
    P[(std::size_t)l+1].prolong_add(C.x.data(),x,l+1);
    A.apply(x,A.tmp.data(),l);g4_residual_kernel<<<g4grid(A.n),G4B>>>(A.n,b,A.tmp.data(),A.r.data());NODALS_CUDA(cudaGetLastError());
    g4_copy_kernel<<<g4grid(A.n),G4B>>>(A.n,A.r.data(),A.b.data());NODALS_CUDA(cudaGetLastError());
    smooth_explicit(A,A.b.data(),A.corr.data(),levelLambda[(std::size_t)l],l);
    g4_axpy_kernel<<<g4grid(A.n),G4B>>>(A.n,1.0,A.corr.data(),x);NODALS_CUDA(cudaGetLastError());
  }
  void apply(const AMGReal*b,AMGReal*z){
    smooth_fine(b,z);
    fine->apply_pc(z,fine_tmp.data());g4_residual_kernel<<<g4grid(fine->nc),G4B>>>(fine->nc,b,fine_tmp.data(),fine_r.data());NODALS_CUDA(cudaGetLastError());
    auto&C=L[0];P[0].restrict(fine_r.data(),C.b.data(),0);explicit_vcycle(0,C.b.data(),C.x.data());P[0].prolong_add(C.x.data(),z,0);
    fine->apply_pc(z,fine_tmp.data());g4_residual_kernel<<<g4grid(fine->nc),G4B>>>(fine->nc,b,fine_tmp.data(),fine_r.data());NODALS_CUDA(cudaGetLastError());
    g4_copy_kernel<<<g4grid(fine->nc),G4B>>>(fine->nc,fine_r.data(),fine_rhs.data());NODALS_CUDA(cudaGetLastError());
    smooth_fine(fine_rhs.data(),fine_corr.data());g4_axpy_kernel<<<g4grid(fine->nc),G4B>>>(fine->nc,1.0,fine_corr.data(),z);NODALS_CUDA(cudaGetLastError());
  }
  std::size_t bytes()const{std::size_t b=fine_r.bytes()+fine_tmp.bytes()+fine_corr.bytes()+fine_rhs.bytes()+fine_l1.bytes()+terminal_inv.bytes();for(const auto&t:P)b+=t.bytes();for(const auto&x:L)b+=x.bytes();return b;}
};
struct G4PCGWorkspace{DeviceBuffer<StateReal>x,r,z,p,q;};
struct G4PCGResult{int its=0;double rel=1;bool ok=false;};
static G4PCGResult pressure_pcg(G4PressureFine&F,G4AMG&A,const StateReal*rhs,double rtol,double atol,int maxit,G4PCGWorkspace&W,DeviceBuffer<double>&dotScratch){
  int n=F.nc;NODALS_CUDA(cudaMemset(W.x.data(),0,W.x.bytes()));g4_copy_kernel<<<g4grid(n),G4B>>>(n,rhs,W.r.data());NODALS_CUDA(cudaGetLastError());double r0=h0_pressure_norm2(W.r.data(),n,dotScratch);if(!std::isfinite(r0))throw std::runtime_error("G4 pressure initial norm nonfinite");if(r0<=atol){NODALS_CUDA(cudaMemset(W.x.data(),0,W.x.bytes()));return {0,0,true};}
  int qamg=h0_begin(H0_AMG_TOTAL);A.apply(W.r.data(),W.z.data());h0_end(qamg);g4_copy_kernel<<<g4grid(n),G4B>>>(n,W.z.data(),W.p.data());NODALS_CUDA(cudaGetLastError());double rho=h0_pressure_dot(W.r.data(),W.z.data(),n,dotScratch);if(!(rho>0)||!std::isfinite(rho))throw std::runtime_error("G4 pressure PCG initial rho nonpositive");G4PCGResult R;
  double target=std::max(atol,rtol*r0);for(int k=0;k<maxit;++k){F.apply_live(W.p.data(),W.q.data());double pq=h0_pressure_dot(W.p.data(),W.q.data(),n,dotScratch);if(!(pq>0)||!std::isfinite(pq))throw std::runtime_error("G4 pressure PCG pAp nonpositive");double alpha=rho/pq;g4_pcg_xr_kernel<<<g4grid(n),G4B>>>(n,alpha,W.p.data(),W.q.data(),W.x.data(),W.r.data());NODALS_CUDA(cudaGetLastError());double rn=h0_pressure_norm2(W.r.data(),n,dotScratch);R.its=k+1;R.rel=rn/std::max(r0,1e-300);if(rn<=target){R.ok=true;break;}qamg=h0_begin(H0_AMG_TOTAL);A.apply(W.r.data(),W.z.data());h0_end(qamg);double rhon=h0_pressure_dot(W.r.data(),W.z.data(),n,dotScratch);if(!(rhon>0)||!std::isfinite(rhon))throw std::runtime_error("G4 pressure PCG rho nonpositive");g4_pcg_p_kernel<<<g4grid(n),G4B>>>(n,rhon/rho,W.z.data(),W.p.data());NODALS_CUDA(cudaGetLastError());rho=rhon;}return R;
}

static G4PCGResult pressure_richardson(
    G4PressureFine&F,G4AMG&A,const StateReal*rhs,
    double rtol,double atol,int maxit,double omega,
    G4PCGWorkspace&W,DeviceBuffer<double>&dotScratch)
{
  const int n=F.nc;
  NODALS_CUDA(cudaMemset(W.x.data(),0,W.x.bytes()));
  g4_copy_kernel<<<g4grid(n),G4B>>>(n,rhs,W.r.data());
  NODALS_CUDA(cudaGetLastError());

  const double r0=h0_pressure_norm2(W.r.data(),n,dotScratch);
  if(!std::isfinite(r0))
    throw std::runtime_error("G12 Richardson initial norm nonfinite");
  if(r0<=atol)return {0,0,true};

  const double target=std::max(atol,rtol*r0);
  G4PCGResult R;

  for(int k=0;k<maxit;++k){
    const int qamg=h0_begin(H0_AMG_TOTAL);
    A.apply(W.r.data(),W.z.data());
    h0_end(qamg);

    F.apply_live(W.z.data(),W.q.data());
    g4_pcg_xr_kernel<<<g4grid(n),G4B>>>(
      n,omega,W.z.data(),W.q.data(),W.x.data(),W.r.data());
    NODALS_CUDA(cudaGetLastError());

    const double rn=h0_pressure_norm2(W.r.data(),n,dotScratch);
    R.its=k+1;
    R.rel=rn/std::max(r0,1e-300);
    if(!std::isfinite(R.rel))
      throw std::runtime_error("G12 Richardson residual nonfinite");
    if(rn<=target){R.ok=true;break;}
  }
  return R;
}

static double pressure_precond_power(
    G4PressureFine&F,G4AMG&A,int its,double safety,
    G4PCGWorkspace&W,DeviceBuffer<double>&dotScratch,
    const char*tag)
{
  if(its<1)throw std::runtime_error("G12 power iterations must be positive");
  if(!(safety>1.0))throw std::runtime_error("G12 power safety must exceed one");

  const int n=F.nc;
  H0Profiler*oldProfiler=g_h0;
  g_h0=nullptr;

  g5_power_init_kernel<<<g4grid(n),G4B>>>(n,W.p.data());
  NODALS_CUDA(cudaGetLastError());

  double vn=device_norm2(W.p.data(),n,dotScratch);
  if(!(vn>0.0)||!std::isfinite(vn))
    throw std::runtime_error("G12 power initial vector invalid");
  device_scale(W.p.data(),(std::size_t)n,AMGReal(1.0/vn));

  double est=0.0,last=0.0;
  CudaEventTimer T;T.start();

  for(int k=0;k<its;++k){
    F.apply_live(W.p.data(),W.q.data());
    A.apply(W.q.data(),W.z.data());

    const double zn=device_norm2(W.z.data(),n,dotScratch);
    const double ray=device_dot(W.p.data(),W.z.data(),n,dotScratch);
    if(!(zn>0.0)||!std::isfinite(zn)||!std::isfinite(ray))
      throw std::runtime_error("G12 preconditioned power failed");

    last=std::abs(ray);
    est=std::max(est,std::max(last,zn));

    g4_copy_kernel<<<g4grid(n),G4B>>>(n,W.z.data(),W.p.data());
    NODALS_CUDA(cudaGetLastError());
    device_scale(W.p.data(),(std::size_t)n,AMGReal(1.0/zn));
  }

  NODALS_CUDA(cudaDeviceSynchronize());
  const double ms=T.stop();
  g_h0=oldProfiler;

  const double hi=safety*est;
  if(!(hi>0.0)||!std::isfinite(hi))
    throw std::runtime_error("G12 preconditioned lambda invalid");

  std::printf(
    "NODALS_GPU_G12_PRECOND_POWER tag=%s its=%d safety=%.6f "
    "lastRay=%.12e rawBound=%.12e lambdaMax=%.12e ms=%.6f "
    "operator=M_INV_A setupOnly=1 profilerExcluded=1 status=PASS\n",
    tag,its,safety,last,est,hi,ms);

  return hi;
}

static G4PCGResult pressure_cheb_poly(
    G4PressureFine&F,G4AMG&A,const StateReal*rhs,
    double rtol,double atol,int maxit,
    double lambdaMax,double lowFraction,int degree,
    G4PCGWorkspace&W,DeviceBuffer<double>&dotScratch)
{
  if(!(lambdaMax>0.0)||!std::isfinite(lambdaMax))
    throw std::runtime_error("G12 Cheb lambdaMax invalid");
  if(!(lowFraction>0.0&&lowFraction<1.0))
    throw std::runtime_error("G12 Cheb low fraction invalid");
  if(degree<1)
    throw std::runtime_error("G12 Cheb degree invalid");

  const int n=F.nc;
  NODALS_CUDA(cudaMemset(W.x.data(),0,W.x.bytes()));
  g4_copy_kernel<<<g4grid(n),G4B>>>(n,rhs,W.r.data());
  NODALS_CUDA(cudaGetLastError());

  const double r0=h0_pressure_norm2(W.r.data(),n,dotScratch);
  if(!std::isfinite(r0))
    throw std::runtime_error("G12 Cheb initial norm nonfinite");
  if(r0<=atol)return {0,0,true};

  const double target=std::max(atol,rtol*r0);
  const double lo=lowFraction*lambdaMax;
  const double centre=.5*(lambdaMax+lo);
  const double radius=.5*(lambdaMax-lo);
  constexpr double pi=3.141592653589793238462643383279502884;

  G4PCGResult R;

  for(int k=0;k<maxit;++k){
    const int rootIndex=k%degree;
    const double root=
      centre-radius*std::cos(
        pi*(2.0*(double)rootIndex+1.0)/(2.0*(double)degree));
    const double omega=1.0/root;

    const int qamg=h0_begin(H0_AMG_TOTAL);
    A.apply(W.r.data(),W.z.data());
    h0_end(qamg);

    F.apply_live(W.z.data(),W.q.data());
    g4_pcg_xr_kernel<<<g4grid(n),G4B>>>(
      n,omega,W.z.data(),W.q.data(),W.x.data(),W.r.data());
    NODALS_CUDA(cudaGetLastError());

    const double rn=h0_pressure_norm2(W.r.data(),n,dotScratch);
    R.its=k+1;
    R.rel=rn/std::max(r0,1e-300);
    if(!std::isfinite(R.rel))
      throw std::runtime_error("G12 Cheb residual nonfinite");
    if(rn<=target){R.ok=true;break;}
  }

  return R;
}


struct G4Gpu {
  int nv=0,nc=0,ncolors=0;
  bool h7WarpConvection=false;
  bool h7WarpFinalize=false;
  int h7ConvectionBlocks=0;
  int h7FinalizeBlocks=0;
  bool h8PrecomputedContinuity=false;
  DeviceBuffer<G4CellPlanDevice> cells;
  DeviceBuffer<OperatorReal> bcoeff;DeviceBuffer<StateReal>fixed;
  DeviceBuffer<std::int64_t> row;DeviceBuffer<std::int32_t>col,diagPos,colorOffset,colorRows;DeviceBuffer<OperatorReal>diffusion,av,delta,diag,rau;
  DeviceBuffer<StateReal>u0,u1,u2,p,s0,s1,s2,c0,c1,c2,b0,b1,b2,fixedDiv,cont;
  DeviceBuffer<double>momSums;H2FineCSRDevice fineCsr;G4PressureFine pf;G4AMG amg;G4PCGWorkspace pcg;DeviceBuffer<double>dotScratch;
  std::size_t bytes()const{return cells.bytes()+bcoeff.bytes()+fixed.bytes()+row.bytes()+col.bytes()+diagPos.bytes()+colorOffset.bytes()+colorRows.bytes()+diffusion.bytes()+av.bytes()+delta.bytes()+diag.bytes()+rau.bytes()+u0.bytes()+u1.bytes()+u2.bytes()+p.bytes()+s0.bytes()+s1.bytes()+s2.bytes()+c0.bytes()+c1.bytes()+c2.bytes()+b0.bytes()+b1.bytes()+b2.bytes()+fixedDiv.bytes()+cont.bytes()+momSums.bytes()+fineCsr.bytes()+pf.bytes()+amg.bytes()+pcg.x.bytes()+pcg.r.bytes()+pcg.z.bytes()+pcg.p.bytes()+pcg.q.bytes()+dotScratch.bytes();}
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
  G.bcoeff.allocate((std::size_t)G.nc*H8_BCOEFF_PER_CELL);
  h8_build_bcoeff_kernel<<<grid_for(G.bcoeff.size()),kBlock>>>(
    G.cells.data(),G.nc,G.bcoeff.data());
  NODALS_CUDA(cudaGetLastError());
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
  G.fineCsr.bcoeff=G.bcoeff.data();
  G.pf.nc=G.nc;G.pf.nv=G.nv;G.pf.cells=G.cells.data();G.pf.bcoeff=G.bcoeff.data();G.pf.rau_live=G.rau.data();G.pf.csr_pc=&G.fineCsr;G.pf.rau_pc.allocate(ir.size());G.pf.rau_pc.upload(ir.data(),ir.size());auto fdiag=g5e_cast_vec<AMGReal>(H.fineDiag);G.pf.diag_pc.allocate(G.nc);G.pf.diag_pc.upload(fdiag.data(),fdiag.size());G.pf.v0.allocate(G.nv);G.pf.v1.allocate(G.nv);G.pf.v2.allocate(G.nv);
  G.amg.fine=&G.pf;G.amg.fineLambda=H.fineLambda;G.amg.levelLambda=H.levelLambda;G.amg.chebDegree=2;G.amg.powerIts=H.powerIts;G.amg.lambdaSafety=H.lambdaSafety;G.amg.lambdaLowFraction=H.lambdaLowFraction;
  G.amg.fine_r.allocate(G.nc);G.amg.fine_tmp.allocate(G.nc);G.amg.fine_corr.allocate(G.nc);G.amg.fine_rhs.allocate(G.nc);G.amg.fine_l1.allocate(G.nc);
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
  std::printf("NODALS_GPU_H8_SPECTRUM level=0 powerIts=%d hostLambda=%.12e gpuLambda=%.12e ratio=%.6f status=PASS\n",G.amg.powerIts,old,G.amg.fineLambda,G.amg.fineLambda/std::max(old,1e-300));
  for(std::size_t l=0;l+1<G.amg.L.size();++l){double h=(l<H.levelLambda.size()?H.levelLambda[l]:0.0);double q=g5_power_csr_gpu(G,G.amg.L[l],G.amg.powerIts,G.amg.lambdaSafety);G.amg.levelLambda[l]=q;std::printf("NODALS_GPU_H8_SPECTRUM level=%zu powerIts=%d hostLambda=%.12e gpuLambda=%.12e ratio=%.6f status=PASS\n",l+1,G.amg.powerIts,h,q,q/std::max(h,1e-300));}
  NODALS_CUDA(cudaDeviceSynchronize());float ms=T.stop();++G.amg.spectrumRefreshes;std::printf("NODALS_GPU_H8_POWER_SETUP powerIts=%d levels=%zu ms=%.3f scalarD2H_only=1 status=PASS\n",G.amg.powerIts,G.amg.L.size(),ms);
}
static double g5_rau_snapshot_rel(G4Gpu&G){
  g5_diff_kernel<<<g4grid(G.nv),G4B>>>(G.nv,G.rau.data(),G.pf.rau_pc.data(),G.pf.v0.data());NODALS_CUDA(cudaGetLastError());double dn=device_norm2(G.pf.v0.data(),G.nv,G.dotScratch);double rn=device_norm2(G.pf.rau_pc.data(),G.nv,G.dotScratch);return dn/std::max(rn,1e-300);
}


static void h2_action_parity_and_bench(G4Gpu&G,const char*tag,int reps=24){

  if(!gate6_setup_autotune_enabled()){
    std::printf(
      "NODALS_GPU_PM2_SETUP_POLICY component=H2_ACTION "
      "source=FROZEN_VALIDATED physical=EXACT_CURRENT_CSR "
      "finePc=CSR_WARP setupParityBenchmark=0 "
      "diagnosticEnv=NODALS_SETUP_AUTOTUNE status=PASS\n");
    return;
  }

  auto&x=G.amg.fine_rhs;auto&ymf=G.amg.fine_tmp;auto&ycsr=G.amg.fine_r;auto&diff=G.amg.fine_corr;
  g5_power_init_kernel<<<g4grid(G.nc),G4B>>>(G.nc,x.data());NODALS_CUDA(cudaGetLastError());
  G.pf.apply_pc_mf(x.data(),ymf.data());G.fineCsr.apply(x.data(),ycsr.data());
  g5_diff_kernel<<<g4grid(G.nc),G4B>>>(G.nc,ycsr.data(),ymf.data(),diff.data());NODALS_CUDA(cudaGetLastError());
  double dn=device_norm2(diff.data(),G.nc,G.dotScratch),rn=device_norm2(ymf.data(),G.nc,G.dotScratch);
  double rel=dn/std::max(rn,1e-300);
  std::printf("NODALS_GPU_H8_CSR_PARITY tag=%s relL2=%.12e tol=5e-5 status=%s\n",tag,rel,rel<=5e-5?"PASS":"FAIL");
  if(!(rel<=5e-5))throw std::runtime_error("H2B fine CSR action parity failed");

  for(int k=0;k<3;++k){G.pf.apply_pc_mf(x.data(),ymf.data());G.fineCsr.apply(x.data(),ycsr.data());}
  NODALS_CUDA(cudaDeviceSynchronize());
  CudaEventTimer T1;T1.start();for(int k=0;k<reps;++k)G.pf.apply_pc_mf(x.data(),ymf.data());double mf=(double)T1.stop()/reps;
  CudaEventTimer T2;T2.start();for(int k=0;k<reps;++k)G.fineCsr.apply(x.data(),ycsr.data());double cs=(double)T2.stop()/reps;
  std::printf("NODALS_GPU_H8_ACTION_BENCH tag=%s reps=%d matrixFreeMs=%.6f csrWarpMs=%.6f speedup=%.6f savedMsPerAction=%.6f status=PASS\n",
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
  std::printf("NODALS_GPU_H8_REFRESH_BENCH tag=%s reps=%d serialRowMs=%.6f warpRowMs=%.6f speedup=%.6f savedMsPerRefresh=%.6f status=PASS\n",
    tag,reps,serialMs,warpMs,serialMs/warpMs,serialMs-warpMs);
}


static int h6_host_row_max(const CSRHost&A){
  int m=0;
  for(int i=0;i<A.n;++i)
    m=std::max(m,(int)(A.row[(std::size_t)i+1]-A.row[(std::size_t)i]));
  return m;
}

static void h6_select_coarse_spmv(
    G4Gpu&G,const SAHierarchyHost&H,const char*tag,int reps=200)
{

  if(!gate6_setup_autotune_enabled()){
    if(G.amg.L.size()!=H.csr.size())
      throw std::runtime_error("PM2 host/device AMG level mismatch");

    std::printf(
      "NODALS_GPU_PM2_SETUP_POLICY component=H6_COARSE_SPMV "
      "source=FROZEN_VALIDATED explicitLevels=%zu policy=WARP_ROW "
      "terminal=DENSE setupBenchmarkOnly=0 diagnosticEnv=NODALS_SETUP_AUTOTUNE "
      "status=PASS\n",
      G.amg.L.empty()?0:G.amg.L.size()-1);

    for(std::size_t l=0;l<G.amg.L.size();++l){
      const bool terminal=(l+1==G.amg.L.size());
      G.amg.L[l].useWarp=!terminal;
      std::printf(
        "NODALS_GPU_PM2_COARSE_SPMV level=%zu rows=%d nnz=%zu role=%s "
        "selected=%s status=PASS\n",
        l+1,G.amg.L[l].n,
        l<H.csr.size()?H.csr[l].val.size():0,
        terminal?"TERMINAL_DENSE":"EXPLICIT_CSR",
        terminal?"TERMINAL_DENSE":"WARP_ROW");
    }
    return;
  }

  if(G.amg.L.size()!=H.csr.size())
    throw std::runtime_error("H8 host/device AMG level mismatch");

  std::printf("NODALS_GPU_H8_COARSE_SPMV_POLICY tag=%s policy=PER_LEVEL_FASTEST_OF_SCALAR_THREAD_ROW_VS_WARP_ROW benchmarkReps=%d setupOnly=1 numericalOperator=UNCHANGED status=PASS\n",
              tag,reps);

  // Last explicit CSR level is terminal and is inverted by the dense terminal
  // kernel during the V-cycle, so it has no production CSR SpMV to select.
  for(std::size_t l=0;l<G.amg.L.size();++l){
    auto&A=G.amg.L[l];
    const auto&AH=H.csr[l];
    const std::size_t nnz=AH.val.size();
    const double avg=A.n?(double)nnz/(double)A.n:0.0;
    const int rowMax=h6_host_row_max(AH);

    if(l+1==G.amg.L.size()){
      A.useWarp=false;
      std::printf("NODALS_GPU_H8_COARSE_SPMV_SELECT tag=%s amgLevel=%zu rows=%d nnz=%zu avgRow=%.6f maxRow=%d role=TERMINAL_DENSE scalarMs=0 warpMs=0 speedupWarpOverScalar=0 selected=TERMINAL_DENSE parityRelL2=0 status=PASS\n",
                  tag,l+1,A.n,nnz,avg,rowMax);
      continue;
    }

    g5_power_init_kernel<<<g4grid(A.n),G4B>>>(A.n,A.b.data());
    NODALS_CUDA(cudaGetLastError());

    // Action parity on exactly the same vector.
    A.apply_scalar_raw(A.b.data(),A.tmp.data());
    A.apply_warp_raw(A.b.data(),A.r.data());
    h6_amg_diff_kernel<<<g4grid(A.n),G4B>>>(
      A.n,A.tmp.data(),A.r.data(),A.corr.data());
    NODALS_CUDA(cudaGetLastError());
    const double dn=device_norm2(A.corr.data(),A.n,G.dotScratch);
    const double rn=device_norm2(A.tmp.data(),A.n,G.dotScratch);
    const double rel=dn/std::max(rn,1e-300);
    if(!(rel<=5e-6))
      throw std::runtime_error("H8 coarse scalar/warp SpMV parity failed");

    // Warm both launch shapes.
    for(int k=0;k<5;++k){
      A.apply_scalar_raw(A.b.data(),A.tmp.data());
      A.apply_warp_raw(A.b.data(),A.r.data());
    }
    NODALS_CUDA(cudaDeviceSynchronize());

    CudaEventTimer Ts;Ts.start();
    for(int k=0;k<reps;++k)A.apply_scalar_raw(A.b.data(),A.tmp.data());
    const double scalarMs=(double)Ts.stop()/reps;

    CudaEventTimer Tw;Tw.start();
    for(int k=0;k<reps;++k)A.apply_warp_raw(A.b.data(),A.r.data());
    const double warpMs=(double)Tw.stop()/reps;

    // Use the directly measured fastest production kernel on this level.
    A.useWarp=(warpMs<scalarMs);
    const double speedup=scalarMs/std::max(warpMs,1e-300);

    std::printf("NODALS_GPU_H8_COARSE_SPMV_SELECT tag=%s amgLevel=%zu rows=%d nnz=%zu avgRow=%.6f maxRow=%d role=EXPLICIT_CSR scalarMs=%.9f warpMs=%.9f speedupWarpOverScalar=%.6f selected=%s parityRelL2=%.12e status=PASS\n",
                tag,l+1,A.n,nnz,avg,rowMax,scalarMs,warpMs,speedup,
                A.useWarp?"WARP_ROW":"SCALAR_THREAD_ROW",rel);
  }
  NODALS_CUDA(cudaDeviceSynchronize());
}

static void h7_reset_momentum_physical(G4Gpu&G){
  NODALS_CUDA(cudaMemcpyAsync(
    G.av.data(),G.diffusion.data(),G.av.bytes(),cudaMemcpyDeviceToDevice));
  NODALS_CUDA(cudaMemsetAsync(G.c0.data(),0,G.c0.bytes()));
  NODALS_CUDA(cudaMemsetAsync(G.c1.data(),0,G.c1.bytes()));
  NODALS_CUDA(cudaMemsetAsync(G.c2.data(),0,G.c2.bytes()));
}

static void h7_apply_convection_scalar(G4Gpu&G){
  h4_assemble_convection_only_kernel<<<g4grid(G.nc),G4B>>>(
    G.cells.data(),G.nc,G.row.data(),G.fixed.data(),
    G.u0.data(),G.u1.data(),G.u2.data(),
    G.av.data(),G.c0.data(),G.c1.data(),G.c2.data());
  NODALS_CUDA(cudaGetLastError());
}

static void h7_apply_convection_warp(G4Gpu&G,int blocks){
  h7_convection_warp_cell_kernel<<<blocks,H7_BLOCK>>>(
    G.cells.data(),G.nc,G.row.data(),G.fixed.data(),
    G.u0.data(),G.u1.data(),G.u2.data(),
    G.av.data(),G.c0.data(),G.c1.data(),G.c2.data());
  NODALS_CUDA(cudaGetLastError());
}

static void h7_apply_finalize_scalar(G4Gpu&G,double alphaU){
  g4_finalize_relax_kernel<<<g4grid(G.nv),G4B>>>(
    G.nv,G.row.data(),G.col.data(),G.diagPos.data(),OperatorReal(alphaU),
    G.av.data(),G.delta.data(),G.diag.data(),G.rau.data());
  NODALS_CUDA(cudaGetLastError());
}

static void h7_apply_finalize_warp(G4Gpu&G,double alphaU,int blocks){
  h7_finalize_relax_warp_row_kernel<<<blocks,H7_BLOCK>>>(
    G.nv,G.row.data(),G.diagPos.data(),OperatorReal(alphaU),
    G.av.data(),G.delta.data(),G.diag.data(),G.rau.data());
  NODALS_CUDA(cudaGetLastError());
}

static void assemble_live(G4Gpu&G,double alphaU){
  h7_reset_momentum_physical(G);
  if(G.h7WarpConvection)
    h7_apply_convection_warp(G,G.h7ConvectionBlocks);
  else
    h7_apply_convection_scalar(G);

  if(G.h7WarpFinalize)
    h7_apply_finalize_warp(G,alphaU,G.h7FinalizeBlocks);
  else
    h7_apply_finalize_scalar(G,alphaU);
}

template<class T>
static double h7_rel_diff(
    const T*a,const T*b,std::size_t n,
    DeviceBuffer<T>&diff,DeviceBuffer<double>&scratch)
{
  if(diff.size()<n)diff.allocate(n);
  h7_diff_kernel<<<grid_for(n),kBlock>>>(n,a,b,diff.data());
  NODALS_CUDA(cudaGetLastError());
  const double dn=device_norm2(diff.data(),n,scratch);
  const double rn=device_norm2(b,n,scratch);
  return dn/std::max(rn,1e-300);
}

static double h7_time_convection_scalar(G4Gpu&G,int reps){
  CudaEventTimer T;
  double total=0;
  for(int k=0;k<reps;++k){
    h7_reset_momentum_physical(G);
    T.start();
    h7_apply_convection_scalar(G);
    total+=T.stop();
  }
  return total/reps;
}

static double h7_time_convection_warp(G4Gpu&G,int blocks,int reps){
  CudaEventTimer T;
  double total=0;
  for(int k=0;k<reps;++k){
    h7_reset_momentum_physical(G);
    T.start();
    h7_apply_convection_warp(G,blocks);
    total+=T.stop();
  }
  return total/reps;
}

static double h7_time_finalize_scalar(
    G4Gpu&G,const OperatorReal*physical,double alphaU,int reps)
{
  CudaEventTimer T;
  double total=0;
  for(int k=0;k<reps;++k){
    NODALS_CUDA(cudaMemcpyAsync(
      G.av.data(),physical,G.av.bytes(),cudaMemcpyDeviceToDevice));
    T.start();
    h7_apply_finalize_scalar(G,alphaU);
    total+=T.stop();
  }
  return total/reps;
}

static double h7_time_finalize_warp(
    G4Gpu&G,const OperatorReal*physical,double alphaU,int blocks,int reps)
{
  CudaEventTimer T;
  double total=0;
  for(int k=0;k<reps;++k){
    NODALS_CUDA(cudaMemcpyAsync(
      G.av.data(),physical,G.av.bytes(),cudaMemcpyDeviceToDevice));
    T.start();
    h7_apply_finalize_warp(G,alphaU,blocks);
    total+=T.stop();
  }
  return total/reps;
}

static void h7_select_momentum_assembly(
    G4Gpu&G,double alphaU,int smCount,const char*tag)
{

  if(!gate6_setup_autotune_enabled()){
    constexpr bool pm2fp32=std::is_same<OperatorReal,float>::value;
    G.h7WarpConvection=pm2fp32;
    G.h7ConvectionBlocks=
      pm2fp32?h7_blocks_for_sm_factor(smCount,4):0;
    G.h7WarpFinalize=false;
    G.h7FinalizeBlocks=0;

    std::printf(
      "NODALS_GPU_PM2_SETUP_POLICY component=H7_MOMENTUM "
      "source=FROZEN_VALIDATED precision=%s convection=%s "
      "convectionSmFactor=%d convectionBlocks=%d "
      "finalize=SCALAR_THREAD_ROW finalizeBlocks=0 "
      "setupBenchmarkOnly=0 diagnosticEnv=NODALS_SETUP_AUTOTUNE status=PASS\n",
      kPrecisionName,
      G.h7WarpConvection?"WARP_CELL":"SCALAR_THREAD_CELL",
      G.h7WarpConvection?4:0,G.h7ConvectionBlocks);
    return;
  }

  constexpr int reps=8;
  const int factors[4]={4,8,16,32};

  // Temporary reference storage exists only during setup selection.
  DeviceBuffer<OperatorReal> avRef(G.av.size()),diff(G.av.size());
  DeviceBuffer<StateReal> c0Ref(G.c0.size()),c1Ref(G.c1.size()),c2Ref(G.c2.size());
  DeviceBuffer<OperatorReal> diagRef(G.diag.size()),rauRef(G.rau.size());

  // Scalar reference action.
  h7_reset_momentum_physical(G);
  h7_apply_convection_scalar(G);
  NODALS_CUDA(cudaDeviceSynchronize());
  NODALS_CUDA(cudaMemcpy(
    avRef.data(),G.av.data(),G.av.bytes(),cudaMemcpyDeviceToDevice));
  NODALS_CUDA(cudaMemcpy(
    c0Ref.data(),G.c0.data(),G.c0.bytes(),cudaMemcpyDeviceToDevice));
  NODALS_CUDA(cudaMemcpy(
    c1Ref.data(),G.c1.data(),G.c1.bytes(),cudaMemcpyDeviceToDevice));
  NODALS_CUDA(cudaMemcpy(
    c2Ref.data(),G.c2.data(),G.c2.bytes(),cudaMemcpyDeviceToDevice));

  const double scalarConv=h7_time_convection_scalar(G,reps);
  double bestConv=scalarConv;
  int bestConvBlocks=0;

  std::printf(
    "NODALS_GPU_H8_CONVECTION_BENCH tag=%s variant=SCALAR_THREAD_CELL "
    "blocks=%d reps=%d ms=%.9f status=PASS\n",
    tag,g4grid(G.nc),reps,scalarConv);

  for(int f:factors){
    const int blocks=h7_blocks_for_sm_factor(smCount,f);
    const double t=h7_time_convection_warp(G,blocks,reps);
    std::printf(
      "NODALS_GPU_H8_CONVECTION_BENCH tag=%s variant=WARP_CELL "
      "smFactor=%d blocks=%d reps=%d ms=%.9f speedupVsScalar=%.6f status=PASS\n",
      tag,f,blocks,reps,t,scalarConv/std::max(t,1e-300));
    if(t<bestConv){bestConv=t;bestConvBlocks=blocks;}
  }

  // Numerical parity of the winning warp candidate, if any.
  double convParity=0,cParity=0;
  if(bestConvBlocks>0){
    h7_reset_momentum_physical(G);
    h7_apply_convection_warp(G,bestConvBlocks);
    NODALS_CUDA(cudaDeviceSynchronize());
    convParity=h7_rel_diff(
      G.av.data(),avRef.data(),G.av.size(),diff,G.dotScratch);
    cParity=std::max({
      h7_rel_diff(G.c0.data(),c0Ref.data(),G.c0.size(),diff,G.dotScratch),
      h7_rel_diff(G.c1.data(),c1Ref.data(),G.c1.size(),diff,G.dotScratch),
      h7_rel_diff(G.c2.data(),c2Ref.data(),G.c2.size(),diff,G.dotScratch)});
    if(convParity>2e-5 || cParity>2e-5)
      throw std::runtime_error("H8 warp convection parity failed");
  }

  G.h7WarpConvection=(bestConvBlocks>0);
  G.h7ConvectionBlocks=bestConvBlocks;
  std::printf(
    "NODALS_GPU_H8_CONVECTION_SELECT tag=%s scalarMs=%.9f bestMs=%.9f "
    "speedup=%.6f selected=%s selectedBlocks=%d matrixParityRelL2=%.12e "
    "rhsParityRelL2=%.12e status=PASS\n",
    tag,scalarConv,bestConv,scalarConv/std::max(bestConv,1e-300),
    G.h7WarpConvection?"WARP_CELL":"SCALAR_THREAD_CELL",
    G.h7ConvectionBlocks,convParity,cParity);

  // Form the selected physical (unrelaxed) matrix once and retain it in avRef
  // for the finalizer benchmark.
  h7_reset_momentum_physical(G);
  if(G.h7WarpConvection)
    h7_apply_convection_warp(G,G.h7ConvectionBlocks);
  else
    h7_apply_convection_scalar(G);
  NODALS_CUDA(cudaDeviceSynchronize());
  NODALS_CUDA(cudaMemcpy(
    avRef.data(),G.av.data(),G.av.bytes(),cudaMemcpyDeviceToDevice));

  // Scalar finalizer reference.
  NODALS_CUDA(cudaMemcpyAsync(
    G.av.data(),avRef.data(),G.av.bytes(),cudaMemcpyDeviceToDevice));
  h7_apply_finalize_scalar(G,alphaU);
  NODALS_CUDA(cudaDeviceSynchronize());
  NODALS_CUDA(cudaMemcpy(
    diagRef.data(),G.diag.data(),G.diag.bytes(),cudaMemcpyDeviceToDevice));
  NODALS_CUDA(cudaMemcpy(
    rauRef.data(),G.rau.data(),G.rau.bytes(),cudaMemcpyDeviceToDevice));

  const double scalarFinalize=
    h7_time_finalize_scalar(G,avRef.data(),alphaU,reps);
  double bestFinalize=scalarFinalize;
  int bestFinalizeBlocks=0;

  std::printf(
    "NODALS_GPU_H8_FINALIZE_BENCH tag=%s variant=SCALAR_THREAD_ROW "
    "blocks=%d reps=%d ms=%.9f status=PASS\n",
    tag,g4grid(G.nv),reps,scalarFinalize);

  for(int f:factors){
    const int blocks=h7_blocks_for_sm_factor(smCount,f);
    const double t=h7_time_finalize_warp(
      G,avRef.data(),alphaU,blocks,reps);
    std::printf(
      "NODALS_GPU_H8_FINALIZE_BENCH tag=%s variant=WARP_ROW "
      "smFactor=%d blocks=%d reps=%d ms=%.9f speedupVsScalar=%.6f status=PASS\n",
      tag,f,blocks,reps,t,scalarFinalize/std::max(t,1e-300));
    if(t<bestFinalize){bestFinalize=t;bestFinalizeBlocks=blocks;}
  }

  double diagParity=0,rauParity=0;
  if(bestFinalizeBlocks>0){
    NODALS_CUDA(cudaMemcpyAsync(
      G.av.data(),avRef.data(),G.av.bytes(),cudaMemcpyDeviceToDevice));
    h7_apply_finalize_warp(G,alphaU,bestFinalizeBlocks);
    NODALS_CUDA(cudaDeviceSynchronize());
    diagParity=h7_rel_diff(
      G.diag.data(),diagRef.data(),G.diag.size(),diff,G.dotScratch);
    rauParity=h7_rel_diff(
      G.rau.data(),rauRef.data(),G.rau.size(),diff,G.dotScratch);
    if(diagParity>2e-5 || rauParity>2e-5)
      throw std::runtime_error("H8 warp finalizer parity failed");
  }

  G.h7WarpFinalize=(bestFinalizeBlocks>0);
  G.h7FinalizeBlocks=bestFinalizeBlocks;
  std::printf(
    "NODALS_GPU_H8_FINALIZE_SELECT tag=%s scalarMs=%.9f bestMs=%.9f "
    "speedup=%.6f selected=%s selectedBlocks=%d diagParityRelL2=%.12e "
    "rauParityRelL2=%.12e status=PASS\n",
    tag,scalarFinalize,bestFinalize,
    scalarFinalize/std::max(bestFinalize,1e-300),
    G.h7WarpFinalize?"WARP_ROW":"SCALAR_THREAD_ROW",
    G.h7FinalizeBlocks,diagParity,rauParity);

  NODALS_CUDA(cudaDeviceSynchronize());
  std::printf(
    "NODALS_GPU_H8_ASSEMBLY_POLICY tag=%s convection=%s convectionBlocks=%d "
    "finalize=%s finalizeBlocks=%d setupBenchmarkOnly=1 numericalOperator=UNCHANGED "
    "status=PASS\n",
    tag,G.h7WarpConvection?"WARP_CELL":"SCALAR_THREAD_CELL",
    G.h7ConvectionBlocks,
    G.h7WarpFinalize?"WARP_ROW":"SCALAR_THREAD_ROW",
    G.h7FinalizeBlocks);
}


template<class T>
static double h8_rel_diff(
    const T*a,const T*b,std::size_t n,
    DeviceBuffer<T>&diff,DeviceBuffer<double>&scratch)
{
  if(diff.size()<n)diff.allocate(n);
  h8_diff_kernel<<<grid_for(n),kBlock>>>(n,a,b,diff.data());
  NODALS_CUDA(cudaGetLastError());
  const double dn=device_norm2(diff.data(),n,scratch);
  const double rn=device_norm2(b,n,scratch);
  return dn/std::max(rn,1e-300);
}

static double h8_time_refresh_dynamic(G4Gpu&G,int reps){
  CudaEventTimer T;double total=0;
  for(int k=0;k<reps;++k){
    T.start();
    G.fineCsr.refresh_dynamic_raw(
      G.cells.data(),G.rau.data(),G.pf.diag_pc.data());
    total+=T.stop();
  }
  return total/reps;
}

static double h8_time_refresh_precomputed(G4Gpu&G,int reps){
  CudaEventTimer T;double total=0;
  for(int k=0;k<reps;++k){
    T.start();
    G.fineCsr.refresh_precomputed_raw(
      G.cells.data(),G.rau.data(),G.pf.diag_pc.data());
    total+=T.stop();
  }
  return total/reps;
}

static double h8_time_bt_dynamic(
    G4Gpu&G,const StateReal*p,
    StateReal*v0,StateReal*v1,StateReal*v2,int reps)
{
  CudaEventTimer T;double total=0;
  for(int k=0;k<reps;++k){
    NODALS_CUDA(cudaMemsetAsync(v0,0,(std::size_t)G.nv*sizeof(StateReal)));
    NODALS_CUDA(cudaMemsetAsync(v1,0,(std::size_t)G.nv*sizeof(StateReal)));
    NODALS_CUDA(cudaMemsetAsync(v2,0,(std::size_t)G.nv*sizeof(StateReal)));
    T.start();
    g4_bt3_kernel<<<g4grid(G.nc),G4B>>>(
      G.cells.data(),p,G.nc,v0,v1,v2);
    NODALS_CUDA(cudaGetLastError());
    total+=T.stop();
  }
  return total/reps;
}

static double h8_time_bt_precomputed(
    G4Gpu&G,const StateReal*p,
    StateReal*v0,StateReal*v1,StateReal*v2,int reps)
{
  CudaEventTimer T;double total=0;
  for(int k=0;k<reps;++k){
    NODALS_CUDA(cudaMemsetAsync(v0,0,(std::size_t)G.nv*sizeof(StateReal)));
    NODALS_CUDA(cudaMemsetAsync(v1,0,(std::size_t)G.nv*sizeof(StateReal)));
    NODALS_CUDA(cudaMemsetAsync(v2,0,(std::size_t)G.nv*sizeof(StateReal)));
    T.start();
    h8_bt3_kernel<<<g4grid(G.nc),G4B>>>(
      G.cells.data(),G.bcoeff.data(),p,G.nc,v0,v1,v2);
    NODALS_CUDA(cudaGetLastError());
    total+=T.stop();
  }
  return total/reps;
}

static double h8_time_cont_dynamic(
    G4Gpu&G,const StateReal*u0,const StateReal*u1,const StateReal*u2,
    StateReal*out,int reps)
{
  CudaEventTimer T;double total=0;
  for(int k=0;k<reps;++k){
    T.start();
    g4_continuity_kernel<<<g4grid(G.nc),G4B>>>(
      G.cells.data(),G.nc,G.fixedDiv.data(),u0,u1,u2,out);
    NODALS_CUDA(cudaGetLastError());
    total+=T.stop();
  }
  return total/reps;
}

static double h8_time_cont_precomputed(
    G4Gpu&G,const StateReal*u0,const StateReal*u1,const StateReal*u2,
    StateReal*out,int reps)
{
  CudaEventTimer T;double total=0;
  for(int k=0;k<reps;++k){
    T.start();
    h8_continuity_kernel<<<g4grid(G.nc),G4B>>>(
      G.cells.data(),G.bcoeff.data(),G.nc,G.fixedDiv.data(),u0,u1,u2,out);
    NODALS_CUDA(cudaGetLastError());
    total+=T.stop();
  }
  return total/reps;
}

static void h8_select_precomputed_b(G4Gpu&G,const char*tag)
{

  if(!gate6_setup_autotune_enabled()){
    G.fineCsr.usePrecomputedB=true;
    G.pf.usePrecomputedBT=true;
    G.pf.usePrecomputedBAction=true;
    G.h8PrecomputedContinuity=true;

    G.fineCsr.refresh_precomputed_raw(
      G.cells.data(),G.rau.data(),G.pf.diag_pc.data());
    NODALS_CUDA(cudaDeviceSynchronize());

    std::printf(
      "NODALS_GPU_H8_BCOEFF tag=%s valuesPerCell=%d bytesPerCell=%zu "
      "totalMiB=%.3f build=SETUP_ONCE storage=OperatorReal precision=%s "
      "status=PASS\n",
      tag,H8_BCOEFF_PER_CELL,
      (std::size_t)H8_BCOEFF_PER_CELL*sizeof(OperatorReal),
      (double)G.bcoeff.bytes()/(1024.0*1024.0),kPrecisionName);
    std::printf(
      "NODALS_GPU_PM2_SETUP_POLICY component=H8_B_GEOMETRY "
      "source=FROZEN_VALIDATED refresh=PRECOMPUTED_B bt=PRECOMPUTED_B "
      "bAction=PRECOMPUTED_B continuity=PRECOMPUTED_B persistentBytes=%zu "
      "setupBenchmarkOnly=0 diagnosticEnv=NODALS_SETUP_AUTOTUNE "
      "numericalOperator=UNCHANGED status=PASS\n",
      G.bcoeff.bytes());
    return;
  }

  constexpr int reps=8;
  std::printf(
    "NODALS_GPU_H8_BCOEFF tag=%s valuesPerCell=%d bytesPerCell=%zu totalMiB=%.3f "
    "build=SETUP_ONCE storage=OperatorReal precision=%s status=PASS\n",
    tag,H8_BCOEFF_PER_CELL,
    (std::size_t)H8_BCOEFF_PER_CELL*sizeof(OperatorReal),
    (double)G.bcoeff.bytes()/(1024.0*1024.0),kPrecisionName);

  // ---- Exact fine CSR numeric refresh ----
  DeviceBuffer<AMGReal> valRef(G.fineCsr.val.size());
  DeviceBuffer<AMGReal> diagRef(G.nc);
  DeviceBuffer<AMGReal> diffVal(G.fineCsr.val.size());

  G.fineCsr.refresh_dynamic_raw(
    G.cells.data(),G.rau.data(),G.pf.diag_pc.data());
  NODALS_CUDA(cudaDeviceSynchronize());
  NODALS_CUDA(cudaMemcpy(
    valRef.data(),G.fineCsr.val.data(),G.fineCsr.val.bytes(),
    cudaMemcpyDeviceToDevice));
  NODALS_CUDA(cudaMemcpy(
    diagRef.data(),G.pf.diag_pc.data(),G.pf.diag_pc.bytes(),
    cudaMemcpyDeviceToDevice));

  G.fineCsr.refresh_precomputed_raw(
    G.cells.data(),G.rau.data(),G.pf.diag_pc.data());
  NODALS_CUDA(cudaDeviceSynchronize());

  const double refreshValParity=h8_rel_diff(
    G.fineCsr.val.data(),valRef.data(),G.fineCsr.val.size(),
    diffVal,G.dotScratch);
  DeviceBuffer<AMGReal> diffDiag(G.nc);
  const double refreshDiagParity=h8_rel_diff(
    G.pf.diag_pc.data(),diagRef.data(),G.nc,diffDiag,G.dotScratch);

  if(refreshValParity>3e-5 || refreshDiagParity>3e-5)
    throw std::runtime_error("H8 precomputed-B fine CSR refresh parity failed");

  const double refreshDynamic=h8_time_refresh_dynamic(G,reps);
  const double refreshPrecomp=h8_time_refresh_precomputed(G,reps);
  G.fineCsr.usePrecomputedB=(refreshPrecomp<refreshDynamic);

  std::printf(
    "NODALS_GPU_H8_REFRESH_SELECT tag=%s dynamicMs=%.9f precomputedMs=%.9f "
    "speedup=%.6f selected=%s valParityRelL2=%.12e diagParityRelL2=%.12e "
    "status=PASS\n",
    tag,refreshDynamic,refreshPrecomp,
    refreshDynamic/std::max(refreshPrecomp,1e-300),
    G.fineCsr.usePrecomputedB?"PRECOMPUTED_B":"DYNAMIC_GEOMETRY",
    refreshValParity,refreshDiagParity);

  // Release the very large setup-only CSR parity buffers before proceeding.
  diffVal.reset();valRef.reset();diagRef.reset();diffDiag.reset();

  // ---- B^T p ----
  DeviceBuffer<StateReal> ptest(G.nc);
  DeviceBuffer<StateReal> d0(G.nv),d1(G.nv),d2(G.nv);
  DeviceBuffer<StateReal> r0(G.nv),r1(G.nv),r2(G.nv),diffV(G.nv);
  h8_test_vector_kernel<<<g4grid(G.nc),G4B>>>(G.nc,ptest.data(),1);
  NODALS_CUDA(cudaGetLastError());

  NODALS_CUDA(cudaMemset(d0.data(),0,d0.bytes()));
  NODALS_CUDA(cudaMemset(d1.data(),0,d1.bytes()));
  NODALS_CUDA(cudaMemset(d2.data(),0,d2.bytes()));
  g4_bt3_kernel<<<g4grid(G.nc),G4B>>>(
    G.cells.data(),ptest.data(),G.nc,d0.data(),d1.data(),d2.data());
  NODALS_CUDA(cudaGetLastError());

  NODALS_CUDA(cudaMemset(r0.data(),0,r0.bytes()));
  NODALS_CUDA(cudaMemset(r1.data(),0,r1.bytes()));
  NODALS_CUDA(cudaMemset(r2.data(),0,r2.bytes()));
  h8_bt3_kernel<<<g4grid(G.nc),G4B>>>(
    G.cells.data(),G.bcoeff.data(),ptest.data(),G.nc,
    r0.data(),r1.data(),r2.data());
  NODALS_CUDA(cudaGetLastError());
  NODALS_CUDA(cudaDeviceSynchronize());

  const double btParity=std::max({
    h8_rel_diff(r0.data(),d0.data(),G.nv,diffV,G.dotScratch),
    h8_rel_diff(r1.data(),d1.data(),G.nv,diffV,G.dotScratch),
    h8_rel_diff(r2.data(),d2.data(),G.nv,diffV,G.dotScratch)});
  if(btParity>3e-5)
    throw std::runtime_error("H8 precomputed-B BT parity failed");

  const double btDynamic=h8_time_bt_dynamic(
    G,ptest.data(),d0.data(),d1.data(),d2.data(),reps);
  const double btPrecomp=h8_time_bt_precomputed(
    G,ptest.data(),r0.data(),r1.data(),r2.data(),reps);
  G.pf.usePrecomputedBT=(btPrecomp<btDynamic);

  std::printf(
    "NODALS_GPU_H8_BT_SELECT tag=%s dynamicKernelMs=%.9f "
    "precomputedKernelMs=%.9f speedup=%.6f selected=%s "
    "parityRelL2=%.12e status=PASS\n",
    tag,btDynamic,btPrecomp,btDynamic/std::max(btPrecomp,1e-300),
    G.pf.usePrecomputedBT?"PRECOMPUTED_B":"DYNAMIC_GEOMETRY",btParity);

  // Matrix-free B action is not the production physical operator, but use the
  // same selected geometry representation in setup parity/power diagnostics.
  G.pf.usePrecomputedBAction=G.pf.usePrecomputedBT;

  // ---- continuity B u ----
  DeviceBuffer<StateReal> u0(G.nv),u1(G.nv),u2(G.nv);
  DeviceBuffer<StateReal> cDyn(G.nc),cPre(G.nc),diffC(G.nc);
  h8_test_vector_kernel<<<g4grid(G.nv),G4B>>>(G.nv,u0.data(),2);
  h8_test_vector_kernel<<<g4grid(G.nv),G4B>>>(G.nv,u1.data(),3);
  h8_test_vector_kernel<<<g4grid(G.nv),G4B>>>(G.nv,u2.data(),4);
  NODALS_CUDA(cudaGetLastError());

  g4_continuity_kernel<<<g4grid(G.nc),G4B>>>(
    G.cells.data(),G.nc,G.fixedDiv.data(),
    u0.data(),u1.data(),u2.data(),cDyn.data());
  h8_continuity_kernel<<<g4grid(G.nc),G4B>>>(
    G.cells.data(),G.bcoeff.data(),G.nc,G.fixedDiv.data(),
    u0.data(),u1.data(),u2.data(),cPre.data());
  NODALS_CUDA(cudaGetLastError());
  NODALS_CUDA(cudaDeviceSynchronize());

  const double contParity=h8_rel_diff(
    cPre.data(),cDyn.data(),G.nc,diffC,G.dotScratch);
  if(contParity>3e-5)
    throw std::runtime_error("H8 precomputed-B continuity parity failed");

  const double contDynamic=h8_time_cont_dynamic(
    G,u0.data(),u1.data(),u2.data(),cDyn.data(),reps);
  const double contPrecomp=h8_time_cont_precomputed(
    G,u0.data(),u1.data(),u2.data(),cPre.data(),reps);
  G.h8PrecomputedContinuity=(contPrecomp<contDynamic);

  std::printf(
    "NODALS_GPU_H8_CONTINUITY_SELECT tag=%s dynamicMs=%.9f "
    "precomputedMs=%.9f speedup=%.6f selected=%s parityRelL2=%.12e "
    "status=PASS\n",
    tag,contDynamic,contPrecomp,contDynamic/std::max(contPrecomp,1e-300),
    G.h8PrecomputedContinuity?"PRECOMPUTED_B":"DYNAMIC_GEOMETRY",
    contParity);

  // Restore the selected exact fine CSR values before all subsequent setup.
  G.fineCsr.refresh(
    G.cells.data(),G.rau.data(),G.pf.diag_pc.data());
  NODALS_CUDA(cudaDeviceSynchronize());

  std::printf(
    "NODALS_GPU_H8_B_POLICY tag=%s refresh=%s bt=%s continuity=%s "
    "persistentBytes=%zu setupOnlyBenchmark=1 numericalOperator=UNCHANGED "
    "status=PASS\n",
    tag,
    G.fineCsr.usePrecomputedB?"PRECOMPUTED_B":"DYNAMIC_GEOMETRY",
    G.pf.usePrecomputedBT?"PRECOMPUTED_B":"DYNAMIC_GEOMETRY",
    G.h8PrecomputedContinuity?"PRECOMPUTED_B":"DYNAMIC_GEOMETRY",
    G.bcoeff.bytes());
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
  std::printf("NODALS_GPU_H8_PROFILE_SUMMARY tag=%s outer=%d pressureStageTotalMs=%.6f momentumStageTotalMs=%.6f records=%lld eventTiming=NO_EXTRA_PER_SPAN_SYNC status=PASS\n",tag,outer,pressureStageMs,momentumStageMs,P.recordCount);
  for(int c=0;c<H0_CAT_COUNT;++c){for(int sl=0;sl<=H0Profiler::MAX_LEVEL;++sl){if(P.calls[c][sl]==0)continue;int lev=sl==0?-1:sl-1;double t=P.total[c][sl],w=P.warm[c][sl];long long n=P.calls[c][sl],nw=P.warmCalls[c][sl];
    const bool mom=c>=H0_MOM_BT_TOTAL;double denom=mom?momentumStageMs:pressureStageMs;double pct=denom>0?100.0*t/denom:0.0;
    std::printf("NODALS_GPU_H8_PROFILE tag=%s domain=%s category=%s level=%d calls=%lld totalMs=%.6f perCallMs=%.6f perSimpleMs=%.6f pctStage=%.3f warmCalls=%lld warmTotalMs=%.6f warmPerSimpleMs=%.6f\n",tag,mom?"momentum":"pressure",h0_cat_name(c),lev,n,t,t/std::max<long long>(n,1),t/std::max(outer,1),pct,nw,w,w/std::max(outer-1,1));
  }}
}

int main(int argc,char**argv){try{
  std::string mesh,tag="h8",wall="patch_0_0",inlet="patch_2_0",outlet="patch_1_0";
  double re=20,bulk=1,simpleTol=1e-6;
  double alphaU=.5,alphaP=.5;
  double momRtol=1e-6,momAtol=1e-12,momDrop=.1,momOmega=1.0;
  double pRtol=.5,pAtol=1e-12;
  double snapshotTol=5e-6;
  int maxOuter=2500,momMax=20000,pMax=20;
  int fineCsrRefreshEvery=1;
  std::string momentumWork="fgs1",runMode="fixed10",amgSmoother="cheb2",amgHierarchy="sa",cfInterp="direct",pressureSolver="pcg",cfCoarsening="pmis",cfStrength="classical-negative",amgSpectrumPolicy="auto";
  double amgJacobiOmega=.7;
  double pressureRichardsonOmega=1.0;
  double pressureChebLowFraction=.05;
  double pressurePowerSafety=1.15;
  int pressureChebDegree=3;
  int pressurePowerIts=6;
  int cfAggressiveFirst=0;
  int amgTerminal=1000;
  int amgPowerIts=16;
  int amgChebDegree=2;
  double amgLambdaSafety=1.5;
  double amgLambdaLowFraction=.05;
  double cfTheta=.25;
  int cfPmax=4;

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
    else if(a=="--amg-hierarchy"&&i+1<argc)amgHierarchy=argv[++i];
    else if(a=="--cf-theta"&&i+1<argc)cfTheta=std::atof(argv[++i]);
    else if(a=="--cf-pmax"&&i+1<argc)cfPmax=std::atoi(argv[++i]);
    else if(a=="--cf-interp"&&i+1<argc)cfInterp=argv[++i];
    else if(a=="--cf-coarsening"&&i+1<argc)cfCoarsening=argv[++i];
    else if(a=="--cf-strength"&&i+1<argc)cfStrength=argv[++i];
    else if(a=="--cf-aggressive-first"&&i+1<argc)cfAggressiveFirst=std::atoi(argv[++i]);
    else if(a=="--amg-terminal"&&i+1<argc)amgTerminal=std::atoi(argv[++i]);
    else if(a=="--amg-spectrum-policy"&&i+1<argc)amgSpectrumPolicy=argv[++i];
    else if(a=="--amg-power-its"&&i+1<argc)amgPowerIts=std::atoi(argv[++i]);
    else if(a=="--amg-cheb-degree"&&i+1<argc)amgChebDegree=std::atoi(argv[++i]);
    else if(a=="--amg-lambda-safety"&&i+1<argc)amgLambdaSafety=std::atof(argv[++i]);
    else if(a=="--amg-lambda-low-fraction"&&i+1<argc)amgLambdaLowFraction=std::atof(argv[++i]);
    else if(a=="--pressure-solver"&&i+1<argc)pressureSolver=argv[++i];
    else if(a=="--pressure-richardson-omega"&&i+1<argc)pressureRichardsonOmega=std::atof(argv[++i]);
    else if(a=="--pressure-cheb-degree"&&i+1<argc)pressureChebDegree=std::atoi(argv[++i]);
    else if(a=="--pressure-power-its"&&i+1<argc)pressurePowerIts=std::atoi(argv[++i]);
    else if(a=="--pressure-cheb-low-fraction"&&i+1<argc)pressureChebLowFraction=std::atof(argv[++i]);
    else if(a=="--pressure-power-safety"&&i+1<argc)pressurePowerSafety=std::atof(argv[++i]);
    else if(a=="--amg-smoother"&&i+1<argc)amgSmoother=argv[++i];
    else if(a=="--amg-jacobi-omega"&&i+1<argc)amgJacobiOmega=std::atof(argv[++i]);
    else if(a=="--momentum-work"&&i+1<argc)momentumWork=argv[++i];
    else if(a=="--run-mode"&&i+1<argc)runMode=argv[++i];
    else throw std::runtime_error("H8 usage error");
  }
  if(mesh.empty())throw std::runtime_error("--mesh required");
  if(!(simpleTol>0.0) || maxOuter<1)throw std::runtime_error("H8 invalid SIMPLE controls");
  if(!(alphaU>0.0&&alphaU<=1.0) || !(alphaP>0.0&&alphaP<=1.0))throw std::runtime_error("H8 alpha-u/alpha-p must be in (0,1]");
  if(momRtol<0.0 || momAtol<0.0 || momDrop<0.0 || momMax<1 || !(momOmega>0.0))throw std::runtime_error("H8 invalid momentum controls");
  if(pRtol<0.0 || pAtol<0.0 || pMax<1)throw std::runtime_error("H8 invalid pressure controls");
  if(amgHierarchy!="sa"&&amgHierarchy!="cf")throw std::runtime_error("H8 amg-hierarchy must be sa or cf");
  if(!(cfTheta>0.0&&cfTheta<=1.0))throw std::runtime_error("H8 cf-theta must be in (0,1]");
  if(cfPmax<1||cfPmax>16)throw std::runtime_error("H8 cf-pmax must be in [1,16]");
  if(cfInterp!="direct"&&cfInterp!="exti")throw std::runtime_error("H8 cf-interp must be direct or exti");
  if(cfCoarsening!="pmis")
    throw std::runtime_error("H8 cf-coarsening currently supports pmis");
  if(cfStrength!="classical-negative")
    throw std::runtime_error("H8 cf-strength currently supports classical-negative");
  if(cfAggressiveFirst!=0&&cfAggressiveFirst!=1)
    throw std::runtime_error("H8 cf-aggressive-first must be 0 or 1");
  if(amgTerminal<16)
    throw std::runtime_error("H8 amg-terminal must be >=16");
  if(amgSpectrumPolicy!="auto"&&amgSpectrumPolicy!="always"&&amgSpectrumPolicy!="off")
    throw std::runtime_error("H8 amg-spectrum-policy must be auto, always or off");
  if(amgPowerIts<1||amgPowerIts>64)
    throw std::runtime_error("H8 amg-power-its must be in [1,64]");
  if(amgChebDegree<1||amgChebDegree>16)
    throw std::runtime_error("H8 amg-cheb-degree must be in [1,16]");
  if(!(amgLambdaSafety>1.0&&amgLambdaSafety<=3.0))
    throw std::runtime_error("H8 amg-lambda-safety must be in (1,3]");
  if(!(amgLambdaLowFraction>0.0&&amgLambdaLowFraction<1.0))
    throw std::runtime_error("H8 amg-lambda-low-fraction must be in (0,1)");
  if(pressureSolver!="pcg"&&pressureSolver!="richardson"&&pressureSolver!="cheb")
    throw std::runtime_error("G12 pressure-solver must be pcg, richardson or cheb");
  if(!(pressureRichardsonOmega>0.0&&pressureRichardsonOmega<2.0))
    throw std::runtime_error("G12 Richardson omega must be in (0,2)");
  if(pressureChebDegree<1||pressureChebDegree>16)
    throw std::runtime_error("G12 Cheb degree must be in [1,16]");
  if(pressurePowerIts<1||pressurePowerIts>64)
    throw std::runtime_error("G12 pressure power its must be in [1,64]");
  if(!(pressureChebLowFraction>0.0&&pressureChebLowFraction<1.0))
    throw std::runtime_error("G12 Cheb low fraction must be in (0,1)");
  if(!(pressurePowerSafety>1.0&&pressurePowerSafety<=2.0))
    throw std::runtime_error("G12 pressure power safety must be in (1,2]");
  if(amgSmoother!="cheb2"&&amgSmoother!="jacobi"&&amgSmoother!="l1jacobi")throw std::runtime_error("H8 amg-smoother must be cheb2, jacobi or l1jacobi");
  if(!(amgJacobiOmega>0.0&&amgJacobiOmega<=2.0))throw std::runtime_error("H8 amg-jacobi-omega must be in (0,2]");
  if(!(snapshotTol>0.0))throw std::runtime_error("H8 snapshot tolerance must be positive");
  if(fineCsrRefreshEvery!=1)throw std::runtime_error("H8 requires fine CSR refreshEvery=1");
  if(momentumWork!="fgs1")throw std::runtime_error("H8 requires momentum-work=fgs1");
  if(!(runMode=="fixed10"||runMode=="converge"))throw std::runtime_error("H8 run-mode must be fixed10 or converge");
  if(runMode=="fixed10"&&maxOuter!=10)throw std::runtime_error("H8 fixed10 requires maxOuter=10");
  const bool physicalUseCurrentCSR=true;

  NODALS_CUDA(cudaSetDevice(0));cudaDeviceProp prop{};NODALS_CUDA(cudaGetDeviceProperties(&prop,0));upload_tensors();NODALS_CUDA(cudaDeviceSynchronize());
  const auto memBaseline=device_memory_info();
  auto M=load_foam_tet_mesh(mesh);auto S=build_g4_setup(M,re,bulk,wall,inlet,outlet);
  std::array<std::vector<double>,3>U0;for(auto&u:U0)u.assign((std::size_t)S.topo.n,0.0);
  std::vector<double>hostA,initDelta;std::array<std::vector<double>,3>hostConv;
  host_assemble_central_g4(S,U0,hostA,hostConv);auto initRau=host_finalize_relax_g4(S,hostA,alphaU,&initDelta);S.pressure.rAU=initRau;
  auto FH=build_h2_fine_csr_host(S);
  int effectiveAmgPowerIts=0;
  if(amgSpectrumPolicy=="always") effectiveAmgPowerIts=amgPowerIts;
  else if(amgSpectrumPolicy=="auto" && amgSmoother=="cheb2") effectiveAmgPowerIts=amgPowerIts;
  if(amgSmoother=="cheb2" && effectiveAmgPowerIts<=0)
    throw std::runtime_error("H8 Chebyshev AMG smoother requires spectrum estimation");
  SAHierarchyHost H;
  if(amgHierarchy=="sa")
    H=build_sa_hierarchy(M,S.pressure,16,6,18,amgTerminal,8,
                         effectiveAmgPowerIts,amgLambdaSafety,
                         amgLambdaLowFraction,4.0/3.0);
  else
    H=build_cf_hierarchy(S,FH,cfTheta,cfPmax,cfInterp,
                         cfAggressiveFirst!=0,amgTerminal,
                         effectiveAmgPowerIts,amgLambdaSafety,
                         amgLambdaLowFraction);
  std::printf("NODALS_GPU_G8_CF_CONFIG tag=%s hierarchy=%s coarsening=%s strength=%s theta=%.6f interp=%s pmax=%d aggressiveFirst=%d terminal=%d spectrumPolicy=%s effectivePowerIts=%d status=PASS\n",tag.c_str(),amgHierarchy.c_str(),cfCoarsening.c_str(),cfStrength.c_str(),cfTheta,cfInterp.c_str(),cfPmax,cfAggressiveFirst,amgTerminal,amgSpectrumPolicy.c_str(),effectiveAmgPowerIts);

  std::printf("NODALS_GPU_H8_CONFIG tag=%s precision=%s device=%s cc=%d.%d petsc=NONE mpi=NONE cells=%zu runMode=%s momentumWork=%s momentumResidualPolicy=%s physicalOperator=exact_current_CSR fineAMG=explicit_%s_CSR_warp coarseAMGSpMV=per_level_scalar_vs_warp_hybrid fineCsrNumericRefreshEvery=1 refreshKernel=warp_per_row spectrumRefresh=setup_only coarseHierarchyNumeric=setup_snapshot coarseSpMV=PER_LEVEL_SCALAR_WARP_HYBRID momentumDiffusion=PERSISTENT_NUMERIC_CSR momentumConvection=NUMERIC_ONLY_SHARED_XYZ momentumAssemblyExec=SETUP_SELECT_SCALAR_VS_WARP BGeometry=SETUP_PRECOMPUTED_24_%s_PER_CELL_AUTOSELECT alphaU=%.8g alphaP=%.8g simpleTol=%.3e maxOuter=%d momentumOmega=%.8g pressureRtol=%.3e pressureAtol=%.3e pressureMaxIts=%d reductions=FP64 state=%s operator=%s amg=%s\n",
    tag.c_str(),kPrecisionName,prop.name,prop.major,prop.minor,M.tets.size(),runMode.c_str(),momentumWork.c_str(),runMode=="fixed10"?"NONE":"CONVERGENCE_ONLY_PRE_SWEEP",
    kPrecisionName,kPrecisionName,alphaU,alphaP,simpleTol,maxOuter,momOmega,pRtol,pAtol,pMax,kPrecisionName,kPrecisionName,kPrecisionName);
  std::printf("NODALS_GPU_H8_TUNING alphaU=%.8g alphaP=%.8g momentumWork=%s momentumOmega=%.8g momentumAdaptiveTol=DISABLED momentumResidualPolicy=%s pRtol=%.3e pAtol=%.3e pMax=%d simpleTol=%.3e maxOuter=%d snapshotTol=%.3e fineCsrRefreshEvery=1 status=PASS\n",
    alphaU,alphaP,momentumWork.c_str(),momOmega,runMode=="fixed10"?"NONE":"CONVERGENCE_ONLY_PRE_SWEEP",pRtol,pAtol,pMax,simpleTol,maxOuter,snapshotTol);
  std::printf("NODALS_GPU_H8_FINE_CSR_TOPOLOGY tag=%s cells=%zu nnz=%zu rowMean=%.6f rowMax=%u directedContrib=%llu csrMiB=%.3f compactRefreshMetadataMiB=%.3f totalFineCsrMiB=%.3f bytesPerCell=%.3f status=PASS\n",
    tag.c_str(),M.tets.size(),FH.col.size(),FH.rowMean,FH.rowMax,(unsigned long long)FH.directedContrib,
    g5_mib(FH.csr_bytes()),g5_mib(FH.metadata_bytes()),g5_mib(FH.bytes()),(double)FH.bytes()/std::max<std::size_t>(M.tets.size(),1));
  std::printf("NODALS_GPU_H8_SETUP tag=%s cells=%zu freeVel=%d momentumNnz=%zu colors=%d hierarchyLevels=%zu terminal=%d transfer0Nnz=%zu cellPlanHostBytes=%zu cellPlanDeviceBytes=%zu baselineUsedMiB=%.3f totalMiB=%.3f status=PASS\n",
    tag.c_str(),M.tets.size(),S.topo.n,S.topo.val.size(),S.coloring.ncolors,H.csr.size(),H.terminal_n,H.P.empty()?0:H.P[0].val.size(),sizeof(G4CellPlanHost),sizeof(G4CellPlanDevice),g5_used_mib(memBaseline),(double)memBaseline.total_bytes/(1024.0*1024.0));
  std::printf("NODALS_GPU_H8_MOMENTUM_ASSEMBLY_DESIGN tag=%s diffusionCSR=BUILT_ONCE_HOST_UPLOADED_ONCE_DEVICE_PERSISTENT convectionCSRTopology=STATIC convectionNumeric=REFRESH_EACH_OUTER commonOperatorXYZ=YES pressureGradientBTopology=STATIC pressureGradientAction=APPLY_CURRENT_P fixedDiffusionDirichletRHS=PERSISTENT variableViscosityDesign=REFRESH_DIFFUSION_NUMERICS_ONLY_NO_TOPOLOGY_REBUILD status=PASS\n",tag.c_str());

  G4Gpu G=upload_all(S,H,initRau,FH);G.pf.cells=G.cells.data();G.pf.bcoeff=G.bcoeff.data();G.pf.rau_live=G.rau.data();G.pf.csr_pc=&G.fineCsr;G.pf.physicalUseCurrentCSR=physicalUseCurrentCSR;G.amg.fine=&G.pf;G.amg.smoother=amgSmoother;G.amg.jacobiOmega=amgJacobiOmega;G.amg.chebDegree=amgChebDegree;G.amg.powerIts=effectiveAmgPowerIts;G.amg.lambdaSafety=amgLambdaSafety;G.amg.lambdaLowFraction=amgLambdaLowFraction;
  std::printf("NODALS_GPU_G9_L1_CONFIG tag=%s smoother=%s denominator=%s relaxWeight=%.8g fineL1Refresh=%s coarseL1=SETUP_SNAPSHOT definition=SUM_ABS_ROW status=PASS\n",tag.c_str(),amgSmoother.c_str(),amgSmoother=="l1jacobi"?"L1_ROW_ABS_SUM":"DIAGONAL",amgJacobiOmega,amgSmoother=="l1jacobi"?"EVERY_FINE_CSR_REFRESH":"OFF");
  std::printf("NODALS_GPU_H8_AMG_SMOOTHER tag=%s hierarchy=%s smoother=%s jacobiOmega=%.8g chebDegree=%d powerIts=%d hierarchyConstruction=SELECTED_BY_AMG_HIERARCHY spectrumSetup=SETUP_ONLY status=PASS\n",
    tag.c_str(),amgHierarchy.c_str(),amgSmoother.c_str(),amgJacobiOmega,G.amg.chebDegree,G.amg.powerIts);
  h7_select_momentum_assembly(G,alphaU,prop.multiProcessorCount,tag.c_str());
  h8_select_precomputed_b(G,tag.c_str());
  double pressureChebLambdaMax=1.0;
  if(pressureSolver=="cheb"){
    pressureChebLambdaMax=pressure_precond_power(
      G.pf,G.amg,pressurePowerIts,pressurePowerSafety,
      G.pcg,G.dotScratch,tag.c_str());
  }
  std::printf(
    "NODALS_GPU_G12_PRESSURE_SOLVER tag=%s solver=%s richardsonOmega=%.8g "
    "chebDegree=%d powerIts=%d chebLowFraction=%.8g powerSafety=%.8g "
    "lambdaMax=%.12e preconditioner=SELECTED_AMG_CONFIGURATION "
    "stoppingTolerance=UNCHANGED status=PASS\n",
    tag.c_str(),pressureSolver.c_str(),pressureRichardsonOmega,
    pressureChebDegree,pressurePowerIts,pressureChebLowFraction,
    pressurePowerSafety,pressureChebLambdaMax);
  if(gate6_setup_autotune_enabled()){
    std::vector<AMGReal> g6FineVal(G.fineCsr.val.size());
    G.fineCsr.val.download(g6FineVal.data(),g6FineVal.size());
    g6_strength_sweep(FH.n,FH.row,FH.col,g6FineVal,tag.c_str());
    g7_pmis_diagnostics(FH.n,FH.row,FH.col,g6FineVal,0.25,tag.c_str());
  }else{
    std::printf(
      "NODALS_GPU_PM2_AUX_DIAGNOSTICS tag=%s "
      "g6StrengthSweep=SKIP g7PmisDuplicate=SKIP "
      "setupOnly=1 diagnosticEnv=NODALS_SETUP_AUTOTUNE "
      "solverHierarchy=UNCHANGED status=PASS\n",
      tag.c_str());
  }
  h6_select_coarse_spmv(G,H,tag.c_str(),200);
  if(effectiveAmgPowerIts>0){
    g5_gpu_spectrum_refresh(G,H);
    std::printf("NODALS_GPU_G10_CF_SPECTRUM_POLICY tag=%s hierarchy=%s smoother=%s policy=%s hostPower=COMPUTE gpuPower=COMPUTE powerIts=%d status=PASS\n",
      tag.c_str(),amgHierarchy.c_str(),amgSmoother.c_str(),
      amgSpectrumPolicy.c_str(),effectiveAmgPowerIts);
  }else{
    G.amg.fineLambda=1.0;
    for(auto&v:G.amg.levelLambda)v=1.0;
    std::printf("NODALS_GPU_G10_CF_SPECTRUM_POLICY tag=%s hierarchy=%s smoother=%s policy=%s hostPower=SKIP gpuPower=SKIP powerIts=0 status=PASS\n",
      tag.c_str(),amgHierarchy.c_str(),amgSmoother.c_str(),
      amgSpectrumPolicy.c_str());
  }
  h2_action_parity_and_bench(G,tag.c_str(),24);
  H0Profiler H0P;g_h0=&H0P;
  NODALS_CUDA(cudaDeviceSynchronize());const auto memUpload=device_memory_info();
  const double baselineUsed=g5_used_mib(memBaseline),uploadUsed=g5_used_mib(memUpload),explicitMiB=g5_mib(G.bytes());
  std::printf("NODALS_GPU_H8_MEMORY tag=%s point=after_upload cells=%zu baselineUsedMiB=%.3f usedMiB=%.3f deltaFromBaselineMiB=%.3f explicitMiB=%.3f explicitBytesPerCell=%.3f status=PASS\n",
    tag.c_str(),M.tets.size(),baselineUsed,uploadUsed,uploadUsed-baselineUsed,explicitMiB,(double)G.bytes()/std::max<std::size_t>(M.tets.size(),1));

  bool converged=false,pressureAll=true,finiteAll=true;double cont0=-1.0,contRel=1.0;
  long long sumP=0;int convergenceAuditCalls=0;std::array<double,3>lastAuditRel{{NAN,NAN,NAN}};
  double tAssembly=0,tMomentum=0,tContinuity=0,tPressure=0,tPupdate=0;
  auto wall0=std::chrono::steady_clock::now();
  int finalIt=0,fineCsrRefreshCount=0;
  for(int it=1;it<=maxOuter;++it){
    H0P.currentOuter=it;if(H0P.used)throw std::runtime_error("H8 profiler nonempty at outer start");
    G5StageEvents E;E.rec(0);
    assemble_live(G,alphaU);E.rec(1);
    if(it==1){
      double rr=g5_rau_snapshot_rel(G);
      std::printf("NODALS_GPU_H8_SNAPSHOT_PARITY it=1 rAURel=%.3e tol=%.3e status=%s\n",rr,snapshotTol,rr<snapshotTol?"PASS":"FAIL");
      if(!(rr<snapshotTol))throw std::runtime_error("H8 setup rAU snapshot mismatch");
    }
    G.pf.bt_state(G.p.data());
    {int q=h0_begin(H0_MOM_RHS);g4_momentum_rhs_kernel<<<g4grid(G.nv),G4B>>>(G.nv,G.s0.data(),G.s1.data(),G.s2.data(),G.c0.data(),G.c1.data(),G.c2.data(),G.pf.v0.data(),G.pf.v1.data(),G.pf.v2.data(),G.delta.data(),G.u0.data(),G.u1.data(),G.u2.data(),G.b0.data(),G.b1.data(),G.b2.data());NODALS_CUDA(cudaGetLastError());h0_end(q);}
    bool auditedThisOuter=false;
    std::array<double,3>auditRel{{NAN,NAN,NAN}};
    if(runMode=="converge" && it>1 && contRel<=simpleTol){
      auditRel=momentum_initial_rel_audit(G);lastAuditRel=auditRel;++convergenceAuditCalls;auditedThisOuter=true;
    }
    const char* momentumDirection=fixed_momentum_work(G,S.coloring,momOmega,momentumWork,it);E.rec(2);

    if(G.h8PrecomputedContinuity){
      h8_continuity_kernel<<<g4grid(G.nc),G4B>>>(
        G.cells.data(),G.bcoeff.data(),G.nc,G.fixedDiv.data(),
        G.u0.data(),G.u1.data(),G.u2.data(),G.cont.data());
    }else{
      g4_continuity_kernel<<<g4grid(G.nc),G4B>>>(
        G.cells.data(),G.nc,G.fixedDiv.data(),
        G.u0.data(),G.u1.data(),G.u2.data(),G.cont.data());
    }
    NODALS_CUDA(cudaGetLastError());
    double cn=device_norm2(G.cont.data(),G.nc,G.dotScratch);if(it==1)cont0=cn;contRel=cn/std::max(cont0,1e-300);
    g4_negate_kernel<<<g4grid(G.nc),G4B>>>(G.nc,G.cont.data());NODALS_CUDA(cudaGetLastError());E.rec(3);

    if(it==1 || ((it-1)%fineCsrRefreshEvery)==0){
      G.fineCsr.refresh(G.cells.data(),G.rau.data(),G.pf.diag_pc.data(),(amgSmoother=="l1jacobi"?G.amg.fine_l1.data():nullptr));
      ++fineCsrRefreshCount;
    }
    G4PCGResult pr;
    if(pressureSolver=="pcg"){
      pr=pressure_pcg(
        G.pf,G.amg,G.cont.data(),pRtol,pAtol,pMax,G.pcg,G.dotScratch);
    }else if(pressureSolver=="richardson"){
      pr=pressure_richardson(
        G.pf,G.amg,G.cont.data(),pRtol,pAtol,pMax,
        pressureRichardsonOmega,G.pcg,G.dotScratch);
    }else{
      pr=pressure_cheb_poly(
        G.pf,G.amg,G.cont.data(),pRtol,pAtol,pMax,
        pressureChebLambdaMax,pressureChebLowFraction,pressureChebDegree,
        G.pcg,G.dotScratch);
    }
    pressureAll=pressureAll&&pr.ok;if(!pr.ok)throw std::runtime_error("H8 pressure PCG failed requested FP32 inexact target");sumP+=pr.its;E.rec(4);
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
      std::printf("NODALS_GPU_H8_SIMPLE tag=%s it=%d relCont=%.12e momentumWork=%s momentumDirection=%s residualAudit=%s auditInitRel=[%.3e,%.3e,%.3e] pCG=%d pTrueRel=%.3e allMomentumMet=%d converged=%d status=%s\n",
        tag.c_str(),it,contRel,momentumWork.c_str(),momentumDirection,auditedThisOuter?"DONE":"SKIPPED",auditRel[0],auditRel[1],auditRel[2],pr.its,pr.rel,(int)allMomentumMet,(int)converged,(pressureAll&&finiteAll)?"PASS":"FAIL");
    if(runMode=="converge"&&converged)break;
  }
  NODALS_CUDA(cudaDeviceSynchronize());auto wall1=std::chrono::steady_clock::now();const auto memEnd=device_memory_info();

  std::vector<StateReal>pf((std::size_t)G.nc);
  G.p.download(pf.data(),pf.size());
  std::vector<double>pd(pf.size());
  for(std::size_t i=0;i<pf.size();++i)pd[i]=(double)pf[i];

  std::array<std::vector<double>,3> finalU;
  for(int d=0;d<3;++d)
    finalU[(std::size_t)d].resize((std::size_t)G.nv);
  std::vector<StateReal>uf((std::size_t)G.nv);
  G.u0.download(uf.data(),uf.size());
  for(std::size_t i=0;i<uf.size();++i)finalU[0][i]=(double)uf[i];
  G.u1.download(uf.data(),uf.size());
  for(std::size_t i=0;i<uf.size();++i)finalU[1][i]=(double)uf[i];
  G.u2.download(uf.data(),uf.size());
  for(std::size_t i=0;i<uf.size();++i)finalU[2][i]=(double)uf[i];

  const auto hpErr=h8_compute_hp_errors(M,S,finalU,pd);
  double dp=pressure_drop_fit_g4(M,pd),exact=S.pipe.hpDrop,
         dpErr=std::abs(dp-exact)/std::max(std::abs(exact),1e-300);

  std::printf(
    "NODALS_GPU_H8_HP_ERROR tag=%s cells=%zu hEff=%.12e "
    "U_L2=%.12e U_relL2=%.12e "
    "P_shifted_L2=%.12e P_shifted_relL2=%.12e pressureShift=%.12e "
    "pressureDropFit=%.12e exactPressureDrop=%.12e "
    "pressureDropRelErr=%.12e quadrature=duffy5_125 "
    "finalStateD2H=AFTER_LOOP status=%s\n",
    tag.c_str(),M.tets.size(),hpErr.hEff,hpErr.uL2,hpErr.uRelL2,
    hpErr.pShiftedL2,hpErr.pShiftedRelL2,hpErr.pressureShift,
    dp,exact,dpErr,converged?"PASS":"UNCONVERGED");
  double wallMs=std::chrono::duration<double,std::milli>(wall1-wall0).count();
  double usedEnd=g5_used_mib(memEnd);

  if(runMode=="fixed10"){
    std::printf("NODALS_GPU_H8_FIXED10 tag=%s cells=%zu fixedOuter=%d momentumWork=%s momentumPassesPerOuter=1 momentumResidualAudits=0 diffusionPolicy=PERSISTENT_NUMERIC_CSR convectionPolicy=NUMERIC_ONLY_EACH_OUTER fineCsrRefreshEvery=1 fineCsrRefreshCount=%d finalRelCont=%.12e avgPressureIts=%.6f pressureDropFit=%.12e exactPressureDrop=%.12e pressureDropRelErr=%.12e loopMs=%.3f avgSimpleMs=%.6f assemblyAvgMs=%.6f momentumAvgMs=%.6f continuityAvgMs=%.6f pressureAvgMs=%.6f pressureUpdateAvgMs=%.6f status=%s\n",
      tag.c_str(),M.tets.size(),finalIt,momentumWork.c_str(),fineCsrRefreshCount,contRel,finalIt?(double)sumP/finalIt:0.0,dp,exact,dpErr,wallMs,finalIt?wallMs/finalIt:0.0,finalIt?tAssembly/finalIt:0.0,finalIt?tMomentum/finalIt:0.0,finalIt?tContinuity/finalIt:0.0,finalIt?tPressure/finalIt:0.0,finalIt?tPupdate/finalIt:0.0,(pressureAll&&finiteAll&&finalIt==10)?"PASS":"FAIL");
  }else{
    std::printf("NODALS_GPU_H8_FULLCONV tag=%s cells=%zu momentumWork=%s simpleTol=%.3e outerIts=%d converged=%d finalRelCont=%.12e convergenceAuditCalls=%d finalAuditInitRel=[%.3e,%.3e,%.3e] avgPressureIts=%.6f pressureDropFit=%.12e exactPressureDrop=%.12e pressureDropRelErr=%.12e loopMs=%.3f avgSimpleMs=%.6f assemblyAvgMs=%.6f momentumAvgMs=%.6f continuityAvgMs=%.6f pressureAvgMs=%.6f pressureUpdateAvgMs=%.6f status=%s\n",
      tag.c_str(),M.tets.size(),momentumWork.c_str(),simpleTol,finalIt,(int)converged,contRel,convergenceAuditCalls,lastAuditRel[0],lastAuditRel[1],lastAuditRel[2],finalIt?(double)sumP/finalIt:0.0,dp,exact,dpErr,wallMs,finalIt?wallMs/finalIt:0.0,finalIt?tAssembly/finalIt:0.0,finalIt?tMomentum/finalIt:0.0,finalIt?tContinuity/finalIt:0.0,finalIt?tPressure/finalIt:0.0,finalIt?tPupdate/finalIt:0.0,converged?"PASS":"FAIL");
  }
  std::printf("NODALS_GPU_H8_MOMENTUM_AUDIT tag=%s runMode=%s policy=%s calls=%d postSweepAudits=0 status=PASS\n",tag.c_str(),runMode.c_str(),runMode=="fixed10"?"NONE":"CONVERGENCE_ONLY_PRE_SWEEP",convergenceAuditCalls);
  h0_print_profile(tag.c_str(),finalIt,tPressure,tMomentum,H0P);
  std::printf("NODALS_GPU_H8_MEMORY tag=%s point=after_convergence cells=%zu baselineUsedMiB=%.3f usedMiB=%.3f deltaFromBaselineMiB=%.3f explicitMiB=%.3f runtimeDriftMiB=%.3f totalMiB=%.3f status=PASS\n",
    tag.c_str(),M.tets.size(),baselineUsed,usedEnd,usedEnd-baselineUsed,explicitMiB,usedEnd-uploadUsed,(double)memEnd.total_bytes/(1024.0*1024.0));
  std::printf("NODALS_GPU_H8_RESIDENCY tag=%s O_N_H2D_inside_SIMPLE=0 O_N_D2H_inside_SIMPLE=%s finalPressureD2H=AFTER_LOOP reductions=FP64 deviceNumericStorage=%s physicalOperator=CURRENT_EXACT_CSR fineCsrNumericRefreshEvery=1 refreshKernel=WARP_PER_ROW momentumWork=%s momentumResidualPolicy=%s momentumDiffusion=PERSISTENT_NUMERIC_CSR momentumConvection=NUMERIC_ONLY_EACH_OUTER spectrumRefresh=SETUP_ONLY coarseHierarchyNumeric=SETUP_SNAPSHOT coarseSpMV=PER_LEVEL_SCALAR_WARP_HYBRID H8_scope=PRECOMPUTED_B_GEOMETRY status=PASS\n",tag.c_str(),runMode=="fixed10"?"0":"SCALAR_CONVERGENCE_AUDITS_ONLY",kPrecisionName,momentumWork.c_str(),runMode=="fixed10"?"NONE":"CONVERGENCE_ONLY_PRE_SWEEP");
  const bool finalPass=(runMode=="fixed10")?(finalIt==10&&pressureAll&&finiteAll):converged;
  std::printf("NODALS_GPU_RESULT gate=H8 tag=%s cells=%zu precision=%s runMode=%s momentumWork=%s outer=%d simpleTol=%.3e fineCsrRefreshEvery=1 physicalOperator=current_exact_CSR pressureAll=%s finite=%s noPetsc=1 noMPI=1 status=%s\n",
    tag.c_str(),M.tets.size(),kPrecisionName,runMode.c_str(),momentumWork.c_str(),finalIt,simpleTol,pressureAll?"PASS":"FAIL",finiteAll?"PASS":"FAIL",finalPass?"PASS":"FAIL");
  return finalPass?0:35;
}catch(const std::exception&e){std::fprintf(stderr,"NODALS_GPU_H8_EXCEPTION what=%s\n",e.what());return 90;}}
