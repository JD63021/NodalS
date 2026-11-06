#pragma once
// H2B derivative of H2 exact-fine-CSR support.
#include "cuda_runtime.hpp"
#include "precision.hpp"
#include "g4_host.hpp"
#include "g5e_fp32_kernels.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace nodals_gpu {

struct H2FineCSRHost {
  int n=0,nv=0;
  std::vector<std::int32_t> row;
  std::vector<std::int32_t> col;
  std::vector<std::int32_t> diagPos;

  // Persistent exact velocity-DOF incidence.
  std::vector<std::int32_t> incOff;
  std::vector<std::uint32_t> packed; // (cell<<3)|localBasis

  // Compact refresh map.  contribOff[c]: first directed basis contribution for
  // pressure row c.  slot[q] is the uint8 row-local CSR slot for that contribution.
  std::vector<std::int32_t> contribOff;
  std::vector<std::uint8_t> slot;

  std::uint64_t directedContrib=0;
  std::uint32_t rowMax=0;
  double rowMean=0.0;

  std::size_t bytes() const {
    return row.size()*sizeof(std::int32_t)+col.size()*sizeof(std::int32_t)+
           diagPos.size()*sizeof(std::int32_t)+incOff.size()*sizeof(std::int32_t)+
           packed.size()*sizeof(std::uint32_t)+contribOff.size()*sizeof(std::int32_t)+
           slot.size()*sizeof(std::uint8_t)+col.size()*sizeof(AMGReal);
  }
  std::size_t metadata_bytes() const {
    return diagPos.size()*sizeof(std::int32_t)+incOff.size()*sizeof(std::int32_t)+
           packed.size()*sizeof(std::uint32_t)+contribOff.size()*sizeof(std::int32_t)+
           slot.size()*sizeof(std::uint8_t);
  }
  std::size_t csr_bytes() const {
    return row.size()*sizeof(std::int32_t)+col.size()*sizeof(std::int32_t)+
           col.size()*sizeof(AMGReal);
  }
};

inline H2FineCSRHost build_h2_fine_csr_host_reference(const G4SetupHost&S){
  H2FineCSRHost H;
  H.n=(int)S.cells.size();
  H.nv=S.topo.n;
  if((std::uint64_t)H.n >= (1ull<<29))
    throw std::runtime_error("H2 packed cell index exceeds 29 bits");

  // Build exact free velocity basis -> incident pressure cell/local-basis map.
  std::vector<std::int32_t> count((std::size_t)H.nv,0);
  for(int c=0;c<H.n;++c)for(int a=0;a<8;++a){
    int g=S.cells[(std::size_t)c].ref[a];
    if(g>=0){
      if(count[(std::size_t)g]==std::numeric_limits<std::int32_t>::max())
        throw std::runtime_error("H2 incidence count overflow");
      ++count[(std::size_t)g];
    }
  }
  H.incOff.assign((std::size_t)H.nv+1,0);
  std::int64_t nt=0;
  for(int g=0;g<H.nv;++g){
    nt+=count[(std::size_t)g];
    if(nt>std::numeric_limits<std::int32_t>::max())
      throw std::runtime_error("H2 incidence total exceeds int32");
    H.incOff[(std::size_t)g+1]=(std::int32_t)nt;
  }
  H.packed.resize((std::size_t)nt);
  std::vector<std::int32_t> cur=H.incOff;
  for(int c=0;c<H.n;++c)for(int a=0;a<8;++a){
    int g=S.cells[(std::size_t)c].ref[a];
    if(g>=0){
      H.packed[(std::size_t)cur[(std::size_t)g]++]=
        ((std::uint32_t)c<<3)|(std::uint32_t)a;
    }
  }

  H.row.assign((std::size_t)H.n+1,0);
  H.diagPos.assign((std::size_t)H.n,-1);
  H.contribOff.assign((std::size_t)H.n+1,0);

  // H1 measured ~72 nnz/row and ~102 directed contributions/row.  Reserving
  // close to those values avoids large repeated reallocations without affecting
  // correctness on another mesh.
  H.col.reserve((std::size_t)H.n*76u);
  H.slot.reserve((std::size_t)H.n*108u);

  std::vector<std::int32_t> target,uniq;
  target.reserve(192);uniq.reserve(192);

  for(int c=0;c<H.n;++c){
    target.clear();
    const auto&cp=S.cells[(std::size_t)c];
    for(int a=0;a<8;++a){
      int g=cp.ref[a];
      if(g<0)continue;
      for(std::int32_t k=H.incOff[(std::size_t)g];k<H.incOff[(std::size_t)g+1];++k)
        target.push_back((std::int32_t)(H.packed[(std::size_t)k]>>3));
    }

    H.contribOff[(std::size_t)c]=(std::int32_t)H.slot.size();
    uniq=target;
    std::sort(uniq.begin(),uniq.end());
    uniq.erase(std::unique(uniq.begin(),uniq.end()),uniq.end());
    if(uniq.empty())throw std::runtime_error("H2 empty pressure CSR row");
    if(uniq.size()>255)throw std::runtime_error("H2 row width exceeds uint8 slot capacity");

    const std::int64_t next=(std::int64_t)H.col.size()+(std::int64_t)uniq.size();
    if(next>std::numeric_limits<std::int32_t>::max())
      throw std::runtime_error("H2 fine pressure CSR nnz exceeds int32");
    H.row[(std::size_t)c]=(std::int32_t)H.col.size();

    auto dit=std::lower_bound(uniq.begin(),uniq.end(),c);
    if(dit==uniq.end()||*dit!=c)throw std::runtime_error("H2 fine pressure diagonal absent");
    H.diagPos[(std::size_t)c]=(std::int32_t)H.col.size()+(std::int32_t)(dit-uniq.begin());

    H.rowMax=std::max<std::uint32_t>(H.rowMax,(std::uint32_t)uniq.size());
    H.col.insert(H.col.end(),uniq.begin(),uniq.end());

    // Preserve exactly the same nested (local basis, incidence) order used by
    // the numeric refresh kernel.
    for(std::int32_t t:target){
      auto it=std::lower_bound(uniq.begin(),uniq.end(),t);
      if(it==uniq.end()||*it!=t)throw std::runtime_error("H2 slot lookup failed");
      H.slot.push_back((std::uint8_t)(it-uniq.begin()));
    }
    if(H.slot.size()>((std::size_t)std::numeric_limits<std::int32_t>::max()))
      throw std::runtime_error("H2 directed contribution total exceeds int32");
    H.contribOff[(std::size_t)c+1]=(std::int32_t)H.slot.size();
  }
  H.row[(std::size_t)H.n]=(std::int32_t)H.col.size();
  H.directedContrib=H.slot.size();
  H.rowMean=H.n?(double)H.col.size()/(double)H.n:0.0;
  return H;
}

// -----------------------------------------------------------------------------
// P5C / validated Gate 5A:
// parallel fine-pressure CSR symbolic topology + compact uint8 slot map.
//
// The exact free-velocity incidence is retained. Pressure rows are independent.
// Two passes avoid retaining a large per-row symbolic scratch structure.
// -----------------------------------------------------------------------------

inline int p5c_finecsr_threads(){
  int n=0;
  if(const char*e=std::getenv("NODALS_FINECSR_THREADS"))n=std::atoi(e);
  if(n<=0){
    unsigned h=std::thread::hardware_concurrency();
    n=h?std::min((int)h,16):1;
  }
  return std::max(1,std::min(n,64));
}

inline void p5c_build_target(
    const G4SetupHost&S,const H2FineCSRHost&H,int c,
    std::vector<std::int32_t>&target)
{
  target.clear();
  const auto&cp=S.cells[(std::size_t)c];
  for(int a=0;a<8;++a){
    const int g=cp.ref[a];
    if(g<0)continue;
    for(std::int32_t k=H.incOff[(std::size_t)g];
        k<H.incOff[(std::size_t)g+1];++k)
      target.push_back((std::int32_t)(H.packed[(std::size_t)k]>>3));
  }
}

inline void p5c_make_uniq(
    const std::vector<std::int32_t>&target,
    std::vector<std::int32_t>&uniq)
{
  uniq=target;
  std::sort(uniq.begin(),uniq.end());
  uniq.erase(std::unique(uniq.begin(),uniq.end()),uniq.end());
}

inline H2FineCSRHost build_h2_fine_csr_host_parallel(const G4SetupHost&S)
{
  using clock=std::chrono::steady_clock;
  const auto all0=clock::now();

  H2FineCSRHost H;
  H.n=(int)S.cells.size();
  H.nv=S.topo.n;
  if((std::uint64_t)H.n >= (1ull<<29))
    throw std::runtime_error("H2 packed cell index exceeds 29 bits");

  const int nth=p5c_finecsr_threads();

  // Exact reference incidence construction.
  const auto inc0=clock::now();
  std::vector<std::int32_t> count((std::size_t)H.nv,0);

  for(int c=0;c<H.n;++c)
    for(int a=0;a<8;++a){
      const int g=S.cells[(std::size_t)c].ref[a];
      if(g>=0){
        if(count[(std::size_t)g]==std::numeric_limits<std::int32_t>::max())
          throw std::runtime_error("H2 incidence count overflow");
        ++count[(std::size_t)g];
      }
    }

  H.incOff.assign((std::size_t)H.nv+1,0);
  std::int64_t nt=0;
  for(int g=0;g<H.nv;++g){
    nt+=count[(std::size_t)g];
    if(nt>std::numeric_limits<std::int32_t>::max())
      throw std::runtime_error("H2 incidence total exceeds int32");
    H.incOff[(std::size_t)g+1]=(std::int32_t)nt;
  }

  H.packed.resize((std::size_t)nt);
  std::vector<std::int32_t> cur=H.incOff;
  for(int c=0;c<H.n;++c)
    for(int a=0;a<8;++a){
      const int g=S.cells[(std::size_t)c].ref[a];
      if(g>=0)
        H.packed[(std::size_t)cur[(std::size_t)g]++]=
          ((std::uint32_t)c<<3)|(std::uint32_t)a;
    }

  std::vector<std::int32_t>().swap(cur);
  std::vector<std::int32_t>().swap(count);
  const auto inc1=clock::now();

  std::printf(
    "NODALS_GPU_P5C_FINECSR_STAGE stage=incidence status=PASS entries=%zu "
    "seconds=%.6f threads=%d rssMiB=%.3f hwmMiB=%.3f\n",
    H.packed.size(),std::chrono::duration<double>(inc1-inc0).count(),nth,
    g4a_status_mib("VmRSS"),g4a_status_mib("VmHWM"));

  // Pass 1: exact per-row nnz and directed contribution counts.
  const auto p10=clock::now();
  std::vector<std::int32_t> rowCount((std::size_t)H.n,0);
  std::vector<std::int32_t> contribCount((std::size_t)H.n,0);

  std::vector<std::uint64_t> targetThread((std::size_t)nth,0);
  std::vector<std::uint32_t> targetMaxThread((std::size_t)nth,0);
  std::vector<std::uint32_t> rowMaxThread((std::size_t)nth,0);

  g4a_parallel_chunks(H.n,nth,128,[&](int t,int cb,int ce){
    std::vector<std::int32_t>target,uniq;
    target.reserve(192);
    uniq.reserve(192);

    std::uint64_t tsum=0;
    std::uint32_t tmax=0,rmax=0;

    for(int c=cb;c<ce;++c){
      p5c_build_target(S,H,c,target);
      p5c_make_uniq(target,uniq);

      if(uniq.empty())throw std::runtime_error("H2 empty pressure CSR row");
      if(uniq.size()>255)throw std::runtime_error("H2 row width exceeds uint8 slot capacity");
      if(target.size()>(std::size_t)std::numeric_limits<std::int32_t>::max())
        throw std::runtime_error("H2 per-row directed contribution overflow");

      auto dit=std::lower_bound(uniq.begin(),uniq.end(),c);
      if(dit==uniq.end()||*dit!=c)
        throw std::runtime_error("H2 fine pressure diagonal absent");

      rowCount[(std::size_t)c]=(std::int32_t)uniq.size();
      contribCount[(std::size_t)c]=(std::int32_t)target.size();

      tsum+=(std::uint64_t)target.size();
      tmax=std::max<std::uint32_t>(tmax,(std::uint32_t)target.size());
      rmax=std::max<std::uint32_t>(rmax,(std::uint32_t)uniq.size());
    }

    targetThread[(std::size_t)t]+=tsum;
    targetMaxThread[(std::size_t)t]=std::max(targetMaxThread[(std::size_t)t],tmax);
    rowMaxThread[(std::size_t)t]=std::max(rowMaxThread[(std::size_t)t],rmax);
  });
  const auto p11=clock::now();

  // Exact prefix allocation.
  const auto pr0=clock::now();
  H.row.assign((std::size_t)H.n+1,0);
  H.contribOff.assign((std::size_t)H.n+1,0);
  H.diagPos.assign((std::size_t)H.n,-1);

  std::int64_t nnz64=0,dir64=0;
  for(int c=0;c<H.n;++c){
    H.row[(std::size_t)c]=(std::int32_t)nnz64;
    H.contribOff[(std::size_t)c]=(std::int32_t)dir64;

    nnz64+=(std::int64_t)rowCount[(std::size_t)c];
    dir64+=(std::int64_t)contribCount[(std::size_t)c];

    if(nnz64>std::numeric_limits<std::int32_t>::max())
      throw std::runtime_error("H2 fine pressure CSR nnz exceeds int32");
    if(dir64>std::numeric_limits<std::int32_t>::max())
      throw std::runtime_error("H2 directed contribution total exceeds int32");
  }

  H.row[(std::size_t)H.n]=(std::int32_t)nnz64;
  H.contribOff[(std::size_t)H.n]=(std::int32_t)dir64;
  H.col.resize((std::size_t)nnz64);
  H.slot.resize((std::size_t)dir64);
  const auto pr1=clock::now();

  // Pass 2: exact sorted columns, diagonal position and slot map.
  const auto p20=clock::now();
  g4a_parallel_chunks(H.n,nth,128,[&](int,int cb,int ce){
    std::vector<std::int32_t>target,uniq;
    target.reserve(192);
    uniq.reserve(192);

    for(int c=cb;c<ce;++c){
      p5c_build_target(S,H,c,target);
      p5c_make_uniq(target,uniq);

      if((std::int32_t)uniq.size()!=rowCount[(std::size_t)c] ||
         (std::int32_t)target.size()!=contribCount[(std::size_t)c])
        throw std::runtime_error("P5C pass mismatch");

      const std::int32_t rb=H.row[(std::size_t)c];
      std::copy(uniq.begin(),uniq.end(),H.col.begin()+rb);

      auto dit=std::lower_bound(uniq.begin(),uniq.end(),c);
      H.diagPos[(std::size_t)c]=rb+(std::int32_t)(dit-uniq.begin());

      const std::int32_t qb=H.contribOff[(std::size_t)c];
      for(std::size_t q=0;q<target.size();++q){
        auto it=std::lower_bound(uniq.begin(),uniq.end(),target[q]);
        if(it==uniq.end()||*it!=target[q])
          throw std::runtime_error("H2 slot lookup failed");
        H.slot[(std::size_t)qb+q]=(std::uint8_t)(it-uniq.begin());
      }
    }
  });
  const auto p21=clock::now();

  H.directedContrib=H.slot.size();
  H.rowMean=H.n?(double)H.col.size()/(double)H.n:0.0;

  std::uint64_t targetTotal=0;
  std::uint32_t targetMax=0,rowMax=0;
  for(int t=0;t<nth;++t){
    targetTotal+=targetThread[(std::size_t)t];
    targetMax=std::max(targetMax,targetMaxThread[(std::size_t)t]);
    rowMax=std::max(rowMax,rowMaxThread[(std::size_t)t]);
  }
  H.rowMax=rowMax;

  const double scratchMiB=
    (double)(
      rowCount.size()*sizeof(std::int32_t)+
      contribCount.size()*sizeof(std::int32_t)+
      targetThread.size()*sizeof(std::uint64_t)+
      targetMaxThread.size()*sizeof(std::uint32_t)+
      rowMaxThread.size()*sizeof(std::uint32_t)
    )/(1024.0*1024.0);

  std::vector<std::int32_t>().swap(rowCount);
  std::vector<std::int32_t>().swap(contribCount);
  std::vector<std::uint64_t>().swap(targetThread);
  std::vector<std::uint32_t>().swap(targetMaxThread);
  std::vector<std::uint32_t>().swap(rowMaxThread);

  const auto all1=clock::now();

  std::printf(
    "NODALS_GPU_P5C_FINECSR_NEW status=PASS rows=%d nnz=%zu directed=%llu "
    "targetTotal=%llu targetMax=%u rowMax=%u rowMean=%.6f threads=%d "
    "incidenceSeconds=%.6f pass1Seconds=%.6f prefixAllocSeconds=%.6f "
    "pass2Seconds=%.6f totalSeconds=%.6f scratchMiB=%.3f csrMiB=%.3f "
    "metadataMiB=%.3f totalMiB=%.3f rssMiB=%.3f hwmMiB=%.3f\n",
    H.n,H.col.size(),
    (unsigned long long)H.directedContrib,
    (unsigned long long)targetTotal,
    targetMax,H.rowMax,H.rowMean,nth,
    std::chrono::duration<double>(inc1-inc0).count(),
    std::chrono::duration<double>(p11-p10).count(),
    std::chrono::duration<double>(pr1-pr0).count(),
    std::chrono::duration<double>(p21-p20).count(),
    std::chrono::duration<double>(all1-all0).count(),
    scratchMiB,
    (double)H.csr_bytes()/(1024.0*1024.0),
    (double)H.metadata_bytes()/(1024.0*1024.0),
    (double)H.bytes()/(1024.0*1024.0),
    g4a_status_mib("VmRSS"),g4a_status_mib("VmHWM"));

  return H;
}

inline void p5c_validate_finecsr(
    const H2FineCSRHost&R,const H2FineCSRHost&N,
    double refSeconds,double newSeconds)
{
  const bool dims=(R.n==N.n&&R.nv==N.nv);
  const bool rowExact=dims&&R.row==N.row;
  const bool colExact=rowExact&&R.col==N.col;
  const bool diagExact=dims&&R.diagPos==N.diagPos;
  const bool incExact=dims&&R.incOff==N.incOff;
  const bool packedExact=incExact&&R.packed==N.packed;
  const bool contribExact=dims&&R.contribOff==N.contribOff;
  const bool slotExact=contribExact&&R.slot==N.slot;
  const bool statsExact=
    R.directedContrib==N.directedContrib &&
    R.rowMax==N.rowMax &&
    R.rowMean==N.rowMean;

  const bool pass=
    dims&&rowExact&&colExact&&diagExact&&incExact&&
    packedExact&&contribExact&&slotExact&&statsExact;

  std::printf(
    "NODALS_GPU_P5C_FINECSR_PARITY status=%s dimsExact=%d rowExact=%d "
    "colExact=%d diagPosExact=%d incOffExact=%d packedExact=%d "
    "contribOffExact=%d slotExact=%d statsExact=%d "
    "refRows=%d newRows=%d refNnz=%zu newNnz=%zu "
    "refDirected=%llu newDirected=%llu refRowMax=%u newRowMax=%u "
    "refSeconds=%.6f newSeconds=%.6f speedup=%.6f "
    "compareHoldsTwoTopologies=1 productionPeakMustUseModeParallel=1\n",
    pass?"PASS":"FAIL",
    (int)dims,(int)rowExact,(int)colExact,(int)diagExact,
    (int)incExact,(int)packedExact,(int)contribExact,(int)slotExact,
    (int)statsExact,R.n,N.n,R.col.size(),N.col.size(),
    (unsigned long long)R.directedContrib,
    (unsigned long long)N.directedContrib,
    R.rowMax,N.rowMax,
    refSeconds,newSeconds,refSeconds/std::max(newSeconds,1e-300));

  if(!pass)throw std::runtime_error("P5C fine CSR parity failed");
}

inline H2FineCSRHost build_h2_fine_csr_host(const G4SetupHost&S)
{
  const char*e=std::getenv("NODALS_FINECSR_MODE");
  const char*mode=(e&&*e)?e:"parallel";

  if(std::strcmp(mode,"reference")==0)
    return build_h2_fine_csr_host_reference(S);

  if(std::strcmp(mode,"parallel")==0)
    return build_h2_fine_csr_host_parallel(S);

  if(std::strcmp(mode,"compare")==0){
    const auto r0=std::chrono::steady_clock::now();
    auto R=build_h2_fine_csr_host_reference(S);
    const auto r1=std::chrono::steady_clock::now();

    const auto n0=std::chrono::steady_clock::now();
    auto N=build_h2_fine_csr_host_parallel(S);
    const auto n1=std::chrono::steady_clock::now();

    p5c_validate_finecsr(
      R,N,
      std::chrono::duration<double>(r1-r0).count(),
      std::chrono::duration<double>(n1-n0).count());
    return N;
  }

  throw std::runtime_error(
    "NODALS_FINECSR_MODE must be reference, compare, or parallel");
}


// One pressure-row thread owns all writes to its row: no atomics.  Geometry is
// recomputed from the compact cell plan each refresh, keeping refresh metadata
// to roughly one byte per directed basis contribution.
__global__ void h2_fine_csr_refresh_kernel(
    int nc,
    const G4CellPlanDevice* __restrict__ cells,
    const OperatorReal* __restrict__ rau,
    const std::int32_t* __restrict__ row,
    const std::int32_t* __restrict__ diagPos,
    const std::int32_t* __restrict__ incOff,
    const std::uint32_t* __restrict__ packed,
    const std::int32_t* __restrict__ contribOff,
    const std::uint8_t* __restrict__ slot,
    AMGReal* __restrict__ val,
    AMGReal* __restrict__ diag)
{
  int c=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(c>=nc)return;
  const int rb=row[c],re=row[c+1];
  for(int j=rb;j<re;++j)val[j]=AMGReal(0);

  const auto&cp=cells[c];
  int q=contribOff[c];
  for(int a=0;a<8;++a){
    const int g=cp.ref[a];
    if(g<0)continue;
    const OperatorReal rv=rau[g];
    const OperatorReal a0=g4_coeff_cell(cp,a,0);
    const OperatorReal a1=g4_coeff_cell(cp,a,1);
    const OperatorReal a2=g4_coeff_cell(cp,a,2);
    for(int k=incOff[g];k<incOff[g+1];++k,++q){
      const std::uint32_t pk=packed[k];
      const int c2=(int)(pk>>3);
      const int b=(int)(pk&7u);
      const auto&cp2=cells[c2];
      const OperatorReal dot=
        a0*g4_coeff_cell(cp2,b,0)+
        a1*g4_coeff_cell(cp2,b,1)+
        a2*g4_coeff_cell(cp2,b,2);
      val[rb+(int)slot[q]] += AMGReal(rv*dot);
    }
  }
  diag[c]=val[diagPos[c]];
}

// H2B: warp-per-pressure-row numeric refresh.  Topology and compact uint8
// row-slot metadata are unchanged from H2.  Each warp owns exactly one CSR row.
// Row zeroing is cooperative; shared-basis contributions are spread over lanes.
// FP32 atomics are row-local and low-contention (H1 measured ~1.41 contributions
// per unique CSR nonzero on average).
__global__ void h2b_fine_csr_refresh_warp_kernel(
    int nc,
    const G4CellPlanDevice* __restrict__ cells,
    const OperatorReal* __restrict__ rau,
    const std::int32_t* __restrict__ row,
    const std::int32_t* __restrict__ diagPos,
    const std::int32_t* __restrict__ incOff,
    const std::uint32_t* __restrict__ packed,
    const std::int32_t* __restrict__ contribOff,
    const std::uint8_t* __restrict__ slot,
    AMGReal* __restrict__ val,
    AMGReal* __restrict__ diag)
{
  const int tid=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  const int c=tid>>5;
  const int lane=tid&31;
  if(c>=nc)return;

  const int rb=row[c], re=row[c+1];
  for(int j=rb+lane;j<re;j+=32)val[j]=AMGReal(0);
  __syncwarp();

  const auto&cp=cells[c];
  int qbase=contribOff[c];
  for(int a=0;a<8;++a){
    const int g=cp.ref[a];
    if(g<0)continue;
    const int ib=incOff[g], ie=incOff[g+1];
    const OperatorReal rv=rau[g];
    const OperatorReal a0=g4_coeff_cell(cp,a,0);
    const OperatorReal a1=g4_coeff_cell(cp,a,1);
    const OperatorReal a2=g4_coeff_cell(cp,a,2);
    for(int k=ib+lane;k<ie;k+=32){
      const std::uint32_t pk=packed[k];
      const int c2=(int)(pk>>3);
      const int b=(int)(pk&7u);
      const auto&cp2=cells[c2];
      const OperatorReal dot=
        a0*g4_coeff_cell(cp2,b,0)+
        a1*g4_coeff_cell(cp2,b,1)+
        a2*g4_coeff_cell(cp2,b,2);
      const int q=qbase+(k-ib);
      atomicAdd(&val[rb+(int)slot[q]],AMGReal(rv*dot));
    }
    qbase += (ie-ib);
  }
  __syncwarp();
  if(lane==0)diag[c]=val[diagPos[c]];
}

// Warp-per-pressure-row CSR SpMV.  H1 measured ~72 entries/row, making 2–3
// entries/lane typical on the development meshes.
__global__ void h2_fine_csr_spmv_warp_kernel(
    int n,
    const std::int32_t* __restrict__ row,
    const std::int32_t* __restrict__ col,
    const AMGReal* __restrict__ val,
    const AMGReal* __restrict__ x,
    AMGReal* __restrict__ y)
{
  const int tid=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  const int warp=tid>>5;
  const int lane=tid&31;
  if(warp>=n)return;
  AMGReal s=0;
  const int rb=row[warp],re=row[warp+1];
  for(int k=rb+lane;k<re;k+=32)s+=val[k]*x[col[k]];
  for(int off=16;off>0;off>>=1)s+=__shfl_down_sync(0xffffffffu,s,off);
  if(lane==0)y[warp]=s;
}

inline int h2_warp_grid(int n){
  constexpr int block=256;
  const long long threads=(long long)n*32ll;
  return (int)((threads+block-1)/block);
}

} // namespace nodals_gpu
