#pragma once
// H2B derivative of H2 exact-fine-CSR support.
#include "cuda_runtime.hpp"
#include "precision.hpp"
#include "g4_host.hpp"
#include "g5e_fp32_kernels.cuh"
#include <cuda_runtime.h>
#include <algorithm>
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

inline H2FineCSRHost build_h2_fine_csr_host(const G4SetupHost&S){
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
