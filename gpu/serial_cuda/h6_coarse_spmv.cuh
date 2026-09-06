#pragma once
#include "precision.hpp"
#include <cuda_runtime.h>
#include <cstdint>

namespace nodals_gpu {

// One warp owns one explicit AMG CSR row.
// The current scalar reference uses one CUDA thread per row.
__global__ void h6_coarse_csr_spmv_warp_kernel(
    int n,
    const std::int64_t* __restrict__ row,
    const std::int32_t* __restrict__ col,
    const AMGReal* __restrict__ val,
    const AMGReal* __restrict__ x,
    AMGReal* __restrict__ y)
{
  const int tid=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  const int i=tid>>5;
  const int lane=tid&31;
  if(i>=n)return;

  AMGReal s=AMGReal(0);
  const std::int64_t rb=row[i], re=row[i+1];
  for(std::int64_t k=rb+lane;k<re;k+=32)
    s += val[k]*x[col[k]];

  for(int off=16;off>0;off>>=1)
    s += __shfl_down_sync(0xffffffffu,s,off);

  if(lane==0)y[i]=s;
}

__global__ void h6_amg_diff_kernel(
    int n,const AMGReal* a,const AMGReal* b,AMGReal* d)
{
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i<n)d[i]=a[i]-b[i];
}

inline int h6_warp_grid(int n)
{
  constexpr int block=256;
  const long long threads=(long long)n*32ll;
  return (int)((threads+block-1)/block);
}

} // namespace nodals_gpu
