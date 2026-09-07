#pragma once
#include "precision.hpp"
#include <cuda_runtime.h>
#include <cstdint>

namespace nodals_gpu {

__global__ void g9_fine_l1_warp_kernel(
    int n,
    const std::int32_t* __restrict__ row,
    const AMGReal* __restrict__ val,
    AMGReal* __restrict__ l1)
{
  const int tid=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  const int i=tid>>5;
  const int lane=tid&31;
  if(i>=n)return;

  AMGReal s=AMGReal(0);
  for(int k=row[i]+lane;k<row[i+1];k+=32)
    s += (val[k] >= AMGReal(0)) ? val[k] : -val[k];

  for(int off=16;off>0;off>>=1)
    s += __shfl_down_sync(0xffffffffu,s,off);

  if(lane==0)l1[i]=s;
}

} // namespace nodals_gpu
