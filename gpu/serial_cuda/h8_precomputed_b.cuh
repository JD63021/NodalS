#pragma once
#include "g5e_fp32_kernels.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace nodals_gpu {

constexpr int H8_BCOEFF_PER_CELL = 8*3;

__device__ __forceinline__ OperatorReal h8_bc(
    const OperatorReal* __restrict__ bc,int c,int a,int d)
{
  return bc[((std::size_t)c*8u+(std::size_t)a)*3u+(std::size_t)d];
}

// One-time setup construction.  This is the exact existing g4_coeff_cell()
// evaluated once for every (cell, local velocity basis, component).
__global__ void h8_build_bcoeff_kernel(
    const G4CellPlanDevice* __restrict__ cells,
    int nc,
    OperatorReal* __restrict__ bc)
{
  const std::size_t q=(std::size_t)blockIdx.x*blockDim.x+threadIdx.x;
  const std::size_t n=(std::size_t)nc*8u*3u;
  if(q>=n)return;
  const int d=(int)(q%3u);
  const std::size_t t=q/3u;
  const int a=(int)(t%8u);
  const int c=(int)(t/8u);
  bc[q]=g4_coeff_cell(cells[c],a,d);
}

// H8 exact fine Schur CSR numeric refresh.
// Same H2B topology, slots, row-local atomics and rAU.  Only B geometry
// evaluation is replaced by loads from the persistent coefficient table.
__global__ void h8_fine_csr_refresh_warp_kernel(
    int nc,
    const G4CellPlanDevice* __restrict__ cells,
    const OperatorReal* __restrict__ bc,
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

  const int rb=row[c],re=row[c+1];
  for(int j=rb+lane;j<re;j+=32)val[j]=AMGReal(0);
  __syncwarp();

  const auto&cp=cells[c];
  int qbase=contribOff[c];

  for(int a=0;a<8;++a){
    const int g=cp.ref[a];
    if(g<0)continue;

    const int ib=incOff[g],ie=incOff[g+1];
    const OperatorReal rv=rau[g];
    const OperatorReal a0=h8_bc(bc,c,a,0);
    const OperatorReal a1=h8_bc(bc,c,a,1);
    const OperatorReal a2=h8_bc(bc,c,a,2);

    for(int k=ib+lane;k<ie;k+=32){
      const std::uint32_t pk=packed[k];
      const int c2=(int)(pk>>3);
      const int b=(int)(pk&7u);

      const OperatorReal dot=
        a0*h8_bc(bc,c2,b,0)+
        a1*h8_bc(bc,c2,b,1)+
        a2*h8_bc(bc,c2,b,2);

      const int q=qbase+(k-ib);
      atomicAdd(&val[rb+(int)slot[q]],AMGReal(rv*dot));
    }
    qbase+=(ie-ib);
  }

  __syncwarp();
  if(lane==0)diag[c]=val[diagPos[c]];
}

__global__ void h8_bt3_kernel(
    const G4CellPlanDevice* __restrict__ cells,
    const OperatorReal* __restrict__ bc,
    const StateReal* __restrict__ p,
    int nc,
    StateReal* __restrict__ v0,
    StateReal* __restrict__ v1,
    StateReal* __restrict__ v2)
{
  const int c=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(c>=nc)return;
  const auto&cp=cells[c];
  const StateReal pv=p[c];

  for(int a=0;a<8;++a){
    const int g=cp.ref[a];
    if(g<0)continue;
    atomicAdd(v0+g,StateReal(h8_bc(bc,c,a,0))*pv);
    atomicAdd(v1+g,StateReal(h8_bc(bc,c,a,1))*pv);
    atomicAdd(v2+g,StateReal(h8_bc(bc,c,a,2))*pv);
  }
}

__global__ void h8_b3_kernel(
    const G4CellPlanDevice* __restrict__ cells,
    const OperatorReal* __restrict__ bc,
    int nc,
    const StateReal* __restrict__ v0,
    const StateReal* __restrict__ v1,
    const StateReal* __restrict__ v2,
    StateReal* __restrict__ y)
{
  const int c=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(c>=nc)return;
  const auto&cp=cells[c];
  OperatorReal s=0;

  for(int a=0;a<8;++a){
    const int g=cp.ref[a];
    if(g<0)continue;
    s += h8_bc(bc,c,a,0)*v0[g]
       + h8_bc(bc,c,a,1)*v1[g]
       + h8_bc(bc,c,a,2)*v2[g];
  }
  y[c]=StateReal(s);
}

__global__ void h8_continuity_kernel(
    const G4CellPlanDevice* __restrict__ cells,
    const OperatorReal* __restrict__ bc,
    int nc,
    const StateReal* __restrict__ fixedDiv,
    const StateReal* __restrict__ u0,
    const StateReal* __restrict__ u1,
    const StateReal* __restrict__ u2,
    StateReal* __restrict__ r)
{
  const int c=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(c>=nc)return;
  const auto&cp=cells[c];
  OperatorReal s=fixedDiv[c];

  for(int a=0;a<8;++a){
    const int g=cp.ref[a];
    if(g<0)continue;
    s += h8_bc(bc,c,a,0)*u0[g]
       + h8_bc(bc,c,a,1)*u1[g]
       + h8_bc(bc,c,a,2)*u2[g];
  }
  r[c]=StateReal(s);
}

__global__ void h8_test_vector_kernel(int n,StateReal* x,int phase)
{
  const int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i<n){
    const float q=(float)(i+1);
    x[i]=sinf((0.000713f+0.000071f*phase)*q)
        + 0.31f*cosf((0.001117f+0.000053f*phase)*q);
  }
}

template<class T>
__global__ void h8_diff_kernel(std::size_t n,const T*a,const T*b,T*d)
{
  const std::size_t i=(std::size_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n)d[i]=a[i]-b[i];
}

} // namespace nodals_gpu
