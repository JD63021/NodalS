#pragma once
#include "g5e_fp32_kernels.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace nodals_gpu {

constexpr int H7_BLOCK=256;
constexpr int H7_WARPS_PER_BLOCK=H7_BLOCK/32;

// Persistent warp-per-cell convection assembly.
// Each warp cooperatively evaluates one tetrahedral 8x8 local convection
// matrix at a time.  A block-local copy of the fixed 8x8x8x3 reference tensor
// avoids divergent constant-memory accesses between lanes.
__global__ void h7_convection_warp_cell_kernel(
    const G4CellPlanDevice* __restrict__ cells,int nc,
    const std::int64_t* __restrict__ row,
    const StateReal* __restrict__ fixed,
    const StateReal* __restrict__ u0,
    const StateReal* __restrict__ u1,
    const StateReal* __restrict__ u2,
    OperatorReal* __restrict__ aval,
    StateReal* __restrict__ cr0,
    StateReal* __restrict__ cr1,
    StateReal* __restrict__ cr2)
{
  __shared__ OperatorReal sT[8*8*8*3];
  __shared__ OperatorReal sUr[H7_WARPS_PER_BLOCK][8*3];

  const int tid=(int)threadIdx.x;
  for(int k=tid;k<8*8*8*3;k+=H7_BLOCK)
    sT[k]=g4_centT[k];
  __syncthreads();

  const int lane=tid&31;
  const int warpLocal=tid>>5;
  const int warpGlobal=(int)blockIdx.x*H7_WARPS_PER_BLOCK+warpLocal;
  const int warpStride=(int)gridDim.x*H7_WARPS_PER_BLOCK;

  for(int ic=warpGlobal;ic<nc;ic+=warpStride){
    const auto&cp=cells[ic];

    // Eight lanes each own one local velocity basis and form its three
    // transformed advecting-velocity components.
    if(lane<8){
      const int m=lane;
      const OperatorReal q0=OperatorReal(g4_ref_value(cp,m,0,u0,u1,u2,fixed));
      const OperatorReal q1=OperatorReal(g4_ref_value(cp,m,1,u0,u1,u2,fixed));
      const OperatorReal q2=OperatorReal(g4_ref_value(cp,m,2,u0,u1,u2,fixed));
      sUr[warpLocal][3*m+0]=q0*cp.invJ[0]+q1*cp.invJ[1]+q2*cp.invJ[2];
      sUr[warpLocal][3*m+1]=q0*cp.invJ[3]+q1*cp.invJ[4]+q2*cp.invJ[5];
      sUr[warpLocal][3*m+2]=q0*cp.invJ[6]+q1*cp.invJ[7]+q2*cp.invJ[8];
    }
    __syncwarp();

    // 64 (a,b) pairs: every lane evaluates two.
    for(int q=lane;q<64;q+=32){
      const int a=q>>3;
      const int b=q&7;
      const int r=cp.ref[a];
      if(r<0)continue;

      OperatorReal cv=0;
      #pragma unroll
      for(int m=0;m<8;++m){
        const OperatorReal ur0=sUr[warpLocal][3*m+0];
        const OperatorReal ur1=sUr[warpLocal][3*m+1];
        const OperatorReal ur2=sUr[warpLocal][3*m+2];
        const int t0=(((a*8+m)*8+b)*3);
        cv += ur0*sT[t0+0] + ur1*sT[t0+1] + ur2*sT[t0+2];
      }
      cv*=cp.det;

      if(cp.ref[b]>=0){
        const unsigned char sl=cp.rowSlot[q];
        if(sl!=255)atomicAdd(aval+row[r]+sl,cv);
      }else{
        const int fs=-cp.ref[b]-1;
        const StateReal z=StateReal(cv);
        atomicAdd(cr0+r,-z*fixed[3*(std::size_t)fs+0]);
        atomicAdd(cr1+r,-z*fixed[3*(std::size_t)fs+1]);
        atomicAdd(cr2+r,-z*fixed[3*(std::size_t)fs+2]);
      }
    }
    __syncwarp();
  }
}

// Persistent warp-per-momentum-row row-L1/relaxation finalizer.
__global__ void h7_finalize_relax_warp_row_kernel(
    int n,const std::int64_t* __restrict__ rp,
    const std::int32_t* __restrict__ diagPos,
    OperatorReal alpha,OperatorReal* __restrict__ av,
    OperatorReal* __restrict__ delta,
    OperatorReal* __restrict__ diag,
    OperatorReal* __restrict__ rau)
{
  const int tid=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  const int lane=tid&31;
  const int warpGlobal=tid>>5;
  const int warpStride=((int)gridDim.x*(int)blockDim.x)>>5;

  for(int i=warpGlobal;i<n;i+=warpStride){
    OperatorReal m=0;
    for(std::int64_t k=rp[i]+lane;k<rp[i+1];k+=32)
      m += fabsf(av[k]);

    for(int off=16;off>0;off>>=1)
      m += __shfl_down_sync(0xffffffffu,m,off);

    if(lane==0){
      const OperatorReal de=(OperatorReal(1)/alpha-OperatorReal(1))*m;
      const int dp=diagPos[i];
      delta[i]=de;
      av[dp]+=de;
      diag[i]=av[dp];
      rau[i]=OperatorReal(1)/av[dp];
    }
  }
}

template<class T>
__global__ void h7_diff_kernel(std::size_t n,const T*a,const T*b,T*d)
{
  const std::size_t i=(std::size_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n)d[i]=a[i]-b[i];
}

inline int h7_blocks_for_sm_factor(int smCount,int factor)
{
  return smCount*factor;
}

} // namespace nodals_gpu
