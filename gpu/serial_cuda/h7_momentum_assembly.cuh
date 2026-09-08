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

// Positive-weight 4x4x4 collapsed rule used by the CPU fast SUPG path.
__device__ __constant__ double h7_supg_rn[4]={0.0485005494469972764,0.238600737551862341,0.517047295104367421,0.795851417896772828};
__device__ __constant__ double h7_supg_rw[4]={0.110888415611277741,0.143458789799214448,0.0686338871729230970,0.0103522407499180812};
__device__ __constant__ double h7_supg_sn[4]={0.0571041961145177246,0.276843013638123803,0.583590432368916834,0.860240135656219485};
__device__ __constant__ double h7_supg_sw[4]={0.135506913431488518,0.203464568010271102,0.129847547608232333,0.0311809709500080849};
__device__ __constant__ double h7_supg_tn[4]={0.0694318442029737137,0.330009478207571871,0.669990521792428129,0.930568155797026231};
__device__ __constant__ double h7_supg_tw[4]={0.173927422568726897,0.326072577431273103,0.326072577431273103,0.173927422568726897};

__device__ __forceinline__ OperatorReal h7_gl(const G4CellPlanDevice&cp,int i,int d)
{
  if(i==0)return -(cp.invJ[d]+cp.invJ[3+d]+cp.invJ[6+d]);
  return cp.invJ[3*(i-1)+d];
}

// Exact implicit P1+BF3 SUPG action matching the CPU fast/64-point formulation.
// One warp owns a cell. Each lane owns two (test,trial) matrix pairs and
// accumulates across all 64 quadrature points, leaving one global atomic per
// local matrix pair rather than one atomic per quadrature point.
__global__ void h7_supg64_warp_cell_kernel(
    const G4CellPlanDevice* __restrict__ cells,int nc,
    const std::int64_t* __restrict__ row,
    const StateReal* __restrict__ fixed,
    const StateReal* __restrict__ u0,
    const StateReal* __restrict__ u1,
    const StateReal* __restrict__ u2,
    OperatorReal nu,OperatorReal tauScale,OperatorReal supgMagic,
    OperatorReal* __restrict__ aval,
    StateReal* __restrict__ cr0,
    StateReal* __restrict__ cr1,
    StateReal* __restrict__ cr2)
{
  __shared__ OperatorReal sCoeff[H7_WARPS_PER_BLOCK][8*3];
  __shared__ OperatorReal sStream[H7_WARPS_PER_BLOCK][8];
  __shared__ OperatorReal sStrong[H7_WARPS_PER_BLOCK][8];
  __shared__ OperatorReal sWeight[H7_WARPS_PER_BLOCK];

  const int tid=(int)threadIdx.x;
  const int lane=tid&31;
  const int warpLocal=tid>>5;
  const int warpGlobal=(int)blockIdx.x*H7_WARPS_PER_BLOCK+warpLocal;
  const int warpStride=(int)gridDim.x*H7_WARPS_PER_BLOCK;

  for(int ic=warpGlobal;ic<nc;ic+=warpStride){
    const auto&cp=cells[ic];

    if(lane<8){
      const int m=lane;
      sCoeff[warpLocal][3*m+0]=OperatorReal(g4_ref_value(cp,m,0,u0,u1,u2,fixed));
      sCoeff[warpLocal][3*m+1]=OperatorReal(g4_ref_value(cp,m,1,u0,u1,u2,fixed));
      sCoeff[warpLocal][3*m+2]=OperatorReal(g4_ref_value(cp,m,2,u0,u1,u2,fixed));
    }
    __syncwarp();

    OperatorReal acc0=0,acc1=0;
    for(int iq=0;iq<64;++iq){
      const int ir=iq>>4,is=(iq>>2)&3,it=iq&3;
      const OperatorReal r=OperatorReal(h7_supg_rn[ir]);
      const OperatorReal ss=OperatorReal(h7_supg_sn[is]);
      const OperatorReal tt=OperatorReal(h7_supg_tn[it]);
      const OperatorReal omr=OperatorReal(1)-r,oms=OperatorReal(1)-ss;
      const OperatorReal lam[4]={omr*oms*(OperatorReal(1)-tt),r,omr*ss,omr*oms*tt};

      OperatorReal phi=0;
      if(lane<4)phi=lam[lane];
      else if(lane<8){
        const int i=lane-4;
        int js[3],kk=0;for(int j=0;j<4;++j)if(j!=i)js[kk++]=j;
        phi=OperatorReal(27)*lam[js[0]]*lam[js[1]]*lam[js[2]];
      }

      OperatorReal a0=(lane<8)?phi*sCoeff[warpLocal][3*lane+0]:OperatorReal(0);
      OperatorReal a1=(lane<8)?phi*sCoeff[warpLocal][3*lane+1]:OperatorReal(0);
      OperatorReal a2=(lane<8)?phi*sCoeff[warpLocal][3*lane+2]:OperatorReal(0);
      for(int off=16;off>0;off>>=1){
        a0+=__shfl_down_sync(0xffffffffu,a0,off);
        a1+=__shfl_down_sync(0xffffffffu,a1,off);
        a2+=__shfl_down_sync(0xffffffffu,a2,off);
      }
      const OperatorReal adv0=__shfl_sync(0xffffffffu,a0,0);
      const OperatorReal adv1=__shfl_sync(0xffffffffu,a1,0);
      const OperatorReal adv2=__shfl_sync(0xffffffffu,a2,0);

      if(lane<8){
        const int a=lane;
        OperatorReal gx=0,gy=0,gz=0,lap=0;
        if(a<4){
          gx=h7_gl(cp,a,0);gy=h7_gl(cp,a,1);gz=h7_gl(cp,a,2);
        }else{
          const int i=a-4;
          int js[3],kk=0;for(int j=0;j<4;++j)if(j!=i)js[kk++]=j;
          gx=OperatorReal(27)*(lam[js[1]]*lam[js[2]]*h7_gl(cp,js[0],0)+lam[js[0]]*lam[js[2]]*h7_gl(cp,js[1],0)+lam[js[0]]*lam[js[1]]*h7_gl(cp,js[2],0));
          gy=OperatorReal(27)*(lam[js[1]]*lam[js[2]]*h7_gl(cp,js[0],1)+lam[js[0]]*lam[js[2]]*h7_gl(cp,js[1],1)+lam[js[0]]*lam[js[1]]*h7_gl(cp,js[2],1));
          gz=OperatorReal(27)*(lam[js[1]]*lam[js[2]]*h7_gl(cp,js[0],2)+lam[js[0]]*lam[js[2]]*h7_gl(cp,js[1],2)+lam[js[0]]*lam[js[1]]*h7_gl(cp,js[2],2));
          OperatorReal d01=0,d02=0,d12=0;
          for(int d=0;d<3;++d){
            const OperatorReal g0=h7_gl(cp,js[0],d),g1=h7_gl(cp,js[1],d),g2=h7_gl(cp,js[2],d);
            d01+=g0*g1;d02+=g0*g2;d12+=g1*g2;
          }
          lap=OperatorReal(54)*(lam[js[2]]*d01+lam[js[1]]*d02+lam[js[0]]*d12);
        }
        const OperatorReal st=adv0*gx+adv1*gy+adv2*gz;
        sStream[warpLocal][a]=st;
        sStrong[warpLocal][a]=(a<4)?st:(st-nu*lap);
      }

      if(lane==0){
        const OperatorReal speed2=adv0*adv0+adv1*adv1+adv2*adv2;
        const OperatorReal diff=OperatorReal(4)*nu/cp.h2;
        OperatorReal den=OperatorReal(4)*speed2/cp.h2+supgMagic*diff*diff;
        if(den<OperatorReal(1e-30))den=OperatorReal(1e-30);
        const OperatorReal tau=tauScale/sqrt(den);
        const OperatorReal qw=OperatorReal(h7_supg_rw[ir]*h7_supg_sw[is]*h7_supg_tw[it]);
        sWeight[warpLocal]=tau*qw*cp.det;
      }
      __syncwarp();

      const int p0=lane,a_0=p0>>3,b_0=p0&7;
      acc0+=sWeight[warpLocal]*sStream[warpLocal][a_0]*sStrong[warpLocal][b_0];
      const int p1=lane+32,a_1=p1>>3,b_1=p1&7;
      acc1+=sWeight[warpLocal]*sStream[warpLocal][a_1]*sStrong[warpLocal][b_1];
      __syncwarp();
    }

    for(int which=0;which<2;++which){
      const int p=lane+32*which,a=p>>3,b=p&7;
      const int r=cp.ref[a];if(r<0)continue;
      const OperatorReal sv=which?acc1:acc0;
      if(cp.ref[b]>=0){
        const unsigned char sl=cp.rowSlot[p];
        if(sl!=255)atomicAdd(aval+row[r]+sl,sv);
      }else{
        const int fs=-cp.ref[b]-1;const StateReal z=StateReal(sv);
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
      m += fabs(av[k]);

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
