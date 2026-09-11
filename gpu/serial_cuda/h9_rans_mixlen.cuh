#pragma once
#include "h7_momentum_assembly.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace nodals_gpu {

// Gate 1 RANS port: lagged Nikuradse smooth-pipe mixing length only.
//
// This matches the promoted CPU RANS operator:
//   ell/R = max(0, 0.14 - 0.08 eta^2 - 0.06 eta^4), eta=r/R
//   nu_t  = scale * ell^2 * sqrt(2 S:S)
// and adds only
//   int_K nu_t grad(phi_a).grad(phi_b) dV
// to the shared scalar momentum operator.  No turbulence unknowns are added.
// The 64-point positive collapsed rule is the same rule used by the CPU
// dynamic plan when supg_quad_points=64 (the promoted default).

__device__ __forceinline__ OperatorReal h9_basis_grad(
    const G4CellPlanDevice&cp,int a,const OperatorReal lam[4],int d)
{
  if(a<4)return h7_gl(cp,a,d);
  const int i=a-4;
  int js[3],kk=0;
  #pragma unroll
  for(int j=0;j<4;++j)if(j!=i)js[kk++]=j;
  return OperatorReal(27)*(
      lam[js[1]]*lam[js[2]]*h7_gl(cp,js[0],d)
     +lam[js[0]]*lam[js[2]]*h7_gl(cp,js[1],d)
     +lam[js[0]]*lam[js[1]]*h7_gl(cp,js[2],d));
}

__device__ __forceinline__ void h9_atomic_max_nonnegative(double*addr,double v)
{
  auto* p=reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old=*p,assumed;
  do{
    assumed=old;
    if(__longlong_as_double((long long)assumed)>=v)break;
    old=atomicCAS(p,assumed,(unsigned long long)__double_as_longlong(v));
  }while(assumed!=old);
}

// One warp owns one tetrahedron.  Lanes 0..7 build basis gradients, lanes
// 0..8 form the nine entries of grad(U), and all 32 lanes own two local 8x8
// matrix entries.  This keeps the full nonlinear update on the device.
__global__ void h9_nikuradse_mixlen64_warp_cell_kernel(
    const G4CellPlanDevice* __restrict__ cells,int nc,
    const OperatorReal* __restrict__ xy,
    const std::int64_t* __restrict__ row,
    const StateReal* __restrict__ fixed,
    const StateReal* __restrict__ u0,
    const StateReal* __restrict__ u1,
    const StateReal* __restrict__ u2,
    OperatorReal cx,OperatorReal cy,OperatorReal radius,
    OperatorReal nu,OperatorReal scale,
    OperatorReal* __restrict__ aval,
    StateReal* __restrict__ cr0,
    StateReal* __restrict__ cr1,
    StateReal* __restrict__ cr2,
    double* __restrict__ stats)
{
  __shared__ OperatorReal sCoeff[H7_WARPS_PER_BLOCK][8*3];
  __shared__ OperatorReal sGrad[H7_WARPS_PER_BLOCK][8*3];
  __shared__ OperatorReal sGradU[H7_WARPS_PER_BLOCK][9];
  __shared__ OperatorReal sNuT[H7_WARPS_PER_BLOCK];
  __shared__ OperatorReal sQW[H7_WARPS_PER_BLOCK];

  const int tid=(int)threadIdx.x;
  const int lane=tid&31;
  const int warpLocal=tid>>5;
  const int warpGlobal=(int)blockIdx.x*H7_WARPS_PER_BLOCK+warpLocal;
  const int warpStride=(int)gridDim.x*H7_WARPS_PER_BLOCK;

  for(int ic=warpGlobal;ic<nc;ic+=warpStride){
    const auto&cp=cells[ic];

    if(lane<8){
      const int a=lane;
      sCoeff[warpLocal][3*a+0]=OperatorReal(g4_ref_value(cp,a,0,u0,u1,u2,fixed));
      sCoeff[warpLocal][3*a+1]=OperatorReal(g4_ref_value(cp,a,1,u0,u1,u2,fixed));
      sCoeff[warpLocal][3*a+2]=OperatorReal(g4_ref_value(cp,a,2,u0,u1,u2,fixed));
    }
    __syncwarp();

    OperatorReal acc0=0,acc1=0;
    double cellRatioWeighted=0.0,cellWeight=0.0,cellMaxRatio=0.0,cellMaxStrain=0.0;

    #pragma unroll 1
    for(int iq=0;iq<64;++iq){
      const int ir=iq>>4,is=(iq>>2)&3,it=iq&3;
      const OperatorReal r=OperatorReal(h7_supg_rn[ir]);
      const OperatorReal ss=OperatorReal(h7_supg_sn[is]);
      const OperatorReal tt=OperatorReal(h7_supg_tn[it]);
      const OperatorReal omr=OperatorReal(1)-r,oms=OperatorReal(1)-ss;
      const OperatorReal lam[4]={
        omr*oms*(OperatorReal(1)-tt),r,omr*ss,omr*oms*tt};

      if(lane<8){
        const int a=lane;
        sGrad[warpLocal][3*a+0]=h9_basis_grad(cp,a,lam,0);
        sGrad[warpLocal][3*a+1]=h9_basis_grad(cp,a,lam,1);
        sGrad[warpLocal][3*a+2]=h9_basis_grad(cp,a,lam,2);
      }
      __syncwarp();

      if(lane<9){
        const int comp=lane/3,dir=lane%3;
        OperatorReal z=0;
        #pragma unroll
        for(int a=0;a<8;++a)
          z+=sCoeff[warpLocal][3*a+comp]*sGrad[warpLocal][3*a+dir];
        sGradU[warpLocal][lane]=z;
      }
      __syncwarp();

      if(lane==0){
        OperatorReal ssRaw=0;
        #pragma unroll
        for(int i=0;i<3;++i){
          #pragma unroll
          for(int j=0;j<3;++j){
            const OperatorReal sij=OperatorReal(0.5)*(
              sGradU[warpLocal][3*i+j]+sGradU[warpLocal][3*j+i]);
            ssRaw+=sij*sij;
          }
        }
        const OperatorReal strain=sqrt(fmax(OperatorReal(0),OperatorReal(2)*ssRaw));
        const OperatorReal* cxy=xy+(std::size_t)ic*8;
        OperatorReal xq=0,yq=0;
        #pragma unroll
        for(int i=0;i<4;++i){xq+=lam[i]*cxy[2*i+0];yq+=lam[i]*cxy[2*i+1];}
        const OperatorReal dx=xq-cx,dy=yq-cy;
        OperatorReal eta=sqrt(dx*dx+dy*dy)/radius;
        eta=fmin(OperatorReal(1),fmax(OperatorReal(0),eta));
        const OperatorReal eta2=eta*eta,eta4=eta2*eta2;
        const OperatorReal ell=radius*fmax(
          OperatorReal(0),OperatorReal(0.14)-OperatorReal(0.08)*eta2-OperatorReal(0.06)*eta4);
        const OperatorReal nut=scale*ell*ell*strain;
        const OperatorReal qw=OperatorReal(h7_supg_rw[ir]*h7_supg_sw[is]*h7_supg_tw[it])*cp.det;
        sNuT[warpLocal]=nut;
        sQW[warpLocal]=qw;

        const double w=(double)qw;
        const double ratio=(double)(nut/nu);
        cellRatioWeighted+=ratio*w;
        cellWeight+=w;
        cellMaxRatio=fmax(cellMaxRatio,ratio);
        cellMaxStrain=fmax(cellMaxStrain,(double)strain);
      }
      __syncwarp();

      const OperatorReal nutw=sNuT[warpLocal]*sQW[warpLocal];
      const int p0=lane,a0=p0>>3,b0=p0&7;
      const OperatorReal gd0=
          sGrad[warpLocal][3*a0+0]*sGrad[warpLocal][3*b0+0]
         +sGrad[warpLocal][3*a0+1]*sGrad[warpLocal][3*b0+1]
         +sGrad[warpLocal][3*a0+2]*sGrad[warpLocal][3*b0+2];
      acc0+=nutw*gd0;

      const int p1=lane+32,a1=p1>>3,b1=p1&7;
      const OperatorReal gd1=
          sGrad[warpLocal][3*a1+0]*sGrad[warpLocal][3*b1+0]
         +sGrad[warpLocal][3*a1+1]*sGrad[warpLocal][3*b1+1]
         +sGrad[warpLocal][3*a1+2]*sGrad[warpLocal][3*b1+2];
      acc1+=nutw*gd1;
      __syncwarp();
    }

    for(int which=0;which<2;++which){
      const int p=lane+32*which,a=p>>3,b=p&7;
      const int rr=cp.ref[a];
      if(rr<0)continue;
      const OperatorReal kv=which?acc1:acc0;
      if(cp.ref[b]>=0){
        const unsigned char sl=cp.rowSlot[p];
        if(sl!=255)atomicAdd(aval+row[rr]+sl,kv);
      }else{
        const int fs=-cp.ref[b]-1;
        const StateReal z=StateReal(kv);
        atomicAdd(cr0+rr,-z*fixed[3*(std::size_t)fs+0]);
        atomicAdd(cr1+rr,-z*fixed[3*(std::size_t)fs+1]);
        atomicAdd(cr2+rr,-z*fixed[3*(std::size_t)fs+2]);
      }
    }

    if(lane==0 && stats){
      atomicAdd(stats+0,cellRatioWeighted);
      atomicAdd(stats+1,cellWeight);
      h9_atomic_max_nonnegative(stats+2,cellMaxRatio);
      h9_atomic_max_nonnegative(stats+3,cellMaxStrain);
    }
    __syncwarp();
  }
}

} // namespace nodals_gpu
