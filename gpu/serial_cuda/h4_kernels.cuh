#pragma once
#include "g5e_fp32_kernels.cuh"
#include <cuda_runtime.h>

namespace nodals_gpu {

// H4: diffusion numeric CSR is persistent.  This hot kernel updates only
// convection numerics and fixed-Dirichlet convection RHS.
// The same scalar momentum matrix is shared by x/y/z.
__global__ void h4_assemble_convection_only_kernel(
    const G4CellPlanDevice* cells,int nc,const std::int64_t* row,
    const StateReal* fixed,const StateReal* u0,const StateReal* u1,
    const StateReal* u2,OperatorReal* aval,
    StateReal* cr0,StateReal* cr1,StateReal* cr2)
{
  int ic=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(ic>=nc)return;
  const auto&cp=cells[ic];

  OperatorReal I[3][3];
  for(int j=0;j<3;++j)for(int d=0;d<3;++d)I[j][d]=cp.invJ[3*j+d];

  StateReal cf[3][8];
  for(int m=0;m<8;++m)
    for(int d=0;d<3;++d)
      cf[d][m]=g4_ref_value(cp,m,d,u0,u1,u2,fixed);

  OperatorReal ur[8][3]={{0}};
  for(int m=0;m<8;++m)
    for(int j=0;j<3;++j)
      for(int d=0;d<3;++d)
        ur[m][j]+=OperatorReal(cf[d][m])*I[j][d];

  for(int a=0;a<8;++a){
    int r=cp.ref[a];
    if(r<0)continue;
    for(int b=0;b<8;++b){
      OperatorReal cv=0;
      for(int m=0;m<8;++m)
        for(int j=0;j<3;++j)
          cv+=ur[m][j]*g4_centT[(((a*8+m)*8+b)*3+j)];
      cv*=cp.det;

      if(cp.ref[b]>=0){
        unsigned char sl=cp.rowSlot[8*a+b];
        if(sl!=255)atomicAdd(aval+row[r]+sl,cv);
      }else{
        int fs=-cp.ref[b]-1;
        StateReal q=StateReal(cv);
        atomicAdd(cr0+r,-q*fixed[3*(std::size_t)fs]);
        atomicAdd(cr1+r,-q*fixed[3*(std::size_t)fs+1]);
        atomicAdd(cr2+r,-q*fixed[3*(std::size_t)fs+2]);
      }
    }
  }
}

} // namespace nodals_gpu
