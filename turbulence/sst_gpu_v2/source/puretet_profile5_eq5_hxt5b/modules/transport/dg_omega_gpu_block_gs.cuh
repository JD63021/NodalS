#pragma once
// Experimental DG-P1 omega: exact 4x4 element-block multi-color GS.
// This is *element block* GS, not scalar Jacobi. Every color is an independent
// set in the tetra adjacency graph. Different colors execute in serial order
// on CUDA's default stream; all rows within a color execute concurrently.
//
// FGS = 0 -> ncolors-1, SGS = 0 -> ncolors-1 -> 0.
// Caller controls relaxation (1.0 = standard block GS).
#include "dg_omega_gpu_fgmres.cuh"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace nodals {namespace omega_dg_gpu {
struct Coloring {
  std::vector<int> indices;
  std::vector<int> offsets;
  int ncolors=0;
};
// Deterministic greedy tetrahedral-dual graph coloring. A pure tetra mesh has
// <=4 geometric face neighbors, so this produces <=5 colors in the ordinary case.
inline Coloring color_bsr4(int ntet,const std::vector<int>&rowPtr,
                           const std::vector<int>&colInd){
  if(ntet<=0||(int)rowPtr.size()!=ntet+1||rowPtr[0]!=0||
     rowPtr[(std::size_t)ntet]!=(int)colInd.size())
    throw std::runtime_error("DG GS: bad BSR4 row pointers");
  std::vector<int>color((std::size_t)ntet,-1);
  int nc=0;
  for(int i=0;i<ntet;++i){
    bool used[5]={false,false,false,false,false};
    int degree=0;
    for(int j=rowPtr[(std::size_t)i];j<rowPtr[(std::size_t)i+1];++j){
      int k=colInd[(std::size_t)j];
      if(k<0||k>=ntet)throw std::runtime_error("DG GS: invalid block column");
      if(k==i)continue;
      if(++degree>4)throw std::runtime_error("DG GS: not a pure tetra face-neighbor BSR4 graph");
      if(color[(std::size_t)k]>=0){
        if(color[(std::size_t)k]>=5)throw std::runtime_error("DG GS: more than five colors");
        used[color[(std::size_t)k]]=true;
      }
    }
    int c=0;while(c<5&&used[c])++c;
    if(c>=5)throw std::runtime_error("DG GS: five-color limit exceeded");
    color[(std::size_t)i]=c;
    nc=std::max(nc,c+1);
  }
  for(int i=0;i<ntet;++i)
    for(int j=rowPtr[(std::size_t)i];j<rowPtr[(std::size_t)i+1];++j){
      int k=colInd[(std::size_t)j];
      if(k!=i&&color[(std::size_t)i]==color[(std::size_t)k])
        throw std::runtime_error("DG GS: invalid color conflict");
    }
  Coloring result;
  result.ncolors=nc;result.offsets.resize((std::size_t)nc+1,0);
  for(int c:color)++result.offsets[(std::size_t)c+1];
  for(int c=0;c<nc;++c)
    result.offsets[(std::size_t)c+1]+=result.offsets[(std::size_t)c];
  result.indices.resize((std::size_t)ntet);
  auto cursor=result.offsets;
  for(int i=0;i<ntet;++i)
    result.indices[(std::size_t)cursor[(std::size_t)color[(std::size_t)i]]++]=i;
  return result;
}
__global__ void block_gs_color(int count,const int*cellList,
                               const int*rowPtr,const int*colInd,
                               const int*diagonalSlots,const double*blocks,
                               const double*rhs,double*x,double relaxation,
                               unsigned int*errors){
  int q=blockIdx.x*blockDim.x+threadIdx.x;
  if(q>=count)return;
  const int t=cellList[q];
  double v[4]={rhs[4*t],rhs[4*t+1],rhs[4*t+2],rhs[4*t+3]};
  const int diagonal=diagonalSlots[t];
  for(int slot=rowPtr[t];slot<rowPtr[t+1];++slot){
    if(slot==diagonal)continue;
    const int nb=colInd[slot];
    for(int i=0;i<4;++i){
      double sub=0.;
      for(int j=0;j<4;++j)sub+=blocks[16*slot+4*i+j]*x[4*nb+j];
      v[i]-=sub;
    }
  }
  double M[4][5];
  for(int i=0;i<4;++i){
    for(int j=0;j<4;++j)M[i][j]=blocks[16*diagonal+4*i+j];
    M[i][4]=v[i];
  }
  // Pivoted 4x4 dense Gaussian elimination.
  for(int k=0;k<4;++k){
    int piv=k;
    for(int i=k+1;i<4;++i)
      if(fabs(M[i][k])>fabs(M[piv][k]))piv=i;
    if(!(fabs(M[piv][k])>1e-35)||!isfinite(M[piv][k])){
      atomicAdd(errors,1u);return;
    }
    if(piv!=k)for(int j=k;j<5;++j){
      const double s=M[k][j];M[k][j]=M[piv][j];M[piv][j]=s;
    }
    const double inv=1./M[k][k];
    for(int j=k;j<5;++j)M[k][j]*=inv;
    for(int i=0;i<4;++i)if(i!=k){
      const double factor=M[i][k];
      for(int j=k;j<5;++j)M[i][j]-=factor*M[k][j];
    }
  }
  for(int i=0;i<4;++i){
    const double updated=(1.-relaxation)*x[4*t+i]+relaxation*M[i][4];
    if(!isfinite(updated)){atomicAdd(errors,1u);return;}
    x[4*t+i]=updated;
  }
}
struct DeviceColoring {
  Coloring host;
  int*indices=nullptr;
  explicit DeviceColoring(Coloring c):host(std::move(c)){
    cuda_ok(cudaMalloc((void**)&indices,host.indices.size()*sizeof(int)),
            "DG GS cell color list allocation");
    cuda_ok(cudaMemcpy(indices,host.indices.data(),host.indices.size()*sizeof(int),
                       cudaMemcpyHostToDevice),"DG GS color upload");
  }
  DeviceColoring(const DeviceColoring&)=delete;
  DeviceColoring&operator=(const DeviceColoring&)=delete;
  ~DeviceColoring(){if(indices)cudaFree(indices);}
};
struct SweepResult {
  int sweeps=0,ncolors=0,launches=0;
};
inline SweepResult block_gs(const MatrixView&A,const DeviceColoring&colors,
                           const double*rhs,double*x,int sweeps,
                           bool symmetric,double relaxation=1.){
  if(sweeps<0||!std::isfinite(relaxation)||!(relaxation>0.)||
     !(relaxation<=1.5)||!rhs||!x)
    throw std::runtime_error("DG GS: invalid sweep options");
  if((int)colors.host.indices.size()!=A.ntet)
    throw std::runtime_error("DG GS: mismatched coloring and matrix");
  unsigned int*bad=nullptr;
  cuda_ok(cudaMalloc((void**)&bad,sizeof(unsigned int)),"DG GS errors allocate");
  cuda_ok(cudaMemset(bad,0,sizeof(unsigned int)),"DG GS errors clear");
  SweepResult result{sweeps,colors.host.ncolors,0};
  try {
    auto sweep_one=[&](int c) {
      const int first=colors.host.offsets[(std::size_t)c];
      const int count=colors.host.offsets[(std::size_t)c+1]-first;
      if(!count)return;
      block_gs_color<<<(count+127)/128,128>>>(
        count,colors.indices+first,A.rowPtr,A.colInd,A.diagonalSlots,
        A.blocks,rhs,x,relaxation,bad);
      cuda_ok(cudaGetLastError(),"DG GS color launch");
      ++result.launches;
    };
    for(int s=0;s<sweeps;++s){
      for(int c=0;c<colors.host.ncolors;++c)sweep_one(c);
      if(symmetric)for(int c=colors.host.ncolors-1;c>=0;--c)sweep_one(c);
    }
    cuda_ok(cudaDeviceSynchronize(),"DG GS synchronize");
    unsigned int err=0;
    cuda_ok(cudaMemcpy(&err,bad,sizeof(err),cudaMemcpyDeviceToHost),"DG GS error count");
    if(err)throw std::runtime_error("DG GS invalid local 4x4 blocks or update ("+std::to_string(err)+")");
  }catch(...){cudaFree(bad);throw;}
  cudaFree(bad);
  return result;
}
}} // namespace nodals::omega_dg_gpu
