// Stage 6.C.1: READ-ONLY sparsity census of finalized component momentum CSR.
// No matrix storage, numerical path, or production kernel is changed.
#pragma once
namespace stage6c1 {
using Real=hxt4b::Real;
// Counter semantics: [0..2] exact XY/XZ/YZ nnz, [3] union exact nnz,
// [4] rows with any exact difference, [5] changed diagonal entries,
// [6] changed off-diagonal entries, [7] approximate union nnz,
// [8] approximate changed off-diagonal nnz, [9] changed XY or XZ nnz,
// [10] rows with approximate difference.
__global__ void census(int nv,const std::int64_t*row,const std::int32_t*diag,
    const Real*x,const Real*y,const Real*z,unsigned long long*counts){
  const int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i>=nv)return;
  unsigned long long xy=0,xz=0,yz=0,any=0,dc=0,off=0,ap=0,apoff=0,base=0;
  const std::int64_t begin=row[i],end=row[i+1];
  for(std::int64_t j=begin;j<end;++j){
    const Real xx=x[j],yy=y[j],zz=z[j];
    const bool a=xx!=yy,b=xx!=zz,c=yy!=zz;
    xy+=(unsigned long long)a;xz+=(unsigned long long)b;yz+=(unsigned long long)c;
    const bool altered=a||b||c;
    any+=(unsigned long long)altered;base+=(unsigned long long)(a||b);
    if(altered){if(j==(std::int64_t)diag[i])++dc;else ++off;}
    // Diagnostic threshold only; exact differences above are authoritative.
    const double scale=fmax(1.,fmax(fabs((double)xx),fmax(fabs((double)yy),fabs((double)zz))));
    const bool approx=fabs((double)(xx-yy))>1.e-12*scale ||
                      fabs((double)(xx-zz))>1.e-12*scale ||
                      fabs((double)(yy-zz))>1.e-12*scale;
    if(approx){++ap;if(j!=(std::int64_t)diag[i])++apoff;}
  }
  const unsigned long long loc[11]={xy,xz,yz,any,(unsigned long long)(any>0),
      dc,off,ap,apoff,base,(unsigned long long)(ap>0)};
  for(int k=0;k<11;++k)if(loc[k])atomicAdd(counts+k,loc[k]);
}
} // namespace stage6c1
