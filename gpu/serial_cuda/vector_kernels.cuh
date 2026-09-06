#pragma once
#include "cuda_runtime.hpp"
#include "precision.hpp"
#include <cuda_runtime.h>
#include <cstddef>
#include <cmath>

namespace nodals_gpu {

constexpr int kBlock = 256;
inline int grid_for(std::size_t n) { return static_cast<int>((n + kBlock - 1) / kBlock); }

template<class T> __global__ void fill_kernel(T* x, std::size_t n, T a) { const std::size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]=a; }
template<class T> __global__ void scale_kernel(T* x, std::size_t n, T a) { const std::size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]*=a; }
template<class T> __global__ void axpy_kernel(T* y, const T* x, std::size_t n, T a) { const std::size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]=a*x[i]+y[i]; }

template<class T> __global__ void fused_axpy3_kernel(T* y0,T* y1,T* y2,const T* x0,const T* x1,const T* x2,std::size_t n,T a) {
  const std::size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) { y0[i]=a*x0[i]+y0[i]; y1[i]=a*x1[i]+y1[i]; y2[i]=a*x2[i]+y2[i]; }
}

template<class T> __global__ void dot_kernel(const T* x,const T* y,std::size_t n,double* out) {
  __shared__ double s[kBlock];
  const std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;
  double v=(i<n)?static_cast<double>(x[i])*static_cast<double>(y[i]):0.0;
  s[threadIdx.x]=v; __syncthreads();
  for(int stride=kBlock/2; stride>0; stride>>=1) { if(threadIdx.x<stride) s[threadIdx.x]+=s[threadIdx.x+stride]; __syncthreads(); }
  if(threadIdx.x==0) atomicAdd(out,s[0]);
}

template<class T> inline void device_fill(T* x,std::size_t n,T a) { if(n) fill_kernel<<<grid_for(n),kBlock>>>(x,n,a); NODALS_CUDA(cudaGetLastError()); }
template<class T> inline void device_scale(T* x,std::size_t n,T a) { if(n) scale_kernel<<<grid_for(n),kBlock>>>(x,n,a); NODALS_CUDA(cudaGetLastError()); }
template<class T> inline void device_axpy(T* y,const T* x,std::size_t n,T a) { if(n) axpy_kernel<<<grid_for(n),kBlock>>>(y,x,n,a); NODALS_CUDA(cudaGetLastError()); }
template<class T> inline void device_fused_axpy3(T* y0,T* y1,T* y2,const T* x0,const T* x1,const T* x2,std::size_t n,T a) { if(n) fused_axpy3_kernel<<<grid_for(n),kBlock>>>(y0,y1,y2,x0,x1,x2,n,a); NODALS_CUDA(cudaGetLastError()); }

template<class T> inline double device_dot(const T* x,const T* y,std::size_t n,DeviceBuffer<double>& scratch) {
  if(scratch.size()!=1) scratch.allocate(1); NODALS_CUDA(cudaMemset(scratch.data(),0,sizeof(double)));
  if(n) dot_kernel<<<grid_for(n),kBlock>>>(x,y,n,scratch.data()); NODALS_CUDA(cudaGetLastError());
  double h=0.0; scratch.download(&h,1); return h;
}
template<class T> inline double device_norm2(const T* x,std::size_t n,DeviceBuffer<double>& scratch) { return std::sqrt(device_dot(x,x,n,scratch)); }

} // namespace nodals_gpu
