#pragma once
// Experimental DG-P1 omega: GPU-resident right-preconditioned FGMRES(BSR4).
// All Krylov vectors remain in device memory; only reductions and the tiny
// Hessenberg/QR system are on the CPU. Not coupled to the NodalS G2 loop yet.
#include "dg_omega_bsr4.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace nodals {namespace omega_dg_gpu {
inline void cuda_ok(cudaError_t e,const char*where){
  if(e!=cudaSuccess)throw std::runtime_error(std::string("DG omega CUDA ")+where+": "+cudaGetErrorString(e));
}
inline void blas_ok(cublasStatus_t s,const char*where){
  if(s!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error(std::string("DG omega cuBLAS ")+where+" failed status "+std::to_string((int)s));
}
struct DeviceVector {
  double* p=nullptr;std::size_t n=0;
  explicit DeviceVector(std::size_t count):n(count){cuda_ok(cudaMalloc((void**)&p,n*sizeof(double)),"cudaMalloc vector");}
  ~DeviceVector(){if(p)cudaFree(p);}
  DeviceVector(const DeviceVector&)=delete;DeviceVector&operator=(const DeviceVector&)=delete;
  DeviceVector(DeviceVector&&x)noexcept:p(x.p),n(x.n){x.p=nullptr;x.n=0;}
  DeviceVector&operator=(DeviceVector&&)=delete;
  void zero(){cuda_ok(cudaMemset(p,0,n*sizeof(double)),"zero vector");}
  void copy_from(const double*src){cuda_ok(cudaMemcpy(p,src,n*sizeof(double),cudaMemcpyDeviceToDevice),"copy vector");}
};
struct MatrixView {
  int ntet=0;const int*rowPtr=nullptr,*colInd=nullptr,*diagonalSlots=nullptr;
  const double*blocks=nullptr;
};
struct FgmresOptions {
  int restart=20,maxIterations=240;
  double rtol=1e-7,atol=1e-10;
  int printEvery=20;
  bool reorthogonalize=true;
};
struct FgmresResult {
  bool converged=false;
  int iterations=0,restarts=0;
  double initialNorm=0,finalNorm=0,relativeResidual=0;
};
inline void axpy(cublasHandle_t h,int n,double a,const double*x,double*y){
  blas_ok(cublasDaxpy(h,n,&a,x,1,y,1),"Daxpy");
}
inline double norm(cublasHandle_t h,int n,const double*x){
  double value=0;blas_ok(cublasDnrm2(h,n,x,1,&value),"Dnrm2");return value;
}
inline double dotproduct(cublasHandle_t h,int n,const double*x,const double*y){
  double value=0;blas_ok(cublasDdot(h,n,x,1,y,1,&value),"Ddot");return value;
}
inline void scal(cublasHandle_t h,int n,double a,double*x){
  blas_ok(cublasDscal(h,n,&a,x,1),"Dscal");
}
inline void spmv(const MatrixView&A,const double*x,double*y) {
  const int n=4*A.ntet;
  bsr4_spmv<<<(n+127)/128,128>>>(A.ntet,A.rowPtr,A.colInd,A.blocks,x,y);
  cuda_ok(cudaGetLastError(),"BSR4 SpMV launch");
}
inline void inverse_blocks(const MatrixView&A,const double*r,double*z,unsigned int*bad){
  bsr4_block_jacobi<<<(A.ntet+127)/128,128>>>(
    A.ntet,A.blocks,A.diagonalSlots,r,z,bad);
  cuda_ok(cudaGetLastError(),"BSR4 Jacobi launch");
}
// True-residual FGMRES with optional second MGS pass. Does not assume SPD.
inline FgmresResult fgmres(const MatrixView&A,const double*b,double*x,
                          const FgmresOptions&opt) {
  if(A.ntet<=0||!A.rowPtr||!A.colInd||!A.diagonalSlots||!A.blocks||!b||!x)
    throw std::runtime_error("DG FGMRES invalid matrix/field pointers");
  if(opt.restart<2||opt.restart>120||opt.maxIterations<=0||
     !(opt.rtol>0.)||!(opt.atol>=0.))
    throw std::runtime_error("DG FGMRES invalid solver options");
  const int n=4*A.ntet,m=opt.restart;
  cublasHandle_t handle=nullptr;
  blas_ok(cublasCreate(&handle),"cublasCreate");
  struct HandleGuard{cublasHandle_t h;~HandleGuard(){if(h)cublasDestroy(h);}}guard{handle};
  DeviceVector residual((std::size_t)n),work((std::size_t)n);
  // Reserve stable vector objects because the vectors are move-only.
  std::vector<std::unique_ptr<DeviceVector>>V,Z;
  V.reserve((std::size_t)m+1);Z.reserve((std::size_t)m);
  for(int i=0;i<=m;++i)V.emplace_back(new DeviceVector((std::size_t)n));
  for(int i=0;i<m;++i)Z.emplace_back(new DeviceVector((std::size_t)n));
  unsigned int*bad=nullptr;
  cuda_ok(cudaMalloc((void**)&bad,sizeof(unsigned int)),"Jacobi error counter");
  struct BadGuard{unsigned int*p;~BadGuard(){if(p)cudaFree(p);}}bguard{bad};
  cuda_ok(cudaMemset(bad,0,sizeof(unsigned int)),"Jacobi zero error counter");
  const double normB=norm(handle,n,b);
  const double tolerance=std::max(opt.atol,opt.rtol*normB);
  FgmresResult result;
  auto true_residual=[&]()->double {
    spmv(A,x,work.p);
    residual.copy_from(b);
    axpy(handle,n,-1.,work.p,residual.p);
    return norm(handle,n,residual.p);
  };
  double beta=true_residual();
  result.initialNorm=beta;
  result.finalNorm=beta;
  if(!std::isfinite(beta))throw std::runtime_error("DG FGMRES nonfinite initial residual");
  if(beta<=tolerance){
    result.converged=true;
    result.relativeResidual=beta/std::max(normB,1e-300);
    return result;
  }
  std::vector<double>H((std::size_t)(m+1)*m,0.);
  auto hij=[&](int i,int j)->double& {return H[(std::size_t)j*(m+1)+i];};
  std::vector<double>cs((std::size_t)m),sn((std::size_t)m),g((std::size_t)m+1),y((std::size_t)m);
  while(result.iterations<opt.maxIterations){
    ++result.restarts;
    V[0]->copy_from(residual.p);
    scal(handle,n,1./beta,V[0]->p);
    std::fill(H.begin(),H.end(),0.);
    std::fill(g.begin(),g.end(),0.);
    g[0]=beta;
    int k=0;
    for(int j=0;j<m&&result.iterations<opt.maxIterations;++j){
      inverse_blocks(A,V[(std::size_t)j]->p,Z[(std::size_t)j]->p,bad);
      spmv(A,Z[(std::size_t)j]->p,work.p);
      // Modified Gram-Schmidt, optionally double pass.
      for(int pass=0;pass<(opt.reorthogonalize?2:1);++pass)for(int i=0;i<=j;++i){
        const double h=dotproduct(handle,n,work.p,V[(std::size_t)i]->p);
        hij(i,j)+=h;
        axpy(handle,n,-h,V[(std::size_t)i]->p,work.p);
      }
      const double hn=norm(handle,n,work.p);
      if(!std::isfinite(hn))throw std::runtime_error("DG FGMRES nonfinite Arnoldi norm");
      hij(j+1,j)=hn;
      if(hn>0.){
        V[(std::size_t)j+1]->copy_from(work.p);
        scal(handle,n,1./hn,V[(std::size_t)j+1]->p);
      }
      for(int i=0;i<j;++i){
        const double a=hij(i,j),bval=hij(i+1,j);
        hij(i,j)=cs[(std::size_t)i]*a+sn[(std::size_t)i]*bval;
        hij(i+1,j)=-sn[(std::size_t)i]*a+cs[(std::size_t)i]*bval;
      }
      const double a=hij(j,j),bval=hij(j+1,j),rho=std::hypot(a,bval);
      if(!(rho>0.))throw std::runtime_error("DG FGMRES zero Hessenberg diagonal");
      cs[(std::size_t)j]=a/rho;sn[(std::size_t)j]=bval/rho;
      hij(j,j)=rho;hij(j+1,j)=0.;
      g[(std::size_t)j+1]=-sn[(std::size_t)j]*g[(std::size_t)j];
      g[(std::size_t)j]=cs[(std::size_t)j]*g[(std::size_t)j];
      ++result.iterations;k=j+1;
      const double estimated=std::fabs(g[(std::size_t)j+1]);
      if(opt.printEvery>0 && result.iterations%opt.printEvery==0)
        std::printf("DG_OMEGA_GPU_FGMRES_ITER iter=%d estimatedResidual=%.9e\n",
                    result.iterations,estimated);
      if(estimated<=tolerance||hn==0.)break;
    }
    for(int i=k-1;i>=0;--i){
      double v=g[(std::size_t)i];
      for(int j=i+1;j<k;++j)v-=hij(i,j)*y[(std::size_t)j];
      if(!(std::fabs(hij(i,i))>1.e-35))throw std::runtime_error("DG FGMRES singular triangular system");
      y[(std::size_t)i]=v/hij(i,i);
    }
    for(int i=0;i<k;++i)axpy(handle,n,y[(std::size_t)i],Z[(std::size_t)i]->p,x);
    beta=true_residual();
    if(!std::isfinite(beta))throw std::runtime_error("DG FGMRES nonfinite true residual");
    result.finalNorm=beta;
    if(beta<=tolerance){result.converged=true;break;}
  }
  unsigned int err=0;
  cuda_ok(cudaMemcpy(&err,bad,sizeof(err),cudaMemcpyDeviceToHost),"read Jacobi errors");
  if(err)throw std::runtime_error("DG FGMRES singular 4x4 Jacobi blocks ("+std::to_string(err)+")");
  result.relativeResidual=beta/std::max(normB,1e-300);
  return result;
}
}} // namespace nodals::omega_dg_gpu
