#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Restarted right-preconditioned FGMRES for the HXT5B scalar CSR.
//
// Right preconditioner:
//     z_j = M^{-1} v_j
//
// where M^{-1} is one or more FORWARD multi-colour Gauss-Seidel sweeps,
// started from zero.
//
// The Krylov operator is therefore
//
//     A M^{-1}
//
// and the solution update is
//
//     x <- x + sum_j y_j z_j.
//
// All convergence decisions use the true ||b-Ax||_2 residual,
// accumulated through the existing FP64 Reducer.
//
// One workspace is cached per (n,restart), so Ux/Uy/Uz/k/omega reuse
// the same Krylov storage instead of cudaMalloc/cudaFree each solve.

namespace hxt5e {

using Real = hxt4b::Real;

static constexpr int TPB = hxt4b::TPB;

__global__ void zero_masked_kernel(
    int n,const unsigned char*mask,Real*x)
{
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i<n && mask[i]) x[i]=Real(0);
}

__global__ void masked_residual_kernel(
    int n,const unsigned char*mask,
    const Real*b,const Real*Ax,Real*r)
{
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i<n) r[i]=mask[i] ? Real(0) : b[i]-Ax[i];
}

__global__ void copy_scaled_kernel(
    int n,const Real*x,Real*y,Real a)
{
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i<n) y[i]=a*x[i];
}

__global__ void axpy_kernel(
    int n,Real a,const Real*x,Real*y)
{
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i<n) y[i]+=a*x[i];
}

struct Workspace {
  int n=0;
  int restart=0;

  Real* V=nullptr;   // (restart+1) * n
  Real* Z=nullptr;   // restart * n
  Real* w=nullptr;
  Real* r=nullptr;

  Workspace(int n_,int restart_)
    : n(n_),restart(restart_)
  {
    if(n<=0 || restart<=0)
      throw std::runtime_error("HXT5E invalid FGMRES workspace");

    const std::size_t nv=(std::size_t)(restart+1)*(std::size_t)n;
    const std::size_t nz=(std::size_t)restart*(std::size_t)n;

    HXT1_CUDA(cudaMalloc((void**)&V,nv*sizeof(Real)));
    HXT1_CUDA(cudaMalloc((void**)&Z,nz*sizeof(Real)));
    HXT1_CUDA(cudaMalloc((void**)&w,(std::size_t)n*sizeof(Real)));
    HXT1_CUDA(cudaMalloc((void**)&r,(std::size_t)n*sizeof(Real)));

    HXT1_CUDA(cudaMemset(V,0,nv*sizeof(Real)));
    HXT1_CUDA(cudaMemset(Z,0,nz*sizeof(Real)));
    HXT1_CUDA(cudaMemset(w,0,(std::size_t)n*sizeof(Real)));
    HXT1_CUDA(cudaMemset(r,0,(std::size_t)n*sizeof(Real)));

    const std::size_t bytes=(nv+nz+2ull*(std::size_t)n)*sizeof(Real);

    std::printf(
      "NODALS_HXT5E_FGMRES_WORKSPACE "
      "n=%d restart=%d vectors=%d bytes=%zu MiB=%.3f precision=%s status=PASS\n",
      n,restart,2*restart+3,bytes,
      bytes/(1024.0*1024.0),
      hxt4b::PRECISION);
  }

  Workspace(const Workspace&)=delete;
  Workspace&operator=(const Workspace&)=delete;

  ~Workspace()
  {
    if(V) cudaFree(V);
    if(Z) cudaFree(Z);
    if(w) cudaFree(w);
    if(r) cudaFree(r);
  }

  Real* v(int j)
  {
    return V+(std::size_t)j*(std::size_t)n;
  }

  Real* z(int j)
  {
    return Z+(std::size_t)j*(std::size_t)n;
  }
};

inline Workspace& get_workspace(int n,int restart)
{
  static std::vector<std::unique_ptr<Workspace>> cache;

  for(auto& q:cache)
    if(q->n==n && q->restart==restart)
      return *q;

  cache.emplace_back(std::make_unique<Workspace>(n,restart));
  return *cache.back();
}

struct Result {
  int its=0;
  int cycles=0;

  double initial=0.0;
  double final=0.0;
  double rel=0.0;

  bool ok=false;
};

inline void apply(
    hxt4b::GpuCSR&A,
    const Real*val,
    const unsigned char*mask,
    const Real*x,
    Real*y)
{
  hxt5b::spmv<<<(A.n+TPB-1)/TPB,TPB>>>(
    A.n,A.row.p,A.col.p,val,x,y);

  zero_masked_kernel<<<(A.n+TPB-1)/TPB,TPB>>>(
    A.n,mask,y);

  HXT1_CUDA(cudaGetLastError());
}

inline double residual_norm(
    hxt4b::GpuCSR&A,
    const Real*val,
    const Real*b,
    const Real*x,
    const unsigned char*mask,
    Workspace&W,
    hxt4a::Reducer<Real>&red)
{
  hxt5b::spmv<<<(A.n+TPB-1)/TPB,TPB>>>(
    A.n,A.row.p,A.col.p,val,x,W.w);

  masked_residual_kernel<<<(A.n+TPB-1)/TPB,TPB>>>(
    A.n,mask,b,W.w,W.r);

  HXT1_CUDA(cudaGetLastError());

  return red.norm(W.r);
}

inline void right_fgs(
    hxt4b::GpuCSR&A,
    const hxt4b::ColoringHost&C,
    const Real*val,
    const unsigned char*mask,
    const Real*rhs,
    Real*z,
    int sweeps,
    double omega)
{
  HXT1_CUDA(cudaMemset(
    z,0,(std::size_t)A.n*sizeof(Real)));

  for(int s=0;s<sweeps;++s)
    hxt5b::sweep(
      A,C,val,rhs,z,omega,true,mask);

  zero_masked_kernel<<<(A.n+TPB-1)/TPB,TPB>>>(
    A.n,mask,z);

  HXT1_CUDA(cudaGetLastError());
}

inline Result fgmres(
    const char*label,
    hxt4b::GpuCSR&A,
    const hxt4b::ColoringHost&C,
    const Real*val,
    const Real*b,
    Real*x,
    const unsigned char*mask,
    double rtol,
    double atol,
    int maxit,
    int restart,
    int fgsSweeps,
    double fgsOmega,
    hxt4a::Reducer<Real>&red)
{
  if(!(rtol>0.0) || !(atol>=0.0) ||
     maxit<=0 || restart<=0 ||
     fgsSweeps<=0 || !(fgsOmega>0.0))
    throw std::runtime_error(
      std::string("HXT5E invalid FGMRES option label=")+label);

  restart=std::min(restart,maxit);

  Workspace&W=get_workspace(A.n,restart);

  // Match the existing GS solver contract: clamped DOFs are zero.
  zero_masked_kernel<<<(A.n+TPB-1)/TPB,TPB>>>(
    A.n,mask,x);

  HXT1_CUDA(cudaGetLastError());

  Result R;

  R.initial=residual_norm(
    A,val,b,x,mask,W,red);

  R.final=R.initial;

  if(!std::isfinite(R.initial))
    throw std::runtime_error(
      std::string("HXT5E nonfinite initial residual label=")+label);

  const double target=
    std::max(atol,rtol*R.initial);

  auto emit=[&](){
    R.rel=R.final/std::max(R.initial,1e-300);

    std::printf(
      "NODALS_HXT5E_FGMRES "
      "label=%s its=%d cycles=%d restart=%d "
      "fgsSweeps=%d fgsOmega=%.6g "
      "initial=%.12e trueFinal=%.12e rel=%.12e "
      "rtol=%.3e atol=%.3e converged=%d status=%s\n",
      label,R.its,R.cycles,restart,
      fgsSweeps,fgsOmega,
      R.initial,R.final,R.rel,
      rtol,atol,(int)R.ok,
      R.ok?"PASS":"WARN");
  };

  if(R.initial<=target){
    R.ok=true;
    emit();
    return R;
  }

  while(R.its<maxit){
    ++R.cycles;

    const double beta=residual_norm(
      A,val,b,x,mask,W,red);

    if(!std::isfinite(beta))
      throw std::runtime_error(
        std::string("HXT5E nonfinite restart residual label=")+label);

    if(beta<=target){
      R.final=beta;
      R.ok=true;
      emit();
      return R;
    }

    copy_scaled_kernel<<<(A.n+TPB-1)/TPB,TPB>>>(
      A.n,W.r,W.v(0),Real(1.0/beta));

    HXT1_CUDA(cudaGetLastError());

    std::vector<double> H(
      (std::size_t)(restart+1)*(std::size_t)restart,0.0);

    std::vector<double> cs((std::size_t)restart,0.0);
    std::vector<double> sn((std::size_t)restart,0.0);
    std::vector<double> g((std::size_t)restart+1,0.0);

    g[0]=beta;

    auto h=[&](int i,int j)->double&{
      return H[(std::size_t)i*(std::size_t)restart+
               (std::size_t)j];
    };

    int kdim=0;
    bool happyBreakdown=false;

    const int cycleMax=
      std::min(restart,maxit-R.its);

    for(int j=0;j<cycleMax;++j){

      // Flexible right preconditioning:
      //
      //     z_j = M_j^{-1} v_j
      //
      // M_j is currently one or more forward coloured-GS sweeps.
      right_fgs(
        A,C,val,mask,W.v(j),W.z(j),
        fgsSweeps,fgsOmega);

      // w = A z_j
      apply(
        A,val,mask,W.z(j),W.w);

      // Modified Gram-Schmidt.
      for(int i=0;i<=j;++i){
        const double hij=
          red.dot(W.w,W.v(i));

        h(i,j)=hij;

        axpy_kernel<<<(A.n+TPB-1)/TPB,TPB>>>(
          A.n,Real(-hij),W.v(i),W.w);
      }

      HXT1_CUDA(cudaGetLastError());

      const double hn=
        red.norm(W.w);

      if(!std::isfinite(hn))
        throw std::runtime_error(
          std::string("HXT5E Arnoldi norm nonfinite label=")+label);

      h(j+1,j)=hn;

      if(hn>1e-30){
        copy_scaled_kernel<<<(A.n+TPB-1)/TPB,TPB>>>(
          A.n,W.w,W.v(j+1),Real(1.0/hn));
      }else{
        happyBreakdown=true;
      }

      // Apply previous Givens rotations.
      for(int i=0;i<j;++i){
        const double a=h(i,j);
        const double b0=h(i+1,j);

        h(i,j)=
          cs[(std::size_t)i]*a+
          sn[(std::size_t)i]*b0;

        h(i+1,j)=
          -sn[(std::size_t)i]*a+
           cs[(std::size_t)i]*b0;
      }

      // New Givens rotation.
      const double a=h(j,j);
      const double b0=h(j+1,j);
      const double rho=std::hypot(a,b0);

      if(rho>0.0){
        cs[(std::size_t)j]=a/rho;
        sn[(std::size_t)j]=b0/rho;
      }else{
        cs[(std::size_t)j]=1.0;
        sn[(std::size_t)j]=0.0;
      }

      h(j,j)=
        cs[(std::size_t)j]*a+
        sn[(std::size_t)j]*b0;

      h(j+1,j)=0.0;

      const double gj=g[(std::size_t)j];

      g[(std::size_t)j]=
        cs[(std::size_t)j]*gj;

      g[(std::size_t)j+1]=
        -sn[(std::size_t)j]*gj;

      ++R.its;
      kdim=j+1;

      const double estimated=
        std::abs(g[(std::size_t)j+1]);

      if(estimated<=target || happyBreakdown)
        break;
    }

    if(kdim<=0)
      break;

    // Back solve R y = g.
    std::vector<double> y((std::size_t)kdim,0.0);

    for(int i=kdim-1;i>=0;--i){
      double q=g[(std::size_t)i];

      for(int j=i+1;j<kdim;++j)
        q-=h(i,j)*y[(std::size_t)j];

      const double d=h(i,i);

      if(std::abs(d)<=1e-300 ||
         !std::isfinite(d))
        throw std::runtime_error(
          std::string("HXT5E singular Hessenberg label=")+label);

      y[(std::size_t)i]=q/d;
    }

    // x += Z y
    for(int j=0;j<kdim;++j){
      axpy_kernel<<<(A.n+TPB-1)/TPB,TPB>>>(
        A.n,Real(y[(std::size_t)j]),W.z(j),x);
    }

    zero_masked_kernel<<<(A.n+TPB-1)/TPB,TPB>>>(
      A.n,mask,x);

    HXT1_CUDA(cudaGetLastError());

    // Never accept only the Hessenberg residual estimate.
    // Audit the true CSR residual after every restart/candidate update.
    R.final=residual_norm(
      A,val,b,x,mask,W,red);

    if(!std::isfinite(R.final))
      throw std::runtime_error(
        std::string("HXT5E true residual nonfinite label=")+label);

    if(R.final<=target){
      R.ok=true;
      emit();
      return R;
    }

    if(happyBreakdown)
      break;
  }

  R.final=residual_norm(
    A,val,b,x,mask,W,red);

  R.ok=R.final<=target;

  emit();

  return R;
}

// Drop-in wrapper for the existing HXT5B momentum solve.
// Legacy modes are untouched.
inline hxt5b::H5B3MomentumResult momentum_solve(
    const std::string&mode,
    int outer,
    hxt4b::GpuCSR&A,
    const hxt4b::ColoringHost&C,
    const Real*val,
    const Real*b,
    Real*x,
    double omega,
    const unsigned char*clamp,
    double rtol,
    double atol,
    int maxIts,
    int restart,
    int fgsSweeps,
    hxt4a::Reducer<Real>&red,
    const char*component)
{
  if(mode!="fgmres-fgs")
    return hxt5b::h5b3_momentum_solve(
      mode,outer,A,C,val,b,x,
      omega,clamp,
      rtol,atol,maxIts,
      red,component);

  const std::string label=
    std::string("momentum_")+component;

  auto Q=fgmres(
    label.c_str(),
    A,C,val,b,x,clamp,
    rtol,atol,
    maxIts,restart,
    fgsSweeps,omega,
    red);

  hxt5b::H5B3MomentumResult R;

  R.sweeps=Q.its;
  R.initial=Q.initial;
  R.final=Q.final;
  R.rel=Q.rel;
  R.ok=Q.ok;

  return R;
}

} // namespace hxt5e
