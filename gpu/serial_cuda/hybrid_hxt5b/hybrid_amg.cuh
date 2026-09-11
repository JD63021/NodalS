#pragma once

#include "../g8_cf_hierarchy_host.hpp"
#include "hybrid_cf_overload.inc"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace hxt4c {

static constexpr int AMG_TPB=256;

template<class T>
struct Buf {
  T*p=nullptr; std::size_t n=0;
  Buf()=default;
  explicit Buf(std::size_t n_){alloc(n_);}
  explicit Buf(const std::vector<T>&v){upload(v);}
  Buf(const Buf&)=delete; Buf&operator=(const Buf&)=delete;
  Buf(Buf&&o) noexcept:p(o.p),n(o.n){o.p=nullptr;o.n=0;}
  Buf&operator=(Buf&&o) noexcept {if(this!=&o){if(p)cudaFree(p);p=o.p;n=o.n;o.p=nullptr;o.n=0;}return *this;}
  ~Buf(){if(p)cudaFree(p);}
  void alloc(std::size_t n_){if(p)cudaFree(p);p=nullptr;n=n_;if(n)HXT1_CUDA(cudaMalloc((void**)&p,n*sizeof(T)));}
  void zero(){if(n)HXT1_CUDA(cudaMemset(p,0,n*sizeof(T)));}
  void upload(const std::vector<T>&v){alloc(v.size());if(n)HXT1_CUDA(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
  void download(std::vector<T>&v)const{v.resize(n);if(n)HXT1_CUDA(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));}
};

template<class T> inline std::vector<T> castv(const std::vector<double>&x){
  std::vector<T>y(x.size());for(std::size_t i=0;i<x.size();++i)y[i]=(T)x[i];return y;
}

struct HybridFineHost {
  nodals_gpu::CSRHost A;
  std::vector<std::int32_t> entryRow;
  std::vector<std::int32_t> diagPos;
};

inline double hxc_host_schur_entry(const hxt3a::BCSR&B,int i,int j,const std::vector<double>&rau){
  std::int64_t a=B.row[(std::size_t)i],ae=B.row[(std::size_t)i+1];
  std::int64_t b=B.row[(std::size_t)j],be=B.row[(std::size_t)j+1];
  long double s=0;
  while(a<ae && b<be){
    const int ga=B.col[(std::size_t)a],gb=B.col[(std::size_t)b];
    if(ga<gb){++a;continue;}
    if(gb<ga){++b;continue;}
    s+=(long double)rau[(std::size_t)ga]*
      ((long double)B.bx[(std::size_t)a]*B.bx[(std::size_t)b]+
       (long double)B.by[(std::size_t)a]*B.by[(std::size_t)b]+
       (long double)B.bz[(std::size_t)a]*B.bz[(std::size_t)b]);
    ++a;++b;
  }
  return (double)s;
}

inline HybridFineHost build_hybrid_fine_host(const hxt3a::BCSR&B,int pin,const std::vector<double>&rau){
  if((int)rau.size()!=B.nv)throw std::runtime_error("HXT4C rAU host size mismatch");

  std::vector<std::int64_t>incOff((std::size_t)B.nv+1,0);
  for(auto g:B.col){
    if(g<0||g>=B.nv)throw std::runtime_error("HXT4C B column out of range");
    ++incOff[(std::size_t)g+1];
  }
  for(int g=0;g<B.nv;++g)incOff[(std::size_t)g+1]+=incOff[(std::size_t)g];
  std::vector<std::int32_t>incRow(B.col.size());
  auto next=incOff;
  for(int i=0;i<B.np;++i)
    for(std::int64_t k=B.row[(std::size_t)i];k<B.row[(std::size_t)i+1];++k)
      incRow[(std::size_t)next[(std::size_t)B.col[(std::size_t)k]]++]=(std::int32_t)i;

  HybridFineHost H;
  H.A.n=B.np;
  H.A.row.assign((std::size_t)B.np+1,0);
  std::vector<std::int32_t>nb;
  for(int i=0;i<B.np;++i){
    nb.clear();
    if(i==pin){
      nb.push_back((std::int32_t)pin);
    }else{
      nb.push_back((std::int32_t)i);
      for(std::int64_t k=B.row[(std::size_t)i];k<B.row[(std::size_t)i+1];++k){
        const int g=B.col[(std::size_t)k];
        for(std::int64_t q=incOff[(std::size_t)g];q<incOff[(std::size_t)g+1];++q){
          const int j=incRow[(std::size_t)q];
          if(j!=pin)nb.push_back((std::int32_t)j);
        }
      }
      std::sort(nb.begin(),nb.end());
      nb.erase(std::unique(nb.begin(),nb.end()),nb.end());
    }
    H.A.col.insert(H.A.col.end(),nb.begin(),nb.end());
    H.A.row[(std::size_t)i+1]=(std::int64_t)H.A.col.size();
  }

  H.A.val.resize(H.A.col.size());
  H.A.diag.resize((std::size_t)B.np);
  H.entryRow.resize(H.A.col.size());
  H.diagPos.assign((std::size_t)B.np,-1);

  for(int i=0;i<B.np;++i){
    for(std::int64_t k=H.A.row[(std::size_t)i];k<H.A.row[(std::size_t)i+1];++k){
      const int j=H.A.col[(std::size_t)k];
      H.entryRow[(std::size_t)k]=(std::int32_t)i;
      const double v=(i==pin)?(j==pin?1.0:0.0):hxc_host_schur_entry(B,i,j,rau);
      H.A.val[(std::size_t)k]=v;
      if(j==i){H.diagPos[(std::size_t)i]=(std::int32_t)k;H.A.diag[(std::size_t)i]=v;}
    }
    if(H.diagPos[(std::size_t)i]<0)throw std::runtime_error("HXT4C pressure CSR missing diagonal");
    if(!(H.A.diag[(std::size_t)i]>0.0)||!std::isfinite(H.A.diag[(std::size_t)i]))
      throw std::runtime_error("HXT4C pressure CSR nonpositive diagonal");
  }

  std::printf(
    "NODALS_HXT4C_FINE_HOST rows=%d nnz=%zu avgNnz=%.6f pin=%d "
    "source=B_DIAG_RAU_BT_REDUCED_PIN status=PASS\n",
    H.A.n,H.A.val.size(),H.A.n?(double)H.A.val.size()/H.A.n:0.0,pin);
  return H;
}

template<class T>
__global__ void refresh_schur_kernel(
    std::size_t nnz,const std::int32_t*entryRow,const std::int32_t*acol,
    const std::int64_t*brow,const std::int32_t*bcol,
    const T*bx,const T*by,const T*bz,const T*rau,T*aval,int pin)
{
  std::size_t k=(std::size_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(k>=nnz)return;
  const int i=entryRow[k],j=acol[k];
  if(i==pin){aval[k]=(j==pin)?T(1):T(0);return;}
  if(j==pin){aval[k]=T(0);return;}

  std::int64_t a=brow[i],ae=brow[i+1],b=brow[j],be=brow[j+1];
  T s=T(0);
  while(a<ae&&b<be){
    const int ga=bcol[a],gb=bcol[b];
    if(ga<gb){++a;continue;}
    if(gb<ga){++b;continue;}
    s+=rau[ga]*(bx[a]*bx[b]+by[a]*by[b]+bz[a]*bz[b]);
    ++a;++b;
  }
  aval[k]=s;
}

template<class T>
__global__ void diag_l1_kernel(
    int n,const std::int64_t*row,const std::int32_t*diagPos,
    const T*val,T*diag,T*l1,unsigned long long*bad)
{
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i>=n)return;
  const T d=val[diagPos[i]];
  T s=T(0);
  for(std::int64_t k=row[i];k<row[i+1];++k)s+=(T)fabs((double)val[k]);
  diag[i]=d;l1[i]=s;
  if(!(d>T(0))||!isfinite((double)d)||!(s>T(0))||!isfinite((double)s))atomicAdd(bad,1ULL);
}

template<class T>
__global__ void csr_spmv_kernel(int n,const std::int64_t*row,const std::int32_t*col,const T*val,const T*x,T*y){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i>=n)return;
  T s=T(0);for(std::int64_t k=row[i];k<row[i+1];++k)s+=val[k]*x[col[k]];y[i]=s;
}
template<class T>
__global__ void residual_kernel(int n,const T*b,const T*ax,T*r){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)r[i]=b[i]-ax[i];
}
template<class T>
__global__ void jacobi_set_kernel(int n,T omega,const T*b,const T*den,T*x){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)x[i]=omega*b[i]/den[i];
}
template<class T>
__global__ void jacobi_add_kernel(int n,T omega,const T*r,const T*den,T*x){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)x[i]+=omega*r[i]/den[i];
}
template<class T>
__global__ void axpy_kernel(int n,T a,const T*x,T*y){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)y[i]+=a*x[i];
}
template<class T>
__global__ void copy_kernel(int n,const T*x,T*y){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)y[i]=x[i];
}

struct HostColor {
  int ncolors=0,maxColor=0;
  std::vector<std::int32_t>rows,off;
};

inline HostColor build_color(const nodals_gpu::CSRHost&A){
  std::vector<int>color((std::size_t)A.n,-1),mark(64,-1);
  int nc=0;
  for(int i=0;i<A.n;++i){
    for(std::int64_t k=A.row[(std::size_t)i];k<A.row[(std::size_t)i+1];++k){
      const int j=A.col[(std::size_t)k];if(j==i||j>=i)continue;
      const int c=color[(std::size_t)j];
      if(c>=0){if(c>=(int)mark.size())mark.resize((std::size_t)c+32,-1);mark[(std::size_t)c]=i;}
    }
    int c=0;while(c<(int)mark.size()&&mark[(std::size_t)c]==i)++c;
    if(c>=(int)mark.size())mark.resize((std::size_t)c+32,-1);
    color[(std::size_t)i]=c;nc=std::max(nc,c+1);
  }
  HostColor C;C.ncolors=nc;C.off.assign((std::size_t)nc+1,0);
  for(int c:color)++C.off[(std::size_t)c+1];
  for(int c=0;c<nc;++c)C.off[(std::size_t)c+1]+=C.off[(std::size_t)c];
  C.rows.resize((std::size_t)A.n);auto next=C.off;
  for(int i=0;i<A.n;++i)C.rows[(std::size_t)next[(std::size_t)color[(std::size_t)i]]++]=(std::int32_t)i;
  for(int c=0;c<nc;++c)C.maxColor=std::max(C.maxColor,(int)(C.off[(std::size_t)c+1]-C.off[(std::size_t)c]));
  return C;
}

template<class T>
struct DColor {
  int ncolors=0,maxColor=0;
  std::vector<std::int32_t>off;
  Buf<std::int32_t>rows;
  void upload(const HostColor&C){ncolors=C.ncolors;maxColor=C.maxColor;off=C.off;rows.upload(C.rows);}
};

template<class T>
__global__ void mcgs_color_kernel(
    int begin,int count,const std::int32_t*colorRows,
    const std::int64_t*row,const std::int32_t*col,const T*val,const T*diag,
    const T*b,T*x,T omega)
{
  int q=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(q>=count)return;
  const int i=colorRows[begin+q];T off=T(0);
  for(std::int64_t k=row[i];k<row[i+1];++k){int j=col[k];if(j!=i)off+=val[k]*x[j];}
  const T old=x[i],gs=(b[i]-off)/diag[i];x[i]=old+omega*(gs-old);
}

template<class T>
struct DeviceCSR {
  int n=0;
  Buf<std::int64_t>row;
  Buf<std::int32_t>col,diagPos;
  Buf<T>val,diag,l1,b,x,r,tmp,corr;
  DColor<T>color;

  DeviceCSR()=default;
  explicit DeviceCSR(const nodals_gpu::CSRHost&A){
    n=A.n;row.upload(A.row);col.upload(A.col);
    std::vector<std::int32_t>dp((std::size_t)n,-1);
    for(int i=0;i<n;++i){
      auto a=A.row[(std::size_t)i],e=A.row[(std::size_t)i+1];
      auto it=std::lower_bound(A.col.begin()+a,A.col.begin()+e,i);
      if(it==A.col.begin()+e||*it!=i)throw std::runtime_error("HXT4C coarse CSR diagonal missing");
      dp[(std::size_t)i]=(std::int32_t)(it-A.col.begin());
    }
    diagPos.upload(dp);val.upload(castv<T>(A.val));diag.upload(castv<T>(A.diag));
    std::vector<double>lh((std::size_t)n,0);
    for(int i=0;i<n;++i)for(std::int64_t k=A.row[(std::size_t)i];k<A.row[(std::size_t)i+1];++k)lh[(std::size_t)i]+=std::abs(A.val[(std::size_t)k]);
    l1.upload(castv<T>(lh));
    b.alloc((std::size_t)n);x.alloc((std::size_t)n);r.alloc((std::size_t)n);tmp.alloc((std::size_t)n);corr.alloc((std::size_t)n);
  }
  void apply(const T*xx,T*yy)const{
    csr_spmv_kernel<T><<<(n+AMG_TPB-1)/AMG_TPB,AMG_TPB>>>(n,row.p,col.p,val.p,xx,yy);
    HXT1_CUDA(cudaGetLastError());
  }
};

template<class T>
struct FineCSR {
  int n=0,pin=0;
  Buf<std::int64_t>row;
  Buf<std::int32_t>col,entryRow,diagPos;
  Buf<T>val,diag,l1,tmp,r,corr;
  DColor<T>color;

  FineCSR()=default;
  FineCSR(const HybridFineHost&H,int pin_):n(H.A.n),pin(pin_){
    row.upload(H.A.row);col.upload(H.A.col);entryRow.upload(H.entryRow);diagPos.upload(H.diagPos);
    val.upload(castv<T>(H.A.val));diag.upload(castv<T>(H.A.diag));
    std::vector<double>lh((std::size_t)n,0);
    for(int i=0;i<n;++i)for(std::int64_t k=H.A.row[(std::size_t)i];k<H.A.row[(std::size_t)i+1];++k)lh[(std::size_t)i]+=std::abs(H.A.val[(std::size_t)k]);
    l1.upload(castv<T>(lh));tmp.alloc((std::size_t)n);r.alloc((std::size_t)n);corr.alloc((std::size_t)n);
  }
  void apply(const T*x,T*y)const{
    csr_spmv_kernel<T><<<(n+AMG_TPB-1)/AMG_TPB,AMG_TPB>>>(n,row.p,col.p,val.p,x,y);HXT1_CUDA(cudaGetLastError());
  }

  void refresh(const hxt4a::BDevice<T>&B,const T*rau){
    const std::size_t nnz=val.n;
    refresh_schur_kernel<T><<<(nnz+AMG_TPB-1)/AMG_TPB,AMG_TPB>>>(
      nnz,entryRow.p,col.p,B.row.p,B.col.p,B.bx.p,B.by.p,B.bz.p,rau,val.p,pin);
    Buf<unsigned long long>bad(std::vector<unsigned long long>(1,0));
    diag_l1_kernel<T><<<(n+AMG_TPB-1)/AMG_TPB,AMG_TPB>>>(n,row.p,diagPos.p,val.p,diag.p,l1.p,bad.p);
    HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
    unsigned long long q=0;HXT1_CUDA(cudaMemcpy(&q,bad.p,sizeof(q),cudaMemcpyDeviceToHost));
    if(q)throw std::runtime_error("HXT4C refreshed fine Schur has invalid diagonal/L1");
  }
};

template<class T>
struct Transfer {
  int nf=0,nc=0;
  Buf<std::int64_t>row;
  Buf<std::int32_t>col;
  Buf<T>val;
  Transfer()=default;
  explicit Transfer(const nodals_gpu::SATransferHost&P):nf(P.nFine),nc(P.nCoarse){
    row.upload(P.row);col.upload(P.col);val.upload(castv<T>(P.val));
  }
};

template<class T>
__global__ void restrict_kernel(int nf,const std::int64_t*row,const std::int32_t*col,const T*val,const T*f,T*c){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i>=nf)return;const T q=f[i];
  for(std::int64_t k=row[i];k<row[i+1];++k)atomicAdd(c+col[k],val[k]*q);
}
template<class T>
__global__ void prolong_add_kernel(int nf,const std::int64_t*row,const std::int32_t*col,const T*val,const T*c,T*f){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i>=nf)return;T q=T(0);
  for(std::int64_t k=row[i];k<row[i+1];++k)q+=val[k]*c[col[k]];f[i]+=q;
}
template<class T>
__global__ void dense_inv_kernel(int n,const T*inv,const T*b,T*x){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i>=n)return;T s=T(0);
  for(int j=0;j<n;++j)s+=inv[(std::size_t)i*n+j]*b[j];x[i]=s;
}

template<class T>
__global__ void sym_scale_rhs_kernel(int n,const T*b,const T*d,T*bs){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i>=n)return;
  bs[i]=b[i]/sqrt(d[i]);
}
template<class T>
__global__ void sym_unscale_kernel(int n,const T*y,const T*d,T*x){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i>=n)return;
  x[i]=y[i]/sqrt(d[i]);
}
template<class T>
__global__ void sym_scale_inplace_kernel(int n,T*x,const T*d){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i>=n)return;
  x[i]/=sqrt(d[i]);
}
template<class T>
__global__ void sym_cheb_set_kernel(int n,T w,const T*bs,T*y){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)y[i]=w*bs[i];
}
template<class T>
__global__ void sym_cheb_add_kernel(int n,T w,const T*bs,const T*Hy,T*y){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)y[i]+=w*(bs[i]-Hy[i]);
}

struct AMGOptions {
  std::string smoother="jacobi";
  double jacobiOmega=0.7;
  int jacobiFineSweeps=1,jacobiCoarseSweeps=1;
  int chebDegree=4;
  double lambdaLowFraction=0.05;
  int mcgsFineSweeps=1,mcgsCoarseSweeps=1;
  double mcgsOmega=1.0;
  std::string mcgsOrder="symmetric";
};

template<class T>
class HybridAMG {
 public:
  FineCSR<T> fine;
  std::vector<std::unique_ptr<DeviceCSR<T>>> L;
  std::vector<std::unique_ptr<Transfer<T>>> P;
  Buf<T>terminalInv;
  int terminalN=0;
  double fineLambda=0;
  std::vector<double>levelLambda;
  AMGOptions opt;

  HybridAMG(const HybridFineHost&F,const nodals_gpu::SAHierarchyHost&H,int pin,const AMGOptions&o):
    fine(F,pin),terminalN(H.terminal_n),fineLambda(H.fineLambda),levelLambda(H.levelLambda),opt(o)
  {
    if(H.csr.empty()||H.P.empty())throw std::runtime_error("HXT4C AMG hierarchy empty");
    if(H.P.size()!=H.csr.size())throw std::runtime_error("HXT4C AMG transfer/level count mismatch");
    for(const auto&q:H.csr)L.emplace_back(std::make_unique<DeviceCSR<T>>(q));
    for(const auto&q:H.P)P.emplace_back(std::make_unique<Transfer<T>>(q));
    terminalInv.upload(castv<T>(H.terminal_inv));
    if(terminalN!=L.back()->n)throw std::runtime_error("HXT4C terminal size mismatch");
    if(terminalInv.n!=(std::size_t)terminalN*(std::size_t)terminalN)throw std::runtime_error("HXT4C terminal inverse size mismatch");

    if(opt.smoother=="mcgs"){
      auto cf=build_color(F.A);fine.color.upload(cf);
      std::printf("NODALS_HXT4C_AMG_COLOR level=0 rows=%d colors=%d maxColor=%d status=PASS\n",fine.n,cf.ncolors,cf.maxColor);
      for(std::size_t l=0;l+1<L.size();++l){
        auto cc=build_color(H.csr[l]);L[l]->color.upload(cc);
        std::printf("NODALS_HXT4C_AMG_COLOR level=%zu rows=%d colors=%d maxColor=%d status=PASS\n",l+1,L[l]->n,cc.ncolors,cc.maxColor);
      }
    }
    std::printf("NODALS_HXT4C_AMG_UPLOAD levels=%zu transfers=%zu terminal=%d smoother=%s chebScaling=%s status=PASS\n",
      L.size(),P.size(),terminalN,opt.smoother.c_str(),
      opt.smoother=="cheb2"?"SYMMETRIC_D_HALF":"NA");
  }

  void refresh(const hxt4a::BDevice<T>&B,const T*rau){fine.refresh(B,rau);}

  static double cheb_root(double hi,double lo,int k,int degree){
    const double centre=.5*(hi+lo),radius=.5*(hi-lo),pi=3.141592653589793238462643383279502884;
    return centre-radius*std::cos(pi*(2.0*k+1.0)/(2.0*degree));
  }

  template<class A>
  void mcgs(A&Aop,const T*b,T*x,bool fineLevel,int sweeps){
    HXT1_CUDA(cudaMemset(x,0,(std::size_t)Aop.n*sizeof(T)));
    auto one=[&](bool forward){
      if(forward){
        for(int c=0;c<Aop.color.ncolors;++c){
          int beg=Aop.color.off[(std::size_t)c],cnt=Aop.color.off[(std::size_t)c+1]-beg;
          mcgs_color_kernel<T><<<(cnt+AMG_TPB-1)/AMG_TPB,AMG_TPB>>>(
            beg,cnt,Aop.color.rows.p,Aop.row.p,Aop.col.p,Aop.val.p,Aop.diag.p,b,x,(T)opt.mcgsOmega);
        }
      }else{
        for(int c=Aop.color.ncolors-1;c>=0;--c){
          int beg=Aop.color.off[(std::size_t)c],cnt=Aop.color.off[(std::size_t)c+1]-beg;
          mcgs_color_kernel<T><<<(cnt+AMG_TPB-1)/AMG_TPB,AMG_TPB>>>(
            beg,cnt,Aop.color.rows.p,Aop.row.p,Aop.col.p,Aop.val.p,Aop.diag.p,b,x,(T)opt.mcgsOmega);
        }
      }
    };
    for(int s=0;s<sweeps;++s){
      if(opt.mcgsOrder=="symmetric"){one(true);one(false);}
      else if(opt.mcgsOrder=="forward")one(true);
      else if(opt.mcgsOrder=="backward")one(false);
      else throw std::runtime_error("HXT4C bad MCGS order");
    }
    HXT1_CUDA(cudaGetLastError());
    (void)fineLevel;
  }

  template<class A>
  void smooth_zero(A&Aop,const T*b,T*x,T*tmp,T*r,T*corr,int sweeps,double lambda){
    const T*den=(opt.smoother=="l1jacobi")?Aop.l1.p:Aop.diag.p;
    if(opt.smoother=="jacobi"||opt.smoother=="l1jacobi"){
      jacobi_set_kernel<T><<<(Aop.n+AMG_TPB-1)/AMG_TPB,AMG_TPB>>>(Aop.n,(T)opt.jacobiOmega,b,den,x);
      for(int s=1;s<sweeps;++s){
        Aop.apply(x,tmp);residual_kernel<T><<<(Aop.n+AMG_TPB-1)/AMG_TPB,AMG_TPB>>>(Aop.n,b,tmp,r);
        jacobi_add_kernel<T><<<(Aop.n+AMG_TPB-1)/AMG_TPB,AMG_TPB>>>(Aop.n,(T)opt.jacobiOmega,r,den,x);
      }
    }else if(opt.smoother=="cheb2"){
      // PCG-compatible production semantics:
      // H = D^{-1/2} A D^{-1/2}; bs = D^{-1/2} b.
      const double lo=opt.lambdaLowFraction*lambda;
      if(!(lambda>lo&&lo>0))throw std::runtime_error("HXT4C Chebyshev interval invalid");
      sym_scale_rhs_kernel<T><<<(Aop.n+AMG_TPB-1)/AMG_TPB,AMG_TPB>>>(
        Aop.n,b,Aop.diag.p,r);
      double w=1.0/cheb_root(lambda,lo,0,opt.chebDegree);
      sym_cheb_set_kernel<T><<<(Aop.n+AMG_TPB-1)/AMG_TPB,AMG_TPB>>>(
        Aop.n,(T)w,r,x);
      for(int k=1;k<opt.chebDegree;++k){
        sym_unscale_kernel<T><<<(Aop.n+AMG_TPB-1)/AMG_TPB,AMG_TPB>>>(
          Aop.n,x,Aop.diag.p,tmp);
        Aop.apply(tmp,corr);
        sym_scale_inplace_kernel<T><<<(Aop.n+AMG_TPB-1)/AMG_TPB,AMG_TPB>>>(
          Aop.n,corr,Aop.diag.p);
        w=1.0/cheb_root(lambda,lo,k,opt.chebDegree);
        sym_cheb_add_kernel<T><<<(Aop.n+AMG_TPB-1)/AMG_TPB,AMG_TPB>>>(
          Aop.n,(T)w,r,corr,x);
      }
      sym_unscale_kernel<T><<<(Aop.n+AMG_TPB-1)/AMG_TPB,AMG_TPB>>>(
        Aop.n,x,Aop.diag.p,x);
    }else{
      (void)corr;
      throw std::runtime_error("HXT4C smooth_zero called for unsupported smoother");
    }
    HXT1_CUDA(cudaGetLastError());
  }

  void smooth_fine(const T*b,T*x){
    if(opt.smoother=="mcgs"){
      // FineCSR has the same members used by mcgs().
      mcgs(fine,b,x,true,opt.mcgsFineSweeps);
    }else{
      smooth_zero(fine,b,x,fine.tmp.p,fine.r.p,fine.corr.p,opt.jacobiFineSweeps,fineLambda);
    }
  }

  void smooth_coarse(std::size_t l,const T*b,T*x){
    auto&A=*L[l];
    if(opt.smoother=="mcgs")mcgs(A,b,x,false,opt.mcgsCoarseSweeps);
    else{
      double lam=(l<levelLambda.size()?levelLambda[l]:0.0);
      smooth_zero(A,b,x,A.tmp.p,A.r.p,A.corr.p,opt.jacobiCoarseSweeps,lam);
    }
  }

  void restrict_to(std::size_t pidx,const T*f,T*c){
    auto&Tf=*P[pidx];HXT1_CUDA(cudaMemset(c,0,(std::size_t)Tf.nc*sizeof(T)));
    restrict_kernel<T><<<(Tf.nf+AMG_TPB-1)/AMG_TPB,AMG_TPB>>>(Tf.nf,Tf.row.p,Tf.col.p,Tf.val.p,f,c);
    HXT1_CUDA(cudaGetLastError());
  }
  void prolong_add(std::size_t pidx,const T*c,T*f){
    auto&Tf=*P[pidx];prolong_add_kernel<T><<<(Tf.nf+AMG_TPB-1)/AMG_TPB,AMG_TPB>>>(Tf.nf,Tf.row.p,Tf.col.p,Tf.val.p,c,f);
    HXT1_CUDA(cudaGetLastError());
  }

  void vcycle(std::size_t l,const T*b,T*x){
    auto&A=*L[l];
    if(l+1==L.size()){
      dense_inv_kernel<T><<<(A.n+AMG_TPB-1)/AMG_TPB,AMG_TPB>>>(A.n,terminalInv.p,b,x);
      HXT1_CUDA(cudaGetLastError());return;
    }
    smooth_coarse(l,b,x);
    A.apply(x,A.tmp.p);
    residual_kernel<T><<<(A.n+AMG_TPB-1)/AMG_TPB,AMG_TPB>>>(A.n,b,A.tmp.p,A.r.p);
    auto&C=*L[l+1];
    restrict_to(l+1,A.r.p,C.b.p);
    vcycle(l+1,C.b.p,C.x.p);
    prolong_add(l+1,C.x.p,x);
    A.apply(x,A.tmp.p);
    residual_kernel<T><<<(A.n+AMG_TPB-1)/AMG_TPB,AMG_TPB>>>(A.n,b,A.tmp.p,A.r.p);
    copy_kernel<T><<<(A.n+AMG_TPB-1)/AMG_TPB,AMG_TPB>>>(A.n,A.r.p,A.b.p);
    smooth_coarse(l,A.b.p,A.corr.p);
    axpy_kernel<T><<<(A.n+AMG_TPB-1)/AMG_TPB,AMG_TPB>>>(A.n,T(1),A.corr.p,x);
  }

  void apply(const T*b,T*z){
    smooth_fine(b,z);
    fine.apply(z,fine.tmp.p);
    residual_kernel<T><<<(fine.n+AMG_TPB-1)/AMG_TPB,AMG_TPB>>>(fine.n,b,fine.tmp.p,fine.r.p);
    auto&C=*L[0];
    restrict_to(0,fine.r.p,C.b.p);
    vcycle(0,C.b.p,C.x.p);
    prolong_add(0,C.x.p,z);
    fine.apply(z,fine.tmp.p);
    residual_kernel<T><<<(fine.n+AMG_TPB-1)/AMG_TPB,AMG_TPB>>>(fine.n,b,fine.tmp.p,fine.r.p);
    smooth_fine(fine.r.p,fine.corr.p);
    axpy_kernel<T><<<(fine.n+AMG_TPB-1)/AMG_TPB,AMG_TPB>>>(fine.n,T(1),fine.corr.p,z);
    HXT1_CUDA(cudaGetLastError());
  }
};

template<class T,class Action>
hxt4a::CGResult pcg_amg(
    Action&Act,HybridAMG<T>&PC,const T*rhs,T*x,int n,
    double rtol,double atol,int maxit,hxt4a::CGWork<T>&W,int pin)
{
  Act.apply(x,W.q.p);
  hxt4a::residual_kernel<T><<<(n+AMG_TPB-1)/AMG_TPB,AMG_TPB>>>(n,rhs,W.q.p,W.r.p);
  hxt4a::pin_zero_kernel<T><<<1,1>>>(pin,W.r.p);
  double r0=W.red.norm(W.r.p);
  if(!std::isfinite(r0))throw std::runtime_error("HXT4C pressure initial residual nonfinite");
  if(r0<=atol)return {0,0,true};
  const double target=std::max(atol,rtol*r0);

  PC.apply(W.r.p,W.z.p);
  hxt4a::pin_zero_kernel<T><<<1,1>>>(pin,W.z.p);
  HXT1_CUDA(cudaMemcpy(W.p.p,W.z.p,(std::size_t)n*sizeof(T),cudaMemcpyDeviceToDevice));
  double rho=W.red.dot(W.r.p,W.z.p);
  if(!(rho>0)||!std::isfinite(rho)){
    const double zn=W.red.norm(W.z.p);
    std::printf("NODALS_HXT5B4_PCG_RHO_AUDIT stage=initial rNorm=%.12e zNorm=%.12e rho=%.12e status=FAIL\n",
      r0,zn,rho);
    throw std::runtime_error("HXT4C pressure PCG initial rho nonpositive");
  }

  hxt4a::CGResult R;
  for(int k=0;k<maxit;++k){
    Act.apply(W.p.p,W.q.p);
    double pq=W.red.dot(W.p.p,W.q.p);
    if(!(pq>0)||!std::isfinite(pq))throw std::runtime_error("HXT4C pressure PCG pAp nonpositive");
    T a=(T)(rho/pq);
    hxt4a::axpy2_kernel<T><<<(n+AMG_TPB-1)/AMG_TPB,AMG_TPB>>>(n,a,W.p.p,W.q.p,x,W.r.p);
    hxt4a::pin_zero_kernel<T><<<1,1>>>(pin,x);hxt4a::pin_zero_kernel<T><<<1,1>>>(pin,W.r.p);
    double rn=W.red.norm(W.r.p);R.its=k+1;R.rel=rn/r0;
    if(rn<=target){R.ok=true;break;}
    PC.apply(W.r.p,W.z.p);hxt4a::pin_zero_kernel<T><<<1,1>>>(pin,W.z.p);
    double nr=W.red.dot(W.r.p,W.z.p);
    if(!(nr>0)||!std::isfinite(nr))throw std::runtime_error("HXT4C pressure PCG rho nonpositive");
    hxt4a::pupdate_kernel<T><<<(n+AMG_TPB-1)/AMG_TPB,AMG_TPB>>>(n,(T)(nr/rho),W.z.p,W.p.p);
    hxt4a::pin_zero_kernel<T><<<1,1>>>(pin,W.p.p);rho=nr;
  }
  return R;
}

template<class T>
__global__ void diff_kernel(int n,const T*a,const T*b,T*d){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)d[i]=a[i]-b[i];
}

template<class T>
double fine_action_parity(
    HybridAMG<T>&PC,hxt4a::SAction<T>&S,int n,int pin,hxt4a::Reducer<T>&red)
{
  std::vector<T>hx((std::size_t)n);
  for(int i=0;i<n;++i)hx[(std::size_t)i]=(i==pin)?T(0):(T)(std::sin(.00131*(i+1))+.2*std::cos(.00077*(i+1)));
  Buf<T>x(hx),a((std::size_t)n),b((std::size_t)n),d((std::size_t)n);
  S.apply(x.p,a.p);PC.fine.apply(x.p,b.p);
  diff_kernel<T><<<(n+AMG_TPB-1)/AMG_TPB,AMG_TPB>>>(n,a.p,b.p,d.p);
  const double den=std::max(red.norm(a.p),1e-300);
  return red.norm(d.p)/den;
}

} // namespace hxt4c
