#pragma once
#include "g4_host.hpp"
#include "g2_host.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nodals_gpu {

struct SATransferHost {
  int nFine=0,nCoarse=0;
  std::vector<std::int64_t> row;
  std::vector<std::int32_t> col;
  std::vector<double> val;
};

struct SAIncidenceHost {
  std::vector<std::int64_t> row;
  std::vector<std::int32_t> cell;
  std::vector<std::uint8_t> basis;
};

struct SAHierarchyHost {
  std::vector<AggResult> agg;          // one aggregate map per transfer
  std::vector<SATransferHost> P;       // fine->A0, A0->A1, ...
  std::vector<CSRHost> csr;            // explicit levels; last is terminal
  std::vector<double> fineDiag;
  double fineLambda=0.0;
  std::vector<double> levelLambda;     // 0 for terminal
  std::vector<double> terminal_inv;
  int terminal_n=0;
  int powerIts=16;
  int interpMaxNnz=8;
  double lambdaSafety=1.5;
  double lambdaLowFraction=0.05;
  double saDamping=4.0/3.0;
};

inline double sa_elapsed_s(const std::chrono::steady_clock::time_point&a,
                           const std::chrono::steady_clock::time_point&b){
  return std::chrono::duration<double>(b-a).count();
}

inline SAIncidenceHost build_sa_incidence(const PressureSetupHost&S){
  SAIncidenceHost I;
  I.row.assign((std::size_t)S.free_vel+1,0);
  for(std::size_t c=0;c<S.cells.size();++c)
    for(int a=0;a<8;++a){
      int g=S.cells[c].vel[a];
      if(g>=0) ++I.row[(std::size_t)g+1];
    }
  for(int g=0;g<S.free_vel;++g) I.row[(std::size_t)g+1]+=I.row[(std::size_t)g];
  I.cell.resize((std::size_t)I.row.back());
  I.basis.resize((std::size_t)I.row.back());
  auto next=I.row;
  for(std::size_t c=0;c<S.cells.size();++c)
    for(int a=0;a<8;++a){
      int g=S.cells[c].vel[a];
      if(g<0) continue;
      std::size_t k=(std::size_t)next[(std::size_t)g]++;
      I.cell[k]=(std::int32_t)c;
      I.basis[k]=(std::uint8_t)a;
    }
  return I;
}

inline void sa_normalize_prune(std::vector<std::pair<int,double>>&row,int own,int maxNnz){
  if(row.empty()) row.push_back({own,1.0});
  std::sort(row.begin(),row.end(),[](const auto&a,const auto&b){return a.first<b.first;});
  std::size_t w=0;
  for(std::size_t k=0;k<row.size();){
    int c=row[k].first; double v=0.0; std::size_t j=k;
    for(;j<row.size()&&row[j].first==c;++j) v+=row[j].second;
    row[w++]={c,v}; k=j;
  }
  row.resize(w);
  if(maxNnz>0 && (int)row.size()>maxNnz){
    std::vector<std::pair<int,double>> keep;keep.reserve((std::size_t)maxNnz);
    auto it=std::find_if(row.begin(),row.end(),[&](const auto&e){return e.first==own;});
    if(it!=row.end())keep.push_back(*it);else keep.push_back({own,0.0});
    std::vector<std::pair<int,double>> others;others.reserve(row.size());
    for(const auto&e:row)if(e.first!=own)others.push_back(e);
    std::sort(others.begin(),others.end(),[](const auto&a,const auto&b){
      double aa=std::abs(a.second),bb=std::abs(b.second);
      return aa>bb || (aa==bb && a.first<b.first);
    });
    for(const auto&e:others)if((int)keep.size()<maxNnz)keep.push_back(e);
    std::sort(keep.begin(),keep.end(),[](const auto&a,const auto&b){return a.first<b.first;});
    row.swap(keep);
  }
  double sum=0.0;for(const auto&e:row)sum+=e.second;
  if(!std::isfinite(sum)||std::abs(sum)<1e-14){row.clear();row.push_back({own,1.0});return;}
  for(auto&e:row)e.second/=sum;
}

inline void sa_fine_apply_reuse(const PressureSetupHost&S,const std::vector<double>&x,
                                std::vector<double>&y,std::vector<double>&v0,
                                std::vector<double>&v1,std::vector<double>&v2){
  if(x.size()!=S.cells.size())throw std::runtime_error("SA fine apply x size");
  std::fill(v0.begin(),v0.end(),0.0);std::fill(v1.begin(),v1.end(),0.0);std::fill(v2.begin(),v2.end(),0.0);
  for(std::size_t c=0;c<S.cells.size();++c){
    const auto&cp=S.cells[c];const double p=x[c];
    for(int a=0;a<8;++a){int g=cp.vel[a];if(g<0)continue;
      v0[(std::size_t)g]+=coeff(cp,a,0)*p;
      v1[(std::size_t)g]+=coeff(cp,a,1)*p;
      v2[(std::size_t)g]+=coeff(cp,a,2)*p;
    }
  }
  for(int g=0;g<S.free_vel;++g){double r=S.rAU[(std::size_t)g];v0[(std::size_t)g]*=r;v1[(std::size_t)g]*=r;v2[(std::size_t)g]*=r;}
  y.assign(S.cells.size(),0.0);
  for(std::size_t c=0;c<S.cells.size();++c){
    const auto&cp=S.cells[c];double s=0.0;
    for(int a=0;a<8;++a){int g=cp.vel[a];if(g<0)continue;
      s+=coeff(cp,a,0)*v0[(std::size_t)g]+coeff(cp,a,1)*v1[(std::size_t)g]+coeff(cp,a,2)*v2[(std::size_t)g];
    }
    y[c]=s;
  }
}

inline double sa_power_fine(const PressureSetupHost&S,const std::vector<double>&diag,int its,double safety){
  const int n=(int)S.cells.size();
  std::vector<double> v((std::size_t)n),u((std::size_t)n),au,w;
  std::vector<double> t0((std::size_t)S.free_vel),t1((std::size_t)S.free_vel),t2((std::size_t)S.free_vel);
  long double ss=0.0L;
  for(int i=0;i<n;++i){double g=(double)(i+1);v[(std::size_t)i]=std::sin(.731*g)+.27*std::cos(1.117*g);ss+=(long double)v[(std::size_t)i]*v[(std::size_t)i];}
  double vn=std::sqrt((double)ss);for(auto&q:v)q/=vn;
  double ray=0.0;
  for(int it=0;it<its;++it){
    for(int i=0;i<n;++i)u[(std::size_t)i]=v[(std::size_t)i]/std::sqrt(diag[(std::size_t)i]);
    sa_fine_apply_reuse(S,u,au,t0,t1,t2);
    w.resize((std::size_t)n);long double num=0.0L,den=0.0L,nw=0.0L;
    for(int i=0;i<n;++i){w[(std::size_t)i]=au[(std::size_t)i]/std::sqrt(diag[(std::size_t)i]);num+=(long double)v[(std::size_t)i]*w[(std::size_t)i];den+=(long double)v[(std::size_t)i]*v[(std::size_t)i];nw+=(long double)w[(std::size_t)i]*w[(std::size_t)i];}
    ray=(double)(num/std::max(den,(long double)1e-300));double nrm=std::sqrt((double)nw);
    if(!(nrm>0.0)||!std::isfinite(nrm)||!(ray>0.0)||!std::isfinite(ray))throw std::runtime_error("SA fine power iteration failed");
    for(int i=0;i<n;++i)v[(std::size_t)i]=w[(std::size_t)i]/nrm;
  }
  return safety*ray;
}

inline void sa_csr_apply(const CSRHost&A,const std::vector<double>&x,std::vector<double>&y){
  y.assign((std::size_t)A.n,0.0);
  for(int i=0;i<A.n;++i){double s=0.0;for(std::int64_t k=A.row[(std::size_t)i];k<A.row[(std::size_t)i+1];++k)s+=A.val[(std::size_t)k]*x[(std::size_t)A.col[(std::size_t)k]];y[(std::size_t)i]=s;}
}
inline double sa_power_csr(const CSRHost&A,int its,double safety){
  std::vector<double>v((std::size_t)A.n),u((std::size_t)A.n),au,w((std::size_t)A.n);
  long double ss=0;for(int i=0;i<A.n;++i){double g=(double)(i+1);v[(std::size_t)i]=std::sin(.731*g)+.27*std::cos(1.117*g);ss+=(long double)v[(std::size_t)i]*v[(std::size_t)i];}
  double vn=std::sqrt((double)ss);for(auto&q:v)q/=vn;double ray=0.0;
  for(int it=0;it<its;++it){
    for(int i=0;i<A.n;++i)u[(std::size_t)i]=v[(std::size_t)i]/std::sqrt(A.diag[(std::size_t)i]);
    sa_csr_apply(A,u,au);long double num=0,den=0,nw=0;
    for(int i=0;i<A.n;++i){w[(std::size_t)i]=au[(std::size_t)i]/std::sqrt(A.diag[(std::size_t)i]);num+=(long double)v[(std::size_t)i]*w[(std::size_t)i];den+=(long double)v[(std::size_t)i]*v[(std::size_t)i];nw+=(long double)w[(std::size_t)i]*w[(std::size_t)i];}
    ray=(double)(num/std::max(den,(long double)1e-300));double nrm=std::sqrt((double)nw);
    if(!(nrm>0.0)||!std::isfinite(nrm)||!(ray>0.0)||!std::isfinite(ray))throw std::runtime_error("SA coarse power iteration failed");
    for(int i=0;i<A.n;++i)v[(std::size_t)i]=w[(std::size_t)i]/nrm;
  }
  return safety*ray;
}

inline SATransferHost build_sa_fine_transfer_reference(const PressureSetupHost&S,const SAIncidenceHost&I,
                                              const AggResult&G,const std::vector<double>&diag,
                                              double lambdaMax,double damping,int maxNnz){
  const double omega=damping/lambdaMax;SATransferHost P;P.nFine=(int)S.cells.size();P.nCoarse=G.nagg;P.row.assign((std::size_t)P.nFine+1,0);
  P.col.reserve((std::size_t)P.nFine*3);P.val.reserve((std::size_t)P.nFine*3);
  std::vector<std::pair<int,double>> row;row.reserve(96);int maxrow=0;long double defect=0.0L;
  for(int i=0;i<P.nFine;++i){
    row.clear();const int own=G.id[(std::size_t)i];row.push_back({own,1.0});const auto&ci=S.cells[(std::size_t)i];const double invd=1.0/diag[(std::size_t)i];
    for(int a=0;a<8;++a){int g=ci.vel[a];if(g<0)continue;
      const double bi[3]={coeff(ci,a,0),coeff(ci,a,1),coeff(ci,a,2)};
      for(std::int64_t k=I.row[(std::size_t)g];k<I.row[(std::size_t)g+1];++k){
        int j=I.cell[(std::size_t)k],aj=(int)I.basis[(std::size_t)k];const auto&cj=S.cells[(std::size_t)j];
        double dot=0.0;for(int d=0;d<3;++d)dot+=bi[d]*coeff(cj,aj,d);
        row.push_back({G.id[(std::size_t)j],-omega*invd*S.rAU[(std::size_t)g]*dot});
      }
    }
    sa_normalize_prune(row,own,maxNnz);double sum=0.0;for(const auto&e:row){P.col.push_back(e.first);P.val.push_back(e.second);sum+=e.second;}P.row[(std::size_t)i+1]=(std::int64_t)P.col.size();maxrow=std::max(maxrow,(int)row.size());defect=std::max(defect,(long double)std::abs(sum-1.0));
  }
  std::printf("NODALS_GPU_SA_HOST_TRANSFER level=0 rows=%d coarse=%d nnz=%zu avgNnz=%.6f maxRowNnz=%d rowSumDefect=%.3Le omega=%.12e lambdaMax=%.12e status=PASS\n",
              P.nFine,P.nCoarse,P.val.size(),P.nFine?(double)P.val.size()/P.nFine:0.0,maxrow,defect,omega,lambdaMax);
  return P;
}

// -----------------------------------------------------------------------------
// Gate 2: parallel deterministic P0 construction.
//
// Every fine row is independent. Because production interpolationMaxNnz is
// bounded (8 in H8), each worker computes a normalized/pruned row once into a
// fixed-width slot. A short prefix scan then compacts those slots into the
// final SATransferHost CSR. No unordered_map, locks, atomics, or GPU scratch.
// -----------------------------------------------------------------------------

inline int sa_gate2_threads(){
  int n=0;
  if(const char*e=std::getenv("NODALS_P0_THREADS"))n=std::atoi(e);
  if(n<=0){unsigned h=std::thread::hardware_concurrency();n=h?std::min((int)h,16):1;}
  return std::max(1,std::min(n,64));
}

inline double sa_gate2_status_mib(const char*key){
  FILE*f=std::fopen("/proc/self/status","r");
  if(!f)return -1.0;
  char line[256];double out=-1.0;
  const std::size_t n=std::strlen(key);
  while(std::fgets(line,sizeof(line),f)){
    if(std::strncmp(line,key,n)==0){
      unsigned long long kb=0;
      if(std::sscanf(line+n,": %llu kB",&kb)==1)out=(double)kb/1024.0;
      break;
    }
  }
  std::fclose(f);
  return out;
}

template<class F>
inline void sa_gate2_parallel_chunks(int n,int nth,int chunk,F&&fn){
  if(n<=0)return;
  nth=std::max(1,std::min(nth,n));
  std::atomic<int> next{0};
  std::atomic<bool> failed{false};
  std::exception_ptr ep;
  std::mutex em;
  std::vector<std::thread> pool;
  pool.reserve((std::size_t)nth);
  for(int t=0;t<nth;++t){
    pool.emplace_back([&,t](){
      try{
        while(!failed.load(std::memory_order_relaxed)){
          const int b=next.fetch_add(chunk,std::memory_order_relaxed);
          if(b>=n)break;
          fn(t,b,std::min(n,b+chunk));
        }
      }catch(...){
        failed.store(true,std::memory_order_relaxed);
        std::lock_guard<std::mutex>g(em);
        if(!ep)ep=std::current_exception();
      }
    });
  }
  for(auto&th:pool)th.join();
  if(ep)std::rethrow_exception(ep);
}

inline void sa_gate2_make_p0_row(
    const PressureSetupHost&S,const SAIncidenceHost&I,const AggResult&G,
    const std::vector<double>&diag,int i,double omega,int maxNnz,
    std::vector<std::pair<int,double>>&row){
  row.clear();
  const int own=G.id[(std::size_t)i];
  row.push_back({own,1.0});
  const auto&ci=S.cells[(std::size_t)i];
  const double invd=1.0/diag[(std::size_t)i];

  for(int a=0;a<8;++a){
    const int g=ci.vel[a];
    if(g<0)continue;
    const double bi[3]={coeff(ci,a,0),coeff(ci,a,1),coeff(ci,a,2)};
    for(std::int64_t k=I.row[(std::size_t)g];k<I.row[(std::size_t)g+1];++k){
      const int j=I.cell[(std::size_t)k],aj=(int)I.basis[(std::size_t)k];
      const auto&cj=S.cells[(std::size_t)j];
      double dot=0.0;
      for(int d=0;d<3;++d)dot+=bi[d]*coeff(cj,aj,d);
      row.push_back({G.id[(std::size_t)j],
                     -omega*invd*S.rAU[(std::size_t)g]*dot});
    }
  }

  sa_normalize_prune(row,own,maxNnz);
}

inline SATransferHost build_sa_fine_transfer_parallel(
    const PressureSetupHost&S,const SAIncidenceHost&I,
    const AggResult&G,const std::vector<double>&diag,
    double lambdaMax,double damping,int maxNnz){
  using clock=std::chrono::steady_clock;
  const auto all0=clock::now();
  if(maxNnz<=0||maxNnz>255)
    throw std::runtime_error("Gate2 parallel P0 requires 1<=maxNnz<=255");

  const int nth=sa_gate2_threads();
  const double omega=damping/lambdaMax;

  SATransferHost P;
  P.nFine=(int)S.cells.size();
  P.nCoarse=G.nagg;

  const std::size_t nf=(std::size_t)P.nFine;
  const std::size_t width=(std::size_t)maxNnz;
  if(width && nf>std::numeric_limits<std::size_t>::max()/width)
    throw std::runtime_error("Gate2 P0 slot size overflow");
  const std::size_t slots=nf*width;

  std::printf("NODALS_GPU_SA_GATE2_STAGE stage=row_build state=START threads=%d rows=%d width=%d rssMiB=%.3f hwmMiB=%.3f\n",
              nth,P.nFine,maxNnz,sa_gate2_status_mib("VmRSS"),sa_gate2_status_mib("VmHWM"));
  std::fflush(stdout);

  std::vector<std::uint8_t> count(nf,0);
  std::vector<std::int32_t> slotCol(slots);
  std::vector<double> slotVal(slots);
  std::vector<int> threadMax((std::size_t)nth,0);
  std::vector<long double> threadDef((std::size_t)nth,0.0L);
  std::vector<std::vector<std::pair<int,double>>> work((std::size_t)nth);
  for(auto&r:work)r.reserve(96);

  const auto b0=clock::now();
  sa_gate2_parallel_chunks(P.nFine,nth,256,[&](int t,int ib,int ie){
    auto&row=work[(std::size_t)t];
    int lmax=0;
    long double ldef=0.0L;
    for(int i=ib;i<ie;++i){
      sa_gate2_make_p0_row(S,I,G,diag,i,omega,maxNnz,row);
      if(row.empty()||row.size()>width)
        throw std::runtime_error("Gate2 normalized P0 row exceeds fixed width");

      count[(std::size_t)i]=(std::uint8_t)row.size();
      const std::size_t base=(std::size_t)i*width;
      double sum=0.0;
      for(std::size_t k=0;k<row.size();++k){
        slotCol[base+k]=(std::int32_t)row[k].first;
        slotVal[base+k]=row[k].second;
        sum+=row[k].second;
      }
      lmax=std::max(lmax,(int)row.size());
      ldef=std::max(ldef,(long double)std::abs(sum-1.0));
    }
    threadMax[(std::size_t)t]=std::max(threadMax[(std::size_t)t],lmax);
    threadDef[(std::size_t)t]=std::max(threadDef[(std::size_t)t],ldef);
  });
  const auto b1=clock::now();

  std::vector<std::vector<std::pair<int,double>>>().swap(work);

  int maxrow=0;
  long double defect=0.0L;
  for(int t=0;t<nth;++t){
    maxrow=std::max(maxrow,threadMax[(std::size_t)t]);
    defect=std::max(defect,threadDef[(std::size_t)t]);
  }
  std::vector<int>().swap(threadMax);
  std::vector<long double>().swap(threadDef);

  std::printf("NODALS_GPU_SA_GATE2_ROW_BUILD status=PASS rows=%d seconds=%.6f maxRowNnz=%d rowSumDefect=%.3Le scratchSlots=%zu rssMiB=%.3f hwmMiB=%.3f\n",
              P.nFine,sa_elapsed_s(b0,b1),maxrow,defect,slots,
              sa_gate2_status_mib("VmRSS"),sa_gate2_status_mib("VmHWM"));
  std::fflush(stdout);

  const auto s0=clock::now();
  P.row.assign(nf+1,0);
  for(int i=0;i<P.nFine;++i)
    P.row[(std::size_t)i+1]=P.row[(std::size_t)i]+
                            (std::int64_t)count[(std::size_t)i];

  const std::size_t nnz=(std::size_t)P.row.back();
  P.col.resize(nnz);
  P.val.resize(nnz);
  const auto s1=clock::now();

  std::printf("NODALS_GPU_SA_GATE2_STAGE stage=compact state=START nnz=%zu rssMiB=%.3f hwmMiB=%.3f\n",
              nnz,sa_gate2_status_mib("VmRSS"),sa_gate2_status_mib("VmHWM"));
  std::fflush(stdout);

  const auto c0=clock::now();
  sa_gate2_parallel_chunks(P.nFine,nth,1024,[&](int,int ib,int ie){
    for(int i=ib;i<ie;++i){
      const std::size_t src=(std::size_t)i*width;
      std::int64_t dst=P.row[(std::size_t)i];
      const int n=(int)count[(std::size_t)i];
      for(int k=0;k<n;++k){
        P.col[(std::size_t)dst]=(std::int32_t)slotCol[src+(std::size_t)k];
        P.val[(std::size_t)dst]=slotVal[src+(std::size_t)k];
        ++dst;
      }
    }
  });
  const auto c1=clock::now();

  const double scratchMiB=
    (double)(count.size()*sizeof(std::uint8_t)+
             slotCol.size()*sizeof(std::int32_t)+
             slotVal.size()*sizeof(double))/(1024.0*1024.0);

  std::vector<std::uint8_t>().swap(count);
  std::vector<std::int32_t>().swap(slotCol);
  std::vector<double>().swap(slotVal);

  const auto all1=clock::now();

  std::printf("NODALS_GPU_SA_GATE2_P0_NEW status=PASS rows=%d coarse=%d nnz=%zu avgNnz=%.6f maxRowNnz=%d rowSumDefect=%.3Le threads=%d buildSeconds=%.6f scanAllocSeconds=%.6f compactSeconds=%.6f totalSeconds=%.6f scratchMiB=%.3f omega=%.12e lambdaMax=%.12e rssMiB=%.3f hwmMiB=%.3f\n",
              P.nFine,P.nCoarse,P.val.size(),
              P.nFine?(double)P.val.size()/P.nFine:0.0,
              maxrow,defect,nth,
              sa_elapsed_s(b0,b1),sa_elapsed_s(s0,s1),
              sa_elapsed_s(c0,c1),sa_elapsed_s(all0,all1),
              scratchMiB,omega,lambdaMax,
              sa_gate2_status_mib("VmRSS"),sa_gate2_status_mib("VmHWM"));

  std::printf("NODALS_GPU_SA_HOST_TRANSFER level=0 rows=%d coarse=%d nnz=%zu avgNnz=%.6f maxRowNnz=%d rowSumDefect=%.3Le omega=%.12e lambdaMax=%.12e status=PASS\n",
              P.nFine,P.nCoarse,P.val.size(),
              P.nFine?(double)P.val.size()/P.nFine:0.0,
              maxrow,defect,omega,lambdaMax);

  return P;
}

inline void sa_gate2_validate_p0(
    const SATransferHost&R,const SATransferHost&N,
    double refSeconds,double newSeconds){
  const bool dims=(R.nFine==N.nFine&&R.nCoarse==N.nCoarse);
  const bool rowExact=(dims&&R.row==N.row);
  const bool colExact=(rowExact&&R.col==N.col);
  const bool valueExact=(colExact&&R.val==N.val);

  long double d2=0.0L,r2=0.0L;
  double maxAbs=0.0;
  if(colExact&&R.val.size()==N.val.size()){
    for(std::size_t k=0;k<R.val.size();++k){
      const double d=N.val[k]-R.val[k];
      d2+=(long double)d*d;
      r2+=(long double)R.val[k]*R.val[k];
      maxAbs=std::max(maxAbs,std::abs(d));
    }
  }else{
    d2=std::numeric_limits<long double>::infinity();
  }
  const double rel=std::sqrt((double)d2)/
                   std::max(std::sqrt((double)r2),1e-300);

  const bool pass=dims&&rowExact&&colExact&&std::isfinite(rel)&&rel<=5e-15;

  std::printf("NODALS_GPU_SA_GATE2_P0_PARITY status=%s dimsExact=%d rowsExact=%d colsExact=%d valuesBitwiseExact=%d refRows=%d newRows=%d refCoarse=%d newCoarse=%d refNnz=%zu newNnz=%zu valueRelL2=%.12e valueMaxAbs=%.12e refSeconds=%.6f newSeconds=%.6f speedup=%.6f compareHoldsTwoP0=1 productionPeakMustUseModeParallel=1\n",
              pass?"PASS":"FAIL",(int)dims,(int)rowExact,(int)colExact,
              (int)valueExact,R.nFine,N.nFine,R.nCoarse,N.nCoarse,
              R.val.size(),N.val.size(),rel,maxAbs,
              refSeconds,newSeconds,refSeconds/std::max(newSeconds,1e-300));

  if(!pass)throw std::runtime_error("Gate2 P0 reference/parallel parity failed");
}

inline SATransferHost build_sa_fine_transfer_dispatch(
    const PressureSetupHost&S,const SAIncidenceHost&I,
    const AggResult&G,const std::vector<double>&diag,
    double lambdaMax,double damping,int maxNnz){
  const char*e=std::getenv("NODALS_P0_MODE");
  const char*mode=(e&&*e)?e:"parallel";

  if(std::strcmp(mode,"reference")==0)
    return build_sa_fine_transfer_reference(
      S,I,G,diag,lambdaMax,damping,maxNnz);

  if(std::strcmp(mode,"parallel")==0)
    return build_sa_fine_transfer_parallel(
      S,I,G,diag,lambdaMax,damping,maxNnz);

  if(std::strcmp(mode,"compare")==0){
    const auto t0=std::chrono::steady_clock::now();
    auto R=build_sa_fine_transfer_reference(
      S,I,G,diag,lambdaMax,damping,maxNnz);
    const auto t1=std::chrono::steady_clock::now();

    std::printf("NODALS_GPU_SA_GATE2_P0_REFERENCE_DONE seconds=%.6f rssMiB=%.3f hwmMiB=%.3f retainedNnz=%zu\n",
                sa_elapsed_s(t0,t1),sa_gate2_status_mib("VmRSS"),
                sa_gate2_status_mib("VmHWM"),R.val.size());

    const auto t2=std::chrono::steady_clock::now();
    auto N=build_sa_fine_transfer_parallel(
      S,I,G,diag,lambdaMax,damping,maxNnz);
    const auto t3=std::chrono::steady_clock::now();

    sa_gate2_validate_p0(
      R,N,sa_elapsed_s(t0,t1),sa_elapsed_s(t2,t3));
    return N;
  }

  throw std::runtime_error(
    "NODALS_P0_MODE must be reference, compare, or parallel");
}

inline SATransferHost build_sa_explicit_transfer_reference(const CSRHost&A,const AggResult&G,double lambdaMax,
                                                  double damping,int maxNnz,int level){
  const double omega=damping/lambdaMax;SATransferHost P;P.nFine=A.n;P.nCoarse=G.nagg;P.row.assign((std::size_t)A.n+1,0);P.col.reserve(A.n*3ull);P.val.reserve(A.n*3ull);
  std::vector<std::pair<int,double>>row;row.reserve(64);int maxrow=0;long double defect=0.0L;
  for(int i=0;i<A.n;++i){row.clear();int own=G.id[(std::size_t)i];row.push_back({own,1.0});double invd=1.0/A.diag[(std::size_t)i];
    for(std::int64_t k=A.row[(std::size_t)i];k<A.row[(std::size_t)i+1];++k)row.push_back({G.id[(std::size_t)A.col[(std::size_t)k]],-omega*invd*A.val[(std::size_t)k]});
    sa_normalize_prune(row,own,maxNnz);double sum=0.0;for(const auto&e:row){P.col.push_back(e.first);P.val.push_back(e.second);sum+=e.second;}P.row[(std::size_t)i+1]=(std::int64_t)P.col.size();maxrow=std::max(maxrow,(int)row.size());defect=std::max(defect,(long double)std::abs(sum-1.0));
  }
  std::printf("NODALS_GPU_SA_HOST_TRANSFER level=%d rows=%d coarse=%d nnz=%zu avgNnz=%.6f maxRowNnz=%d rowSumDefect=%.3Le omega=%.12e lambdaMax=%.12e status=PASS\n",
              level,P.nFine,P.nCoarse,P.val.size(),P.nFine?(double)P.val.size()/P.nFine:0.0,maxrow,defect,omega,lambdaMax);
  return P;
}

inline CSRHost sa_unordered_rows_to_csr(std::vector<std::unordered_map<int,double>>&rows){
  CSRHost A;A.n=(int)rows.size();A.row.assign((std::size_t)A.n+1,0);
  for(int i=0;i<A.n;++i)A.row[(std::size_t)i+1]=A.row[(std::size_t)i]+(std::int64_t)rows[(std::size_t)i].size();
  A.col.resize((std::size_t)A.row.back());A.val.resize((std::size_t)A.row.back());A.diag.assign((std::size_t)A.n,0.0);
  std::vector<std::pair<int,double>>tmp;
  for(int i=0;i<A.n;++i){tmp.clear();tmp.reserve(rows[(std::size_t)i].size());for(const auto&kv:rows[(std::size_t)i])if(kv.second!=0.0)tmp.push_back(kv);std::sort(tmp.begin(),tmp.end(),[](const auto&a,const auto&b){return a.first<b.first;});
    std::int64_t k=A.row[(std::size_t)i];for(const auto&e:tmp){A.col[(std::size_t)k]=e.first;A.val[(std::size_t)k]=e.second;if(e.first==i)A.diag[(std::size_t)i]=e.second;++k;}
    // rows may have exact-zero merged entries; compact if this occurs.
    A.row[(std::size_t)i+1]=k;
  }
  // Compact because exact zeros may have been skipped.
  std::int64_t write=0;
  std::vector<std::int64_t> nr((std::size_t)A.n+1,0);
  for(int i=0;i<A.n;++i){std::int64_t begin=A.row[(std::size_t)i],end=A.row[(std::size_t)i+1];nr[(std::size_t)i]=write;for(std::int64_t k=begin;k<end;++k){A.col[(std::size_t)write]=A.col[(std::size_t)k];A.val[(std::size_t)write]=A.val[(std::size_t)k];++write;}}
  nr[(std::size_t)A.n]=write;A.row.swap(nr);A.col.resize((std::size_t)write);A.val.resize((std::size_t)write);
  for(int i=0;i<A.n;++i)if(!(A.diag[(std::size_t)i]>0.0)||!std::isfinite(A.diag[(std::size_t)i]))throw std::runtime_error("SA coarse matrix non-positive diagonal");
  return A;
}

struct SAQEntry {int coarse=-1;double b[3]={0.0,0.0,0.0};};

inline CSRHost build_sa_first_coarse_reference(const PressureSetupHost&S,const SAIncidenceHost&I,const SATransferHost&P,unsigned long long*rawOut=nullptr){
  std::vector<std::unordered_map<int,double>> rows((std::size_t)P.nCoarse);
  std::vector<SAQEntry>q;q.reserve(64);unsigned long long raw=0;
  for(int g=0;g<S.free_vel;++g){
    q.clear();
    for(std::int64_t k=I.row[(std::size_t)g];k<I.row[(std::size_t)g+1];++k){
      int c=I.cell[(std::size_t)k],a=(int)I.basis[(std::size_t)k];const auto&cp=S.cells[(std::size_t)c];double bb[3]={coeff(cp,a,0),coeff(cp,a,1),coeff(cp,a,2)};
      for(std::int64_t kp=P.row[(std::size_t)c];kp<P.row[(std::size_t)c+1];++kp){double pv=P.val[(std::size_t)kp];SAQEntry e;e.coarse=P.col[(std::size_t)kp];for(int d=0;d<3;++d)e.b[d]=pv*bb[d];q.push_back(e);}
    }
    std::sort(q.begin(),q.end(),[](const auto&a,const auto&b){return a.coarse<b.coarse;});
    std::size_t w=0;
    for(std::size_t k=0;k<q.size();){int c=q[k].coarse;double bsum[3]={0,0,0};std::size_t j=k;for(;j<q.size()&&q[j].coarse==c;++j)for(int d=0;d<3;++d)bsum[d]+=q[j].b[d];q[w].coarse=c;for(int d=0;d<3;++d)q[w].b[d]=bsum[d];++w;k=j;}q.resize(w);
    double r=S.rAU[(std::size_t)g];
    for(std::size_t a=0;a<q.size();++a)for(std::size_t b=a;b<q.size();++b){double v=r*(q[a].b[0]*q[b].b[0]+q[a].b[1]*q[b].b[1]+q[a].b[2]*q[b].b[2]);if(v==0.0)continue;rows[(std::size_t)q[a].coarse][q[b].coarse]+=v;if(a!=b)rows[(std::size_t)q[b].coarse][q[a].coarse]+=v;raw+=(a==b)?1:2;}
  }
  auto A=sa_unordered_rows_to_csr(rows);if(rawOut)*rawOut=raw;std::printf("NODALS_GPU_SA_HOST_COARSE level=1 rows=%d nnz=%zu avgNnz=%.6f rawTerms=%llu symmetryRel=%.3e status=PASS\n",A.n,A.val.size(),A.n?(double)A.val.size()/A.n:0.0,raw,csr_symmetry_rel(A));return A;
}

// -----------------------------------------------------------------------------
// Gate 1A: same factorized P^T B diag(rAU) B^T P operator as the legacy A1
// builder, but with multicore row ownership and thread-local dense accumulation.
// -----------------------------------------------------------------------------

inline int sa_gate1a_threads(){
  int n=0;
  if(const char*e=std::getenv("NODALS_A1_THREADS")) n=std::atoi(e);
  if(n<=0){unsigned h=std::thread::hardware_concurrency();n=h?std::min((int)h,16):1;}
  return std::max(1,std::min(n,64));
}

inline bool sa_gate1a_symmetry_audit_enabled(){
  const char*e=std::getenv("NODALS_A1_SYMMETRY_AUDIT");
  if(!e||!*e)return false;
  return std::strcmp(e,"0")!=0 &&
         std::strcmp(e,"false")!=0 &&
         std::strcmp(e,"FALSE")!=0 &&
         std::strcmp(e,"off")!=0 &&
         std::strcmp(e,"OFF")!=0;
}

inline double sa_gate1a_status_mib(const char*key){
  FILE*f=std::fopen("/proc/self/status","r");
  if(!f)return -1.0;
  char line[256];double out=-1.0;
  const std::size_t n=std::strlen(key);
  while(std::fgets(line,sizeof(line),f)){
    if(std::strncmp(line,key,n)==0){
      unsigned long long kb=0;
      if(std::sscanf(line+n,": %llu kB",&kb)==1)out=(double)kb/1024.0;
      break;
    }
  }
  std::fclose(f);return out;
}

template<class F>
inline void sa_gate1a_parallel_chunks(int n,int nth,int chunk,F&&fn){
  if(n<=0)return;
  nth=std::max(1,std::min(nth,n));
  std::atomic<int> next{0};
  std::atomic<bool> failed{false};
  std::exception_ptr ep;
  std::mutex em;
  std::vector<std::thread> pool;
  pool.reserve((std::size_t)nth);
  for(int t=0;t<nth;++t){
    pool.emplace_back([&,t](){
      try{
        while(!failed.load(std::memory_order_relaxed)){
          const int b=next.fetch_add(chunk,std::memory_order_relaxed);
          if(b>=n)break;
          fn(t,b,std::min(n,b+chunk));
        }
      }catch(...){
        failed.store(true,std::memory_order_relaxed);
        std::lock_guard<std::mutex>g(em);
        if(!ep)ep=std::current_exception();
      }
    });
  }
  for(auto&th:pool)th.join();
  if(ep)std::rethrow_exception(ep);
}

inline void sa_gate1a_make_q(
    const PressureSetupHost&S,const SAIncidenceHost&I,const SATransferHost&P,
    int g,std::vector<SAQEntry>&q){
  q.clear();
  for(std::int64_t k=I.row[(std::size_t)g];k<I.row[(std::size_t)g+1];++k){
    const int c=I.cell[(std::size_t)k],a=(int)I.basis[(std::size_t)k];
    const auto&cp=S.cells[(std::size_t)c];
    const double bb[3]={coeff(cp,a,0),coeff(cp,a,1),coeff(cp,a,2)};
    for(std::int64_t kp=P.row[(std::size_t)c];kp<P.row[(std::size_t)c+1];++kp){
      const double pv=P.val[(std::size_t)kp];
      SAQEntry e;e.coarse=P.col[(std::size_t)kp];
      for(int d=0;d<3;++d)e.b[d]=pv*bb[d];
      q.push_back(e);
    }
  }
  std::sort(q.begin(),q.end(),[](const auto&a,const auto&b){return a.coarse<b.coarse;});
  std::size_t w=0;
  for(std::size_t k=0;k<q.size();){
    const int c=q[k].coarse;double bsum[3]={0,0,0};std::size_t j=k;
    for(;j<q.size()&&q[j].coarse==c;++j)
      for(int d=0;d<3;++d)bsum[d]+=q[j].b[d];
    q[w].coarse=c;
    for(int d=0;d<3;++d)q[w].b[d]=bsum[d];
    ++w;k=j;
  }
  q.resize(w);
}

struct SAQCompactGate1A{
  std::vector<std::int64_t> row;
  std::vector<std::int32_t> col;
  std::vector<double> b0,b1,b2;
};

inline CSRHost build_sa_first_coarse_parallel(
    const PressureSetupHost&S,const SAIncidenceHost&I,const SATransferHost&P,
    unsigned long long*rawOut=nullptr){
  using clock=std::chrono::steady_clock;
  const auto all0=clock::now();
  const int nth=sa_gate1a_threads();
  const int nv=S.free_vel,nc=P.nCoarse;
  if(nv<0||nc<=0)throw std::runtime_error("Gate1A invalid A1 dimensions");

  std::printf("NODALS_GPU_SA_GATE1A_STAGE stage=q_compress state=START threads=%d rssMiB=%.3f hwmMiB=%.3f\n",
              nth,sa_gate1a_status_mib("VmRSS"),sa_gate1a_status_mib("VmHWM"));
  std::fflush(stdout);

  SAQCompactGate1A Q;
  Q.row.assign((std::size_t)nv+1,0);
  std::vector<std::vector<SAQEntry>> qs((std::size_t)nth);
  for(auto&x:qs)x.reserve(128);

  sa_gate1a_parallel_chunks(nv,nth,1024,[&](int t,int b,int e){
    auto&q=qs[(std::size_t)t];
    for(int g=b;g<e;++g){
      sa_gate1a_make_q(S,I,P,g,q);
      Q.row[(std::size_t)g+1]=(std::int64_t)q.size();
    }
  });
  for(int g=0;g<nv;++g)Q.row[(std::size_t)g+1]+=Q.row[(std::size_t)g];
  if(Q.row.back()<0)throw std::runtime_error("Gate1A negative q nnz");
  const std::size_t qnnz=(std::size_t)Q.row.back();

  Q.col.resize(qnnz);Q.b0.resize(qnnz);Q.b1.resize(qnnz);Q.b2.resize(qnnz);

  sa_gate1a_parallel_chunks(nv,nth,512,[&](int t,int b,int e){
    auto&q=qs[(std::size_t)t];
    for(int g=b;g<e;++g){
      sa_gate1a_make_q(S,I,P,g,q);
      std::int64_t out=Q.row[(std::size_t)g];
      const std::int64_t end=Q.row[(std::size_t)g+1];
      if(end-out!=(std::int64_t)q.size())throw std::runtime_error("Gate1A q count/fill mismatch");
      for(const auto&z:q){
        Q.col[(std::size_t)out]=(std::int32_t)z.coarse;
        Q.b0[(std::size_t)out]=z.b[0];
        Q.b1[(std::size_t)out]=z.b[1];
        Q.b2[(std::size_t)out]=z.b[2];
        ++out;
      }
    }
  });
  std::vector<std::vector<SAQEntry>>().swap(qs);

  int qmax=0;
  for(int g=0;g<nv;++g)
    qmax=std::max(qmax,(int)(Q.row[(std::size_t)g+1]-Q.row[(std::size_t)g]));
  const auto q1=clock::now();
  std::printf("NODALS_GPU_SA_GATE1A_Q status=PASS velocities=%d qNnz=%zu avgSupport=%.6f maxSupport=%d seconds=%.6f rssMiB=%.3f hwmMiB=%.3f\n",
              nv,qnnz,nv?(double)qnnz/(double)nv:0.0,qmax,sa_elapsed_s(all0,q1),
              sa_gate1a_status_mib("VmRSS"),sa_gate1a_status_mib("VmHWM"));
  std::fflush(stdout);

  std::printf("NODALS_GPU_SA_GATE1A_STAGE stage=reverse_incidence state=START\n");
  std::fflush(stdout);
  const auto r0=clock::now();
  std::vector<std::int64_t> rptr((std::size_t)nc+1,0);
  for(std::size_t k=0;k<qnnz;++k){
    const int c=(int)Q.col[k];
    if(c<0||c>=nc)throw std::runtime_error("Gate1A q coarse column out of range");
    ++rptr[(std::size_t)c+1];
  }
  for(int i=0;i<nc;++i)rptr[(std::size_t)i+1]+=rptr[(std::size_t)i];

  std::vector<std::int32_t> rvel(qnnz);
  std::vector<std::int64_t> rq(qnnz);
  auto rcur=rptr;
  for(int g=0;g<nv;++g){
    for(std::int64_t k=Q.row[(std::size_t)g];k<Q.row[(std::size_t)g+1];++k){
      const int c=(int)Q.col[(std::size_t)k];
      const std::int64_t p=rcur[(std::size_t)c]++;
      rvel[(std::size_t)p]=(std::int32_t)g;
      rq[(std::size_t)p]=k;
    }
  }
  std::vector<std::int64_t>().swap(rcur);
  const auto r1=clock::now();
  std::printf("NODALS_GPU_SA_GATE1A_REVERSE status=PASS entries=%zu seconds=%.6f rssMiB=%.3f hwmMiB=%.3f\n",
              qnnz,sa_elapsed_s(r0,r1),sa_gate1a_status_mib("VmRSS"),sa_gate1a_status_mib("VmHWM"));
  std::fflush(stdout);

  std::printf("NODALS_GPU_SA_GATE1A_STAGE stage=row_accumulate state=START threads=%d\n",nth);
  std::fflush(stdout);
  const auto a0=clock::now();
  std::vector<std::vector<std::pair<int,double>>> rows((std::size_t)nc);
  std::vector<std::vector<double>> acc((std::size_t)nth);
  std::vector<std::vector<int>> mark((std::size_t)nth);
  std::vector<std::vector<int>> touched((std::size_t)nth);
  for(int t=0;t<nth;++t){
    acc[(std::size_t)t].resize((std::size_t)nc);
    mark[(std::size_t)t].assign((std::size_t)nc,-1);
    touched[(std::size_t)t].reserve(512);
  }
  std::vector<unsigned long long> rawThread((std::size_t)nth,0);

  sa_gate1a_parallel_chunks(nc,nth,32,[&](int t,int ib,int ie){
    auto&av=acc[(std::size_t)t];
    auto&mk=mark[(std::size_t)t];
    auto&tv=touched[(std::size_t)t];
    unsigned long long raw=0;
    for(int i=ib;i<ie;++i){
      tv.clear();
      for(std::int64_t rp=rptr[(std::size_t)i];rp<rptr[(std::size_t)i+1];++rp){
        const int g=(int)rvel[(std::size_t)rp];
        const std::int64_t qi=rq[(std::size_t)rp];
        const double bi0=Q.b0[(std::size_t)qi],bi1=Q.b1[(std::size_t)qi],bi2=Q.b2[(std::size_t)qi];
        const double rau=S.rAU[(std::size_t)g];
        for(std::int64_t qj=Q.row[(std::size_t)g];qj<Q.row[(std::size_t)g+1];++qj){
          const int c=(int)Q.col[(std::size_t)qj];
          const double v=rau*(bi0*Q.b0[(std::size_t)qj]+bi1*Q.b1[(std::size_t)qj]+bi2*Q.b2[(std::size_t)qj]);
          if(v==0.0)continue;
          if(mk[(std::size_t)c]!=i){
            mk[(std::size_t)c]=i;av[(std::size_t)c]=v;tv.push_back(c);
          }else av[(std::size_t)c]+=v;
          ++raw;
        }
      }
      std::sort(tv.begin(),tv.end());
      auto&rr=rows[(std::size_t)i];
      rr.clear();rr.reserve(tv.size());
      for(int c:tv){
        const double v=av[(std::size_t)c];
        if(v!=0.0)rr.push_back({c,v});
      }
    }
    rawThread[(std::size_t)t]+=raw;
  });

  unsigned long long raw=0;
  for(auto x:rawThread)raw+=x;
  std::vector<unsigned long long>().swap(rawThread);

  std::size_t rowPayload=0;
  for(const auto&r:rows)rowPayload+=r.size();
  const auto a1=clock::now();
  std::printf("NODALS_GPU_SA_GATE1A_ROWS status=PASS rows=%d retainedPairs=%zu rawTerms=%llu seconds=%.6f rssMiB=%.3f hwmMiB=%.3f\n",
              nc,rowPayload,raw,sa_elapsed_s(a0,a1),sa_gate1a_status_mib("VmRSS"),sa_gate1a_status_mib("VmHWM"));
  std::fflush(stdout);

  const double scratchEstMiB=
    (double)(
      Q.row.size()*sizeof(std::int64_t)+Q.col.size()*sizeof(std::int32_t)+
      (Q.b0.size()+Q.b1.size()+Q.b2.size())*sizeof(double)+
      rptr.size()*sizeof(std::int64_t)+rvel.size()*sizeof(std::int32_t)+rq.size()*sizeof(std::int64_t)+
      (std::size_t)nth*(std::size_t)nc*(sizeof(double)+sizeof(int))+
      rowPayload*sizeof(std::pair<int,double>)
    )/(1024.0*1024.0);

  std::vector<std::int64_t>().swap(Q.row);
  std::vector<std::int32_t>().swap(Q.col);
  std::vector<double>().swap(Q.b0);
  std::vector<double>().swap(Q.b1);
  std::vector<double>().swap(Q.b2);
  std::vector<std::int64_t>().swap(rptr);
  std::vector<std::int32_t>().swap(rvel);
  std::vector<std::int64_t>().swap(rq);
  std::vector<std::vector<double>>().swap(acc);
  std::vector<std::vector<int>>().swap(mark);
  std::vector<std::vector<int>>().swap(touched);

  std::printf("NODALS_GPU_SA_GATE1A_STAGE stage=csr_pack state=START rssMiB=%.3f hwmMiB=%.3f\n",
              sa_gate1a_status_mib("VmRSS"),sa_gate1a_status_mib("VmHWM"));
  std::fflush(stdout);
  const auto p0=clock::now();

  CSRHost A;A.n=nc;A.row.assign((std::size_t)nc+1,0);
  for(int i=0;i<nc;++i)
    A.row[(std::size_t)i+1]=A.row[(std::size_t)i]+(std::int64_t)rows[(std::size_t)i].size();
  A.col.resize((std::size_t)A.row.back());
  A.val.resize((std::size_t)A.row.back());
  A.diag.assign((std::size_t)nc,0.0);

  sa_gate1a_parallel_chunks(nc,nth,256,[&](int,int ib,int ie){
    for(int i=ib;i<ie;++i){
      std::int64_t k=A.row[(std::size_t)i];
      for(const auto&e:rows[(std::size_t)i]){
        A.col[(std::size_t)k]=e.first;
        A.val[(std::size_t)k]=e.second;
        if(e.first==i)A.diag[(std::size_t)i]=e.second;
        ++k;
      }
    }
  });
  std::vector<std::vector<std::pair<int,double>>>().swap(rows);

  for(int i=0;i<nc;++i)
    if(!(A.diag[(std::size_t)i]>0.0)||!std::isfinite(A.diag[(std::size_t)i]))
      throw std::runtime_error("Gate1A coarse matrix non-positive diagonal");

  const auto p1=clock::now();
  const auto all1=clock::now();
  if(rawOut)*rawOut=raw;

  const bool symmetryAudit=sa_gate1a_symmetry_audit_enabled();
  const double symmetryRel=symmetryAudit?csr_symmetry_rel(A):-1.0;
  std::printf("NODALS_GPU_SA_GATE1A_NEW status=PASS rows=%d nnz=%zu avgNnz=%.6f rawTerms=%llu threads=%d qNnz=%zu qAvg=%.6f qMax=%d qSeconds=%.6f reverseSeconds=%.6f accumulateSeconds=%.6f packSeconds=%.6f totalSeconds=%.6f scratchEstimateMiB=%.3f symmetryAudit=%s symmetryRel=%.3e rssMiB=%.3f hwmMiB=%.3f\n",
              A.n,A.val.size(),A.n?(double)A.val.size()/A.n:0.0,raw,nth,qnnz,
              nv?(double)qnnz/(double)nv:0.0,qmax,sa_elapsed_s(all0,q1),sa_elapsed_s(r0,r1),
              sa_elapsed_s(a0,a1),sa_elapsed_s(p0,p1),sa_elapsed_s(all0,all1),scratchEstMiB,
              symmetryAudit?"ON":"OFF",symmetryRel,sa_gate1a_status_mib("VmRSS"),sa_gate1a_status_mib("VmHWM"));
  return A;
}

inline void sa_gate1a_validate(const CSRHost&R,const CSRHost&N,
                               unsigned long long rawR,unsigned long long rawN,
                               double refSeconds,double newSeconds){
  const bool rowExact=(R.n==N.n && R.row==N.row);
  const bool colExact=(rowExact && R.col==N.col);
  long double d2=0.0L,r2=0.0L,dd2=0.0L,dr2=0.0L;
  double maxAbs=0.0,diagMax=0.0;

  if(colExact && R.val.size()==N.val.size()){
    for(std::size_t k=0;k<R.val.size();++k){
      const double d=N.val[k]-R.val[k];
      d2+=(long double)d*d;
      r2+=(long double)R.val[k]*R.val[k];
      maxAbs=std::max(maxAbs,std::abs(d));
    }
    for(std::size_t i=0;i<R.diag.size();++i){
      const double d=N.diag[i]-R.diag[i];
      dd2+=(long double)d*d;
      dr2+=(long double)R.diag[i]*R.diag[i];
      diagMax=std::max(diagMax,std::abs(d));
    }
  }else{
    d2=dd2=std::numeric_limits<long double>::infinity();
  }

  const double valRel=std::sqrt((double)d2)/std::max(std::sqrt((double)r2),1e-300);
  const double diagRel=std::sqrt((double)dd2)/std::max(std::sqrt((double)dr2),1e-300);

  double actionRel=std::numeric_limits<double>::infinity();
  if(colExact){
    std::vector<double>x((std::size_t)R.n),yr,yn;
    for(int i=0;i<R.n;++i){
      const double g=(double)(i+1);
      x[(std::size_t)i]=std::sin(.017*g)+.31*std::cos(.011*g);
    }
    sa_csr_apply(R,x,yr);
    sa_csr_apply(N,x,yn);
    long double ad=0.0L,ar=0.0L;
    for(int i=0;i<R.n;++i){
      const double d=yn[(std::size_t)i]-yr[(std::size_t)i];
      ad+=(long double)d*d;
      ar+=(long double)yr[(std::size_t)i]*yr[(std::size_t)i];
    }
    actionRel=std::sqrt((double)ad)/std::max(std::sqrt((double)ar),1e-300);
  }

  const double sym=csr_symmetry_rel(N);
  const bool pass=rowExact&&colExact&&rawR==rawN&&std::isfinite(valRel)&&
                  valRel<=5e-12&&diagRel<=5e-12&&actionRel<=5e-12&&sym<=5e-12;

  std::printf("NODALS_GPU_SA_GATE1A_PARITY status=%s rowsExact=%d colsExact=%d refRows=%d newRows=%d refNnz=%zu newNnz=%zu rawRef=%llu rawNew=%llu rawExact=%d valueRelL2=%.12e valueMaxAbs=%.12e diagRelL2=%.12e diagMaxAbs=%.12e actionRelL2=%.12e symmetryRel=%.12e refSeconds=%.6f newSeconds=%.6f speedup=%.6f compareHoldsTwoA1=1 productionPeakMustUseModeParallel=1\n",
              pass?"PASS":"FAIL",(int)rowExact,(int)colExact,R.n,N.n,R.val.size(),N.val.size(),
              rawR,rawN,(int)(rawR==rawN),valRel,maxAbs,diagRel,diagMax,actionRel,sym,
              refSeconds,newSeconds,refSeconds/std::max(newSeconds,1e-300));

  if(!pass)throw std::runtime_error("Gate1A A1 reference/parallel parity failed");
}

inline CSRHost build_sa_first_coarse_dispatch(
    const PressureSetupHost&S,const SAIncidenceHost&I,const SATransferHost&P){
  const char*e=std::getenv("NODALS_A1_MODE");
  const char*mode=(e&&*e)?e:"parallel";

  if(std::strcmp(mode,"reference")==0)
    return build_sa_first_coarse_reference(S,I,P);

  if(std::strcmp(mode,"parallel")==0)
    return build_sa_first_coarse_parallel(S,I,P);

  if(std::strcmp(mode,"compare")==0){
    unsigned long long rr=0,rn=0;
    const auto t0=std::chrono::steady_clock::now();
    auto R=build_sa_first_coarse_reference(S,I,P,&rr);
    const auto t1=std::chrono::steady_clock::now();

    std::printf("NODALS_GPU_SA_GATE1A_REFERENCE_DONE seconds=%.6f rssMiB=%.3f hwmMiB=%.3f retainedNnz=%zu\n",
                sa_elapsed_s(t0,t1),sa_gate1a_status_mib("VmRSS"),sa_gate1a_status_mib("VmHWM"),R.val.size());

    const auto t2=std::chrono::steady_clock::now();
    auto N=build_sa_first_coarse_parallel(S,I,P,&rn);
    const auto t3=std::chrono::steady_clock::now();

    sa_gate1a_validate(R,N,rr,rn,sa_elapsed_s(t0,t1),sa_elapsed_s(t2,t3));
    return N;
  }

  throw std::runtime_error("NODALS_A1_MODE must be reference, compare, or parallel");
}

inline CSRHost build_sa_ptap_reference(const CSRHost&A,const SATransferHost&P,int nextLevel){
  if(P.nFine!=A.n)throw std::runtime_error("SA PtAP transfer size mismatch");
  std::vector<std::unordered_map<int,double>> rows((std::size_t)P.nCoarse);
  std::vector<std::pair<int,double>>q;q.reserve(128);unsigned long long raw=0;
  for(int i=0;i<A.n;++i){
    q.clear();
    for(std::int64_t k=A.row[(std::size_t)i];k<A.row[(std::size_t)i+1];++k){
      int j=A.col[(std::size_t)k];double a=A.val[(std::size_t)k];
      for(std::int64_t kp=P.row[(std::size_t)j];kp<P.row[(std::size_t)j+1];++kp)q.push_back({P.col[(std::size_t)kp],a*P.val[(std::size_t)kp]});
    }
    std::sort(q.begin(),q.end(),[](const auto&a,const auto&b){return a.first<b.first;});std::size_t w=0;
    for(std::size_t k=0;k<q.size();){int c=q[k].first;double v=0.0;std::size_t j=k;for(;j<q.size()&&q[j].first==c;++j)v+=q[j].second;q[w++]={c,v};k=j;}q.resize(w);
    for(std::int64_t kp=P.row[(std::size_t)i];kp<P.row[(std::size_t)i+1];++kp){int I=P.col[(std::size_t)kp];double pi=P.val[(std::size_t)kp];for(const auto&e:q){double v=pi*e.second;if(v!=0.0){rows[(std::size_t)I][e.first]+=v;++raw;}}}
  }
  auto C=sa_unordered_rows_to_csr(rows);std::printf("NODALS_GPU_SA_HOST_COARSE level=%d rows=%d nnz=%zu avgNnz=%.6f rawTerms=%llu symmetryRel=%.3e status=PASS\n",nextLevel,C.n,C.val.size(),C.n?(double)C.val.size()/C.n:0.0,raw,csr_symmetry_rel(C));return C;
}

// -----------------------------------------------------------------------------
// Gate 3: recursive coarse SA transfer + PtAP, multicore host implementation.
//
// Transfer:
//   Each fine row is independent and the pruned row width is bounded by the
//   existing interpolationMaxNnz. Compute once into fixed-width row slots,
//   prefix-scan counts, compact to SATransferHost.
//
// PtAP:
//   q_i = sum_j A_ij P_j is formed once per fine row using the exact legacy
//   local sort/merge arithmetic. Reverse P incidence maps each coarse output
//   row I to the fine rows i carrying P_iI. Workers own output rows, therefore
//   numerical accumulation is lock-free. Reverse incidence is filled in
//   increasing fine-row order to preserve the legacy per-(I,K) += order.
// -----------------------------------------------------------------------------

inline int sa_gate3_threads(){
  int n=0;
  if(const char*e=std::getenv("NODALS_RECURSIVE_THREADS"))n=std::atoi(e);
  if(n<=0){unsigned h=std::thread::hardware_concurrency();n=h?std::min((int)h,16):1;}
  return std::max(1,std::min(n,64));
}

inline SATransferHost build_sa_explicit_transfer_parallel(
    const CSRHost&A,const AggResult&G,double lambdaMax,
    double damping,int maxNnz,int level){
  using clock=std::chrono::steady_clock;
  const auto all0=clock::now();
  if(maxNnz<=0||maxNnz>255)
    throw std::runtime_error("Gate3 recursive transfer requires 1<=maxNnz<=255");

  const int nth=sa_gate3_threads();
  const double omega=damping/lambdaMax;

  SATransferHost P;
  P.nFine=A.n;
  P.nCoarse=G.nagg;

  const std::size_t nf=(std::size_t)P.nFine;
  const std::size_t width=(std::size_t)maxNnz;
  const std::size_t slots=nf*width;

  std::vector<std::uint8_t> count(nf,0);
  std::vector<std::int32_t> slotCol(slots);
  std::vector<double> slotVal(slots);
  std::vector<std::vector<std::pair<int,double>>> work((std::size_t)nth);
  for(auto&r:work)r.reserve(256);
  std::vector<int> threadMax((std::size_t)nth,0);
  std::vector<long double> threadDef((std::size_t)nth,0.0L);

  const auto b0=clock::now();
  sa_gate1a_parallel_chunks(P.nFine,nth,128,[&](int t,int ib,int ie){
    auto&row=work[(std::size_t)t];
    int lmax=0;
    long double ldef=0.0L;

    for(int i=ib;i<ie;++i){
      row.clear();
      const int own=G.id[(std::size_t)i];
      row.push_back({own,1.0});
      const double invd=1.0/A.diag[(std::size_t)i];

      for(std::int64_t k=A.row[(std::size_t)i];k<A.row[(std::size_t)i+1];++k)
        row.push_back({
          G.id[(std::size_t)A.col[(std::size_t)k]],
          -omega*invd*A.val[(std::size_t)k]});

      sa_normalize_prune(row,own,maxNnz);
      if(row.empty()||row.size()>width)
        throw std::runtime_error("Gate3 recursive transfer row exceeds fixed width");

      count[(std::size_t)i]=(std::uint8_t)row.size();
      const std::size_t base=(std::size_t)i*width;
      double sum=0.0;
      for(std::size_t k=0;k<row.size();++k){
        slotCol[base+k]=(std::int32_t)row[k].first;
        slotVal[base+k]=row[k].second;
        sum+=row[k].second;
      }
      lmax=std::max(lmax,(int)row.size());
      ldef=std::max(ldef,(long double)std::abs(sum-1.0));
    }

    threadMax[(std::size_t)t]=std::max(threadMax[(std::size_t)t],lmax);
    threadDef[(std::size_t)t]=std::max(threadDef[(std::size_t)t],ldef);
  });
  const auto b1=clock::now();

  int maxrow=0;
  long double defect=0.0L;
  for(int t=0;t<nth;++t){
    maxrow=std::max(maxrow,threadMax[(std::size_t)t]);
    defect=std::max(defect,threadDef[(std::size_t)t]);
  }
  std::vector<std::vector<std::pair<int,double>>>().swap(work);
  std::vector<int>().swap(threadMax);
  std::vector<long double>().swap(threadDef);

  const auto p0=clock::now();
  P.row.assign(nf+1,0);
  for(int i=0;i<P.nFine;++i)
    P.row[(std::size_t)i+1]=P.row[(std::size_t)i]+
                            (std::int64_t)count[(std::size_t)i];
  P.col.resize((std::size_t)P.row.back());
  P.val.resize((std::size_t)P.row.back());
  const auto p1=clock::now();

  const auto c0=clock::now();
  sa_gate1a_parallel_chunks(P.nFine,nth,512,[&](int,int ib,int ie){
    for(int i=ib;i<ie;++i){
      const std::size_t src=(std::size_t)i*width;
      std::int64_t dst=P.row[(std::size_t)i];
      const int n=(int)count[(std::size_t)i];
      for(int k=0;k<n;++k){
        P.col[(std::size_t)dst]=slotCol[src+(std::size_t)k];
        P.val[(std::size_t)dst]=slotVal[src+(std::size_t)k];
        ++dst;
      }
    }
  });
  const auto c1=clock::now();

  const double scratchMiB=
    (double)(count.size()*sizeof(std::uint8_t)+
             slotCol.size()*sizeof(std::int32_t)+
             slotVal.size()*sizeof(double))/(1024.0*1024.0);

  std::vector<std::uint8_t>().swap(count);
  std::vector<std::int32_t>().swap(slotCol);
  std::vector<double>().swap(slotVal);

  const auto all1=clock::now();

  std::printf(
    "NODALS_GPU_SA_GATE3_TRANSFER_NEW level=%d status=PASS rows=%d coarse=%d nnz=%zu avgNnz=%.6f maxRowNnz=%d rowSumDefect=%.3Le threads=%d buildSeconds=%.6f scanAllocSeconds=%.6f compactSeconds=%.6f totalSeconds=%.6f scratchMiB=%.3f omega=%.12e lambdaMax=%.12e rssMiB=%.3f hwmMiB=%.3f\n",
    level,P.nFine,P.nCoarse,P.val.size(),
    P.nFine?(double)P.val.size()/P.nFine:0.0,
    maxrow,defect,nth,
    sa_elapsed_s(b0,b1),sa_elapsed_s(p0,p1),
    sa_elapsed_s(c0,c1),sa_elapsed_s(all0,all1),
    scratchMiB,omega,lambdaMax,
    sa_gate1a_status_mib("VmRSS"),sa_gate1a_status_mib("VmHWM"));

  return P;
}

inline void sa_gate3_validate_transfer(
    const SATransferHost&R,const SATransferHost&N,int level,
    double refSeconds,double newSeconds){
  const bool dims=(R.nFine==N.nFine&&R.nCoarse==N.nCoarse);
  const bool rowExact=dims&&R.row==N.row;
  const bool colExact=rowExact&&R.col==N.col;
  const bool valueExact=colExact&&R.val==N.val;

  long double d2=0.0L,r2=0.0L;
  double maxAbs=0.0;
  if(colExact&&R.val.size()==N.val.size()){
    for(std::size_t k=0;k<R.val.size();++k){
      const double d=N.val[k]-R.val[k];
      d2+=(long double)d*d;
      r2+=(long double)R.val[k]*R.val[k];
      maxAbs=std::max(maxAbs,std::abs(d));
    }
  }else d2=std::numeric_limits<long double>::infinity();

  const double rel=std::sqrt((double)d2)/
                   std::max(std::sqrt((double)r2),1e-300);
  const bool pass=dims&&rowExact&&colExact&&std::isfinite(rel)&&rel<=5e-15;

  std::printf(
    "NODALS_GPU_SA_GATE3_TRANSFER_PARITY level=%d status=%s dimsExact=%d rowsExact=%d colsExact=%d valuesBitwiseExact=%d refNnz=%zu newNnz=%zu valueRelL2=%.12e valueMaxAbs=%.12e refSeconds=%.6f newSeconds=%.6f speedup=%.6f\n",
    level,pass?"PASS":"FAIL",(int)dims,(int)rowExact,(int)colExact,
    (int)valueExact,R.val.size(),N.val.size(),rel,maxAbs,
    refSeconds,newSeconds,refSeconds/std::max(newSeconds,1e-300));

  if(!pass)throw std::runtime_error("Gate3 recursive transfer parity failed");
}

struct SAGate3QRow{
  std::vector<std::pair<int,double>> e;
};

struct SAGate3Rev{
  std::int32_t fine=-1;
  double p=0.0;
};

inline CSRHost build_sa_ptap_parallel(
    const CSRHost&A,const SATransferHost&P,int nextLevel,
    unsigned long long*rawOut=nullptr){
  using clock=std::chrono::steady_clock;
  if(P.nFine!=A.n)throw std::runtime_error("Gate3 PtAP transfer size mismatch");

  const auto all0=clock::now();
  const int nth=sa_gate3_threads();
  const int nf=A.n,nc=P.nCoarse;

  std::printf(
    "NODALS_GPU_SA_GATE3_STAGE level=%d stage=q_rows state=START fineRows=%d coarseRows=%d threads=%d rssMiB=%.3f hwmMiB=%.3f\n",
    nextLevel,nf,nc,nth,
    sa_gate1a_status_mib("VmRSS"),sa_gate1a_status_mib("VmHWM"));
  std::fflush(stdout);

  // Exact legacy q_i arithmetic, one independent row per worker.
  std::vector<SAGate3QRow> qrows((std::size_t)nf);
  const auto q0=clock::now();
  sa_gate1a_parallel_chunks(nf,nth,32,[&](int,int ib,int ie){
    std::vector<std::pair<int,double>>q;
    q.reserve(512);

    for(int i=ib;i<ie;++i){
      q.clear();
      for(std::int64_t k=A.row[(std::size_t)i];k<A.row[(std::size_t)i+1];++k){
        const int j=A.col[(std::size_t)k];
        const double a=A.val[(std::size_t)k];
        for(std::int64_t kp=P.row[(std::size_t)j];kp<P.row[(std::size_t)j+1];++kp)
          q.push_back({P.col[(std::size_t)kp],a*P.val[(std::size_t)kp]});
      }

      std::sort(q.begin(),q.end(),[](const auto&a,const auto&b){
        return a.first<b.first;
      });

      std::size_t w=0;
      for(std::size_t k=0;k<q.size();){
        const int c=q[k].first;
        double v=0.0;
        std::size_t j=k;
        for(;j<q.size()&&q[j].first==c;++j)v+=q[j].second;
        q[w++]={c,v};
        k=j;
      }
      q.resize(w);
      qrows[(std::size_t)i].e=q;
    }
  });
  const auto q1=clock::now();

  std::size_t qnnz=0;
  int qmax=0;
  for(const auto&r:qrows){
    qnnz+=r.e.size();
    qmax=std::max(qmax,(int)r.e.size());
  }

  std::printf(
    "NODALS_GPU_SA_GATE3_Q level=%d status=PASS qNnz=%zu qAvg=%.6f qMax=%d seconds=%.6f rssMiB=%.3f hwmMiB=%.3f\n",
    nextLevel,qnnz,nf?(double)qnnz/nf:0.0,qmax,
    sa_elapsed_s(q0,q1),
    sa_gate1a_status_mib("VmRSS"),sa_gate1a_status_mib("VmHWM"));
  std::fflush(stdout);

  // Reverse P incidence. Serial fill preserves increasing fine-row order.
  const auto r0=clock::now();
  std::vector<std::int64_t> rptr((std::size_t)nc+1,0);
  for(int i=0;i<nf;++i)
    for(std::int64_t kp=P.row[(std::size_t)i];kp<P.row[(std::size_t)i+1];++kp)
      ++rptr[(std::size_t)P.col[(std::size_t)kp]+1];

  for(int I=0;I<nc;++I)
    rptr[(std::size_t)I+1]+=rptr[(std::size_t)I];

  std::vector<SAGate3Rev> rev((std::size_t)rptr.back());
  auto cur=rptr;
  for(int i=0;i<nf;++i){
    for(std::int64_t kp=P.row[(std::size_t)i];kp<P.row[(std::size_t)i+1];++kp){
      const int I=P.col[(std::size_t)kp];
      const std::int64_t z=cur[(std::size_t)I]++;
      rev[(std::size_t)z].fine=(std::int32_t)i;
      rev[(std::size_t)z].p=P.val[(std::size_t)kp];
    }
  }
  std::vector<std::int64_t>().swap(cur);
  const auto r1=clock::now();

  std::printf(
    "NODALS_GPU_SA_GATE3_REVERSE level=%d status=PASS entries=%zu seconds=%.6f rssMiB=%.3f hwmMiB=%.3f\n",
    nextLevel,rev.size(),sa_elapsed_s(r0,r1),
    sa_gate1a_status_mib("VmRSS"),sa_gate1a_status_mib("VmHWM"));
  std::fflush(stdout);

  // Output coarse rows are independent.
  const auto a0=clock::now();
  std::vector<std::vector<std::pair<int,double>>> rows((std::size_t)nc);
  std::vector<std::vector<double>> acc((std::size_t)nth);
  std::vector<std::vector<int>> mark((std::size_t)nth);
  std::vector<std::vector<int>> touched((std::size_t)nth);
  for(int t=0;t<nth;++t){
    acc[(std::size_t)t].resize((std::size_t)nc);
    mark[(std::size_t)t].assign((std::size_t)nc,-1);
    touched[(std::size_t)t].reserve(512);
  }
  std::vector<unsigned long long> rawThread((std::size_t)nth,0);

  sa_gate1a_parallel_chunks(nc,nth,16,[&](int t,int ib,int ie){
    auto&av=acc[(std::size_t)t];
    auto&mk=mark[(std::size_t)t];
    auto&tv=touched[(std::size_t)t];
    unsigned long long raw=0;

    for(int I=ib;I<ie;++I){
      tv.clear();

      for(std::int64_t z=rptr[(std::size_t)I];z<rptr[(std::size_t)I+1];++z){
        const int i=(int)rev[(std::size_t)z].fine;
        const double pi=rev[(std::size_t)z].p;

        for(const auto&e:qrows[(std::size_t)i].e){
          const double v=pi*e.second;
          if(v==0.0)continue;

          const int K=e.first;
          if(mk[(std::size_t)K]!=I){
            mk[(std::size_t)K]=I;
            av[(std::size_t)K]=v;
            tv.push_back(K);
          }else{
            av[(std::size_t)K]+=v;
          }
          ++raw;
        }
      }

      std::sort(tv.begin(),tv.end());
      auto&rr=rows[(std::size_t)I];
      rr.clear();
      rr.reserve(tv.size());
      for(int K:tv){
        const double v=av[(std::size_t)K];
        if(v!=0.0)rr.push_back({K,v});
      }
    }

    rawThread[(std::size_t)t]+=raw;
  });

  unsigned long long raw=0;
  for(auto x:rawThread)raw+=x;
  if(rawOut)*rawOut=raw;

  std::size_t retained=0;
  for(const auto&r:rows)retained+=r.size();
  const auto a1=clock::now();

  std::printf(
    "NODALS_GPU_SA_GATE3_ROWS level=%d status=PASS rows=%d retainedPairs=%zu rawTerms=%llu seconds=%.6f rssMiB=%.3f hwmMiB=%.3f\n",
    nextLevel,nc,retained,raw,sa_elapsed_s(a0,a1),
    sa_gate1a_status_mib("VmRSS"),sa_gate1a_status_mib("VmHWM"));
  std::fflush(stdout);

  const double scratchEstimateMiB=
    (double)(
      qnnz*sizeof(std::pair<int,double>) +
      qrows.size()*sizeof(SAGate3QRow) +
      rptr.size()*sizeof(std::int64_t) +
      rev.size()*sizeof(SAGate3Rev) +
      (std::size_t)nth*(std::size_t)nc*(sizeof(double)+sizeof(int)) +
      retained*sizeof(std::pair<int,double>)
    )/(1024.0*1024.0);

  std::vector<SAGate3QRow>().swap(qrows);
  std::vector<std::int64_t>().swap(rptr);
  std::vector<SAGate3Rev>().swap(rev);
  std::vector<std::vector<double>>().swap(acc);
  std::vector<std::vector<int>>().swap(mark);
  std::vector<std::vector<int>>().swap(touched);
  std::vector<unsigned long long>().swap(rawThread);

  const auto p0=clock::now();
  CSRHost C;
  C.n=nc;
  C.row.assign((std::size_t)nc+1,0);
  for(int I=0;I<nc;++I)
    C.row[(std::size_t)I+1]=C.row[(std::size_t)I]+
                            (std::int64_t)rows[(std::size_t)I].size();

  C.col.resize((std::size_t)C.row.back());
  C.val.resize((std::size_t)C.row.back());
  C.diag.assign((std::size_t)nc,0.0);

  sa_gate1a_parallel_chunks(nc,nth,128,[&](int,int ib,int ie){
    for(int I=ib;I<ie;++I){
      std::int64_t k=C.row[(std::size_t)I];
      for(const auto&e:rows[(std::size_t)I]){
        C.col[(std::size_t)k]=e.first;
        C.val[(std::size_t)k]=e.second;
        if(e.first==I)C.diag[(std::size_t)I]=e.second;
        ++k;
      }
    }
  });
  std::vector<std::vector<std::pair<int,double>>>().swap(rows);

  for(int I=0;I<nc;++I)
    if(!(C.diag[(std::size_t)I]>0.0)||!std::isfinite(C.diag[(std::size_t)I]))
      throw std::runtime_error("Gate3 PtAP coarse matrix non-positive diagonal");

  const auto p1=clock::now();
  const auto all1=clock::now();

  std::printf(
    "NODALS_GPU_SA_GATE3_PTAP_NEW level=%d status=PASS rows=%d nnz=%zu avgNnz=%.6f rawTerms=%llu threads=%d qNnz=%zu qAvg=%.6f qMax=%d qSeconds=%.6f reverseSeconds=%.6f accumulateSeconds=%.6f packSeconds=%.6f totalSeconds=%.6f scratchEstimateMiB=%.3f rssMiB=%.3f hwmMiB=%.3f\n",
    nextLevel,C.n,C.val.size(),C.n?(double)C.val.size()/C.n:0.0,
    raw,nth,qnnz,nf?(double)qnnz/nf:0.0,qmax,
    sa_elapsed_s(q0,q1),sa_elapsed_s(r0,r1),
    sa_elapsed_s(a0,a1),sa_elapsed_s(p0,p1),
    sa_elapsed_s(all0,all1),scratchEstimateMiB,
    sa_gate1a_status_mib("VmRSS"),sa_gate1a_status_mib("VmHWM"));

  return C;
}

inline void sa_gate3_validate_ptap(
    const CSRHost&R,const CSRHost&N,int nextLevel,
    unsigned long long rawR,unsigned long long rawN,
    double refSeconds,double newSeconds){
  const bool dims=R.n==N.n;
  const bool rowExact=dims&&R.row==N.row;
  const bool colExact=rowExact&&R.col==N.col;
  const bool valueExact=colExact&&R.val==N.val;

  long double d2=0.0L,r2=0.0L,dd2=0.0L,dr2=0.0L;
  double maxAbs=0.0,diagMax=0.0;

  if(colExact&&R.val.size()==N.val.size()){
    for(std::size_t k=0;k<R.val.size();++k){
      const double d=N.val[k]-R.val[k];
      d2+=(long double)d*d;
      r2+=(long double)R.val[k]*R.val[k];
      maxAbs=std::max(maxAbs,std::abs(d));
    }
    for(std::size_t i=0;i<R.diag.size();++i){
      const double d=N.diag[i]-R.diag[i];
      dd2+=(long double)d*d;
      dr2+=(long double)R.diag[i]*R.diag[i];
      diagMax=std::max(diagMax,std::abs(d));
    }
  }else{
    d2=dd2=std::numeric_limits<long double>::infinity();
  }

  const double valRel=std::sqrt((double)d2)/
                      std::max(std::sqrt((double)r2),1e-300);
  const double diagRel=std::sqrt((double)dd2)/
                       std::max(std::sqrt((double)dr2),1e-300);

  double actionRel=std::numeric_limits<double>::infinity();
  if(colExact){
    std::vector<double>x((std::size_t)R.n),yr,yn;
    for(int i=0;i<R.n;++i){
      const double g=(double)(i+1);
      x[(std::size_t)i]=std::sin(.019*g)+.29*std::cos(.013*g);
    }
    sa_csr_apply(R,x,yr);
    sa_csr_apply(N,x,yn);
    long double ad=0.0L,ar=0.0L;
    for(int i=0;i<R.n;++i){
      const double d=yn[(std::size_t)i]-yr[(std::size_t)i];
      ad+=(long double)d*d;
      ar+=(long double)yr[(std::size_t)i]*yr[(std::size_t)i];
    }
    actionRel=std::sqrt((double)ad)/
              std::max(std::sqrt((double)ar),1e-300);
  }

  const bool pass=dims&&rowExact&&colExact&&rawR==rawN&&
                  std::isfinite(valRel)&&valRel<=5e-13&&
                  diagRel<=5e-13&&actionRel<=5e-13;

  std::printf(
    "NODALS_GPU_SA_GATE3_PTAP_PARITY level=%d status=%s dimsExact=%d rowsExact=%d colsExact=%d valuesBitwiseExact=%d refNnz=%zu newNnz=%zu rawRef=%llu rawNew=%llu rawExact=%d valueRelL2=%.12e valueMaxAbs=%.12e diagRelL2=%.12e diagMaxAbs=%.12e actionRelL2=%.12e refSeconds=%.6f newSeconds=%.6f speedup=%.6f\n",
    nextLevel,pass?"PASS":"FAIL",(int)dims,(int)rowExact,(int)colExact,
    (int)valueExact,R.val.size(),N.val.size(),
    rawR,rawN,(int)(rawR==rawN),valRel,maxAbs,diagRel,diagMax,
    actionRel,refSeconds,newSeconds,
    refSeconds/std::max(newSeconds,1e-300));

  if(!pass)throw std::runtime_error("Gate3 recursive PtAP parity failed");
}

inline const char* sa_gate3_mode(){
  const char*e=std::getenv("NODALS_RECURSIVE_MODE");
  return (e&&*e)?e:"parallel";
}

inline SATransferHost build_sa_explicit_transfer_dispatch(
    const CSRHost&A,const AggResult&G,double lambdaMax,
    double damping,int maxNnz,int level){
  const char*mode=sa_gate3_mode();

  if(std::strcmp(mode,"reference")==0)
    return build_sa_explicit_transfer_reference(
      A,G,lambdaMax,damping,maxNnz,level);

  if(std::strcmp(mode,"parallel")==0)
    return build_sa_explicit_transfer_parallel(
      A,G,lambdaMax,damping,maxNnz,level);

  if(std::strcmp(mode,"compare")==0){
    const auto t0=std::chrono::steady_clock::now();
    auto R=build_sa_explicit_transfer_reference(
      A,G,lambdaMax,damping,maxNnz,level);
    const auto t1=std::chrono::steady_clock::now();

    const auto t2=std::chrono::steady_clock::now();
    auto N=build_sa_explicit_transfer_parallel(
      A,G,lambdaMax,damping,maxNnz,level);
    const auto t3=std::chrono::steady_clock::now();

    sa_gate3_validate_transfer(
      R,N,level,sa_elapsed_s(t0,t1),sa_elapsed_s(t2,t3));
    return N;
  }

  throw std::runtime_error(
    "NODALS_RECURSIVE_MODE must be reference, compare, or parallel");
}

inline CSRHost build_sa_ptap_dispatch(
    const CSRHost&A,const SATransferHost&P,int nextLevel){
  const char*mode=sa_gate3_mode();

  if(std::strcmp(mode,"reference")==0)
    return build_sa_ptap_reference(A,P,nextLevel);

  if(std::strcmp(mode,"parallel")==0)
    return build_sa_ptap_parallel(A,P,nextLevel);

  if(std::strcmp(mode,"compare")==0){
    unsigned long long rr=0,rn=0;

    // Reference function does not expose raw count, so reproduce its count from
    // the exact reference q rows while leaving the reference implementation
    // untouched. This count is cheap relative to the full old unordered-map
    // accumulation and is used only in compare mode.
    const auto t0=std::chrono::steady_clock::now();
    auto R=build_sa_ptap_reference(A,P,nextLevel);
    const auto t1=std::chrono::steady_clock::now();

    // Derive reference raw count: one nonzero pi*q contribution per legacy
    // insertion. q_i is reconstructed with identical local sort/merge.
    for(int i=0;i<A.n;++i){
      std::vector<std::pair<int,double>>q;
      for(std::int64_t k=A.row[(std::size_t)i];k<A.row[(std::size_t)i+1];++k){
        const int j=A.col[(std::size_t)k];
        const double a=A.val[(std::size_t)k];
        for(std::int64_t kp=P.row[(std::size_t)j];kp<P.row[(std::size_t)j+1];++kp)
          q.push_back({P.col[(std::size_t)kp],a*P.val[(std::size_t)kp]});
      }
      std::sort(q.begin(),q.end(),[](const auto&a,const auto&b){return a.first<b.first;});
      std::size_t w=0;
      for(std::size_t k=0;k<q.size();){
        const int c=q[k].first;double v=0.0;std::size_t j=k;
        for(;j<q.size()&&q[j].first==c;++j)v+=q[j].second;
        q[w++]={c,v};k=j;
      }
      q.resize(w);
      for(std::int64_t kp=P.row[(std::size_t)i];kp<P.row[(std::size_t)i+1];++kp){
        const double pi=P.val[(std::size_t)kp];
        for(const auto&e:q)if(pi*e.second!=0.0)++rr;
      }
    }

    const auto t2=std::chrono::steady_clock::now();
    auto N=build_sa_ptap_parallel(A,P,nextLevel,&rn);
    const auto t3=std::chrono::steady_clock::now();

    sa_gate3_validate_ptap(
      R,N,nextLevel,rr,rn,
      sa_elapsed_s(t0,t1),sa_elapsed_s(t2,t3));
    return N;
  }

  throw std::runtime_error(
    "NODALS_RECURSIVE_MODE must be reference, compare, or parallel");
}


inline SAHierarchyHost build_sa_hierarchy(const SerialTetMesh&M,PressureSetupHost&S,
                                           int target=16,int minsz=6,int softmax=18,int terminal=1000,
                                           int maxPnnz=8,int powerIts=16,double safety=1.5,
                                           double lowFrac=0.05,double damping=4.0/3.0){
  SAHierarchyHost H;H.powerIts=powerIts;H.interpMaxNnz=maxPnnz;H.lambdaSafety=safety;H.lambdaLowFraction=lowFrac;H.saDamping=damping;
  auto t0=std::chrono::steady_clock::now();auto I=build_sa_incidence(S);auto t1=std::chrono::steady_clock::now();
  std::printf("NODALS_GPU_SA_HOST_STAGE stage=incidence entries=%zu seconds=%.6f status=PASS\n",I.cell.size(),sa_elapsed_s(t0,t1));

  auto G0=aggregate_graph(face_graph(M),target,minsz,softmax);H.agg.push_back(G0);
  H.fineDiag=cpu_fine_diag(S);
  auto p0=std::chrono::steady_clock::now();H.fineLambda=sa_power_fine(S,H.fineDiag,powerIts,safety);auto p1=std::chrono::steady_clock::now();
  std::printf("NODALS_GPU_SA_HOST_SPECTRUM level=0 powerIts=%d lambdaMax=%.12e seconds=%.6f status=PASS\n",powerIts,H.fineLambda,sa_elapsed_s(p0,p1));
  auto tr0=std::chrono::steady_clock::now();H.P.push_back(build_sa_fine_transfer_dispatch(S,I,G0,H.fineDiag,H.fineLambda,damping,maxPnnz));auto tr1=std::chrono::steady_clock::now();
  std::printf("NODALS_GPU_SA_HOST_STAGE stage=P0 seconds=%.6f status=PASS\n",sa_elapsed_s(tr0,tr1));
  auto c0=std::chrono::steady_clock::now();H.csr.push_back(build_sa_first_coarse_dispatch(S,I,H.P[0]));auto c1=std::chrono::steady_clock::now();
  std::printf("NODALS_GPU_SA_HOST_STAGE stage=A1 seconds=%.6f status=PASS\n",sa_elapsed_s(c0,c1));

  // Incidence is no longer needed once P0 and A1 are formed.
  SAIncidenceHost().row.swap(I.row);SAIncidenceHost().cell.swap(I.cell);SAIncidenceHost().basis.swap(I.basis);

  while(H.csr.back().n>terminal){
    const int lev=(int)H.csr.size(); // explicit operator index lev-1; transfer level lev
    auto &A=H.csr.back();double lam=sa_power_csr(A,powerIts,safety);
    if(H.levelLambda.size()<H.csr.size())H.levelLambda.resize(H.csr.size(),0.0);
    H.levelLambda[H.csr.size()-1]=lam;
    std::printf("NODALS_GPU_SA_HOST_SPECTRUM level=%d powerIts=%d lambdaMax=%.12e status=PASS\n",lev,powerIts,lam);
    auto G=aggregate_graph(csr_graph(A),target,minsz,softmax);H.agg.push_back(G);
    auto P=build_sa_explicit_transfer_dispatch(A,G,lam,damping,maxPnnz,lev);H.P.push_back(std::move(P));
    H.csr.push_back(build_sa_ptap_dispatch(A,H.P.back(),lev+1));
  }
  if(H.levelLambda.size()<H.csr.size())H.levelLambda.resize(H.csr.size(),0.0);
  H.terminal_n=H.csr.back().n;H.terminal_inv=dense_inverse_cholesky(H.csr.back());
  std::printf("NODALS_GPU_SA_HOST_HIERARCHY levels=%zu transfers=%zu terminal=%d fineLambda=%.12e interpMaxNnz=%d powerIts=%d safety=%.6f lowFraction=%.6f damping=%.6f status=PASS\n",
              H.csr.size(),H.P.size(),H.terminal_n,H.fineLambda,maxPnnz,powerIts,safety,lowFrac,damping);
  return H;
}

} // namespace nodals_gpu
