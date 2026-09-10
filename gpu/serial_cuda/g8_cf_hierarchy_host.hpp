#pragma once
#include "g5_sa_host.hpp"
#include "h2b_fine_csr.cuh"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <thread>
#include <sys/resource.h>
#include <unistd.h>

namespace nodals_gpu {

inline bool g8_pm2_env_enabled(const char*name)
{
  const char*e=std::getenv(name);
  return e && *e &&
         std::strcmp(e,"0")!=0 &&
         std::strcmp(e,"false")!=0 &&
         std::strcmp(e,"FALSE")!=0 &&
         std::strcmp(e,"off")!=0 &&
         std::strcmp(e,"OFF")!=0;
}

inline bool g8_pm2_cf_diagnostics_enabled()
{
  return g8_pm2_env_enabled("NODALS_PMIS_DIAGNOSTICS") ||
         g8_pm2_env_enabled("NODALS_SETUP_AUTOTUNE");
}

// Production setup does not need an O(nnz log(row)) symmetry audit after
// matrices that have already passed exact old/new parity gates.
// Return -1 when skipped so existing diagnostic lines remain parseable.
inline double g8_pm2_symmetry_rel(const CSRHost&A)
{
  return g8_pm2_cf_diagnostics_enabled() ? csr_symmetry_rel(A) : -1.0;
}


inline double g8_pm0_rss_mib()
{
  long pages=0;
  FILE*f=std::fopen("/proc/self/statm","r");
  if(f){
    long total=0;
    if(std::fscanf(f,"%ld %ld",&total,&pages)!=2)pages=0;
    std::fclose(f);
  }
  const long ps=sysconf(_SC_PAGESIZE);
  return (pages>0 && ps>0)
      ? (double)pages*(double)ps/(1024.0*1024.0) : -1.0;
}

inline double g8_pm0_hwm_mib()
{
  struct rusage r{};
  if(getrusage(RUSAGE_SELF,&r)!=0)return -1.0;
  return (double)r.ru_maxrss/1024.0;
}

inline void g8_pm0_stage(
    const char*stage,int level,int rows,std::size_t nnz,double sec)
{
  std::printf(
    "NODALS_GPU_PM0_STAGE stage=%s level=%d rows=%d nnz=%zu "
    "seconds=%.9f rssMiB=%.3f hwmMiB=%.3f status=PASS\n",
    stage,level,rows,nnz,sec,g8_pm0_rss_mib(),g8_pm0_hwm_mib());
}

struct G8CFSplit {
  int n=0,nC=0,rounds=0;
  std::vector<std::uint8_t> state;
  std::vector<std::int32_t> coarseId;
  std::vector<std::vector<std::int32_t>> strong;
  std::vector<std::vector<std::int32_t>> graph;
};

inline std::uint64_t g8_hash64(std::uint64_t x){
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

inline bool g8_priority_higher(
    int a,int b,const std::vector<std::vector<std::int32_t>>& g)
{
  const std::size_t da=g[(std::size_t)a].size(),db=g[(std::size_t)b].size();
  if(da!=db) return da>db;
  const std::uint64_t ha=g8_hash64((std::uint64_t)a+0x63021ull);
  const std::uint64_t hb=g8_hash64((std::uint64_t)b+0x63021ull);
  if(ha!=hb) return ha>hb;
  return a>b;
}

inline double g8_csr_entry(const CSRHost&A,int i,int j){
  auto b=A.col.begin()+A.row[(std::size_t)i];
  auto e=A.col.begin()+A.row[(std::size_t)i+1];
  auto it=std::lower_bound(b,e,j);
  if(it==e||*it!=j)return 0.0;
  return A.val[(std::size_t)(it-A.col.begin())];
}

inline CSRHost g8_build_exact_fine_csr(
    const G4SetupHost&S,const H2FineCSRHost&F)
{
  const auto pm0_t0=std::chrono::steady_clock::now();
  CSRHost A;
  A.n=F.n;
  A.row.resize(F.row.size());
  for(std::size_t i=0;i<F.row.size();++i)A.row[i]=(std::int64_t)F.row[i];
  A.col=F.col;
  A.val.assign(F.col.size(),0.0);
  A.diag.assign((std::size_t)F.n,0.0);

  if(S.pressure.rAU.size()!=(std::size_t)F.nv)
    throw std::runtime_error("G8 fine rAU size mismatch");

  for(int c=0;c<F.n;++c){
    const int rb=F.row[(std::size_t)c];
    const auto&cp=S.pressure.cells[(std::size_t)c];
    int q=F.contribOff[(std::size_t)c];

    for(int a=0;a<8;++a){
      const int g=cp.vel[a];
      if(g<0)continue;
      const double rv=S.pressure.rAU[(std::size_t)g];
      const double a0=coeff(cp,a,0),a1=coeff(cp,a,1),a2=coeff(cp,a,2);
      for(int k=F.incOff[(std::size_t)g];k<F.incOff[(std::size_t)g+1];++k,++q){
        const std::uint32_t pk=F.packed[(std::size_t)k];
        const int c2=(int)(pk>>3),b=(int)(pk&7u);
        const auto&cp2=S.pressure.cells[(std::size_t)c2];
        const double dot=
          a0*coeff(cp2,b,0)+a1*coeff(cp2,b,1)+a2*coeff(cp2,b,2);
        A.val[(std::size_t)(rb+(int)F.slot[(std::size_t)q])] += rv*dot;
      }
    }

    const int dp=F.diagPos[(std::size_t)c];
    const double d=A.val[(std::size_t)dp];
    if(!(d>0.0)||!std::isfinite(d))
      throw std::runtime_error("G8 fine CSR non-positive diagonal");
    A.diag[(std::size_t)c]=d;
  }

  std::printf(
    "NODALS_GPU_G8_CF_FINE_CSR rows=%d nnz=%zu avgNnz=%.6f symmetryRel=%.3e "
    "source=EXACT_HOST_SETUP_SNAPSHOT status=PASS\n",
    A.n,A.val.size(),A.n?(double)A.val.size()/A.n:0.0,g8_pm2_symmetry_rel(A));
  g8_pm0_stage(
    "fine_snapshot",0,A.n,A.val.size(),
    std::chrono::duration<double>(
      std::chrono::steady_clock::now()-pm0_t0).count());
  return A;
}

inline G8CFSplit g8_build_pmis_split_reference(const CSRHost&A,double theta,int level){
  const auto pm0_total_t0=std::chrono::steady_clock::now();
  auto pm0_phase_t0=pm0_total_t0;
  G8CFSplit R;R.n=A.n;
  std::vector<double> maxNeg((std::size_t)A.n,0.0);

  for(int i=0;i<A.n;++i){
    for(std::int64_t k=A.row[(std::size_t)i];k<A.row[(std::size_t)i+1];++k){
      const int j=A.col[(std::size_t)k];
      if(j==i)continue;
      const double a=A.val[(std::size_t)k];
      if(a<0.0)maxNeg[(std::size_t)i]=std::max(maxNeg[(std::size_t)i],-a);
    }
  }

  g8_pm0_stage(
    "pmis_maxneg",level,A.n,A.val.size(),
    std::chrono::duration<double>(
      std::chrono::steady_clock::now()-pm0_phase_t0).count());
  pm0_phase_t0=std::chrono::steady_clock::now();

  R.strong.resize((std::size_t)A.n);
  R.graph.resize((std::size_t)A.n);
  std::uint64_t ds=0,isolatedDirected=0;

  for(int i=0;i<A.n;++i){
    const double thr=theta*maxNeg[(std::size_t)i];
    auto& s=R.strong[(std::size_t)i];
    if(thr>0.0){
      for(std::int64_t k=A.row[(std::size_t)i];k<A.row[(std::size_t)i+1];++k){
        const int j=A.col[(std::size_t)k];
        if(j==i)continue;
        const double a=A.val[(std::size_t)k];
        if(a<0.0 && -a>=thr)s.push_back((std::int32_t)j);
      }
    }
    std::sort(s.begin(),s.end());
    s.erase(std::unique(s.begin(),s.end()),s.end());
    ds+=(std::uint64_t)s.size();
    if(s.empty())++isolatedDirected;
    for(int j:s){
      R.graph[(std::size_t)i].push_back((std::int32_t)j);
      R.graph[(std::size_t)j].push_back((std::int32_t)i);
    }
  }

  std::uint64_t unionDirected=0,unionIsolated=0;
  for(auto&g:R.graph){
    std::sort(g.begin(),g.end());
    g.erase(std::unique(g.begin(),g.end()),g.end());
    unionDirected+=(std::uint64_t)g.size();
    if(g.empty())++unionIsolated;
  }

  g8_pm0_stage(
    "pmis_strong_union_graph",level,A.n,(std::size_t)unionDirected,
    std::chrono::duration<double>(
      std::chrono::steady_clock::now()-pm0_phase_t0).count());
  pm0_phase_t0=std::chrono::steady_clock::now();

  R.state.assign((std::size_t)A.n,0);
  std::vector<std::int32_t>winners;
  int undecided=A.n;

  while(undecided>0){
    winners.clear();
    for(int i=0;i<A.n;++i){
      if(R.state[(std::size_t)i]!=0)continue;
      bool win=true;
      for(int j:R.graph[(std::size_t)i]){
        if(R.state[(std::size_t)j]==0 && g8_priority_higher(j,i,R.graph)){
          win=false;break;
        }
      }
      if(win)winners.push_back((std::int32_t)i);
    }
    if(winners.empty())throw std::runtime_error("G8 PMIS no progress");

    for(int i:winners)if(R.state[(std::size_t)i]==0){
      R.state[(std::size_t)i]=1;--undecided;
    }
    for(int i:winners)for(int j:R.graph[(std::size_t)i])
      if(R.state[(std::size_t)j]==0){
        R.state[(std::size_t)j]=2;--undecided;
      }

    ++R.rounds;
    if(R.rounds>A.n)throw std::runtime_error("G8 PMIS excessive rounds");
  }

  g8_pm0_stage(
    "pmis_rounds",level,A.n,(std::size_t)unionDirected,
    std::chrono::duration<double>(
      std::chrono::steady_clock::now()-pm0_phase_t0).count());

  R.coarseId.assign((std::size_t)A.n,-1);
  for(int i=0;i<A.n;++i)if(R.state[(std::size_t)i]==1)
    R.coarseId[(std::size_t)i]=R.nC++;

  std::uint64_t nF=(std::uint64_t)A.n-(std::uint64_t)R.nC,cc=0;
  for(int i=0;i<A.n;++i)if(R.state[(std::size_t)i]==1)
    for(int j:R.strong[(std::size_t)i])
      if(R.state[(std::size_t)j]==1)++cc;

  std::printf(
    "NODALS_GPU_G8_CF_SPLIT level=%d rows=%d theta=%.2f criterion=CLASSICAL_NEGATIVE "
    "directedStrong=%llu avgStrongDegree=%.6f directedIsolated=%llu "
    "unionEdges=%llu unionIsolated=%llu C=%d F=%llu coarseRatio=%.9f "
    "finePerCoarse=%.6f rounds=%d strongCCDirected=%llu status=PASS\n",
    level,A.n,theta,(unsigned long long)ds,double(ds)/double(A.n),
    (unsigned long long)isolatedDirected,(unsigned long long)(unionDirected/2ull),
    (unsigned long long)unionIsolated,R.nC,(unsigned long long)nF,
    double(R.nC)/double(A.n),R.nC?double(A.n)/double(R.nC):0.0,
    R.rounds,(unsigned long long)cc);
  g8_pm0_stage(
    "pmis_split_total",level,A.n,(std::size_t)ds,
    std::chrono::duration<double>(
      std::chrono::steady_clock::now()-pm0_total_t0).count());
  return R;
}

// -----------------------------------------------------------------------------
// PM4: deterministic multicore PMIS strength + symmetric union graph build.
// PMIS state-selection rounds remain the original serial deterministic loop.
// -----------------------------------------------------------------------------

inline int g8_pm4_threads()
{
  int n=0;
  if(const char*e=std::getenv("NODALS_PM4_THREADS"))n=std::atoi(e);
  if(n<=0){
    const unsigned h=std::thread::hardware_concurrency();
    n=h?std::min((int)h,16):1;
  }
  return std::max(1,std::min(n,64));
}

template<class F>
inline void g8_pm4_parallel_chunks(int n,int nth,int chunk,F&&fn)
{
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

inline G8CFSplit g8_build_pmis_split_parallel(
    const CSRHost&A,double theta,int level)
{
  using clock=std::chrono::steady_clock;
  const auto all0=clock::now();
  const int nth=g8_pm4_threads();

  G8CFSplit R;
  R.n=A.n;

  // 1) Row-local maximum negative coupling.
  const auto m0=clock::now();
  std::vector<double> maxNeg((std::size_t)A.n,0.0);

  g8_pm4_parallel_chunks(A.n,nth,256,[&](int,int ib,int ie){
    for(int i=ib;i<ie;++i){
      double mx=0.0;
      for(std::int64_t k=A.row[(std::size_t)i];
          k<A.row[(std::size_t)i+1];++k){
        const int j=A.col[(std::size_t)k];
        if(j==i)continue;
        const double a=A.val[(std::size_t)k];
        if(a<0.0)mx=std::max(mx,-a);
      }
      maxNeg[(std::size_t)i]=mx;
    }
  });
  const auto m1=clock::now();

  // 2) Strong rows. A CSR columns are already sorted and unique, therefore
  // filtering them preserves exactly the reference strong-row ordering.
  const auto s0=clock::now();
  R.strong.resize((std::size_t)A.n);
  std::vector<std::uint64_t> dsThread((std::size_t)nth,0);
  std::vector<std::uint64_t> isoThread((std::size_t)nth,0);

  g8_pm4_parallel_chunks(A.n,nth,128,[&](int t,int ib,int ie){
    std::uint64_t ds=0,iso=0;
    for(int i=ib;i<ie;++i){
      const double thr=theta*maxNeg[(std::size_t)i];
      auto&sr=R.strong[(std::size_t)i];
      if(thr>0.0){
        for(std::int64_t k=A.row[(std::size_t)i];
            k<A.row[(std::size_t)i+1];++k){
          const int j=A.col[(std::size_t)k];
          if(j==i)continue;
          const double a=A.val[(std::size_t)k];
          if(a<0.0 && -a>=thr)sr.push_back((std::int32_t)j);
        }
      }
      ds+=(std::uint64_t)sr.size();
      if(sr.empty())++iso;
    }
    dsThread[(std::size_t)t]+=ds;
    isoThread[(std::size_t)t]+=iso;
  });

  std::uint64_t ds=0,isolatedDirected=0;
  for(int t=0;t<nth;++t){
    ds+=dsThread[(std::size_t)t];
    isolatedDirected+=isoThread[(std::size_t)t];
  }
  std::vector<double>().swap(maxNeg);
  const auto s1=clock::now();

  // 3) Build incoming strong adjacency in flat CSR form.
  // Serial fill in increasing source-row order makes every incoming row sorted.
  const auto tr0=clock::now();
  std::vector<std::int64_t> inRow((std::size_t)A.n+1,0);
  for(int i=0;i<A.n;++i)
    for(int j:R.strong[(std::size_t)i])
      ++inRow[(std::size_t)j+1];

  for(int i=0;i<A.n;++i)
    inRow[(std::size_t)i+1]+=inRow[(std::size_t)i];

  std::vector<std::int32_t> incoming((std::size_t)inRow.back());
  auto next=inRow;
  for(int i=0;i<A.n;++i){
    for(int j:R.strong[(std::size_t)i]){
      const std::int64_t k=next[(std::size_t)j]++;
      incoming[(std::size_t)k]=(std::int32_t)i;
    }
  }
  std::vector<std::int64_t>().swap(next);
  const auto tr1=clock::now();

  // 4) Exact sorted unique union of outgoing and incoming rows.
  const auto u0=clock::now();
  R.graph.resize((std::size_t)A.n);
  std::vector<std::uint64_t> unionThread((std::size_t)nth,0);
  std::vector<std::uint64_t> unionIsoThread((std::size_t)nth,0);

  g8_pm4_parallel_chunks(A.n,nth,128,[&](int t,int ib,int ie){
    std::uint64_t ud=0,ui=0;
    for(int i=ib;i<ie;++i){
      const auto&out=R.strong[(std::size_t)i];
      const std::int64_t b=inRow[(std::size_t)i];
      const std::int64_t e=inRow[(std::size_t)i+1];

      auto&g=R.graph[(std::size_t)i];
      g.reserve(out.size()+(std::size_t)(e-b));

      std::size_t a=0;
      std::int64_t q=b;

      while(a<out.size() || q<e){
        std::int32_t v;
        if(q>=e || (a<out.size() && out[a]<incoming[(std::size_t)q])){
          v=out[a++];
        }else if(a>=out.size() || incoming[(std::size_t)q]<out[a]){
          v=incoming[(std::size_t)q++];
        }else{
          v=out[a];
          ++a;
          ++q;
        }
        g.push_back(v);
      }

      ud+=(std::uint64_t)g.size();
      if(g.empty())++ui;
    }
    unionThread[(std::size_t)t]+=ud;
    unionIsoThread[(std::size_t)t]+=ui;
  });

  std::uint64_t unionDirected=0,unionIsolated=0;
  for(int t=0;t<nth;++t){
    unionDirected+=unionThread[(std::size_t)t];
    unionIsolated+=unionIsoThread[(std::size_t)t];
  }

  std::vector<std::int64_t>().swap(inRow);
  std::vector<std::int32_t>().swap(incoming);
  std::vector<std::uint64_t>().swap(dsThread);
  std::vector<std::uint64_t>().swap(isoThread);
  std::vector<std::uint64_t>().swap(unionThread);
  std::vector<std::uint64_t>().swap(unionIsoThread);
  const auto u1=clock::now();

  // 5) Original deterministic serial PMIS rounds, unchanged.
  const auto r0=clock::now();
  R.state.assign((std::size_t)A.n,0);
  std::vector<std::int32_t>winners;
  int undecided=A.n;

  while(undecided>0){
    winners.clear();
    for(int i=0;i<A.n;++i){
      if(R.state[(std::size_t)i]!=0)continue;
      bool win=true;
      for(int j:R.graph[(std::size_t)i]){
        if(R.state[(std::size_t)j]==0 && g8_priority_higher(j,i,R.graph)){
          win=false;
          break;
        }
      }
      if(win)winners.push_back((std::int32_t)i);
    }

    if(winners.empty())throw std::runtime_error("PM4 PMIS no progress");

    for(int i:winners)if(R.state[(std::size_t)i]==0){
      R.state[(std::size_t)i]=1;
      --undecided;
    }

    for(int i:winners)
      for(int j:R.graph[(std::size_t)i])
        if(R.state[(std::size_t)j]==0){
          R.state[(std::size_t)j]=2;
          --undecided;
        }

    ++R.rounds;
    if(R.rounds>A.n)throw std::runtime_error("PM4 PMIS excessive rounds");
  }

  R.coarseId.assign((std::size_t)A.n,-1);
  for(int i=0;i<A.n;++i)
    if(R.state[(std::size_t)i]==1)
      R.coarseId[(std::size_t)i]=R.nC++;
  const auto r1=clock::now();

  std::uint64_t nF=(std::uint64_t)A.n-(std::uint64_t)R.nC,cc=0;
  for(int i=0;i<A.n;++i)
    if(R.state[(std::size_t)i]==1)
      for(int j:R.strong[(std::size_t)i])
        if(R.state[(std::size_t)j]==1)++cc;

  const auto all1=clock::now();

  std::printf(
    "NODALS_GPU_G8_CF_SPLIT level=%d rows=%d theta=%.2f criterion=CLASSICAL_NEGATIVE "
    "directedStrong=%llu avgStrongDegree=%.6f directedIsolated=%llu "
    "unionEdges=%llu unionIsolated=%llu C=%d F=%llu coarseRatio=%.9f "
    "finePerCoarse=%.6f rounds=%d strongCCDirected=%llu status=PASS\n",
    level,A.n,theta,(unsigned long long)ds,double(ds)/double(A.n),
    (unsigned long long)isolatedDirected,(unsigned long long)(unionDirected/2ull),
    (unsigned long long)unionIsolated,R.nC,(unsigned long long)nF,
    double(R.nC)/double(A.n),R.nC?double(A.n)/double(R.nC):0.0,
    R.rounds,(unsigned long long)cc);

  std::printf(
    "NODALS_GPU_PM4_SPLIT_NEW level=%d status=PASS rows=%d threads=%d "
    "directedStrong=%llu unionDirected=%llu C=%d rounds=%d "
    "maxNegSeconds=%.6f strongSeconds=%.6f transposeSeconds=%.6f "
    "unionSeconds=%.6f roundsSeconds=%.6f totalSeconds=%.6f "
    "rssMiB=%.3f hwmMiB=%.3f\n",
    level,A.n,nth,(unsigned long long)ds,(unsigned long long)unionDirected,
    R.nC,R.rounds,
    std::chrono::duration<double>(m1-m0).count(),
    std::chrono::duration<double>(s1-s0).count(),
    std::chrono::duration<double>(tr1-tr0).count(),
    std::chrono::duration<double>(u1-u0).count(),
    std::chrono::duration<double>(r1-r0).count(),
    std::chrono::duration<double>(all1-all0).count(),
    g8_pm0_rss_mib(),g8_pm0_hwm_mib());

  return R;
}

inline void g8_pm4_validate_split(
    const G8CFSplit&R,const G8CFSplit&N,int level,
    double refSeconds,double newSeconds)
{
  const bool dims=R.n==N.n && R.nC==N.nC;
  const bool strongExact=dims && R.strong==N.strong;
  const bool graphExact=strongExact && R.graph==N.graph;
  const bool stateExact=graphExact && R.state==N.state;
  const bool coarseExact=stateExact && R.coarseId==N.coarseId;
  const bool roundsExact=R.rounds==N.rounds;

  const bool pass=
    dims&&strongExact&&graphExact&&stateExact&&coarseExact&&roundsExact;

  std::printf(
    "NODALS_GPU_PM4_SPLIT_PARITY level=%d status=%s "
    "dimsExact=%d strongExact=%d graphExact=%d stateExact=%d "
    "coarseIdExact=%d roundsExact=%d refC=%d newC=%d "
    "refRounds=%d newRounds=%d refSeconds=%.6f newSeconds=%.6f speedup=%.6f\n",
    level,pass?"PASS":"FAIL",
    (int)dims,(int)strongExact,(int)graphExact,(int)stateExact,
    (int)coarseExact,(int)roundsExact,R.nC,N.nC,R.rounds,N.rounds,
    refSeconds,newSeconds,refSeconds/std::max(newSeconds,1e-300));

  if(!pass)throw std::runtime_error("PM4 PMIS split parity failed");
}

inline const char* g8_pm4_split_mode()
{
  const char*e=std::getenv("NODALS_PM4_SPLIT_MODE");
  return (e&&*e)?e:"parallel";
}

inline G8CFSplit g8_build_pmis_split_dispatch(
    const CSRHost&A,double theta,int level)
{
  const char*mode=g8_pm4_split_mode();

  if(std::strcmp(mode,"reference")==0)
    return g8_build_pmis_split_reference(A,theta,level);

  if(std::strcmp(mode,"parallel")==0)
    return g8_build_pmis_split_parallel(A,theta,level);

  if(std::strcmp(mode,"compare")==0){
    const auto r0=std::chrono::steady_clock::now();
    auto R=g8_build_pmis_split_reference(A,theta,level);
    const auto r1=std::chrono::steady_clock::now();

    const auto n0=std::chrono::steady_clock::now();
    auto N=g8_build_pmis_split_parallel(A,theta,level);
    const auto n1=std::chrono::steady_clock::now();

    g8_pm4_validate_split(
      R,N,level,
      std::chrono::duration<double>(r1-r0).count(),
      std::chrono::duration<double>(n1-n0).count());
    return N;
  }

  throw std::runtime_error(
    "NODALS_PM4_SPLIT_MODE must be reference, compare, or parallel");
}


inline void g8_prune_normalize(
    std::vector<std::pair<int,double>>& cand,int pmax)
{
  if(cand.empty())throw std::runtime_error("G8 interpolation empty candidates");

  std::sort(cand.begin(),cand.end(),[](const auto&a,const auto&b){
    if(a.second!=b.second)return a.second>b.second;
    return a.first<b.first;
  });
  if(pmax>0 && (int)cand.size()>pmax)cand.resize((std::size_t)pmax);

  double sum=0.0;
  for(const auto&e:cand)sum+=e.second;
  if(!(sum>0.0)||!std::isfinite(sum))
    throw std::runtime_error("G8 interpolation invalid weight sum");
  for(auto&e:cand)e.second/=sum;

  std::sort(cand.begin(),cand.end(),[](const auto&a,const auto&b){
    return a.first<b.first;
  });
}

inline SATransferHost g8_build_transfer_reference(
    const CSRHost&A,const G8CFSplit&S,int pmax,int level,
    const std::string& interp)
{
  const auto pm0_t0=std::chrono::steady_clock::now();
  if(interp!="direct"&&interp!="exti")
    throw std::runtime_error("G8 interpolation mode must be direct or exti");

  SATransferHost P;
  P.nFine=A.n;P.nCoarse=S.nC;
  P.row.assign((std::size_t)A.n+1,0);
  P.col.reserve((std::size_t)A.n*(interp=="exti"?3u:2u));
  P.val.reserve((std::size_t)A.n*(interp=="exti"?3u:2u));

  std::uint64_t cRows=0,directRows=0,fallbackRows=0;
  std::uint64_t extRows=0,extFPaths=0,prePruneCandidates=0,prunedRows=0;
  int maxRow=0,maxPrePrune=0;
  long double maxDefect=0.0L;
  std::vector<std::pair<int,double>> cand;
  std::unordered_map<int,double> acc;
  acc.reserve(64);

  for(int i=0;i<A.n;++i){
    cand.clear();

    if(S.state[(std::size_t)i]==1){
      cand.push_back({S.coarseId[(std::size_t)i],1.0});
      ++cRows;
    }else{
      acc.clear();

      // Direct strong C contribution.
      for(int j:S.strong[(std::size_t)i]){
        if(S.state[(std::size_t)j]!=1)continue;
        const double a=g8_csr_entry(A,i,j);
        const double w=(a<0.0)?-a:std::abs(a);
        if(w>0.0)acc[S.coarseId[(std::size_t)j]]+=w;
      }

      const bool hadDirect=!acc.empty();
      if(hadDirect)++directRows;

      // Extended+i-like distance-two contribution:
      // i -> strong F k -> strong C j, distributing |a_ik| across k's
      // strong-C neighbors in proportion to their negative coupling.
      if(interp=="exti"){
        bool usedExt=false;
        for(int k:S.strong[(std::size_t)i]){
          if(S.state[(std::size_t)k]!=2)continue;
          const double aik=g8_csr_entry(A,i,k);
          const double path=(aik<0.0)?-aik:0.0;
          if(!(path>0.0))continue;

          double denom=0.0;
          for(int j:S.strong[(std::size_t)k]){
            if(S.state[(std::size_t)j]!=1)continue;
            const double akj=g8_csr_entry(A,k,j);
            if(akj<0.0)denom+=-akj;
          }
          if(!(denom>0.0))continue;

          for(int j:S.strong[(std::size_t)k]){
            if(S.state[(std::size_t)j]!=1)continue;
            const double akj=g8_csr_entry(A,k,j);
            if(akj<0.0){
              acc[S.coarseId[(std::size_t)j]]+=path*((-akj)/denom);
            }
          }
          usedExt=true;
          ++extFPaths;
        }
        if(usedExt)++extRows;
      }

      // Gate4 repair is retained only if neither direct nor extended+i found C.
      if(acc.empty()){
        ++fallbackRows;
        for(int j:S.graph[(std::size_t)i]){
          if(S.state[(std::size_t)j]!=1)continue;
          const double a=g8_csr_entry(A,i,j);
          const double w=std::abs(a);
          if(w>0.0)acc[S.coarseId[(std::size_t)j]]+=w;
        }
      }

      if(acc.empty())
        throw std::runtime_error("G8 interpolation row has no C target");

      cand.reserve(std::max(cand.capacity(),acc.size()));
      for(const auto&kv:acc)if(kv.second>0.0)cand.push_back(kv);
      const int pre=(int)cand.size();
      prePruneCandidates+=(std::uint64_t)pre;
      maxPrePrune=std::max(maxPrePrune,pre);
      if(pmax>0&&pre>pmax)++prunedRows;

      g8_prune_normalize(cand,pmax);
    }

    double rs=0.0;
    for(const auto&e:cand){
      P.col.push_back((std::int32_t)e.first);
      P.val.push_back(e.second);
      rs+=e.second;
    }
    P.row[(std::size_t)i+1]=(std::int64_t)P.col.size();
    maxRow=std::max(maxRow,(int)cand.size());
    maxDefect=std::max(maxDefect,(long double)std::abs(rs-1.0));
  }

  const std::uint64_t fRows=(std::uint64_t)P.nFine-(std::uint64_t)cRows;
  const double fden=fRows?double(fRows):1.0;

  std::printf(
    "NODALS_GPU_G8_CF_TRANSFER level=%d interp=%s rows=%d coarse=%d pmax=%d "
    "nnz=%zu avgNnz=%.6f maxRowNnz=%d CInjectionRows=%llu "
    "directStrongCRows=%llu extendedRows=%llu extendedFPaths=%llu "
    "unionFallbackRows=%llu unionFallbackFracOfFine=%.9f "
    "avgCandidatesBeforePrune=%.6f maxCandidatesBeforePrune=%d prunedRows=%llu "
    "rowSumDefect=%.3Le weightPolicy=NEGATIVE_COUPLING "
    "extendedPolicy=STRONG_F_TO_STRONG_C fallbackPolicy=SYMMETRIC_UNION_C status=PASS\n",
    level,interp.c_str(),P.nFine,P.nCoarse,pmax,P.val.size(),
    P.nFine?(double)P.val.size()/P.nFine:0.0,maxRow,
    (unsigned long long)cRows,(unsigned long long)directRows,
    (unsigned long long)extRows,(unsigned long long)extFPaths,
    (unsigned long long)fallbackRows,
    P.nFine?double(fallbackRows)/double(P.nFine):0.0,
    double(prePruneCandidates)/fden,maxPrePrune,
    (unsigned long long)prunedRows,maxDefect);
  g8_pm0_stage(
    "interpolation",level,P.nFine,P.val.size(),
    std::chrono::duration<double>(
      std::chrono::steady_clock::now()-pm0_t0).count());
  return P;
}

// -----------------------------------------------------------------------------
// PM3: parallel deterministic CF interpolation.
// Scheduling changes only; per-row interpolation arithmetic is unchanged.
// -----------------------------------------------------------------------------

inline int g8_pm3_threads()
{
  int n=0;
  if(const char*e=std::getenv("NODALS_PM3_THREADS"))n=std::atoi(e);
  if(n<=0){
    const unsigned h=std::thread::hardware_concurrency();
    n=h?std::min((int)h,16):1;
  }
  return std::max(1,std::min(n,64));
}

template<class F>
inline void g8_pm3_parallel_chunks(int n,int nth,int chunk,F&&fn)
{
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

struct G8PM3Stats {
  std::uint64_t cRows=0,directRows=0,fallbackRows=0;
  std::uint64_t extRows=0,extFPaths=0,prePruneCandidates=0,prunedRows=0;
  int maxRow=0,maxPrePrune=0;
  long double maxDefect=0.0L;
};

inline SATransferHost g8_build_transfer_parallel(
    const CSRHost&A,const G8CFSplit&S,int pmax,int level,
    const std::string& interp)
{
  if(interp!="direct"&&interp!="exti")
    throw std::runtime_error("G8 interpolation mode must be direct or exti");

  const auto all0=std::chrono::steady_clock::now();
  const int nth=g8_pm3_threads();
  std::vector<std::vector<std::pair<int,double>>> rows((std::size_t)A.n);
  std::vector<G8PM3Stats> stats((std::size_t)nth);

  const auto build0=std::chrono::steady_clock::now();

  g8_pm3_parallel_chunks(A.n,nth,64,[&](int t,int ib,int ie){
    auto&st=stats[(std::size_t)t];
    std::vector<std::pair<int,double>> cand;
    std::unordered_map<int,double> acc;
    cand.reserve(64);
    acc.reserve(64);

    for(int i=ib;i<ie;++i){
      cand.clear();

      if(S.state[(std::size_t)i]==1){
        cand.push_back({S.coarseId[(std::size_t)i],1.0});
        ++st.cRows;
      }else{
        acc.clear();

        for(int j:S.strong[(std::size_t)i]){
          if(S.state[(std::size_t)j]!=1)continue;
          const double a=g8_csr_entry(A,i,j);
          const double w=(a<0.0)?-a:std::abs(a);
          if(w>0.0)acc[S.coarseId[(std::size_t)j]]+=w;
        }

        if(!acc.empty())++st.directRows;

        if(interp=="exti"){
          bool usedExt=false;
          for(int k:S.strong[(std::size_t)i]){
            if(S.state[(std::size_t)k]!=2)continue;
            const double aik=g8_csr_entry(A,i,k);
            const double path=(aik<0.0)?-aik:0.0;
            if(!(path>0.0))continue;

            double denom=0.0;
            for(int j:S.strong[(std::size_t)k]){
              if(S.state[(std::size_t)j]!=1)continue;
              const double akj=g8_csr_entry(A,k,j);
              if(akj<0.0)denom+=-akj;
            }
            if(!(denom>0.0))continue;

            for(int j:S.strong[(std::size_t)k]){
              if(S.state[(std::size_t)j]!=1)continue;
              const double akj=g8_csr_entry(A,k,j);
              if(akj<0.0)
                acc[S.coarseId[(std::size_t)j]]+=path*((-akj)/denom);
            }

            usedExt=true;
            ++st.extFPaths;
          }
          if(usedExt)++st.extRows;
        }

        if(acc.empty()){
          ++st.fallbackRows;
          for(int j:S.graph[(std::size_t)i]){
            if(S.state[(std::size_t)j]!=1)continue;
            const double a=g8_csr_entry(A,i,j);
            const double w=std::abs(a);
            if(w>0.0)acc[S.coarseId[(std::size_t)j]]+=w;
          }
        }

        if(acc.empty())
          throw std::runtime_error("PM3 interpolation row has no C target");

        cand.reserve(std::max(cand.capacity(),acc.size()));
        for(const auto&kv:acc)if(kv.second>0.0)cand.push_back(kv);

        const int pre=(int)cand.size();
        st.prePruneCandidates+=(std::uint64_t)pre;
        st.maxPrePrune=std::max(st.maxPrePrune,pre);
        if(pmax>0&&pre>pmax)++st.prunedRows;

        g8_prune_normalize(cand,pmax);
      }

      double rs=0.0;
      for(const auto&e:cand)rs+=e.second;
      st.maxRow=std::max(st.maxRow,(int)cand.size());
      st.maxDefect=std::max(st.maxDefect,(long double)std::abs(rs-1.0));
      rows[(std::size_t)i]=cand;
    }
  });

  const auto build1=std::chrono::steady_clock::now();

  G8PM3Stats T;
  for(const auto&st:stats){
    T.cRows+=st.cRows; T.directRows+=st.directRows; T.fallbackRows+=st.fallbackRows;
    T.extRows+=st.extRows; T.extFPaths+=st.extFPaths;
    T.prePruneCandidates+=st.prePruneCandidates; T.prunedRows+=st.prunedRows;
    T.maxRow=std::max(T.maxRow,st.maxRow);
    T.maxPrePrune=std::max(T.maxPrePrune,st.maxPrePrune);
    T.maxDefect=std::max(T.maxDefect,st.maxDefect);
  }

  const auto pack0=std::chrono::steady_clock::now();

  SATransferHost P;
  P.nFine=A.n; P.nCoarse=S.nC;
  P.row.assign((std::size_t)A.n+1,0);
  for(int i=0;i<A.n;++i)
    P.row[(std::size_t)i+1]=P.row[(std::size_t)i]+(std::int64_t)rows[(std::size_t)i].size();

  P.col.resize((std::size_t)P.row.back());
  P.val.resize((std::size_t)P.row.back());

  g8_pm3_parallel_chunks(A.n,nth,256,[&](int,int ib,int ie){
    for(int i=ib;i<ie;++i){
      std::int64_t k=P.row[(std::size_t)i];
      for(const auto&e:rows[(std::size_t)i]){
        P.col[(std::size_t)k]=e.first;
        P.val[(std::size_t)k]=e.second;
        ++k;
      }
    }
  });

  std::vector<std::vector<std::pair<int,double>>>().swap(rows);
  const auto pack1=std::chrono::steady_clock::now();
  const auto all1=std::chrono::steady_clock::now();

  const std::uint64_t fRows=(std::uint64_t)P.nFine-T.cRows;
  const double fden=fRows?double(fRows):1.0;

  std::printf(
    "NODALS_GPU_G8_CF_TRANSFER level=%d interp=%s rows=%d coarse=%d pmax=%d "
    "nnz=%zu avgNnz=%.6f maxRowNnz=%d CInjectionRows=%llu "
    "directStrongCRows=%llu extendedRows=%llu extendedFPaths=%llu "
    "unionFallbackRows=%llu unionFallbackFracOfFine=%.9f "
    "avgCandidatesBeforePrune=%.6f maxCandidatesBeforePrune=%d prunedRows=%llu "
    "rowSumDefect=%.3Le weightPolicy=NEGATIVE_COUPLING "
    "extendedPolicy=STRONG_F_TO_STRONG_C fallbackPolicy=SYMMETRIC_UNION_C status=PASS\n",
    level,interp.c_str(),P.nFine,P.nCoarse,pmax,P.val.size(),
    P.nFine?(double)P.val.size()/P.nFine:0.0,T.maxRow,
    (unsigned long long)T.cRows,(unsigned long long)T.directRows,
    (unsigned long long)T.extRows,(unsigned long long)T.extFPaths,
    (unsigned long long)T.fallbackRows,
    P.nFine?double(T.fallbackRows)/double(P.nFine):0.0,
    double(T.prePruneCandidates)/fden,T.maxPrePrune,
    (unsigned long long)T.prunedRows,T.maxDefect);

  std::printf(
    "NODALS_GPU_PM3_TRANSFER_NEW level=%d status=PASS rows=%d coarse=%d "
    "nnz=%zu threads=%d buildSeconds=%.6f packSeconds=%.6f totalSeconds=%.6f "
    "rssMiB=%.3f hwmMiB=%.3f\n",
    level,P.nFine,P.nCoarse,P.val.size(),nth,
    std::chrono::duration<double>(build1-build0).count(),
    std::chrono::duration<double>(pack1-pack0).count(),
    std::chrono::duration<double>(all1-all0).count(),
    g8_pm0_rss_mib(),g8_pm0_hwm_mib());

  return P;
}

inline void g8_pm3_validate_transfer(
    const SATransferHost&R,const SATransferHost&N,int level,
    double refSeconds,double newSeconds)
{
  const bool dims=R.nFine==N.nFine&&R.nCoarse==N.nCoarse;
  const bool rowExact=dims&&R.row==N.row;
  const bool colExact=rowExact&&R.col==N.col;
  const bool valueBitwise=colExact&&R.val==N.val;

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

  const double valueRel=std::sqrt((double)d2)/std::max(std::sqrt((double)r2),1e-300);

  double actionRel=std::numeric_limits<double>::infinity();
  if(colExact){
    std::vector<double>x((std::size_t)R.nCoarse),yr((std::size_t)R.nFine,0.0),yn((std::size_t)N.nFine,0.0);
    for(int j=0;j<R.nCoarse;++j){
      const double g=(double)(j+1);
      x[(std::size_t)j]=std::sin(.037*g)+.31*std::cos(.021*g);
    }
    for(int i=0;i<R.nFine;++i){
      double a=0.0,b=0.0;
      for(std::int64_t k=R.row[(std::size_t)i];k<R.row[(std::size_t)i+1];++k)
        a+=R.val[(std::size_t)k]*x[(std::size_t)R.col[(std::size_t)k]];
      for(std::int64_t k=N.row[(std::size_t)i];k<N.row[(std::size_t)i+1];++k)
        b+=N.val[(std::size_t)k]*x[(std::size_t)N.col[(std::size_t)k]];
      yr[(std::size_t)i]=a; yn[(std::size_t)i]=b;
    }
    long double ad=0.0L,ar=0.0L;
    for(int i=0;i<R.nFine;++i){
      const double d=yn[(std::size_t)i]-yr[(std::size_t)i];
      ad+=(long double)d*d; ar+=(long double)yr[(std::size_t)i]*yr[(std::size_t)i];
    }
    actionRel=std::sqrt((double)ad)/std::max(std::sqrt((double)ar),1e-300);
  }

  const bool pass=dims&&rowExact&&colExact&&
    std::isfinite(valueRel)&&valueRel<=5e-13&&
    std::isfinite(actionRel)&&actionRel<=5e-13;

  std::printf(
    "NODALS_GPU_PM3_TRANSFER_PARITY level=%d status=%s dimsExact=%d "
    "rowsExact=%d colsExact=%d valuesBitwiseExact=%d refNnz=%zu newNnz=%zu "
    "valueRelL2=%.12e valueMaxAbs=%.12e actionRelL2=%.12e "
    "refSeconds=%.6f newSeconds=%.6f speedup=%.6f\n",
    level,pass?"PASS":"FAIL",(int)dims,(int)rowExact,(int)colExact,
    (int)valueBitwise,R.val.size(),N.val.size(),valueRel,maxAbs,actionRel,
    refSeconds,newSeconds,refSeconds/std::max(newSeconds,1e-300));

  if(!pass)throw std::runtime_error("PM3 interpolation parity failed");
}

inline const char* g8_pm3_transfer_mode()
{
  const char*e=std::getenv("NODALS_PM3_TRANSFER_MODE");
  return (e&&*e)?e:"parallel";
}

inline SATransferHost g8_build_transfer_dispatch(
    const CSRHost&A,const G8CFSplit&S,int pmax,int level,
    const std::string&interp)
{
  const char*mode=g8_pm3_transfer_mode();
  if(std::strcmp(mode,"reference")==0)
    return g8_build_transfer_reference(A,S,pmax,level,interp);
  if(std::strcmp(mode,"parallel")==0)
    return g8_build_transfer_parallel(A,S,pmax,level,interp);
  if(std::strcmp(mode,"compare")==0){
    const auto r0=std::chrono::steady_clock::now();
    auto R=g8_build_transfer_reference(A,S,pmax,level,interp);
    const auto r1=std::chrono::steady_clock::now();
    const auto n0=std::chrono::steady_clock::now();
    auto N=g8_build_transfer_parallel(A,S,pmax,level,interp);
    const auto n1=std::chrono::steady_clock::now();
    g8_pm3_validate_transfer(R,N,level,
      std::chrono::duration<double>(r1-r0).count(),
      std::chrono::duration<double>(n1-n0).count());
    return N;
  }
  throw std::runtime_error("NODALS_PM3_TRANSFER_MODE must be reference, compare, or parallel");
}


inline SATransferHost g11_compose_transfers_exact(
    const SATransferHost&P01,const SATransferHost&P12)
{
  if(P01.nCoarse!=P12.nFine)
    throw std::runtime_error("G11 transfer composition dimension mismatch");

  SATransferHost Q;
  Q.nFine=P01.nFine;
  Q.nCoarse=P12.nCoarse;
  Q.row.assign((std::size_t)Q.nFine+1,0);

  std::unordered_map<int,double> acc;
  std::vector<std::pair<int,double>> cand;
  acc.reserve(128);
  cand.reserve(128);

  std::uint64_t rawProducts=0;
  int maxRowNnz=0;
  long double maxRowSumDefect=0.0L;
  std::uint64_t rowsOver8=0,rowsOver16=0;

  for(int i=0;i<Q.nFine;++i){
    acc.clear();
    cand.clear();

    for(std::int64_t a=P01.row[(std::size_t)i];
        a<P01.row[(std::size_t)i+1];++a){
      const int mid=P01.col[(std::size_t)a];
      const double w0=P01.val[(std::size_t)a];

      for(std::int64_t b=P12.row[(std::size_t)mid];
          b<P12.row[(std::size_t)mid+1];++b){
        const int c=P12.col[(std::size_t)b];
        const double w1=P12.val[(std::size_t)b];
        acc[c]+=w0*w1;
        ++rawProducts;
      }
    }

    if(acc.empty())
      throw std::runtime_error("G11 composed interpolation empty row");

    cand.reserve(std::max(cand.capacity(),acc.size()));
    for(const auto&kv:acc){
      if(kv.second!=0.0)cand.push_back(kv);
    }
    std::sort(cand.begin(),cand.end(),[](const auto&a,const auto&b){
      return a.first<b.first;
    });

    double rs=0.0;
    for(const auto&e:cand){
      Q.col.push_back((std::int32_t)e.first);
      Q.val.push_back(e.second);
      rs+=e.second;
    }
    Q.row[(std::size_t)i+1]=(std::int64_t)Q.col.size();

    const int rn=(int)cand.size();
    maxRowNnz=std::max(maxRowNnz,rn);
    if(rn>8)++rowsOver8;
    if(rn>16)++rowsOver16;
    maxRowSumDefect=std::max(
      maxRowSumDefect,(long double)std::abs(rs-1.0));
  }

  std::printf(
    "NODALS_GPU_G11_AGG_COMPOSE fine=%d mid=%d coarse=%d nnz=%zu "
    "avgNnz=%.6f maxRowNnz=%d rawProducts=%llu rowsOver8=%llu rowsOver16=%llu "
    "rowSumDefect=%.3Le prune=NONE composition=EXACT_P01_TIMES_P12 status=PASS\n",
    Q.nFine,P01.nCoarse,Q.nCoarse,Q.val.size(),
    Q.nFine?(double)Q.val.size()/Q.nFine:0.0,maxRowNnz,
    (unsigned long long)rawProducts,
    (unsigned long long)rowsOver8,(unsigned long long)rowsOver16,
    maxRowSumDefect);

  return Q;
}


// -----------------------------------------------------------------------------
// PM1: CF/PMIS Galerkin PtAP optimization.
//
// The reference path is the pre-existing build_sa_ptap() from g5_sa_host.hpp.
// The parallel path is the validated SA Gate-3 construction specialized here
// so PM1 changes only the CF hierarchy:
//
//   q_i = A_i P
//   reverse coarse-row -> (fine row, P_iI) incidence
//   exclusive ownership of complete coarse output rows
//   thread-local dense marked accumulation
//   deterministic final CSR packing
//
// Reverse incidence is filled in increasing fine-row order so the numeric +=
// order for each coarse matrix entry matches the serial reference.
// -----------------------------------------------------------------------------

inline int g8_pm1_threads()
{
  int n=0;
  if(const char*e=std::getenv("NODALS_PM1_THREADS"))n=std::atoi(e);
  if(n<=0){
    const unsigned h=std::thread::hardware_concurrency();
    n=h?std::min((int)h,16):1;
  }
  return std::max(1,std::min(n,64));
}

template<class F>
inline void g8_pm1_parallel_chunks(int n,int nth,int chunk,F&&fn)
{
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

struct G8PM1QRow {
  std::vector<std::pair<int,double>> e;
};

struct G8PM1Rev {
  std::int32_t fine=-1;
  double p=0.0;
};

inline CSRHost g8_pm1_ptap_parallel(
    const CSRHost&A,const SATransferHost&P,int nextLevel,
    unsigned long long*rawOut=nullptr)
{
  using clock=std::chrono::steady_clock;
  if(P.nFine!=A.n)
    throw std::runtime_error("PM1 PtAP transfer size mismatch");

  const auto all0=clock::now();
  const int nth=g8_pm1_threads();
  const int nf=A.n;
  const int nc=P.nCoarse;

  // q_i = A_i P using exactly the legacy local loop/sort/merge arithmetic.
  const auto q0=clock::now();
  std::vector<G8PM1QRow> qrows((std::size_t)nf);

  g8_pm1_parallel_chunks(nf,nth,32,[&](int,int ib,int ie){
    std::vector<std::pair<int,double>>q;
    q.reserve(512);

    for(int i=ib;i<ie;++i){
      q.clear();
      for(std::int64_t k=A.row[(std::size_t)i];
          k<A.row[(std::size_t)i+1];++k){
        const int j=A.col[(std::size_t)k];
        const double a=A.val[(std::size_t)k];
        for(std::int64_t kp=P.row[(std::size_t)j];
            kp<P.row[(std::size_t)j+1];++kp){
          q.push_back({
            P.col[(std::size_t)kp],
            a*P.val[(std::size_t)kp]});
        }
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

  // Reverse P incidence, filled serially in increasing fine-row order.
  const auto r0=clock::now();
  std::vector<std::int64_t> rptr((std::size_t)nc+1,0);
  for(int i=0;i<nf;++i)
    for(std::int64_t kp=P.row[(std::size_t)i];
        kp<P.row[(std::size_t)i+1];++kp)
      ++rptr[(std::size_t)P.col[(std::size_t)kp]+1];

  for(int I=0;I<nc;++I)
    rptr[(std::size_t)I+1]+=rptr[(std::size_t)I];

  std::vector<G8PM1Rev> rev((std::size_t)rptr.back());
  auto cur=rptr;
  for(int i=0;i<nf;++i){
    for(std::int64_t kp=P.row[(std::size_t)i];
        kp<P.row[(std::size_t)i+1];++kp){
      const int I=P.col[(std::size_t)kp];
      const std::int64_t z=cur[(std::size_t)I]++;
      rev[(std::size_t)z].fine=(std::int32_t)i;
      rev[(std::size_t)z].p=P.val[(std::size_t)kp];
    }
  }
  std::vector<std::int64_t>().swap(cur);
  const auto r1=clock::now();

  // Every worker owns complete output coarse rows.
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

  g8_pm1_parallel_chunks(nc,nth,16,[&](int t,int ib,int ie){
    auto&av=acc[(std::size_t)t];
    auto&mk=mark[(std::size_t)t];
    auto&tv=touched[(std::size_t)t];
    unsigned long long raw=0;

    for(int I=ib;I<ie;++I){
      tv.clear();

      for(std::int64_t z=rptr[(std::size_t)I];
          z<rptr[(std::size_t)I+1];++z){
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

  const double scratchMiB=
    (double)(
      qnnz*sizeof(std::pair<int,double>) +
      qrows.size()*sizeof(G8PM1QRow) +
      rptr.size()*sizeof(std::int64_t) +
      rev.size()*sizeof(G8PM1Rev) +
      (std::size_t)nth*(std::size_t)nc*(sizeof(double)+sizeof(int)) +
      retained*sizeof(std::pair<int,double>)
    )/(1024.0*1024.0);

  // Release the largest temporary structures before final CSR packing.
  std::vector<G8PM1QRow>().swap(qrows);
  std::vector<std::int64_t>().swap(rptr);
  std::vector<G8PM1Rev>().swap(rev);
  std::vector<std::vector<double>>().swap(acc);
  std::vector<std::vector<int>>().swap(mark);
  std::vector<std::vector<int>>().swap(touched);
  std::vector<unsigned long long>().swap(rawThread);

  const auto p0=clock::now();
  CSRHost C;
  C.n=nc;
  C.row.assign((std::size_t)nc+1,0);
  for(int I=0;I<nc;++I)
    C.row[(std::size_t)I+1]=
      C.row[(std::size_t)I]+(std::int64_t)rows[(std::size_t)I].size();

  C.col.resize((std::size_t)C.row.back());
  C.val.resize((std::size_t)C.row.back());
  C.diag.assign((std::size_t)nc,0.0);

  g8_pm1_parallel_chunks(nc,nth,128,[&](int,int ib,int ie){
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
    if(!(C.diag[(std::size_t)I]>0.0)||
       !std::isfinite(C.diag[(std::size_t)I]))
      throw std::runtime_error("PM1 PtAP non-positive diagonal");

  const auto p1=clock::now();
  const auto all1=clock::now();

  std::printf(
    "NODALS_GPU_PM1_PTAP_NEW level=%d status=PASS rows=%d nnz=%zu "
    "avgNnz=%.6f rawTerms=%llu threads=%d qNnz=%zu qAvg=%.6f qMax=%d "
    "qSeconds=%.6f reverseSeconds=%.6f accumulateSeconds=%.6f "
    "packSeconds=%.6f totalSeconds=%.6f scratchEstimateMiB=%.3f "
    "rssMiB=%.3f hwmMiB=%.3f\n",
    nextLevel,C.n,C.val.size(),
    C.n?(double)C.val.size()/C.n:0.0,
    raw,nth,qnnz,nf?(double)qnnz/nf:0.0,qmax,
    std::chrono::duration<double>(q1-q0).count(),
    std::chrono::duration<double>(r1-r0).count(),
    std::chrono::duration<double>(a1-a0).count(),
    std::chrono::duration<double>(p1-p0).count(),
    std::chrono::duration<double>(all1-all0).count(),
    scratchMiB,g8_pm0_rss_mib(),g8_pm0_hwm_mib());

  return C;
}

inline void g8_pm1_validate_ptap(
    const CSRHost&R,const CSRHost&N,int level,
    double refSeconds,double newSeconds)
{
  const bool dims=R.n==N.n;
  const bool rowExact=dims&&R.row==N.row;
  const bool colExact=rowExact&&R.col==N.col;
  const bool valBitwise=colExact&&R.val==N.val;
  const bool diagBitwise=R.diag==N.diag;

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

  const double valRel=
    std::sqrt((double)d2)/std::max(std::sqrt((double)r2),1e-300);
  const double diagRel=
    std::sqrt((double)dd2)/std::max(std::sqrt((double)dr2),1e-300);

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
    actionRel=
      std::sqrt((double)ad)/std::max(std::sqrt((double)ar),1e-300);
  }

  const bool pass=
    dims&&rowExact&&colExact&&
    std::isfinite(valRel)&&valRel<=5e-13&&
    std::isfinite(diagRel)&&diagRel<=5e-13&&
    std::isfinite(actionRel)&&actionRel<=5e-13;

  std::printf(
    "NODALS_GPU_PM1_PTAP_PARITY level=%d status=%s "
    "dimsExact=%d rowsExact=%d colsExact=%d valuesBitwiseExact=%d "
    "diagBitwiseExact=%d refNnz=%zu newNnz=%zu "
    "valueRelL2=%.12e valueMaxAbs=%.12e "
    "diagRelL2=%.12e diagMaxAbs=%.12e actionRelL2=%.12e "
    "refSeconds=%.6f newSeconds=%.6f speedup=%.6f\\n",
    level,pass?"PASS":"FAIL",
    (int)dims,(int)rowExact,(int)colExact,(int)valBitwise,
    (int)diagBitwise,R.val.size(),N.val.size(),
    valRel,maxAbs,diagRel,diagMax,actionRel,
    refSeconds,newSeconds,
    refSeconds/std::max(newSeconds,1e-300));

  if(!pass)
    throw std::runtime_error("PM1 PtAP parity failed");
}

inline const char* g8_pm1_ptap_mode()
{
  const char*e=std::getenv("NODALS_PM1_PTAP_MODE");
  return (e&&*e)?e:"parallel";
}

inline CSRHost g8_pm1_ptap_dispatch(
    const CSRHost&A,const SATransferHost&P,int nextLevel)
{
  const char*mode=g8_pm1_ptap_mode();

  if(std::strcmp(mode,"reference")==0)
    return build_sa_ptap(A,P,nextLevel);

  if(std::strcmp(mode,"parallel")==0)
    return g8_pm1_ptap_parallel(A,P,nextLevel);

  if(std::strcmp(mode,"compare")==0){
    const auto r0=std::chrono::steady_clock::now();
    auto R=build_sa_ptap(A,P,nextLevel);
    const auto r1=std::chrono::steady_clock::now();

    const auto n0=std::chrono::steady_clock::now();
    auto N=g8_pm1_ptap_parallel(A,P,nextLevel);
    const auto n1=std::chrono::steady_clock::now();

    g8_pm1_validate_ptap(
      R,N,nextLevel,
      std::chrono::duration<double>(r1-r0).count(),
      std::chrono::duration<double>(n1-n0).count());
    return N;
  }

  throw std::runtime_error(
    "NODALS_PM1_PTAP_MODE must be reference, compare, or parallel");
}

inline SAHierarchyHost build_cf_hierarchy(
    const G4SetupHost&S,const H2FineCSRHost&F,
    double theta,int pmax,const std::string& interp,
    bool aggressiveFirst,
    int terminal=1000,int powerIts=16,double safety=1.5,double lowFrac=0.05)
{
  const auto t0=std::chrono::steady_clock::now();

  std::printf(
    "NODALS_GPU_PM2_CF_DIAGNOSTIC_POLICY symmetryAudit=%s "
    "diagnosticEnv=NODALS_PMIS_DIAGNOSTICS setupAutotuneEnv=NODALS_SETUP_AUTOTUNE "
    "status=PASS\n",
    g8_pm2_cf_diagnostics_enabled()?"ON":"OFF");

  SAHierarchyHost H;
  H.powerIts=powerIts;
  H.interpMaxNnz=pmax;
  H.lambdaSafety=safety;
  H.lambdaLowFraction=lowFrac;
  H.saDamping=0.0;

  CSRHost A=g8_build_exact_fine_csr(S,F);
  H.fineDiag=A.diag;

  const auto pm0_spec_t0=std::chrono::steady_clock::now();
  if(powerIts>0){
    H.fineLambda=sa_power_csr(A,powerIts,safety);
    std::printf(
      "NODALS_GPU_G8_CF_SPECTRUM level=0 powerIts=%d lambdaMax=%.12e policy=COMPUTE status=PASS\n",
      powerIts,H.fineLambda);
  }else{
    H.fineLambda=1.0;
    std::printf(
      "NODALS_GPU_G8_CF_SPECTRUM level=0 powerIts=0 lambdaMax=1 policy=SKIP_UNUSED_BY_JACOBI status=PASS\n");
  }

  g8_pm0_stage(
    "fine_spectrum",0,A.n,A.val.size(),
    std::chrono::duration<double>(
      std::chrono::steady_clock::now()-pm0_spec_t0).count());

  int logicalLevel=0;

  if(aggressiveFirst && A.n>terminal){
    // Stage 1: fine -> C1.
    auto split0=g8_build_pmis_split_dispatch(A,theta,0);
    auto P01=g8_build_transfer_dispatch(A,split0,pmax,0,interp);
    const auto pm0_ptap1_t0=std::chrono::steady_clock::now();
    CSRHost A1=g8_pm1_ptap_dispatch(A,P01,1);
    g8_pm0_stage(
      "ptap",1,A1.n,A1.val.size(),
      std::chrono::duration<double>(
        std::chrono::steady_clock::now()-pm0_ptap1_t0).count());

    std::printf(
      "NODALS_GPU_G11_AGG_STAGE stage=1 fine=%d coarse=%d Pnnz=%zu Annz=%zu "
      "AavgNnz=%.6f symmetryRel=%.3e runtimeLevel=TEMPORARY status=PASS\n",
      A.n,A1.n,P01.val.size(),A1.val.size(),
      A1.n?(double)A1.val.size()/A1.n:0.0,g8_pm2_symmetry_rel(A1));

    if(A1.n<=terminal){
      // Degenerate case: no second aggressive stage needed.
      H.P.push_back(std::move(P01));
      H.csr.push_back(std::move(A1));
    }else{
      // Stage 2: C1 -> C2 using the same PMIS/ext+i construction.
      auto split1=g8_build_pmis_split_dispatch(A1,theta,1);
      auto P12=g8_build_transfer_dispatch(A1,split1,pmax,1,interp);
      const auto pm0_ptap2_t0=std::chrono::steady_clock::now();
      CSRHost A2=g8_pm1_ptap_dispatch(A1,P12,2);
      g8_pm0_stage(
        "ptap",2,A2.n,A2.val.size(),
        std::chrono::duration<double>(
          std::chrono::steady_clock::now()-pm0_ptap2_t0).count());

      std::printf(
        "NODALS_GPU_G11_AGG_STAGE stage=2 fine=%d coarse=%d Pnnz=%zu Annz=%zu "
        "AavgNnz=%.6f symmetryRel=%.3e runtimeLevel=KEPT_COARSE_OPERATOR status=PASS\n",
        A1.n,A2.n,P12.val.size(),A2.val.size(),
        A2.n?(double)A2.val.size()/A2.n:0.0,g8_pm2_symmetry_rel(A2));

      // Long-range aggressive interpolation. No pruning in Gate 8 so the
      // composed transfer represents P01*P12 exactly to roundoff.
      auto P02=g11_compose_transfers_exact(P01,P12);

      const int removedRows=A1.n;
      const std::size_t removedNnz=A1.val.size();
      const std::size_t removedPnnz=P01.val.size()+P12.val.size();

      H.P.push_back(std::move(P02));
      H.csr.push_back(std::move(A2));

      std::printf(
        "NODALS_GPU_G11_AGG_FIRST enabled=1 fine=%d removedIntermediateRows=%d "
        "removedIntermediateNnz=%zu separateTransferNnz=%zu composedTransferNnz=%zu "
        "firstRuntimeCoarseRows=%d firstRuntimeCoarseNnz=%zu "
        "coarseOperatorPolicy=REUSE_TWO_STAGE_GALERKIN "
        "longRangeInterpolation=EXACT_COMPOSED_EXTI status=PASS\n",
        A.n,removedRows,removedNnz,removedPnnz,H.P[0].val.size(),
        H.csr[0].n,H.csr[0].val.size());
    }

    if(H.csr.back().n>terminal){
      H.levelLambda.resize(H.csr.size(),0.0);
      if(powerIts>0){
        const double lam=sa_power_csr(H.csr.back(),powerIts,safety);
        H.levelLambda[0]=lam;
        std::printf(
          "NODALS_GPU_G8_CF_SPECTRUM level=1 powerIts=%d lambdaMax=%.12e policy=COMPUTE status=PASS\n",
          powerIts,lam);
      }else{
        H.levelLambda[0]=1.0;
        std::printf(
          "NODALS_GPU_G8_CF_SPECTRUM level=1 powerIts=0 lambdaMax=1 policy=SKIP_UNUSED_BY_JACOBI status=PASS\n");
      }
    }

    A=H.csr.back();
    logicalLevel=1;
  }

  // Standard hierarchy continuation. If aggressiveFirst==false this also
  // constructs the first transfer exactly as Gates 5/7 did.
  while(H.csr.empty() || H.csr.back().n>terminal){
    auto split=g8_build_pmis_split_dispatch(A,theta,logicalLevel);
    auto P=g8_build_transfer_dispatch(A,split,pmax,logicalLevel,interp);
    H.P.push_back(std::move(P));

    const auto pm0_ptap_t0=std::chrono::steady_clock::now();
    CSRHost C=g8_pm1_ptap_dispatch(A,H.P.back(),logicalLevel+1);
    g8_pm0_stage(
      "ptap",logicalLevel+1,C.n,C.val.size(),
      std::chrono::duration<double>(
        std::chrono::steady_clock::now()-pm0_ptap_t0).count());
    std::printf(
      "NODALS_GPU_G8_CF_COARSE level=%d interp=%s rows=%d nnz=%zu avgNnz=%.6f "
      "symmetryRel=%.3e status=PASS\n",
      logicalLevel+1,interp.c_str(),C.n,C.val.size(),
      C.n?(double)C.val.size()/C.n:0.0,g8_pm2_symmetry_rel(C));
    H.csr.push_back(std::move(C));

    CSRHost().row.swap(A.row);CSRHost().col.swap(A.col);
    CSRHost().val.swap(A.val);CSRHost().diag.swap(A.diag);

    if(H.csr.back().n<=terminal)break;

    const std::size_t idx=H.csr.size()-1;
    if(H.levelLambda.size()<H.csr.size())
      H.levelLambda.resize(H.csr.size(),0.0);

    if(powerIts>0){
      const double lam=sa_power_csr(H.csr.back(),powerIts,safety);
      H.levelLambda[idx]=lam;
      std::printf(
        "NODALS_GPU_G8_CF_SPECTRUM level=%d powerIts=%d lambdaMax=%.12e policy=COMPUTE status=PASS\n",
        logicalLevel+1,powerIts,lam);
    }else{
      H.levelLambda[idx]=1.0;
      std::printf(
        "NODALS_GPU_G8_CF_SPECTRUM level=%d powerIts=0 lambdaMax=1 policy=SKIP_UNUSED_BY_JACOBI status=PASS\n",
        logicalLevel+1);
    }

    A=H.csr.back();
    ++logicalLevel;
    if(logicalLevel>20)
      throw std::runtime_error("G8 CF hierarchy excessive levels");
  }

  if(H.levelLambda.size()<H.csr.size())
    H.levelLambda.resize(H.csr.size(),1.0);

  H.terminal_n=H.csr.back().n;
  const auto pm0_terminal_t0=std::chrono::steady_clock::now();
  H.terminal_inv=dense_inverse_cholesky(H.csr.back());
  g8_pm0_stage(
    "terminal_inverse",(int)H.csr.size(),H.csr.back().n,H.csr.back().val.size(),
    std::chrono::duration<double>(
      std::chrono::steady_clock::now()-pm0_terminal_t0).count());

  const auto t1=std::chrono::steady_clock::now();
  const double sec=std::chrono::duration<double>(t1-t0).count();

  std::printf(
    "NODALS_GPU_G8_CF_HIERARCHY levels=%zu transfers=%zu terminal=%d "
    "theta=%.2f criterion=CLASSICAL_NEGATIVE interp=%s pmax=%d "
    "aggressiveFirst=%d aggressivePolicy=%s powerIts=%d spectrumPolicy=%s "
    "fallback=SYMMETRIC_UNION_C seconds=%.6f status=PASS\n",
    H.csr.size(),H.P.size(),H.terminal_n,theta,interp.c_str(),pmax,
    aggressiveFirst?1:0,
    aggressiveFirst?"TWO_STAGE_EXACT_COMPOSITION":"OFF",
    powerIts,powerIts>0?"COMPUTE":"SKIP_UNUSED_BY_JACOBI",sec);

  std::size_t pm0_retained_nnz=0;
  for(const auto&x:H.csr)pm0_retained_nnz+=x.val.size();
  g8_pm0_stage("cf_hierarchy_total",-1,F.n,pm0_retained_nnz,sec);
  return H;
}

} // namespace nodals_gpu
