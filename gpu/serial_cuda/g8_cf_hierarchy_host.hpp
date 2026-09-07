#pragma once
#include "g5_sa_host.hpp"
#include "h2b_fine_csr.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nodals_gpu {

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
    A.n,A.val.size(),A.n?(double)A.val.size()/A.n:0.0,csr_symmetry_rel(A));
  return A;
}

inline G8CFSplit g8_build_pmis_split(const CSRHost&A,double theta,int level){
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
  return R;
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

inline SATransferHost g8_build_transfer(
    const CSRHost&A,const G8CFSplit&S,int pmax,int level,
    const std::string& interp)
{
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
  return P;
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

inline SAHierarchyHost build_cf_hierarchy(
    const G4SetupHost&S,const H2FineCSRHost&F,
    double theta,int pmax,const std::string& interp,
    bool aggressiveFirst,
    int terminal=1000,int powerIts=16,double safety=1.5,double lowFrac=0.05)
{
  const auto t0=std::chrono::steady_clock::now();

  SAHierarchyHost H;
  H.powerIts=powerIts;
  H.interpMaxNnz=pmax;
  H.lambdaSafety=safety;
  H.lambdaLowFraction=lowFrac;
  H.saDamping=0.0;

  CSRHost A=g8_build_exact_fine_csr(S,F);
  H.fineDiag=A.diag;

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

  int logicalLevel=0;

  if(aggressiveFirst && A.n>terminal){
    // Stage 1: fine -> C1.
    auto split0=g8_build_pmis_split(A,theta,0);
    auto P01=g8_build_transfer(A,split0,pmax,0,interp);
    CSRHost A1=build_sa_ptap(A,P01,1);

    std::printf(
      "NODALS_GPU_G11_AGG_STAGE stage=1 fine=%d coarse=%d Pnnz=%zu Annz=%zu "
      "AavgNnz=%.6f symmetryRel=%.3e runtimeLevel=TEMPORARY status=PASS\n",
      A.n,A1.n,P01.val.size(),A1.val.size(),
      A1.n?(double)A1.val.size()/A1.n:0.0,csr_symmetry_rel(A1));

    if(A1.n<=terminal){
      // Degenerate case: no second aggressive stage needed.
      H.P.push_back(std::move(P01));
      H.csr.push_back(std::move(A1));
    }else{
      // Stage 2: C1 -> C2 using the same PMIS/ext+i construction.
      auto split1=g8_build_pmis_split(A1,theta,1);
      auto P12=g8_build_transfer(A1,split1,pmax,1,interp);
      CSRHost A2=build_sa_ptap(A1,P12,2);

      std::printf(
        "NODALS_GPU_G11_AGG_STAGE stage=2 fine=%d coarse=%d Pnnz=%zu Annz=%zu "
        "AavgNnz=%.6f symmetryRel=%.3e runtimeLevel=KEPT_COARSE_OPERATOR status=PASS\n",
        A1.n,A2.n,P12.val.size(),A2.val.size(),
        A2.n?(double)A2.val.size()/A2.n:0.0,csr_symmetry_rel(A2));

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
    auto split=g8_build_pmis_split(A,theta,logicalLevel);
    auto P=g8_build_transfer(A,split,pmax,logicalLevel,interp);
    H.P.push_back(std::move(P));

    CSRHost C=build_sa_ptap(A,H.P.back(),logicalLevel+1);
    std::printf(
      "NODALS_GPU_G8_CF_COARSE level=%d interp=%s rows=%d nnz=%zu avgNnz=%.6f "
      "symmetryRel=%.3e status=PASS\n",
      logicalLevel+1,interp.c_str(),C.n,C.val.size(),
      C.n?(double)C.val.size()/C.n:0.0,csr_symmetry_rel(C));
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
  H.terminal_inv=dense_inverse_cholesky(H.csr.back());

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

  return H;
}

} // namespace nodals_gpu
