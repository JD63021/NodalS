#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

namespace nodals_gpu {

inline std::uint64_t g7_splitmix64(std::uint64_t x){
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

inline bool g7_priority_higher(
    int a,int b,
    const std::vector<std::vector<std::int32_t>>& graph)
{
  const std::size_t da=graph[(std::size_t)a].size();
  const std::size_t db=graph[(std::size_t)b].size();
  if(da!=db) return da>db;
  const std::uint64_t ha=g7_splitmix64((std::uint64_t)a+0x63021ull);
  const std::uint64_t hb=g7_splitmix64((std::uint64_t)b+0x63021ull);
  if(ha!=hb) return ha>hb;
  return a>b;
}

template<class T>
inline void g7_pmis_diagnostics(
    int n,
    const std::vector<std::int32_t>& row,
    const std::vector<std::int32_t>& col,
    const std::vector<T>& val,
    double theta,
    const char* tag)
{
  if(n<=0) throw std::runtime_error("G7 PMIS invalid row count");
  if(row.size()!=(std::size_t)n+1 || col.size()!=val.size())
    throw std::runtime_error("G7 PMIS CSR size mismatch");

  const auto t0=std::chrono::steady_clock::now();

  std::vector<double> maxNeg((std::size_t)n,0.0);
  for(int i=0;i<n;++i){
    for(std::int32_t k=row[(std::size_t)i];k<row[(std::size_t)i+1];++k){
      const int j=col[(std::size_t)k];
      if(j==i) continue;
      const double a=(double)val[(std::size_t)k];
      if(a<0.0) maxNeg[(std::size_t)i]=std::max(maxNeg[(std::size_t)i],-a);
    }
  }

  std::vector<std::vector<std::int32_t>> strong((std::size_t)n);
  std::uint64_t directedStrong=0;
  int strongMin=std::numeric_limits<int>::max(),strongMax=0,strongIsolated=0;

  for(int i=0;i<n;++i){
    const double thr=theta*maxNeg[(std::size_t)i];
    auto& out=strong[(std::size_t)i];
    if(thr>0.0){
      for(std::int32_t k=row[(std::size_t)i];k<row[(std::size_t)i+1];++k){
        const int j=col[(std::size_t)k];
        if(j==i) continue;
        const double a=(double)val[(std::size_t)k];
        if(a<0.0 && -a>=thr) out.push_back((std::int32_t)j);
      }
    }
    std::sort(out.begin(),out.end());
    out.erase(std::unique(out.begin(),out.end()),out.end());
    const int d=(int)out.size();
    directedStrong+=(std::uint64_t)d;
    strongMin=std::min(strongMin,d);
    strongMax=std::max(strongMax,d);
    if(d==0)++strongIsolated;
  }

  std::vector<std::vector<std::int32_t>> graph((std::size_t)n);
  for(int i=0;i<n;++i){
    for(int j:strong[(std::size_t)i]){
      graph[(std::size_t)i].push_back((std::int32_t)j);
      graph[(std::size_t)j].push_back((std::int32_t)i);
    }
  }

  std::uint64_t unionDirected=0;
  int unionMin=std::numeric_limits<int>::max(),unionMax=0,unionIsolated=0;
  for(auto& g:graph){
    std::sort(g.begin(),g.end());
    g.erase(std::unique(g.begin(),g.end()),g.end());
    const int d=(int)g.size();
    unionDirected+=(std::uint64_t)d;
    unionMin=std::min(unionMin,d);
    unionMax=std::max(unionMax,d);
    if(d==0)++unionIsolated;
  }

  std::vector<std::uint8_t> state((std::size_t)n,0); // 0 undecided, 1 C, 2 F
  std::vector<std::int32_t> winners;
  winners.reserve((std::size_t)n/8u+1u);
  int undecided=n,rounds=0;

  while(undecided>0){
    winners.clear();

    for(int i=0;i<n;++i){
      if(state[(std::size_t)i]!=0) continue;
      bool win=true;
      for(int j:graph[(std::size_t)i]){
        if(state[(std::size_t)j]==0 && g7_priority_higher(j,i,graph)){
          win=false;
          break;
        }
      }
      if(win) winners.push_back((std::int32_t)i);
    }

    if(winners.empty())
      throw std::runtime_error("G7 PMIS made no progress");

    for(int i:winners){
      if(state[(std::size_t)i]==0){
        state[(std::size_t)i]=1;
        --undecided;
      }
    }

    for(int i:winners){
      for(int j:graph[(std::size_t)i]){
        if(state[(std::size_t)j]==0){
          state[(std::size_t)j]=2;
          --undecided;
        }
      }
    }

    ++rounds;
    if(rounds>n) throw std::runtime_error("G7 PMIS excessive rounds");
  }

  std::uint64_t nC=0,nF=0;
  for(auto s:state){
    if(s==1)++nC;
    else if(s==2)++nF;
    else throw std::runtime_error("G7 PMIS undecided row after completion");
  }

  std::uint64_t ccDirected=0;
  for(int i=0;i<n;++i) if(state[(std::size_t)i]==1){
    for(int j:strong[(std::size_t)i])
      if(state[(std::size_t)j]==1) ++ccDirected;
  }

  std::uint64_t fZeroOut=0,fOneOut=0,fTwoPlusOut=0,fZeroUnion=0;
  std::uint64_t outCsum=0,unionCsum=0;
  int outCmax=0,unionCmax=0;

  for(int i=0;i<n;++i){
    if(state[(std::size_t)i]!=2) continue;

    int outC=0;
    for(int j:strong[(std::size_t)i])
      if(state[(std::size_t)j]==1) ++outC;

    int unionC=0;
    for(int j:graph[(std::size_t)i])
      if(state[(std::size_t)j]==1) ++unionC;

    outCsum+=(std::uint64_t)outC;
    unionCsum+=(std::uint64_t)unionC;
    outCmax=std::max(outCmax,outC);
    unionCmax=std::max(unionCmax,unionC);

    if(outC==0)++fZeroOut;
    else if(outC==1)++fOneOut;
    else ++fTwoPlusOut;

    if(unionC==0)++fZeroUnion;
  }

  std::uint64_t checksum=0xcbf29ce484222325ull;
  for(int i=0;i<n;++i) if(state[(std::size_t)i]==1){
    checksum ^= g7_splitmix64((std::uint64_t)i+1ull);
    checksum *= 0x100000001b3ull;
  }

  const auto t1=std::chrono::steady_clock::now();
  const double sec=std::chrono::duration<double>(t1-t0).count();
  const double fden=nF?double(nF):1.0;

  std::printf(
    "NODALS_GPU_G7_PMIS_GRAPH tag=%s criterion=CLASSICAL_NEGATIVE theta=%.2f "
    "rows=%d directedStrong=%llu avgDirectedDegree=%.6f minDirectedDegree=%d "
    "maxDirectedDegree=%d directedIsolated=%d unionUndirectedEdges=%llu "
    "avgUnionDegree=%.6f minUnionDegree=%d maxUnionDegree=%d unionIsolated=%d "
    "setupOnly=1 solverHierarchy=UNCHANGED status=PASS\n",
    tag,theta,n,(unsigned long long)directedStrong,double(directedStrong)/double(n),
    strongMin==std::numeric_limits<int>::max()?0:strongMin,strongMax,strongIsolated,
    (unsigned long long)(unionDirected/2ull),double(unionDirected)/double(n),
    unionMin==std::numeric_limits<int>::max()?0:unionMin,unionMax,unionIsolated);

  std::printf(
    "NODALS_GPU_G7_PMIS_SPLIT tag=%s algorithm=DETERMINISTIC_SERIAL_PMIS_LIKE "
    "graph=SYMMETRIC_UNION_OF_CLASSICAL_NEGATIVE_STRENGTH theta=%.2f "
    "C=%llu F=%llu coarseRatio=%.9f finePerCoarse=%.6f rounds=%d "
    "strongCCDirected=%llu deterministicChecksum=%llu seconds=%.6f "
    "setupOnly=1 solverHierarchy=UNCHANGED status=PASS\n",
    tag,theta,(unsigned long long)nC,(unsigned long long)nF,
    double(nC)/double(n),nC?double(n)/double(nC):0.0,rounds,
    (unsigned long long)ccDirected,(unsigned long long)checksum,sec);

  std::printf(
    "NODALS_GPU_G7_PMIS_F_COVERAGE tag=%s theta=%.2f F=%llu "
    "outgoingStrongC_zero=%llu outgoingStrongC_one=%llu outgoingStrongC_twoPlus=%llu "
    "outgoingStrongC_avg=%.6f outgoingStrongC_max=%d "
    "unionC_zero=%llu unionC_avg=%.6f unionC_max=%d "
    "directInterpolationCoverageFrac=%.9f unionCoverageFrac=%.9f "
    "setupOnly=1 solverHierarchy=UNCHANGED status=PASS\n",
    tag,theta,(unsigned long long)nF,
    (unsigned long long)fZeroOut,(unsigned long long)fOneOut,
    (unsigned long long)fTwoPlusOut,double(outCsum)/fden,outCmax,
    (unsigned long long)fZeroUnion,double(unionCsum)/fden,unionCmax,
    1.0-double(fZeroOut)/fden,1.0-double(fZeroUnion)/fden);

  std::printf(
    "NODALS_GPU_G7_PMIS_SUMMARY tag=%s theta=%.2f criterion=CLASSICAL_NEGATIVE "
    "splitBuilt=1 transferBuilt=0 galerkinBuilt=0 solverPathChanged=0 status=PASS\n",
    tag,theta);
}

} // namespace nodals_gpu
