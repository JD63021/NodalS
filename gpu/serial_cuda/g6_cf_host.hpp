#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

namespace nodals_gpu {

template<class T>
inline T g6_row_value(
    int rowId,int colId,
    const std::vector<std::int32_t>& row,
    const std::vector<std::int32_t>& col,
    const std::vector<T>& val)
{
  const auto b=col.begin()+row[(std::size_t)rowId];
  const auto e=col.begin()+row[(std::size_t)rowId+1];
  const auto it=std::lower_bound(b,e,colId);
  if(it==e || *it!=colId) return T(0);
  return val[(std::size_t)(it-col.begin())];
}

template<class T>
inline void g6_strength_sweep(
    int n,
    const std::vector<std::int32_t>& row,
    const std::vector<std::int32_t>& col,
    const std::vector<T>& val,
    const char* tag)
{
  if(n<=0) throw std::runtime_error("G6 strength invalid row count");
  if(row.size()!=(std::size_t)n+1) throw std::runtime_error("G6 strength row size mismatch");
  if(col.size()!=val.size()) throw std::runtime_error("G6 strength col/val size mismatch");
  if((std::uint64_t)row.back()!=val.size()) throw std::runtime_error("G6 strength row end mismatch");

  std::vector<double> maxAbs((std::size_t)n,0.0),maxNeg((std::size_t)n,0.0);
  std::uint64_t off=0,pos=0,neg=0,zero=0;
  int rowsNoOffdiag=0,rowsNoNegative=0;
  int matrixMinDeg=std::numeric_limits<int>::max(),matrixMaxDeg=0;

  for(int i=0;i<n;++i){
    int deg=0;
    for(std::int32_t k=row[(std::size_t)i];k<row[(std::size_t)i+1];++k){
      const int j=col[(std::size_t)k];
      if(j==i) continue;
      ++deg;++off;
      const double a=(double)val[(std::size_t)k];
      if(!std::isfinite(a)) throw std::runtime_error("G6 strength nonfinite matrix value");
      maxAbs[(std::size_t)i]=std::max(maxAbs[(std::size_t)i],std::abs(a));
      if(a<0.0){++neg;maxNeg[(std::size_t)i]=std::max(maxNeg[(std::size_t)i],-a);}
      else if(a>0.0)++pos;
      else ++zero;
    }
    matrixMinDeg=std::min(matrixMinDeg,deg);
    matrixMaxDeg=std::max(matrixMaxDeg,deg);
    if(deg==0)++rowsNoOffdiag;
    if(maxNeg[(std::size_t)i]==0.0)++rowsNoNegative;
  }

  const double denom=off?double(off):1.0;
  std::printf(
    "NODALS_GPU_G6_STRENGTH_MATRIX tag=%s source=ACTUAL_REFRESHED_FP32_FINE_CSR "
    "rows=%d nnz=%zu offdiag=%llu avgDegree=%.6f minDegree=%d maxDegree=%d "
    "positiveOffdiag=%llu negativeOffdiag=%llu zeroOffdiag=%llu "
    "positiveFrac=%.9f negativeFrac=%.9f zeroFrac=%.9f "
    "rowsNoOffdiag=%d rowsNoNegative=%d setupOnly=1 solverHierarchy=UNCHANGED status=PASS\n",
    tag,n,val.size(),(unsigned long long)off,double(off)/double(n),
    matrixMinDeg==std::numeric_limits<int>::max()?0:matrixMinDeg,matrixMaxDeg,
    (unsigned long long)pos,(unsigned long long)neg,(unsigned long long)zero,
    double(pos)/denom,double(neg)/denom,double(zero)/denom,
    rowsNoOffdiag,rowsNoNegative);

  const double thetaList[3]={0.10,0.25,0.50};
  for(double theta:thetaList){
    for(int mode=0;mode<2;++mode){
      const bool absMode=(mode==0);
      std::uint64_t edges=0,mutual=0,strongPos=0,strongNeg=0;
      int minDeg=std::numeric_limits<int>::max(),maxDeg=0,isolated=0,mutualIsolated=0;

      for(int i=0;i<n;++i){
        int d=0,md=0;
        const double ti=theta*(absMode?maxAbs[(std::size_t)i]:maxNeg[(std::size_t)i]);
        for(std::int32_t k=row[(std::size_t)i];k<row[(std::size_t)i+1];++k){
          const int j=col[(std::size_t)k];
          if(j==i) continue;
          const double aij=(double)val[(std::size_t)k];
          const double sij=absMode?std::abs(aij):(aij<0.0?-aij:0.0);
          if(!(ti>0.0 && sij>=ti)) continue;

          ++d;++edges;
          if(aij>0.0)++strongPos;
          else if(aij<0.0)++strongNeg;

          const double aji=(double)g6_row_value(j,i,row,col,val);
          const double tj=theta*(absMode?maxAbs[(std::size_t)j]:maxNeg[(std::size_t)j]);
          const double sji=absMode?std::abs(aji):(aji<0.0?-aji:0.0);
          if(tj>0.0 && sji>=tj){++md;++mutual;}
        }

        minDeg=std::min(minDeg,d);
        maxDeg=std::max(maxDeg,d);
        if(d==0)++isolated;
        if(md==0)++mutualIsolated;
      }

      if(minDeg==std::numeric_limits<int>::max()) minDeg=0;
      const double eDen=edges?double(edges):1.0;
      std::printf(
        "NODALS_GPU_G6_STRENGTH tag=%s theta=%.2f criterion=%s "
        "directedEdges=%llu retainedFrac=%.9f avgDegree=%.6f minDegree=%d maxDegree=%d "
        "isolatedRows=%d isolatedFrac=%.9f "
        "mutualDirectedEdges=%llu mutualAvgDegree=%.6f mutualIsolatedRows=%d "
        "mutualIsolatedFrac=%.9f strongPositive=%llu strongNegative=%llu "
        "strongPositiveFrac=%.9f strongNegativeFrac=%.9f "
        "mutualRule=RECIPROCAL_EDGE_PASSES_OWN_ROW_THRESHOLD "
        "setupOnly=1 solverHierarchy=UNCHANGED status=PASS\n",
        tag,theta,absMode?"ABS_ROW_MAX":"CLASSICAL_NEGATIVE",
        (unsigned long long)edges,double(edges)/denom,double(edges)/double(n),
        minDeg,maxDeg,isolated,double(isolated)/double(n),
        (unsigned long long)mutual,double(mutual)/double(n),mutualIsolated,
        double(mutualIsolated)/double(n),
        (unsigned long long)strongPos,(unsigned long long)strongNeg,
        double(strongPos)/eDen,double(strongNeg)/eDen);
    }
  }

  std::printf(
    "NODALS_GPU_G6_STRENGTH_SWEEP tag=%s thetaList=0.10,0.25,0.50 "
    "criteria=ABS_ROW_MAX,CLASSICAL_NEGATIVE source=ACTUAL_REFRESHED_FP32_FINE_CSR "
    "persistentGraphBuilt=0 solverPathChanged=0 status=PASS\n",tag);
}

} // namespace nodals_gpu
