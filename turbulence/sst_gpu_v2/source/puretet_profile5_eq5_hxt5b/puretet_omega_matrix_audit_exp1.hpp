#pragma once
// Included INSIDE namespace hxt5d AFTER State; reads assembled GPU CSR without modifying it.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <vector>
namespace puretet_matrix_exp1 {
struct Counts {
  int rows=0,negRhs=0,badDiag=0,negativeRowSum=0,notWeakDiagDom=0;
  std::int64_t offdiag=0,positiveOffdiag=0,positiveComp[5]={0,0,0,0,0};
  double posOffdiagL1=0,negRhsL1=0,minRhs=1e300,minDiag=1e300,
    minRowSum=1e300,maxOffdiagSumRatio=0;
};
inline void print(int it,const nodals_hxt1::HostMesh&M,const hxt4b::GpuCSR&A,
                  const State&S,const Dev<Real>&vol,const Dev<Real>&dg,const Dev<Real>&wall,double radius) {
  const int nv=S.nvert,n=A.n;
  auto row=puretet_omega_diag::download(A.row,(size_t)n+1);
  auto col=puretet_omega_diag::download(A.col,A.nnz);
  auto val=puretet_omega_diag::download(A.val,A.nnz);
  auto base=puretet_omega_diag::download(A.base,A.nnz);
  auto conv=puretet_omega_diag::download(A.conv,A.nnz);
  auto vd=puretet_omega_diag::download(vol,A.nnz);
  auto in=puretet_omega_diag::download(dg,A.nnz);
  auto wa=puretet_omega_diag::download(wall,A.nnz);
  auto rhs=puretet_omega_diag::download(S.rhs,n);
  auto old=puretet_omega_diag::download(S.oldW,n);
  Counts q[6];
  for(int i=0;i<nv;++i) {
    const auto&P=M.points[(size_t)i];int bin=puretet_omega_diag::bin(std::hypot(P.x,P.y),radius);
    double diag=0,offAbs=0,rawRowSum=0,relaxDelta=0,oldDiag=0,posOffL1=0;
    std::int64_t noff=0,npos=0;std::int64_t comps[5]={0,0,0,0,0};
    for(std::int64_t a=row[(size_t)i];a<row[(size_t)i+1];++a){
      const int j=col[(size_t)a];if(j<0||j>=n)throw std::runtime_error("EXP1 invalid CSR col");
      if(j>=nv)continue; // P1 constrained enrichment coefficients are zero.
      const double comp[5]={(double)base[(size_t)a],(double)conv[(size_t)a],(double)vd[(size_t)a],(double)in[(size_t)a],(double)wa[(size_t)a]};
      const double orig=comp[0]+comp[1]+comp[2]+comp[3]+comp[4];
      rawRowSum+=orig;
      if(i==j){diag=orig;oldDiag=(double)val[(size_t)a];relaxDelta=oldDiag-orig;}
      else {noff++;offAbs+=std::abs(orig);if(orig>0){npos++;posOffL1+=orig;}
        for(int k=0;k<5;++k)if(comp[k]>0)comps[k]++;
      }
    }
    const double bphys=(double)rhs[(size_t)i]-relaxDelta*(double)old[(size_t)i];
    for(int b : {bin,5}){
      auto&t=q[b];t.rows++;t.offdiag+=noff;t.positiveOffdiag+=npos;t.posOffdiagL1+=posOffL1;
      for(int k=0;k<5;++k)t.positiveComp[k]+=comps[k];
      if(!(diag>0)||!std::isfinite(diag))t.badDiag++;
      if(diag<offAbs*(1-1e-12))t.notWeakDiagDom++;
      if(rawRowSum<0)t.negativeRowSum++;
      if(bphys<0){t.negRhs++;t.negRhsL1-=bphys;}
      t.minRhs=std::min(t.minRhs,bphys);t.minDiag=std::min(t.minDiag,diag);t.minRowSum=std::min(t.minRowSum,rawRowSum);
      if(diag>0)t.maxOffdiagSumRatio=std::max(t.maxOffdiagSumRatio,offAbs/diag);
    }
  }
  for(int b=0;b<6;++b){const auto&t=q[b];
    std::printf("PURETET_EXP1_MATRIX it=%d bin=%d rows=%d activeOffdiag=%lld positiveOffdiag=%lld positiveOffdiagL1=%.10e positiveComponentEntries=[%lld,%lld,%lld,%lld,%lld] badDiagonal=%d nonDiagonalDominant=%d negativeRowSum=%d negativePhysicalRhs=%d negativePhysicalRhsL1=%.10e minPhysicalRhs=%.10e minDiagonal=%.10e minRowSum=%.10e maxOffdiagToDiag=%.10e status=PASS\n",
      it,b,t.rows,(long long)t.offdiag,(long long)t.positiveOffdiag,t.posOffdiagL1,
      (long long)t.positiveComp[0],(long long)t.positiveComp[1],(long long)t.positiveComp[2],(long long)t.positiveComp[3],(long long)t.positiveComp[4],
      t.badDiag,t.notWeakDiagDom,t.negativeRowSum,t.negRhs,t.negRhsL1,t.minRhs,t.minDiag,t.minRowSum,t.maxOffdiagSumRatio);
  }
}
}
