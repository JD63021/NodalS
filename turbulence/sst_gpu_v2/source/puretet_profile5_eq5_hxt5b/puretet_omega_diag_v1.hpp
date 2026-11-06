#pragma once
// PURETET_OMEGA_DIAG_V1 - host-side audit only. This file is included within
// namespace hxt5d AFTER State and StepResult are defined.
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>
namespace puretet_omega_diag {
inline int interval(){
  const char* p=std::getenv("PURETET_OMEGA_AUDIT_EVERY");
  if(!p||!*p)return 0;
  const int n=std::atoi(p);return n>0?n:0;
}
inline bool scheduled(int it){
  const int n=interval();return n>0&&(it<=3||it%n==0);
}
template<class T> inline std::vector<T> download(const Dev<T>&d,std::size_t count){
  std::vector<T> h(count);
  if(count)HXT1_CUDA(cudaMemcpy(h.data(),d.p,count*sizeof(T),cudaMemcpyDeviceToHost));
  return h;
}
inline int bin(double r,double radius){
  int b=(int)std::floor(5.0*r/radius);
  return b<0?0:(b>4?4:b);
}
inline void report(int it,const nodals_hxt1::HostMesh&M,
                   const hxt4b::GpuCSR&A,const State&S,
                   const Dev<Real>&vol,const Dev<Real>&dg,const Dev<Real>&wall,
                   const hxt5c::Options&O,double radius,
                   const std::vector<Real>&raw){
  // q1 and pure-TET are checked at the call site. Rows 0:nvert are active.
  const int nv=S.nvert,n=A.n;
  if((int)raw.size()!=nv||nv<=0)return;
  const auto after=download(S.omega,n);
  const auto old=download(S.oldW,n);
  const auto rhs=download(S.rhs,n);
  const auto row=download(A.row,(std::size_t)n+1);
  const auto col=download(A.col,A.nnz);
  const auto relaxed=download(A.val,A.nnz);
  const auto base=download(A.base,A.nnz);
  const auto convection=download(A.conv,A.nnz);
  const auto volume=download(vol,A.nnz);
  const auto inlet=download(dg,A.nnz);
  const auto walls=download(wall,A.nnz);
  std::vector<Real> before=after;
  for(int i=0;i<nv;++i)before[(std::size_t)i]=raw[(std::size_t)i];
  double preRelax2=0,postRelax2=0,prePhysical2=0,postPhysical2=0;
  double preRelaxInf=0,postRelaxInf=0,prePhysicalInf=0,postPhysicalInf=0;
  double minRaw=std::numeric_limits<double>::infinity(),minAfter=std::numeric_limits<double>::infinity();
  int below=0,neg=0,nan=0,atFloor=0;double clipL1=0,clipL2=0;
  struct Region{
    int cnt=0,below=0,negative=0,atFloor=0;
    double sumPreRelax=0,sumPostRelax=0,sumPrePhysical=0,sumPostPhysical=0;
    double convection=0,base=0,volume=0,inlet=0,wall=0,rhsPhysical=0;
    double deltaL1=0;
  } reg[5];
  for(int i=0;i<nv;++i){
    const double x=(double)raw[(std::size_t)i],y=(double)after[(std::size_t)i];
    const auto&P=M.points[(std::size_t)i];
    const int b=bin(std::hypot(P.x,P.y),radius);auto&r=reg[b];++r.cnt;
    if(!std::isfinite(x))++nan;
    if(std::isfinite(x))minRaw=std::min(minRaw,x);
    if(std::isfinite(y))minAfter=std::min(minAfter,y);
    if(!(x>O.omegaFloor)){++below;++r.below;}
    if(x<0.0){++neg;++r.negative;}
    if(y<=(double)O.omegaFloor*(1.0+1e-12)){++atFloor;++r.atFloor;}
    double d=std::isfinite(x)?y-x:0;
    clipL1+=std::abs(d);clipL2+=d*d;
    double preSumRelax=0,postSumRelax=0,preSumPhysical=0,postSumPhysical=0;
    double ar0=0,ac=0,av=0,ai=0,aw=0,deltaDiagonal=0;
    for(std::int64_t j=row[(std::size_t)i];j<row[(std::size_t)i+1];++j){
      const auto q=(std::size_t)j;const int v=col[q];
      const double original=(double)base[q]+(double)convection[q]+(double)volume[q]+(double)inlet[q]+(double)walls[q];
      const double a=(double)relaxed[q];
      if(i==v)deltaDiagonal+=a-original;
      preSumRelax+=a*(double)before[(std::size_t)v];
      postSumRelax+=a*(double)after[(std::size_t)v];
      preSumPhysical+=original*(double)before[(std::size_t)v];
      postSumPhysical+=original*(double)after[(std::size_t)v];
      ar0+=(double)base[q]*(double)after[(std::size_t)v];
      ac+=(double)convection[q]*(double)after[(std::size_t)v];
      av+=(double)volume[q]*(double)after[(std::size_t)v];
      ai+=(double)inlet[q]*(double)after[(std::size_t)v];
      aw+=(double)walls[q]*(double)after[(std::size_t)v];
    }
    const double bRel=(double)rhs[(std::size_t)i];
    const double bPhys=bRel-deltaDiagonal*(double)old[(std::size_t)i];
    const double preR=bRel-preSumRelax,postR=bRel-postSumRelax;
    const double preP=bPhys-preSumPhysical,postP=bPhys-postSumPhysical;
    preRelax2+=preR*preR;postRelax2+=postR*postR;
    prePhysical2+=preP*preP;postPhysical2+=postP*postP;
    preRelaxInf=std::max(preRelaxInf,std::abs(preR));postRelaxInf=std::max(postRelaxInf,std::abs(postR));
    prePhysicalInf=std::max(prePhysicalInf,std::abs(preP));postPhysicalInf=std::max(postPhysicalInf,std::abs(postP));
    r.sumPreRelax+=preR;r.sumPostRelax+=postR;r.sumPrePhysical+=preP;r.sumPostPhysical+=postP;
    r.base+=ar0;r.convection+=ac;r.volume+=av;r.inlet+=ai;r.wall+=aw;r.rhsPhysical+=bPhys;
    r.deltaL1+=std::abs(d);
  }
  std::printf("PURETET_OMEGA_CLIP_AUDIT it=%d nVertex=%d floor=%.9e rawMin=%.9e postMin=%.9e rawBelow=%d rawNegative=%d rawNonfinite=%d postAtFloor=%d clipL1=%.9e clipL2=%.9e preRelaxedResidual=%.9e postRelaxedResidual=%.9e residualAmplification=%.9e prePhysicalResidual=%.9e postPhysicalResidual=%.9e physicalAmplification=%.9e preRelaxedMax=%.9e postRelaxedMax=%.9e prePhysicalMax=%.9e postPhysicalMax=%.9e status=PASS\n",
    it,nv,O.omegaFloor,minRaw,minAfter,below,neg,nan,atFloor,clipL1,std::sqrt(clipL2),std::sqrt(preRelax2),std::sqrt(postRelax2),std::sqrt(postRelax2)/std::max(std::sqrt(preRelax2),1e-300),std::sqrt(prePhysical2),std::sqrt(postPhysical2),std::sqrt(postPhysical2)/std::max(std::sqrt(prePhysical2),1e-300),preRelaxInf,postRelaxInf,prePhysicalInf,postPhysicalInf);
  for(int b=0;b<5;++b){const auto&r=reg[b];
    std::printf("PURETET_OMEGA_CLIP_RADIAL it=%d bin=%d rMin=%.1f rMax=%.1f vertices=%d rawBelow=%d rawNegative=%d postAtFloor=%d clipL1=%.9e preRelaxedSum=%.9e postRelaxedSum=%.9e prePhysicalSum=%.9e postPhysicalSum=%.9e status=PASS\n",it,b,b*.2,(b+1)*.2,r.cnt,r.below,r.negative,r.atFloor,r.deltaL1,r.sumPreRelax,r.sumPostRelax,r.sumPrePhysical,r.sumPostPhysical);
    std::printf("PURETET_OMEGA_WEAK_RADIAL it=%d bin=%d base=%.9e convection=%.9e volumeDiffusionReaction=%.9e inlet=%.9e wall=%.9e physicalRhs=%.9e physicalResidual=%.9e status=PASS\n",it,b,r.base,r.convection,r.volume,r.inlet,r.wall,r.rhsPhysical,r.sumPostPhysical);
  }
}
} // namespace puretet_omega_diag
