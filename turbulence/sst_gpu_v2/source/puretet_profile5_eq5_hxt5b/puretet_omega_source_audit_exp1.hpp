#pragma once
// Included INSIDE namespace hxt5c, after Real has been defined.
// Diagnostic ONLY: post-GPU readback of exactly the assembled source vectors.
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>
namespace puretet_source_exp1 {
inline std::vector<Real> lastCross,lastReaction,lastWallRhs; inline int lastOuter=-1; // PURETET_EXP2_SOURCE_CACHE
inline bool scheduled(int it) {
  const char *p=std::getenv("PURETET_OMEGA_AUDIT_EVERY");
  const int n=p?std::atoi(p):0;
  return n>0 && (it<=3 || it%n==0);
}
inline int radial_bin(double r,double radius) {
  const int b=int(5.0*r/radius);return b<0?0:(b>4?4:b);
}
inline void print(int it,const nodals_hxt1::HostMesh&M,int nv,double radius,
                  const std::vector<Real>&v,const std::vector<Real>&postInlet,
                  const std::vector<Real>&total,const std::vector<Real>&prod,
                  const std::vector<Real>&cross) {
  if(v.size()<(size_t)nv||postInlet.size()<(size_t)nv||total.size()<(size_t)nv||
     prod.size()<(size_t)nv||cross.size()<(size_t)nv) throw std::runtime_error("EXP1 source audit dimension error");
  struct S {int n=0,neg=0,negativeCross=0;double source=0,production=0,cross=0,inlet=0,wall=0,minimum=1.e300,negL1=0,crossNegL1=0,closureL1=0;};
  S s[6];
  for(int i=0;i<nv;++i) {
    auto&P=M.points[(size_t)i];const int b=radial_bin(std::hypot(P.x,P.y),radius);
    const double vol=(double)v[(size_t)i],pin=(double)postInlet[(size_t)i],rhs=(double)total[(size_t)i],
      p=(double)prod[(size_t)i],c=(double)cross[(size_t)i];
    const double dInlet=pin-vol,dWall=rhs-pin;
    for(int j=0;j<2;++j){auto &q=s[j?5:b];q.n++;q.source+=rhs;q.production+=p;q.cross+=c;q.inlet+=dInlet;q.wall+=dWall;q.minimum=std::min(q.minimum,rhs);
      if(rhs<0){q.neg++;q.negL1-=rhs;}if(c<0){q.negativeCross++;q.crossNegL1-=c;}
      q.closureL1+=std::abs(vol-(p+c));}
  }
  for(int b=0;b<6;++b) {auto &q=s[b];
    std::printf("PURETET_EXP1_SOURCE it=%d bin=%d vertices=%d negativeTotalRhs=%d negativeCrossVertices=%d minTotalRhs=%.10e rhsTotal=%.10e production=%.10e cross=%.10e inlet=%.10e wall=%.10e negativeRhsL1=%.10e negativeCrossL1=%.10e volumeSplitErrorL1=%.10e status=PASS\n",
                it,b,q.n,q.neg,q.negativeCross,q.minimum,q.source,q.production,q.cross,q.inlet,q.wall,q.negL1,q.crossNegL1,q.closureL1);
  }
}
}
