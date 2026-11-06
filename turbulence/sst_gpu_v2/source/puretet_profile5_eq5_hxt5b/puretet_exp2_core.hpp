#pragma once
// EXP2 host arithmetic, no CUDA dependency. This header is also unit-tested with C++17.
#include <vector>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <limits>
namespace puretet_exp2_core {
struct Snapshot {
  int n=0,nv=0;
  std::vector<int> row,col;
  std::vector<double> relaxed,base,conv,volume,inlet,wall,reaction,rhs,old,cross;
  void validate() const {
    if(n<=0||nv<=0||nv>n ||row.size()!=size_t(n+1))throw std::runtime_error("EXP2 invalid dimensions");
    const auto nz=col.size();
    for(const auto *v : {&relaxed,&base,&conv,&volume,&inlet,&wall,&reaction})if(v->size()!=nz)throw std::runtime_error("EXP2 invalid CSR vector length");
    for(const auto *v : {&rhs,&old,&cross})if(v->size()!=size_t(n))throw std::runtime_error("EXP2 invalid rhs vector length");
    if(row.front()!=0||row.back()!=int(nz))throw std::runtime_error("EXP2 CSR bounds mismatch");
    for(int i=0;i<n;++i){if(row[i+1]<row[i])throw std::runtime_error("EXP2 CSR unordered");for(int a=row[i];a<row[i+1];++a)if(col[a]<0||col[a]>=n)throw std::runtime_error("EXP2 CSR invalid col");}
  }
  double orig(int a) const {return base[a]+conv[a]+volume[a]+inlet[a]+wall[a];}
  std::vector<double> physical_rhs() const {
    std::vector<double> b=rhs;
    for(int i=0;i<nv;++i){
      double d=0;bool found=false;
      for(int a=row[i];a<row[i+1];++a)if(col[a]==i){d+=relaxed[a]-orig(a);found=true;}
      if(!found)throw std::runtime_error("EXP2 missing CSR diagonal");
      b[i]-=d*old[i];
    }
    return b;
  }
  std::vector<double> rhs_nonnegative_physical() const {
    auto b=rhs;auto p=physical_rhs();
    for(int i=0;i<nv;++i)if(p[i]<0)b[i]-=p[i];
    return b;
  }
  std::vector<double> rhs_without_cross() const {
    auto b=rhs;
    for(int i=0;i<nv;++i)b[i]-=cross[i];
    return b;
  }
  enum Block {REACTION,WALL};
  std::vector<double> lump_offdiag(Block block) const {
    auto v=relaxed;
    const auto&term=(block==REACTION)?reaction:wall;
    for(int i=0;i<nv;++i){
      int d=-1;double shift=0;
      for(int a=row[i];a<row[i+1];++a){int j=col[a];if(j==i)d=a;else if(j<nv){v[a]-=term[a];shift+=term[a];}}
      if(d<0)throw std::runtime_error("EXP2 missing diagonal in matrix lumping");
      v[d]+=shift;
    }
    return v;
  }
  double residual_norm(const std::vector<double>&v,const std::vector<double>&b,const std::vector<double>&x) const {
    if(v.size()!=col.size()||b.size()!=size_t(n)||x.size()!=size_t(n))throw std::runtime_error("EXP2 residual dimension mismatch");
    double s=0;
    for(int i=0;i<nv;++i){double r=b[i];for(int a=row[i];a<row[i+1];++a)r-=v[a]*x[col[a]];s+=r*r;}
    return std::sqrt(s);
  }
};
} // namespace puretet_exp2_core
