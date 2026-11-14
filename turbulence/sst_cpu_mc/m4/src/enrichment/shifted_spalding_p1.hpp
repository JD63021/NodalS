#pragma once
// Staged Spalding inverse, analytically differentiated, as verified in Stage 1.
#include <array>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>
namespace nodals::cpu_mc::enrichment {
using V3=std::array<double,3>;
inline V3 add(V3 a,V3 b){return {a[0]+b[0],a[1]+b[1],a[2]+b[2]};}
inline V3 sub(V3 a,V3 b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
inline V3 scale(double x,V3 a){return {x*a[0],x*a[1],x*a[2]};}
inline double dot(V3 a,V3 b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
struct WallLaw {
 double nu=0,utau=0,kappa=.41,B=5.17;
 std::array<double,2> F(double u)const{
  const double x=kappa*u;if(x==0)return {0,1};
  double rem=0,der=0;
  if(x<.35){double term=1.;for(int k=1;k<=4;++k)term*=x/k;
    for(int k=5;k<=40;++k){term*=x/k;rem+=term;der+=term*k/x;
      if(term<1e-16*std::max(1.,rem))break;}}
  else {const double e=std::exp(x);rem=e-(1+x+x*x/2+x*x*x/6+x*x*x*x/24);
        der=e-(1+x+x*x/2+x*x*x/6);}
  const double c=std::exp(-kappa*B);
  return {u+c*rem,1+c*kappa*der};
 }
 std::array<double,2> value_gradient(double y)const{
  if(!(nu>0&&utau>0&&y>=0&&std::isfinite(y)))throw std::runtime_error("invalid enrichment wall parameters");
  const double yp=utau*y/nu;
  if(yp==0)return {0.,utau*utau/nu};
  double lo=0,hi=1.;
  while(F(hi)[0]<yp)hi*=2;
  double u=std::min(yp,hi*.5);
  for(int n=0;n<80;++n){
    const auto a=F(u);const double err=a[0]-yp;
    if(std::abs(err)<1e-12*std::max(1.,yp))return {utau*u,utau*utau/(nu*a[1])};
    if(err>0)hi=u;else lo=u;
    double un=u-err/a[1];
    if(!(un>lo&&un<hi)||!std::isfinite(un))un=.5*(lo+hi);
    u=un;
  }
  const auto a=F(u);
  if(std::abs(a[0]-yp)>1e-9*std::max(1.,yp))throw std::runtime_error("Spalding inverse failed");
  return {utau*u,utau*utau/(nu*a[1])};
 }
};
struct Eval {
 std::array<double,4>E{};
 std::array<V3,4>dE{};
 double W=0,shift=0;V3 gradShift{};
};
inline Eval evaluate(const std::array<double,4>&l,
  const std::array<V3,4>&gl,const std::array<double,4>&yv,
  const std::array<bool,4>&active,const WallLaw&law){
 double y=0,Wi=0;V3 gy{},gWi{};
 for(int a=0;a<4;++a){
   if(yv[a]<-1e-15)throw std::runtime_error("negative wall distance");
   y+=l[a]*yv[a];gy=add(gy,scale(yv[a],gl[a]));
   const double w=law.value_gradient(std::max(0.,yv[a]))[0];
   Wi+=l[a]*w;gWi=add(gWi,scale(w,gl[a]));
 }
 if(y<0&&y>-1e-12)y=0;
 const auto W=law.value_gradient(y);
 Eval ret;ret.W=W[0];ret.shift=W[0]-Wi;ret.gradShift=sub(scale(W[1],gy),gWi);
 for(int a=0;a<4;++a)if(active[a]){
    ret.E[a]=l[a]*ret.shift;
    ret.dE[a]=add(scale(ret.shift,gl[a]),scale(l[a],ret.gradShift));
 }
 return ret;
}
}