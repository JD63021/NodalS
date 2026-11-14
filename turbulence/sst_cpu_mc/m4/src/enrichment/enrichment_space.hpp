#pragma once
// Experimental shared shifted Spalding P1 enrichment for CPU SST M4.
// Extra coefficients are globally shared vertex modes, not tet-local bubbles.
// Their no-slip trace is zero (and their elementwise divergence is nonzero).
#include "shifted_spalding_p1.hpp"
#include "../mesh/tet_mesh.hpp"
#include "../case/case_config.hpp"
#include <vector>
#include <array>
#include <cmath>
#include <stdexcept>
#include <algorithm>
namespace nodals::cpu_mc::m4 {
using namespace nodals::cpu_mc::m2;
struct EnrichmentSpace {
  const Mesh& mesh;
  std::vector<double> distance;
  std::vector<double> nodalW; // evaluate Spalding inverse at vertices once
  std::vector<int> index;
  std::vector<unsigned char> fixed;
  int ne=0, nr=0;
  bool enabled=false;
  nodals::cpu_mc::enrichment::WallLaw law;
  explicit EnrichmentSpace(const Mesh&m,const std::vector<double>&y,const m3::CaseConfig&cfg)
   :mesh(m),distance(y),ne((int)m.points.size()+(int)m.faces.size()),enabled(cfg.enrichEnabled) {
    index.assign(m.points.size(),-1);
    fixed.assign((std::size_t)ne,0);
    if(!enabled)return;
    if(!(cfg.enrichUTau>0 && cfg.enrichCutoff>0 && cfg.nu>0))
      throw std::runtime_error("enrichment: invalid utau/cutoff/nu");
    law.nu=cfg.nu;law.utau=cfg.enrichUTau;
    nodalW.resize(m.points.size());
    for(size_t i=0;i<m.points.size();++i)
      nodalW[i]=law.value_gradient(std::max(0.,y[i]))[0];
    std::vector<unsigned char> wallVertices(m.points.size(),0);
    for(size_t f=0;f<m.faces.size();++f)if(m.faces[f].patch==3){
      fixed[m.points.size()+f]=1;
      for(int v:m.faces[f].v)wallVertices[v]=1;
    }
    for(size_t i=0;i<m.points.size();++i){
      if(wallVertices[i])fixed[i]=1;
      if(wallVertices[i] || y[i]*law.utau/law.nu<=cfg.enrichCutoff)index[i]=nr++;
    }
    fixed.resize((std::size_t)ne+nr,0);
    if(nr==0)throw std::runtime_error("enrichment selected zero vertices");
  }
  int nent()const{return ne+nr;}
  int entity(int c,int a)const{
    const auto&t=mesh.tets[(std::size_t)c];
    if(a<4)return t.v[a];
    if(a<8)return (int)mesh.points.size()+t.face[a-4];
    const int idx=index[(std::size_t)t.v[a-8]];
    return idx<0?-1:ne+idx;
  }
  bool is_fixed(int e)const{
    return enabled && e>=0 && e<(int)fixed.size() && fixed[(size_t)e]!=0;
  }
  // Local values and physical gradients for 8 original + 4 enriched shapes.
  // An inactive enriched mode is exactly zero and has entity=-1.
  void shape(int cell,const std::array<double,4>&l,
             double phi[12],Vec3 grad[12])const {
    const auto&t=mesh.tets[(size_t)cell];
    const Vec3 p0=mesh.points[(size_t)t.v[0]];
    const Vec3 a=mesh.points[(size_t)t.v[1]]-p0,
               b=mesh.points[(size_t)t.v[2]]-p0,
               c=mesh.points[(size_t)t.v[3]]-p0;
    const double det=dot(a,cross(b,c));
    if(!(det>0))throw std::runtime_error("enrichment invalid oriented tet");
    const Vec3 g1=(1.0/det)*cross(b,c),g2=(1.0/det)*cross(c,a),g3=(1.0/det)*cross(a,b);
    Vec3 gl[4]={-1.0*(g1+g2+g3),g1,g2,g3};
    for(int j=0;j<4;++j){phi[j]=l[j];grad[j]=gl[j];}
    for(int j=0;j<4;++j){
      int z[3],k=0;for(int q=0;q<4;++q)if(q!=j)z[k++]=q;
      phi[4+j]=27*l[z[0]]*l[z[1]]*l[z[2]];
      grad[4+j]={};
      for(int k2=0;k2<3;++k2)
        grad[4+j]=grad[4+j]+
          (27.0*l[z[(k2+1)%3]]*l[z[(k2+2)%3]])*gl[z[k2]];
    }
    for(int j=8;j<12;++j){phi[j]=0;grad[j]={};}
    if(!enabled)return;
    bool present=false;
    for(int j=0;j<4;++j)if(index[(size_t)t.v[j]]>=0)present=true;
    if(!present)return;
    double y=0,linearW=0;Vec3 gy{},gLinearW{};
    for(int j=0;j<4;++j){
      const double yd=distance[(size_t)t.v[j]],Wj=nodalW[(size_t)t.v[j]];
      y+=l[j]*yd;linearW+=l[j]*Wj;
      gy=gy+yd*gl[j];gLinearW=gLinearW+Wj*gl[j];
    }
    const auto W=law.value_gradient(std::max(0.,y));
    const double shift=W[0]-linearW;
    const Vec3 gradShift=W[1]*gy-gLinearW;
    for(int j=0;j<4;++j)if(index[(size_t)t.v[j]]>=0){
      phi[8+j]=l[j]*shift;
      grad[8+j]=shift*gl[j]+l[j]*gradShift;
    }
  }
  // Gauss-Legendre rule on [0,1]. Non-polynomial wall-law shapes need
  // increased face/volume quadrature, never a polynomial-only low-order rule.
  static void gauss(int n,std::vector<double>&x,std::vector<double>&w){
    x.resize(n);w.resize(n);
    const double pi=3.1415926535897932384626433832795;
    for(int i=0;i<(n+1)/2;++i){
      double z=std::cos(pi*(i+0.75)/(n+0.5)),dp=0;
      for(int it=0;it<60;++it){
        double p0=1,p1=z;
        for(int k=2;k<=n;++k){double t=((2*k-1)*z*p1-(k-1)*p0)/k;p0=p1;p1=t;}
        dp=n*(z*p1-p0)/(z*z-1);
        double dz=p1/dp;z-=dz;
        if(std::abs(dz)<1e-15)break;
      }
      const double weight=1./((1.-z*z)*dp*dp);
      x[i]=(1-z)*.5;x[n-1-i]=(1+z)*.5;
      w[i]=weight;w[n-1-i]=weight;
    }
  }
  // Exact weak pressure coupling for each enriched mode is the elementwise
  // flux integral ∫_∂T E_i n dS (not an assumed zero "bubble" coupling).
  // Each triangle uses a high-order Duffy rule. Coordinates on a shared face
  // are consistent because the face's sorted vertex IDs are used by both tets.
  void face_flux(int c,int lf,std::array<Vec3,4>&B,int order=12)const {
    B={};
    if(!enabled)return;
    const auto&t=mesh.tets[(size_t)c];
    const Vec3 ar=outward_area(mesh,c,lf); // vector with magnitude Area
    std::vector<double>x,w;gauss(order,x,w);
    int js[3],k=0;
    for(int j=0;j<4;++j)if(j!=lf)js[k++]=j;
    for(int i=0;i<order;++i)for(int j=0;j<order;++j){
      const double a=x[i],b=x[j];std::array<double,4> l{};
      l[js[0]]=a;l[js[1]]=(1-a)*b;l[js[2]]=(1-a)*(1-b);
      double ph[12];Vec3 gp[12];shape(c,l,ph,gp);
      const double wt=2.*(1-a)*w[i]*w[j];
      for(int m=0;m<4;++m)if(entity(c,8+m)>=0)
        B[m]=B[m]+(wt*ph[8+m])*ar;
    }
  }
  void divergence(int c,std::array<Vec3,4>&B)const{
    B={};if(!enabled)return;
    for(int f=0;f<4;++f){std::array<Vec3,4> b{};face_flux(c,f,b);
      for(int m=0;m<4;++m)B[m]=B[m]+b[m];
    }
  }
  Vec3 velocity(int cell,const std::array<double,4>&l,const std::vector<double>&u)const {
    double ph[12];Vec3 gp[12];shape(cell,l,ph,gp);
    if((int)u.size()!=3*nent())throw std::runtime_error("enrichment velocity length mismatch");
    Vec3 out{};
    for(int a=0;a<12;++a){const int id=entity(cell,a);if(id<0)continue;
      out.x+=ph[a]*u[3*(size_t)id];out.y+=ph[a]*u[3*(size_t)id+1];out.z+=ph[a]*u[3*(size_t)id+2];
    }
    return out;
  }
  void add_gradient(int cell,const std::array<double,4>&l,const std::vector<double>&u,double g[9])const{
    if(!enabled)return;
    double ph[12];Vec3 gp[12];shape(cell,l,ph,gp);
    for(int a=8;a<12;++a){int id=entity(cell,a);if(id<0)continue;
      const Vec3 v=gp[a];const double grad[3]={v.x,v.y,v.z};
      for(int i=0;i<3;++i)for(int d=0;d<3;++d)
        g[3*i+d]+=u[3*(size_t)id+i]*grad[d];
    }
  }
};
} // namespace nodals::cpu_mc::m4
