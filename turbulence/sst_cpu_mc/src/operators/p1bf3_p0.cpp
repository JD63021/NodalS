#include "p1bf3_p0.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nodals::cpu_mc::m2 {
Bcell divergence_coefficients(const Mesh&m,int c){
  Bcell r{};
  for(int i=0;i<4;++i){const Vec3 s=outward_area(m,c,i);r.b[i]=(-1.0/3.0)*s;r.b[4+i]=0.45*s;}
  return r;
}
Element24 deviatoric_element(const Mesh&m,int c,double nu){
  if(!(nu>0))throw std::runtime_error("viscosity must be positive");
  const auto&t=m.tets[c];
  const Vec3 a=m.points[t.v[0]],e1=m.points[t.v[1]]-a,e2=m.points[t.v[2]]-a,e3=m.points[t.v[3]]-a;
  const double det=dot(e1,cross(e2,e3));
  if(!(det>0))throw std::runtime_error("negative oriented tet");
  std::array<Vec3,4> gl{};
  gl[1]=(1.0/det)*cross(e2,e3);gl[2]=(1.0/det)*cross(e3,e1);gl[3]=(1.0/det)*cross(e1,e2);
  gl[0]=-1.0*(gl[1]+gl[2]+gl[3]);
  const double x[4]={0.06943184420297371,0.33000947820757187,0.6699905217924281,0.9305681557970262};
  const double w[4]={0.1739274225687269,0.3260725774312731,0.3260725774312731,0.1739274225687269};
  Element24 K{};
  for(int ir=0;ir<4;++ir)for(int is=0;is<4;++is)for(int it=0;it<4;++it){
    const double r=x[ir],s=x[is],z=x[it],omr=1.0-r,oms=1.0-s;
    const double weight=w[ir]*w[is]*w[it]*omr*omr*oms*det; // Duffy: dr ds dt to barycentric
    std::array<double,4> l{omr*oms*(1.0-z),r,omr*s,omr*oms*z};
    std::array<Vec3,8> g{};
    for(int i=0;i<4;++i){
      g[i]=gl[i];
      for(int j=0;j<4;++j)if(j!=i){double p=27.0;
        for(int k=0;k<4;++k)if(k!=i&&k!=j)p*=l[k];
        g[4+i]=g[4+i]+p*gl[j];
      }
    }
    for(int aa=0;aa<8;++aa)for(int bb=0;bb<8;++bb){
      const Vec3 u=g[aa],v=g[bb];
      const double ga[3]={u.x,u.y,u.z},gb[3]={v.x,v.y,v.z};
      const double uv=dot(u,v);
      for(int i=0;i<3;++i)for(int j=0;j<3;++j){
        K[3*aa+i][3*bb+j]+=weight*nu*((i==j?uv:0.0)+ga[j]*gb[i]-(2.0/3.0)*ga[i]*gb[j]);
      }
    }
  }
  return K;
}
}
