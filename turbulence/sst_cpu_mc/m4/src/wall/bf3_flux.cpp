#include "bf3_flux.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
namespace nodals::cpu_mc::m3 {
BF3Flux::BF3Flux(const Mesh& mesh):m_(mesh){
 const int nv=(int)m_.points.size();wall_.reserve(m_.boundary[3]);
 for(int f=0;f<(int)m_.faces.size();++f){const auto& face=m_.faces[f];if(face.patch!=3)continue;
   int lf=-1;for(int k=0;k<4;++k)if(m_.tets[face.cell0].face[k]==f){lf=k;break;}
   if(lf<0)throw std::runtime_error("wall face without cell-local opposite");
   const Vec3 sf=outward_area(m_,face.cell0,lf);const double a=std::sqrt(dot(sf,sf));
   if(!(a>0))throw std::runtime_error("degenerate wall face");
   WallFace w{};w.bubble=nv+f;w.n=(1.0/a)*sf;w.area=a;
   for(int k=0;k<3;++k)w.v[k]=face.v[k];wall_.push_back(w);
 }
}
void BF3Flux::Q(std::vector<double>&x)const{
 for(const auto&f:wall_){int b=3*f.bubble;const auto n=f.n;
  const double bn=n.x*x[b]+n.y*x[b+1]+n.z*x[b+2];
  x[b]-=n.x*bn;x[b+1]-=n.y*bn;x[b+2]-=n.z*bn;
 }
}
void BF3Flux::T(std::vector<double>&x)const{
 for(const auto&f:wall_){int b=3*f.bubble;const auto n=f.n;
  double sn=0;for(int v:f.v){const int a=3*v;sn+=n.x*x[a]+n.y*x[a+1]+n.z*x[a+2];}
  const double bn=n.x*x[b]+n.y*x[b+1]+n.z*x[b+2];
  const double shift=-bn-(20.0/27.0)*sn;
  x[b]+=n.x*shift;x[b+1]+=n.y*shift;x[b+2]+=n.z*shift;
 }
}
void BF3Flux::Tt(std::vector<double>&x)const{
 for(const auto&f:wall_){int b=3*f.bubble;const auto n=f.n;
  const double bn=n.x*x[b]+n.y*x[b+1]+n.z*x[b+2];
  x[b]-=n.x*bn;x[b+1]-=n.y*bn;x[b+2]-=n.z*bn;
  const double s=-(20.0/27.0)*bn;
  for(int v:f.v){const int a=3*v;x[a]+=s*n.x;x[a+1]+=s*n.y;x[a+2]+=s*n.z;}
 }
}
double BF3Flux::max_mean_flux(const std::vector<double>&x)const{
 double maxflux=0;
 for(const auto&f:wall_){const int b=3*f.bubble;const auto n=f.n;
   double avg=(9.0/20.0)*(n.x*x[b]+n.y*x[b+1]+n.z*x[b+2]);
   for(int v:f.v){const int a=3*v;avg+=(n.x*x[a]+n.y*x[a+1]+n.z*x[a+2])/3.0;}
   maxflux=std::max(maxflux,std::abs(avg)*f.area);
 }
 return maxflux;
}
}
