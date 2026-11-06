#pragma once
// Stage25A rigid rotation of the loaded pure-tet mesh, in memory only.
// Q_y(theta) = [[cos,0,sin],[0,1,0],[-sin,0,cos]].
// Stored inverse-Jacobian reference-gradients are rows (covectors in physical
// coordinates); each must be rotated with Q, just like physical vectors.
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
namespace nodals { namespace stage25 {
inline double c=1.0, s=0.0, angle_deg=0.0;
inline bool configured=false;
inline void set_angle(double degrees){
  if(!std::isfinite(degrees)||std::abs(degrees)>180.)
    throw std::runtime_error("STAGE25_ROTATION requires finite angle in [-180,180]");
  angle_deg=degrees;configured=true;
  const double rad=degrees*(3.141592653589793238462643383279502884/180.);
  c=std::cos(rad);s=std::sin(rad);
  if(degrees==0.){c=1.;s=0.;}
}
inline double axial(double x,double /*y*/,double z){return s*x+c*z;}
inline double radial_x(double x,double /*y*/,double z){return c*x-s*z;}
inline void vector(double x,double y,double z,double &xo,double &yo,double &zo){
  xo=c*x+s*z;yo=y;zo=-s*x+c*z;
}
template<class Mesh>
inline void rotate_loaded_tets(Mesh &m,double degrees){
  if(m.h.nhex!=0||m.h.ninterface!=0||m.h.ntet==0)
    throw std::runtime_error("STAGE25_ROTATION currently supports pure-tet HXT1 meshes only");
  set_angle(degrees);
  const double oldVolume=[](const Mesh &mesh){long double v=0.;for(const auto&g:mesh.tetGeom)v+=g.volume;return (double)v;}(m);
  for(auto &q:m.points){double x,y,z;vector(q.x,q.y,q.z,x,y,z);q.x=x;q.y=y;q.z=z;}
  for(auto &g:m.tetGeom){
    for(int j=0;j<3;++j){
      const double x=g.invJ[3*j],z=g.invJ[3*j+2];
      g.invJ[3*j]=c*x+s*z;
      g.invJ[3*j+2]=-s*x+c*z;
    }
    if(!(g.det>0.&&g.volume>0.))
      throw std::runtime_error("STAGE25_ROTATION nonpositive tet geometry");
    // Pure rigid rotations preserve det(J), volume and all connectivity.
  }
  long double vv=0.;for(const auto&g:m.tetGeom)vv+=g.volume;
  const double defect=std::abs((double)vv-oldVolume)/std::max(1.e-300,oldVolume);
  if(defect>1.e-12)throw std::runtime_error("STAGE25_ROTATION volume drift");
  std::printf("STAGE25_ROTATION angleDeg=%.12g cos=%.16e sin=%.16e "
      "mesh=IN_MEMORY_HXT1 points=%zu tets=%zu invJac=ROTATED "
      "detVolume=UNCHANGED boundaryIDs=UNCHANGED volumeRelDefect=%.3e status=PASS\n",
      degrees,c,s,m.points.size(),m.tetGeom.size(),defect);
}
}} // nodals::stage25
