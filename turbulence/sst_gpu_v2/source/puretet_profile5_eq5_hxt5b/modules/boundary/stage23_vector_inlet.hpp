#pragma once
// Stage 23A: portable geometry-based inlet representation.
// The velocity vector is in physical/global Cartesian coordinates. Normal is outward.
#include <array>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <sstream>
namespace nodals { namespace stage23 {
struct Vec3 { double x=0, y=0, z=0; };
inline double dot(Vec3 a,Vec3 b) {return a.x*b.x+a.y*b.y+a.z*b.z;}
inline double norm(Vec3 a) {return std::sqrt(dot(a,a));}
inline Vec3 mul(double a,Vec3 v) {return {a*v.x,a*v.y,a*v.z};}
inline Vec3 from_csv(const std::string& text) {
  std::istringstream s(text); Vec3 v;char c1=0,c2=0;
  if(!(s>>v.x>>c1>>v.y>>c2>>v.z) || c1!=','||c2!=',')
    throw std::runtime_error("SST23_INLET_VELOCITY must be x,y,z");
  std::string tail;if(s>>tail)throw std::runtime_error("SST23_INLET_VELOCITY has trailing text");
  if(!std::isfinite(v.x)||!std::isfinite(v.y)||!std::isfinite(v.z)||!(norm(v)>0))
    throw std::runtime_error("SST23_INLET_VELOCITY must be finite and nonzero");
  return v;
}
inline Vec3 from_environment(double old_bulk) {
  const char* e=std::getenv("SST23_INLET_VELOCITY");
  return e ? from_csv(e) : Vec3{0.,0.,old_bulk};
}
inline double signed_flux(Vec3 u,Vec3 outward_normal,double area) {
  return area*dot(u,outward_normal);
}
inline double upwind_inflow(Vec3 u,Vec3 outward_normal) {
  return std::fmax(0.,-dot(u,outward_normal));
}
inline bool old_pipe_exact(Vec3 v,double bulk) {
  const double scale=std::fmax(1.,std::fabs(bulk));
  return std::fabs(v.x)<=1e-14*scale && std::fabs(v.y)<=1e-14*scale &&
    std::fabs(v.z-bulk)<=1e-14*scale;
}
} }
