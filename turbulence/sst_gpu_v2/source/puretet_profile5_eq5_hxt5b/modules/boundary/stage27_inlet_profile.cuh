#pragma once
// STAGE27_DG_PROFILE_V1 -- 2D P1 inlet velocity, no turbulence-BC change.
// The host imports only the 3 velocity columns from the profile CSV.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <set>

namespace stage27_inlet {
using Real=hxt4b::Real;
__device__ __constant__ int profileActive=0;
__device__ __constant__ const Real* velocityX=nullptr;
__device__ __constant__ const Real* velocityY=nullptr;
__device__ __constant__ const Real* velocityZ=nullptr;

struct HostProfile {
    std::vector<Real> ux,uy,uz;
    double inwardFlux=0.,minInward=1.e100,maxSpeed=0.;
    std::size_t nVertices=0;
};

inline HostProfile load(const nodals_hxt1::HostMesh&M,
                        const hxt5b::BoundaryHost&BH,const std::string&csv) {
  if(M.h.nhex!=0 || BH.inT.empty())
    throw std::runtime_error("STAGE27_DG_PROFILE supports PURE TET inlet only");
  std::ifstream f(csv);
  if(!f)throw std::runtime_error("Cannot open STAGE27_INLET_PROFILE_CSV="+csv);
  std::string header;
  if(!std::getline(f,header))throw std::runtime_error("Empty inlet profile CSV");
  // Positional schema is deliberately strict to avoid silent wrong-column imports.
  const std::string prefix="x_m,y_m,z_m,Ux_m_s,Uy_m_s,Uz_m_s,k_m2_s2,omega_1_s";
  if(header!=prefix && header!=prefix+"\r")
    throw std::runtime_error("Unexpected inlet CSV header. Need x_m,y_m,z_m,Ux_m_s,Uy_m_s,Uz_m_s,k_m2_s2,omega_1_s");
  struct Row {double y,z,u,v,w;};
  std::vector<Row> table;
  std::string line;
  while(std::getline(f,line)){
    if(line.empty()||line=="\r")continue;
    std::istringstream s(line);
    std::string part;
    std::vector<double> a;
    while(std::getline(s,part,',')){
      std::size_t p=0;double d=std::stod(part,&p);
      if(p!=part.size() && !(p+1==part.size()&&part[p]=='\r'))
        throw std::runtime_error("Non-numeric CSV token: "+part);
      if(!std::isfinite(d))throw std::runtime_error("Nonfinite inlet CSV value");
      a.push_back(d);
    }
    if(a.size()!=8)throw std::runtime_error("Expected eight columns per inlet CSV record");
    table.push_back({a[1],a[2],a[3],a[4],a[5]});
  }
  std::set<int> vertices;
  for(const auto&f:BH.inT){
    const auto&tet=M.tets[(std::size_t)f.cell];
    for(int j=0;j<3;++j)
      vertices.insert((int)tet.v[hxt2::TET_FACE[f.lf][j]]);
  }
  if(table.size()!=vertices.size())
    throw std::runtime_error("CSV row count does not match inlet boundary vertex count");
  HostProfile P;
  P.ux.assign((std::size_t)M.h.nv,Real(0));
  P.uy.assign((std::size_t)M.h.nv,Real(0));
  P.uz.assign((std::size_t)M.h.nv,Real(0));
  std::vector<unsigned char> assigned(table.size(),0);
  for(int vid:vertices){
    const auto&v=M.points[(std::size_t)vid];
    std::size_t best=table.size();double bd2=1.e100;
    for(std::size_t j=0;j<table.size();++j){
      double dy=v.y-table[j].y,dz=v.z-table[j].z,d2=dy*dy+dz*dz;
      if(d2<bd2){bd2=d2;best=j;}
    }
    if(best==table.size()||bd2>1.e-18 || assigned[best])
      throw std::runtime_error("Inlet P1 node not matched uniquely to CSV y,z within 1e-9 m");
    assigned[best]=1;
    P.ux[(std::size_t)vid]=(Real)table[best].u;
    P.uy[(std::size_t)vid]=(Real)table[best].v;
    P.uz[(std::size_t)vid]=(Real)table[best].w;
    P.maxSpeed=std::max(P.maxSpeed,std::sqrt(table[best].u*table[best].u+
                               table[best].v*table[best].v+table[best].w*table[best].w));
  }
  P.nVertices=vertices.size();
  for(const auto&face:BH.inT){
    const auto&tet=M.tets[(std::size_t)face.cell];
    double ux=0,uy=0,uz=0;
    for(int j=0;j<3;++j){
      const std::size_t i=(std::size_t)tet.v[hxt2::TET_FACE[face.lf][j]];
      ux+=P.ux[i]/3.;uy+=P.uy[i]/3.;uz+=P.uz[i]/3.;
    }
    const double vn=ux*face.n[0]+uy*face.n[1]+uz*face.n[2];
    P.minInward=std::min(P.minInward,-vn);
    P.inwardFlux-=face.area*vn;
  }
  if(!(P.inwardFlux>0.)||!(P.minInward>0.))
    throw std::runtime_error("STAGE27_DG_PROFILE requires strictly inward velocity on every inlet face");
  return P;
}

inline void adjust_pressure_source(hxt5b::BoundaryHost&BH,
                                   const nodals_hxt1::HostMesh&M,
                                   const HostProfile&P,nodals::stage23::Vec3 oldU) {
  const int nh=(int)M.h.nhex;
  double oldSum=BH.sourceSum;
  for(const auto&face:BH.inT){
    const auto&tet=M.tets[(std::size_t)face.cell];
    double ux=0,uy=0,uz=0;
    for(int j=0;j<3;++j){
      std::size_t v=(std::size_t)tet.v[hxt2::TET_FACE[face.lf][j]];
      ux+=(double)P.ux[v]/3.;uy+=(double)P.uy[v]/3.;uz+=(double)P.uz[v]/3.;
    }
    const double newFlux=face.area*(ux*face.n[0]+uy*face.n[1]+uz*face.n[2]);
    const double oldFlux=face.area*(oldU.x*face.n[0]+oldU.y*face.n[1]+oldU.z*face.n[2]);
    BH.csrc[(std::size_t)(nh+face.cell)]+=newFlux-oldFlux;
    BH.sourceSum+=newFlux-oldFlux;
  }
  double sum=0;for(double x:BH.csrc)sum+=x;
  if(std::abs(sum+P.inwardFlux)>1.e-8*std::max(1.,P.inwardFlux))
    throw std::runtime_error("STAGE27_DG_PROFILE pressure source/trace flux mismatch");
  std::printf("STAGE27_DG_PROFILE_PRESSURE oldFlux=%.12e newFlux=%.12e sourceSum=%.12e status=PASS\n",
              oldSum,-P.inwardFlux,sum);
}
} // namespace stage27_inlet
