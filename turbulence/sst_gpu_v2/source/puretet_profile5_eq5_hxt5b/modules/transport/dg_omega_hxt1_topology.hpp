#pragma once
// Host-side topology bridge: HXT1 pure-tetra mesh -> DG omega BSR4.
// No change to the CG solver or the original mesh files.
// HXT1 contains explicit inlet/outlet tetra boundary records; every other
// exterior tetra face is wall on the currently supported one-wall topology.
// Uses globally sorted face keys for reproducible local face quadrature mapping.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "dg_omega_bsr4.cuh"
#include "../../../hybrid_hxt1/hxt1_mesh.hpp"

namespace nodals { namespace omega_dg_gpu {
struct BSR4Topology {
  std::vector<Cell> cells;
  std::vector<Face> faces;
  std::vector<int> rowPtr,colInd,diagonalSlots;
  std::size_t internalFaces=0,wallFaces=0,inletFaces=0,outletFaces=0;
  std::size_t nblocks() const{return colInd.size();}
  std::size_t byte_estimate() const{
    return 16*nblocks()*sizeof(double)+4*cells.size()*sizeof(double)+
      cells.size()*sizeof(Cell)+faces.size()*sizeof(Face)+
      rowPtr.size()*sizeof(int)+colInd.size()*sizeof(int);
  }
};
inline Vec3 subtract(Vec3 a,Vec3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
inline Vec3 cross(Vec3 a,Vec3 b) {
  return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};
}
inline Vec3 scale(Vec3 x,double a){return {x.x*a,x.y*a,x.z*a};}
inline double magnitude(Vec3 x){return std::sqrt(dot(x,x));}
inline Vec3 position(const nodals_hxt1::Point&x){return {x.x,x.y,x.z};}
inline std::array<int,3> local_face_vertices(const nodals_hxt1::TetConn&t,int opposite){
  if(opposite<0||opposite>3)throw std::runtime_error("DG topology invalid local face");
  std::array<int,3> key{};int q=0;
  for(int a=0;a<4;++a)if(a!=opposite)key[(std::size_t)q++]=t.v[a];
  std::sort(key.begin(),key.end());return key;
}
inline std::array<int,3> face_key(const nodals_hxt1::HostMesh&M,int cell,int opposite){
  if(cell<0||(std::size_t)cell>=M.tets.size())throw std::runtime_error("DG topology boundary cell out of range");
  return local_face_vertices(M.tets[(std::size_t)cell],opposite);
}
inline Cell geometry(const nodals_hxt1::HostMesh&M,int t) {
  const auto &T=M.tets[(std::size_t)t];
  Vec3 x[4];
  for(int a=0;a<4;++a){
    if(T.v[a]<0||(std::size_t)T.v[a]>=M.points.size())throw std::runtime_error("DG topology tetra vertex out of range");
    x[a]=position(M.points[(std::size_t)T.v[a]]);
  }
  const Vec3 e1=subtract(x[1],x[0]),e2=subtract(x[2],x[0]),e3=subtract(x[3],x[0]);
  const double determinant=dot(e1,cross(e2,e3));
  if(!std::isfinite(determin)||std::fabs(determin)<1.e-30)throw std::runtime_error("DG topology degenerate tetra");
  Cell c{};
  c.volume=std::fabs(determin)/6.;
  c.grad[1]=scale(cross(e2,e3),1./determinant);
  c.grad[2]=scale(cross(e3,e1),1./determinant);
  c.grad[3]=scale(cross(e1,e2),1./determinant);
  c.grad[0]=scale({c.grad[1].x+c.grad[2].x+c.grad[3].x,
                   c.grad[1].y+c.grad[2].y+c.grad[3].y,
                   c.grad[1].z+c.grad[2].z+c.grad[3].z},-1.);
  // The HXT1 mesh geometry should describe exactly the same physical tetra.
  const double vHXT=M.tetGeom[(std::size_t)t].volume;
  if(!(vHXT>0.)||std::fabs(vHXT-c.volume)>1.e-6*std::max(vHXT,c.volume))
    throw std::runtime_error("DG topology tetra volume disagreement with native HXT1 geometry");
  return c;
}
inline BSR4Topology build_topology(const nodals_hxt1::HostMesh&M){
  if(M.h.nhex||M.h.ninterface||M.h.ntet==0||M.tets.size()!=M.h.ntet ||
     M.tetGeom.size()!=M.h.ntet||M.points.size()!=M.h.nv)
    throw std::runtime_error("DG topology pure-tet HXT1 mesh required");
  if(M.h.ntet>(std::uint64_t)std::numeric_limits<int>::max()/8)
    throw std::runtime_error("DG topology int32 DOF or slot overflow");
  BSR4Topology p;
  const int nt=(int)M.h.ntet;
  p.cells.reserve((std::size_t)nt);
  p.faces.reserve((std::size_t)nt*3);
  struct Owner{int cell,opposite,face;};
  std::map<std::array<int,3>,Owner> owners;
  for(int t=0;t<nt;++t){
    const auto &tet=M.tets[(std::size_t)t];
    std::array<int,4> verts{tet.v[0],tet.v[1],tet.v[2],tet.v[3]};
    auto sorted=verts;std::sort(sorted.begin(),sorted.end());
    if(std::adjacent_find(sorted.begin(),sorted.end())!=sorted.end())
      throw std::runtime_error("DG topology duplicated vertices within tetrahedron");
    p.cells.push_back(geometry(M,t));
    for(int opposite=0;opposite<4;++opposite){
      const auto key=local_face_vertices(tet,opposite);
      auto it=owners.find(key);
      if(it==owners.end()){
        Face f{};f.left=t;f.right=-1;
        for(int j=0;j<3;++j){f.vertexL[j]=-1;f.vertexR[j]=-1;}
        f.slotLL=f.slotLR=f.slotRL=f.slotRR=-1;
        p.faces.push_back(f);
        owners.emplace(key,Owner{t,opposite,(int)p.faces.size()-1});
      }else{
        Face&f=p.faces[(std::size_t)it->second.face];
        if(f.right>=0)throw std::runtime_error("DG topology nonmanifold tetrahedral face");
        f.right=t;
      }
    }
  }
  // Recover face keys, per-side index maps and outward-oriented normals.
  for(const auto&item:owners){
    const auto&key=item.first;
    Face&f=p.faces[(std::size_t)item.second.face];
    const auto&leftTet=M.tets[(std::size_t)f.left];
    Vec3 x0=position(M.points[(std::size_t)key[0]]);
    Vec3 x1=position(M.points[(std::size_t)key[1]]);
    Vec3 x2=position(M.points[(std::size_t)key[2]]);
    Vec3 normal=cross(subtract(x1,x0),subtract(x2,x0));
    const double area2=magnitude(normal);
    if(!(area2>0.)||!std::isfinite(area2))throw std::runtime_error("DG topology degenerate face");
    Vec3 opposite=position(M.points[(std::size_t)leftTet.v[item.second.opposite]]);
    if(dot(normal,subtract(opposite,x0))>0.)normal=scale(normal,-1.);
    f.area=area2*.5;
    f.normal=scale(normal,1./area2);
    for(int j=0;j<3;++j){
      bool foundL=false,foundR=f.right<0;
      for(int a=0;a<4;++a){
        if(leftTet.v[a]==key[(std::size_t)j]){f.vertexL[j]=a;foundL=true;}
        if(f.right>=0&&M.tets[(std::size_t)f.right].v[a]==key[(std::size_t)j]){
          f.vertexR[j]=a;foundR=true;
        }
      }
      if(!foundL||!foundR)throw std::runtime_error("DG topology broken local face map");
    }
    if(f.right>=0){
      ++p.internalFaces;
      const auto&rightTet=M.tets[(std::size_t)f.right];
      bool found=false;
      for(int a=0;a<4;++a)if(rightTet.v[a]!=key[0]&&rightTet.v[a]!=key[1]&&rightTet.v[a]!=key[2]){
        Vec3 other=position(M.points[(std::size_t)rightTet.v[a]]);
        if(!(dot(f.normal,subtract(other,x0))>0.))
          throw std::runtime_error("DG topology internal face not oriented left-to-right");
        found=true;break;
      }
      if(!found)throw std::runtime_error("DG topology missing right opposite vertex");
    }else f.boundary=Wall;
  }
  // Override the inferred wall faces using native HXT1 explicit inlet/outlet.
  const auto set_bc=[&](const std::vector<nodals_hxt1::BoundaryRec>&recs,int kind){
    for(const auto&r:recs){
      const auto key=face_key(M,r.cell,r.localFace);
      auto it=owners.find(key);
      if(it==owners.end())throw std::runtime_error("DG topology boundary not in tetra mesh");
      Face&f=p.faces[(std::size_t)it->second.face];
      if(f.right>=0||f.left!=r.cell||f.boundary!=Wall)
        throw std::runtime_error("DG topology boundary record duplicated or wrong owner");
      f.boundary=kind;
    }
  };
  set_bc(M.inTet,Inlet);
  set_bc(M.outTet,Outlet);
  for(const auto&f:p.faces)if(f.right<0){
    if(f.boundary==Wall)++p.wallFaces;
    else if(f.boundary==Inlet)++p.inletFaces;
    else if(f.boundary==Outlet)++p.outletFaces;
    else throw std::runtime_error("DG topology unknown exterior face");
  }
  if(p.inletFaces!=M.inTet.size()||p.outletFaces!=M.outTet.size())
    throw std::runtime_error("DG topology lost inlet or outlet boundary record");
  if(!p.wallFaces||!p.inletFaces||!p.outletFaces)
    throw std::runtime_error("DG topology requires one inlet, one outlet and wall");
  std::vector<std::set<int>> adj((std::size_t)nt);
  for(int t=0;t<nt;++t)adj[(std::size_t)t].insert(t);
  for(const auto&f:p.faces)if(f.right>=0){
    adj[(std::size_t)f.left].insert(f.right);
    adj[(std::size_t)f.right].insert(f.left);
  }
  std::map<std::pair<int,int>,int> slot;
  p.rowPtr.reserve((std::size_t)nt+1);
  p.diagonalSlots.reserve((std::size_t)nt);
  p.rowPtr.push_back(0);
  for(int t=0;t<nt;++t){
    for(int v:adj[(std::size_t)t]){
      slot.emplace(std::make_pair(t,v),(int)p.colInd.size());
      p.colInd.push_back(v);
    }
    p.diagonalSlots.push_back(slot.at({t,t}));
    p.rowPtr.push_back((int)p.colInd.size());
  }
  for(auto&f:p.faces){
    f.slotLL=slot.at({f.left,f.left});
    if(f.right>=0){
      f.slotLR=slot.at({f.left,f.right});
      f.slotRL=slot.at({f.right,f.left});
      f.slotRR=slot.at({f.right,f.right});
    }
  }
  return p;
}
}} // namespace nodals::omega_dg_gpu
