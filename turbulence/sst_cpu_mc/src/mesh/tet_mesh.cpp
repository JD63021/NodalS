#include "tet_mesh.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace nodals::cpu_mc::m2 {
namespace {
struct ArrHash {std::size_t operator()(const std::array<int,3>&a) const noexcept {
  std::size_t h=1469598103934665603ull;
  for(int v:a){h^=(unsigned int)v;h*=1099511628211ull;}return h;
}};
std::array<int,3> sorted_face(const std::array<int,4>&v,int skip){
  std::array<int,3> f{};int k=0;for(int i=0;i<4;++i)if(i!=skip)f[k++]=v[i];
  std::sort(f.begin(),f.end());return f;
}
}
Mesh read_msh2(const std::string& path){
  std::ifstream in(path);if(!in)throw std::runtime_error("cannot open mesh "+path);
  Mesh m;std::string key;bool hasFmt=false,hasNodes=false,hasTets=false;
  std::unordered_map<std::int64_t,int> nodeid;
  std::unordered_map<std::array<int,3>,int,ArrHash> boundary;
  while(in>>key){
    if(key=="$MeshFormat"){
      double version=0;int filetype=1,dataSize=0;
      in>>version>>filetype>>dataSize;
      if(version<2.0||version>=3.0||filetype!=0)
        throw std::runtime_error("M2 requires ASCII Gmsh MSH2 mesh");
      hasFmt=true;
    } else if(key=="$Nodes"){
      std::size_t n=0;in>>n;if(!n)throw std::runtime_error("empty mesh nodes");
      m.points.reserve(n);nodeid.reserve(n*2);
      for(std::size_t i=0;i<n;++i){std::int64_t id;Vec3 p{};
        if(!(in>>id>>p.x>>p.y>>p.z))throw std::runtime_error("truncated nodes");
        if(!nodeid.emplace(id,(int)i).second)throw std::runtime_error("duplicate node ID");
        m.points.push_back(p);
      }
      hasNodes=true;
    } else if(key=="$Elements"){
      if(!hasNodes)throw std::runtime_error("nodes must precede elements");
      std::size_t n=0;in>>n;
      std::string line;std::getline(in,line);
      m.tets.reserve(n);
      for(std::size_t e=0;e<n;++e){
        if(!std::getline(in,line))throw std::runtime_error("truncated elements");
        std::istringstream ls(line);std::int64_t id=0;int typ=0,ntags=0;
        if(!(ls>>id>>typ>>ntags)||ntags<0||ntags>100)throw std::runtime_error("malformed element line");
        int physical=0;
        for(int tag=0,v;tag<ntags;++tag){if(!(ls>>v))throw std::runtime_error("truncated element tags");if(tag==0)physical=v;}
        if(typ==2 || typ==4){
          std::array<int,4> v{};
          for(int i=0;i<(typ==2?3:4);++i){std::int64_t nid;
            if(!(ls>>nid))throw std::runtime_error("missing element vertex");
            const auto it=nodeid.find(nid);if(it==nodeid.end())throw std::runtime_error("unknown node ID");
            v[i]=it->second;
          }
          if(typ==4){
            Tet t{};t.v=v;
            const Vec3 a=m.points[v[0]],b=m.points[v[1]],c=m.points[v[2]],d=m.points[v[3]];
            const double det=dot(b-a,cross(c-a,d-a));
            if(!(std::isfinite(det)&&std::abs(det)>1e-30))throw std::runtime_error("degenerate tet");
            if(det<0.0)std::swap(t.v[1],t.v[2]);
            t.volume=std::abs(det)/6.0;
            m.totalVolume+=t.volume;
            m.tets.push_back(t);
          } else {
            if(physical<1||physical>3)throw std::runtime_error("unexpected boundary physical tag");
            auto f=std::array<int,3>{v[0],v[1],v[2]};std::sort(f.begin(),f.end());
            if(!boundary.emplace(f,physical).second)throw std::runtime_error("duplicated boundary triangle");
          }
        }
      }
      hasTets=true;
    }
  }
  if(!hasFmt||!hasNodes||!hasTets||m.tets.empty())throw std::runtime_error("incomplete MSH2 mesh");
  std::unordered_map<std::array<int,3>,int,ArrHash> faceMap;
  faceMap.reserve(m.tets.size()*3);
  for(int c=0;c<(int)m.tets.size();++c){
    auto &t=m.tets[c];
    for(int i=0;i<4;++i){
      const auto f=sorted_face(t.v,i);
      auto [it,inserted]=faceMap.emplace(f,(int)m.faces.size());
      if(inserted){Face nf{};nf.v=f;nf.cell0=c;m.faces.push_back(nf);}
      else {auto&ff=m.faces[it->second];if(ff.cell1>=0)throw std::runtime_error("non-manifold face");ff.cell1=c;}
      t.face[i]=it->second;
    }
  }
  for(auto& f:m.faces){
    const auto it=boundary.find(f.v);
    if(f.cell1<0){if(it==boundary.end())throw std::runtime_error("untagged boundary face");f.patch=it->second;++m.boundary[f.patch];}
    else if(it!=boundary.end())throw std::runtime_error("boundary triangle is actually internal");
  }
  if(m.boundary[1]+m.boundary[2]+m.boundary[3]!=(std::int64_t)boundary.size())
    throw std::runtime_error("boundary triangles not matched one-to-one");
  for(int tag=1;tag<=3;++tag)if(!m.boundary[tag])throw std::runtime_error("missing inlet/outlet/wall patch");
  return m;
}
Vec3 outward_area(const Mesh& m,int cell,int localFace){
  const auto&t=m.tets.at((std::size_t)cell);const auto&f=m.faces.at((std::size_t)t.face[localFace]);
  const Vec3 a=m.points[f.v[0]],b=m.points[f.v[1]],c=m.points[f.v[2]],op=m.points[t.v[localFace]];
  Vec3 s=0.5*cross(b-a,c-a);
  if(dot(s,op-a)>0)s=-1.0*s;
  return s;
}
}
