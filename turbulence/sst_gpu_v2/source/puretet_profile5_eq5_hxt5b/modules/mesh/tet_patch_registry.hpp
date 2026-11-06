#pragma once
// Stage19: non-invasive native patch registry and geometric preflight.
// No CUDA, turbulence, DG or SIMPLE arithmetic lives here.
// The existing boundary operators remain the accepted Stage18 implementation.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nodals { namespace mesh {

struct Face {
  int cell=0,localFace=0;
  std::string kind,patch;
  double area=0,nx=0,ny=0,nz=0;
};
struct Registry {
  std::vector<Face> faces;
  std::map<std::pair<std::string,std::string>,std::pair<std::size_t,double>> patches;
  std::size_t inlet=0,outlet=0,wall=0;
};
using FaceId=std::pair<int,int>;
using FaceKey=std::array<int,3>;
inline constexpr int FACE_LIDS[4][3]={{1,2,3},{0,3,2},{0,1,3},{0,2,1}};
inline bool valid_name(const std::string&s){
  if(s.empty()||!(s[0]>='a'&&s[0]<='z'))return false;
  for(char c:s)if(!((c>='a'&&c<='z')||(c>='0'&&c<='9')||c=='_'))return false;
  return true;
}
inline bool valid_kind(const std::string&s){return s=="inlet"||s=="outlet"||s=="wall";}
struct Descriptor {
  std::string mode="legacy";
  std::set<std::pair<std::string,std::string>> patches;
  std::map<FaceId,std::pair<std::string,std::string>> overrides;
};
inline Descriptor read_descriptor(const std::string&path){
  Descriptor d;
  if(path.empty()){
    d.patches={{"inlet","main"},{"outlet","main"},{"wall","main"}};
    return d;
  }
  std::ifstream f(path);
  if(!f)throw std::runtime_error("STAGE19 missing patch descriptor: "+path);
  std::string line;int lineNo=0;
  if(!std::getline(f,line)||line!="SST_PATCHES_V1")throw std::runtime_error("STAGE19 invalid patch descriptor header");
  ++lineNo;
  while(std::getline(f,line)){
    ++lineNo;
    auto pos=line.find('#');if(pos!=std::string::npos)line.resize(pos);
    std::istringstream in(line);
    std::string cmd;in>>cmd;if(cmd.empty())continue;
    if(cmd=="mode"){
      if(!(in>>d.mode)||(d.mode!="legacy"&&d.mode!="explicit"))throw std::runtime_error("STAGE19 invalid patch mode");
      std::string extra;if(in>>extra)throw std::runtime_error("STAGE19 extra mode argument");
    }else if(cmd=="patch"){
      std::string kind,name,extra;
      if(!(in>>kind>>name)||!valid_kind(kind)||!valid_name(name)||(in>>extra))throw std::runtime_error("STAGE19 invalid patch declaration line "+std::to_string(lineNo));
      if(!d.patches.emplace(kind,name).second)throw std::runtime_error("STAGE19 duplicate patch declaration");
    }else if(cmd=="face"){
      int cell=-1,lf=-1;std::string kind,name,extra;
      if(!(in>>cell>>lf>>kind>>name)||cell<0||lf<0||lf>3||!valid_kind(kind)||!valid_name(name)||(in>>extra))throw std::runtime_error("STAGE19 invalid face declaration line "+std::to_string(lineNo));
      if(!d.overrides.emplace(FaceId{cell,lf},std::make_pair(kind,name)).second)throw std::runtime_error("STAGE19 duplicate face mapping line "+std::to_string(lineNo));
    }else throw std::runtime_error("STAGE19 unknown patch descriptor instruction "+cmd);
  }
  for(auto kind:{"inlet","outlet","wall"}){
    bool found=false;
    for(const auto&p:d.patches)if(p.first==kind){found=true;break;}
    if(!found)throw std::runtime_error(std::string("STAGE19 missing declared ")+kind+" patch");
  }
  for(const auto&o:d.overrides)if(!d.patches.count(o.second))throw std::runtime_error("STAGE19 undeclared patch in face mapping");
  return d;
}

template<class Mesh>
inline Registry build_registry(const Mesh&M,const std::string&path){
  if(M.h.nhex!=0||M.h.ntet==0)throw std::runtime_error("STAGE19 registry currently requires pure TET HXT1 mesh");
  const Descriptor d=read_descriptor(path);
  std::map<FaceKey,std::vector<FaceId>> incidence;
  auto face_key=[&](int cell,int lf){
    if(cell<0||static_cast<std::size_t>(cell)>=M.tets.size()||lf<0||lf>3)throw std::runtime_error("STAGE19 face index outside mesh");
    FaceKey k{};
    const auto&t=M.tets[static_cast<std::size_t>(cell)];
    for(int j=0;j<3;++j){
      const int idx=t.v[FACE_LIDS[lf][j]];
      if(idx<0||static_cast<std::size_t>(idx)>=M.points.size())throw std::runtime_error("STAGE19 invalid tetrahedron connectivity");
      k[j]=idx;
    }
    std::sort(k.begin(),k.end());
    if(k[0]==k[1]||k[1]==k[2])throw std::runtime_error("STAGE19 degenerate face vertices");
    return k;
  };
  for(int c=0;c<(int)M.tets.size();++c)for(int lf=0;lf<4;++lf)
    incidence[face_key(c,lf)].emplace_back(c,lf);
  std::map<FaceId,std::string> native;
  auto native_add=[&](const auto&recs,const std::string&kind){
    for(const auto&r:recs){
      const FaceId id{r.cell,r.localFace};
      (void)face_key(id.first,id.second);
      if(!native.emplace(id,kind).second)throw std::runtime_error("STAGE19 duplicate HXT1 boundary label");
    }
  };
  native_add(M.inTet,"inlet");native_add(M.outTet,"outlet");
  Registry result;
  std::set<FaceId> seen;
  for(const auto&kv:incidence){
    if(kv.second.size()>2)throw std::runtime_error("STAGE19 nonmanifold tetrahedral face");
    if(kv.second.size()==2){
      for(const auto&id:kv.second){
        if(native.count(id))throw std::runtime_error("STAGE19 HXT1 label refers to internal face");
        if(d.overrides.count(id))throw std::runtime_error("STAGE19 patch mapping refers to internal face");
      }
      continue;
    }
    const FaceId id=kv.second.front();seen.insert(id);
    std::string kind="wall",name="main";
    auto label=native.find(id);if(label!=native.end())kind=label->second;
    if(d.mode=="explicit"&&!d.overrides.count(id))throw std::runtime_error("STAGE19 explicit patch map has unlabeled exterior face");
    auto user=d.overrides.find(id);
    if(user!=d.overrides.end()){
      if(user->second.first!=kind)throw std::runtime_error("STAGE19 label conflicts with existing HXT1 inlet/outlet/wall operator (Stage23 required)");
      name=user->second.second;
    }
    if(!d.patches.count({kind,name}))throw std::runtime_error("STAGE19 face references undeclared patch "+kind+"."+name);
    const auto&t=M.tets[static_cast<std::size_t>(id.first)];
    const auto&a=M.points[t.v[FACE_LIDS[id.second][0]]];
    const auto&b=M.points[t.v[FACE_LIDS[id.second][1]]];
    const auto&c=M.points[t.v[FACE_LIDS[id.second][2]]];
    const double ux=b.x-a.x,uy=b.y-a.y,uz=b.z-a.z;
    const double vx=c.x-a.x,vy=c.y-a.y,vz=c.z-a.z;
    double nx=uy*vz-uz*vy,ny=uz*vx-ux*vz,nz=ux*vy-uy*vx;
    const double magnitude=std::sqrt(nx*nx+ny*ny+nz*nz);
    if(!(magnitude>0)||!std::isfinite(magnitude))throw std::runtime_error("STAGE19 zero or invalid exterior face area");
    nx/=magnitude;ny/=magnitude;nz/=magnitude;
    const auto&opposite=M.points[t.v[id.second]];
    const double fcX=(a.x+b.x+c.x)/3.0,fcY=(a.y+b.y+c.y)/3.0,fcZ=(a.z+b.z+c.z)/3.0;
    if(nx*(opposite.x-fcX)+ny*(opposite.y-fcY)+nz*(opposite.z-fcZ)>0){nx=-nx;ny=-ny;nz=-nz;}
    Face face{id.first,id.second,kind,name,0.5*magnitude,nx,ny,nz};
    result.faces.push_back(face);
    auto&stat=result.patches[{kind,name}];++stat.first;stat.second+=face.area;
    if(kind=="inlet")++result.inlet;
    else if(kind=="outlet")++result.outlet;
    else ++result.wall;
  }
  for(const auto&label:native)if(!seen.count(label.first))throw std::runtime_error("STAGE19 boundary label is not on exterior");
  for(const auto&label:d.overrides)if(!seen.count(label.first))throw std::runtime_error("STAGE19 patch mapping not on exterior");
  for(const auto&p:d.patches)if(result.patches[p].first==0)throw std::runtime_error("STAGE19 empty declared patch "+p.first+"."+p.second);
  if(result.inlet==0||result.outlet==0||result.wall==0)throw std::runtime_error("STAGE19 requires nonempty inlet/outlet/wall");
  for(const auto&v:result.patches)
    std::printf("SST_PATCH patch=%s.%s faces=%zu area=%.12e status=PASS\n",v.first.first.c_str(),v.first.second.c_str(),v.second.first,v.second.second);
  std::printf("SST_PATCH_REGISTRY faces=%zu inlet=%zu outlet=%zu wall=%zu mode=%s coverage=COMPLETE normals=OUTWARD status=PASS\n",result.faces.size(),result.inlet,result.outlet,result.wall,d.mode.c_str());
  return result;
}

template<class BoundaryHost,class Mesh>
inline void audit_legacy_operators(const Registry&R,const BoundaryHost&B,const Mesh&M){
  if(R.inlet!=M.inTet.size()||R.outlet!=M.outTet.size()||R.wall!=B.wallT.size()||R.inlet!=B.inT.size())
    throw std::runtime_error("STAGE19 registry/operator face totals mismatch");
  std::vector<FaceId> actual,expected;
  for(const auto&f:B.wallT)actual.emplace_back(f.cell,f.lf);
  for(const auto&f:R.faces)if(f.kind=="wall")expected.emplace_back(f.cell,f.localFace);
  if(actual!=expected)throw std::runtime_error("STAGE19 registry/legacy wall-face ordering mismatch");
  std::printf("SST_PATCH_OPERATOR_AUDIT dgInlet=%zu outlet=%zu spaldingWall=%zu order=EXACT numericalBoundary=STAGE18_UNCHANGED status=PASS\n",R.inlet,R.outlet,R.wall);
}
}} // namespace nodals::mesh
