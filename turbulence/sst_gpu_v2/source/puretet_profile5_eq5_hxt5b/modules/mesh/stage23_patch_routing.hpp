#pragma once
// Stage23C: physically route SST_PATCHES_V1 tags to the live pure-TET
// inlet/outlet operator lists, before B, DG and wall caches are built.
// Operates on an in-memory mesh copy; never modifies the HXT1 file.
#include "tet_patch_registry.hpp"
#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace nodals { namespace mesh {

template <class Mesh>
inline void route_operator_patches(Mesh &M,const std::string &descriptor_path){
  if(descriptor_path.empty()){
    std::printf("STAGE23C_PATCH_ROUTE mode=legacy_native inlet=%zu outlet=%zu operatorLists=UNCHANGED status=PASS\n",M.inTet.size(),M.outTet.size());
    return;
  }
  const Descriptor d=read_descriptor(descriptor_path);
  if(d.mode=="legacy"){
    // Only names may change; the existing Stage19 registry checks type parity.
    std::printf("STAGE23C_PATCH_ROUTE mode=legacy_names inlet=%zu outlet=%zu operatorLists=UNCHANGED status=PASS\n",M.inTet.size(),M.outTet.size());
    return;
  }
  if(M.h.nhex!=0||M.h.ntet==0)throw std::runtime_error("STAGE23C explicit routing presently requires pure TET HXT1");
  if(M.tets.size()!=static_cast<std::size_t>(M.h.ntet)||M.tetVel.size()!=M.tets.size())
    throw std::runtime_error("STAGE23C inconsistent tetrahedron DOF tables");

  std::map<FaceKey,std::vector<FaceId>> owner;
  auto key=[&](int c,int lf){
    if(c<0||static_cast<std::size_t>(c)>=M.tets.size()||lf<0||lf>=4)
      throw std::runtime_error("STAGE23C invalid face index");
    FaceKey k{};const auto &T=M.tets[static_cast<std::size_t>(c)];
    for(int j=0;j<3;++j){
      int i=T.v[FACE_LIDS[lf][j]];
      if(i<0||static_cast<std::size_t>(i)>=M.points.size())
        throw std::runtime_error("STAGE23C invalid face vertex");
      k[j]=i;
    }
    std::sort(k.begin(),k.end());
    if(k[0]==k[1]||k[1]==k[2])throw std::runtime_error("STAGE23C degenerate triangle");
    return k;
  };
  for(std::size_t c=0;c<M.tets.size();++c)
    for(int lf=0;lf<4;++lf)owner[key(static_cast<int>(c),lf)].push_back({static_cast<int>(c),lf});
  std::map<FaceId,std::string> wanted;
  for(const auto &e:owner){
    if(e.second.size()>2)throw std::runtime_error("STAGE23C nonmanifold tetrahedral face");
    if(e.second.size()!=1)continue;
    const FaceId id=e.second.front();
    const auto it=d.overrides.find(id);
    if(it==d.overrides.end())throw std::runtime_error("STAGE23C explicit map missing exterior face cell="+std::to_string(id.first)+" lf="+std::to_string(id.second));
    wanted.emplace(id,it->second.first);
    const auto &v=M.tetVel[static_cast<std::size_t>(id.first)];
    if(v.g[4+id.second]<0||static_cast<std::uint64_t>(v.g[4+id.second])>=M.h.nvel)
      throw std::runtime_error("STAGE23C invalid BF3 face DOF");
  }
  if(wanted.size()!=d.overrides.size())throw std::runtime_error("STAGE23C descriptor labels interior or invalid face");
  std::size_t ni=0,no=0,nw=0;
  for(const auto &w:wanted){
    if(w.second=="inlet")++ni;
    else if(w.second=="outlet")++no;
    else if(w.second=="wall")++nw;
    else throw std::runtime_error("STAGE23C invalid operator kind");
  }
  if(!ni||!no||!nw)throw std::runtime_error("STAGE23C requires nonempty inlet, outlet and wall");
  // Preserve original HXT1 face order wherever possible, important for exact
  // +Z parity. Add genuinely reassigned faces in a deterministic order.
  using Rec=typename decltype(M.inTet)::value_type;
  std::vector<Rec> inlet,outlet;
  std::set<FaceId> handled;
  auto preserve=[&](const auto& old,std::vector<Rec>& dst,const std::string&kind){
    for(const auto&r:old){
      FaceId id{r.cell,r.localFace};
      auto it=wanted.find(id);
      if(it==wanted.end())throw std::runtime_error("STAGE23C native tagged face not exterior");
      if(it->second==kind){
        if(!handled.insert(id).second)throw std::runtime_error("STAGE23C repeated native label");
        dst.push_back(r);
      }
    }
  };
  preserve(M.inTet,inlet,"inlet");
  preserve(M.outTet,outlet,"outlet");
  for(const auto&w:wanted){
    if(w.second=="wall"||handled.count(w.first))continue;
    const int c=w.first.first,lf=w.first.second;
    const Rec r{c,lf,M.tetVel[static_cast<std::size_t>(c)].g[4+lf]};
    (w.second=="inlet"?inlet:outlet).push_back(r);
    handled.insert(w.first);
  }
  if(inlet.size()!=ni||outlet.size()!=no)
    throw std::runtime_error("STAGE23C routed face counts inconsistent");
  M.inTet=std::move(inlet);M.outTet=std::move(outlet);
  M.h.nintet=static_cast<std::uint64_t>(ni);
  M.h.nouttet=static_cast<std::uint64_t>(no);
  std::printf("STAGE23C_PATCH_ROUTE mode=explicit source=%s inlet=%zu outlet=%zu wall=%zu operatorLists=ROUTED naturalOutlet=YES status=PASS\n",descriptor_path.c_str(),ni,no,nw);
}
}} // namespace nodals::mesh
