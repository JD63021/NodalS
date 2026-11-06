#pragma once
// Stage 2: production-shaped PURE TET Bernstein P2 scalar CSR graph.
// This is not the old 8x8 P1+BF3 velocity graph and is not a coupled SST solver.
#include "bernstein_p2_tet.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace nodals { namespace p2 {
using int32=std::int32_t;
struct CSR {
  Connectivity conn;
  std::vector<std::int64_t> row;
  std::vector<int32> col,diag,slot100;
  // Each mesh vertex and mesh edge participates in exactly one scalar unknown.
  // Boundary masks are built from explicit triangulated boundary faces, never
  // from geometric coordinate thresholds (not robust on a curved duct).
  std::vector<std::uint8_t> boundary;
};
inline std::uint64_t csr_key(std::uint32_t i,std::uint32_t j){
  return (std::uint64_t(i)<<32)|std::uint64_t(j);
}
inline CSR build_scalar_csr(int32 nvert,const std::vector<Tet4>&tet,
                            const std::vector<std::array<int32,3>>&boundary_faces={}){
  CSR C;C.conn=build_connectivity(nvert,tet);
  const std::size_t ne=tet.size(),nd=C.conn.ndof;
  std::vector<std::uint64_t> keys;
  keys.reserve(ne*100u);
  for(const auto &g:C.conn.tetra){
    for(int a=0;a<10;++a)for(int b=0;b<10;++b)
      keys.push_back(csr_key((std::uint32_t)g[a],(std::uint32_t)g[b]));
  }
  std::sort(keys.begin(),keys.end());keys.erase(std::unique(keys.begin(),keys.end()),keys.end());
  if(keys.size()>=static_cast<std::size_t>(std::numeric_limits<int32>::max()))
    throw std::runtime_error("P2: 32-bit local slot overflow; implement int64 slot first");
  C.row.resize(nd+1,0);C.col.resize(keys.size());C.diag.resize(nd,-1);
  for(auto k:keys){auto i=(std::uint32_t)(k>>32);if(i>=nd)throw std::runtime_error("P2 csr row overflow");
    ++C.row[i+1];}
  for(std::size_t i=1;i<C.row.size();++i)C.row[i]+=C.row[i-1];
  std::vector<std::int64_t>next=C.row;
  for(auto k:keys){const auto i=(std::uint32_t)(k>>32);C.col[next[i]++]=(int32)(std::uint32_t)k;}
  for(std::size_t i=0;i<nd;++i){
    auto lo=C.row[i],hi=C.row[i+1];auto it=std::lower_bound(C.col.begin()+lo,C.col.begin()+hi,(int32)i);
    if(it==C.col.begin()+hi||*it!=(int32)i)throw std::runtime_error("P2 csr missing diagonal");
    C.diag[i]=(int32)(it-C.col.begin());
  }
  C.slot100.resize(ne*100u);
  for(std::size_t e=0;e<ne;++e){
    const auto &g=C.conn.tetra[e];
    for(int a=0;a<10;++a)for(int b=0;b<10;++b){
      int i=g[a],j=g[b];auto lo=C.row[i],hi=C.row[i+1];
      auto it=std::lower_bound(C.col.begin()+lo,C.col.begin()+hi,j);
      if(it==C.col.begin()+hi||*it!=j)throw std::runtime_error("P2 local slot missing");
      C.slot100[e*100u+a*10+b]=(int32)(it-C.col.begin());
    }
  }
  C.boundary.assign(nd,0);
  for(const auto &tri:boundary_faces){
    for(int a=0;a<3;++a){if(tri[a]<0 || tri[a]>=nvert)throw std::runtime_error("P2 boundary vertex invalid");
      C.boundary[tri[a]]=1;}
    for(int a=0;a<3;++a){int32 i=tri[a],j=tri[(a+1)%3];
      Edge key={std::min(i,j),std::max(i,j)};
      auto it=std::lower_bound(C.conn.edges.begin(),C.conn.edges.end(),key);
      if(it==C.conn.edges.end()||*it!=key)throw std::runtime_error("P2 boundary edge missing");
      C.boundary[nvert+(it-C.conn.edges.begin())]=1;
    }
  }
  return C;
}
// Lift vertex-only P1 data to a genuinely P2 field exactly representing the
// SAME piecewise-linear continuous function, not a fake zero-bubble mapping.
template<class T>
inline std::vector<T> lift_vertex_q1(const Connectivity &C,const std::vector<T>&vertex){
  if(vertex.size()!=(std::size_t)C.nvert)throw std::runtime_error("P2 lift: vertex size mismatch");
  std::vector<T> q(C.ndof);
  std::copy(vertex.begin(),vertex.end(),q.begin());
  for(std::size_t e=0;e<C.edges.size();++e){auto v=C.edges[e];
    q[C.nvert+e]=(vertex[v[0]]+vertex[v[1]])/T(2);
  }
  return q;
}
// The 10 genuine Bernstein controls are the vertex/edge coefficients, not
// point values at the six physical edge midpoints.
template<class T> inline std::size_t project_controls_lower(std::vector<T>&field,T floor){
  std::size_t altered=0;
  for(auto &v:field)if(v<floor){v=floor;++altered;}
  return altered;
}
}} // namespace nodals::p2
