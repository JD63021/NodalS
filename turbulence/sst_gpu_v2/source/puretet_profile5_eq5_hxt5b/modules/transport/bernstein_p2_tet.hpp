#pragma once
// TRUE scalar CG Bernstein P2 tetra: 4 vertices + 6 globally shared EDGES.
// Portable C++17 / nvcc-ready basis, gradients, and host connectivity.
// IMPORTANT: this header is NOT YET wired to the coupled NodalS SST solver.
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

#if defined(__CUDACC__)
#define NODALS_P2_HD __host__ __device__ __forceinline__
#else
#define NODALS_P2_HD inline
#endif

namespace nodals { namespace p2 {
inline constexpr int EDGE[6][2]={{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};

template<class T>
NODALS_P2_HD void basis(const T l[4], T phi[10], T grad[10][3]) {
  // Reference tetra coords (r,s,t) = (l1,l2,l3).
  const T d[4][3]={{T(-1),T(-1),T(-1)},
                   {T(1),T(0),T(0)}, {T(0),T(1),T(0)}, {T(0),T(0),T(1)}};
  for(int i=0;i<4;++i){
    phi[i]=l[i]*l[i];
    for(int k=0;k<3;++k)grad[i][k]=T(2)*l[i]*d[i][k];
  }
  for(int e=0;e<6;++e){
    // Device-safe edge mapping. Namespace EDGE[][] is a host constexpr array;
    // it must not be dereferenced in __host__ __device__ CUDA functions.
    int i=0, j=1;
    switch(e){
      case 0: i=0; j=1; break;
      case 1: i=0; j=2; break;
      case 2: i=0; j=3; break;
      case 3: i=1; j=2; break;
      case 4: i=1; j=3; break;
      case 5: i=2; j=3; break;
    }
    phi[4+e]=T(2)*l[i]*l[j];
    for(int k=0;k<3;++k)
      grad[4+e][k]=T(2)*(l[i]*d[j][k]+l[j]*d[i][k]);
  }
}

// gradB_physical[a][d] = sum_j gradB_ref[a][j] * inverseJ[j][d]
template<class T>
NODALS_P2_HD void physical_gradient(const T gradRef[10][3],
                                   const T inverseJ[9], T out[10][3]) {
  for(int a=0;a<10;++a)
    for(int d=0;d<3;++d)
      out[a][d]=gradRef[a][0]*inverseJ[d] +
                gradRef[a][1]*inverseJ[3+d]+gradRef[a][2]*inverseJ[6+d];
}

using Tet4=std::array<std::int32_t,4>;
using Tet10=std::array<std::int32_t,10>;
using Edge=std::array<std::int32_t,2>;
struct Connectivity {
  std::int32_t nvert=0, ndof=0;
  std::vector<Edge> edges;
  std::vector<Tet10> tetra;
};
inline Connectivity build_connectivity(std::int32_t nvert,
                                     const std::vector<Tet4>&tets) {
  if(nvert<=0)throw std::runtime_error("P2: vertex count <=0");
  Connectivity C;C.nvert=nvert;
  C.edges.reserve(tets.size()*6u);
  for(const auto&t:tets)for(int e=0;e<6;++e){
    const int v=t[EDGE[e][0]], w=t[EDGE[e][1]];
    if(v<0||w<0||v>=nvert||w>=nvert||v==w)
      throw std::runtime_error("P2: invalid tet edge");
    C.edges.push_back({std::min(v,w),std::max(v,w)});
  }
  std::sort(C.edges.begin(),C.edges.end());
  C.edges.erase(std::unique(C.edges.begin(),C.edges.end()),C.edges.end());
  if(C.edges.size()>static_cast<std::size_t>(INT32_MAX-nvert))
    throw std::runtime_error("P2: too many edges for int32 IDs");
  C.ndof=nvert+static_cast<std::int32_t>(C.edges.size());
  C.tetra.resize(tets.size());
  for(std::size_t c=0;c<tets.size();++c){
    const auto&t=tets[c]; auto&g=C.tetra[c];
    for(int a=0;a<4;++a)g[a]=t[a];
    for(int e=0;e<6;++e){
      int v=t[EDGE[e][0]],w=t[EDGE[e][1]];
      Edge key={std::min(v,w),std::max(v,w)};
      auto it=std::lower_bound(C.edges.begin(),C.edges.end(),key);
      if(it==C.edges.end()||*it!=key)throw std::runtime_error("P2: edge lookup failed");
      g[4+e]=nvert+static_cast<std::int32_t>(it-C.edges.begin());
    }
  }
  return C;
}

// Sufficient positivity condition; applies to EVERY physical point of a tet.
// For pure CG P2, each shared edge has ONE coefficient (not four BF3 faces).
template<class T> inline std::size_t floor_controls(std::vector<T>&c,T lower){
  std::size_t changed=0;
  for(T&x:c) if(x<lower){x=lower;++changed;}
  return changed;
}

}} // namespace nodals::p2
#undef NODALS_P2_HD
