#pragma once
#include "g4_host.hpp"
#include "g2_host.hpp"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace nodals_gpu {

struct G4AMGRefreshHost {
  std::vector<std::int64_t> supportRow;
  std::vector<std::int32_t> supportCell;
  std::vector<std::uint8_t> supportBasis;
  std::vector<std::vector<std::int32_t>> coarsenSlot;
};

inline G4AMGRefreshHost build_g4_amg_refresh_host(const G4SetupHost& S,const HierarchyHost& H) {
  G4AMGRefreshHost R;
  const int nv=S.topo.n;
  R.supportRow.assign((std::size_t)nv+1,0);
  for(std::size_t c=0;c<S.cells.size();++c)
    for(int a=0;a<8;++a) {
      const int g=S.cells[c].ref[a];
      if(g>=0) ++R.supportRow[(std::size_t)g+1];
    }
  for(int g=0;g<nv;++g) R.supportRow[(std::size_t)g+1]+=R.supportRow[(std::size_t)g];

  const std::size_t ns=(std::size_t)R.supportRow.back();
  R.supportCell.assign(ns,-1);
  R.supportBasis.assign(ns,0);
  std::vector<std::int64_t> next=R.supportRow;
  for(std::size_t c=0;c<S.cells.size();++c)
    for(int a=0;a<8;++a) {
      const int g=S.cells[c].ref[a];
      if(g<0) continue;
      const std::size_t k=(std::size_t)next[(std::size_t)g]++;
      R.supportCell[k]=(std::int32_t)c;
      R.supportBasis[k]=(std::uint8_t)a;
    }

  if(H.csr.size()!=H.agg.size())
    throw std::runtime_error("G4B hierarchy aggregate/CSR size mismatch");

  if(H.csr.size()>=2) R.coarsenSlot.resize(H.csr.size()-1);
  for(std::size_t l=0;l+1<H.csr.size();++l) {
    const auto& A=H.csr[l];
    const auto& B=H.csr[l+1];
    const auto& G=H.agg[l+1];
    if((int)G.id.size()!=A.n) throw std::runtime_error("G4B coarse aggregate size mismatch");
    auto& map=R.coarsenSlot[l];
    map.assign(A.val.size(),-1);
    for(int i=0;i<A.n;++i) {
      const int ai=G.id[(std::size_t)i];
      if(ai<0 || ai>=B.n) throw std::runtime_error("G4B invalid coarse row aggregate");
      for(std::int64_t k=A.row[(std::size_t)i];k<A.row[(std::size_t)i+1];++k) {
        const int j=A.col[(std::size_t)k];
        const int aj=G.id[(std::size_t)j];
        auto first=B.col.begin()+B.row[(std::size_t)ai];
        auto last =B.col.begin()+B.row[(std::size_t)ai+1];
        auto it=std::lower_bound(first,last,aj);
        if(it==last || *it!=aj) throw std::runtime_error("G4B coarse numeric destination slot missing");
        const auto dst=(std::int64_t)(it-B.col.begin());
        if(dst>INT32_MAX) throw std::runtime_error("G4B coarse destination slot exceeds int32");
        map[(std::size_t)k]=(std::int32_t)dst;
      }
    }
  }
  return R;
}

} // namespace nodals_gpu
