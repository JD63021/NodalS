// Native HXT1 to DG-P1 BSR4 topology smoke test.
// Usage: RUN_TOPOLOGY.sh [path/to/HXT1_mesh.bin]
// Synthetic mesh first; optional actual pure-tet mesh validates all faces.
#include "../dg_omega_hxt1_topology.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>
using nodals_hxt1::HostMesh;
namespace d=nodals::omega_dg_gpu;
static HostMesh synthetic(){
  HostMesh m;
  m.points={{0,0,0},{1,0,0},{0,1,0},{0,0,1},{0,0,-1}};
  m.tets.resize(2);
  for(int j=0;j<4;++j)m.tets[0].v[j]=std::array<int,4>{0,1,2,3}[(size_t)j];
  for(int j=0;j<4;++j)m.tets[1].v[j]=std::array<int,4>{0,2,1,4}[(size_t)j];
  m.tetGeom.resize(2);
  for(auto&g:m.tetGeom)g.volume=1./6.;
  m.h.nv=m.points.size();m.h.ntet=m.tets.size();
  // Inlet: outer face opposite vertex #0 on tet #0; outlet: same on tet #1.
  m.inTet.push_back({0,0,0});
  m.outTet.push_back({1,0,0});
  m.h.nintet=1;m.h.nouttet=1;
  return m;
}
static void verify(const HostMesh&m,const d::BSR4Topology&p){
  const auto nt=m.tets.size();
  if(p.cells.size()!=nt||p.rowPtr.size()!=nt+1||p.diagonalSlots.size()!=nt)
    throw std::runtime_error("DG_TOPOLOGY_BAD_ROW_DIMENSIONS");
  if(p.rowPtr.front()!=0||p.rowPtr.back()!=(int)p.colInd.size())
    throw std::runtime_error("DG_TOPOLOGY_BAD_BSR_INDICES");
  if(p.faces.size()!=p.internalFaces+p.wallFaces+p.inletFaces+p.outletFaces)
    throw std::runtime_error("DG_TOPOLOGY_FACE_COUNT_MISMATCH");
  if(p.internalFaces*2+p.wallFaces+p.inletFaces+p.outletFaces!=4*nt)
    throw std::runtime_error("DG_TOPOLOGY_FACE_INCIDENCE_MISMATCH");
  if(p.inletFaces!=m.inTet.size()||p.outletFaces!=m.outTet.size())
    throw std::runtime_error("DG_TOPOLOGY_PATCH_MISMATCH");
  for(size_t t=0;t<nt;++t){
    const int begin=p.rowPtr[t],end=p.rowPtr[t+1],slot=p.diagonalSlots[t];
    if(begin>=end||slot<begin||slot>=end||p.colInd[(size_t)slot]!=(int)t)
      throw std::runtime_error("DG_TOPOLOGY_INVALID_DIAGONAL");
    double sum[3]={0,0,0};
    for(auto g:p.cells[t].grad){sum[0]+=g.x;sum[1]+=g.y;sum[2]+=g.z;}
    if(std::sqrt(sum[0]*sum[0]+sum[1]*sum[1]+sum[2]*sum[2])>1e-8)
      throw std::runtime_error("DG_TOPOLOGY_GRADIENT_SUM");
  }
  for(const auto&f:p.faces){
    if(f.left<0||f.slotLL<0||!(f.area>0))
      throw std::runtime_error("DG_TOPOLOGY_INVALID_FACE");
    if(f.right>=0){
      if(f.slotLR<0||f.slotRL<0||f.slotRR<0)
        throw std::runtime_error("DG_TOPOLOGY_MISSING_CROSS_BLOCK");
    }
  }
  std::printf("DG_OMEGA_HXT1_TOPOLOGY_PASS tets=%zu internal=%zu wall=%zu inlet=%zu outlet=%zu blocks=%zu predictedBytes=%zu\n",
     nt,p.internalFaces,p.wallFaces,p.inletFaces,p.outletFaces,p.nblocks(),p.byte_estimate());
}
int main(int argc,char**argv){
  try{
    auto m=synthetic();
    d::BSR4Topology a=d::build_topology(m);
    verify(m,a);
    if(a.internalFaces!=1||a.wallFaces!=4||a.inletFaces!=1||a.outletFaces!=1)
      throw std::runtime_error("DG_TOPOLOGY_SYNTHETIC_FACE_LABELS");
    if(argc>=2){
      auto real=nodals_hxt1::load(argv[1]);
      auto b=d::build_topology(real);
      verify(real,b);
    }
    std::puts("DG_OMEGA_HXT1_TOPOLOGY_ALL_PASS");
    return 0;
  }catch(const std::exception&e){
    std::fprintf(stderr,"DG_OMEGA_HXT1_TOPOLOGY_FAIL reason=%s\n",e.what());return 1;
  }
}
