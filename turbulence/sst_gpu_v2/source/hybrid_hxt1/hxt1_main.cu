#include "hxt1_mesh.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace nodals_hxt1;

__global__ void audit_hex_kernel(const HexConn* c,const HexVel* v,const HexGeom* g,
                                 std::uint64_t n,std::uint64_t nv,std::uint64_t nvel,
                                 unsigned long long*err){
  std::uint64_t i=(std::uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=n)return;
  for(int a=0;a<8;++a) if(c[i].v[a]<0 || (std::uint64_t)c[i].v[a]>=nv) atomicAdd(err,1ULL);
  for(int a=0;a<14;++a) if(v[i].g[a]<0 || (std::uint64_t)v[i].g[a]>=nvel) atomicAdd(err,1ULL);
  if(!(g[i].volume>0.0) || !(g[i].h2>0.0)) atomicAdd(err,1ULL);
  for(int q=0;q<8;++q) if(!(g[i].detJ[q]>0.0) || !isfinite(g[i].detJ[q])) atomicAdd(err,1ULL);
}
__global__ void audit_tet_kernel(const TetConn* c,const TetVel* v,const TetGeom* g,
                                 std::uint64_t n,std::uint64_t nv,std::uint64_t nvel,
                                 unsigned long long*err){
  std::uint64_t i=(std::uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=n)return;
  for(int a=0;a<4;++a) if(c[i].v[a]<0 || (std::uint64_t)c[i].v[a]>=nv) atomicAdd(err,1ULL);
  for(int a=0;a<8;++a) if(v[i].g[a]<0 || (std::uint64_t)v[i].g[a]>=nvel) atomicAdd(err,1ULL);
  if(!(g[i].volume>0.0) || !(g[i].h2>0.0) || !(g[i].det>0.0) || !isfinite(g[i].det)) atomicAdd(err,1ULL);
}
__global__ void audit_interface_kernel(const InterfaceRec*r,std::uint64_t n,
                                       std::uint64_t nh,std::uint64_t nt,std::uint64_t nvel,
                                       unsigned long long*err){
  std::uint64_t i=(std::uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=n)return;
  const auto&a=r[i];
  if(a.hexCell<0 || (std::uint64_t)a.hexCell>=nh || a.hexLocalFace<0 || a.hexLocalFace>=6) atomicAdd(err,1ULL);
  if(a.tet0<0 || (std::uint64_t)a.tet0>=nt || a.tet0LocalFace<0 || a.tet0LocalFace>=4) atomicAdd(err,1ULL);
  if(a.tet1<0 || (std::uint64_t)a.tet1>=nt || a.tet1LocalFace<0 || a.tet1LocalFace>=4) atomicAdd(err,1ULL);
  if(a.hexBubble<0 || a.tet0Bubble<0 || a.tet1Bubble<0 ||
     (std::uint64_t)a.hexBubble>=nvel || (std::uint64_t)a.tet0Bubble>=nvel ||
     (std::uint64_t)a.tet1Bubble>=nvel) atomicAdd(err,1ULL);
  if(a.hexBubble==a.tet0Bubble || a.hexBubble==a.tet1Bubble || a.tet0Bubble==a.tet1Bubble) atomicAdd(err,1ULL);
}
__global__ void audit_boundary_kernel(const BoundaryRec*r,std::uint64_t n,
                                      std::uint64_t nc,int arity,std::uint64_t nvel,
                                      unsigned long long*err){
  std::uint64_t i=(std::uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=n)return;
  if(r[i].cell<0 || (std::uint64_t)r[i].cell>=nc ||
     r[i].localFace<0 || r[i].localFace>=arity ||
     r[i].bubble<0 || (std::uint64_t)r[i].bubble>=nvel) atomicAdd(err,1ULL);
}

template<class T>
static std::size_t bytes(const std::vector<T>&v){return v.size()*sizeof(T);}

int main(int argc,char**argv){
  try{
    std::string mesh;
    for(int i=1;i<argc;++i){
      std::string a=argv[i];
      if(a=="--mesh" && i+1<argc) mesh=argv[++i];
      else if(a=="--help"){
        std::printf("usage: %s --mesh HXT1_mesh.bin\n",argv[0]); return 0;
      } else throw std::runtime_error("unknown/incomplete argument: "+a);
    }
    if(mesh.empty()) throw std::runtime_error("--mesh required");

    HostMesh M=load(mesh);
    const auto&H=M.h;
    bool host=true;
    host &= M.hexes.size()==H.nhex && M.tets.size()==H.ntet;
    host &= M.hexVel.size()==H.nhex && M.tetVel.size()==H.ntet;
    host &= M.hexGeom.size()==H.nhex && M.tetGeom.size()==H.ntet;
    host &= M.interface.size()==H.ninterface;
    host &= H.npressure==H.nhex+H.ntet;
    host &= H.nvel==H.nv+H.nhexface+H.ntetface;
    if(!host) throw std::runtime_error("HXT1 host header/vector consistency failed");

    double minHex=1e300,minTet=1e300,vol=0.0;
    for(const auto&g:M.hexGeom){vol+=g.volume;for(double d:g.detJ)minHex=std::min(minHex,d);}
    for(const auto&g:M.tetGeom){vol+=g.volume;minTet=std::min(minTet,g.det);}
    if(!(minHex>0.0 && minTet>0.0)) throw std::runtime_error("non-positive geometry in host plan");

    std::printf("NODALS_HXT1_HOST_COUNTS vertices=%llu hex=%llu tet=%llu hexFaceBubbles=%llu tetFaceBubbles=%llu scalarVelocityDofs=%llu pressureDofs=%llu interface=%llu status=PASS\n",
      (unsigned long long)H.nv,(unsigned long long)H.nhex,(unsigned long long)H.ntet,
      (unsigned long long)H.nhexface,(unsigned long long)H.ntetface,
      (unsigned long long)H.nvel,(unsigned long long)H.npressure,(unsigned long long)H.ninterface);
    std::printf("NODALS_HXT1_HOST_GEOMETRY minHexGaussDetJ=%.12e minTetDet=%.12e totalVolume=%.12e status=PASS\n",minHex,minTet,vol);
    std::printf("NODALS_HXT1_ELEMENT_CONTRACT hex=Q1_PLUS_BF2 localScalar=14 tet=P1_PLUS_BF3 localScalar=8 pressure=Q0_P0_ONE_PER_CELL interface=WEAK_NOT_YET_ASSEMBLED status=PASS\n");

    int dev=0; cudaDeviceProp prop{};
    HXT1_CUDA(cudaGetDevice(&dev)); HXT1_CUDA(cudaGetDeviceProperties(&prop,dev));

    Dev<Point>d_points(M.points);
    Dev<HexConn>d_hex(M.hexes); Dev<TetConn>d_tet(M.tets);
    Dev<HexVel>d_hv(M.hexVel); Dev<TetVel>d_tv(M.tetVel);
    Dev<HexGeom>d_hg(M.hexGeom); Dev<TetGeom>d_tg(M.tetGeom);
    Dev<InterfaceRec>d_if(M.interface);
    Dev<BoundaryRec>d_wall(M.wallHex),d_ih(M.inHex),d_it(M.inTet),d_oh(M.outHex),d_ot(M.outTet);

    std::size_t total=
      bytes(M.points)+bytes(M.hexes)+bytes(M.tets)+bytes(M.hexVel)+bytes(M.tetVel)+
      bytes(M.hexGeom)+bytes(M.tetGeom)+bytes(M.interface)+
      bytes(M.wallHex)+bytes(M.inHex)+bytes(M.inTet)+bytes(M.outHex)+bytes(M.outTet);
    std::printf("NODALS_HXT1_GPU_UPLOAD device=%s cc=%d.%d persistentMeshPlanBytes=%zu persistentMeshPlanMiB=%.3f hostToDeviceSetupOnly=1 status=PASS\n",
      prop.name,prop.major,prop.minor,total,total/(1024.0*1024.0));

    unsigned long long *d_err=nullptr,err=0;
    HXT1_CUDA(cudaMalloc((void**)&d_err,sizeof(*d_err)));
    HXT1_CUDA(cudaMemset(d_err,0,sizeof(*d_err)));
    const int B=256;
    if(H.nhex) audit_hex_kernel<<<(H.nhex+B-1)/B,B>>>(d_hex.p,d_hv.p,d_hg.p,H.nhex,H.nv,H.nvel,d_err);
    if(H.ntet) audit_tet_kernel<<<(H.ntet+B-1)/B,B>>>(d_tet.p,d_tv.p,d_tg.p,H.ntet,H.nv,H.nvel,d_err);
    if(H.ninterface) audit_interface_kernel<<<(H.ninterface+B-1)/B,B>>>(d_if.p,H.ninterface,H.nhex,H.ntet,H.nvel,d_err);
    if(H.nwallhex) audit_boundary_kernel<<<(H.nwallhex+B-1)/B,B>>>(d_wall.p,H.nwallhex,H.nhex,6,H.nvel,d_err);
    if(H.ninhex) audit_boundary_kernel<<<(H.ninhex+B-1)/B,B>>>(d_ih.p,H.ninhex,H.nhex,6,H.nvel,d_err);
    if(H.nintet) audit_boundary_kernel<<<(H.nintet+B-1)/B,B>>>(d_it.p,H.nintet,H.ntet,4,H.nvel,d_err);
    if(H.nouthex) audit_boundary_kernel<<<(H.nouthex+B-1)/B,B>>>(d_oh.p,H.nouthex,H.nhex,6,H.nvel,d_err);
    if(H.nouttet) audit_boundary_kernel<<<(H.nouttet+B-1)/B,B>>>(d_ot.p,H.nouttet,H.ntet,4,H.nvel,d_err);
    HXT1_CUDA(cudaGetLastError());
    HXT1_CUDA(cudaDeviceSynchronize());
    HXT1_CUDA(cudaMemcpy(&err,d_err,sizeof(err),cudaMemcpyDeviceToHost));
    HXT1_CUDA(cudaFree(d_err));

    std::printf("NODALS_HXT1_DEVICE_AUDIT errors=%llu geometryPlansResident=1 dofPlansResident=1 interfaceMapResident=1 boundaryMapsResident=1 status=%s\n",
      err,err?"FAIL":"PASS");
    if(err) throw std::runtime_error("HXT1 device audit failed");

    std::printf("HXT1_STATUS=PASS\n");
    return 0;
  }catch(const std::exception&e){
    std::fprintf(stderr,"HXT1_ERROR: %s\n",e.what());
    std::fprintf(stderr,"HXT1_STATUS=FAIL\n");
    return 2;
  }
}
