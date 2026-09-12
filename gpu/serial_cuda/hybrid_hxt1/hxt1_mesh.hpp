#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace nodals_hxt1 {

inline void cuda_check(cudaError_t e,const char*expr,const char*file,int line){
  if(e!=cudaSuccess) throw std::runtime_error(
    std::string("CUDA ")+expr+" failed at "+file+":"+std::to_string(line)+": "+cudaGetErrorString(e));
}
#define HXT1_CUDA(x) ::nodals_hxt1::cuda_check((x),#x,__FILE__,__LINE__)

struct Header {
  std::uint64_t nv=0,nhex=0,ntet=0,nhexface=0,ntetface=0,nvel=0,npressure=0,ninterface=0;
  std::uint64_t nwallhex=0,ninhex=0,nintet=0,nouthex=0,nouttet=0;
};
struct Point { double x,y,z; };
struct HexConn { std::int32_t v[8]; };
struct TetConn { std::int32_t v[4]; };
struct HexVel { std::int32_t g[14]; };
struct TetVel { std::int32_t g[8]; };
struct HexGeom { double volume,h2,detJ[8],invJ[72]; };
struct TetGeom { double volume,h2,det,invJ[9]; };
struct InterfaceRec {
  std::int32_t hexCell,hexLocalFace,hexBubble;
  std::int32_t tet0,tet0LocalFace,tet0Bubble;
  std::int32_t tet1,tet1LocalFace,tet1Bubble;
};
struct BoundaryRec { std::int32_t cell,localFace,bubble; };

static_assert(sizeof(Point)==3*sizeof(double));
static_assert(sizeof(HexGeom)==82*sizeof(double));
static_assert(sizeof(TetGeom)==12*sizeof(double));
static_assert(sizeof(InterfaceRec)==9*sizeof(std::int32_t));
static_assert(sizeof(BoundaryRec)==3*sizeof(std::int32_t));

template<class T>
inline void read_exact(std::ifstream&f,T&v,const char*what){
  f.read(reinterpret_cast<char*>(&v),sizeof(T));
  if(!f) throw std::runtime_error(std::string("short read: ")+what);
}
template<class T>
inline void read_vec(std::ifstream&f,std::vector<T>&v,std::uint64_t n,const char*what){
  v.resize((std::size_t)n);
  if(n){
    f.read(reinterpret_cast<char*>(v.data()),(std::streamsize)(n*sizeof(T)));
    if(!f) throw std::runtime_error(std::string("short read: ")+what);
  }
}

struct HostMesh {
  Header h;
  std::vector<Point> points;
  std::vector<HexConn> hexes;
  std::vector<TetConn> tets;
  std::vector<HexVel> hexVel;
  std::vector<TetVel> tetVel;
  std::vector<std::int32_t> hexPressure,tetPressure;
  std::vector<HexGeom> hexGeom;
  std::vector<TetGeom> tetGeom;
  std::vector<InterfaceRec> interface;
  std::vector<BoundaryRec> wallHex,inHex,inTet,outHex,outTet;
};

inline HostMesh load(const std::string&path){
  std::ifstream f(path,std::ios::binary);
  if(!f) throw std::runtime_error("cannot open "+path);
  char magic[8]; f.read(magic,8);
  if(!f || std::string(magic,8)!="HXT1BIN1") throw std::runtime_error("bad HXT1 magic");
  std::uint32_t version=0,reserved=0;
  read_exact(f,version,"version"); read_exact(f,reserved,"reserved");
  if(version!=1) throw std::runtime_error("unsupported HXT1 version");
  HostMesh M;
  std::uint64_t* q=&M.h.nv;
  for(int i=0;i<13;++i) read_exact(f,q[i],"header count");

  read_vec(f,M.points,M.h.nv,"points");
  read_vec(f,M.hexes,M.h.nhex,"hexes");
  read_vec(f,M.tets,M.h.ntet,"tets");
  read_vec(f,M.hexVel,M.h.nhex,"hex velocity dofs");
  read_vec(f,M.tetVel,M.h.ntet,"tet velocity dofs");
  read_vec(f,M.hexPressure,M.h.nhex,"hex pressure ids");
  read_vec(f,M.tetPressure,M.h.ntet,"tet pressure ids");
  read_vec(f,M.hexGeom,M.h.nhex,"hex geometry");
  read_vec(f,M.tetGeom,M.h.ntet,"tet geometry");
  read_vec(f,M.interface,M.h.ninterface,"interface");
  read_vec(f,M.wallHex,M.h.nwallhex,"wall hex");
  read_vec(f,M.inHex,M.h.ninhex,"inlet hex");
  read_vec(f,M.inTet,M.h.nintet,"inlet tet");
  read_vec(f,M.outHex,M.h.nouthex,"outlet hex");
  read_vec(f,M.outTet,M.h.nouttet,"outlet tet");
  char extra=0;
  f.read(&extra,1);
  if(f.gcount()!=0) throw std::runtime_error("trailing bytes in HXT1 mesh");
  return M;
}

template<class T>
struct Dev {
  T* p=nullptr; std::size_t n=0;
  Dev()=default;
  explicit Dev(const std::vector<T>&v){upload(v);}
  Dev(const Dev&)=delete; Dev&operator=(const Dev&)=delete;
  ~Dev(){if(p)cudaFree(p);}
  void upload(const std::vector<T>&v){
    n=v.size(); if(!n)return;
    HXT1_CUDA(cudaMalloc((void**)&p,n*sizeof(T)));
    HXT1_CUDA(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));
  }
};

} // namespace
