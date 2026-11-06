#pragma once
// Standalone versioned binary checkpoint for mixed momentum P1+BF3/P0,
// scalar CG Bernstein P2. Never reinterpret Stage31 R1 scalar face coefficients.
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace nodals { namespace p2 {
inline std::uint64_t hash64(std::uint64_t h,const void*p,std::size_t n){
 const auto*b=static_cast<const unsigned char*>(p);
 for(std::size_t i=0;i<n;++i){h^=b[i];h*=1099511628211ull;}return h;
}
struct CheckpointHeader{
 char magic[16]{};
 std::uint32_t format=2,realBytes=8;
 std::uint64_t velocityDofs=0,pressureDofs=0,vertices=0,scalarP2Dofs=0;
 std::uint64_t iteration=0;
 double prevMeanNut=0;
 char inputSha256[65]{},connectivitySha256[65]{};
 std::uint64_t payloadHash=0;
};
template<class T>struct FieldsP2{
 std::vector<T> u0,u1,u2,pressure,kP2,omegaP2;
};
inline void assert_sha(const std::string&h){
 if(h.size()!=64)throw std::runtime_error("P2 checkpoint: SHA256 length not 64");
 for(char c:h)if(!((c>='0'&&c<='9')||(c>='a'&&c<='f')))
   throw std::runtime_error("P2 checkpoint: invalid SHA256 string");
}
template<class T>inline void check_shapes(const CheckpointHeader&h,const FieldsP2<T>&f){
 if(sizeof(T)!=8||h.velocityDofs==0||h.pressureDofs==0||h.vertices==0||
    h.scalarP2Dofs<=h.vertices || h.vertices>h.velocityDofs ||
    f.u0.size()!=h.velocityDofs||f.u1.size()!=h.velocityDofs||
    f.u2.size()!=h.velocityDofs||f.pressure.size()!=h.pressureDofs||
    f.kP2.size()!=h.scalarP2Dofs||f.omegaP2.size()!=h.scalarP2Dofs)
   throw std::runtime_error("P2 checkpoint: inconsistent dimensions/fields");
}
template<class T>void write_p2(const std::string&path,CheckpointHeader h,const FieldsP2<T>&f){
 assert_sha(std::string(h.inputSha256,64));
 assert_sha(std::string(h.connectivitySha256,64));
 if(h.iteration==0||!std::isfinite(h.prevMeanNut))throw std::runtime_error("P2 checkpoint invalid iteration/nut");
 check_shapes(h,f);
 std::memcpy(h.magic,"NODALS_P2_SST2",14);
 h.payloadHash=14695981039346656037ull;
 auto update=[&](const std::vector<T>&x){h.payloadHash=hash64(h.payloadHash,x.data(),x.size()*sizeof(T));};
 update(f.u0);update(f.u1);update(f.u2);update(f.pressure);update(f.kP2);update(f.omegaP2);
 auto write=[&](std::ofstream&o,const std::vector<T>&x){o.write(reinterpret_cast<const char*>(x.data()),x.size()*sizeof(T));};
 const std::string tmp=path+".partial";
 {std::ofstream o(tmp,std::ios::binary|std::ios::trunc);
  if(!o)throw std::runtime_error("P2 checkpoint cannot create file");
  o.write(reinterpret_cast<const char*>(&h),sizeof(h));
  write(o,f.u0);write(o,f.u1);write(o,f.u2);write(o,f.pressure);write(o,f.kP2);write(o,f.omegaP2);
  o.flush();if(!o)throw std::runtime_error("P2 checkpoint write error");}
 if(std::rename(tmp.c_str(),path.c_str())!=0)throw std::runtime_error("P2 checkpoint rename failed");
}
template<class T>CheckpointHeader read_p2(const std::string&path,
   std::uint64_t nv,std::uint64_t np,std::uint64_t nvert,std::uint64_t np2,
   const std::string&inputSha,const std::string&connectivitySha,FieldsP2<T>&f){
 std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("P2 checkpoint missing");
 CheckpointHeader h{};in.read(reinterpret_cast<char*>(&h),sizeof(h));
 if(!in||std::memcmp(h.magic,"NODALS_P2_SST2",14)!=0||h.format!=2||h.realBytes!=8)
   throw std::runtime_error("P2 checkpoint incompatible version/magic");
 if(h.velocityDofs!=nv||h.pressureDofs!=np||h.vertices!=nvert||h.scalarP2Dofs!=np2)
   throw std::runtime_error("P2 checkpoint DOF dimension mismatch");
 if(std::string(h.inputSha256,64)!=inputSha||
    std::string(h.connectivitySha256,64)!=connectivitySha)
   throw std::runtime_error("P2 checkpoint mesh/inlet/edge graph fingerprint mismatch");
 if(h.iteration==0||!std::isfinite(h.prevMeanNut))throw std::runtime_error("P2 checkpoint invalid metadata");
 f.u0.resize(nv);f.u1.resize(nv);f.u2.resize(nv);f.pressure.resize(np);
 f.kP2.resize(np2);f.omegaP2.resize(np2);
 auto rd=[&](std::vector<T>&x){in.read(reinterpret_cast<char*>(x.data()),x.size()*sizeof(T));};
 rd(f.u0);rd(f.u1);rd(f.u2);rd(f.pressure);rd(f.kP2);rd(f.omegaP2);
 if(!in||in.peek()!=std::char_traits<char>::eof())throw std::runtime_error("P2 checkpoint size/trailing bytes mismatch");
 std::uint64_t checksum=14695981039346656037ull;
 auto up=[&](const std::vector<T>&x){checksum=hash64(checksum,x.data(),x.size()*sizeof(T));
    for(T v:x)if(!std::isfinite(v))throw std::runtime_error("P2 checkpoint non-finite coefficient");};
 up(f.u0);up(f.u1);up(f.u2);up(f.pressure);up(f.kP2);up(f.omegaP2);
 if(checksum!=h.payloadHash)throw std::runtime_error("P2 checkpoint payload corruption");
 return h;
}
}} // namespace nodals::p2
