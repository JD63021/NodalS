#pragma once
// Stage31: portable-within-build FP64 warm-restart state, NOT a VTU checkpoint.
// Checkpoint must be written at the end of a completed SIMPLE iteration.
// No dependency on CUDA in this header: the caller transfers device arrays.
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <type_traits>

namespace stage31_restart {
inline std::string env_string(const char*key,const std::string& fallback="") {
  const char*p=std::getenv(key);
  return (p && *p)?std::string(p):fallback;
}
inline int cadence(){
  auto s=env_string("STAGE31_CHECKPOINT_EVERY","25");
  std::size_t pos=0;
  int n=std::stoi(s,&pos);
  if(pos!=s.size() || n<0 || n>1000000) throw std::runtime_error("STAGE31_CHECKPOINT_EVERY outside 0..1000000");
  return n;
}
inline std::string fingerprint(){
  auto s=env_string("STAGE31_INPUT_FINGERPRINT");
  if(s.size()!=64)throw std::runtime_error("STAGE31_INPUT_FINGERPRINT missing/not SHA256");
  for(char c:s)if(!((c>='0'&&c<='9')||(c>='a'&&c<='f')))throw std::runtime_error("STAGE31_INPUT_FINGERPRINT invalid hex");
  return s;
}
inline std::uint64_t update_hash(std::uint64_t h,const void*p,std::size_t n){
  const auto*b=static_cast<const unsigned char*>(p);
  for(std::size_t i=0;i<n;++i){h^=b[i];h*=1099511628211ull;}
  return h;
}
struct Header {
  char magic[16]{};
  std::uint32_t format=1;
  std::uint32_t fpBytes=8;
  std::uint64_t nv=0,np=0,nvert=0;
  std::uint64_t iteration=0;
  double prevMeanNut=-1;
  char fingerprint[65]{};
  std::uint64_t dataHash=0;
};
template<class T>struct Fields{
  std::vector<T> u0,u1,u2,p,k,omega;
};
template<class T>void write(const std::string&path,int iteration,int nv,int np,int nvert,
                              double prevMeanNut,const std::string&fp,const Fields<T>&d){
  static_assert(sizeof(T)==8,"Stage31 supports FP64 only");
  if(iteration<1 || nv<=0 || np<=0 || nvert<=0 || nvert>nv || !std::isfinite(prevMeanNut))
     throw std::runtime_error("Stage31 write invalid header numbers");
  if(d.u0.size()!=(std::size_t)nv || d.u1.size()!=(std::size_t)nv ||
     d.u2.size()!=(std::size_t)nv || d.k.size()!=(std::size_t)nv ||
     d.omega.size()!=(std::size_t)nv || d.p.size()!=(std::size_t)np)
     throw std::runtime_error("Stage31 write field sizes differ from solver DOFs");
  Header hdr;
  std::memcpy(hdr.magic,"NODALS_S31_R1",13);
  hdr.nv=(std::uint64_t)nv;hdr.np=(std::uint64_t)np;hdr.nvert=(std::uint64_t)nvert;
  hdr.iteration=(std::uint64_t)iteration;hdr.prevMeanNut=prevMeanNut;
  if(fp.size()!=64)throw std::runtime_error("Stage31 write fingerprint wrong length");
  std::memcpy(hdr.fingerprint,fp.data(),64);
  std::uint64_t hash=14695981039346656037ull;
  auto checksum=[&](const std::vector<T>&v){hash=update_hash(hash,v.data(),v.size()*sizeof(T));};
  checksum(d.u0);checksum(d.u1);checksum(d.u2);checksum(d.p);checksum(d.k);checksum(d.omega);
  hdr.dataHash=hash;
  const std::string tmp=path+".partial";
  {
    std::ofstream f(tmp,std::ios::binary|std::ios::trunc);
    if(!f)throw std::runtime_error("Stage31 cannot open temporary checkpoint: "+tmp);
    f.write(reinterpret_cast<const char*>(&hdr),sizeof(hdr));
    auto append=[&](const std::vector<T>&v){f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T));};
    append(d.u0);append(d.u1);append(d.u2);append(d.p);append(d.k);append(d.omega);
    f.flush();
    if(!f)throw std::runtime_error("Stage31 checkpoint write/flush failed: "+tmp);
  }
  if(std::rename(tmp.c_str(),path.c_str())!=0)
    throw std::runtime_error("Stage31 checkpoint atomic rename failed: "+std::to_string(errno));
}
template<class T>int read(const std::string&path,int nv,int np,int nvert,const std::string&fp,
                         Fields<T>&d,double&prevMeanNut){
  static_assert(sizeof(T)==8,"Stage31 supports FP64 only");
  std::ifstream f(path,std::ios::binary);
  if(!f)throw std::runtime_error("Stage31 checkpoint not found: "+path);
  Header hdr{};
  f.read(reinterpret_cast<char*>(&hdr),sizeof(hdr));
  if(!f || std::memcmp(hdr.magic,"NODALS_S31_R1",13)!=0 || hdr.format!=1 || hdr.fpBytes!=8)
    throw std::runtime_error("Stage31 checkpoint header/format mismatch");
  if(hdr.nv!=(std::uint64_t)nv || hdr.np!=(std::uint64_t)np || hdr.nvert!=(std::uint64_t)nvert)
    throw std::runtime_error("Stage31 checkpoint DOF dimensions mismatch: refusing restart");
  if(std::string(hdr.fingerprint,64)!=fp)
    throw std::runtime_error("Stage31 checkpoint mesh/inlet/physical-parameter SHA256 mismatch");
  if(hdr.iteration<1 || hdr.iteration>100000000 || !std::isfinite(hdr.prevMeanNut))
    throw std::runtime_error("Stage31 checkpoint iteration/state invalid");
  d.u0.resize(nv);d.u1.resize(nv);d.u2.resize(nv);d.p.resize(np);d.k.resize(nv);d.omega.resize(nv);
  auto fetch=[&](std::vector<T>&v){f.read(reinterpret_cast<char*>(v.data()),v.size()*sizeof(T));};
  fetch(d.u0);fetch(d.u1);fetch(d.u2);fetch(d.p);fetch(d.k);fetch(d.omega);
  if(!f || f.peek()!=std::char_traits<char>::eof())
    throw std::runtime_error("Stage31 checkpoint truncated or trailing data");
  std::uint64_t hash=14695981039346656037ull;
  auto checksum=[&](const std::vector<T>&v){hash=update_hash(hash,v.data(),v.size()*sizeof(T));};
  checksum(d.u0);checksum(d.u1);checksum(d.u2);checksum(d.p);checksum(d.k);checksum(d.omega);
  if(hash!=hdr.dataHash)throw std::runtime_error("Stage31 checkpoint payload checksum mismatch");
  auto finite=[&](const std::vector<T>&v){for(T x:v)if(!std::isfinite(x))return false;return true;};
  if(!finite(d.u0)||!finite(d.u1)||!finite(d.u2)||!finite(d.p)||!finite(d.k)||!finite(d.omega))
    throw std::runtime_error("Stage31 checkpoint contains nonfinite values");
  prevMeanNut=hdr.prevMeanNut;
  return (int)hdr.iteration;
}
}
