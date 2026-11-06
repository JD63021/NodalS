#pragma once
// Stage 6.A -- host-only CUDA allocation ledger. No physics/solver code changed.
// This file MUST be included AFTER cuda_runtime.h and BEFORE the cudaMalloc/free
// macros at the top of the solver's single CUDA translation unit.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace stage6_mem {
struct Rec {size_t bytes=0, seq=0; std::string owner; const char* file=nullptr; int line=0;};
inline bool active(){ static const bool yes=[](){const char*p=std::getenv("STAGE6_MEMORY_AUDIT");return p&&std::strcmp(p,"1")==0;}();return yes;}
inline bool verbose(){static const bool yes=[](){const char*p=std::getenv("STAGE6_MEMORY_VERBOSE");return p&&std::strcmp(p,"1")==0;}();return yes;}
struct Ledger {
  std::unordered_map<void*,Rec> alloc;
  size_t live=0, peak=0, seq=0;
  std::string phase="startup";
  std::ofstream csv;
  Ledger(){if(active()){const char*path=std::getenv("STAGE6_MEMORY_CSV");if(path&&*path){csv.open(path,std::ios::out|std::ios::trunc);if(csv)csv<<"event,phase,sequence,pointer,bytes,live_bytes,peak_bytes,owner,source,line\n";}}}
  ~Ledger(){if(active())std::printf("STAGE6_MEM_FINAL trackedPeakMiB=%.6f trackedLiveAtShutdownMiB=%.6f allocationsRemaining=%zu\n",peak/1048576.,live/1048576.,alloc.size());}
  void write(const char*event,void*p,const Rec&r){if(!csv)return;
    csv<<event<<','<<phase<<','<<r.seq<<','<<p<<','<<r.bytes<<','<<live<<','<<peak<<','<<r.owner<<','<<(r.file?r.file:"")<<','<<r.line<<'\n';csv.flush();}
};
inline Ledger& ledger(){static Ledger l;return l;}
inline cudaError_t alloc(void**ptr,size_t bytes,const char*file,int line){
  const cudaError_t err=::cudaMalloc(ptr,bytes);
  if(active() && err==cudaSuccess && ptr && *ptr){
    auto&l=ledger();Rec r{bytes,++l.seq,std::string(file)+":"+std::to_string(line),file,line};
    if(l.alloc.count(*ptr))std::fprintf(stderr,"STAGE6_MEM_ERROR double_allocation ptr=%p\n",*ptr);
    l.alloc[*ptr]=r;l.live+=bytes;l.peak=std::max(l.peak,l.live);l.write("ALLOC",*ptr,r);
    if(verbose())std::printf("STAGE6_MEM_ALLOC ptr=%p owner=%s bytes=%zu liveMiB=%.3f\n",*ptr,r.owner.c_str(),bytes,l.live/1048576.);
  }
  return err;
}
inline cudaError_t free(void*ptr,const char*file,int line){
  if(active()&&ptr){auto&l=ledger();auto it=l.alloc.find(ptr);
    if(it!=l.alloc.end()){const Rec r=it->second;l.live-=r.bytes;l.write("FREE",ptr,r);l.alloc.erase(it);}else{
      std::fprintf(stderr,"STAGE6_MEM_UNTRACKED_FREE ptr=%p source=%s:%d\n",ptr,file,line);
    }}
  return ::cudaFree(ptr);
}
inline void tag(void*ptr,const char*name){if(!active()||!ptr)return;auto&l=ledger();auto it=l.alloc.find(ptr);if(it!=l.alloc.end())it->second.owner=name;else std::printf("STAGE6_MEM_UNTRACKED_TAG name=%s ptr=%p\n",name,ptr);}
inline void phase(const char*name){if(active())ledger().phase=name;}
inline void snapshot(const char*name,int topN=16){
  if(!active())return;
  auto&l=ledger();l.phase=name;
  const cudaError_t sync=::cudaDeviceSynchronize();
  size_t freeBytes=0,totalBytes=0;
  const cudaError_t err=::cudaMemGetInfo(&freeBytes,&totalBytes);
  const double MiB=1048576.;
  std::printf("STAGE6_MEM_PHASE phase=%s trackedLiveMiB=%.6f trackedPeakMiB=%.6f deviceUsedAllProcessesMiB=%.6f deviceTotalMiB=%.6f allocations=%zu cudaSync=%d cudaMemInfo=%d\n",
    name,l.live/MiB,l.peak/MiB,err==cudaSuccess?(totalBytes-freeBytes)/MiB:-1.,err==cudaSuccess?totalBytes/MiB:-1.,l.alloc.size(),(int)sync,(int)err);
  std::vector<std::pair<void*,Rec>> bySize(l.alloc.begin(),l.alloc.end());
  std::sort(bySize.begin(),bySize.end(),[](const auto&a,const auto&b){return a.second.bytes>b.second.bytes;});
  for(int i=0;i<topN&&i<(int)bySize.size();++i){const auto&a=bySize[i];
    std::printf("STAGE6_MEM_TOP phase=%s rank=%d owner=%s ptr=%p bytes=%zu MiB=%.6f source=%s:%d\n",
      name,i+1,a.second.owner.c_str(),a.first,a.second.bytes,a.second.bytes/MiB,a.second.file?a.second.file:"unknown",a.second.line);}
  std::map<std::string,size_t> categories;
  for(const auto&q:l.alloc){std::string k=q.second.owner;auto pos=k.find('.');
    if(pos!=std::string::npos&&k.find(":")!=std::string::npos)k="UNTAGGED_SOURCE_ALLOCATION";
    else if(auto x=k.find('/');x!=std::string::npos)k=k.substr(0,x);
    categories[k]+=q.second.bytes;}
  for(const auto&c:categories)std::printf("STAGE6_MEM_CATEGORY phase=%s name=%s bytes=%zu MiB=%.6f\n",name,c.first.c_str(),c.second,c.second/MiB);
  std::fflush(stdout);
}
} // namespace stage6_mem
