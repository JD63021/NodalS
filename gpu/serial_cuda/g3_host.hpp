#pragma once
#include "foam_mesh_g2.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <exception>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>

namespace nodals_gpu {

struct MomentumSetupHost {
  std::vector<std::int32_t> g2free;
  std::vector<unsigned char> fixed;
  std::int32_t free_vel=0,fixed_vertices=0,fixed_faces=0,free_vertices=0,free_faces=0;
  int outlet_patch=-1;
};

inline MomentumSetupHost build_momentum_setup(const SerialTetMesh& M,int outlet_patch) {
  MomentumSetupHost S; S.outlet_patch=outlet_patch;
  const std::int32_t nv=(std::int32_t)M.points.size(), nf=(std::int32_t)M.faces.size(), ni=(std::int32_t)M.neighbour.size();
  S.fixed.assign((std::size_t)nv+nf,0);
  // Same serial flow boundary-elimination mask used by G2:
  // all wall/inlet boundary velocity entities are fixed; outlet entities remain free.
  for(std::int32_t f=ni;f<nf;++f) {
    const int p=M.face_patch[(std::size_t)f];
    if(p!=outlet_patch) {
      S.fixed[(std::size_t)nv+f]=1;
      for(auto v:M.faces[(std::size_t)f].v) S.fixed[(std::size_t)v]=1;
    }
  }
  S.g2free.assign(S.fixed.size(),-1);
  std::int32_t next=0;
  for(std::int32_t v=0;v<nv;++v) {
    if(S.fixed[(std::size_t)v]) ++S.fixed_vertices;
    else { S.g2free[(std::size_t)v]=next++; ++S.free_vertices; }
  }
  for(std::int32_t f=0;f<nf;++f) {
    if(S.fixed[(std::size_t)nv+f]) ++S.fixed_faces;
    else { S.g2free[(std::size_t)nv+f]=next++; ++S.free_faces; }
  }
  S.free_vel=next;
  return S;
}

struct Quad5 { std::array<double,4> lam; double w; };
inline std::vector<Quad5> tet_duffy5_g3() {
  const double rn[5]={0.034578939918215090,0.17348032077169567,0.38988638706551931,0.63433347263088680,0.85105421294701644};
  const double rw[5]={0.081764784285771011,0.12619896189991137,0.089200161221590066,0.032055600722961895,0.0041138252030990035};
  const double sn[5]={0.039809857051468722,0.19801341787360821,0.43797481024738616,0.69546427335363614,0.90146491420117358};
  const double sw[5]={0.096781590226651476,0.16717463809436969,0.14638698708466985,0.073908870072616678,0.015747914521692299};
  const double tn[5]={0.046910077030668018,0.23076534494715845,0.50000000000000000,0.76923465505284150,0.95308992296933193};
  const double tw[5]={0.11846344252809449,0.23931433524968326,0.28444444444444450,0.23931433524968326,0.11846344252809449};
  std::vector<Quad5> q; q.reserve(125);
  for(int ir=0;ir<5;++ir) for(int is=0;is<5;++is) for(int it=0;it<5;++it) {
    const double r=rn[ir], ss=sn[is], t=tn[it], omr=1-r, oms=1-ss;
    q.push_back({{omr*oms*(1-t),r,omr*ss,omr*oms*t},rw[ir]*sw[is]*tw[it]});
  }
  return q;
}

inline void basis_grad_ref_g3(const std::array<double,4>& l,double gr[8][3]) {
  const double gl[4][3]={{-1,-1,-1},{1,0,0},{0,1,0},{0,0,1}};
  for(int i=0;i<4;++i) for(int d=0;d<3;++d) gr[i][d]=gl[i][d];
  for(int i=0;i<4;++i) {
    int js[3],k=0; for(int j=0;j<4;++j) if(j!=i) js[k++]=j;
    for(int d=0;d<3;++d) gr[4+i][d]=0.0;
    for(int a=0;a<3;++a) {
      const int j=js[a],o1=js[(a+1)%3],o2=js[(a+2)%3];
      for(int d=0;d<3;++d) gr[4+i][d]+=27.0*l[o1]*l[o2]*gl[j][d];
    }
  }
}

struct DiffusionTensorG3 { double t[8][8][3][3] = {}; };
inline const DiffusionTensorG3& diffusion_tensor_g3() {
  static const DiffusionTensorG3 T=[](){
    DiffusionTensorG3 out;
    const auto Q=tet_duffy5_g3();
    for(const auto& q:Q) {
      double gr[8][3]; basis_grad_ref_g3(q.lam,gr);
      for(int a=0;a<8;++a) for(int b=0;b<8;++b)
        for(int j=0;j<3;++j) for(int k=0;k<3;++k)
          out.t[a][b][j][k]+=gr[a][j]*gr[b][k]*q.w;
    }
    return out;
  }();
  return T;
}

struct MomentumCSRHost {
  int n=0;
  std::vector<std::int64_t> row;
  std::vector<std::int32_t> col;
  std::vector<double> val,diag;
  double nu=1.0,alpha_u=0.7;
  double offdiag_over_diag_mean=0.0,offdiag_over_diag_max=0.0;
};

inline MomentumCSRHost build_static_relaxed_momentum_csr_reference(const SerialTetMesh& M,const MomentumSetupHost& S,double nu=1.0,double alpha_u=0.7) {
  if(!(nu>0.0) || !(alpha_u>0.0 && alpha_u<=1.0)) throw std::runtime_error("invalid G3 nu/alpha_u");
  MomentumCSRHost A; A.n=S.free_vel; A.nu=nu; A.alpha_u=alpha_u;
  const std::int32_t nv=(std::int32_t)M.points.size();
  std::vector<std::vector<std::int32_t>> rows((std::size_t)A.n);
  for(std::size_t c=0;c<M.tets.size();++c) {
    std::int32_t g[8];
    for(int i=0;i<4;++i) g[i]=S.g2free[(std::size_t)M.tets[c][i]];
    for(int i=0;i<4;++i) g[4+i]=S.g2free[(std::size_t)nv+M.opp_face[c][i]];
    for(int a=0;a<8;++a) if(g[a]>=0) {
      auto& rr=rows[(std::size_t)g[a]];
      for(int b=0;b<8;++b) if(g[b]>=0) rr.push_back(g[b]);
    }
  }
  A.row.assign((std::size_t)A.n+1,0);
  for(int i=0;i<A.n;++i) {
    auto& rr=rows[(std::size_t)i];
    std::sort(rr.begin(),rr.end()); rr.erase(std::unique(rr.begin(),rr.end()),rr.end());
    if(!std::binary_search(rr.begin(),rr.end(),i)) throw std::runtime_error("G3 CSR row missing diagonal");
    A.row[(std::size_t)i+1]=A.row[(std::size_t)i]+(std::int64_t)rr.size();
  }
  A.col.resize((std::size_t)A.row.back());
  for(int i=0;i<A.n;++i) {
    auto& rr=rows[(std::size_t)i];
    std::copy(rr.begin(),rr.end(),A.col.begin()+A.row[(std::size_t)i]);
  }
  std::vector<std::vector<std::int32_t>>().swap(rows);
  A.val.assign(A.col.size(),0.0); A.diag.assign((std::size_t)A.n,0.0);
  const auto& T=diffusion_tensor_g3();

  for(std::size_t c=0;c<M.tets.size();++c) {
    std::int32_t g[8];
    for(int i=0;i<4;++i) g[i]=S.g2free[(std::size_t)M.tets[c][i]];
    for(int i=0;i<4;++i) g[4+i]=S.g2free[(std::size_t)nv+M.opp_face[c][i]];
    auto t=M.tets[c];
    const Vec3d X[4]={M.points[(std::size_t)t[0]],M.points[(std::size_t)t[1]],M.points[(std::size_t)t[2]],M.points[(std::size_t)t[3]]};
    double J[3][3]={{X[1].x-X[0].x,X[2].x-X[0].x,X[3].x-X[0].x},
                    {X[1].y-X[0].y,X[2].y-X[0].y,X[3].y-X[0].y},
                    {X[1].z-X[0].z,X[2].z-X[0].z,X[3].z-X[0].z}},I[3][3];
    const double det=det3(J); if(!(det>0.0)) throw std::runtime_error("G3 non-positive tet orientation");
    inv3(J,I);
    double K[8][8]={{0}};
    for(int a=0;a<8;++a) for(int b=0;b<8;++b) {
      double s=0.0;
      for(int j=0;j<3;++j) for(int k=0;k<3;++k) for(int d=0;d<3;++d)
        s += T.t[a][b][j][k]*I[j][d]*I[k][d];
      K[a][b]=nu*det*s;
    }
    for(int a=0;a<8;++a) if(g[a]>=0) {
      const int r=g[a];
      for(int b=0;b<8;++b) if(g[b]>=0) {
        auto first=A.col.begin()+A.row[(std::size_t)r], last=A.col.begin()+A.row[(std::size_t)r+1];
        auto it=std::lower_bound(first,last,g[b]);
        if(it==last || *it!=g[b]) throw std::runtime_error("G3 CSR local support lookup failed");
        A.val[(std::size_t)(it-A.col.begin())]+=K[a][b];
      }
    }
  }
  // SIMPLE momentum under-relaxation: only the physical diagonal is inflated by 1/alpha_u.
  for(int i=0;i<A.n;++i) {
    auto first=A.col.begin()+A.row[(std::size_t)i],last=A.col.begin()+A.row[(std::size_t)i+1];
    auto it=std::lower_bound(first,last,i); if(it==last||*it!=i)throw std::runtime_error("G3 diag lookup failed");
    std::size_t k=(std::size_t)(it-A.col.begin());
    const double phys=A.val[k]; if(!(phys>0.0)||!std::isfinite(phys))throw std::runtime_error("G3 nonpositive diffusion diagonal");
    A.val[k]=phys/alpha_u; A.diag[(std::size_t)i]=A.val[k];
  }
  double sumratio=0.0,maxratio=0.0;
  for(int i=0;i<A.n;++i) {
    double off=0.0;
    for(std::int64_t k=A.row[(std::size_t)i];k<A.row[(std::size_t)i+1];++k)
      if(A.col[(std::size_t)k]!=i) off+=std::abs(A.val[(std::size_t)k]);
    const double q=off/A.diag[(std::size_t)i]; sumratio+=q; maxratio=std::max(maxratio,q);
  }
  A.offdiag_over_diag_mean=A.n?sumratio/A.n:0.0; A.offdiag_over_diag_max=maxratio;
  return A;
}

// -----------------------------------------------------------------------------
// Gate 4A: parallel row-owned momentum CSR construction.
//
// The legacy algorithm pushes cell supports into millions of std::vectors,
// sorts each row, then assembles diffusion by a serial cell loop.  This path
// preserves the same final sorted CSR and the same cell-order floating-point
// accumulation while assigning complete output rows to CPU workers.
//
// Reverse incidence is filled in ascending cell order.  Therefore, for any
// output row, contributions from incident cells are accumulated in exactly the
// same cell order as the reference cell-major loop.
// -----------------------------------------------------------------------------

inline double g4a_elapsed(
    const std::chrono::steady_clock::time_point&a,
    const std::chrono::steady_clock::time_point&b){
  return std::chrono::duration<double>(b-a).count();
}

inline double g4a_status_mib(const char*key){
  FILE*f=std::fopen("/proc/self/status","r");
  if(!f)return -1.0;
  char line[256];double out=-1.0;
  const std::size_t n=std::strlen(key);
  while(std::fgets(line,sizeof(line),f)){
    if(std::strncmp(line,key,n)==0){
      unsigned long long kb=0;
      if(std::sscanf(line+n,": %llu kB",&kb)==1)out=(double)kb/1024.0;
      break;
    }
  }
  std::fclose(f);
  return out;
}

inline int g4a_threads(){
  int n=0;
  if(const char*e=std::getenv("NODALS_G4_CSR_THREADS"))n=std::atoi(e);
  if(n<=0){unsigned h=std::thread::hardware_concurrency();n=h?std::min((int)h,16):1;}
  return std::max(1,std::min(n,64));
}

template<class F>
inline void g4a_parallel_chunks(int n,int nth,int chunk,F&&fn){
  if(n<=0)return;
  nth=std::max(1,std::min(nth,n));
  std::atomic<int> next{0};
  std::atomic<bool> failed{false};
  std::exception_ptr ep;
  std::mutex em;
  std::vector<std::thread> pool;
  pool.reserve((std::size_t)nth);
  for(int t=0;t<nth;++t){
    pool.emplace_back([&,t](){
      try{
        while(!failed.load(std::memory_order_relaxed)){
          const int b=next.fetch_add(chunk,std::memory_order_relaxed);
          if(b>=n)break;
          fn(t,b,std::min(n,b+chunk));
        }
      }catch(...){
        failed.store(true,std::memory_order_relaxed);
        std::lock_guard<std::mutex>g(em);
        if(!ep)ep=std::current_exception();
      }
    });
  }
  for(auto&th:pool)th.join();
  if(ep)std::rethrow_exception(ep);
}

struct G4AMomInc {
  std::int32_t cell;
  std::uint8_t local;
};

struct G4AGeom {
  double det;
  double invJ[9];
};

inline std::int32_t g4a_cell_gid(
    const SerialTetMesh&M,const MomentumSetupHost&S,
    std::size_t c,int a,std::int32_t nv){
  if(a<4)return S.g2free[(std::size_t)M.tets[c][a]];
  return S.g2free[(std::size_t)nv+M.opp_face[c][a-4]];
}

inline MomentumCSRHost build_static_relaxed_momentum_csr_parallel(
    const SerialTetMesh&M,const MomentumSetupHost&S,
    double nu=1.0,double alpha_u=0.7){
  using clock=std::chrono::steady_clock;
  if(!(nu>0.0)||!(alpha_u>0.0&&alpha_u<=1.0))
    throw std::runtime_error("invalid G4A nu/alpha_u");

  const auto all0=clock::now();
  const int nth=g4a_threads();
  const std::int32_t nv=(std::int32_t)M.points.size();
  const int nc=(int)M.tets.size();

  MomentumCSRHost A;
  A.n=S.free_vel;A.nu=nu;A.alpha_u=alpha_u;

  // Reverse free-row -> (cell,local-a) incidence.
  const auto i0=clock::now();
  std::vector<std::int64_t> iptr((std::size_t)A.n+1,0);
  for(int c=0;c<nc;++c)
    for(int a=0;a<8;++a){
      const int g=g4a_cell_gid(M,S,(std::size_t)c,a,nv);
      if(g>=0)++iptr[(std::size_t)g+1];
    }
  for(int r=0;r<A.n;++r)iptr[(std::size_t)r+1]+=iptr[(std::size_t)r];

  std::vector<G4AMomInc> inc((std::size_t)iptr.back());
  auto cur=iptr;
  for(int c=0;c<nc;++c)
    for(int a=0;a<8;++a){
      const int g=g4a_cell_gid(M,S,(std::size_t)c,a,nv);
      if(g>=0){
        const auto z=cur[(std::size_t)g]++;
        inc[(std::size_t)z]={(std::int32_t)c,(std::uint8_t)a};
      }
    }
  std::vector<std::int64_t>().swap(cur);
  const auto i1=clock::now();

  std::printf(
    "NODALS_GPU_G4A_STAGE stage=incidence status=PASS entries=%zu seconds=%.6f threads=%d rssMiB=%.3f hwmMiB=%.3f\n",
    inc.size(),g4a_elapsed(i0,i1),nth,
    g4a_status_mib("VmRSS"),g4a_status_mib("VmHWM"));
  std::fflush(stdout);

  // Row-owned topology.
  const auto t0=clock::now();
  std::vector<std::vector<std::int32_t>> rows((std::size_t)A.n);
  g4a_parallel_chunks(A.n,nth,256,[&](int,int rb,int re){
    for(int r=rb;r<re;++r){
      auto&rr=rows[(std::size_t)r];
      const std::size_t ninc=(std::size_t)(
        iptr[(std::size_t)r+1]-iptr[(std::size_t)r]);
      rr.reserve(ninc*8);
      for(std::int64_t z=iptr[(std::size_t)r];z<iptr[(std::size_t)r+1];++z){
        const std::size_t c=(std::size_t)inc[(std::size_t)z].cell;
        for(int b=0;b<8;++b){
          const int g=g4a_cell_gid(M,S,c,b,nv);
          if(g>=0)rr.push_back((std::int32_t)g);
        }
      }
      std::sort(rr.begin(),rr.end());
      rr.erase(std::unique(rr.begin(),rr.end()),rr.end());
      if(!std::binary_search(rr.begin(),rr.end(),r))
        throw std::runtime_error("G4A CSR row missing diagonal");
    }
  });

  A.row.assign((std::size_t)A.n+1,0);
  for(int r=0;r<A.n;++r)
    A.row[(std::size_t)r+1]=A.row[(std::size_t)r]+
                            (std::int64_t)rows[(std::size_t)r].size();

  A.col.resize((std::size_t)A.row.back());
  g4a_parallel_chunks(A.n,nth,512,[&](int,int rb,int re){
    for(int r=rb;r<re;++r)
      std::copy(rows[(std::size_t)r].begin(),rows[(std::size_t)r].end(),
                A.col.begin()+A.row[(std::size_t)r]);
  });
  std::vector<std::vector<std::int32_t>>().swap(rows);
  const auto t1=clock::now();

  std::printf(
    "NODALS_GPU_G4A_STAGE stage=topology status=PASS rows=%d nnz=%zu avgRow=%.6f seconds=%.6f rssMiB=%.3f hwmMiB=%.3f\n",
    A.n,A.col.size(),A.n?(double)A.col.size()/A.n:0.0,
    g4a_elapsed(t0,t1),g4a_status_mib("VmRSS"),g4a_status_mib("VmHWM"));
  std::fflush(stdout);

  // Geometry once per cell, in parallel.
  const auto g0=clock::now();
  std::vector<G4AGeom> geom((std::size_t)nc);
  g4a_parallel_chunks(nc,nth,512,[&](int,int cb,int ce){
    for(int c=cb;c<ce;++c){
      auto tet=M.tets[(std::size_t)c];
      const Vec3d X[4]={
        M.points[(std::size_t)tet[0]],M.points[(std::size_t)tet[1]],
        M.points[(std::size_t)tet[2]],M.points[(std::size_t)tet[3]]};
      double J[3][3]={
        {X[1].x-X[0].x,X[2].x-X[0].x,X[3].x-X[0].x},
        {X[1].y-X[0].y,X[2].y-X[0].y,X[3].y-X[0].y},
        {X[1].z-X[0].z,X[2].z-X[0].z,X[3].z-X[0].z}};
      double I[3][3];
      const double det=det3(J);
      if(!(det>0.0))throw std::runtime_error("G4A non-positive tet orientation");
      inv3(J,I);
      auto&gg=geom[(std::size_t)c];
      gg.det=det;
      for(int j=0;j<3;++j)for(int d=0;d<3;++d)
        gg.invJ[3*j+d]=I[j][d];
    }
  });
  const auto g1=clock::now();

  std::printf(
    "NODALS_GPU_G4A_STAGE stage=geometry status=PASS cells=%d seconds=%.6f scratchMiB=%.3f rssMiB=%.3f hwmMiB=%.3f\n",
    nc,g4a_elapsed(g0,g1),
    (double)(geom.size()*sizeof(G4AGeom))/(1024.0*1024.0),
    g4a_status_mib("VmRSS"),g4a_status_mib("VmHWM"));
  std::fflush(stdout);

  // Row-owned numeric assembly. Incident cells are in ascending cell order.
  const auto n0=clock::now();
  A.val.assign(A.col.size(),0.0);
  A.diag.assign((std::size_t)A.n,0.0);
  const auto&T=diffusion_tensor_g3();

  g4a_parallel_chunks(A.n,nth,64,[&](int,int rb,int re){
    for(int r=rb;r<re;++r){
      for(std::int64_t z=iptr[(std::size_t)r];z<iptr[(std::size_t)r+1];++z){
        const auto&ii=inc[(std::size_t)z];
        const std::size_t c=(std::size_t)ii.cell;
        const int a=(int)ii.local;
        const auto&gg=geom[c];

        for(int b=0;b<8;++b){
          const int gb=g4a_cell_gid(M,S,c,b,nv);
          if(gb<0)continue;

          double v=0.0;
          for(int j=0;j<3;++j)
            for(int k=0;k<3;++k)
              for(int d=0;d<3;++d)
                v+=T.t[a][b][j][k]*
                   gg.invJ[3*j+d]*gg.invJ[3*k+d];
          v*=nu*gg.det;

          auto first=A.col.begin()+A.row[(std::size_t)r];
          auto last =A.col.begin()+A.row[(std::size_t)r+1];
          auto it=std::lower_bound(first,last,gb);
          if(it==last||*it!=gb)
            throw std::runtime_error("G4A CSR local support lookup failed");
          A.val[(std::size_t)(it-A.col.begin())]+=v;
        }
      }
    }
  });
  const auto n1=clock::now();

  std::vector<G4AGeom>().swap(geom);
  std::vector<G4AMomInc>().swap(inc);
  std::vector<std::int64_t>().swap(iptr);

  // Same relaxation/final diagnostics as reference.
  const auto f0=clock::now();
  g4a_parallel_chunks(A.n,nth,512,[&](int,int rb,int re){
    for(int i=rb;i<re;++i){
      auto first=A.col.begin()+A.row[(std::size_t)i];
      auto last=A.col.begin()+A.row[(std::size_t)i+1];
      auto it=std::lower_bound(first,last,i);
      if(it==last||*it!=i)throw std::runtime_error("G4A diag lookup failed");
      const std::size_t k=(std::size_t)(it-A.col.begin());
      const double phys=A.val[k];
      if(!(phys>0.0)||!std::isfinite(phys))
        throw std::runtime_error("G4A nonpositive diffusion diagonal");
      A.val[k]=phys/alpha_u;
      A.diag[(std::size_t)i]=A.val[k];
    }
  });

  double sumratio=0.0,maxratio=0.0;
  for(int i=0;i<A.n;++i){
    double off=0.0;
    for(std::int64_t k=A.row[(std::size_t)i];k<A.row[(std::size_t)i+1];++k)
      if(A.col[(std::size_t)k]!=i)off+=std::abs(A.val[(std::size_t)k]);
    const double q=off/A.diag[(std::size_t)i];
    sumratio+=q;maxratio=std::max(maxratio,q);
  }
  A.offdiag_over_diag_mean=A.n?sumratio/A.n:0.0;
  A.offdiag_over_diag_max=maxratio;
  const auto f1=clock::now();
  const auto all1=clock::now();

  std::printf(
    "NODALS_GPU_G4A_MOM_CSR_NEW status=PASS rows=%d nnz=%zu threads=%d incidenceSeconds=%.6f topologySeconds=%.6f geometrySeconds=%.6f numericSeconds=%.6f finalizeSeconds=%.6f totalSeconds=%.6f rssMiB=%.3f hwmMiB=%.3f\n",
    A.n,A.val.size(),nth,
    g4a_elapsed(i0,i1),g4a_elapsed(t0,t1),g4a_elapsed(g0,g1),
    g4a_elapsed(n0,n1),g4a_elapsed(f0,f1),g4a_elapsed(all0,all1),
    g4a_status_mib("VmRSS"),g4a_status_mib("VmHWM"));

  return A;
}

inline void g4a_cpu_spmv3(
    const MomentumCSRHost&A,
    const std::vector<double>&x0,
    const std::vector<double>&x1,
    const std::vector<double>&x2,
    std::vector<double>&y0,
    std::vector<double>&y1,
    std::vector<double>&y2){
  if((int)x0.size()!=A.n||(int)x1.size()!=A.n||(int)x2.size()!=A.n)
    throw std::runtime_error("G4A cpu_spmv3 size");
  y0.assign((std::size_t)A.n,0.0);
  y1.assign((std::size_t)A.n,0.0);
  y2.assign((std::size_t)A.n,0.0);
  for(int i=0;i<A.n;++i){
    double a0=0.0,a1=0.0,a2=0.0;
    for(std::int64_t k=A.row[(std::size_t)i];k<A.row[(std::size_t)i+1];++k){
      const int j=A.col[(std::size_t)k];
      const double v=A.val[(std::size_t)k];
      a0+=v*x0[(std::size_t)j];
      a1+=v*x1[(std::size_t)j];
      a2+=v*x2[(std::size_t)j];
    }
    y0[(std::size_t)i]=a0;
    y1[(std::size_t)i]=a1;
    y2[(std::size_t)i]=a2;
  }
}

inline void g4a_validate_momentum_csr(
    const MomentumCSRHost&R,const MomentumCSRHost&N,
    double refSeconds,double newSeconds){
  const bool dims=(R.n==N.n);
  const bool rowExact=dims&&R.row==N.row;
  const bool colExact=rowExact&&R.col==N.col;
  const bool valBitwise=colExact&&R.val==N.val;
  const bool diagBitwise=dims&&R.diag==N.diag;

  long double d2=0.0L,r2=0.0L,dd2=0.0L,dr2=0.0L;
  double maxAbs=0.0,diagMax=0.0;
  if(colExact&&R.val.size()==N.val.size()){
    for(std::size_t k=0;k<R.val.size();++k){
      const double d=N.val[k]-R.val[k];
      d2+=(long double)d*d;r2+=(long double)R.val[k]*R.val[k];
      maxAbs=std::max(maxAbs,std::abs(d));
    }
    for(std::size_t i=0;i<R.diag.size();++i){
      const double d=N.diag[i]-R.diag[i];
      dd2+=(long double)d*d;dr2+=(long double)R.diag[i]*R.diag[i];
      diagMax=std::max(diagMax,std::abs(d));
    }
  }else{
    d2=dd2=std::numeric_limits<long double>::infinity();
  }

  const double valRel=std::sqrt((double)d2)/
                      std::max(std::sqrt((double)r2),1e-300);
  const double diagRel=std::sqrt((double)dd2)/
                       std::max(std::sqrt((double)dr2),1e-300);

  double actionRel=std::numeric_limits<double>::infinity();
  if(colExact){
    std::vector<double>x0((std::size_t)R.n),x1((std::size_t)R.n),x2((std::size_t)R.n);
    for(int i=0;i<R.n;++i){
      const double q=(double)(i+1);
      x0[(std::size_t)i]=std::sin(.0031*q);
      x1[(std::size_t)i]=std::cos(.0023*q);
      x2[(std::size_t)i]=std::sin(.0017*q)+.2*std::cos(.0009*q);
    }
    std::vector<double>r0,r1,r2v,n0,n1,n2;
    g4a_cpu_spmv3(R,x0,x1,x2,r0,r1,r2v);
    g4a_cpu_spmv3(N,x0,x1,x2,n0,n1,n2);
    long double ad=0.0L,ar=0.0L;
    for(int i=0;i<R.n;++i){
      const double e0=n0[(std::size_t)i]-r0[(std::size_t)i];
      const double e1=n1[(std::size_t)i]-r1[(std::size_t)i];
      const double e2=n2[(std::size_t)i]-r2v[(std::size_t)i];
      ad+=(long double)e0*e0+(long double)e1*e1+(long double)e2*e2;
      ar+=(long double)r0[(std::size_t)i]*r0[(std::size_t)i]+
          (long double)r1[(std::size_t)i]*r1[(std::size_t)i]+
          (long double)r2v[(std::size_t)i]*r2v[(std::size_t)i];
    }
    actionRel=std::sqrt((double)ad)/
              std::max(std::sqrt((double)ar),1e-300);
  }

  const bool pass=dims&&rowExact&&colExact&&
                  std::isfinite(valRel)&&valRel<=5e-14&&
                  diagRel<=5e-14&&actionRel<=5e-14;

  std::printf(
    "NODALS_GPU_G4A_MOM_CSR_PARITY status=%s dimsExact=%d rowsExact=%d colsExact=%d valuesBitwiseExact=%d diagBitwiseExact=%d refRows=%d newRows=%d refNnz=%zu newNnz=%zu valueRelL2=%.12e valueMaxAbs=%.12e diagRelL2=%.12e diagMaxAbs=%.12e actionRelL2=%.12e refSeconds=%.6f newSeconds=%.6f speedup=%.6f\n",
    pass?"PASS":"FAIL",(int)dims,(int)rowExact,(int)colExact,
    (int)valBitwise,(int)diagBitwise,R.n,N.n,R.val.size(),N.val.size(),
    valRel,maxAbs,diagRel,diagMax,actionRel,
    refSeconds,newSeconds,refSeconds/std::max(newSeconds,1e-300));

  if(!pass)throw std::runtime_error("Gate4A momentum CSR parity failed");
}

inline MomentumCSRHost build_static_relaxed_momentum_csr(
    const SerialTetMesh&M,const MomentumSetupHost&S,
    double nu=1.0,double alpha_u=0.7){
  const char*e=std::getenv("NODALS_G4_CSR_MODE");
  const char*mode=(e&&*e)?e:"parallel";

  if(std::strcmp(mode,"reference")==0)
    return build_static_relaxed_momentum_csr_reference(M,S,nu,alpha_u);

  if(std::strcmp(mode,"parallel")==0)
    return build_static_relaxed_momentum_csr_parallel(M,S,nu,alpha_u);

  if(std::strcmp(mode,"compare")==0){
    const auto t0=std::chrono::steady_clock::now();
    auto R=build_static_relaxed_momentum_csr_reference(M,S,nu,alpha_u);
    const auto t1=std::chrono::steady_clock::now();

    std::printf(
      "NODALS_GPU_G4A_MOM_CSR_REFERENCE_DONE seconds=%.6f rows=%d nnz=%zu rssMiB=%.3f hwmMiB=%.3f\n",
      g4a_elapsed(t0,t1),R.n,R.val.size(),
      g4a_status_mib("VmRSS"),g4a_status_mib("VmHWM"));

    const auto t2=std::chrono::steady_clock::now();
    auto N=build_static_relaxed_momentum_csr_parallel(M,S,nu,alpha_u);
    const auto t3=std::chrono::steady_clock::now();

    g4a_validate_momentum_csr(R,N,g4a_elapsed(t0,t1),g4a_elapsed(t2,t3));
    return N;
  }

  throw std::runtime_error(
    "NODALS_G4_CSR_MODE must be reference, compare, or parallel");
}


inline void cpu_spmv3(const MomentumCSRHost& A,
                      const std::vector<double>& x0,const std::vector<double>& x1,const std::vector<double>& x2,
                      std::vector<double>& y0,std::vector<double>& y1,std::vector<double>& y2) {
  if((int)x0.size()!=A.n||(int)x1.size()!=A.n||(int)x2.size()!=A.n)throw std::runtime_error("G3 cpu_spmv3 size");
  y0.assign(A.n,0.0);y1.assign(A.n,0.0);y2.assign(A.n,0.0);
  for(int i=0;i<A.n;++i) {
    double a0=0,a1=0,a2=0;
    for(std::int64_t k=A.row[(std::size_t)i];k<A.row[(std::size_t)i+1];++k) {
      const int j=A.col[(std::size_t)k]; const double v=A.val[(std::size_t)k];
      a0+=v*x0[(std::size_t)j]; a1+=v*x1[(std::size_t)j]; a2+=v*x2[(std::size_t)j];
    }
    y0[(std::size_t)i]=a0;y1[(std::size_t)i]=a1;y2[(std::size_t)i]=a2;
  }
}

struct ColoringHost {
  int ncolors=0,max_degree=0,min_size=0,max_size=0;
  double mean_size=0.0;
  std::vector<std::int32_t> color,offset,rows;
};

inline ColoringHost greedy_csr_coloring(const MomentumCSRHost& A) {
  ColoringHost C; C.color.assign((std::size_t)A.n,-1);
  std::vector<int> mark(64,-1);
  int maxc=-1;
  for(int i=0;i<A.n;++i) {
    int deg=0;
    for(std::int64_t k=A.row[(std::size_t)i];k<A.row[(std::size_t)i+1];++k) {
      int j=A.col[(std::size_t)k]; if(j==i)continue; ++deg;
      if(j<i) {
        int c=C.color[(std::size_t)j];
        if(c>=0) {
          if(c>=(int)mark.size()) mark.resize((std::size_t)c+32,-1);
          mark[(std::size_t)c]=i;
        }
      }
    }
    C.max_degree=std::max(C.max_degree,deg);
    int c=0; while(c<(int)mark.size() && mark[(std::size_t)c]==i)++c;
    if(c==(int)mark.size())mark.resize(mark.size()+32,-1);
    C.color[(std::size_t)i]=c; maxc=std::max(maxc,c);
  }
  C.ncolors=maxc+1;
  std::vector<int> sz((std::size_t)C.ncolors,0);
  for(int c:C.color)++sz[(std::size_t)c];
  C.offset.assign((std::size_t)C.ncolors+1,0);
  for(int c=0;c<C.ncolors;++c)C.offset[(std::size_t)c+1]=C.offset[(std::size_t)c]+sz[(std::size_t)c];
  C.rows.resize((std::size_t)A.n); std::vector<int> next(C.offset.begin(),C.offset.end()-1);
  for(int i=0;i<A.n;++i)C.rows[(std::size_t)next[(std::size_t)C.color[(std::size_t)i]]++]=i;
  C.min_size=C.ncolors?*std::min_element(sz.begin(),sz.end()):0;
  C.max_size=C.ncolors?*std::max_element(sz.begin(),sz.end()):0;
  C.mean_size=C.ncolors?(double)A.n/C.ncolors:0.0;
  // Exact edge audit.
  for(int i=0;i<A.n;++i) for(std::int64_t k=A.row[(std::size_t)i];k<A.row[(std::size_t)i+1];++k) {
    int j=A.col[(std::size_t)k];
    if(j!=i && C.color[(std::size_t)i]==C.color[(std::size_t)j])
      throw std::runtime_error("G3 graph coloring conflict");
  }
  return C;
}

inline double vec_rel_l2(const std::vector<double>& a,const std::vector<double>& b) {
  if(a.size()!=b.size())throw std::runtime_error("G3 vec size");
  long double d=0,r=0;
  for(std::size_t i=0;i<a.size();++i){long double q=(long double)a[i]-b[i];d+=q*q;r+=(long double)b[i]*b[i];}
  return std::sqrt((double)d)/std::max(std::sqrt((double)r),1e-300);
}
inline double vec_scaled_inf(const std::vector<double>& a,const std::vector<double>& b) {
  if(a.size()!=b.size())throw std::runtime_error("G3 vec size");
  double e=0,s=1.0;for(std::size_t i=0;i<a.size();++i){e=std::max(e,std::abs(a[i]-b[i]));s=std::max({s,std::abs(a[i]),std::abs(b[i])});}return e/s;
}
inline double vec_norm2(const std::vector<double>& a) {
  long double s=0;for(double v:a)s+=(long double)v*v;return std::sqrt((double)s);
}

} // namespace nodals_gpu
