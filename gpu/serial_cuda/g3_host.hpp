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

inline MomentumCSRHost build_static_relaxed_momentum_csr(const SerialTetMesh& M,const MomentumSetupHost& S,double nu=1.0,double alpha_u=0.7) {
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
