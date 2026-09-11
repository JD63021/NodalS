#pragma once
#include "foam_mesh_g2.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nodals_gpu {

struct CellBPlanHost {
  std::int32_t vel[8];
  double base[12];
  std::int8_t inletOpp=-1;
  double inletSf[3]={0.0,0.0,0.0};
  std::uint8_t wallBasis[8]={0,0,0,0,0,0,0,0};
};
struct PressureSetupHost {
  std::vector<std::int32_t> g2free;
  std::vector<unsigned char> fixed;
  std::vector<CellBPlanHost> cells;
  std::vector<double> rAU;
  std::int32_t free_vel=0,fixed_vertices=0,fixed_faces=0,free_vertices=0,free_faces=0;
  int outlet_patch=-1;
};

inline Vec3d face_outward_area_vector_g2(const SerialTetMesh& M,int f){
  const auto&F=M.faces[(std::size_t)f];
  const Vec3d&a=M.points[(std::size_t)F.v[0]],&b=M.points[(std::size_t)F.v[1]],&c=M.points[(std::size_t)F.v[2]];
  Vec3d u{b.x-a.x,b.y-a.y,b.z-a.z},v{c.x-a.x,c.y-a.y,c.z-a.z};
  Vec3d sf{0.5*(u.y*v.z-u.z*v.y),0.5*(u.z*v.x-u.x*v.z),0.5*(u.x*v.y-u.y*v.x)};
  const int cell=M.owner[(std::size_t)f];
  Vec3d fc{(a.x+b.x+c.x)/3.0,(a.y+b.y+c.y)/3.0,(a.z+b.z+c.z)/3.0},cc{};
  for(int i=0;i<4;++i){const auto&x=M.points[(std::size_t)M.tets[(std::size_t)cell][i]];cc.x+=0.25*x.x;cc.y+=0.25*x.y;cc.z+=0.25*x.z;}
  const double dot=sf.x*(fc.x-cc.x)+sf.y*(fc.y-cc.y)+sf.z*(fc.z-cc.z);
  if(dot<0.0){sf.x=-sf.x;sf.y=-sf.y;sf.z=-sf.z;}
  return sf;
}
inline bool cell_inlet_face_g2(const SerialTetMesh&M,int inlet_patch,int c,int&opp,Vec3d&sf){
  opp=-1;sf={};
  if(inlet_patch<0)return false;
  for(int i=0;i<4;++i){const int f=M.opp_face[(std::size_t)c][i];if(f<(int)M.neighbour.size())continue;if(M.face_patch[(std::size_t)f]!=inlet_patch)continue;if(opp>=0)throw std::runtime_error("tet has multiple DG inlet faces");opp=i;sf=face_outward_area_vector_g2(M,f);}
  return opp>=0;
}

inline PressureSetupHost build_pressure_setup(const SerialTetMesh& M,int outlet_patch,int inlet_patch=-1,bool dg_inlet=false,int wall_patch=-1,bool weak_wall=false){
  PressureSetupHost S;S.outlet_patch=outlet_patch;const std::int32_t nv=(std::int32_t)M.points.size(),nf=(std::int32_t)M.faces.size(),ni=(std::int32_t)M.neighbour.size();S.fixed.assign((std::size_t)nv+nf,0);
  std::vector<unsigned char> wallEntity((std::size_t)nv+nf,0);
  for(std::int32_t f=ni;f<nf;++f){
    const int p=M.face_patch[(std::size_t)f];
    if(weak_wall&&p==wall_patch){wallEntity[(std::size_t)nv+f]=1;for(auto v:M.faces[(std::size_t)f].v)wallEntity[(std::size_t)v]=1;}
    const bool freeBoundary=(p==outlet_patch)||(dg_inlet&&p==inlet_patch)||(weak_wall&&p==wall_patch);
    if(!freeBoundary){S.fixed[(std::size_t)nv+f]=1;for(auto v:M.faces[(std::size_t)f].v)S.fixed[(std::size_t)v]=1;}
  }
  S.g2free.assign(S.fixed.size(),-1);std::int32_t next=0;for(std::int32_t v=0;v<nv;++v){if(S.fixed[v])++S.fixed_vertices;else{S.g2free[v]=next++;++S.free_vertices;}}for(std::int32_t f=0;f<nf;++f){if(S.fixed[(std::size_t)nv+f])++S.fixed_faces;else{S.g2free[(std::size_t)nv+f]=next++;++S.free_faces;}}S.free_vel=next;
  S.cells.resize(M.tets.size());
  for(std::size_t c=0;c<M.tets.size();++c){auto &cp=S.cells[c];for(int i=0;i<4;++i){cp.vel[i]=S.g2free[(std::size_t)M.tets[c][i]];cp.wallBasis[i]=wallEntity[(std::size_t)M.tets[c][i]]?1:0;}for(int i=0;i<4;++i){const std::size_t e=(std::size_t)nv+M.opp_face[c][i];cp.vel[4+i]=S.g2free[e];cp.wallBasis[4+i]=wallEntity[e]?1:0;}auto t=M.tets[c];const Vec3d X[4]={M.points[t[0]],M.points[t[1]],M.points[t[2]],M.points[t[3]]};double J[3][3]={{X[1].x-X[0].x,X[2].x-X[0].x,X[3].x-X[0].x},{X[1].y-X[0].y,X[2].y-X[0].y,X[3].y-X[0].y},{X[1].z-X[0].z,X[2].z-X[0].z,X[3].z-X[0].z}},I[3][3];double det=det3(J);if(!(det>0))throw std::runtime_error("non-positive tet orientation");inv3(J,I);double vol=det/6.0;const double gr[4][3]={{-1,-1,-1},{1,0,0},{0,1,0},{0,0,1}};for(int i=0;i<4;++i)for(int d=0;d<3;++d){double g=0;for(int j=0;j<3;++j)g+=gr[i][j]*I[j][d];cp.base[3*i+d]=vol*g;}if(dg_inlet){int io=-1;Vec3d sf{};if(cell_inlet_face_g2(M,inlet_patch,(int)c,io,sf)){cp.inletOpp=(std::int8_t)io;cp.inletSf[0]=sf.x;cp.inletSf[1]=sf.y;cp.inletSf[2]=sf.z;}}}
  S.rAU.resize((std::size_t)S.free_vel);for(std::int32_t g=0;g<S.free_vel;++g){double q=(double)(g+1);S.rAU[g]=0.85+0.15*(0.5+0.5*std::sin(0.000731*q+0.2*std::cos(0.000113*q)));}
  return S;
}
inline double coeff(const CellBPlanHost& cp,int a,int d){
  if(d<2 && cp.wallBasis[a])return 0.0;
  const int i=(a<4)?a:a-4;
  double v=cp.base[3*i+d]*((a<4)?1.0:-(27.0/20.0));
  if(cp.inletOpp>=0){
    if(a<4 && a!=(int)cp.inletOpp)v-=cp.inletSf[d]/3.0;
    else if(a==4+(int)cp.inletOpp)v-=(9.0/20.0)*cp.inletSf[d];
  }
  return v;
}
inline void cpu_schur_apply(const PressureSetupHost& S,const std::vector<double>& x,std::vector<double>& y){if(x.size()!=S.cells.size())throw std::runtime_error("cpu schur x size");std::vector<double>v0(S.free_vel,0),v1(S.free_vel,0),v2(S.free_vel,0);for(std::size_t c=0;c<S.cells.size();++c){const auto&cp=S.cells[c];double p=x[c];for(int a=0;a<8;++a){int g=cp.vel[a];if(g<0)continue;v0[g]+=coeff(cp,a,0)*p;v1[g]+=coeff(cp,a,1)*p;v2[g]+=coeff(cp,a,2)*p;}}for(int g=0;g<S.free_vel;++g){v0[g]*=S.rAU[g];v1[g]*=S.rAU[g];v2[g]*=S.rAU[g];}y.assign(S.cells.size(),0);for(std::size_t c=0;c<S.cells.size();++c){auto&cp=S.cells[c];double s=0;for(int a=0;a<8;++a){int g=cp.vel[a];if(g<0)continue;s+=coeff(cp,a,0)*v0[g]+coeff(cp,a,1)*v1[g]+coeff(cp,a,2)*v2[g];}y[c]=s;}}
inline std::vector<double> cpu_fine_diag(const PressureSetupHost& S){std::vector<double>d(S.cells.size(),0);for(std::size_t c=0;c<S.cells.size();++c){auto&cp=S.cells[c];double s=0;for(int a=0;a<8;++a){int g=cp.vel[a];if(g<0)continue;for(int k=0;k<3;++k){double b=coeff(cp,a,k);s+=S.rAU[g]*b*b;}}d[c]=s;}return d;}

struct CSRHost {int n=0;std::vector<std::int64_t> row;std::vector<std::int32_t> col;std::vector<double> val,diag;};
inline std::vector<std::vector<std::int32_t>> face_graph(const SerialTetMesh& M){int n=(int)M.tets.size();std::vector<std::vector<std::int32_t>>g(n);for(std::size_t f=0;f<M.neighbour.size();++f){int a=M.owner[f],b=M.neighbour[f];g[a].push_back(b);g[b].push_back(a);}for(auto&v:g){std::sort(v.begin(),v.end());v.erase(std::unique(v.begin(),v.end()),v.end());}return g;}
inline std::vector<std::vector<std::int32_t>> csr_graph(const CSRHost& A){std::vector<std::vector<std::int32_t>>g(A.n);for(int i=0;i<A.n;++i)for(std::int64_t k=A.row[i];k<A.row[i+1];++k){int j=A.col[(std::size_t)k];if(j!=i)g[i].push_back(j);}return g;}
struct AggResult {std::vector<std::int32_t> id;int nagg=0,min_size=0,max_size=0,under=0,over=0;double mean=0;};
inline AggResult aggregate_graph(const std::vector<std::vector<std::int32_t>>& g,int target=16,int minsz=6,int softmax=18){const int n=(int)g.size();std::vector<int>id(n,-1);std::vector<std::vector<int>>m;std::vector<int>q;q.reserve(target);for(int seed=0;seed<n;++seed)if(id[seed]<0){int a=(int)m.size();m.push_back({});q.clear();q.push_back(seed);id[seed]=a;for(std::size_t h=0;h<q.size() && (int)q.size()<target;++h){int i=q[h];for(int j:g[i])if(id[j]<0 && (int)q.size()<target){id[j]=a;q.push_back(j);}}m.back()=q;}
  for(int a=0;a<(int)m.size();++a){if(m[a].empty()||(int)m[a].size()>=minsz)continue;int best=-1,bestsize=INT_MAX;bool bestfit=false;for(int i:m[a])for(int j:g[i]){int b=id[j];if(b<0||b==a||m[b].empty())continue;int combined=(int)m[a].size()+(int)m[b].size();bool fit=combined<=softmax;if(best<0||(fit&&!bestfit)||(fit==bestfit&&(int)m[b].size()<bestsize)){best=b;bestsize=(int)m[b].size();bestfit=fit;}}if(best>=0){for(int i:m[a]){id[i]=best;m[best].push_back(i);}m[a].clear();}}
  std::vector<int>compact(m.size(),-1);int na=0;for(int a=0;a<(int)m.size();++a)if(!m[a].empty())compact[a]=na++;AggResult R;R.id.resize(n);std::vector<int>sz(na,0);for(int i=0;i<n;++i){R.id[i]=compact[id[i]];if(R.id[i]<0)throw std::runtime_error("aggregation lost row");++sz[R.id[i]];}R.nagg=na;R.min_size=na?*std::min_element(sz.begin(),sz.end()):0;R.max_size=na?*std::max_element(sz.begin(),sz.end()):0;R.mean=na?(double)n/na:0;for(int s:sz){if(s<minsz)++R.under;if(s>softmax)++R.over;}return R;}

inline CSRHost maps_to_csr(const std::vector<std::map<int,double>>& rows){CSRHost A;A.n=(int)rows.size();A.row.resize((std::size_t)A.n+1,0);for(int i=0;i<A.n;++i)A.row[i+1]=A.row[i]+(std::int64_t)rows[i].size();A.col.resize((std::size_t)A.row.back());A.val.resize((std::size_t)A.row.back());A.diag.assign(A.n,0);for(int i=0;i<A.n;++i){std::int64_t k=A.row[i];for(auto [j,v]:rows[i]){A.col[(std::size_t)k]=j;A.val[(std::size_t)k]=v;if(j==i)A.diag[i]=v;++k;}if(!(A.diag[i]>0)||!std::isfinite(A.diag[i]))throw std::runtime_error("coarse matrix non-positive diagonal");}return A;}
inline CSRHost build_first_coarse(const PressureSetupHost& S,const AggResult& G){struct C{int cell;double b[3];};std::vector<std::vector<C>>sup((std::size_t)S.free_vel);for(int c=0;c<(int)S.cells.size();++c){auto&cp=S.cells[c];for(int a=0;a<8;++a){int g=cp.vel[a];if(g<0)continue;C z{c,{coeff(cp,a,0),coeff(cp,a,1),coeff(cp,a,2)}};sup[g].push_back(z);}}
  std::vector<std::map<int,double>>rows((std::size_t)G.nagg);for(int g=0;g<S.free_vel;++g){std::map<int,std::array<double,3>>s;for(auto&c:sup[g]){auto &v=s[G.id[c.cell]];for(int d=0;d<3;++d)v[d]+=c.b[d];}for(auto &aa:s)for(auto &bb:s){double v=S.rAU[g]*(aa.second[0]*bb.second[0]+aa.second[1]*bb.second[1]+aa.second[2]*bb.second[2]);rows[aa.first][bb.first]+=v;}}return maps_to_csr(rows);}
inline CSRHost coarsen_csr(const CSRHost&A,const AggResult&G){std::vector<std::map<int,double>>rows((std::size_t)G.nagg);for(int i=0;i<A.n;++i){int ai=G.id[i];for(std::int64_t k=A.row[i];k<A.row[i+1];++k)rows[ai][G.id[A.col[(std::size_t)k]]]+=A.val[(std::size_t)k];}return maps_to_csr(rows);}
inline double csr_symmetry_rel(const CSRHost&A){double d2=0,a2=0;std::vector<std::unordered_map<int,double>>r(A.n);for(int i=0;i<A.n;++i)for(std::int64_t k=A.row[i];k<A.row[i+1];++k)r[i][A.col[(std::size_t)k]]=A.val[(std::size_t)k];for(int i=0;i<A.n;++i)for(auto [j,v]:r[i]){double w=0;auto it=r[j].find(i);if(it!=r[j].end())w=it->second;double d=v-w;d2+=d*d;a2+=v*v;}return std::sqrt(d2)/std::max(std::sqrt(a2),1e-300);}

inline double fine_jacobi_gershgorin_bound(const PressureSetupHost& S){
  std::vector<std::array<double,3>> absSupport((std::size_t)S.free_vel);
  for(auto& q:absSupport) q={{0.0,0.0,0.0}};
  for(const auto& cp:S.cells) for(int a=0;a<8;++a){
    const int g=cp.vel[a]; if(g<0) continue;
    for(int d=0;d<3;++d) absSupport[(std::size_t)g][(std::size_t)d]+=std::abs(coeff(cp,a,d));
  }
  double bound=0.0;
  for(const auto& cp:S.cells){
    double diag=0.0,rowAbsUpper=0.0;
    for(int a=0;a<8;++a){
      const int g=cp.vel[a]; if(g<0) continue;
      for(int d=0;d<3;++d){
        const double b=coeff(cp,a,d);
        diag += S.rAU[(std::size_t)g]*b*b;
        rowAbsUpper += std::abs(b)*S.rAU[(std::size_t)g]*absSupport[(std::size_t)g][(std::size_t)d];
      }
    }
    if(!(diag>0.0) || !std::isfinite(diag)) throw std::runtime_error("invalid fine diagonal in Jacobi bound");
    bound=std::max(bound,rowAbsUpper/diag);
  }
  if(!(bound>0.0) || !std::isfinite(bound)) throw std::runtime_error("invalid fine Jacobi Gershgorin bound");
  return bound;
}
inline double csr_jacobi_gershgorin_bound(const CSRHost& A){
  double bound=0.0;
  for(int i=0;i<A.n;++i){
    double s=0.0;
    for(std::int64_t k=A.row[(std::size_t)i];k<A.row[(std::size_t)i+1];++k) s+=std::abs(A.val[(std::size_t)k]);
    const double d=A.diag[(std::size_t)i];
    if(!(d>0.0) || !std::isfinite(d)) throw std::runtime_error("invalid coarse diagonal in Jacobi bound");
    bound=std::max(bound,s/d);
  }
  if(!(bound>0.0) || !std::isfinite(bound)) throw std::runtime_error("invalid coarse Jacobi Gershgorin bound");
  return bound;
}
inline double safe_jacobi_omega(double bound){
  // Symmetric pre/post weighted Jacobi is SPD when omega*lambda_max(D^-1 A)<2.
  // Use the Gershgorin upper bound with a conservative product target 1.5.
  return std::min(0.80,1.50/bound);
}

struct HierarchyHost {std::vector<AggResult> agg;std::vector<CSRHost> csr;std::vector<double> terminal_inv;int terminal_n=0;};
inline std::vector<double> dense_inverse_cholesky(const CSRHost&A){const int n=A.n;std::vector<double>M((std::size_t)n*n,0),L((std::size_t)n*n,0);for(int i=0;i<n;++i)for(std::int64_t k=A.row[i];k<A.row[i+1];++k)M[(std::size_t)i*n+A.col[(std::size_t)k]]=A.val[(std::size_t)k];double md=0;for(int i=0;i<n;++i)md=std::max(md,std::abs(M[(std::size_t)i*n+i]));for(int i=0;i<n;++i){for(int j=0;j<=i;++j){double s=M[(std::size_t)i*n+j];for(int k=0;k<j;++k)s-=L[(std::size_t)i*n+k]*L[(std::size_t)j*n+k];if(i==j){if(!(s>std::max(1e-30,md*1e-20))||!std::isfinite(s))throw std::runtime_error("terminal Cholesky pivot failure row="+std::to_string(i)+" pivot="+std::to_string(s));L[(std::size_t)i*n+j]=std::sqrt(s);}else L[(std::size_t)i*n+j]=s/L[(std::size_t)j*n+j];}}
  std::vector<double>inv((std::size_t)n*n,0),y(n),x(n);for(int col=0;col<n;++col){for(int i=0;i<n;++i){double s=(i==col)?1.0:0.0;for(int k=0;k<i;++k)s-=L[(std::size_t)i*n+k]*y[k];y[i]=s/L[(std::size_t)i*n+i];}for(int i=n-1;i>=0;--i){double s=y[i];for(int k=i+1;k<n;++k)s-=L[(std::size_t)k*n+i]*x[k];x[i]=s/L[(std::size_t)i*n+i];}for(int i=0;i<n;++i)inv[(std::size_t)i*n+col]=x[i];}for(int i=0;i<n;++i)for(int j=i+1;j<n;++j){double v=0.5*(inv[(std::size_t)i*n+j]+inv[(std::size_t)j*n+i]);inv[(std::size_t)i*n+j]=inv[(std::size_t)j*n+i]=v;}return inv;}
inline HierarchyHost build_hierarchy(const SerialTetMesh&M,const PressureSetupHost&S,int target=16,int minsz=6,int softmax=18,int terminal_target=1000){HierarchyHost H;auto G=aggregate_graph(face_graph(M),target,minsz,softmax);H.agg.push_back(G);H.csr.push_back(build_first_coarse(S,G));while(H.csr.back().n>terminal_target){auto Gc=aggregate_graph(csr_graph(H.csr.back()),target,minsz,softmax);H.agg.push_back(Gc);H.csr.push_back(coarsen_csr(H.csr.back(),Gc));}H.terminal_n=H.csr.back().n;H.terminal_inv=dense_inverse_cholesky(H.csr.back());return H;}

} // namespace nodals_gpu
