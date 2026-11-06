
#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include <map>
#include "modules/wall/portable/spalding_core.hpp"
#include "modules/wall/portable/tet_ray_trace.hpp"

#include "modules/wall/portable/spalding_vector_core.hpp"
#include "modules/wall/portable/spalding_active.hpp"
// Stage24B: shared runtime selection; default 5/point3 retains frozen physics.
__device__ __constant__ int stage24b_active_points=5;
// Stage27 experiment: CUDA switch symbol is defined exactly once in spalding_active.hpp.
__device__ __constant__ int stage24b_omega_slot=2;
// Stage 2.B.1 standalone compact ray-cache layout: default legacy identity.
__device__ __constant__ int stage27_s02b_stride=5;
__device__ __constant__ int stage27_s02b_mapping[5]={0,1,2,3,4};
__device__ __constant__ double stage27_s02b_heights[5]={0.,0.,0.,0.,0.};
#include "modules/wall/portable/stage22c_wall_split.hpp"
#include "modules/boundary/stage23_vector_inlet.hpp"
namespace hxt5b {

using Real=hxt4b::Real;
using nodals_hxt1::Dev;
using nodals_hxt1::HostMesh;

static constexpr double GX4[4]={
  -0.8611363115940525752,-0.3399810435848562648,
   0.3399810435848562648, 0.8611363115940525752};
static constexpr double GW4[4]={
  0.3478548451374538574,0.6521451548625461426,
  0.6521451548625461426,0.3478548451374538574};

__device__ __constant__ double QL0[12]={
  0.063089014491502,0.063089014491502,0.873821971016996,
  0.249286745170910,0.249286745170910,0.501426509658180,
  0.053145049844816,0.053145049844816,0.310352451033785,
  0.310352451033785,0.636502499121399,0.636502499121399};
__device__ __constant__ double QL1[12]={
  0.063089014491502,0.873821971016996,0.063089014491502,
  0.249286745170910,0.501426509658180,0.249286745170910,
  0.310352451033785,0.636502499121399,0.053145049844816,
  0.636502499121399,0.053145049844816,0.310352451033785};
__device__ __constant__ double QL2[12]={
  0.873821971016996,0.063089014491502,0.063089014491502,
  0.501426509658180,0.249286745170910,0.249286745170910,
  0.636502499121399,0.310352451033785,0.636502499121399,
  0.053145049844816,0.310352451033785,0.053145049844816};
__device__ __constant__ double QW12[12]={
  0.050844906370207,0.050844906370207,0.050844906370207,
  0.116786275726379,0.116786275726379,0.116786275726379,
  0.082851075618374,0.082851075618374,0.082851075618374,
  0.082851075618374,0.082851075618374,0.082851075618374};

static constexpr double HL0[12]={
  0.063089014491502,0.063089014491502,0.873821971016996,
  0.249286745170910,0.249286745170910,0.501426509658180,
  0.053145049844816,0.053145049844816,0.310352451033785,
  0.310352451033785,0.636502499121399,0.636502499121399};
static constexpr double HL1[12]={
  0.063089014491502,0.873821971016996,0.063089014491502,
  0.249286745170910,0.501426509658180,0.249286745170910,
  0.310352451033785,0.636502499121399,0.053145049844816,
  0.636502499121399,0.053145049844816,0.310352451033785};
static constexpr double HL2[12]={
  0.873821971016996,0.063089014491502,0.063089014491502,
  0.501426509658180,0.249286745170910,0.249286745170910,
  0.636502499121399,0.310352451033785,0.636502499121399,
  0.053145049844816,0.310352451033785,0.053145049844816};
static constexpr double HW12[12]={
  0.050844906370207,0.050844906370207,0.050844906370207,
  0.116786275726379,0.116786275726379,0.116786275726379,
  0.082851075618374,0.082851075618374,0.082851075618374,
  0.082851075618374,0.082851075618374,0.082851075618374};

struct V3{double x=0,y=0,z=0;};
inline V3 vadd(V3 a,V3 b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
inline V3 vsub(V3 a,V3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
inline V3 vmul(double s,V3 a){return {s*a.x,s*a.y,s*a.z};}
inline V3 vcross(V3 a,V3 b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
inline double vdot(V3 a,V3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
inline double vnorm(V3 a){return std::sqrt(vdot(a,a));}
inline V3 vp(const nodals_hxt1::Point&p){return {p.x,p.y,p.z};}

struct HexFacePlan{
  std::int32_t cell=0;
  std::int8_t lf=0,fd=0,d0=1,d1=2;
  double fv=0;
  double n[3]={0,0,0};
};

// Prescribed physical wall-model matching point.
struct WallMatchPoint{
  std::int32_t cell=-1;       // diagnostic only
  std::int32_t g[14]={0};     // precomputed global velocity DOFs
  double b[14]={0};           // Q1+BF2 basis at match point
  double y=0;
};

#include "modules/wall/tet_wall_types.inc"
#include "modules/wall/stage27_s02b_ray_accessor.inc"

// ============================================================
// PERF_WALL_HEX_PERSISTENT_CACHE
//
// Static data for one wall face / quadrature point.
//
// Built once:
//   wall basis
//   sample basis
//   surface quadrature measure
//   wall -> sample distance
//
// Runtime remains:
//   sample U
//   Spalding solve
//   beta
//   wall matrix/stats
// ============================================================
struct WallHexQuadCache{
  double bw[14];
  double bs[14];
  double w=0;
  double y=0;
};


inline void href(int lf,int&fd,double&fv,int&d0,int&d1){
  fd=-1;
  for(int d=0;d<3;++d){
    int s=hxt2::HEX_SIGN[hxt2::HEX_FACE[lf][0]][d];bool same=true;
    for(int k=1;k<4;++k) if(hxt2::HEX_SIGN[hxt2::HEX_FACE[lf][k]][d]!=s) same=false;
    if(same){fd=d;fv=(double)s;break;}
  }
  if(fd<0)throw std::runtime_error("HXT5B HEX face reference identification failed");
  int q[2],m=0;for(int d=0;d<3;++d)if(d!=fd)q[m++]=d;d0=q[0];d1=q[1];
}
inline void hbasis(double x,double y,double z,double ph[14]){
  double c[3]={x,y,z};
  for(int a=0;a<8;++a){double q=1;for(int d=0;d<3;++d)q*=.5*(1+hxt2::HEX_SIGN[a][d]*c[d]);ph[a]=q;}
  for(int f=0;f<6;++f){int fd,d0,d1;double fv;href(f,fd,fv,d0,d1);ph[8+f]=.5*(1+fv*c[fd])*(1-c[d0]*c[d0])*(1-c[d1]*c[d1]);}
}
inline void tbasis(const double l[4],double ph[8]){
  for(int a=0;a<4;++a)ph[a]=l[a];
  for(int i=0;i<4;++i){double q=27;for(int j=0;j<4;++j)if(j!=i)q*=l[j];ph[4+i]=q;}
}
inline void q1host(const HostMesh&M,const nodals_hxt1::HexConn&h,double x,double y,double z,V3&X,V3 dX[3]){
  double c[3]={x,y,z};X={};dX[0]=dX[1]=dX[2]={};
  for(int a=0;a<8;++a){
    double f[3],df[3];for(int d=0;d<3;++d){double s=hxt2::HEX_SIGN[a][d];f[d]=.5*(1+s*c[d]);df[d]=.5*s;}
    double N=f[0]*f[1]*f[2],Nd[3]={df[0]*f[1]*f[2],f[0]*df[1]*f[2],f[0]*f[1]*df[2]};V3 P=vp(M.points[h.v[a]]);
    X=vadd(X,vmul(N,P));for(int d=0;d<3;++d)dX[d]=vadd(dX[d],vmul(Nd[d],P));
  }
}
inline HexFacePlan hexplan(const HostMesh&M,const nodals_hxt1::BoundaryRec&r){
  HexFacePlan P;P.cell=r.cell;
  int fd=0,d0=0,d1=0;double fv=0;
  href(r.localFace,fd,fv,d0,d1);
  if(fd<0||fd>2||d0<0||d0>2||d1<0||d1>2)
    throw std::runtime_error("HXT5B invalid HEX reference-face direction");
  P.lf=(std::int8_t)r.localFace;
  P.fd=(std::int8_t)fd;P.fv=fv;
  P.d0=(std::int8_t)d0;P.d1=(std::int8_t)d1;
  const auto&h=M.hexes[r.cell];V3 q[4];for(int k=0;k<4;++k)q[k]=vp(M.points[h.v[hxt2::HEX_FACE[r.localFace][k]]]);
  V3 av=vmul(.5,vadd(vcross(vsub(q[1],q[0]),vsub(q[2],q[0])),vcross(vsub(q[2],q[0]),vsub(q[3],q[0]))));
  V3 fc{};for(int k=0;k<4;++k)fc=vadd(fc,vmul(.25,q[k]));V3 cc{};for(int a=0;a<8;++a)cc=vadd(cc,vmul(.125,vp(M.points[h.v[a]])));
  if(vdot(av,vsub(fc,cc))<0)av=vmul(-1,av);double A=vnorm(av);if(!(A>0))throw std::runtime_error("HXT5B HEX boundary area invalid");
  av=vmul(1/A,av);P.n[0]=av.x;P.n[1]=av.y;P.n[2]=av.z;return P;
}


// ---------------------------------------------------------------------------
// Prescribed matching-height cache.
// X_m = X_wall - y_match * n_out.
// The containing HEX is located once on the host through conforming HEX-face
// adjacency plus a Q1 inverse map.
// ---------------------------------------------------------------------------
inline bool wall_match_solve_cols(
    V3 c0,V3 c1,V3 c2,V3 r,double d[3])
{
  const V3 c12=vcross(c1,c2),cR2=vcross(r,c2),c1R=vcross(c1,r);
  const double det=vdot(c0,c12);
  const double sc=std::max({vnorm(c0),vnorm(c1),vnorm(c2),1e-300});
  if(!std::isfinite(det)||std::fabs(det)<1e-14*sc*sc*sc)return false;
  d[0]=vdot(r,c12)/det;
  d[1]=vdot(c0,cR2)/det;
  d[2]=vdot(c0,c1R)/det;
  return std::isfinite(d[0])&&std::isfinite(d[1])&&std::isfinite(d[2]);
}

inline bool wall_match_hex_ref_newton(
    const nodals_hxt1::HostMesh&M,int c,const V3&P,
    const double seed[3],double r[3])
{
  const auto H=M.hexes[(std::size_t)c];

  double lo[3]={1e300,1e300,1e300},hi[3]={-1e300,-1e300,-1e300};
  for(int a=0;a<8;++a){
    const auto Q=M.points[(std::size_t)H.v[a]];
    const double x[3]={Q.x,Q.y,Q.z};
    for(int d=0;d<3;++d){lo[d]=std::min(lo[d],x[d]);hi[d]=std::max(hi[d],x[d]);}
  }
  const double sc=std::sqrt(
      (hi[0]-lo[0])*(hi[0]-lo[0])+
      (hi[1]-lo[1])*(hi[1]-lo[1])+
      (hi[2]-lo[2])*(hi[2]-lo[2]));
  const double tol=1e-10*std::max(sc,1e-12);

  r[0]=seed[0];r[1]=seed[1];r[2]=seed[2];

  for(int it=0;it<40;++it){
    V3 X,dX[3];q1host(M,H,r[0],r[1],r[2],X,dX);
    const V3 e=vsub(P,X);
    if(vnorm(e)<=tol)break;

    double q[3];
    if(!wall_match_solve_cols(dX[0],dX[1],dX[2],e,q))return false;

    double a=1.0;
    const double qm=std::max({std::fabs(q[0]),std::fabs(q[1]),std::fabs(q[2])});
    if(qm>0.75)a=0.75/qm;

    r[0]+=a*q[0];r[1]+=a*q[1];r[2]+=a*q[2];

    if(std::fabs(r[0])>3.0||std::fabs(r[1])>3.0||std::fabs(r[2])>3.0)
      return false;
  }

  V3 X,dX[3];q1host(M,H,r[0],r[1],r[2],X,dX);
  return vnorm(vsub(P,X))<=10*tol &&
         r[0]>=-1-1e-8&&r[0]<=1+1e-8 &&
         r[1]>=-1-1e-8&&r[1]<=1+1e-8 &&
         r[2]>=-1-1e-8&&r[2]<=1+1e-8;
}

inline bool wall_match_hex_ref(
    const nodals_hxt1::HostMesh&M,int c,const V3&P,double r[3])
{
  const auto H=M.hexes[(std::size_t)c];

  double lo[3]={1e300,1e300,1e300},hi[3]={-1e300,-1e300,-1e300};
  for(int a=0;a<8;++a){
    const auto Q=M.points[(std::size_t)H.v[a]];
    const double x[3]={Q.x,Q.y,Q.z};
    for(int d=0;d<3;++d){lo[d]=std::min(lo[d],x[d]);hi[d]=std::max(hi[d],x[d]);}
  }

  const double pp[3]={P.x,P.y,P.z};
  double seed[3];
  for(int d=0;d<3;++d){
    const double m=.5*(lo[d]+hi[d]),h=.5*(hi[d]-lo[d]);
    seed[d]=h>0?(pp[d]-m)/h:0.0;
    seed[d]=std::max(-1.0,std::min(1.0,seed[d]));
  }
  if(wall_match_hex_ref_newton(M,c,P,seed,r))return true;

  const double center[3]={0.0,0.0,0.0};
  if(wall_match_hex_ref_newton(M,c,P,center,r))return true;

  static constexpr double S=0.55;
  for(int sx=-1;sx<=1;sx+=2)
    for(int sy=-1;sy<=1;sy+=2)
      for(int sz=-1;sz<=1;sz+=2){
        const double qseed[3]={S*sx,S*sy,S*sz};
        if(wall_match_hex_ref_newton(M,c,P,qseed,r))return true;
      }
  return false;
}

// Rare-path inverse map for highly distorted trilinear HEX8 cells.
// Used only after the fast wall-seeded BFS fails and the candidate HEX
// has passed a physical-space AABB test.
inline bool wall_match_hex_ref_dense_fallback(
    const nodals_hxt1::HostMesh&M,int c,const V3&P,double r[3])
{
  if(wall_match_hex_ref(M,c,P,r))return true;

  static constexpr double S[7]={
    -0.95,-0.67,-0.33,0.0,0.33,0.67,0.95
  };

  const auto H=M.hexes[(std::size_t)c];

  // Pick the closest point on a coarse 7^3 reference grid as the first
  // Newton seed.
  double bestSeed[3]={0.0,0.0,0.0};
  double bestD2=1e300;
  for(int i=0;i<7;++i)
    for(int j=0;j<7;++j)
      for(int k=0;k<7;++k){
        V3 X,dX[3];
        q1host(M,H,S[i],S[j],S[k],X,dX);
        const double dx=P.x-X.x;
        const double dy=P.y-X.y;
        const double dz=P.z-X.z;
        const double d2=dx*dx+dy*dy+dz*dz;
        if(d2<bestD2){
          bestD2=d2;
          bestSeed[0]=S[i];
          bestSeed[1]=S[j];
          bestSeed[2]=S[k];
        }
      }

  if(wall_match_hex_ref_newton(M,c,P,bestSeed,r))return true;

  // Exhaust the same dense grid only on this rare path.
  for(int i=0;i<7;++i)
    for(int j=0;j<7;++j)
      for(int k=0;k<7;++k){
        const double seed[3]={S[i],S[j],S[k]};
        if(wall_match_hex_ref_newton(M,c,P,seed,r))return true;
      }

  return false;
}

inline std::array<std::int32_t,4> wall_match_face_key(
    const nodals_hxt1::HostMesh&M,int c,int localFace)
{
  // Use the solver's host-side reference HEX face table directly.
  // This avoids any assumption about local-face ordering and avoids
  // reading CUDA __constant__ data from host code.
  std::array<std::int32_t,4> k{{-1,-1,-1,-1}};
  const auto H=M.hexes[(std::size_t)c];
  for(int j=0;j<4;++j)
    k[(std::size_t)j]=H.v[hxt2::HEX_FACE[localFace][j]];
  std::sort(k.begin(),k.end());
  return k;
}

inline std::vector<WallMatchPoint> build_wall_match_height_cache(
    const nodals_hxt1::HostMesh&M,
    const std::vector<HexFacePlan>&wallH,
    double yMatch)
{
  if(!(yMatch>0.0))throw std::runtime_error("wall match y must be positive");
  if(M.h.ntet)throw std::runtime_error("prescribed match-height cache currently requires all-HEX mesh");
  // Host copy of the four Gauss-Legendre wall-face abscissae.
  static constexpr double gx4[4]={
    -8.61136311594052575e-01,
    -3.39981043584856265e-01,
     3.39981043584856265e-01,
     8.61136311594052575e-01
  };

  const int nh=(int)M.h.nhex;
  constexpr std::size_t WALL_MATCH_MAX_SEARCH_CELLS=4096;
  std::vector<std::array<int,6>> nb((std::size_t)nh);
  for(auto&a:nb)a.fill(-1);

  using Key=std::array<std::int32_t,4>;
  std::map<Key,std::pair<int,int>> owner;
  for(int c=0;c<nh;++c){
    for(int lf=0;lf<6;++lf){
      const Key k=wall_match_face_key(M,c,lf);
      auto it=owner.find(k);
      if(it==owner.end())owner.emplace(k,std::make_pair(c,lf));
      else{
        const int c0=it->second.first,lf0=it->second.second;
        nb[(std::size_t)c][(std::size_t)lf]=c0;
        nb[(std::size_t)c0][(std::size_t)lf0]=c;
      }
    }
  }

  std::vector<WallMatchPoint> out(wallH.size()*16);
  std::size_t sameCell=0,crossCell=0;
  int maxVisited=0;
  std::size_t globalFallbackPoints=0;
  std::size_t globalFallbackCandidates=0;


  for(std::size_t fi=0;fi<wallH.size();++fi){
    const auto&f=wallH[fi];
    for(int q=0;q<16;++q){
      const int i=q>>2,j=q&3;
      double cw[3]={0,0,0};
      cw[f.fd]=f.fv;
      cw[f.d0]=gx4[i];
      cw[f.d1]=gx4[j];

      V3 Xw,dX[3];q1host(M,M.hexes[(std::size_t)f.cell],cw[0],cw[1],cw[2],Xw,dX);
      V3 cr=vcross(dX[f.d0],dX[f.d1]);
      if(cr.x*f.n[0]+cr.y*f.n[1]+cr.z*f.n[2]<0.0){cr.x=-cr.x;cr.y=-cr.y;cr.z=-cr.z;}
      const double cn=vnorm(cr);
      if(!(cn>0.0))throw std::runtime_error("wall match zero wall Jacobian");
      const V3 n{cr.x/cn,cr.y/cn,cr.z/cn};
      const V3 P{Xw.x-yMatch*n.x,Xw.y-yMatch*n.y,Xw.z-yMatch*n.z};

      std::vector<int> todo;
      todo.reserve(256);
      todo.push_back(f.cell);
      std::vector<unsigned char> seen((std::size_t)nh,0);
      seen[(std::size_t)f.cell]=1;

      int found=-1;double rr[3]={0,0,0};

      // Best seed for the wall-adjacent HEX: the exact reference point
      // from which the physical matching point was constructed.
      double rtWall[3];
      if(wall_match_hex_ref_newton(M,f.cell,P,cw,rtWall)){
        found=f.cell;
        rr[0]=rtWall[0];rr[1]=rtWall[1];rr[2]=rtWall[2];
      }

      std::size_t head=0;
      while(found<0 &&
            head<todo.size() &&
            head<WALL_MATCH_MAX_SEARCH_CELLS){
        const int c=todo[head++];
        double rt[3];
        if(wall_match_hex_ref(M,c,P,rt)){
          found=c;rr[0]=rt[0];rr[1]=rt[1];rr[2]=rt[2];break;
        }
        for(int lf=0;lf<6;++lf){
          const int cnbr=nb[(std::size_t)c][(std::size_t)lf];
          if(cnbr<0)continue;
          if(!seen[(std::size_t)cnbr]){
            seen[(std::size_t)cnbr]=1;
            todo.push_back(cnbr);
          }
        }
      }
      maxVisited=std::max(maxVisited,(int)head);

      // WALL_MATCH_GLOBAL_AABB_FALLBACK
      // The wall-seeded BFS is deliberately capped for the common fast path.
      // On highly irregular all-HEX meshes a valid containing cell can lie
      // outside that topological search ball even though the physical point is
      // perfectly valid.  For those rare points only, scan the remaining HEXes,
      // reject impossible cells by their physical AABB, and then apply the same
      // robust Q1 inverse map.  This removes the arbitrary search-cap failure
      // without changing the wall-model sample position or physics.
      if(found<0){
        ++globalFallbackPoints;
        for(int c=0;c<nh && found<0;++c){
          if(seen[(std::size_t)c])continue;

          const auto H=M.hexes[(std::size_t)c];
          double lo[3]={1e300,1e300,1e300};
          double hi[3]={-1e300,-1e300,-1e300};
          for(int aa=0;aa<8;++aa){
            const auto Q=M.points[(std::size_t)H.v[aa]];
            const double xx[3]={Q.x,Q.y,Q.z};
            for(int d=0;d<3;++d){
              lo[d]=std::min(lo[d],xx[d]);
              hi[d]=std::max(hi[d],xx[d]);
            }
          }

          const double dx=hi[0]-lo[0];
          const double dy=hi[1]-lo[1];
          const double dz=hi[2]-lo[2];
          const double pad=1e-10*std::max({dx,dy,dz,1e-12});

          if(P.x<lo[0]-pad || P.x>hi[0]+pad ||
             P.y<lo[1]-pad || P.y>hi[1]+pad ||
             P.z<lo[2]-pad || P.z>hi[2]+pad)
            continue;

          ++globalFallbackCandidates;
          double rt[3];
          if(wall_match_hex_ref_dense_fallback(M,c,P,rt)){
            found=c;
            rr[0]=rt[0];rr[1]=rt[1];rr[2]=rt[2];
            break;
          }
        }
      }

      if(found<0){
        throw std::runtime_error(
          "wall match point location failed face="+std::to_string(fi)+
          " q="+std::to_string(q)+
          " wallCell="+std::to_string(f.cell)+
          " y="+std::to_string(yMatch)+
          " Xw=["+std::to_string(Xw.x)+","+std::to_string(Xw.y)+","+std::to_string(Xw.z)+"]"+
          " P=["+std::to_string(P.x)+","+std::to_string(P.y)+","+std::to_string(P.z)+"]"+
          " searched="+std::to_string(head)+
          " queued="+std::to_string(todo.size())+
          " searchCap="+std::to_string(WALL_MATCH_MAX_SEARCH_CELLS));
      }

      auto&mp=out[fi*16+(std::size_t)q];
      mp.cell=found;
      mp.y=yMatch;
      hbasis(rr[0],rr[1],rr[2],mp.b);
      const auto HV=M.hexVel[(std::size_t)found];
      for(int a=0;a<14;++a){
        mp.g[a]=HV.g[a];
        if(mp.g[a]<0)
          throw std::runtime_error("wall match cache encountered negative velocity DOF");
        if(!std::isfinite(mp.b[a]))
          throw std::runtime_error("wall match cache encountered nonfinite basis value");
      }
      if(found==f.cell)++sameCell;else ++crossCell;
    }
  }

  std::printf(
    "NODALS_SST_MATCH_HEIGHT_CACHE targetY=%.12e wallFaces=%zu points=%zu "
    "sameCell=%zu crossCell=%zu maxSearchCells=%d searchCap=%zu geometry=INWARD_PHYSICAL_WALL_NORMAL "
    "locator=WALL_SEEDED_Q1_NEWTON_PLUS_DEEP_HEX_FACE_BFS_MULTISTART status=PASS\n",
    yMatch,wallH.size(),out.size(),sameCell,crossCell,maxVisited,
    (std::size_t)WALL_MATCH_MAX_SEARCH_CELLS);
  std::printf(
    "NODALS_SST_MATCH_HEIGHT_GLOBAL_FALLBACK points=%zu candidates=%zu "
    "policy=FAST_BFS_THEN_GLOBAL_AABB_Q1_INVERSE status=PASS\n",
    globalFallbackPoints,globalFallbackCandidates);
  return out;
}

inline std::vector<WallHexQuadCache> build_wall_hex_cache(
    const HostMesh&M,
    const std::vector<HexFacePlan>&F,
    double sf)
{
  if(!(sf>0.0&&sf<1.0))
    throw std::runtime_error(
      "HXT5B wall cache bad sample fraction");

  std::vector<WallHexQuadCache> C(F.size()*16ull);

  for(std::size_t fi=0;fi<F.size();++fi){

    const auto&f=F[fi];
    const auto&h=M.hexes[(std::size_t)f.cell];

    for(int q=0;q<16;++q){

      const int i=q>>2;
      const int j=q&3;

      double cw[3]={0,0,0};
      cw[f.fd]=f.fv;
      cw[f.d0]=GX4[i];
      cw[f.d1]=GX4[j];

      double cs[3]={cw[0],cw[1],cw[2]};
      cs[f.fd]=f.fv+sf*(-2.0*f.fv);

      auto&Q=C[fi*16ull+(std::size_t)q];

      // Same Q1+BF2 basis values previously evaluated in wall_hex.
      hbasis(cw[0],cw[1],cw[2],Q.bw);
      hbasis(cs[0],cs[1],cs[2],Q.bs);

      V3 Xw,dXw[3],Xs,dXs[3];

      q1host(M,h,cw[0],cw[1],cw[2],Xw,dXw);
      q1host(M,h,cs[0],cs[1],cs[2],Xs,dXs);

      const double rr=std::hypot(Xw.x,Xw.y);

      if(!(rr>0.0))
        throw std::runtime_error(
          "HXT5B wall cache invalid radius");

      const V3 nr{
        Xw.x/rr,
        Xw.y/rr,
        0.0
      };

      // Orientation does not affect |cross|.  Original kernel
      // flipped this vector before taking its magnitude.
      const V3 cr=vcross(
        dXw[f.d0],
        dXw[f.d1]);

      const double jac=vnorm(cr);

      const double y=
        vdot(
          vsub(Xw,Xs),
          nr);

      if(!(jac>0.0) ||
         !(y>0.0) ||
         !std::isfinite(jac) ||
         !std::isfinite(y))
        throw std::runtime_error(
          "HXT5B wall cache invalid sample geometry");

      Q.w=GW4[i]*GW4[j]*jac;
      Q.y=y;
    }
  }

  const double mib=
    (double)(C.size()*sizeof(WallHexQuadCache))/
    (1024.0*1024.0);

  std::printf(
    "NODALS_HXT5B_WALL_CACHE "
    "faces=%zu quadrature=16 entries=%zu MiB=%.3f "
    "geometry=PERSISTENT basis=PERSISTENT "
    "runtime=SPALDING_NUMERIC_ONLY status=PASS\n",
    F.size(),C.size(),mib);

  return C;
}


#include "modules/wall/tet_wall_geometry.inc"

struct BoundaryHost{
  hxt3a::BCSR B;
  std::vector<double> csrc;
  std::vector<unsigned char> wallMask;
  std::vector<HexFacePlan> inH,wallH;
  std::vector<TetFacePlan> inT,wallT;
  double inletArea=0,sourceSum=0,constResidual=0;
};
inline void repack(hxt3a::BCSR&A){
  for(auto&r:A.rows)hxt3a::cleanup(r);A.row.assign((size_t)A.np+1,0);
  for(int i=0;i<A.np;++i)A.row[i+1]=A.row[i]+(std::int64_t)A.rows[i].size();
  size_t nnz=(size_t)A.row.back();A.col.resize(nnz);A.bx.resize(nnz);A.by.resize(nnz);A.bz.resize(nnz);
  for(int i=0;i<A.np;++i){size_t k=A.row[i];for(auto&kv:A.rows[i]){A.col[k]=kv.first;A.bx[k]=kv.second.x;A.by[k]=kv.second.y;A.bz[k]=kv.second.z;++k;}}
}
#include "modules/wall/stage22d3_wall_b_host.inc"
inline BoundaryHost build_boundaries(const HostMesh&M,const hxt3a::BCSR&B0,nodals::stage23::Vec3 inletU){
  BoundaryHost H;H.B=B0;H.csrc.assign(B0.np,0);H.wallMask.assign(B0.nv,0);int nh=M.h.nhex;
  for(auto&r:M.inHex){
    auto P=hexplan(M,r);H.inH.push_back(P);std::array<V3,14>I{};V3 Avec{};const auto&hc=M.hexes[r.cell];
    for(int i=0;i<4;++i)for(int j=0;j<4;++j){double c[3]={0,0,0};c[P.fd]=P.fv;c[P.d0]=GX4[i];c[P.d1]=GX4[j];V3 X,dX[3];q1host(M,hc,c[0],c[1],c[2],X,dX);
      V3 av=vcross(dX[P.d0],dX[P.d1]);V3 n{P.n[0],P.n[1],P.n[2]};if(vdot(av,n)<0)av=vmul(-1,av);V3 nds=vmul(GW4[i]*GW4[j],av);Avec=vadd(Avec,nds);
      double ph[14];hbasis(c[0],c[1],c[2],ph);for(int a=0;a<14;++a)I[a]=vadd(I[a],vmul(ph[a],nds));}
    auto&row=H.B.rows[r.cell];auto&v=M.hexVel[r.cell];for(int a=0;a<14;++a)hxt3a::add(row,v.g[a],{-I[a].x,-I[a].y,-I[a].z});
    const double Q=inletU.x*Avec.x+inletU.y*Avec.y+inletU.z*Avec.z;
    H.csrc[r.cell]+=Q;H.inletArea+=vnorm(Avec);H.sourceSum+=Q;
  }
  for(auto&r:M.inTet){
    auto P=tetplan(M,r);H.inT.push_back(P);std::array<double,8>I{};int a0=hxt2::TET_FACE[r.localFace][0],a1=hxt2::TET_FACE[r.localFace][1],a2=hxt2::TET_FACE[r.localFace][2];
    for(int q=0;q<12;++q){double l[4]={0,0,0,0};l[a0]=HL0[q];l[a1]=HL1[q];l[a2]=HL2[q];double ph[8];tbasis(l,ph);for(int a=0;a<8;++a)I[a]+=HW12[q]*P.area*ph[a];}
    auto&row=H.B.rows[nh+r.cell];auto&v=M.tetVel[r.cell];for(int a=0;a<8;++a)hxt3a::add(row,v.g[a],{-I[a]*P.n[0],-I[a]*P.n[1],-I[a]*P.n[2]});
    const double Q=P.area*(inletU.x*P.n[0]+inletU.y*P.n[1]+inletU.z*P.n[2]);
    H.csrc[nh+r.cell]+=Q;H.inletArea+=P.area;H.sourceSum+=Q;
  }
  for(auto&r:M.wallHex){H.wallH.push_back(hexplan(M,r));auto&h=M.hexes[r.cell];H.wallMask[r.bubble]=1;for(int k=0;k<4;++k)H.wallMask[h.v[hxt2::HEX_FACE[r.localFace][k]]]=1;}
  // PURETET_PROFILE5_WALL: classify exterior faces by topological incidence.
  // HXT1BIN1 stores TET inlet/outlet but no explicit TET wall list. This is
  // intentionally limited to all-TET meshes (no HEX/TET interface ambiguity).
  if(M.h.nhex==0 && M.h.ntet>0){
    using FaceKey=std::array<std::int32_t,3>;
    std::map<FaceKey,std::vector<std::pair<int,int>>> owners;
    auto facekey=[&](int c,int lf){FaceKey k{};const auto &T=M.tets[(std::size_t)c];
      for(int j=0;j<3;++j)k[(std::size_t)j]=T.v[hxt2::TET_FACE[lf][j]];
      std::sort(k.begin(),k.end());return k;};
    for(int c=0;c<(int)M.h.ntet;++c)for(int lf=0;lf<4;++lf)
      owners[facekey(c,lf)].push_back({c,lf});
    std::map<FaceKey,int> tagged;
    for(const auto&r:M.inTet)++tagged[facekey(r.cell,r.localFace)];
    for(const auto&r:M.outTet)++tagged[facekey(r.cell,r.localFace)];
    for(auto&entry:owners){
      const auto&v=entry.second;
      if(v.size()>2)throw std::runtime_error("pure-tet nonmanifold face");
      auto it=tagged.find(entry.first);
      if(v.size()==2){if(it!=tagged.end())throw std::runtime_error("tagged interior tet face");continue;}
      if(it!=tagged.end()){
        if(it->second!=1)throw std::runtime_error("duplicate tet boundary label");
        continue;
      }
      const int c=v[0].first,lf=v[0].second;
      nodals_hxt1::BoundaryRec r{c,lf,M.tetVel[(std::size_t)c].g[4+lf]};
      H.wallT.push_back(tetplan(M,r));
      H.wallMask[(std::size_t)r.bubble]=1;
      for(int j=0;j<3;++j)H.wallMask[(std::size_t)M.tets[(std::size_t)c].v[hxt2::TET_FACE[lf][j]]]=1;
    }
    if(H.wallT.empty()||M.inTet.empty()||M.outTet.empty())
      throw std::runtime_error("pure-tet requires nonzero wall/inlet/outlet faces");
    double wa=0,ia=0,oa=0;
    for(const auto&f:H.wallT)wa+=f.area;
    for(const auto&f:H.inT)ia+=f.area;
    for(const auto&r:M.outTet)oa+=tetplan(M,r).area;
    std::printf("PURETET_WALL_CLASSIFY wallFaces=%zu inletFaces=%zu outletFaces=%zu wallArea=%.12e inletArea=%.12e outletArea=%.12e status=PASS\n",H.wallT.size(),M.inTet.size(),M.outTet.size(),wa,ia,oa);
  }
  repack(H.B);
  std::vector<double>x(B0.nv,0),y=x,z=x;
  for(std::uint64_t v=0;v<M.h.nv;++v){x[v]=inletU.x;y[v]=inletU.y;z[v]=inletU.z;}
  auto c=hxt3a::b_host(H.B,x,y,z);
  for(int i=0;i<H.B.np;++i)H.constResidual=std::max(H.constResidual,std::abs(c[i]+H.csrc[i]));
  if(H.constResidual>1e-10*std::max(1.0,std::abs(H.sourceSum)))throw std::runtime_error("HXT5B DG continuity trace/source audit failed");
  return H;
}
inline BoundaryHost build_boundaries(const HostMesh&M,const hxt3a::BCSR&B0,double bulk){
  return build_boundaries(M,B0,nodals::stage23::Vec3{0.,0.,bulk});
}
inline hxt4b::FixedData plug_initial(const HostMesh&M,nodals::stage23::Vec3 inletU){
  hxt4b::FixedData F;F.mask.assign(M.h.nvel,0);F.u0.assign(M.h.nvel,0);F.u1.assign(M.h.nvel,0);F.u2.assign(M.h.nvel,0);for(std::uint64_t v=0;v<M.h.nv;++v){F.u0[v]=(Real)inletU.x;F.u1[v]=(Real)inletU.y;F.u2[v]=(Real)inletU.z;}return F;
}

inline hxt4b::FixedData plug_initial(const HostMesh&M,double bulk){
  return plug_initial(M,nodals::stage23::Vec3{0.,0.,bulk});
}
__device__ __forceinline__ void q1dev(const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn&h,double x,double y,double z,double X[3],double dX[3][3]){
  double c[3]={x,y,z};X[0]=X[1]=X[2]=0;for(int d=0;d<3;++d)for(int j=0;j<3;++j)dX[d][j]=0;
  for(int a=0;a<8;++a){double f[3],df[3];for(int d=0;d<3;++d){double s=hxt4a::h4_hex_sign[a][d];f[d]=.5*(1+s*c[d]);df[d]=.5*s;}double N=f[0]*f[1]*f[2],Nd[3]={df[0]*f[1]*f[2],f[0]*df[1]*f[2],f[0]*f[1]*df[2]};auto P=pts[h.v[a]];X[0]+=N*P.x;X[1]+=N*P.y;X[2]+=N*P.z;for(int d=0;d<3;++d){dX[d][0]+=Nd[d]*P.x;dX[d][1]+=Nd[d]*P.y;dX[d][2]+=Nd[d]*P.z;}}
}
#include "modules/wall/spalding_law_root.inc"

// ============================================================
// HXT5B_SPALDING_PROFILE_INTEGRAL_V1
//
// FE-native wall-law inversion over the whole wall-normal line of the
// wall-adjacent HEX element.  At each surface quadrature point, hold the two
// tangential reference coordinates fixed and use a 4-point Gauss rule through
// the complete wall-normal reference interval.  A single u_tau is chosen so
// that the weighted mean Spalding y+ residual vanishes:
//
//   integral [ y+_Spalding(U_h(y)/u_tau) - y*u_tau/nu ] dy = 0.
//
// This removes the arbitrary single off-wall velocity sample from the
// Spalding inversion.  The Picard wall traction uses the same wall-normal
// quadrature to form a profile-averaged FE trial functional Ubar and applies
// beta = u_tau^2/|Ubar|, so beta*Ubar reproduces the target wall stress.
// ============================================================
__device__ __forceinline__
bool spalding_profile_gauss_point(int nq,int m,double&xi,double&w)
{
  if(nq==4){
    if(m<0||m>=4)return false;
    xi=hxt5a::h5_hex_x[m];w=hxt5a::h5_hex_w[m];return true;
  }
  if(nq==8){
    const double X[8]={
      -0.96028985649753618,-0.79666647741362673,-0.52553240991632899,-0.18343464249564978,
       0.18343464249564978, 0.52553240991632899, 0.79666647741362673, 0.96028985649753618};
    const double W[8]={
       0.10122853629037706,0.22238103445337443,0.31370664587788688,0.36268378337836166,
       0.36268378337836166,0.31370664587788688,0.22238103445337443,0.10122853629037706};
    if(m<0||m>=8)return false;xi=X[m];w=W[m];return true;
  }
  if(nq==12){
    const double X[12]={
      -0.98156063424671924,-0.90411725637047480,-0.76990267419430469,-0.58731795428661748,
      -0.36783149899818018,-0.12523340851146891, 0.12523340851146891, 0.36783149899818018,
       0.58731795428661748, 0.76990267419430469, 0.90411725637047480, 0.98156063424671924};
    const double W[12]={
       0.047175336386511411,0.10693932599531907,0.16007832854334642,0.20316742672306573,
       0.23349253653835461,0.24914704581340269,0.24914704581340269,0.23349253653835461,
       0.20316742672306573,0.16007832854334642,0.10693932599531907,0.047175336386511411};
    if(m<0||m>=12)return false;xi=X[m];w=W[m];return true;
  }
  return false;
}

__device__ __forceinline__
double spalding_profile_residual(
    const double U[12],const double y[12],const double w[12],int nq,
    double nu,double ut)
{
  if(!(ut>0.0)||!(nu>0.0)||!isfinite(ut))return 1.0e300;
  double r=0.0,sw=0.0;
  for(int m=0;m<nq;++m){
    if(!(w[m]>0.0))continue;
    const double up=U[m]/ut;
    const double sp=spy(up,.4,5.5);
    if(!isfinite(sp))return 1.0e300;
    r+=w[m]*(sp-y[m]*ut/nu);
    sw+=w[m];
  }
  return sw>0.0 ? r/sw : 1.0e300;
}

template<class T>
__device__ __forceinline__
bool spalding_profile_hex(
    const nodals_hxt1::Point*pts,
    const nodals_hxt1::HexConn&cell,
    const HexFacePlan&f,
    const double cw[3],
    const T U14[14],
    double nu,int nq,
    double&ut,double&yp,double&beta,double&ubar,double&ybar,
    double basisAvg[14],double&rmsResidual,double&pointUTauRelSpan)
{
  if(!(nu>0.0)||(nq!=4&&nq!=8&&nq!=12))return false;

  double Xw[3],dXw[3][3];
  q1dev(pts,cell,cw[0],cw[1],cw[2],Xw,dXw);
  double cr[3]={
    dXw[f.d0][1]*dXw[f.d1][2]-dXw[f.d0][2]*dXw[f.d1][1],
    dXw[f.d0][2]*dXw[f.d1][0]-dXw[f.d0][0]*dXw[f.d1][2],
    dXw[f.d0][0]*dXw[f.d1][1]-dXw[f.d0][1]*dXw[f.d1][0]
  };
  if(cr[0]*f.n[0]+cr[1]*f.n[1]+cr[2]*f.n[2]<0.0){
    cr[0]=-cr[0];cr[1]=-cr[1];cr[2]=-cr[2];
  }
  const double jac=sqrt(cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]);
  if(!(jac>0.0)||!isfinite(jac))return false;
  const double n[3]={cr[0]/jac,cr[1]/jac,cr[2]/jac};

  double Um[12]={0},ym[12]={0},wm[12]={0};
  for(int a=0;a<14;++a)basisAvg[a]=0.0;
  double sw=0.0,su=0.0,sy=0.0;
  double minPointUT=1.0e300,maxPointUT=0.0;
  int positivePointRoots=0;

  for(int m=0;m<nq;++m){
    double xi=0.0,wref=0.0;
    if(!spalding_profile_gauss_point(nq,m,xi,wref))return false;
    double c[3]={cw[0],cw[1],cw[2]};
    // Complete wall-normal reference line.  If the wall is xi=+1 the order
    // is reversed, but the same Gauss-Legendre abscissae are used.
    c[f.fd]=-f.fv*xi;

    double ph[14],gr[14][3];
    hxt2::hex_basis_ref(c[0],c[1],c[2],ph,gr);
    double Xm[3],dXm[3][3];
    q1dev(pts,cell,c[0],c[1],c[2],Xm,dXm);

    const double y=(Xw[0]-Xm[0])*n[0]+(Xw[1]-Xm[1])*n[1]+(Xw[2]-Xm[2])*n[2];
    const double dy_dxi=fabs(dXm[f.fd][0]*n[0]+dXm[f.fd][1]*n[1]+dXm[f.fd][2]*n[2]);
    const double w=wref*dy_dxi;
    if(!(y>0.0)||!(w>0.0)||!isfinite(y)||!isfinite(w))return false;

    double us=0.0;
    for(int a=0;a<14;++a){
      us+=(double)U14[a]*ph[a];
      basisAvg[a]+=w*ph[a];
    }
    Um[m]=fabs(us);ym[m]=y;wm[m]=w;
    sw+=w;su+=w*us;sy+=w*y;

    double upoint=0.0,ypp=0.0,betap=0.0;
    if(spalding(us,y,nu,upoint,ypp,betap)&&upoint>0.0&&isfinite(upoint)){
      minPointUT=fmin(minPointUT,upoint);
      maxPointUT=fmax(maxPointUT,upoint);
      ++positivePointRoots;
    }
  }

  if(!(sw>0.0))return false;
  ubar=su/sw;
  ybar=sy/sw;
  for(int a=0;a<14;++a)basisAvg[a]/=sw;

  if(positivePointRoots==0){
    ut=0.0;yp=0.0;beta=nu/fmax(ybar,1.0e-30);
    rmsResidual=0.0;pointUTauRelSpan=0.0;
    return isfinite(beta)&&beta>0.0;
  }

  double lo=fmax(0.5*minPointUT,1.0e-14);
  double hi=fmax(2.0*maxPointUT,2.0*lo);
  double rlo=spalding_profile_residual(Um,ym,wm,nq,nu,lo);
  double rhi=spalding_profile_residual(Um,ym,wm,nq,nu,hi);

  for(int k=0;k<20 && (!(rlo>=0.0)||!isfinite(rlo));++k){
    lo*=0.5;
    rlo=spalding_profile_residual(Um,ym,wm,nq,nu,lo);
  }
  for(int k=0;k<20 && (!(rhi<=0.0)||!isfinite(rhi));++k){
    hi*=2.0;
    rhi=spalding_profile_residual(Um,ym,wm,nq,nu,hi);
  }
  if(!isfinite(rlo)||!isfinite(rhi)||rlo<0.0||rhi>0.0)return false;

  for(int k=0;k<36;++k){
    const double mid=0.5*(lo+hi);
    const double rm=spalding_profile_residual(Um,ym,wm,nq,nu,mid);
    if(!isfinite(rm))return false;
    if(rm>0.0)lo=mid;else hi=mid;
  }
  ut=0.5*(lo+hi);
  yp=ybar*ut/nu;

  const double UbarAbs=fabs(ubar);
  beta=(UbarAbs>1.0e-14)?ut*ut/UbarAbs:nu/fmax(ybar,1.0e-30);
  if(!(beta>0.0)||!isfinite(beta)||!isfinite(yp)||!isfinite(ut))return false;

  double rr=0.0;
  for(int m=0;m<nq;++m){
    const double rm=spy(Um[m]/ut,.4,5.5)-ym[m]*ut/nu;
    rr+=wm[m]*rm*rm;
  }
  rmsResidual=sqrt(rr/sw);
  pointUTauRelSpan=(maxPointUT>0.0&&minPointUT<1.0e299)
    ?(maxPointUT-minPointUT)/fmax(ut,1.0e-30):0.0;
  return isfinite(rmsResidual)&&isfinite(pointUTauRelSpan);
}


// ============================================================
// HXT5B_SPALDING_PROFILE_AVERAGE_V1
//
// Average-first wall-law inversion.  NQ is used only to obtain accurate
// wall-normal volume averages of U_h, y and the FE trial functional.  One
// Spalding inversion is then performed at (Ubar,ybar):
//
//   y+_Spalding(|Ubar|/u_tau) = ybar*u_tau/nu.
//
// This is deliberately different from spalding_profile_hex(), which fits one
// u_tau to the nonlinear residual at every normal quadrature point.  The
// average-first variant is intended for Robin/traction walls whose physical
// FE trace is not strongly zero; it interprets the wall-adjacent column as one
// unresolved wall-model state rather than demanding a pointwise wall-law
// profile through the element.
//
// The averaging weight is the physical HEX volume Jacobian at fixed tangential
// reference coordinates.  basisAvg therefore represents the NQ volume-average
// trial functional used by the traction linearization.
// ============================================================

template<class T>
__device__ __forceinline__
bool spalding_match_height_point(
    const WallMatchPoint*mp,int fi,int q,
    const T*u,
    double nu,double&ut,double&yp,double&beta,double&um)
{
  if(!mp)return false;
  const WallMatchPoint&m=mp[(std::size_t)fi*16u+(std::size_t)q];
  if(m.cell<0||!(m.y>0.0))return false;
  um=0.0;
  #pragma unroll
  for(int a=0;a<14;++a){
    const int g=m.g[a];
    if(g<0)return false;
    um+=(double)u[g]*m.b[a];
  }
  return isfinite(um)&&spalding(um,m.y,nu,ut,yp,beta);
}


__device__ __forceinline__
double spalding_profile3_objective(
    const double U[3],const double y[3],double nu,double ut)
{
  if(!(ut>0.0)||!(nu>0.0)||!isfinite(ut))return 1.0e300;
  double J=0.0;
  #pragma unroll
  for(int m=0;m<3;++m){
    const double r=spy(U[m]/ut,.4,5.5)-y[m]*ut/nu;
    if(!isfinite(r))return 1.0e300;
    J+=r*r;
  }
  return J/3.0;
}

template<class T>
__device__ __forceinline__
bool spalding_match_profile3(
    const WallMatchPoint*mp,int fi,int q,
    const T*u,double nu,
    double&ut,double&ypMid,double&beta,double&uMid,
    double&fitRms,double&pointUTauRelSpan)
{
  if(!mp||!(nu>0.0))return false;

  double U[3]={0,0,0},Usigned[3]={0,0,0},y[3]={0,0,0};
  double root[3]={0,0,0};
  double minRoot=1.0e300,maxRoot=0.0,sumRoot=0.0;
  int nRoot=0;

  #pragma unroll
  for(int m=0;m<3;++m){
    const std::size_t idx=((std::size_t)fi*16u+(std::size_t)q)*3u+(std::size_t)m;
    const WallMatchPoint&pm=mp[idx];
    if(pm.cell<0||!(pm.y>0.0))return false;

    double um=0.0;
    #pragma unroll
    for(int a=0;a<14;++a){
      const int g=pm.g[a];
      if(g<0)return false;
      um+=(double)u[g]*pm.b[a];
    }
    if(!isfinite(um))return false;

    Usigned[m]=um;
    U[m]=fabs(um);
    y[m]=pm.y;

    // Individual point inversions are ONLY seed/diagnostic information.
    // A zero-speed point legitimately gives u_tau=0, and a numerically
    // awkward point inversion must not abort the 3-point least-squares fit.
    double r=0.0,yp=0.0,bt=0.0;
    const bool rok=spalding(U[m],pm.y,nu,r,yp,bt);
    if(rok && r>0.0 && isfinite(r)){
      root[m]=r;
      minRoot=fmin(minRoot,r);
      maxRoot=fmax(maxRoot,r);
      sumRoot+=r;
      ++nRoot;
    }
  }

  if(!(y[0]<y[1]&&y[1]<y[2]))return false;

  const double Umax=fmax(U[0],fmax(U[1],U[2]));

  // Exact all-zero state is a valid stationary-wall state, not a root failure.
  if(Umax<=1.0e-14){
    ut=0.0;
    ypMid=0.0;
    uMid=Usigned[1];
    beta=nu/y[1];
    fitRms=0.0;
    pointUTauRelSpan=0.0;
    return isfinite(beta)&&beta>0.0;
  }

  // Seed from every successful point inversion.  If none survives, use the
  // viscous dimensional estimate u_tau ~ sqrt(nu U / y) as a finite scale.
  double scale=0.0;
  if(nRoot>0)
    scale=sumRoot/(double)nRoot;
  else
    scale=sqrt(nu*Umax/fmax(y[1],1.0e-30));

  if(!(scale>0.0)||!isfinite(scale))
    scale=1.0;

  // Broad positive trust box.  Successful point roots tighten it, but no
  // individual point is authoritative.
  double lo=fmax(0.05*scale,1.0e-10);
  double hi=fmax(20.0*scale,100.0*lo);
  if(nRoot>0){
    lo=fmax(fmin(lo,0.20*minRoot),1.0e-10);
    hi=fmax(hi,5.0*maxRoot);
  }

  ut=fmin(hi,fmax(lo,scale));

  // Fast path: safeguarded Gauss-Newton, as before.  Unlike the old code,
  // failure of this local iteration does not abort the wall assembly.
  bool gnOK=true;
  #pragma unroll
  for(int it=0;it<12;++it){
    const double J0=spalding_profile3_objective(U,y,nu,ut);
    if(!isfinite(J0) || J0>=1.0e299){gnOK=false;break;}

    double g=0.0,h=0.0;
    #pragma unroll
    for(int m=0;m<3;++m){
      const double up=U[m]/ut;
      const double r=spy(up,.4,5.5)-y[m]*ut/nu;
      const double dr=-(up/ut)*spyd_newton(up,.4,5.5)-y[m]/nu;
      if(!isfinite(r)||!isfinite(dr)){gnOK=false;break;}
      g+=r*dr;
      h+=dr*dr;
    }
    if(!gnOK||!(h>0.0)||!isfinite(h)){gnOK=false;break;}

    double step=g/h;
    double trial=fmin(hi,fmax(lo,ut-step));
    double Jt=spalding_profile3_objective(U,y,nu,trial);

    #pragma unroll
    for(int ls=0;ls<10 && (!isfinite(Jt)||Jt>=1.0e299||Jt>J0);++ls){
      step*=0.5;
      trial=fmin(hi,fmax(lo,ut-step));
      Jt=spalding_profile3_objective(U,y,nu,trial);
    }

    if(!isfinite(Jt)||Jt>=1.0e299||Jt>J0){
      gnOK=false;
      break;
    }

    const double rel=fabs(trial-ut)/fmax(ut,1.0e-30);
    ut=trial;
    if(rel<1.0e-10)break;
  }

  double J=spalding_profile3_objective(U,y,nu,ut);

  // Robust fallback: derivative-free minimization in log(u_tau).  This is
  // entered only if the fast Gauss-Newton path becomes non-finite or invalid.
  // It still fits ALL THREE physical samples; no sample is silently dropped.
  if(!gnOK || !(ut>0.0) || !isfinite(ut) || !isfinite(J) || J>=1.0e299){
    double xa=log(lo), xb=log(hi);
    const double phi=0.6180339887498948482;
    double x1=xb-phi*(xb-xa);
    double x2=xa+phi*(xb-xa);
    double f1=spalding_profile3_objective(U,y,nu,exp(x1));
    double f2=spalding_profile3_objective(U,y,nu,exp(x2));

    #pragma unroll
    for(int it=0;it<28;++it){
      if(f1<=f2){
        xb=x2; x2=x1; f2=f1;
        x1=xb-phi*(xb-xa);
        f1=spalding_profile3_objective(U,y,nu,exp(x1));
      }else{
        xa=x1; x1=x2; f1=f2;
        x2=xa+phi*(xb-xa);
        f2=spalding_profile3_objective(U,y,nu,exp(x2));
      }
    }

    const double xbest=(f1<=f2)?x1:x2;
    ut=exp(xbest);
    J=(f1<=f2)?f1:f2;
  }

  if(!(ut>0.0)||!isfinite(ut)||!isfinite(J)||J>=1.0e299)
    return false;

  ypMid=y[1]*ut/nu;
  uMid=Usigned[1];
  const double Ua=fabs(uMid);
  beta=(Ua>1.0e-14)?ut*ut/Ua:nu/y[1];
  fitRms=sqrt(fmax(J,0.0));

  // Diagnostic span uses only point inversions that were actually valid.
  pointUTauRelSpan=(nRoot>=2 && maxRoot>=minRoot && minRoot<1.0e299)
    ?(maxRoot-minRoot)/fmax(ut,1.0e-30)
    :0.0;

  return isfinite(ypMid)&&isfinite(beta)&&beta>0.0&&
         isfinite(fitRms)&&isfinite(pointUTauRelSpan);
}


#include "modules/wall/spalding_profile5_objective.inc"

#include "modules/wall/tet_wall_match_cache.inc"
#include "modules/wall/tet_wall_ray_cache.inc"

template<class T>
__device__ __forceinline__
bool spalding_match_profile5(
    const WallMatchPoint*mp,int fi,int q,
    const T*u,double nu,
    double&ut,double&ypMid,double&beta,double&uMid,
    double&fitRms,double&pointUTauRelSpan)
{
  if(!mp||!(nu>0.0))return false;

  double U[5]={0,0,0,0,0},Usigned[5]={0,0,0,0,0},y[5]={0,0,0,0,0};
  double root[5]={0,0,0,0,0};
  double minRoot=1.0e300,maxRoot=0.0,sumRoot=0.0,Umax=0.0;
  int nRoot=0;

  #pragma unroll
  for(int m=0;m<5;++m){
    const std::size_t idx=((std::size_t)fi*16u+(std::size_t)q)*5u+(std::size_t)m;
    const WallMatchPoint&pm=mp[idx];
    if(pm.cell<0||!(pm.y>0.0))return false;
    double um=0.0;
    #pragma unroll
    for(int a=0;a<14;++a){
      const int g=pm.g[a];
      if(g<0)return false;
      um+=(double)u[g]*pm.b[a];
    }
    if(!isfinite(um))return false;
    Usigned[m]=um; U[m]=fabs(um); y[m]=pm.y; Umax=fmax(Umax,U[m]);

    double r=0.0,yp=0.0,bt=0.0;
    const bool rok=spalding(U[m],pm.y,nu,r,yp,bt);
    if(rok && r>0.0 && isfinite(r)){
      root[m]=r; minRoot=fmin(minRoot,r); maxRoot=fmax(maxRoot,r);
      sumRoot+=r; ++nRoot;
    }
  }

  #pragma unroll
  for(int m=1;m<5;++m)if(!(y[m-1]<y[m]))return false;

  if(Umax<=1.0e-14){
    ut=0.0; ypMid=0.0; uMid=Usigned[2]; beta=nu/y[2];
    fitRms=0.0; pointUTauRelSpan=0.0;
    return isfinite(beta)&&beta>0.0;
  }

  double scale=0.0;
  if(nRoot>0)scale=sumRoot/(double)nRoot;
  else scale=sqrt(nu*Umax/fmax(y[2],1.0e-30));
  if(!(scale>0.0)||!isfinite(scale))scale=1.0;

  double lo=fmax(0.05*scale,1.0e-10);
  double hi=fmax(20.0*scale,100.0*lo);
  if(nRoot>0){
    lo=fmax(fmin(lo,0.20*minRoot),1.0e-10);
    hi=fmax(hi,5.0*maxRoot);
  }
  ut=fmin(hi,fmax(lo,scale));

  bool gnOK=true;
  #pragma unroll
  for(int it=0;it<12;++it){
    const double J0=spalding_profile5_objective(U,y,nu,ut);
    if(!isfinite(J0)||J0>=1.0e299){gnOK=false;break;}
    double g=0.0,h=0.0;
    #pragma unroll
    for(int m=0;m<5;++m){
      const double up=U[m]/ut;
      const double r=spy(up,.4,5.5)-y[m]*ut/nu;
      const double dr=-(up/ut)*spyd_newton(up,.4,5.5)-y[m]/nu;
      if(!isfinite(r)||!isfinite(dr)){gnOK=false;break;}
      g+=r*dr; h+=dr*dr;
    }
    if(!gnOK||!(h>0.0)||!isfinite(h)){gnOK=false;break;}
    double step=g/h;
    double trial=fmin(hi,fmax(lo,ut-step));
    double Jt=spalding_profile5_objective(U,y,nu,trial);
    #pragma unroll
    for(int ls=0;ls<10 && (!isfinite(Jt)||Jt>=1.0e299||Jt>J0);++ls){
      step*=0.5; trial=fmin(hi,fmax(lo,ut-step));
      Jt=spalding_profile5_objective(U,y,nu,trial);
    }
    if(!isfinite(Jt)||Jt>=1.0e299||Jt>J0){gnOK=false;break;}
    const double rel=fabs(trial-ut)/fmax(ut,1.0e-30);
    ut=trial; if(rel<1.0e-10)break;
  }

  double J=spalding_profile5_objective(U,y,nu,ut);
  if(!gnOK || !(ut>0.0) || !isfinite(ut) || !isfinite(J) || J>=1.0e299){
    double xa=log(lo),xb=log(hi);
    const double phi=0.6180339887498948482;
    double x1=xb-phi*(xb-xa),x2=xa+phi*(xb-xa);
    double f1=spalding_profile5_objective(U,y,nu,exp(x1));
    double f2=spalding_profile5_objective(U,y,nu,exp(x2));
    #pragma unroll
    for(int it=0;it<28;++it){
      if(f1<=f2){
        xb=x2; x2=x1; f2=f1; x1=xb-phi*(xb-xa);
        f1=spalding_profile5_objective(U,y,nu,exp(x1));
      }else{
        xa=x1; x1=x2; f1=f2; x2=xa+phi*(xb-xa);
        f2=spalding_profile5_objective(U,y,nu,exp(x2));
      }
    }
    const double xbest=(f1<=f2)?x1:x2;
    ut=exp(xbest); J=(f1<=f2)?f1:f2;
  }

  if(!(ut>0.0)||!isfinite(ut)||!isfinite(J)||J>=1.0e299)return false;
  ypMid=y[2]*ut/nu; uMid=Usigned[2];
  const double Ua=fabs(uMid);
  beta=(Ua>1.0e-14)?ut*ut/Ua:nu/y[2];
  fitRms=sqrt(fmax(J,0.0));
  pointUTauRelSpan=(nRoot>=2 && maxRoot>=minRoot && minRoot<1.0e299)
    ?(maxRoot-minRoot)/fmax(ut,1.0e-30):0.0;
  return isfinite(ypMid)&&isfinite(beta)&&beta>0.0&&
         isfinite(fitRms)&&isfinite(pointUTauRelSpan);
}

#include "modules/wall/tet_wall_picard_cache.inc"

#include "modules/wall/spalding_profile5_tet_fit.inc"

template<class T>
__device__ __forceinline__
bool profile_average_velocity_basis_hex(
    const nodals_hxt1::Point*pts,
    const nodals_hxt1::HexConn&cell,
    const HexFacePlan&f,
    const double cw[3],
    const T U14[14],
    int nq,
    double&ubar,
    double basisAvg[14])
{
  if(nq!=4&&nq!=8&&nq!=12)return false;

  for(int a=0;a<14;++a)basisAvg[a]=0.0;
  double sw=0.0,su=0.0;

  for(int m=0;m<nq;++m){
    double xi=0.0,wref=0.0;
    if(!spalding_profile_gauss_point(nq,m,xi,wref))return false;

    double c[3]={cw[0],cw[1],cw[2]};
    c[f.fd]=-f.fv*xi;

    double ph[14],gr[14][3],I[9],det=0.0;
    hxt2::hex_basis_ref(c[0],c[1],c[2],ph,gr);
    if(!hxt2::hex_metric(pts,cell,c[0],c[1],c[2],I,det))return false;
    if(!(det>0.0)||!isfinite(det))return false;

    const double w=wref*det;
    if(!(w>0.0)||!isfinite(w))return false;

    double us=0.0;
    for(int a=0;a<14;++a){
      us+=(double)U14[a]*ph[a];
      basisAvg[a]+=w*ph[a];
    }
    sw+=w;
    su+=w*us;
  }

  if(!(sw>0.0)||!isfinite(sw)||!isfinite(su))return false;

  ubar=su/sw;
  for(int a=0;a<14;++a)basisAvg[a]/=sw;

  if(!isfinite(ubar))return false;
  for(int a=0;a<14;++a)
    if(!isfinite(basisAvg[a]))return false;

  return true;
}

template<class T>
__device__ __forceinline__
bool spalding_profile_average_hex(
    const nodals_hxt1::Point*pts,
    const nodals_hxt1::HexConn&cell,
    const HexFacePlan&f,
    const double cw[3],
    const T U14[14],
    double nu,int nq,
    double&ut,double&yp,double&beta,double&ubar,double&ybar,
    double basisAvg[14],double&rmsResidual,double&pointUTauRelSpan)
{
  if(!(nu>0.0)||(nq!=4&&nq!=8&&nq!=12))return false;

  double Xw[3],dXw[3][3];
  q1dev(pts,cell,cw[0],cw[1],cw[2],Xw,dXw);
  double cr[3]={
    dXw[f.d0][1]*dXw[f.d1][2]-dXw[f.d0][2]*dXw[f.d1][1],
    dXw[f.d0][2]*dXw[f.d1][0]-dXw[f.d0][0]*dXw[f.d1][2],
    dXw[f.d0][0]*dXw[f.d1][1]-dXw[f.d0][1]*dXw[f.d1][0]
  };
  if(cr[0]*f.n[0]+cr[1]*f.n[1]+cr[2]*f.n[2]<0.0){
    cr[0]=-cr[0];cr[1]=-cr[1];cr[2]=-cr[2];
  }
  const double jac=sqrt(cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]);
  if(!(jac>0.0)||!isfinite(jac))return false;
  const double n[3]={cr[0]/jac,cr[1]/jac,cr[2]/jac};

  double Um[12]={0},ym[12]={0},wm[12]={0};
  for(int a=0;a<14;++a)basisAvg[a]=0.0;
  double sw=0.0,su=0.0,sy=0.0;
  double minPointUT=1.0e300,maxPointUT=0.0;
  int positivePointRoots=0;

  for(int m=0;m<nq;++m){
    double xi=0.0,wref=0.0;
    if(!spalding_profile_gauss_point(nq,m,xi,wref))return false;
    double c[3]={cw[0],cw[1],cw[2]};
    c[f.fd]=-f.fv*xi;

    double ph[14],gr[14][3],I[9],det=0.0;
    hxt2::hex_basis_ref(c[0],c[1],c[2],ph,gr);
    if(!hxt2::hex_metric(pts,cell,c[0],c[1],c[2],I,det))return false;
    if(!(det>0.0)||!isfinite(det))return false;

    double Xm[3],dXm[3][3];
    q1dev(pts,cell,c[0],c[1],c[2],Xm,dXm);
    const double y=(Xw[0]-Xm[0])*n[0]+(Xw[1]-Xm[1])*n[1]+(Xw[2]-Xm[2])*n[2];
    const double w=wref*det;
    if(!(y>0.0)||!(w>0.0)||!isfinite(y)||!isfinite(w))return false;

    double us=0.0;
    for(int a=0;a<14;++a){
      us+=(double)U14[a]*ph[a];
      basisAvg[a]+=w*ph[a];
    }
    Um[m]=fabs(us);ym[m]=y;wm[m]=w;
    sw+=w;su+=w*us;sy+=w*y;

    double upoint=0.0,ypp=0.0,betap=0.0;
    if(spalding(us,y,nu,upoint,ypp,betap)&&upoint>0.0&&isfinite(upoint)){
      minPointUT=fmin(minPointUT,upoint);
      maxPointUT=fmax(maxPointUT,upoint);
      ++positivePointRoots;
    }
  }

  if(!(sw>0.0))return false;
  ubar=su/sw;
  ybar=sy/sw;
  for(int a=0;a<14;++a)basisAvg[a]/=sw;

  if(!spalding(ubar,ybar,nu,ut,yp,beta))return false;
  if(!(beta>0.0)||!isfinite(beta)||!isfinite(yp)||!isfinite(ut))return false;

  // Diagnostics only: quantify how unlike a pointwise Spalding profile the
  // resolved FE column is.  These residuals do NOT enter the average-first root.
  double rr=0.0;
  if(ut>0.0){
    for(int m=0;m<nq;++m){
      const double rm=spy(Um[m]/ut,.4,5.5)-ym[m]*ut/nu;
      rr+=wm[m]*rm*rm;
    }
    rmsResidual=sqrt(rr/sw);
  }else rmsResidual=0.0;
  pointUTauRelSpan=(positivePointRoots>0&&maxPointUT>0.0&&minPointUT<1.0e299)
    ?(maxPointUT-minPointUT)/fmax(ut,1.0e-30):0.0;
  return isfinite(rmsResidual)&&isfinite(pointUTauRelSpan);
}


// ============================================================
// HXT5B_SPALDING_PROFILE_AVERAGE_RELATIVE_WALL_V1
// NQ average-first Robin-wall experiment.
// Live wall-law state: Delta Ubar = Ubar - Uwall.
// Live momentum trial: Delta Bbar = Bbar - Bwall.
// Therefore beta*Delta Ubar = u_tau^2 remains exact.
// Physical wall test functional remains unchanged.
// ============================================================
template<class T>
__device__ __forceinline__
bool spalding_profile_average_relative_wall_hex(
    const nodals_hxt1::Point*pts,
    const nodals_hxt1::HexConn&cell,
    const HexFacePlan&f,
    const double cw[3],
    const T U14[14],
    double nu,int nq,
    double&ut,double&yp,double&beta,double&ubar,double&ybar,
    double basisAvg[14],double&rmsResidual,double&pointUTauRelSpan)
{
  if(!(nu>0.0)||(nq!=4&&nq!=8&&nq!=12))return false;

  double Xw[3],dXw[3][3];
  q1dev(pts,cell,cw[0],cw[1],cw[2],Xw,dXw);
  double cr[3]={
    dXw[f.d0][1]*dXw[f.d1][2]-dXw[f.d0][2]*dXw[f.d1][1],
    dXw[f.d0][2]*dXw[f.d1][0]-dXw[f.d0][0]*dXw[f.d1][2],
    dXw[f.d0][0]*dXw[f.d1][1]-dXw[f.d0][1]*dXw[f.d1][0]
  };
  if(cr[0]*f.n[0]+cr[1]*f.n[1]+cr[2]*f.n[2]<0.0){
    cr[0]=-cr[0];cr[1]=-cr[1];cr[2]=-cr[2];
  }
  const double jac=sqrt(cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]);
  if(!(jac>0.0)||!isfinite(jac))return false;
  const double n[3]={cr[0]/jac,cr[1]/jac,cr[2]/jac};
  double phWall[14],grWall[14][3];
  hxt2::hex_basis_ref(cw[0],cw[1],cw[2],phWall,grWall);
  double uw=0.0;
  for(int a=0;a<14;++a)uw+=(double)U14[a]*phWall[a];

  double Um[12]={0},ym[12]={0},wm[12]={0};
  for(int a=0;a<14;++a)basisAvg[a]=0.0;
  double sw=0.0,su=0.0,sy=0.0;
  double minPointUT=1.0e300,maxPointUT=0.0;
  int positivePointRoots=0;

  for(int m=0;m<nq;++m){
    double xi=0.0,wref=0.0;
    if(!spalding_profile_gauss_point(nq,m,xi,wref))return false;
    double c[3]={cw[0],cw[1],cw[2]};
    c[f.fd]=-f.fv*xi;

    double ph[14],gr[14][3],I[9],det=0.0;
    hxt2::hex_basis_ref(c[0],c[1],c[2],ph,gr);
    if(!hxt2::hex_metric(pts,cell,c[0],c[1],c[2],I,det))return false;
    if(!(det>0.0)||!isfinite(det))return false;

    double Xm[3],dXm[3][3];
    q1dev(pts,cell,c[0],c[1],c[2],Xm,dXm);
    const double y=(Xw[0]-Xm[0])*n[0]+(Xw[1]-Xm[1])*n[1]+(Xw[2]-Xm[2])*n[2];
    const double w=wref*det;
    if(!(y>0.0)||!(w>0.0)||!isfinite(y)||!isfinite(w))return false;

    double us=0.0;
    for(int a=0;a<14;++a){
      us+=(double)U14[a]*ph[a];
      basisAvg[a]+=w*(ph[a]-phWall[a]);
    }
    const double du=us-uw;
    Um[m]=fabs(du);ym[m]=y;wm[m]=w;
    sw+=w;su+=w*du;sy+=w*y;

    double upoint=0.0,ypp=0.0,betap=0.0;
    if(spalding(us-uw,y,nu,upoint,ypp,betap)&&upoint>0.0&&isfinite(upoint)){
      minPointUT=fmin(minPointUT,upoint);
      maxPointUT=fmax(maxPointUT,upoint);
      ++positivePointRoots;
    }
  }

  if(!(sw>0.0))return false;
  ubar=su/sw;
  ybar=sy/sw;
  for(int a=0;a<14;++a)basisAvg[a]/=sw;

  if(!spalding(ubar,ybar,nu,ut,yp,beta))return false;
  if(!(beta>0.0)||!isfinite(beta)||!isfinite(yp)||!isfinite(ut))return false;

  // Diagnostics only: quantify how unlike a pointwise Spalding profile the
  // resolved FE column is.  These residuals do NOT enter the average-first root.
  double rr=0.0;
  if(ut>0.0){
    for(int m=0;m<nq;++m){
      const double rm=spy(Um[m]/ut,.4,5.5)-ym[m]*ut/nu;
      rr+=wm[m]*rm*rm;
    }
    rmsResidual=sqrt(rr/sw);
  }else rmsResidual=0.0;
  pointUTauRelSpan=(positivePointRoots>0&&maxPointUT>0.0&&minPointUT<1.0e299)
    ?(maxPointUT-minPointUT)/fmax(ut,1.0e-30):0.0;
  return isfinite(rmsResidual)&&isfinite(pointUTauRelSpan);
}


// ============================================================
// HXT5B_LOG_LAW_PROFILE_AVERAGE_V1
// Pure high-Re log law used as a Robin wall-stress closure:
//   U+ = (1/kappa) ln(E y+)
// with kappa=0.4 and E=exp(0.4*5.5)=9.025013499434122.
// NQ is used only to volume-average U_h, y and the FE trial
// functional; one log-law inversion is performed at (Ubar,ybar).
// ============================================================
__device__ __forceinline__
double loglaw_g(double yp,double reY)
{
  constexpr double KAPPA=0.4;
  constexpr double E=9.025013499434122;
  const double a=E*yp;
  if(!(a>1.0))return -reY;
  return (yp/KAPPA)*log(a)-reY;
}

__device__ __forceinline__
bool loglaw(double slip,double y,double nu,double&ut,double&yp,double&beta)
{
  constexpr double E=9.025013499434122;
  const double U=fabs(slip);
  if(!(U>0.0)||!(y>0.0)||!(nu>0.0)||!isfinite(U))return false;
  const double reY=y*U/nu;
  double lo=(1.0+1.0e-12)/E;
  double hi=fmax(30.0,sqrt(fmax(reY,0.0))+1.0);
  for(int it=0;it<80 && loglaw_g(hi,reY)<0.0;++it)hi*=2.0;
  if(!(loglaw_g(hi,reY)>=0.0)||!isfinite(hi))return false;
  for(int it=0;it<80;++it){
    const double mid=0.5*(lo+hi);
    if(loglaw_g(mid,reY)>0.0)hi=mid;else lo=mid;
  }
  yp=0.5*(lo+hi);
  ut=yp*nu/y;
  beta=ut*ut/U;
  return ut>0.0&&yp>0.0&&beta>0.0&&isfinite(ut)&&isfinite(yp)&&isfinite(beta);
}

// NQ physical-volume average followed by ONE pure log-law inversion.
template<class T>
__device__ __forceinline__
bool loglaw_profile_average_hex(
    const nodals_hxt1::Point*pts,
    const nodals_hxt1::HexConn&cell,
    const HexFacePlan&f,
    const double cw[3],
    const T U14[14],
    double nu,int nq,
    double&ut,double&yp,double&beta,double&ubar,double&ybar,
    double basisAvg[14],double&rmsResidual,double&pointUTauRelSpan)
{
  if(!(nu>0.0)||(nq!=4&&nq!=8&&nq!=12))return false;

  double Xw[3],dXw[3][3];
  q1dev(pts,cell,cw[0],cw[1],cw[2],Xw,dXw);
  double cr[3]={
    dXw[f.d0][1]*dXw[f.d1][2]-dXw[f.d0][2]*dXw[f.d1][1],
    dXw[f.d0][2]*dXw[f.d1][0]-dXw[f.d0][0]*dXw[f.d1][2],
    dXw[f.d0][0]*dXw[f.d1][1]-dXw[f.d0][1]*dXw[f.d1][0]
  };
  if(cr[0]*f.n[0]+cr[1]*f.n[1]+cr[2]*f.n[2]<0.0){
    cr[0]=-cr[0];cr[1]=-cr[1];cr[2]=-cr[2];
  }
  const double jac=sqrt(cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]);
  if(!(jac>0.0)||!isfinite(jac))return false;
  const double n[3]={cr[0]/jac,cr[1]/jac,cr[2]/jac};

  double Um[12]={0},ym[12]={0},wm[12]={0};
  for(int a=0;a<14;++a)basisAvg[a]=0.0;
  double sw=0.0,su=0.0,sy=0.0;
  double minPointUT=1.0e300,maxPointUT=0.0;
  int positivePointRoots=0;

  for(int m=0;m<nq;++m){
    double xi=0.0,wref=0.0;
    if(!spalding_profile_gauss_point(nq,m,xi,wref))return false;
    double c[3]={cw[0],cw[1],cw[2]};
    c[f.fd]=-f.fv*xi;

    double ph[14],gr[14][3],I[9],det=0.0;
    hxt2::hex_basis_ref(c[0],c[1],c[2],ph,gr);
    if(!hxt2::hex_metric(pts,cell,c[0],c[1],c[2],I,det))return false;
    if(!(det>0.0)||!isfinite(det))return false;

    double Xm[3],dXm[3][3];
    q1dev(pts,cell,c[0],c[1],c[2],Xm,dXm);
    const double y=(Xw[0]-Xm[0])*n[0]+(Xw[1]-Xm[1])*n[1]+(Xw[2]-Xm[2])*n[2];
    const double w=wref*det;
    if(!(y>0.0)||!(w>0.0)||!isfinite(y)||!isfinite(w))return false;

    double us=0.0;
    for(int a=0;a<14;++a){
      us+=(double)U14[a]*ph[a];
      basisAvg[a]+=w*ph[a];
    }
    Um[m]=fabs(us);ym[m]=y;wm[m]=w;
    sw+=w;su+=w*us;sy+=w*y;

    double upoint=0.0,ypp=0.0,betap=0.0;
    if(loglaw(us,y,nu,upoint,ypp,betap)&&upoint>0.0&&isfinite(upoint)){
      minPointUT=fmin(minPointUT,upoint);
      maxPointUT=fmax(maxPointUT,upoint);
      ++positivePointRoots;
    }
  }

  if(!(sw>0.0))return false;
  ubar=su/sw;
  ybar=sy/sw;
  for(int a=0;a<14;++a)basisAvg[a]/=sw;

  if(!loglaw(ubar,ybar,nu,ut,yp,beta))return false;
  if(!(beta>0.0)||!isfinite(beta)||!isfinite(yp)||!isfinite(ut))return false;

  // Diagnostics only: quantify how unlike a pointwise Spalding profile the
  // resolved FE column is.  These residuals do NOT enter the average-first root.
  double rr=0.0;
  if(ut>0.0){
    for(int m=0;m<nq;++m){
      const double arg=9.025013499434122*ym[m]*ut/nu;
      const double rm=(arg>1.0)?(Um[m]/ut-log(arg)/0.4):0.0;
      rr+=wm[m]*rm*rm;
    }
    rmsResidual=sqrt(rr/sw);
  }else rmsResidual=0.0;
  pointUTauRelSpan=(positivePointRoots>0&&maxPointUT>0.0&&minPointUT<1.0e299)
    ?(maxPointUT-minPointUT)/fmax(ut,1.0e-30):0.0;
  return isfinite(rmsResidual)&&isfinite(pointUTauRelSpan);
}

__device__ __forceinline__ void amax(double*addr,double v){auto*p=(unsigned long long*)addr;unsigned long long old=*p,ass;do{ass=old;if(__longlong_as_double((long long)ass)>=v)break;old=atomicCAS(p,ass,(unsigned long long)__double_as_longlong(v));}while(ass!=old);}

__global__ void dg_hex(const HexFacePlan*F,int nf,const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,const nodals_hxt1::HexVel*hv,const std::int32_t*slot,const Real*u0,const Real*u1,const Real*u2,Real*mat,Real*diag,Real*rhsz,double bulk,double R,double nu,double ms,unsigned long long*bad){
  int fi=blockIdx.x;if(fi>=nf)return;auto f=F[fi];auto H=hv[f.cell];__shared__ Real U0[14],U1[14],U2[14],ph[16][14],gp[16][14][3],nue[16],wq[16];
  if(threadIdx.x<14){int g=H.g[threadIdx.x];U0[threadIdx.x]=u0[g];U1[threadIdx.x]=u1[g];U2[threadIdx.x]=u2[g];}__syncthreads();
  if(threadIdx.x<16){int q=threadIdx.x,i=q>>2,j=q&3;double c[3]={0,0,0};c[f.fd]=f.fv;c[f.d0]=hxt5a::h5_hex_x[i];c[f.d1]=hxt5a::h5_hex_x[j];double pp[14],gr[14][3],gg[14][3],I[9],det;hxt2::hex_basis_ref(c[0],c[1],c[2],pp,gr);if(!hxt2::hex_metric(pts,hc[f.cell],c[0],c[1],c[2],I,det)){atomicAdd(bad,1ULL);wq[q]=0;return;}hxt2::phys_grad14(gr,I,gg);double X[3],dX[3][3];q1dev(pts,hc[f.cell],c[0],c[1],c[2],X,dX);double cr[3]={dX[f.d0][1]*dX[f.d1][2]-dX[f.d0][2]*dX[f.d1][1],dX[f.d0][2]*dX[f.d1][0]-dX[f.d0][0]*dX[f.d1][2],dX[f.d0][0]*dX[f.d1][1]-dX[f.d0][1]*dX[f.d1][0]};if(cr[0]*f.n[0]+cr[1]*f.n[1]+cr[2]*f.n[2]<0){cr[0]*=-1;cr[1]*=-1;cr[2]*=-1;}double jac=sqrt(cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]);Real gu[9]={0};for(int a=0;a<14;++a){ph[q][a]=pp[a];for(int d=0;d<3;++d){Real gd=gg[a][d];gp[q][a][d]=gd;gu[d]+=U0[a]*gd;gu[3+d]+=U1[a]*gd;gu[6+d]+=U2[a]*gd;}}Real nut=hxt5a::h5_nikuradse_nut(X[0],X[1],R,ms,gu,nullptr);nue[q]=nu+nut;wq[q]=hxt5a::h5_hex_w[i]*hxt5a::h5_hex_w[j]*jac;}__syncthreads();
  Real inflow=fmax(Real(0),Real(-bulk*f.n[2]));
  for(int p=threadIdx.x;p<196;p+=blockDim.x){int a=p/14,b=p%14;Real z=0;for(int q=0;q<16;++q){Real dna=gp[q][a][0]*f.n[0]+gp[q][a][1]*f.n[1]+gp[q][a][2]*f.n[2],dnb=gp[q][b][0]*f.n[0]+gp[q][b][1]*f.n[1]+gp[q][b][2]*f.n[2];z+=wq[q]*(inflow*ph[q][a]*ph[q][b]+nue[q]*(-ph[q][a]*dnb+dna*ph[q][b]));}atomicAdd(mat+slot[(size_t)f.cell*196+p],z);if(a==b)atomicAdd(diag+H.g[a],z);}
  for(int a=threadIdx.x;a<14;a+=blockDim.x){Real z=0;for(int q=0;q<16;++q){Real dna=gp[q][a][0]*f.n[0]+gp[q][a][1]*f.n[1]+gp[q][a][2]*f.n[2];z+=wq[q]*(inflow*ph[q][a]*bulk+nue[q]*dna*bulk);}atomicAdd(rhsz+H.g[a],z);}
}
__global__ void dg_tet(const TetFacePlan*F,int nf,const nodals_hxt1::Point*pts,const nodals_hxt1::TetConn*tc,const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,const std::int32_t*slot,const Real*u0,const Real*u1,const Real*u2,Real*mat,Real*diag,Real*rhsz,double bulk,double R,double nu,double ms){
  int fi=blockIdx.x;if(fi>=nf)return;auto f=F[fi];auto T=tv[f.cell];__shared__ Real U0[8],U1[8],U2[8],ph[12][8],gp[12][8][3],nue[12],wq[12];if(threadIdx.x<8){int g=T.g[threadIdx.x];U0[threadIdx.x]=u0[g];U1[threadIdx.x]=u1[g];U2[threadIdx.x]=u2[g];}__syncthreads();
  if(threadIdx.x<12){int q=threadIdx.x,a0=hxt4a::h4_tet_face[f.lf][0],a1=hxt4a::h4_tet_face[f.lf][1],a2=hxt4a::h4_tet_face[f.lf][2];double l[4]={0,0,0,0};l[a0]=QL0[q];l[a1]=QL1[q];l[a2]=QL2[q];double pp[8],gr[8][3],gg[8][3];hxt2::tet_basis(l[0],l[1],l[2],l[3],pp,gr);hxt2::tet_phys_grad(gr,tg[f.cell],gg);Real gu[9]={0};double x=0,y=0;for(int a=0;a<8;++a){ph[q][a]=pp[a];for(int d=0;d<3;++d){Real gd=gg[a][d];gp[q][a][d]=gd;gu[d]+=U0[a]*gd;gu[3+d]+=U1[a]*gd;gu[6+d]+=U2[a]*gd;}}for(int a=0;a<4;++a){auto P=pts[tc[f.cell].v[a]];x+=l[a]*P.x;y+=l[a]*P.y;}Real nut=hxt5a::h5_nikuradse_nut(x,y,R,ms,gu,nullptr);nue[q]=nu+nut;wq[q]=QW12[q]*f.area;}__syncthreads();Real inflow=fmax(Real(0),Real(-bulk*f.n[2]));
  for(int p=threadIdx.x;p<64;p+=blockDim.x){int a=p/8,b=p%8;Real z=0;for(int q=0;q<12;++q){Real dna=gp[q][a][0]*f.n[0]+gp[q][a][1]*f.n[1]+gp[q][a][2]*f.n[2],dnb=gp[q][b][0]*f.n[0]+gp[q][b][1]*f.n[1]+gp[q][b][2]*f.n[2];z+=wq[q]*(inflow*ph[q][a]*ph[q][b]+nue[q]*(-ph[q][a]*dnb+dna*ph[q][b]));}atomicAdd(mat+slot[(size_t)f.cell*64+p],z);if(a==b)atomicAdd(diag+T.g[a],z);}
  for(int a=threadIdx.x;a<8;a+=blockDim.x){Real z=0;for(int q=0;q<12;++q){Real dna=gp[q][a][0]*f.n[0]+gp[q][a][1]*f.n[1]+gp[q][a][2]*f.n[2];z+=wq[q]*(inflow*ph[q][a]*bulk+nue[q]*dna*bulk);}atomicAdd(rhsz+T.g[a],z);}
}

__global__ void wall_hex_cached(
    const HexFacePlan*F,
    int nf,
    const WallHexQuadCache*C,
    const nodals_hxt1::HexVel*hv,
    const std::int32_t*slot,
    const Real*u2,
    Real*mat,
    Real*diag,
    double nu,
    double*stats)
{
  int fi=blockIdx.x;
  if(fi>=nf)return;

  auto f=F[fi];
  auto H=hv[f.cell];

  __shared__ Real U[14];
  __shared__ Real bw[16][14];
  __shared__ Real bs[16][14];
  __shared__ Real betaw[16];

  __shared__ double ar[16];
  __shared__ double ut2[16];
  __shared__ double ypw[16];
  __shared__ double slw[16];
  __shared__ double my[16];
  __shared__ double mu[16];

  if(threadIdx.x<14)
    U[threadIdx.x]=u2[H.g[threadIdx.x]];

  __syncthreads();

  // One thread per quadrature point.
  // Geometry + basis now come directly from persistent cache.
  if(threadIdx.x<16){

    const int q=threadIdx.x;

    const WallHexQuadCache*Q=
      C+(std::size_t)fi*16ull+(std::size_t)q;

    double sl=0;
    double ut=0;
    double yp=0;
    double beta=0;

    for(int a=0;a<14;++a){

      const double wa=Q->bw[a];
      const double sa=Q->bs[a];

      bw[q][a]=(Real)wa;
      bs[q][a]=(Real)sa;

      sl+=(double)U[a]*sa;
    }

    // Exactly the existing Spalding solve and fallback.
    if(!spalding(
          sl,Q->y,nu,
          ut,yp,beta))
    {
      beta=nu/Q->y;
      ut=0;
      yp=0;

      atomicAdd(stats+6,1.0);
    }

    betaw[q]=(Real)(beta*Q->w);

    ar[q]=Q->w;
    ut2[q]=ut*ut*Q->w;
    ypw[q]=yp*Q->w;
    slw[q]=fabs(sl)*Q->w;
    my[q]=yp;
    mu[q]=ut;
  }

  __syncthreads();

  // Same 14x14 local wall matrix.
  for(int p=threadIdx.x;
      p<196;
      p+=blockDim.x)
  {
    const int a=p/14;
    const int b=p%14;

    Real z=0;

    for(int q=0;q<16;++q)
      z+=betaw[q]*
         bw[q][a]*
         bs[q][b];

    atomicAdd(
      mat+
      slot[
        (std::size_t)f.cell*196ull+
        (std::size_t)p],
      z);

    if(a==b)
      atomicAdd(
        diag+H.g[a],
        z);
  }

  // Same wall statistics.
  if(threadIdx.x==0){

    double A=0;
    double U2=0;
    double Y=0;
    double S=0;
    double MY=0;
    double MU=0;

    for(int q=0;q<16;++q){
      A+=ar[q];
      U2+=ut2[q];
      Y+=ypw[q];
      S+=slw[q];
      MY=fmax(MY,my[q]);
      MU=fmax(MU,mu[q]);
    }

    atomicAdd(stats+0,A);
    atomicAdd(stats+1,U2);
    atomicAdd(stats+2,Y);
    atomicAdd(stats+3,S);

    amax(stats+4,MY);
    amax(stats+5,MU);
  }
}


__global__ void wall_hex(const HexFacePlan*F,int nf,const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,const nodals_hxt1::HexVel*hv,const std::int32_t*slot,const Real*u2,Real*mat,Real*diag,double nu,double sf,double*stats,unsigned long long*bad){
  int fi=blockIdx.x;if(fi>=nf)return;auto f=F[fi];auto H=hv[f.cell];__shared__ Real U[14],bw[16][14],bs[16][14],betaw[16];__shared__ double ar[16],ut2[16],ypw[16],slw[16],my[16],mu[16];if(threadIdx.x<14)U[threadIdx.x]=u2[H.g[threadIdx.x]];__syncthreads();
  if(threadIdx.x<16){int q=threadIdx.x,i=q>>2,j=q&3;double cw[3]={0,0,0};cw[f.fd]=f.fv;cw[f.d0]=hxt5a::h5_hex_x[i];cw[f.d1]=hxt5a::h5_hex_x[j];double cs[3]={cw[0],cw[1],cw[2]};cs[f.fd]=f.fv+sf*(-2*f.fv);double pw[14],gw[14][3],ps[14],gs[14][3];hxt2::hex_basis_ref(cw[0],cw[1],cw[2],pw,gw);hxt2::hex_basis_ref(cs[0],cs[1],cs[2],ps,gs);double Xw[3],dXw[3][3],Xs[3],dXs[3][3];q1dev(pts,hc[f.cell],cw[0],cw[1],cw[2],Xw,dXw);q1dev(pts,hc[f.cell],cs[0],cs[1],cs[2],Xs,dXs);double rr=hypot(Xw[0],Xw[1]),jac=0,y=0,sl=0,ut=0,yp=0,beta=0;bool ok=rr>0;if(ok){double nr[3]={Xw[0]/rr,Xw[1]/rr,0};double cr[3]={dXw[f.d0][1]*dXw[f.d1][2]-dXw[f.d0][2]*dXw[f.d1][1],dXw[f.d0][2]*dXw[f.d1][0]-dXw[f.d0][0]*dXw[f.d1][2],dXw[f.d0][0]*dXw[f.d1][1]-dXw[f.d0][1]*dXw[f.d1][0]};if(cr[0]*nr[0]+cr[1]*nr[1]+cr[2]*nr[2]<0){cr[0]*=-1;cr[1]*=-1;cr[2]*=-1;}jac=sqrt(cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]);y=(Xw[0]-Xs[0])*nr[0]+(Xw[1]-Xs[1])*nr[1];ok=y>0&&jac>0;}for(int a=0;a<14;++a){bw[q][a]=ok?pw[a]:0;bs[q][a]=ok?ps[a]:0;if(ok)sl+=(double)U[a]*ps[a];}if(ok&&!spalding(sl,y,nu,ut,yp,beta)){beta=nu/y;ut=yp=0;atomicAdd(stats+6,1.0);}if(!ok){atomicAdd(bad,1ULL);beta=ut=yp=sl=jac=0;}double w=hxt5a::h5_hex_w[i]*hxt5a::h5_hex_w[j]*jac;betaw[q]=beta*w;ar[q]=w;ut2[q]=ut*ut*w;ypw[q]=yp*w;slw[q]=fabs(sl)*w;my[q]=yp;mu[q]=ut;}__syncthreads();
  for(int p=threadIdx.x;p<196;p+=blockDim.x){int a=p/14,b=p%14;Real z=0;for(int q=0;q<16;++q)z+=betaw[q]*bw[q][a]*bs[q][b];atomicAdd(mat+slot[(size_t)f.cell*196+p],z);if(a==b)atomicAdd(diag+H.g[a],z);}
  if(threadIdx.x==0){double A=0,U2=0,Y=0,S=0,MY=0,MU=0;for(int q=0;q<16;++q){A+=ar[q];U2+=ut2[q];Y+=ypw[q];S+=slw[q];MY=fmax(MY,my[q]);MU=fmax(MU,mu[q]);}atomicAdd(stats+0,A);atomicAdd(stats+1,U2);atomicAdd(stats+2,Y);atomicAdd(stats+3,S);amax(stats+4,MY);amax(stats+5,MU);}
}

struct WallStats{double area=0,fDarcy=0,yPlusMean=0,slipMean=0,yPlusMax=0,uTauMax=0,rootFailures=0;};
inline WallStats wallstats(const Dev<double>&d,double bulk){std::vector<double>h(d.n,0.0);if(d.n)HXT1_CUDA(cudaMemcpy(h.data(),d.p,d.n*sizeof(double),cudaMemcpyDeviceToHost));WallStats q;if(h.size()<7)return q;q.area=h[0];if(q.area>0){q.fDarcy=8*(h[1]/q.area)/(bulk*bulk);q.yPlusMean=h[2]/q.area;q.slipMean=h[3]/q.area;}q.yPlusMax=h[4];q.uTauMax=h[5];q.rootFailures=h[6];return q;}
#include "modules/wall/spalding_wall_tet_traction.inc"
#include "modules/wall/face_projected_axial_tet.inc"
#include "modules/wall/segregated_vector_tet.inc"
#include "modules/wall/segregated_vector_tet_22c.inc"
#include "modules/wall/stage22d4_nitsche_tet.inc"

inline void assemble_boundary(const BoundaryHost&H,const Dev<HexFacePlan>&ih,const Dev<TetFacePlan>&it,const Dev<HexFacePlan>&wh,const Dev<WallHexQuadCache>&wc,const Dev<nodals_hxt1::Point>&pts,const Dev<nodals_hxt1::HexConn>&hc,const Dev<nodals_hxt1::TetConn>&tc,const Dev<nodals_hxt1::HexVel>&hv,const Dev<nodals_hxt1::TetVel>&tv,const Dev<nodals_hxt1::TetGeom>&tg,hxt4b::GpuCSR&A,const Real*u0,const Real*u1,const Real*u2,Dev<Real>&dg,Dev<Real>&ddg,Dev<Real>&rz,Dev<Real>&wall,Dev<Real>&dwall,Dev<double>&ws,Dev<unsigned long long>&bad,double bulk,double R,double nu,double ms,double sf,
    const TetFacePlan*wt=nullptr,const TetWallQuadCache*wq=nullptr,const TetWallPackedPoint*wm=nullptr){
  if(dg.n)HXT1_CUDA(cudaMemset(dg.p,0,dg.n*sizeof(Real)));if(ddg.n)HXT1_CUDA(cudaMemset(ddg.p,0,ddg.n*sizeof(Real)));if(rz.n)HXT1_CUDA(cudaMemset(rz.p,0,rz.n*sizeof(Real)));if(wall.n)HXT1_CUDA(cudaMemset(wall.p,0,wall.n*sizeof(Real)));if(dwall.n)HXT1_CUDA(cudaMemset(dwall.p,0,dwall.n*sizeof(Real)));if(ws.n)HXT1_CUDA(cudaMemset(ws.p,0,ws.n*sizeof(double)));if(bad.n)HXT1_CUDA(cudaMemset(bad.p,0,bad.n*sizeof(unsigned long long)));if(!H.inH.empty())dg_hex<<<H.inH.size(),64>>>(ih.p,H.inH.size(),pts.p,hc.p,hv.p,A.hexSlot.p,u0,u1,u2,dg.p,ddg.p,rz.p,bulk,R,nu,ms,bad.p);if(!H.inT.empty())dg_tet<<<H.inT.size(),64>>>(it.p,H.inT.size(),pts.p,tc.p,tv.p,tg.p,A.tetSlot.p,u0,u1,u2,dg.p,ddg.p,rz.p,bulk,R,nu,ms);if(!H.wallH.empty())wall_hex_cached<<<H.wallH.size(),64>>>(wh.p,H.wallH.size(),wc.p,hv.p,A.hexSlot.p,u2,wall.p,dwall.p,nu,ws.p);if(!H.wallT.empty())wall_tet_profile5<<<H.wallT.size(),64>>>(wt,(int)H.wallT.size(),wq,wm,tv.p,A.tetSlot.p,u2,wall.p,dwall.p,nu,ws.p,bad.p);HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());unsigned long long q=0;HXT1_CUDA(cudaMemcpy(&q,bad.p,sizeof(q),cudaMemcpyDeviceToHost));if(q)throw std::runtime_error("HXT5B boundary geometry failure count="+std::to_string(q));
}

__global__ void combine(std::size_t n,const Real*b,const Real*c,const Real*t,const Real*dg,const Real*w,Real*xy,Real*z){size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n){Real q=b[i]+c[i]+t[i]+dg[i];xy[i]=q;z[i]=q+w[i];}}
__global__ void finalize(int n,const unsigned char*fixed,const unsigned char*wm,const std::int32_t*dp,Real*xy,Real*z,const Real*drb,const Real*conv,const Real*dt,const Real*ddg,const Real*dw,Real*rxy,Real*rz,Real*dxy,Real*dz,double au,double rs,unsigned long long*bad){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;int d=dp[i];Real ox=xy[d],oz=z[d];dxy[i]=ox;dz[i]=oz;if(fixed[i]){rxy[i]=rz[i]=0;return;}Real den=drb[i]+conv[d]+dt[i]+ddg[i],denz=den+dw[i];if(!(ox>0&&oz>0&&den>0&&denz>0)){atomicAdd(bad,1ULL);rxy[i]=rz[i]=0;return;}xy[d]=ox/au;z[d]=oz/au;rxy[i]=wm[i]?0:rs*au/den;rz[i]=rs*au/denz;
}
__global__ void rhs(int n,const unsigned char*cl,const Real*bt,const Real*extra,const Real*u,const Real*d,double au,Real*r){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)r[i]=cl[i]?0:bt[i]+(extra?extra[i]:0)+((1-au)/au)*d[i]*u[i];}
__global__ void mcgs(int begin,int count,const std::int32_t*rows,const std::int64_t*row,const std::int32_t*col,const Real*val,const std::int32_t*dp,const Real*b,Real*x,double om,const unsigned char*cl){int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=count)return;int i=rows[begin+q];if(cl[i]){x[i]=0;return;}Real off=0;for(auto k=row[i];k<row[i+1];++k)if(col[k]!=i)off+=val[k]*x[col[k]];Real gs=(b[i]-off)/val[dp[i]];x[i]+=om*(gs-x[i]);}
inline void sweep(hxt4b::GpuCSR&A,const hxt4b::ColoringHost&C,const Real*val,const Real*b,Real*x,double om,bool f,const unsigned char*cl){if(f){for(int c=0;c<C.ncolors;++c){int s=C.off[c],n=C.off[c+1]-s;mcgs<<<(n+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(s,n,A.colorRows.p,A.row.p,A.col.p,val,A.diagPos.p,b,x,om,cl);}}else{for(int c=C.ncolors-1;c>=0;--c){int s=C.off[c],n=C.off[c+1]-s;mcgs<<<(n+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(s,n,A.colorRows.p,A.row.p,A.col.p,val,A.diagPos.p,b,x,om,cl);}}}
inline void work(const std::string&m,int it,hxt4b::GpuCSR&A,const hxt4b::ColoringHost&C,const Real*val,const Real*b,Real*x,double om,const unsigned char*cl){if(m=="fgs1")sweep(A,C,val,b,x,om,true,cl);else if(m=="altgs1")sweep(A,C,val,b,x,om,(it&1)!=0,cl);else if(m=="sgs1"){sweep(A,C,val,b,x,om,true,cl);sweep(A,C,val,b,x,om,false,cl);}else throw std::runtime_error("bad momentum work");HXT1_CUDA(cudaGetLastError());}
__global__ void spmv(int n,const std::int64_t*r,const std::int32_t*c,const Real*v,const Real*x,Real*y){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){Real s=0;for(auto k=r[i];k<r[i+1];++k)s+=v[k]*x[c[k]];y[i]=s;}}
__global__ void pres(int n,const unsigned char*cl,const Real*bt,const Real*ex,const Real*u,const Real*d,double au,const Real*Ax,Real*r){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)r[i]=cl[i]?0:bt[i]+(ex?ex[i]:0)-(Ax[i]-((1-au)/au)*d[i]*u[i]);}
inline double pnorm(hxt4b::GpuCSR&A,const Real*val,const unsigned char*cl,const Real*bt,const Real*ex,const Real*u,const Real*d,double au,hxt4a::Reducer<Real>&red){spmv<<<(A.n+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(A.n,A.row.p,A.col.p,val,u,A.tmp.p);pres<<<(A.n+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(A.n,cl,bt,ex,u,d,au,A.tmp.p,A.res.p);return red.norm(A.res.p);}
__global__ void addsrc(int n,const Real*s,Real*x){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]+=s[i];}

__device__ __forceinline__ void h5b_atomic_max_double(double*addr,double v){
  auto*p=(unsigned long long*)addr;unsigned long long old=*p,ass;
  do{ass=old;if(__longlong_as_double((long long)ass)>=v)break;
     old=atomicCAS(p,ass,(unsigned long long)__double_as_longlong(v));}while(ass!=old);
}
__global__ void finite_audit_kernel(std::size_t n,const Real*x,unsigned long long*bad,double*maxabs){
  std::size_t i=(std::size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  double v=(double)x[i];
  if(!isfinite(v))atomicAdd(bad,1ULL);
  else h5b_atomic_max_double(maxabs,fabs(v));
}
struct FiniteAudit { unsigned long long nonfinite=0; double maxabs=0; };
inline FiniteAudit audit_real(const char*label,const Real*x,std::size_t n,bool failOnBad=true){
  Dev<unsigned long long>bad(std::vector<unsigned long long>(1,0));
  Dev<double>mx(std::vector<double>(1,0));
  if(n)finite_audit_kernel<<<(n+255)/256,256>>>(n,x,bad.p,mx.p);
  HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
  FiniteAudit A;
  HXT1_CUDA(cudaMemcpy(&A.nonfinite,bad.p,sizeof(A.nonfinite),cudaMemcpyDeviceToHost));
  HXT1_CUDA(cudaMemcpy(&A.maxabs,mx.p,sizeof(A.maxabs),cudaMemcpyDeviceToHost));
  std::printf("NODALS_HXT5B_FINITE_AUDIT label=%s n=%zu nonfinite=%llu maxAbs=%.12e status=%s\n",
    label,n,A.nonfinite,A.maxabs,A.nonfinite?"FAIL":"PASS");
  if(failOnBad&&A.nonfinite)throw std::runtime_error(std::string("HXT5B nonfinite array at ")+label);
  return A;
}
inline void audit_diag_host(const char*label,const Real*x,int n){
  std::vector<Real>h((std::size_t)n);
  HXT1_CUDA(cudaMemcpy(h.data(),x,(std::size_t)n*sizeof(Real),cudaMemcpyDeviceToHost));
  double mn=std::numeric_limits<double>::infinity(),mx=0;std::size_t bad=0,nonpos=0;
  for(Real q:h){double v=(double)q;if(!std::isfinite(v)){++bad;continue;}if(!(v>0)){++nonpos;continue;}mn=std::min(mn,v);mx=std::max(mx,v);}
  double ratio=(std::isfinite(mn)&&mn>0)?mx/mn:std::numeric_limits<double>::infinity();
  std::printf("NODALS_HXT5B_DIAG_AUDIT label=%s n=%d nonfinite=%zu nonpositive=%zu minPositive=%.12e maxPositive=%.12e maxOverMin=%.12e status=%s\n",
    label,n,bad,nonpos,mn,mx,ratio,(bad||nonpos)?"WARN":"PASS");
}

template<class T>__global__ void scale3(int n,const T*rx,const T*ry,const T*rz,T*x,T*y,T*z){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){x[i]*=rx[i];y[i]*=ry[i];z[i]*=rz[i];}}
template<class T>struct SAniso{hxt4a::BDevice<T>&B;const T*rx,*ry,*rz;int pin;Dev<T>x,y,z;SAniso(hxt4a::BDevice<T>&b,const T*a,const T*c,const T*d,int p):B(b),rx(a),ry(c),rz(d),pin(p),x(std::vector<T>(b.nv,0)),y(std::vector<T>(b.nv,0)),z(std::vector<T>(b.nv,0)){}void apply(const T*p,T*out){hxt4a::BT_apply(B,p,x.p,y.p,z.p);scale3<<<(B.nv+255)/256,256>>>(B.nv,rx,ry,rz,x.p,y.p,z.p);hxt4a::B_apply(B,x.p,y.p,z.p,out);hxt4a::pin_output_zero_kernel<T><<<1,1>>>(pin,out);}};
template<class T>__global__ void sdiag(int np,const std::int64_t*r,const std::int32_t*c,const T*bx,const T*by,const T*bz,const T*rx,const T*ry,const T*rz,T*d,int pin){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=np)return;if(i==pin){d[i]=1;return;}T s=0;for(auto k=r[i];k<r[i+1];++k){int g=c[k];s+=rx[g]*bx[k]*bx[k]+ry[g]*by[k]*by[k]+rz[g]*bz[k]*bz[k];}d[i]=s;}

inline double host_sentry(const hxt3a::BCSR&B,int i,int j,const std::vector<double>&rx,const std::vector<double>&ry,const std::vector<double>&rz){
  auto a=B.row[(std::size_t)i],ae=B.row[(std::size_t)i+1],b=B.row[(std::size_t)j],be=B.row[(std::size_t)j+1];long double q=0;
  while(a<ae&&b<be){int ga=B.col[(std::size_t)a],gb=B.col[(std::size_t)b];if(ga<gb){++a;continue;}if(gb<ga){++b;continue;}q+=(long double)rx[(std::size_t)ga]*B.bx[(std::size_t)a]*B.bx[(std::size_t)b]+(long double)ry[(std::size_t)ga]*B.by[(std::size_t)a]*B.by[(std::size_t)b]+(long double)rz[(std::size_t)ga]*B.bz[(std::size_t)a]*B.bz[(std::size_t)b];++a;++b;}return (double)q;
}
inline void retune_host_fine(hxt4c::HybridFineHost&H,const hxt3a::BCSR&B,int pin,const std::vector<double>&rx,const std::vector<double>&ry,const std::vector<double>&rz){
  if((int)rx.size()!=B.nv||(int)ry.size()!=B.nv||(int)rz.size()!=B.nv)throw std::runtime_error("HXT5B host directional rAU size mismatch");
  if(H.A.n!=B.np)throw std::runtime_error("HXT5B fine CSR row mismatch");
  for(int i=0;i<H.A.n;++i){
    bool haveDiag=false;
    for(auto k=H.A.row[(std::size_t)i];k<H.A.row[(std::size_t)i+1];++k){int j=H.A.col[(std::size_t)k];double v=(i==pin)?(j==pin?1.0:0.0):(j==pin?0.0:host_sentry(B,i,j,rx,ry,rz));H.A.val[(std::size_t)k]=v;if(j==i){H.A.diag[(std::size_t)i]=v;haveDiag=true;}}
    if(!haveDiag||!(H.A.diag[(std::size_t)i]>0.0)||!std::isfinite(H.A.diag[(std::size_t)i]))throw std::runtime_error("HXT5B exact directional setup Schur has invalid diagonal");
  }
  std::printf("NODALS_HXT5B_DIRECTIONAL_SETUP rows=%d nnz=%zu source=BX_RX_BXT_PLUS_BY_RY_BYT_PLUS_BZ_RZ_BZT status=PASS\\n",H.A.n,H.A.val.size());
}

template<class T>__global__ void refresh(std::size_t nnz,const std::int32_t*er,const std::int32_t*ac,const std::int64_t*br,const std::int32_t*bc,const T*bx,const T*by,const T*bz,const T*rx,const T*ry,const T*rz,T*av,int pin){size_t k=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(k>=nnz)return;int i=er[k],j=ac[k];if(i==pin){av[k]=(j==pin);return;}if(j==pin){av[k]=0;return;}auto a=br[i],ae=br[i+1],b=br[j],be=br[j+1];T s=0;while(a<ae&&b<be){int ga=bc[a],gb=bc[b];if(ga<gb){++a;continue;}if(gb<ga){++b;continue;}s+=rx[ga]*bx[a]*bx[b]+ry[ga]*by[a]*by[b]+rz[ga]*bz[a]*bz[b];++a;++b;}av[k]=s;}
template<class T>inline void refresh_pc(hxt4c::HybridAMG<T>&P,hxt4a::BDevice<T>&B,const T*rx,const T*ry,const T*rz){auto&F=P.fine;refresh<<<(F.val.n+255)/256,256>>>(F.val.n,F.entryRow.p,F.col.p,B.row.p,B.col.p,B.bx.p,B.by.p,B.bz.p,rx,ry,rz,F.val.p,F.pin);hxt4c::Buf<unsigned long long>bad(std::vector<unsigned long long>(1,0));hxt4c::diag_l1_kernel<<<(F.n+255)/256,256>>>(F.n,F.row.p,F.diagPos.p,F.val.p,F.diag.p,F.l1.p,bad.p);HXT1_CUDA(cudaDeviceSynchronize());unsigned long long q=0;HXT1_CUDA(cudaMemcpy(&q,bad.p,sizeof(q),cudaMemcpyDeviceToHost));if(q)throw std::runtime_error("HXT5B anisotropic fine Schur invalid");}
template<class T,class S>inline double parity(hxt4c::HybridAMG<T>&P,S&Sop,int n,int pin,hxt4a::Reducer<T>&red){std::vector<T>h(n);for(int i=0;i<n;++i)h[i]=(i==pin)?0:sin(.00131*(i+1))+.2*cos(.00077*(i+1));hxt4c::Buf<T>x(h),a(n),b(n),d(n);Sop.apply(x.p,a.p);P.fine.apply(x.p,b.p);hxt4c::diff_kernel<<<(n+255)/256,256>>>(n,a.p,b.p,d.p);return red.norm(d.p)/std::max(red.norm(a.p),1e-300);}


// HXT5B3_ROW_L1_IMPLEMENTATION
// Mirrors the validated turbulent NodalS momentum policy:
//   relaxMetric_i = sum_j |A_ij|
//   delta_i       = (1/alphaU - 1) * relaxMetric_i
//   Arel_ii       = Aphys_ii + delta_i
// while retaining the hybrid-specific pressure rule that the direct Nitsche
// penalty is excluded from the physical rAU denominator.

__global__ void h5b3_finalize_row_l1(
    int n,const unsigned char*fixed,const unsigned char*wallMask,
    const std::int64_t*row,const std::int32_t*diagPos,
    Real*xy,Real*z,
    const Real*diagRBase,const Real*conv,const Real*diagRTurb,
    const Real*diagDG,const Real*diagWall,
    Real*rxy,Real*rz,Real*diagOrigXY,Real*diagOrigZ,
    Real*relaxDeltaXY,Real*relaxDeltaZ,
    double momNitscheGamma,double schurNitscheGamma,
    double alphaU,double rauScale,unsigned long long*bad)
{
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i>=n)return;
  const int d=diagPos[i];
  const Real ox=xy[d],oz=z[d];
  diagOrigXY[i]=ox;diagOrigZ[i]=oz;

  Real l1x=Real(0),l1z=Real(0);
  for(std::int64_t k=row[i];k<row[i+1];++k){
    l1x+=(Real)fabs((double)xy[k]);
    l1z+=(Real)fabs((double)z[k]);
  }
  const Real fac=(Real)(1.0/alphaU-1.0);
  const Real dx=fac*l1x,dz=fac*l1z;
  relaxDeltaXY[i]=dx;relaxDeltaZ[i]=dz;

  if(fixed[i]){
    rxy[i]=rz[i]=Real(0);
    return;
  }

  // Penalty-free physical diagonal.  It is allowed to be locally non-positive:
  // row-L1 equation relaxation is part of the actual rAU denominator.
  const Real noPenXY=diagRBase[i]+conv[d]+diagRTurb[i]+diagDG[i];
  const Real noPenZ =noPenXY+diagWall[i];

  // The difference between the full assembled diagonal and noPen is exactly the
  // direct HEX/TET Nitsche penalty contribution (molecular + current SST part).
  // Since that contribution is linear in gamma, an independent Schur gamma can
  // admit any desired fraction without changing the momentum operator.
  const Real penXY=ox-noPenXY;
  const Real penZ =oz-noPenZ;
  const Real schurScale=(momNitscheGamma>0.0)
      ? (Real)(schurNitscheGamma/momNitscheGamma) : Real(0);
  const Real schurPenXY=schurScale*penXY;
  const Real schurPenZ =schurScale*penZ;

  // The momentum equation is solved with row-L1 relaxed diagonals.
  // A raw (unrelaxed) diagonal may be negative in convection-dominated
  // P1+BF3 wall rows.  Do not reject it if the actual relaxed momentum
  // diagonal and the pressure Schur denominator are both valid.
  const Real relaxedXY=ox+dx,relaxedZ=oz+dz;
  if(!(relaxedXY>Real(0))||!(relaxedZ>Real(0))||
     !(l1x>Real(0))||!(l1z>Real(0))||
     !isfinite((double)ox)||!isfinite((double)oz)||
     !isfinite((double)dx)||!isfinite((double)dz)||
     !isfinite((double)noPenXY)||!isfinite((double)noPenZ)||
     !isfinite((double)penXY)||!isfinite((double)penZ)||
     !isfinite((double)schurPenXY)||!isfinite((double)schurPenZ)||
     (schurNitscheGamma>0.0 && (!(penXY>=Real(0))||!(penZ>=Real(0))))){
    atomicAdd(bad,1ULL);rxy[i]=rz[i]=Real(0);return;
  }

  xy[d]=relaxedXY;
  z[d] =relaxedZ;

  // Validate the denominator actually used by the Schur approximation.
  // With schurNitscheGamma=0 this is noPen + row-L1 delta, preserving the
  // previous pressure approximation except for removing the premature raw-noPen gate.
  const Real mx=noPenXY+schurPenXY+dx,mz=noPenZ+schurPenZ+dz;
  if(!(mx>Real(0))||!(mz>Real(0))||
     !isfinite((double)mx)||!isfinite((double)mz)){
    atomicAdd(bad,1ULL);rxy[i]=rz[i]=Real(0);return;
  }
  rxy[i]=wallMask[i]?Real(0):(Real)(rauScale/(double)mx);
  rz[i]=(Real)(rauScale/(double)mz);
}

__global__ void s22b_finalize_row_l1(
    int n,const unsigned char*fixed,const unsigned char*wallMask,
    const std::int64_t*row,const std::int32_t*diagPos,
    Real*xy,Real*z,
    const Real*diagRBase,const Real*conv,const Real*diagRTurb,
    const Real*diagDG,const Real*diagWall,
    Real*rxy,Real*rz,Real*diagOrigXY,Real*diagOrigZ,
    Real*relaxDeltaXY,Real*relaxDeltaZ,
    double momNitscheGamma,double schurNitscheGamma,
    double alphaU,double rauScale,unsigned long long*bad)
{
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i>=n)return;
  const int d=diagPos[i];
  const Real ox=xy[d],oz=z[d];
  diagOrigXY[i]=ox;diagOrigZ[i]=oz;

  Real l1x=Real(0),l1z=Real(0);
  for(std::int64_t k=row[i];k<row[i+1];++k){
    l1x+=(Real)fabs((double)xy[k]);
    l1z+=(Real)fabs((double)z[k]);
  }
  const Real fac=(Real)(1.0/alphaU-1.0);
  const Real dx=fac*l1x,dz=fac*l1z;
  relaxDeltaXY[i]=dx;relaxDeltaZ[i]=dz;

  if(fixed[i]){
    rxy[i]=rz[i]=Real(0);
    return;
  }

  // Penalty-free physical diagonal.  It is allowed to be locally non-positive:
  // row-L1 equation relaxation is part of the actual rAU denominator.
  const Real noPenXY=diagRBase[i]+conv[d]+diagRTurb[i]+diagDG[i]+diagWall[i];
  const Real noPenZ =noPenXY;

  // The difference between the full assembled diagonal and noPen is exactly the
  // direct HEX/TET Nitsche penalty contribution (molecular + current SST part).
  // Since that contribution is linear in gamma, an independent Schur gamma can
  // admit any desired fraction without changing the momentum operator.
  const Real penXY=ox-noPenXY;
  const Real penZ =oz-noPenZ;
  const Real schurScale=(momNitscheGamma>0.0)
      ? (Real)(schurNitscheGamma/momNitscheGamma) : Real(0);
  const Real schurPenXY=schurScale*penXY;
  const Real schurPenZ =schurScale*penZ;

  // The momentum equation is solved with row-L1 relaxed diagonals.
  // A raw (unrelaxed) diagonal may be negative in convection-dominated
  // P1+BF3 wall rows.  Do not reject it if the actual relaxed momentum
  // diagonal and the pressure Schur denominator are both valid.
  const Real relaxedXY=ox+dx,relaxedZ=oz+dz;
  if(!(relaxedXY>Real(0))||!(relaxedZ>Real(0))||
     !(l1x>Real(0))||!(l1z>Real(0))||
     !isfinite((double)ox)||!isfinite((double)oz)||
     !isfinite((double)dx)||!isfinite((double)dz)||
     !isfinite((double)noPenXY)||!isfinite((double)noPenZ)||
     !isfinite((double)penXY)||!isfinite((double)penZ)||
     !isfinite((double)schurPenXY)||!isfinite((double)schurPenZ)||
     (schurNitscheGamma>0.0 && (!(penXY>=Real(0))||!(penZ>=Real(0))))){
    atomicAdd(bad,1ULL);rxy[i]=rz[i]=Real(0);return;
  }

  xy[d]=relaxedXY;
  z[d] =relaxedZ;

  // Validate the denominator actually used by the Schur approximation.
  // With schurNitscheGamma=0 this is noPen + row-L1 delta, preserving the
  // previous pressure approximation except for removing the premature raw-noPen gate.
  const Real mx=noPenXY+schurPenXY+dx,mz=noPenZ+schurPenZ+dz;
  if(!(mx>Real(0))||!(mz>Real(0))||
     !isfinite((double)mx)||!isfinite((double)mz)){
    atomicAdd(bad,1ULL);rxy[i]=rz[i]=Real(0);return;
  }
  rxy[i]=(Real)(rauScale/(double)mx);
  rz[i]=(Real)(rauScale/(double)mz);
}


__global__ void h5b3_rhs(
    int n,const unsigned char*clamp,const Real*bt,const Real*extra,
    const Real*uold,const Real*relaxDelta,Real*rhs)
{
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i<n)rhs[i]=clamp[i]?Real(0):
    bt[i]+(extra?extra[i]:Real(0))+relaxDelta[i]*uold[i];
}

__global__ void h5b3_rhs2(
    int n,const unsigned char*clamp,const Real*bt,const Real*extra0,const Real*extra1,
    const Real*uold,const Real*relaxDelta,Real*rhs)
{
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i<n)rhs[i]=clamp[i]?Real(0):
    bt[i]+(extra0?extra0[i]:Real(0))+(extra1?extra1[i]:Real(0))+relaxDelta[i]*uold[i];
}

__global__ void h5b3_residual(
    int n,const unsigned char*clamp,const Real*b,const Real*Ax,Real*r)
{
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i<n)r[i]=clamp[i]?Real(0):(b[i]-Ax[i]);
}

inline double h5b3_relaxed_residual_norm(
    hxt4b::GpuCSR&A,const Real*val,const unsigned char*clamp,
    const Real*b,const Real*x,hxt4a::Reducer<Real>&red)
{
  hxt5b::spmv<<<(A.n+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(
    A.n,A.row.p,A.col.p,val,x,A.tmp.p);
  h5b3_residual<<<(A.n+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(
    A.n,clamp,b,A.tmp.p,A.res.p);
  HXT1_CUDA(cudaGetLastError());
  return red.norm(A.res.p);
}

struct H5B3MomentumResult {
  int sweeps=0;
  double initial=0.0;
  double final=0.0;
  double rel=0.0;
  bool ok=false;
};

inline H5B3MomentumResult h5b3_momentum_solve(
    const std::string&mode,int outer,hxt4b::GpuCSR&A,
    const hxt4b::ColoringHost&C,const Real*val,const Real*b,Real*x,
    double omega,const unsigned char*clamp,double rtol,double atol,int maxSweeps,
    hxt4a::Reducer<Real>&red,const char*component)
{
  H5B3MomentumResult R;
  R.initial=h5b3_relaxed_residual_norm(A,val,clamp,b,x,red);
  if(!std::isfinite(R.initial))
    throw std::runtime_error(std::string("HXT5B3 nonfinite initial momentum residual comp=")+component);
  if(R.initial<=atol){R.final=R.initial;R.rel=0.0;R.ok=true;return R;}
  const double target=std::max(atol,rtol*R.initial);

  for(int q=0;q<maxSweeps;++q){
    if(mode=="sgs1"){
      hxt5b::sweep(A,C,val,b,x,omega,true,clamp);
      hxt5b::sweep(A,C,val,b,x,omega,false,clamp);
    }else if(mode=="fgs1"){
      hxt5b::sweep(A,C,val,b,x,omega,true,clamp);
    }else if(mode=="altgs1"){
      hxt5b::sweep(A,C,val,b,x,omega,((outer+q)&1)!=0,clamp);
    }else throw std::runtime_error("HXT5B3 bad momentum work mode");

    R.sweeps=q+1;
    R.final=h5b3_relaxed_residual_norm(A,val,clamp,b,x,red);
    R.rel=R.final/std::max(R.initial,1e-300);
    if(!std::isfinite(R.final))
      throw std::runtime_error(std::string("HXT5B3 momentum SGS produced NaN/Inf comp=")+component);
    if(outer==1 && q<6)
      std::printf("NODALS_HXT5B3_MOM_SWEEP comp=%s sweep=%d initial=%.12e current=%.12e rel=%.12e targetRel=%.6g status=PASS\n",
        component,q+1,R.initial,R.final,R.rel,rtol);
    if(R.final<=target){R.ok=true;break;}
  }
  return R;
}

__global__ void h5b3_physical_residual(
    int n,const unsigned char*clamp,const Real*bt,const Real*extra,
    const Real*u,const Real*relaxDelta,const Real*ArelU,Real*r)
{
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i<n){
    if(clamp[i]){r[i]=Real(0);return;}
    r[i]=bt[i]+(extra?extra[i]:Real(0))-(ArelU[i]-relaxDelta[i]*u[i]);
  }
}

inline double h5b3_physical_residual_norm(
    hxt4b::GpuCSR&A,const Real*val,const unsigned char*clamp,
    const Real*bt,const Real*extra,const Real*u,const Real*relaxDelta,
    hxt4a::Reducer<Real>&red)
{
  hxt5b::spmv<<<(A.n+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(
    A.n,A.row.p,A.col.p,val,u,A.tmp.p);
  h5b3_physical_residual<<<(A.n+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(
    A.n,clamp,bt,extra,u,relaxDelta,A.tmp.p,A.res.p);
  HXT1_CUDA(cudaGetLastError());
  return red.norm(A.res.p);
}

__global__ void h5b3_physical_residual2(
    int n,const unsigned char*clamp,const Real*bt,const Real*extra0,const Real*extra1,
    const Real*u,const Real*relaxDelta,const Real*ArelU,Real*r)
{
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i<n){
    if(clamp[i]){r[i]=Real(0);return;}
    r[i]=bt[i]+(extra0?extra0[i]:Real(0))+(extra1?extra1[i]:Real(0))-(ArelU[i]-relaxDelta[i]*u[i]);
  }
}

inline double h5b3_physical_residual_norm2(
    hxt4b::GpuCSR&A,const Real*val,const unsigned char*clamp,
    const Real*bt,const Real*extra0,const Real*extra1,const Real*u,const Real*relaxDelta,
    hxt4a::Reducer<Real>&red)
{
  hxt5b::spmv<<<(A.n+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(
    A.n,A.row.p,A.col.p,val,u,A.tmp.p);
  h5b3_physical_residual2<<<(A.n+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(
    A.n,clamp,bt,extra0,extra1,u,relaxDelta,A.tmp.p,A.res.p);
  HXT1_CUDA(cudaGetLastError());
  return red.norm(A.res.p);
}

__global__ void h5b3_subtract(int n,const Real*a,const Real*b,Real*r){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i<n)r[i]=a[i]-b[i];
}

inline void h5b3_relax_audit(
    const char*label,const Real*diagOrig,const Real*delta,int n)
{
  std::vector<Real>d((std::size_t)n),q((std::size_t)n);
  HXT1_CUDA(cudaMemcpy(d.data(),diagOrig,(std::size_t)n*sizeof(Real),cudaMemcpyDeviceToHost));
  HXT1_CUDA(cudaMemcpy(q.data(),delta,(std::size_t)n*sizeof(Real),cudaMemcpyDeviceToHost));
  double minRatio=std::numeric_limits<double>::infinity(),maxRatio=0,mean=0;std::size_t used=0;
  for(int i=0;i<n;++i){
    const double a=(double)d[(std::size_t)i],b=(double)q[(std::size_t)i];
    if(a>0&&std::isfinite(a)&&std::isfinite(b)){
      const double r=b/a;minRatio=std::min(minRatio,r);maxRatio=std::max(maxRatio,r);mean+=r;++used;
    }
  }
  if(used)mean/=used;
  std::printf("NODALS_HXT5B3_ROWL1_AUDIT label=%s rows=%zu deltaOverPhysDiagMin=%.12e mean=%.12e max=%.12e status=PASS\n",
    label,used,minRatio,mean,maxRatio);
}

} // namespace hxt5b
