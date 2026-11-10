#include "wall_sampler.hpp"
#include "tet_ray_trace.hpp"
#include "spalding_active.hpp"
#include <mpi.h>
#include <array>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <string>

namespace nodals::cpu_mc::m3 {
namespace {
using namespace nodals::cpu_mc::m2;
using P=std::array<double,3>;
using T=std::array<int,4>;
constexpr double L0[12]={.063089014491502,.063089014491502,.873821971016996,.249286745170910,.249286745170910,.501426509658180,.053145049844816,.053145049844816,.310352451033785,.310352451033785,.636502499121399,.636502499121399};
constexpr double L1[12]={.063089014491502,.873821971016996,.063089014491502,.249286745170910,.501426509658180,.249286745170910,.310352451033785,.636502499121399,.053145049844816,.636502499121399,.053145049844816,.310352451033785};
constexpr double L2[12]={.873821971016996,.063089014491502,.063089014491502,.501426509658180,.249286745170910,.249286745170910,.636502499121399,.310352451033785,.636502499121399,.053145049844816,.310352451033785,.053145049844816};
constexpr double QW[12]={.050844906370207,.050844906370207,.050844906370207,.116786275726379,.116786275726379,.116786275726379,.082851075618374,.082851075618374,.082851075618374,.082851075618374,.082851075618374,.082851075618374};
constexpr double GX4[4]={-.86113631159405257522,-.33998104358485626480,.33998104358485626480,.86113631159405257522};
constexpr double GW4[4]={.34785484513745385737,.65214515486254614263,.65214515486254614263,.34785484513745385737};
P p(Vec3 v){return {v.x,v.y,v.z};}
Vec3 v(P q){return {q[0],q[1],q[2]};}
P mix(P a,P b,double t){return {a[0]+t*b[0],a[1]+t*b[1],a[2]+t*b[2]};}
void basis(const std::array<double,4>&l,double ph[8]){
 for(int i=0;i<4;++i)ph[i]=l[i];
 for(int i=0;i<4;++i){double b=27.;for(int j=0;j<4;++j)if(j!=i)b*=l[j];ph[4+i]=b;}
}
Vec3 interpolate(const Mesh&m,const std::vector<double>&u,int c,const std::array<double,4>&lam){
 const int nv=(int)m.points.size();double ph[8];basis(lam,ph);
 const auto&t=m.tets[c];Vec3 out{};
 for(int a=0;a<8;++a){const int e=a<4?t.v[a]:nv+t.face[a-4];
   out.x+=ph[a]*u[(std::size_t)3*e];
   out.y+=ph[a]*u[(std::size_t)3*e+1];
   out.z+=ph[a]*u[(std::size_t)3*e+2];
 }return out;
}
}
WallFaceFit sample_spalding_faces(const m2::Mesh&m,const m2::Ownership&own,
  int rank,const CaseConfig&cfg,const std::vector<double>&lag){
 if(cfg.wallSamples!=1)throw std::runtime_error("M3 wall sampler currently requires wall_function.samples=1, matching captured spiral case");
 using ::nodals::wall::ray::TetRayTrace;
 using ::nodals::wall::vector3::V3;
 const int nv=(int)m.points.size(),ne=nv+(int)m.faces.size();
 if(lag.size()!=(std::size_t)3*ne)throw std::runtime_error("wall ray sampler: velocity field size mismatch");
 std::vector<P>pts;pts.reserve(m.points.size());for(Vec3 x:m.points)pts.push_back(p(x));
 std::vector<T>tets;tets.reserve(m.tets.size());for(const auto&t:m.tets)tets.push_back(t.v);
 const TetRayTrace trace(std::move(pts),std::move(tets));
 WallFaceFit out;out.beta.assign(m.faces.size(),0.0);out.uTau.assign(m.faces.size(),0.0);out.yPlus.assign(m.faces.size(),0.0);
 out.qBeta.assign(12*m.faces.size(),0.0);out.qUTau.assign(12*m.faces.size(),0.0);out.qYPlus.assign(12*m.faces.size(),0.0);
 const int nf=(int)m.faces.size();
 for(int f=0;f<nf;++f){
  const auto&face=m.faces[f];if(face.patch!=3||own.cell[face.cell0]!=rank)continue;
  const int c=face.cell0;const auto&t=m.tets[c];
  int lf=-1;for(int j=0;j<4;++j)if(t.face[j]==f)lf=j;
  if(lf<0)throw std::runtime_error("wall sampler face topology invalid");
  int a[3],idx=0;for(int j=0;j<4;++j)if(j!=lf)a[idx++]=j;
  const Vec3 area=outward_area(m,c,lf);const double amag=std::sqrt(dot(area,area));
  if(!(amag>0))throw std::runtime_error("wall sampler degenerate face");
  const Vec3 normal=(1./amag)*area,inward=-1.*normal;
  const P inwardP=p(inward);
  // GPU f2_face_averaged_spalding: form the physical-area weighted VECTOR
  // averages first, then call the nonlinear Spalding law ONCE per face.
  // In contrast, omega trace_log deliberately continues to use its separate
  // per-quadrature uTau values (qUTau): F2 only changes momentum friction.
  V3 faceSamples[5]{};
  V3 facePic{};
  double faceHeights[5]{};
  double sumW=0.;
  for(int q=0;q<12;++q){
   const double w=QW[q]*amag;sumW+=w;
   const Vec3 origin=L0[q]*m.points[t.v[a[0]]]+L1[q]*m.points[t.v[a[1]]]+L2[q]*m.points[t.v[a[2]]];
   const P originP=p(origin);
   const auto bary0=trace.bary(c,originP);
   // GPU Picard carrier: true line average over [0,0.9*first-tet-exit].
   const double dh=1e-7*std::cbrt(6*t.volume);
   const auto bary1=trace.bary(c,mix(originP,inwardP,dh));
   std::array<double,4> dr{};
   double exit=1e300;
   for(int j=0;j<4;++j){dr[j]=(bary1[j]-bary0[j])/dh;
    if(dr[j]<-1e-15)exit=std::min(exit,-bary0[j]/dr[j]);}
   if(!(exit>0) || !std::isfinite(exit))throw std::runtime_error("wall sampler invalid first-tet exit");
   Vec3 pic{};
   for(int k=0;k<4;++k){const double s=.45*exit*(1.+GX4[k]);
     auto bary=bary0;for(int j=0;j<4;++j)bary[j]+=s*dr[j];
     pic=pic+(.5*GW4[k])*interpolate(m,lag,c,bary);
   }
   double y=cfg.spaldingY3;
   ::nodals::wall::ray::Hit hit;
   try{hit=trace.trace(c,originP,inwardP,{y},cfg.maxCrossings,cfg.baryTolerance).front();}
   catch(const std::exception&e){
     if(!cfg.localRayAdapt)throw std::runtime_error("M4 RAY_TRACE wall face="+std::to_string(f)+" q="+std::to_string(q)+" "+e.what());
     double lo=0,hi=y;for(int it=0;it<28;++it){const double mid=.5*(lo+hi);
       try{(void)trace.trace(c,originP,inwardP,{mid},cfg.maxCrossings,cfg.baryTolerance);lo=mid;}
       catch(const std::exception&){hi=mid;}}
     const double chosen=std::min(y,.9*lo);
     if(!(chosen>=.25*y && chosen<y))throw std::runtime_error("M4 LOCAL_RAY_UNSAFE face="+std::to_string(f));
     y=chosen;hit=trace.trace(c,originP,inwardP,{y},cfg.maxCrossings,cfg.baryTolerance).front();++out.shortened;
   }
   const Vec3 uv=interpolate(m,lag,hit.cell,hit.bary);
   V3 samples[5]{};samples[2]=V3{uv.x,uv.y,uv.z};
   const double heights[5]={y/3.,2.*y/3.,y,4.*y/3.,5.*y/3.};
   const V3 n{normal.x,normal.y,normal.z},pbar{pic.x,pic.y,pic.z};
   auto fit=nodals::wall::vector3::fit_active(samples,heights,n,pbar,V3{0,0,0},cfg.nu,1);
   if(cfg.zeroLimit && fit.uTau==0.0)fit.beta=cfg.nu/y;
   if(!fit.valid || !(fit.beta>0)||!std::isfinite(fit.beta))
    throw std::runtime_error("M4 pointwise Spalding fit invalid face="+std::to_string(f)+" q="+std::to_string(q));
   const size_t iq=(size_t)f*12u+(size_t)q;
   out.qBeta[iq]=fit.beta;out.qUTau[iq]=fit.uTau;out.qYPlus[iq]=fit.yPlus;
   // The GPU F2 fit averages vector samples, never the scalar speeds,
   // uTau or pointwise beta. Active slots for samples=1 contain only slot 2.
   faceSamples[2].x+=w*uv.x;
   faceSamples[2].y+=w*uv.y;
   faceSamples[2].z+=w*uv.z;
   faceHeights[2]+=w*y;
   facePic.x+=w*pic.x;
   facePic.y+=w*pic.y;
   facePic.z+=w*pic.z;
   ++out.rays;if(hit.cell!=c)++out.crossed;
  }
  if(!(sumW>0))throw std::runtime_error("wall sampler zero quadrature area");
  const double invW=1.0/sumW;
  faceSamples[2].x*=invW;
  faceSamples[2].y*=invW;
  faceSamples[2].z*=invW;
  faceHeights[2]*=invW;
  facePic.x*=invW;
  facePic.y*=invW;
  facePic.z*=invW;
  // Inactive heights are not used by fit_active(count=1); keep them
  // positive and increasing for clarity and future diagnostic support.
  for(int i=0;i<5;++i)if(i!=2)faceHeights[i]=faceHeights[2]*(i+1)/3.0;
  const V3 n{normal.x,normal.y,normal.z};
  auto faceFit=nodals::wall::vector3::fit_active(
      faceSamples,faceHeights,n,facePic,V3{0,0,0},cfg.nu,cfg.wallSamples);
  if(cfg.zeroLimit && faceFit.uTau==0.0)faceFit.beta=cfg.nu/faceHeights[2];
  if(!faceFit.valid || !(faceFit.beta>0.) || !std::isfinite(faceFit.beta) ||
     !(faceFit.uTau>=0.) || !std::isfinite(faceFit.uTau) || !std::isfinite(faceFit.yPlus))
    throw std::runtime_error("M4 F2 face-averaged Spalding fit invalid face="+std::to_string(f));
  out.beta[f]=faceFit.beta;
  out.uTau[f]=faceFit.uTau;
  out.yPlus[f]=faceFit.yPlus;
  // Momentum assembly uses qBeta. Every quadrature point now receives
  // the same face F2 coefficient, as in the GPU f2FaceFit lookup.
  // The independent omega trace still uses qUTau/qYPlus unchanged.
  for(int q=0;q<12;++q)out.qBeta[(std::size_t)f*12u+(std::size_t)q]=faceFit.beta;
 }
 MPI_Allreduce(MPI_IN_PLACE,out.beta.data(),nf,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
 MPI_Allreduce(MPI_IN_PLACE,out.uTau.data(),nf,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
 MPI_Allreduce(MPI_IN_PLACE,out.yPlus.data(),nf,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
 MPI_Allreduce(MPI_IN_PLACE,out.qBeta.data(),nf*12,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
 MPI_Allreduce(MPI_IN_PLACE,out.qUTau.data(),nf*12,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
 MPI_Allreduce(MPI_IN_PLACE,out.qYPlus.data(),nf*12,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
 double local[3]={(double)out.rays,(double)out.crossed,(double)out.shortened},global[3]={};
 MPI_Allreduce(local,global,3,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
 out.rays=(std::size_t)global[0];out.crossed=(std::size_t)global[1];out.shortened=(std::size_t)global[2];
 if(rank==0)std::printf("NODALS_CPU_SST_M3_WALL_F2 status=PASS wallFaces=%zu tracedRays=%zu crossed=%zu shortened=%zu samples=1 fit=GPU_F2_FACE_AVERAGED_VECTOR momentumBeta=ONE_FIT_PER_FACE omegaTrace=QUADRATURE_VALUES source=UNCHANGED_GPU_SPALDING_ACTIVE\n",m.boundary[3],out.rays,out.crossed,out.shortened);
 return out;
}
}
