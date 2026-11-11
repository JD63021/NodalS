#pragma once
// NodalS DG-P1 omega: true SST-2003m constitutive coefficients on a pure-tet mesh.
//
// This kernel reads the *native* P1+BF3 velocity, native masked-Q1 k and
// independent cell-local DG omega. No CG omega projection is involved.
// It evaluates the original NodalS SST-2003m physics at four physical
// degree-two tetra quadrature points, assembling the complete local 4x4
// volume matrix and RHS. Face terms still use cell-averaged mu and velocity
// via dg_omega_bsr4.cuh (temporary face coefficient approximation).
//
// Requires actual geometry-specific wall distance at each mesh vertex.
// The existing G2 Stage20 wall-distance field can be supplied directly.
// Never silently use the pipe radius for an arbitrary U-bend.
// This is an opt-in standalone kernel, not a live G2 dispatch yet.
#include "dg_omega_bsr4.cuh"
#include "../physics/sst2003m_distance_core.hpp"
#include "../../../hybrid_hxt1/hxt1_mesh.hpp"
#include <cuda_runtime.h>
#include <cmath>
namespace nodals {namespace omega_dg_gpu {
struct PhysicalSSTConfig {
  double radius,nu,kFloor,omegaFloor;
  int deviatoricProduction;
};
struct PhysicsAudit {
  unsigned long long invalid;
  unsigned long long omegaFloored;
  unsigned long long kFloored;
};
__device__ __forceinline__ void native_tet_u(
 const nodals_hxt1::TetVel &vel,const Cell&g,const double lambda[4],
 const double*u0,const double*u1,const double*u2,
 double U[3],double gradU[9]){
 for(int d=0;d<3;++d)U[d]=0.;
 for(int j=0;j<9;++j)gradU[j]=0.;
 for(int a=0;a<8;++a){
   double shape=0.,gr[3]{0.,0.,0.};
   if(a<4){shape=lambda[a];gr[0]=g.grad[a].x;gr[1]=g.grad[a].y;gr[2]=g.grad[a].z;}
   else {
     const int excluded=a-4;
     // NodalS tet bubble #excluded: 27 * product of other 3 barycentric coords.
     double product=27.;
     for(int j=0;j<4;++j)if(j!=excluded)product*=lambda[j];
     shape=product;
     for(int d=0;d<3;++d){
       double dg=0.;
       for(int j=0;j<4;++j)if(j!=excluded){
         double v=27.;
         for(int h=0;h<4;++h)if(h!=excluded&&h!=j)v*=lambda[h];
         const Vec3 gj=g.grad[j];
         dg+=v*(d==0?gj.x:(d==1?gj.y:gj.z));
       }
       gr[d]=dg;
     }
   }
   const int id=vel.g[a];
   const double val[3]={u0[id],u1[id],u2[id]};
   for(int d=0;d<3;++d){
     U[d]+=shape*val[d];
     for(int j=0;j<3;++j)gradU[3*d+j]+=val[d]*gr[j];
   }
 }
}
__device__ __forceinline__ bool quadrature_sst(
 int t,const double lambda[4],const Cell&g,
 const nodals_hxt1::TetConn*tc,const nodals_hxt1::TetVel*tv,
 const double*u0,const double*u1,const double*u2,
 const double*kCG,const double*oldOmegaDG,const double*wallDistance,
 const PhysicalSSTConfig opt,double&mu,double&react,double&src,
 Vec3&v,PhysicsAudit*audit){
 double k=0.,w=0.,gk[3]{0.,0.,0.},gw[3]{0.,0.,0.},d=0.;
 for(int i=0;i<4;++i){
   const int vertex=tc[t].v[i];
   if(vertex<0)return false;
   const double kv=kCG[vertex],wv=oldOmegaDG[4*t+i],dist=wallDistance[vertex];
   if(!isfinite(kv)||!isfinite(wv)||!(dist>=0.)||!isfinite(dist))return false;
   k+=lambda[i]*kv;w+=lambda[i]*wv;d+=lambda[i]*dist;
   const Vec3 gradient=g.grad[i];
   const double gg[3]={gradient.x,gradient.y,gradient.z};
   for(int j=0;j<3;++j){gk[j]+=kv*gg[j];gw[j]+=wv*gg[j];}
 }
 double u[3],gradU[9];
 native_tet_u(tv[t],g,lambda,u0,u1,u2,u,gradU);
 for(int j=0;j<3;++j)if(!isfinite(u[j]))return false;
 for(int j=0;j<9;++j)if(!isfinite(gradU[j]))return false;
 const auto s=::nodals::sst2003m::evaluate_at_distance<double>(
    d,opt.radius,opt.nu,gradU,k,w,gk,gw,
    opt.kFloor,opt.omegaFloor,opt.deviatoricProduction);
 if(!(s.nut>=0.)||!isfinite(s.nut)||!(s.sigmaW>0.)||
    !isfinite(s.reactOmega)||!isfinite(s.srcOmega))return false;
 mu=opt.nu+s.sigmaW*s.nut;
 react=s.reactOmega;
 src=s.srcOmega;
 v={u[0],u[1],u[2]};
 if(audit){
   if(s.omegaFloored)atomicAdd(&audit->omegaFloored,1ULL);
   if(s.kFloored)atomicAdd(&audit->kFloored,1ULL);
 }
 return isfinite(mu)&&isfinite(react)&&isfinite(src)&&mu>=opt.nu;
}
__global__ void sst_dg_omega_volume(
 int nt,Cell*c,const nodals_hxt1::TetConn*tc,
 const nodals_hxt1::TetVel*tv,const double*u0,const double*u1,
 const double*u2,const double*kCG,const double*oldOmegaDG,
 const double*wallDistance,const int*diagonalSlots,
 double*blocks,double*rhs,PhysicalSSTConfig opt,PhysicsAudit*audit){
 const int t=blockIdx.x*blockDim.x+threadIdx.x;
 if(t>=nt)return;
 if(!(opt.nu>0.)||!(opt.kFloor>0.)||!(opt.omegaFloor>0.)||
    !wallDistance||!oldOmegaDG||!kCG||!tc||!tv){
   atomicAdd(&audit->invalid,1ULL);return;
 }
 Cell g=c[t];
 constexpr double a=0.58541019662496845446;
 constexpr double b=0.13819660112501051518;
 double local[16]{},localRhs[4]{},sumMu=0.,sumReact=0.,sumSource=0.;
 Vec3 sumU{};
 for(int q=0;q<4;++q){
   double lambda[4]={b,b,b,b};lambda[q]=a;
   double mu=0.,reaction=0.,source=0.;Vec3 velocity{};
   if(!quadrature_sst(t,lambda,g,tc,tv,u0,u1,u2,kCG,
                     oldOmegaDG,wallDistance,opt,mu,reaction,source,velocity,audit)){
     atomicAdd(&audit->invalid,1ULL);return;
   }
   sumMu+=mu*.25;sumReact+=reaction*.25;sumSource+=source*.25;
   sumU.x+=velocity.x*.25;sumU.y+=velocity.y*.25;sumU.z+=velocity.z*.25;
   const double weight=g.volume*.25;
   for(int i=0;i<4;++i){
     localRhs[i]+=weight*source*lambda[i];
     const Vec3 gi=g.grad[i];
     for(int j=0;j<4;++j){
       const Vec3 gj=g.grad[j];
       local[4*i+j]+=weight*(mu*dot(gi,gj)
          +reaction*lambda[i]*lambda[j]
          +lambda[i]*dot(velocity,gj));
     }
   }
 }
 g.mu=sumMu;g.reaction=sumReact;g.source=sumSource;g.u=sumU;
 c[t]=g;
 const int base=16*diagonalSlots[t];
 for(int i=0;i<4;++i){
   rhs[4*t+i]=localRhs[i];
   for(int j=0;j<4;++j)blocks[base+4*i+j]=local[4*i+j];
 }
}
}} // namespace nodals::omega_dg_gpu
