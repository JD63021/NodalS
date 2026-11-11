#pragma once
// Physical-wall omega DG-P1 symmetric Nitsche at the ORIGINAL 12-point
// tetra face quadrature. Target omega and effective diffusivity are supplied
// by the existing adaptive five-point Spalding / of_auto point-3 kernel.
// There is NO replacement wall law and no CG projection into the physics.
#include "dg_omega_bsr4.cuh"
namespace nodals { namespace omega_dg_gpu {
__global__ void physical_trace_log_wall_q12(
  int nf,const hxt5b::TetFacePlan*wallPlans,const int*wallToBlock,
  const nodals_hxt1::TetGeom*geo,
  const int*diagonalSlots,const double*targetQ12,const double*muQ12,
  double traceGamma,double*blocks,double*rhs,unsigned int*bad){
 const int wall=(int)blockIdx.x*blockDim.x+threadIdx.x;
 if(wall>=nf)return;
 const auto f=wallPlans[wall];
 const int cell=f.cell,slot=diagonalSlots[cell];
 const double h=geo[cell].det/(2.*f.area);
 if(!(h>0.)||!(f.area>0.)||wallToBlock[wall]<0){
   atomicAdd(bad,1u);return;
 }
 for(int q=0;q<12;++q){
   const double target=targetQ12[(std::size_t)wall*12+q];
   const double mu=muQ12[(std::size_t)wall*12+q];
   const double tau=traceGamma*mu/h;
   const double w=hxt5b::QW12[q]*f.area;
   if(!(target>0.)||!(mu>0.)||!(tau>0.)||
      !isfinite(target)||!isfinite(mu)||!isfinite(tau)||!isfinite(w)){
     atomicAdd(bad,1u);return;
   }
   double ph[4]{0.,0.,0.,0.};
   ph[hxt4a::h4_tet_face[f.lf][0]]=hxt5b::QL0[q];
   ph[hxt4a::h4_tet_face[f.lf][1]]=hxt5b::QL1[q];
   ph[hxt4a::h4_tet_face[f.lf][2]]=hxt5b::QL2[q];
   double gp[4][3];
   for(int a=0;a<4;++a){
     const double*J=geo[cell].invJ;
     const double r[3]={a==0?-1.:(a==1?1.:0.),
                        a==0?-1.:(a==2?1.:0.),
                        a==0?-1.:(a==3?1.:0.)};
     for(int d=0;d<3;++d)gp[a][d]=r[0]*J[d]+r[1]*J[3+d]+r[2]*J[6+d];
   }
   double dn[4];
   for(int a=0;a<4;++a)
     dn[a]=gp[a][0]*f.n[0]+gp[a][1]*f.n[1]+gp[a][2]*f.n[2];
   for(int a=0;a<4;++a){
     atomicAdd(rhs+4*cell+a,w*(-mu*dn[a]+tau*ph[a])*target);
     for(int b=0;b<4;++b)
       atomicAdd(blocks+16*slot+4*a+b,w*
         (-mu*dn[b]*ph[a]-mu*dn[a]*ph[b]+tau*ph[a]*ph[b]));
   }
 }
}
}} // namespace nodals::omega_dg_gpu
