#pragma once
// DG masked-Q1 experiment: original NodalS row-L1 fixed-point relaxation,
// now evaluated for genuinely independent tetrahedral P1 physical unknowns.
//
// Every tet has four P1 omega DOFs (physical) and four *virtual*, FIXED-ZERO
// face BF3 modes b_e = 27*product_{a!=e}(lambda_a). Like the validated CG
// masked-Q1 scalar path, only physical P1 unknowns are solved; couplings from
// the constrained BF3 columns are retained in each physical row's FULL L1
// relaxation norm. The physical operator is the unchanged DG-P1 BSR4 SIPG/LF
// matrix. There is NO second PDE, no unconstrained BF3 solution, and no CG
// projection into the constitutive physics.
//
// We assemble P1-to-BF3 columns (volume, interior face, inlet/outlet and
// physical Spalding/Nitsche wall), preserving their algebraic sums before
// taking absolute values. Neighbor BF3 columns exist on at most one face.
// Then apply (A+Delta)omega_new = b+Delta*omega_old with
// Delta_i=(1/alpha-1)*(sum_P1|Aij|+sum_BF3|Aij|).
// This is exactly the converged fixed point of the DG-P1 physical equation.
// For parity with the original masked-Q1 code, nodal floor is used instead
// of a cell-mean Bernstein scaling limiter. Flooring is explicitly counted;
// it can alter DG cell means, so it is an experimental comparison only.
#include "dg_omega_sst_volume.cuh"
#include "dg_omega_wall_q12.cuh"
#include <cuda_runtime.h>
#include <cmath>

namespace nodals { namespace omega_dg_gpu {
__device__ __forceinline__ Vec3 scaled(Vec3 x,double s){
  return {s*x.x,s*x.y,s*x.z};
}
__device__ __forceinline__ void bf3_values(const Cell&g,const double l[4],
                                           double p[4],Vec3 gp[4]){
  for(int e=0;e<4;++e){
    double value=27.;
    Vec3 grad{0.,0.,0.};
    for(int k=0;k<4;++k)if(k!=e)value*=l[k];
    for(int j=0;j<4;++j)if(j!=e){
      double factor=27.;
      for(int k=0;k<4;++k)if(k!=e && k!=j)factor*=l[k];
      grad.x+=factor*g.grad[j].x;
      grad.y+=factor*g.grad[j].y;
      grad.z+=factor*g.grad[j].z;
    }
    p[e]=value;gp[e]=grad;
  }
}
// Volume P1-to-constrained-BF3 columns: use the original 24-point
// tetrahedral integration (not the DG-P1 pilot's four-point rule), because
// the virtual BF3 trial is cubic and its mixed stiffness requires more points.
__global__ void masked_q1_volume_bf3(
  int nt,const Cell*c,const nodals_hxt1::TetConn*tc,
  const nodals_hxt1::TetVel*tv,const double*u0,const double*u1,
  const double*u2,const double*kCG,const double*oldOmegaDG,
  const double*wallDistance,PhysicalSSTConfig physics,
  double*selfPB,unsigned int*bad){
 const int t=blockIdx.x*blockDim.x+threadIdx.x;
 if(t>=nt)return;
 const Cell g=c[t];
 double matrix[16]{};
 for(int q=0;q<24;++q){
   double l[4]={hxt5a::h5_tet24_l0[q],hxt5a::h5_tet24_l1[q],
                hxt5a::h5_tet24_l2[q],hxt5a::h5_tet24_l3[q]};
   double mu=0.,reaction=0.,src=0.;Vec3 u{};
   if(!quadrature_sst(t,l,g,tc,tv,u0,u1,u2,kCG,oldOmegaDG,
                       wallDistance,physics,mu,reaction,src,u,nullptr)){
     atomicAdd(bad,1u);return;
   }
   double pb[4];Vec3 gb[4];
   bf3_values(g,l,pb,gb);
   const double weight=hxt5a::h5_tet24_w[q]*g.volume;
   for(int a=0;a<4;++a)
     for(int b=0;b<4;++b)
       matrix[4*a+b]+=weight*(mu*dot(g.grad[a],gb[b])+
                             reaction*l[a]*pb[b]+l[a]*dot(u,gb[b]));
 }
 for(int i=0;i<16;++i)selfPB[(std::size_t)t*16+i]=matrix[i];
}
// Face couplings of a P1 test to a virtual BF3 trial, with the SAME
// SIPG and theta*abs(un) LF coefficients as the BSR4 DG-P1 operator.
// A same-element bubble column receives volume and all local face terms;
// a neighboring bubble column has only the one shared-face contribution.
__global__ void masked_q1_face_bf3(
  int nf,const Face*faces,const Cell*c,Controls opt,
  double*selfPB,double*crossL1,unsigned int*bad){
 const int fi=blockIdx.x*blockDim.x+threadIdx.x;
 if(fi>=nf)return;
 const Face f=faces[fi];const Cell L=c[f.left];
 const double hL=3.*L.volume/f.area;
 if(!(f.area>0.)||!(hL>0.)){atomicAdd(bad,1u);return;}
 if(f.right>=0){
   const Cell R=c[f.right];const double hR=3.*R.volume/f.area;
   if(!(hR>0.)){atomicAdd(bad,1u);return;}
   const double eta=4.*opt.penaltyC*fmax(L.mu/hL,R.mu/hR);
   const double un=dot(avg(L.u,R.u),f.normal);
   const double alpha=opt.theta*fabs(un);
   const double conv[2]={.5*(alpha-un),.5*(alpha+un)};
   double M[2][2][16]{};
   for(int q=0;q<12;++q){
     double ll[4]{0.,0.,0.,0.},lr[4]{0.,0.,0.,0.};
     for(int j=0;j<3;++j){
       const double value=j==0?hxt5b::QL0[q]:(j==1?hxt5b::QL1[q]:hxt5b::QL2[q]);
       ll[f.vertexL[j]]=value;
       lr[f.vertexR[j]]=value;
     }
     double pbl[4],pbr[4];Vec3 gbl[4],gbr[4];
     bf3_values(L,ll,pbl,gbl);bf3_values(R,lr,pbr,gbr);
     for(int sa=0;sa<2;++sa){
       const Cell A=sa?R:L;
       const double *la=sa?lr:ll;
       const double signA=sa?-1.:1.;
       for(int sb=0;sb<2;++sb){
         const Cell B=sb?R:L;
         const double *pb=sb?pbr:pbl;
         const Vec3 *gb=sb?gbr:gbl;
         const double signB=sb?-1.:1.;
         for(int a=0;a<4;++a){
           const double pa=la[a],jv=signA*pa;
           const double fluxV=.5*A.mu*dot(A.grad[a],f.normal);
           for(int b=0;b<4;++b){
             const double jw=signB*pb[b];
             const double fluxW=.5*B.mu*dot(gb[b],f.normal);
             double value=-fluxW*jv-fluxV*jw+eta*jv*jw;
             value+=conv[sa]*pa*(sa==sb?pb[b]:-pb[b]);
             M[sa][sb][4*a+b]+=(f.area*hxt5b::QW12[q])*value;
           }
         }
       }
     }
   }
   for(int sa=0;sa<2;++sa){
     const int t=sa?f.right:f.left;
     for(int a=0;a<4;++a){
       double outL1=0.;
       for(int b=0;b<4;++b){
         atomicAdd(selfPB+(std::size_t)t*16+4*a+b,M[sa][sa][4*a+b]);
         outL1+=fabs(M[sa][1-sa][4*a+b]);
       }
       atomicAdd(crossL1+4*t+a,outL1);
     }
   }
 }else{
   if(f.boundary==Wall)return; // Exact 12-point physical wall handled below.
   const double un=dot(L.u,f.normal);
   const bool inlet=f.boundary==Inlet;
   const bool backflow=f.boundary==Outlet && un<0.;
   double M[16]{};
   for(int q=0;q<12;++q){
     double l[4]{0.,0.,0.,0.};
     for(int j=0;j<3;++j)l[f.vertexL[j]]=j==0?hxt5b::QL0[q]:(j==1?hxt5b::QL1[q]:hxt5b::QL2[q]);
     double pb[4];Vec3 gb[4];
     bf3_values(L,l,pb,gb);
     const double weight=f.area*hxt5b::QW12[q];
     for(int a=0;a<4;++a)for(int b=0;b<4;++b){
       double v=0.;
       if(inlet){
         const double tau=opt.traceGamma*L.mu/hL;
         v+=-L.mu*dot(gb[b],f.normal)*l[a]-
             L.mu*dot(L.grad[a],f.normal)*pb[b]+
             tau*l[a]*pb[b];
       }
       if(un<0. && (inlet||backflow))v+=(-un)*l[a]*pb[b];
       M[4*a+b]+=weight*v;
     }
   }
   for(int j=0;j<16;++j)
     atomicAdd(selfPB+(std::size_t)f.left*16+j,M[j]);
 }
}
// The physical wall is the ORIGINAL Q12 trace-log Nitsche law and mu
// sampled by the shared five-point Spalding fit. Its 4 BF3 trial columns
// contribute to the masked relaxation, but BF3 physical coefficients stay 0.
__global__ void masked_q1_wall_bf3_q12(
  int nw,const hxt5b::TetFacePlan*plans,const Cell*c,
  const double*wallMu,double traceGamma,
  double*selfPB,unsigned int*bad){
 const int wi=blockIdx.x*blockDim.x+threadIdx.x;
 if(wi>=nw)return;
 const auto f=plans[wi];const int t=f.cell;
 const Cell g=c[t];
 const double h=3.*g.volume/f.area;
 if(!(h>0.)||!(f.area>0.)||f.lf<0||f.lf>3){
   atomicAdd(bad,1u);return;
 }
 double M[16]{};
 const Vec3 n{f.n[0],f.n[1],f.n[2]};
 for(int q=0;q<12;++q){
   const double mu=wallMu[(std::size_t)wi*12+q];
   if(!(mu>0.)||!isfinite(mu)){atomicAdd(bad,1u);return;}
   double l[4]{0.,0.,0.,0.};
   l[hxt4a::h4_tet_face[f.lf][0]]=hxt5b::QL0[q];
   l[hxt4a::h4_tet_face[f.lf][1]]=hxt5b::QL1[q];
   l[hxt4a::h4_tet_face[f.lf][2]]=hxt5b::QL2[q];
   double pb[4];Vec3 gb[4];
   bf3_values(g,l,pb,gb);
   const double tau=traceGamma*mu/h;
   const double weight=hxt5b::QW12[q]*f.area;
   for(int a=0;a<4;++a)
     for(int b=0;b<4;++b)
       M[4*a+b]+=weight*(-mu*dot(gb[b],n)*l[a]-
                              mu*dot(g.grad[a],n)*pb[b]+
                              tau*l[a]*pb[b]);
 }
 for(int j=0;j<16;++j)
   atomicAdd(selfPB+(std::size_t)t*16+j,M[j]);
}
// Original masked-Q1 stabilization: virtual bubble columns enter only the
// row-L1 norm. The DG-P1 physical fixed point is IDENTICAL for a converged
// linear solve because Delta*old disappears as omega_new->omega_old.
__global__ void masked_q1_row_l1_relax(
  int nt,const int*row,const int*col,const int*diag,
  double*matrix,double*rhs,const double*previous,
  const double*selfPB,const double*crossL1,double alpha,
  unsigned int*bad){
 const int t=blockIdx.x*blockDim.x+threadIdx.x;
 if(t>=nt)return;
 if(!(alpha>0. && alpha<=1.)){atomicAdd(bad,1u);return;}
 for(int a=0;a<4;++a){
   double rowL1=crossL1[4*t+a];
   for(int b=0;b<4;++b)rowL1+=fabs(selfPB[(std::size_t)t*16+4*a+b]);
   for(int slot=row[t];slot<row[t+1];++slot)
     for(int b=0;b<4;++b)
       rowL1+=fabs(matrix[16*slot+4*a+b]);
   const int k=16*diag[t]+5*a;
   const double original=matrix[k];
   const double delta=(1./alpha-1.)*rowL1;
   if(!(rowL1>0.)||!isfinite(rowL1)||!isfinite(delta)||
      !isfinite(original)||!(original+delta>0.)){
     atomicAdd(bad,1u);return;
   }
   matrix[k]=original+delta;
   rhs[4*t+a]+=delta*previous[4*t+a];
 }
}
// Match the original CG masked-Q1 vertex floor rather than the DG P1
// cell-mean-preserving Bernstein rescaling. Explicitly audit mass changes.
// This is not positivity-conservative; that is the purpose of the comparison.
__global__ void masked_q1_nodal_floor(
 int nt,const double*candidate,double*updated,double floor,
 unsigned long long*clippedDofs,unsigned long long*clippedCells,
 double*positiveMassAdjustment,unsigned int*bad){
 const int t=blockIdx.x*blockDim.x+threadIdx.x;
 if(t>=nt)return;
 bool any=false;double delta=0.;
 for(int a=0;a<4;++a){
   const double value=candidate[4*t+a];
   if(!isfinite(value)){atomicAdd(bad,1u);return;}
   const double use=value<floor?floor:value;
   updated[4*t+a]=use;
   if(use!=value){
     any=true;atomicAdd(clippedDofs,1ULL);delta+=0.25*(use-value);
   }
 }
 if(any){
   atomicAdd(clippedCells,1ULL);
   atomicAdd(positiveMassAdjustment,delta);
 }
}
}} // namespace nodals::omega_dg_gpu
