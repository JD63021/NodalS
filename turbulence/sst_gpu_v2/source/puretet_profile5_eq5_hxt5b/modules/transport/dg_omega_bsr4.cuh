#pragma once
// Experimental NodalS omega-only DG-P1 GPU BSR4 assembler.
// Independent and opt-in: it does not alter the coupled CG SST executable.
// Numeric reference: dg_omega_p1_reference.hpp.
// Future G2 bridge must populate the actual quadrature-point SST coefficients
// and shared Spalding/trace_log targets before enabling physics feedback.
#include <cuda_runtime.h>
#include <cstdint>

namespace nodals { namespace omega_dg_gpu {
struct Vec3 {double x,y,z;};
struct Cell {
  double volume;
  Vec3 grad[4];
  double mu, reaction, source;
  Vec3 u;
};
enum Boundary : int { Natural=0, Inlet=1, Wall=2, Outlet=3 };
struct Face {
  int left,right;
  int vertexL[3],vertexR[3]; // sorted global face triangle's local vertex indices
  int slotLL,slotLR,slotRL,slotRR; // global block slots in BSR adjacency
  int boundary;
  Vec3 normal; // unit normal outward from left
  double area;
  double target[3]; // per-face triangle quadrature point; already computed on host
};
struct Controls {
  double penaltyC=10.0;
  double theta=1.0;
  double traceGamma=100000.0;
  double wallNormalTolerance=1.e-8;
};
__host__ __device__ inline double dot(Vec3 a,Vec3 b) {
  return a.x*b.x+a.y*b.y+a.z*b.z;
}
__host__ __device__ inline Vec3 avg(Vec3 a,Vec3 b) {
  return {(a.x+b.x)*.5,(a.y+b.y)*.5,(a.z+b.z)*.5};
}
__device__ inline double phi(int a,const int map[3],int q) {
  // Same three degree-2 triangle quadrature points as the portable reference.
  for(int j=0;j<3;++j)if(map[j]==a)return j==q?2./3.:1./6.;
  return 0.;
}
__global__ void zero(int n,double*blocks,double*rhs,int nBlocks) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<16*nBlocks)blocks[i]=0.;
  if(i<n)rhs[i]=0.;
}
__global__ void volume(int n,const Cell*c,const int*diagonalSlots,
                       double*blocks,double*rhs,unsigned int*errors) {
  const int t=blockIdx.x*blockDim.x+threadIdx.x;
  if(t>=n)return;
  const Cell z=c[t];
  if(!(z.volume>0&&z.mu>=0&&isfinite(z.volume)&&isfinite(z.mu)&&
       isfinite(z.source)&&isfinite(z.reaction))){
    atomicAdd(errors,1u);return;
  }
  const int slot=diagonalSlots[t];
  for(int a=0;a<4;++a) {
    rhs[4*t+a]=z.volume*.25*z.source;
    for(int b=0;b<4;++b) {
      const double diffus=z.mu*z.volume*dot(z.grad[a],z.grad[b]);
      const double react=z.reaction*z.volume*(a==b?.1:.05);
      const double adv=.25*z.volume*dot(z.u,z.grad[b]);
      blocks[16*slot+4*a+b]=diffus+react+adv;
    }
  }
}
__device__ inline void block_add(double*blocks,int slot,int a,int b,double value) {
  if(value!=0.)atomicAdd(blocks+16*slot+4*a+b,value);
}
__global__ void faces(int n,const Face*f,const Cell*c,Controls opt,
                      double*blocks,double*rhs,unsigned int*errors) {
  const int fi=blockIdx.x*blockDim.x+threadIdx.x;
  if(fi>=n)return;
  const Face F=f[fi];
  const Cell L=c[F.left];
  const double hL=3.*L.volume/F.area;
  if(!(hL>0.&&F.area>0.&&isfinite(hL))){atomicAdd(errors,1u);return;}
  if(F.right>=0){
    const Cell R=c[F.right];
    const double hR=3.*R.volume/F.area;
    if(!(hR>0.&&isfinite(hR))){atomicAdd(errors,1u);return;}
    const double eta=4.*opt.penaltyC*fmax(L.mu/hL,R.mu/hR);
    const double un=dot(avg(L.u,R.u),F.normal);
    const double alpha=opt.theta*fabs(un);
    const double convL=.5*(alpha-un);
    const double convR=.5*(alpha+un);
    for(int q=0;q<3;++q){
      const double w=F.area/3.;
      for(int sa=0;sa<2;++sa){
        const Cell A=sa?R:L;
        const int*mapA=sa?F.vertexR:F.vertexL;
        const double signA=sa?-1.:1.;
        const int slotAA=sa?F.slotRR:F.slotLL;
        const int slotAB=sa?F.slotRL:F.slotLR;
        for(int a=0;a<4;++a){
          const double pa=phi(a,mapA,q),jv=signA*pa;
          const double fluxV=.5*A.mu*dot(A.grad[a],F.normal);
          for(int sb=0;sb<2;++sb){
            const Cell B=sb?R:L;
            const int*mapB=sb?F.vertexR:F.vertexL;
            const double signB=sb?-1.:1.;
            const int slot=sb==sa?slotAA:slotAB;
            for(int b=0;b<4;++b){
              const double pb=phi(b,mapB,q),jw=signB*pb;
              const double fluxW=.5*B.mu*dot(B.grad[b],F.normal);
              double value=w*(-fluxW*jv-fluxV*jw+eta*jv*jw);
              // Advective-volume DG face correction, exact oracle parity.
              if(sa==0)value+=w*convL*pa*(sb==0?pb:-pb);
              else value+=w*convR*pa*(sb==1?pb:-pb);
              block_add(blocks,slot,a,b,value);
            }
          }
        }
      }
    }
  }else{
    const double un=dot(L.u,F.normal);
    if(F.boundary==Wall && fabs(un)>opt.wallNormalTolerance){
      atomicAdd(errors,1u);return;
    }
    const bool dirichlet=F.boundary==Wall||F.boundary==Inlet;
    const bool backflow=F.boundary==Outlet&&un<0.;
    for(int q=0;q<3;++q){
      const double w=F.area/3.;
      const double target=F.target[q];
      if((dirichlet||(un<0.&&(F.boundary==Inlet||backflow)))&&
          (!(target>0.)||!isfinite(target))){
        atomicAdd(errors,1u);return;
      }
      for(int a=0;a<4;++a){
        const double pa=phi(a,F.vertexL,q);
        const double gA=dot(L.grad[a],F.normal);
        if(dirichlet) {
          const double eta=opt.traceGamma*L.mu/hL;
          atomicAdd(rhs+4*F.left+a,w*(-L.mu*gA+eta*pa)*target);
          for(int b=0;b<4;++b){
            const double pb=phi(b,F.vertexL,q);
            const double gB=dot(L.grad[b],F.normal);
            block_add(blocks,F.slotLL,a,b,
              w*(-L.mu*gB*pa-L.mu*gA*pb+eta*pa*pb));
          }
        }
        if(un<0. && (F.boundary==Inlet||backflow)){
          atomicAdd(rhs+4*F.left+a,-un*w*target*pa);
          for(int b=0;b<4;++b)
            block_add(blocks,F.slotLL,a,b,-un*w*pa*phi(b,F.vertexL,q));
        }
      }
    }
  }
}
__global__ void bsr4_spmv(int nTet,const int*rowPtr,const int*colInd,
                          const double*blocks,const double*x,double*y){
  const int row=blockIdx.x*blockDim.x+threadIdx.x;
  if(row>=4*nTet)return;
  const int t=row/4,a=row%4;
  double acc=0.;
  for(int j=rowPtr[t];j<rowPtr[t+1];++j){
    const int other=colInd[j];
    for(int b=0;b<4;++b)acc+=blocks[16*j+4*a+b]*x[4*other+b];
  }
  y[row]=acc;
}
// Solve exact 4x4 diagonal block for one element: pivoted elimination.
// This preconditioner is available to the future device FGMRES implementation.
__global__ void bsr4_block_jacobi(int nTet,const double*blocks,
                                 const int*diagonalSlots,const double*r,
                                 double*z,unsigned int*errors){
  const int t=blockIdx.x*blockDim.x+threadIdx.x;
  if(t>=nTet)return;
  double M[4][5];
  const int bslot=diagonalSlots[t];
  for(int i=0;i<4;++i){
    for(int j=0;j<4;++j)M[i][j]=blocks[16*bslot+4*i+j];
    M[i][4]=r[4*t+i];
  }
  for(int k=0;k<4;++k){
    int piv=k;
    for(int i=k+1;i<4;++i)if(fabs(M[i][k])>fabs(M[piv][k]))piv=i;
    if(!(fabs(M[piv][k])>1.e-35)||!isfinite(M[piv][k])){
      atomicAdd(errors,1u);return;
    }
    if(piv!=k)for(int j=k;j<5;++j){double tmp=M[piv][j];M[piv][j]=M[k][j];M[k][j]=tmp;}
    const double inv=1./M[k][k];
    for(int j=k;j<5;++j)M[k][j]*=inv;
    for(int i=0;i<4;++i)if(i!=k){
      const double f=M[i][k];
      for(int j=k;j<5;++j)M[i][j]-=f*M[k][j];
    }
  }
  for(int i=0;i<4;++i)z[4*t+i]=M[i][4];
}
}} // namespace nodals::omega_dg_gpu
