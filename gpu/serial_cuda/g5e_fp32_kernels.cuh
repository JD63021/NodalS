#pragma once
#include "cuda_runtime.hpp"
#include "precision.hpp"
#include "g4_host.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <type_traits>

namespace nodals_gpu {

using DeviceReal = StateReal;
static_assert(std::is_same<DeviceReal,float>::value,"G5E requires full FP32 state");
static_assert(std::is_same<OperatorReal,float>::value,"G5E requires FP32 operator");
static_assert(std::is_same<AMGReal,float>::value,"G5E requires FP32 AMG");

constexpr int G4B=256;
inline int g4grid(std::size_t n){return (int)((n+G4B-1)/G4B);}

struct G4CellPlanDevice {
  std::int32_t ref[8];
  std::uint8_t rowSlot[64];
  OperatorReal det;
  OperatorReal invJ[9];
};

__constant__ OperatorReal g4_diffT[8*8*3*3];
__constant__ OperatorReal g4_centT[8*8*8*3];

__device__ inline OperatorReal g4_coeff_cell(const G4CellPlanDevice&cp,int a,int d){
  const OperatorReal g1=cp.invJ[d],g2=cp.invJ[3+d],g3=cp.invJ[6+d];
  const OperatorReal gl=(a%4)==0?-(g1+g2+g3):((a%4)==1?g1:((a%4)==2?g2:g3));
  const OperatorReal base=(cp.det/OperatorReal(6))*gl;
  return (a<4)?base:OperatorReal(-27.0/20.0)*base;
}
__device__ inline StateReal g4_ref_value(const G4CellPlanDevice&cp,int a,int d,
                                         const StateReal*u0,const StateReal*u1,const StateReal*u2,
                                         const StateReal*fixed){
  int r=cp.ref[a];if(r>=0)return d==0?u0[r]:(d==1?u1[r]:u2[r]);
  int s=-r-1;return fixed[3*(std::size_t)s+d];
}

__global__ void g4_assemble_physical_kernel(const G4CellPlanDevice*cells,int nc,const std::int64_t*row,
                                             const StateReal*fixed,const StateReal*u0,const StateReal*u1,
                                             const StateReal*u2,OperatorReal nu,OperatorReal*aval,
                                             StateReal*cr0,StateReal*cr1,StateReal*cr2){
  int ic=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(ic>=nc)return;const auto&cp=cells[ic];
  OperatorReal I[3][3];for(int j=0;j<3;++j)for(int d=0;d<3;++d)I[j][d]=cp.invJ[3*j+d];
  OperatorReal metric[3][3]={{0}};for(int j=0;j<3;++j)for(int k=0;k<3;++k)for(int d=0;d<3;++d)metric[j][k]+=I[j][d]*I[k][d];
  StateReal cf[3][8];for(int m=0;m<8;++m)for(int d=0;d<3;++d)cf[d][m]=g4_ref_value(cp,m,d,u0,u1,u2,fixed);
  OperatorReal ur[8][3]={{0}};for(int m=0;m<8;++m)for(int j=0;j<3;++j)for(int d=0;d<3;++d)ur[m][j]+=OperatorReal(cf[d][m])*I[j][d];
  for(int a=0;a<8;++a){int r=cp.ref[a];if(r<0)continue;for(int b=0;b<8;++b){
    OperatorReal kd=0;for(int j=0;j<3;++j)for(int k=0;k<3;++k)kd+=g4_diffT[((a*8+b)*3+j)*3+k]*metric[j][k];kd*=nu*cp.det;
    OperatorReal cv=0;for(int m=0;m<8;++m)for(int j=0;j<3;++j)cv+=ur[m][j]*g4_centT[(((a*8+m)*8+b)*3+j)];cv*=cp.det;
    if(cp.ref[b]>=0){unsigned char sl=cp.rowSlot[8*a+b];if(sl!=255)atomicAdd(aval+row[r]+sl,kd+cv);}
    else{int fs=-cp.ref[b]-1;StateReal q=StateReal(cv);atomicAdd(cr0+r,-q*fixed[3*(std::size_t)fs]);atomicAdd(cr1+r,-q*fixed[3*(std::size_t)fs+1]);atomicAdd(cr2+r,-q*fixed[3*(std::size_t)fs+2]);}
  }}
}

__global__ void g4_finalize_relax_kernel(int n,const std::int64_t*rp,const std::int32_t*ci,
                                          const std::int32_t*diagPos,OperatorReal alpha,
                                          OperatorReal*av,OperatorReal*delta,OperatorReal*diag,OperatorReal*rau){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i>=n)return;OperatorReal m=0;
  for(std::int64_t k=rp[i];k<rp[i+1];++k)m+=fabsf(av[k]);
  OperatorReal de=(OperatorReal(1)/alpha-OperatorReal(1))*m;int dp=diagPos[i];
  delta[i]=de;av[dp]+=de;diag[i]=av[dp];rau[i]=OperatorReal(1)/av[dp];
}
__global__ void g4_bt3_kernel(const G4CellPlanDevice*cells,const StateReal*p,int nc,StateReal*v0,StateReal*v1,StateReal*v2){
  int c=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(c>=nc)return;const auto&cp=cells[c];StateReal pv=p[c];
  for(int a=0;a<8;++a){int g=cp.ref[a];if(g<0)continue;atomicAdd(v0+g,StateReal(g4_coeff_cell(cp,a,0))*pv);atomicAdd(v1+g,StateReal(g4_coeff_cell(cp,a,1))*pv);atomicAdd(v2+g,StateReal(g4_coeff_cell(cp,a,2))*pv);}
}
__global__ void g4_rau3_kernel(StateReal*v0,StateReal*v1,StateReal*v2,const OperatorReal*r,int n){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n){OperatorReal q=r[i];v0[i]*=q;v1[i]*=q;v2[i]*=q;}
}
__global__ void g4_b3_kernel(const G4CellPlanDevice*cells,int nc,const StateReal*v0,const StateReal*v1,const StateReal*v2,StateReal*y){
  int c=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(c>=nc)return;const auto&cp=cells[c];OperatorReal s=0;
  for(int a=0;a<8;++a){int g=cp.ref[a];if(g<0)continue;s+=g4_coeff_cell(cp,a,0)*v0[g]+g4_coeff_cell(cp,a,1)*v1[g]+g4_coeff_cell(cp,a,2)*v2[g];}y[c]=StateReal(s);
}
__global__ void g4_momentum_rhs_kernel(int n,const StateReal*s0,const StateReal*s1,const StateReal*s2,
                                       const StateReal*c0,const StateReal*c1,const StateReal*c2,
                                       const StateReal*bt0,const StateReal*bt1,const StateReal*bt2,
                                       const OperatorReal*delta,const StateReal*u0,const StateReal*u1,const StateReal*u2,
                                       StateReal*b0,StateReal*b1,StateReal*b2){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n){
    b0[i]=s0[i]+c0[i]+bt0[i]+delta[i]*u0[i];b1[i]=s1[i]+c1[i]+bt1[i]+delta[i]*u1[i];b2[i]=s2[i]+c2[i]+bt2[i]+delta[i]*u2[i];
  }
}
__global__ void g4_mcgs_color_kernel(int count,const std::int32_t*rows,const std::int64_t*rp,const std::int32_t*ci,
                                     const OperatorReal*av,const OperatorReal*diag,const StateReal*b0,const StateReal*b1,
                                     const StateReal*b2,StateReal*x0,StateReal*x1,StateReal*x2,OperatorReal omega,
                                     int a0,int a1,int a2){
  int q=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(q>=count)return;int i=rows[q];OperatorReal s0=0,s1=0,s2=0;
  for(std::int64_t k=rp[i];k<rp[i+1];++k){int j=ci[k];if(j==i)continue;OperatorReal a=av[k];if(a0)s0+=a*x0[j];if(a1)s1+=a*x1[j];if(a2)s2+=a*x2[j];}
  OperatorReal iv=OperatorReal(1)/diag[i];
  if(a0){OperatorReal z=(b0[i]-s0)*iv;x0[i]=StateReal((OperatorReal(1)-omega)*x0[i]+omega*z);}
  if(a1){OperatorReal z=(b1[i]-s1)*iv;x1[i]=StateReal((OperatorReal(1)-omega)*x1[i]+omega*z);}
  if(a2){OperatorReal z=(b2[i]-s2)*iv;x2[i]=StateReal((OperatorReal(1)-omega)*x2[i]+omega*z);}
}
__global__ void g4_mom_norms_kernel(int n,const std::int64_t*rp,const std::int32_t*ci,const OperatorReal*av,
                                    const StateReal*b0,const StateReal*b1,const StateReal*b2,
                                    const StateReal*x0,const StateReal*x1,const StateReal*x2,double*sums){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i>=n)return;OperatorReal ax0=0,ax1=0,ax2=0;
  for(std::int64_t k=rp[i];k<rp[i+1];++k){int j=ci[k];OperatorReal a=av[k];ax0+=a*x0[j];ax1+=a*x1[j];ax2+=a*x2[j];}
  double db0=(double)b0[i],db1=(double)b1[i],db2=(double)b2[i];
  double r0=(double)(b0[i]-ax0),r1=(double)(b1[i]-ax1),r2=(double)(b2[i]-ax2);
  atomicAdd(sums+0,db0*db0);atomicAdd(sums+1,db1*db1);atomicAdd(sums+2,db2*db2);
  atomicAdd(sums+3,r0*r0);atomicAdd(sums+4,r1*r1);atomicAdd(sums+5,r2*r2);
}
__global__ void g4_continuity_kernel(const G4CellPlanDevice*cells,int nc,const StateReal*fixedDiv,
                                      const StateReal*u0,const StateReal*u1,const StateReal*u2,StateReal*r){
  int c=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(c>=nc)return;const auto&cp=cells[c];OperatorReal s=fixedDiv[c];
  for(int a=0;a<8;++a){int g=cp.ref[a];if(g<0)continue;s+=g4_coeff_cell(cp,a,0)*u0[g]+g4_coeff_cell(cp,a,1)*u1[g]+g4_coeff_cell(cp,a,2)*u2[g];}r[c]=StateReal(s);
}
__global__ void g4_negate_kernel(int n,StateReal*x){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)x[i]=-x[i];}
__global__ void g4_axpy_kernel(int n,double a,const StateReal*x,StateReal*y){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)y[i]+=StateReal(a)*x[i];}
__global__ void g4_csr_spmv_kernel(int n,const std::int64_t*row,const std::int32_t*col,const AMGReal*val,const AMGReal*x,AMGReal*y){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n){AMGReal s=0;for(std::int64_t k=row[i];k<row[i+1];++k)s+=val[k]*x[col[k]];y[i]=s;}
}
__global__ void g4_jacobi_zero_kernel(int n,double omega,const AMGReal*b,const AMGReal*d,AMGReal*x){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)x[i]=AMGReal(omega)*b[i]/d[i];}
__global__ void g4_residual_kernel(int n,const AMGReal*b,const AMGReal*Ax,AMGReal*r){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)r[i]=b[i]-Ax[i];}
__global__ void g4_jacobi_add_kernel(int n,double omega,const AMGReal*r,const AMGReal*d,AMGReal*x){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)x[i]+=AMGReal(omega)*r[i]/d[i];}
__global__ void g4_dense_mv_kernel(int n,const AMGReal*A,const AMGReal*b,AMGReal*x){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n){AMGReal s=0;for(int j=0;j<n;++j)s+=A[(std::size_t)i*n+j]*b[j];x[i]=s;}}
__global__ void g4_copy_kernel(int n,const StateReal*a,StateReal*b){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)b[i]=a[i];}
__global__ void g4_pcg_xr_kernel(int n,double alpha,const StateReal*p,const StateReal*q,StateReal*x,StateReal*r){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n){StateReal a=StateReal(alpha);x[i]+=a*p[i];r[i]-=a*q[i];}}
__global__ void g4_pcg_p_kernel(int n,double beta,const StateReal*z,StateReal*p){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)p[i]=z[i]+StateReal(beta)*p[i];}

__global__ void g5_sa_restrict_kernel(int n,const std::int64_t*rp,const std::int32_t*ci,const AMGReal*pv,const AMGReal*f,AMGReal*c){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i>=n)return;AMGReal x=f[i];for(std::int64_t k=rp[i];k<rp[i+1];++k)atomicAdd(c+ci[k],pv[k]*x);
}
__global__ void g5_sa_prolong_add_kernel(int n,const std::int64_t*rp,const std::int32_t*ci,const AMGReal*pv,const AMGReal*c,AMGReal*f){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i>=n)return;AMGReal s=0;for(std::int64_t k=rp[i];k<rp[i+1];++k)s+=pv[k]*c[ci[k]];f[i]+=s;
}
__global__ void g5_power_init_kernel(int n,AMGReal*v){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n){float g=(float)(i+1);v[i]=sinf(.731f*g)+.27f*cosf(1.117f*g);}}
__global__ void g5_div_sqrt_diag_kernel(int n,const AMGReal*x,const AMGReal*d,AMGReal*y){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)y[i]=x[i]/sqrtf(d[i]);}
__global__ void g5_diff_kernel(int n,const OperatorReal*a,const OperatorReal*b,OperatorReal*y){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)y[i]=a[i]-b[i];}

} // namespace nodals_gpu
