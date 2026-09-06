#pragma once
#include "cuda_runtime.hpp"
#include "g4_host.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace nodals_gpu {
constexpr int G4B=256;
inline int g4grid(std::size_t n){return (int)((n+G4B-1)/G4B);}

__constant__ double g4_diffT[8*8*3*3];
__constant__ double g4_centT[8*8*8*3];

__device__ inline double g4_coeff_cell(const G4CellPlanHost&cp,int a,int d){
  double g1=cp.invJ[d],g2=cp.invJ[3+d],g3=cp.invJ[6+d];
  double gl=(a%4)==0?-(g1+g2+g3):((a%4)==1?g1:((a%4)==2?g2:g3));
  double base=(cp.det/6.0)*gl;return (a<4)?base:-(27.0/20.0)*base;
}
__device__ inline double g4_ref_value(const G4CellPlanHost&cp,int a,int d,const double*u0,const double*u1,const double*u2,const double*fixed){
  int r=cp.ref[a];if(r>=0)return d==0?u0[r]:(d==1?u1[r]:u2[r]);int s=-r-1;return fixed[3*(std::size_t)s+d];
}
__global__ void g4_assemble_physical_kernel(const G4CellPlanHost*cells,int nc,const std::int64_t*row,const double*fixed,
                                             const double*u0,const double*u1,const double*u2,double nu,
                                             double*aval,double*cr0,double*cr1,double*cr2){
  int ic=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(ic>=nc)return;const auto&cp=cells[ic];
  double I[3][3];for(int j=0;j<3;++j)for(int d=0;d<3;++d)I[j][d]=cp.invJ[3*j+d];
  double metric[3][3]={{0}};for(int j=0;j<3;++j)for(int k=0;k<3;++k)for(int d=0;d<3;++d)metric[j][k]+=I[j][d]*I[k][d];
  double cf[3][8];for(int m=0;m<8;++m)for(int d=0;d<3;++d)cf[d][m]=g4_ref_value(cp,m,d,u0,u1,u2,fixed);
  double ur[8][3]={{0}};for(int m=0;m<8;++m)for(int j=0;j<3;++j)for(int d=0;d<3;++d)ur[m][j]+=cf[d][m]*I[j][d];
  for(int a=0;a<8;++a){int r=cp.ref[a];if(r<0)continue;for(int b=0;b<8;++b){
      double kd=0.0;for(int j=0;j<3;++j)for(int k=0;k<3;++k)kd+=g4_diffT[((a*8+b)*3+j)*3+k]*metric[j][k];kd*=nu*cp.det;
      double cv=0.0;for(int m=0;m<8;++m)for(int j=0;j<3;++j)cv+=ur[m][j]*g4_centT[(((a*8+m)*8+b)*3+j)];cv*=cp.det;
      if(cp.ref[b]>=0){unsigned char s=cp.rowSlot[8*a+b];if(s!=255)atomicAdd(aval+row[r]+s,kd+cv);}else{int fs=-cp.ref[b]-1;double q=cv;atomicAdd(cr0+r,-q*fixed[3*(std::size_t)fs]);atomicAdd(cr1+r,-q*fixed[3*(std::size_t)fs+1]);atomicAdd(cr2+r,-q*fixed[3*(std::size_t)fs+2]);}
  }}
}
__global__ void g4_finalize_relax_kernel(int n,const std::int64_t*rp,const std::int32_t*ci,const std::int32_t*diagPos,double alpha,double*av,double*delta,double*diag,double*rau){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i>=n)return;double m=0;for(std::int64_t k=rp[i];k<rp[i+1];++k)m+=fabs(av[k]);double de=(1.0/alpha-1.0)*m;int dp=diagPos[i];delta[i]=de;av[dp]+=de;diag[i]=av[dp];rau[i]=1.0/av[dp];
}
__global__ void g4_bt3_kernel(const G4CellPlanHost*cells,const double*p,int nc,double*v0,double*v1,double*v2){int c=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(c>=nc)return;const auto&cp=cells[c];double pv=p[c];for(int a=0;a<8;++a){int g=cp.ref[a];if(g<0)continue;atomicAdd(v0+g,g4_coeff_cell(cp,a,0)*pv);atomicAdd(v1+g,g4_coeff_cell(cp,a,1)*pv);atomicAdd(v2+g,g4_coeff_cell(cp,a,2)*pv);}}
__global__ void g4_rau3_kernel(double*v0,double*v1,double*v2,const double*r,int n){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n){double q=r[i];v0[i]*=q;v1[i]*=q;v2[i]*=q;}}
__global__ void g4_b3_kernel(const G4CellPlanHost*cells,int nc,const double*v0,const double*v1,const double*v2,double*y){int c=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(c>=nc)return;const auto&cp=cells[c];double s=0;for(int a=0;a<8;++a){int g=cp.ref[a];if(g<0)continue;s+=g4_coeff_cell(cp,a,0)*v0[g]+g4_coeff_cell(cp,a,1)*v1[g]+g4_coeff_cell(cp,a,2)*v2[g];}y[c]=s;}
__global__ void g4_fine_diag_kernel(const G4CellPlanHost*cells,int nc,const double*r,double*d){int c=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(c>=nc)return;const auto&cp=cells[c];double s=0;for(int a=0;a<8;++a){int g=cp.ref[a];if(g<0)continue;for(int q=0;q<3;++q){double b=g4_coeff_cell(cp,a,q);s+=r[g]*b*b;}}d[c]=s;}
__global__ void g4_momentum_rhs_kernel(int n,const double*s0,const double*s1,const double*s2,const double*c0,const double*c1,const double*c2,const double*bt0,const double*bt1,const double*bt2,const double*delta,const double*u0,const double*u1,const double*u2,double*b0,double*b1,double*b2){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n){b0[i]=s0[i]+c0[i]+bt0[i]+delta[i]*u0[i];b1[i]=s1[i]+c1[i]+bt1[i]+delta[i]*u1[i];b2[i]=s2[i]+c2[i]+bt2[i]+delta[i]*u2[i];}}
__global__ void g4_mcgs_color_kernel(int count,const std::int32_t*rows,const std::int64_t*rp,const std::int32_t*ci,const double*av,const double*diag,const double*b0,const double*b1,const double*b2,double*x0,double*x1,double*x2,double omega,int a0,int a1,int a2){int q=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(q>=count)return;int i=rows[q];double s0=0,s1=0,s2=0;for(std::int64_t k=rp[i];k<rp[i+1];++k){int j=ci[k];if(j==i)continue;double a=av[k];if(a0)s0+=a*x0[j];if(a1)s1+=a*x1[j];if(a2)s2+=a*x2[j];}double iv=1.0/diag[i];if(a0){double n=(b0[i]-s0)*iv;x0[i]=(1-omega)*x0[i]+omega*n;}if(a1){double n=(b1[i]-s1)*iv;x1[i]=(1-omega)*x1[i]+omega*n;}if(a2){double n=(b2[i]-s2)*iv;x2[i]=(1-omega)*x2[i]+omega*n;}}
__global__ void g4_mom_norms_kernel(int n,const std::int64_t*rp,const std::int32_t*ci,const double*av,const double*b0,const double*b1,const double*b2,const double*x0,const double*x1,const double*x2,double*sums){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i>=n)return;double ax0=0,ax1=0,ax2=0;for(std::int64_t k=rp[i];k<rp[i+1];++k){int j=ci[k];double a=av[k];ax0+=a*x0[j];ax1+=a*x1[j];ax2+=a*x2[j];}double r0=b0[i]-ax0,r1=b1[i]-ax1,r2=b2[i]-ax2;atomicAdd(sums+0,b0[i]*b0[i]);atomicAdd(sums+1,b1[i]*b1[i]);atomicAdd(sums+2,b2[i]*b2[i]);atomicAdd(sums+3,r0*r0);atomicAdd(sums+4,r1*r1);atomicAdd(sums+5,r2*r2);}
__global__ void g4_continuity_kernel(const G4CellPlanHost*cells,int nc,const double*fixedDiv,const double*u0,const double*u1,const double*u2,double*r){int c=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(c>=nc)return;const auto&cp=cells[c];double s=fixedDiv[c];for(int a=0;a<8;++a){int g=cp.ref[a];if(g<0)continue;s+=g4_coeff_cell(cp,a,0)*u0[g]+g4_coeff_cell(cp,a,1)*u1[g]+g4_coeff_cell(cp,a,2)*u2[g];}r[c]=s;}
__global__ void g4_negate_kernel(int n,double*x){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)x[i]=-x[i];}
__global__ void g4_axpy_kernel(int n,double a,const double*x,double*y){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)y[i]+=a*x[i];}
__global__ void g4_csr_spmv_kernel(int n,const std::int64_t*row,const std::int32_t*col,const double*val,const double*x,double*y){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n){double s=0;for(std::int64_t k=row[i];k<row[i+1];++k)s+=val[k]*x[col[k]];y[i]=s;}}
__global__ void g4_jacobi_zero_kernel(int n,double omega,const double*b,const double*d,double*x){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)x[i]=omega*b[i]/d[i];}
__global__ void g4_residual_kernel(int n,const double*b,const double*Ax,double*r){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)r[i]=b[i]-Ax[i];}
__global__ void g4_jacobi_add_kernel(int n,double omega,const double*r,const double*d,double*x){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)x[i]+=omega*r[i]/d[i];}
__global__ void g4_restrict_kernel(int n,const std::int32_t*agg,const double*r,double*bc){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)atomicAdd(bc+agg[i],r[i]);}
__global__ void g4_prolong_add_kernel(int n,const std::int32_t*agg,const double*xc,double*x){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)x[i]+=xc[agg[i]];}
__global__ void g4_dense_mv_kernel(int n,const double*A,const double*b,double*x){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n){double s=0;for(int j=0;j<n;++j)s+=A[(std::size_t)i*n+j]*b[j];x[i]=s;}}
__global__ void g4_copy_kernel(int n,const double*a,double*b){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)b[i]=a[i];}
__global__ void g4_pcg_xr_kernel(int n,double alpha,const double*p,const double*q,double*x,double*r){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n){x[i]+=alpha*p[i];r[i]-=alpha*q[i];}}
__global__ void g4_pcg_p_kernel(int n,double beta,const double*z,double*p){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)p[i]=z[i]+beta*p[i];}
__device__ inline void g4_atomic_max_positive(double*addr,double v){
  if(!(v>=0.0) || !isfinite(v)) return;
  auto *u=reinterpret_cast<unsigned long long*>(addr);
  atomicMax(u,__double_as_longlong(v));
}

__global__ void g4_extract_diag_kernel(int n,const std::int32_t*diagPos,const double*val,double*diag){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)diag[i]=val[diagPos[i]];
}

__global__ void g4_refresh_first_coarse_kernel(
  int nv,const std::int64_t*supportRow,const std::int32_t*supportCell,const std::uint8_t*supportBasis,
  const G4CellPlanHost*cells,const std::int32_t*fineAgg,const double*rau,
  const std::int64_t*crp,const std::int32_t*cci,double*cval)
{
  int g=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(g>=nv)return;
  const double rg=rau[g];
  const std::int64_t b=supportRow[g],e=supportRow[g+1];
  for(std::int64_t ka=b;ka<e;++ka){
    const int ca=supportCell[ka],aa=(int)supportBasis[ka];
    const int ra=fineAgg[ca];
    double ba[3]={g4_coeff_cell(cells[ca],aa,0),g4_coeff_cell(cells[ca],aa,1),g4_coeff_cell(cells[ca],aa,2)};
    for(std::int64_t kb=b;kb<e;++kb){
      const int cb=supportCell[kb],ab=(int)supportBasis[kb];
      const int cc=fineAgg[cb];
      double v=0.0;
      for(int d=0;d<3;++d)v+=ba[d]*g4_coeff_cell(cells[cb],ab,d);
      v*=rg;
      std::int64_t lo=crp[ra],hi=crp[ra+1];
      while(lo<hi){std::int64_t m=lo+(hi-lo)/2;int q=cci[m];if(q<cc)lo=m+1;else hi=m;}
      if(lo<crp[ra+1] && cci[lo]==cc) atomicAdd(cval+lo,v);
    }
  }
}

__global__ void g4_refresh_coarse_values_kernel(std::int64_t nnz,const double*src,const std::int32_t*dstSlot,double*dst){
  std::int64_t k=(std::int64_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(k<nnz) atomicAdd(dst+dstSlot[k],src[k]);
}

__global__ void g4_fine_bound_kernel(
  int nc,const G4CellPlanHost*cells,const std::int64_t*supportRow,
  const std::int32_t*supportCell,const std::uint8_t*supportBasis,
  const double*rau,const double*diag,double*bound)
{
  int c=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(c>=nc)return;
  const auto&cp=cells[c];double rowUpper=0.0;
  for(int a=0;a<8;++a){
    int g=cp.ref[a];if(g<0)continue;
    for(int d=0;d<3;++d){
      const double bc=fabs(g4_coeff_cell(cp,a,d));
      double sup=0.0;
      for(std::int64_t k=supportRow[g];k<supportRow[g+1];++k)
        sup+=fabs(g4_coeff_cell(cells[supportCell[k]],(int)supportBasis[k],d));
      rowUpper+=bc*rau[g]*sup;
    }
  }
  g4_atomic_max_positive(bound,rowUpper/diag[c]);
}

__global__ void g4_csr_bound_kernel(int n,const std::int64_t*rp,const double*val,const double*diag,double*bound){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i>=n)return;
  double s=0.0;for(std::int64_t k=rp[i];k<rp[i+1];++k)s+=fabs(val[k]);
  g4_atomic_max_positive(bound,s/diag[i]);
}

__global__ void g4_csr_to_dense_kernel(int n,const std::int64_t*rp,const std::int32_t*ci,const double*val,double*A){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)
    for(std::int64_t k=rp[i];k<rp[i+1];++k)A[(std::size_t)i*n+ci[k]]=val[k];
}

__global__ void g4_cholesky_inverse_kernel(int n,double*A,double*inv,int*status){
  if(blockIdx.x!=0)return;
  const int tid=(int)threadIdx.x;
  __shared__ double threshold;
  if(tid==0){
    *status=0;double md=0.0;
    for(int i=0;i<n;++i)md=fmax(md,fabs(A[(std::size_t)i*n+i]));
    threshold=fmax(1e-30,md*1e-20);
  }
  __syncthreads();

  for(int k=0;k<n;++k){
    if(tid==0){
      double s=A[(std::size_t)k*n+k];
      for(int j=0;j<k;++j){double q=A[(std::size_t)k*n+j];s-=q*q;}
      if(!(s>threshold) || !isfinite(s)){*status=k+1;A[(std::size_t)k*n+k]=1.0;}
      else A[(std::size_t)k*n+k]=sqrt(s);
    }
    __syncthreads();
    if(*status) return;
    const double piv=A[(std::size_t)k*n+k];
    for(int i=k+1+tid;i<n;i+=blockDim.x){
      double s=A[(std::size_t)i*n+k];
      for(int j=0;j<k;++j)s-=A[(std::size_t)i*n+j]*A[(std::size_t)k*n+j];
      A[(std::size_t)i*n+k]=s/piv;
    }
    __syncthreads();
  }

  for(int col=tid;col<n;col+=blockDim.x){
    for(int i=0;i<n;++i){
      double s=(i==col)?1.0:0.0;
      for(int k=0;k<i;++k)s-=A[(std::size_t)i*n+k]*inv[(std::size_t)k*n+col];
      inv[(std::size_t)i*n+col]=s/A[(std::size_t)i*n+i];
    }
    for(int i=n-1;i>=0;--i){
      double s=inv[(std::size_t)i*n+col];
      for(int k=i+1;k<n;++k)s-=A[(std::size_t)k*n+i]*inv[(std::size_t)k*n+col];
      inv[(std::size_t)i*n+col]=s/A[(std::size_t)i*n+i];
    }
  }
}

__global__ void g4_symmetrize_dense_kernel(int n,double*A){
  std::size_t k=(std::size_t)blockIdx.x*blockDim.x+threadIdx.x;
  const std::size_t nn=(std::size_t)n*n;if(k>=nn)return;
  int i=(int)(k/n),j=(int)(k-(std::size_t)i*n);
  if(j>i){double v=0.5*(A[(std::size_t)i*n+j]+A[(std::size_t)j*n+i]);A[(std::size_t)i*n+j]=v;A[(std::size_t)j*n+i]=v;}
}

__global__ void g5_sa_restrict_kernel(int n,const std::int64_t*rp,const std::int32_t*ci,const double*pv,const double*f,double*c){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i>=n)return;double x=f[i];
  for(std::int64_t k=rp[i];k<rp[i+1];++k)atomicAdd(c+ci[k],pv[k]*x);
}
__global__ void g5_sa_prolong_add_kernel(int n,const std::int64_t*rp,const std::int32_t*ci,const double*pv,const double*c,double*f){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i>=n)return;double s=0.0;
  for(std::int64_t k=rp[i];k<rp[i+1];++k)s+=pv[k]*c[ci[k]];f[i]+=s;
}
__global__ void g5_power_init_kernel(int n,double*v){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n){double g=(double)(i+1);v[i]=sin(.731*g)+.27*cos(1.117*g);}
}
__global__ void g5_div_sqrt_diag_kernel(int n,const double*x,const double*d,double*y){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)y[i]=x[i]/sqrt(d[i]);
}
__global__ void g5_diff_kernel(int n,const double*a,const double*b,double*y){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)y[i]=a[i]-b[i];
}

} // namespace nodals_gpu
