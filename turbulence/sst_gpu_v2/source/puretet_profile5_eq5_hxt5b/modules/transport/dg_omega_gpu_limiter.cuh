#pragma once
// Device-only omega DG-P1 update: preserve cell mean during positivity limiting.
// Coordinates are tetra-local: omega[4*tet + 0..3].
//
// For a Bernstein P1 polynomial the four control values are the extrema,
// so scaling about the cell mean is a positivity-preserving polynomial limiter.
// This operation is after the inexact linear solve and under-relaxation;
// do not clip each DOF independently (that changes the cell mean).
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
namespace nodals {namespace omega_dg_gpu {
struct LimiterCounters {
 unsigned long long limitedCells;
 unsigned long long meanDeficitCells;
 unsigned long long negativeCandidateDofs;
 unsigned long long nonfiniteCells;
};
__global__ void relax_and_limit(int nt,const double*previous,const double*candidate,
                               double*updated,double alpha,double omegaFloor,
                               LimiterCounters*counts){
 const int t=blockIdx.x*blockDim.x+threadIdx.x;
 if(t>=nt)return;
 double w[4],mean=0.,minval=1.e300;
 for(int i=0;i<4;++i){
   const double x=candidate[4*t+i];
   if(x<0.)atomicAdd(&counts->negativeCandidateDofs,1ULL);
   w[i]=previous[4*t+i]+alpha*(x-previous[4*t+i]);
   if(!isfinite(w[i])){atomicAdd(&counts->nonfiniteCells,1ULL);return;}
   mean+=.25*w[i];minval=fmin(minval,w[i]);
 }
 if(!(omegaFloor>0.)||!isfinite(mean)){atomicAdd(&counts->nonfiniteCells,1ULL);return;}
 if(mean<omegaFloor){
   // Cannot preserve both an inadmissible sub-floor mean and positivity.
   // Expose it in diagnostics instead of hiding the event.
   atomicAdd(&counts->meanDeficitCells,1ULL);
   for(int i=0;i<4;++i)updated[4*t+i]=omegaFloor;
   return;
 }
 double theta=1.;
 if(minval<omegaFloor){
   theta=(mean-omegaFloor)/(mean-minval);
   theta=fmin(1.,fmax(0.,theta));
   atomicAdd(&counts->limitedCells,1ULL);
 }
 for(int i=0;i<4;++i){
   double x=mean+theta*(w[i]-mean);
   updated[4*t+i]=fmax(x,omegaFloor);
 }
}
__global__ void cg_vertex_visualization_projection(
  int nt,const int*cellVertex,int nvertex,const double*omegaDG,
  double*sum,unsigned int*count){
 const int t=blockIdx.x*blockDim.x+threadIdx.x;
 if(t>=nt)return;
 for(int a=0;a<4;++a){
   int v=cellVertex[4*t+a];
   if(v>=0&&v<nvertex){
     atomicAdd(sum+v,omegaDG[4*t+a]);
     atomicAdd(count+v,1u);
   }
 }
}
__global__ void cg_vertex_visualization_finalize(int nvertex,double*sum,const unsigned int*count){
 const int v=blockIdx.x*blockDim.x+threadIdx.x;
 if(v<nvertex && count[v])sum[v]/=(double)count[v];
}
// The projection above is ONLY for output. It must not feed into F1/F2,
// the omega/K RHS, or the momentum eddy-viscosity calculation.
}} // namespace nodals::omega_dg_gpu
