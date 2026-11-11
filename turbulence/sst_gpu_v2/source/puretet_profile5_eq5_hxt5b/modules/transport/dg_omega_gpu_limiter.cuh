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
 unsigned long long stalledCells; // local alpha reduced to 0; no positive mass destroyed
 unsigned long long heavilyBacktrackedCells; // alphaLocal < 0.1 requested alpha
};
__global__ void relax_and_limit(int nt,const double*previous,const double*candidate,
                               double*updated,double alpha,double omegaFloor,
                               LimiterCounters*counts){
 const int t=blockIdx.x*blockDim.x+threadIdx.x;
 if(t>=nt)return;
 // Preserve the previous positivity-constrained DG cell average. If an
 // inexact block-GS candidate would drive the *under-relaxed cell mean* below
 // omegaFloor, backtrack the local Picard step rather than replacing the
 // negative mean with the floor (a hidden nonconservative source).
 double oldMean=0.,candidateMean=0.;
 for(int i=0;i<4;++i){
   const double x=candidate[4*t+i],p=previous[4*t+i];
   if(!isfinite(x)||!isfinite(p)){
     atomicAdd(&counts->nonfiniteCells,1ULL);return;
   }
   if(x<0.)atomicAdd(&counts->negativeCandidateDofs,1ULL);
   oldMean+=0.25*p;candidateMean+=0.25*x;
 }
 if(!(omegaFloor>0.)||!(alpha>0.&&alpha<=1.)||
    !isfinite(oldMean)||!isfinite(candidateMean)||
    oldMean<omegaFloor*(1.-1.e-10)){
   atomicAdd(&counts->nonfiniteCells,1ULL);return;
 }
 double alphaLocal=alpha;
 const double requestedMean=oldMean+alpha*(candidateMean-oldMean);
 if(requestedMean<omegaFloor){
   atomicAdd(&counts->meanDeficitCells,1ULL); // REQUIRED backtrack count
   if(!(candidateMean<oldMean)){
     atomicAdd(&counts->nonfiniteCells,1ULL);return;
   }
   const double allowed=(oldMean-omegaFloor)/(oldMean-candidateMean);
   alphaLocal=fmax(0.,fmin(alpha,0.95*fmax(0.,allowed)));
   if(alphaLocal==0.)atomicAdd(&counts->stalledCells,1ULL);
   if(alphaLocal<alpha*0.1)
     atomicAdd(&counts->heavilyBacktrackedCells,1ULL);
 }
 double w[4],mean=0.,minval=1.e300;
 for(int i=0;i<4;++i){
   w[i]=previous[4*t+i]+alphaLocal*
        (candidate[4*t+i]-previous[4*t+i]);
   if(!isfinite(w[i])){atomicAdd(&counts->nonfiniteCells,1ULL);return;}
   mean+=0.25*w[i];minval=fmin(minval,w[i]);
 }
 if(mean<omegaFloor*(1.-1.e-10)){
   atomicAdd(&counts->nonfiniteCells,1ULL);return;
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
