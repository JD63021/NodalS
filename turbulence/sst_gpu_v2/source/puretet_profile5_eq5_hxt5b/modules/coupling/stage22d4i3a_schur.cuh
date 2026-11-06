// STAGE 22D.4I.3A: constrained SIMPLE Schur from a GAMMA-FREE scalar row-L1
// approximation, lifted via the exact BF3 T/T^T map.
// Included within namespace hxt5d.  Does NOT modify the physical momentum
// solve; does NOT remove the existing legacy X/Y wall clamps (I3B).
//
// C = T Q D0^{-1} Q T^T, where Q kills redundant bubble-normal coordinates.
// S = B C B^T, and the correction MUST be u <- u - C B^T dp.
// D0 is the penalty-free componentwise row-L1 surrogate (NOT the exact row-L1
// of the full T^T A0 T operator).  This is the intentionally cheap I3A test.
__global__ void s22d4i3a_refinv(int nv, const std::int64_t*row,
 const std::int32_t*diagPos,
 const Real*Ax,const Real*Ay,const Real*Az,
 const Real*penX,const Real*penY,const Real*penZ,
 const Real*relX,const Real*relY,const Real*relZ,
 const unsigned char*fixXY,const unsigned char*fixZ,
 double urU,double scale,Real*rx,Real*ry,Real*rz,
 unsigned long long*bad){
 const int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i>=nv)return;
 const int d=diagPos[i];
 const Real* v[3]={Ax,Ay,Az};
 const Real* pen[3]={penX,penY,penZ};
 const Real* delta[3]={relX,relY,relZ};
 Real* r[3]={rx,ry,rz};
 const double fac=1./urU-1.;
 for(int c=0;c<3;++c){
   if((c==2?fixZ[i]:fixXY[i])){r[c][i]=Real(0);continue;}
   double l1=0., diag=0.;
   for(std::int64_t k=row[i];k<row[i+1];++k){
     const double raw=(double)v[c][k]-(double)pen[c][k]
                      -(k==(std::int64_t)d?(double)delta[c][i]:0.);
     if(k==(std::int64_t)d)diag=raw;
     l1+=fabs(raw);
   }
   const double denom=diag+fac*l1;
   if(!(denom>0.)||!isfinite(denom)||!isfinite(l1)){
     atomicAdd(bad,1ULL);r[c][i]=Real(0);
   }else r[c][i]=(Real)(scale/denom);
 }
}

__global__ void s22d4i3a_velocity_subtract(int nv,Real*ux,Real*uy,Real*uz,
 const Real*dx,const Real*dy,const Real*dz){
 const int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i>=nv)return;
 ux[i]-=dx[i];uy[i]-=dy[i];uz[i]-=dz[i];
}

// Diagnostic identity: post - pre + S(dp) = 0.
__global__ void s22d4i3a_continuity_identity(int n,const Real*pre,
 const Real*post,const Real*sdp,Real*err){
 const int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
 if(i<n)err[i]=post[i]-pre[i]+sdp[i];
}

// Allocations belong to app lifetime, not to each pressure matvec.
template<class T>struct S22D4I3ASchur {
 hxt4a::BDevice<T>&B;
 const hxt5b::TetFacePlan*faces;
 const nodals_hxt1::TetVel*tv;
 int nfaces,pin;
 const T*rx,*ry,*rz;
 T *ux,*uy,*uz;
 void correct(const T*dp){
   const int nv=B.nv;
   hxt4a::BT_apply(B,dp,ux,uy,uz);
   s22d4i_Tt_apply<<<(nfaces+255)/256,256>>>(nfaces,faces,tv,ux,uy,uz);
   hxt5b::scale3<<<(nv+255)/256,256>>>(nv,rx,ry,rz,ux,uy,uz);
   // Componentwise diagonal scaling need not preserve bubble tangency.
   s22d4i_Q_apply<<<(nfaces+255)/256,256>>>(nfaces,faces,tv,ux,uy,uz);
   s22d4i_T_apply<<<(nfaces+255)/256,256>>>(nfaces,faces,tv,ux,uy,uz);
   HXT1_CUDA(cudaGetLastError());
 }
 void apply(const T*p,T*out){
   correct(p);
   hxt4a::B_apply(B,ux,uy,uz,out);
   hxt4a::pin_output_zero_kernel<T><<<1,1>>>(pin,out);
 }
};
