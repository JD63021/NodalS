// Stage22D.4I.2 — residual of FROZEN PICARD MOMENTUM FORM at post-SIMPLE state.
// This file is included inside namespace hxt5d in hxt5d_sst_g2.cuh.
// b_rel is the T^T-transformed predictor RHS after removing lagged off-block
// terms; Arel is the full coupled, row-L1-relaxed Nitsche/Robin operator.
// Identity: r_phys^red = T^T[b_rel-Arel*u_new
//   + relaxationDelta*(u_new-u_old) - alphaP*(BT(dp))].
// This is a pressure-consistent, de-relaxed, reduced FROZEN-PICARD residual.
// It is NOT a reassembled Jacobian-free nonlinear residual at updated SST fields.
__global__ void s22d4i2_adjust(int n,const Real*ux,const Real*uy,const Real*uz,
    const Real*uold,const Real*rx,const Real*rz,
    const Real*dp0,const Real*dp1,const Real*dp2,double alphaP,Real*adj){
  const int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i>=3*n)return;
  const int k=i/n,j=i%n;
  const Real*un = k==0?ux:(k==1?uy:uz);
  const Real*pd = k==0?dp0:(k==1?dp1:dp2);
  const Real*rel = k==2?rz:rx;
  adj[i]=(Real)((double)rel[j]*((double)un[j]-(double)uold[i])
             -alphaP*(double)pd[j]);
}
__global__ void s22d4i2_residual(int n,const Real*b,const Real*Au,
    const Real*adj,const unsigned char*fixedXY,const unsigned char*fixedZ,
    Real*r){
  const int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i>=3*n)return;
  const int k=i/n,j=i%n;
  r[i]=((k==2?fixedZ[j]:fixedXY[j]))?Real(0):
       (Real)((double)b[i]-(double)Au[i]+(double)adj[i]);
}
