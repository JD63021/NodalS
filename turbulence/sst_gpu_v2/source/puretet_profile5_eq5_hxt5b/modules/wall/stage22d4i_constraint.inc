// Stage22D4I1: Exact wall-face MEAN flux constraint for TET P1+BF3.
// On each affine triangle, b_f = 27 l1*l2*l3 and <b_f> = 9/20.
// Face BF3 coefficients are globally independent and shared vertex P1
// coefficients are left free.  This is a TRUE Galerkin elimination:
//    u = T x,   A_red x = T^T A T x,   rhs_red = T^T rhs.
// x carries a redundant normal bubble coordinate, fixed to zero with Q.
// Included inside hxt5d namespace; uses existing TetFacePlan, TetVel.

__global__ void s22d4i_T_apply(int nf,const hxt5b::TetFacePlan*faces,
 const nodals_hxt1::TetVel*tv,Real*ux,Real*uy,Real*uz){
 const int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i>=nf)return;
 const auto f=faces[i];const auto t=tv[f.cell];
 const int b=t.g[4+(int)f.lf];
 const double nx=f.n[0],ny=f.n[1],nz=f.n[2];
 double sn=0.;
 for(int a=0;a<3;++a){
   const int j=t.g[hxt2::c_tet_face[(int)f.lf][a]];
   sn+=nx*(double)ux[j]+ny*(double)uy[j]+nz*(double)uz[j];
 }
 const double bn=nx*(double)ux[b]+ny*(double)uy[b]+nz*(double)uz[b];
 const double delta=-bn-(20./27.)*sn;
 ux[b]=(Real)((double)ux[b]+nx*delta);
 uy[b]=(Real)((double)uy[b]+ny*delta);
 uz[b]=(Real)((double)uz[b]+nz*delta);
}

// Q removes only redundant BF3-normal coordinates in PARAMETER space.
__global__ void s22d4i_Q_apply(int nf,const hxt5b::TetFacePlan*faces,
 const nodals_hxt1::TetVel*tv,Real*ux,Real*uy,Real*uz){
 const int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i>=nf)return;
 const auto f=faces[i];const int b=tv[f.cell].g[4+(int)f.lf];
 const double nx=f.n[0],ny=f.n[1],nz=f.n[2];
 const double bn=nx*(double)ux[b]+ny*(double)uy[b]+nz*(double)uz[b];
 ux[b]=(Real)((double)ux[b]-nx*bn);
 uy[b]=(Real)((double)uy[b]-ny*bn);
 uz[b]=(Real)((double)uz[b]-nz*bn);
}

// T^T acts on momentum TEST functions. Bubble-normal is eliminated,
// and its residual force is transferred to the three P1 vertex residuals.
// Each wall-bubble DOF is unique; P1 nodes can be shared and use atomics.
__global__ void s22d4i_Tt_apply(int nf,const hxt5b::TetFacePlan*faces,
 const nodals_hxt1::TetVel*tv,Real*yx,Real*yy,Real*yz){
 const int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i>=nf)return;
 const auto f=faces[i];const auto t=tv[f.cell];
 const int b=t.g[4+(int)f.lf];
 const double nx=f.n[0],ny=f.n[1],nz=f.n[2];
 const double bn=nx*(double)yx[b]+ny*(double)yy[b]+nz*(double)yz[b];
 yx[b]=(Real)((double)yx[b]-nx*bn);
 yy[b]=(Real)((double)yy[b]-ny*bn);
 yz[b]=(Real)((double)yz[b]-nz*bn);
 const double coeff=-(20./27.)*bn;
 for(int a=0;a<3;++a){
   const int j=t.g[hxt2::c_tet_face[(int)f.lf][a]];
   atomicAdd(yx+j,(Real)(coeff*nx));
   atomicAdd(yy+j,(Real)(coeff*ny));
   atomicAdd(yz+j,(Real)(coeff*nz));
 }
}
