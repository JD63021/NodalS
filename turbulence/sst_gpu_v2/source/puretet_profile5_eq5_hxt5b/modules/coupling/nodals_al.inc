// NODALS_AL_B_P0_V1: exact algebraic augmentation on the active P0 rows.
// Included INSIDE hxt5d namespace by stage22d4d_coupled_matrixfree.inc.
// Continuity: B*u + csrc = 0.  Thus the absolute momentum RHS shift is
// -gamma * B^T * Mp^-1 * csrc. No standalone grad-div weak form.
// Pure tet only: P0 row t corresponds to tvel[t] and geom[t]; Mp(t,t)=|Kt|.
// The pinned (removed) continuity equation is excluded from both pieces.
__global__ void nodals_al_mass(int np,const nodals_hxt1::TetGeom*g,
    Real*z,double gamma,int pin){
  const int t=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(t>=np)return;
  z[t]=(t==pin)?Real(0):(Real)((double)z[t]*gamma/g[t].volume);
}
__global__ void nodals_al_source_mass(int np,const nodals_hxt1::TetGeom*g,
    const Real*csrc,Real*z,double gamma,int pin){
  const int t=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(t>=np)return;
  z[t]=(t==pin)?Real(0):(Real)(-gamma*(double)csrc[t]/g[t].volume);
}
__global__ void nodals_al_add(int nv,const Real*ax,const Real*ay,const Real*az,
    Real*x,Real*y,Real*z){
  const int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i>=nv)return;
  x[i]+=ax[i];y[i]+=ay[i];z[i]+=az[i];
}
// Add gamma B^T Mp^-1 B to the physical M3 9-block CSR, BEFORE
// the existing T^T(..)T BF3 tangential projection. B includes DG inlet.
// Missing local columns after row cleanup represent exactly zero entries.
__global__ void nodals_al_m3_add(int np,
    const std::int64_t*brow,const std::int32_t*bcol,
    const Real*bx,const Real*by,const Real*bz,
    const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*g,
    const std::int32_t*tetSlot,std::size_t nnz,
    Real*k,double gamma,int pin,unsigned long long*bad){
  const int t=(int)blockIdx.x,p=(int)threadIdx.x;
  if(t>=np||p>=64||t==pin)return;
  const std::int64_t start=brow[t],end=brow[t+1];
  if(end-start>8 || !(g[t].volume>0.) || !isfinite(g[t].volume)){
    if(p==0)atomicAdd(bad,1ULL);return;
  }
  const auto local=tv[t];
  if(p==0){
    for(auto j=start;j<end;++j){
      bool found=false;
      for(int l=0;l<8;++l)found|=(bcol[j]==local.g[l]);
      if(!found)atomicAdd(bad,1ULL);
    }
  }
  const int a=p/8,b=p%8;
  const int ia=local.g[a],ib=local.g[b];
  double va[3]={0.,0.,0.},vb[3]={0.,0.,0.};
  for(auto j=start;j<end;++j){
    if(bcol[j]==ia){va[0]=bx[j];va[1]=by[j];va[2]=bz[j];}
    if(bcol[j]==ib){vb[0]=bx[j];vb[1]=by[j];vb[2]=bz[j];}
  }
  const int slot=tetSlot[(std::size_t)t*64u+(std::size_t)p];
  if(slot<0){atomicAdd(bad,1ULL);return;}
  const double fac=gamma/g[t].volume;
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){
    const double value=fac*va[i]*vb[j];
    if(value!=0.)atomicAdd(k+((std::size_t)i*3+j)*nnz+(std::size_t)slot,(Real)value);
  }
}
// Diagonal surrogate for the *augmented* momentum inverse in I3A.
// IMPORTANT: this is a SIMPLE approximation, NOT the exact Schur of A_gamma.
// I3A applies identical rx/ry/rz to the pressure action and velocity correction.
__global__ void nodals_al_diag(int np,
    const std::int64_t*row,const std::int32_t*col,
    const Real*bx,const Real*by,const Real*bz,
    const nodals_hxt1::TetGeom*g,double gamma,int pin,
    Real*dx,Real*dy,Real*dz){
  const int t=(int)blockIdx.x,k=(int)threadIdx.x;
  if(t>=np||t==pin)return;
  const auto idx=row[t]+k;
  if(idx>=row[t+1])return;
  const int node=col[idx];
  const double f=gamma/g[t].volume;
  atomicAdd(dx+node,(Real)(f*(double)bx[idx]*(double)bx[idx]));
  atomicAdd(dy+node,(Real)(f*(double)by[idx]*(double)by[idx]));
  atomicAdd(dz+node,(Real)(f*(double)bz[idx]*(double)bz[idx]));
}
__global__ void nodals_al_i3_diag_update(int n,
    const Real*dx,const Real*dy,const Real*dz,Real*rx,Real*ry,Real*rz){
  const int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i>=n)return;
  const double x=rx[i],y=ry[i],z=rz[i];
  rx[i]=(Real)(x/(1.+x*(double)dx[i]));
  ry[i]=(Real)(y/(1.+y*(double)dy[i]));
  rz[i]=(Real)(z/(1.+z*(double)dz[i]));
}
