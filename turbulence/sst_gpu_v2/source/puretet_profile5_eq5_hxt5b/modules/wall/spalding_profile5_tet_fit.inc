// Stage13: tetrahedral interpolation retained in CUDA; five-point fit is backend-neutral.
template<class T>
__device__ __forceinline__ bool spalding_match_profile5_tet(
    const TetWallPackedPoint*mp,int fi,int q,const T*u,double nu,
    double&ut,double&ypMid,double&beta,double&uMid,double&fitRms,double&pointUTauRelSpan)
{
  if(!mp||!(nu>0.0))return false;
  double U[5]={0,0,0,0,0},Usigned[5]={0,0,0,0,0},y[5]={0,0,0,0,0};
  const unsigned stage27ActiveMask=
      ::nodals::wall::active::active_slot_mask(::stage24b_active_points);
  #pragma unroll
  for(int m=0;m<5;++m){
    const TetWallMatchPoint&pm=stage27_s02b_at(mp,fi,q,m);
    if(pm.cell<0||!(pm.y>0.0))return false;
    if(!(stage27ActiveMask&(1u<<m))){
      // Retain legacy topology validity checks even when not interpolating.
      for(int a=0;a<8;++a)if(pm.g[a]<0)return false;
      y[m]=stage27_s02b_y(m);
      continue;
    }
    double um=0.0;
    #pragma unroll
    for(int a=0;a<8;++a){
      const int g=pm.g[a];
      if(g<0)return false;
      um+=(double)u[g]*pm.b[a];
    }
    if(!isfinite(um))return false;
    Usigned[m]=um;U[m]=fabs(um);y[m]=pm.y;
  }
  return nodals::wall::active::fit_values(
      U,Usigned,y,::stage24b_active_points,nu,ut,ypMid,beta,uMid,fitRms,pointUTauRelSpan);
}

// STAGE25A_FIX2_OF_AUTO_VECTOR_FIT: evaluate exactly the same physical
// five-slot wall samples as momentum, using the full face-tangential speed.
// Keep the legacy scalar-Z function above for backward compatibility.
// Tangential speed is invariant under rigid rotation of both velocity and n.
template<class T>
__device__ __forceinline__ bool spalding_match_profile5_tet_vector(
    const TetWallPackedPoint*mp,int fi,int q,
    const T*u0,const T*u1,const T*u2,const double normal[3],double nu,
    double&ut,double&ypMid,double&beta,double&uMid,
    double&fitRms,double&pointUTauRelSpan,
    ::nodals::wall::shared27::Record* stage27Saved=nullptr)
{
  if(stage27Saved)stage27Saved->valid=0;
  if(!mp||!u0||!u1||!u2||!normal||!(nu>0.0))return false;
  const double n2=normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2];
  if(!(fabs(n2-1.0)<1.0e-8))return false;
  double U[5]={0.,0.,0.,0.,0.},Usigned[5]={0.,0.,0.,0.,0.};
  double y[5]={0.,0.,0.,0.,0.};
  const unsigned stage27ActiveMask=
      ::nodals::wall::active::active_slot_mask(::stage24b_active_points);
  #pragma unroll
  for(int m=0;m<5;++m){
    const TetWallMatchPoint&pm=stage27_s02b_at(mp,fi,q,m);
    if(pm.cell<0||!(pm.y>0.0))return false;
    if(!(stage27ActiveMask&(1u<<m))){
      for(int a=0;a<8;++a)if(pm.g[a]<0)return false;
      y[m]=stage27_s02b_y(m);
      continue;
    }
    double uv[3]={0.,0.,0.};
    #pragma unroll
    for(int a=0;a<8;++a){
      const int g=pm.g[a];
      if(g<0)return false;
      const double ph=pm.b[a];
      uv[0]+=(double)u0[g]*ph;
      uv[1]+=(double)u1[g]*ph;
      uv[2]+=(double)u2[g]*ph;
    }
    if(!isfinite(uv[0])||!isfinite(uv[1])||!isfinite(uv[2]))return false;
    const double un=uv[0]*normal[0]+uv[1]*normal[1]+uv[2]*normal[2];
    const double ut0=uv[0]-un*normal[0];
    const double ut1=uv[1]-un*normal[1];
    const double ut2=uv[2]-un*normal[2];
    const double mag=sqrt(ut0*ut0+ut1*ut1+ut2*ut2);
    if(!isfinite(mag))return false;
    U[m]=mag;Usigned[m]=mag;y[m]=pm.y;
  }
  const bool fitOK=nodals::wall::active::fit_values(
      U,Usigned,y,::stage24b_active_points,nu,
      ut,ypMid,beta,uMid,fitRms,pointUTauRelSpan);
  if(stage27Saved && fitOK){
    for(int m=0;m<5;++m)stage27Saved->magnitude[m]=U[m];
    stage27Saved->ut=ut;stage27Saved->yp=ypMid;
    stage27Saved->betaFit=beta;stage27Saved->uReference=uMid;
    stage27Saved->rms=fitRms;stage27Saved->span=pointUTauRelSpan;
    stage27Saved->nu=nu;stage27Saved->count=::stage24b_active_points;
    stage27Saved->valid=1;
  }
  return fitOK;
}

