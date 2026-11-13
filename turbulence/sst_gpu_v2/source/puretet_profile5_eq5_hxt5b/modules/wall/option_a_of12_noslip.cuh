// OPTION_A_OF12_NOSLIP_V2: experimental strong no-slip + shear-only replacement.
// Velocity wall vertex/face-BF3 DOFs are exactly fixed; scalar fields unchanged.
// F2 five-point Spalding supplies u_tau. For unique wall-owner tetrahedra
// nu_wall=max(nu,u_tau^2/|P_t grad_n U|). Unlike v1, this coefficient is used
// ONLY in the wall-normal tangential deviatoric shear subspace via the V2
// correction operator; unrelated volume gradient directions use SST nu_eff.
// A fail-fast prevents unbounded viscosity from low-gradient cancellation.
static __device__ int option_a_noslip_device = 0;

__global__ void option_a_build_wall_nu(
    const hxt5b::TetFacePlan* faces,int nf,
    const nodals_hxt1::TetVel* tv,
    const nodals_hxt1::TetGeom* tg,
    const Real* ux,const Real* uy,const Real* uz,
    const double* spaldingFaceFit,
    Real* wallNu,double nu,
    double* stats,unsigned long long* bad)
{
  const int fi=(int)blockIdx.x*blockDim.x+(int)threadIdx.x;
  if(fi>=nf)return;
  const auto f=faces[fi];
  const auto t=tv[f.cell];
  const auto geo=tg[f.cell];
  const double n[3]={f.n[0],f.n[1],f.n[2]};
  const double utau=spaldingFaceFit[3ull*(std::size_t)fi+1ull];
  if(!(isfinite(utau)&&utau>=0.&&isfinite(nu)&&nu>0.)){
    atomicAdd(bad,1ULL);return;
  }

  double derivative[3]={0.,0.,0.},weightSum=0.;
  for(int q=0;q<12;++q){
    double lam[4]={0.,0.,0.,0.};
    lam[hxt2::c_tet_face[(int)f.lf][0]]=hxt5b::QL0[q];
    lam[hxt2::c_tet_face[(int)f.lf][1]]=hxt5b::QL1[q];
    lam[hxt2::c_tet_face[(int)f.lf][2]]=hxt5b::QL2[q];
    double phi[8],gref[8][3],gphys[8][3];
    hxt2::tet_basis(lam[0],lam[1],lam[2],lam[3],phi,gref);
    hxt2::tet_phys_grad(gref,geo,gphys);
    double gradn[3]={0.,0.,0.};
    for(int b=0;b<8;++b){
      const int id=t.g[b];
      const double dn=gphys[b][0]*n[0]+
                      gphys[b][1]*n[1]+gphys[b][2]*n[2];
      gradn[0]+=(double)ux[id]*dn;
      gradn[1]+=(double)uy[id]*dn;
      gradn[2]+=(double)uz[id]*dn;
    }
    const double w=hxt5b::QW12[q];
    for(int j=0;j<3;++j)derivative[j]+=w*gradn[j];
    weightSum+=w;
  }
  if(!(weightSum>0.&&isfinite(weightSum))){atomicAdd(bad,1ULL);return;}
  for(int i=0;i<3;++i)derivative[i]/=weightSum;
  const double gn=derivative[0]*n[0]+derivative[1]*n[1]+derivative[2]*n[2];
  double g2=0.;
  for(int i=0;i<3;++i){
    const double x=derivative[i]-gn*n[i];
    g2+=x*x;
  }
  const double g=sqrt(fmax(0.,g2));
  double nw;
  if(g<1.e-12){
    if(utau>1.e-8){atomicAdd(bad,1ULL);return;}
    nw=nu;
  }else nw=fmax(nu,utau*utau/g);
  // Prevent an almost-cancelled face-gradient from silently creating a
  // catastrophically large finite matrix coefficient. This is a diagnostic
  // failure threshold, NOT a clipped viscosity model or tuning parameter.
  if(!(nw>=nu && isfinite(nw)) || nw/nu>1.e6){
    const auto prior=atomicAdd(bad,1ULL);
    if(prior<12ULL)
      printf("NODALS_OPTION_A_UNBOUNDED_FACE face=%d utau=%.8e g=%.8e nuwall=%.8e ratio=%.8e\n",
             fi,utau,g,nw,nw/nu);
    return;
  }
  wallNu[f.cell]=(Real)nw; // unique face-owner tet is asserted on host.
  if(stats){
    atomicAdd(stats+0,f.area);
    atomicAdd(stats+1,f.area*utau*utau);
    atomicAdd(stats+2,f.area*(nw/nu));
  }
}


// Post-pressure-correction audit of the exact constrained P1+BF3 wall trace.
// Each wall face has three P1 vertex DOFs and one independent BF3 face DOF.
__global__ void option_a_audit_wall_dofs(
    const hxt5b::TetFacePlan*faces,int nf,
    const nodals_hxt1::TetVel*tv,
    const Real*ux,const Real*uy,const Real*uz,double*sumSq)
{
  const int fi=(int)blockIdx.x*blockDim.x+(int)threadIdx.x;
  if(fi>=nf)return;
  const auto f=faces[fi];
  const auto t=tv[f.cell];
  double v2=0.;
  for(int k=0;k<4;++k){
    const int a=k==3 ? 4+(int)f.lf : hxt2::c_tet_face[(int)f.lf][k];
    const int gid=t.g[a];
    const double x=(double)ux[gid],y=(double)uy[gid],z=(double)uz[gid];
    v2+=x*x+y*y+z*z;
  }
  atomicAdd(sumSq,v2);
}
