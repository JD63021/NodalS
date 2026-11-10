// Bazilevs-B experiment: add ONLY the symmetric tangential Nitsche
// consistency pair to the existing Spalding Robin wall law.
// Pure tetrahedral P1+BF3, homogeneous stationary wall, physical velocity trace.
// Mode 0: disabled; 1: molecular nu; 2: nu + SST nut frozen from F1B/F2.
// The BF3 mean normal-flux constraint, normal penalty, omega wall state,
// Spalding fit and all existing preconditioners are unchanged.
// Included inside namespace hxt5d via stage22d4d_coupled_matrixfree.inc.

__device__ __forceinline__ double bazb_tangent_entry(
    int i,int j,const double n[3],
    double pa,double pb,const double ga[3],const double gb[3],double mu){
  const double dna=ga[0]*n[0]+ga[1]*n[1]+ga[2]*n[2];
  const double dnb=gb[0]*n[0]+gb[1]*n[1]+gb[2]*n[2];
  const double pij=(i==j?1.0:0.0)-n[i]*n[j];
  // -<w_t,2 mu D(u)n> - <2 mu D(w)n,u_t>.
  // P_t removes the isotropic -2/3 mu div(u) I contribution.
  return -mu*((pa*dnb+dna*pb)*pij
      +pa*(gb[i]-n[i]*dnb)*n[j]
      +pb*n[i]*(ga[j]-n[j]*dna));
}

__device__ __forceinline__ double bazb_mu(
    const double*coeff,int fi,int q,double molecularNu,int viscosityMode){
  return viscosityMode==1?molecularNu:
    coeff[2u*((std::size_t)fi*12u+(std::size_t)q)];
}

// Called once per SIMPLE iteration by the true 9-block M3 CSR constructor,
// BEFORE its BF3 T^t A T transformation. One thread per local (a,b) pair.
__global__ void bazb_m3_add(
    const hxt5b::TetFacePlan*F,int nf,
    const hxt5b::TetWallQuadCache*qc,
    const nodals_hxt1::TetGeom*tg,
    const std::int32_t*tetSlot,const double*coeff,
    double molecularNu,int viscosityMode,std::size_t nnz,
    Real*blocks,unsigned long long*bad){
  const int fi=(int)blockIdx.x,pair=(int)threadIdx.x;
  if(fi>=nf||pair>=64)return;
  const int a=pair/8,b=pair%8;
  const auto face=F[fi];const auto geom=tg[face.cell];
  const double n[3]={face.n[0],face.n[1],face.n[2]};
  const int position=tetSlot[(std::size_t)face.cell*64u+(std::size_t)pair];
  if(position<0 || (std::size_t)position>=nnz){
    atomicAdd(bad,1ULL);return;
  }
  double sum[3][3]{};
  for(int q=0;q<12;++q){
    const auto&ca=qc[(std::size_t)fi*12u+(std::size_t)q];
    const double mu=bazb_mu(coeff,fi,q,molecularNu,viscosityMode);
    if(!(mu>0.0)||!isfinite(mu)||!(ca.w>0.0)||!isfinite(ca.w)){
      atomicAdd(bad,1ULL);return;
    }
    double lam[4]={0.,0.,0.,0.};
    lam[hxt2::c_tet_face[(int)face.lf][0]]=hxt5b::QL0[q];
    lam[hxt2::c_tet_face[(int)face.lf][1]]=hxt5b::QL1[q];
    lam[hxt2::c_tet_face[(int)face.lf][2]]=hxt5b::QL2[q];
    double ph[8],gr[8][3],gp[8][3];
    hxt2::tet_basis(lam[0],lam[1],lam[2],lam[3],ph,gr);
    hxt2::tet_phys_grad(gr,geom,gp);
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)
      sum[i][j]+=ca.w*bazb_tangent_entry(
        i,j,n,ca.bw[a],ca.bw[b],gp[a],gp[b],mu);
  }
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)
    atomicAdd(blocks+((std::size_t)i*3u+(std::size_t)j)*nnz
      +(std::size_t)position,(Real)sum[i][j]);
}

// Reference matrix-free evaluation for the existing full frozen-operator
// residual/parity checks. Called BEFORE T^t projection, just like F1B.
__global__ void bazb_tangent_action(
    const hxt5b::TetFacePlan*F,int nf,
    const hxt5b::TetWallQuadCache*qc,
    const nodals_hxt1::TetVel*tv,
    const nodals_hxt1::TetGeom*tg,
    const double*coeff,double molecularNu,int viscosityMode,
    int nv,const Real*x,Real*y){
  const int fi=(int)blockIdx.x,a=(int)threadIdx.x;
  if(fi>=nf||a>=8)return;
  const auto face=F[fi];const auto geom=tg[face.cell];
  const auto cell=tv[face.cell];
  const double n[3]={face.n[0],face.n[1],face.n[2]};
  double sum[3]={};
  for(int q=0;q<12;++q){
    const auto&ca=qc[(std::size_t)fi*12u+(std::size_t)q];
    const double mu=bazb_mu(coeff,fi,q,molecularNu,viscosityMode);
    double lam[4]={0.,0.,0.,0.};
    lam[hxt2::c_tet_face[(int)face.lf][0]]=hxt5b::QL0[q];
    lam[hxt2::c_tet_face[(int)face.lf][1]]=hxt5b::QL1[q];
    lam[hxt2::c_tet_face[(int)face.lf][2]]=hxt5b::QL2[q];
    double ph[8],gr[8][3],gp[8][3];
    hxt2::tet_basis(lam[0],lam[1],lam[2],lam[3],ph,gr);
    hxt2::tet_phys_grad(gr,geom,gp);
    for(int b=0;b<8;++b){
      const int ib=cell.g[b];
      for(int i=0;i<3;++i)for(int j=0;j<3;++j){
        const double kab=ca.w*bazb_tangent_entry(
          i,j,n,ca.bw[a],ca.bw[b],gp[a],gp[b],mu);
        sum[i]+=kab*(double)x[(std::size_t)j*(std::size_t)nv
          +(std::size_t)ib];
      }
    }
  }
  const int ia=cell.g[a];
  for(int i=0;i<3;++i)
    atomicAdd(y+(std::size_t)i*(std::size_t)nv+(std::size_t)ia,(Real)sum[i]);
}
