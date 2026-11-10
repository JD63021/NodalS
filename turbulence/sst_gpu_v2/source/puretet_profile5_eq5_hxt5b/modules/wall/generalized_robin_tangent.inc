// Generalized Robin-Nitsche correction for the existing physical Spalding
// Robin momentum wall operator. Reference: Juntunen & Stenberg (2009),
// applied componentwise to projected deviatoric viscous traction.
// Included INSIDE namespace hxt5d, after bazilevs_b_tangent.inc.
//
// Existing F1B/F2 matrix already contains + beta_w <w_t,u_t>. We add only
// (A_generalized - A_original) to avoid double counting wall resistance.
// beta_w comes from the UNMODIFIED off-wall F2 Spalding fit.
// beta_N = stabilizationC * nu_c / (3*tetVolume/wallFaceArea).
// C_stab controls numerical stabilization; F2 ray/sample y is independent.
// 0=off; 1=molecular nu; 2=frozen nu+nut consistency. Both symmetric
// consistency terms and the negative traction-traction correction are kept.

__device__ __forceinline__ double genrobin_entry(
   int i,int j,const double n[3],double pa,double pb,
   const double ga[3],const double gb[3],
   double mu,double beta,double betaN){
  const double den=beta+betaN;
  const double alpha=beta/den;
  const double dni=ga[0]*n[0]+ga[1]*n[1]+ga[2]*n[2];
  const double dnj=gb[0]*n[0]+gb[1]*n[1]+gb[2]*n[2];
  const double pij=(i==j?1.0:0.0)-n[i]*n[j];

  // t_t(phi_a e_i) = mu * [P e_i * dn_a + n_i P grad(phi_a)].
  double dot=0.0;
  for(int k=0;k<3;++k){
    const double pki=(k==i?1.0:0.0)-n[k]*n[i];
    const double pkj=(k==j?1.0:0.0)-n[k]*n[j];
    const double tA=mu*(pki*dni+n[i]*(ga[k]-n[k]*dni));
    const double tB=mu*(pkj*dnj+n[j]*(gb[k]-n[k]*dnj));
    dot+=tA*tB;
  }
  // alpha * (-<t(u),w_t>-<t(w),u_t>)
  // -beta*alpha <u_t,w_t> = a0 - beta, a0 = beta*betaN/den
  // -(1/den)<t(u),t(w)> is required for Robin consistency.
  return alpha*bazb_tangent_entry(i,j,n,pa,pb,ga,gb,mu)
       -beta*alpha*pa*pb*pij-dot/den;
}

__device__ __forceinline__ bool genrobin_params(
    const double*coeff,int fi,int q,double molecularNu,int viscosityMode,
    double stabilizationC,double h,double&mu,double&beta,double&betaN){
  const std::size_t qi=(std::size_t)fi*12u+(std::size_t)q;
  mu=viscosityMode==1?molecularNu:coeff[2u*qi];
  beta=coeff[2u*qi+1u];
  betaN=stabilizationC*mu/h;
  return mu>0.0 && beta>0.0 && betaN>0.0 &&
      isfinite(mu) && isfinite(beta) && isfinite(betaN) &&
      isfinite(beta+betaN);
}

// Assembled 3x3 M3-CSR increment in PHYSICAL FE basis; M3 then applies
// the same BF3 T^t A T transforms as in the original coupled solver.
__global__ void genrobin_m3_add(
    const hxt5b::TetFacePlan*F,int nf,
    const hxt5b::TetWallQuadCache*qc,
    const nodals_hxt1::TetGeom*tg,
    const std::int32_t*tetSlot,const double*coeff,
    double molecularNu,int viscosityMode,double stabilizationC,
    std::size_t nnz,Real*blocks,unsigned long long*bad){
  const int fi=(int)blockIdx.x,pair=(int)threadIdx.x;
  if(fi>=nf||pair>=64)return;
  const int a=pair/8,b=pair%8;
  const auto face=F[fi];const auto geom=tg[face.cell];
  const double h=3.0*geom.volume/face.area;
  const double n[3]={face.n[0],face.n[1],face.n[2]};
  const int pos=tetSlot[(std::size_t)face.cell*64u+(std::size_t)pair];
  if(pos<0 || (std::size_t)pos>=nnz || !(h>0.0) || !isfinite(h)){
    atomicAdd(bad,1ULL);return;
  }
  double K[3][3]{};
  for(int q=0;q<12;++q){
    const auto&ca=qc[(std::size_t)fi*12u+(std::size_t)q];
    double mu=0.,beta=0.,betaN=0.;
    if(!genrobin_params(coeff,fi,q,molecularNu,viscosityMode,
                        stabilizationC,h,mu,beta,betaN) ||
       !(ca.w>0.0) || !isfinite(ca.w)){
      atomicAdd(bad,1ULL);return;
    }
    double lam[4]={0.,0.,0.,0.};
    lam[hxt2::c_tet_face[(int)face.lf][0]]=hxt5b::QL0[q];
    lam[hxt2::c_tet_face[(int)face.lf][1]]=hxt5b::QL1[q];
    lam[hxt2::c_tet_face[(int)face.lf][2]]=hxt5b::QL2[q];
    double phi[8],gr[8][3],gp[8][3];
    hxt2::tet_basis(lam[0],lam[1],lam[2],lam[3],phi,gr);
    hxt2::tet_phys_grad(gr,geom,gp);
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)
      K[i][j]+=ca.w*genrobin_entry(
          i,j,n,ca.bw[a],ca.bw[b],gp[a],gp[b],mu,beta,betaN);
  }
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)
    atomicAdd(blocks+((std::size_t)i*3u+(std::size_t)j)*nnz+
                    (std::size_t)pos,(Real)K[i][j]);
}

// The TRUE matrix-free coupled FGMRES action gets precisely the same
// increment before T^t, using lagged F2 beta/nu_c from this SIMPLE step.
__global__ void genrobin_tangent_action(
    const hxt5b::TetFacePlan*F,int nf,
    const hxt5b::TetWallQuadCache*qc,
    const nodals_hxt1::TetVel*tv,
    const nodals_hxt1::TetGeom*tg,
    const double*coeff,double molecularNu,int viscosityMode,
    double stabilizationC,int nv,const Real*x,Real*y){
  const int fi=(int)blockIdx.x,a=(int)threadIdx.x;
  if(fi>=nf||a>=8)return;
  const auto face=F[fi];const auto geom=tg[face.cell];
  const auto cell=tv[face.cell];
  const double h=3.0*geom.volume/face.area;
  const double n[3]={face.n[0],face.n[1],face.n[2]};
  double Kx[3]={};
  for(int q=0;q<12;++q){
    const auto&ca=qc[(std::size_t)fi*12u+(std::size_t)q];
    double mu=0.,beta=0.,betaN=0.;
    // Validation also runs on the assembly path; never silently apply NaNs.
    if(!genrobin_params(coeff,fi,q,molecularNu,viscosityMode,
                        stabilizationC,h,mu,beta,betaN))return;
    double lam[4]={0.,0.,0.,0.};
    lam[hxt2::c_tet_face[(int)face.lf][0]]=hxt5b::QL0[q];
    lam[hxt2::c_tet_face[(int)face.lf][1]]=hxt5b::QL1[q];
    lam[hxt2::c_tet_face[(int)face.lf][2]]=hxt5b::QL2[q];
    double phi[8],gr[8][3],gp[8][3];
    hxt2::tet_basis(lam[0],lam[1],lam[2],lam[3],phi,gr);
    hxt2::tet_phys_grad(gr,geom,gp);
    for(int b=0;b<8;++b){
      const int ib=cell.g[b];
      for(int i=0;i<3;++i)for(int j=0;j<3;++j)
        Kx[i]+=ca.w*genrobin_entry(
            i,j,n,ca.bw[a],ca.bw[b],gp[a],gp[b],mu,beta,betaN)*
            (double)x[(std::size_t)j*(std::size_t)nv+(std::size_t)ib];
    }
  }
  const int ia=cell.g[a];
  for(int i=0;i<3;++i)
    atomicAdd(y+(std::size_t)i*(std::size_t)nv+(std::size_t)ia,(Real)Kx[i]);
}
