// Stage33B/C: independent, optional residual-form SUPG for k / omega.
// Pure affine tetrahedra, P1+BF3 scalar trial/test, warp-per-tet GPU assembly.
// Zero new persistent GPU storage: contributes to existing scalar vol CSR + rhs.
// Quadrature: same 24-point positive tet rule as the stock SST scalar volume.
// Frozen/Picard transport residual used for each equation:
//   R_q = U . grad(q) - (nu + sigma_q*nut)*Laplacian(q)
//         + reaction_q*q - source_q.
// Physical variable diffusivity derivative grad(sigma*nut).grad(q) is omitted;
// wall boundary source/penalty is not included in a volume strong residual.
// Hence this is a *frozen-coefficient* residual approximation, not an exact
// nonlinear variable-diffusion SST residual. Do not claim full consistency.
// tau=scale/sqrt(4|U|^2/h2+magic*(4*nu_eff/h2)^2).
// Kernels integrate tau (U.grad test)(U.grad trial - nu_eff lap trial
//            + react*trial) to CSR, tau (U.grad test)*source to RHS.

__global__ void stage33bc_scalar_supg24(
    const nodals_hxt1::Point* __restrict__ pts,
    const nodals_hxt1::TetConn* __restrict__ tc,
    const nodals_hxt1::TetVel* __restrict__ tv,
    const nodals_hxt1::TetGeom* __restrict__ tg,
    const std::int32_t* __restrict__ slot,
    const Real* __restrict__ u0,const Real* __restrict__ u1,const Real* __restrict__ u2,
    const Real* __restrict__ k,const Real* __restrict__ omega,
    Real* __restrict__ volumeMatrix,Real* __restrict__ rhs,
    std::uint64_t nc,double radius,double nu,int equation,
    double kFloor,double omegaFloor,double tauScale,double magic,
    unsigned long long* __restrict__ bad)
{
  constexpr int NW=8;
  const int lane=(int)threadIdx.x&31;
  const int wi=(int)threadIdx.x>>5;
  const std::uint64_t stride=(std::uint64_t)gridDim.x*NW;
  __shared__ double coeff[NW][8][5];
  __shared__ double adv[NW][3],eff[NW],react[NW],src[NW],weight[NW];
  __shared__ double stream[NW][8],strong[NW][8];
  for(std::uint64_t c=(std::uint64_t)blockIdx.x*NW+(std::uint64_t)wi;c<nc;c+=stride){
    const auto& G=tg[c];
    const auto& V=tv[c];
    if(lane<8){
      const int id=V.g[lane];
      coeff[wi][lane][0]=(double)u0[id];
      coeff[wi][lane][1]=(double)u1[id];
      coeff[wi][lane][2]=(double)u2[id];
      coeff[wi][lane][3]=(double)k[id];
      coeff[wi][lane][4]=(double)omega[id];
    }
    __syncwarp();
    if(!(G.h2>0.0 && G.det>0.0 && isfinite(G.h2) && isfinite(G.det))){
      if(lane==0)atomicAdd(bad,1ULL);
      __syncwarp();
      continue;
    }
    double aa=0.0,bb=0.0,ar=0.0;
    for(int iq=0;iq<24;++iq){
      const double lam[4]={hxt5a::h5_tet24_l0[iq],hxt5a::h5_tet24_l1[iq],
                          hxt5a::h5_tet24_l2[iq],hxt5a::h5_tet24_l3[iq]};
      if(lane==0){
        double ph[8],gr[8][3],gp[8][3];
        hxt2::tet_basis(lam[0],lam[1],lam[2],lam[3],ph,gr);
        hxt2::tet_phys_grad(gr,G,gp);
        Real gu[9]={0,0,0,0,0,0,0,0,0};
        Real gk[3]={0,0,0},gw[3]={0,0,0},kv=0,wv=0;
        double a[3]={0,0,0};
        for(int j=0;j<8;++j){
          const Real p=(Real)ph[j];
          kv+=(Real)coeff[wi][j][3]*p;
          wv+=(Real)coeff[wi][j][4]*p;
          for(int d=0;d<3;++d){
            const Real gd=(Real)gp[j][d];
            a[d]+=ph[j]*coeff[wi][j][d];
            gu[d]+=(Real)coeff[wi][j][0]*gd;
            gu[3+d]+=(Real)coeff[wi][j][1]*gd;
            gu[6+d]+=(Real)coeff[wi][j][2]*gd;
            gk[d]+=(Real)coeff[wi][j][3]*gd;
            gw[d]+=(Real)coeff[wi][j][4]*gd;
          }
        }
        double x=0,y=0;
        for(int j=0;j<4;++j){
          const auto P=pts[tc[c].v[j]];
          x+=lam[j]*P.x;y+=lam[j]*P.y;
        }
        const auto ss=local_sst((Real)x,(Real)y,(Real)radius,(Real)nu,
             gu,kv,wv,gk,gw,(Real)kFloor,(Real)omegaFloor,
             stage20_distance(tc,(int)c,lam));
        const double nuEff=nu+(equation==K_EQ?(double)ss.sigmaK:(double)ss.sigmaW)*(double)ss.nut;
        const double rr=equation==K_EQ?(double)ss.reactK:(double)ss.reactOmega;
        const double ssr=equation==K_EQ?(double)ss.pk:(double)ss.srcOmega;
        adv[wi][0]=a[0];adv[wi][1]=a[1];adv[wi][2]=a[2];
        eff[wi]=nuEff;react[wi]=rr;src[wi]=ssr;
        const double speed2=a[0]*a[0]+a[1]*a[1]+a[2]*a[2];
        const double diff=4.0*nuEff/G.h2;
        const double den=fmax(4.0*speed2/G.h2+magic*diff*diff,1e-30);
        const double tau=tauScale/sqrt(den);
        if(!(isfinite(nuEff)&&nuEff>0.0&&isfinite(rr)&&isfinite(ssr)&&isfinite(tau)&&tau>=0.0)){
          atomicAdd(bad,1ULL);weight[wi]=0.0;
        }else{
          weight[wi]=tau*G.det*hxt5a::h5_tet24_w[iq];
        }
      }
      __syncwarp();
      if(lane<8){
        double ph[8],gr[8][3],gp[8][3];
        hxt2::tet_basis(lam[0],lam[1],lam[2],lam[3],ph,gr);
        hxt2::tet_phys_grad(gr,G,gp);
        const int a=lane;
        const double st=adv[wi][0]*gp[a][0]+adv[wi][1]*gp[a][1]+adv[wi][2]*gp[a][2];
        // Affine P1 Laplacian = 0. BF3 Laplacian is nonzero.
        double lap=0.0;
        if(a>=4){
          const int face=a-4;
          int js[3],z=0;
          for(int j=0;j<4;++j)if(j!=face)js[z++]=j;
          double dot01=0,dot02=0,dot12=0;
          for(int d=0;d<3;++d){
            const double g0=js[0]==0?-(G.invJ[d]+G.invJ[3+d]+G.invJ[6+d]):G.invJ[3*(js[0]-1)+d];
            const double g1=js[1]==0?-(G.invJ[d]+G.invJ[3+d]+G.invJ[6+d]):G.invJ[3*(js[1]-1)+d];
            const double g2=js[2]==0?-(G.invJ[d]+G.invJ[3+d]+G.invJ[6+d]):G.invJ[3*(js[2]-1)+d];
            dot01+=g0*g1;dot02+=g0*g2;dot12+=g1*g2;
          }
          lap=54.0*(lam[js[2]]*dot01+lam[js[1]]*dot02+lam[js[0]]*dot12);
        }
        stream[wi][a]=st;
        strong[wi][a]=st-eff[wi]*lap+react[wi]*ph[a];
      }
      __syncwarp();
      const int p0=lane,p1=lane+32;
      aa+=weight[wi]*stream[wi][p0/8]*strong[wi][p0%8];
      bb+=weight[wi]*stream[wi][p1/8]*strong[wi][p1%8];
      if(lane<8)ar+=weight[wi]*stream[wi][lane]*src[wi];
      __syncwarp();
    }
    const int p0=lane,p1=lane+32;
    atomicAdd(volumeMatrix+slot[c*64ull+(std::size_t)p0],(Real)aa);
    atomicAdd(volumeMatrix+slot[c*64ull+(std::size_t)p1],(Real)bb);
    if(lane<8)atomicAdd(rhs+V.g[lane],(Real)ar);
    __syncwarp();
  }
}
