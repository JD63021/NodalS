// Stage33A: optional, pure-tetrahedron P1+BF3 momentum SUPG.
// Port of NodalS GPU H7 warp-per-cell SUPG64 kernel to Stage32's 8x8 CSR slots.
// This is an experimental *frozen-coefficient Oseen* strong-residual operator:
// R_i = a . grad(u_i) - nu_eff(centroid) * Laplacian(u_i),
// where a is the Picard-frozen velocity at the quadrature point.
// Since pressure is P0, its element-interior gradient is zero.  The strong
// derivative of spatially varying nut is NOT represented.  Therefore this is
// NOT an exact variable-viscosity RANS strong residual, and not yet a full
// residual-consistent RANS SUPG discretization.  Treat as Stage33A experiment.
// No additional persistent GPU buffers are allocated.

__device__ __constant__ double s33_rn[4] = {
  .0485005494469972764,.238600737551862341,.517047295104367421,.795851417896772828};
__device__ __constant__ double s33_rw[4] = {
  .110888415611277741,.143458789799214448,.0686338871729230970,.0103522407499180812};
__device__ __constant__ double s33_sn[4] = {
  .0571041961145177246,.276843013638123803,.583590432368916834,.860240135656219485};
__device__ __constant__ double s33_sw[4] = {
  .135506913431488518,.203464568010271102,.129847547608232333,.0311809709500080849};
__device__ __constant__ double s33_tn[4] = {
  .0694318442029737137,.330009478207571871,.669990521792428129,.930568155797026231};
__device__ __constant__ double s33_tw[4] = {
  .173927422568726897,.326072577431273103,.326072577431273103,.173927422568726897};

__device__ __forceinline__ double s33_grad_lam(const nodals_hxt1::TetGeom&g,int i,int d) {
  return i==0 ? -(g.invJ[d]+g.invJ[3+d]+g.invJ[6+d]) : g.invJ[3*(i-1)+d];
}

// 256 threads = eight warps; each warp assembles exactly one tetrahedron.
// One atomic per *local* 8x8 CSR entry instead of one per quadrature point.
__global__ void stage33a_momentum_supg64(
    const nodals_hxt1::Point* __restrict__ pts,
    const nodals_hxt1::TetConn* __restrict__ tc,
    const nodals_hxt1::TetVel* __restrict__ tv,
    const nodals_hxt1::TetGeom* __restrict__ tg,
    const std::int32_t* __restrict__ slot,
    const Real* __restrict__ u0,const Real* __restrict__ u1,const Real* __restrict__ u2,
    const Real* __restrict__ k,const Real* __restrict__ omega,
    Real* __restrict__ momAdd,Real* __restrict__ diagAdd,
    std::uint64_t nc,double radius,double nu,double kFloor,double omegaFloor,
    double tauScale,double magic,int streamlineOnly,unsigned long long* __restrict__ bad)
{
  constexpr int NW=8;
  const int lane=(int)threadIdx.x&31;
  const int w=(int)threadIdx.x>>5;
  const std::uint64_t start=(std::uint64_t)blockIdx.x*NW+(std::uint64_t)w;
  const std::uint64_t stride=(std::uint64_t)gridDim.x*NW;
  __shared__ double coeff[NW][8][5]; // Uxyz, k, omega
  __shared__ double stream[NW][8],strong[NW][8],weight[NW],nuEff[NW];
  for(std::uint64_t c=start;c<nc;c+=stride){
    const auto& V=tv[c];
    const auto& G=tg[c];
    if(lane<8){
      const int id=V.g[lane];
      coeff[w][lane][0]=(double)u0[id];
      coeff[w][lane][1]=(double)u1[id];
      coeff[w][lane][2]=(double)u2[id];
      coeff[w][lane][3]=(double)k[id];
      coeff[w][lane][4]=(double)omega[id];
    }
    __syncwarp();
    if(lane==0){
      // Freeze SST turbulent diffusivity at the tetra centroid (no new cache).
      const double l0=.25;
      double ph[8],gr[8][3],gp[8][3];
      hxt2::tet_basis(l0,l0,l0,l0,ph,gr);
      hxt2::tet_phys_grad(gr,G,gp);
      Real gu[9]={0},gk[3]={0},gw[3]={0},kv=0,wv=0;
      for(int j=0;j<8;++j){
        const Real pj=(Real)ph[j];
        kv+=(Real)(coeff[w][j][3])*pj;
        wv+=(Real)(coeff[w][j][4])*pj;
        for(int d=0;d<3;++d){
          const Real gj=(Real)gp[j][d];
          gu[d]+=(Real)coeff[w][j][0]*gj;
          gu[3+d]+=(Real)coeff[w][j][1]*gj;
          gu[6+d]+=(Real)coeff[w][j][2]*gj;
          gk[d]+=(Real)coeff[w][j][3]*gj;
          gw[d]+=(Real)coeff[w][j][4]*gj;
        }
      }
      double x=0,y=0;
      for(int i=0;i<4;++i){
        const auto p=pts[tc[c].v[i]];
        x+=.25*p.x;y+=.25*p.y;
      }
      auto ss=hxt5c::local_sst((Real)x,(Real)y,(Real)radius,(Real)nu,
          gu,kv,wv,gk,gw,(Real)kFloor,(Real)omegaFloor);
      const double ne=nu+(double)ss.nut;
      if(!(G.h2>0&&G.det>0&&ne>0&&isfinite(ne))){
        atomicAdd(bad,1ULL);
        nuEff[w]=nu;
      } else nuEff[w]=ne;
    }
    __syncwarp();
    double acc0=0,acc1=0;
    for(int iq=0;iq<64;++iq){
      const int ir=iq>>4,is=(iq>>2)&3,it=iq&3;
      const double r=s33_rn[ir],s=s33_sn[is],t=s33_tn[it];
      const double omr=1-r,oms=1-s;
      const double lam[4]={omr*oms*(1-t),r,omr*s,omr*oms*t};
      double phi=0;
      if(lane<4)phi=lam[lane];
      else if(lane<8){
        const int i=lane-4;int js[3],z=0;
        for(int j=0;j<4;++j)if(j!=i)js[z++]=j;
        phi=27*lam[js[0]]*lam[js[1]]*lam[js[2]];
      }
      double ax=lane<8?phi*coeff[w][lane][0]:0.0;
      double ay=lane<8?phi*coeff[w][lane][1]:0.0;
      double az=lane<8?phi*coeff[w][lane][2]:0.0;
      for(int off=16;off>0;off>>=1){
        ax+=__shfl_down_sync(0xffffffffu,ax,off);
        ay+=__shfl_down_sync(0xffffffffu,ay,off);
        az+=__shfl_down_sync(0xffffffffu,az,off);
      }
      ax=__shfl_sync(0xffffffffu,ax,0);
      ay=__shfl_sync(0xffffffffu,ay,0);
      az=__shfl_sync(0xffffffffu,az,0);
      if(lane<8){
        const int a=lane;
        double gx=0,gy=0,gz=0,lap=0;
        if(a<4){
          gx=s33_grad_lam(G,a,0);gy=s33_grad_lam(G,a,1);gz=s33_grad_lam(G,a,2);
        }else{
          const int i=a-4;int js[3],z=0;
          for(int j=0;j<4;++j)if(j!=i)js[z++]=j;
          const double g0[3]={s33_grad_lam(G,js[0],0),s33_grad_lam(G,js[0],1),s33_grad_lam(G,js[0],2)};
          const double g1[3]={s33_grad_lam(G,js[1],0),s33_grad_lam(G,js[1],1),s33_grad_lam(G,js[1],2)};
          const double g2[3]={s33_grad_lam(G,js[2],0),s33_grad_lam(G,js[2],1),s33_grad_lam(G,js[2],2)};
          gx=27*(lam[js[1]]*lam[js[2]]*g0[0]+lam[js[0]]*lam[js[2]]*g1[0]+lam[js[0]]*lam[js[1]]*g2[0]);
          gy=27*(lam[js[1]]*lam[js[2]]*g0[1]+lam[js[0]]*lam[js[2]]*g1[1]+lam[js[0]]*lam[js[1]]*g2[1]);
          gz=27*(lam[js[1]]*lam[js[2]]*g0[2]+lam[js[0]]*lam[js[2]]*g1[2]+lam[js[0]]*lam[js[1]]*g2[2]);
          double d01=0,d02=0,d12=0;
          for(int d=0;d<3;++d){
            d01+=g0[d]*g1[d];d02+=g0[d]*g2[d];d12+=g1[d]*g2[d];
          }
          lap=54*(lam[js[2]]*d01+lam[js[1]]*d02+lam[js[0]]*d12);
        }
        const double st=ax*gx+ay*gy+az*gz;
        stream[w][a]=st;
        strong[w][a]=streamlineOnly ? st : st-nuEff[w]*lap; // NODALS_SKEW_SD_DIAGNOSTIC_V1
      }
      if(lane==0){
        const double speed2=ax*ax+ay*ay+az*az;
        const double diff=4*nuEff[w]/G.h2;
        const double den=fmax(4*speed2/G.h2+magic*diff*diff,1e-30);
        const double tau=tauScale/sqrt(den);
        weight[w]=tau*G.det*s33_rw[ir]*s33_sw[is]*s33_tw[it];
      }
      __syncwarp();
      const int p0=lane,p1=lane+32;
      acc0+=weight[w]*stream[w][p0/8]*strong[w][p0%8];
      acc1+=weight[w]*stream[w][p1/8]*strong[w][p1%8];
      __syncwarp();
    }
    const int p0=lane,p1=lane+32;
    const int a0=p0/8,b0=p0%8,a1=p1/8,b1=p1%8;
    atomicAdd(momAdd+slot[c*64ull+p0],(Real)acc0);
    atomicAdd(momAdd+slot[c*64ull+p1],(Real)acc1);
    if(a0==b0)atomicAdd(diagAdd+V.g[a0],(Real)acc0);
    if(a1==b1)atomicAdd(diagAdd+V.g[a1],(Real)acc1);
    __syncwarp();
  }
}
