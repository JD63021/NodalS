// Stage 2 P2 scalar *VOLUME* transport kernel. Intended for inclusion INSIDE
// namespace hxt5c, after LocalSST, local_sst, stage20_distance, and the
// existing tetrahedral 24-point quadrature and velocity basis definitions.
// This kernel is real 10x10 P2 Galerkin transport: NO face-BF3 scalar slots.
// Velocity remains unchanged: 8-mode P1+BF3 interpolation.
// IMPORTANT: NOT YET CONNECTED to the G2 application / wall / restart.
//
// For each q: physical k,w and their derivatives use *P2* 10 controls.
// Scalar diffusion, reaction, nonlinear production and advective test/trial
// functions are likewise all P2. Never use this kernel with 8x8 CSR slots.
// The required P2 DOF indices and 100 matrix slots are distinct arrays.

__global__ void sst_p2_tet_volume24(
   const nodals_hxt1::Point*pts,const nodals_hxt1::TetConn*tc,
   const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,
   const nodals::p2::Tet10Device*p2cells,const std::int32_t*slot100,
   const Real*u0,const Real*u1,const Real*u2,
   const Real*kP2,const Real*omegaP2,Real*mat,Real*rhs,
   std::uint64_t n,double radius,double nu,int equation,
   double kFloor,double omegaFloor)
{
  const std::uint64_t c=blockIdx.x;if(c>=n)return;
  __shared__ Real U[8][3],K[10],W[10];
  __shared__ Real PH[24][10],GP[24][10][3],VD[24][3];
  __shared__ Real DIF[24],REA[24],SRC[24],WT[24];
  if(threadIdx.x<10){const int a=threadIdx.x;
    const int id=p2cells[c].g[a];K[a]=kP2[id];W[a]=omegaP2[id];
    if(a<8){const int ui=tv[c].g[a];U[a][0]=u0[ui];U[a][1]=u1[ui];U[a][2]=u2[ui];}
  }
  __syncthreads();
  const int q=threadIdx.x;
  if(q<24){
    const double l[4]={hxt5a::h5_tet24_l0[q],hxt5a::h5_tet24_l1[q],
       hxt5a::h5_tet24_l2[q],hxt5a::h5_tet24_l3[q]};
    double ph[10],gr[10][3],gp[10][3];
    nodals::p2::basis(l,ph,gr);
    nodals::p2::physical_gradient(gr,tg[c].invJ,gp);
    double velPh[8],velGr[8][3],velGp[8][3];
    hxt2::tet_basis(l[0],l[1],l[2],l[3],velPh,velGr);
    hxt2::tet_phys_grad(velGr,tg[c],velGp);
    Real gu[9]={},gk[3]={},gw[3]={},kv=Real(0),wv=Real(0),uv[3]={};
    for(int a=0;a<10;++a){
      PH[q][a]=(Real)ph[a];kv+=K[a]*(Real)ph[a];wv+=W[a]*(Real)ph[a];
      for(int d=0;d<3;++d){GP[q][a][d]=(Real)gp[a][d];
        gk[d]+=K[a]*(Real)gp[a][d];gw[d]+=W[a]*(Real)gp[a][d];}
    }
    for(int a=0;a<8;++a){
      for(int d=0;d<3;++d){uv[d]+=U[a][d]*(Real)velPh[a];
        for(int j=0;j<3;++j)gu[3*d+j]+=U[a][d]*(Real)velGp[a][j];}
    }
    Real x=0,y=0;for(int a=0;a<4;++a){const auto X=pts[tc[c].v[a]];
      x+=(Real)(l[a]*X.x);y+=(Real)(l[a]*X.y);}
    auto ss=local_sst(x,y,(Real)radius,(Real)nu,gu,kv,wv,gk,gw,
      (Real)kFloor,(Real)omegaFloor,stage20_distance(tc,(int)c,l));
    const Real weightedVolume=(Real)(tg[c].det*hxt5a::h5_tet24_w[q]);
    const auto coeff=nodals::transport::volume_coefficients<Real>(
      equation,K_EQ,weightedVolume,ss);
    DIF[q]=coeff.diffusion+weightedVolume*(Real)nu;REA[q]=coeff.reaction;SRC[q]=coeff.source;
    WT[q]=weightedVolume;
    for(int d=0;d<3;++d)VD[q][d]=uv[d];
  }
  __syncthreads();
  for(int p=(int)threadIdx.x;p<100;p+=(int)blockDim.x){
    const int a=p/10,b=p%10;Real value=0;
    for(int q=0;q<24;++q){
      Real gradDot=0,adv=0;
      for(int d=0;d<3;++d){
        gradDot+=GP[q][a][d]*GP[q][b][d];
        adv+=VD[q][d]*GP[q][b][d];
      }
      value+=DIF[q]*gradDot + REA[q]*PH[q][a]*PH[q][b]
           + WT[q]*PH[q][a]*adv;
    }
    atomicAdd(mat+slot100[c*100ull+(std::size_t)p],value);
  }
  for(int a=(int)threadIdx.x;a<10;a+=(int)blockDim.x){
    Real value=0;for(int q=0;q<24;++q)value+=SRC[q]*PH[q][a];
    atomicAdd(rhs+p2cells[c].g[a],value);
  }
}
