__global__ void sst_tet_volume(
    const nodals_hxt1::Point*pts,const nodals_hxt1::TetConn*tc,const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,
    const std::int32_t*slot,const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,
    Real*mat,Real*rhs,Real*productionAudit,Real*crossAudit,Real*reactionAudit,std::uint64_t n,double radius,double nu,int equation,double kFloor,double omegaFloor)
{
  std::uint64_t c=(std::uint64_t)blockIdx.x;if(c>=n)return;
  __shared__ Real U0[8],U1[8],U2[8],K[8],W[8];
  // PERF_V4_TET24_SCALAR_VOLUME
  __shared__ Real PH[24][8],GP[24][8][3],DW[24],RW[24],SW[24],SP[24],SC[24];
  if(threadIdx.x<8){int g=tv[c].g[threadIdx.x];U0[threadIdx.x]=u0[g];U1[threadIdx.x]=u1[g];U2[threadIdx.x]=u2[g];K[threadIdx.x]=k[g];W[threadIdx.x]=omega[g];}
  __syncthreads();
  int iq=(int)threadIdx.x;
  if(iq<24){
    double l[4]={
      hxt5a::h5_tet24_l0[iq],hxt5a::h5_tet24_l1[iq],
      hxt5a::h5_tet24_l2[iq],hxt5a::h5_tet24_l3[iq]};
    double qw=hxt5a::h5_tet24_w[iq];
    double ph[8],gr[8][3],gp[8][3];hxt2::tet_basis(l[0],l[1],l[2],l[3],ph,gr);hxt2::tet_phys_grad(gr,tg[c],gp);
    Real gu[9]={0,0,0,0,0,0,0,0,0},gk[3]={0,0,0},gw[3]={0,0,0},kv=0,wv=0;
    for(int a=0;a<8;++a){Real pa=(Real)ph[a];PH[iq][a]=pa;kv+=K[a]*pa;wv+=W[a]*pa;for(int d=0;d<3;++d){Real gd=(Real)gp[a][d];GP[iq][a][d]=gd;gk[d]+=K[a]*gd;gw[d]+=W[a]*gd;gu[d]+=U0[a]*gd;gu[3+d]+=U1[a]*gd;gu[6+d]+=U2[a]*gd;}}
    double xq=0,yq=0;for(int a=0;a<4;++a){auto X=pts[tc[c].v[a]];xq+=l[a]*X.x;yq+=l[a]*X.y;}
    LocalSST ss=local_sst((Real)xq,(Real)yq,(Real)radius,(Real)nu,gu,kv,wv,gk,gw,(Real)kFloor,(Real)omegaFloor,stage20_distance(tc,(int)c,l));Real wt=(Real)(tg[c].det*qw);
    const auto tcfs=nodals::transport::volume_coefficients<Real>(equation,K_EQ,wt,ss);DW[iq]=tcfs.diffusion;RW[iq]=tcfs.reaction;SW[iq]=tcfs.source;SP[iq]=tcfs.omegaProductionAudit;SC[iq]=tcfs.omegaCrossAudit;
  }
  __syncthreads();
  for(int p=(int)threadIdx.x;p<64;p+=(int)blockDim.x){int a=p/8,b=p%8;Real z=0;for(int q=0;q<24;++q){Real gd=GP[q][a][0]*GP[q][b][0]+GP[q][a][1]*GP[q][b][1]+GP[q][a][2]*GP[q][b][2];z+=DW[q]*gd+RW[q]*PH[q][a]*PH[q][b];}atomicAdd(mat+slot[c*64ull+(std::size_t)p],z);if(reactionAudit){Real zr=0;for(int q=0;q<24;++q)zr+=RW[q]*PH[q][a]*PH[q][b];atomicAdd(reactionAudit+slot[c*64ull+(std::size_t)p],zr);}}
  for(int a=(int)threadIdx.x;a<8;a+=(int)blockDim.x){Real z=0,zp=0,zc=0;for(int q=0;q<24;++q){z+=SW[q]*PH[q][a];if(productionAudit){zp+=SP[q]*PH[q][a];zc+=SC[q]*PH[q][a];}}atomicAdd(rhs+tv[c].g[a],z);if(productionAudit){atomicAdd(productionAudit+tv[c].g[a],zp);atomicAdd(crossAudit+tv[c].g[a],zc);}}
}


