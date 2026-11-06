// True P2 SST constitutive feedback to ORIGINAL P1+BF3 momentum matrix.
// Separate 10-control k/w scalar interpolation, 8-mode velocity test/trial.
// Uses exact P2 gradients in the local_sst model, no BF3 approximation of k/w.
// Intended for inclusion inside namespace hxt5d after sst_mom_tet.
// This kernel alone does not constitute full coupled P2 solver wiring.
__global__ void sst_mom_p2_tet(
 const nodals_hxt1::Point*pts,const nodals_hxt1::TetConn*tc,
 const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,
 const std::int32_t*slot64,const nodals::p2::Tet10Device*p2cells,
 const Real*u0,const Real*u1,const Real*u2,
 const Real*kP2,const Real*omegaP2,
 Real*turb,Real*diagRTurb,std::uint64_t n,double radius,double nu,
 double kFloor,double omegaFloor)
{
  std::uint64_t c=blockIdx.x;if(c>=n)return;
  __shared__ Real U[8][3],K[10],W[10];
  __shared__ Real GUV[24][8][3],NW[24];
  if(threadIdx.x<10){const int a=threadIdx.x;
    K[a]=kP2[p2cells[c].g[a]];W[a]=omegaP2[p2cells[c].g[a]];
    if(a<8){const int g=tv[c].g[a];U[a][0]=u0[g];U[a][1]=u1[g];U[a][2]=u2[g];}
  }
  __syncthreads();
  const int q=threadIdx.x;
  if(q<24){
    const double l[4]={hxt5a::h5_tet24_l0[q],hxt5a::h5_tet24_l1[q],
      hxt5a::h5_tet24_l2[q],hxt5a::h5_tet24_l3[q]};
    const double w=hxt5a::h5_tet24_w[q];
    double pv[8],gv[8][3],gphysV[8][3];
    hxt2::tet_basis(l[0],l[1],l[2],l[3],pv,gv);
    hxt2::tet_phys_grad(gv,tg[c],gphysV);
    double ps[10],gs[10][3],gphysS[10][3];
    nodals::p2::basis(l,ps,gs);
    nodals::p2::physical_gradient(gs,tg[c].invJ,gphysS);
    Real gu[9]={},gk[3]={},gw[3]={},kv=0,wv=0;
    for(int a=0;a<8;++a)for(int d=0;d<3;++d){
      GUV[q][a][d]=(Real)gphysV[a][d];
      for(int comp=0;comp<3;++comp)
        gu[3*comp+d]+=U[a][comp]*(Real)gphysV[a][d];
    }
    for(int a=0;a<10;++a){
      kv+=K[a]*(Real)ps[a];wv+=W[a]*(Real)ps[a];
      for(int d=0;d<3;++d){gk[d]+=K[a]*(Real)gphysS[a][d];gw[d]+=W[a]*(Real)gphysS[a][d];}
    }
    double x=0,y=0;for(int a=0;a<4;++a){auto X=pts[tc[c].v[a]];
      x+=l[a]*X.x;y+=l[a]*X.y;}
    auto ss=hxt5c::local_sst((Real)x,(Real)y,(Real)radius,(Real)nu,
       gu,kv,wv,gk,gw,(Real)kFloor,(Real)omegaFloor,
       hxt5c::stage20_distance(tc,(int)c,l));
    NW[q]=ss.nut*(Real)(tg[c].det*w);
  }
  __syncthreads();
  for(int p=(int)threadIdx.x;p<64;p+=(int)blockDim.x){
    int a=p/8,b=p%8;Real v=0;
    for(int q=0;q<24;++q){Real dot=0;
      for(int d=0;d<3;++d)dot+=GUV[q][a][d]*GUV[q][b][d];
      v+=NW[q]*dot;
    }
    atomicAdd(turb+slot64[c*64ull+(std::size_t)p],v);
    if(a==b)atomicAdd(diagRTurb+tv[c].g[a],v);
  }
}

#if defined(NODALS_P2_STAGE4_COUPLED)
__global__ void p2_stage4_diff(std::size_t n,const Real*trueP2,const Real*projectedP1,double*sum){
  const std::size_t i=(std::size_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=n)return;
  const double a=(double)trueP2[i],b=(double)projectedP1[i];
  if(!isfinite(a)||!isfinite(b)){atomicAdd(sum+2,1.);return;}
  atomicAdd(sum+0,fabs(a-b));atomicAdd(sum+1,fabs(a));
}
#endif
