// True Bernstein P2 tetra INLET: 10x10 matrix/10 RHS entries, using exact
// physical P2 k,omega values/gradients and unchanged enriched velocity.
// Intended for inclusion inside namespace hxt5c; not yet G2-wired.
__global__ void sst_p2_inlet_tet12(
    const hxt5b::TetFacePlan*F,int nf,
    const nodals_hxt1::Point*pts,const nodals_hxt1::TetConn*tc,
    const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,
    const nodals::p2::Tet10Device*p2cells,const std::int32_t*slot100,
    const Real*u0,const Real*u1,const Real*u2,
    const Real*kP2,const Real*omegaP2,Real*mat,Real*rhs,
    double inletValue,double inletX,double inletY,double inletZ,
    double radius,double nu,int equation,double kFloor,double omegaFloor)
{
  const int fi=blockIdx.x;if(fi>=nf)return;
  const auto f=F[fi];const int c=f.cell;
  __shared__ Real U[8][3],K[10],W[10];
  __shared__ Real PH[12][10],DN[12][10],NU[12],FLUX[12],WT[12];
  if(threadIdx.x<10){int a=threadIdx.x;const int id=p2cells[c].g[a];
    K[a]=kP2[id];W[a]=omegaP2[id];
    if(a<8){const int idU=tv[c].g[a];U[a][0]=u0[idU];U[a][1]=u1[idU];U[a][2]=u2[idU];}}
  __syncthreads();
  const int q=threadIdx.x;
  if(q<12){
    const int a0=hxt4a::h4_tet_face[f.lf][0],a1=hxt4a::h4_tet_face[f.lf][1],
              a2=hxt4a::h4_tet_face[f.lf][2];
    double l[4]={0,0,0,0};l[a0]=hxt5b::QL0[q];l[a1]=hxt5b::QL1[q];l[a2]=hxt5b::QL2[q];
    double ph[10],gr[10][3],gp[10][3];nodals::p2::basis(l,ph,gr);
    nodals::p2::physical_gradient(gr,tg[c].invJ,gp);
    double velPh[8],velGr[8][3],velGp[8][3];
    hxt2::tet_basis(l[0],l[1],l[2],l[3],velPh,velGr);
    hxt2::tet_phys_grad(velGr,tg[c],velGp);
    Real gu[9]={},gk[3]={},gw[3]={},kv=0,wv=0;
    for(int a=0;a<10;++a){kv+=K[a]*(Real)ph[a];wv+=W[a]*(Real)ph[a];
      PH[q][a]=(Real)ph[a];DN[q][a]=(Real)(gp[a][0]*f.n[0]+gp[a][1]*f.n[1]+gp[a][2]*f.n[2]);
      for(int d=0;d<3;++d){gk[d]+=K[a]*(Real)gp[a][d];gw[d]+=W[a]*(Real)gp[a][d];}}
    for(int a=0;a<8;++a)for(int d=0;d<3;++d)for(int j=0;j<3;++j)
      gu[3*d+j]+=U[a][d]*(Real)velGp[a][j];
    Real x=0,y=0;for(int a=0;a<4;++a){auto p=pts[tc[c].v[a]];
      x+=(Real)(l[a]*p.x);y+=(Real)(l[a]*p.y);}
    const auto ss=local_sst(x,y,(Real)radius,(Real)nu,gu,kv,wv,gk,gw,
      (Real)kFloor,(Real)omegaFloor,stage20_distance(tc,c,l));
    NU[q]=(Real)nu+(equation==K_EQ?ss.sigmaK:ss.sigmaW)*ss.nut;
    WT[q]=(Real)(hxt5b::QW12[q]*f.area);
    Real vx=(Real)inletX,vy=(Real)inletY,vz=(Real)inletZ;
    if(stage27_inlet::profileActive){vx=vy=vz=Real(0);
      for(int a=0;a<4;++a){auto idx=tc[c].v[a];
        vx+=(Real)l[a]*stage27_inlet::velocityX[idx];
        vy+=(Real)l[a]*stage27_inlet::velocityY[idx];
        vz+=(Real)l[a]*stage27_inlet::velocityZ[idx];}}
    FLUX[q]=fmax(Real(0),-(vx*(Real)f.n[0]+vy*(Real)f.n[1]+vz*(Real)f.n[2]));
  }
  __syncthreads();
  for(int p=(int)threadIdx.x;p<100;p+=(int)blockDim.x){
    int a=p/10,b=p%10;Real v=0;
    for(int q=0;q<12;++q)
      v+=WT[q]*(FLUX[q]*PH[q][a]*PH[q][b]
        +NU[q]*(-PH[q][a]*DN[q][b]+DN[q][a]*PH[q][b]));
    atomicAdd(mat+slot100[(std::size_t)c*100u+p],v);
  }
  for(int a=(int)threadIdx.x;a<10;a+=(int)blockDim.x){Real v=0;
    for(int q=0;q<12;++q)
      v+=WT[q]*(FLUX[q]*PH[q][a]*(Real)inletValue+NU[q]*DN[q][a]*(Real)inletValue);
    atomicAdd(rhs+p2cells[c].g[a],v);
  }
}
