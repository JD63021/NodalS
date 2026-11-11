__global__ void sst_inlet_tet(
    const hxt5b::TetFacePlan*F,int nf,const nodals_hxt1::Point*pts,const nodals_hxt1::TetConn*tc,const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,const std::int32_t*slot,
    const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,Real*mat,Real*rhs,double qIn,double inletX,double inletY,double inletZ,double radius,double nu,int equation,double kFloor,double omegaFloor)
{
  int fi=(int)blockIdx.x;if(fi>=nf)return;auto f=F[fi];auto T=tv[f.cell];
  __shared__ Real U0[8],U1[8],U2[8],K[8],W[8],PH[12][8],GP[12][8][3],NUE[12],WQ[12],QIN[12];if(threadIdx.x<8){int g=T.g[threadIdx.x];U0[threadIdx.x]=u0[g];U1[threadIdx.x]=u1[g];U2[threadIdx.x]=u2[g];K[threadIdx.x]=k[g];W[threadIdx.x]=omega[g];}__syncthreads();
  if(threadIdx.x<12){int q=(int)threadIdx.x,a0=hxt4a::h4_tet_face[f.lf][0],a1=hxt4a::h4_tet_face[f.lf][1],a2=hxt4a::h4_tet_face[f.lf][2];double l[4]={0,0,0,0};l[a0]=hxt5b::QL0[q];l[a1]=hxt5b::QL1[q];l[a2]=hxt5b::QL2[q];double ph[8],gr[8][3],gp[8][3];hxt2::tet_basis(l[0],l[1],l[2],l[3],ph,gr);hxt2::tet_phys_grad(gr,tg[f.cell],gp);
    Real gu[9]={0,0,0,0,0,0,0,0,0},gk[3]={0,0,0},gw[3]={0,0,0},kv=0,wv=0;for(int a=0;a<8;++a){Real pa=(Real)ph[a];PH[q][a]=pa;kv+=K[a]*pa;wv+=W[a]*pa;for(int d=0;d<3;++d){Real gd=(Real)gp[a][d];GP[q][a][d]=gd;gk[d]+=K[a]*gd;gw[d]+=W[a]*gd;gu[d]+=U0[a]*gd;gu[3+d]+=U1[a]*gd;gu[6+d]+=U2[a]*gd;}}
    hxt5c::eval_dg_omega_p1(f.cell,l,tg[f.cell],wv,gw);
    double x=0,y=0;for(int a=0;a<4;++a){auto X=pts[tc[f.cell].v[a]];x+=l[a]*X.x;y+=l[a]*X.y;}LocalSST ss=local_sst((Real)x,(Real)y,(Real)radius,(Real)nu,gu,kv,wv,gk,gw,(Real)kFloor,(Real)omegaFloor,stage20_distance(tc,f.cell,l));Real extra=(equation==K_EQ?ss.sigmaK:ss.sigmaW)*ss.nut;NUE[q]=(Real)nu+extra;WQ[q]=(Real)(hxt5b::QW12[q]*f.area);
    Real vx=(Real)inletX,vy=(Real)inletY,vz=(Real)inletZ;
    if(stage27_inlet::profileActive){
      auto conn=tc[f.cell];vx=vy=vz=Real(0);
      for(int a=0;a<4;++a){std::size_t id=(std::size_t)conn.v[a];
        vx+=(Real)l[a]*stage27_inlet::velocityX[id];
        vy+=(Real)l[a]*stage27_inlet::velocityY[id];
        vz+=(Real)l[a]*stage27_inlet::velocityZ[id];}
    }
    QIN[q]=fmax(Real(0),Real(-(vx*f.n[0]+vy*f.n[1]+vz*f.n[2])));
  }__syncthreads();Real inflow=fmax(Real(0),Real(-(inletX*f.n[0]+inletY*f.n[1]+inletZ*f.n[2])));
  for(int p=(int)threadIdx.x;p<64;p+=(int)blockDim.x){int a=p/8,b=p%8;Real z=0;for(int q=0;q<12;++q){Real dna=GP[q][a][0]*(Real)f.n[0]+GP[q][a][1]*(Real)f.n[1]+GP[q][a][2]*(Real)f.n[2],dnb=GP[q][b][0]*(Real)f.n[0]+GP[q][b][1]*(Real)f.n[1]+GP[q][b][2]*(Real)f.n[2];z+=WQ[q]*((stage27_inlet::profileActive?QIN[q]:inflow)*PH[q][a]*PH[q][b]+NUE[q]*(-PH[q][a]*dnb+dna*PH[q][b]));}atomicAdd(mat+slot[(std::size_t)f.cell*64ull+(std::size_t)p],z);}
  for(int a=(int)threadIdx.x;a<8;a+=(int)blockDim.x){Real z=0;for(int q=0;q<12;++q){Real dna=GP[q][a][0]*(Real)f.n[0]+GP[q][a][1]*(Real)f.n[1]+GP[q][a][2]*(Real)f.n[2];z+=WQ[q]*((stage27_inlet::profileActive?QIN[q]:inflow)*PH[q][a]*(Real)qIn+NUE[q]*dna*(Real)qIn);}atomicAdd(rhs+T.g[a],z);}
}

