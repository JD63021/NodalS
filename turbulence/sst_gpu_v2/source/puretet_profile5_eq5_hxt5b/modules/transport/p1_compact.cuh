// Compact tetra P1 scalar operators. The velocity remains 8-mode P1+BF3.
// For affine tets, molecular gradNi.gradNj is constant and exact without
// numerical quadrature; the old exact convection tensor is reused for the
// cubic BF3 velocity instead of underintegrating it with four-point quadrature.
__global__ void p1_compact_molecular_base(
 const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,
 const std::int32_t*slot,Real*base,std::uint64_t n,double nu){
 const std::uint64_t c=blockIdx.x;if(c>=n || threadIdx.x>=16)return;
 int a=threadIdx.x/4,b=threadIdx.x%4;
 const double G[4][3]={{-1,-1,-1},{1,0,0},{0,1,0},{0,0,1}};
 double ga[3]={0,0,0},gb[3]={0,0,0};
 for(int d=0;d<3;++d)for(int j=0;j<3;++j){
   ga[d]+=G[a][j]*tg[c].invJ[3*j+d];
   gb[d]+=G[b][j]*tg[c].invJ[3*j+d];
 }
 double dot=ga[0]*gb[0]+ga[1]*gb[1]+ga[2]*gb[2];
 atomicAdd(base+slot[c*64ull+(size_t)a*8+b],(Real)(nu*tg[c].det*dot/6.0));
}

__global__ void p1_compact_convection_exact(
 const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,
 const std::int32_t*slot,const Real*u0,const Real*u1,const Real*u2,
 Real*conv,std::uint64_t n){
 const std::uint64_t c=blockIdx.x;if(c>=n || threadIdx.x>=16)return;
 int a=threadIdx.x/4,b=threadIdx.x%4;
 const auto&g=tg[c];double z=0;
 for(int m=0;m<8;++m){int id=tv[c].g[m];
   for(int j=0;j<3;++j){
     double uj=(double)u0[id]*g.invJ[3*j]+(double)u1[id]*g.invJ[3*j+1]
              +(double)u2[id]*g.invJ[3*j+2];
     z+=uj*hxt4b::h4b_tet_convT[(((a*8+m)*8+b)*3+j)];
   }
 }
 atomicAdd(conv+slot[c*64ull+(size_t)a*8+b],(Real)(z*g.det));
}

// Four-point degree-two exact rule for affine P1 mass with constant factors.
// SST coefficients are NONPOLYNOMIAL; the 4-point rule is an explicit accuracy
// experiment. Use 24 for original Keast-6 integration of nonlinear coefficients.
__global__ void p1_compact_sst_tet_volume(
 const nodals_hxt1::Point*pts,const nodals_hxt1::TetConn*tc,
 const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,
 const std::int32_t*slot,const Real*u0,const Real*u1,const Real*u2,
 const Real*k,const Real*omega,Real*mat,Real*rhs,
 std::uint64_t n,double radius,double nu,int equation,double kFloor,
 double omegaFloor,int quadrature){
 std::uint64_t c=blockIdx.x;if(c>=n)return;
 __shared__ Real V[8][3],K[4],W[4],DD[24],RR[24],SS[24];
 if(threadIdx.x<8){int id=tv[c].g[threadIdx.x];
  V[threadIdx.x][0]=u0[id];V[threadIdx.x][1]=u1[id];V[threadIdx.x][2]=u2[id];
  if(threadIdx.x<4){K[threadIdx.x]=k[id];W[threadIdx.x]=omega[id];}
 }
 __syncthreads();
 const int nq=(quadrature==4)?4:24;
 int q=threadIdx.x;
 if(q<nq){
   double l[4],w;
   if(quadrature==4){
     const double a=0.58541019662496845446,b=0.13819660112501051518;
     for(int j=0;j<4;++j)l[j]=(j==q)?a:b;
     w=1.0/24.0;
   }else{
     l[0]=hxt5a::h5_tet24_l0[q];l[1]=hxt5a::h5_tet24_l1[q];
     l[2]=hxt5a::h5_tet24_l2[q];l[3]=hxt5a::h5_tet24_l3[q];
     w=hxt5a::h5_tet24_w[q];
   }
   double ph[8],gr[8][3],gp[8][3];
   hxt2::tet_basis(l[0],l[1],l[2],l[3],ph,gr);
   hxt2::tet_phys_grad(gr,tg[c],gp);
   Real gu[9]={Real(0)},gk[3]={Real(0)},gw[3]={Real(0)},kv=0,wv=0;
   for(int a=0;a<8;++a)for(int d=0;d<3;++d){
     Real gd=(Real)gp[a][d];
     gu[d]+=V[a][0]*gd;gu[3+d]+=V[a][1]*gd;gu[6+d]+=V[a][2]*gd;
     if(a<4){gk[d]+=K[a]*gd;gw[d]+=W[a]*gd;}
   }
   for(int a=0;a<4;++a){kv+=K[a]*(Real)l[a];wv+=W[a]*(Real)l[a];}
   Real xx=0,yy=0;for(int a=0;a<4;++a){auto X=pts[tc[c].v[a]];
     xx+=(Real)(l[a]*X.x);yy+=(Real)(l[a]*X.y);}
   LocalSST ss=local_sst(xx,yy,(Real)radius,(Real)nu,gu,kv,wv,gk,gw,
       (Real)kFloor,(Real)omegaFloor,stage20_distance(tc,(int)c,l));
   auto coef=nodals::transport::volume_coefficients<Real>(
       equation,K_EQ,(Real)(tg[c].det*w),ss);
   DD[q]=coef.diffusion;RR[q]=coef.reaction;SS[q]=coef.source;
 }
 __syncthreads();
 const double GL[4][3]={{-1,-1,-1},{1,0,0},{0,1,0},{0,0,1}};
 if(threadIdx.x<16){int a=threadIdx.x/4,b=threadIdx.x%4;
   double gab[3]={0,0,0},gbb[3]={0,0,0};
   for(int d=0;d<3;++d)for(int j=0;j<3;++j){
     gab[d]+=GL[a][j]*tg[c].invJ[3*j+d];
     gbb[d]+=GL[b][j]*tg[c].invJ[3*j+d];
   }
   double prod=gab[0]*gbb[0]+gab[1]*gbb[1]+gab[2]*gbb[2];
   Real v=0;
   for(int i=0;i<nq;++i){
     double na,nb;
     if(quadrature==4){const double aa=0.58541019662496845446,bb=0.13819660112501051518;
       na=(a==i)?aa:bb;nb=(b==i)?aa:bb;
     }else{
       double l[4]={hxt5a::h5_tet24_l0[i],hxt5a::h5_tet24_l1[i],
                    hxt5a::h5_tet24_l2[i],hxt5a::h5_tet24_l3[i]};
       na=l[a];nb=l[b];
     }
     v+=DD[i]*(Real)prod+RR[i]*(Real)(na*nb);
   }
   atomicAdd(mat+slot[c*64ull+(size_t)a*8+b],v);
 }
 if(threadIdx.x<4){int a=threadIdx.x;Real v=0;
  for(int i=0;i<nq;++i){double na;
    if(quadrature==4)na=(i==a)?0.58541019662496845446:0.13819660112501051518;
    else {double l[4]={hxt5a::h5_tet24_l0[i],hxt5a::h5_tet24_l1[i],
                  hxt5a::h5_tet24_l2[i],hxt5a::h5_tet24_l3[i]};na=l[a];}
    v+=SS[i]*(Real)na;
  }
  atomicAdd(rhs+tv[c].g[a],v);
 }
}
