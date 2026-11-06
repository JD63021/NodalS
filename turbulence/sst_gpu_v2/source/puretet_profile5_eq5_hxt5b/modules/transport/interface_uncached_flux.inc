__global__ void sst_interface_diffusion(
    const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,const nodals_hxt1::TetConn*tc,
    const nodals_hxt1::HexVel*hv,const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,
    const hxt2::InterfacePlan*P,const std::int64_t*row,const std::int32_t*col,
    const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,Real*mat,std::uint64_t n,
    double radius,double nu,double gammaPenalty,int equation,double kFloor,double omegaFloor,unsigned long long*bad)
{
  std::uint64_t iside=(std::uint64_t)blockIdx.x;if(iside>=2*n)return;
  int ir=(int)(iside>>1),side=(int)(iside&1);const auto&p=P[ir];const auto&HG=hv[p.hexCell];int tcell=p.tetCell[side],tf=p.tetFace[side];const auto&TG=tv[tcell];
  __shared__ int ug[22];__shared__ int un;
  __shared__ Real hu0[14],hu1[14],hu2[14],hk[14],hw[14],tu0[8],tu1[8],tu2[8],tk[8],tw[8];
  // PERF_V3_Q12_SCALAR_INTERFACE: positive degree-6 triangle rule
  __shared__ Real sph[12][14],sgph[12][14][3],spt[12][8],sgpt[12][8][3],sdh[12],sdt[12],swq[12];
  if(threadIdx.x==0){int m=0;for(int a=0;a<14;++a)ug[m++]=HG.g[a];for(int b=0;b<8;++b){int g=TG.g[b],found=0;for(int q=0;q<m;++q)if(ug[q]==g)found=1;if(!found)ug[m++]=g;}un=m;}
  if(threadIdx.x<14){int g=HG.g[threadIdx.x];hu0[threadIdx.x]=u0[g];hu1[threadIdx.x]=u1[g];hu2[threadIdx.x]=u2[g];hk[threadIdx.x]=k[g];hw[threadIdx.x]=omega[g];}
  if(threadIdx.x<8){int g=TG.g[threadIdx.x];tu0[threadIdx.x]=u0[g];tu1[threadIdx.x]=u1[g];tu2[threadIdx.x]=u2[g];tk[threadIdx.x]=k[g];tw[threadIdx.x]=omega[g];}
  __syncthreads();
  for(int q=(int)threadIdx.x;q<12;q+=(int)blockDim.x){
    double L[3]={hxt5b::QL0[q],hxt5b::QL1[q],hxt5b::QL2[q]};
    swq[q]=(Real)(hxt5b::QW12[q]*p.areaTri[side]);
    double xr=0,yr=0,zr=0;for(int m=0;m<3;++m){int hl=p.hexTriLocal[side][m];xr+=L[m]*hxt4a::h4_hex_sign[hl][0];yr+=L[m]*hxt4a::h4_hex_sign[hl][1];zr+=L[m]*hxt4a::h4_hex_sign[hl][2];}
    double ph[14],grh[14][3],gph[14][3],I[9],det;hxt2::hex_basis_ref(xr,yr,zr,ph,grh);
    if(!hxt2::hex_metric(pts,hc[p.hexCell],xr,yr,zr,I,det)){atomicAdd(bad,1ULL);sdh[q]=sdt[q]=0;continue;}hxt2::phys_grad14(grh,I,gph);
    double lam[4]={0,0,0,0};for(int m=0;m<3;++m)lam[hxt4a::h4_tet_face[tf][m]]=L[m];double pt[8],grt[8][3],gpt[8][3];hxt2::tet_basis(lam[0],lam[1],lam[2],lam[3],pt,grt);hxt2::tet_phys_grad(grt,tg[tcell],gpt);
    Real guh[9]={0,0,0,0,0,0,0,0,0},gut[9]={0,0,0,0,0,0,0,0,0},gkh[3]={0,0,0},gwh[3]={0,0,0},gkt[3]={0,0,0},gwt[3]={0,0,0};Real kvh=0,wvh=0,kvt=0,wvt=0;
    for(int a=0;a<14;++a){sph[q][a]=(Real)ph[a];kvh+=hk[a]*(Real)ph[a];wvh+=hw[a]*(Real)ph[a];for(int d=0;d<3;++d){Real gd=(Real)gph[a][d];sgph[q][a][d]=gd;gkh[d]+=hk[a]*gd;gwh[d]+=hw[a]*gd;guh[d]+=hu0[a]*gd;guh[3+d]+=hu1[a]*gd;guh[6+d]+=hu2[a]*gd;}}
    for(int a=0;a<8;++a){spt[q][a]=(Real)pt[a];kvt+=tk[a]*(Real)pt[a];wvt+=tw[a]*(Real)pt[a];for(int d=0;d<3;++d){Real gd=(Real)gpt[a][d];sgpt[q][a][d]=gd;gkt[d]+=tk[a]*gd;gwt[d]+=tw[a]*gd;gut[d]+=tu0[a]*gd;gut[3+d]+=tu1[a]*gd;gut[6+d]+=tu2[a]*gd;}}
    double xh=0,yh=0;for(int a=0;a<8;++a){double N=.125*(1+hxt4a::h4_hex_sign[a][0]*xr)*(1+hxt4a::h4_hex_sign[a][1]*yr)*(1+hxt4a::h4_hex_sign[a][2]*zr);auto X=pts[hc[p.hexCell].v[a]];xh+=N*X.x;yh+=N*X.y;}
    double xt=0,yt=0;for(int a=0;a<4;++a){auto X=pts[tc[tcell].v[a]];xt+=lam[a]*X.x;yt+=lam[a]*X.y;}
    LocalSST sh=local_sst((Real)xh,(Real)yh,(Real)radius,(Real)nu,guh,kvh,wvh,gkh,gwh,(Real)kFloor,(Real)omegaFloor);LocalSST st=local_sst((Real)xt,(Real)yt,(Real)radius,(Real)nu,gut,kvt,wvt,gkt,gwt,(Real)kFloor,(Real)omegaFloor);
    sdh[q]=(equation==K_EQ?sh.sigmaK:sh.sigmaW)*sh.nut;sdt[q]=(equation==K_EQ?st.sigmaK:st.sigmaW)*st.nut;
  }
  __syncthreads();
  const Real nx=(Real)p.normal[0],ny=(Real)p.normal[1],nz=(Real)p.normal[2];const Real tauBase=(Real)(gammaPenalty*nu*fmax(p.invHHex,p.invHTet[side]));int npair=un*un;
  for(int pair=(int)threadIdx.x;pair<npair;pair+=(int)blockDim.x){int ii=pair/un,jj=pair%un,gi=ug[ii],gj=ug[jj];Real kij=0;
    for(int q=0;q<12;++q){Real ji=0,jjv=0,di=0,dj=0;for(int a=0;a<14;++a){if(HG.g[a]==gi){ji+=sph[q][a];di+=Real(.5)*sdh[q]*(sgph[q][a][0]*nx+sgph[q][a][1]*ny+sgph[q][a][2]*nz);}if(HG.g[a]==gj){jjv+=sph[q][a];dj+=Real(.5)*sdh[q]*(sgph[q][a][0]*nx+sgph[q][a][1]*ny+sgph[q][a][2]*nz);}}
      for(int a=0;a<8;++a){if(TG.g[a]==gi){ji-=spt[q][a];di+=Real(.5)*sdt[q]*(sgpt[q][a][0]*nx+sgpt[q][a][1]*ny+sgpt[q][a][2]*nz);}if(TG.g[a]==gj){jjv-=spt[q][a];dj+=Real(.5)*sdt[q]*(sgpt[q][a][0]*nx+sgpt[q][a][1]*ny+sgpt[q][a][2]*nz);}}
      Real cons=-(dj*ji+di*jjv);Real tauFull=(Real)gammaPenalty*fmax(((Real)nu+sdh[q])*(Real)p.invHHex,((Real)nu+sdt[q])*(Real)p.invHTet[side]);Real deltaTau=fmax(Real(0),tauFull-tauBase);kij+=swq[q]*(cons+deltaTau*ji*jjv);
    }int sl=hxt4b::csr_find(row,col,gi,gj);atomicAdd(mat+sl,kij);
  }
}

