__global__ void sst_interface_diffusion_cached(
    const nodals_hxt1::HexVel*hv,
    const nodals_hxt1::TetVel*tv,
    const hxt2::InterfacePlan*P,
    const SSTInterfaceQuadCache*Q,
    const SSTInterfaceSideCache*SC,
    const Real*u0,const Real*u1,const Real*u2,
    const Real*k,const Real*omega,
    Real*mat,
    std::uint64_t n,
    double radius,double nu,double gammaPenalty,
    int equation,
    double kFloor,double omegaFloor)
{
  const std::uint64_t iside=(std::uint64_t)blockIdx.x;

  if(iside>=2*n)
    return;

  const int ir=(int)(iside>>1);
  const int side=(int)(iside&1);

  const auto&p=P[ir];
  const auto&HG=hv[p.hexCell];

  const int tcell=p.tetCell[side];
  const auto&TG=tv[tcell];

  const auto&S=SC[iside];
  const int un=S.un;

  __shared__ Real hu0[14],hu1[14],hu2[14];
  __shared__ Real hk[14],hw[14];

  __shared__ Real tu0[8],tu1[8],tu2[8];
  __shared__ Real tk[8],tw[8];

  __shared__ Real sdh[12],sdt[12];

  if(threadIdx.x<14){
    const int g=HG.g[threadIdx.x];

    hu0[threadIdx.x]=u0[g];
    hu1[threadIdx.x]=u1[g];
    hu2[threadIdx.x]=u2[g];

    hk[threadIdx.x]=k[g];
    hw[threadIdx.x]=omega[g];
  }

  if(threadIdx.x<8){
    const int g=TG.g[threadIdx.x];

    tu0[threadIdx.x]=u0[g];
    tu1[threadIdx.x]=u1[g];
    tu2[threadIdx.x]=u2[g];

    tk[threadIdx.x]=k[g];
    tw[threadIdx.x]=omega[g];
  }

  __syncthreads();

  const int q=(int)threadIdx.x;

  if(q<12){

    const auto&C=
      Q[iside*12ull+(std::size_t)q];

    Real guh[9]={0,0,0,0,0,0,0,0,0};
    Real gut[9]={0,0,0,0,0,0,0,0,0};

    Real gkh[3]={0,0,0};
    Real gwh[3]={0,0,0};

    Real gkt[3]={0,0,0};
    Real gwt[3]={0,0,0};

    Real kvh=0,wvh=0;
    Real kvt=0,wvt=0;

    for(int a=0;a<14;++a){

      const Real pa=C.ph[a];

      kvh+=hk[a]*pa;
      wvh+=hw[a]*pa;

      for(int d=0;d<3;++d){

        const Real gd=C.gph[a][d];

        gkh[d]+=hk[a]*gd;
        gwh[d]+=hw[a]*gd;

        guh[d]+=hu0[a]*gd;
        guh[3+d]+=hu1[a]*gd;
        guh[6+d]+=hu2[a]*gd;
      }
    }

    for(int a=0;a<8;++a){

      const Real pa=C.pt[a];

      kvt+=tk[a]*pa;
      wvt+=tw[a]*pa;

      for(int d=0;d<3;++d){

        const Real gd=C.gpt[a][d];

        gkt[d]+=tk[a]*gd;
        gwt[d]+=tw[a]*gd;

        gut[d]+=tu0[a]*gd;
        gut[3+d]+=tu1[a]*gd;
        gut[6+d]+=tu2[a]*gd;
      }
    }

    const LocalSST sh=
      local_sst(
        C.xh,C.yh,
        (Real)radius,(Real)nu,
        guh,
        kvh,wvh,
        gkh,gwh,
        (Real)kFloor,
        (Real)omegaFloor);

    const LocalSST st=
      local_sst(
        C.xt,C.yt,
        (Real)radius,(Real)nu,
        gut,
        kvt,wvt,
        gkt,gwt,
        (Real)kFloor,
        (Real)omegaFloor);

    sdh[q]=
      (equation==K_EQ ? sh.sigmaK : sh.sigmaW)
      *sh.nut;

    sdt[q]=
      (equation==K_EQ ? st.sigmaK : st.sigmaW)
      *st.nut;
  }

  __syncthreads();

  const Real tauBase=
    (Real)(
      gammaPenalty*nu*
      fmax(
        p.invHHex,
        p.invHTet[side]));

  const int npair=un*un;

  for(int pair=(int)threadIdx.x;
      pair<npair;
      pair+=(int)blockDim.x)
  {
    const int ii=pair/un;
    const int jj=pair%un;

    Real kij=0;

    for(int q2=0;q2<12;++q2){

      const auto&C=
        Q[iside*12ull+(std::size_t)q2];

      const Real ji=C.jump[ii];
      const Real jjv=C.jump[jj];

      const Real di=
        Real(.5)*
        (sdh[q2]*C.hn[ii]
        +sdt[q2]*C.tn[ii]);

      const Real dj=
        Real(.5)*
        (sdh[q2]*C.hn[jj]
        +sdt[q2]*C.tn[jj]);

      const Real cons=
        -(dj*ji+di*jjv);

      const Real tauFull=
        (Real)gammaPenalty*
        fmax(
          ((Real)nu+sdh[q2])*
            (Real)p.invHHex,
          ((Real)nu+sdt[q2])*
            (Real)p.invHTet[side]);

      const Real deltaTau=
        fmax(
          Real(0),
          tauFull-tauBase);

      kij+=
        C.wq*
        (cons+
         deltaTau*ji*jjv);
    }

    const int sl=
      S.slot[ii*22+jj];

    atomicAdd(
      mat+sl,
      kij);
  }
}


