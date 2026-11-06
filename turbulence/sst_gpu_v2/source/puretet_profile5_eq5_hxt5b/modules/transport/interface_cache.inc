// ============================================================
// HXT5C_SST_INTERFACE_CACHE_V1
//
// Immutable hybrid HEX/TET interface data.
//
// Cached once:
//   Q12 basis values
//   physical gradients
//   physical quadrature x,y
//   surface weights
//   jump basis
//   HEX/TET normal derivatives
//   union interface DOFs
//   CSR slots
//
// Runtime remains purely dynamic SST state evaluation.
// ============================================================

struct SSTInterfaceQuadCache {
  Real ph[14];
  Real gph[14][3];

  Real pt[8];
  Real gpt[8][3];

  Real xh=0, yh=0;
  Real xt=0, yt=0;
  Real wq=0;

  Real jump[22];
  Real hn[22];
  Real tn[22];
};

struct SSTInterfaceSideCache {
  int un=0;
  int ug[22];
  int slot[22*22];
};

struct SSTInterfaceCache {
  SSTInterfaceQuadCache* q=nullptr;
  SSTInterfaceSideCache* side=nullptr;
  std::uint64_t nside=0;

  SSTInterfaceCache()=default;

  explicit SSTInterfaceCache(std::uint64_t ninterface)
    : nside(2*ninterface)
  {
    if(!nside) return;

    HXT1_CUDA(cudaMalloc(
      (void**)&q,
      (std::size_t)nside*12ull*
      sizeof(SSTInterfaceQuadCache)));

    HXT1_CUDA(cudaMalloc(
      (void**)&side,
      (std::size_t)nside*
      sizeof(SSTInterfaceSideCache)));
  }

  SSTInterfaceCache(const SSTInterfaceCache&)=delete;
  SSTInterfaceCache&operator=(const SSTInterfaceCache&)=delete;

  ~SSTInterfaceCache(){
    if(q) cudaFree(q);
    if(side) cudaFree(side);
  }
};


__global__ void sst_interface_cache_setup(
    const nodals_hxt1::Point*pts,
    const nodals_hxt1::HexConn*hc,
    const nodals_hxt1::TetConn*tc,
    const nodals_hxt1::HexVel*hv,
    const nodals_hxt1::TetVel*tv,
    const nodals_hxt1::TetGeom*tg,
    const hxt2::InterfacePlan*P,
    const std::int64_t*row,
    const std::int32_t*col,
    SSTInterfaceQuadCache*Q,
    SSTInterfaceSideCache*S,
    std::uint64_t n,
    unsigned long long*bad)
{
  const std::uint64_t iside=
    (std::uint64_t)blockIdx.x;

  if(iside>=2*n) return;

  const int ir=(int)(iside>>1);
  const int side=(int)(iside&1);

  const auto&p=P[ir];

  const auto&HG=hv[p.hexCell];

  const int tcell=p.tetCell[side];
  const int tf=p.tetFace[side];

  const auto&TG=tv[tcell];

  __shared__ int ug[22];
  __shared__ int un;

  if(threadIdx.x==0){
    int m=0;

    for(int a=0;a<14;++a)
      ug[m++]=HG.g[a];

    for(int a=0;a<8;++a){
      const int g=TG.g[a];
      bool found=false;

      for(int j=0;j<m;++j)
        if(ug[j]==g){
          found=true;
          break;
        }

      if(!found)
        ug[m++]=g;
    }

    un=m;
    S[iside].un=m;
  }

  __syncthreads();

  for(int m=(int)threadIdx.x;
      m<22;
      m+=(int)blockDim.x)
  {
    S[iside].ug[m]=(m<un)?ug[m]:-1;
  }

  const int npair=un*un;

  for(int pair=(int)threadIdx.x;
      pair<npair;
      pair+=(int)blockDim.x)
  {
    const int ii=pair/un;
    const int jj=pair%un;

    S[iside].slot[ii*22+jj]=
      hxt4b::csr_find(
        row,col,
        ug[ii],ug[jj]);
  }

  const int iq=(int)threadIdx.x;

  if(iq<12){

    auto&C=Q[iside*12ull+(std::size_t)iq];

    const double L[3]={
      hxt5b::QL0[iq],
      hxt5b::QL1[iq],
      hxt5b::QL2[iq]
    };

    C.wq=(Real)(
      hxt5b::QW12[iq]*
      p.areaTri[side]);

    double xr=0,yr=0,zr=0;

    for(int m=0;m<3;++m){
      const int hl=
        p.hexTriLocal[side][m];

      xr+=L[m]*hxt4a::h4_hex_sign[hl][0];
      yr+=L[m]*hxt4a::h4_hex_sign[hl][1];
      zr+=L[m]*hxt4a::h4_hex_sign[hl][2];
    }

    double ph[14],grh[14][3],gph[14][3];
    double I[9],det;

    hxt2::hex_basis_ref(
      xr,yr,zr,
      ph,grh);

    if(!hxt2::hex_metric(
         pts,hc[p.hexCell],
         xr,yr,zr,
         I,det))
    {
      atomicAdd(bad,1ULL);
      return;
    }

    hxt2::phys_grad14(
      grh,I,gph);

    double lam[4]={0,0,0,0};

    for(int m=0;m<3;++m)
      lam[hxt4a::h4_tet_face[tf][m]]=L[m];

    double pt[8],grt[8][3],gpt[8][3];

    hxt2::tet_basis(
      lam[0],lam[1],
      lam[2],lam[3],
      pt,grt);

    hxt2::tet_phys_grad(
      grt,tg[tcell],gpt);

    for(int a=0;a<14;++a){
      C.ph[a]=(Real)ph[a];

      for(int d=0;d<3;++d)
        C.gph[a][d]=(Real)gph[a][d];
    }

    for(int a=0;a<8;++a){
      C.pt[a]=(Real)pt[a];

      for(int d=0;d<3;++d)
        C.gpt[a][d]=(Real)gpt[a][d];
    }

    double xh=0,yh=0;

    for(int a=0;a<8;++a){
      const double N=
        .125*
        (1+hxt4a::h4_hex_sign[a][0]*xr)*
        (1+hxt4a::h4_hex_sign[a][1]*yr)*
        (1+hxt4a::h4_hex_sign[a][2]*zr);

      const auto X=
        pts[hc[p.hexCell].v[a]];

      xh+=N*X.x;
      yh+=N*X.y;
    }

    double xt=0,yt=0;

    for(int a=0;a<4;++a){
      const auto X=
        pts[tc[tcell].v[a]];

      xt+=lam[a]*X.x;
      yt+=lam[a]*X.y;
    }

    C.xh=(Real)xh;
    C.yh=(Real)yh;
    C.xt=(Real)xt;
    C.yt=(Real)yt;

    const Real nx=(Real)p.normal[0];
    const Real ny=(Real)p.normal[1];
    const Real nz=(Real)p.normal[2];

    for(int m=0;m<22;++m){
      C.jump[m]=Real(0);
      C.hn[m]=Real(0);
      C.tn[m]=Real(0);
    }

    for(int m=0;m<un;++m){

      const int g=ug[m];

      Real j=0;
      Real hn=0;
      Real tn=0;

      for(int a=0;a<14;++a)
        if(HG.g[a]==g){
          j+=C.ph[a];

          hn+=
            C.gph[a][0]*nx+
            C.gph[a][1]*ny+
            C.gph[a][2]*nz;
        }

      for(int a=0;a<8;++a)
        if(TG.g[a]==g){
          j-=C.pt[a];

          tn+=
            C.gpt[a][0]*nx+
            C.gpt[a][1]*ny+
            C.gpt[a][2]*nz;
        }

      C.jump[m]=j;
      C.hn[m]=hn;
      C.tn[m]=tn;
    }
  }
}


inline void setup_sst_interface_cache(
    std::uint64_t ninterface,
    const Dev<nodals_hxt1::Point>&pts,
    const Dev<nodals_hxt1::HexConn>&hc,
    const Dev<nodals_hxt1::TetConn>&tc,
    const Dev<nodals_hxt1::HexVel>&hv,
    const Dev<nodals_hxt1::TetVel>&tv,
    const Dev<nodals_hxt1::TetGeom>&tg,
    const Dev<hxt2::InterfacePlan>&ip,
    hxt4b::GpuCSR&A,
    SSTInterfaceCache&C,
    Dev<unsigned long long>&bad)
{
  HXT1_CUDA(cudaMemset(
    bad.p,0,
    sizeof(unsigned long long)));

  if(ninterface)
    sst_interface_cache_setup
      <<<2*ninterface,128>>>(
        pts.p,hc.p,tc.p,
        hv.p,tv.p,tg.p,
        ip.p,
        A.row.p,A.col.p,
        C.q,C.side,
        ninterface,
        bad.p);

  HXT1_CUDA(cudaGetLastError());
  HXT1_CUDA(cudaDeviceSynchronize());

  unsigned long long nb=0;

  HXT1_CUDA(cudaMemcpy(
    &nb,bad.p,sizeof(nb),
    cudaMemcpyDeviceToHost));

  if(nb)
    throw std::runtime_error(
      "SST persistent interface-cache geometry failure count="+
      std::to_string(nb));

  const double mib=
    (double)(
      C.nside*
      (12ull*sizeof(SSTInterfaceQuadCache)+
       sizeof(SSTInterfaceSideCache)))/
    (1024.0*1024.0);

  std::printf(
    "NODALS_SST_INTERFACE_CACHE "
    "interfaces=%llu sides=%llu quadrature=12 "
    "MiB=%.3f "
    "geometry=PERSISTENT topology=PERSISTENT "
    "csrSlots=PERSISTENT runtime=SST_NUMERIC_ONLY "
    "status=PASS\n",
    (unsigned long long)ninterface,
    (unsigned long long)C.nside,
    mib);
}



// ============================================================
// HXT5C_SST_INTERFACE_CACHED_RUNTIME_V1
//
// Same scalar hybrid Nitsche operator as sst_interface_diffusion.
//
// Cached:
//   basis
//   physical gradients
//   quadrature coordinates/weights
//   union DOF topology
//   jump basis
//   normal derivatives
//   CSR slots
//
// Dynamic:
//   U, k, omega
//   local_sst()
//   sigma*nut
// ============================================================
