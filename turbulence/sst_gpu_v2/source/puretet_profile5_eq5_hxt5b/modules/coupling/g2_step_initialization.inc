inline StepResult step_turbulence(
    State&S,int outer,const nodals_hxt1::HostMesh&M,const hxt5b::BoundaryHost&BH,
    const Dev<hxt5b::HexFacePlan>&inH,const Dev<hxt5b::TetFacePlan>&inT,const Dev<hxt5b::HexFacePlan>&wallH,
    const Dev<nodals_hxt1::Point>&pts,const Dev<nodals_hxt1::HexConn>&hc,const Dev<nodals_hxt1::TetConn>&tc,const Dev<nodals_hxt1::HexVel>&hv,const Dev<nodals_hxt1::TetVel>&tv,const Dev<nodals_hxt1::TetGeom>&tg,const Dev<hxt2::InterfacePlan>&ip,
    const Dev<double>&hexConvGrad,const Dev<double>&hexConvDetW,
    hxt4b::GpuCSR&A,const hxt4b::ColoringHost&C,const Dev<unsigned char>&fixed,const Real*u0,const Real*u1,const Real*u2,
    Dev<Real>&vol,Dev<Real>&dg,Dev<Real>&wall,Dev<unsigned long long>&bad,hxt4a::Reducer<Real>&red,
    double bulk,double radius,double nu,double gamma,double wallSampleFraction,const hxt5c::Options&O,
    const hxt5c::SSTInterfaceCache*ifCache=nullptr,
    const Real*calibratedK=nullptr,const Real*calibratedOmega=nullptr)
{
  const char* compactFlag=std::getenv("SST_COMPACT_P1");
  const bool compactQ1=compactFlag && std::string(compactFlag)=="1";
  // P1_CONVFIX: assemble convection exactly once in g2_step_convection.inc.
  HXT1_CUDA(cudaMemcpy(S.oldK.p,S.k.p,(std::size_t)A.n*sizeof(Real),cudaMemcpyDeviceToDevice));HXT1_CUDA(cudaMemcpy(S.oldW.p,S.omega.p,(std::size_t)A.n*sizeof(Real),cudaMemcpyDeviceToDevice));
  const int stateMode=hxt5c::sst_state_space_mode_host();
  const bool q1Transport=(stateMode==hxt5c::SST_STATE_Q1);
  const bool bernsteinBF3=std::getenv("NODALS_BF3_BERNSTEIN") &&
    std::string(std::getenv("NODALS_BF3_BERNSTEIN"))=="1";
  if(bernsteinBF3 && (stateMode!=hxt5c::SST_STATE_Q1BF2_UNLIMITED || compactQ1 || M.h.nhex))
    throw std::runtime_error("BF3_BERNSTEIN requires active tetrahedral BF3, compact=0");
  const Dev<unsigned char>& scalarFixed=compactQ1?fixed:(q1Transport?S.q1Fixed:fixed);
  // BF3_FAILFAST_20261103: log and validate the ACTUAL native scalar mode,
  // including the first step of a Stage31 restart (which may be iteration 201).
  const bool bf3Require = std::getenv("NODALS_BF3_REQUIRE_ACTIVE") &&
    std::string(std::getenv("NODALS_BF3_REQUIRE_ACTIVE"))=="1";
  static bool bf3FirstStepChecked=false;
  if(bf3Require && !bf3FirstStepChecked){
    const int nVertex=S.nvert;
    const int nTotal=A.n;
    const int nBubble=nTotal-nVertex;
    std::printf("NODALS_BF3_EFFECTIVE_MODE source=BF3_FAILFAST_20261103 it=%d env=%s mode=%s "
       "compact=%d scalarDofs=%d vertexDofs=%d bubbleDofs=%d localTetModes=8\n",
       outer,std::getenv("SST_STATE_SPACE")?std::getenv("SST_STATE_SPACE"):"UNSET",
       hxt5c::sst_state_space_name(stateMode),(int)compactQ1,nTotal,nVertex,nBubble);
    if(stateMode!=hxt5c::SST_STATE_Q1BF2_UNLIMITED || compactQ1)
      throw std::runtime_error("BF3_FAILFAST: native solver is NOT in active Q1_BF2_UNLIMITED mode");
    if(nBubble<=0)throw std::runtime_error("BF3_FAILFAST: no scalar bubble DOFs");
    std::vector<unsigned char> hostMask((std::size_t)nTotal,0);
    HXT1_CUDA(cudaMemcpy(hostMask.data(),scalarFixed.p,
       (std::size_t)nTotal*sizeof(unsigned char),cudaMemcpyDeviceToHost));
    int freeBubble=0,maskedBubble=0,maskedVertex=0;
    for(int i=0;i<nVertex;++i)if(hostMask[(std::size_t)i])++maskedVertex;
    for(int i=nVertex;i<nTotal;++i){
      if(hostMask[(std::size_t)i])++maskedBubble;
      else ++freeBubble;
    }
    std::printf("NODALS_BF3_ACTIVE_ROWS it=%d freeBubble=%d maskedBubble=%d "
       "totalBubble=%d maskedVertex=%d status=%s\n",outer,freeBubble,maskedBubble,nBubble,maskedVertex,
       freeBubble>0?"PASS":"FAIL");
    std::fflush(stdout);
    if(freeBubble<=0)throw std::runtime_error("BF3_FAILFAST: all scalar bubble rows are fixed");
    bf3FirstStepChecked=true;
  }
  if(outer==1 && q1Transport){
    std::vector<unsigned char> seen((std::size_t)S.nvert,0);
    int count=0,duplicate=0;
    for(int row:C.rows){
      if(row<0 || row>=S.nvert)continue;
      if(seen[(std::size_t)row])++duplicate;
      else {seen[(std::size_t)row]=1;++count;}
    }
    std::printf("P1_PARITY_COLOR mode=%s vertexRowsInColoring=%d expected=%d duplicates=%d status=%s\n",
       compactQ1?"compact":"masked",count,S.nvert,duplicate,
       (count==S.nvert && duplicate==0)?"PASS":"MISSING_SCALAR_ROWS");
  }
  if(outer==1 || (bf3Require && outer>1)){
    std::printf("NODALS_SST_STATE_SPACE mode=%s transport=%s constitutive=%s activeDofs=%d constrainedEnrichmentDofs=%d positivity=BERNSTEIN_CONTROL_NET_IF_Q2 status=PASS\n",
      hxt5c::sst_state_space_name(stateMode),
      q1Transport?"VERTEX_Q1":"FULL_14DOF",
      stateMode==hxt5c::SST_STATE_BERNSTEIN_Q2?"BERNSTEIN_Q2_POSITIVE_LIMITED14":
        (stateMode==hxt5c::SST_STATE_Q1BF2_UNLIMITED?"Q1_BF2_DIRECT_14DOF_UNLIMITED":"VERTEX_Q1"),
      q1Transport?S.nvert:A.n,
      compactQ1?0:(q1Transport?(A.n-S.nvert):0));
  }
