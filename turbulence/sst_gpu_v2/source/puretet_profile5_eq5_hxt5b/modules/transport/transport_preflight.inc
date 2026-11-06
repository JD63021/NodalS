inline void assemble_equation(
    Equation eq,const nodals_hxt1::HostMesh&M,const hxt5b::BoundaryHost&BH,
    const Dev<hxt5b::HexFacePlan>&inH,const Dev<hxt5b::TetFacePlan>&inT,const Dev<hxt5b::HexFacePlan>&wallH,
    const Dev<nodals_hxt1::Point>&pts,const Dev<nodals_hxt1::HexConn>&hc,const Dev<nodals_hxt1::TetConn>&tc,const Dev<nodals_hxt1::HexVel>&hv,const Dev<nodals_hxt1::TetVel>&tv,const Dev<nodals_hxt1::TetGeom>&tg,const Dev<hxt2::InterfacePlan>&ip,
    hxt4b::GpuCSR&A,const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,
    Dev<Real>&vol,Dev<Real>&dg,Dev<Real>&wall,Dev<Real>&rhs,Dev<double>&wallStats,Dev<unsigned long long>&bad,Dev<unsigned long long>&diagAudit,
    double qIn,double bulk,double radius,double nu,double interfaceGamma,double wallSampleFraction,const Options&O,int nonlinearIt,
    const SSTInterfaceCache*ifCache=nullptr, const Real*calibratedL1=nullptr)
{
  // Stage33B/C: independently selectable k/omega SUPG. Parsed once per
  // scalar assembly call; existing Stage32/33A kernel path is untouched if OFF.
  const auto s33_read_pos=[](const char* key,const char* fallback)->double{
    const char* p=std::getenv(key);
    const double v=std::stod(p&&*p?p:fallback);
    if(!std::isfinite(v)||v<=0.0)
      throw std::runtime_error(std::string("STAGE33BC invalid ")+key);
    return v;
  };
  const auto s33_read_flag=[](const char* key)->bool{
    const char* p=std::getenv(key);
    const std::string v=(p&&*p)?p:"0";
    if(v!="0" && v!="1")
      throw std::runtime_error(std::string("STAGE33BC ")+key+" must be 0/1");
    return v=="1";
  };
  const char* compactEnv=std::getenv("SST_COMPACT_P1");
  const bool compactQ1=compactEnv && std::string(compactEnv)=="1";
  if(compactQ1 && (M.h.nhex || M.h.ninterface || !q1_transport_enabled()))
    throw std::runtime_error("SST_COMPACT_P1 requires pure-tet SST_STATE_SPACE=q1");
  const char* quadEnv=std::getenv("SST_COMPACT_QUAD");
  const int compactQuad=(quadEnv&&*quadEnv)?std::stoi(quadEnv):4;
  if(compactQ1 && compactQuad!=4 && compactQuad!=24)
    throw std::runtime_error("SST_COMPACT_QUAD must be 4 or 24");
  const bool stage33ScalarSupg=(eq==K_EQ)?s33_read_flag("STAGE33BC_K_SUPG"):
                                             s33_read_flag("STAGE33BC_OMEGA_SUPG");
  const double stage33ScalarTauScale=s33_read_pos(
      eq==K_EQ?"STAGE33BC_K_TAU_SCALE":"STAGE33BC_OMEGA_TAU_SCALE","0.05");
  const double stage33ScalarMagic=s33_read_pos("STAGE33BC_SCALAR_MAGIC","9");
  if(nonlinearIt==1){
    std::printf("STAGE33BC_SCALAR_SUPG equation=%s enabled=%d tauScale=%.8g magic=%.8g quadrature=24 "
      "residual=FROZEN_CONVECTION_DIFFUSION_REACTION_SOURCE ",
      eq==K_EQ?"k":"omega",(int)stage33ScalarSupg,stage33ScalarTauScale,stage33ScalarMagic);
    std::printf("strongVariableDiffusivityGradient=OMITTED wallSources=BOUNDARY_UNMODIFIED "
      "matrix=EXISTING_SCALAR_CSR status=PASS\n");
  }
  if(compactQ1 && stage33ScalarSupg)
    throw std::runtime_error("compact P1 scalar SUPG not implemented; disable scalar SUPG");
  if(stage33ScalarSupg && (M.h.nhex>0 || M.h.ninterface>0 || M.h.ntet==0))
      throw std::runtime_error("STAGE33BC scalar SUPG requires pure tetrahedral mesh");
  // PURETET_AB5_MODE_MASK_V1: fail fast for unsupported configuration.
  const char* eq5Mode=std::getenv("PURETET_EQ5_SAMPLE_SET");
  const std::string eq5ModeText=eq5Mode?std::string(eq5Mode):std::string("C");
  const int eq5Mask=(eq5ModeText=="A")?4:((eq5ModeText=="B")?21:((eq5ModeText=="S1")?1:((eq5ModeText=="S5")?16:31)));
  if(eq5ModeText!="A" && eq5ModeText!="B" && eq5ModeText!="C" && eq5ModeText!="S1" && eq5ModeText!="S5")
    throw std::runtime_error("PURETET_EQ5_SAMPLE_SET must be A, B, C, S1, or S5");
  if((eq5Mask!=31) && !(M.h.ntet>0 && M.h.nhex==0 && O.omegaWallMode==OMEGA_WALL_PROFILE5_EQ_SST))
    throw std::runtime_error("EQ5 A/B sample selection supports PURE TET PROFILE5_EQ_SST only");
  const auto inlet23=nodals::stage23::from_environment(bulk);
  const std::size_t scalarBufferCount=compactQ1?(A.nnz+1):A.nnz;
  if(compactQ1 && (vol.n<scalarBufferCount || dg.n<scalarBufferCount || wall.n<scalarBufferCount))
    throw std::runtime_error("compact P1 boundary discard buffer too short");
  HXT1_CUDA(cudaMemset(vol.p,0,scalarBufferCount*sizeof(Real)));HXT1_CUDA(cudaMemset(dg.p,0,scalarBufferCount*sizeof(Real)));HXT1_CUDA(cudaMemset(wall.p,0,scalarBufferCount*sizeof(Real)));HXT1_CUDA(cudaMemset(rhs.p,0,rhs.n*sizeof(Real)));HXT1_CUDA(cudaMemset(wallStats.p,0,wallStats.n*sizeof(double)));HXT1_CUDA(cudaMemset(bad.p,0,bad.n*sizeof(unsigned long long)));
  // PURETET_EXP1_SOURCE: all buffers are additional, never used in assembly.
  const bool exp1=(!compactQ1 && eq==OMEGA_EQ && M.h.ntet>0 && M.h.nhex==0 && q1_transport_enabled() && puretet_source_exp1::scheduled(nonlinearIt));
  Dev<Real> expProd,expCross,expReaction;
  if(exp1){expProd.upload(std::vector<Real>((size_t)A.n,Real(0)));expCross.upload(std::vector<Real>((size_t)A.n,Real(0)));expReaction.upload(std::vector<Real>((size_t)A.nnz,Real(0)));}
  std::vector<Real> expVolume,expInlet,expTotal,expP,expC;
  auto takeRhs=[&](std::vector<Real>&v){v.resize((size_t)A.n);HXT1_CUDA(cudaMemcpy(v.data(),rhs.p,(size_t)A.n*sizeof(Real),cudaMemcpyDeviceToHost));};
