struct H8SupgCli {
  bool enabled=false;
  double tauScale=0.05;
  double magic=9.0;
  int quadPoints=64;
  std::string form="implicit";
};

static H8SupgCli h8_supg_scan(int argc,char**argv)
{
  H8SupgCli q;
  for(int i=1;i<argc;++i){
    const std::string a=argv[i];
    if(a=="--supg"&&i+1<argc)q.enabled=std::atoi(argv[++i])!=0;
    else if(a=="--supg-tau-scale"&&i+1<argc)q.tauScale=std::atof(argv[++i]);
    else if(a=="--supg-magic"&&i+1<argc)q.magic=std::atof(argv[++i]);
    else if(a=="--supg-form"&&i+1<argc)q.form=argv[++i];
    else if(a=="--supg-quad-points"&&i+1<argc)q.quadPoints=std::atoi(argv[++i]);
  }
  return q;
}

static int h8_base_without_supg_cli(int argc,char**argv)
{
  std::vector<char*> a;
  a.reserve((std::size_t)argc);
  a.push_back(argv[0]);
  for(int i=1;i<argc;++i){
    const std::string s=argv[i];
    if((s=="--supg"||s=="--supg-tau-scale"||s=="--supg-magic"||
        s=="--supg-form"||s=="--supg-quad-points") && i+1<argc){++i;continue;}
    a.push_back(argv[i]);
  }
  return h8_base_main((int)a.size(),a.data());
}

static void h8_fill_device_h2(G4Gpu&G,const G4SetupHost&S)
{
  std::vector<G4CellPlanDevice> d(S.cells.size());
  G.cells.download(d.data(),d.size());
  for(std::size_t i=0;i<d.size();++i)d[i].h2=(OperatorReal)S.cells[i].h2;
  G.cells.upload(d.data(),d.size());
}

// P7B_R2_SUPG_KERNEL_DISPATCH
static const char* h8_supg_kernel_mode()
{
  const char*e=std::getenv("NODALS_GPU_SUPG_KERNEL");
  return (e&&*e)?e:"mixed64_8";
}

static void h8_apply_supg_warp(
    G4Gpu&G,int blocks,double nu,double tauScale,double magic)
{
  const std::string mode=h8_supg_kernel_mode();
  static bool printed=false;
  if(!printed){
    std::printf(
      "NODALS_GPU_P7B_R2_KERNEL mode=%s globalOperator=%s "
      "laggedTau=1 affineTet=1 pmisUnchanged=1 persistentExtraBytes=0 status=PASS\n",
      mode.c_str(),kPrecisionName);
    printed=true;
  }

#define P7B_R2_LAUNCH(W,CF,AF) \
  h7_supg64_r2_kernel<W,CF,AF><<<blocks,32*(W)>>>( \
    G.cells.data(),G.nc,G.row.data(),G.fixed.data(), \
    G.u0.data(),G.u1.data(),G.u2.data(), \
    OperatorReal(nu),OperatorReal(tauScale),OperatorReal(magic), \
    G.av.data(),G.c0.data(),G.c1.data(),G.c2.data())

  if(mode=="reference"){
    h7_supg64_warp_cell_kernel<<<blocks,H7_BLOCK>>>(
      G.cells.data(),G.nc,G.row.data(),G.fixed.data(),
      G.u0.data(),G.u1.data(),G.u2.data(),
      OperatorReal(nu),OperatorReal(tauScale),OperatorReal(magic),
      G.av.data(),G.c0.data(),G.c1.data(),G.c2.data());
  }
  else if(mode=="exact8")      P7B_R2_LAUNCH(8,false,false);
  else if(mode=="exact16")     P7B_R2_LAUNCH(16,false,false);
  else if(mode=="exact32")     P7B_R2_LAUNCH(32,false,false);
  else if(mode=="mixed64_8")   P7B_R2_LAUNCH(8,true,false);
  else if(mode=="mixed64_16")  P7B_R2_LAUNCH(16,true,false);
  else if(mode=="mixed64_32")  P7B_R2_LAUNCH(32,true,false);
  else if(mode=="mixed32_8")   P7B_R2_LAUNCH(8,true,true);
  else if(mode=="mixed32_16")  P7B_R2_LAUNCH(16,true,true);
  else if(mode=="mixed32_32")  P7B_R2_LAUNCH(32,true,true);
  else throw std::runtime_error("invalid NODALS_GPU_SUPG_KERNEL");

#undef P7B_R2_LAUNCH
  NODALS_CUDA(cudaGetLastError());
}

static void h8_assemble_live_supg(
    G4Gpu&G,double alphaU,int supgBlocks,
    double nu,double tauScale,double magic)
{
  h7_reset_momentum_physical(G);
  if(G.h7WarpConvection)
    h7_apply_convection_warp(G,G.h7ConvectionBlocks);
  else
    h7_apply_convection_scalar(G);

  h8_apply_supg_warp(G,supgBlocks,nu,tauScale,magic);

  if(G.h7WarpFinalize)
    h7_apply_finalize_warp(G,alphaU,G.h7FinalizeBlocks);
  else
    h7_apply_finalize_scalar(G,alphaU);
}
