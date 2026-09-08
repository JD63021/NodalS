  NODALS_CUDA(cudaSetDevice(0));cudaDeviceProp prop{};NODALS_CUDA(cudaGetDeviceProperties(&prop,0));upload_tensors();NODALS_CUDA(cudaDeviceSynchronize());
  const auto memBaseline=device_memory_info();
  auto M=load_foam_tet_mesh(mesh);auto S=build_g4_setup(M,re,bulk,wall,inlet,outlet);
  std::array<std::vector<double>,3>U0;for(auto&u:U0)u.assign((std::size_t)S.topo.n,0.0);
  std::vector<double>hostA,initDelta;std::array<std::vector<double>,3>hostConv;
  host_assemble_central_g4(S,U0,hostA,hostConv);
  host_add_supg64_g4(S,U0,supgTauScale,supgMagic,hostA,hostConv);
  auto initRau=host_finalize_relax_g4(S,hostA,alphaU,&initDelta);S.pressure.rAU=initRau;
  auto FH=build_h2_fine_csr_host(S);
  int effectiveAmgPowerIts=0;
  if(amgSpectrumPolicy=="always") effectiveAmgPowerIts=amgPowerIts;
  else if(amgSpectrumPolicy=="auto" && amgSmoother=="cheb2") effectiveAmgPowerIts=amgPowerIts;
  if(amgSmoother=="cheb2" && effectiveAmgPowerIts<=0)throw std::runtime_error("H8 Chebyshev AMG smoother requires spectrum estimation");
  SAHierarchyHost H;
  if(amgHierarchy=="sa")H=build_sa_hierarchy(M,S.pressure,16,6,18,amgTerminal,8,effectiveAmgPowerIts,amgLambdaSafety,amgLambdaLowFraction,4.0/3.0);
  else H=build_cf_hierarchy(S,FH,cfTheta,cfPmax,cfInterp,cfAggressiveFirst!=0,amgTerminal,effectiveAmgPowerIts,amgLambdaSafety,amgLambdaLowFraction);
  std::printf("NODALS_GPU_G8_CF_CONFIG tag=%s hierarchy=%s coarsening=%s strength=%s theta=%.6f interp=%s pmax=%d aggressiveFirst=%d terminal=%d spectrumPolicy=%s effectivePowerIts=%d status=PASS\n",tag.c_str(),amgHierarchy.c_str(),cfCoarsening.c_str(),cfStrength.c_str(),cfTheta,cfInterp.c_str(),cfPmax,cfAggressiveFirst,amgTerminal,amgSpectrumPolicy.c_str(),effectiveAmgPowerIts);

  std::printf("NODALS_GPU_H8_CONFIG tag=%s precision=%s device=%s cc=%d.%d petsc=NONE mpi=NONE cells=%zu runMode=%s momentumWork=%s momentumResidualPolicy=%s physicalOperator=exact_current_CSR fineAMG=explicit_%s_CSR_warp coarseAMGSpMV=per_level_scalar_vs_warp_hybrid fineCsrNumericRefreshEvery=1 refreshKernel=warp_per_row spectrumRefresh=setup_only coarseHierarchyNumeric=setup_snapshot coarseSpMV=PER_LEVEL_SCALAR_WARP_HYBRID momentumDiffusion=PERSISTENT_NUMERIC_CSR momentumConvection=NUMERIC_ONLY_SHARED_XYZ momentumAssemblyExec=SETUP_SELECT_SCALAR_VS_WARP BGeometry=SETUP_PRECOMPUTED_24_%s_PER_CELL_AUTOSELECT alphaU=%.8g alphaP=%.8g simpleTol=%.3e maxOuter=%d momentumOmega=%.8g pressureRtol=%.3e pressureAtol=%.3e pressureMaxIts=%d reductions=FP64 state=%s operator=%s amg=%s\n",
    tag.c_str(),kPrecisionName,prop.name,prop.major,prop.minor,M.tets.size(),runMode.c_str(),momentumWork.c_str(),runMode=="fixed10"?"NONE":"CONVERGENCE_ONLY_PRE_SWEEP",kPrecisionName,kPrecisionName,alphaU,alphaP,simpleTol,maxOuter,momOmega,pRtol,pAtol,pMax,kPrecisionName,kPrecisionName,kPrecisionName);
  std::printf("NODALS_GPU_H8_TUNING alphaU=%.8g alphaP=%.8g momentumWork=%s momentumOmega=%.8g momentumAdaptiveTol=DISABLED momentumResidualPolicy=%s pRtol=%.3e pAtol=%.3e pMax=%d simpleTol=%.3e maxOuter=%d snapshotTol=%.3e fineCsrRefreshEvery=1 status=PASS\n",alphaU,alphaP,momentumWork.c_str(),momOmega,runMode=="fixed10"?"NONE":"CONVERGENCE_ONLY_PRE_SWEEP",pRtol,pAtol,pMax,simpleTol,maxOuter,snapshotTol);
  std::printf("NODALS_GPU_H8_FINE_CSR_TOPOLOGY tag=%s cells=%zu nnz=%zu rowMean=%.6f rowMax=%u directedContrib=%llu csrMiB=%.3f compactRefreshMetadataMiB=%.3f totalFineCsrMiB=%.3f bytesPerCell=%.3f status=PASS\n",tag.c_str(),M.tets.size(),FH.col.size(),FH.rowMean,FH.rowMax,(unsigned long long)FH.directedContrib,g5_mib(FH.csr_bytes()),g5_mib(FH.metadata_bytes()),g5_mib(FH.bytes()),(double)FH.bytes()/std::max<std::size_t>(M.tets.size(),1));
  std::printf("NODALS_GPU_H8_SETUP tag=%s cells=%zu freeVel=%d momentumNnz=%zu colors=%d hierarchyLevels=%zu terminal=%d transfer0Nnz=%zu cellPlanHostBytes=%zu cellPlanDeviceBytes=%zu baselineUsedMiB=%.3f totalMiB=%.3f status=PASS\n",tag.c_str(),M.tets.size(),S.topo.n,S.topo.val.size(),S.coloring.ncolors,H.csr.size(),H.terminal_n,H.P.empty()?0:H.P[0].val.size(),sizeof(G4CellPlanHost),sizeof(G4CellPlanDevice),g5_used_mib(memBaseline),(double)memBaseline.total_bytes/(1024.0*1024.0));
  std::printf("NODALS_GPU_H8_MOMENTUM_ASSEMBLY_DESIGN tag=%s diffusionCSR=BUILT_ONCE_HOST_UPLOADED_ONCE_DEVICE_PERSISTENT convectionCSRTopology=STATIC convectionNumeric=REFRESH_EACH_OUTER commonOperatorXYZ=YES pressureGradientBTopology=STATIC pressureGradientAction=APPLY_CURRENT_P fixedDiffusionDirichletRHS=PERSISTENT variableViscosityDesign=REFRESH_DIFFUSION_NUMERICS_ONLY_NO_TOPOLOGY_REBUILD status=PASS\n",tag.c_str());

  G4Gpu G=upload_all(S,H,initRau,FH);h8_fill_device_h2(G,S);
  G.pf.cells=G.cells.data();G.pf.bcoeff=G.bcoeff.data();G.pf.rau_live=G.rau.data();G.pf.csr_pc=&G.fineCsr;G.pf.physicalUseCurrentCSR=physicalUseCurrentCSR;G.amg.fine=&G.pf;G.amg.smoother=amgSmoother;G.amg.jacobiOmega=amgJacobiOmega;G.amg.chebDegree=amgChebDegree;G.amg.powerIts=effectiveAmgPowerIts;G.amg.lambdaSafety=amgLambdaSafety;G.amg.lambdaLowFraction=amgLambdaLowFraction;
  std::printf("NODALS_GPU_G9_L1_CONFIG tag=%s smoother=%s denominator=%s relaxWeight=%.8g fineL1Refresh=%s coarseL1=SETUP_SNAPSHOT definition=SUM_ABS_ROW status=PASS\n",tag.c_str(),amgSmoother.c_str(),amgSmoother=="l1jacobi"?"L1_ROW_ABS_SUM":"DIAGONAL",amgJacobiOmega,amgSmoother=="l1jacobi"?"EVERY_FINE_CSR_REFRESH":"OFF");
  std::printf("NODALS_GPU_H8_AMG_SMOOTHER tag=%s hierarchy=%s smoother=%s jacobiOmega=%.8g chebDegree=%d powerIts=%d hierarchyConstruction=SELECTED_BY_AMG_HIERARCHY spectrumSetup=SETUP_ONLY status=PASS\n",tag.c_str(),amgHierarchy.c_str(),amgSmoother.c_str(),amgJacobiOmega,G.amg.chebDegree,G.amg.powerIts);
  h7_select_momentum_assembly(G,alphaU,prop.multiProcessorCount,tag.c_str());
  const int supgBlocks=h7_blocks_for_sm_factor(prop.multiProcessorCount,8);
  std::printf("NODALS_GPU_H8_SUPG tag=%s enabled=1 form=%s quadPoints=%d tauScale=%.8g magic=%.8g blocks=%d setupSnapshotIncludesSUPG=1 status=PASS\n",tag.c_str(),supgForm.c_str(),supgQuadPoints,supgTauScale,supgMagic,supgBlocks);

  // The setup-only momentum autotuner leaves rAU for the historical central-only
  // operator. Restore the actual production operator before any pressure-CSR
  // parity/selection work. This is required when the host hierarchy snapshot
  // already includes SUPG.
  h8_assemble_live_supg(G,alphaU,supgBlocks,S.pipe.nu,supgTauScale,supgMagic);
  NODALS_CUDA(cudaDeviceSynchronize());
  std::printf("NODALS_GPU_H8_PRODUCTION_OPERATOR_RESTORE tag=%s supg=1 status=PASS\n",tag.c_str());

  h8_select_precomputed_b(G,tag.c_str());
  double pressureChebLambdaMax=1.0;
  if(pressureSolver=="cheb")pressureChebLambdaMax=pressure_precond_power(G.pf,G.amg,pressurePowerIts,pressurePowerSafety,G.pcg,G.dotScratch,tag.c_str());
  std::printf("NODALS_GPU_G12_PRESSURE_SOLVER tag=%s solver=%s richardsonOmega=%.8g chebDegree=%d powerIts=%d chebLowFraction=%.8g powerSafety=%.8g lambdaMax=%.12e preconditioner=SELECTED_AMG_CONFIGURATION stoppingTolerance=UNCHANGED status=PASS\n",tag.c_str(),pressureSolver.c_str(),pressureRichardsonOmega,pressureChebDegree,pressurePowerIts,pressureChebLowFraction,pressurePowerSafety,pressureChebLambdaMax);
  {
    std::vector<AMGReal> g6FineVal(G.fineCsr.val.size());G.fineCsr.val.download(g6FineVal.data(),g6FineVal.size());
    g6_strength_sweep(FH.n,FH.row,FH.col,g6FineVal,tag.c_str());
    g7_pmis_diagnostics(FH.n,FH.row,FH.col,g6FineVal,0.25,tag.c_str());
  }
  h6_select_coarse_spmv(G,H,tag.c_str(),200);
  if(effectiveAmgPowerIts>0){g5_gpu_spectrum_refresh(G,H);std::printf("NODALS_GPU_G10_CF_SPECTRUM_POLICY tag=%s hierarchy=%s smoother=%s policy=%s hostPower=COMPUTE gpuPower=COMPUTE powerIts=%d status=PASS\n",tag.c_str(),amgHierarchy.c_str(),amgSmoother.c_str(),amgSpectrumPolicy.c_str(),effectiveAmgPowerIts);}
  else{G.amg.fineLambda=1.0;for(auto&v:G.amg.levelLambda)v=1.0;std::printf("NODALS_GPU_G10_CF_SPECTRUM_POLICY tag=%s hierarchy=%s smoother=%s policy=%s hostPower=SKIP gpuPower=SKIP powerIts=0 status=PASS\n",tag.c_str(),amgHierarchy.c_str(),amgSmoother.c_str(),amgSpectrumPolicy.c_str());}
  h2_action_parity_and_bench(G,tag.c_str(),24);
  H0Profiler H0P;g_h0=&H0P;
  NODALS_CUDA(cudaDeviceSynchronize());const auto memUpload=device_memory_info();
  const double baselineUsed=g5_used_mib(memBaseline),uploadUsed=g5_used_mib(memUpload),explicitMiB=g5_mib(G.bytes());
  std::printf("NODALS_GPU_H8_MEMORY tag=%s point=after_upload cells=%zu baselineUsedMiB=%.3f usedMiB=%.3f deltaFromBaselineMiB=%.3f explicitMiB=%.3f explicitBytesPerCell=%.3f status=PASS\n",tag.c_str(),M.tets.size(),baselineUsed,uploadUsed,uploadUsed-baselineUsed,explicitMiB,(double)G.bytes()/std::max<std::size_t>(M.tets.size(),1));

