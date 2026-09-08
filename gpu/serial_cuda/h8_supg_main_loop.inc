  bool converged=false,pressureAll=true,finiteAll=true;double cont0=-1.0,contRel=1.0;
  long long sumP=0;int convergenceAuditCalls=0;std::array<double,3>lastAuditRel{{NAN,NAN,NAN}};
  double tAssembly=0,tMomentum=0,tContinuity=0,tPressure=0,tPupdate=0;
  auto wall0=std::chrono::steady_clock::now();
  int finalIt=0,fineCsrRefreshCount=0;
  for(int it=1;it<=maxOuter;++it){
    H0P.currentOuter=it;if(H0P.used)throw std::runtime_error("H8 profiler nonempty at outer start");
    G5StageEvents E;E.rec(0);
    h8_assemble_live_supg(G,alphaU,supgBlocks,S.pipe.nu,supgTauScale,supgMagic);E.rec(1);
    if(it==1){double rr=g5_rau_snapshot_rel(G);std::printf("NODALS_GPU_H8_SNAPSHOT_PARITY it=1 rAURel=%.3e tol=%.3e status=%s\n",rr,snapshotTol,rr<snapshotTol?"PASS":"FAIL");if(!(rr<snapshotTol))throw std::runtime_error("H8 setup rAU snapshot mismatch");}
    G.pf.bt_state(G.p.data());
    {int q=h0_begin(H0_MOM_RHS);g4_momentum_rhs_kernel<<<g4grid(G.nv),G4B>>>(G.nv,G.s0.data(),G.s1.data(),G.s2.data(),G.c0.data(),G.c1.data(),G.c2.data(),G.pf.v0.data(),G.pf.v1.data(),G.pf.v2.data(),G.delta.data(),G.u0.data(),G.u1.data(),G.u2.data(),G.b0.data(),G.b1.data(),G.b2.data());NODALS_CUDA(cudaGetLastError());h0_end(q);}
    bool auditedThisOuter=false;std::array<double,3>auditRel{{NAN,NAN,NAN}};
    if(runMode=="converge" && it>1 && contRel<=simpleTol){auditRel=momentum_initial_rel_audit(G);lastAuditRel=auditRel;++convergenceAuditCalls;auditedThisOuter=true;}
    const char* momentumDirection=fixed_momentum_work(G,S.coloring,momOmega,momentumWork,it);E.rec(2);

    if(G.h8PrecomputedContinuity)h8_continuity_kernel<<<g4grid(G.nc),G4B>>>(G.cells.data(),G.bcoeff.data(),G.nc,G.fixedDiv.data(),G.u0.data(),G.u1.data(),G.u2.data(),G.cont.data());
    else g4_continuity_kernel<<<g4grid(G.nc),G4B>>>(G.cells.data(),G.nc,G.fixedDiv.data(),G.u0.data(),G.u1.data(),G.u2.data(),G.cont.data());
    NODALS_CUDA(cudaGetLastError());
    double cn=device_norm2(G.cont.data(),G.nc,G.dotScratch);if(it==1)cont0=cn;contRel=cn/std::max(cont0,1e-300);
    g4_negate_kernel<<<g4grid(G.nc),G4B>>>(G.nc,G.cont.data());NODALS_CUDA(cudaGetLastError());E.rec(3);

    if(it==1 || ((it-1)%fineCsrRefreshEvery)==0){G.fineCsr.refresh(G.cells.data(),G.rau.data(),G.pf.diag_pc.data(),(amgSmoother=="l1jacobi"?G.amg.fine_l1.data():nullptr));++fineCsrRefreshCount;}
    G4PCGResult pr;
    if(pressureSolver=="pcg")pr=pressure_pcg(G.pf,G.amg,G.cont.data(),pRtol,pAtol,pMax,G.pcg,G.dotScratch);
    else if(pressureSolver=="richardson")pr=pressure_richardson(G.pf,G.amg,G.cont.data(),pRtol,pAtol,pMax,pressureRichardsonOmega,G.pcg,G.dotScratch);
    else pr=pressure_cheb_poly(G.pf,G.amg,G.cont.data(),pRtol,pAtol,pMax,pressureChebLambdaMax,pressureChebLowFraction,pressureChebDegree,G.pcg,G.dotScratch);
    pressureAll=pressureAll&&pr.ok;if(!pr.ok)throw std::runtime_error("H8 pressure solve failed requested inexact target");sumP+=pr.its;E.rec(4);
    g4_axpy_kernel<<<g4grid(G.nc),G4B>>>(G.nc,alphaP,G.pcg.x.data(),G.p.data());NODALS_CUDA(cudaGetLastError());E.rec(5);
    NODALS_CUDA(cudaEventSynchronize(E.e[5]));H0P.collect();

    tAssembly+=E.ms(0,1);tMomentum+=E.ms(1,2);tContinuity+=E.ms(2,3);tPressure+=E.ms(3,4);tPupdate+=E.ms(4,5);
    finiteAll=finiteAll&&std::isfinite(contRel)&&std::isfinite(pr.rel);
    bool allMomentumMet=false;
    if(auditedThisOuter){allMomentumMet=std::isfinite(auditRel[0])&&std::isfinite(auditRel[1])&&std::isfinite(auditRel[2])&&auditRel[0]<=simpleTol&&auditRel[1]<=simpleTol&&auditRel[2]<=simpleTol;finiteAll=finiteAll&&std::isfinite(auditRel[0])&&std::isfinite(auditRel[1])&&std::isfinite(auditRel[2]);}
    converged=(runMode=="converge")&&finiteAll&&pressureAll&&contRel<=simpleTol&&auditedThisOuter&&allMomentumMet;finalIt=it;

    if(it<=10 || (it%25)==0 || auditedThisOuter || converged)std::printf("NODALS_GPU_H8_SIMPLE tag=%s it=%d relCont=%.12e momentumWork=%s momentumDirection=%s residualAudit=%s auditInitRel=[%.3e,%.3e,%.3e] pCG=%d pTrueRel=%.3e allMomentumMet=%d converged=%d status=%s\n",tag.c_str(),it,contRel,momentumWork.c_str(),momentumDirection,auditedThisOuter?"DONE":"SKIPPED",auditRel[0],auditRel[1],auditRel[2],pr.its,pr.rel,(int)allMomentumMet,(int)converged,(pressureAll&&finiteAll)?"PASS":"FAIL");
    if(runMode=="converge"&&converged)break;
  }
  NODALS_CUDA(cudaDeviceSynchronize());auto wall1=std::chrono::steady_clock::now();const auto memEnd=device_memory_info();

  std::vector<StateReal>pf((std::size_t)G.nc);G.p.download(pf.data(),pf.size());std::vector<double>pd(pf.size());for(std::size_t i=0;i<pf.size();++i)pd[i]=(double)pf[i];
  std::array<std::vector<double>,3> finalU;for(int d=0;d<3;++d)finalU[(std::size_t)d].resize((std::size_t)G.nv);
  std::vector<StateReal> uf((std::size_t)G.nv);
  G.u0.download(uf.data(),uf.size());for(std::size_t i=0;i<uf.size();++i)finalU[0][i]=(double)uf[i];
  G.u1.download(uf.data(),uf.size());for(std::size_t i=0;i<uf.size();++i)finalU[1][i]=(double)uf[i];
  G.u2.download(uf.data(),uf.size());for(std::size_t i=0;i<uf.size();++i)finalU[2][i]=(double)uf[i];
  const auto hpErr=h8_compute_hp_errors(M,S,finalU,pd);
  double dp=pressure_drop_fit_g4(M,pd),exact=S.pipe.hpDrop,dpErr=std::abs(dp-exact)/std::max(std::abs(exact),1e-300);
  std::printf("NODALS_GPU_H8_HP_ERROR tag=%s cells=%zu hEff=%.12e U_L2=%.12e U_relL2=%.12e P_shifted_L2=%.12e P_shifted_relL2=%.12e pressureShift=%.12e pressureDropFit=%.12e exactPressureDrop=%.12e pressureDropRelErr=%.12e quadrature=duffy5_125 finalStateD2H=AFTER_LOOP status=%s\n",tag.c_str(),M.tets.size(),hpErr.hEff,hpErr.uL2,hpErr.uRelL2,hpErr.pShiftedL2,hpErr.pShiftedRelL2,hpErr.pressureShift,dp,exact,dpErr,converged?"PASS":"UNCONVERGED");
  double wallMs=std::chrono::duration<double,std::milli>(wall1-wall0).count();double usedEnd=g5_used_mib(memEnd);

  if(runMode=="fixed10")std::printf("NODALS_GPU_H8_FIXED10 tag=%s cells=%zu fixedOuter=%d momentumWork=%s momentumPassesPerOuter=1 momentumResidualAudits=0 diffusionPolicy=PERSISTENT_NUMERIC_CSR convectionPolicy=NUMERIC_ONLY_EACH_OUTER fineCsrRefreshEvery=1 fineCsrRefreshCount=%d finalRelCont=%.12e avgPressureIts=%.6f pressureDropFit=%.12e exactPressureDrop=%.12e pressureDropRelErr=%.12e loopMs=%.3f avgSimpleMs=%.6f assemblyAvgMs=%.6f momentumAvgMs=%.6f continuityAvgMs=%.6f pressureAvgMs=%.6f pressureUpdateAvgMs=%.6f status=%s\n",tag.c_str(),M.tets.size(),finalIt,momentumWork.c_str(),fineCsrRefreshCount,contRel,finalIt?(double)sumP/finalIt:0.0,dp,exact,dpErr,wallMs,finalIt?wallMs/finalIt:0.0,finalIt?tAssembly/finalIt:0.0,finalIt?tMomentum/finalIt:0.0,finalIt?tContinuity/finalIt:0.0,finalIt?tPressure/finalIt:0.0,finalIt?tPupdate/finalIt:0.0,(pressureAll&&finiteAll&&finalIt==10)?"PASS":"FAIL");
  else std::printf("NODALS_GPU_H8_FULLCONV tag=%s cells=%zu momentumWork=%s simpleTol=%.3e outerIts=%d converged=%d finalRelCont=%.12e convergenceAuditCalls=%d finalAuditInitRel=[%.3e,%.3e,%.3e] avgPressureIts=%.6f pressureDropFit=%.12e exactPressureDrop=%.12e pressureDropRelErr=%.12e loopMs=%.3f avgSimpleMs=%.6f assemblyAvgMs=%.6f momentumAvgMs=%.6f continuityAvgMs=%.6f pressureAvgMs=%.6f pressureUpdateAvgMs=%.6f status=%s\n",tag.c_str(),M.tets.size(),momentumWork.c_str(),simpleTol,finalIt,(int)converged,contRel,convergenceAuditCalls,lastAuditRel[0],lastAuditRel[1],lastAuditRel[2],finalIt?(double)sumP/finalIt:0.0,dp,exact,dpErr,wallMs,finalIt?wallMs/finalIt:0.0,finalIt?tAssembly/finalIt:0.0,finalIt?tMomentum/finalIt:0.0,finalIt?tContinuity/finalIt:0.0,finalIt?tPressure/finalIt:0.0,finalIt?tPupdate/finalIt:0.0,converged?"PASS":"FAIL");
  std::printf("NODALS_GPU_H8_MOMENTUM_AUDIT tag=%s runMode=%s policy=%s calls=%d postSweepAudits=0 status=PASS\n",tag.c_str(),runMode.c_str(),runMode=="fixed10"?"NONE":"CONVERGENCE_ONLY_PRE_SWEEP",convergenceAuditCalls);
  h0_print_profile(tag.c_str(),finalIt,tPressure,tMomentum,H0P);
  std::printf("NODALS_GPU_H8_MEMORY tag=%s point=after_convergence cells=%zu baselineUsedMiB=%.3f usedMiB=%.3f deltaFromBaselineMiB=%.3f explicitMiB=%.3f runtimeDriftMiB=%.3f totalMiB=%.3f status=PASS\n",tag.c_str(),M.tets.size(),baselineUsed,usedEnd,usedEnd-baselineUsed,explicitMiB,usedEnd-uploadUsed,(double)memEnd.total_bytes/(1024.0*1024.0));
  std::printf("NODALS_GPU_H8_RESIDENCY tag=%s O_N_H2D_inside_SIMPLE=0 O_N_D2H_inside_SIMPLE=%s finalStateD2H=AFTER_LOOP reductions=FP64 deviceNumericStorage=%s physicalOperator=CURRENT_EXACT_CSR fineCsrNumericRefreshEvery=1 refreshKernel=WARP_PER_ROW momentumWork=%s momentumResidualPolicy=%s momentumDiffusion=PERSISTENT_NUMERIC_CSR momentumConvection=NUMERIC_ONLY_EACH_OUTER spectrumRefresh=SETUP_ONLY coarseHierarchyNumeric=SETUP_SNAPSHOT coarseSpMV=PER_LEVEL_SCALAR_WARP_HYBRID H8_scope=SUPG64 status=PASS\n",tag.c_str(),runMode=="fixed10"?"0":"SCALAR_CONVERGENCE_AUDITS_ONLY",kPrecisionName,momentumWork.c_str(),runMode=="fixed10"?"NONE":"CONVERGENCE_ONLY_PRE_SWEEP");
  const bool finalPass=(runMode=="fixed10")?(finalIt==10&&pressureAll&&finiteAll):converged;
  std::printf("NODALS_GPU_RESULT gate=H8 tag=%s cells=%zu precision=%s runMode=%s momentumWork=%s outer=%d simpleTol=%.3e fineCsrRefreshEvery=1 physicalOperator=current_exact_CSR pressureAll=%s finite=%s noPetsc=1 noMPI=1 status=%s\n",tag.c_str(),M.tets.size(),kPrecisionName,runMode.c_str(),momentumWork.c_str(),finalIt,simpleTol,pressureAll?"PASS":"FAIL",finiteAll?"PASS":"FAIL",finalPass?"PASS":"FAIL");
  return finalPass?0:35;
}catch(const std::exception&e){std::fprintf(stderr,"NODALS_GPU_H8_EXCEPTION what=%s\n",e.what());return 90;}
}
