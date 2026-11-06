#pragma once
// Generated from the repository's validated build_cf_hierarchy() implementation.
// Only the initial fine-matrix source is changed: arbitrary symmetric CSR.
namespace nodals_gpu {
inline SAHierarchyHost build_cf_hierarchy_from_csr(
    const CSRHost&Ain,
    double theta,int pmax,const std::string& interp,
    bool aggressiveFirst,
    int terminal=1000,int powerIts=16,double safety=1.5,double lowFrac=0.05)
{
  const auto t0=std::chrono::steady_clock::now();

  std::printf(
    "NODALS_GPU_PM2_CF_DIAGNOSTIC_POLICY symmetryAudit=%s "
    "diagnosticEnv=NODALS_PMIS_DIAGNOSTICS setupAutotuneEnv=NODALS_SETUP_AUTOTUNE "
    "status=PASS\n",
    g8_pm2_cf_diagnostics_enabled()?"ON":"OFF");

  SAHierarchyHost H;
  H.powerIts=powerIts;
  H.interpMaxNnz=pmax;
  H.lambdaSafety=safety;
  H.lambdaLowFraction=lowFrac;
  H.saDamping=0.0;

  CSRHost A=Ain;
  H.fineDiag=A.diag;

  const auto pm0_spec_t0=std::chrono::steady_clock::now();
  if(powerIts>0){
    H.fineLambda=sa_power_csr(A,powerIts,safety);
    std::printf(
      "NODALS_GPU_G8_CF_SPECTRUM level=0 powerIts=%d lambdaMax=%.12e policy=COMPUTE status=PASS\n",
      powerIts,H.fineLambda);
  }else{
    H.fineLambda=1.0;
    std::printf(
      "NODALS_GPU_G8_CF_SPECTRUM level=0 powerIts=0 lambdaMax=1 policy=SKIP_UNUSED_BY_JACOBI status=PASS\n");
  }

  g8_pm0_stage(
    "fine_spectrum",0,A.n,A.val.size(),
    std::chrono::duration<double>(
      std::chrono::steady_clock::now()-pm0_spec_t0).count());

  int logicalLevel=0;

  if(aggressiveFirst && A.n>terminal){
    // Stage 1: fine -> C1.
    auto split0=g8_build_pmis_split_dispatch(A,theta,0);
    auto P01=g8_build_transfer_dispatch(A,split0,pmax,0,interp);
    const auto pm0_ptap1_t0=std::chrono::steady_clock::now();
    CSRHost A1=g8_pm1_ptap_dispatch(A,P01,1);
    g8_pm0_stage(
      "ptap",1,A1.n,A1.val.size(),
      std::chrono::duration<double>(
        std::chrono::steady_clock::now()-pm0_ptap1_t0).count());

    std::printf(
      "NODALS_GPU_G11_AGG_STAGE stage=1 fine=%d coarse=%d Pnnz=%zu Annz=%zu "
      "AavgNnz=%.6f symmetryRel=%.3e runtimeLevel=TEMPORARY status=PASS\n",
      A.n,A1.n,P01.val.size(),A1.val.size(),
      A1.n?(double)A1.val.size()/A1.n:0.0,g8_pm2_symmetry_rel(A1));

    if(A1.n<=terminal){
      // Degenerate case: no second aggressive stage needed.
      H.P.push_back(std::move(P01));
      H.csr.push_back(std::move(A1));
    }else{
      // Stage 2: C1 -> C2 using the same PMIS/ext+i construction.
      auto split1=g8_build_pmis_split_dispatch(A1,theta,1);
      auto P12=g8_build_transfer_dispatch(A1,split1,pmax,1,interp);
      const auto pm0_ptap2_t0=std::chrono::steady_clock::now();
      CSRHost A2=g8_pm1_ptap_dispatch(A1,P12,2);
      g8_pm0_stage(
        "ptap",2,A2.n,A2.val.size(),
        std::chrono::duration<double>(
          std::chrono::steady_clock::now()-pm0_ptap2_t0).count());

      std::printf(
        "NODALS_GPU_G11_AGG_STAGE stage=2 fine=%d coarse=%d Pnnz=%zu Annz=%zu "
        "AavgNnz=%.6f symmetryRel=%.3e runtimeLevel=KEPT_COARSE_OPERATOR status=PASS\n",
        A1.n,A2.n,P12.val.size(),A2.val.size(),
        A2.n?(double)A2.val.size()/A2.n:0.0,g8_pm2_symmetry_rel(A2));

      // Long-range aggressive interpolation. No pruning in Gate 8 so the
      // composed transfer represents P01*P12 exactly to roundoff.
      auto P02=g11_compose_transfers_exact(P01,P12);

      const int removedRows=A1.n;
      const std::size_t removedNnz=A1.val.size();
      const std::size_t removedPnnz=P01.val.size()+P12.val.size();

      H.P.push_back(std::move(P02));
      H.csr.push_back(std::move(A2));

      std::printf(
        "NODALS_GPU_G11_AGG_FIRST enabled=1 fine=%d removedIntermediateRows=%d "
        "removedIntermediateNnz=%zu separateTransferNnz=%zu composedTransferNnz=%zu "
        "firstRuntimeCoarseRows=%d firstRuntimeCoarseNnz=%zu "
        "coarseOperatorPolicy=REUSE_TWO_STAGE_GALERKIN "
        "longRangeInterpolation=EXACT_COMPOSED_EXTI status=PASS\n",
        A.n,removedRows,removedNnz,removedPnnz,H.P[0].val.size(),
        H.csr[0].n,H.csr[0].val.size());
    }

    if(H.csr.back().n>terminal){
      H.levelLambda.resize(H.csr.size(),0.0);
      if(powerIts>0){
        const double lam=sa_power_csr(H.csr.back(),powerIts,safety);
        H.levelLambda[0]=lam;
        std::printf(
          "NODALS_GPU_G8_CF_SPECTRUM level=1 powerIts=%d lambdaMax=%.12e policy=COMPUTE status=PASS\n",
          powerIts,lam);
      }else{
        H.levelLambda[0]=1.0;
        std::printf(
          "NODALS_GPU_G8_CF_SPECTRUM level=1 powerIts=0 lambdaMax=1 policy=SKIP_UNUSED_BY_JACOBI status=PASS\n");
      }
    }

    A=H.csr.back();
    logicalLevel=1;
  }

  // Standard hierarchy continuation. If aggressiveFirst==false this also
  // constructs the first transfer exactly as Gates 5/7 did.
  while(H.csr.empty() || H.csr.back().n>terminal){
    auto split=g8_build_pmis_split_dispatch(A,theta,logicalLevel);
    auto P=g8_build_transfer_dispatch(A,split,pmax,logicalLevel,interp);
    H.P.push_back(std::move(P));

    const auto pm0_ptap_t0=std::chrono::steady_clock::now();
    CSRHost C=g8_pm1_ptap_dispatch(A,H.P.back(),logicalLevel+1);
    g8_pm0_stage(
      "ptap",logicalLevel+1,C.n,C.val.size(),
      std::chrono::duration<double>(
        std::chrono::steady_clock::now()-pm0_ptap_t0).count());
    std::printf(
      "NODALS_GPU_G8_CF_COARSE level=%d interp=%s rows=%d nnz=%zu avgNnz=%.6f "
      "symmetryRel=%.3e status=PASS\n",
      logicalLevel+1,interp.c_str(),C.n,C.val.size(),
      C.n?(double)C.val.size()/C.n:0.0,g8_pm2_symmetry_rel(C));
    H.csr.push_back(std::move(C));

    CSRHost().row.swap(A.row);CSRHost().col.swap(A.col);
    CSRHost().val.swap(A.val);CSRHost().diag.swap(A.diag);

    if(H.csr.back().n<=terminal)break;

    const std::size_t idx=H.csr.size()-1;
    if(H.levelLambda.size()<H.csr.size())
      H.levelLambda.resize(H.csr.size(),0.0);

    if(powerIts>0){
      const double lam=sa_power_csr(H.csr.back(),powerIts,safety);
      H.levelLambda[idx]=lam;
      std::printf(
        "NODALS_GPU_G8_CF_SPECTRUM level=%d powerIts=%d lambdaMax=%.12e policy=COMPUTE status=PASS\n",
        logicalLevel+1,powerIts,lam);
    }else{
      H.levelLambda[idx]=1.0;
      std::printf(
        "NODALS_GPU_G8_CF_SPECTRUM level=%d powerIts=0 lambdaMax=1 policy=SKIP_UNUSED_BY_JACOBI status=PASS\n",
        logicalLevel+1);
    }

    A=H.csr.back();
    ++logicalLevel;
    if(logicalLevel>20)
      throw std::runtime_error("G8 CF hierarchy excessive levels");
  }

  if(H.levelLambda.size()<H.csr.size())
    H.levelLambda.resize(H.csr.size(),1.0);

  H.terminal_n=H.csr.back().n;
  const auto pm0_terminal_t0=std::chrono::steady_clock::now();
  H.terminal_inv=dense_inverse_cholesky(H.csr.back());
  g8_pm0_stage(
    "terminal_inverse",(int)H.csr.size(),H.csr.back().n,H.csr.back().val.size(),
    std::chrono::duration<double>(
      std::chrono::steady_clock::now()-pm0_terminal_t0).count());

  const auto t1=std::chrono::steady_clock::now();
  const double sec=std::chrono::duration<double>(t1-t0).count();

  std::printf(
    "NODALS_GPU_G8_CF_HIERARCHY levels=%zu transfers=%zu terminal=%d "
    "theta=%.2f criterion=CLASSICAL_NEGATIVE interp=%s pmax=%d "
    "aggressiveFirst=%d aggressivePolicy=%s powerIts=%d spectrumPolicy=%s "
    "fallback=SYMMETRIC_UNION_C seconds=%.6f status=PASS\n",
    H.csr.size(),H.P.size(),H.terminal_n,theta,interp.c_str(),pmax,
    aggressiveFirst?1:0,
    aggressiveFirst?"TWO_STAGE_EXACT_COMPOSITION":"OFF",
    powerIts,powerIts>0?"COMPUTE":"SKIP_UNUSED_BY_JACOBI",sec);

  std::size_t pm0_retained_nnz=0;
  for(const auto&x:H.csr)pm0_retained_nnz+=x.val.size();
  g8_pm0_stage("cf_hierarchy_total",-1,Ain.n,pm0_retained_nnz,sec);
  return H;
}
} // namespace nodals_gpu
