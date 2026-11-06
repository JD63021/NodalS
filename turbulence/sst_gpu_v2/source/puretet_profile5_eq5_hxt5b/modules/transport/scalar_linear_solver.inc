struct LinearResult{int works=0;double r0=0,rn=0,rel=0;bool ok=false;};
inline double scalar_residual(hxt4b::GpuCSR&A,const Dev<unsigned char>&fixed,const Real*rhs,const Real*x,hxt4a::Reducer<Real>&red){A.spmv(x,A.tmp.p);hxt4b::residual_free_kernel<<<(A.n+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(A.n,fixed.p,rhs,A.tmp.p,A.res.p);HXT1_CUDA(cudaGetLastError());return red.norm(A.res.p);}
inline LinearResult scalar_solve(
    hxt4b::GpuCSR&A,
    const hxt4b::ColoringHost&C,
    const Dev<unsigned char>&fixed,
    const Real*rhs,
    Real*x,
    hxt4a::Reducer<Real>&red,
    const Options&O,
    int nonlinearIt,
    const char*label="sst")
{
  // Both G1 and G2 call this routine with equation label "k" or "omega".
  // Apply equation-specific thresholds to the ACTUAL scalar Krylov/GS solve,
  // not just to the run-script log. Other labels retain common settings.
  const bool isK = label && label[0]=='k' && label[1]=='\0';
  const bool isOmega = label && label[0]=='o' && label[1]=='m';
  const double rtol = isK && O.kLinearRtol>=0.0 ? O.kLinearRtol :
                      isOmega && O.omegaLinearRtol>=0.0 ? O.omegaLinearRtol : O.linearRtol;
  const double atol = isK && O.kLinearAtol>=0.0 ? O.kLinearAtol :
                      isOmega && O.omegaLinearAtol>=0.0 ? O.omegaLinearAtol : O.linearAtol;
  const std::string& stage5aOverride=isK?O.s05aKMode:(isOmega?O.s05aOmegaMode:O.linearSolver);
  const std::string stage5aMode=stage5aOverride=="inherit"?O.linearSolver:stage5aOverride;
  if(stage5aMode=="fgs-fixed" || stage5aMode=="fgs-guarded"){
    // Run a FIXED number of existing colored-GS sweeps; no residual SpMV,
    // norm reduction or synchronization inside the sweep loop. The old
    // FGMRES and fgs1/sgs1 paths below remain byte-for-byte unchanged.
    const int sweeps=isK?O.s05aKSweeps:O.s05aOmegaSweeps;
    const bool audited=O.s05aCheckEvery>0 &&
      (nonlinearIt%O.s05aCheckEvery==0);
    if(stage5aMode=="fgs-guarded" && O.s05aCheckEvery<=0)
      throw std::runtime_error("Stage5.A guarded FGS requires residual checks");
    LinearResult R;
    R.works=sweeps;
    const double nan=std::numeric_limits<double>::quiet_NaN();
    R.r0=nan;R.rn=nan;R.rel=nan;R.ok=false; // UNVERIFIED is never 'ok'.
    if(audited){
      R.r0=scalar_residual(A,fixed,rhs,x,red);
      if(!std::isfinite(R.r0))throw std::runtime_error("Stage5.A nonfinite initial scalar true residual");
    }
    for(int j=0;j<sweeps;++j)
      hxt5b::sweep(A,C,A.val.p,rhs,x,O.gsOmega,true,fixed.p);
    HXT1_CUDA(cudaGetLastError());
    bool fallback=false;
    if(audited){
      R.rn=scalar_residual(A,fixed,rhs,x,red);
      if(!std::isfinite(R.rn))throw std::runtime_error("Stage5.A nonfinite final scalar true residual");
      R.rel=R.r0>0.0 ? R.rn/R.r0 : (R.rn==0.0?0.0:std::numeric_limits<double>::infinity());
      R.ok=R.rn<=std::max(atol,rtol*R.r0);
      if(!R.ok && stage5aMode=="fgs-guarded"){
        // Reuse robust frozen FGMRES as a *selectable safeguard*. No changes
        // to the matrix, scalar boundary conditions or BF3 momentum solve.
        auto Q=hxt5e::fgmres(label,A,C,A.val.p,rhs,x,fixed.p,
          rtol,atol,O.linearMax,O.fgmresRestart,O.fgsSweeps,O.gsOmega,red);
        R.works+=Q.its;R.rn=Q.final;
        R.rel=R.r0>0.0?R.rn/R.r0:0.0;
        R.ok=R.rn<=std::max(atol,rtol*R.r0);
        fallback=true;
      }
    }
    if(nonlinearIt<=3 || nonlinearIt%10==0 || fallback){
      std::printf("STAGE27_S05A_SCALAR it=%d equation=%s method=%s "
        "sweeps=%d innerAudit=%s trueRel=%.6e acceptable=%d fallbackFGMRES=%d status=PASS\n",
        nonlinearIt,label,stage5aMode.c_str(),sweeps,
        audited?"MEASURED":"SKIPPED",R.rel,(int)R.ok,(int)fallback);
    }
    return R;
  }
  if(stage5aMode=="fgmres-fgs"){
    auto Q=hxt5e::fgmres(
      label,A,C,A.val.p,rhs,x,fixed.p,
      rtol,atol,
      O.linearMax,O.fgmresRestart,
      O.fgsSweeps,O.gsOmega,
      red);

    LinearResult R;
    R.works=Q.its;
    R.r0=Q.initial;
    R.rn=Q.final;
    R.rel=Q.rel;
    R.ok=Q.ok;
    return R;
  }

  const char*workMode=nullptr;

  if(stage5aMode=="sgs1")
    workMode="sgs1";
  else if(stage5aMode=="fgs1")
    workMode="fgs1";
  else
    throw std::runtime_error("bad SST scalar linear solver");

  LinearResult R;

  R.r0=scalar_residual(
    A,fixed,rhs,x,red);

  R.rn=R.r0;

  if(R.r0==0){
    R.ok=true;
    return R;
  }

  const double target=
    std::max(atol,rtol*R.r0);

  for(int j=0;j<O.linearMax;++j){
    hxt5b::work(
      workMode,
      nonlinearIt+j,
      A,C,A.val.p,rhs,x,
      O.gsOmega,fixed.p);

    R.works=j+1;

    R.rn=scalar_residual(
      A,fixed,rhs,x,red);

    if(!std::isfinite(R.rn))
      break;

    if(R.rn<=target){
      R.ok=true;
      break;
    }
  }

  R.rel=
    R.rn/std::max(R.r0,1e-300);

  return R;
}

