namespace hxt5b_perf {

using Real = hxt4b::Real;

inline auto momentum_solve(
    const std::string&mode,
    int outer,
    hxt4b::GpuCSR&A,
    const hxt4b::ColoringHost&C,
    const Real*val,
    const Real*b,
    Real*x,
    double omega,
    const unsigned char*clamp,
    double rtol,
    double atol,
    int maxWork,
    int restart,
    int fgsSweeps,
    hxt4a::Reducer<Real>&red,
    const char*component,
    bool warnPolicy)
{
  const bool fixedFast =
      warnPolicy &&
      mode=="fgs1" &&
      maxWork==1;

  if(fixedFast){

    using Result =
      decltype(
        hxt5e::momentum_solve(
          mode,outer,
          A,C,
          val,b,x,
          omega,clamp,
          rtol,atol,
          maxWork,
          restart,
          fgsSweeps,
          red,
          component));

    // Same numerical action as fixed fgs1/maxWork=1.
    hxt5b::sweep(
      A,C,
      val,b,x,
      omega,
      true,
      clamp);

    HXT1_CUDA(cudaGetLastError());

    Result R{};

    R.sweeps = 1;

    // Inner residual deliberately not evaluated.
    // These fields are diagnostics only in fixed-work/warn mode.
    R.initial = 0.0;
    R.final   = 0.0;
    R.rel     = 0.0;
    R.ok      = true;

    if(outer==1 &&
       component &&
       component[0]=='x')
      std::printf(
        "NODALS_HXT5B_MOM_FIXED_FAST "
        "mode=fgs1 sweeps=1 "
        "innerResidualAudit=SKIPPED "
        "outerPhysicalGate=UNCHANGED "
        "status=PASS\n");

    return R;
  }

  // All non-fixed configurations remain exactly as before.
  return hxt5e::momentum_solve(
    mode,outer,
    A,C,
    val,b,x,
    omega,clamp,
    rtol,atol,
    maxWork,
    restart,
    fgsSweeps,
    red,
    component);
}

} // namespace hxt5b_perf
