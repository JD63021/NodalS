  // Omega-only true DG branch: keep the accepted G2 k/P1+BF3 momentum
  // path, but capture the ORIGINAL adaptive Spalding trace-log omega targets
  // and solve the distinct 4*ntet DG system, never the CG omega system.
  hxt5c::LinearResult lw{};
  bool omegaDiagNow=false;
  std::vector<Real> omegaDiagRaw;
  HXT1_CUDA(cudaMemset(S.clipW.p,0,sizeof(unsigned long long)));
  if(S.dgOmega){
    S.dgOmega->prepare_wall_capture();
    hxt5c::assemble_equation(hxt5c::OMEGA_EQ,M,BH,inH,inT,wallH,
       pts,hc,tc,hv,tv,tg,ip,A,u0,u1,u2,
       S.k.p,S.oldW.p,vol,dg,wall,S.rhs,S.wallStats,bad,S.diagAudit,
       S.omegaIn,bulk,radius,nu,gamma,wallSampleFraction,O,outer,ifCache,calibratedOmega);
    S.dgOmega->finish_wall_capture();
    S.dgOmega->advance(outer,tc,tv,tg,O.tetWallPlans,
       u0,u1,u2,S.k.p,O,radius,nu,S.omega.p,S.clipW.p);
    lw.works=2;
    lw.r0=lw.rn=lw.rel=std::numeric_limits<double>::quiet_NaN();
    lw.ok=false; // two fixed FGS sweeps: inner residual is NOT measured
  }else{
    hxt5c::assemble_equation(hxt5c::OMEGA_EQ,M,BH,inH,inT,wallH,
       pts,hc,tc,hv,tv,tg,ip,A,u0,u1,u2,
       S.k.p,S.oldW.p,vol,dg,wall,S.rhs,S.wallStats,bad,S.diagAudit,
       S.omegaIn,bulk,radius,nu,gamma,wallSampleFraction,O,outer,ifCache,calibratedOmega);
    lw=hxt5c::scalar_solve(A,C,scalarFixed,S.rhs.p,S.omega.p,red,O,outer,"omega");
    if(q1Transport&&A.n>S.nvert)
       hxt5c::zero_enrichment_kernel<<<(A.n-S.nvert+hxt4b::TPB-1)/hxt4b::TPB,
                                     hxt4b::TPB>>>(S.nvert,A.n,S.omega.p);
    omegaDiagNow=(!compactQ1 && M.h.nhex==0 && M.h.ntet>0 &&
                  q1Transport && puretet_omega_diag::scheduled(outer));
    if(omegaDiagNow){
      omegaDiagRaw.resize((std::size_t)S.nvert);
      HXT1_CUDA(cudaMemcpy(omegaDiagRaw.data(),S.omega.p,
            (std::size_t)S.nvert*sizeof(Real),cudaMemcpyDeviceToHost));
    }
    hxt5c::floor_vertex_kernel<<<(M.h.nv+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(
      (int)M.h.nv,(Real)O.omegaFloor,S.omega.p,S.clipW.p);
    if(bernsteinBF3)
      bf3_bernstein_apply(S,M,S.omega.p,S.oldW.p,(Real)O.omegaFloor,outer,"omega");
  }
  HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
  if(omegaDiagNow){
    puretet_omega_diag::report(outer,M,A,S,vol,dg,wall,O,radius,omegaDiagRaw);
    puretet_matrix_exp1::print(outer,M,A,S,vol,dg,wall,radius);
    if(O.omegaWallMode==hxt5c::OMEGA_WALL_PROFILE5_EQ_SST){
      puretet_frozen_exp2::run(outer,M,A,C,scalarFixed,S,vol,dg,wall,lw,red,O,radius,omegaDiagRaw);
      puretet_gamma_exp3::run(outer,M,A,C,scalarFixed,S,vol,dg,wall,red,O,radius,omegaDiagRaw);
    }
  }

