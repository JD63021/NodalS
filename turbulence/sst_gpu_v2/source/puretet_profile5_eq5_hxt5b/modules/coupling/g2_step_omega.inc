  hxt5c::assemble_equation(hxt5c::OMEGA_EQ,M,BH,inH,inT,wallH,pts,hc,tc,hv,tv,tg,ip,A,u0,u1,u2,S.k.p,S.oldW.p,vol,dg,wall,S.rhs,S.wallStats,bad,S.diagAudit,S.omegaIn,bulk,radius,nu,gamma,wallSampleFraction,O,outer,ifCache,calibratedOmega);
  auto lw=hxt5c::scalar_solve(A,C,scalarFixed,S.rhs.p,S.omega.p,red,O,outer,"omega");if(q1Transport&&A.n>S.nvert)hxt5c::zero_enrichment_kernel<<<(A.n-S.nvert+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(S.nvert,A.n,S.omega.p);const bool omegaDiagNow=(!compactQ1 && M.h.nhex==0 && M.h.ntet>0 && q1Transport && puretet_omega_diag::scheduled(outer));
  std::vector<Real> omegaDiagRaw;
  if(omegaDiagNow){omegaDiagRaw.resize((std::size_t)S.nvert);HXT1_CUDA(cudaMemcpy(omegaDiagRaw.data(),S.omega.p,(std::size_t)S.nvert*sizeof(Real),cudaMemcpyDeviceToHost));}
  HXT1_CUDA(cudaMemset(S.clipW.p,0,sizeof(unsigned long long)));hxt5c::floor_vertex_kernel<<<(M.h.nv+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>((int)M.h.nv,(Real)O.omegaFloor,S.omega.p,S.clipW.p);
  if(bernsteinBF3)bf3_bernstein_apply(S,M,S.omega.p,S.oldW.p,(Real)O.omegaFloor,outer,"omega");
  HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
  if(omegaDiagNow){
    puretet_omega_diag::report(outer,M,A,S,vol,dg,wall,O,radius,omegaDiagRaw);
    puretet_matrix_exp1::print(outer,M,A,S,vol,dg,wall,radius);
    if(O.omegaWallMode==hxt5c::OMEGA_WALL_PROFILE5_EQ_SST){
      puretet_frozen_exp2::run(outer,M,A,C,scalarFixed,S,vol,dg,wall,lw,red,O,radius,omegaDiagRaw);
      puretet_gamma_exp3::run(outer,M,A,C,scalarFixed,S,vol,dg,wall,red,O,radius,omegaDiagRaw);
    }
  }

