inline hxt5c::Stats assemble_momentum_sst(
    const nodals_hxt1::HostMesh&M,const Dev<nodals_hxt1::Point>&pts,const Dev<nodals_hxt1::HexConn>&hc,const Dev<nodals_hxt1::TetConn>&tc,
    const Dev<nodals_hxt1::HexVel>&hv,const Dev<nodals_hxt1::TetVel>&tv,const Dev<nodals_hxt1::TetGeom>&tg,const Dev<hxt2::InterfacePlan>&ip,
    hxt4b::GpuCSR&A,const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,Dev<Real>&turb,Dev<Real>&diagRTurb,
    double radius,double nu,double gamma,const hxt5c::Options&O,Dev<double>&stats,Dev<unsigned long long>&bad,
    Dev<Real>*devRhs0=nullptr,Dev<Real>*devRhs1=nullptr,Dev<Real>*devRhs2=nullptr,
    const hxt5c::SSTInterfaceCache*ifCache=nullptr
#if defined(NODALS_P2_STAGE4_COUPLED)
    ,const nodals::p2::Tet10Device *p2cells=nullptr,const Real *p2k=nullptr,const Real *p2w=nullptr
#endif
    )
{
  HXT1_CUDA(cudaMemset(turb.p,0,turb.n*sizeof(Real)));HXT1_CUDA(cudaMemset(diagRTurb.p,0,diagRTurb.n*sizeof(Real)));HXT1_CUDA(cudaMemset(bad.p,0,bad.n*sizeof(unsigned long long)));
  if(devRhs0)HXT1_CUDA(cudaMemset(devRhs0->p,0,devRhs0->n*sizeof(Real)));
  if(devRhs1)HXT1_CUDA(cudaMemset(devRhs1->p,0,devRhs1->n*sizeof(Real)));
  if(devRhs2)HXT1_CUDA(cudaMemset(devRhs2->p,0,devRhs2->n*sizeof(Real)));
  if(O.deviatoricMomentum){
    if(!devRhs0||!devRhs1||!devRhs2)throw std::runtime_error("SST deviatoric stress requires three explicit RHS buffers");
    if(M.h.ninterface)throw std::runtime_error("SST deviatoric stress V1 is all-HEX/all-conforming only; hybrid interface correction not yet implemented");
    if(M.h.nhex)sst_mom_dev_rhs_hex<<<M.h.nhex,128>>>(pts.p,hc.p,hv.p,u0,u1,u2,k,omega,devRhs0->p,devRhs1->p,devRhs2->p,M.h.nhex,radius,nu,O.kFloor,O.omegaFloor,bad.p);
    if(M.h.ntet)sst_mom_dev_rhs_tet<<<M.h.ntet,64>>>(pts.p,tc.p,tv.p,tg.p,u0,u1,u2,k,omega,devRhs0->p,devRhs1->p,devRhs2->p,M.h.ntet,radius,nu,O.kFloor,O.omegaFloor);
  }
  if(M.h.nhex)sst_mom_hex<<<M.h.nhex,128>>>(pts.p,hc.p,hv.p,A.hexSlot.p,u0,u1,u2,k,omega,turb.p,diagRTurb.p,M.h.nhex,radius,nu,O.kFloor,O.omegaFloor,bad.p);
  if(M.h.ntet){
#if defined(NODALS_P2_STAGE4_COUPLED)
    if(p2cells){
      if(!p2k||!p2w||M.h.nhex||M.h.ninterface||O.deviatoricMomentum)
        throw std::runtime_error("P2_STAGE4: volume coupling requires all pure-tet P2 fields, scalar momentum stress");
      sst_mom_p2_tet<<<M.h.ntet,128>>>(pts.p,tc.p,tv.p,tg.p,A.tetSlot.p,p2cells,u0,u1,u2,p2k,p2w,turb.p,diagRTurb.p,M.h.ntet,radius,nu,O.kFloor,O.omegaFloor);
    }else
#endif
      sst_mom_tet<<<M.h.ntet,64>>>(pts.p,tc.p,tv.p,tg.p,A.tetSlot.p,u0,u1,u2,k,omega,turb.p,diagRTurb.p,M.h.ntet,radius,nu,O.kFloor,O.omegaFloor);
  }
  if(M.h.ninterface){
    if(ifCache)
      sst_mom_interface_cached<<<2*M.h.ninterface,64>>>(
        hv.p,tv.p,ip.p,
        ifCache->q,ifCache->side,
        u0,u1,u2,k,omega,
        turb.p,diagRTurb.p,
        M.h.ninterface,
        radius,nu,gamma,
        O.kFloor,O.omegaFloor);
    else
      sst_mom_interface<<<2*M.h.ninterface,64>>>(
        pts.p,hc.p,tc.p,hv.p,tv.p,tg.p,
        ip.p,A.row.p,A.col.p,
        u0,u1,u2,k,omega,
        turb.p,diagRTurb.p,
        M.h.ninterface,
        radius,nu,gamma,
        O.kFloor,O.omegaFloor,
        bad.p);
  }
  HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());unsigned long long nb=0;HXT1_CUDA(cudaMemcpy(&nb,bad.p,sizeof(nb),cudaMemcpyDeviceToHost));if(nb)throw std::runtime_error("SST G2 momentum diffusion geometry/interface failure count="+std::to_string(nb));
  // PERF_V1: caller discards these momentum-stage SST stats.
  // Accepted post-k/omega statistics are collected once in step_turbulence().
  (void)stats;
  return {};
}

