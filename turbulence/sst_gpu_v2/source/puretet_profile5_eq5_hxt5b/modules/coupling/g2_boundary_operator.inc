inline void assemble_boundary_sst(const hxt5b::BoundaryHost&H,const Dev<hxt5b::HexFacePlan>&ih,const Dev<hxt5b::TetFacePlan>&it,const Dev<hxt5b::HexFacePlan>&wh,const Dev<nodals_hxt1::Point>&pts,const Dev<nodals_hxt1::HexConn>&hc,const Dev<nodals_hxt1::TetConn>&tc,const Dev<nodals_hxt1::HexVel>&hv,const Dev<nodals_hxt1::TetVel>&tv,const Dev<nodals_hxt1::TetGeom>&tg,hxt4b::GpuCSR&A,const Real*u0,const Real*u1,const Real*u2,const Real*k,const Real*omega,Dev<Real>&dg,Dev<Real>&ddg,Dev<Real>&rx,Dev<Real>&ry,Dev<Real>&rz,Dev<Real>&wall,Dev<Real>&dwall,Dev<double>&ws,Dev<unsigned long long>&bad,double bulk,double radius,double nu,double sf,const hxt5c::Options&O,int momentumWallMode,int spaldingMolecularConsistency,double weakWallGamma,double weakWallHScale,double weakWallNutScale,
 bool stage22Vector=false,double stage22Penalty=0.,
 Real*stage22Rhs0=nullptr,Real*stage22Rhs1=nullptr,Real*stage22Rhs2=nullptr,
 Real*stage22WallY=nullptr,Real*stage22WallZ=nullptr,
 Real*stage22DiagY=nullptr,Real*stage22DiagZ=nullptr,
 bool stage22dFaceFit=false,bool stage22d4Live=false
#if defined(NODALS_P2_STAGE4_COUPLED)
 ,const nodals::p2::Tet10Device *p2cells=nullptr,const Real *p2k=nullptr,const Real *p2w=nullptr
#endif
 ){
  HXT1_CUDA(cudaMemset(dg.p,0,dg.n*sizeof(Real)));HXT1_CUDA(cudaMemset(ddg.p,0,ddg.n*sizeof(Real)));HXT1_CUDA(cudaMemset(rx.p,0,rx.n*sizeof(Real)));HXT1_CUDA(cudaMemset(ry.p,0,ry.n*sizeof(Real)));HXT1_CUDA(cudaMemset(rz.p,0,rz.n*sizeof(Real)));HXT1_CUDA(cudaMemset(wall.p,0,wall.n*sizeof(Real)));HXT1_CUDA(cudaMemset(dwall.p,0,dwall.n*sizeof(Real)));HXT1_CUDA(cudaMemset(ws.p,0,ws.n*sizeof(double)));HXT1_CUDA(cudaMemset(bad.p,0,bad.n*sizeof(unsigned long long)));
  const auto inlet23=nodals::stage23::from_environment(bulk);
  if(!H.inH.empty())sst_dg_hex<<<H.inH.size(),64>>>(ih.p,(int)H.inH.size(),pts.p,hc.p,hv.p,A.hexSlot.p,u0,u1,u2,k,omega,dg.p,ddg.p,rx.p,ry.p,rz.p,inlet23.x,inlet23.y,inlet23.z,radius,nu,O.kFloor,O.omegaFloor,bad.p);
  if(!H.inT.empty())sst_dg_tet<<<H.inT.size(),64>>>(it.p,(int)H.inT.size(),pts.p,tc.p,tv.p,tg.p,A.tetSlot.p,u0,u1,u2,k,omega,dg.p,ddg.p,rx.p,ry.p,rz.p,inlet23.x,inlet23.y,inlet23.z,radius,nu,O.kFloor,O.omegaFloor
#if defined(NODALS_P2_STAGE4_COUPLED)
      ,p2cells,p2k,p2w
#endif
      );
  if(!H.wallH.empty()){
    if(momentumWallMode==MOM_WALL_SPALDING){
      hxt5b::wall_hex<<<H.wallH.size(),64>>>(wh.p,(int)H.wallH.size(),pts.p,hc.p,hv.p,A.hexSlot.p,u2,wall.p,dwall.p,nu,sf,ws.p,bad.p);

      // SPALDING_MOLECULAR_CONSISTENCY_V1
      // Historical form:
      //   existing Spalding Robin/wall-law contribution
      //   + ONLY the two molecular Nitsche consistency terms
      //
      //   -<v,nu dn(u)> - <nu dn(v),u>
      //
      // No extra gamma*nu/h penalty is added here.
      if(spaldingMolecularConsistency){
        sst_wall_nitsche_hex<<<H.wallH.size(),64>>>(
          wh.p,(int)H.wallH.size(),pts.p,hc.p,hv.p,A.hexSlot.p,
          u0,u1,u2,k,omega,wall.p,dwall.p,radius,nu,sf,
          0.0,1.0,0.0,0,
          O.kFloor,O.omegaFloor,ws.p,bad.p);
      }
    }else if(momentumWallMode==MOM_WALL_SPALDING_TRACE){
      sst_wall_spalding_trace_hex<<<H.wallH.size(),64>>>(
        wh.p,(int)H.wallH.size(),pts.p,hc.p,hv.p,A.hexSlot.p,
        u2,wall.p,dwall.p,nu,sf,spaldingMolecularConsistency,
        ws.p,bad.p);
    }else if(momentumWallMode==MOM_WALL_SPALDING_ALIGNED){
      sst_wall_spalding_aligned_hex<<<H.wallH.size(),64>>>(
        wh.p,(int)H.wallH.size(),pts.p,hc.p,hv.p,A.hexSlot.p,
        u2,wall.p,dwall.p,nu,sf,spaldingMolecularConsistency,
        weakWallGamma,ws.p,bad.p);
    }else if(momentumWallMode==MOM_WALL_SPALDING_PROFILE_INTEGRAL){
      sst_wall_spalding_profile_integral_hex<<<H.wallH.size(),64>>>(
        wh.p,(int)H.wallH.size(),pts.p,hc.p,hv.p,A.hexSlot.p,
        u2,wall.p,dwall.p,nu,O.profileIntegralNQ,(int)O.profileAverageClosure,
        O.profileAverageWallLaw,O.profileMatchPoints,ws.p,bad.p);
    }else if(momentumWallMode==MOM_WALL_SPALDING_COMPATIBLE){
      sst_wall_spalding_compatible_hex<<<H.wallH.size(),64>>>(
        wh.p,(int)H.wallH.size(),pts.p,hc.p,hv.p,A.hexSlot.p,
        u2,k,wall.p,dwall.p,nu,sf,O.kFloor,ws.p,bad.p);
    }else if(momentumWallMode==MOM_WALL_SST_CONSISTENT_TABLE){
      sst_wall_consistent_table_hex<<<H.wallH.size(),64>>>(
        wh.p,(int)H.wallH.size(),pts.p,hc.p,hv.p,A.hexSlot.p,
        u2,wall.p,dwall.p,nu,sf,O.profileIntegralNQ,(int)O.profileAverageClosure,ws.p,bad.p);
    }else{
      sst_wall_nitsche_hex<<<H.wallH.size(),64>>>(
        wh.p,(int)H.wallH.size(),pts.p,hc.p,hv.p,A.hexSlot.p,
        u0,u1,u2,k,omega,wall.p,dwall.p,radius,nu,sf,
        weakWallGamma,weakWallHScale,weakWallNutScale,
        momentumWallMode==MOM_WALL_WEAK_EFFECTIVE,
        O.kFloor,O.omegaFloor,ws.p,bad.p);
    }
  }
  if(stage22dFaceFit && (stage22Vector||H.wallT.empty()||!H.wallH.empty()))
    throw std::runtime_error("STAGE22D face fit diagnostic requires axial pure-tet legacy constraints");
  if(stage22Vector){
    if(H.wallT.empty()||!H.wallH.empty()||!stage22Rhs0||!stage22Rhs1||!stage22Rhs2||
       !(stage22Penalty>=0.)||!std::isfinite(stage22Penalty))
      throw std::runtime_error("STAGE22B vector wall requires pure-tet cache and three lag RHS arrays");
    HXT1_CUDA(cudaMemset(stage22Rhs0,0,A.n*sizeof(Real)));
    HXT1_CUDA(cudaMemset(stage22Rhs1,0,A.n*sizeof(Real)));
    HXT1_CUDA(cudaMemset(stage22Rhs2,0,A.n*sizeof(Real)));
    if(!stage22WallY||!stage22WallZ||!stage22DiagY||!stage22DiagZ)
      throw std::runtime_error("STAGE22C component-specific wall arrays missing");
    HXT1_CUDA(cudaMemset(stage22WallY,0,A.nnz*sizeof(Real)));
    HXT1_CUDA(cudaMemset(stage22WallZ,0,A.nnz*sizeof(Real)));
    HXT1_CUDA(cudaMemset(stage22DiagY,0,A.n*sizeof(Real)));
    HXT1_CUDA(cudaMemset(stage22DiagZ,0,A.n*sizeof(Real)));
    hxt5b::s22c_wall_tet_component<<<H.wallT.size(),64>>>(
      O.tetWallPlans,(int)H.wallT.size(),O.tetWallQuads,O.profileTetMatchPoints,
      tv.p,tg.p,A.tetSlot.p,u0,u1,u2,
      wall.p,stage22WallY,stage22WallZ,
      dwall.p,stage22DiagY,stage22DiagZ,
      stage22Rhs0,stage22Rhs1,stage22Rhs2,
      nu,stage22Penalty,ws.p,bad.p);
  }
  if(!stage22Vector&&!stage22d4Live&&!H.wallT.empty()){
    // The same five-point Spalding TET velocity fit is valid for either
    // turbulence closure. The scalar wall dispatch is handled separately.
    if(momentumWallMode!=MOM_WALL_SPALDING_PROFILE_INTEGRAL ||
       !O.profileAverageClosure || O.profileAverageWallLaw!=5 ||
       (O.omegaWallMode!=hxt5c::OMEGA_WALL_PROFILE5_EQ_SST &&
        O.omegaWallMode!=hxt5c::OMEGA_WALL_OF_AUTO))
      throw std::runtime_error("pure-tet momentum requires PROFILE5 with EQ5 or OF_AUTO");
    if(stage22dFaceFit){
      hxt5b::s22d_face_fit_axial<<<H.wallT.size(),64>>>(
        O.tetWallPlans,(int)H.wallT.size(),O.tetWallQuads,O.profileTetMatchPoints,
        tv.p,A.tetSlot.p,u0,u1,u2,wall.p,dwall.p,nu,ws.p,bad.p);
    }else{
      hxt5b::wall_tet_profile5<<<H.wallT.size(),64>>>(
        O.tetWallPlans,(int)H.wallT.size(),O.tetWallQuads,O.profileTetMatchPoints,
        tv.p,A.tetSlot.p,u2,wall.p,dwall.p,nu,ws.p,bad.p);
    }
  }
  HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());unsigned long long q=0;HXT1_CUDA(cudaMemcpy(&q,bad.p,sizeof(q),cudaMemcpyDeviceToHost));if(q)throw std::runtime_error("SST G2 momentum boundary failure count="+std::to_string(q));
}


