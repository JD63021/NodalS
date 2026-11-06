  if(M.h.ninterface){
    if(ifCache)
      sst_interface_diffusion_cached<<<2*M.h.ninterface,64>>>(
        hv.p,tv.p,ip.p,
        ifCache->q,ifCache->side,
        u0,u1,u2,k,omega,
        vol.p,M.h.ninterface,
        radius,nu,interfaceGamma,
        (int)eq,
        O.kFloor,O.omegaFloor);
    else
      sst_interface_diffusion<<<2*M.h.ninterface,64>>>(
        pts.p,hc.p,tc.p,hv.p,tv.p,tg.p,
        ip.p,A.row.p,A.col.p,
        u0,u1,u2,k,omega,
        vol.p,M.h.ninterface,
        radius,nu,interfaceGamma,
        (int)eq,
        O.kFloor,O.omegaFloor,
        bad.p);
  }
  if(!BH.inH.empty())sst_inlet_hex<<<BH.inH.size(),64>>>(inH.p,(int)BH.inH.size(),pts.p,hc.p,hv.p,A.hexSlot.p,u0,u1,u2,k,omega,dg.p,rhs.p,qIn,inlet23.x,inlet23.y,inlet23.z,radius,nu,(int)eq,O.kFloor,O.omegaFloor,bad.p);
  if(!BH.inT.empty())sst_inlet_tet<<<BH.inT.size(),64>>>(inT.p,(int)BH.inT.size(),pts.p,tc.p,tv.p,tg.p,A.tetSlot.p,u0,u1,u2,k,omega,dg.p,rhs.p,qIn,inlet23.x,inlet23.y,inlet23.z,radius,nu,(int)eq,O.kFloor,O.omegaFloor);
  if(exp1)takeRhs(expInlet);
  if(O.profileAverageClosure && O.profileAverageWallLaw==5 &&
           O.omegaWallMode==OMEGA_WALL_PROFILE5_EQ_SST && !BH.wallH.empty()){
    sst_profile5_equilibrium_closure_hex<<<BH.wallH.size(),256>>>(
      wallH.p,(int)BH.wallH.size(),pts.p,hc.p,hv.p,A.hexSlot.p,
      u2,k,omega,wall.p,rhs.p,nu,O.profile5EqGamma,(int)eq,
      O.profileMatchPoints,wallStats.p,bad.p);
  }else if(O.profileAverageClosure && O.omegaWallMode==OMEGA_WALL_CONSISTENT_TABLE && !BH.wallH.empty()){
    sst_table_profile_average_closure_hex<<<BH.wallH.size(),256>>>(
      wallH.p,(int)BH.wallH.size(),pts.p,hc.p,hv.p,A.hexSlot.p,
      u0,u1,u2,k,omega,wall.p,rhs.p,
      radius,nu,O.wallPenaltyGamma,O.kFloor,O.omegaFloor,
      (int)eq,O.profileIntegralNQ,wallStats.p,bad.p);
  }else if(O.profileAverageClosure &&
           (O.omegaWallMode==OMEGA_WALL_SAMPLE_LOG||O.omegaWallMode==OMEGA_WALL_TRACE_LOG) &&
           eq==OMEGA_EQ && !BH.wallH.empty()){
    sst_log_profile_average_closure_hex<<<BH.wallH.size(),256>>>(
      wallH.p,(int)BH.wallH.size(),pts.p,hc.p,hv.p,A.hexSlot.p,
      u0,u1,u2,k,omega,wall.p,rhs.p,
      radius,nu,O.wallPenaltyGamma,O.kFloor,O.omegaFloor,
      O.omegaWallMode,O.profileIntegralNQ,wallStats.p,bad.p);
  }else if(O.profileAverageClosure && O.omegaWallMode==OMEGA_WALL_OF_AUTO && !BH.wallH.empty()){
    sst_of_profile_average_closure_hex<<<BH.wallH.size(),256>>>(
      wallH.p,(int)BH.wallH.size(),pts.p,hc.p,hv.p,A.hexSlot.p,
      u0,u1,u2,k,omega,wall.p,rhs.p,
      radius,nu,O.wallPenaltyGamma,O.kFloor,O.omegaFloor,
      O.ofKappa,O.ofE,O.ofBeta1,O.ofReBlend,O.ofProductionLimit,
      O.ofProductionScale,O.ofWallVolumeScale,(int)eq,
      (int)O.sharedSpaldingWallState,O.profileIntegralNQ,
      O.profileAverageWallLaw,O.profileMatchPoints,
      (int)O.ofProduction,wallStats.p,bad.p);
    if(O.profileAverageWallLaw==3 || O.profileAverageWallLaw==4 || O.profileAverageWallLaw==5){
      HXT1_CUDA(cudaGetLastError());
      HXT1_CUDA(cudaDeviceSynchronize());
    }
  }else if(O.profileIntegralClosure && O.omegaWallMode==OMEGA_WALL_OF_AUTO && !BH.wallH.empty()){
    sst_of_profile_closure_hex<<<BH.wallH.size(),256>>>(
      wallH.p,(int)BH.wallH.size(),pts.p,hc.p,hv.p,A.hexSlot.p,
      u0,u1,u2,k,omega,wall.p,rhs.p,
      radius,nu,O.wallPenaltyGamma,O.kFloor,O.omegaFloor,
      O.ofKappa,O.ofE,O.ofBeta1,O.ofReBlend,O.ofProductionLimit,
      O.ofProductionScale,O.ofWallVolumeScale,(int)eq,
      (int)O.sharedSpaldingWallState,O.profileIntegralNQ,
      O.profileClosureYPlusMin,(int)O.ofProduction,
      wallStats.p,bad.p);
  }else{
    if((eq==OMEGA_EQ||(eq==K_EQ&&(O.omegaWallMode==OMEGA_WALL_CONSISTENT_TABLE||O.omegaWallMode==OMEGA_WALL_LOWRE_RESOLVED)))&&!BH.wallH.empty())sst_omega_wall_hex<<<BH.wallH.size(),64>>>(wallH.p,(int)BH.wallH.size(),pts.p,hc.p,hv.p,A.hexSlot.p,u0,u1,u2,k,omega,wall.p,rhs.p,radius,nu,wallSampleFraction,O.wallPenaltyGamma,O.kFloor,O.omegaFloor,(int)eq,O.omegaWallMode,O.ofKappa,O.ofE,O.ofBeta1,O.ofReBlend,O.ofProductionLimit,(int)O.sharedSpaldingWallState,(int)O.profileIntegralUTau,O.profileIntegralNQ,O.lowReOmegaFactor,wallStats.p,bad.p);
    if((O.omegaWallMode==OMEGA_WALL_OF_AUTO||O.omegaWallMode==OMEGA_WALL_COMPATIBLE)&&O.ofProduction&&!BH.wallH.empty())sst_of_wall_production_hex<<<BH.wallH.size(),64>>>(wallH.p,(int)BH.wallH.size(),pts.p,hc.p,hv.p,u0,u1,u2,k,omega,rhs.p,radius,nu,wallSampleFraction,O.kFloor,O.omegaFloor,O.ofKappa,O.ofE,O.ofBeta1,O.ofReBlend,O.ofProductionLimit,O.ofProductionScale,O.ofWallVolumeScale,O.omegaWallMode,(int)eq,(int)O.sharedSpaldingWallState,(int)O.profileIntegralUTau,O.profileIntegralNQ,bad.p);
  }
  if(!BH.wallT.empty()){
    if(!(O.profileAverageClosure && O.profileAverageWallLaw==5))
      throw std::runtime_error("pure-tet wall requires PROFILE5 five-point Spalding sampling");
    if(O.omegaWallMode==OMEGA_WALL_PROFILE5_EQ_SST){
      sst_profile5_equilibrium_closure_tet<<<BH.wallT.size(),256>>>(
        O.tetWallPlans,(int)BH.wallT.size(),pts.p,tc.p,tv.p,A.tetSlot.p,
        u2,k,omega,wall.p,rhs.p,nu,O.profile5EqGamma,(int)eq,eq5Mask,
        O.profileTetMatchPoints,wallStats.p,bad.p);
    }else if(O.omegaWallMode==OMEGA_WALL_OF_AUTO){
      if(!O.sharedSpaldingWallState)
        throw std::runtime_error("pure-tet of_auto requires shared five-point Spalding u_tau");
      if(eq==OMEGA_EQ || O.ofProduction){
        sst_tet_of_auto_spalding5_point3<<<BH.wallT.size(),64>>>(
          O.tetWallPlans,(int)BH.wallT.size(),pts.p,tc.p,tg.p,tv.p,A.tetSlot.p,
          u0,u1,u2,k,omega,wall.p,rhs.p,radius,nu,O.wallPenaltyGamma,
          O.kFloor,O.omegaFloor,O.ofKappa,O.ofE,O.ofBeta1,
          O.ofReBlend,O.ofProductionLimit,(int)O.ofProduction,
          O.ofProductionScale,O.ofWallVolumeScale,(int)eq,
          O.profileTetMatchPoints,wallStats.p,bad.p,O.stage27SavedWallFit);
      }
    }else{
      throw std::runtime_error("pure-tet wall supports PROFILE5 EQ5 or PROFILE5 OF_AUTO only");
    }
  }
