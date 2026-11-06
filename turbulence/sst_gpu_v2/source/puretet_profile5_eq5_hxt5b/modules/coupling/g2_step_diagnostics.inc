  hxt5c::field_diff_kernel<<<(A.n+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(A.n,S.k.p,S.oldK.p,A.tmp.p);double dk=red.norm(A.tmp.p),nk=red.norm(S.k.p);hxt5c::field_diff_kernel<<<(A.n+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(A.n,S.omega.p,S.oldW.p,A.tmp.p);double dw=red.norm(A.tmp.p),nw=red.norm(S.omega.p);
  StepResult R;R.kRel=dk/std::max(nk,1e-300);R.omegaRel=dw/std::max(nw,1e-300);R.stats=hxt5c::collect_stats(M.h.nhex,M.h.ntet,pts,hc,tc,hv,tv,tg,u0,u1,u2,S.k.p,S.omega.p,radius,nu,O,S.stats,bad);R.nutRel=S.prevMeanNut>0?std::abs(R.stats.meanNutOverNu-S.prevMeanNut)/std::max(R.stats.meanNutOverNu,1e-300):std::numeric_limits<double>::infinity();S.prevMeanNut=R.stats.meanNutOverNu;R.kLinear=lk;R.omegaLinear=lw;
  // PURETET_AB5 all-five mismatch is measured on the pre-solve old field.
  if(M.h.ntet>0 && M.h.nhex==0 && O.omegaWallMode==hxt5c::OMEGA_WALL_PROFILE5_EQ_SST &&
     (outer<=3 || outer%5==0)){
    std::vector<double> five(33,0);
    HXT1_CUDA(cudaMemcpy(five.data(),S.wallStats.p,33*sizeof(double),cudaMemcpyDeviceToHost));
    double sumT=0,sumE=0;
    double per[5];
    for(int m=0;m<5;++m){
      const double target2=five[18+m], err2=five[23+m];
      sumT+=target2;sumE+=err2;
      per[m]=std::sqrt(std::max(0.0,err2)/std::max(1.e-300,target2));
    }
    const char *subset=std::getenv("PURETET_EQ5_SAMPLE_SET");
    const char *mode=(subset && *subset)?subset:"C";
    const double relative=std::sqrt(std::max(0.0,sumE)/std::max(1.e-300,sumT));
    std::printf("PURETET_AB5_FIVEPOINT it=%d subset=%s gamma=%.9e preSolveAll5RelRMS=%.9e perSample=[%.9e,%.9e,%.9e,%.9e,%.9e] totalFullC=%.9e selectedC=%.9e selectedRawC=%.9e effectiveWeightRatio=%.9e pointEvaluations=%.0f status=%s\n",
       outer,mode,O.profile5EqGamma,relative,per[0],per[1],per[2],per[3],per[4],
       five[7],five[8],five[10],five[8]/std::max(1.e-300,five[7]),five[12],
       std::abs(five[8]-five[7])<=1.e-9*std::max(1.0,five[7])?"PASS":"WARNING");
  }
  std::vector<double>hws(13,0);HXT1_CUDA(cudaMemcpy(hws.data(),S.wallStats.p,13*sizeof(double),cudaMemcpyDeviceToHost));R.wallOmegaMean=hws[0]>0?hws[1]/hws[0]:0;R.wallYPlusMean=hws[0]>0?hws[2]/hws[0]:0;R.wallOmegaSampleMean=hws[0]>0?hws[3]/hws[0]:0;R.wallOmegaTraceMean=hws[0]>0?hws[4]/hws[0]:0;
  if(outer<=5||outer%O.printEvery==0)std::printf("NODALS_SST_OMEGA_WALL mode=%s outer=%d targetMean=%.6e sampleMean=%.6e traceMean=%.6e sampleOverTarget=%.6e traceOverTarget=%.6e yPlusMean=%.6e penaltyGamma=%.6e status=PASS\n",hxt5c::omega_wall_mode_name(O.omegaWallMode),outer,R.wallOmegaMean,R.wallOmegaSampleMean,R.wallOmegaTraceMean,R.wallOmegaMean>0?R.wallOmegaSampleMean/R.wallOmegaMean:0,R.wallOmegaMean>0?R.wallOmegaTraceMean/R.wallOmegaMean:0,R.wallYPlusMean,O.wallPenaltyGamma);if((outer<=5||outer%O.printEvery==0)&&O.omegaWallMode==hxt5c::OMEGA_WALL_OF_AUTO&&hws[0]>0)std::printf("NODALS_SST_OF_AUTO outer=%d ReYMean=%.6e lamFracMean=%.6e omegaVisMean=%.6e omegaLogMean=%.6e pkBulkMean=%.6e pkAutoMean=%.6e pkRatio=%.6e nutSpaldingOverNuMean=%.6e nutSSTOverNuSampleMean=%.6e production=%d prodScale=%.6e wallVolumeScale=%.6e status=PASS\n",outer,hws[5]/hws[0],hws[6]/hws[0],hws[7]/hws[0],hws[8]/hws[0],hws[9]/hws[0],hws[10]/hws[0],hws[9]!=0?hws[10]/hws[9]:0,hws[11]/hws[0],hws[12]/hws[0],(int)O.ofProduction,O.ofProductionScale,O.ofWallVolumeScale);if((outer<=5||outer%O.printEvery==0)&&O.omegaWallMode==hxt5c::OMEGA_WALL_COMPATIBLE&&hws[0]>0)std::printf("NODALS_SST_COMPATIBLE outer=%d uTauMean=%.6e uTauViscMean=%.6e uStarLogMean=%.6e uStarMean=%.6e uStarOverUTau=%.6e tauCompatMean=%.6e pkBulkMean=%.6e pkCompatMean=%.6e pkRatio=%.6e production=%d prodScale=%.6e wallVolumeScale=%.6e status=PASS\n",outer,hws[5]/hws[0],hws[6]/hws[0],hws[7]/hws[0],hws[8]/hws[0],hws[5]!=0?hws[8]/hws[5]:0,hws[11]/hws[0],hws[9]/hws[0],hws[10]/hws[0],hws[9]!=0?hws[10]/hws[9]:0,(int)O.ofProduction,O.ofProductionScale,O.ofWallVolumeScale);if((outer<=5||outer%O.printEvery==0)&&O.omegaWallMode==hxt5c::OMEGA_WALL_CONSISTENT_TABLE&&hws[0]>0)std::printf("NODALS_SST_CONSISTENT_TABLE outer=%d uTauMean=%.6e UPlusMean=%.6e kPlusMean=%.6e omegaPlusMean=%.6e kTargetMean=%.6e omegaTargetMean=%.6e kSampleMean=%.6e nutSSTOverNuSampleMean=%.6e status=PASS\n",outer,hws[5]/hws[0],hws[6]/hws[0],hws[7]/hws[0],hws[8]/hws[0],hws[9]/hws[0],hws[10]/hws[0],hws[11]/hws[0],hws[12]/hws[0]);
  HXT1_CUDA(cudaMemcpy(&R.kClips,S.clipK.p,sizeof(R.kClips),cudaMemcpyDeviceToHost));HXT1_CUDA(cudaMemcpy(&R.omegaClips,S.clipW.p,sizeof(R.omegaClips),cudaMemcpyDeviceToHost));S.totalKClips+=R.kClips;S.totalOmegaClips+=R.omegaClips;
  if(!std::isfinite(R.kRel)||!std::isfinite(R.omegaRel)||!std::isfinite(R.stats.meanNutOverNu)||!std::isfinite(R.stats.maxNutOverNu))throw std::runtime_error("SST G2 nonfinite turbulence update");if(R.stats.meanNutOverNu>1e8||R.stats.maxNutOverNu>1e10)throw std::runtime_error("SST G2 eddy viscosity divergence guard");
