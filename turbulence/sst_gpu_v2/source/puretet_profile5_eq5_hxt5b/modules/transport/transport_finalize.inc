  if(exp1){takeRhs(expTotal);
    puretet_source_exp1::lastWallRhs.resize((size_t)A.n);
    for(size_t i=0;i<(size_t)A.n;++i)
      puretet_source_exp1::lastWallRhs[i]=expTotal[i]-expInlet[i];
    puretet_source_exp1::print(nonlinearIt,M,(int)M.h.nv,radius,expVolume,expInlet,expTotal,expP,expC);}
  combine_scalar<<<(A.nnz+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(A.nnz,A.base.p,A.conv.p,vol.p,dg.p,wall.p,A.val.p);
  HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());unsigned long long nb=0;HXT1_CUDA(cudaMemcpy(&nb,bad.p,sizeof(nb),cudaMemcpyDeviceToHost));if(nb)throw std::runtime_error("SST G1 assembly geometry/root failure count="+std::to_string(nb));
  // Optional one-time reference export. The enriched CSR is constructed ONLY
  // by the already-existing masked reference run, never by compact P1.
  if(!compactQ1 && nonlinearIt==1 && q1_transport_enabled()){
    const char* exportDir=std::getenv("SST_P1_L1_EXPORT_DIR");
    if(exportDir && *exportDir)
      p1_l1_dump_reference(exportDir,eq==K_EQ?"k":"omega",A,(int)M.h.nv);
  }
  const char* parityDir=std::getenv("SST_P1_PARITY_DIR");
  const char* maskedVertexEnv=std::getenv("SST_P1_MASKED_VERTEX_L1");
  const bool maskedVertexL1=!compactQ1 && q1_transport_enabled() &&
      maskedVertexEnv && std::string(maskedVertexEnv)=="1";
  const char* parityMode=compactQ1?"compact":(maskedVertexL1?"maskedvertex":"masked");
  const char* parityEq=eq==K_EQ?"k":"omega";
  if(parityDir && *parityDir && nonlinearIt==1 && q1_transport_enabled())
    p1_parity_dump_vertex_block(parityDir,parityMode,parityEq,"raw",A,rhs,(int)M.h.nv);
  HXT1_CUDA(cudaMemset(diagAudit.p,0,diagAudit.n*sizeof(unsigned long long)));
  const Real alpha=(Real)(eq==K_EQ?O.alphaK:O.alphaOmega);
  const char *boostRaw=std::getenv(eq==K_EQ?"SST_P1_L1_K_BOOST":"SST_P1_L1_OMEGA_BOOST");
  const double boostD=(boostRaw&&*boostRaw)?std::stod(boostRaw):1.0;
  if(!std::isfinite(boostD)||boostD<0.5||boostD>20.0)
    throw std::runtime_error("P1_L1_CALIB boost must lie in [0.5,20]");
  if(compactQ1 && nonlinearIt==1)
    std::printf("P1_L1_ACTIVE equation=%s calibrated=%d boost=%.8g alpha=%.8g status=%s\n",
      eq==K_EQ?"k":"omega",(int)(calibratedL1!=nullptr),boostD,(double)alpha,
      calibratedL1?"PASS":"UNCALIBRATED_REFERENCE_ONLY");
  scalar_row_l1_relax_kernel<<<(A.n+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(
      A.n,A.row.p,A.col.p,A.diagPos.p,A.val.p,rhs.p,eq==K_EQ?k:omega,alpha,
      diagAudit.p,(int)M.h.nv,(int)maskedVertexL1,calibratedL1,(Real)boostD);
  if(nonlinearIt==1 && q1_transport_enabled())
    std::printf("P1_PARITY_RELAX equation=%s mode=%s l1Policy=%s status=PASS\n",
       parityEq,parityMode,maskedVertexL1?"VERTEX_ONLY":"EXISTING_FULL_ROW");
  HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
  if(parityDir && *parityDir && nonlinearIt==1 && q1_transport_enabled())
    p1_parity_dump_vertex_block(parityDir,parityMode,parityEq,"relaxed",A,rhs,(int)M.h.nv);
  unsigned long long ha[2]={0,0};HXT1_CUDA(cudaMemcpy(ha,diagAudit.p,2*sizeof(unsigned long long),cudaMemcpyDeviceToHost));
  if(nonlinearIt<=5||nonlinearIt%O.printEvery==0||ha[1])std::printf("NODALS_SST_G1_DIAG equation=%s nonlinearIt=%d rawNonpositive=%llu relaxedNonpositive=%llu alpha=%.6g relaxation=ROW_L1_FIXED_POINT status=%s\n",eq==K_EQ?"k":"omega",nonlinearIt,ha[0],ha[1],(double)alpha,ha[1]?"FAIL":"PASS");
  if(ha[1])throw std::runtime_error("SST G1 relaxed scalar diagonal invalid count="+std::to_string(ha[1]));
}

