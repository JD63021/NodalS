  if(M.h.nhex)sst_hex_volume<<<M.h.nhex,128>>>(pts.p,hc.p,hv.p,A.hexSlot.p,u0,u1,u2,k,omega,vol.p,rhs.p,M.h.nhex,radius,nu,(int)eq,O.kFloor,O.omegaFloor,bad.p);
  if(M.h.ntet && compactQ1)
    p1_compact_sst_tet_volume<<<M.h.ntet,64>>>(pts.p,tc.p,tv.p,tg.p,A.tetSlot.p,
      u0,u1,u2,k,omega,vol.p,rhs.p,M.h.ntet,radius,nu,(int)eq,O.kFloor,O.omegaFloor,compactQuad);
  else if(M.h.ntet)sst_tet_volume<<<M.h.ntet,64>>>(pts.p,tc.p,tv.p,tg.p,A.tetSlot.p,u0,u1,u2,k,omega,vol.p,rhs.p,exp1?expProd.p:nullptr,exp1?expCross.p:nullptr,exp1?expReaction.p:nullptr,M.h.ntet,radius,nu,(int)eq,O.kFloor,O.omegaFloor);
  if(stage33ScalarSupg){
    stage33bc_scalar_supg24<<<(M.h.ntet+7)/8,256>>>(
        pts.p,tc.p,tv.p,tg.p,A.tetSlot.p,
        u0,u1,u2,k,omega,vol.p,rhs.p,M.h.ntet,
        radius,nu,(int)eq,O.kFloor,O.omegaFloor,
        stage33ScalarTauScale,stage33ScalarMagic,bad.p);
    HXT1_CUDA(cudaGetLastError());
    HXT1_CUDA(cudaDeviceSynchronize());
    if(nonlinearIt<=5||nonlinearIt%O.printEvery==0)
      std::printf("STAGE33BC_SCALAR_ASSEMBLED equation=%s it=%d cells=%llu "
        "tauScale=%.8g magic=%.8g status=PASS\n",
        eq==K_EQ?"k":"omega",nonlinearIt,(unsigned long long)M.h.ntet,
        stage33ScalarTauScale,stage33ScalarMagic);
  }
  if(exp1){takeRhs(expVolume);expP.resize((size_t)A.n);expC.resize((size_t)A.n);HXT1_CUDA(cudaMemcpy(expP.data(),expProd.p,(size_t)A.n*sizeof(Real),cudaMemcpyDeviceToHost));HXT1_CUDA(cudaMemcpy(expC.data(),expCross.p,(size_t)A.n*sizeof(Real),cudaMemcpyDeviceToHost));puretet_source_exp1::lastOuter=nonlinearIt;puretet_source_exp1::lastCross=expC;puretet_source_exp1::lastReaction.resize((size_t)A.nnz);HXT1_CUDA(cudaMemcpy(puretet_source_exp1::lastReaction.data(),expReaction.p,(size_t)A.nnz*sizeof(Real),cudaMemcpyDeviceToHost));}
