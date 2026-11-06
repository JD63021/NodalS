  // P1_CONVFIX: This is the ONLY scalar-convection assembly site.
  // The compact CSR contains ONLY 4x4 P1 vertex slots. Its 8x8 compatibility
  // slot table uses a discard sentinel = nnz for bubble entries, and conv
  // has exactly nnz entries. Running legacy 8x8 tet convection here is OOB.
  // Assemble 16 vertex entries directly, never using the discard sentinel.
  HXT1_CUDA(cudaMemset(A.conv.p,0,A.nnz*sizeof(Real)));
  HXT1_CUDA(cudaMemset(bad.p,0,bad.n*sizeof(unsigned long long)));

  if(compactQ1){
    if(M.h.nhex || M.h.ninterface || M.h.ntet==0 || A.n!=S.nvert)
      throw std::runtime_error("P1_CONVFIX compact convection requires puretet vertex CSR");
    hxt5c::p1_compact_convection_exact<<<M.h.ntet,32>>>(
      tv.p,tg.p,A.tetSlot.p,u0,u1,u2,A.conv.p,M.h.ntet);
    if(outer==1)
      std::printf("P1_CONVFIX_CONVECTION it=1 path=COMPACT_VERTEX_ONLY "
        "localPairs=16 discardWrites=0 doubleAssembly=0 status=PASS\n");
  }else{
    if(M.h.nhex)
      hxt4b::hex_conv_assemble_cached<<<M.h.nhex,128>>>(
        hv.p,A.hexSlot.p,u0,u1,u2,
        hexConvGrad.p,hexConvDetW.p,A.conv.p,M.h.nhex);

    if(M.h.ntet)
      hxt4b::tet_conv_assemble_thread_per_cell
        <<<(M.h.ntet+127)/128,128>>>(
          tv.p,tg.p,A.tetSlot.p,u0,u1,u2,
          A.conv.p,M.h.ntet);
  }
  HXT1_CUDA(cudaGetLastError());
  HXT1_CUDA(cudaDeviceSynchronize());
