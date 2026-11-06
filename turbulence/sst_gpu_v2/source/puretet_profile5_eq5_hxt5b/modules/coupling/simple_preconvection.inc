        // PERF_V1: step_turbulence() leaves A.conv assembled with the
        // pressure-corrected velocity, so it is already current for the
        // next momentum step. Only G2 outer 1 needs this pre-momentum build.
        if(git==stage31Start+1){
          HXT1_CUDA(cudaMemset(A.conv.p,0,A.nnz*sizeof(T)));HXT1_CUDA(cudaMemset(d_bad.p,0,sizeof(unsigned long long)));
          if(M.h.nhex)hex_conv_assemble_cached<<<M.h.nhex,128>>>(
          d_hv.p,A.hexSlot.p,
          d_u0.p,d_u1.p,d_u2.p,
          d_hexConvGrad.p,d_hexConvDetW.p,
          A.conv.p,M.h.nhex);
          if(M.h.ntet){
            if(momentumSkew)
              tet_conv_assemble_skew_thread_per_cell<<<(M.h.ntet+127)/128,128>>>(
                d_tv.p,d_tg.p,A.tetSlot.p,d_u0.p,d_u1.p,d_u2.p,A.conv.p,M.h.ntet);
            else if(momentumConservative)
              tet_conv_assemble_conservative_thread_per_cell<<<(M.h.ntet+127)/128,128>>>(
                d_tv.p,d_tg.p,A.tetSlot.p,d_u0.p,d_u1.p,d_u2.p,A.conv.p,M.h.ntet);
            else
              tet_conv_assemble_thread_per_cell<<<(M.h.ntet+127)/128,128>>>(
                d_tv.p,d_tg.p,A.tetSlot.p,d_u0.p,d_u1.p,d_u2.p,A.conv.p,M.h.ntet);
          }
        }
