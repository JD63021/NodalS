        auto tr=nodals::coupling::advance_turbulence(git,[&](int iteration){return hxt5d::step_turbulence(TS,iteration,M,BH,d_inH,d_inT,d_wallH,d_pts,d_hc,d_tc,d_hv,d_tv,d_tg,d_ip,d_hexConvGrad,d_hexConvDetW,(p1Compact?*p1Matrix:A),(p1Compact?p1Colors:VC),(p1Compact?p1ScalarFixed:d_scalarFixed),d_u0.p,d_u1.p,d_u2.p,d_turb,d_dg,d_wall,d_bad,sstRed,bulk,R,nu,gamma,wallSampleFraction,sstG1,&d_sstInterfaceCache,(p1Compact?d_p1L1K.p:nullptr),(p1Compact?d_p1L1Omega.p:nullptr));});g2FinalStats=tr.stats;g2KRel=tr.kRel;g2WRel=tr.omegaRel;g2NutRel=tr.nutRel;g2Outer=git;
        // NODALS_CONSERVATIVE_PICARD_V1: scalar equations used the ORIGINAL
        // advective convection matrix. Reassemble A.conv for the NEXT momentum
        // Picard step only, using the current pressure-corrected carrier U*.
        // NODALS_SKEW_PICARD_V1: only update momentum A.conv AFTER scalar transport.
        if(momentumDivergenceCorrection){
          HXT1_CUDA(cudaMemset(A.conv.p,0,A.nnz*sizeof(T)));
          if(momentumSkew)
            hxt4b::tet_conv_assemble_skew_thread_per_cell
              <<<(M.h.ntet+127)/128,128>>>(d_tv.p,d_tg.p,A.tetSlot.p,
                d_u0.p,d_u1.p,d_u2.p,A.conv.p,M.h.ntet);
          else
            hxt4b::tet_conv_assemble_conservative_thread_per_cell
              <<<(M.h.ntet+127)/128,128>>>(d_tv.p,d_tg.p,A.tetSlot.p,
                d_u0.p,d_u1.p,d_u2.p,A.conv.p,M.h.ntet);
          HXT1_CUDA(cudaGetLastError());
          if(git==stage31Start+1 || git%sstG2PrintEvery==0)
            std::printf("NODALS_MOMENTUM_CONVECTION_REFRESH it=%d "
              "form=%s carrier=PRESSURE_CORRECTED_PICARD_U "
              "scalars=ORIGINAL_ADVECTIVE status=PASS\n",
              git,momentumConvectionForm.c_str());
        }
        // BF3_FAILFAST: inspect live scalar GPU arrays AFTER both scalar solves.
        // At restart, first audit is absolute iteration stage31Start+1, not 1.
        if(std::getenv("NODALS_BF3_REQUIRE_ACTIVE") &&
           std::string(std::getenv("NODALS_BF3_REQUIRE_ACTIVE"))=="1" &&
           (git==stage31Start+1 || git==stage31Start+5 || git==stage31Start+10)){
          const std::size_t nBubble=(std::size_t)nv-(std::size_t)M.h.nv;
          if(nBubble==0)throw std::runtime_error("BF3_FAILFAST: scalar bubble extent is zero");
          std::vector<T> bk(nBubble),bw(nBubble);
          HXT1_CUDA(cudaMemcpy(bk.data(),TS.k.p+M.h.nv,nBubble*sizeof(T),cudaMemcpyDeviceToHost));
          HXT1_CUDA(cudaMemcpy(bw.data(),TS.omega.p+M.h.nv,nBubble*sizeof(T),cudaMemcpyDeviceToHost));
          double maxK=0.,maxW=0.,sumK=0.,sumW=0.;
          std::size_t nonzeroK=0,nonzeroW=0;
          for(std::size_t i=0;i<nBubble;++i){
            const double k=std::abs((double)bk[i]),w=std::abs((double)bw[i]);
            if(!std::isfinite(k)||!std::isfinite(w))
              throw std::runtime_error("BF3_FAILFAST: nonfinite scalar bubble coefficient");
            maxK=std::max(maxK,k);maxW=std::max(maxW,w);
            sumK+=k*k;sumW+=w*w;
            if(k!=0.)++nonzeroK;
            if(w!=0.)++nonzeroW;
          }
          std::printf("NODALS_BF3_POSTSOLVE it=%d bubbleDofs=%zu kNonzero=%zu omegaNonzero=%zu "
             "kMax=%.12e omegaMax=%.12e kL2=%.12e omegaL2=%.12e status=%s\n",
             git,nBubble,nonzeroK,nonzeroW,maxK,maxW,std::sqrt(sumK),std::sqrt(sumW),
             (nonzeroK&&nonzeroW)?"PASS":"ZERO_BUBBLE_FAIL");
          std::fflush(stdout);
          if(!nonzeroK||!nonzeroW)
            throw std::runtime_error("BF3_FAILFAST: scalar BF3 modes remain identically zero after FGMRES");
        }


