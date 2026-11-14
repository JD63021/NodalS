        const double pAbsFinal=pAbs0*pr.rel;pIts+=pr.its;pressure_update_minus_kernel<<<(np+TPB-1)/TPB,TPB>>>(np,pin,(T)alphaP,d_dp.p,d_p.p);
        const double cnPreCorrectionG2=cn;
        if(simpleCoupling=="corrected"){
          // Correct the predictor velocity with the same rAU used to build
          // S = B rAU B^T, then recompute the ACTUAL continuity residual.
          if(stage22d4i3Enable){
            schurI3.correct(d_dp.p);
            hxt5d::s22d4i3a_velocity_subtract<<<(nv+TPB-1)/TPB,TPB>>>(
              nv,d_u0.p,d_u1.p,d_u2.p,d_i3cx.p,d_i3cy.p,d_i3cz.p);
            HXT1_CUDA(cudaGetLastError());
          }else{
          hxt4a::BT_apply(B,d_dp.p,d_bt0.p,d_bt1.p,d_bt2.p);
          hxt5b_simple_velocity_correct<T><<<(nv+TPB-1)/TPB,TPB>>>(nv,d_clampXY.p,A.rau.p,d_bt0.p,d_u0.p);
          hxt5b_simple_velocity_correct<T><<<(nv+TPB-1)/TPB,TPB>>>(nv,d_clampXY.p,stage22Vector?d_s22RauY.p:A.rau.p,d_bt1.p,d_u1.p);
          hxt5b_simple_velocity_correct<T><<<(nv+TPB-1)/TPB,TPB>>>(nv,d_fixed.p,d_rauZ.p,d_bt2.p,d_u2.p);
          if(stage22d4iEnable){
            // The original Nitsche B_N annihilates wall-normal BF3
            // coefficients: B_N*T == B_N.  Therefore projecting the SIMPLE
            // correction retains its pressure-equation mass balance.
            hxt5d::s22d4i_T_apply<<<(BH.wallT.size()+255)/256,256>>>(
              (int)BH.wallT.size(),sstG1.tetWallPlans,d_tv.p,d_u0.p,d_u1.p,d_u2.p);
            HXT1_CUDA(cudaGetLastError());
          }
          } // legacy correction; I3A correction is in the other branch.

          hxt4a::B_apply(B,d_u0.p,d_u1.p,d_u2.p,d_cont.p);
          hxt5b::addsrc<<<(np+TPB-1)/TPB,TPB>>>(np,d_contSource.p,d_cont.p);
          cstatsFull=hxt5b_compact::continuity_stats(np,d_cont.p,d_contStats);

          pinContValue=T(0);
          if(pin>=0) HXT1_CUDA(cudaMemcpy(&pinContValue,d_cont.p+pin,sizeof(T),cudaMemcpyDeviceToHost));
          contPinAbs=(pin>=0)?std::abs((double)pinContValue):0.0;
          hxt4a::pin_zero_kernel<T><<<1,1>>>(pin,d_cont.p);
          cstatsActive=hxt5b_compact::continuity_stats(np,d_cont.p,d_contStats);
          cn=pred.norm(d_cont.p);
          if(!std::isfinite(cn)||!std::isfinite(cstatsActive.localMax)||!std::isfinite(cstatsFull.globalAbs))
            throw std::runtime_error("SST G2 corrected SIMPLE nonfinite continuity");

          if(stage22d4i3Enable && nodalsDiagnosticsDue(git)){
            schurI3.apply(d_dp.p,d_i3schurDp.p);
            hxt5d::s22d4i3a_continuity_identity<<<(np+TPB-1)/TPB,TPB>>>(
              np,d_i3contPre.p,d_cont.p,d_i3schurDp.p,d_i3contErr.p);
            HXT1_CUDA(cudaGetLastError());
            const double defect=pred.norm(d_i3contErr.p);
            const double relative=defect/std::max(cnPreCorrectionG2,1.e-300);
            std::printf("STAGE22D4I3A_IDENTITY it=%d defectL2=%.12e relativeToContPre=%.12e "
                        "definition=CONTPOST_MINUS_CONTPRE_PLUS_SCHUR_DP status=%s\n",
                        git,defect,relative,(std::isfinite(relative)&&relative<=1.e-6)?"PASS":"FAIL");
            if(!std::isfinite(relative)||relative>1.e-6)
              throw std::runtime_error("I3A Schur/velocity-correction identity failed");
          }
          if(stage22d4i3Enable && (git<=5||git%sstG2PrintEvery==0))
            std::printf("STAGE22D4I3A_PRESSURE it=%d "
              "schur=BF3_CONSTRAINED_ROW_L1_GAMMA_FREE "
              "correction=MATCHED contPre=%.12e contPost=%.12e "
              "pressureIts=%d status=PASS\n",git,cnPreCorrectionG2,cn,pr.its);
          if(git<=5||git%sstG2PrintEvery==0)
            std::printf("NODALS_TRUE_SIMPLE_CORRECTION scope=G2 it=%d contPre=%.12e contPost=%.12e ratio=%.12e localMaxPost=%.12e pCG=%d pRel=%.12e status=PASS\n",
              git,cnPreCorrectionG2,cn,cn/std::max(cnPreCorrectionG2,1e-300),cstatsActive.localMax,pr.its,pr.rel);
        }
        if(stage22Vector)
          hxt5b::s22b_add_extra<<<(nv+TPB-1)/TPB,TPB>>>(nv,d_s22Rhs2.p,d_sstDevRhs2.p,d_s22ZExtra.p);
        double s22cPostWallAxial=0.;
        if(stage22Vector && !stage22d4Live && (git==1||git%sstG2PrintEvery==0)){
          HXT1_CUDA(cudaMemset(d_s22PostStats.p,0,7*sizeof(double)));
          HXT1_CUDA(cudaMemset(d_bad.p,0,sizeof(unsigned long long)));
          hxt5b::s22c_wall_post_audit<<<BH.wallT.size(),64>>>(
            sstG1.tetWallPlans,(int)BH.wallT.size(),sstG1.tetWallQuads,
            sstG1.profileTetMatchPoints,d_tv.p,d_tg.p,
            d_u0.p,d_u1.p,d_u2.p,nu,stage22Penalty,d_s22PostStats.p,d_bad.p);
          HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
          unsigned long long badPost=0;
          HXT1_CUDA(cudaMemcpy(&badPost,d_bad.p,sizeof(badPost),cudaMemcpyDeviceToHost));
          if(badPost)throw std::runtime_error("STAGE22C post-correction wall audit invalid samples="+std::to_string(badPost));
          double s22Post[7]={};
          HXT1_CUDA(cudaMemcpy(s22Post,d_s22PostStats.p,sizeof(s22Post),cudaMemcpyDeviceToHost));
          if(!(s22Post[0]>0.)||!std::isfinite(s22Post[1])||!std::isfinite(s22Post[5]))
            throw std::runtime_error("STAGE22C nonfinite post-correction wall audit");
          std::printf("SST22_WALL_NORMAL_AUDIT it=%d meanAbsNormal=%.12e signedNormal=%.12e "
            "meanAbsNormalOverBulk=%.12e penaltyRatio=%.8g state=POST_CORRECTION status=PASS\n",
            git,s22Post[1]/s22Post[0],s22Post[2]/s22Post[0],
            s22Post[1]/(s22Post[0]*std::max(fabs(bulk),1.e-30)),stage22Penalty);
          s22cPostWallAxial=s22Post[5];
          std::printf("SST22C_WALL_FORCE_AUDIT it=%d tractionIntegral=[%.12e,%.12e,%.12e] "
            "excessNormalCoeffAreaIntegral=%.12e state=POST_CORRECTION forceBalance=NOT_CLOSED status=PASS\n",
            git,s22Post[3],s22Post[4],s22Post[5],s22Post[6]);
        }
        if(stage22d4Live){
          const bool due=stage27AuditPolicy.due(git,sstG2Max,"CORRECTED");
          if(due)stage22d4cAudit(git,"CORRECTED");
          std::printf("STAGE27_AUDIT_SCHEDULE it=%d phase=CORRECTED executed=%d mode=%s\n",
                      git,(int)due,stage27AuditPolicy.name());
        }
        // Lightweight wall-normal diagnostic remains on *every* iteration
        // in checkpoint mode. This retains the original affine-face flux check.
        if(stage22d4Live && nodalsDiagnosticsDue(git)){
          HXT1_CUDA(cudaMemset(d_s22PostStats.p,0,4*sizeof(double)));
          hxt5d::s22d4b_wall_normal_audit<<<BH.wallT.size(),64>>>(
            sstG1.tetWallPlans,(int)BH.wallT.size(),sstG1.tetWallQuads,
            d_tv.p,d_u0.p,d_u1.p,d_u2.p,d_s22PostStats.p);
          HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
          double wallN[4]={};
          HXT1_CUDA(cudaMemcpy(wallN,d_s22PostStats.p,sizeof(wallN),cudaMemcpyDeviceToHost));
          if(!(wallN[0]>0.)||!std::isfinite(wallN[1])||!std::isfinite(wallN[2]))
            throw std::runtime_error("STAGE22D4B post-correction wall normal audit invalid");
          std::printf("STAGE22D4B_WALL_NORMAL it=%d meanAbsNormal=%.12e signedNormal=%.12e "
            "wallFlux=%.12e wallFluxOverInlet=%.12e wallArea=%.12e "
            "state=POST_CORRECTION geometry=AFFINE_FACE status=PASS\n",
            git,wallN[1]/wallN[0],wallN[2]/wallN[0],wallN[2],
            fabs(wallN[2])/std::max(fabs(bulk)*stage22d4InletArea,1e-30),wallN[0]);
        }
        double ma0=0.0,ma1=0.0,ma2=0.0;
        // These full momentum residuals are *diagnostics*, not a SIMPLE solve.
        // Still evaluate if the user enabled a momentum-convergence condition.
        if(git%sstG2PrintEvery==0 || nodalsDiagnosticsDue(git) || sstG2MomAtol>0){
        hxt4a::BT_apply(B,d_p.p,d_bt0.p,d_bt1.p,d_bt2.p);ma0=hxt5b::h5b3_physical_residual_norm2(A,A.val.p,d_clampXY.p,d_bt0.p,d_stage23DevRhs0.p,stage22Vector?d_s22Rhs0.p:nullptr,d_u0.p,d_relaxDeltaXY.p,ured),ma1=hxt5b::h5b3_physical_residual_norm2(A,stage22Vector?d_s22ValY.p:A.val.p,d_clampXY.p,d_bt1.p,d_stage23DevRhs1.p,stage22Vector?d_s22Rhs1.p:nullptr,d_u1.p,stage22Vector?d_s22RelaxDeltaY.p:d_relaxDeltaXY.p,ured),ma2=hxt5b::h5b3_physical_residual_norm2(A,d_valZ.p,d_fixed.p,d_bt2.p,d_inletRhsZ.p,stage22Vector?d_s22ZExtra.p:d_sstDevRhs2.p,d_u2.p,d_relaxDeltaZ.p,ured);
        }
        if(stage22d4iEnable && (git%sstG2PrintEvery==0 || nodalsDiagnosticsDue(git) || sstG2MomAtol>0)){
          const std::size_t nbyte=(std::size_t)nv*sizeof(T);
          HXT1_CUDA(cudaMemcpy(d_s22d4i2X.p,d_u0.p,nbyte,cudaMemcpyDeviceToDevice));
          HXT1_CUDA(cudaMemcpy(d_s22d4i2X.p+nv,d_u1.p,nbyte,cudaMemcpyDeviceToDevice));
          HXT1_CUDA(cudaMemcpy(d_s22d4i2X.p+2*(std::size_t)nv,d_u2.p,nbyte,cudaMemcpyDeviceToDevice));
          hxt5d::s22d4i_Q_apply<<<(BH.wallT.size()+255)/256,256>>>(
            (int)BH.wallT.size(),sstG1.tetWallPlans,d_tv.p,
            d_s22d4i2X.p,d_s22d4i2X.p+nv,d_s22d4i2X.p+2*(std::size_t)nv);
          hxt5d::S22D4DOperator i2op{A,VC,A.val.p,d_s22ValY.p,d_valZ.p,
            d_clampXY.p,d_fixed.p,sstG1.tetWallPlans,sstG1.tetWallQuads,
            d_tv.p,d_tg.p,d_s22d4dCoeff.p,(int)BH.wallT.size(),s22d4Gamma,
            d_s22d4iScratch.p,true,sampledConductance};
          if(alGamma!=0.)
            i2op.set_al(&B,d_tg.p,d_alQ.p,d_alX.p,d_alY.p,d_alZ.p,alGamma,pin);
          i2op.set_implicit_dev(implicitDevStress,
            d_pts.p,d_tc.p,d_tv.p,d_tg.p,
            d_u0.p,d_u1.p,d_u2.p,TS.k.p,TS.omega.p,
            M.h.ntet,R,nu,sstG1.kFloor,sstG1.omegaFloor);
          i2op.apply(d_s22d4i2X.p,d_s22d4i2Au.p);  // T^T * Arel * T * x
          // dp still holds the unrelaxed pressure correction. The actual
          // pressure update was p_new = p_old - alphaP*dp.
          hxt4a::BT_apply(B,d_dp.p,d_s22d4i2Dp0.p,d_s22d4i2Dp1.p,d_s22d4i2Dp2.p);
          hxt5d::s22d4i2_adjust<<<(3*nv+255)/256,256>>>(nv,d_u0.p,d_u1.p,d_u2.p,
            d_s22d4i2Old.p,d_relaxDeltaXY.p,d_relaxDeltaZ.p,
            d_s22d4i2Dp0.p,d_s22d4i2Dp1.p,d_s22d4i2Dp2.p,alphaP,d_s22d4i2Adj.p);
          hxt5d::s22d4i_Tt_apply<<<(BH.wallT.size()+255)/256,256>>>(
            (int)BH.wallT.size(),sstG1.tetWallPlans,d_tv.p,
            d_s22d4i2Adj.p,d_s22d4i2Adj.p+nv,d_s22d4i2Adj.p+2*(std::size_t)nv);
          hxt5d::s22d4i2_residual<<<(3*nv+255)/256,256>>>(nv,
            d_s22d4dB.p,d_s22d4i2Au.p,d_s22d4i2Adj.p,
            d_clampXY.p,d_fixed.p,d_s22d4i2R.p);
          HXT1_CUDA(cudaGetLastError());
          s22d4i2Mx=ured.norm(d_s22d4i2R.p);
          s22d4i2My=ured.norm(d_s22d4i2R.p+nv);
          s22d4i2Mz=ured.norm(d_s22d4i2R.p+2*(std::size_t)nv);
          const double i2norm=stage22d4dRed.norm(d_s22d4i2R.p);
          if(!std::isfinite(i2norm)||!std::isfinite(s22d4i2Mx)||
             !std::isfinite(s22d4i2My)||!std::isfinite(s22d4i2Mz))
             throw std::runtime_error("Stage22D4I2 nonfinite reduced post-SIMPLE momentum");
          if(nodalsDiagnosticsDue(git)) std::printf("STAGE22D4I2_REDUCED_MOMENTUM it=%d "
            "projectedPhysical=[%.12e,%.12e,%.12e] combined=%.12e "
            "legacyUnprojected=[%.12e,%.12e,%.12e] "
            "meaning=FROZEN_PICARD_POST_SIMPLE_DERELAXED pressureUpdate=INCLUDED "
            "linearFGMRESGate=NOT_USED status=PASS\n",
            git,s22d4i2Mx,s22d4i2My,s22d4i2Mz,i2norm,ma0,ma1,ma2);
        }
