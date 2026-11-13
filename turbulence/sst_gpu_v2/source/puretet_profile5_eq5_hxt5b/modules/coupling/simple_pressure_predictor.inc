        if(stage22d4i3Enable){
          HXT1_CUDA(cudaMemset(d_bad.p,0,sizeof(unsigned long long)));
          hxt5d::s22d4i3a_refinv<<<(nv+TPB-1)/TPB,TPB>>>(
            nv,A.row.p,A.diagPos.p,A.val.p,d_s22ValY.p,d_valZ.p,
            d_i3penX.p,d_i3penY.p,d_i3penZ.p,
            d_relaxDeltaXY.p,d_s22RelaxDeltaY.p,d_relaxDeltaZ.p,
            d_clampXY.p,d_fixed.p,alphaU,rauScale,
            d_i3rx.p,d_i3ry.p,d_i3rz.p,d_bad.p);
          HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
          unsigned long long badI3=0;
          HXT1_CUDA(cudaMemcpy(&badI3,d_bad.p,sizeof(badI3),cudaMemcpyDeviceToHost));
          if(badI3)throw std::runtime_error(
            "I3A invalid gamma-free row-L1 denominator rows="+std::to_string(badI3));
          if(git==1||git%sstG2PrintEvery==0)
            std::printf("STAGE22D4I3A_D0 it=%d gammaStab=%.8g "
              "gammaPenaltyCSR=SUBTRACTED rowL1=UNRELAXED_NONPENALIZED "
              "spalding=INCLUDED status=PASS\n",git,s22d4Gamma);
        }
        hxt5b::sdiag<<<(np+TPB-1)/TPB,TPB>>>(np,B.row.p,B.col.p,B.bx.p,B.by.p,B.bz.p,A.rau.p,(stage22Vector?d_s22RauY.p:A.rau.p),d_rauZ.p,d_sd.p,pin);hxt4a::invert_diag_kernel<T><<<(np+TPB-1)/TPB,TPB>>>(np,d_sd.p,d_isd.p,pin);hxt5b::SAniso<T>Sg2(B,A.rau.p,(stage22Vector?d_s22RauY.p:A.rau.p),d_rauZ.p,pin);
        predAuditMark(1);
        // P50: Sg2 above is the LIVE Schur equation, rebuilt from CURRENT rAU.
        // PC.fine is only a numerical preconditioner; it can be reused.
        if(amgFineMode=="explicit"){
          const bool p50DoRefresh=(git==stage31Start+1)||
            ((git-1)%p50RefreshEvery==0);
          double p50ThisMs=0.0;
          if(p50DoRefresh){
            const auto p50T0=std::chrono::steady_clock::now();
            hxt5b::refresh_pc(PC,B,A.rau.p,
              (stage22Vector?d_s22RauY.p:A.rau.p),d_rauZ.p);
            // refresh_pc already performs a GPU device synchronization.
            p50ThisMs=std::chrono::duration<double,std::milli>(
              std::chrono::steady_clock::now()-p50T0).count();
            p50RefreshTotalMs+=p50ThisMs;
            p50LastRefresh=git;
            ++p50RefreshCount;
          }
          const int p50NextRefresh=p50LastRefresh+p50RefreshEvery;
          std::printf("NODALS_P50_REFRESH it=%d action=%s "
            "cadence=%d lastRefresh=%d nextRefresh=%d "
            "refreshMs=%.6f cumulativeRefreshMs=%.6f refreshCount=%d "
            "fineMatrix=%s livePressureOperator=YES currentRAU=YES "
            "status=PASS\n",git,p50DoRefresh?"REFRESH":"REUSE",
            p50RefreshEvery,p50LastRefresh,p50NextRefresh,p50ThisMs,
            p50RefreshTotalMs,p50RefreshCount,
            p50DoRefresh?"UPDATED":"FROZEN");
        }
        predAuditMark(2);
        hxt4a::BT_apply(B,d_p.p,d_bt0.p,d_bt1.p,d_bt2.p);
        hxt5b::h5b3_rhs2<<<(nv+TPB-1)/TPB,TPB>>>(nv,d_clampXY.p,d_bt0.p,d_stage23DevRhs0.p,stage22Vector?d_s22Rhs0.p:nullptr,d_u0.p,d_relaxDeltaXY.p,d_rhs0.p);hxt5b::h5b3_rhs2<<<(nv+TPB-1)/TPB,TPB>>>(nv,d_clampXY.p,d_bt1.p,d_stage23DevRhs1.p,stage22Vector?d_s22Rhs1.p:nullptr,d_u1.p,stage22Vector?d_s22RelaxDeltaY.p:d_relaxDeltaXY.p,d_rhs1.p);hxt5b::h5b3_rhs2<<<(nv+TPB-1)/TPB,TPB>>>(nv,d_fixed.p,d_bt2.p,d_inletRhsZ.p,d_sstDevRhs2.p,d_u2.p,d_relaxDeltaZ.p,d_rhs2.p);
        if(stage22Vector)
          hxt5b::s22b_add_rhs<<<(nv+TPB-1)/TPB,TPB>>>(nv,d_s22Rhs2.p,d_rhs2.p);
        if(sstG1.deviatoricMomentum && (git<=5||git%sstG2PrintEvery==0))
          std::printf("NODALS_SST_DEVIATORIC_RHS it=%d norm=[%.12e,%.12e,%.12e] status=PASS\n",git,ured.norm(d_sstDevRhs0.p),ured.norm(d_sstDevRhs1.p),ured.norm(d_sstDevRhs2.p));
        // NODALS_UNIFIED_FULL_MOMENTUM_SUPG_V1: complete element-interior
        // residual with matching Picard carrier and matched 64-point tau.
        // P0 pressure has zero cell-interior gradient; no face jumps included.
        if(momentumFullSupg){
          HXT1_CUDA(cudaMemset(d_bad.p,0,sizeof(unsigned long long)));
          hxt5d::momentum_full_deviatoric_supg_rhs64<<<M.h.ntet,64>>>(
            d_pts.p,d_tc.p,d_tv.p,d_tg.p,
            d_u0.p,d_u1.p,d_u2.p,TS.k.p,TS.omega.p,
            d_rhs0.p,d_rhs1.p,d_rhs2.p,M.h.ntet,R,nu,
            sstG1.kFloor,sstG1.omegaFloor,stage33TauScale,stage33Magic,
            supgViscosityFdRel,convectionDivFactor,d_bad.p);
          HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
          unsigned long long fullBad=0;
          HXT1_CUDA(cudaMemcpy(&fullBad,d_bad.p,sizeof(fullBad),cudaMemcpyDeviceToHost));
          if(fullBad)throw std::runtime_error("Full momentum SUPG RHS invalid cell count="+
            std::to_string(fullBad));
          if(git==1 || git%sstG2PrintEvery==0)
            std::printf("NODALS_MOMENTUM_FULL_SUPG_RHS_ASSEMBLED it=%d cells=%llu "
              "divCarrierFactor=%.1f strong=FULL_DEVIATORIC_ELEMENT_INTERIOR "
              "nuGrad=FD_LOCAL_SST bf3Hessian=ANALYTIC pressure=P0_CELL_ZERO "
              "rhs=PICARD_DEFERRED status=PASS\n",
              git,(unsigned long long)M.h.ntet,convectionDivFactor);
        }
        predAuditMark(3);
        double s22d4dR0=0.,s22d4dR1=0.;
        if(stage22d4Live){
          hxt5d::s22d4d_pack<<<(nv+255)/256,256>>>(nv,d_u0.p,d_u1.p,d_u2.p,
            d_rhs0.p,d_rhs1.p,d_rhs2.p,d_s22Rhs0.p,d_s22Rhs1.p,d_s22Rhs2.p,
            d_clampXY.p,d_fixed.p,d_s22d4dX.p,d_s22d4dB.p);
          hxt5d::S22D4DOperator d4op{A,VC,A.val.p,d_s22ValY.p,d_valZ.p,
            d_clampXY.p,d_fixed.p,sstG1.tetWallPlans,sstG1.tetWallQuads,
            d_tv.p,d_tg.p,d_s22d4dCoeff.p,(int)BH.wallT.size(),s22d4Gamma,d_s22d4iScratch.p,stage22d4iEnable,sampledConductance,
            f1bFrozenWall?d_f1bWallCross.p:nullptr};
          // NODALS_EMAC_PICARD_V1: pressure-corrected Picard carrier for 9-block EMAC.
          d4op.set_emac_picard(momentumEmac,d_u0.p,d_u1.p,d_u2.p,M.h.ntet);
          d4op.set_implicit_dev(implicitDevStress,
            d_pts.p,d_tc.p,d_tv.p,d_tg.p,
            d_u0.p,d_u1.p,d_u2.p,TS.k.p,TS.omega.p,
            M.h.ntet,R,nu,sstG1.kFloor,sstG1.omegaFloor);
          d4op.set_option_a_shear(optionAShearActive,d_optionANu.p);
          if(stage22d4iEnable){
            hxt5d::s22d4i_Q_apply<<<(BH.wallT.size()+255)/256,256>>>(
               (int)BH.wallT.size(),sstG1.tetWallPlans,d_tv.p,
               d_s22d4dX.p,d_s22d4dX.p+nv,d_s22d4dX.p+2*(std::size_t)nv);
            hxt5d::s22d4i_Tt_apply<<<(BH.wallT.size()+255)/256,256>>>(
               (int)BH.wallT.size(),sstG1.tetWallPlans,d_tv.p,
               d_s22d4dB.p,d_s22d4dB.p+nv,d_s22d4dB.p+2*(std::size_t)nv);
          }
          if(f1bFrozenWall && f1bGpuGate && git==1){
            // Evaluate the FULL frozen coupled vector operator, including
            // diagonal CSR, BF3 T/Tt and the same fixed-DOF masks, on exactly
            // the same reduced-coordinate input. Only the off-wall cross action
            // differs. This gate runs once and is NOT enabled for timing.
            hxt5d::S22D4DOperator reference=d4op;
            reference.f1bFaceCoupling=nullptr;
            reference.apply(d_s22d4dX.p,d_f1bGateLegacy.p);
            d4op.apply(d_s22d4dX.p,d_f1bGateCached.p);
            hxt5d::f1b_difference<<<(3*nv+255)/256,256>>>(3*nv,
              d_f1bGateLegacy.p,d_f1bGateCached.p,d_f1bGateDiff.p);
            HXT1_CUDA(cudaGetLastError());
            const double normReference=stage22d4dRed.norm(d_f1bGateLegacy.p);
            const double normDiff=stage22d4dRed.norm(d_f1bGateDiff.p);
            const double relative=normDiff/std::max(normReference,1.e-300);
            std::printf("NODALS_F1B_GPU_OPERATOR_GATE it=%d relative=%.12e "
              "absolute=%.12e refNorm=%.12e tol=1e-10 status=%s\n",git,
              relative,normDiff,normReference,
              std::isfinite(relative)&&relative<=1.e-10?"PASS":"FAIL");
            if(!std::isfinite(relative)||relative>1.e-10)
              throw std::runtime_error("F1B GPU full coupled operator does not match legacy");
          }
          if(nodalsDiagnosticsDue(git) && s05b1Active)
            s22d4dR0=hxt5d::s05b1_full_residual(d4op,d_s22d4dB.p,d_s22d4dX.p,
              d_s05b1Ax.p,d_s05b1Residual.p,stage22d4dRed,true);
          else if(nodalsDiagnosticsDue(git)) {
            auto& d4work=hxt5e::get_workspace(3*nv,std::min(stage22d4dRestart,stage22d4dMaxIts));
            s22d4dR0=hxt5d::s22d4d_residual_norm(d4op,d_s22d4dB.p,d_s22d4dX.p,d4work,stage22d4dRed);
          }
          if(stage22d4dMode=="coupled"){
            hxt5d::S22D4DResult q{};
            if(s05b1Active){
              if(s05b3Active){
                q=hxt5d::s05b3_segregated(d4op,d_s22d4dX.p,d_s22d4dB.p,
                  d_s05b1Embed.p,d_s05b1Ax.p,d_s05b1Residual.p,
                  d_s05b1Backup.p,d_s05b1Rhs.p,d_s05b1Delta.p,
                  s05b3Blocks,s05b3InnerIts,s05b3InnerFgs,momOmega,s05b3Damping,
                  s05b3Mode=="bicgstab",s05b3Audit!=0,s05b3Require!=0,
                  momRtol,momAtol,stage22d4dRed,s05b1ScalarRed,git);
                if(!q.ok){
                  std::printf("STAGE27_S05B3_FALLBACK it=%d reason=PROJECTED_RESIDUAL "
                    "solver=COUPLED_FGMRES status=PASS\n",git);
                  q=hxt5d::s22d4d_block_fgmres(d4op,d_s22d4dX.p,d_s22d4dB.p,
                    stage22d4dRestart,stage22d4dMaxIts,momRtol,momAtol,
                    momFgsSweeps,momOmega,stage22d4dRed,git);
                }
              }else{
              q=hxt5d::s05b1_segregated_fgmres(d4op,d_s22d4dX.p,d_s22d4dB.p,
                d_s05b1Embed.p,d_s05b1Ax.p,d_s05b1Residual.p,
                d_s05b1Backup.p,d_s05b1Rhs.p,d_s05b1Delta.p,
                s05b1BlockMax,s05b1ComponentMax,s05b1Restart,
                s05b1PrecondSweeps,momOmega,momRtol,momAtol,
                stage22d4dRed,s05b1ScalarRed,git);
              if(!q.ok && s05b1Guarded){
                std::printf("STAGE27_S05B1_FALLBACK it=%d reason=PROJECTED_RESIDUAL "
                  "solver=COUPLED_FGMRES status=PASS\n",git);
                q=hxt5d::s22d4d_block_fgmres(d4op,d_s22d4dX.p,d_s22d4dB.p,
                  stage22d4dRestart,stage22d4dMaxIts,momRtol,momAtol,
                  momFgsSweeps,momOmega,stage22d4dRed,git);
              }
              }
            }else{
              predAuditMark(4);
              if(m3Active){
                if(m3Mode=="m4")
                  q=hxt5d::m4_fixed_hybrid_csr(d4op,d_s22d4dX.p,d_s22d4dB.p,
                    d_m3K.p,d_m3Temp.p,m3Sweeps,m3Omega,
                    d_m3BubbleFace.p,d_m3Bad.p,git,d_m4Coupled.p,
                    d_m3Check.p,d_m3CheckDiff.p,stage22d4dRed);
                else
                  q=hxt5d::m3_fixed_coupled_csr(d4op,d_s22d4dX.p,d_s22d4dB.p,
                     d_m3K.p,d_m3Temp.p,m3Sweeps,m3Omega,m3Mode=="sgs",
                     d_m3BubbleFace.p,d_m3Bad.p,git,
                     // NODALS_EMAC_M3_TRUE_CSR_RESIDUAL_DIAG_V1: first two diagnostic iterations only.
                     (nodalsDiagnosticsDue(git) && git<=2)?&stage22d4dRed:nullptr,
                     d_m3Check.p,d_m3CheckDiff.p);
                if(git==1 && nodalsDiagnosticsDue(git)){
                  // Optional assembly parity gate against the ORIGINAL exact
                  // matrix-free application. Once at startup, not per sweep.
                  d4op.apply(d_s22d4dX.p,d_m3Check.p);
                  hxt5d::m3_spmv<<<(nv+255)/256,256>>>(nv,A.nnz,A.row.p,A.col.p,
                    d_m3K.p,d_s22d4dX.p,d_m3CheckDiff.p,
                    d_clampXY.p,d_fixed.p);
                  hxt5d::m3_sub<<<(3*nv+255)/256,256>>>(3*nv,
                    d_m3Check.p,d_m3CheckDiff.p,d_m3CheckDiff.p);
                  const double difference=stage22d4dRed.norm(d_m3CheckDiff.p);
                  const double baseline=stage22d4dRed.norm(d_m3Check.p);
                  const double rel=difference/std::max(baseline,1e-300);
                  std::printf("NODALS_M3_OPERATOR_PARITY it=1 relative=%.12e "
                    "tol=1e-9 status=%s\n",rel,(std::isfinite(rel)&&rel<=1e-9)?"PASS":"FAIL");
                  if(!std::isfinite(rel)||rel>1e-9)
                    throw std::runtime_error("M3 exact CSR and BF3 reference operator disagree");
                }
              }else{
                q=hxt5d::s22d4d_block_fgmres(d4op,d_s22d4dX.p,d_s22d4dB.p,
                  stage22d4dRestart,stage22d4dMaxIts,momRtol,momAtol,
                  momFgsSweeps,momOmega,stage22d4dRed,git);
              }
              predAuditMark(5);
            }
            if(stage22d4iEnable)
              hxt5d::s22d4i_T_apply<<<(BH.wallT.size()+255)/256,256>>>(
                (int)BH.wallT.size(),sstG1.tetWallPlans,d_tv.p,
                d_s22d4dX.p,d_s22d4dX.p+nv,d_s22d4dX.p+2*(std::size_t)nv);
            hxt5d::s22d4d_unpack<<<(nv+255)/256,256>>>(nv,d_s22d4dX.p,d_u0.p,d_u1.p,d_u2.p);
            if(!q.ok && momCapPolicy=="fail")
              throw std::runtime_error("Stage22D4D coupled block FGMRES did not reach requested momentum rtol");
          }
        }
        hxt5b::H5B3MomentumResult mx{},my{},mz{};
        if(stage22d4dMode=="segregated"){
          mx=hxt5b_perf::momentum_solve(momWork,git,A,VC,A.val.p,d_rhs0.p,d_u0.p,momOmega,d_clampXY.p,momRtol,momAtol,momMax,momFgmresRestart,momFgsSweeps,ured,"x",momCapPolicy=="warn");
          my=hxt5b_perf::momentum_solve(momWork,git,A,VC,stage22Vector?d_s22ValY.p:A.val.p,d_rhs1.p,d_u1.p,momOmega,d_clampXY.p,momRtol,momAtol,momMax,momFgmresRestart,momFgsSweeps,ured,"y",momCapPolicy=="warn");
          mz=hxt5b_perf::momentum_solve(momWork,git,A,VC,d_valZ.p,d_rhs2.p,d_u2.p,momOmega,d_fixed.p,momRtol,momAtol,momMax,momFgmresRestart,momFgsSweeps,ured,"z",momCapPolicy=="warn");
          if((!mx.ok||!my.ok||!mz.ok)&&momCapPolicy=="fail")throw std::runtime_error("SST G2 momentum residual-drop target not reached");
        }
        predAuditMark(6);
        if(stage22d4Live){
          // Repack final predictor velocities. The true RHS is unchanged,
          // because the frozen old-velocity contributions are unchanged.
          hxt5d::s22d4d_pack<<<(nv+255)/256,256>>>(nv,d_u0.p,d_u1.p,d_u2.p,
            d_rhs0.p,d_rhs1.p,d_rhs2.p,d_s22Rhs0.p,d_s22Rhs1.p,d_s22Rhs2.p,
            d_clampXY.p,d_fixed.p,d_s22d4dX.p,d_s22d4dB.p);
          hxt5d::S22D4DOperator d4op{A,VC,A.val.p,d_s22ValY.p,d_valZ.p,
            d_clampXY.p,d_fixed.p,sstG1.tetWallPlans,sstG1.tetWallQuads,
            d_tv.p,d_tg.p,d_s22d4dCoeff.p,(int)BH.wallT.size(),s22d4Gamma,d_s22d4iScratch.p,stage22d4iEnable,sampledConductance,
            f1bFrozenWall?d_f1bWallCross.p:nullptr};
          // NODALS_EMAC_PICARD_V1: pressure-corrected Picard carrier for 9-block EMAC.
          d4op.set_emac_picard(momentumEmac,d_u0.p,d_u1.p,d_u2.p,M.h.ntet);
          d4op.set_implicit_dev(implicitDevStress,
            d_pts.p,d_tc.p,d_tv.p,d_tg.p,
            d_u0.p,d_u1.p,d_u2.p,TS.k.p,TS.omega.p,
            M.h.ntet,R,nu,sstG1.kFloor,sstG1.omegaFloor);
          if(stage22d4iEnable){
            hxt5d::s22d4i_Q_apply<<<(BH.wallT.size()+255)/256,256>>>(
               (int)BH.wallT.size(),sstG1.tetWallPlans,d_tv.p,
               d_s22d4dX.p,d_s22d4dX.p+nv,d_s22d4dX.p+2*(std::size_t)nv);
            hxt5d::s22d4i_Tt_apply<<<(BH.wallT.size()+255)/256,256>>>(
               (int)BH.wallT.size(),sstG1.tetWallPlans,d_tv.p,
               d_s22d4dB.p,d_s22d4dB.p+nv,d_s22d4dB.p+2*(std::size_t)nv);
          }
          if(nodalsDiagnosticsDue(git) && s05b1Active)
            s22d4dR1=hxt5d::s05b1_full_residual(d4op,d_s22d4dB.p,d_s22d4dX.p,
              d_s05b1Ax.p,d_s05b1Residual.p,stage22d4dRed,true);
          else if(nodalsDiagnosticsDue(git)) {
            auto& dw=hxt5e::get_workspace(3*nv,std::min(stage22d4dRestart,stage22d4dMaxIts));
            s22d4dR1=hxt5d::s22d4d_residual_norm(d4op,d_s22d4dB.p,d_s22d4dX.p,dw,stage22d4dRed);
          }
          if(nodalsDiagnosticsDue(git))
            std::printf("STAGE22D4D_FROZEN_VECTOR_RESIDUAL it=%d mode=%s gamma=%.8g startL2=%.12e finalL2=%.12e trueRelative=%.12e isFullLinearizedVectorOperator=1 schur=UNCHANGED status=PASS\n",
                 git,stage22d4dMode.c_str(),s22d4Gamma,s22d4dR0,s22d4dR1,
                 s22d4dR1/std::max(s22d4dR0,1.e-300));
        }
        if(stage22d4Live){
          const bool due=stage27AuditPolicy.due(git,sstG2Max,"PREDICTOR");
          if(due)stage22d4cAudit(git,"PREDICTOR");
          std::printf("STAGE27_AUDIT_SCHEDULE it=%d phase=PREDICTOR executed=%d mode=%s\n",
                      git,(int)due,stage27AuditPolicy.name());
        }
        predAuditMark(7);
        hxt4a::B_apply(B,d_u0.p,d_u1.p,d_u2.p,d_cont.p);hxt5b::addsrc<<<(np+TPB-1)/TPB,TPB>>>(np,d_contSource.p,d_cont.p);// NODALS_SST_COMPACT_CONT_ACTIVE_V2
        auto cstatsFull=
          hxt5b_compact::continuity_stats(
            np,d_cont.p,d_contStats);

        // A pinned-pressure formulation has one continuity row
        // removed from the active pressure equation, so audit that row
        // separately.  In natural-pressure mode pin == -1 and ALL
        // continuity rows are physical/active: never dereference
        // d_cont[-1].
        T pinContValue=T(0);
        if(pin >= 0){
          HXT1_CUDA(cudaMemcpy(
            &pinContValue,
            d_cont.p+pin,
            sizeof(T),
            cudaMemcpyDeviceToHost));
        }

        double contPinAbs=
          (pin >= 0)
            ? std::abs((double)pinContValue)
            : 0.0;

        // Pressure-reference equation is not an active
        // continuity equation in the pressure solve.
        hxt4a::pin_zero_kernel<T><<<1,1>>>(
          pin,d_cont.p);

        auto cstatsActive=
          hxt5b_compact::continuity_stats(
            np,d_cont.p,d_contStats);

        double cn=pred.norm(d_cont.p);

        if(!std::isfinite(cn) ||
           !std::isfinite(cstatsActive.localMax) ||
           !std::isfinite(cstatsFull.globalAbs) ||
           !std::isfinite(contPinAbs))
          throw std::runtime_error(
            "SST G2 nonfinite continuity");

        if(git==stage31Start+1){
          g2ContActiveL2Initial=cn;
          g2ContActiveMaxInitial=cstatsActive.localMax;
          g2ContGlobalInitial=cstatsFull.globalAbs;
          g2ContPinInitial=contPinAbs;
        }

        // This is exactly the initial pressure-equation
        // residual norm because the pin row has been removed.
        if(stage22d4i3Enable && (git<=2||git%100==0||nodalsDiagnosticsDue(git)))
          HXT1_CUDA(cudaMemcpy(d_i3contPre.p,d_cont.p,(std::size_t)np*sizeof(T),cudaMemcpyDeviceToDevice));
        const double pAbs0=cn;
        const bool podUsedThisSolve=pressurePOD.active();
        pressurePOD.begin_solve(git);
        HXT1_CUDA(cudaMemset(d_dp.p,0,np*sizeof(T)));hxt4a::CGResult pr;
        predAuditMark(8); // Start PCG (continuity RHS and stats have finished).
        if(stage22d4i3Enable){
          if(pressureSolver!="pcg_amg")
            throw std::runtime_error("I3A supports only pcg_amg pressure solver");
          if(pressurePOD.enabled())
            throw std::runtime_error("I3A requires pressure POD disabled");
          pr=hxt4c::pcg_amg(schurI3,PC,d_cont.p,d_dp.p,np,
            pRtol,pAtol,pMax,WP,pin);
        }else if(pressureSolver=="pcg_amg")pr=(amgFineMode=="matrixfree"
          ? hxt4c::pcg_amg_mf_fine(
              Sg2,PC,d_sd.p,
              d_cont.p,d_dp.p,np,
              pRtol,pAtol,pMax,WP,pin)
          : hxt4c::pcg_amg(
              Sg2,PC,
              d_cont.p,d_dp.p,np,
              pRtol,pAtol,pMax,WP,pin,
              pressurePOD.enabled()?&pressurePOD:nullptr));else pr=pcg_prod(Sg2,d_cont.p,d_dp.p,d_isd.p,np,pRtol,pAtol,pMax,WP,pin);
        predAuditMark(9); // End PCG; next interval is post-PCG bookkeeping.
        if(!pr.ok){
          std::printf(
            "NODALS_SST_G2_PRESSURE_FAIL "
            "git=%d solver=%s pCG=%d pRel=%.12e "
            "pAbs0=%.12e pAbsFinal=%.12e "
            "pRtol=%.12e pAtol=%.12e pMax=%d status=FAIL\\n",
            git,pressureSolver.c_str(),pr.its,pr.rel,
            pAbs0,pAbs0*pr.rel,
            pRtol,pAtol,pMax);
          throw std::runtime_error("SST G2 pressure solve failed");
        }
        pressurePOD.note_solve(git,pr.its,podUsedThisSolve);
        pressurePOD.capture_after_solve(d_dp.p,git);

        // Adaptive fixed-Krylov-tail learning: after a complete residual batch
        // is available, build a CURRENT-A-orthogonal tentative block and test
        // it on this exact same pressure system.  The just-completed solve is
        // the old-basis baseline; the probe uses a separate x=0 buffer and
        // never changes the physical pressure correction d_dp.
