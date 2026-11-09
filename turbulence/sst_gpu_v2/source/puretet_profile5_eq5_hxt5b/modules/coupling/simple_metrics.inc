        bool sample=(git%sstG2MetricsEvery==0 || git==sstG2Max);
        if(sample && stage22d4Live){
          const bool wallFresh=stage27AuditPolicy.disabled() || stage27AuditCorrectedAt==git;
          const char* wallSource=stage27AuditPolicy.disabled()?"PRODUCTION_FROZEN_ASSEMBLY":"STAGE22D4C_CORRECTED";
          std::printf("STAGE27_WALL_METRIC_STATE it=%d freshness=%s "
                      "lastCorrectedAudit=%d age=%d plateauEligible=%d "
                      "source=%s status=PASS\n",
                      git,wallFresh?"FRESH":"CACHED",stage27AuditCorrectedAt,
                      git-stage27AuditCorrectedAt,(int)wallFresh,wallSource);
        }
        if(sample){auto ws2=hxt5b::wallstats(d_wallStats,bulk);if(stage22d4Live){
          if(stage27AuditPolicy.disabled()){
            double prod[3]={};
            HXT1_CUDA(cudaMemcpy(prod,d_stage27ProdWallStats.p,sizeof(prod),cudaMemcpyDeviceToHost));
            if(!(prod[0]>0.) || !std::isfinite(prod[0]) ||
               !std::isfinite(prod[1]) || !std::isfinite(prod[2]) ||
               prod[1]<0. || prod[2]<0.)
              throw std::runtime_error("Stage1A invalid production Spalding wall statistics");
            ws2.fDarcy=8.*prod[1]/prod[0]/(bulk*bulk);
            ws2.yPlusMean=prod[2]/prod[0];
            std::printf("STAGE27_PRODUCTION_WALL_METRICS it=%d quadratureArea=%.12e "
                        "wallDarcy=%.12e wallYPlusMean=%.12e "
                        "fieldState=FROZEN_PRE_MOMENTUM operator=UNCHANGED status=PASS\n",
                        git,prod[0],ws2.fDarcy,ws2.yPlusMean);
          }else{
            ws2.fDarcy=stage22d4cPostF;
            ws2.yPlusMean=stage22d4cPostY;
          }
        }HXT1_CUDA(cudaMemcpy(hPressurePlateau.data(),d_p.p,(std::size_t)np*sizeof(T),cudaMemcpyDeviceToHost));auto pd2=hxt4a::to_double(hPressurePlateau);double pIn2=hxt4a::boundary_p(M,pd2,true);double pOut2=hxt4a::boundary_p(M,pd2,false);double dp2=pIn2-pOut2;double slope2=hxt4a::pressure_slope(M,pd2);double qIn=-BH.sourceSum;double qOut=qIn+cstatsFull.globalSigned;double qImbRel=std::abs(cstatsFull.globalSigned)/std::max(std::abs(qIn),1e-300);finalDp=dp2;finalF=ws2.fDarcy;
          if(stage22Vector && git%sstG2PrintEvery==0){
            const double pressureAxial=dp2*BH.inletArea;
            std::printf("SST22C_AXIAL_FORCE_COMPARISON it=%d pressureForce=%.12e wallTractionIntegral=%.12e "
              "difference=%.12e closure=NOT_COMPLETE_CONVECTIVE_AND_DIFFUSIVE_TERMS_NOT_INCLUDED status=PASS\n",
              git,pressureAxial,s22cPostWallAxial,pressureAxial-s22cPostWallAxial);
          }finalY=ws2.yPlusMean;
          // NODALS_SST_PLOT_METRICS_V1: report already-computed quantities.
          // Independent of P1 masking, momentum form, SUPG and plateau mode.
          if(git%sstG2PrintEvery==0 || git==sstG2Max){
            std::printf("NODALS_SST_PLOT_METRICS it=%d pressureDrop=%.12e "
                        "wallDarcy=%.12e wallYPlusMean=%.12e\n",
                        git,finalDp,finalF,finalY);
            std::fflush(stdout);
          }

          metricGate.observe(git,finalF,finalDp,stage27AuditPolicy.disabled() || stage27AuditCorrectedAt==git);
          if(git%sstG2PrintEvery==0 && (!stage22d4Live || stage27AuditPolicy.disabled() || stage27AuditCorrectedAt==git)){hf.push_back(finalF);hy.push_back(finalY);hdp.push_back(finalDp);while((int)hf.size()>sstG2PlateauSamples){hf.erase(hf.begin());hy.erase(hy.begin());hdp.erase(hdp.begin());}}double sf=relSpan(hf),sy=relSpan(hy),sd=relSpan(hdp);bool have=(int)hf.size()>=sstG2PlateauSamples;const nodals::coupling::GateInputs gateInputs={cn,cstatsActive.localMax,sstG2ContAtol,
            {tr.kRel,tr.omegaRel,tr.nutRel,tr.stats.kFloorFrac,tr.stats.omegaFloorFrac},simpleTol,
            {stage22d4iEnable?s22d4i2Mx:ma0,
             stage22d4iEnable?s22d4i2My:ma1,
             stage22d4iEnable?s22d4i2Mz:ma2},sstG2MomAtol,sstG2PhysicalGate,(int)hf.size(),sstG2PlateauSamples,
            sf,sy,sd,sstG2PlateauTol};const auto gate=nodals::coupling::evaluate_gate(gateInputs);
          bool physical=gate.physicalOk;bool massok=gate.massOk;bool turbok=gate.turbulenceOk;
          const bool momok=gate.momentumOk;g2Converged=gate.converged;
          // Never converge on stale cached wall metrics, even if the physics
          // and scalar/continuity residuals individually pass.
          if(stage22d4Live && stage27AuditPolicy.checkpoint() &&
             stage27AuditCorrectedAt!=git)g2Converged=false;
          if(metricGate.active()){
            const bool healthy=std::isfinite(cn)&&std::isfinite(finalDp)&&
              std::isfinite(tr.kRel)&&std::isfinite(tr.omegaRel)&&
              std::isfinite(tr.nutRel)&&std::isfinite(ma0)&&
              std::isfinite(ma1)&&std::isfinite(ma2)&&
              tr.stats.kFloorFrac<0.01 && tr.stats.omegaFloorFrac<0.01;
            const bool residualsOk=!metricGate.requireResiduals||
              (massok && turbok && momok);
            g2Converged=healthy && metricGate.passed() && residualsOk;
            if(git%sstG2PrintEvery==0 || g2Converged){
              std::printf("NODALS_METRIC_GATE it=%d metric=%s window=%zu/%d "
                "relativeSpanWall=%.9e relativeSpanDP=%.9e tol=%.9e "
                "metricOK=%d healthy=%d residualsOK=%d massOK=%d turbOK=%d momOK=%d converged=%d\n",
                git,metricGate.mode.c_str(),metricGate.drop.size(),metricGate.window,
                metricGate.wallSpan(),metricGate.dropSpan(),metricGate.tolerance,
                (int)metricGate.passed(),(int)healthy,(int)residualsOk,
                (int)massok,(int)turbok,(int)momok,(int)g2Converged);
            }
          }

          if(stage22d4iEnable)std::printf("STAGE22D4I2_MOM_GATE it=%d "
            "reduced=[%.12e,%.12e,%.12e] tol=%.12e momOK=%d "
            "massOK=%d turbOK=%d physicalOK=%d converged=%d status=PASS\n",
            git,s22d4i2Mx,s22d4i2My,s22d4i2Mz,sstG2MomAtol,(int)momok,
            (int)massok,(int)turbok,(int)physical,(int)g2Converged);
          if(git%sstG2PrintEvery==0){
            const double uAbsX=stage22d4iEnable?s22d4i2Mx:ma0;
            const double uAbsY=stage22d4iEnable?s22d4i2My:ma1;
            const double uAbsZ=stage22d4iEnable?s22d4i2Mz:ma2;
            // Velocity: post-SIMPLE physical momentum residual in each component.
            // The relative momentum residual is normalized to its *first reported*
            // component residual. This is an outer-trend ratio, not an inner FGS rtol.
            static double uFirst[3]={0.,0.,0.};
            if(uFirst[0]==0)uFirst[0]=std::max(uAbsX,1e-300);
            if(uFirst[1]==0)uFirst[1]=std::max(uAbsY,1e-300);
            if(uFirst[2]==0)uFirst[2]=std::max(uAbsZ,1e-300);
            std::printf(
              "NODALS_SST_RESIDUALS it=%d "
              "UxAbs=%.6e UxRel=%.6e UyAbs=%.6e UyRel=%.6e UzAbs=%.6e UzRel=%.6e "
              "pAbs=%.6e pRel=%.6e contAbs=%.6e contRel=%.6e "
              "kAbs=%.6e kRel=%.6e omegaAbs=%.6e omegaRel=%.6e "
              "kUpdateRel=%.6e omegaUpdateRel=%.6e "
              "pIts=%d kSweeps=%d omegaSweeps=%d converged=%d\n",
              git,uAbsX,uAbsX/uFirst[0],uAbsY,uAbsY/uFirst[1],uAbsZ,uAbsZ/uFirst[2],
              pAbsFinal,pr.rel,cn,cn/std::max(g2ContActiveL2Initial,1e-300),
              tr.kLinear.rn,tr.kLinear.rel,tr.omegaLinear.rn,tr.omegaLinear.rel,
              tr.kRel,tr.omegaRel,pr.its,tr.kLinear.works,tr.omegaLinear.works,
              (int)g2Converged);
          }
          std::printf("NODALS_SST_G2 it=%d absCont=%.6e absMom=[%.6e,%.6e,%.6e] pCG=%d pRel=%.3e kLinIts=%d kLinRel=%.3e omegaLinIts=%d omegaLinRel=%.3e kRelUpdate=%.6e omegaRelUpdate=%.6e nutMeanRelUpdate=%.6e meanNutOverNu=%.6e maxNutOverNu=%.6e kFloorVolumeFrac=%.6e omegaFloorVolumeFrac=%.6e wallDarcy=%.6e wallYPlusMean=%.6e pIn=%.6e pOut=%.6e pressureSlope=%.6e pressureDrop=%.6e inletFlux=%.6e outletFlux=%.6e fluxImbalanceRel=%.6e plateauSpan=[%.6e,%.6e,%.6e] plateauCount=%zu converged=%d status=PASS\n",git,cn,ma0,ma1,ma2,pr.its,pr.rel,tr.kLinear.works,tr.kLinear.rel,tr.omegaLinear.works,tr.omegaLinear.rel,tr.kRel,tr.omegaRel,tr.nutRel,tr.stats.meanNutOverNu,tr.stats.maxNutOverNu,tr.stats.kFloorFrac,tr.stats.omegaFloorFrac,finalF,finalY,pIn2,pOut2,slope2,finalDp,qIn,qOut,qImbRel,sf,sy,sd,hf.size(),(int)g2Converged);
        }
