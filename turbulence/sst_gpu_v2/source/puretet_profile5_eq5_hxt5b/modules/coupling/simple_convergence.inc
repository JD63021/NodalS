        const double contRel=
          cn/std::max(g2ContActiveL2Initial,1e-300);

        const bool massNow=nodals::coupling::mass_ok(cn,cstatsActive.localMax,sstG2ContAtol);

        const bool turbNow=nodals::coupling::turbulence_ok(
          {tr.kRel,tr.omegaRel,tr.nutRel,tr.stats.kFloorFrac,tr.stats.omegaFloorFrac},simpleTol);

        const double spanF=relSpan(hf);
        const double spanY=relSpan(hy);
        const double spanDP=relSpan(hdp);

        const bool plateauNow=nodals::coupling::plateau_ok(
          (int)hf.size(),sstG2PlateauSamples,spanF,spanY,spanDP,sstG2PlateauTol);

        if(stage22d4dMode=="coupled"){
          std::printf("NODALS_SST_COMPACT_COUPLED it=%d "
            "fullVectorInitial=%.3e fullVectorFinal=%.3e fullVectorRel=%.3e "
            "k=%.3e->%.3e omega=%.3e->%.3e "
            "p=%.3e->%.3e pIts=%d continuity=%.3e status=PASS\n",
            git,s22d4dR0,s22d4dR1,
            s22d4dR1/std::max(s22d4dR0,1.e-300),
            tr.kLinear.r0,tr.kLinear.rn,
            tr.omegaLinear.r0,tr.omegaLinear.rn,
            pAbs0,pAbsFinal,pr.its,contRel);
        } else {
        std::printf(
          "NODALS_SST_COMPACT_ABS "
          "it=%d "
          "Ux=%.3e->%.3e "
          "Uy=%.3e->%.3e "
          "Uz=%.3e->%.3e "
          "k=%.3e->%.3e "
          "omega=%.3e->%.3e "
          "p=%.3e->%.3e "
          "pIts=%d\n",
          git,
          mx.initial,mx.final,
          my.initial,my.final,
          mz.initial,mz.final,
          tr.kLinear.r0,tr.kLinear.rn,
          tr.omegaLinear.r0,tr.omegaLinear.rn,
          pAbs0,pAbsFinal,pr.its);

        std::printf(
          "NODALS_SST_COMPACT_REL "
          "it=%d "
          "Ux=%.3e Uy=%.3e Uz=%.3e "
          "k=%.3e omega=%.3e "
          "continuity=%.3e\n",
          git,
          mx.rel,my.rel,mz.rel,
          tr.kLinear.rel,tr.omegaLinear.rel,
          contRel);
        }
        std::printf(
          "NODALS_SST_COMPACT_CONT "
          "it=%d "
          "activeL2=%.3e->%.3e "
          "activeMax=%.3e->%.3e "
          "pinResidual=%.3e->%.3e "
          "globalAbs=%.3e->%.3e "
          "tol=%.3e "
          "massOK=%d\n",
          git,
          g2ContActiveL2Initial,cn,
          g2ContActiveMaxInitial,cstatsActive.localMax,
          g2ContPinInitial,contPinAbs,
          g2ContGlobalInitial,cstatsFull.globalAbs,
          sstG2ContAtol,
          (int)massNow);

        std::printf(
          "NODALS_SST_COMPACT_OUTER "
          "it=%d "
          "dK=%.3e dOmega=%.3e dNut=%.3e "
          "turbTol=%.3e "
          "massOK=%d turbOK=%d plateauOK=%d "
          "plateau=%zu/%d "
          "converged=%d\n",
          git,
          tr.kRel,tr.omegaRel,tr.nutRel,
          simpleTol,
          (int)massNow,(int)turbNow,(int)plateauNow,
          hf.size(),sstG2PlateauSamples,
          (int)g2Converged);

        if(sample){
          std::printf(
            "NODALS_SST_COMPACT_PHYS "
            "it=%d "
            "wallYPlus=%.6e "
            "wallDarcy=%.6e "
            "pressureDrop=%.6e "
            "plateauSpan=[%.3e,%.3e,%.3e]\n",
            git,finalY,finalF,finalDp,
            spanY,spanF,spanDP);
        }

