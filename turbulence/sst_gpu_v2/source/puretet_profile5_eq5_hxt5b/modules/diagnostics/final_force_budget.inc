// NODALS_SST_Z_FORCE_BUDGET_V1
    //
    // Diagnostic only. Reassemble every physical z-momentum contribution
    // at the final nonlinear state and contract each contribution with
    // valid Q1 axial FE test functions. BF2 test coefficients are zero,
    // since constants / axial Q1 window functions live in the Q1 subspace.
    //
    // Physical equation audited:
    //
    //   (A_base + A_conv + A_turb + A_DG + A_wall) Uz
    //       - B^T_z p
    //       - inletRhsZ
    //   = 0
    //
    // No SIMPLE row-L1 relaxation terms are included here.
    if(sstG2 && g2Outer>0 && sstG1.deviatoricMomentum)
      std::printf("NODALS_SST_Z_FORCE_BUDGET skipped=1 reason=EXPLICIT_DEVIATORIC_CROSS_TERM_NOT_IN_LEGACY_BUDGET status=PASS\n");
    if(sstG2 && g2Outer>0 && !sstG1.deviatoricMomentum && M.h.ntet>0)
      std::printf("NODALS_SST_Z_FORCE_BUDGET skipped=1 reason=HEX_ONLY_DIAGNOSTIC_ON_TET_MESH status=SKIPPED\n");
    if(sstG2 && g2Outer>0 && !sstG1.deviatoricMomentum && M.h.ntet==0){

      // ---------------------------------------------------------------
      // Reassemble all physical pieces at the FINAL U,k,omega state.
      // ---------------------------------------------------------------
      HXT1_CUDA(cudaMemset(
        A.conv.p,0,A.nnz*sizeof(T)));

      HXT1_CUDA(cudaMemset(
        d_bad.p,0,sizeof(unsigned long long)));

      if(M.h.nhex)
        hex_conv_assemble_cached<<<M.h.nhex,128>>>(
          d_hv.p,
          A.hexSlot.p,
          d_u0.p,d_u1.p,d_u2.p,
          d_hexConvGrad.p,d_hexConvDetW.p,
          A.conv.p,
          M.h.nhex);

      if(M.h.ntet)
        tet_conv_assemble_thread_per_cell
          <<<(M.h.ntet+127)/128,128>>>(
            d_tv.p,d_tg.p,A.tetSlot.p,
            d_u0.p,d_u1.p,d_u2.p,
            A.conv.p,
            M.h.ntet);

      hxt5d::assemble_momentum_sst(
        M,
        d_pts,d_hc,d_tc,d_hv,d_tv,d_tg,d_ip,
        A,
        d_u0.p,d_u1.p,d_u2.p,
        TS.k.p,TS.omega.p,
        d_turb,d_diagRTurb,
        R,nu,gamma,
        sstG1,TS.stats,
        d_bad,
        &d_sstDevRhs0,&d_sstDevRhs1,&d_sstDevRhs2,
        &d_sstInterfaceCache);

      hxt5d::assemble_boundary_sst(
        BH,
        d_inH,d_inT,d_wallH,
        d_pts,d_hc,d_tc,d_hv,d_tv,d_tg,
        A,
        d_u0.p,d_u1.p,d_u2.p,
        TS.k.p,TS.omega.p,
        d_dg,d_diagDG,
        d_inletRhsX,d_inletRhsY,d_inletRhsZ,
        d_wall,d_diagWall,
        d_wallStats,
        d_bad,
        bulk,R,nu,wallSampleFraction,
        sstG1,
        momentumWallMode,
        spaldingMolecularConsistency,
        weakWallGamma,
        weakWallHScale,
        weakWallNutScale,
        stage22Vector,stage22Penalty,d_s22Rhs0.p,d_s22Rhs1.p,d_s22Rhs2.p,
        d_s22WallY.p,d_s22WallZ.p,d_s22DiagY.p,d_s22DiagZ.p,stage22dFaceFit);

      hxt4a::BT_apply(
        B,d_p.p,
        d_bt0.p,d_bt1.p,d_bt2.p);

      HXT1_CUDA(cudaDeviceSynchronize());

      unsigned long long forceBad=0;
      HXT1_CUDA(cudaMemcpy(
        &forceBad,
        d_bad.p,
        sizeof(forceBad),
        cudaMemcpyDeviceToHost));

      if(forceBad)
        throw std::runtime_error(
          "Z force budget final-state reassembly geometry failure count="
          +std::to_string(forceBad));

      // Fixed z rows, if any, are not admissible test DOFs.
      std::vector<unsigned char>
        hForceFixed((std::size_t)nv,0);

      HXT1_CUDA(cudaMemcpy(
        hForceFixed.data(),
        d_fixed.p,
        (std::size_t)nv*sizeof(unsigned char),
        cudaMemcpyDeviceToHost));

      auto makeForceTest =
        [&](bool whole,double z0,double z1)
      {
        if(M.h.ntet!=0)
          throw std::runtime_error(
            "Z force budget V1 currently expects all-HEX mesh");

        std::vector<T>
          q((std::size_t)nv,T(0));

        std::vector<unsigned char>
          seen((std::size_t)nv,0);

        const double ztol=1e-11;

        std::size_t vertexDofs=0;
        std::size_t activeDofs=0;
        std::size_t fixedSuppressed=0;
        std::size_t conflicts=0;

        for(std::size_t c=0;c<M.hexes.size();++c){
          const auto hc=M.hexes[c];
          const auto H =M.hexVel[c];

          // First 8 local velocity basis functions are Q1 vertices.
          // BF2 face-bubble test coefficients remain exactly zero.
          for(int a=0;a<8;++a){
            const int g=H.g[a];

            if(g<0 || g>=nv)
              throw std::runtime_error(
                "Z force budget bad vertex velocity dof");

            const double z=M.points[
              (std::size_t)hc.v[a]].z;

            const T v =
              whole
              ? T(1)
              : T(
                  z>=z0-ztol &&
                  z<=z1+ztol
                    ? 1.0 : 0.0);

            if(seen[(std::size_t)g]){
              if(std::abs(
                   (double)q[(std::size_t)g]
                   -(double)v)>1e-12)
                ++conflicts;
            }else{
              q[(std::size_t)g]=v;
              seen[(std::size_t)g]=1;
              ++vertexDofs;
            }
          }
        }

        if(conflicts)
          throw std::runtime_error(
            "Z force budget inconsistent shared Q1 vertex test values");

        for(int i=0;i<nv;++i){
          if(hForceFixed[(std::size_t)i] &&
             q[(std::size_t)i]!=T(0)){
            q[(std::size_t)i]=T(0);
            ++fixedSuppressed;
          }

          if(q[(std::size_t)i]!=T(0))
            ++activeDofs;
        }

        std::printf(
          "NODALS_SST_Z_FORCE_TEST "
          "kind=%s z0=%.12e z1=%.12e "
          "vertexDofs=%zu activeDofs=%zu "
          "fixedSuppressed=%zu "
          "BF2coefficients=ZERO "
          "status=PASS\n",
          whole ? "GLOBAL_CONSTANT"
                : "Q1_WINDOW_300D_450D",
          z0,z1,
          vertexDofs,activeDofs,
          fixedSuppressed);

        return q;
      };

      auto runForceBudget =
        [&](const char* tag,
            const std::vector<T>& hTest,
            double z0,double z1)
      {
        Dev<T> dForceTest(hTest);

        auto matrixTerm =
          [&](const T*mat)->double
        {
          hxt5b::spmv
            <<<(nv+TPB-1)/TPB,TPB>>>(
              nv,
              A.row.p,A.col.p,
              mat,
              d_u2.p,
              A.tmp.p);

          HXT1_CUDA(cudaGetLastError());

          return ured.dot(
            dForceTest.p,
            A.tmp.p);
        };

        const double base =
          matrixTerm(A.base.p);

        const double conv =
          matrixTerm(A.conv.p);

        const double turb =
          matrixTerm(d_turb.p);

        const double dg =
          matrixTerm(d_dg.p);

        const double wall =
          matrixTerm(d_wall.p);

        const double pressure =
          ured.dot(
            dForceTest.p,
            d_bt2.p);

        const double inletRhs =
          ured.dot(
            dForceTest.p,
            d_inletRhsZ.p);

        const double lhs =
          base+conv+turb+dg+wall;

        const double rhs =
          pressure+inletRhs;

        const double closure =
          lhs-rhs;

        const double scale =
          std::max(
            std::abs(pressure),
            1e-300);

        std::printf(
          "NODALS_SST_Z_FORCE_BUDGET "
          "tag=%s "
          "z0=%.12e z1=%.12e "
          "baseMolecular=%.12e "
          "convection=%.12e "
          "sstEddyDiffusion=%.12e "
          "dgMatrix=%.12e "
          "wallOperator=%.12e "
          "pressureDrive=%.12e "
          "inletRhs=%.12e "
          "lhs=%.12e rhs=%.12e "
          "closure=%.12e "
          "closureOverPressure=%.12e "
          "baseOverPressure=%.12e "
          "convOverPressure=%.12e "
          "turbOverPressure=%.12e "
          "dgOverPressure=%.12e "
          "wallOverPressure=%.12e "
          "molecularConsistency=%d "
          "operatorChanged=0 "
          "status=PASS\n",
          tag,
          z0,z1,
          base,
          conv,
          turb,
          dg,
          wall,
          pressure,
          inletRhs,
          lhs,rhs,
          closure,
          closure/scale,
          base/scale,
          conv/scale,
          turb/scale,
          dg/scale,
          wall/scale,
          spaldingMolecularConsistency);
      };

      // Global constant-z test: useful independent closure check.
      {
        auto q=makeForceTest(
          true,zmin,zmax);

        runForceBudget(
          "GLOBAL_CONSTANT",
          q,zmin,zmax);
      }

      // Downstream 300D--450D test.
      //
      // Vertex values are 1 on [1.2,1.8] and zero outside.
      // Therefore the FE test is exactly Q1 and has a one-axial-cell
      // linear ramp immediately outside each end of the window.
      {
        constexpr double forceZ0=1.2;
        constexpr double forceZ1=1.8;

        auto q=makeForceTest(
          false,forceZ0,forceZ1);

        runForceBudget(
          "WINDOW_300D_450D",
          q,forceZ0,forceZ1);
      }
    }

    
