        if(stage22d4iEnable){
          hxt5d::s22d4i_T_apply<<<(BH.wallT.size()+255)/256,256>>>(
            (int)BH.wallT.size(),sstG1.tetWallPlans,d_tv.p,d_u0.p,d_u1.p,d_u2.p);
          HXT1_CUDA(cudaGetLastError());
        }
        if(stage22d4iEnable && (git%sstG2PrintEvery==0 || nodalsDiagnosticsDue(git) || sstG2MomAtol>0)){
          // Snapshot constrained velocity only when post-SIMPLE diagnostic is due.
          const std::size_t nbyte=(std::size_t)nv*sizeof(T);
          HXT1_CUDA(cudaMemcpy(d_s22d4i2Old.p,d_u0.p,nbyte,cudaMemcpyDeviceToDevice));
          HXT1_CUDA(cudaMemcpy(d_s22d4i2Old.p+nv,d_u1.p,nbyte,cudaMemcpyDeviceToDevice));
          HXT1_CUDA(cudaMemcpy(d_s22d4i2Old.p+2*(std::size_t)nv,d_u2.p,nbyte,cudaMemcpyDeviceToDevice));
        }
        if(optionAShearActive){
          HXT1_CUDA(cudaMemset(d_optionANu.p,0,(std::size_t)M.h.ntet*sizeof(T)));
          HXT1_CUDA(cudaMemset(d_optionAStats.p,0,3*sizeof(double)));
          HXT1_CUDA(cudaMemset(d_bad.p,0,sizeof(unsigned long long)));
          hxt5d::f2_face_averaged_spalding<<<(BH.wallT.size()+127)/128,128>>>(
              sstG1.tetWallPlans,(int)BH.wallT.size(),sstG1.tetWallQuads,
              sstG1.profileTetMatchPoints,d_tv.p,
              d_u0.p,d_u1.p,d_u2.p,nu,d_f2FaceFit.p,d_bad.p);
          HXT1_CUDA(cudaGetLastError());
          hxt5d::option_a_build_wall_nu<<<(BH.wallT.size()+127)/128,128>>>(
              sstG1.tetWallPlans,(int)BH.wallT.size(),
              d_tv.p,d_tg.p,d_u0.p,d_u1.p,d_u2.p,
              d_f2FaceFit.p,d_optionANu.p,nu,d_optionAStats.p,d_bad.p);
          HXT1_CUDA(cudaGetLastError());
          HXT1_CUDA(cudaDeviceSynchronize());
          unsigned long long wallBad=0;
          HXT1_CUDA(cudaMemcpy(&wallBad,d_bad.p,sizeof(wallBad),cudaMemcpyDeviceToHost));
          if(wallBad)throw std::runtime_error("OPTION_A invalid Spalding/normal-gradient wall face count="+std::to_string(wallBad));
          if(git==1 || git%sstG2PrintEvery==0){
            double st[3]={};
            HXT1_CUDA(cudaMemcpy(st,d_optionAStats.p,sizeof(st),cudaMemcpyDeviceToHost));
            if(!(st[0]>0.))throw std::runtime_error("OPTION_A zero sampled wall area");
            std::printf("NODALS_OPTION_A_WALL it=%d wallFaces=%zu wallArea=%.10e meanUtau2=%.10e meanNuWallOverNu=%.10e clipMolecular=ON model=REPLACE_CELL_NU status=PASS\n",
              git,BH.wallT.size(),st[0],st[1]/st[0],st[2]/st[0]);
          }
        }
        hxt5d::assemble_momentum_sst(M,d_pts,d_hc,d_tc,d_hv,d_tv,d_tg,d_ip,A,d_u0.p,d_u1.p,d_u2.p,TS.k.p,TS.omega.p,d_turb,d_diagRTurb,R,nu,gamma,sstG1,TS.stats,d_bad,&d_sstDevRhs0,&d_sstDevRhs1,&d_sstDevRhs2,&d_sstInterfaceCache
#if defined(NODALS_P2_STAGE4_COUPLED)
          ,p2MomentumLive?p2Stage4->d_tets.p:nullptr,
          p2MomentumLive?p2Stage4->k.p:nullptr,
          p2MomentumLive?p2Stage4->w.p:nullptr
#endif
        );
#if defined(NODALS_P2_STAGE4_COUPLED)
        if(p2Stage4Active && git==stage31Start+2){
          Dev<T> q1ProjectedTurb(std::vector<T>((std::size_t)A.nnz,T(0)));
          Dev<T> q1ProjectedDiag(std::vector<T>((std::size_t)nv,T(0)));
          hxt5d::sst_mom_tet<<<M.h.ntet,64>>>(d_pts.p,d_tc.p,d_tv.p,d_tg.p,
            A.tetSlot.p,d_u0.p,d_u1.p,d_u2.p,TS.k.p,TS.omega.p,
            q1ProjectedTurb.p,q1ProjectedDiag.p,M.h.ntet,R,nu,
            sstG1.kFloor,sstG1.omegaFloor);
          HXT1_CUDA(cudaGetLastError());
          Dev<double> cmp(std::vector<double>(3,0.0));
          hxt5d::p2_stage4_diff<<<(A.nnz+255)/256,256>>>(
            (std::size_t)A.nnz,d_turb.p,q1ProjectedTurb.p,cmp.p);
          HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
          double h[3]={};HXT1_CUDA(cudaMemcpy(h,cmp.p,sizeof(h),cudaMemcpyDeviceToHost));
          const double rel=h[0]/std::max(h[1],1e-300);
          if(!(std::isfinite(rel) && h[1]>0 && rel>1e-14) || h[2]>0)
            throw std::runtime_error("P2_STAGE4: no finite edge-mode feedback in physical momentum volume");
          std::printf("P2_STAGE4_MOMENTUM_APPLIED=PASS it=%d scope=PHYSICAL_CSR_VOLUME relativeToProjectedP1=%.12e l1P2=%.12e invalid=%g oldProjection=VERTICES_ONLY\n",
            git,rel,h[1],h[2]);
          std::fflush(stdout);
        }
#endif
        // Full SUPG implicit part (a.grad v, tau a.grad u); always paired with
        // complete strong-residual Picard RHS in simple_pressure_predictor.inc.
        if(momentumFullSupg){ // NODALS_UNIFIED_FULL_MOMENTUM_SUPG_V1
          if(!M.h.ntet || M.h.nhex || M.h.ninterface)
            throw std::runtime_error("Full momentum SUPG requires pure tetrahedra");
          hxt5d::stage33a_momentum_supg64<<<(M.h.ntet+7)/8,256>>>(
            d_pts.p,d_tc.p,d_tv.p,d_tg.p,A.tetSlot.p,
            d_u0.p,d_u1.p,d_u2.p,TS.k.p,TS.omega.p,
            d_turb.p,d_diagRTurb.p,M.h.ntet,R,nu,
            sstG1.kFloor,sstG1.omegaFloor,stage33TauScale,stage33Magic,
            1,d_bad.p);
          HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
          unsigned long long supgBad=0;
          HXT1_CUDA(cudaMemcpy(&supgBad,d_bad.p,sizeof(supgBad),cudaMemcpyDeviceToHost));
          if(supgBad)throw std::runtime_error("Full momentum SUPG LHS invalid cell count="+
            std::to_string(supgBad));
          if(git<=2||git%sstG2PrintEvery==0)
            std::printf("NODALS_MOMENTUM_FULL_SUPG_LHS_ASSEMBLED it=%d cells=%llu "
              "tauScale=%.8g quadrature=64 pairedStrongRhs=YES status=PASS\n",
              git,(unsigned long long)M.h.ntet,stage33TauScale);
        }
        hxt5d::assemble_boundary_sst(BH,d_inH,d_inT,d_wallH,d_pts,d_hc,d_tc,d_hv,d_tv,d_tg,A,d_u0.p,d_u1.p,d_u2.p,TS.k.p,TS.omega.p,d_dg,d_diagDG,d_inletRhsX,d_inletRhsY,d_inletRhsZ,d_wall,d_diagWall,d_wallStats,d_bad,bulk,R,nu,wallSampleFraction,sstG1,momentumWallMode,spaldingMolecularConsistency,weakWallGamma,weakWallHScale,weakWallNutScale,
          (stage22Vector && !stage22d4Live),stage22Penalty,d_s22Rhs0.p,d_s22Rhs1.p,d_s22Rhs2.p,
          stage6bWallY,stage6bWallZ,d_s22DiagY.p,d_s22DiagZ.p,stage22dFaceFit,stage22d4Live
#if defined(NODALS_P2_STAGE4_COUPLED)
          ,p2MomentumLive?p2Stage4->d_tets.p:nullptr,
          p2MomentumLive?p2Stage4->k.p:nullptr,
          p2MomentumLive?p2Stage4->w.p:nullptr
#endif
          );
#if defined(NODALS_P2_STAGE4_COUPLED)
        if(p2MomentumLive){
          if(BH.inT.empty())throw std::runtime_error("P2_STAGE4: no inlet faces");
          std::printf("P2_STAGE4_INLET_APPLIED=PASS it=%d path=DG_TET12 constitutive=P2_VALUE_AND_GRADIENT faces=%zu status=PASS\n",git,BH.inT.size());
        }
#endif
        // STAGE25A_FIX1_VECTOR_DG: strict constant-plug cancellation audit
        // on the ACTUAL SST G2 XYZ numerical inlet operator, before any
        // momentum solve or wall coefficients are added. A rotated inlet
        // must cancel componentwise: G_DG * Uin[i] == rhsDG[i].
        if(git==1 && !stage27ProfileActive){
          // Stage27 potential initialization replaces fixed.u* with a
          // spatially varying flow field. The Stage25A *operator* regression
          // must use a prescribed CONSTANT inlet plug as its independent
          // test vector, not the velocity currently being initialized.
          // Keep the original plug-run code path and strict residual gate.
          hxt4b::FixedData stage27DgReference;
          const hxt4b::FixedData* stage27DgTest=&fixed;
          if(stage27PotentialStart){
            stage27DgReference=hxt5b::plug_initial(M,inlet23);
            stage27DgTest=&stage27DgReference;
          }
          std::printf("STAGE27_DG_AUDIT_REFERENCE initial=%s "
            "testVector=INDEPENDENT_CONSTANT_INLET_PLUG "
            "operator=ACTUAL_SST_G2_DG rhs=ASSEMBLED_DG "
            "relativeGate=UNCHANGED status=PASS\n",
            stage27PotentialStart?"POTENTIAL":"PLUG");
          Dev<T> dgPlug0(stage27DgTest->u0),dgPlug1(stage27DgTest->u1),dgPlug2(stage27DgTest->u2);
          const T* dgInputs[3]={dgPlug0.p,dgPlug1.p,dgPlug2.p};
          const T* dgRight[3]={d_inletRhsX.p,d_inletRhsY.p,d_inletRhsZ.p};
          const char* component[3]={"x","y","z"};
          double errSquared=0.,rhsSquared=0.;
          const double auditLimit=std::is_same<T,float>::value?5.e-5:2.e-11;
          for(int componentIndex=0;componentIndex<3;++componentIndex){
            hxt5b::spmv<<<(nv+TPB-1)/TPB,TPB>>>(nv,A.row.p,A.col.p,
              d_dg.p,dgInputs[componentIndex],A.tmp.p);
            hxt5b::h5b3_subtract<<<(nv+TPB-1)/TPB,TPB>>>(nv,
              A.tmp.p,dgRight[componentIndex],A.res.p);
            const double e=ured.norm(A.res.p),r=ured.norm(dgRight[componentIndex]);
            errSquared+=e*e;rhsSquared+=r*r;
            std::printf("STAGE25A_FIX1_SST_DG_COMPONENT component=%s "
              "defect=%.12e rhsNorm=%.12e status=%s\n",component[componentIndex],
              e,r,std::isfinite(e)&&std::isfinite(r)?"PASS":"FAIL");
          }
          const double relative=std::sqrt(errSquared)/
            std::max(std::sqrt(rhsSquared),1.e-300);
          std::printf("STAGE25A_FIX1_SST_DG_VECTOR "
            "frame=PHYSICAL_XYZ normAbs=%.12e normRHS=%.12e relative=%.12e "
            "limit=%.12e status=%s\n",std::sqrt(errSquared),
            std::sqrt(rhsSquared),relative,auditLimit,
            std::isfinite(relative)&&relative<=auditLimit?"PASS":"FAIL");
          if(!std::isfinite(relative)||relative>auditLimit)
            throw std::runtime_error("STAGE25A_FIX1 SST vector DG does not preserve constant inlet plug");
        }
        // Stage23B: assemble the full X/Y physical momentum source once.
        // Z remains handled by its existing two-input RHS path.
        hxt5b::s22b_add_extra<<<(nv+TPB-1)/TPB,TPB>>>(nv,
          d_sstDevRhs0.p,d_inletRhsX.p,d_stage23DevRhs0.p);
        hxt5b::s22b_add_extra<<<(nv+TPB-1)/TPB,TPB>>>(nv,
          d_sstDevRhs1.p,d_inletRhsY.p,d_stage23DevRhs1.p);
        if(stage22d4Live){
          // Copy exactly the Picard state whose off-component action the wall
          // RHS lags. These buffers are *not* updated by the scalar solves.
          HXT1_CUDA(cudaMemcpy(d_s22d4cLagU0.p,d_u0.p,nv*sizeof(T),cudaMemcpyDeviceToDevice));
          HXT1_CUDA(cudaMemcpy(d_s22d4cLagU1.p,d_u1.p,nv*sizeof(T),cudaMemcpyDeviceToDevice));
          HXT1_CUDA(cudaMemcpy(d_s22d4cLagU2.p,d_u2.p,nv*sizeof(T),cudaMemcpyDeviceToDevice));
          if(stage22d4i3Enable){
            HXT1_CUDA(cudaMemset(d_i3penX.p,0,(std::size_t)A.nnz*sizeof(T)));
            HXT1_CUDA(cudaMemset(d_i3penY.p,0,(std::size_t)A.nnz*sizeof(T)));
            HXT1_CUDA(cudaMemset(d_i3penZ.p,0,(std::size_t)A.nnz*sizeof(T)));
          }
          HXT1_CUDA(cudaMemset(d_wall.p,0,A.nnz*sizeof(T)));
          HXT1_CUDA(cudaMemset(stage6bWallY,0,A.nnz*sizeof(T)));
          HXT1_CUDA(cudaMemset(stage6bWallZ,0,A.nnz*sizeof(T)));
          HXT1_CUDA(cudaMemset(d_s22Rhs0.p,0,nv*sizeof(T)));
          HXT1_CUDA(cudaMemset(d_s22Rhs1.p,0,nv*sizeof(T)));
          HXT1_CUDA(cudaMemset(d_s22Rhs2.p,0,nv*sizeof(T)));
          HXT1_CUDA(cudaMemset(d_bad.p,0,sizeof(unsigned long long)));
          // Each active quadrature thread writes its own hit flag; no
          // contended global hit-counter atomic in the production GPU kernel.
          // A fresh current-FROZEN-state wall statistic is built every outer
          // in off mode, using the production fit, never an independent audit.
          if(stage27AuditPolicy.disabled())
            HXT1_CUDA(cudaMemset(d_stage27ProdWallStats.p,0,3*sizeof(double)));
          // F2: exactly ONE nonlinear Spalding fit per physical wall face.
          // All q-dependent nu_eff, normal/BF3 terms and omega equation remain
          // unchanged. The face coefficient is frozen for this SIMPLE step.
          if(f2FaceSpalding && !optionAShearActive){
            hxt5d::f2_face_averaged_spalding<<<(BH.wallT.size()+127)/128,128>>>(
              sstG1.tetWallPlans,(int)BH.wallT.size(),sstG1.tetWallQuads,
              sstG1.profileTetMatchPoints,d_tv.p,
              d_u0.p,d_u1.p,d_u2.p,nu,d_f2FaceFit.p,d_bad.p);
            HXT1_CUDA(cudaGetLastError());
            HXT1_CUDA(cudaDeviceSynchronize());
            unsigned long long f2Bad=0;
            HXT1_CUDA(cudaMemcpy(&f2Bad,d_bad.p,sizeof(f2Bad),cudaMemcpyDeviceToHost));
            if(f2Bad)throw std::runtime_error("F2 face-level Spalding invalid faces="+std::to_string(f2Bad));
            if(git==1 || git%sstG2PrintEvery==0)
              std::printf("NODALS_F2_FACE_FIT it=%d faces=%zu fits=%zu "
                "quadrature=12 momentumOnly=YES status=PASS\n",
                git,BH.wallT.size(),BH.wallT.size());
          }
          // F1A compiles a distinct kernel for each accumulation strategy;
          // disabled mode retains the original CUDA kernel specialization.
          if(f1FaceLocal){
            hxt5d::s22d4_nitsche_tet_component<true><<<BH.wallT.size(),64>>>(
             sstG1.tetWallPlans,(int)BH.wallT.size(),sstG1.tetWallQuads,
             sstG1.profileTetMatchPoints,d_tv.p,d_tg.p,d_pts.p,d_tc.p,A.tetSlot.p,
             d_u0.p,d_u1.p,d_u2.p,TS.k.p,TS.omega.p,
             d_wall.p,stage6bWallY,stage6bWallZ,
             d_s22Rhs0.p,d_s22Rhs1.p,d_s22Rhs2.p,
             nu,R,sstG1.kFloor,sstG1.omegaFloor,s22d4Gamma,d_bad.p,d_s22d4dCoeff.p,stage22d4iEnable,
            stage22d4i3Enable?d_i3penX.p:nullptr,
            stage22d4i3Enable?d_i3penY.p:nullptr,
            stage22d4i3Enable?d_i3penZ.p:nullptr,
            stage27AuditPolicy.disabled()?d_stage27ProdWallStats.p:nullptr,
            sampledConductance,
            git>stage31Start+1?sstG1.stage27SavedWallFit:nullptr,d_stage27C3ReuseFlags.p,
            f2FaceSpalding?d_f2FaceFit.p:nullptr
#if defined(NODALS_P2_STAGE4_COUPLED)
            ,p2MomentumLive?p2Stage4->d_tets.p:nullptr,
            p2MomentumLive?p2Stage4->k.p:nullptr,
            p2MomentumLive?p2Stage4->w.p:nullptr
#endif
            );
          }else{
            hxt5d::s22d4_nitsche_tet_component<false><<<BH.wallT.size(),64>>>(
             sstG1.tetWallPlans,(int)BH.wallT.size(),sstG1.tetWallQuads,
             sstG1.profileTetMatchPoints,d_tv.p,d_tg.p,d_pts.p,d_tc.p,A.tetSlot.p,
             d_u0.p,d_u1.p,d_u2.p,TS.k.p,TS.omega.p,
             d_wall.p,stage6bWallY,stage6bWallZ,
             d_s22Rhs0.p,d_s22Rhs1.p,d_s22Rhs2.p,
             nu,R,sstG1.kFloor,sstG1.omegaFloor,s22d4Gamma,d_bad.p,d_s22d4dCoeff.p,stage22d4iEnable,
            stage22d4i3Enable?d_i3penX.p:nullptr,
            stage22d4i3Enable?d_i3penY.p:nullptr,
            stage22d4i3Enable?d_i3penZ.p:nullptr,
            stage27AuditPolicy.disabled()?d_stage27ProdWallStats.p:nullptr,
            sampledConductance,
            git>stage31Start+1?sstG1.stage27SavedWallFit:nullptr,d_stage27C3ReuseFlags.p,
            f2FaceSpalding?d_f2FaceFit.p:nullptr
#if defined(NODALS_P2_STAGE4_COUPLED)
            ,p2MomentumLive?p2Stage4->d_tets.p:nullptr,
            p2MomentumLive?p2Stage4->k.p:nullptr,
            p2MomentumLive?p2Stage4->w.p:nullptr
#endif
            );
          }
          HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
          if(!stage27Flags.empty())HXT1_CUDA(cudaMemcpy(stage27Flags.data(),
            d_stage27C3ReuseFlags.p,stage27Flags.size(),cudaMemcpyDeviceToHost));
          unsigned long long stage27Reused=0;
          for(unsigned char value:stage27Flags)stage27Reused+=(value!=0);
          std::printf("STAGE27_S02C3_REUSE it=%d hits=%llu possible=%zu "
            "guard=%s fallback=STAGE2C1 status=PASS\n",
            git,stage27Reused,BH.wallT.size()*12u,
            f2FaceSpalding?"F2_FACE_FIT_NO_QUAD_REUSE":"EXACT_TANGENTIAL_SPEED");
          unsigned long long stage4Bad=0;
          HXT1_CUDA(cudaMemcpy(&stage4Bad,d_bad.p,sizeof(stage4Bad),cudaMemcpyDeviceToHost));
          if(stage4Bad)throw std::runtime_error("STAGE22D4B live Nitsche face assembly invalid="+std::to_string(stage4Bad));
#if defined(NODALS_P2_STAGE4_COUPLED)
          if(p2MomentumLive)
            std::printf("P2_STAGE4_WALL_APPLIED=PASS it=%d path=F1B_NITSCHE12 constitutive=P2_VALUE_AND_GRADIENT faces=%zu status=PASS\n",git,BH.wallT.size());
#endif
          if(git==1||git%sstG2PrintEvery==0)
            std::printf("STAGE22D4B_MOMENTUM it=%d faces=%zu block=XYZ_SCALAR "
              "off_component=PICARD_LAGGED viscosity=NU_PLUS_SST_NUT gammaN=%.8g "
              "spalding=FACE_TANGENTIAL mode=%s status=PASS\n",git,BH.wallT.size(),s22d4Gamma,
              sampledWallMode.c_str());
        }
        if(f1bFrozenWall){
          // The source wall assembly has already finalized this Picard step's
          // 12-point coefficients. Rebuild the compact cache ONCE per outer.
          // Subsequent FGMRES matvecs reuse the exact frozen cross tensor.
          hxt5d::f1b_build_face_coupling<<<BH.wallT.size(),64>>>(
            sstG1.tetWallPlans,(int)BH.wallT.size(),sstG1.tetWallQuads,
            d_tg.p,d_s22d4dCoeff.p,d_f1bWallCross.p,s22d4Gamma,
            stage22d4iEnable,sampledConductance);
          HXT1_CUDA(cudaGetLastError());
          if(git==1 || git%sstG2PrintEvery==0)
            std::printf("NODALS_F1B_CACHE it=%d faces=%zu entries=%zu "
              "bytes=%zu refresh=EACH_SIMPLE sampledBasis=%s status=PASS\n",
              git,BH.wallT.size(),(std::size_t)64*BH.wallT.size(),
              (std::size_t)64*BH.wallT.size()*sizeof(double),
              sampledConductance?"CA_BP":"CA_BW");
        }
        if(momentumWallMode==hxt5d::MOM_WALL_SPALDING_ALIGNED && git%sstG2PrintEvery==0)
          hxt5d::print_spalding_aligned_stats(d_wallStats,git);
        if(momentumWallMode==hxt5d::MOM_WALL_SPALDING_PROFILE_INTEGRAL && git%sstG2PrintEvery==0)
          hxt5d::print_spalding_profile_integral_stats(d_wallStats,git);
        if(stage22Vector)
          hxt5b::s22c_combine<<<(A.nnz+TPB-1)/TPB,TPB>>>(A.nnz,A.base.p,A.conv.p,
            d_turb.p,d_dg.p,d_wall.p,stage6bWallY,stage6bWallZ,A.val.p,d_s22ValY.p,d_valZ.p);
        else
          hxt5b::combine<<<(A.nnz+TPB-1)/TPB,TPB>>>(A.nnz,A.base.p,A.conv.p,d_turb.p,d_dg.p,d_wall.p,A.val.p,d_valZ.p);
        HXT1_CUDA(cudaMemset(d_bad.p,0,sizeof(unsigned long long)));
        if(stage22Vector){
          hxt5b::s22c_finalize_scalar<<<(nv+TPB-1)/TPB,TPB>>>(nv,d_fixed.p,
            A.row.p,A.diagPos.p,A.val.p,A.rau.p,A.diagOriginal.p,d_relaxDeltaXY.p,alphaU,rauScale,d_bad.p);
          hxt5b::s22c_finalize_scalar<<<(nv+TPB-1)/TPB,TPB>>>(nv,d_fixed.p,
            A.row.p,A.diagPos.p,d_s22ValY.p,d_s22RauY.p,d_s22DiagOrigY.p,d_s22RelaxDeltaY.p,alphaU,rauScale,d_bad.p);
          hxt5b::s22c_finalize_scalar<<<(nv+TPB-1)/TPB,TPB>>>(nv,d_fixed.p,
            A.row.p,A.diagPos.p,d_valZ.p,d_rauZ.p,d_diagOrigZ.p,d_relaxDeltaZ.p,alphaU,rauScale,d_bad.p);
        }
        else
        hxt5b::h5b3_finalize_row_l1<<<(nv+TPB-1)/TPB,TPB>>>(nv,d_fixed.p,d_wallMask.p,A.row.p,A.diagPos.p,A.val.p,d_valZ.p,A.diagRBase.p,A.conv.p,d_diagRTurb.p,d_diagDG.p,d_diagWall.p,A.rau.p,d_rauZ.p,A.diagOriginal.p,d_diagOrigZ.p,d_relaxDeltaXY.p,d_relaxDeltaZ.p,gamma,schurNitscheGamma,alphaU,rauScale,d_bad.p);
        HXT1_CUDA(cudaDeviceSynchronize());HXT1_CUDA(cudaMemcpy(&bad,d_bad.p,sizeof(bad),cudaMemcpyDeviceToHost));if(bad)throw std::runtime_error("SST G2 nonpositive momentum relaxed/rAU diagonal count="+std::to_string(bad));
        // Stage6.C.1: one read-only census of exact component differences.
        // This runs only with an explicit opt-in and changes NO operator.
        if(git==1 && stage22Vector && stage22d4Live &&
           std::getenv("STAGE6C1_DELTA_AUDIT") &&
           std::string(std::getenv("STAGE6C1_DELTA_AUDIT"))=="1"){
          Dev<unsigned long long> stage6c1Counts(std::vector<unsigned long long>(11,0ULL));
          stage6c1::census<<<(nv+255)/256,256>>>(nv,A.row.p,A.diagPos.p,
            A.val.p,d_s22ValY.p,d_valZ.p,stage6c1Counts.p);
          HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
          unsigned long long cc[11]={};
          HXT1_CUDA(cudaMemcpy(cc,stage6c1Counts.p,sizeof(cc),cudaMemcpyDeviceToHost));
          const double totalNnz=(double)A.nnz;
          const double fullExtra=2.*totalNnz*sizeof(T)/1048576.;
          const double compactCandidate=(12.*((double)cc[0]+(double)cc[1])+
            8.*((double)nv+1.))/1048576.;
          std::printf("STAGE6C1_DELTA_CENSUS nnz=%zu nv=%d exactXY=%llu exactXZ=%llu "
            "exactYZ=%llu union=%llu changedRows=%llu changedDiagonal=%llu "
            "changedOffDiagonal=%llu tolerance1e-12Union=%llu "
            "tolerance1e-12OffDiagonal=%llu status=PASS\n",
            (std::size_t)A.nnz,nv,cc[0],cc[1],cc[2],cc[3],cc[4],cc[5],cc[6],cc[7],cc[8]);
          std::printf("STAGE6C1_DELTA_ESTIMATE fullYZMiB=%.6f "
            "candidateSparseYZMiB=%.6f savingUpperBoundMiB=%.6f "
            "ratioExactYZToNnz=%.8f "
            "candidate=NON_IMPLEMENTED_INT32_INDEX_FP64_ABSOLUTE_OVERRIDE "
            "notes=NO_SAVING_YET status=PASS\n",
            fullExtra,compactCandidate,fullExtra-compactCandidate,
            ((double)cc[0]+(double)cc[1])/std::max(1.,2.*totalNnz));
        }


