        if(pressureSolver=="pcg_amg"&&pressurePOD.has_pending_batch()){
          const bool haveCandidate=pressurePOD.build_tentative(Sg2,PC,WP.red,git);
          if(haveCandidate){
            if(pressurePOD.no_probe()){
              pressurePOD.accept_tentative_without_probe(git,pr.its);
            }else{
              T*probeX=pressurePOD.probe_x();
              HXT1_CUDA(cudaMemset(probeX,0,(std::size_t)np*sizeof(T)));
              const int probeMax=pressurePOD.probe_max(pr.its,pMax);
              auto probePr=hxt4c::pcg_amg(
                Sg2,PC,d_cont.p,probeX,np,pRtol,pAtol,probeMax,WP,pin,&pressurePOD,false);
              pressurePOD.resolve_tentative(git,pr.its,probePr.its,probePr.ok);
            }
          }
        }

