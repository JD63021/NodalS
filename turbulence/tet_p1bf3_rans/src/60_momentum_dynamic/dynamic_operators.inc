static PetscErrorCode assembleCentralConvectionCustom(const Discrete& D,const CustomDynamicRuntimePlan& P,CustomMomentumCSR& A) {
  PetscFunctionBeginUser;
  const auto& T=centralTensor(); for(auto& r:A.convRhs) std::fill(r.begin(),r.end(),0.0);
  for(const auto& cp:P.cells) {
    double coeff[3][8]={{0}};
    for(int m=0;m<8;++m) { if(dynRefFree(cp.ref[m])) for(int d=0;d<3;++d) coeff[d][m]=customMomentumFieldValue(A,d,cp.ref[m]); else { const PetscInt e=dynFixedEntity(cp.ref[m]); for(int d=0;d<3;++d) coeff[d][m]=entityDirValue(D,d,e); } }
    double uref[8][3]={{0}}; for(int m=0;m<8;++m) for(int j=0;j<3;++j) for(int d=0;d<3;++d) uref[m][j]+=coeff[d][m]*cp.invJ[j][d];
    double Cl[8][8]={{0}}; for(int a=0;a<8;++a) for(int b=0;b<8;++b) { double v=0.0; for(int m=0;m<8;++m) for(int j=0;j<3;++j) v+=uref[m][j]*T.t[a][m][b][j]; Cl[a][b]=cp.det*v; }
    for(int ii=0;ii<(int)cp.nOwnedRows;++ii) { const int a=(int)cp.ownedBasis[ii]; const PetscInt lr=cp.ref[a]; for(int b=0;b<8;++b) { if(dynRefFree(cp.ref[b])) { const std::uint8_t slot=cp.rowSlot[a][b]; if(slot==255) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"compact central missing row slot"); A.aRel[(std::size_t)(A.rowPtr[(std::size_t)lr]+slot)]+=Cl[a][b]; } else { const PetscInt e=dynFixedEntity(cp.ref[b]); for(int d=0;d<3;++d) A.convRhs[d][(std::size_t)lr]-=Cl[a][b]*entityDirValue(D,d,e); } } }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
static inline double supgNikuradseNuT(
    const double coeff[3][8],const double gradBasis[8][3],const std::array<double,4>& lam,
    const double xy[4][2],const PipeGeometry& pipe,double nutScale) {
  if(!(nutScale>0.0) || !(pipe.R>0.0)) return 0.0;
  double gradU[3][3]={{0}};
  for(int i=0;i<3;++i) for(int j=0;j<3;++j)
    for(int a=0;a<8;++a) gradU[i][j]+=coeff[i][a]*gradBasis[a][j];
  double ss=0.0;
  for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
    const double sij=0.5*(gradU[i][j]+gradU[j][i]);
    ss+=sij*sij;
  }
  const double strain=std::sqrt(std::max(0.0,2.0*ss));
  double xq=0.0,yq=0.0;
  for(int i=0;i<4;++i) { xq+=lam[(std::size_t)i]*xy[i][0]; yq+=lam[(std::size_t)i]*xy[i][1]; }
  const double dx=xq-pipe.cx,dy=yq-pipe.cy;
  const double eta=std::min(1.0,std::max(0.0,std::sqrt(dx*dx+dy*dy)/pipe.R));
  const double eta2=eta*eta,eta4=eta2*eta2;
  const double ell=pipe.R*std::max(0.0,0.14-0.08*eta2-0.06*eta4);
  return nutScale*ell*ell*strain;
}


static const std::array<TriangleQuadPoint,12>& triangleDunavant12Degree6() {
  static const std::array<TriangleQuadPoint,12> Q={{
    {{0.063089014491502,0.063089014491502,0.873821971016996},0.050844906370207},
    {{0.063089014491502,0.873821971016996,0.063089014491502},0.050844906370207},
    {{0.873821971016996,0.063089014491502,0.063089014491502},0.050844906370207},
    {{0.249286745170910,0.249286745170910,0.501426509658180},0.116786275726379},
    {{0.249286745170910,0.501426509658180,0.249286745170910},0.116786275726379},
    {{0.501426509658180,0.249286745170910,0.249286745170910},0.116786275726379},
    {{0.053145049844816,0.310352451033785,0.636502499121399},0.082851075618374},
    {{0.053145049844816,0.636502499121399,0.310352451033785},0.082851075618374},
    {{0.310352451033785,0.053145049844816,0.636502499121399},0.082851075618374},
    {{0.310352451033785,0.636502499121399,0.053145049844816},0.082851075618374},
    {{0.636502499121399,0.053145049844816,0.310352451033785},0.082851075618374},
    {{0.636502499121399,0.310352451033785,0.053145049844816},0.082851075618374}
  }};
  return Q;
}

static inline double dgInletNikuradseNuT(
    const double coeff[3][8],const double gradBasis[8][3],const std::array<double,4>& lam,
    const double xy[4][2],const PipeGeometry& pipe,double nutScale,PetscBool deviatoricStrain) {
  if(!(nutScale>0.0) || !(pipe.R>0.0)) return 0.0;
  double gradU[3][3]={{0}};
  for(int i=0;i<3;++i) for(int j=0;j<3;++j)
    for(int a=0;a<8;++a) gradU[i][j]+=coeff[i][a]*gradBasis[a][j];
  double S[3][3]={{0}},ssRaw=0.0;
  for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
    S[i][j]=0.5*(gradU[i][j]+gradU[j][i]);
    ssRaw+=S[i][j]*S[i][j];
  }
  const double divU=gradU[0][0]+gradU[1][1]+gradU[2][2];
  const double thirdDiv=divU/3.0;
  double ssDev=0.0;
  for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
    const double sd=S[i][j]-((i==j)?thirdDiv:0.0);
    ssDev+=sd*sd;
  }
  const double strain=std::sqrt(std::max(0.0,2.0*(deviatoricStrain?ssDev:ssRaw)));
  double xq=0.0,yq=0.0;
  for(int i=0;i<4;++i) { xq+=lam[(std::size_t)i]*xy[i][0]; yq+=lam[(std::size_t)i]*xy[i][1]; }
  const double dx=xq-pipe.cx,dy=yq-pipe.cy;
  const double eta=std::min(1.0,std::max(0.0,std::sqrt(dx*dx+dy*dy)/pipe.R));
  const double eta2=eta*eta,eta4=eta2*eta2;
  const double ell=pipe.R*std::max(0.0,0.14-0.08*eta2-0.06*eta4);
  return nutScale*ell*ell*strain;
}

static PetscErrorCode assembleDgNumericalTraceInletCustom(
    const Discrete& D,const CustomDynamicRuntimePlan& P,CustomMomentumCSR& A,
    const BoundaryGeometry& B,const PipeGeometry& pipe,double nu,
    PetscBool mixingLength,double mixlenScale,PetscBool deviatoricStrain) {
  PetscFunctionBeginUser;
  for(int d=0;d<3;++d) {
    if((PetscInt)A.inletRhs[d].size()!=A.nOwned) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"weak DG inlet RHS storage not enabled");
    std::fill(A.inletRhs[d].begin(),A.inletRhs[d].end(),0.0);
  }

  const double uhat[3]={B.inletVelocity.x,B.inletVelocity.y,B.inletVelocity.z};
  const auto& FQ=triangleDunavant12Degree6();
  double localArea=0.0,localFlux=0.0;
  double localCancel2=0.0,localCancelMax=0.0,localRhs2=0.0;
  unsigned long long localFaces=0;
  static PetscBool printed=PETSC_FALSE;

  for(const auto& inf:P.inletFaces) {
    const auto& cp=P.cells[(std::size_t)inf.cellIndex];
    double coeff[3][8]={{0}};
    for(int m=0;m<8;++m) {
      if(dynRefFree(cp.ref[m])) for(int d=0;d<3;++d) coeff[d][m]=customMomentumFieldValue(A,d,cp.ref[m]);
      else {
        const PetscInt e=dynFixedEntity(cp.ref[m]);
        for(int d=0;d<3;++d) coeff[d][m]=entityDirValue(D,d,e);
      }
    }

    int fv[3],kk=0;
    for(int i=0;i<4;++i) if(i!=(int)inf.opp) fv[kk++]=i;
    double G[8][8]={{0}},Rhs[3][8]={{0}};
    const double bn=uhat[0]*inf.normal[0]+uhat[1]*inf.normal[1]+uhat[2]*inf.normal[2];
    const double inflow=std::max(0.0,-bn);

    for(const auto& q:FQ) {
      std::array<double,4> lam{{0.0,0.0,0.0,0.0}};
      for(int j=0;j<3;++j) lam[(std::size_t)fv[j]]=q.l[j];

      double val[8],grr[8][3],gradBasis[8][3]={{0}};
      basis(lam,val,grr);
      for(int a=0;a<8;++a) for(int d=0;d<3;++d)
        for(int j=0;j<3;++j) gradBasis[a][d]+=grr[a][j]*cp.invJ[j][d];

      const double nuT=mixingLength?dgInletNikuradseNuT(coeff,gradBasis,lam,cp.xy,pipe,mixlenScale,deviatoricStrain):0.0;
      const double nuEff=nu+nuT;
      const double w=q.w*inf.area;
      double dn[8]={0};
      for(int a=0;a<8;++a)
        dn[a]=gradBasis[a][0]*inf.normal[0]+gradBasis[a][1]*inf.normal[1]+gradBasis[a][2]*inf.normal[2];

      for(int a=0;a<8;++a) {
        for(int b=0;b<8;++b)
          G[a][b]+=w*(inflow*val[a]*val[b] + nuEff*(-val[a]*dn[b] + dn[a]*val[b]));
        for(int d=0;d<3;++d)
          Rhs[d][a]+=w*(inflow*val[a]*uhat[d] + nuEff*dn[a]*uhat[d]);
      }
    }

    // Exact local consistency gate for the initialized plug:
    // for U == Uhat, the weak inlet contribution must cancel elementwise,
    //   G * Uhat - rhs == 0.
    // This checks inflow + Nitsche signs, basis traces, normal derivatives,
    // quadrature, and Uhat orientation independently of pressure/SIMPLE.
    if(!printed) {
      for(int d=0;d<3;++d) for(int a=0;a<8;++a) {
        double rr=-Rhs[d][a];
        for(int b=0;b<8;++b) rr+=G[a][b]*coeff[d][b];
        localCancel2+=rr*rr;
        localCancelMax=std::max(localCancelMax,std::abs(rr));
        localRhs2+=Rhs[d][a]*Rhs[d][a];
      }
    }

    for(int ii=0;ii<(int)cp.nOwnedRows;++ii) {
      const int a=(int)cp.ownedBasis[ii]; const PetscInt lr=cp.ref[a];
      for(int d=0;d<3;++d) A.inletRhs[d][(std::size_t)lr]+=Rhs[d][a];
      for(int b=0;b<8;++b) {
        if(dynRefFree(cp.ref[b])) {
          const std::uint8_t slot=cp.rowSlot[a][b];
          if(slot==255) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"weak DG inlet missing row slot");
          A.aRel[(std::size_t)(A.rowPtr[(std::size_t)lr]+slot)]+=G[a][b];
        } else {
          const PetscInt e=dynFixedEntity(cp.ref[b]);
          for(int d=0;d<3;++d) A.inletRhs[d][(std::size_t)lr]-=G[a][b]*entityDirValue(D,d,e);
        }
      }
    }

    if(inf.diagnosticOwner) {
      localArea+=inf.area; localFlux+=bn*inf.area; ++localFaces;
    }
  }

  if(!printed) {
    double area=0.0,flux=0.0; unsigned long long faces=0;
    double cancel2=0.0,cancelMax=0.0,rhs2=0.0;
    PetscCallMPI(MPI_Allreduce(&localArea,&area,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(&localFlux,&flux,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(&localFaces,&faces,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(&localCancel2,&cancel2,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(&localCancelMax,&cancelMax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(&localRhs2,&rhs2,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
    if(faces==0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"dg_numerical_trace global inlet-face audit found zero inlet faces");
    const double cancelRel=std::sqrt(cancel2)/std::max(std::sqrt(rhs2),1.0e-300);
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_DG_INLET_PLUG_MOMENTUM_CANCELLATION rel=%.12e maxAbs=%.12e rhsNorm=%.12e expected=roundoff status=%s\n",
      cancelRel,cancelMax,std::sqrt(rhs2),cancelRel<=1.0e-11?"PASS":"FAIL"));
    if(cancelRel>1.0e-11) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"DG weak inlet does not exactly preserve prescribed constant plug");
    const double target=B.signedNormalSpeed*B.inletProjectedArea;
    const double rel=std::abs(flux-target)/std::max(std::abs(target),1e-300);
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_DG_INLET_TRACE mode=dg_numerical_trace faces=%llu area=%.12e projectedArea=%.12e Uhat=[%.12e,%.12e,%.12e] prescribedFlux=%.12e targetFlux=%.12e relFluxError=%.3e Btrace=P0_exact_P1_Aover3_BF3_9Aover20 momentum=inflow_plus_nonsymmetric_Nitsche inletQuad=Dunavant12_degree6_exact_BF3xBF3 nuEff=nu_plus_lagged_nuT status=%s\n",
      faces,area,B.inletProjectedArea,uhat[0],uhat[1],uhat[2],flux,target,rel,rel<1e-12?"PASS":"CHECK"));
    printed=PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode assembleSupgCustom(const Discrete& D,const CustomDynamicRuntimePlan& P,CustomMomentumCSR& A,double tauScale,double supgMagic,const std::string& form,PetscBool mixingLength,const PipeGeometry& pipe,double mixlenScale,PetscBool supgStrongMolecular,SupgStats& stats) {
  PetscFunctionBeginUser;
  const bool implicit=(form=="implicit"); if(!implicit && form!="explicit") SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"SUPG form must be implicit or explicit");
  for(auto& r:A.supgRhs) std::fill(r.begin(),r.end(),0.0); double localTauMin=1e300,localTauMax=0.0,localTauWeighted=0.0,localWeight=0.0;
  for(std::size_t ic=0;ic<P.cells.size();++ic) {
    const auto& cp=P.cells[ic]; double gradLambda[4][3]; dynGradLambdaFromInvJ(cp.invJ,gradLambda);
    double coeff[3][8]={{0}}; for(int m=0;m<8;++m) { if(dynRefFree(cp.ref[m])) for(int d=0;d<3;++d) coeff[d][m]=customMomentumFieldValue(A,d,cp.ref[m]); else { const PetscInt e=dynFixedEntity(cp.ref[m]); for(int d=0;d<3;++d) coeff[d][m]=entityDirValue(D,d,e); } }
    double Sl[8][8]={{0}},Vr[3][8]={{0}},gradLambdaDot[4][4]={{0}}; for(int i=0;i<4;++i) for(int j=i;j<4;++j){double v=0;for(int d=0;d<3;++d)v+=gradLambda[i][d]*gradLambda[j][d];gradLambdaDot[i][j]=gradLambdaDot[j][i]=v;}
    for(PetscInt q=0;q<P.nQ;++q) {
      const auto& rq=P.ref[(std::size_t)q]; const std::size_t iq=(ic*(std::size_t)P.nQ+(std::size_t)q)*3; double adv[3]={0,0,0}; for(int d=0;d<3;++d) for(int m=0;m<8;++m) adv[d]+=coeff[d][m]*rq.phi[m]; const double speed2=adv[0]*adv[0]+adv[1]*adv[1]+adv[2]*adv[2];
      double gradBasis[8][3]={{0}},stream[8]={0},strongTrial[8]={0}; for(int a=0;a<4;++a) for(int d=0;d<3;++d) gradBasis[a][d]=gradLambda[a][d];
      for(int i=0;i<4;++i){int js[3],kk=0;for(int j=0;j<4;++j)if(j!=i)js[kk++]=j;for(int d=0;d<3;++d)gradBasis[4+i][d]=27.0*(rq.lam[js[1]]*rq.lam[js[2]]*gradLambda[js[0]][d]+rq.lam[js[0]]*rq.lam[js[2]]*gradLambda[js[1]][d]+rq.lam[js[0]]*rq.lam[js[1]]*gradLambda[js[2]][d]);}
      for(int a=0;a<8;++a)stream[a]=adv[0]*gradBasis[a][0]+adv[1]*gradBasis[a][1]+adv[2]*gradBasis[a][2]; for(int a=0;a<4;++a)strongTrial[a]=stream[a];
      // Runtime A/B gate for the FIX4 SUPG strong-residual change.  The
      // physical momentum diffusion remains nu+nu_t in both modes; only the
      // viscous Laplacian inside the SUPG strong residual is switched.
      const double nuTStrong=(!supgStrongMolecular && mixingLength)?supgNikuradseNuT(coeff,gradBasis,rq.lam,cp.xy,pipe,mixlenScale):0.0;
      const double nuStrong=supgStrongMolecular?P.nu:(P.nu+nuTStrong);
      for(int i=0;i<4;++i){int js[3],kk=0;for(int j=0;j<4;++j)if(j!=i)js[kk++]=j;const double lap=54.0*(rq.lam[js[2]]*gradLambdaDot[js[0]][js[1]]+rq.lam[js[1]]*gradLambdaDot[js[0]][js[2]]+rq.lam[js[0]]*gradLambdaDot[js[1]][js[2]]);strongTrial[4+i]=-nuStrong*lap+stream[4+i];}
      const double diff=4.0*P.nu/cp.h2,den=std::max(4.0*speed2/cp.h2+supgMagic*diff*diff,1e-30),tau=tauScale/std::sqrt(den),w=tau*rq.w*cp.det,volumeW=rq.w*cp.det; localTauMin=std::min(localTauMin,tau);localTauMax=std::max(localTauMax,tau);localTauWeighted+=tau*volumeW;localWeight+=volumeW;
      if(implicit){for(int a=0;a<8;++a){const double ta=w*stream[a];for(int b=0;b<8;++b)Sl[a][b]+=ta*strongTrial[b];if(!P.forcingZero)for(int d=0;d<3;++d)Vr[d][a]+=ta*P.forcing[iq+d];}}
      else {double sr[3]={0,0,0};for(int d=0;d<3;++d){for(int b=0;b<8;++b)sr[d]+=coeff[d][b]*strongTrial[b];if(!P.forcingZero)sr[d]-=P.forcing[iq+d];}for(int a=0;a<8;++a){const double ta=-w*stream[a];for(int d=0;d<3;++d)Vr[d][a]+=ta*sr[d];}}
    }
    for(int ii=0;ii<(int)cp.nOwnedRows;++ii){const int a=(int)cp.ownedBasis[ii];const PetscInt lr=cp.ref[a];if(implicit){for(int b=0;b<8;++b){if(dynRefFree(cp.ref[b])){const std::uint8_t slot=cp.rowSlot[a][b];if(slot==255)SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"compact SUPG missing row slot");A.aRel[(std::size_t)(A.rowPtr[(std::size_t)lr]+slot)]+=Sl[a][b];}else{const PetscInt e=dynFixedEntity(cp.ref[b]);for(int d=0;d<3;++d)A.supgRhs[d][(std::size_t)lr]-=Sl[a][b]*entityDirValue(D,d,e);}}}for(int d=0;d<3;++d)A.supgRhs[d][(std::size_t)lr]+=Vr[d][a];}
  }
  double globalMin=0,globalMax=0,globalTW=0,globalW=0; PetscCallMPI(MPI_Allreduce(&localTauMin,&globalMin,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD));PetscCallMPI(MPI_Allreduce(&localTauMax,&globalMax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));PetscCallMPI(MPI_Allreduce(&localTauWeighted,&globalTW,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));PetscCallMPI(MPI_Allreduce(&localWeight,&globalW,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));stats.tauMin=globalMin;stats.tauMax=globalMax;stats.tauMean=globalW?globalTW/globalW:0.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode assembleStaticDiffusionCustom(const CustomDynamicAssemblyPlan& P,CustomMomentumCSR& A,double scale) {
  PetscFunctionBeginUser;
  if(A.aRel.size()!=(std::size_t)A.rowPtr.back()) A.aRel.assign((std::size_t)A.rowPtr.back(),0.0);
  else std::fill(A.aRel.begin(),A.aRel.end(),0.0);
  const auto& R=diffusionTensor();
  for(const auto& cp:P.cells) {
    double metric[3][3]={{0}};
    for(int j=0;j<3;++j) for(int k=0;k<3;++k)
      for(int d=0;d<3;++d) metric[j][k]+=cp.invJ[j][d]*cp.invJ[k][d];
    double Kl[8][8]={{0}};
    for(int a=0;a<8;++a) for(int b=0;b<8;++b) {
      double v=0.0;
      for(int j=0;j<3;++j) for(int k=0;k<3;++k) v+=R.t[a][b][j][k]*metric[j][k];
      Kl[a][b]=scale*cp.det*v;
    }
    for(PetscInt ii=0;ii<cp.nOwnedRows;++ii) {
      const int a=(int)cp.ownedBasis[ii];
      const PetscInt lr=cp.gid[a]-A.rstart;
      for(int b=0;b<8;++b) if(cp.gid[b]>=0) {
        const std::uint8_t slot=cp.rowSlot[a][b];
        if(slot==255) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M1M2 static diffusion missing precomputed CSR slot");
        A.aRel[(std::size_t)(A.rowPtr[(std::size_t)lr]+slot)]+=Kl[a][b];
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode assembleStaticDiffusionCustomLegacyQuadrature(const CustomDynamicAssemblyPlan& P,CustomMomentumCSR& A,double scale) {
  PetscFunctionBeginUser;
  if(A.aRel.size()!=(std::size_t)A.rowPtr.back()) A.aRel.assign((std::size_t)A.rowPtr.back(),0.0);
  else std::fill(A.aRel.begin(),A.aRel.end(),0.0);
  const auto Q=tetDuffy5();
  for(const auto& cp:P.cells) {
    double Kl[8][8]={{0}};
    for(const auto& q:Q) {
      double val[8],grr[8][3],gr[8][3]; basis(q.lam,val,grr);
      for(int a=0;a<8;++a) for(int d=0;d<3;++d) {
        gr[a][d]=0.0; for(int j=0;j<3;++j) gr[a][d]+=grr[a][j]*cp.invJ[j][d];
      }
      const double w=q.w*cp.det;
      for(int a=0;a<8;++a) for(int b=0;b<8;++b) {
        double dot=0.0; for(int d=0;d<3;++d) dot+=gr[a][d]*gr[b][d];
        Kl[a][b]+=dot*w;
      }
    }
    for(PetscInt ii=0;ii<cp.nOwnedRows;++ii) {
      const int a=(int)cp.ownedBasis[ii]; const PetscInt lr=cp.gid[a]-A.rstart;
      for(int b=0;b<8;++b) if(cp.gid[b]>=0) {
        const std::uint8_t slot=cp.rowSlot[a][b];
        if(slot==255) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M1M2 legacy static diffusion missing precomputed CSR slot");
        A.aRel[(std::size_t)(A.rowPtr[(std::size_t)lr]+slot)]+=scale*Kl[a][b];
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode assembleStaticMomentumRhsNative(const Mesh& M,const Discrete& D,const ProblemConfig& problem,const CustomDynamicAssemblyPlan& P,CustomMomentumCSR& A,std::array<std::vector<double>,3>& rhs) {
  PetscFunctionBeginUser;
  for(int d=0;d<3;++d) rhs[(std::size_t)d].assign((std::size_t)A.nOwned,0.0);
  const auto Q=(problem.mode!=ProblemMode::MMS)?tetDuffy5():tetDuffy7();
  for(const auto& cp:P.cells){
    const auto t=M.tets[(std::size_t)cp.cell]; const Vec3 X[4]={M.points[t[0]],M.points[t[1]],M.points[t[2]],M.points[t[3]]};
    double Al[8][8]={{0}},fl[3][8]={{0}};
    for(const auto& q:Q){ double val[8],grr[8][3],gr[8][3]; basis(q.lam,val,grr);
      for(int a=0;a<8;++a) for(int d=0;d<3;++d){gr[a][d]=0.0;for(int j=0;j<3;++j)gr[a][d]+=grr[a][j]*cp.invJ[j][d];}
      double x=0,y=0,z=0;for(int i=0;i<4;++i){x+=q.lam[i]*X[i].x;y+=q.lam[i]*X[i].y;z+=q.lam[i]*X[i].z;} double ff[3];problemForcing(problem,x,y,z,ff);const double w=q.w*cp.det;
      for(int a=0;a<8;++a){for(int d=0;d<3;++d)fl[d][a]+=ff[d]*val[a]*w;for(int b=0;b<8;++b){double dot=0;for(int d=0;d<3;++d)dot+=gr[a][d]*gr[b][d];Al[a][b]+=dot*w;}}}
    for(PetscInt ii=0;ii<cp.nOwnedRows;++ii){const int a=(int)cp.ownedBasis[(std::size_t)ii];const PetscInt lr=cp.gid[a]-A.rstart;
      for(int d=0;d<3;++d) rhs[(std::size_t)d][(std::size_t)lr]+=fl[d][a];
      for(int b=0;b<8;++b) if(cp.gid[b]<0) for(int d=0;d<3;++d){const double ud=entityDirValue(D,d,cp.entity[b]);if(ud!=0.0)rhs[(std::size_t)d][(std::size_t)lr]-=problem.nu*Al[a][b]*ud;}
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customMomentumAddEntry(CustomMomentumCSR& A,PetscInt rowGid,PetscInt colGid,double v) {
  PetscFunctionBeginUser;
  if(A.colGid.empty() && A.rowPtr.back()>0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"M1 colGid released; use precomputed dynamic rowSlot for live assembly");
  if(rowGid<A.rstart || rowGid>=A.rend) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"M3A attempted non-owned row insertion");
  const PetscInt i=rowGid-A.rstart,b=A.rowPtr[(std::size_t)i],e=A.rowPtr[(std::size_t)i+1];
  auto it=std::lower_bound(A.colGid.begin()+b,A.colGid.begin()+e,colGid);
  if(it==A.colGid.begin()+e || *it!=colGid) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M3A custom CSR missing element column");
  A.aRel[(std::size_t)(it-A.colGid.begin())]+=v;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode assembleCentralConvectionCustom(const Discrete& D,const CustomDynamicAssemblyPlan& P,CustomMomentumCSR& A) {
  PetscFunctionBeginUser;
  const auto& T=centralTensor();
  for(auto& r:A.convRhs) std::fill(r.begin(),r.end(),0.0);
  for(const auto& cp:P.cells) {
    double coeff[3][8]={{0}};
    for(int m=0;m<8;++m) {
      if(cp.gid[m]>=0) for(int d=0;d<3;++d) coeff[d][m]=customMomentumFieldValue(A,d,cp.localIndex[m]);
      else for(int d=0;d<3;++d) coeff[d][m]=entityDirValue(D,d,cp.entity[m]);
    }
    double uref[8][3]={{0}};
    for(int m=0;m<8;++m) for(int j=0;j<3;++j) for(int d=0;d<3;++d) uref[m][j]+=coeff[d][m]*cp.invJ[j][d];
    double Cl[8][8]={{0}};
    for(int a=0;a<8;++a) for(int b=0;b<8;++b) {
      double v=0.0; for(int m=0;m<8;++m) for(int j=0;j<3;++j) v+=uref[m][j]*T.t[a][m][b][j]; Cl[a][b]=cp.det*v;
    }
    for(PetscInt ii=0;ii<cp.nOwnedRows;++ii) {
      const int a=(int)cp.ownedBasis[ii]; const PetscInt row=cp.gid[a],lr=row-A.rstart;
      for(int b=0;b<8;++b) {
        if(cp.gid[b]>=0) { const std::uint8_t slot=cp.rowSlot[a][b]; if(slot==255) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M3A central missing precomputed slot"); A.aRel[(std::size_t)(A.rowPtr[(std::size_t)lr]+slot)]+=Cl[a][b]; }
        else for(int d=0;d<3;++d) A.convRhs[d][(std::size_t)lr]-=Cl[a][b]*entityDirValue(D,d,cp.entity[b]);
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode assembleSupgCustom(const Discrete& D,const CustomDynamicAssemblyPlan& P,CustomMomentumCSR& A,
  double tauScale,double supgMagic,const std::string& form,PetscBool mixingLength,const PipeGeometry& pipe,double mixlenScale,PetscBool supgStrongMolecular,SupgStats& stats) {
  PetscFunctionBeginUser;
  const bool implicit=(form=="implicit"); if(!implicit && form!="explicit") SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"SUPG form must be implicit or explicit");
  for(auto& r:A.supgRhs) std::fill(r.begin(),r.end(),0.0);
  double localTauMin=1e300,localTauMax=0.0,localTauWeighted=0.0,localWeight=0.0;
  for(std::size_t ic=0;ic<P.cells.size();++ic) {
    const auto& cp=P.cells[ic];
    double coeff[3][8]={{0}};
    for(int m=0;m<8;++m) {
      if(cp.gid[m]>=0) for(int d=0;d<3;++d) coeff[d][m]=customMomentumFieldValue(A,d,cp.localIndex[m]);
      else for(int d=0;d<3;++d) coeff[d][m]=entityDirValue(D,d,cp.entity[m]);
    }
    double Sl[8][8]={{0}},Vr[3][8]={{0}},gradLambdaDot[4][4]={{0}};
    for(int i=0;i<4;++i) for(int j=i;j<4;++j) { double v=0; for(int d=0;d<3;++d)v+=cp.gradLambda[i][d]*cp.gradLambda[j][d]; gradLambdaDot[i][j]=gradLambdaDot[j][i]=v; }
    for(PetscInt q=0;q<P.nQ;++q) {
      const auto& rq=P.ref[(std::size_t)q]; const std::size_t iq=(ic*(std::size_t)P.nQ+(std::size_t)q)*3;
      double adv[3]={0,0,0}; for(int d=0;d<3;++d) for(int m=0;m<8;++m) adv[d]+=coeff[d][m]*rq.phi[m];
      const double speed2=adv[0]*adv[0]+adv[1]*adv[1]+adv[2]*adv[2];
      double gradBasis[8][3]={{0}},stream[8]={0},strongTrial[8]={0};
      for(int a=0;a<4;++a) for(int d=0;d<3;++d) gradBasis[a][d]=cp.gradLambda[a][d];
      for(int i=0;i<4;++i) { int js[3],kk=0; for(int j=0;j<4;++j) if(j!=i) js[kk++]=j; for(int d=0;d<3;++d)
        gradBasis[4+i][d]=27.0*(rq.lam[js[1]]*rq.lam[js[2]]*cp.gradLambda[js[0]][d]+rq.lam[js[0]]*rq.lam[js[2]]*cp.gradLambda[js[1]][d]+rq.lam[js[0]]*rq.lam[js[1]]*cp.gradLambda[js[2]][d]); }
      for(int a=0;a<8;++a) stream[a]=adv[0]*gradBasis[a][0]+adv[1]*gradBasis[a][1]+adv[2]*gradBasis[a][2];
      for(int a=0;a<4;++a) strongTrial[a]=stream[a];
      const double nuTStrong=(!supgStrongMolecular && mixingLength)?supgNikuradseNuT(coeff,gradBasis,rq.lam,cp.xy,pipe,mixlenScale):0.0;
      const double nuStrong=supgStrongMolecular?P.nu:(P.nu+nuTStrong);
      for(int i=0;i<4;++i) { int js[3],kk=0; for(int j=0;j<4;++j) if(j!=i) js[kk++]=j; const double lap=54.0*(rq.lam[js[2]]*gradLambdaDot[js[0]][js[1]]+rq.lam[js[1]]*gradLambdaDot[js[0]][js[2]]+rq.lam[js[0]]*gradLambdaDot[js[1]][js[2]]); strongTrial[4+i]=-nuStrong*lap+stream[4+i]; }
      const double diff=4.0*P.nu/cp.h2,den=std::max(4.0*speed2/cp.h2+supgMagic*diff*diff,1e-30),tau=tauScale/std::sqrt(den),w=tau*rq.w*cp.det,volumeW=rq.w*cp.det;
      localTauMin=std::min(localTauMin,tau); localTauMax=std::max(localTauMax,tau); localTauWeighted+=tau*volumeW; localWeight+=volumeW;
      if(implicit) {
        for(int a=0;a<8;++a) { const double ta=w*stream[a]; for(int b=0;b<8;++b) Sl[a][b]+=ta*strongTrial[b]; if(!P.forcingZero) for(int d=0;d<3;++d) Vr[d][a]+=ta*P.forcing[iq+d]; }
      } else {
        double sr[3]={0,0,0}; for(int d=0;d<3;++d){for(int b=0;b<8;++b)sr[d]+=coeff[d][b]*strongTrial[b];if(!P.forcingZero)sr[d]-=P.forcing[iq+d];}
        for(int a=0;a<8;++a){const double ta=-w*stream[a];for(int d=0;d<3;++d)Vr[d][a]+=ta*sr[d];}
      }
    }
    for(PetscInt ii=0;ii<cp.nOwnedRows;++ii) {
      const int a=(int)cp.ownedBasis[ii]; const PetscInt row=cp.gid[a],lr=row-A.rstart;
      if(implicit) {
        for(int b=0;b<8;++b) {
          if(cp.gid[b]>=0) { const std::uint8_t slot=cp.rowSlot[a][b]; if(slot==255) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M3A SUPG missing precomputed slot"); A.aRel[(std::size_t)(A.rowPtr[(std::size_t)lr]+slot)]+=Sl[a][b]; }
          else for(int d=0;d<3;++d) A.supgRhs[d][(std::size_t)lr]-=Sl[a][b]*entityDirValue(D,d,cp.entity[b]);
        }
      }
      for(int d=0;d<3;++d) A.supgRhs[d][(std::size_t)lr]+=Vr[d][a];
    }
  }
  double globalMin=0,globalMax=0,globalTW=0,globalW=0;
  PetscCallMPI(MPI_Allreduce(&localTauMin,&globalMin,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localTauMax,&globalMax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localTauWeighted,&globalTW,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localWeight,&globalW,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  stats.tauMin=globalMin;stats.tauMax=globalMax;stats.tauMean=globalW?globalTW/globalW:0.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

struct SchurTerm {
  PetscInt localRau=-1;
  double coeff=0.0;
};

struct SchurColumnPlan {
  PetscInt col=-1;
  std::vector<SchurTerm> terms;
};

struct SchurRowPlan {
  PetscInt row=-1;
  std::vector<SchurColumnPlan> columns;
};

