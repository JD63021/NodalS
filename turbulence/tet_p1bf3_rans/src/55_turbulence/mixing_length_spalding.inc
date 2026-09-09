struct MixingLengthMaxAudit {
  long long cellGid=-1;
  int q=-1,rank=-1;
  double x=0.0,y=0.0,z=0.0,zOverL=0.0,rOverR=0.0,yPlusRefBlasius=0.0,lmOverR=0.0;
  // nuTRatio/twoSdotS/strain are the ACTIVE strain-mode values used to build
  // the momentum diffusion. Raw/deviatoric values are always carried so one
  // run can compare both models at exactly the same physical point.
  double nuTRatio=0.0,twoSdotS=0.0,strain=0.0;
  double divU=0.0;
  double nuTRatioRaw=0.0,twoSdotSRaw=0.0,strainRaw=0.0;
  double nuTRatioDev=0.0,twoSdotSDev=0.0,strainDev=0.0;
  double Sxx=0.0,Syy=0.0,Szz=0.0,Sxy=0.0,Sxz=0.0,Syz=0.0;
};
struct MixingLengthStats {
  double nuTRatioMin=0.0,nuTRatioMean=0.0,nuTRatioMax=0.0;
  double ellMin=0.0,ellMax=0.0,strainMax=0.0;
  double nuTRatioP95=0.0,nuTRatioP99=0.0,nuTRatioP999=0.0;
  double volFracGT100=0.0,volFracGT300=0.0,volFracGT1000=0.0;
  // Raw-vs-deviatoric shadow statistics. Diagnostic only: the active operator
  // still uses exactly one mode selected by -mixlen_strain_mode.
  double rawP95=0.0,rawP99=0.0,rawP999=0.0,rawMax=0.0,rawStrainMax=0.0;
  double devP95=0.0,devP99=0.0,devP999=0.0,devMax=0.0,devStrainMax=0.0;
  MixingLengthMaxAudit maxAudit{};
  unsigned long long quadratureSamples=0;
};
static inline PetscBool dynRefFree(PetscInt r){ return r>=0 ? PETSC_TRUE:PETSC_FALSE; }
static inline PetscInt dynFixedEntity(PetscInt r){ return -r-1; }
static inline void dynGradLambdaFromInvJ(const double invJ[3][3],double g[4][3]) {
  for(int d=0;d<3;++d) {
    g[1][d]=invJ[0][d]; g[2][d]=invJ[1][d]; g[3][d]=invJ[2][d];
    g[0][d]=-(g[1][d]+g[2][d]+g[3][d]);
  }
}
static PetscErrorCode compactCustomDynamicPlan(const Mesh& M,const Discrete& D,const ProblemConfig& problem,CustomDynamicAssemblyPlan& P,CustomDynamicRuntimePlan& R) {
  PetscFunctionBeginUser;
  int rank=0; PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  R.cells.clear(); R.cells.reserve(P.cells.size());
  R.wallFaces.clear();
  R.inletFaces.clear();
  for(const auto& cp:P.cells) {
    CustomDynamicRuntimeCellPlan rp;
    for(int a=0;a<8;++a) {
      rp.ref[a]=(cp.gid[a]>=0)?cp.localIndex[a]:-(cp.entity[a]+1);
      rp.ownedBasis[a]=(std::uint8_t)cp.ownedBasis[a];
      for(int b=0;b<8;++b) rp.rowSlot[a][b]=cp.rowSlot[a][b];
    }
    rp.nOwnedRows=(std::uint8_t)cp.nOwnedRows; rp.det=cp.det; rp.h2=cp.h2;
    rp.diagnosticOwner=(std::uint8_t)(D.cellOwner[(std::size_t)cp.cell]==rank ? 1 : 0);
    for(int j=0;j<3;++j) for(int d=0;d<3;++d) rp.invJ[j][d]=cp.invJ[j][d];
    const auto t=M.tets[(std::size_t)cp.cell];
    for(int i=0;i<4;++i) {
      const Vec3& X=M.points[(std::size_t)t[i]];
      rp.xy[i][0]=X.x; rp.xy[i][1]=X.y; rp.z[i]=(float)X.z;
    }
    rp.diagnosticCellGid=D.pGid[(std::size_t)cp.cell];
    const std::uint32_t runtimeCellIndex=(std::uint32_t)R.cells.size();
    if(!D.wallEntity.empty()) {
      const PetscInt nv=(PetscInt)M.points.size();
      for(int i=0;i<4;++i) {
        const PetscInt f=M.oppFace[(std::size_t)cp.cell][i];
        if(f<0 || nv+f>=(PetscInt)D.wallEntity.size() || !D.wallEntity[(std::size_t)(nv+f)]) continue;
        const Vec3 sf=faceOutwardAreaVector(M,f);
        const double area=norm3(sf);
        if(!(area>0.0)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Stage-3 degenerate wall face");
        CustomDynamicRuntimeWallFace wf;
        wf.cellIndex=runtimeCellIndex; wf.opp=(std::uint8_t)i; wf.diagnosticOwner=rp.diagnosticOwner;
        wf.area=area; wf.hn=cp.det/(2.0*area); // 3*(det/6)/A
        wf.normal[0]=sf.x/area; wf.normal[1]=sf.y/area; wf.normal[2]=sf.z/area;
        R.wallFaces.push_back(wf);
      }
    }
    if(dgNumericalTraceInlet(problem)) {
      int io=-1; Vec3 sf{};
      if(dgNumericalTraceFaceGeom(M,problem,cp.cell,io,sf)) {
        const double area=norm3(sf);
        if(!(area>0.0)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"weak DG inlet has degenerate face");
        CustomDynamicRuntimeWallFace inf;
        inf.cellIndex=runtimeCellIndex; inf.opp=(std::uint8_t)io; inf.diagnosticOwner=rp.diagnosticOwner;
        inf.area=area; inf.normal[0]=sf.x/area; inf.normal[1]=sf.y/area; inf.normal[2]=sf.z/area;
        R.inletFaces.push_back(inf);
      }
    }
    R.cells.push_back(rp);
  }
  R.nQ=P.nQ; R.ref=std::move(P.ref); R.forcing=std::move(P.forcing);
  R.forcingZero=P.forcingZero; R.nu=P.nu;
  const unsigned long long oldLocal=(unsigned long long)P.cells.capacity()*sizeof(CustomDynamicCellPlan);
  const unsigned long long newLocal=(unsigned long long)R.cells.capacity()*sizeof(CustomDynamicRuntimeCellPlan)
    +(unsigned long long)(R.wallFaces.capacity()+R.inletFaces.capacity())*sizeof(CustomDynamicRuntimeWallFace);
  unsigned long long oldGlobal=0,newGlobal=0;
  PetscCallMPI(MPI_Allreduce(&oldLocal,&oldGlobal,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&newLocal,&newGlobal,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  std::vector<CustomDynamicCellPlan>().swap(P.cells);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_DYNPLAN_COMPACT fullCellBytes=%zu compactCellBytes=%zu oldCellMiB=%.3f compactCellMiB=%.3f savedCellMiB=%.3f ratio=%.6f status=PASS\n",
    sizeof(CustomDynamicCellPlan),sizeof(CustomDynamicRuntimeCellPlan),(double)oldGlobal/(1024.0*1024.0),(double)newGlobal/(1024.0*1024.0),(double)(oldGlobal-newGlobal)/(1024.0*1024.0),oldGlobal?(double)newGlobal/(double)oldGlobal:0.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode assembleStaticDiffusionCustom(const CustomDynamicRuntimePlan& P,CustomMomentumCSR& A,double scale) {
  PetscFunctionBeginUser;
  if(A.aRel.size()!=(std::size_t)A.rowPtr.back()) A.aRel.assign((std::size_t)A.rowPtr.back(),0.0);
  else std::fill(A.aRel.begin(),A.aRel.end(),0.0);
  const auto& R=diffusionTensor();
  for(const auto& cp:P.cells) {
    double metric[3][3]={{0}};
    for(int j=0;j<3;++j) for(int k=0;k<3;++k) for(int d=0;d<3;++d) metric[j][k]+=cp.invJ[j][d]*cp.invJ[k][d];
    double Kl[8][8]={{0}};
    for(int a=0;a<8;++a) for(int b=0;b<8;++b) { double v=0.0; for(int j=0;j<3;++j) for(int k=0;k<3;++k) v+=R.t[a][b][j][k]*metric[j][k]; Kl[a][b]=scale*cp.det*v; }
    for(int ii=0;ii<(int)cp.nOwnedRows;++ii) {
      const int a=(int)cp.ownedBasis[ii]; const PetscInt lr=cp.ref[a];
      for(int b=0;b<8;++b) if(dynRefFree(cp.ref[b])) { const std::uint8_t slot=cp.rowSlot[a][b]; if(slot==255) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"compact static diffusion missing row slot"); A.aRel[(std::size_t)(A.rowPtr[(std::size_t)lr]+slot)]+=Kl[a][b]; }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode customMomentumResetPhysical(const CustomDynamicRuntimePlan& P,CustomMomentumCSR& A,double nu,PetscBool) {
  PetscFunctionBeginUser;
  PetscCall(assembleStaticDiffusionCustom(P,A,nu));
  for(int d=0;d<3;++d) std::fill(A.mixlenRhs[d].begin(),A.mixlenRhs[d].end(),0.0);
  PetscFunctionReturn(PETSC_SUCCESS);
}

// -----------------------------------------------------------------------------
// Stage-2 algebraic RANS: Nikuradse smooth-pipe mixing length.
//
// Molecular nu*K is already in A.aRel. This adds only the lagged eddy-viscosity
// contribution int_K nu_t grad(phi_a).grad(phi_b) dV. The three component
// momentum solves therefore remain scalar and share the same operator.
//
// For the present straight pipe, U=(0,0,Uz(r)), this is the exact axial
// Boussinesq stress contribution. The general variable-viscosity transpose-
// gradient term is intentionally deferred beyond Stage 2.
//
// Frozen NodalS pipe convention: pipe axis=z, radial plane=(x,y).
// -----------------------------------------------------------------------------
static PetscErrorCode assembleNikuradseMixingLengthCustom(
    const Discrete& D,const CustomDynamicRuntimePlan& P,CustomMomentumCSR& A,
    const PipeGeometry& pipe,double nu,double nutScale,PetscBool deviatoricStrain,
    PetscBool detailedAudit,MixingLengthStats& stats) {
  PetscFunctionBeginUser;
  if(!(nu>0.0) || !(pipe.R>0.0)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"mixing-length model requires nu>0 and pipe.R>0");
  if(nutScale<0.0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"mixlen scale must be >= 0");
  for(int d=0;d<3;++d) {
    if((PetscInt)A.mixlenRhs[d].size()!=A.nOwned) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"mixing-length RHS storage not enabled");
    std::fill(A.mixlenRhs[d].begin(),A.mixlenRhs[d].end(),0.0);
  }

  double localMin=1e300,localMax=0.0,localWeighted=0.0,localWeight=0.0;
  double localEllMin=1e300,localEllMax=0.0,localStrainMax=0.0;
  double localRawMax=0.0,localDevMax=0.0,localRawStrainMax=0.0,localDevStrainMax=0.0;
  double localWeightGT100=0.0,localWeightGT300=0.0,localWeightGT1000=0.0;
  unsigned long long localSamples=0;
  int rank=0; PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  MixingLengthMaxAudit localMaxAudit{}; localMaxAudit.rank=rank;
  constexpr int auditBins=2048;
  constexpr double auditMaxRatio=1.0e6;
  const double auditLogMax=std::log1p(auditMaxRatio);
  std::vector<double> localHist,localRawHist,localDevHist;
  if(detailedAudit) {
    localHist.assign(auditBins,0.0);
    localRawHist.assign(auditBins,0.0);
    localDevHist.assign(auditBins,0.0);
  }

  for(const auto& cp:P.cells) {
    double coeff[3][8]={{0}};
    for(int m=0;m<8;++m) {
      if(dynRefFree(cp.ref[m])) {
        for(int d=0;d<3;++d) coeff[d][m]=customMomentumFieldValue(A,d,cp.ref[m]);
      } else {
        const PetscInt e=dynFixedEntity(cp.ref[m]);
        for(int d=0;d<3;++d) coeff[d][m]=entityDirValue(D,d,e);
      }
    }

    double Kt[8][8]={{0}};
    for(PetscInt q=0;q<P.nQ;++q) {
      const auto& rq=P.ref[(std::size_t)q];

      double gradBasis[8][3]={{0}};
      for(int a=0;a<8;++a) for(int d=0;d<3;++d)
        for(int j=0;j<3;++j) gradBasis[a][d]+=rq.gradRef[a][j]*cp.invJ[j][d];

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
      const double twoSdotSRaw=std::max(0.0,2.0*ssRaw);
      const double twoSdotSDev=std::max(0.0,2.0*ssDev);
      const double strainRaw=std::sqrt(twoSdotSRaw);
      const double strainDev=std::sqrt(twoSdotSDev);
      const double twoSdotS=deviatoricStrain?twoSdotSDev:twoSdotSRaw;
      const double strain=deviatoricStrain?strainDev:strainRaw;

      double xq=0.0,yq=0.0;
      for(int i=0;i<4;++i) { xq+=rq.lam[i]*cp.xy[i][0]; yq+=rq.lam[i]*cp.xy[i][1]; }
      const double dx=xq-pipe.cx,dy=yq-pipe.cy;
      const double r=std::sqrt(dx*dx+dy*dy);
      const double eta=std::min(1.0,std::max(0.0,r/pipe.R));
      const double eta2=eta*eta,eta4=eta2*eta2;
      const double ell=pipe.R*std::max(0.0,0.14-0.08*eta2-0.06*eta4);
      const double nuTRaw=nutScale*ell*ell*strainRaw;
      const double nuTDev=nutScale*ell*ell*strainDev;
      const double nuT=deviatoricStrain?nuTDev:nuTRaw;
      const double w=rq.w*cp.det;

      for(int a=0;a<8;++a) for(int b=0;b<8;++b) {
        double gd=0.0; for(int d=0;d<3;++d) gd+=gradBasis[a][d]*gradBasis[b][d];
        Kt[a][b]+=nuT*gd*w;
      }

      if(cp.diagnosticOwner) {
        const double ratio=nuT/nu;
        const double ratioRaw=nuTRaw/nu,ratioDev=nuTDev/nu;
        localMin=std::min(localMin,ratio);
        localWeighted+=ratio*w; localWeight+=w;
        localEllMin=std::min(localEllMin,ell); localEllMax=std::max(localEllMax,ell);
        localStrainMax=std::max(localStrainMax,strain);
        localRawMax=std::max(localRawMax,ratioRaw); localDevMax=std::max(localDevMax,ratioDev);
        localRawStrainMax=std::max(localRawStrainMax,strainRaw); localDevStrainMax=std::max(localDevStrainMax,strainDev);
        ++localSamples;
        if(detailedAudit) {
          if(ratio>100.0) localWeightGT100+=w;
          if(ratio>300.0) localWeightGT300+=w;
          if(ratio>1000.0) localWeightGT1000+=w;
          auto addAuditHist=[&](double v,std::vector<double>& hist) {
            const double clipped=std::min(auditMaxRatio,std::max(0.0,v));
            int ib=(int)std::floor((std::log1p(clipped)/auditLogMax)*(auditBins-1));
            ib=std::max(0,std::min(auditBins-1,ib));
            hist[(std::size_t)ib]+=w;
          };
          addAuditHist(ratio,localHist);
          addAuditHist(ratioRaw,localRawHist);
          addAuditHist(ratioDev,localDevHist);
        }
        if(ratio>localMax) {
          localMax=ratio;
          double zq=0.0; for(int i=0;i<4;++i) zq+=rq.lam[i]*(double)cp.z[i];
          const double fRef=(pipe.re>0.0)?0.3164*std::pow(pipe.re,-0.25):0.0;
          const double uTauRef=std::abs(pipe.bulkVelocity)*std::sqrt(std::max(0.0,fRef/8.0));
          localMaxAudit.cellGid=(long long)cp.diagnosticCellGid;
          localMaxAudit.q=(int)q; localMaxAudit.x=xq; localMaxAudit.y=yq; localMaxAudit.z=zq;
          localMaxAudit.zOverL=(pipe.L>0.0)?(zq-pipe.zIn)/pipe.L:0.0;
          localMaxAudit.rOverR=(pipe.R>0.0)?r/pipe.R:0.0;
          localMaxAudit.yPlusRefBlasius=(pipe.R>0.0)?std::max(0.0,pipe.R-r)*uTauRef/nu:0.0;
          localMaxAudit.lmOverR=(pipe.R>0.0)?ell/pipe.R:0.0;
          localMaxAudit.nuTRatio=ratio; localMaxAudit.twoSdotS=twoSdotS; localMaxAudit.strain=strain;
          localMaxAudit.divU=divU;
          localMaxAudit.nuTRatioRaw=ratioRaw; localMaxAudit.twoSdotSRaw=twoSdotSRaw; localMaxAudit.strainRaw=strainRaw;
          localMaxAudit.nuTRatioDev=ratioDev; localMaxAudit.twoSdotSDev=twoSdotSDev; localMaxAudit.strainDev=strainDev;
          localMaxAudit.Sxx=S[0][0]; localMaxAudit.Syy=S[1][1]; localMaxAudit.Szz=S[2][2];
          localMaxAudit.Sxy=S[0][1]; localMaxAudit.Sxz=S[0][2]; localMaxAudit.Syz=S[1][2];
        }
      }
    }

    for(int ii=0;ii<(int)cp.nOwnedRows;++ii) {
      const int a=(int)cp.ownedBasis[ii]; const PetscInt lr=cp.ref[a];
      for(int b=0;b<8;++b) {
        if(dynRefFree(cp.ref[b])) {
          const std::uint8_t slot=cp.rowSlot[a][b];
          if(slot==255) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"mixing-length diffusion missing row slot");
          A.aRel[(std::size_t)(A.rowPtr[(std::size_t)lr]+slot)]+=Kt[a][b];
        } else {
          const PetscInt e=dynFixedEntity(cp.ref[b]);
          for(int d=0;d<3;++d) {
            const double ud=entityDirValue(D,d,e);
            if(ud!=0.0) A.mixlenRhs[d][(std::size_t)lr]-=Kt[a][b]*ud;
          }
        }
      }
    }
  }

  double globalMin=0.0,globalMax=0.0,globalWeighted=0.0,globalWeight=0.0;
  double globalEllMin=0.0,globalEllMax=0.0,globalStrainMax=0.0;
  double globalRawMax=0.0,globalDevMax=0.0,globalRawStrainMax=0.0,globalDevStrainMax=0.0;
  unsigned long long globalSamples=0;
  PetscCallMPI(MPI_Allreduce(&localMin,&globalMin,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localMax,&globalMax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localWeighted,&globalWeighted,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localWeight,&globalWeight,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localEllMin,&globalEllMin,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localEllMax,&globalEllMax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localStrainMax,&globalStrainMax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localRawMax,&globalRawMax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localDevMax,&globalDevMax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localRawStrainMax,&globalRawStrainMax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localDevStrainMax,&globalDevStrainMax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localSamples,&globalSamples,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));

  stats.nuTRatioMin=globalSamples?globalMin:0.0;
  stats.nuTRatioMean=(globalWeight>0.0)?globalWeighted/globalWeight:0.0;
  stats.nuTRatioMax=globalSamples?globalMax:0.0;
  stats.ellMin=globalSamples?globalEllMin:0.0;
  stats.ellMax=globalSamples?globalEllMax:0.0;
  stats.strainMax=globalStrainMax;
  stats.nuTRatioP95=stats.nuTRatioP99=stats.nuTRatioP999=0.0;
  stats.volFracGT100=stats.volFracGT300=stats.volFracGT1000=0.0;
  stats.rawP95=stats.rawP99=stats.rawP999=0.0; stats.rawMax=globalRawMax; stats.rawStrainMax=globalRawStrainMax;
  stats.devP95=stats.devP99=stats.devP999=0.0; stats.devMax=globalDevMax; stats.devStrainMax=globalDevStrainMax;
  stats.maxAudit={};
  stats.quadratureSamples=globalSamples;

  if(detailedAudit && globalSamples) {
    double localGT[3]={localWeightGT100,localWeightGT300,localWeightGT1000},globalGT[3]={0,0,0};
    PetscCallMPI(MPI_Allreduce(localGT,globalGT,3,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
    if(globalWeight>0.0) {
      stats.volFracGT100=globalGT[0]/globalWeight; stats.volFracGT300=globalGT[1]/globalWeight; stats.volFracGT1000=globalGT[2]/globalWeight;
    }
    std::vector<double> globalHist(auditBins,0.0),globalRawHist(auditBins,0.0),globalDevHist(auditBins,0.0);
    PetscCallMPI(MPI_Allreduce(localHist.data(),globalHist.data(),auditBins,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(localRawHist.data(),globalRawHist.data(),auditBins,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(localDevHist.data(),globalDevHist.data(),auditBins,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
    auto histQuantile=[&](const std::vector<double>& hist,double qv) {
      if(!(globalWeight>0.0)) return 0.0;
      const double target=qv*globalWeight; double c=0.0; int ib=auditBins-1;
      for(int i=0;i<auditBins;++i) { c+=hist[(std::size_t)i]; if(c>=target) { ib=i; break; } }
      const double t=((double)ib+0.5)/(double)(auditBins-1);
      return std::expm1(std::min(1.0,t)*auditLogMax);
    };
    stats.nuTRatioP95=histQuantile(globalHist,0.95); stats.nuTRatioP99=histQuantile(globalHist,0.99); stats.nuTRatioP999=histQuantile(globalHist,0.999);
    stats.rawP95=histQuantile(globalRawHist,0.95); stats.rawP99=histQuantile(globalRawHist,0.99); stats.rawP999=histQuantile(globalRawHist,0.999);
    stats.devP95=histQuantile(globalDevHist,0.95); stats.devP99=histQuantile(globalDevHist,0.99); stats.devP999=histQuantile(globalDevHist,0.999);

    struct { double value; int rank; } localPair{localMax,rank},globalPair{0.0,0};
    PetscCallMPI(MPI_Allreduce(&localPair,&globalPair,1,MPI_DOUBLE_INT,MPI_MAXLOC,PETSC_COMM_WORLD));
    if(rank==globalPair.rank) stats.maxAudit=localMaxAudit;
    PetscCallMPI(MPI_Bcast(&stats.maxAudit,(int)sizeof(stats.maxAudit),MPI_BYTE,globalPair.rank,PETSC_COMM_WORLD));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}


// -----------------------------------------------------------------------------
// Stage-3 weak Spalding wall function for the straight z-axis pipe.
//
// The cylindrical wall has n_z=0, therefore the axial Uz trace is exactly
// tangential.  We keep Ux/Uy strongly zero through the algebraic wall clamp and
// impose only the axial condition weakly:
//
//   -int_Gamma nu (dn u) w
//   -int_Gamma nu (dn w) u
//   +int_Gamma beta_w u w,
//       beta_w = u_tau^2 / |u_t|
//
// with u_tau obtained from Spalding's all-y+ law using y = distanceFactor*h_n,
// h_n=3V/A_f.  At U_t -> 0 the coefficient tends smoothly to nu/y.
//
// No extra C*nu/h tangential penalty is added: beta_w itself is the wall-model
// penalty.  The legacy/default linearization is the lagged secant beta_w.
// An optional consistent-tangent mode replaces only the wall Jacobian and adds
// the compensating RHS required to preserve the exact same nonlinear fixed point.
// -----------------------------------------------------------------------------
struct WallFunctionStats {
  double betaMin=0.0,betaMean=0.0,betaMax=0.0;
  double tangentMin=0.0,tangentMean=0.0,tangentMax=0.0;
  double usedJacMin=0.0,usedJacMean=0.0,usedJacMax=0.0;
  double tangentRatioMin=0.0,tangentRatioMean=0.0,tangentRatioMax=0.0;
  double tangentRhsAbsMin=0.0,tangentRhsAbsMean=0.0,tangentRhsAbsMax=0.0;
  double yPlusMin=0.0,yPlusMean=0.0,yPlusMax=0.0;
  double uTauMin=0.0,uTauMean=0.0,uTauMax=0.0;
  double uTau2Mean=0.0,fWallSpalding=0.0;
  std::array<double,10> axialArea{};
  std::array<double,10> axialUTau2Integral{};
  double slipMin=0.0,slipMean=0.0,slipMax=0.0;
  double yMin=0.0,yMean=0.0,yMax=0.0;
  double maxAbsNormalZ=0.0;
  unsigned long long wallFaces=0,quadratureSamples=0,rootFailures=0,tangentFailures=0;
};

struct TriangleQuadPoint {
  double l[3];
  double w;
};

static const std::array<TriangleQuadPoint,7>& triangleDunavant7() {
  static const std::array<TriangleQuadPoint,7> Q={{
    {{1.0/3.0,1.0/3.0,1.0/3.0},0.225000000000000},
    {{0.059715871789770,0.470142064105115,0.470142064105115},0.132394152788506},
    {{0.470142064105115,0.059715871789770,0.470142064105115},0.132394152788506},
    {{0.470142064105115,0.470142064105115,0.059715871789770},0.132394152788506},
    {{0.797426985353087,0.101286507323456,0.101286507323456},0.125939180544827},
    {{0.101286507323456,0.797426985353087,0.101286507323456},0.125939180544827},
    {{0.101286507323456,0.101286507323456,0.797426985353087},0.125939180544827}
  }};
  return Q;
}

static double spaldingYPlusFromUPlus(double up,double kappa,double B) {
  if(!(up>=0.0)) return std::numeric_limits<double>::quiet_NaN();
  const double x=kappa*up;
  double rem=0.0;
  if(std::abs(x)<1e-3) {
    const double x2=x*x,x4=x2*x2;
    rem=x4*(1.0/24.0 + x/120.0 + x2/720.0 + x2*x/5040.0);
  } else {
    rem=std::expm1(x)-x-0.5*x*x-(x*x*x)/6.0;
  }
  return up + std::exp(-kappa*B)*rem;
}

// Derivative d(y+)/d(u+) of the Spalding map above.  A small-x series
// avoids cancellation in exp(x)-1-x-x^2/2.
static double spaldingYPlusDerivativeFromUPlus(double up,double kappa,double B) {
  if(!(up>=0.0)) return std::numeric_limits<double>::quiet_NaN();
  const double x=kappa*up;
  double remPrimeOverKappa=0.0;
  if(std::abs(x)<1e-3) {
    const double x2=x*x,x3=x2*x;
    remPrimeOverKappa=x3*(1.0/6.0 + x/24.0 + x2/120.0 + x3/720.0);
  } else {
    remPrimeOverKappa=std::expm1(x)-x-0.5*x*x;
  }
  return 1.0 + std::exp(-kappa*B)*kappa*remPrimeOverKappa;
}

// Consistent scalar tangent of signed Spalding wall traction
// tau_w(U)=u_tau(|U|)^2 sign(U).  beta is the secant tau_w/U.
// For U -> 0 the wall law is linear and tangent == beta == nu/y.
static bool spaldingTangentFromSolvedState(
    double uPlus,double beta,double kappa,double B,double& tangent,double& tangentRatio) {
  if(!(beta>0.0) || !std::isfinite(beta) || !(uPlus>=0.0) || !std::isfinite(uPlus)) return false;
  if(uPlus<=1e-14) { tangent=beta; tangentRatio=1.0; return true; }
  const double F=spaldingYPlusFromUPlus(uPlus,kappa,B);
  const double Fp=spaldingYPlusDerivativeFromUPlus(uPlus,kappa,B);
  const double den=F+uPlus*Fp;
  if(!(F>0.0) || !(Fp>0.0) || !(den>0.0) || !std::isfinite(den)) return false;
  tangentRatio=2.0*uPlus*Fp/den;
  tangent=beta*tangentRatio;
  return std::isfinite(tangent) && tangent>0.0 && std::isfinite(tangentRatio) && tangentRatio>0.0;
}

static bool spaldingPenaltyFromSlip(double slip,double y,double nu,double kappa,double B,
                                    double& uTau,double& yPlus,double& uPlus,double& beta) {
  if(!(y>0.0) || !(nu>0.0) || !(kappa>0.0) || !(B>0.0) || !std::isfinite(slip))
    return false;
  const double U=std::abs(slip);
  const double reY=y*U/nu;
  if(reY<=1e-14) {
    uTau=0.0; yPlus=0.0; uPlus=0.0; beta=nu/y;
    return std::isfinite(beta);
  }

  auto g=[&](double up)->double {
    return up*spaldingYPlusFromUPlus(up,kappa,B)-reY;
  };

  double lo=0.0,hi=std::max(1.0,std::sqrt(reY)+1.0);
  int expand=0;
  while(g(hi)<0.0 && expand<40) { hi*=2.0; ++expand; }
  if(!std::isfinite(g(hi)) || g(hi)<0.0) return false;

  for(int it=0;it<70;++it) {
    const double mid=0.5*(lo+hi);
    if(g(mid)>0.0) hi=mid; else lo=mid;
  }
  uPlus=0.5*(lo+hi);
  if(!(uPlus>0.0) || !std::isfinite(uPlus)) return false;
  uTau=U/uPlus;
  yPlus=y*uTau/nu;
  beta=(uTau*uTau)/U;
  return std::isfinite(uTau) && std::isfinite(yPlus) && std::isfinite(beta) && beta>0.0;
}

// NODALS_TET_WALL_FIREDRAKE_PARITY_BCD_20260908
// NODALS_TET_WALL_AXIAL_FRICTION_BINS_20260908
// legacy_face_trace + molecularConsistency=1 remains the historical oracle.
// offwall_quarterplane evaluates the exact P1+BF3 trial trace at the tet
// barycentric plane lambda_opp=distanceFactor and measures the physical radial
// wall-to-sample distance.  Molecular Nitsche consistency is independently
// switchable so B/C/D isolate the two changes.
static PetscErrorCode assembleSpaldingWeakWallCustom(
    const Discrete& D,const CustomDynamicRuntimePlan& P,CustomMomentumCSR& A,
    double nu,double kappa,double B,double distanceFactor,double betaScale,double pipeCx,double pipeCy,
    const std::string& wallLinearization,double wallTangentBlend,
    const std::string& wallSampleMode,PetscBool wallMolecularConsistency,double bulkVelocity,
    double pipeZIn,double pipeLength,
    WallFunctionStats& stats) {
  PetscFunctionBeginUser;
  if(!(nu>0.0) || !(kappa>0.0) || !(B>0.0) || !(distanceFactor>0.0) || betaScale<0.0)
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"invalid Stage-3 Spalding wall parameters");
  if(wallLinearization!="secant" && wallLinearization!="consistent_tangent")
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"wall linearization must be secant or consistent_tangent");
  if(!(wallTangentBlend>=0.0 && wallTangentBlend<=1.0))
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"wall tangent blend must satisfy 0 <= blend <= 1");
  const bool useConsistentTangent=(wallLinearization=="consistent_tangent");
  const bool offwallSample=(wallSampleMode=="offwall_quarterplane");
  if(wallSampleMode!="legacy_face_trace" && !offwallSample)
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"wall sample mode must be legacy_face_trace or offwall_quarterplane");
  if(offwallSample && !(distanceFactor>0.0 && distanceFactor<1.0))
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"offwall_quarterplane requires 0 < wall_distance_factor < 1");
  if(!(bulkVelocity>0.0) || !std::isfinite(bulkVelocity))
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Spalding wall diagnostic requires positive bulk velocity");
  if(!(pipeLength>0.0) || !std::isfinite(pipeZIn) || !std::isfinite(pipeLength))
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Spalding axial-bin diagnostic requires finite zIn and positive pipe length");
  for(int d=0;d<3;++d) {
    if((PetscInt)A.wallRhs[d].size()!=A.nOwned) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Stage-3 wall RHS storage not enabled");
    std::fill(A.wallRhs[d].begin(),A.wallRhs[d].end(),0.0);
  }

  double lBetaMin=1e300,lBetaMax=0.0,lBetaW=0.0;
  double lTanMin=1e300,lTanMax=0.0,lTanW=0.0;
  double lUsedJacMin=1e300,lUsedJacMax=0.0,lUsedJacW=0.0;
  double lTanRatioMin=1e300,lTanRatioMax=0.0,lTanRatioW=0.0;
  double lTanRhsMin=1e300,lTanRhsMax=0.0,lTanRhsW=0.0;
  double lYpMin=1e300,lYpMax=0.0,lYpW=0.0;
  double lUtMin=1e300,lUtMax=0.0,lUtW=0.0,lUt2W=0.0;
  double lSlipMin=1e300,lSlipMax=0.0,lSlipW=0.0;
  double lYMin=1e300,lYMax=0.0,lYW=0.0,lAreaW=0.0,lNz=0.0;
  std::array<double,10> lAxialArea{};
  std::array<double,10> lAxialUTau2Integral{};
  unsigned long long lFaces=0,lSamples=0,lFail=0,lTanFail=0;

  const auto& FQ=triangleDunavant7();
  for(const auto& wf:P.wallFaces) {
    if((std::size_t)wf.cellIndex>=P.cells.size())
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Stage-3 wall-face references invalid runtime cell");
    const auto& cp=P.cells[(std::size_t)wf.cellIndex];
    const int opp=(int)wf.opp;
    if(opp<0 || opp>3 || !(wf.area>0.0) || !(wf.hn>0.0))
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"invalid Stage-3 wall-face geometry");
    const double legacyY=distanceFactor*wf.hn;
    // Individual planar facets on the triangulated cylinder can have n_z != 0
    // even though the underlying physical cylinder is exactly z-aligned. Use
    // the analytic radial cylinder normal for the weak wall flux and retain the
    // facet |n_z| only as a geometry diagnostic.
    const double nz=std::abs(wf.normal[2]);

    double coeff[3][8]={{0}};
    for(int m=0;m<8;++m) {
      if(dynRefFree(cp.ref[m])) {
        for(int d=0;d<3;++d) coeff[d][m]=customMomentumFieldValue(A,d,cp.ref[m]);
      } else {
        const PetscInt e=dynFixedEntity(cp.ref[m]);
        for(int d=0;d<3;++d) coeff[d][m]=entityDirValue(D,d,e);
      }
    }

    int fv[3],kk=0;
    for(int i=0;i<4;++i) if(i!=opp) fv[kk++]=i;
    double W[8][8]={{0}};
    double Rtangent[8]={0};

    for(const auto& q:FQ) {
      std::array<double,4> lam{{0.0,0.0,0.0,0.0}};
      for(int j=0;j<3;++j) lam[fv[j]]=q.l[j];
      double phiWall[8],gradRefWall[8][3],gradWall[8][3]={{0}};
      basis(lam,phiWall,gradRefWall);
      for(int a=0;a<8;++a) for(int d=0;d<3;++d)
        for(int j=0;j<3;++j) gradWall[a][d]+=gradRefWall[a][j]*cp.invJ[j][d];

      double xq=0.0,yq=0.0;
      for(int i=0;i<4;++i) { xq+=lam[i]*cp.xy[i][0]; yq+=lam[i]*cp.xy[i][1]; }
      const double rx=xq-pipeCx, ry=yq-pipeCy, rr=std::hypot(rx,ry);
      if(!(rr>0.0) || !std::isfinite(rr))
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"invalid Stage-3 analytic radial wall normal");
      const double nwx=rx/rr, nwy=ry/rr; // analytic cylinder normal; n_w,z = 0 exactly

      // Trial trace used by the Spalding traction.  The legacy path samples on
      // the wall face.  Firedrake-parity mode moves the wall quadrature point
      // to a parallel interior tet plane:
      //   lambda_opp = distanceFactor,
      //   lambda_i   = (1-distanceFactor)*lambda_i_wall, i != opp.
      // With distanceFactor=0.25 this is the affine-tet centroid depth because
      // a tet centroid has all four barycentric coordinates equal to 1/4.
      double phiSample[8];
      for(int a=0;a<8;++a) phiSample[a]=phiWall[a];
      double y=legacyY;
      if(offwallSample) {
        std::array<double,4> lamSample=lam;
        for(int i=0;i<4;++i) if(i!=opp) lamSample[i]*=(1.0-distanceFactor);
        lamSample[opp]=distanceFactor;
        double gradRefSampleDummy[8][3];
        basis(lamSample,phiSample,gradRefSampleDummy);

        double xsx=0.0,xsy=0.0;
        for(int i=0;i<4;++i) { xsx+=lamSample[i]*cp.xy[i][0]; xsy+=lamSample[i]*cp.xy[i][1]; }
        y=std::abs((xq-xsx)*nwx + (yq-xsy)*nwy);
        if(!(y>0.0) || !std::isfinite(y))
          SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"invalid tet off-wall Spalding sample distance");
      }

      double uz=0.0;
      for(int a=0;a<8;++a) uz+=coeff[2][a]*phiSample[a];
      const double slip=std::abs(uz);

      double uTau=0.0,yPlus=0.0,uPlus=0.0,beta=0.0;
      const bool ok=spaldingPenaltyFromSlip(slip,y,nu,kappa,B,uTau,yPlus,uPlus,beta);
      if(!ok) {
        if(wf.diagnosticOwner) ++lFail;
        beta=nu/y; uTau=0.0; yPlus=0.0; uPlus=0.0;
      }

      // beta is the current secant tau_w/U.  The consistent tangent is the
      // exact derivative d(tau_w)/dU of the same signed Spalding traction.
      // betaScale multiplies both so changing the linearization does not alter
      // the nonlinear wall law.  In consistent_tangent mode a blend theta is
      // allowed between the old secant and the full tangent.
      double tangent=beta,tangentRatio=1.0;
      if(ok && !spaldingTangentFromSolvedState(uPlus,beta,kappa,B,tangent,tangentRatio)) {
        if(wf.diagnosticOwner) ++lTanFail;
        tangent=beta; tangentRatio=1.0;
      }
      beta*=betaScale;
      tangent*=betaScale;
      const double usedJac=useConsistentTangent ? beta+wallTangentBlend*(tangent-beta) : beta;
      // Residual has +tau_w(U).  Newton/Picard linearization
      // tau_w(U_new) ~= usedJac*U_new + (beta-usedJac)*U_old, hence
      // A*U_new = ... + (usedJac-beta)*U_old.  This compensating RHS is
      // essential: omitting it would change the converged wall condition.
      const double tangentRhsDensity=(usedJac-beta)*uz;
      const double tangentRhsAbs=std::abs(tangentRhsDensity);
      const double w=q.w*wf.area;

      double dnWall[8]={0};
      for(int a=0;a<8;++a)
        dnWall[a]=gradWall[a][0]*nwx+gradWall[a][1]*nwy;

      for(int a=0;a<8;++a) {
        for(int b=0;b<8;++b) {
          // Keep the historical molecular Nitsche pair strictly wall-wall when
          // enabled.  This makes B isolate sampling and C isolate consistency.
          const double molecular=wallMolecularConsistency
            ? (-nu*dnWall[b]*phiWall[a]-nu*dnWall[a]*phiWall[b]) : 0.0;
          // Spalding Robin traction is wall-test x sampled-trial in off-wall
          // mode, and reduces exactly to wall-test x wall-trial in legacy mode.
          W[a][b]+=(molecular+usedJac*phiWall[a]*phiSample[b])*w;
        }
        Rtangent[a]+=tangentRhsDensity*phiWall[a]*w;
      }

      if(wf.diagnosticOwner) {
        lBetaMin=std::min(lBetaMin,beta); lBetaMax=std::max(lBetaMax,beta); lBetaW+=beta*w;
        lTanMin=std::min(lTanMin,tangent); lTanMax=std::max(lTanMax,tangent); lTanW+=tangent*w;
        lUsedJacMin=std::min(lUsedJacMin,usedJac); lUsedJacMax=std::max(lUsedJacMax,usedJac); lUsedJacW+=usedJac*w;
        lTanRatioMin=std::min(lTanRatioMin,tangentRatio); lTanRatioMax=std::max(lTanRatioMax,tangentRatio); lTanRatioW+=tangentRatio*w;
        lTanRhsMin=std::min(lTanRhsMin,tangentRhsAbs); lTanRhsMax=std::max(lTanRhsMax,tangentRhsAbs); lTanRhsW+=tangentRhsAbs*w;
        lYpMin=std::min(lYpMin,yPlus); lYpMax=std::max(lYpMax,yPlus); lYpW+=yPlus*w;
        lUtMin=std::min(lUtMin,uTau); lUtMax=std::max(lUtMax,uTau); lUtW+=uTau*w; lUt2W+=uTau*uTau*w;
        lSlipMin=std::min(lSlipMin,slip); lSlipMax=std::max(lSlipMax,slip); lSlipW+=slip*w;
        lYMin=std::min(lYMin,y); lYMax=std::max(lYMax,y); lYW+=y*w;
        lAreaW+=w; lNz=std::max(lNz,nz); ++lSamples;
        double zq=0.0;
        for(int i=0;i<4;++i) zq+=lam[i]*(double)cp.z[i];
        const double zFrac=(zq-pipeZIn)/pipeLength;
        int axialBin=(int)std::floor(10.0*zFrac);
        axialBin=std::max(0,std::min(9,axialBin));
        lAxialArea[(std::size_t)axialBin]+=w;
        lAxialUTau2Integral[(std::size_t)axialBin]+=uTau*uTau*w;
      }
    }

    if(wf.diagnosticOwner) ++lFaces;

    // The matrix is shared by all scalar components.  For Ux/Uy every wall
    // trace DOF is clamped to zero, so these Nitsche rows/columns are inactive.
    // For Uz the same entries provide the weak wall-function operator.
    for(int ii=0;ii<(int)cp.nOwnedRows;++ii) {
      const int a=(int)cp.ownedBasis[ii];
      const PetscInt lr=cp.ref[a];
      A.wallRhs[2][(std::size_t)lr]+=Rtangent[a];
      for(int b=0;b<8;++b) {
        if(dynRefFree(cp.ref[b])) {
          const std::uint8_t slot=cp.rowSlot[a][b];
          if(slot==255) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Stage-3 wall operator missing row slot");
          A.aRel[(std::size_t)(A.rowPtr[(std::size_t)lr]+slot)]+=W[a][b];
        } else {
          const PetscInt e=dynFixedEntity(cp.ref[b]);
          const double uzFixed=entityDirValue(D,2,e);
          if(uzFixed!=0.0) A.wallRhs[2][(std::size_t)lr]-=W[a][b]*uzFixed;
        }
      }
    }
  }

  auto redMin=[&](double l,double& g){PetscCallMPI(MPI_Allreduce(&l,&g,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD));return PETSC_SUCCESS;};
  auto redMax=[&](double l,double& g){PetscCallMPI(MPI_Allreduce(&l,&g,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));return PETSC_SUCCESS;};
  auto redSum=[&](double l,double& g){PetscCallMPI(MPI_Allreduce(&l,&g,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));return PETSC_SUCCESS;};
  double gBetaMin=0,gBetaMax=0,gBetaW=0,gTanMin=0,gTanMax=0,gTanW=0,gUsedJacMin=0,gUsedJacMax=0,gUsedJacW=0;
  double gTanRatioMin=0,gTanRatioMax=0,gTanRatioW=0,gTanRhsMin=0,gTanRhsMax=0,gTanRhsW=0;
  double gYpMin=0,gYpMax=0,gYpW=0,gUtMin=0,gUtMax=0,gUtW=0,gUt2W=0;
  double gSlipMin=0,gSlipMax=0,gSlipW=0,gYMin=0,gYMax=0,gYW=0,gAreaW=0,gNz=0;
  std::array<double,10> gAxialArea{};
  std::array<double,10> gAxialUTau2Integral{};
  PetscCall(redMin(lBetaMin,gBetaMin)); PetscCall(redMax(lBetaMax,gBetaMax)); PetscCall(redSum(lBetaW,gBetaW));
  PetscCall(redMin(lTanMin,gTanMin)); PetscCall(redMax(lTanMax,gTanMax)); PetscCall(redSum(lTanW,gTanW));
  PetscCall(redMin(lUsedJacMin,gUsedJacMin)); PetscCall(redMax(lUsedJacMax,gUsedJacMax)); PetscCall(redSum(lUsedJacW,gUsedJacW));
  PetscCall(redMin(lTanRatioMin,gTanRatioMin)); PetscCall(redMax(lTanRatioMax,gTanRatioMax)); PetscCall(redSum(lTanRatioW,gTanRatioW));
  PetscCall(redMin(lTanRhsMin,gTanRhsMin)); PetscCall(redMax(lTanRhsMax,gTanRhsMax)); PetscCall(redSum(lTanRhsW,gTanRhsW));
  PetscCall(redMin(lYpMin,gYpMin)); PetscCall(redMax(lYpMax,gYpMax)); PetscCall(redSum(lYpW,gYpW));
  PetscCall(redMin(lUtMin,gUtMin)); PetscCall(redMax(lUtMax,gUtMax)); PetscCall(redSum(lUtW,gUtW)); PetscCall(redSum(lUt2W,gUt2W));
  PetscCall(redMin(lSlipMin,gSlipMin)); PetscCall(redMax(lSlipMax,gSlipMax)); PetscCall(redSum(lSlipW,gSlipW));
  PetscCall(redMin(lYMin,gYMin)); PetscCall(redMax(lYMax,gYMax)); PetscCall(redSum(lYW,gYW));
  PetscCall(redSum(lAreaW,gAreaW)); PetscCall(redMax(lNz,gNz));
  PetscCallMPI(MPI_Allreduce(lAxialArea.data(),gAxialArea.data(),10,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(lAxialUTau2Integral.data(),gAxialUTau2Integral.data(),10,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  unsigned long long gFaces=0,gSamples=0,gFail=0,gTanFail=0;
  PetscCallMPI(MPI_Allreduce(&lFaces,&gFaces,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&lSamples,&gSamples,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&lFail,&gFail,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&lTanFail,&gTanFail,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));

  if(gFaces==0 || gSamples==0 || !(gAreaW>0.0))
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Stage-3 weak wall enabled but no cylindrical wall-face quadrature samples were found");
  stats.wallFaces=gFaces; stats.quadratureSamples=gSamples; stats.rootFailures=gFail; stats.tangentFailures=gTanFail; stats.maxAbsNormalZ=gNz;
  if(gSamples && gAreaW>0.0) {
    stats.betaMin=gBetaMin; stats.betaMean=gBetaW/gAreaW; stats.betaMax=gBetaMax;
    stats.tangentMin=gTanMin; stats.tangentMean=gTanW/gAreaW; stats.tangentMax=gTanMax;
    stats.usedJacMin=gUsedJacMin; stats.usedJacMean=gUsedJacW/gAreaW; stats.usedJacMax=gUsedJacMax;
    stats.tangentRatioMin=gTanRatioMin; stats.tangentRatioMean=gTanRatioW/gAreaW; stats.tangentRatioMax=gTanRatioMax;
    stats.tangentRhsAbsMin=gTanRhsMin; stats.tangentRhsAbsMean=gTanRhsW/gAreaW; stats.tangentRhsAbsMax=gTanRhsMax;
    stats.yPlusMin=gYpMin; stats.yPlusMean=gYpW/gAreaW; stats.yPlusMax=gYpMax;
    stats.uTauMin=gUtMin; stats.uTauMean=gUtW/gAreaW; stats.uTauMax=gUtMax;
    stats.uTau2Mean=gUt2W/gAreaW;
    stats.fWallSpalding=8.0*betaScale*stats.uTau2Mean/(bulkVelocity*bulkVelocity);
    stats.axialArea=gAxialArea;
    stats.axialUTau2Integral=gAxialUTau2Integral;
    stats.slipMin=gSlipMin; stats.slipMean=gSlipW/gAreaW; stats.slipMax=gSlipMax;
    stats.yMin=gYMin; stats.yMean=gYW/gAreaW; stats.yMax=gYMax;
  } else stats={};
  PetscFunctionReturn(PETSC_SUCCESS);
}

