struct SupgStats {
  PetscReal tauMin=0,tauMean=0,tauMax=0;
};

static PetscErrorCode assembleSupgLegacyMPI(
    const Mesh& M,const Discrete& D,int rank,const GhostPlan& G,Vec U[3],
    const ProblemConfig& P,double tauScale,double supgMagic,Mat Sg,Vec supgRhs[3],SupgStats& stats) {
  PetscFunctionBeginUser;
  PetscCall(MatZeroEntries(Sg));
  for(int d=0;d<3;++d) PetscCall(VecSet(supgRhs[d],0.0));

  Vec Ul[3]={nullptr,nullptr,nullptr};
  const PetscScalar* ua[3]={nullptr,nullptr,nullptr};
  for(int d=0;d<3;++d) {
    PetscCall(VecGhostUpdateBegin(U[d],INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecGhostUpdateEnd(U[d],INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecGhostGetLocalForm(U[d],&Ul[d]));
    PetscCall(VecGetArrayRead(Ul[d],&ua[d]));
  }

  const auto Q=tetDuffy5();
  const PetscInt nv=(PetscInt)M.points.size();
  double localTauMin=1.0e300,localTauMax=0.0,localTauWeighted=0.0,localWeight=0.0;

  for(PetscInt c=0;c<(PetscInt)M.tets.size();++c) if(D.cellOwner[c]==rank) {
    const auto t=M.tets[c];
    const Vec3 X[4]={M.points[t[0]],M.points[t[1]],M.points[t[2]],M.points[t[3]]};
    double J[3][3]={{X[1].x-X[0].x,X[2].x-X[0].x,X[3].x-X[0].x},
                    {X[1].y-X[0].y,X[2].y-X[0].y,X[3].y-X[0].y},
                    {X[1].z-X[0].z,X[2].z-X[0].z,X[3].z-X[0].z}},invJ[3][3];
    const double det=det3(J);
    inv3(J,invJ);
    const double h=tetDiameter(X),h2=h*h;
    if(!(h2>0.0)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"zero tetrahedron diameter in SUPG");

    PetscInt lg[8];
    for(int i=0;i<4;++i) lg[i]=t[i];
    for(int i=0;i<4;++i) lg[4+i]=nv+M.oppFace[c][i];
    double coeff[3][8]={{0}};
    for(int m=0;m<8;++m) {
      const PetscInt gid=D.g2free[lg[m]];
      if(gid>=0) {
        const PetscInt li=velocityLocalIndex(G,gid);
        for(int d=0;d<3;++d) coeff[d][m]=PetscRealPart(ua[d][li]);
      } else {
        for(int d=0;d<3;++d) coeff[d][m]=entityDirValue(D,d,lg[m]);
      }
    }

    double Sl[8][8]={{0}};
    double Fr[3][8]={{0}};
    for(const auto& q:Q) {
      double val[8],grr[8][3],Href[8][3][3];
      basis(q.lam,val,grr);
      referenceHessian(q.lam,Href);
      double gr[8][3]={{0}},lap[8]={0};
      for(int a=0;a<8;++a) {
        for(int d=0;d<3;++d) for(int j=0;j<3;++j) gr[a][d]+=grr[a][j]*invJ[j][d];
        for(int d=0;d<3;++d) for(int r=0;r<3;++r) for(int ss=0;ss<3;++ss)
          lap[a]+=Href[a][r][ss]*invJ[r][d]*invJ[ss][d];
      }

      double adv[3]={0,0,0};
      for(int d=0;d<3;++d) for(int m=0;m<8;++m) adv[d]+=coeff[d][m]*val[m];
      double speed2=0.0; for(int d=0;d<3;++d) speed2+=adv[d]*adv[d];
      const double diffusive=4.0*P.nu/h2;
      const double denominator=std::max(4.0*speed2/h2 + supgMagic*diffusive*diffusive,1.0e-30);
      const double tau=tauScale/std::sqrt(denominator);
      const double w=q.w*det;
      localTauMin=std::min(localTauMin,tau); localTauMax=std::max(localTauMax,tau);
      localTauWeighted+=tau*w; localWeight+=w;

      double stream[8]={0},strongTrial[8]={0};
      for(int a=0;a<8;++a) {
        for(int d=0;d<3;++d) stream[a]+=adv[d]*gr[a][d];
        strongTrial[a]=-P.nu*lap[a]+stream[a];
      }
      double x=0,y=0,z=0;
      for(int i=0;i<4;++i){x+=q.lam[i]*X[i].x;y+=q.lam[i]*X[i].y;z+=q.lam[i]*X[i].z;}
      double ff[3]; problemForcing(P,x,y,z,ff);

      for(int a=0;a<8;++a) {
        const double tv=tau*stream[a]*w;
        for(int b=0;b<8;++b) Sl[a][b]+=tv*strongTrial[b];
        for(int d=0;d<3;++d) Fr[d][a]+=tv*ff[d];
      }
    }

    for(int a=0;a<8;++a) {
      const PetscInt ia=D.g2free[lg[a]];
      if(ia<0) continue;
      for(int b=0;b<8;++b) {
        const PetscInt ib=D.g2free[lg[b]];
        if(ib>=0) PetscCall(MatSetValue(Sg,ia,ib,Sl[a][b],ADD_VALUES));
        else for(int d=0;d<3;++d) Fr[d][a]-=Sl[a][b]*entityDirValue(D,d,lg[b]);
      }
      for(int d=0;d<3;++d) PetscCall(VecSetValue(supgRhs[d],ia,Fr[d][a],ADD_VALUES));
    }
  }

  for(int d=0;d<3;++d) {
    PetscCall(VecRestoreArrayRead(Ul[d],&ua[d]));
    PetscCall(VecGhostRestoreLocalForm(U[d],&Ul[d]));
  }
  PetscCall(MatAssemblyBegin(Sg,MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(Sg,MAT_FINAL_ASSEMBLY));
  for(int d=0;d<3;++d){PetscCall(VecAssemblyBegin(supgRhs[d]));PetscCall(VecAssemblyEnd(supgRhs[d]));}

  double globalMin=0,globalMax=0,globalTauWeighted=0,globalWeight=0;
  PetscCallMPI(MPI_Allreduce(&localTauMin,&globalMin,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localTauMax,&globalMax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localTauWeighted,&globalTauWeighted,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localWeight,&globalWeight,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  stats.tauMin=globalMin; stats.tauMax=globalMax;
  stats.tauMean=(globalWeight>0)?globalTauWeighted/globalWeight:0.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}


struct SupgReferencePoint {
  std::array<double,4> lam{};
  double w=0.0;
  double phi[8]={0};
  double gradRef[8][3]={{0}};
  double hessRef[8][3][3]={{{0}}};
};

struct SupgCellPlan {
  PetscInt cell=-1;
  PetscInt entity[8]={0,0,0,0,0,0,0,0};
  PetscInt gid[8]={-1,-1,-1,-1,-1,-1,-1,-1};
  PetscInt localIndex[8]={-1,-1,-1,-1,-1,-1,-1,-1};
  PetscInt freeBasis[8]={0,0,0,0,0,0,0,0};
  PetscInt freeGid[8]={0,0,0,0,0,0,0,0};
  PetscInt nfree=0;
  double det=0.0;
  double h2=0.0;
  // Compact affine geometry: physical gradients of the four barycentric
  // coordinates.  P1 gradients are these directly.  BF3 gradients and
  // Laplacians are reconstructed analytically from lambda(q) at runtime.
  // This replaces the old O(nQ*8) per-cell physical-gradient/Hessian caches.
  double gradLambda[4][3]={{0}};
};

struct SupgAssemblyPlan {
  PetscInt nQ=0;
  std::vector<SupgReferencePoint> ref;
  std::vector<SupgCellPlan> cells;
  // No per-cell/per-quadrature physical derivative caches.  Affine BF3
  // derivatives are reconstructed from SupgCellPlan::gradLambda and lambda(q).
  // Only allocated for MMS.  Pipe/generic-flow forcing is identically zero.
  std::vector<double> forcing;
  PetscBool forcingZero=PETSC_TRUE;
  double nu=0.0;
};

static PetscErrorCode buildSupgAssemblyPlan(
    const Mesh& M,const Discrete& D,int rank,const GhostPlan& G,const ProblemConfig& problem,
    PetscInt nQ,SupgAssemblyPlan& P) {
  PetscFunctionBeginUser;
  const auto Q=supgQuadrature(nQ);
  P.nQ=(PetscInt)Q.size(); P.nu=problem.nu;
  P.ref.resize(Q.size());
  double wsum=0.0,wabs=0.0,wmin=1.0e300,wmax=-1.0e300;
  for(std::size_t q=0;q<Q.size();++q) {
    P.ref[q].lam=Q[q].lam; P.ref[q].w=Q[q].w;
    basis(Q[q].lam,P.ref[q].phi,P.ref[q].gradRef);
    referenceHessian(Q[q].lam,P.ref[q].hessRef);
    wsum+=Q[q].w; wabs+=std::abs(Q[q].w); wmin=std::min(wmin,Q[q].w); wmax=std::max(wmax,Q[q].w);
  }
  P.cells.clear(); P.cells.reserve(D.cellCount[rank]);
  const PetscInt nv=(PetscInt)M.points.size();
  for(PetscInt c=0;c<(PetscInt)M.tets.size();++c) if(D.cellOwner[c]==rank) {
    SupgCellPlan cp; cp.cell=c;
    const auto t=M.tets[c];
    const Vec3 X[4]={M.points[t[0]],M.points[t[1]],M.points[t[2]],M.points[t[3]]};
    double J[3][3]={{X[1].x-X[0].x,X[2].x-X[0].x,X[3].x-X[0].x},
                    {X[1].y-X[0].y,X[2].y-X[0].y,X[3].y-X[0].y},
                    {X[1].z-X[0].z,X[2].z-X[0].z,X[3].z-X[0].z}},invJ[3][3];
    cp.det=det3(J);
    if(cp.det<=0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"non-positive tet orientation in SUPG plan at cell %" PetscInt_FMT,c);
    inv3(J,invJ);
    // Reference barycentric gradients are (-1,-1,-1), (1,0,0), (0,1,0),
    // (0,0,1).  Transform them once per affine cell and retain only these
    // 12 doubles; all BF3 q-point derivatives are reconstructed from them.
    const double gradLambdaRef[4][3]={{-1.0,-1.0,-1.0},{1.0,0.0,0.0},{0.0,1.0,0.0},{0.0,0.0,1.0}};
    for(int i=0;i<4;++i) for(int d=0;d<3;++d) {
      double g=0.0;
      for(int j=0;j<3;++j) g+=gradLambdaRef[i][j]*invJ[j][d];
      cp.gradLambda[i][d]=g;
    }
    const double h=tetDiameter(X); cp.h2=h*h;
    if(!(cp.h2>0.0)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"zero tetrahedron diameter in SUPG plan");
    for(int i=0;i<4;++i) cp.entity[i]=t[i];
    for(int i=0;i<4;++i) cp.entity[4+i]=nv+M.oppFace[c][i];
    for(int a=0;a<8;++a) {
      cp.gid[a]=D.g2free[cp.entity[a]];
      if(cp.gid[a]>=0) {
        cp.localIndex[a]=velocityLocalIndex(G,cp.gid[a]);
        cp.freeBasis[cp.nfree]=a; cp.freeGid[cp.nfree]=cp.gid[a]; cp.nfree++;
      }
    }
    P.cells.push_back(cp);
  }

  const std::size_t nqtot=P.cells.size()*Q.size();
  P.forcingZero=(problem.mode==ProblemMode::MMS)?PETSC_FALSE:PETSC_TRUE;
  if(!P.forcingZero) {
    P.forcing.assign(nqtot*3,0.0);
    for(std::size_t ic=0;ic<P.cells.size();++ic) {
      const auto& cp=P.cells[ic];
      const auto t=M.tets[cp.cell];
      const Vec3 X[4]={M.points[t[0]],M.points[t[1]],M.points[t[2]],M.points[t[3]]};
      for(std::size_t q=0;q<Q.size();++q) {
        const std::size_t iq=ic*Q.size()+q;
        double x=0.0,y=0.0,z=0.0;
        for(int i=0;i<4;++i) { x+=P.ref[q].lam[i]*X[i].x; y+=P.ref[q].lam[i]*X[i].y; z+=P.ref[q].lam[i]*X[i].z; }
        double f[3]; problemForcing(problem,x,y,z,f);
        for(int d=0;d<3;++d) P.forcing[iq*3+d]=f[d];
      }
    }
  } else P.forcing.clear();

  const double compactGeomMiB=(double)(P.cells.size()*12*sizeof(double))/(1024.0*1024.0);
  const double forceMiB=(double)(P.forcing.size()*sizeof(double))/(1024.0*1024.0);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_SUPG_PLAN quadraturePoints=%" PetscInt_FMT " ownedCells=%zu affinePhysicalGradCached=0 viscousStrongCached=0 compactGradLambdaCached=1 compactGeomMiB=%.3f p1LaplacianSkipped=1 forcingCached=%d forcingMiB=%.3f weightSum=%.16e weightAbsSum=%.16e weightRange=[%.6e,%.6e]\n",
    P.nQ,P.cells.size(),compactGeomMiB,P.forcingZero?0:1,forceMiB,wsum,wabs,wmin,wmax));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode gatherSupgCellCoefficients(
    const Discrete& D,const GhostPlan& G,const SupgCellPlan& cp,const PetscScalar* ua[3],double coeff[3][8]) {
  PetscFunctionBeginUser;
  for(int m=0;m<8;++m) {
    if(cp.gid[m]>=0) {
      const PetscInt li=cp.localIndex[m];
      for(int d=0;d<3;++d) coeff[d][m]=PetscRealPart(ua[d][li]);
    } else {
      for(int d=0;d<3;++d) coeff[d][m]=entityDirValue(D,d,cp.entity[m]);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode assembleSupgFastMPI(
    const Discrete& D,const GhostPlan& G,const SupgAssemblyPlan& P,Vec U[3],
    double tauScale,double supgMagic,const std::string& form,Mat Sg,Vec supgRhs[3],SupgStats& stats) {
  PetscFunctionBeginUser;
  const bool implicit=(form=="implicit");
  if(!implicit && form!="explicit") SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"SUPG form must be implicit or explicit");
  if(implicit) PetscCall(MatZeroEntries(Sg));
  for(int d=0;d<3;++d) PetscCall(VecSet(supgRhs[d],0.0));

  Vec Ul[3]={nullptr,nullptr,nullptr};
  const PetscScalar* ua[3]={nullptr,nullptr,nullptr};
  for(int d=0;d<3;++d) {
    PetscCall(VecGhostUpdateBegin(U[d],INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecGhostUpdateEnd(U[d],INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecGhostGetLocalForm(U[d],&Ul[d]));
    PetscCall(VecGetArrayRead(Ul[d],&ua[d]));
  }

  double localTauMin=1.0e300,localTauMax=0.0,localTauWeighted=0.0,localWeight=0.0;
  for(std::size_t ic=0;ic<P.cells.size();++ic) {
    const auto& cp=P.cells[ic];
    double coeff[3][8]={{0}};
    PetscCall(gatherSupgCellCoefficients(D,G,cp,ua,coeff));
    double Sl[8][8]={{0}};
    double Vr[3][8]={{0}};
    double gradLambdaDot[4][4]={{0}};
    for(int i=0;i<4;++i) for(int j=i;j<4;++j) {
      double v=0.0; for(int d=0;d<3;++d) v+=cp.gradLambda[i][d]*cp.gradLambda[j][d];
      gradLambdaDot[i][j]=gradLambdaDot[j][i]=v;
    }

    for(PetscInt q=0;q<P.nQ;++q) {
      const std::size_t iq=ic*(std::size_t)P.nQ+(std::size_t)q;
      const auto& rq=P.ref[(std::size_t)q];
      double adv[3]={0.0,0.0,0.0};
      for(int d=0;d<3;++d) for(int m=0;m<8;++m) adv[d]+=coeff[d][m]*rq.phi[m];
      const double speed2=adv[0]*adv[0]+adv[1]*adv[1]+adv[2]*adv[2];
      // tau is deliberately lagged: adv comes from the previous SIMPLE state U
      // supplied to this assembly call.  No derivative of tau enters either form.
      double stream[8]={0},strongTrial[8]={0};
      // Compact affine reconstruction.  P1 gradients are the physical
      // barycentric gradients.  For b_i = 27*prod_{j!=i} lambda_j,
      //   grad b_i = 27*sum_{j!=i}(prod_{m!=i,j}lambda_m) grad lambda_j,
      //   Delta b_i = 54*sum_pair lambda_remaining*(grad lambda_j.grad lambda_k).
      // Pairwise barycentric-gradient dots are cell constants and are formed
      // once per cell outside the q loop below.
      double gradBasis[8][3]={{0}};
      for(int a=0;a<4;++a) for(int d=0;d<3;++d) gradBasis[a][d]=cp.gradLambda[a][d];
      for(int i=0;i<4;++i) {
        int js[3],kk=0; for(int j=0;j<4;++j) if(j!=i) js[kk++]=j;
        for(int d=0;d<3;++d) {
          gradBasis[4+i][d]=27.0*(
              rq.lam[js[1]]*rq.lam[js[2]]*cp.gradLambda[js[0]][d]
             +rq.lam[js[0]]*rq.lam[js[2]]*cp.gradLambda[js[1]][d]
             +rq.lam[js[0]]*rq.lam[js[1]]*cp.gradLambda[js[2]][d]);
        }
      }
      for(int a=0;a<8;++a)
        stream[a]=adv[0]*gradBasis[a][0]+adv[1]*gradBasis[a][1]+adv[2]*gradBasis[a][2];
      for(int a=0;a<4;++a) strongTrial[a]=stream[a]; // Delta(P1)=0 exactly.
      for(int i=0;i<4;++i) {
        int js[3],kk=0; for(int j=0;j<4;++j) if(j!=i) js[kk++]=j;
        const double lap=54.0*(
            rq.lam[js[2]]*gradLambdaDot[js[0]][js[1]]
           +rq.lam[js[1]]*gradLambdaDot[js[0]][js[2]]
           +rq.lam[js[0]]*gradLambdaDot[js[1]][js[2]]);
        strongTrial[4+i]=-P.nu*lap+stream[4+i];
      }
      const double diff=4.0*P.nu/cp.h2;
      const double denominator=std::max(4.0*speed2/cp.h2 + supgMagic*diff*diff,1.0e-30);
      const double tau=tauScale/std::sqrt(denominator);
      const double w=tau*rq.w*cp.det;
      const double volumeW=rq.w*cp.det;
      localTauMin=std::min(localTauMin,tau); localTauMax=std::max(localTauMax,tau);
      localTauWeighted+=tau*volumeW; localWeight+=volumeW;

      if(implicit) {
        // Rank-one outer product at each quadrature point:
        //   tau*w * stream_test[:] \otimes strongTrial[:].
        for(int a=0;a<8;++a) {
          const double ta=w*stream[a];
          for(int b=0;b<8;++b) Sl[a][b]+=ta*strongTrial[b];
          if(!P.forcingZero) for(int d=0;d<3;++d) Vr[d][a]+=ta*P.forcing[iq*3+d];
        }
      } else {
        // Fully explicit SUPG residual from the previous SIMPLE iterate:
        // RHS += -(S(Ulag)*Ulag - F(Ulag)).  This avoids a global SUPG matrix.
        double strongResidual[3]={0.0,0.0,0.0};
        for(int d=0;d<3;++d) {
          for(int b=0;b<8;++b) strongResidual[d]+=coeff[d][b]*strongTrial[b];
          if(!P.forcingZero) strongResidual[d]-=P.forcing[iq*3+d];
        }
        for(int a=0;a<8;++a) {
          const double ta=-w*stream[a];
          for(int d=0;d<3;++d) Vr[d][a]+=ta*strongResidual[d];
        }
      }
    }

    if(cp.nfree>0) {
      if(implicit) {
        PetscScalar vals[64];
        for(PetscInt ii=0;ii<cp.nfree;++ii) {
          const int a=(int)cp.freeBasis[ii];
          for(PetscInt jj=0;jj<cp.nfree;++jj) {
            const int b=(int)cp.freeBasis[jj];
            vals[ii*cp.nfree+jj]=(PetscScalar)Sl[a][b];
          }
        }
        PetscCall(MatSetValues(Sg,cp.nfree,cp.freeGid,cp.nfree,cp.freeGid,vals,ADD_VALUES));
        for(int d=0;d<3;++d) {
          PetscScalar rv[8]; PetscBool any=PETSC_FALSE;
          for(PetscInt ii=0;ii<cp.nfree;++ii) {
            const int a=(int)cp.freeBasis[ii]; double v=Vr[d][a];
            for(int b=0;b<8;++b) if(cp.gid[b]<0) v-=Sl[a][b]*entityDirValue(D,d,cp.entity[b]);
            rv[ii]=(PetscScalar)v; if(v!=0.0) any=PETSC_TRUE;
          }
          if(any) PetscCall(VecSetValues(supgRhs[d],cp.nfree,cp.freeGid,rv,ADD_VALUES));
        }
      } else {
        for(int d=0;d<3;++d) {
          PetscScalar rv[8]; PetscBool any=PETSC_FALSE;
          for(PetscInt ii=0;ii<cp.nfree;++ii) {
            const int a=(int)cp.freeBasis[ii]; rv[ii]=(PetscScalar)Vr[d][a]; if(Vr[d][a]!=0.0) any=PETSC_TRUE;
          }
          if(any) PetscCall(VecSetValues(supgRhs[d],cp.nfree,cp.freeGid,rv,ADD_VALUES));
        }
      }
    }
  }

  for(int d=0;d<3;++d) {
    PetscCall(VecRestoreArrayRead(Ul[d],&ua[d]));
    PetscCall(VecGhostRestoreLocalForm(U[d],&Ul[d]));
  }
  if(implicit) { PetscCall(MatAssemblyBegin(Sg,MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(Sg,MAT_FINAL_ASSEMBLY)); }
  for(int d=0;d<3;++d) { PetscCall(VecAssemblyBegin(supgRhs[d])); PetscCall(VecAssemblyEnd(supgRhs[d])); }

  double globalMin=0,globalMax=0,globalTauWeighted=0,globalWeight=0;
  PetscCallMPI(MPI_Allreduce(&localTauMin,&globalMin,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localTauMax,&globalMax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localTauWeighted,&globalTauWeighted,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localWeight,&globalWeight,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  stats.tauMin=globalMin; stats.tauMax=globalMax; stats.tauMean=(globalWeight!=0.0)?globalTauWeighted/globalWeight:0.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}


// -----------------------------------------------------------------------------
// M2B direct owned-row dynamic momentum assembly
// -----------------------------------------------------------------------------
struct CustomDynamicCellPlan {
  PetscInt cell=-1;
  PetscInt entity[8]={0,0,0,0,0,0,0,0};
  PetscInt gid[8]={-1,-1,-1,-1,-1,-1,-1,-1};
  PetscInt localIndex[8]={-1,-1,-1,-1,-1,-1,-1,-1};
  PetscInt ownedBasis[8]={0,0,0,0,0,0,0,0};
  PetscInt nOwnedRows=0;
  // Per-cell row-local CSR slots: 255 means fixed/inactive.  A P1+BF3 cell has
  // at most eight trial columns, so one byte is sufficient and removes all
  // sparse-column searches from the nonlinear assembly hot path.
  std::uint8_t rowSlot[8][8]={{0}};
  double det=0.0,invJ[3][3]={{0}},gradLambda[4][3]={{0}},h2=0.0;
};

struct CustomDynamicAssemblyPlan {
  std::vector<CustomDynamicCellPlan> cells;
  PetscInt nQ=0;
  std::vector<SupgReferencePoint> ref;
  std::vector<double> forcing;
  PetscBool forcingZero=PETSC_TRUE;
  double nu=0.0;
};

static PetscErrorCode buildCustomDynamicAssemblyPlan(const Mesh& M,const Discrete& D,const ProblemConfig& problem,
  const CustomMomentumCSR& A,PetscInt nQ,CustomDynamicAssemblyPlan& P) {
  PetscFunctionBeginUser;
  const PetscInt nv=(PetscInt)M.points.size();
  const auto Q=supgQuadrature(nQ);
  P.nQ=(PetscInt)Q.size(); P.nu=problem.nu; P.ref.resize(Q.size());
  double wsum=0.0,wabs=0.0,wmin=1e300,wmax=-1e300;
  for(std::size_t q=0;q<Q.size();++q) {
    P.ref[q].lam=Q[q].lam; P.ref[q].w=Q[q].w;
    basis(Q[q].lam,P.ref[q].phi,P.ref[q].gradRef); referenceHessian(Q[q].lam,P.ref[q].hessRef);
    wsum+=Q[q].w; wabs+=std::abs(Q[q].w); wmin=std::min(wmin,Q[q].w); wmax=std::max(wmax,Q[q].w);
  }
  P.cells.clear();
  // Row-support halo: include a tet on this rank iff at least one of its free
  // scalar velocity rows is owned here. Each global row is therefore assembled
  // exactly once, while partition-boundary cells may be evaluated by two ranks.
  for(PetscInt c=0;c<(PetscInt)M.tets.size();++c) {
    CustomDynamicCellPlan cp; cp.cell=c;
    for(int a=0;a<8;++a) for(int b=0;b<8;++b) cp.rowSlot[a][b]=255;
    const auto t=M.tets[(std::size_t)c];
    for(int i=0;i<4;++i) cp.entity[i]=t[i];
    for(int i=0;i<4;++i) cp.entity[4+i]=nv+M.oppFace[(std::size_t)c][i];
    for(int a=0;a<8;++a) {
      cp.gid[a]=D.g2free[(std::size_t)cp.entity[a]];
      if(cp.gid[a]>=A.rstart && cp.gid[a]<A.rend) cp.ownedBasis[cp.nOwnedRows++]=a;
    }
    if(cp.nOwnedRows==0) continue;
    for(int a=0;a<8;++a) if(cp.gid[a]>=0) {
      cp.localIndex[a]=customMomentumLocalIndex(A,cp.gid[a]);
      if(cp.localIndex[a]<0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M3A dynamic plan free gid absent from custom halo");
    }
    for(PetscInt ii=0;ii<cp.nOwnedRows;++ii) {
      const int a=(int)cp.ownedBasis[ii]; const PetscInt lr=cp.gid[a]-A.rstart;
      const PetscInt rb=A.rowPtr[(std::size_t)lr],re=A.rowPtr[(std::size_t)lr+1];
      if(re-rb>=255) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M3A one-byte cell CSR slot overflow");
      for(int b=0;b<8;++b) if(cp.gid[b]>=0) {
        auto it=std::lower_bound(A.colGid.begin()+rb,A.colGid.begin()+re,cp.gid[b]);
        if(it==A.colGid.begin()+re || *it!=cp.gid[b]) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M3A dynamic plan missing CSR column");
        cp.rowSlot[a][b]=(std::uint8_t)(it-(A.colGid.begin()+rb));
      }
    }
    const Vec3 X[4]={M.points[t[0]],M.points[t[1]],M.points[t[2]],M.points[t[3]]};
    double J[3][3]={{X[1].x-X[0].x,X[2].x-X[0].x,X[3].x-X[0].x},
                    {X[1].y-X[0].y,X[2].y-X[0].y,X[3].y-X[0].y},
                    {X[1].z-X[0].z,X[2].z-X[0].z,X[3].z-X[0].z}};
    cp.det=det3(J); if(cp.det<=0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"non-positive tet in M3A dynamic plan");
    inv3(J,cp.invJ);
    const double gradLambdaRef[4][3]={{-1,-1,-1},{1,0,0},{0,1,0},{0,0,1}};
    for(int i=0;i<4;++i) for(int d=0;d<3;++d) for(int j=0;j<3;++j) cp.gradLambda[i][d]+=gradLambdaRef[i][j]*cp.invJ[j][d];
    const double h=tetDiameter(X); cp.h2=h*h; if(!(cp.h2>0)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"zero tet diameter in M2B plan");
    P.cells.push_back(cp);
  }
  P.forcingZero=(problem.mode==ProblemMode::MMS)?PETSC_FALSE:PETSC_TRUE;
  if(!P.forcingZero) {
    P.forcing.assign(P.cells.size()*Q.size()*3,0.0);
    for(std::size_t ic=0;ic<P.cells.size();++ic) {
      const auto t=M.tets[(std::size_t)P.cells[ic].cell];
      const Vec3 X[4]={M.points[t[0]],M.points[t[1]],M.points[t[2]],M.points[t[3]]};
      for(std::size_t q=0;q<Q.size();++q) {
        double x=0,y=0,z=0; for(int i=0;i<4;++i){x+=P.ref[q].lam[i]*X[i].x;y+=P.ref[q].lam[i]*X[i].y;z+=P.ref[q].lam[i]*X[i].z;}
        double f[3]; problemForcing(problem,x,y,z,f); const std::size_t iq=(ic*Q.size()+q)*3;
        for(int d=0;d<3;++d) P.forcing[iq+d]=f[d];
      }
    }
  } else P.forcing.clear();
  unsigned long long lc=(unsigned long long)P.cells.size(),gc=0,lrows=0,grows=0;
  for(const auto& cp:P.cells) lrows+=(unsigned long long)cp.nOwnedRows;
  PetscCallMPI(MPI_Allreduce(&lc,&gc,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&lrows,&grows,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_M3A_DYNAMIC_PLAN summedRowSupportCells=%llu globalCells=%zu supportReplication=%.6f ownedRowIncidences=%llu quadraturePoints=%" PetscInt_FMT " matrixAssembly=direct_custom_owned_rows no_offrank_insertion=1 weightSum=%.16e weightAbsSum=%.16e weightRange=[%.6e,%.6e]\n",
    gc,M.tets.size(),M.tets.empty()?0.0:(double)gc/(double)M.tets.size(),grows,P.nQ,wsum,wabs,wmin,wmax));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode assembleStaticDiffusionCustom(const CustomDynamicAssemblyPlan& P,CustomMomentumCSR& A) {
  PetscFunctionBeginUser;
  std::fill(A.kNu.begin(),A.kNu.end(),0.0);
  // The old FE startup used the collapsed 5^3 rule for non-MMS flow.  Degree 4
  // is sufficient for all P1+BF3 diffusion products on affine tetrahedra, so
  // this is an exact one-time construction of the same scalar K topology.
  const auto Q=tetDuffy5();
  for(const auto& cp:P.cells) {
    double Kl[8][8]={{0}};
    for(const auto& q:Q) {
      double val[8],grr[8][3],gr[8][3]; basis(q.lam,val,grr);
      for(int a=0;a<8;++a) for(int d=0;d<3;++d) {
        gr[a][d]=0.0;
        for(int j=0;j<3;++j) gr[a][d]+=grr[a][j]*cp.invJ[j][d];
      }
      const double w=q.w*cp.det;
      for(int a=0;a<8;++a) for(int b=0;b<8;++b) {
        double dot=0.0; for(int d=0;d<3;++d) dot+=gr[a][d]*gr[b][d];
        Kl[a][b]+=dot*w;
      }
    }
    for(PetscInt ii=0;ii<cp.nOwnedRows;++ii) {
      const int a=(int)cp.ownedBasis[ii];
      const PetscInt lr=cp.gid[a]-A.rstart;
      for(int b=0;b<8;++b) if(cp.gid[b]>=0) {
        const std::uint8_t slot=cp.rowSlot[a][b];
        if(slot==255) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M3A static diffusion missing precomputed CSR slot");
        A.kNu[(std::size_t)(A.rowPtr[(std::size_t)lr]+slot)]+=Kl[a][b];
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
  double tauScale,double supgMagic,const std::string& form,SupgStats& stats) {
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
      for(int i=0;i<4;++i) { int js[3],kk=0; for(int j=0;j<4;++j) if(j!=i) js[kk++]=j; const double lap=54.0*(rq.lam[js[2]]*gradLambdaDot[js[0]][js[1]]+rq.lam[js[1]]*gradLambdaDot[js[0]][js[2]]+rq.lam[js[0]]*gradLambdaDot[js[1]][js[2]]); strongTrial[4+i]=-P.nu*lap+stream[4+i]; }
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

