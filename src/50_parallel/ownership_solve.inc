static std::vector<Vec3> cellCentroids(const Mesh& M) {
  std::vector<Vec3> C(M.tets.size());
  for(size_t c=0;c<M.tets.size();++c) {
    Vec3 q{};
    for(int i=0;i<4;++i) { const auto &x=M.points[M.tets[c][i]]; q.x+=x.x; q.y+=x.y; q.z+=x.z; }
    C[c]={q.x*.25,q.y*.25,q.z*.25};
  }
  return C;
}

static std::vector<int> makeRCBCellOwners(const Mesh& M, int nranks) {
  const PetscInt nc=(PetscInt)M.tets.size();
  if(nranks<1 || nranks>nc) throw std::runtime_error("invalid MPI rank count for RCB partition");
  auto C=cellCentroids(M);
  std::vector<PetscInt> ids(nc); std::iota(ids.begin(),ids.end(),0);
  std::vector<int> own(nc,-1);
  std::function<void(PetscInt,PetscInt,int,int)> rec;
  rec=[&](PetscInt b,PetscInt e,int r0,int r1) {
    if(r1-r0==1) { for(PetscInt k=b;k<e;++k) own[ids[k]]=r0; return; }
    double mn[3]={1e300,1e300,1e300}, mx[3]={-1e300,-1e300,-1e300};
    for(PetscInt k=b;k<e;++k){const auto&q=C[ids[k]];double a[3]={q.x,q.y,q.z};for(int d=0;d<3;++d){mn[d]=std::min(mn[d],a[d]);mx[d]=std::max(mx[d],a[d]);}}
    int axis=0; if(mx[1]-mn[1]>mx[axis]-mn[axis]) axis=1; if(mx[2]-mn[2]>mx[axis]-mn[axis]) axis=2;
    int rm=(r0+r1)/2; int nRanks=r1-r0, leftRanks=rm-r0;
    PetscInt n=e-b; PetscInt nLeft=(PetscInt)((long long)n*leftRanks/nRanks);
    nLeft=std::max<PetscInt>(1,std::min<PetscInt>(n-1,nLeft));
    auto coord=[&](PetscInt c){return axis==0?C[c].x:(axis==1?C[c].y:C[c].z);};
    std::nth_element(ids.begin()+b,ids.begin()+b+nLeft,ids.begin()+e,[&](PetscInt a,PetscInt z){double ca=coord(a),cz=coord(z);return ca<cz || (ca==cz && a<z);});
    rec(b,b+nLeft,r0,rm); rec(b+nLeft,e,rm,r1);
  };
  rec(0,nc,0,nranks);
  return own;
}

static PetscErrorCode buildOwnership(const Mesh& M, int /*rank*/, int size,const ProblemConfig& P, Discrete& D) {
  PetscFunctionBeginUser;
  const PetscInt nv=(PetscInt)M.points.size(), nf=(PetscInt)M.faces.size(), ni=(PetscInt)M.neighbour.size(), nc=(PetscInt)M.tets.size();
  prepareBoundaryData(M,P,D);
  D.cellOwner=makeRCBCellOwners(M,size);
  D.cellCount.assign(size,0); for(PetscInt c=0;c<nc;++c) D.cellCount[D.cellOwner[c]]++;
  std::vector<PetscInt> pOff(size+1,0); for(int r=0;r<size;++r) pOff[r+1]=pOff[r]+D.cellCount[r];
  D.pGid.assign(nc,-1); std::vector<PetscInt> pn=pOff;
  for(PetscInt c=0;c<nc;++c) D.pGid[c]=pn[D.cellOwner[c]]++;

  std::vector<int> entOwner(nv+nf,-1);
  for(PetscInt c=0;c<nc;++c) {
    const int r=D.cellOwner[c];
    for(int i=0;i<4;++i) {
      const PetscInt v=M.tets[c][i];
      if(!D.fixedEntity[v]) entOwner[v]=(entOwner[v]<0)?r:std::min(entOwner[v],r);
    }
  }
  for(PetscInt f=0;f<nf;++f) if(!D.fixedEntity[nv+f]) {
    if(f<ni) {
      const int r0=D.cellOwner[M.owner[f]],r1=D.cellOwner[M.neighbour[f]];
      entOwner[nv+f]=std::min(r0,r1);
    } else entOwner[nv+f]=D.cellOwner[M.owner[f]];
  }

  D.velCount.assign(size,0); D.g2free.assign(nv+nf,-1);
  D.freeVertices=D.fixedVertices=D.freeFaces=D.fixedFaces=0;
  for(PetscInt v=0;v<nv;++v) {
    if(D.fixedEntity[v]) {D.fixedVertices++;continue;}
    if(entOwner[v]<0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"free vertex has no owner");
    D.velCount[entOwner[v]]++;D.freeVertices++;
  }
  for(PetscInt f=0;f<nf;++f) {
    if(D.fixedEntity[nv+f]) {D.fixedFaces++;continue;}
    if(entOwner[nv+f]<0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"free face has no owner");
    D.velCount[entOwner[nv+f]]++;D.freeFaces++;
  }
  std::vector<PetscInt> vOff(size+1,0); for(int r=0;r<size;++r) vOff[r+1]=vOff[r]+D.velCount[r];
  D.ns=vOff[size]; std::vector<PetscInt> vn=vOff;
  for(PetscInt v=0;v<nv;++v) if(!D.fixedEntity[v]) D.g2free[v]=vn[entOwner[v]]++;
  for(PetscInt f=0;f<nf;++f) if(!D.fixedEntity[nv+f]) D.g2free[nv+f]=vn[entOwner[nv+f]]++;

  PetscInt cmin=*std::min_element(D.cellCount.begin(),D.cellCount.end()), cmax=*std::max_element(D.cellCount.begin(),D.cellCount.end());
  PetscInt vmin=*std::min_element(D.velCount.begin(),D.velCount.end()), vMax=*std::max_element(D.velCount.begin(),D.velCount.end());
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_MPI_PARTITION type=geometric_rcb ranks=%d cellMin=%" PetscInt_FMT " cellMax=%" PetscInt_FMT " velDofMin=%" PetscInt_FMT " velDofMax=%" PetscInt_FMT " sharedEntityOwner=min_adjacent_rank\n",size,cmin,cmax,vmin,vMax));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_BC_DOF freeVertices=%" PetscInt_FMT " fixedVertices=%" PetscInt_FMT " freeFacesBF3=%" PetscInt_FMT " fixedFacesBF3=%" PetscInt_FMT "\n",D.freeVertices,D.fixedVertices,D.freeFaces,D.fixedFaces));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode assembleMPI(const Mesh& M, int rank, int size,const ProblemConfig& P, Discrete& D,
  PetscBool buildDiffusionReference=PETSC_FALSE,PetscBool buildBReference=PETSC_FALSE,PetscBool buildMomentumRhsReference=PETSC_FALSE) {
  PetscFunctionBeginUser;
  const PetscInt nv=(PetscInt)M.points.size(), nc=(PetscInt)M.tets.size();
  PetscCall(buildOwnership(M,rank,size,P,D));
  PetscInt nlv=D.velCount[rank], nlp=D.cellCount[rank];
  // M3A production path does not create a PETSc scalar momentum matrix at all.
  // A temporary D.A may be requested only by the 40k reference gate so the
  // direct custom static diffusion action can be compared against the frozen
  // PETSc assembly before D.A is destroyed.
  if(buildDiffusionReference) {
    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD,nlv,nlv,D.ns,D.ns,72,nullptr,72,nullptr,&D.A));
    PetscCall(MatSetOption(D.A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));
    PetscCall(MatSetOption(D.A,MAT_SYMMETRIC,PETSC_TRUE));
  }
  // M4B: PETSc B matrices are reference-only.  Production pressure physics
  // uses the custom FP64 MPI B/B^T plan, so no sparse B storage is created.
  // The element B coefficients are still integrated below because fixedDiv is
  // a physical Dirichlet contribution and must remain exactly unchanged.
  PetscInt vStart=0,pStart=0;
  for(int r=0;r<rank;++r){vStart+=D.velCount[r];pStart+=D.cellCount[r];}
  const PetscInt vEnd=vStart+nlv;
  std::vector<PetscInt> bDnnz,bOnnz;
  PetscInt gbD=0,gbO=0;
  if(buildBReference) {
    bDnnz.assign((size_t)nlp,0); bOnnz.assign((size_t)nlp,0);
    for(PetscInt c=0;c<nc;++c) if(D.cellOwner[c]==rank) {
      const PetscInt li=D.pGid[c]-pStart;
      if(li<0 || li>=nlp) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"pressure row ownership mismatch during B reference preallocation");
      PetscInt entity[8]; for(int i=0;i<4;++i)entity[i]=M.tets[c][i]; for(int i=0;i<4;++i)entity[4+i]=nv+M.oppFace[c][i];
      for(int a=0;a<8;++a) {
        const PetscInt gid=D.g2free[entity[a]]; if(gid<0) continue;
        if(gid>=vStart && gid<vEnd) ++bDnnz[(size_t)li]; else ++bOnnz[(size_t)li];
      }
    }
    PetscInt lbD=0,lbO=0; for(PetscInt i=0;i<nlp;++i){lbD+=bDnnz[(size_t)i];lbO+=bOnnz[(size_t)i];}
    PetscCallMPI(MPI_Allreduce(&lbD,&gbD,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(&lbO,&gbO,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  }
  for(int d=0;d<3;++d) {
    if(buildBReference) {
      PetscCall(MatCreateAIJ(PETSC_COMM_WORLD,nlp,nlv,nc,D.ns,0,bDnnz.data(),0,bOnnz.data(),&D.B[d]));
      PetscCall(MatSetOption(D.B[d],MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));
    }
    if(buildMomentumRhsReference){PetscCall(VecCreateMPI(PETSC_COMM_WORLD,nlv,D.ns,&D.rhs[d])); PetscCall(VecSet(D.rhs[d],0));}
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_M4B_B_STORAGE PETScB=%s physicalB=custom_FP64_MPI referenceDiagNnz=%" PetscInt_FMT " referenceOffdiagNnz=%" PetscInt_FMT "\n",
    buildBReference?"temporary_reference":"never_created_in_production",gbD,gbO));
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD,nlp,nc,&D.volumes)); PetscCall(VecSet(D.volumes,0));
  PetscCall(VecDuplicate(D.volumes,&D.fixedDiv)); PetscCall(VecSet(D.fixedDiv,0));
  D.volumesOwnedFP64.assign((std::size_t)nlp,0.0);
  D.fixedDivOwnedFP64.assign((std::size_t)nlp,0.0);

  // On affine P1+BF3 tets, diffusion integrands have degree <=4 and B degree <=2.
  // For the pipe forcing is zero, so the degree-8 exact 5^3 collapsed rule is ample
  // and avoids the older 7^3 startup quadrature.  MMS retains the 7^3 rule below.
  auto Q=(P.mode!=ProblemMode::MMS)?tetDuffy5():tetDuffy7();
  double lmin=1e300,lmax=0,lsum=0;
  for(PetscInt c=0;c<nc;++c) if(D.cellOwner[c]==rank) {
    auto t=M.tets[c]; Vec3 X[4]={M.points[t[0]],M.points[t[1]],M.points[t[2]],M.points[t[3]]};
    double J[3][3]={{X[1].x-X[0].x,X[2].x-X[0].x,X[3].x-X[0].x},{X[1].y-X[0].y,X[2].y-X[0].y,X[3].y-X[0].y},{X[1].z-X[0].z,X[2].z-X[0].z,X[3].z-X[0].z}}, invJ[3][3];
    double det=det3(J); if(det<=0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"non-positive tet orientation at cell %" PetscInt_FMT,c); inv3(J,invJ);
    double Al[8][8]={{0}}, Bl[3][8]={{0}}, fl[3][8]={{0}};
    for(const auto& q:Q) {
      double val[8],grr[8][3],gr[8][3]; basis(q.lam,val,grr);
      for(int a=0;a<8;++a) for(int d=0;d<3;++d){gr[a][d]=0;for(int j=0;j<3;++j)gr[a][d]+=grr[a][j]*invJ[j][d];}
      double x=0,y=0,z=0; for(int i=0;i<4;++i){x+=q.lam[i]*X[i].x;y+=q.lam[i]*X[i].y;z+=q.lam[i]*X[i].z;} double ff[3]; problemForcing(P,x,y,z,ff); double w=q.w*det;
      for(int a=0;a<8;++a){for(int d=0;d<3;++d){Bl[d][a]+=gr[a][d]*w;fl[d][a]+=ff[d]*val[a]*w;} for(int b=0;b<8;++b){double dot=0;for(int d=0;d<3;++d)dot+=gr[a][d]*gr[b][d];Al[a][b]+=dot*w;}}
    }
    PetscInt lg[8]; for(int i=0;i<4;++i)lg[i]=t[i]; for(int i=0;i<4;++i)lg[4+i]=nv+M.oppFace[c][i];
    PetscInt pr=D.pGid[c]; double fixedDiv=0.0;
    for(int a=0;a<8;++a) {
      PetscInt ia=D.g2free[lg[a]];
      if(ia>=0) {
        for(int d=0;d<3;++d) {
          if(buildMomentumRhsReference) PetscCall(VecSetValue(D.rhs[d],ia,fl[d][a],ADD_VALUES));
          if(buildBReference) PetscCall(MatSetValue(D.B[d],pr,ia,Bl[d][a],ADD_VALUES));
        }
        for(int b=0;b<8;++b) {
          PetscInt ib=D.g2free[lg[b]];
          if(ib>=0) { if(buildDiffusionReference) PetscCall(MatSetValue(D.A,ia,ib,Al[a][b],ADD_VALUES)); }
          else for(int d=0;d<3;++d) {
            const double ud=entityDirValue(D,d,lg[b]);
            if(ud!=0.0 && buildMomentumRhsReference) PetscCall(VecSetValue(D.rhs[d],ia,-P.nu*Al[a][b]*ud,ADD_VALUES));
          }
        }
      } else {
        for(int d=0;d<3;++d) fixedDiv += Bl[d][a]*entityDirValue(D,d,lg[a]);
      }
    }
    const PetscInt pLocal=pr-pStart;
    if(pLocal<0 || pLocal>=nlp) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M6A owned pressure row mismatch during native FP64 scalar assembly");
    D.fixedDivOwnedFP64[(std::size_t)pLocal]+=fixedDiv;
    if(fixedDiv!=0.0) PetscCall(VecSetValue(D.fixedDiv,pr,fixedDiv,ADD_VALUES));
    double vol=det/6.0; D.volumesOwnedFP64[(std::size_t)pLocal]=vol; PetscCall(VecSetValue(D.volumes,pr,vol,INSERT_VALUES)); lmin=std::min(lmin,vol); lmax=std::max(lmax,vol); lsum+=vol;
  }
  if(buildDiffusionReference) { PetscCall(MatAssemblyBegin(D.A,MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(D.A,MAT_FINAL_ASSEMBLY)); }
  for(int d=0;d<3;++d){
    if(buildBReference){PetscCall(MatAssemblyBegin(D.B[d],MAT_FINAL_ASSEMBLY));PetscCall(MatAssemblyEnd(D.B[d],MAT_FINAL_ASSEMBLY));}
    if(buildMomentumRhsReference){PetscCall(VecAssemblyBegin(D.rhs[d]));PetscCall(VecAssemblyEnd(D.rhs[d]));}
  }
  if(buildBReference) {
    MatInfo bi; PetscCall(MatGetInfo(D.B[0],MAT_GLOBAL_SUM,&bi)); PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_M4B_B_REFERENCE_ALLOCATION used=%.0f allocated=%.0f allocOverUsed=%.12f status=%s\n",
      bi.nz_used,bi.nz_allocated,bi.nz_used?bi.nz_allocated/bi.nz_used:0.0,(bi.nz_used>0.0 && std::abs(bi.nz_allocated-bi.nz_used)<0.5)?"PASS":"CHECK"));
  }
  PetscCall(VecAssemblyBegin(D.volumes)); PetscCall(VecAssemblyEnd(D.volumes));
  PetscCall(VecAssemblyBegin(D.fixedDiv)); PetscCall(VecAssemblyEnd(D.fixedDiv));
  double gmin,gmax,gsum; PetscCallMPI(MPI_Allreduce(&lmin,&gmin,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD)); PetscCallMPI(MPI_Allreduce(&lmax,&gmax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD)); PetscCallMPI(MPI_Allreduce(&lsum,&gsum,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_FE cells=%" PetscInt_FMT " scalarVelDofs=%" PetscInt_FMT " velocityDofs=%" PetscInt_FMT " pressureDofs=%" PetscInt_FMT " freeVertices=%" PetscInt_FMT " freeBF3Faces=%" PetscInt_FMT " totalVolume=%.16e minVol=%.6e maxVol=%.6e hEff=%.12e assembly=%s\n",nc,D.ns,3*D.ns,nc,D.freeVertices,D.freeFaces,gsum,gmin,gmax,std::cbrt(gsum/nc),buildDiffusionReference?(buildBReference?"temporary_DA_plus_B_reference":"temporary_DA_reference_no_B"):(buildBReference?"temporary_B_reference_no_DA":"custom_pressure_B_no_PETSc_B_or_DA")));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode smoothSolveMPI(Mat A, Vec b, Vec x, PetscReal rtol, PetscReal relDrop, PetscInt maxIts, PetscInt checkEvery, PetscReal omega, PetscInt localSweeps, PetscInt *parallelIts, PetscReal *relres) {
  PetscFunctionBeginUser;
  Vec r; PetscReal bn=0,rn=0,rnInitial=-1.0; PetscCall(VecDuplicate(b,&r)); PetscCall(VecNorm(b,NORM_2,&bn));
  if(PetscIsInfOrNanReal(bn)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"momentum RHS norm is NaN/Inf before local SGS");
  if(bn==0) bn=1;
  // Optional OpenFOAM-like inner relative stopping: reduce the residual from
  // the current warm-start value by relDrop (e.g. 0.1).  Disabled at 0 so the
  // historical tight ||r||/||b|| criterion is unchanged by default.
  if(relDrop>0.0) {
    PetscCall(MatMult(A,x,r)); PetscCall(VecAYPX(r,-1.0,b)); PetscCall(VecNorm(r,NORM_2,&rnInitial));
    if(PetscIsInfOrNanReal(rnInitial)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"momentum initial residual is NaN/Inf");
    if(rnInitial/bn<rtol) { *parallelIts=0; *relres=rnInitial/bn; PetscCall(VecDestroy(&r)); PetscFunctionReturn(PETSC_SUCCESS); }
  }
  PetscInt it=0;
  while(it<maxIts) {
    PetscInt chunk=PetscMin(checkEvery,maxIts-it);
    PetscCall(MatSOR(A,b,omega,SOR_LOCAL_SYMMETRIC_SWEEP,0.0,chunk,localSweeps,x)); it+=chunk;
    PetscCall(MatMult(A,x,r)); PetscCall(VecAYPX(r,-1.0,b)); PetscCall(VecNorm(r,NORM_2,&rn));
    if(PetscIsInfOrNanReal(rn)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"momentum local SGS generated NaN/Inf after %" PetscInt_FMT " sweeps",it);
    if(rn/bn<rtol) break;
    if(relDrop>0.0 && rnInitial>0.0 && rn<=relDrop*rnInitial) break;
  }
  *parallelIts=it; *relres=rn/bn; PetscCall(VecDestroy(&r)); PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode gatherToZero(Vec x, Vec *x0) {
  PetscFunctionBeginUser; VecScatter sc; PetscCall(VecScatterCreateToZero(x,&sc,x0)); PetscCall(VecScatterBegin(sc,x,*x0,INSERT_VALUES,SCATTER_FORWARD)); PetscCall(VecScatterEnd(sc,x,*x0,INSERT_VALUES,SCATTER_FORWARD)); PetscCall(VecScatterDestroy(&sc)); PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customGatherOwnedVelocityToZero(const std::array<std::vector<double>,3>& U,const std::vector<PetscInt>& counts,std::array<std::vector<double>,3>& global) {
  PetscFunctionBeginUser; int rank=0,size=1;PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));std::vector<int> c((std::size_t)size),d((std::size_t)size);int off=0;for(int r=0;r<size;++r){c[(std::size_t)r]=(int)counts[(std::size_t)r];d[(std::size_t)r]=off;off+=c[(std::size_t)r];}
  for(int q=0;q<3;++q){if(rank==0)global[(std::size_t)q].resize((std::size_t)off);PetscCallMPI(MPI_Gatherv(U[(std::size_t)q].data(),(int)U[(std::size_t)q].size(),MPI_DOUBLE,rank==0?global[(std::size_t)q].data():nullptr,c.data(),d.data(),MPI_DOUBLE,0,PETSC_COMM_WORLD));}PetscFunctionReturn(PETSC_SUCCESS);
}

