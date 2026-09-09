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
  // Retained only in the non-compact diagnostic/runtime path. PLAN-C1 already
  // carries the same x/y vertices in CustomDynamicRuntimeCellPlan.
  double xy[4][2]={{0}};
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
  // A2 memory hygiene: exact two-pass reserve for distributed row-support cells.
  // This avoids geometric vector-capacity jumps (e.g. ~2x retained capacity)
  // without changing which cells are included in the plan.
  std::size_t rowSupportCount=0;
  for(PetscInt c=0;c<(PetscInt)M.tets.size();++c) {
    const auto t=M.tets[(std::size_t)c];
    bool any=false;
    for(int i=0;i<4 && !any;++i) { const PetscInt g=D.g2free[(std::size_t)t[i]]; if(g>=A.rstart && g<A.rend) any=true; }
    for(int i=0;i<4 && !any;++i) { const PetscInt g=D.g2free[(std::size_t)(nv+M.oppFace[(std::size_t)c][i])]; if(g>=A.rstart && g<A.rend) any=true; }
    if(any) ++rowSupportCount;
  }
  P.cells.clear(); P.cells.reserve(rowSupportCount);
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
    for(int i=0;i<4;++i) { cp.xy[i][0]=X[i].x; cp.xy[i][1]=X[i].y; }
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
    gc,(std::size_t)distributedGlobalCellCount(D),distributedGlobalCellCount(D)?(double)gc/(double)distributedGlobalCellCount(D):0.0,grows,P.nQ,wsum,wabs,wmin,wmax));
  PetscFunctionReturn(PETSC_SUCCESS);
}


// -----------------------------------------------------------------------------
// PLAN-C1 compact live dynamic assembly plan.
// The full CustomDynamicCellPlan is useful during setup, but several fields are
// redundant once static RHS/topology construction is complete.  Keep only the
// data required by repeated diffusion/convection/SUPG assembly.
// ref[a] >= 0 : custom momentum local index for a free velocity entity.
// ref[a] <  0 : encoded fixed entity id as -(entity+1).
// For an owned basis, ref[a] is exactly the owned local row index.
// -----------------------------------------------------------------------------
struct CustomDynamicRuntimeCellPlan {
  PetscInt ref[8]={-1,-1,-1,-1,-1,-1,-1,-1};
  std::uint8_t ownedBasis[8]={0,0,0,0,0,0,0,0};
  std::uint8_t nOwnedRows=0;
  std::uint8_t diagnosticOwner=0;
  std::uint8_t rowSlot[8][8]={{0}};
  double det=0.0,invJ[3][3]={{0}},h2=0.0;
  // Stage-2 RANS pipe geometry.  x/y are already required by the physical
  // mixing-length model.  The compact float z coordinates and pressure/cell
  // global id are diagnostic-only metadata used by -mixlen_audit.
  double xy[4][2]={{0}};
  float z[4]={0.0f,0.0f,0.0f,0.0f};
  PetscInt diagnosticCellGid=-1;
};
struct CustomDynamicRuntimeWallFace {
  std::uint32_t cellIndex=0;
  std::uint8_t opp=0;
  std::uint8_t diagnosticOwner=0;
  double hn=0.0,area=0.0,normal[3]={0.0,0.0,0.0};
};
struct CustomDynamicRuntimePlan {
  std::vector<CustomDynamicRuntimeCellPlan> cells;
  std::vector<CustomDynamicRuntimeWallFace> wallFaces;
  std::vector<CustomDynamicRuntimeWallFace> inletFaces;
  PetscInt nQ=0;
  std::vector<SupgReferencePoint> ref;
  std::vector<double> forcing;
  PetscBool forcingZero=PETSC_TRUE;
  double nu=0.0;
};

