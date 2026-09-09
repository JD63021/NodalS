struct CustomMomentumCSR {
  PetscInt rstart=0,rend=0,nOwned=0;
  std::vector<PetscInt> rowPtr;
  std::vector<PetscInt> colGid;
  std::vector<PetscInt> colLocal; // [0,nOwned) owned, [nOwned,...) custom ghosts
  std::vector<PetscInt> diagPos;

  std::vector<PetscInt> ghostGid; // grouped by owning rank because global IDs are rank-contiguous
  std::vector<int> reqSendCounts,reqSendDispls,reqRecvCounts,reqRecvDispls;
  std::vector<PetscInt> reqRecvGid,reqRecvLocalOffset;
  std::vector<double> exchangeSend,ghostValues;
  std::vector<MPI_Request> exchangeRequests;

  // M1+M2 memory-hygiene storage: only one active numeric CSR array is retained.
  // kNu remains as an intentionally-empty compatibility field for legacy shadow
  // diagnostics; static diffusion is re-integrated directly into aRel on every
  // physical-operator rebuild.  No persistent second nnz-sized FP64 array exists.
  std::vector<double> kNu,aRel;
  // Legacy M1/M2A shadow arrays are kept as empty fields so old diagnostic helpers
  // remain source-compatible; they consume no retained numeric payload in M2B.
  std::vector<double> convection,supg,aPhys;
  std::vector<double> physDiag,relaxDelta,relaxedDiag,metric,rAU;
  // Stage-3 pipe weak wall: owned scalar velocity rows that lie on the
  // cylindrical wall.  Ux/Uy use this mask as an exact algebraic clamp;
  // Uz remains unconstrained and receives the weak wall operator.
  std::vector<char> wallOwned;

  // Direct dynamic assembly gathers the three velocity components through the
  // same peer-only custom halo used by MatVec/SGS and accumulates owned-row RHS.
  std::array<std::vector<double>,3> fieldOwned,fieldGhost,convRhs,supgRhs,mixlenRhs,wallRhs,inletRhs;

  // Reused live-solver work buffers. Keep the custom path allocation-free.
  std::vector<double> workB,workX,workY,workBeff;
};


// -----------------------------------------------------------------------------
// Cell-principal-block Jacobi rAU plan
// -----------------------------------------------------------------------------
// For -rau_mode cell_block_diag, replace scalar 1/diag(A_rel) by the average
// diagonal of the inverse of each incident tetrahedron's assembled GLOBAL
// principal block A_rel[I_K,I_K]. This is not an element-matrix inverse: every
// block entry is read from the assembled relaxed CSR, so all global
// contributions to each row/column pair are retained.
//
// Each rank stores complete CSR rows only for its velocity ownership interval.
// The existing velocity halo identifies every off-rank row touched by an owned
// row. The plan below exchanges those complete ghost-row VALUES once per SIMPLE
// operator rebuild. Row topology is reconstructed once from the replicated mesh.
struct CellBlockJacobiPlan {
  PetscBool built=PETSC_FALSE;
  // Sparse principal-entry topology retained for OWNED rows. Production M1
  // releases CustomMomentumCSR::colGid after setup; cell-block rAU therefore
  // must not depend on that released array during SIMPLE refreshes.  Each
  // retained entry stores its column GID and permanent position in A.aRel.
  std::vector<PetscInt> ownedRowPtr;
  std::vector<PetscInt> ownedRowColGid;
  std::vector<PetscInt> ownedRowValuePos;

  std::vector<PetscInt> ghostRowPtr;
  std::vector<PetscInt> ghostRowColGid;
  std::vector<double> ghostRowValues;

  // Sparse (row,column) value halo used only by cell-block rAU.  A remote
  // assembled momentum row can contain many columns that this rank never uses;
  // request exactly the principal-block entries required by local tetrahedra.
  // sendValuePos is the owned-CSR slot for every value requested BY a peer.
  // recvDest maps each value requested FROM a peer back into ghostRowValues.
  std::vector<int> sendCounts,sendDispls,recvCounts,recvDispls;
  std::vector<PetscInt> sendValuePos,recvDest;
  std::vector<double> sendValues,recvPackedValues;

  std::vector<PetscInt> supportCells;
  std::vector<PetscInt> ownedMultiplicity;
  unsigned long long exchangedValuesPerRefresh=0;
};


static int customMomentumOwnerOfGid(const std::vector<PetscInt>& off,PetscInt gid) {
  auto it=std::upper_bound(off.begin(),off.end(),gid);
  if(it==off.begin() || (it==off.end() && gid>=off.back())) return -1;
  return (int)((it-off.begin())-1);
}

static PetscErrorCode buildCustomMomentumCSR(const Mesh& M,const Discrete& D,int rank,CustomMomentumCSR& A) {
  PetscFunctionBeginUser;
  const PetscInt nv=(PetscInt)M.points.size();
  std::vector<PetscInt> off(D.velCount.size()+1,0);
  for(std::size_t r=0;r<D.velCount.size();++r) off[r+1]=off[r]+D.velCount[r];
  A.rstart=off[(std::size_t)rank]; A.rend=off[(std::size_t)rank+1]; A.nOwned=A.rend-A.rstart;
  bool hasOwnedFreeWall=false;
  if(!D.wallEntity.empty()) for(std::size_t e=0;e<D.wallEntity.size();++e) if(D.wallEntity[e]) {
    const PetscInt g=D.g2free[e]; if(g>=A.rstart && g<A.rend) { hasOwnedFreeWall=true; break; }
  }
  if(hasOwnedFreeWall) {
    A.wallOwned.assign((std::size_t)A.nOwned,0);
    for(std::size_t e=0;e<D.wallEntity.size();++e) if(D.wallEntity[e]) {
      const PetscInt g=D.g2free[e];
      if(g>=A.rstart && g<A.rend) A.wallOwned[(std::size_t)(g-A.rstart)]=1;
    }
  } else std::vector<char>().swap(A.wallOwned);

  std::vector<std::vector<PetscInt>> rows((std::size_t)A.nOwned);
  // Global mesh is still replicated in M1.  Evaluate row support from every tet
  // touching an owned row.  This exactly models the owned-row + halo assembly
  // that will remain after mesh localization.
  for(PetscInt c=0;c<(PetscInt)M.tets.size();++c) {
    PetscInt entity[8],gid[8];
    for(int i=0;i<4;++i) entity[i]=M.tets[(std::size_t)c][i];
    for(int i=0;i<4;++i) entity[4+i]=nv+M.oppFace[(std::size_t)c][i];
    for(int i=0;i<8;++i) gid[i]=D.g2free[(std::size_t)entity[i]];
    for(int a=0;a<8;++a) if(gid[a]>=A.rstart && gid[a]<A.rend) {
      auto& rr=rows[(std::size_t)(gid[a]-A.rstart)];
      for(int b=0;b<8;++b) if(gid[b]>=0) rr.push_back(gid[b]);
    }
  }

  A.rowPtr.assign((std::size_t)A.nOwned+1,0);
  for(PetscInt i=0;i<A.nOwned;++i) {
    auto& rr=rows[(std::size_t)i];
    std::sort(rr.begin(),rr.end()); rr.erase(std::unique(rr.begin(),rr.end()),rr.end());
    const PetscInt diag=A.rstart+i;
    if(!std::binary_search(rr.begin(),rr.end(),diag))
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom momentum CSR row is missing its diagonal");
    A.rowPtr[(std::size_t)i+1]=A.rowPtr[(std::size_t)i]+(PetscInt)rr.size();
  }
  A.colGid.resize((std::size_t)A.rowPtr.back());
  A.diagPos.assign((std::size_t)A.nOwned,-1);
  for(PetscInt i=0;i<A.nOwned;++i) {
    const auto& rr=rows[(std::size_t)i];
    PetscInt p=A.rowPtr[(std::size_t)i];
    for(PetscInt k=0;k<(PetscInt)rr.size();++k) {
      A.colGid[(std::size_t)(p+k)]=rr[(std::size_t)k];
      if(rr[(std::size_t)k]==A.rstart+i) A.diagPos[(std::size_t)i]=p+k;
    }
  }

  A.ghostGid.clear();
  for(PetscInt g:A.colGid) if(g<A.rstart || g>=A.rend) A.ghostGid.push_back(g);
  std::sort(A.ghostGid.begin(),A.ghostGid.end());
  A.ghostGid.erase(std::unique(A.ghostGid.begin(),A.ghostGid.end()),A.ghostGid.end());

  const int nr=(int)D.velCount.size();
  A.reqSendCounts.assign((std::size_t)nr,0); A.reqRecvCounts.assign((std::size_t)nr,0);
  for(PetscInt g:A.ghostGid) {
    const int owner=customMomentumOwnerOfGid(off,g);
    if(owner<0 || owner>=nr) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"invalid owner for custom momentum ghost gid");
    ++A.reqSendCounts[(std::size_t)owner];
  }
  A.reqSendDispls.assign((std::size_t)nr,0); A.reqRecvDispls.assign((std::size_t)nr,0);
  for(int r=1;r<nr;++r) A.reqSendDispls[(std::size_t)r]=A.reqSendDispls[(std::size_t)r-1]+A.reqSendCounts[(std::size_t)r-1];
  PetscCallMPI(MPI_Alltoall(A.reqSendCounts.data(),1,MPI_INT,A.reqRecvCounts.data(),1,MPI_INT,PETSC_COMM_WORLD));
  for(int r=1;r<nr;++r) A.reqRecvDispls[(std::size_t)r]=A.reqRecvDispls[(std::size_t)r-1]+A.reqRecvCounts[(std::size_t)r-1];
  int nRecvReq=0; for(int v:A.reqRecvCounts) nRecvReq+=v;
  A.reqRecvGid.assign((std::size_t)nRecvReq,-1);
  PetscCallMPI(MPI_Alltoallv(A.ghostGid.empty()?nullptr:A.ghostGid.data(),A.reqSendCounts.data(),A.reqSendDispls.data(),MPIU_INT,
                             A.reqRecvGid.empty()?nullptr:A.reqRecvGid.data(),A.reqRecvCounts.data(),A.reqRecvDispls.data(),MPIU_INT,PETSC_COMM_WORLD));
  A.reqRecvLocalOffset.resize(A.reqRecvGid.size());
  for(std::size_t k=0;k<A.reqRecvGid.size();++k) {
    const PetscInt g=A.reqRecvGid[k];
    if(g<A.rstart || g>=A.rend) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom momentum peer requested a non-owned gid");
    A.reqRecvLocalOffset[k]=g-A.rstart;
  }

  A.colLocal.resize(A.colGid.size());
  for(std::size_t k=0;k<A.colGid.size();++k) {
    const PetscInt g=A.colGid[k];
    if(g>=A.rstart && g<A.rend) A.colLocal[k]=g-A.rstart;
    else {
      auto it=std::lower_bound(A.ghostGid.begin(),A.ghostGid.end(),g);
      if(it==A.ghostGid.end() || *it!=g) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom momentum ghost lookup failed");
      A.colLocal[k]=A.nOwned+(PetscInt)(it-A.ghostGid.begin());
    }
  }

  const std::size_t nnz=A.colGid.size();
  A.kNu.clear(); A.kNu.shrink_to_fit(); // M2: persistent static K payload eliminated
  A.aRel.assign(nnz,0.0);
  A.convection.clear(); A.supg.clear(); A.aPhys.clear();
  A.physDiag.assign((std::size_t)A.nOwned,0.0); A.relaxDelta.assign((std::size_t)A.nOwned,0.0);
  A.relaxedDiag.assign((std::size_t)A.nOwned,0.0); A.metric.assign((std::size_t)A.nOwned,0.0);
  A.rAU.assign((std::size_t)A.nOwned,0.0);
  for(int d=0;d<3;++d) {
    A.fieldOwned[d].assign((std::size_t)A.nOwned,0.0);
    A.fieldGhost[d].assign(A.ghostGid.size(),0.0);
    A.convRhs[d].assign((std::size_t)A.nOwned,0.0);
    A.supgRhs[d].assign((std::size_t)A.nOwned,0.0);
  }
  A.workB.assign((std::size_t)A.nOwned,0.0); A.workX.assign((std::size_t)A.nOwned,0.0);
  A.workY.assign((std::size_t)A.nOwned,0.0); A.workBeff.assign((std::size_t)A.nOwned,0.0);
  A.exchangeSend.assign(A.reqRecvGid.size(),0.0); A.ghostValues.assign(A.ghostGid.size(),0.0);
  int peerOps=0; for(int r=0;r<nr;++r) { if(A.reqSendCounts[(std::size_t)r]>0) ++peerOps; if(A.reqRecvCounts[(std::size_t)r]>0) ++peerOps; }
  A.exchangeRequests.resize((std::size_t)peerOps);

  unsigned long long localNnz=(unsigned long long)nnz,globalNnz=0;
  PetscCallMPI(MPI_Allreduce(&localNnz,&globalNnz,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  PetscInt lg=(PetscInt)A.ghostGid.size(),gmin=0,gmax=0;
  PetscCallMPI(MPI_Allreduce(&lg,&gmin,1,MPIU_INT,MPI_MIN,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&lg,&gmax,1,MPIU_INT,MPI_MAX,PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_CUSTOM_MOM_CSR rows=%" PetscInt_FMT " globalNnz=%llu avgNnzPerRow=%.6f customGhostMin=%" PetscInt_FMT " customGhostMax=%" PetscInt_FMT " semantics=owned_rows_all_incident_tets_custom_MPI_exchange\n",
    D.ns,globalNnz,D.ns?((double)globalNnz/(double)D.ns):0.0,gmin,gmax));
  unsigned long long localBytes=0,globalBytes=0;
  localBytes += (unsigned long long)(A.rowPtr.capacity()+A.colGid.capacity()+A.colLocal.capacity()+A.diagPos.capacity()+A.ghostGid.capacity()+A.reqRecvGid.capacity()+A.reqRecvLocalOffset.capacity())*sizeof(PetscInt);
  localBytes += (unsigned long long)(A.reqSendCounts.capacity()+A.reqSendDispls.capacity()+A.reqRecvCounts.capacity()+A.reqRecvDispls.capacity())*sizeof(int);
  localBytes += (unsigned long long)(A.exchangeSend.capacity()+A.ghostValues.capacity()+A.aRel.capacity()+A.physDiag.capacity()+A.relaxDelta.capacity()+A.relaxedDiag.capacity()+A.metric.capacity()+A.rAU.capacity()+A.workB.capacity()+A.workX.capacity()+A.workY.capacity()+A.workBeff.capacity())*sizeof(double);
  localBytes += (unsigned long long)A.wallOwned.capacity()*sizeof(char);
  for(int d=0;d<3;++d) localBytes += (unsigned long long)(A.fieldOwned[d].capacity()+A.fieldGhost[d].capacity()+A.convRhs[d].capacity()+A.supgRhs[d].capacity()+A.mixlenRhs[d].capacity()+A.wallRhs[d].capacity()+A.inletRhs[d].capacity())*sizeof(double);
  PetscCallMPI(MPI_Allreduce(&localBytes,&globalBytes,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_CUSTOM_MOM_MEMORY summedRetainedMiB=%.3f bytesPerGlobalVelDof=%.3f note=M1M2_single_activeA_static_diffusion_reintegrated_no_persistent_kNu_no_PETSc_momentum_matrices\n",
    (double)globalBytes/(1024.0*1024.0),D.ns?((double)globalBytes/(double)D.ns):0.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}


// -----------------------------------------------------------------------------
// M4A custom FP64 pressure B/B^T shadow architecture
// -----------------------------------------------------------------------------
// The final mixed-precision design cannot leave the physical divergence/gradient
// operators in PETSc, because a single-precision PETSc build would then make B
// itself single precision.  M4A therefore builds genuine custom MPI B and B^T
// actions now, while the validated FP64 PETSc pressure path remains live.
//
// B x:       owned pressure rows, velocity owned+halo input.
// B^T p:     owned velocity rows, all incident pressure cells, pressure halo.
// S x:       B [ rAU .* (B^T x) ], entirely custom FP64 arithmetic.
//
// No global sparse insertion is used.  Runtime communication is peer-only; the
// one-time MPI_Alltoall[v] below is only for discovering which GIDs each peer
// needs, exactly as in the custom momentum halo.
