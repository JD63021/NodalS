struct GhostPlan {
  PetscInt rstart=0,rend=0,nOwned=0;
  std::vector<PetscInt> ghosts;
  std::unordered_map<PetscInt,PetscInt> ghostLocal;
};

static PetscErrorCode buildVelocityGhostPlan(const Mesh& M,const Discrete& D,int rank,GhostPlan& G) {
  PetscFunctionBeginUser;
  G.rstart=0; for(int r=0;r<rank;++r) G.rstart+=D.velCount[(std::size_t)r];
  G.rend=G.rstart+D.velCount[(std::size_t)rank];
  G.nOwned=G.rend-G.rstart;
  const PetscInt nv=(PetscInt)M.points.size();
  std::set<PetscInt> need;
  for(PetscInt c=0;c<(PetscInt)M.tets.size();++c) if(D.cellOwner[c]==rank) {
    PetscInt lg[8];
    for(int i=0;i<4;++i) lg[i]=M.tets[c][i];
    for(int i=0;i<4;++i) lg[4+i]=nv+M.oppFace[c][i];
    for(int a=0;a<8;++a) {
      const PetscInt gid=D.g2free[lg[a]];
      if(gid>=0 && (gid<G.rstart || gid>=G.rend)) need.insert(gid);
    }
  }
  G.ghosts.assign(need.begin(),need.end());
  G.ghostLocal.reserve(G.ghosts.size()*2+1);
  for(PetscInt k=0;k<(PetscInt)G.ghosts.size();++k) G.ghostLocal.emplace(G.ghosts[k],G.nOwned+k);
  PetscInt localGhosts=(PetscInt)G.ghosts.size(), minGhosts=0,maxGhosts=0;
  PetscCallMPI(MPI_Allreduce(&localGhosts,&minGhosts,1,MPIU_INT,MPI_MIN,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localGhosts,&maxGhosts,1,MPIU_INT,MPI_MAX,PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_MPI_GHOSTS velocityGhostMin=%" PetscInt_FMT " velocityGhostMax=%" PetscInt_FMT " purpose=central_convection_local_element_values\n",minGhosts,maxGhosts));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscInt velocityLocalIndex(const GhostPlan& G,PetscInt gid) {
  if(gid>=G.rstart && gid<G.rend) return gid-G.rstart;
  auto it=G.ghostLocal.find(gid);
  if(it==G.ghostLocal.end()) throw std::runtime_error("velocity ghost plan missing element DOF");
  return it->second;
}

// -----------------------------------------------------------------------------
// M2B direct custom FP64 momentum live architecture
// -----------------------------------------------------------------------------
// This is deliberately independent of PETSc's MPIAIJ row storage and of the
// existing velocity ghost plan.  An owned velocity row receives FE
// contributions from every incident tetrahedron, including tetrahedra owned by
// neighbouring cell partitions.  Therefore the custom row-support halo is
// built from all cells touching an owned row, not merely from rank-owned cells.
//
// M2B removes PETSc dynamic momentum matrices entirely. Static diffusion is copied
// once from the validated FE matrix, then all convection/SUPG element contributions
// are assembled directly into owned custom CSR rows from an all-incident-tet row
// support plan.  PETSc remains only for pressure-side objects and vectors.
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

  // Direct dynamic assembly gathers the three velocity components through the
  // same peer-only custom halo used by MatVec/SGS and accumulates owned-row RHS.
  std::array<std::vector<double>,3> fieldOwned,fieldGhost,convRhs,supgRhs;

  // Reused live-solver work buffers. Keep the custom path allocation-free.
  std::vector<double> workB,workX,workY,workBeff;
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
  for(int d=0;d<3;++d) localBytes += (unsigned long long)(A.fieldOwned[d].capacity()+A.fieldGhost[d].capacity()+A.convRhs[d].capacity()+A.supgRhs[d].capacity())*sizeof(double);
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
struct CustomPeerHalo {
  PetscInt start=0,end=0,nOwned=0;
  std::vector<PetscInt> ghostGid;
  std::vector<int> reqSendCounts,reqSendDispls,reqRecvCounts,reqRecvDispls;
  std::vector<PetscInt> reqRecvGid,reqRecvLocalOffset;
  std::vector<double> exchangeSend,ghostValues;
  std::vector<MPI_Request> exchangeRequests;
};

static PetscErrorCode buildCustomPeerHalo(const std::vector<PetscInt>& counts,int rank,
  std::vector<PetscInt> neededGhosts,CustomPeerHalo& H,const char *label) {
  PetscFunctionBeginUser;
  std::vector<PetscInt> off(counts.size()+1,0);
  for(std::size_t r=0;r<counts.size();++r) off[r+1]=off[r]+counts[r];
  H.start=off[(std::size_t)rank]; H.end=off[(std::size_t)rank+1]; H.nOwned=H.end-H.start;
  std::sort(neededGhosts.begin(),neededGhosts.end());
  neededGhosts.erase(std::unique(neededGhosts.begin(),neededGhosts.end()),neededGhosts.end());
  H.ghostGid.clear(); H.ghostGid.reserve(neededGhosts.size());
  for(PetscInt g:neededGhosts) if(g<H.start || g>=H.end) H.ghostGid.push_back(g);
  const int nr=(int)counts.size();
  H.reqSendCounts.assign((std::size_t)nr,0); H.reqRecvCounts.assign((std::size_t)nr,0);
  for(PetscInt g:H.ghostGid) {
    const int owner=customMomentumOwnerOfGid(off,g);
    if(owner<0 || owner>=nr) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M4A halo ghost GID has invalid owner");
    ++H.reqSendCounts[(std::size_t)owner];
  }
  H.reqSendDispls.assign((std::size_t)nr,0); H.reqRecvDispls.assign((std::size_t)nr,0);
  for(int r=1;r<nr;++r) H.reqSendDispls[(std::size_t)r]=H.reqSendDispls[(std::size_t)r-1]+H.reqSendCounts[(std::size_t)r-1];
  PetscCallMPI(MPI_Alltoall(H.reqSendCounts.data(),1,MPI_INT,H.reqRecvCounts.data(),1,MPI_INT,PETSC_COMM_WORLD));
  for(int r=1;r<nr;++r) H.reqRecvDispls[(std::size_t)r]=H.reqRecvDispls[(std::size_t)r-1]+H.reqRecvCounts[(std::size_t)r-1];
  int nRecvReq=0; for(int v:H.reqRecvCounts) nRecvReq+=v;
  H.reqRecvGid.assign((std::size_t)nRecvReq,-1);
  PetscCallMPI(MPI_Alltoallv(H.ghostGid.empty()?nullptr:H.ghostGid.data(),H.reqSendCounts.data(),H.reqSendDispls.data(),MPIU_INT,
                             H.reqRecvGid.empty()?nullptr:H.reqRecvGid.data(),H.reqRecvCounts.data(),H.reqRecvDispls.data(),MPIU_INT,PETSC_COMM_WORLD));
  H.reqRecvLocalOffset.resize(H.reqRecvGid.size());
  for(std::size_t k=0;k<H.reqRecvGid.size();++k) {
    const PetscInt g=H.reqRecvGid[k];
    if(g<H.start || g>=H.end) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M4A peer requested a non-owned GID");
    H.reqRecvLocalOffset[k]=g-H.start;
  }
  H.exchangeSend.assign(H.reqRecvGid.size(),0.0); H.ghostValues.assign(H.ghostGid.size(),0.0);
  int peerOps=0; for(int r=0;r<nr;++r){if(H.reqSendCounts[(std::size_t)r]>0)++peerOps;if(H.reqRecvCounts[(std::size_t)r]>0)++peerOps;}
  H.exchangeRequests.resize((std::size_t)peerOps);
  PetscInt lg=(PetscInt)H.ghostGid.size(),gmin=0,gmax=0;
  PetscCallMPI(MPI_Allreduce(&lg,&gmin,1,MPIU_INT,MPI_MIN,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&lg,&gmax,1,MPIU_INT,MPI_MAX,PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_M4B_HALO kind=%s ghostMin=%" PetscInt_FMT " ghostMax=%" PetscInt_FMT " runtime=peer_only_nonblocking\n",label,gmin,gmax));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscInt customPeerLocalIndex(const CustomPeerHalo& H,PetscInt gid) {
  if(gid>=H.start && gid<H.end) return gid-H.start;
  auto it=std::lower_bound(H.ghostGid.begin(),H.ghostGid.end(),gid);
  if(it==H.ghostGid.end() || *it!=gid) return -1;
  return H.nOwned+(PetscInt)(it-H.ghostGid.begin());
}

static PetscErrorCode customPeerExchange(CustomPeerHalo& H,const std::vector<double>& xOwned,int tag) {
  PetscFunctionBeginUser;
  if((PetscInt)xOwned.size()!=H.nOwned) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"M4A halo owned vector size mismatch");
  for(std::size_t k=0;k<H.reqRecvLocalOffset.size();++k) H.exchangeSend[k]=xOwned[(std::size_t)H.reqRecvLocalOffset[k]];
  const int nr=(int)H.reqSendCounts.size(); int q=0;
  for(int r=0;r<nr;++r) if(H.reqSendCounts[(std::size_t)r]>0)
    PetscCallMPI(MPI_Irecv(H.ghostValues.data()+H.reqSendDispls[(std::size_t)r],H.reqSendCounts[(std::size_t)r],MPI_DOUBLE,r,tag,PETSC_COMM_WORLD,&H.exchangeRequests[(std::size_t)q++]));
  for(int r=0;r<nr;++r) if(H.reqRecvCounts[(std::size_t)r]>0)
    PetscCallMPI(MPI_Isend(H.exchangeSend.data()+H.reqRecvDispls[(std::size_t)r],H.reqRecvCounts[(std::size_t)r],MPI_DOUBLE,r,tag,PETSC_COMM_WORLD,&H.exchangeRequests[(std::size_t)q++]));
  if(q!=(int)H.exchangeRequests.size()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M4A halo peer request count changed");
  if(q>0) PetscCallMPI(MPI_Waitall(q,H.exchangeRequests.data(),MPI_STATUSES_IGNORE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline double customPeerValue(const CustomPeerHalo& H,const std::vector<double>& owned,PetscInt li) {
  return (li<H.nOwned)?owned[(std::size_t)li]:H.ghostValues[(std::size_t)(li-H.nOwned)];
}

static PetscErrorCode customVecOwnedRange(Vec v,PetscInt start,PetscInt end,std::vector<double>& out) {
  PetscFunctionBeginUser;
  PetscInt s=0,e=0; PetscCall(VecGetOwnershipRange(v,&s,&e));
  if(s!=start || e!=end) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"M4A PETSc Vec ownership does not match custom plan");
  const PetscScalar *a=nullptr; PetscCall(VecGetArrayRead(v,&a)); out.resize((std::size_t)(end-start));
  for(PetscInt i=0;i<end-start;++i) out[(std::size_t)i]=(double)PetscRealPart(a[i]);
  PetscCall(VecRestoreArrayRead(v,&a)); PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customVecWriteOwnedRange(Vec v,PetscInt start,PetscInt end,const std::vector<double>& in) {
  PetscFunctionBeginUser;
  PetscInt s=0,e=0; PetscCall(VecGetOwnershipRange(v,&s,&e));
  if(s!=start || e!=end || (PetscInt)in.size()!=end-start) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"M4B PETSc Vec ownership does not match custom pressure plan");
  PetscScalar *a=nullptr; PetscCall(VecGetArray(v,&a));
  for(PetscInt i=0;i<end-start;++i) a[i]=(PetscScalar)in[(std::size_t)i];
  PetscCall(VecRestoreArray(v,&a)); PetscFunctionReturn(PETSC_SUCCESS);
}

struct CustomPressureForwardCell {
  PetscInt pLocal=-1;
  PetscInt velLocal[8]={-1,-1,-1,-1,-1,-1,-1,-1};
  double vol=0.0,gradLambda[4][3]={{0}};
};
struct CustomPressureTransposeCell {
  PetscInt pLocal=-1;
  PetscInt nOwnedVel=0;
  std::uint8_t basis[8]={0};
  PetscInt velOwnedLocal[8]={0};
  double vol=0.0,gradLambda[4][3]={{0}};
};
struct CustomPressureBPlan {
  CustomPeerHalo velocityHalo,pressureHalo;
  std::vector<CustomPressureForwardCell> forwardCells;
  std::vector<CustomPressureTransposeCell> transposeCells;
  std::vector<double> velOwned,pOwned,pressureWork,velocityWork;
};

static inline double customPressureBCoeff(double vol,const double gradLambda[4][3],int d,int a) {
  const int i=(a<4)?a:(a-4);
  const double base=vol*gradLambda[i][d];
  return (a<4)?base:-(27.0/20.0)*base;
}

static void fillCustomPressureGeom(const Mesh& M,PetscInt c,double& vol,double gradLambda[4][3]) {
  const auto t=M.tets[(std::size_t)c];
  const Vec3 X[4]={M.points[t[0]],M.points[t[1]],M.points[t[2]],M.points[t[3]]};
  double J[3][3]={{X[1].x-X[0].x,X[2].x-X[0].x,X[3].x-X[0].x},
                  {X[1].y-X[0].y,X[2].y-X[0].y,X[3].y-X[0].y},
                  {X[1].z-X[0].z,X[2].z-X[0].z,X[3].z-X[0].z}},invJ[3][3];
  const double det=det3(J); if(!(det>0.0)) throw std::runtime_error("M4A non-positive tet orientation");
  inv3(J,invJ); vol=det/6.0;
  const double gr[4][3]={{-1,-1,-1},{1,0,0},{0,1,0},{0,0,1}};
  for(int i=0;i<4;++i) for(int d=0;d<3;++d) { gradLambda[i][d]=0.0; for(int j=0;j<3;++j) gradLambda[i][d]+=gr[i][j]*invJ[j][d]; }
}

static PetscErrorCode buildCustomPressureBPlan(const Mesh& M,const Discrete& D,int rank,CustomPressureBPlan& P) {
  PetscFunctionBeginUser;
  const PetscInt nv=(PetscInt)M.points.size(),nc=(PetscInt)M.tets.size();
  std::vector<PetscInt> vOff(D.velCount.size()+1,0),pOff(D.cellCount.size()+1,0);
  for(std::size_t r=0;r<D.velCount.size();++r){vOff[r+1]=vOff[r]+D.velCount[r];pOff[r+1]=pOff[r]+D.cellCount[r];}
  const PetscInt vStart=vOff[(std::size_t)rank],vEnd=vOff[(std::size_t)rank+1];
  const PetscInt pStart=pOff[(std::size_t)rank],pEnd=pOff[(std::size_t)rank+1];

  std::vector<PetscInt> neededVel,neededP;
  P.forwardCells.clear(); P.forwardCells.reserve((std::size_t)D.cellCount[(std::size_t)rank]);
  for(PetscInt c=0;c<nc;++c) if(D.cellOwner[(std::size_t)c]==rank) {
    CustomPressureForwardCell cp; cp.pLocal=D.pGid[(std::size_t)c]-pStart;
    PetscInt ent[8]; for(int i=0;i<4;++i)ent[i]=M.tets[(std::size_t)c][i]; for(int i=0;i<4;++i)ent[4+i]=nv+M.oppFace[(std::size_t)c][i];
    for(int a=0;a<8;++a) { const PetscInt g=D.g2free[(std::size_t)ent[a]]; if(g>=0 && (g<vStart || g>=vEnd)) neededVel.push_back(g); }
    fillCustomPressureGeom(M,c,cp.vol,cp.gradLambda); P.forwardCells.push_back(cp);
  }
  PetscCall(buildCustomPeerHalo(D.velCount,rank,std::move(neededVel),P.velocityHalo,"velocity_for_B"));
  // Fill velocity local indices only after the halo is finalized.
  for(PetscInt c=0;c<nc;++c) if(D.cellOwner[(std::size_t)c]==rank) {
    auto& cp=P.forwardCells[(std::size_t)(D.pGid[(std::size_t)c]-pStart)];
    PetscInt ent[8]; for(int i=0;i<4;++i)ent[i]=M.tets[(std::size_t)c][i]; for(int i=0;i<4;++i)ent[4+i]=nv+M.oppFace[(std::size_t)c][i];
    for(int a=0;a<8;++a) { const PetscInt g=D.g2free[(std::size_t)ent[a]]; cp.velLocal[a]=(g>=0)?customPeerLocalIndex(P.velocityHalo,g):-1; if(g>=0 && cp.velLocal[a]<0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M4A B forward halo missing velocity gid"); }
  }

  // A2 memory hygiene: reserve the exact transpose support count.  The old
  // push_back growth could leave nearly 2x capacity depending on mesh/rank.
  std::size_t transposeSupportCount=0;
  for(PetscInt c=0;c<nc;++c) {
    PetscInt ent[8]; for(int i=0;i<4;++i)ent[i]=M.tets[(std::size_t)c][i]; for(int i=0;i<4;++i)ent[4+i]=nv+M.oppFace[(std::size_t)c][i];
    bool any=false;
    for(int a=0;a<8;++a) { const PetscInt g=D.g2free[(std::size_t)ent[a]]; if(g>=vStart && g<vEnd) { any=true; break; } }
    if(any) ++transposeSupportCount;
  }
  P.transposeCells.clear(); P.transposeCells.reserve(transposeSupportCount);
  for(PetscInt c=0;c<nc;++c) {
    PetscInt ent[8],gid[8]; for(int i=0;i<4;++i)ent[i]=M.tets[(std::size_t)c][i]; for(int i=0;i<4;++i)ent[4+i]=nv+M.oppFace[(std::size_t)c][i];
    CustomPressureTransposeCell cp;
    for(int a=0;a<8;++a) { gid[a]=D.g2free[(std::size_t)ent[a]]; if(gid[a]>=vStart && gid[a]<vEnd) { cp.basis[cp.nOwnedVel]=(std::uint8_t)a; cp.velOwnedLocal[cp.nOwnedVel]=gid[a]-vStart; ++cp.nOwnedVel; } }
    if(cp.nOwnedVel==0) continue;
    const PetscInt pg=D.pGid[(std::size_t)c]; if(pg<pStart || pg>=pEnd) neededP.push_back(pg);
    fillCustomPressureGeom(M,c,cp.vol,cp.gradLambda); P.transposeCells.push_back(cp);
  }
  PetscCall(buildCustomPeerHalo(D.cellCount,rank,std::move(neededP),P.pressureHalo,"pressure_for_Bt"));
  std::size_t jt=0;
  for(PetscInt c=0;c<nc;++c) {
    bool any=false; PetscInt ent[8]; for(int i=0;i<4;++i)ent[i]=M.tets[(std::size_t)c][i]; for(int i=0;i<4;++i)ent[4+i]=nv+M.oppFace[(std::size_t)c][i];
    for(int a=0;a<8;++a){const PetscInt g=D.g2free[(std::size_t)ent[a]];if(g>=vStart && g<vEnd){any=true;break;}}
    if(!any) continue;
    auto& cp=P.transposeCells[jt++]; cp.pLocal=customPeerLocalIndex(P.pressureHalo,D.pGid[(std::size_t)c]);
    if(cp.pLocal<0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M4A Bt halo missing pressure gid");
  }
  if(jt!=P.transposeCells.size()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M4A Bt support fill mismatch");
  P.velOwned.assign((std::size_t)(vEnd-vStart),0.0); P.pOwned.assign((std::size_t)(pEnd-pStart),0.0);
  P.pressureWork.assign((std::size_t)(pEnd-pStart),0.0); P.velocityWork.assign((std::size_t)(vEnd-vStart),0.0);
  unsigned long long lf=(unsigned long long)P.forwardCells.size(),lt=(unsigned long long)P.transposeCells.size(),gf=0,gt=0;
  PetscCallMPI(MPI_Allreduce(&lf,&gf,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD)); PetscCallMPI(MPI_Allreduce(&lt,&gt,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  unsigned long long lb=(unsigned long long)P.forwardCells.capacity()*sizeof(CustomPressureForwardCell)+(unsigned long long)P.transposeCells.capacity()*sizeof(CustomPressureTransposeCell);
  auto haloBytes=[](const CustomPeerHalo& H)->unsigned long long{return (unsigned long long)(H.ghostGid.capacity()+H.reqRecvGid.capacity()+H.reqRecvLocalOffset.capacity())*sizeof(PetscInt)+(unsigned long long)(H.reqSendCounts.capacity()+H.reqSendDispls.capacity()+H.reqRecvCounts.capacity()+H.reqRecvDispls.capacity())*sizeof(int)+(unsigned long long)(H.exchangeSend.capacity()+H.ghostValues.capacity())*sizeof(double);};
  lb+=haloBytes(P.velocityHalo)+haloBytes(P.pressureHalo)+(unsigned long long)(P.velOwned.capacity()+P.pOwned.capacity()+P.pressureWork.capacity()+P.velocityWork.capacity())*sizeof(double);
  unsigned long long gb=0; PetscCallMPI(MPI_Allreduce(&lb,&gb,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_M4B_B_PLAN pressureRows=%llu transposeSupportCells=%llu transposeSupportReplication=%.6f retainedMiB=%.3f semantics=custom_FP64_owned_rows_peer_halo\n",gf,gt,distributedGlobalCellCount(D)?(double)gt/(double)distributedGlobalCellCount(D):0.0,(double)gb/(1024.0*1024.0)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static unsigned long long customPressurePlanLocalBytes(const CustomPressureBPlan& P) {
  auto haloBytes=[](const CustomPeerHalo& H)->unsigned long long {
    return (unsigned long long)(H.ghostGid.capacity()+H.reqRecvGid.capacity()+H.reqRecvLocalOffset.capacity())*sizeof(PetscInt)
      +(unsigned long long)(H.reqSendCounts.capacity()+H.reqSendDispls.capacity()+H.reqRecvCounts.capacity()+H.reqRecvDispls.capacity())*sizeof(int)
      +(unsigned long long)(H.exchangeSend.capacity()+H.ghostValues.capacity())*sizeof(double)
      +(unsigned long long)H.exchangeRequests.capacity()*sizeof(MPI_Request);
  };
  return (unsigned long long)P.forwardCells.capacity()*sizeof(CustomPressureForwardCell)
    +(unsigned long long)P.transposeCells.capacity()*sizeof(CustomPressureTransposeCell)
    +haloBytes(P.velocityHalo)+haloBytes(P.pressureHalo)
    +(unsigned long long)(P.velOwned.capacity()+P.pOwned.capacity()+P.pressureWork.capacity()+P.velocityWork.capacity())*sizeof(double);
}

static PetscErrorCode customPressureBApply(CustomPressureBPlan& P,int d,const std::vector<double>& xOwned,std::vector<double>& yOwned) {
  PetscFunctionBeginUser; PetscCall(customPeerExchange(P.velocityHalo,xOwned,48201+d));
  yOwned.assign((std::size_t)P.pressureHalo.nOwned,0.0);
  for(const auto& cp:P.forwardCells) { double s=0.0; for(int a=0;a<8;++a) if(cp.velLocal[a]>=0) s+=customPressureBCoeff(cp.vol,cp.gradLambda,d,a)*customPeerValue(P.velocityHalo,xOwned,cp.velLocal[a]); yOwned[(std::size_t)cp.pLocal]=s; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customPressureBtApply(CustomPressureBPlan& P,int d,const std::vector<double>& pOwned,std::vector<double>& yOwned) {
  PetscFunctionBeginUser; PetscCall(customPeerExchange(P.pressureHalo,pOwned,48211+d));
  yOwned.assign((std::size_t)P.velocityHalo.nOwned,0.0);
  for(const auto& cp:P.transposeCells) { const double pv=customPeerValue(P.pressureHalo,pOwned,cp.pLocal); for(PetscInt j=0;j<cp.nOwnedVel;++j) { const int a=(int)cp.basis[j]; yOwned[(std::size_t)cp.velOwnedLocal[j]]+=customPressureBCoeff(cp.vol,cp.gradLambda,d,a)*pv; } }
  PetscFunctionReturn(PETSC_SUCCESS);
}

struct CustomPressureParityNorm { double rel=0.0,maxAbs=0.0; };
static PetscErrorCode customPressureCompareOwned(const std::vector<double>& got,Vec ref,CustomPressureParityNorm& out) {
  PetscFunctionBeginUser; PetscInt s=0,e=0; PetscCall(VecGetOwnershipRange(ref,&s,&e)); std::vector<double> rv; PetscCall(customVecOwnedRange(ref,s,e,rv));
  if(got.size()!=rv.size()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"M4A parity vector size mismatch");
  double ld2=0.0,lr2=0.0,lmax=0.0; for(std::size_t i=0;i<got.size();++i){const double q=got[i]-rv[i];ld2+=q*q;lr2+=rv[i]*rv[i];lmax=std::max(lmax,std::abs(q));}
  double gd2=0.0,gr2=0.0,gmax=0.0; PetscCallMPI(MPI_Allreduce(&ld2,&gd2,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD)); PetscCallMPI(MPI_Allreduce(&lr2,&gr2,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD)); PetscCallMPI(MPI_Allreduce(&lmax,&gmax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  out.rel=std::sqrt(gd2)/std::max(std::sqrt(gr2),1e-300); out.maxAbs=gmax; PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customPressureBStaticParity(const Discrete& D,CustomPressureBPlan& P,double tol) {
  PetscFunctionBeginUser;
  Vec xv=nullptr,pv=nullptr,yP=nullptr,yV=nullptr; PetscCall(VecDuplicate(D.rhs[0],&xv)); PetscCall(VecDuplicate(D.volumes,&pv)); PetscCall(VecDuplicate(D.volumes,&yP)); PetscCall(VecDuplicate(D.rhs[0],&yV));
  PetscInt vs=0,ve=0,ps=0,pe=0; PetscCall(VecGetOwnershipRange(xv,&vs,&ve)); PetscCall(VecGetOwnershipRange(pv,&ps,&pe));
  { PetscScalar *a=nullptr; PetscCall(VecGetArray(xv,&a)); for(PetscInt i=0;i<ve-vs;++i){const double g=(double)(vs+i+1);a[i]=(PetscScalar)(std::sin(0.001731*g)+0.25*std::cos(0.000913*g));} PetscCall(VecRestoreArray(xv,&a)); }
  { PetscScalar *a=nullptr; PetscCall(VecGetArray(pv,&a)); for(PetscInt i=0;i<pe-ps;++i){const double g=(double)(ps+i+1);a[i]=(PetscScalar)(std::cos(0.001127*g)-0.17*std::sin(0.000719*g));} PetscCall(VecRestoreArray(pv,&a)); }
  PetscCall(customVecOwnedRange(xv,vs,ve,P.velOwned)); PetscCall(customVecOwnedRange(pv,ps,pe,P.pOwned));
  bool ok=true;
  for(int d=0;d<3;++d) {
    PetscCall(customPressureBApply(P,d,P.velOwned,P.pressureWork)); PetscCall(MatMult(D.B[d],xv,yP)); CustomPressureParityNorm bn; PetscCall(customPressureCompareOwned(P.pressureWork,yP,bn));
    PetscCall(customPressureBtApply(P,d,P.pOwned,P.velocityWork)); PetscCall(MatMultTranspose(D.B[d],pv,yV)); CustomPressureParityNorm btn; PetscCall(customPressureCompareOwned(P.velocityWork,yV,btn));
    const bool dok=(bn.rel<=tol && btn.rel<=tol); ok=ok&&dok;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_M4A_B_PARITY comp=%d BRel=%.3e BMaxAbs=%.3e BtRel=%.3e BtMaxAbs=%.3e tol=%.3e status=%s\n",d,bn.rel,bn.maxAbs,btn.rel,btn.maxAbs,tol,dok?"PASS":"CHECK"));
  }
  PetscCall(VecDestroy(&xv)); PetscCall(VecDestroy(&pv)); PetscCall(VecDestroy(&yP)); PetscCall(VecDestroy(&yV));
  if(!ok) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M4A custom B/Bt static parity failed");
  PetscFunctionReturn(PETSC_SUCCESS);
}

struct M10PressurePCGProfile {
  PetscBool enabled=PETSC_FALSE;
  unsigned long long solves=0,iterations=0,schurCalls=0,btCalls=0,bCalls=0,pcApplyCalls=0,reductionCalls=0;
  double totalPcg=0.0;
  double schurTotal=0.0,bt=0.0,rau=0.0,b=0.0,schurAccum=0.0;
  double pcApplyTotal=0.0,pcBridgeIn=0.0,pcKernel=0.0,pcBridgeOut=0.0,pcProject=0.0;
  double reductions=0.0,vectorOps=0.0;
};
static void m10ProfileAdd(M10PressurePCGProfile& a,const M10PressurePCGProfile& b) {
  a.solves+=b.solves; a.iterations+=b.iterations; a.schurCalls+=b.schurCalls; a.btCalls+=b.btCalls; a.bCalls+=b.bCalls; a.pcApplyCalls+=b.pcApplyCalls; a.reductionCalls+=b.reductionCalls;
  a.totalPcg+=b.totalPcg; a.schurTotal+=b.schurTotal; a.bt+=b.bt; a.rau+=b.rau; a.b+=b.b; a.schurAccum+=b.schurAccum;
  a.pcApplyTotal+=b.pcApplyTotal; a.pcBridgeIn+=b.pcBridgeIn; a.pcKernel+=b.pcKernel; a.pcBridgeOut+=b.pcBridgeOut; a.pcProject+=b.pcProject;
  a.reductions+=b.reductions; a.vectorOps+=b.vectorOps;
}

static PetscErrorCode customPressureSchurApply(CustomPressureBPlan& P,const CustomMomentumCSR& M,const std::vector<double>& xOwned,std::vector<double>& yOwned,M10PressurePCGProfile *prof=nullptr) {
  PetscFunctionBeginUser;
  PetscLogDouble tall0=0,tall1=0,t0=0,t1=0;
  if(prof && prof->enabled) { PetscCall(PetscTime(&tall0)); prof->schurCalls++; }
  if(prof && prof->enabled) PetscCall(PetscTime(&t0));
  yOwned.assign((std::size_t)P.pressureHalo.nOwned,0.0);
  std::vector<double> partP;
  if(prof && prof->enabled) { PetscCall(PetscTime(&t1)); prof->schurAccum += (double)(t1-t0); }
  for(int d=0;d<3;++d) {
    if(prof && prof->enabled) PetscCall(PetscTime(&t0));
    PetscCall(customPressureBtApply(P,d,xOwned,P.velocityWork));
    if(prof && prof->enabled) { PetscCall(PetscTime(&t1)); prof->bt += (double)(t1-t0); prof->btCalls++; PetscCall(PetscTime(&t0)); }
    for(std::size_t i=0;i<P.velocityWork.size();++i) P.velocityWork[i]*=M.rAU[i];
    if(prof && prof->enabled) { PetscCall(PetscTime(&t1)); prof->rau += (double)(t1-t0); PetscCall(PetscTime(&t0)); }
    PetscCall(customPressureBApply(P,d,P.velocityWork,partP));
    if(prof && prof->enabled) { PetscCall(PetscTime(&t1)); prof->b += (double)(t1-t0); prof->bCalls++; PetscCall(PetscTime(&t0)); }
    for(std::size_t i=0;i<yOwned.size();++i) yOwned[i]+=partP[i];
    if(prof && prof->enabled) { PetscCall(PetscTime(&t1)); prof->schurAccum += (double)(t1-t0); }
  }
  if(prof && prof->enabled) { PetscCall(PetscTime(&tall1)); prof->schurTotal += (double)(tall1-tall0); }
  PetscFunctionReturn(PETSC_SUCCESS);
}


// M5B: custom outer pressure PCG in FP64. PETSc owns only the already-built
// pressure preconditioner; every Krylov vector, dot product, recurrence and
// exact Schur application remains in custom C++ FP64/MPI.
struct CustomPressurePCGWorkspace {
  std::vector<double> x,r,z,p;
};
struct CustomPressurePCGResult {
  PetscInt its=0;
  double finalPreconditionedRel=0.0;
  PetscBool converged=PETSC_FALSE;
};

static PetscErrorCode customPressureDot(const std::vector<double>& a,const std::vector<double>& b,double *gdot) {
  PetscFunctionBeginUser;
  if(a.size()!=b.size()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"M5B pressure dot size mismatch");
  double l=0.0; for(std::size_t i=0;i<a.size();++i) l+=a[i]*b[i];
  PetscCallMPI(MPI_Allreduce(&l,gdot,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode customPressureNorm2(const std::vector<double>& a,double *gnorm) {
  PetscFunctionBeginUser;
  double l=0.0,g=0.0; for(double v:a) l+=v*v;
  PetscCallMPI(MPI_Allreduce(&l,&g,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  *gnorm=std::sqrt(std::max(0.0,g));
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode customPressureProjectConstant(std::vector<double>& a) {
  PetscFunctionBeginUser;
  double ls=0.0,gs=0.0; unsigned long long ln=(unsigned long long)a.size(),gn=0;
  for(double v:a) ls+=v;
  PetscCallMPI(MPI_Allreduce(&ls,&gs,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&ln,&gn,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  if(gn) { const double m=gs/(double)gn; for(double& v:a) v-=m; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// M6A: native FP64 pressure-state helpers.  These never touch PetscScalar and
// therefore remain FP64 even when the eventual PETSc backend is configured
// --with-precision=single.
static PetscErrorCode customPressureVolumeMeanShift(std::vector<double>& p,const std::vector<double>& vol,double globalVol) {
  PetscFunctionBeginUser;
  if(p.size()!=vol.size()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"M6A pressure/volume size mismatch");
  double local=0.0,global=0.0; for(std::size_t i=0;i<p.size();++i) local+=vol[i]*p[i];
  PetscCallMPI(MPI_Allreduce(&local,&global,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  const double shift=global/std::max(globalVol,1e-300); for(double& v:p) v-=shift;
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode customGatherOwnedPressureToZero(const std::vector<double>& local,const std::vector<PetscInt>& counts,std::vector<double>& global) {
  PetscFunctionBeginUser;
  int rank=0,size=1; PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank)); PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  if((int)counts.size()!=size) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"M6A pressure gather count size mismatch");
  std::vector<int> cnt((std::size_t)size),disp((std::size_t)size,0); PetscInt total=0;
  for(int r=0;r<size;++r) { if(counts[(std::size_t)r]>(PetscInt)INT_MAX) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"M6A pressure gather count exceeds int MPI limit"); cnt[(std::size_t)r]=(int)counts[(std::size_t)r]; if(r) disp[(std::size_t)r]=disp[(std::size_t)r-1]+cnt[(std::size_t)r-1]; total+=counts[(std::size_t)r]; }
  if((PetscInt)local.size()!=counts[(std::size_t)rank]) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"M6A local pressure gather size mismatch");
  if(rank==0) global.assign((std::size_t)total,0.0); else global.clear();
  PetscCallMPI(MPI_Gatherv(local.empty()?nullptr:local.data(),cnt[(std::size_t)rank],MPI_DOUBLE,rank==0?global.data():nullptr,cnt.data(),disp.data(),MPI_DOUBLE,0,PETSC_COMM_WORLD));
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode customPressurePCApply(PC pc,CustomPressureBPlan& B,Vec pcIn,Vec pcOut,
  const std::vector<double>& in,std::vector<double>& out,PetscBool projectConstant,M10PressurePCGProfile *prof=nullptr) {
  PetscFunctionBeginUser;
  PetscLogDouble tall0=0,tall1=0,t0=0,t1=0;
  if(prof && prof->enabled) { PetscCall(PetscTime(&tall0)); prof->pcApplyCalls++; PetscCall(PetscTime(&t0)); }
  PetscCall(customVecWriteOwnedRange(pcIn,B.pressureHalo.start,B.pressureHalo.end,in));
  if(prof && prof->enabled) { PetscCall(PetscTime(&t1)); prof->pcBridgeIn += (double)(t1-t0); PetscCall(PetscTime(&t0)); }
  PetscCall(PCApply(pc,pcIn,pcOut));
  if(prof && prof->enabled) { PetscCall(PetscTime(&t1)); prof->pcKernel += (double)(t1-t0); PetscCall(PetscTime(&t0)); }
  PetscCall(customVecOwnedRange(pcOut,B.pressureHalo.start,B.pressureHalo.end,out));
  if(prof && prof->enabled) { PetscCall(PetscTime(&t1)); prof->pcBridgeOut += (double)(t1-t0); }
  if(projectConstant) {
    if(prof && prof->enabled) PetscCall(PetscTime(&t0));
    PetscCall(customPressureProjectConstant(out));
    if(prof && prof->enabled) { PetscCall(PetscTime(&t1)); prof->pcProject += (double)(t1-t0); prof->reductionCalls += 2; }
  }
  if(prof && prof->enabled) { PetscCall(PetscTime(&tall1)); prof->pcApplyTotal += (double)(tall1-tall0); }
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode customPressurePCG(CustomPressureBPlan& B,const CustomMomentumCSR& M,PC pc,
  Vec pcIn,Vec pcOut,const std::vector<double>& rhs,double rtol,double atol,double dtol,PetscInt maxIts,
  PetscBool projectConstant,CustomPressurePCGWorkspace& W,CustomPressurePCGResult& R,M10PressurePCGProfile *prof=nullptr) {
  PetscFunctionBeginUser;
  const std::size_t n=rhs.size();
  PetscLogDouble t0=0,t1=0;
  if(prof && prof->enabled) PetscCall(PetscTime(&t0));
  W.x.assign(n,0.0); W.r=rhs; W.z.assign(n,0.0); W.p.assign(n,0.0);
  if(prof && prof->enabled) { PetscCall(PetscTime(&t1)); prof->vectorOps += (double)(t1-t0); }
  if(projectConstant) {
    if(prof && prof->enabled) PetscCall(PetscTime(&t0));
    PetscCall(customPressureProjectConstant(W.r));
    if(prof && prof->enabled) { PetscCall(PetscTime(&t1)); prof->reductions += (double)(t1-t0); prof->reductionCalls += 2; }
  }
  PetscCall(customPressurePCApply(pc,B,pcIn,pcOut,W.r,W.z,projectConstant,prof));
  double n0=0.0;
  if(prof && prof->enabled) PetscCall(PetscTime(&t0));
  PetscCall(customPressureNorm2(W.z,&n0));
  if(prof && prof->enabled) { PetscCall(PetscTime(&t1)); prof->reductions += (double)(t1-t0); prof->reductionCalls++; }
  const double target=std::max(atol,rtol*n0);
  if(n0<=target) { R.its=0; R.finalPreconditionedRel=0.0; R.converged=PETSC_TRUE; PetscFunctionReturn(PETSC_SUCCESS); }
  if(prof && prof->enabled) PetscCall(PetscTime(&t0));
  W.p=W.z;
  if(prof && prof->enabled) { PetscCall(PetscTime(&t1)); prof->vectorOps += (double)(t1-t0); }
  double rho=0.0;
  if(prof && prof->enabled) PetscCall(PetscTime(&t0));
  PetscCall(customPressureDot(W.r,W.z,&rho));
  if(prof && prof->enabled) { PetscCall(PetscTime(&t1)); prof->reductions += (double)(t1-t0); prof->reductionCalls++; }
  if(!(rho>0.0) || !std::isfinite(rho)) { R.its=0; R.finalPreconditionedRel=1.0; R.converged=PETSC_FALSE; PetscFunctionReturn(PETSC_SUCCESS); }
  R.converged=PETSC_FALSE; R.finalPreconditionedRel=1.0; R.its=0;
  for(PetscInt k=0;k<maxIts;++k) {
    PetscCall(customPressureSchurApply(B,M,W.p,B.pressureWork,prof));
    double pq=0.0;
    if(prof && prof->enabled) PetscCall(PetscTime(&t0));
    PetscCall(customPressureDot(W.p,B.pressureWork,&pq));
    if(prof && prof->enabled) { PetscCall(PetscTime(&t1)); prof->reductions += (double)(t1-t0); prof->reductionCalls++; }
    if(!(pq>0.0) || !std::isfinite(pq)) { R.its=k; PetscFunctionReturn(PETSC_SUCCESS); }
    const double alpha=rho/pq;
    if(prof && prof->enabled) PetscCall(PetscTime(&t0));
    for(std::size_t i=0;i<n;++i) { W.x[i]+=alpha*W.p[i]; W.r[i]-=alpha*B.pressureWork[i]; }
    if(prof && prof->enabled) { PetscCall(PetscTime(&t1)); prof->vectorOps += (double)(t1-t0); }
    if(projectConstant) {
      if(prof && prof->enabled) PetscCall(PetscTime(&t0));
      PetscCall(customPressureProjectConstant(W.r));
      if(prof && prof->enabled) { PetscCall(PetscTime(&t1)); prof->reductions += (double)(t1-t0); prof->reductionCalls += 2; }
    }
    PetscCall(customPressurePCApply(pc,B,pcIn,pcOut,W.r,W.z,projectConstant,prof));
    double zn=0.0;
    if(prof && prof->enabled) PetscCall(PetscTime(&t0));
    PetscCall(customPressureNorm2(W.z,&zn));
    if(prof && prof->enabled) { PetscCall(PetscTime(&t1)); prof->reductions += (double)(t1-t0); prof->reductionCalls++; }
    R.its=k+1; R.finalPreconditionedRel=zn/std::max(n0,1e-300);
    if(zn<=target) { R.converged=PETSC_TRUE; if(prof && prof->enabled) prof->iterations += (unsigned long long)R.its; PetscFunctionReturn(PETSC_SUCCESS); }
    if(dtol>0.0 && zn>dtol*n0) { if(prof && prof->enabled) prof->iterations += (unsigned long long)R.its; PetscFunctionReturn(PETSC_SUCCESS); }
    double rhoNew=0.0;
    if(prof && prof->enabled) PetscCall(PetscTime(&t0));
    PetscCall(customPressureDot(W.r,W.z,&rhoNew));
    if(prof && prof->enabled) { PetscCall(PetscTime(&t1)); prof->reductions += (double)(t1-t0); prof->reductionCalls++; }
    if(!(rhoNew>0.0) || !std::isfinite(rhoNew)) { if(prof && prof->enabled) prof->iterations += (unsigned long long)R.its; PetscFunctionReturn(PETSC_SUCCESS); }
    const double beta=rhoNew/rho;
    if(prof && prof->enabled) PetscCall(PetscTime(&t0));
    for(std::size_t i=0;i<n;++i) W.p[i]=W.z[i]+beta*W.p[i];
    if(prof && prof->enabled) { PetscCall(PetscTime(&t1)); prof->vectorOps += (double)(t1-t0); }
    rho=rhoNew;
  }
  if(prof && prof->enabled) prof->iterations += (unsigned long long)R.its;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode m10PrintPressurePCGProfile(const char *scope,const M10PressurePCGProfile& p,PetscInt mixedDof) {
  PetscFunctionBeginUser;
  double local[13]={p.totalPcg,p.schurTotal,p.bt,p.rau,p.b,p.schurAccum,p.pcApplyTotal,p.pcBridgeIn,p.pcKernel,p.pcBridgeOut,p.pcProject,p.reductions,p.vectorOps};
  double g[13]={0}; PetscCallMPI(MPI_Allreduce(local,g,13,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  unsigned long long lc[7]={p.solves,p.iterations,p.schurCalls,p.btCalls,p.bCalls,p.pcApplyCalls,p.reductionCalls},gc[7]={0};
  PetscCallMPI(MPI_Allreduce(lc,gc,7,MPI_UNSIGNED_LONG_LONG,MPI_MAX,PETSC_COMM_WORLD));
  const double top=g[1]+g[6]+g[11]+g[12];
  const double unaccounted=std::max(0.0,g[0]-top);
  const double itden=gc[1]?double(gc[1]):1.0, pcden=gc[5]?double(gc[5]):1.0, schden=gc[2]?double(gc[2]):1.0;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_M10_PCG_PROFILE scope=%s mixedDof=%" PetscInt_FMT " solves=%llu iterations=%llu avgIts=%.6f totalMs=%.6f totalMsPerIt=%.6f schurCalls=%llu schurMs=%.6f schurMsPerIt=%.6f btMsPerIt=%.6f rauMsPerIt=%.6f bMsPerIt=%.6f schurAccumMsPerIt=%.6f pcApplyCalls=%llu pcApplyTotalMs=%.6f pcApplyMsPerIt=%.6f pcKernelMsPerCall=%.6f bridgeInMsPerCall=%.6f bridgeOutMsPerCall=%.6f pcProjectMsPerCall=%.6f reductionCalls=%llu reductionsMsPerIt=%.6f vectorOpsMsPerIt=%.6f unaccountedMsPerIt=%.6f schurInternalClosure=%.6f pcInternalClosure=%.6f\\n",
    scope,mixedDof,gc[0],gc[1],gc[0]?double(gc[1])/double(gc[0]):0.0,1e3*g[0],1e3*g[0]/itden,gc[2],1e3*g[1],1e3*g[1]/itden,
    1e3*g[2]/itden,1e3*g[3]/itden,1e3*g[4]/itden,1e3*g[5]/itden,gc[5],1e3*g[6],1e3*g[6]/itden,1e3*g[8]/pcden,1e3*g[7]/pcden,1e3*g[9]/pcden,1e3*g[10]/pcden,gc[6],1e3*g[11]/itden,1e3*g[12]/itden,1e3*unaccounted/itden,
    g[1]>0?(g[2]+g[3]+g[4]+g[5])/g[1]:0.0,g[6]>0?(g[7]+g[8]+g[9]+g[10])/g[6]:0.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static unsigned long long customPressurePCGWorkspaceBytes(const CustomPressurePCGWorkspace& W) {
  return (unsigned long long)(W.x.capacity()+W.r.capacity()+W.z.capacity()+W.p.capacity())*sizeof(double);
}

// M4B live physical pressure MatShell. PETSc owns Krylov vectors and GAMG only;
// B/B^T/rAU arithmetic is custom C++ FP64 with peer-only MPI communication.
struct CustomFactoredPressureContext {
  CustomPressureBPlan *B=nullptr;
  const CustomMomentumCSR *M=nullptr;
  std::vector<double> xOwned,yOwned;
};

static PetscErrorCode customFactoredPressureMult(Mat A,Vec x,Vec y) {
  PetscFunctionBeginUser;
  CustomFactoredPressureContext *C=nullptr; PetscCall(MatShellGetContext(A,&C));
  if(!C || !C->B || !C->M) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M4B custom factored pressure context is incomplete");
  PetscCall(customVecOwnedRange(x,C->B->pressureHalo.start,C->B->pressureHalo.end,C->xOwned));
  PetscCall(customPressureSchurApply(*C->B,*C->M,C->xOwned,C->yOwned));
  PetscCall(customVecWriteOwnedRange(y,C->B->pressureHalo.start,C->B->pressureHalo.end,C->yOwned));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode createCustomFactoredPressure(CustomPressureBPlan& B,const CustomMomentumCSR& M,Vec pressureTemplate,
  CustomFactoredPressureContext& C,Mat *A) {
  PetscFunctionBeginUser;
  C.B=&B; C.M=&M;
  PetscInt plocal=0,pglobal=0; PetscCall(VecGetLocalSize(pressureTemplate,&plocal)); PetscCall(VecGetSize(pressureTemplate,&pglobal));
  PetscCall(MatCreateShell(PETSC_COMM_WORLD,plocal,plocal,pglobal,pglobal,&C,A));
  PetscCall(MatShellSetOperation(*A,MATOP_MULT,(void(*)(void))customFactoredPressureMult));
  PetscCall(MatSetOption(*A,MAT_SYMMETRIC,PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customPressureSchurParity(Mat factored,const Discrete& D,CustomPressureBPlan& P,const CustomMomentumCSR& M,double tol) {
  PetscFunctionBeginUser;
  Vec x=nullptr,y=nullptr; PetscCall(VecDuplicate(D.volumes,&x)); PetscCall(VecDuplicate(D.volumes,&y)); PetscInt s=0,e=0; PetscCall(VecGetOwnershipRange(x,&s,&e));
  { PetscScalar *a=nullptr; PetscCall(VecGetArray(x,&a)); for(PetscInt i=0;i<e-s;++i){const double g=(double)(s+i+1);a[i]=(PetscScalar)(std::sin(0.001013*g)+0.31*std::cos(0.000617*g));} PetscCall(VecRestoreArray(x,&a)); }
  PetscCall(customVecOwnedRange(x,s,e,P.pOwned)); PetscCall(customPressureSchurApply(P,M,P.pOwned,P.pressureWork)); PetscCall(MatMult(factored,x,y)); CustomPressureParityNorm n; PetscCall(customPressureCompareOwned(P.pressureWork,y,n));
  const bool ok=n.rel<=tol; PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_M4A_SCHUR_PARITY SRel=%.3e SMaxAbs=%.3e tol=%.3e arithmetic=custom_FP64_B_rAU_Bt reference=PETSc_PetscScalar_factored status=%s\n",n.rel,n.maxAbs,tol,ok?"PASS":"CHECK"));
  PetscCall(VecDestroy(&x)); PetscCall(VecDestroy(&y)); if(!ok) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M4A custom factored Schur parity failed"); PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customPressureLiveSchurParity(Mat explicitRef,const Discrete& D,CustomPressureBPlan& P,const CustomMomentumCSR& M,double tol) {
  PetscFunctionBeginUser;
  Vec x=nullptr,y=nullptr; PetscCall(VecDuplicate(D.volumes,&x)); PetscCall(VecDuplicate(D.volumes,&y)); PetscInt s=0,e=0; PetscCall(VecGetOwnershipRange(x,&s,&e));
  { PetscScalar *a=nullptr; PetscCall(VecGetArray(x,&a)); for(PetscInt i=0;i<e-s;++i){const double g=(double)(s+i+1);a[i]=(PetscScalar)(std::sin(0.001013*g)+0.31*std::cos(0.000617*g));} PetscCall(VecRestoreArray(x,&a)); }
  PetscCall(customVecOwnedRange(x,s,e,P.pOwned)); PetscCall(customPressureSchurApply(P,M,P.pOwned,P.pressureWork)); PetscCall(MatMult(explicitRef,x,y)); CustomPressureParityNorm n; PetscCall(customPressureCompareOwned(P.pressureWork,y,n));
  const bool ok=n.rel<=tol; PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_M4B_LIVE_SCHUR_PARITY SRel=%.3e SMaxAbs=%.3e tol=%.3e live=custom_FP64_B_rAU_Bt reference=explicit_FP64_Pmat_snapshot status=%s\n",n.rel,n.maxAbs,tol,ok?"PASS":"CHECK"));
  PetscCall(VecDestroy(&x)); PetscCall(VecDestroy(&y)); if(!ok) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M4B live custom pressure operator parity failed"); PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customMomentumLoadFromPetsc(Mat M,const CustomMomentumCSR& A,std::vector<double>& vals) {
  PetscFunctionBeginUser;
  if(A.colGid.empty() && A.rowPtr.back()>0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"M1 colGid was released after setup; PETSc matrix-value shadow loading is setup-only");
  if(vals.size()!=A.colGid.size()) vals.assign(A.colGid.size(),0.0); else std::fill(vals.begin(),vals.end(),0.0);
  for(PetscInt i=0;i<A.nOwned;++i) {
    const PetscInt row=A.rstart+i; PetscInt ncols=0; const PetscInt *cols=nullptr; const PetscScalar *mv=nullptr;
    PetscCall(MatGetRow(M,row,&ncols,&cols,&mv));
    const PetscInt b=A.rowPtr[(std::size_t)i],e=A.rowPtr[(std::size_t)i+1];
    for(PetscInt k=0;k<ncols;++k) {
      auto it=std::lower_bound(A.colGid.begin()+b,A.colGid.begin()+e,cols[k]);
      if(it==A.colGid.begin()+e || *it!=cols[k]) {
        PetscCall(MatRestoreRow(M,row,&ncols,&cols,&mv));
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"PETSc momentum matrix contains a column absent from custom owned-row topology");
      }
      vals[(std::size_t)(it-A.colGid.begin())]=(double)PetscRealPart(mv[k]);
    }
    PetscCall(MatRestoreRow(M,row,&ncols,&cols,&mv));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customMomentumExchange(CustomMomentumCSR& A,const std::vector<double>& xOwned) {
  PetscFunctionBeginUser;
  if((PetscInt)xOwned.size()!=A.nOwned) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"custom momentum owned vector size mismatch");
  for(std::size_t k=0;k<A.reqRecvLocalOffset.size();++k) A.exchangeSend[k]=xOwned[(std::size_t)A.reqRecvLocalOffset[k]];
  // Reverse the one-time GID request traffic with peer-only value messages.
  // No per-iteration all-to-all is used: receive only from ranks that own our
  // ghosts and send only to ranks that actually requested one of our values.
  int nr=0; PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&nr)); int q=0; constexpr int tag=48173;
  for(int r=0;r<nr;++r) if(A.reqSendCounts[(std::size_t)r]>0)
    PetscCallMPI(MPI_Irecv(A.ghostValues.data()+A.reqSendDispls[(std::size_t)r],A.reqSendCounts[(std::size_t)r],MPI_DOUBLE,r,tag,PETSC_COMM_WORLD,&A.exchangeRequests[(std::size_t)q++]));
  for(int r=0;r<nr;++r) if(A.reqRecvCounts[(std::size_t)r]>0)
    PetscCallMPI(MPI_Isend(A.exchangeSend.data()+A.reqRecvDispls[(std::size_t)r],A.reqRecvCounts[(std::size_t)r],MPI_DOUBLE,r,tag,PETSC_COMM_WORLD,&A.exchangeRequests[(std::size_t)q++]));
  if(q!=(int)A.exchangeRequests.size()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom momentum peer request count changed unexpectedly");
  if(q>0) PetscCallMPI(MPI_Waitall(q,A.exchangeRequests.data(),MPI_STATUSES_IGNORE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customMomentumVecOwned(Vec v,const CustomMomentumCSR& A,std::vector<double>& out);


static PetscInt customMomentumLocalIndex(const CustomMomentumCSR& A,PetscInt gid) {
  if(gid>=A.rstart && gid<A.rend) return gid-A.rstart;
  auto it=std::lower_bound(A.ghostGid.begin(),A.ghostGid.end(),gid);
  if(it==A.ghostGid.end() || *it!=gid) return -1;
  return A.nOwned+(PetscInt)(it-A.ghostGid.begin());
}

static PetscErrorCode customMomentumGatherVelocity(CustomMomentumCSR& A,Vec U[3]) {
  PetscFunctionBeginUser;
  for(int d=0;d<3;++d) {
    PetscCall(customMomentumVecOwned(U[d],A,A.fieldOwned[d]));
    PetscCall(customMomentumExchange(A,A.fieldOwned[d]));
    A.fieldGhost[d]=A.ghostValues;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customMomentumGatherVelocityNative(CustomMomentumCSR& A,const std::array<std::vector<double>,3>& U) {
  PetscFunctionBeginUser;
  for(int d=0;d<3;++d) {
    if((PetscInt)U[(std::size_t)d].size()!=A.nOwned) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"M6B native velocity size mismatch");
    A.fieldOwned[(std::size_t)d]=U[(std::size_t)d];
    PetscCall(customMomentumExchange(A,A.fieldOwned[(std::size_t)d]));
    A.fieldGhost[(std::size_t)d]=A.ghostValues;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline double customMomentumFieldValue(const CustomMomentumCSR& A,int d,PetscInt li) {
  return (li<A.nOwned)?A.fieldOwned[d][(std::size_t)li]:A.fieldGhost[d][(std::size_t)(li-A.nOwned)];
}

struct CustomDynamicAssemblyPlan;
static PetscErrorCode assembleStaticDiffusionCustom(const CustomDynamicAssemblyPlan& P,CustomMomentumCSR& A,double scale);

static PetscErrorCode customMomentumResetPhysical(const CustomDynamicAssemblyPlan& P,CustomMomentumCSR& A,double nu,PetscBool legacyReference) {
  PetscFunctionBeginUser;
  // Gate-only legacy reference retains cached nu*K. Production M2 rebuilds nu*K
  // directly into the sole active CSR values from the compact reference tensor.
  if(legacyReference) {
    if(A.kNu.size()!=A.aRel.size()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M1M2 legacy reference missing cached kNu");
    A.aRel=A.kNu;
  } else PetscCall(assembleStaticDiffusionCustom(P,A,nu));
  for(int d=0;d<3;++d) { std::fill(A.convRhs[d].begin(),A.convRhs[d].end(),0.0); std::fill(A.supgRhs[d].begin(),A.supgRhs[d].end(),0.0); }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customMomentumFinalizeRelaxation(CustomMomentumCSR& A,
  const std::string& uRelaxMode,double alphaU,const std::string& simpleVariant,const std::string& rauMode,
  double simplecBlend,double floorFraction,const std::string& fallback,double rauScale) {
  PetscFunctionBeginUser;
  const double relaxFactor=1.0/alphaU-1.0;
  for(PetscInt i=0;i<A.nOwned;++i) {
    const PetscInt dpos=A.diagPos[(std::size_t)i];
    const double d=A.aRel[(std::size_t)dpos]; A.physDiag[(std::size_t)i]=d;
    double relaxMetric=d;
    if(uRelaxMode=="row_l1") {
      relaxMetric=0.0;
      for(PetscInt k=A.rowPtr[(std::size_t)i];k<A.rowPtr[(std::size_t)i+1];++k) relaxMetric+=std::abs(A.aRel[(std::size_t)k]);
    }
    const double delta=relaxFactor*relaxMetric; A.relaxDelta[(std::size_t)i]=delta;
    A.aRel[(std::size_t)dpos]+=delta; A.relaxedDiag[(std::size_t)i]=A.aRel[(std::size_t)dpos];
  }
  for(PetscInt i=0;i<A.nOwned;++i) {
    double use=0.0;
    if(simpleVariant=="simplec") {
      double raw=0.0; for(PetscInt k=A.rowPtr[(std::size_t)i];k<A.rowPtr[(std::size_t)i+1];++k) raw+=A.aRel[(std::size_t)k];
      const double d=A.relaxedDiag[(std::size_t)i], blended=(1.0-simplecBlend)*d+simplecBlend*raw, threshold=floorFraction*d;
      use=blended;
      if(!(use>threshold) || !std::isfinite(use)) {
        if(fallback=="diag") use=d;
        else if(fallback=="floor") use=std::max(threshold,std::numeric_limits<double>::min());
        else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"custom SIMPLEC hit fallback=error");
      }
    } else if(rauMode=="diag") use=A.relaxedDiag[(std::size_t)i];
    else { use=0.0; for(PetscInt k=A.rowPtr[(std::size_t)i];k<A.rowPtr[(std::size_t)i+1];++k) use+=std::abs(A.aRel[(std::size_t)k]); }
    if(!(use>0.0) || !std::isfinite(use)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"invalid custom momentum correction metric");
    A.metric[(std::size_t)i]=use; A.rAU[(std::size_t)i]=rauScale/use;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customMomentumMatVec(CustomMomentumCSR& A,const std::vector<double>& vals,const std::vector<double>& xOwned,std::vector<double>& yOwned) {
  PetscFunctionBeginUser;
  if(vals.size()!=(std::size_t)A.rowPtr.back()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"custom momentum matrix value size mismatch");
  PetscCall(customMomentumExchange(A,xOwned));
  if((PetscInt)yOwned.size()!=A.nOwned) yOwned.resize((std::size_t)A.nOwned);
  std::fill(yOwned.begin(),yOwned.end(),0.0);
  for(PetscInt i=0;i<A.nOwned;++i) {
    double s=0.0;
    for(PetscInt k=A.rowPtr[(std::size_t)i];k<A.rowPtr[(std::size_t)i+1];++k) {
      const PetscInt li=A.colLocal[(std::size_t)k];
      const double x=(li<A.nOwned)?xOwned[(std::size_t)li]:A.ghostValues[(std::size_t)(li-A.nOwned)];
      s+=vals[(std::size_t)k]*x;
    }
    yOwned[(std::size_t)i]=s;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customMomentumDerive(CustomMomentumCSR& A,bool centralConvection,bool implicitSupg,
  const std::string& uRelaxMode,double alphaU,const std::string& simpleVariant,const std::string& rauMode,
  double simplecBlend,double floorFraction,const std::string& fallback,double rauScale) {
  PetscFunctionBeginUser;
  if(A.kNu.size()!=A.aRel.size()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"legacy custom momentum shadow derive requires persistent kNu and is disabled in M1+M2 production branch");
  const double relaxFactor=1.0/alphaU-1.0;
  for(std::size_t k=0;k<A.aPhys.size();++k)
    A.aPhys[k]=A.kNu[k]+(centralConvection?A.convection[k]:0.0)+(implicitSupg?A.supg[k]:0.0);
  A.aRel=A.aPhys;
  for(PetscInt i=0;i<A.nOwned;++i) {
    const PetscInt dpos=A.diagPos[(std::size_t)i];
    const double d=A.aPhys[(std::size_t)dpos]; A.physDiag[(std::size_t)i]=d;
    double relaxMetric=d;
    if(uRelaxMode=="row_l1") {
      relaxMetric=0.0;
      for(PetscInt k=A.rowPtr[(std::size_t)i];k<A.rowPtr[(std::size_t)i+1];++k) relaxMetric+=std::abs(A.aPhys[(std::size_t)k]);
    }
    const double delta=relaxFactor*relaxMetric; A.relaxDelta[(std::size_t)i]=delta;
    A.aRel[(std::size_t)dpos]+=delta; A.relaxedDiag[(std::size_t)i]=A.aRel[(std::size_t)dpos];
  }
  for(PetscInt i=0;i<A.nOwned;++i) {
    double use=0.0;
    if(simpleVariant=="simplec") {
      double raw=0.0; for(PetscInt k=A.rowPtr[(std::size_t)i];k<A.rowPtr[(std::size_t)i+1];++k) raw+=A.aRel[(std::size_t)k];
      const double d=A.relaxedDiag[(std::size_t)i]; const double blended=(1.0-simplecBlend)*d+simplecBlend*raw; const double threshold=floorFraction*d;
      use=blended;
      if(!(use>threshold) || !std::isfinite(use)) {
        if(fallback=="diag") use=d;
        else if(fallback=="floor") use=std::max(threshold,std::numeric_limits<double>::min());
        else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"custom SIMPLEC shadow hit fallback=error");
      }
    } else if(rauMode=="diag") use=A.relaxedDiag[(std::size_t)i];
    else {
      use=0.0; for(PetscInt k=A.rowPtr[(std::size_t)i];k<A.rowPtr[(std::size_t)i+1];++k) use+=std::abs(A.aRel[(std::size_t)k]);
    }
    if(!(use>0.0) || !std::isfinite(use)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"invalid custom momentum correction metric");
    A.metric[(std::size_t)i]=use; A.rAU[(std::size_t)i]=rauScale/use;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

struct CustomParityNorm { double rel=0.0,maxAbs=0.0; };

static PetscErrorCode customMomentumCompareArrays(const std::vector<double>& got,const std::vector<double>& ref,CustomParityNorm& out) {
  PetscFunctionBeginUser;
  if(got.size()!=ref.size()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"custom momentum comparison size mismatch");
  double ld2=0.0,lr2=0.0,lmax=0.0;
  for(std::size_t i=0;i<got.size();++i) { const double d=got[i]-ref[i]; ld2+=d*d; lr2+=ref[i]*ref[i]; lmax=std::max(lmax,std::abs(d)); }
  double gd2=0.0,gr2=0.0,gmax=0.0;
  PetscCallMPI(MPI_Allreduce(&ld2,&gd2,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&lr2,&gr2,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&lmax,&gmax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  out.rel=std::sqrt(gd2)/std::max(std::sqrt(gr2),1.0e-300); out.maxAbs=gmax;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customMomentumVecOwned(Vec v,const CustomMomentumCSR& A,std::vector<double>& out) {
  PetscFunctionBeginUser;
  PetscInt s=0,e=0; PetscCall(VecGetOwnershipRange(v,&s,&e));
  if(s!=A.rstart || e!=A.rend) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"custom momentum Vec ownership mismatch");
  const PetscScalar *va=nullptr; PetscCall(VecGetArrayRead(v,&va)); out.resize((std::size_t)A.nOwned);
  for(PetscInt i=0;i<A.nOwned;++i) out[(std::size_t)i]=(double)PetscRealPart(va[i]);
  PetscCall(VecRestoreArrayRead(v,&va)); PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customMomentumMatrixParity(Mat M,const CustomMomentumCSR& A,const std::vector<double>& got,CustomParityNorm& out) {
  PetscFunctionBeginUser;
  std::vector<double> ref; PetscCall(customMomentumLoadFromPetsc(M,A,ref));
  PetscCall(customMomentumCompareArrays(got,ref,out)); PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customMomentumVecParity(Vec v,const CustomMomentumCSR& A,const std::vector<double>& got,CustomParityNorm& out) {
  PetscFunctionBeginUser;
  std::vector<double> ref; PetscCall(customMomentumVecOwned(v,A,ref));
  PetscCall(customMomentumCompareArrays(got,ref,out)); PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customMomentumActionParity(Mat M,CustomMomentumCSR& A,const std::vector<double>& vals,CustomParityNorm& out) {
  PetscFunctionBeginUser;
  Vec x=nullptr,y=nullptr; PetscCall(MatCreateVecs(M,&x,&y));
  PetscScalar *xa=nullptr; PetscCall(VecGetArray(x,&xa)); std::vector<double> xo((std::size_t)A.nOwned,0.0);
  for(PetscInt i=0;i<A.nOwned;++i) {
    const double g=(double)(A.rstart+i+1); const double v=std::sin(0.017*g)+0.37*std::cos(0.031*g);
    xa[i]=(PetscScalar)v; xo[(std::size_t)i]=v;
  }
  PetscCall(VecRestoreArray(x,&xa)); PetscCall(MatMult(M,x,y));
  std::vector<double> yc,yr; PetscCall(customMomentumMatVec(A,vals,xo,yc)); PetscCall(customMomentumVecOwned(y,A,yr));
  PetscCall(customMomentumCompareArrays(yc,yr,out)); PetscCall(VecDestroy(&x)); PetscCall(VecDestroy(&y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customMomentumSymmetricGS(CustomMomentumCSR& A,const std::vector<double>& b,std::vector<double>& x,double omega,PetscInt localSweeps) {
  PetscFunctionBeginUser;
  if((PetscInt)b.size()!=A.nOwned || (PetscInt)x.size()!=A.nOwned) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"custom SGS vector size mismatch");
  auto& beff=A.workBeff;
  if((PetscInt)beff.size()!=A.nOwned) beff.assign((std::size_t)A.nOwned,0.0);
  // MPIAIJ local-SOR semantics: freeze the off-rank values for this outer SOR
  // application, form the effective local RHS once, then perform the requested
  // number of local symmetric sweeps on the owned-owned block.
  PetscCall(customMomentumExchange(A,x));
  for(PetscInt i=0;i<A.nOwned;++i) {
    double v=b[(std::size_t)i];
    for(PetscInt k=A.rowPtr[(std::size_t)i];k<A.rowPtr[(std::size_t)i+1];++k) {
      const PetscInt li=A.colLocal[(std::size_t)k]; if(li>=A.nOwned) v-=A.aRel[(std::size_t)k]*A.ghostValues[(std::size_t)(li-A.nOwned)];
    }
    beff[(std::size_t)i]=v;
  }
  for(PetscInt sweep=0;sweep<localSweeps;++sweep) {
    for(PetscInt i=0;i<A.nOwned;++i) {
      const double old=x[(std::size_t)i],diag=A.aRel[(std::size_t)A.diagPos[(std::size_t)i]]; double rhs=beff[(std::size_t)i];
      for(PetscInt k=A.rowPtr[(std::size_t)i];k<A.rowPtr[(std::size_t)i+1];++k) {
        const PetscInt li=A.colLocal[(std::size_t)k]; if(li<A.nOwned && li!=i) rhs-=A.aRel[(std::size_t)k]*x[(std::size_t)li];
      }
      x[(std::size_t)i]=(1.0-omega)*old+omega*(rhs/diag);
    }
    for(PetscInt ii=A.nOwned;ii>0;--ii) {
      const PetscInt i=ii-1; const double old=x[(std::size_t)i],diag=A.aRel[(std::size_t)A.diagPos[(std::size_t)i]]; double rhs=beff[(std::size_t)i];
      for(PetscInt k=A.rowPtr[(std::size_t)i];k<A.rowPtr[(std::size_t)i+1];++k) {
        const PetscInt li=A.colLocal[(std::size_t)k]; if(li<A.nOwned && li!=i) rhs-=A.aRel[(std::size_t)k]*x[(std::size_t)li];
      }
      x[(std::size_t)i]=(1.0-omega)*old+omega*(rhs/diag);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customMomentumWriteOwned(Vec v,const CustomMomentumCSR& A,const std::vector<double>& in) {
  PetscFunctionBeginUser;
  if((PetscInt)in.size()!=A.nOwned) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"custom momentum write size mismatch");
  PetscInt s=0,e=0; PetscCall(VecGetOwnershipRange(v,&s,&e));
  if(s!=A.rstart || e!=A.rend) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"custom momentum Vec ownership mismatch on write");
  PetscScalar *va=nullptr; PetscCall(VecGetArray(v,&va));
  for(PetscInt i=0;i<A.nOwned;++i) va[i]=(PetscScalar)in[(std::size_t)i];
  PetscCall(VecRestoreArray(v,&va));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customMomentumNorm2(const std::vector<double>& x,double& nrm) {
  PetscFunctionBeginUser;
  double local=0.0; for(double v:x) local+=v*v;
  double global=0.0; PetscCallMPI(MPI_Allreduce(&local,&global,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  nrm=std::sqrt(global); PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customGlobalMinMax(const std::vector<double>& v,double& gmin,double& gmax) {
  PetscFunctionBeginUser;
  if(v.empty()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"M6B empty native vector in min/max");
  const double lmin=*std::min_element(v.begin(),v.end()),lmax=*std::max_element(v.begin(),v.end());
  PetscCallMPI(MPI_Allreduce(&lmin,&gmin,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&lmax,&gmax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customMomentumResidualNorm(CustomMomentumCSR& A,const std::vector<double>& b,const std::vector<double>& x,double& rn) {
  PetscFunctionBeginUser;
  PetscCall(customMomentumMatVec(A,A.aRel,x,A.workY));
  double local=0.0;
  for(PetscInt i=0;i<A.nOwned;++i) { const double r=b[(std::size_t)i]-A.workY[(std::size_t)i]; local+=r*r; }
  double global=0.0; PetscCallMPI(MPI_Allreduce(&local,&global,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  rn=std::sqrt(global);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode smoothSolveCustomMomentumMPI(CustomMomentumCSR& A,Vec b,Vec x, PetscReal rtol, PetscReal relDrop,
  PetscInt maxIts, PetscInt checkEvery, PetscReal omega, PetscInt localSweeps, PetscInt *parallelIts, PetscReal *relres) {
  PetscFunctionBeginUser;
  PetscCall(customMomentumVecOwned(b,A,A.workB)); PetscCall(customMomentumVecOwned(x,A,A.workX));
  double bn=0.0,rn=0.0,rnInitial=-1.0; PetscCall(customMomentumNorm2(A.workB,bn));
  if(!std::isfinite(bn)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"custom momentum RHS norm is NaN/Inf before local SGS");
  if(bn==0.0) bn=1.0;

  if(relDrop>0.0) {
    PetscCall(customMomentumResidualNorm(A,A.workB,A.workX,rn)); rnInitial=rn;
    if(!std::isfinite(rnInitial)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"custom momentum initial residual is NaN/Inf");
    if(rnInitial/bn<rtol) { *parallelIts=0; *relres=(PetscReal)(rnInitial/bn); PetscCall(customMomentumWriteOwned(x,A,A.workX)); PetscFunctionReturn(PETSC_SUCCESS); }
  }

  PetscInt it=0;
  while(it<maxIts) {
    const PetscInt chunk=PetscMin(checkEvery,maxIts-it);
    // Match MPIAIJ MatSOR semantics: each outer SOR iteration refreshes the
    // processor-boundary values, then performs localSweeps symmetric GS sweeps.
    for(PetscInt q=0;q<chunk;++q) PetscCall(customMomentumSymmetricGS(A,A.workB,A.workX,(double)omega,localSweeps));
    it+=chunk;
    PetscCall(customMomentumResidualNorm(A,A.workB,A.workX,rn));
    if(!std::isfinite(rn)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"custom momentum local SGS generated NaN/Inf after %" PetscInt_FMT " sweeps",it);
    if(rn/bn<rtol) break;
    if(relDrop>0.0 && rnInitial>0.0 && rn<=relDrop*rnInitial) break;
  }
  PetscCall(customMomentumWriteOwned(x,A,A.workX));
  *parallelIts=it; *relres=(PetscReal)(rn/bn);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode smoothSolveCustomMomentumNative(CustomMomentumCSR& A,const std::vector<double>& b,std::vector<double>& x, double rtol,double atol,double relDrop,
  PetscInt maxIts,PetscInt checkEvery,double omega,PetscInt localSweeps,PetscInt *parallelIts,double *relres) {
  PetscFunctionBeginUser;
  if((PetscInt)b.size()!=A.nOwned || (PetscInt)x.size()!=A.nOwned) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"M6B native momentum state size mismatch");
  double bn=0.0,rn=0.0,rnInitial=-1.0; PetscCall(customMomentumNorm2(b,bn));
  if(!std::isfinite(bn)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"M6B native momentum RHS norm NaN/Inf");
  if(bn==0.0) bn=1.0;
  PetscCall(customMomentumResidualNorm(A,b,x,rn)); rnInitial=rn; if(!std::isfinite(rnInitial)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"M6B native initial momentum residual NaN/Inf");
  const double target=std::max((double)atol,(double)rtol*bn);
  if(rnInitial<=target){*parallelIts=0;*relres=rnInitial/bn;PetscFunctionReturn(PETSC_SUCCESS);}
  PetscInt it=0; while(it<maxIts){ const PetscInt chunk=PetscMin(checkEvery,maxIts-it); for(PetscInt q=0;q<chunk;++q) PetscCall(customMomentumSymmetricGS(A,b,x,omega,localSweeps)); it+=chunk;
    PetscCall(customMomentumResidualNorm(A,b,x,rn)); if(!std::isfinite(rn)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"M6B native SGS NaN/Inf");
    if(rn<=target) break; if(relDrop>0.0 && rnInitial>0.0 && rn<=relDrop*rnInitial) break; }
  *parallelIts=it; *relres=rn/bn; PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customMomentumSgsParity(Mat Ar,Vec b,Vec x0,CustomMomentumCSR& A,double omega,PetscInt localSweeps,CustomParityNorm& out) {
  PetscFunctionBeginUser;
  Vec xp=nullptr; PetscCall(VecDuplicate(b,&xp)); PetscCall(VecCopy(x0,xp));
  std::vector<double> bc,xc,xpOwned; PetscCall(customMomentumVecOwned(b,A,bc)); PetscCall(customMomentumVecOwned(x0,A,xc));
  PetscCall(MatSOR(Ar,b,omega,SOR_LOCAL_SYMMETRIC_SWEEP,0.0,1,localSweeps,xp));
  PetscCall(customMomentumSymmetricGS(A,bc,xc,omega,localSweeps)); PetscCall(customMomentumVecOwned(xp,A,xpOwned));
  PetscCall(customMomentumCompareArrays(xc,xpOwned,out)); PetscCall(VecDestroy(&xp)); PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customMomentumShadowGate(PetscInt it,CustomMomentumCSR& A,Mat C,Mat Sg,Mat Aphys,Mat Ar,
  Vec diag,Vec relaxDiag,Vec relaxedDiag,Vec metric,Vec rAU,bool centralConvection,bool implicitSupg,
  const std::string& uRelaxMode,double alphaU,const std::string& simpleVariant,const std::string& rauMode,
  double simplecBlend,double floorFraction,const std::string& fallback,double rauScale,double tol,PetscBool strict) {
  PetscFunctionBeginUser;
  if(centralConvection) PetscCall(customMomentumLoadFromPetsc(C,A,A.convection)); else std::fill(A.convection.begin(),A.convection.end(),0.0);
  if(implicitSupg) PetscCall(customMomentumLoadFromPetsc(Sg,A,A.supg)); else std::fill(A.supg.begin(),A.supg.end(),0.0);
  PetscCall(customMomentumDerive(A,centralConvection,implicitSupg,uRelaxMode,alphaU,simpleVariant,rauMode,simplecBlend,floorFraction,fallback,rauScale));

  CustomParityNorm ap,ar,act,pd,rd,rdiag,met,rau;
  PetscCall(customMomentumMatrixParity(Aphys,A,A.aPhys,ap)); PetscCall(customMomentumMatrixParity(Ar,A,A.aRel,ar));
  PetscCall(customMomentumActionParity(Ar,A,A.aRel,act)); PetscCall(customMomentumVecParity(diag,A,A.physDiag,pd));
  PetscCall(customMomentumVecParity(relaxDiag,A,A.relaxDelta,rd)); PetscCall(customMomentumVecParity(relaxedDiag,A,A.relaxedDiag,rdiag));
  if(metric) PetscCall(customMomentumVecParity(metric,A,A.metric,met)); else { met.rel=0.0; met.maxAbs=0.0; }
  PetscCall(customMomentumVecParity(rAU,A,A.rAU,rau));
  const double worst=std::max({ap.rel,ar.rel,act.rel,pd.rel,rd.rel,rdiag.rel,met.rel,rau.rel});
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_CUSTOM_MOM_SHADOW it=%" PetscInt_FMT " AphysRel=%.3e ArRel=%.3e actionRel=%.3e physDiagRel=%.3e relaxDeltaRel=%.3e relaxedDiagRel=%.3e metricRel=%.3e rAURel=%.3e maxAbsAction=%.3e tol=%.3e status=%s\n",
    it,ap.rel,ar.rel,act.rel,pd.rel,rd.rel,rdiag.rel,met.rel,rau.rel,act.maxAbs,tol,worst<=tol?"PASS":"CHECK"));
  if(strict && worst>tol) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom momentum shadow parity exceeded tolerance");
  PetscFunctionReturn(PETSC_SUCCESS);
}

struct CentralCellPlan {
  PetscInt cell=-1;
  PetscInt entity[8]={0,0,0,0,0,0,0,0};
  PetscInt gid[8]={-1,-1,-1,-1,-1,-1,-1,-1};
  PetscInt localIndex[8]={-1,-1,-1,-1,-1,-1,-1,-1};
  PetscInt freeBasis[8]={0,0,0,0,0,0,0,0};
  PetscInt freeGid[8]={0,0,0,0,0,0,0,0};
  PetscInt nfree=0;
  double det=0.0;
  double invJ[3][3]={{0}};
};

struct CentralAssemblyPlan {
  std::vector<CentralCellPlan> cells;
};

static PetscErrorCode buildCentralAssemblyPlan(const Mesh& M,const Discrete& D,int rank,const GhostPlan& G,CentralAssemblyPlan& P) {
  PetscFunctionBeginUser;
  const PetscInt nv=(PetscInt)M.points.size();
  P.cells.clear();
  P.cells.reserve(D.cellCount[rank]);
  for(PetscInt c=0;c<(PetscInt)M.tets.size();++c) if(D.cellOwner[c]==rank) {
    CentralCellPlan cp; cp.cell=c;
    const auto t=M.tets[c];
    const Vec3 X[4]={M.points[t[0]],M.points[t[1]],M.points[t[2]],M.points[t[3]]};
    double J[3][3]={{X[1].x-X[0].x,X[2].x-X[0].x,X[3].x-X[0].x},
                    {X[1].y-X[0].y,X[2].y-X[0].y,X[3].y-X[0].y},
                    {X[1].z-X[0].z,X[2].z-X[0].z,X[3].z-X[0].z}};
    cp.det=det3(J);
    if(cp.det<=0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"non-positive tet orientation at cell %" PetscInt_FMT,c);
    inv3(J,cp.invJ);
    for(int i=0;i<4;++i) cp.entity[i]=t[i];
    for(int i=0;i<4;++i) cp.entity[4+i]=nv+M.oppFace[c][i];
    for(int a=0;a<8;++a) {
      cp.gid[a]=D.g2free[cp.entity[a]];
      if(cp.gid[a]>=0) {
        cp.localIndex[a]=velocityLocalIndex(G,cp.gid[a]);
        cp.freeBasis[cp.nfree]=a;
        cp.freeGid[cp.nfree]=cp.gid[a];
        cp.nfree++;
      }
    }
    P.cells.push_back(cp);
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_CENTRAL_PLAN cachedAffineGeometry=1 cachedEntityMaps=1 ownedCells=%zu tensorQuadrature=collapsed_5x5x5_degree8 matrixInsertion=batched_element_MatSetValues\n",
    P.cells.size()));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode assembleCentralConvectionFastMPI(const Mesh& M,const Discrete& D,const GhostPlan& G,const CentralAssemblyPlan& P,Vec U[3],Mat C,Vec dirRhs[3]) {
  PetscFunctionBeginUser;
  PetscCall(MatZeroEntries(C));
  for(int d=0;d<3;++d) PetscCall(VecSet(dirRhs[d],0.0));
  Vec Ul[3]={nullptr,nullptr,nullptr};
  const PetscScalar* ua[3]={nullptr,nullptr,nullptr};
  for(int d=0;d<3;++d) {
    PetscCall(VecGhostUpdateBegin(U[d],INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecGhostUpdateEnd(U[d],INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecGhostGetLocalForm(U[d],&Ul[d]));
    PetscCall(VecGetArrayRead(Ul[d],&ua[d]));
  }
  const auto& T=centralTensor();
  for(const auto& cp:P.cells) {
    double coeff[3][8]={{0}};
    for(int m=0;m<8;++m) {
      if(cp.gid[m]>=0) {
        const PetscInt li=cp.localIndex[m];
        for(int d=0;d<3;++d) coeff[d][m]=PetscRealPart(ua[d][li]);
      } else {
        for(int d=0;d<3;++d) coeff[d][m]=entityDirValue(D,d,cp.entity[m]);
      }
    }
    double uref[8][3]={{0}};
    for(int m=0;m<8;++m)
      for(int j=0;j<3;++j)
        for(int d=0;d<3;++d)
          uref[m][j] += coeff[d][m]*cp.invJ[j][d];

    double Cl[8][8]={{0}};
    for(int a=0;a<8;++a)
      for(int b=0;b<8;++b) {
        double v=0.0;
        for(int m=0;m<8;++m)
          for(int j=0;j<3;++j)
            v += uref[m][j]*T.t[a][m][b][j];
        Cl[a][b]=cp.det*v;
      }

    if(cp.nfree>0) {
      PetscScalar vals[64];
      for(PetscInt ii=0;ii<cp.nfree;++ii) {
        const int a=(int)cp.freeBasis[ii];
        for(PetscInt jj=0;jj<cp.nfree;++jj) {
          const int b=(int)cp.freeBasis[jj];
          vals[ii*cp.nfree+jj]=(PetscScalar)Cl[a][b];
        }
      }
      PetscCall(MatSetValues(C,cp.nfree,cp.freeGid,cp.nfree,cp.freeGid,vals,ADD_VALUES));

      for(int d=0;d<3;++d) {
        PetscScalar rv[8];
        PetscBool any=PETSC_FALSE;
        for(PetscInt ii=0;ii<cp.nfree;++ii) {
          const int a=(int)cp.freeBasis[ii];
          double v=0.0;
          for(int b=0;b<8;++b) if(cp.gid[b]<0) {
            const double ud=entityDirValue(D,d,cp.entity[b]);
            if(ud!=0.0) v-=Cl[a][b]*ud;
          }
          rv[ii]=(PetscScalar)v;
          if(v!=0.0) any=PETSC_TRUE;
        }
        if(any) PetscCall(VecSetValues(dirRhs[d],cp.nfree,cp.freeGid,rv,ADD_VALUES));
      }
    }
  }
  for(int d=0;d<3;++d) {
    PetscCall(VecRestoreArrayRead(Ul[d],&ua[d]));
    PetscCall(VecGhostRestoreLocalForm(U[d],&Ul[d]));
  }
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  for(int d=0;d<3;++d){PetscCall(VecAssemblyBegin(dirRhs[d]));PetscCall(VecAssemblyEnd(dirRhs[d]));}
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode assembleCentralConvectionMPI(const Mesh& M,const Discrete& D,int rank,const GhostPlan& G,Vec U[3],Mat C,Vec dirRhs[3]) {
  PetscFunctionBeginUser;
  PetscCall(MatZeroEntries(C));
  for(int d=0;d<3;++d) PetscCall(VecSet(dirRhs[d],0.0));
  Vec Ul[3]={nullptr,nullptr,nullptr};
  const PetscScalar* ua[3]={nullptr,nullptr,nullptr};
  for(int d=0;d<3;++d) {
    PetscCall(VecGhostUpdateBegin(U[d],INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecGhostUpdateEnd(U[d],INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecGhostGetLocalForm(U[d],&Ul[d]));
    PetscCall(VecGetArrayRead(Ul[d],&ua[d]));
  }
  const auto& T=centralTensor();
  const PetscInt nv=(PetscInt)M.points.size();
  for(PetscInt c=0;c<(PetscInt)M.tets.size();++c) if(D.cellOwner[c]==rank) {
    const auto t=M.tets[c];
    const Vec3 X[4]={M.points[t[0]],M.points[t[1]],M.points[t[2]],M.points[t[3]]};
    double J[3][3]={{X[1].x-X[0].x,X[2].x-X[0].x,X[3].x-X[0].x},
                    {X[1].y-X[0].y,X[2].y-X[0].y,X[3].y-X[0].y},
                    {X[1].z-X[0].z,X[2].z-X[0].z,X[3].z-X[0].z}},invJ[3][3];
    const double det=det3(J);
    inv3(J,invJ);
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
    double uref[8][3]={{0}};
    for(int m=0;m<8;++m)
      for(int j=0;j<3;++j)
        for(int d=0;d<3;++d)
          uref[m][j] += coeff[d][m]*invJ[j][d];

    double Cl[8][8]={{0}};
    for(int a=0;a<8;++a)
      for(int b=0;b<8;++b) {
        double v=0;
        for(int m=0;m<8;++m)
          for(int j=0;j<3;++j)
            v += uref[m][j]*T.t[a][m][b][j];
        Cl[a][b]=det*v;
      }

    for(int a=0;a<8;++a) {
      const PetscInt ia=D.g2free[lg[a]];
      if(ia<0) continue;
      for(int b=0;b<8;++b) {
        const PetscInt ib=D.g2free[lg[b]];
        if(ib>=0) PetscCall(MatSetValue(C,ia,ib,Cl[a][b],ADD_VALUES));
        else for(int d=0;d<3;++d) {
          const double ud=entityDirValue(D,d,lg[b]);
          if(ud!=0.0) PetscCall(VecSetValue(dirRhs[d],ia,-Cl[a][b]*ud,ADD_VALUES));
        }
      }
    }
  }
  for(int d=0;d<3;++d) {
    PetscCall(VecRestoreArrayRead(Ul[d],&ua[d]));
    PetscCall(VecGhostRestoreLocalForm(U[d],&Ul[d]));
  }
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  for(int d=0;d<3;++d){PetscCall(VecAssemblyBegin(dirRhs[d]));PetscCall(VecAssemblyEnd(dirRhs[d]));}
  PetscFunctionReturn(PETSC_SUCCESS);
}


