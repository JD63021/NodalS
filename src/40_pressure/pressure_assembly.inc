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

struct PressureAssemblyPlan {
  Mat S=nullptr;
  Mat Pcompact=nullptr; // compact face-neighbour GAMG surrogate
  PetscInt globalSchurNnz=0; // structural NNZ of explicitly expanded B diag(rAU) B^T
  std::vector<std::vector<PetscInt>> vertexCells;
  std::vector<std::array<double,24>> Bcell; // [d*8+a]
  // Geometry-only compact FV/LSQ data, indexed by cell/local tet face.
  // fvLsqGeom is |S_f dot c_Kf| where c_Kf is the direct-neighbour
  // coefficient in a one-ring weighted LSQ gradient.  fvTpfaGeom is a
  // positive nonorthogonality-capped two-point fallback.
  std::vector<std::array<double,4>> fvLsqGeom;
  std::vector<std::array<double,4>> fvTpfaGeom;
  std::vector<std::array<double,4>> fvNonorthCos;
  // Pressure-cell diagonal exchange used by the FV/LSQ surrogate.  The
  // velocity ghost plan contains the DOFs of owned elements only, so it is
  // deliberately NOT used to inspect a neighbouring off-rank cell.  Instead
  // each rank computes exact Schur diagonals for its owned pressure cells and
  // ghosts only the face-neighbour cell diagonals needed by the compact Pmat.
  Vec fvCellDiag=nullptr;
  PetscInt fvCellDiagStart=0;
  std::unordered_map<PetscInt,PetscInt> fvCellDiagLocal; // pressure gid -> local-form index
  std::vector<SchurRowPlan> rows;
  std::vector<PetscInt> flatRowGid,flatRowColOffset,flatColGid,flatColTermOffset,flatTermLocalRau;
  std::vector<double> flatTermCoeff;
};


static unsigned long long u64bytes(std::size_t n) {
  return (unsigned long long)n;
}

template<class T>
static unsigned long long vectorUsefulBytes(const std::vector<T>& v) {
  return u64bytes(v.size()) * u64bytes(sizeof(T));
}

template<class T>
static unsigned long long vectorRetainedBytes(const std::vector<T>& v) {
  return u64bytes(sizeof(v)) + u64bytes(v.capacity()) * u64bytes(sizeof(T));
}

template<class K,class V>
static unsigned long long unorderedMapRetainedEstimateBytes(const std::unordered_map<K,V>& m) {
  // Lower-bound-ish retained estimate: hash buckets plus one pair and two link/hash
  // words per node.  The allocator's own metadata is intentionally not guessed.
  return u64bytes(sizeof(m))
       + u64bytes(m.bucket_count()) * u64bytes(sizeof(void*))
       + u64bytes(m.size()) * (u64bytes(sizeof(std::pair<const K,V>)) + 2u*u64bytes(sizeof(void*)));
}

static PetscErrorCode printMemoryAuditBytes(const char *stage,const char *name,
                                            unsigned long long localUseful,
                                            unsigned long long localRetained,
                                            PetscInt cells,const char *scope,
                                            const char *note="") {
  PetscFunctionBeginUser;
  unsigned long long useful=0,retained=0,retMin=0,retMax=0;
  PetscCallMPI(MPI_Allreduce(&localUseful,&useful,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localRetained,&retained,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localRetained,&retMin,1,MPI_UNSIGNED_LONG_LONG,MPI_MIN,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localRetained,&retMax,1,MPI_UNSIGNED_LONG_LONG,MPI_MAX,PETSC_COMM_WORLD));
  const double mib=retained/(1024.0*1024.0);
  const double bpc=cells>0?(double)retained/(double)cells:0.0;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_MEMORY_AUDIT_OBJECT stage=%s name=%s scope=%s usefulBytes=%llu retainedEstimateBytes=%llu retainedMiB=%.3f retainedBytesPerCell=%.3f minRankRetainedBytes=%llu maxRankRetainedBytes=%llu note=%s\n",
    stage,name,scope,useful,retained,mib,bpc,retMin,retMax,note));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode printMemoryAuditMat(const char *stage,const char *name,Mat A,PetscInt cells) {
  PetscFunctionBeginUser;
  if(!A) PetscFunctionReturn(PETSC_SUCCESS);
  MatInfo info;
  PetscInt nr=0,nc=0;
  PetscCall(MatGetSize(A,&nr,&nc));
  PetscCall(MatGetInfo(A,MAT_GLOBAL_SUM,&info));
  const double ratio=info.nz_used>0.0?info.nz_allocated/info.nz_used:0.0;
  const double memMiB=info.memory/(1024.0*1024.0);
  const double memBpc=cells>0?info.memory/(double)cells:0.0;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_MEMORY_AUDIT_MAT stage=%s name=%s rows=%" PetscInt_FMT " cols=%" PetscInt_FMT
    " nzUsed=%.0f nzAllocated=%.0f nzUnneeded=%.0f allocOverUsed=%.6f mallocs=%.0f memoryBytes=%.0f memoryMiB=%.3f memoryBytesPerCell=%.3f\n",
    stage,name,nr,nc,info.nz_used,info.nz_allocated,info.nz_unneeded,ratio,info.mallocs,info.memory,memMiB,memBpc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode auditMeshMemory(const Mesh& M,PetscInt cells,const char *scope="replicated_per_rank") {
  PetscFunctionBeginUser;
  unsigned long long useful=0,retained=0;
  useful += vectorUsefulBytes(M.points); retained += vectorRetainedBytes(M.points);
  PetscCall(printMemoryAuditBytes("after_mesh","mesh.points",vectorUsefulBytes(M.points),vectorRetainedBytes(M.points),cells,scope));
  const unsigned long long fu=vectorUsefulBytes(M.faces),fr=vectorRetainedBytes(M.faces);
  useful+=fu;retained+=fr;
  PetscCall(printMemoryAuditBytes("after_mesh","mesh.faces_compact",fu,fr,cells,scope,"fixed_std_array_3_vertices_no_per_face_heap_allocation"));
  auto addvec=[&](const char *name,const auto& v){
    const auto u=vectorUsefulBytes(v),r=vectorRetainedBytes(v); useful+=u;retained+=r;
    return printMemoryAuditBytes("after_mesh",name,u,r,cells,scope);
  };
  PetscCall(addvec("mesh.owner",M.owner));
  PetscCall(addvec("mesh.neighbour",M.neighbour));
  PetscCall(addvec("mesh.tets",M.tets));
  PetscCall(addvec("mesh.oppFace",M.oppFace));
  PetscCall(addvec("mesh.facePatch",M.facePatch));
  unsigned long long pu=vectorUsefulBytes(M.patches),pr=u64bytes(sizeof(M.patches))+u64bytes(M.patches.capacity())*u64bytes(sizeof(Patch));
  for(const auto& p:M.patches){pu+=u64bytes(p.name.size()+1);pr+=u64bytes(p.name.capacity()+1);}
  useful+=pu;retained+=pr;
  PetscCall(printMemoryAuditBytes("after_mesh","mesh.patches",pu,pr,cells,scope));
  PetscCall(printMemoryAuditBytes("after_mesh","mesh.TOTAL_CPP",useful,retained,cells,scope,"excludes_allocator_metadata_and_temporary_parser_strings"));
  PetscFunctionReturn(PETSC_SUCCESS);
}


static unsigned long long meshRetainedEstimateLocal(const Mesh& M) {
  unsigned long long r=0;
  r+=vectorRetainedBytes(M.points)+vectorRetainedBytes(M.faces)+vectorRetainedBytes(M.owner)+vectorRetainedBytes(M.neighbour)
    +vectorRetainedBytes(M.tets)+vectorRetainedBytes(M.oppFace)+vectorRetainedBytes(M.facePatch);
  r+=u64bytes(sizeof(M.patches))+u64bytes(M.patches.capacity())*u64bytes(sizeof(Patch));
  for(const auto& p:M.patches) r+=u64bytes(p.name.capacity()+1);
  return r;
}
static PetscErrorCode auditRootGlobalMeshMemory(const Mesh& Mroot,int rank,PetscInt cells) {
  PetscFunctionBeginUser;
  const unsigned long long b=(rank==0)?meshRetainedEstimateLocal(Mroot):0ull;
  PetscCall(printMemoryAuditBytes("after_mesh","mesh.rootGlobal.TOTAL_CPP",b,b,cells,"root_only_single_global_copy","retained_only_on_rank0_for_postprocess_and_global_oracle"));
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode auditRootGlobalOwnershipMemory(const Discrete& Droot,int rank,PetscInt cells) {
  PetscFunctionBeginUser; unsigned long long b=0;
  if(rank==0) b=vectorRetainedBytes(Droot.g2free)+vectorRetainedBytes(Droot.pGid)+vectorRetainedBytes(Droot.cellOwner)+vectorRetainedBytes(Droot.velCount)+vectorRetainedBytes(Droot.cellCount)+vectorRetainedBytes(Droot.fixedEntity)+vectorRetainedBytes(Droot.fixedDirValue);
  PetscCall(printMemoryAuditBytes("after_fe_assembly","discrete.rootGlobalOwnership",b,b,cells,"root_only_single_global_copy","global_gid_owner_maps_retained_on_rank0_for_final_diagnostics"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode auditDiscreteMemory(const Discrete& D,PetscInt cells,PetscBool distributed=PETSC_FALSE) {
  PetscFunctionBeginUser;
  int rank=0; PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  unsigned long long useful=0,retained=0;
  const char *mapScope=distributed?"distributed_rank_local_support":"replicated_per_rank";
  auto emit=[&](const char *name,const auto& v,const char *scope=nullptr)->PetscErrorCode{
    const auto u=vectorUsefulBytes(v),r=vectorRetainedBytes(v); useful+=u;retained+=r;
    return printMemoryAuditBytes("after_fe_assembly",name,u,r,cells,scope?scope:mapScope);
  };
  PetscCall(emit("discrete.g2free",D.g2free));
  PetscCall(emit("discrete.pGid",D.pGid));
  PetscCall(emit("discrete.cellOwner",D.cellOwner));
  PetscCall(emit("discrete.velCount",D.velCount,"small_rank_metadata"));
  PetscCall(emit("discrete.cellCount",D.cellCount,"small_rank_metadata"));
  PetscCall(emit("discrete.fixedEntity",D.fixedEntity));
  PetscCall(emit("discrete.fixedDirValueCompact",D.fixedDirValue));
  PetscCall(emit("discrete.volumesOwnedFP64",D.volumesOwnedFP64,"distributed_owned_pressure_rows"));
  PetscCall(emit("discrete.fixedDivOwnedFP64",D.fixedDivOwnedFP64,"distributed_owned_pressure_rows"));
  for(int d=0;d<3;++d){std::string n="discrete.rhsOwnedFP64_"+std::to_string(d);PetscCall(emit(n.c_str(),D.rhsOwnedFP64[(std::size_t)d],"distributed_owned_velocity_rows"));}
  PetscCall(printMemoryAuditBytes("after_fe_assembly","discrete.TOTAL_CPP",useful,retained,cells,distributed?"mixed_distributed":"mixed",distributed?"local_entity_cell_maps_plus_distributed_owned_payloads":"most_large_arrays_are_global_and_replicated_per_rank"));
  PetscCall(printMemoryAuditMat("after_fe_assembly","D.A_diffusion",D.A,cells));
  for(int d=0;d<3;++d){std::string n="D.B"+std::to_string(d);PetscCall(printMemoryAuditMat("after_fe_assembly",n.c_str(),D.B[d],cells));}
  unsigned long long rhsVecs=0;for(int d=0;d<3;++d)if(D.rhs[d])++rhsVecs;
  const unsigned long long vecPayload=(rhsVecs*(unsigned long long)D.velCount[rank] + 2ull*(unsigned long long)D.cellCount[rank])*sizeof(PetscScalar);
  PetscCall(printMemoryAuditBytes("after_fe_assembly","petsc.initial_vec_numeric_payload",vecPayload,vecPayload,cells,"distributed_owned_payload","M6B_pressure_layout_bridges_plus_optional_velocity_rhs_reference"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode auditPlanMemory(const GhostPlan& G,const PressureAssemblyPlan& P,
                                      const CentralAssemblyPlan& C,const SupgAssemblyPlan& S,
                                      PetscInt cells) {
  PetscFunctionBeginUser;
  // Velocity ghost plan
  unsigned long long gu=vectorUsefulBytes(G.ghosts),gr=vectorRetainedBytes(G.ghosts);
  gu += u64bytes(G.ghostLocal.size())*u64bytes(sizeof(std::pair<const PetscInt,PetscInt>));
  gr += unorderedMapRetainedEstimateBytes(G.ghostLocal);
  PetscCall(printMemoryAuditBytes("after_plans","ghost.velocity",gu,gr,cells,"distributed_owned_plus_halo","unordered_map_retained_is_lower_bound_estimate"));

  // Pressure plan: break out the globally replicated and local pieces.
  unsigned long long vu=u64bytes(sizeof(P.vertexCells))+u64bytes(P.vertexCells.size())*u64bytes(sizeof(std::vector<PetscInt>));
  unsigned long long vr=u64bytes(sizeof(P.vertexCells))+u64bytes(P.vertexCells.capacity())*u64bytes(sizeof(std::vector<PetscInt>));
  for(const auto& v:P.vertexCells){vu+=vectorUsefulBytes(v);vr+=u64bytes(v.capacity())*u64bytes(sizeof(PetscInt));}
  PetscCall(printMemoryAuditBytes("after_plans","pressure.vertexCells",vu,vr,cells,"replicated_per_rank","nested_support_vectors"));
  PetscCall(printMemoryAuditBytes("after_plans","pressure.Bcell",vectorUsefulBytes(P.Bcell),vectorRetainedBytes(P.Bcell),cells,"replicated_per_rank","24_doubles_per_global_cell_per_rank"));
  PetscCall(printMemoryAuditBytes("after_plans","pressure.fvLsqGeom",vectorUsefulBytes(P.fvLsqGeom),vectorRetainedBytes(P.fvLsqGeom),cells,"replicated_per_rank"));
  PetscCall(printMemoryAuditBytes("after_plans","pressure.fvTpfaGeom",vectorUsefulBytes(P.fvTpfaGeom),vectorRetainedBytes(P.fvTpfaGeom),cells,"replicated_per_rank"));
  PetscCall(printMemoryAuditBytes("after_plans","pressure.fvNonorthCos",vectorUsefulBytes(P.fvNonorthCos),vectorRetainedBytes(P.fvNonorthCos),cells,"replicated_per_rank"));

  unsigned long long ru=u64bytes(sizeof(P.rows))+u64bytes(P.rows.size())*u64bytes(sizeof(SchurRowPlan));
  unsigned long long rr=u64bytes(sizeof(P.rows))+u64bytes(P.rows.capacity())*u64bytes(sizeof(SchurRowPlan));
  unsigned long long rows=0,cols=0,terms=0;
  for(const auto& r:P.rows){
    ++rows;
    ru+=u64bytes(r.columns.size())*u64bytes(sizeof(SchurColumnPlan));
    rr+=u64bytes(r.columns.capacity())*u64bytes(sizeof(SchurColumnPlan));
    cols+=r.columns.size();
    for(const auto& c:r.columns){
      ru+=u64bytes(c.terms.size())*u64bytes(sizeof(SchurTerm));
      rr+=u64bytes(c.terms.capacity())*u64bytes(sizeof(SchurTerm));
      terms+=c.terms.size();
    }
  }
  PetscCall(printMemoryAuditBytes("after_plans","pressure.schurNestedRows",ru,rr,cells,"distributed_owned_rows","retained_includes_vector_objects_and_capacities_excludes_allocator_metadata"));
  const unsigned long long fu=vectorUsefulBytes(P.flatRowGid)+vectorUsefulBytes(P.flatRowColOffset)+vectorUsefulBytes(P.flatColGid)+vectorUsefulBytes(P.flatColTermOffset)+vectorUsefulBytes(P.flatTermLocalRau)+vectorUsefulBytes(P.flatTermCoeff);
  const unsigned long long fr=vectorRetainedBytes(P.flatRowGid)+vectorRetainedBytes(P.flatRowColOffset)+vectorRetainedBytes(P.flatColGid)+vectorRetainedBytes(P.flatColTermOffset)+vectorRetainedBytes(P.flatTermLocalRau)+vectorRetainedBytes(P.flatTermCoeff);
  PetscCall(printMemoryAuditBytes("after_plans","pressure.schurFlatPlan",fu,fr,cells,"distributed_owned_rows","flat_row_col_term_arrays_no_nested_vectors"));
  unsigned long long mapu=u64bytes(P.fvCellDiagLocal.size())*u64bytes(sizeof(std::pair<const PetscInt,PetscInt>));
  unsigned long long mapr=unorderedMapRetainedEstimateBytes(P.fvCellDiagLocal);
  PetscCall(printMemoryAuditBytes("after_plans","pressure.fvCellDiagLocal_map",mapu,mapr,cells,"distributed_owned_plus_halo","lower_bound_map_estimate"));
  const unsigned long long fvDiagVecLocal=u64bytes(P.fvCellDiagLocal.size())*u64bytes(sizeof(PetscScalar));
  PetscCall(printMemoryAuditBytes("after_plans","pressure.fvCellDiag_vec_numeric_payload",fvDiagVecLocal,fvDiagVecLocal,cells,"distributed_owned_plus_halo","PETSc_Vec_numeric_slots_only"));
  PetscCall(printMemoryAuditMat("after_plans","pressure.S_explicit_snapshot",P.S,cells));
  PetscCall(printMemoryAuditMat("after_plans","pressure.Pcompact",P.Pcompact,cells));
  if(!P.flatRowGid.empty()){rows=P.flatRowGid.size();cols=P.flatColGid.size();terms=P.flatTermLocalRau.size();}
  unsigned long long grow=rows,gcol=cols,gterm=terms;
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE,&grow,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE,&gcol,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE,&gterm,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_MEMORY_AUDIT_SCHUR_COUNTS ownedRows=%llu uniqueColumns=%llu algebraicTerms=%llu avgColumnsPerRow=%.3f avgTermsPerColumn=%.3f sizeofRow=%zu sizeofColumn=%zu sizeofTerm=%zu\n",
    grow,gcol,gterm,grow?(double)gcol/(double)grow:0.0,gcol?(double)gterm/(double)gcol:0.0,sizeof(SchurRowPlan),sizeof(SchurColumnPlan),sizeof(SchurTerm)));
  if(!P.flatRowGid.empty()) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_M5A_FLAT_SCHUR rows=%llu columns=%llu terms=%llu BcellBytes=%zu nestedRows=%zu\n",grow,gcol,gterm,P.Bcell.size()*sizeof(std::array<double,24>),P.rows.size()));

  // Central plan is owned-cell only.
  PetscCall(printMemoryAuditBytes("after_plans","central.cells",vectorUsefulBytes(C.cells),vectorRetainedBytes(C.cells),cells,"distributed_owned_cells"));

  // SUPG: reference table is replicated tiny data; cells/caches are owned-cell only.
  PetscCall(printMemoryAuditBytes("after_plans","supg.reference",vectorUsefulBytes(S.ref),vectorRetainedBytes(S.ref),cells,"replicated_per_rank"));
  PetscCall(printMemoryAuditBytes("after_plans","supg.cells",vectorUsefulBytes(S.cells),vectorRetainedBytes(S.cells),cells,"distributed_owned_cells"));
  PetscCall(printMemoryAuditBytes("after_plans","supg.grad_physical_q_basis_xyz",0,0,cells,"eliminated","reconstructed_from_12_gradLambda_doubles_per_owned_cell"));
  PetscCall(printMemoryAuditBytes("after_plans","supg.viscStrong_q_basis",0,0,cells,"eliminated","BF3_laplacian_reconstructed_from_gradLambda_dot_products"));
  PetscCall(printMemoryAuditBytes("after_plans","supg.forcing",vectorUsefulBytes(S.forcing),vectorRetainedBytes(S.forcing),cells,"distributed_owned_cells","empty_for_pipe_and_generic_flow"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode auditCustomMomentumDetailedMemory(const CustomMomentumCSR& A,PetscInt cells) {
  PetscFunctionBeginUser;
  auto ints=[&](const char *name,const auto&... vecs)->PetscErrorCode {
    unsigned long long u=0,r=0;
    ((u+=vectorUsefulBytes(vecs),r+=vectorRetainedBytes(vecs)),...);
    return printMemoryAuditBytes("after_state_objects",name,u,r,cells,"distributed_owned_velocity_rows_plus_halo");
  };
  auto doubles=[&](const char *name,const auto&... vecs)->PetscErrorCode {
    unsigned long long u=0,r=0;
    ((u+=vectorUsefulBytes(vecs),r+=vectorRetainedBytes(vecs)),...);
    return printMemoryAuditBytes("after_state_objects",name,u,r,cells,"distributed_owned_velocity_rows_plus_halo");
  };

  // CSR/index topology retained by the current assembled-SGS predictor.
  PetscCall(ints("momentum.csr.rowPtr",A.rowPtr));
  PetscCall(ints("momentum.csr.colGid_RELEASED_M1",A.colGid));
  PetscCall(ints("momentum.csr.colLocal",A.colLocal));
  PetscCall(ints("momentum.csr.diagPos",A.diagPos));
  PetscCall(ints("momentum.halo.gids_offsets",A.ghostGid,A.reqRecvGid,A.reqRecvLocalOffset));

  unsigned long long iu=0,ir=0;
  auto addInt=[&](const auto& v){iu+=vectorUsefulBytes(v);ir+=vectorRetainedBytes(v);};
  addInt(A.reqSendCounts);addInt(A.reqSendDispls);addInt(A.reqRecvCounts);addInt(A.reqRecvDispls);
  PetscCall(printMemoryAuditBytes("after_state_objects","momentum.halo.counts_displs",iu,ir,cells,"small_rank_peer_metadata"));
  const unsigned long long requ=(unsigned long long)A.exchangeRequests.size()*sizeof(MPI_Request);
  const unsigned long long reqr=(unsigned long long)sizeof(A.exchangeRequests)+(unsigned long long)A.exchangeRequests.capacity()*sizeof(MPI_Request);
  PetscCall(printMemoryAuditBytes("after_state_objects","momentum.halo.mpi_requests",requ,reqr,cells,"small_rank_peer_metadata"));

  // M1+M2 actual payload: kNu is intentionally empty; only active aRel remains.
  PetscCall(doubles("momentum.matrix.kNu_ELIMINATED_M2",A.kNu));
  PetscCall(doubles("momentum.matrix.aRel_FP64",A.aRel));

  // Diagonal/coupling quantities. A matrix-free rewrite must KEEP or reproduce
  // these because SIMPLE/SIMPLEC pressure coupling uses the resulting rAU.
  PetscCall(doubles("momentum.diag.physDiag",A.physDiag));
  PetscCall(doubles("momentum.diag.relaxDelta",A.relaxDelta));
  PetscCall(doubles("momentum.diag.relaxedDiag",A.relaxedDiag));
  PetscCall(doubles("momentum.diag.metric",A.metric));
  PetscCall(doubles("momentum.diag.rAU",A.rAU));

  // Current custom halo numeric buffers.
  PetscCall(doubles("momentum.halo.exchange_numeric",A.exchangeSend,A.ghostValues));

  unsigned long long fou=0,forr=0,fgu=0,fgr=0,cru=0,crr=0,sru=0,srr=0;
  for(int d=0;d<3;++d){
    fou+=vectorUsefulBytes(A.fieldOwned[d]);forr+=vectorRetainedBytes(A.fieldOwned[d]);
    fgu+=vectorUsefulBytes(A.fieldGhost[d]);fgr+=vectorRetainedBytes(A.fieldGhost[d]);
    cru+=vectorUsefulBytes(A.convRhs[d]);crr+=vectorRetainedBytes(A.convRhs[d]);
    sru+=vectorUsefulBytes(A.supgRhs[d]);srr+=vectorRetainedBytes(A.supgRhs[d]);
  }
  PetscCall(printMemoryAuditBytes("after_state_objects","momentum.fields.owned3_FP64",fou,forr,cells,"distributed_owned_velocity_rows"));
  PetscCall(printMemoryAuditBytes("after_state_objects","momentum.fields.ghost3_FP64",fgu,fgr,cells,"distributed_velocity_halo"));
  PetscCall(printMemoryAuditBytes("after_state_objects","momentum.rhs.convection3_FP64",cru,crr,cells,"distributed_owned_velocity_rows"));
  PetscCall(printMemoryAuditBytes("after_state_objects","momentum.rhs.supg3_FP64",sru,srr,cells,"distributed_owned_velocity_rows"));
  PetscCall(doubles("momentum.work.sgs4_FP64",A.workB,A.workX,A.workY,A.workBeff));

  // Legacy shadow arrays should remain empty in the accepted M2B/M3A path.
  PetscCall(doubles("momentum.legacy.empty_shadows",A.convection,A.supg,A.aPhys));

  // Category totals. Keep them non-overlapping so their slopes can be added.
  unsigned long long topology=0,matrix=0,diag=0,halo=0,fields=0,rhs=0,work=0;
  auto vr=[&](const auto& v){return vectorRetainedBytes(v);};
  topology=vr(A.rowPtr)+vr(A.colGid)+vr(A.colLocal)+vr(A.diagPos);
  matrix=vr(A.kNu)+vr(A.aRel);
  diag=vr(A.physDiag)+vr(A.relaxDelta)+vr(A.relaxedDiag)+vr(A.metric)+vr(A.rAU);
  halo=vr(A.ghostGid)+vr(A.reqRecvGid)+vr(A.reqRecvLocalOffset)+vr(A.reqSendCounts)+vr(A.reqSendDispls)+vr(A.reqRecvCounts)+vr(A.reqRecvDispls)+vr(A.exchangeSend)+vr(A.ghostValues)+reqr;
  for(int d=0;d<3;++d){fields+=vr(A.fieldOwned[d])+vr(A.fieldGhost[d]);rhs+=vr(A.convRhs[d])+vr(A.supgRhs[d]);}
  work=vr(A.workB)+vr(A.workX)+vr(A.workY)+vr(A.workBeff);
  const unsigned long long total=topology+matrix+diag+halo+fields+rhs+work+vr(A.convection)+vr(A.supg)+vr(A.aPhys);
  PetscCall(printMemoryAuditBytes("after_state_objects","momentum.CAT_topology",topology,topology,cells,"distributed_nonoverlap_total","rowPtr_colLocal_diagPos_colGid_released"));
  PetscCall(printMemoryAuditBytes("after_state_objects","momentum.CAT_matrix_values",matrix,matrix,cells,"distributed_nonoverlap_total","aRel_only_kNu_eliminated"));
  PetscCall(printMemoryAuditBytes("after_state_objects","momentum.CAT_diag_coupling",diag,diag,cells,"distributed_nonoverlap_total","must_preserve_physDiag_relaxedDiag_metric_rAU_semantics"));
  PetscCall(printMemoryAuditBytes("after_state_objects","momentum.CAT_halo",halo,halo,cells,"distributed_nonoverlap_total","peer_metadata_plus_exchange_numeric"));
  PetscCall(printMemoryAuditBytes("after_state_objects","momentum.CAT_fields",fields,fields,cells,"distributed_nonoverlap_total","three_owned_plus_three_ghost_velocity_buffers"));
  PetscCall(printMemoryAuditBytes("after_state_objects","momentum.CAT_dynamic_rhs",rhs,rhs,cells,"distributed_nonoverlap_total","convection3_plus_supg3"));
  PetscCall(printMemoryAuditBytes("after_state_objects","momentum.CAT_work",work,work,cells,"distributed_nonoverlap_total","four_FP64_SGS_residual_work_vectors"));
  PetscCall(printMemoryAuditBytes("after_state_objects","custom.momentumCSR_DETAILED_TOTAL",total,total,cells,"distributed_nonoverlap_total","sum_of_detailed_custom_momentum_components"));

  // M1+M2 are now actual allocations, not projections.  What remains removable
  // by the later M3 matrix-free step is active aRel plus rowPtr/colLocal/diagPos.
  const unsigned long long matfreeCore=diag+halo+fields+rhs+work;
  PetscCall(printMemoryAuditBytes("after_state_objects","momentum.ACTUAL_M1_colGid_released",vr(A.colGid),vr(A.colGid),cells,"actual_allocation","must_be_zero_after_setup"));
  PetscCall(printMemoryAuditBytes("after_state_objects","momentum.ACTUAL_M2_kNu_eliminated",vr(A.kNu),vr(A.kNu),cells,"actual_allocation","must_be_zero_persistent_static_diffusion_reintegrated"));
  PetscCall(printMemoryAuditBytes("after_state_objects","momentum.PROJECTION_M3_matfree_core_retained",matfreeCore,matfreeCore,cells,"accounting_projection_only","excludes_dynamicPlan; assumes_alternative_nonCSR_smoother"));
  const unsigned long long removableMatfree=(topology+matrix);
  PetscCall(printMemoryAuditBytes("after_state_objects","momentum.PROJECTION_M3_remaining_CSR_matrix_removable",removableMatfree,removableMatfree,cells,"accounting_projection_only","rowPtr_colLocal_diagPos_plus_aRel_after_M1M2"));

  unsigned long long lnnz=(unsigned long long)A.colLocal.size(),gnnz=0,lnowned=(unsigned long long)A.nOwned,gnowned=0,lghost=(unsigned long long)A.ghostGid.size(),gghost=0;
  PetscCallMPI(MPI_Allreduce(&lnnz,&gnnz,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&lnowned,&gnowned,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&lghost,&gghost,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_MOMENTUM_MEMORY_COUNTS globalOwnedVel=%llu globalCSRnnz=%llu avgNnzPerVelRow=%.6f aggregateGhostSlots=%llu sizeofPetscInt=%zu sizeofDouble=%zu sizeofMPIRequest=%zu\n",
    gnowned,gnnz,gnowned?(double)gnnz/(double)gnowned:0.0,gghost,sizeof(PetscInt),sizeof(double),sizeof(MPI_Request)));
  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode auditStateObjects(Mat C,Mat Sg,Mat Knu,Mat Aphys,Mat Ar,
                                        const Discrete& D,const GhostPlan& G,
                                        PetscInt cells) {
  PetscFunctionBeginUser;
  int rank=0; PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  if(C) PetscCall(printMemoryAuditMat("after_state_objects","C_convection",C,cells));
  if(Sg) PetscCall(printMemoryAuditMat("after_state_objects","Sg_supg",Sg,cells));
  if(Knu) PetscCall(printMemoryAuditMat("after_state_objects","Knu_static_diffusion",Knu,cells));
  if(Aphys) PetscCall(printMemoryAuditMat("after_state_objects","Aphys",Aphys,cells));
  if(Ar) PetscCall(printMemoryAuditMat("after_state_objects","Ar_relaxed",Ar,cells));
  const unsigned long long ghostLocal=(unsigned long long)G.ghosts.size();
  // Velocity-sized state/work vectors existing at this point.
  (void)ghostLocal;
  const unsigned long long pressureCount=2ull; // M6B: pcIn/pcOut only
  const unsigned long long localP=(unsigned long long)D.cellCount[rank];
  const unsigned long long localPayload=pressureCount*localP*sizeof(PetscScalar);
  PetscCall(printMemoryAuditBytes("after_state_objects","petsc.state_work_vec_numeric_payload",localPayload,localPayload,cells,"distributed_global_payload","M6B_pcIn_pcOut_only_excludes_PETSc_vec_headers"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode auditGAMGMemory(KSP pksp,PetscInt cells) {
  PetscFunctionBeginUser;
  if(!pksp) PetscFunctionReturn(PETSC_SUCCESS);
  PC pc=nullptr; const char *pct=nullptr;
  PetscCall(KSPGetPC(pksp,&pc)); PetscCall(PCGetType(pc,&pct));
  // M8 diagnostic-only generalization: PCMGGetLevels() is valid for GAMG/MG
  // but not for PCHYPRE/BoomerAMG.  Non-GAMG cases are still fully measured
  // by the common RSS/HWM resource marks; skip only the PETSc-MG internals audit.
  if(!pct || std::string(pct)!="gamg") {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_MEMORY_AUDIT_AMG_INTERNALS_SKIPPED pc=%s reason=non_GAMG use=RSS_HWM_resource_marks\n",pct?pct:"?"));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscInt levels=0;
  PetscCall(PCMGGetLevels(pc,&levels));
  double totalMatrixMemory=0.0,totalNnz=0.0,totalAllocated=0.0;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_MEMORY_AUDIT_GAMG_BEGIN pc=%s levels=%" PetscInt_FMT "\n",pct?pct:"?",levels));
  for(PetscInt lev=0;lev<levels;++lev) {
    KSP lksp=nullptr;
    if(lev==0) PetscCall(PCMGGetCoarseSolve(pc,&lksp));
    else PetscCall(PCMGGetSmoother(pc,lev,&lksp));
    Mat A=nullptr,Pm=nullptr; PetscCall(KSPGetOperators(lksp,&A,&Pm)); if(!Pm) Pm=A;
    if(Pm) {
      MatInfo i; PetscInt nr=0,nc=0; PetscCall(MatGetSize(Pm,&nr,&nc)); PetscCall(MatGetInfo(Pm,MAT_GLOBAL_SUM,&i));
      totalMatrixMemory+=i.memory; totalNnz+=i.nz_used; totalAllocated+=i.nz_allocated;
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "P1BF3_MEMORY_AUDIT_GAMG_LEVEL level=%" PetscInt_FMT " rows=%" PetscInt_FMT " cols=%" PetscInt_FMT
        " nzUsed=%.0f nzAllocated=%.0f allocOverUsed=%.6f memoryMiB=%.3f\n",
        lev,nr,nc,i.nz_used,i.nz_allocated,i.nz_used?i.nz_allocated/i.nz_used:0.0,i.memory/(1024.0*1024.0)));
    }
    if(lev>0) {
      Mat I=nullptr; PetscCall(PCMGGetInterpolation(pc,lev,&I));
      if(I) {
        MatInfo ii; PetscInt nr=0,nc=0; PetscCall(MatGetSize(I,&nr,&nc)); PetscCall(MatGetInfo(I,MAT_GLOBAL_SUM,&ii));
        totalMatrixMemory+=ii.memory; totalNnz+=ii.nz_used; totalAllocated+=ii.nz_allocated;
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
          "P1BF3_MEMORY_AUDIT_GAMG_TRANSFER fineLevel=%" PetscInt_FMT " rows=%" PetscInt_FMT " cols=%" PetscInt_FMT
          " nzUsed=%.0f nzAllocated=%.0f allocOverUsed=%.6f memoryMiB=%.3f\n",
          lev,nr,nc,ii.nz_used,ii.nz_allocated,ii.nz_used?ii.nz_allocated/ii.nz_used:0.0,ii.memory/(1024.0*1024.0)));
      }
    }
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_MEMORY_AUDIT_GAMG_TOTAL matrixAndTransferMemoryMiB=%.3f bytesPerCell=%.3f nzUsed=%.0f nzAllocated=%.0f allocOverUsed=%.6f note=excludes_KSP_PC_vectors_and_nonmatrix_metadata\n",
    totalMatrixMemory/(1024.0*1024.0),cells?totalMatrixMemory/(double)cells:0.0,totalNnz,totalAllocated,totalNnz?totalAllocated/totalNnz:0.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static int localBasisForEntity(const Mesh& M,PetscInt cell,PetscInt entity) {
  const PetscInt nv=(PetscInt)M.points.size();
  if(entity<nv) {
    for(int i=0;i<4;++i) if(M.tets[cell][i]==entity) return i;
  } else {
    const PetscInt f=entity-nv;
    for(int i=0;i<4;++i) if(M.oppFace[cell][i]==f) return 4+i;
  }
  return -1;
}

static PetscErrorCode buildPressureAssemblyPlan(const Mesh& M,const Discrete& D,int rank,const GhostPlan& G,const std::string& pPmatMode,PetscBool buildExpandedSchur,PressureAssemblyPlan& P) {
  PetscFunctionBeginUser;
  const PetscInt nv=(PetscInt)M.points.size(), ni=(PetscInt)M.neighbour.size(), nc=(PetscInt)M.tets.size();
  const PetscInt globalNc=distributedGlobalCellCount(D);
  // Only full/legacy surrogate modes need the global vertex-star support.
  // native_face is intentionally lean: its compact face graph is built directly
  // from the custom live B geometry and never stores vertexCells or Bcell.
  P.vertexCells.clear();
  if(pPmatMode!="native_face") {
    P.vertexCells.assign(nv,{});
    for(PetscInt c=0;c<nc;++c) for(int i=0;i<4;++i) P.vertexCells[M.tets[c][i]].push_back(c);
  }

  // Production full Pmat reconstructs B analytically; legacy surrogate modes keep the old B cache.
  P.Bcell.clear();
  if(pPmatMode!="full" && pPmatMode!="native_face") {
    P.Bcell.resize(nc);
    const double glref[4][3]={{-1,-1,-1},{1,0,0},{0,1,0},{0,0,1}};
    for(PetscInt c=0;c<nc;++c) {
      const auto t=M.tets[c];
      const Vec3 X[4]={M.points[t[0]],M.points[t[1]],M.points[t[2]],M.points[t[3]]};
      double J[3][3]={{X[1].x-X[0].x,X[2].x-X[0].x,X[3].x-X[0].x},
                      {X[1].y-X[0].y,X[2].y-X[0].y,X[3].y-X[0].y},
                      {X[1].z-X[0].z,X[2].z-X[0].z,X[3].z-X[0].z}},invJ[3][3];
      const double det=det3(J),vol=det/6.0; inv3(J,invJ); double gl[4][3]={{0}};
      for(int i=0;i<4;++i) for(int d=0;d<3;++d) for(int j=0;j<3;++j) gl[i][d]+=glref[i][j]*invJ[j][d];
      for(int i=0;i<4;++i) for(int d=0;d<3;++d) { P.Bcell[c][d*8+i]=vol*gl[i][d]; P.Bcell[c][d*8+4+i]=-(27.0/20.0)*vol*gl[i][d]; }
    }
  }

  // Precompute a compact one-ring LSQ geometry for an FV-like pressure
  // surrogate.  The LSQ normal coefficient is used only to define the
  // face-neighbour PRECONDITIONING graph; the exact FE Schur remains the KSP
  // operator.  Boundary faces enter the LSQ metric through reflected
  // centroid pseudo-neighbours so wall/outlet cells retain a full-rank 3-D
  // geometry metric without adding pressure unknowns.
  if(pPmatMode=="fv_lsq") {
    const auto fvCellC = cellCentroids(M);
    P.fvLsqGeom.assign(nc,{});
    P.fvTpfaGeom.assign(nc,{});
    P.fvNonorthCos.assign(nc,{});
    auto faceCentre = [&](PetscInt f)->Vec3 {
      Vec3 q{}; const auto& F=M.faces[f];
      for(PetscInt v:F.v) { const auto& x=M.points[v]; q.x+=x.x; q.y+=x.y; q.z+=x.z; }
      const double z=1.0/(double)F.v.size(); return {q.x*z,q.y*z,q.z*z};
    };
    auto faceAreaVec = [&](PetscInt f)->Vec3 {
      const auto& F=M.faces[f];
      if(F.v.size()!=3) return {};
      const Vec3& x0=M.points[F.v[0]]; const Vec3& x1=M.points[F.v[1]]; const Vec3& x2=M.points[F.v[2]];
      const Vec3 cr=cross3(sub3(x1,x0),sub3(x2,x0)); return {0.5*cr.x,0.5*cr.y,0.5*cr.z};
    };
    for(PetscInt c=0;c<nc;++c) {
      double H[3][3]={{0}};
      std::array<Vec3,4> dFace{};
      std::array<double,4> wFace{};
      for(int i=0;i<4;++i) {
        const PetscInt f=M.oppFace[c][i];
        Vec3 d{};
        if(f<ni) {
          const PetscInt L=(M.owner[f]==c)?M.neighbour[f]:M.owner[f];
          d=sub3(fvCellC[L],fvCellC[c]);
        } else {
          const Vec3 fc=faceCentre(f);
          const Vec3 h=sub3(fc,fvCellC[c]); d={2.0*h.x,2.0*h.y,2.0*h.z};
        }
        const double d2=d.x*d.x+d.y*d.y+d.z*d.z;
        const double w=1.0/std::max(d2,1e-30);
        dFace[i]=d; wFace[i]=w;
        const double a[3]={d.x,d.y,d.z};
        for(int r=0;r<3;++r) for(int q=0;q<3;++q) H[r][q]+=w*a[r]*a[q];
      }
      const double tr=H[0][0]+H[1][1]+H[2][2];
      const double reg=std::max(1e-12,1e-10*tr/3.0);
      for(int d=0;d<3;++d) H[d][d]+=reg;
      double HI[3][3]; inv3(H,HI);
      for(int i=0;i<4;++i) {
        const PetscInt f=M.oppFace[c][i];
        if(f>=ni) { P.fvLsqGeom[c][i]=0.0; P.fvTpfaGeom[c][i]=0.0; P.fvNonorthCos[c][i]=0.0; continue; }
        const Vec3 d=dFace[i]; const double w=wFace[i];
        double cv[3]={0,0,0},a[3]={d.x,d.y,d.z};
        for(int r=0;r<3;++r) for(int q=0;q<3;++q) cv[r]+=HI[r][q]*w*a[q];
        const Vec3 S=faceAreaVec(f); const double A=norm3(S), dl=norm3(d);
        const double Sdotc=S.x*cv[0]+S.y*cv[1]+S.z*cv[2];
        const double Sdotd=std::abs(S.x*d.x+S.y*d.y+S.z*d.z);
        const double coso=(A>0.0 && dl>0.0)?Sdotd/(A*dl):0.0;
        P.fvNonorthCos[c][i]=coso;
        P.fvLsqGeom[c][i]=std::abs(Sdotc);
        // Positive TPFA fallback with a 0.1 cosine floor: it cannot blow up on
        // nearly tangential centre-to-centre lines and is used only as a small
        // floor under the LSQ coefficient.
        const double denom=std::max(Sdotd,0.1*A*dl);
        P.fvTpfaGeom[c][i]=(denom>0.0)?A*A/denom:0.0;
      }
    }

  } else {
    P.fvLsqGeom.clear(); P.fvTpfaGeom.clear(); P.fvNonorthCos.clear();
  }

  // M5A: flat full-Schur cache. No retained Bcell or nested vectors in production.
  P.rows.clear(); P.flatRowGid.clear(); P.flatRowColOffset.clear(); P.flatColGid.clear();
  P.flatColTermOffset.clear(); P.flatTermLocalRau.clear(); P.flatTermCoeff.clear();
  size_t localColumns=0,localTerms=0;
  if(buildExpandedSchur) {
    if(pPmatMode=="full") {
      P.flatRowGid.reserve((std::size_t)D.cellCount[rank]); P.flatRowColOffset.reserve((std::size_t)D.cellCount[rank]+1);
      P.flatRowColOffset.push_back(0); P.flatColTermOffset.push_back(0);
      for(PetscInt K=0;K<nc;++K) if(D.cellOwner[K]==rank) {
        PetscInt entity[8]; for(int i=0;i<4;++i) entity[i]=M.tets[K][i]; for(int i=0;i<4;++i) entity[4+i]=nv+M.oppFace[K][i];
        double kVol=0.0,kGrad[4][3]={{0}}; fillCustomPressureGeom(M,K,kVol,kGrad);
        std::unordered_map<PetscInt,std::array<double,13>> geom; geom.reserve(96);
        auto getGeom=[&](PetscInt C)->const std::array<double,13>& {
          auto it=geom.find(C); if(it!=geom.end()) return it->second;
          double v=0.0,g[4][3]={{0}}; fillCustomPressureGeom(M,C,v,g); std::array<double,13> q{}; q[0]=v; int z=1;
          for(int i=0;i<4;++i) for(int d=0;d<3;++d) q[(std::size_t)z++]=g[i][d]; return geom.emplace(C,q).first->second;
        };
        std::map<PetscInt,std::vector<SchurTerm>> grouped;
        for(int a=0;a<8;++a) {
          const PetscInt gid=D.g2free[entity[a]]; if(gid<0) continue; const PetscInt li=velocityLocalIndex(G,gid);
          double BK[3]; for(int d=0;d<3;++d) BK[d]=customPressureBCoeff(kVol,kGrad,d,a);
          auto addSupport=[&](PetscInt L) {
            const int b=localBasisForEntity(M,L,entity[a]); if(b<0) throw std::runtime_error("M5A support/local-basis mismatch");
            const auto& q=getGeom(L); double lg[4][3]={{0}}; int z=1; for(int i=0;i<4;++i) for(int d=0;d<3;++d) lg[i][d]=q[(std::size_t)z++];
            double dot=0.0; for(int d=0;d<3;++d) dot+=BK[d]*customPressureBCoeff(q[0],lg,d,b); grouped[D.pGid[L]].push_back({li,dot});
          };
          if(entity[a]<nv) for(PetscInt L:P.vertexCells[entity[a]]) addSupport(L);
          else { const PetscInt f=entity[a]-nv; addSupport(M.owner[f]); if(f<ni) addSupport(M.neighbour[f]); }
        }
        P.flatRowGid.push_back(D.pGid[K]);
        for(auto& kv:grouped) { P.flatColGid.push_back(kv.first); for(const auto& t:kv.second){P.flatTermLocalRau.push_back(t.localRau);P.flatTermCoeff.push_back(t.coeff);} P.flatColTermOffset.push_back((PetscInt)P.flatTermLocalRau.size()); }
        P.flatRowColOffset.push_back((PetscInt)P.flatColGid.size());
      }
      localColumns=P.flatColGid.size(); localTerms=P.flatTermLocalRau.size();
    } else {
      P.rows.reserve(D.cellCount[rank]);
      for(PetscInt K=0;K<nc;++K) if(D.cellOwner[K]==rank) {
        PetscInt entity[8]; for(int i=0;i<4;++i) entity[i]=M.tets[K][i]; for(int i=0;i<4;++i) entity[4+i]=nv+M.oppFace[K][i];
        std::map<PetscInt,std::vector<SchurTerm>> grouped;
        for(int a=0;a<8;++a) { const PetscInt gid=D.g2free[entity[a]]; if(gid<0) continue; const PetscInt li=velocityLocalIndex(G,gid); double BK[3]; for(int d=0;d<3;++d) BK[d]=P.Bcell[K][d*8+a];
          auto add=[&](PetscInt L){const int b=localBasisForEntity(M,L,entity[a]);double dot=0;for(int d=0;d<3;++d)dot+=BK[d]*P.Bcell[L][d*8+b];grouped[D.pGid[L]].push_back({li,dot});};
          if(entity[a]<nv) for(PetscInt L:P.vertexCells[entity[a]]) add(L); else {const PetscInt f=entity[a]-nv;add(M.owner[f]);if(f<ni)add(M.neighbour[f]);}
        }
        SchurRowPlan rp; rp.row=D.pGid[K]; rp.columns.reserve(grouped.size()); for(auto& kv:grouped){SchurColumnPlan cp;cp.col=kv.first;cp.terms=std::move(kv.second);localTerms+=cp.terms.size();rp.columns.push_back(std::move(cp));} localColumns+=rp.columns.size();P.rows.push_back(std::move(rp));
      }
    }
  }

  // FULLFAST-FP64-HIST: vertexCells is setup-only for the full flat-Schur plan.
  // Once flatRow/Col/Term arrays are complete, release the nested star map.
  if(pPmatMode=="full") {
    std::vector<std::vector<PetscInt>>().swap(P.vertexCells);
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_FULLFAST_PLAN_COMPACT vertexCells=RELEASED flatTermCoeffBytes=%zu coeffStorage=float64 exactPhysicalSchur=FP64_factored\n",
      sizeof(double)));
  }

  const PetscInt nlp=D.cellCount[rank];

  // Build FV pressure-side support only for the fv_lsq surrogate.  Production
  // p_pmat=full does not need any FV geometry, face-neighbour diagonal ghosts,
  // or compact Pmat storage.
  P.fvCellDiagStart=0;
  P.fvCellDiagLocal.clear();
  if(pPmatMode=="fv_lsq") {
    for(int r=0;r<rank;++r) P.fvCellDiagStart += D.cellCount[r];
    std::vector<PetscInt> fvDiagGhosts;
    for(PetscInt K=0;K<nc;++K) if(D.cellOwner[K]==rank) {
      for(int i=0;i<4;++i) {
        const PetscInt f=M.oppFace[K][i];
        if(f>=ni) continue;
        const PetscInt L=(M.owner[f]==K)?M.neighbour[f]:M.owner[f];
        if(D.cellOwner[L]!=rank) fvDiagGhosts.push_back(D.pGid[L]);
      }
    }
    std::sort(fvDiagGhosts.begin(),fvDiagGhosts.end());
    fvDiagGhosts.erase(std::unique(fvDiagGhosts.begin(),fvDiagGhosts.end()),fvDiagGhosts.end());
    PetscCall(VecCreateGhost(PETSC_COMM_WORLD,nlp,globalNc,(PetscInt)fvDiagGhosts.size(),
                             fvDiagGhosts.empty()?nullptr:fvDiagGhosts.data(),&P.fvCellDiag));
    for(PetscInt i=0;i<nlp;++i) P.fvCellDiagLocal[P.fvCellDiagStart+i]=i;
    for(PetscInt j=0;j<(PetscInt)fvDiagGhosts.size();++j)
      P.fvCellDiagLocal[fvDiagGhosts[j]]=nlp+j;
  }

  // Exact full-Schur preallocation from the already-built row topology.
  // This replaces the legacy blanket 128/128 allocation that was 3.81x used NNZ.
  if(buildExpandedSchur) {
    PetscInt pStart=0; for(int r=0;r<rank;++r) pStart+=D.cellCount[r];
    const PetscInt pEnd=pStart+nlp;
    std::vector<PetscInt> dnnz((size_t)nlp,0),onnz((size_t)nlp,0);
    if(pPmatMode=="full") {
      for(std::size_t r=0;r<P.flatRowGid.size();++r) { const PetscInt li=P.flatRowGid[r]-pStart; if(li<0 || li>=nlp) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"flat Schur row ownership mismatch");
        for(PetscInt j=P.flatRowColOffset[r];j<P.flatRowColOffset[r+1];++j){const PetscInt c=P.flatColGid[(std::size_t)j];if(c>=pStart&&c<pEnd)++dnnz[(size_t)li];else++onnz[(size_t)li];} }
    } else {
      for(const auto& rp:P.rows) { const PetscInt li=rp.row-pStart; if(li<0 || li>=nlp) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Schur row ownership mismatch during exact preallocation"); for(const auto& cp:rp.columns){if(cp.col>=pStart&&cp.col<pEnd)++dnnz[(size_t)li];else++onnz[(size_t)li];} }
    }
    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD,nlp,nlp,globalNc,globalNc,0,dnnz.data(),0,onnz.data(),&P.S));
    PetscCall(MatSetOption(P.S,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));
    PetscCall(MatSetOption(P.S,MAT_SYMMETRIC,PETSC_TRUE));
    PetscInt ld=0,lo=0; for(PetscInt i=0;i<nlp;++i){ld+=dnnz[(size_t)i];lo+=onnz[(size_t)i];}
    PetscInt gd=0,go=0; PetscCallMPI(MPI_Allreduce(&ld,&gd,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD)); PetscCallMPI(MPI_Allreduce(&lo,&go,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_M3B_PMAT_PREALLOC mode=exact_from_schur_topology diagNnz=%" PetscInt_FMT " offdiagNnz=%" PetscInt_FMT " totalNnz=%" PetscInt_FMT " allocationError=ON\n",gd,go,gd+go));
  }
  if(pPmatMode!="full") {
    // Compact pressure preconditioning matrix.  native_face uses exact row
    // preallocation for diagonal + physical face neighbours; legacy modes keep
    // their historical 5/5 upper bound.
    if(pPmatMode=="native_face") {
      PetscInt pStart=0; for(int r=0;r<rank;++r) pStart+=D.cellCount[r];
      const PetscInt pEnd=pStart+nlp;
      std::vector<PetscInt> dnnz((std::size_t)nlp,1),onnz((std::size_t)nlp,0);
      for(PetscInt K=0;K<nc;++K) if(D.cellOwner[(std::size_t)K]==rank) {
        const PetscInt lr=D.pGid[(std::size_t)K]-pStart;
        for(int i=0;i<4;++i) {
          const PetscInt f=M.oppFace[(std::size_t)K][i]; if(f>=ni) continue;
          const PetscInt L=(M.owner[(std::size_t)f]==K)?M.neighbour[(std::size_t)f]:M.owner[(std::size_t)f];
          const PetscInt pg=D.pGid[(std::size_t)L]; if(pg>=pStart && pg<pEnd) ++dnnz[(std::size_t)lr]; else ++onnz[(std::size_t)lr];
        }
      }
      PetscCall(MatCreateAIJ(PETSC_COMM_WORLD,nlp,nlp,globalNc,globalNc,0,dnnz.data(),0,onnz.data(),&P.Pcompact));
      PetscCall(MatSetOption(P.Pcompact,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));
      PetscCall(MatSetOption(P.Pcompact,MAT_SYMMETRIC,PETSC_TRUE));
      PetscInt ld=0,lo=0; for(PetscInt i=0;i<nlp;++i){ld+=dnnz[(std::size_t)i];lo+=onnz[(std::size_t)i];}
      PetscInt gd=0,go=0; PetscCallMPI(MPI_Allreduce(&ld,&gd,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD)); PetscCallMPI(MPI_Allreduce(&lo,&go,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "P1BF3_B1_COMPACT_PREALLOC mode=native_face diagNnz=%" PetscInt_FMT " offdiagNnz=%" PetscInt_FMT " totalNnz=%" PetscInt_FMT " avgNnzPerRow=%.6f allocationError=ON graph=cell_plus_face_neighbours\n",
        gd,go,gd+go,(double)(gd+go)/(double)globalNc));
    } else {
      PetscCall(MatCreateAIJ(PETSC_COMM_WORLD,nlp,nlp,globalNc,globalNc,5,nullptr,5,nullptr,&P.Pcompact));
      PetscCall(MatSetOption(P.Pcompact,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));
      PetscCall(MatSetOption(P.Pcompact,MAT_SYMMETRIC,PETSC_TRUE));
    }
  }
  if(pPmatMode=="full") PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_M5A_FULL_PMAT_STORAGE fvGeometry=ELIMINATED fvCellDiag=ELIMINATED Pcompact=ELIMINATED physicalBcell=ELIMINATED nestedSchur=ELIMINATED flatSchur=ACTIVE\n"));
  if(buildExpandedSchur) {
    unsigned long long lc=(unsigned long long)localColumns,lt=(unsigned long long)localTerms,gc=0,gt=0;
    PetscCallMPI(MPI_Allreduce(&lc,&gc,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(&lt,&gt,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
    P.globalSchurNnz=(PetscInt)gc;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_SCHUR_PLAN cachedTopology=1 storage=%s pressureRows=%" PetscInt_FMT " uniqueRowColumns=%llu algebraicTerms=%llu update=batched_row_MatSetValues\n",
      pPmatMode=="full"?"flat_CSR_terms":"nested_legacy",globalNc,gc,gt));
  } else {
    P.globalSchurNnz=-1;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_SCHUR_PLAN cachedTopology=0 pressureRows=%" PetscInt_FMT " explicitExpandedSchur=NOT_MATERIALIZED operator=factored_B_rAU_Bt\n",globalNc));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode reindexFlatSchurRauToCustom(PressureAssemblyPlan& P,const GhostPlan& G,const CustomMomentumCSR& A) {
  PetscFunctionBeginUser;
  for(auto& li:P.flatTermLocalRau){PetscInt gid=-1;if(li<G.nOwned)gid=G.rstart+li;else{const PetscInt q=li-G.nOwned;if(q<0 || q>=(PetscInt)G.ghosts.size())SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M6B flat Schur legacy rAU index out of range");gid=G.ghosts[(std::size_t)q];}const PetscInt cli=customMomentumLocalIndex(A,gid);if(cli<0)SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M6B flat Schur rAU gid missing custom halo");li=cli;}
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode updatePressureSchurFullNative(CustomMomentumCSR& A,PetscBool assembleFullSchur,PressureAssemblyPlan& P) {
  PetscFunctionBeginUser;
  if(!assembleFullSchur) PetscFunctionReturn(PETSC_SUCCESS);
  if(!P.S) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M6B explicit full Schur requested without matrix");
  PetscCall(customMomentumExchange(A,A.rAU)); PetscCall(MatZeroEntries(P.S)); std::vector<PetscScalar> vals;
  auto rauAt=[&](PetscInt li)->double{return li<A.nOwned?A.rAU[(std::size_t)li]:A.ghostValues[(std::size_t)(li-A.nOwned)];};
  for(std::size_t r=0;r<P.flatRowGid.size();++r){const PetscInt c0=P.flatRowColOffset[r],c1=P.flatRowColOffset[r+1],n=c1-c0;vals.resize((std::size_t)n);for(PetscInt jj=0;jj<n;++jj){const PetscInt cj=c0+jj,t0=P.flatColTermOffset[(std::size_t)cj],t1=P.flatColTermOffset[(std::size_t)cj+1];double v=0;for(PetscInt t=t0;t<t1;++t)v+=rauAt(P.flatTermLocalRau[(std::size_t)t])*P.flatTermCoeff[(std::size_t)t];vals[(std::size_t)jj]=(PetscScalar)v;}if(n)PetscCall(MatSetValues(P.S,1,&P.flatRowGid[r],n,P.flatColGid.data()+c0,vals.data(),INSERT_VALUES));}
  PetscCall(MatAssemblyBegin(P.S,MAT_FINAL_ASSEMBLY));PetscCall(MatAssemblyEnd(P.S,MAT_FINAL_ASSEMBLY));PetscFunctionReturn(PETSC_SUCCESS);
}


// M11 B1+B2: lean compact FE-informed face-neighbour Pmat for PETSc GAMG.
// The exact outer PCG operator remains custom FP64 B diag(rAU) B^T.
// Pnative keeps the exact Schur diagonal, exact BF3 internal-face coupling,
// and conservatively redistributes P1 vertex diagonal energy onto physical
// face-neighbour edges.  No full Schur topology, Bcell cache, or vertexCells
// cache is retained.
static PetscErrorCode updatePressureCompactNative(const Mesh& M,const Discrete& D,int rank,
                                                   CustomPressureBPlan& B,const CustomMomentumCSR& A,
                                                   double p1Strength,PressureAssemblyPlan& P) {
  PetscFunctionBeginUser;
  if(!P.Pcompact) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M11 native compact Pmat requested without matrix");
  if(p1Strength<0.0 || p1Strength>1.0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"M11 native compact P1 strength must satisfy 0 <= strength <= 1");
  PetscCall(customPeerExchange(B.velocityHalo,A.rAU,48331));
  PetscCall(MatZeroEntries(P.Pcompact));
  const PetscInt nv=(PetscInt)M.points.size(),ni=(PetscInt)M.neighbour.size(),nc=(PetscInt)M.tets.size();
  PetscInt pStart=0; for(int r=0;r<rank;++r) pStart+=D.cellCount[(std::size_t)r];
  std::array<PetscInt,5> cols{}; std::array<PetscScalar,5> vals{};
  double localDiag=0.0,localP1Diag=0.0,localRedistributed=0.0,localBF3Abs=0.0;

  auto rAtLocal=[&](PetscInt li)->double {
    if(li<0) return 0.0;
    return customPeerValue(B.velocityHalo,A.rAU,li);
  };
  auto internalFaceDegreeAtVertex=[&](PetscInt C,int lv)->int {
    int deg=0; for(int i=0;i<4;++i) if(i!=lv && M.oppFace[(std::size_t)C][i]<ni) ++deg; return deg;
  };
  auto vertexDiagWithR=[&](PetscInt C,int lv,double rauV)->double {
    if(!(rauV>0.0)) return 0.0;
    double vol=0.0,g[4][3]={{0}}; fillCustomPressureGeom(M,C,vol,g);
    double b2=0.0; for(int d=0;d<3;++d){const double q=customPressureBCoeff(vol,g,d,lv);b2+=q*q;} return rauV*b2;
  };

  for(PetscInt K=0;K<nc;++K) if(D.cellOwner[(std::size_t)K]==rank) {
    const PetscInt pl=D.pGid[(std::size_t)K]-pStart;
    if(pl<0 || pl>=(PetscInt)B.forwardCells.size()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M11 pressure forward-cell indexing mismatch");
    const auto& cp=B.forwardCells[(std::size_t)pl];
    PetscInt ncol=1; cols[0]=D.pGid[(std::size_t)K]; vals[0]=0.0;
    double p1DiagK=0.0;
    // Exact full Schur diagonal from all local P1+BF3 basis functions.
    for(int a=0;a<8;++a) if(cp.velLocal[a]>=0) {
      double b2=0.0; for(int d=0;d<3;++d){const double q=customPressureBCoeff(cp.vol,cp.gradLambda,d,a);b2+=q*q;}
      const double da=rAtLocal(cp.velLocal[a])*b2; vals[0]+=(PetscScalar)da; if(a<4) p1DiagK+=da;
    }
    localDiag += PetscRealPart(vals[0]); localP1Diag += p1DiagK;

    for(int i=0;i<4;++i) {
      const PetscInt f=M.oppFace[(std::size_t)K][i]; if(f>=ni) continue;
      const PetscInt L=(M.owner[(std::size_t)f]==K)?M.neighbour[(std::size_t)f]:M.owner[(std::size_t)f];
      int j=-1; for(int q=0;q<4;++q) if(M.oppFace[(std::size_t)L][q]==f){j=q;break;}
      if(j<0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M11 neighbour local face not found");
      const PetscInt liFace=cp.velLocal[4+i];
      if(liFace<0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M11 internal BF3 face unexpectedly constrained/missing from velocity halo");
      const double rauF=rAtLocal(liFace);
      double volL=0.0,gL[4][3]={{0}}; fillCustomPressureGeom(M,L,volL,gL);
      double dot=0.0; for(int d=0;d<3;++d) dot += customPressureBCoeff(cp.vol,cp.gradLambda,d,4+i)*customPressureBCoeff(volL,gL,d,4+j);
      const double bf3off=rauF*dot; localBF3Abs += std::abs(bf3off);

      double qK=0.0,qL=0.0;
      if(p1Strength>0.0) {
        // Only the three vertices on this shared physical face participate.
        for(int lvK=0;lvK<4;++lvK) if(lvK!=i) {
          const PetscInt v=M.tets[(std::size_t)K][lvK];
          const PetscInt lvL=localBasisForEntity(M,L,v);
          if(lvL<0 || lvL>=4) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M11 shared face vertex missing in neighbour tet");
          const PetscInt liV=cp.velLocal[lvK];
          if(liV<0) continue; // constrained Dirichlet velocity vertex: zero pressure response
          const double rauV=rAtLocal(liV);
          const int degK=internalFaceDegreeAtVertex(K,lvK),degL=internalFaceDegreeAtVertex(L,lvL);
          if(degK>0) qK += vertexDiagWithR(K,lvK,rauV)/(double)degK;
          if(degL>0) qL += vertexDiagWithR(L,lvL,rauV)/(double)degL;
        }
      }
      const double w=p1Strength*std::max(0.0,std::min(qK,qL));
      localRedistributed += w;
      cols[ncol]=D.pGid[(std::size_t)L]; vals[ncol]=(PetscScalar)(bf3off-w); ++ncol;
    }
    PetscCall(MatSetValues(P.Pcompact,1,&cols[0],ncol,cols.data(),vals.data(),INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(P.Pcompact,MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(P.Pcompact,MAT_FINAL_ASSEMBLY));

  double gDiag=0.0,gP1=0.0,gRed=0.0,gBF3=0.0;
  PetscCallMPI(MPI_Allreduce(&localDiag,&gDiag,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localP1Diag,&gP1,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localRedistributed,&gRed,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localBF3Abs,&gBF3,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  MatInfo mi; PetscCall(MatGetInfo(P.Pcompact,MAT_GLOBAL_SUM,&mi));
  PetscBool sym=PETSC_FALSE; PetscCall(MatIsSymmetric(P.Pcompact,1e-12,&sym));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_B1_COMPACT_VALUES mode=native_face p1Strength=%.6g pmatNnz=%.0f pmatAllocated=%.0f avgNnzPerRow=%.6f exactDiagSum=%.12e p1DiagSum=%.12e redistributedDirectedDegree=%.12e redistributedFraction=%.6f bf3OffAbsRowSum=%.12e symmetric=%d exactOperator=custom_FP64_B_rAU_Bt compactRole=GAMG_preconditioner_only\n",
    p1Strength,mi.nz_used,mi.nz_allocated,mi.nz_used/(double)distributedGlobalCellCount(D),gDiag,gP1,gRed,gP1>0.0?gRed/gP1:0.0,gBF3,(int)sym));
  if(!sym) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M11 native compact pressure Pmat lost symmetry");
  PetscFunctionReturn(PETSC_SUCCESS);
}



// Gate 9G: compact FE/SIMPLE-energy face Laplacian.
// Keep exactly the native physical face-neighbour pressure graph, but choose
// each internal edge conductance from the actual B diag(rAU) B^T energy of a
// unit pressure jump across that face, restricted to velocity basis functions
// shared by the two adjacent tetrahedra:
//
//   a_f = 1/4 sum_{j shared(P,N)} rAU_j || B_Pj - B_Nj ||_2^2 .
//
// For q=[+1,-1] on the two cells, q^T [[a,-a],[-a,a]] q = 4 a, so the factor
// 1/4 matches the exact shared-DOF FE pressure-jump energy.  This is positive
// by construction, symmetric, uses the live SIMPLE/SIMPLEC mobility, and keeps
// only one owner-neighbour edge per physical face.  The outlet p=0 anchor is
// built on the same FE scale from velocity basis functions with nonzero trace
// on the outlet face.  Finally a single global trace match rescales the compact
// matrix to the exact Schur diagonal sum; this preserves relative face weights
// while removing a global scaling error that matters to Richardson/GAMG-only.
struct Gate9gFeFaceAudit {
  PetscReal coeffMin=PETSC_MAX_REAL,coeffMax=0.0,coeffMean=0.0;
  PetscReal anchorMin=PETSC_MAX_REAL,anchorMax=0.0,anchorMean=0.0;
  PetscReal exactDiagSum=0.0,compactDiagSumBeforeScale=0.0,traceScale=1.0;
  PetscReal exactSharedOffAbs=0.0,energyEdgeSum=0.0;
  PetscInt internalFaces=0,outletFaces=0,positiveExactPair=0,nonfinite=0;
};

static PetscErrorCode updatePressureCompactFeFaceEnergy(const Mesh& M,const Discrete& D,const ProblemConfig& Pcfg,
                                                         int rank,CustomPressureBPlan& B,const CustomMomentumCSR& A,
                                                         PressureAssemblyPlan& P,Gate9gFeFaceAudit& audit) {
  PetscFunctionBeginUser;
  if(!P.Pcompact) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate-9G FE-face Kp requested without compact Pmat");
  PetscCall(customPeerExchange(B.velocityHalo,A.rAU,59371));
  PetscCall(MatZeroEntries(P.Pcompact));
  PetscCall(MatSetOption(P.Pcompact,MAT_SYMMETRIC,PETSC_TRUE));
  const PetscInt nv=(PetscInt)M.points.size(),ni=(PetscInt)M.neighbour.size(),nc=(PetscInt)M.tets.size();
  PetscInt pStart=0; for(int r=0;r<rank;++r) pStart+=D.cellCount[(std::size_t)r];
  std::array<PetscInt,5> cols{}; std::array<PetscScalar,5> vals{};
  double localCoeffMin=PETSC_MAX_REAL,localCoeffMax=0.0,localCoeffSum=0.0;
  double localAnchorMin=PETSC_MAX_REAL,localAnchorMax=0.0,localAnchorSum=0.0;
  double localExactDiag=0.0,localCompactDiag=0.0,localExactOffAbs=0.0,localEnergyEdge=0.0;
  PetscInt localInternal=0,localOutlet=0,localPositivePair=0,localNonfinite=0;

  auto rAtLocal=[&](PetscInt li)->double { return li<0?0.0:customPeerValue(B.velocityHalo,A.rAU,li); };
  auto basisB2=[&](double vol,const double g[4][3],int a)->double {
    double q=0.0; for(int d=0;d<3;++d){const double b=customPressureBCoeff(vol,g,d,a);q+=b*b;} return q;
  };

  for(PetscInt K=0;K<nc;++K) if(D.cellOwner[(std::size_t)K]==rank) {
    const PetscInt pl=D.pGid[(std::size_t)K]-pStart;
    if(pl<0 || pl>=(PetscInt)B.forwardCells.size()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate-9G pressure forward-cell indexing mismatch");
    const auto& cp=B.forwardCells[(std::size_t)pl];
    PetscInt ncol=1; cols[0]=D.pGid[(std::size_t)K]; vals[0]=0.0;

    // Exact full Schur diagonal trace contribution for scaling/audit.
    double exactDiagK=0.0;
    for(int a=0;a<8;++a) if(cp.velLocal[a]>=0) exactDiagK += rAtLocal(cp.velLocal[a])*basisB2(cp.vol,cp.gradLambda,a);
    localExactDiag += exactDiagK;

    for(int i=0;i<4;++i) {
      const PetscInt f=M.oppFace[(std::size_t)K][i];
      if(f<ni) {
        const PetscInt L=(M.owner[(std::size_t)f]==K)?M.neighbour[(std::size_t)f]:M.owner[(std::size_t)f];
        int j=-1; for(int q=0;q<4;++q) if(M.oppFace[(std::size_t)L][q]==f){j=q;break;}
        if(j<0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate-9G neighbour local face not found");
        double volL=0.0,gL[4][3]={{0}}; fillCustomPressureGeom(M,L,volL,gL);
        double jumpEnergy=0.0,exactPair=0.0;

        // Three P1 vertices shared by the physical face.
        for(int lvK=0;lvK<4;++lvK) if(lvK!=i) {
          const PetscInt v=M.tets[(std::size_t)K][lvK];
          const PetscInt lvL=localBasisForEntity(M,L,v);
          if(lvL<0 || lvL>=4) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate-9G shared face vertex missing in neighbour tet");
          const PetscInt li=cp.velLocal[lvK];
          if(li<0) continue; // constrained velocity trace has no pressure response
          const double rr=rAtLocal(li);
          double d2=0.0,dot=0.0;
          for(int d=0;d<3;++d) {
            const double bK=customPressureBCoeff(cp.vol,cp.gradLambda,d,lvK);
            const double bL=customPressureBCoeff(volL,gL,d,lvL);
            const double db=bK-bL; d2+=db*db; dot+=bK*bL;
          }
          jumpEnergy += rr*d2; exactPair += rr*dot;
        }

        // Shared BF3 face bubble.
        const PetscInt liF=cp.velLocal[4+i];
        if(liF>=0) {
          const double rr=rAtLocal(liF); double d2=0.0,dot=0.0;
          for(int d=0;d<3;++d) {
            const double bK=customPressureBCoeff(cp.vol,cp.gradLambda,d,4+i);
            const double bL=customPressureBCoeff(volL,gL,d,4+j);
            const double db=bK-bL; d2+=db*db; dot+=bK*bL;
          }
          jumpEnergy += rr*d2; exactPair += rr*dot;
        }
        const double af=0.25*jumpEnergy;
        if(!std::isfinite(af) || af<0.0) {++localNonfinite; continue;}
        vals[0]+=(PetscScalar)af;
        cols[ncol]=D.pGid[(std::size_t)L]; vals[ncol]=(PetscScalar)(-af); ++ncol;
        localCoeffMin=std::min(localCoeffMin,af); localCoeffMax=std::max(localCoeffMax,af); localCoeffSum+=af;
        localExactOffAbs+=std::abs(exactPair); localEnergyEdge+=af; ++localInternal;
        if(exactPair>0.0) ++localPositivePair;
      } else {
        const int pi=M.facePatch[(std::size_t)f];
        if(pi==Pcfg.boundary.outlet) {
          // FE-scaled Dirichlet-to-zero edge.  Only P1 vertices on this face
          // and its BF3 trace bubble participate in the pressure boundary jump.
          double ab=0.0;
          for(int lv=0;lv<4;++lv) if(lv!=i && cp.velLocal[lv]>=0)
            ab += rAtLocal(cp.velLocal[lv])*basisB2(cp.vol,cp.gradLambda,lv);
          if(cp.velLocal[4+i]>=0) ab += rAtLocal(cp.velLocal[4+i])*basisB2(cp.vol,cp.gradLambda,4+i);
          if(!std::isfinite(ab) || ab<0.0) {++localNonfinite; ab=0.0;}
          vals[0]+=(PetscScalar)ab;
          localAnchorMin=std::min(localAnchorMin,ab); localAnchorMax=std::max(localAnchorMax,ab); localAnchorSum+=ab; ++localOutlet;
        } else if(pi==Pcfg.boundary.inlet || isWallPatch(Pcfg.boundary,pi)) {
          // homogeneous Neumann for the auxiliary pressure operator
        } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate-9G encountered unclassified pressure boundary face");
      }
    }
    localCompactDiag += PetscRealPart(vals[0]);
    PetscCall(MatSetValues(P.Pcompact,1,&cols[0],ncol,cols.data(),vals.data(),INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(P.Pcompact,MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(P.Pcompact,MAT_FINAL_ASSEMBLY));

  double gCoeffMin=0.0,gCoeffMax=0.0,gCoeffSum=0.0,gAnchorMin=0.0,gAnchorMax=0.0,gAnchorSum=0.0;
  double gExactDiag=0.0,gCompactDiag=0.0,gExactOffAbs=0.0,gEnergyEdge=0.0;
  PetscInt gInternal=0,gOutlet=0,gPositivePair=0,gNonfinite=0;
  PetscCallMPI(MPI_Allreduce(&localCoeffMin,&gCoeffMin,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localCoeffMax,&gCoeffMax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localCoeffSum,&gCoeffSum,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localAnchorMin,&gAnchorMin,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localAnchorMax,&gAnchorMax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localAnchorSum,&gAnchorSum,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localExactDiag,&gExactDiag,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localCompactDiag,&gCompactDiag,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localExactOffAbs,&gExactOffAbs,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localEnergyEdge,&gEnergyEdge,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localInternal,&gInternal,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localOutlet,&gOutlet,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localPositivePair,&gPositivePair,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localNonfinite,&gNonfinite,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));

  const double traceScale=(gCompactDiag>0.0)?gExactDiag/gCompactDiag:1.0;
  if(!(traceScale>0.0) || !std::isfinite(traceScale)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"Gate-9G invalid FE-face trace scale");
  PetscCall(MatScale(P.Pcompact,traceScale));
  MatInfo mi{}; PetscCall(MatGetInfo(P.Pcompact,MAT_GLOBAL_SUM,&mi));
  PetscBool sym=PETSC_FALSE; PetscCall(MatIsSymmetric(P.Pcompact,1e-12,&sym));
  if(!sym) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate-9G FE-face compact Kp lost symmetry");

  audit.coeffMin=(PetscReal)(gInternal?gCoeffMin*traceScale:0.0);
  audit.coeffMax=(PetscReal)(gCoeffMax*traceScale);
  audit.coeffMean=(PetscReal)(gInternal?gCoeffSum*traceScale/(double)gInternal:0.0);
  audit.anchorMin=(PetscReal)(gOutlet?gAnchorMin*traceScale:0.0);
  audit.anchorMax=(PetscReal)(gAnchorMax*traceScale);
  audit.anchorMean=(PetscReal)(gOutlet?gAnchorSum*traceScale/(double)gOutlet:0.0);
  audit.exactDiagSum=(PetscReal)gExactDiag; audit.compactDiagSumBeforeScale=(PetscReal)gCompactDiag; audit.traceScale=(PetscReal)traceScale;
  audit.exactSharedOffAbs=(PetscReal)gExactOffAbs; audit.energyEdgeSum=(PetscReal)(gEnergyEdge*traceScale);
  audit.internalFaces=gInternal/2; audit.outletFaces=gOutlet; audit.positiveExactPair=gPositivePair/2; audit.nonfinite=gNonfinite;

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_GATE9G_FE_FACE_KP_VALUES rows=%" PetscInt_FMT " nnz=%.0f avgNnzPerRow=%.6f internalFaces=%" PetscInt_FMT " outletFaces=%" PetscInt_FMT " coeffMin=%.12e coeffMean=%.12e coeffMax=%.12e anchorMin=%.12e anchorMean=%.12e anchorMax=%.12e exactDiagSum=%.12e compactDiagBeforeScale=%.12e traceScale=%.12e positiveExactSharedPair=%" PetscInt_FMT " exactSharedOffAbs=%.12e energyEdgeSumScaled=%.12e symmetric=%d nonfinite=%" PetscInt_FMT " formula=quarter_shared_FE_jump_energy graph=cell_plus_face_neighbours exactOperator=UNCHANGED_custom_FP64_B_rAU_Bt\n",
    distributedGlobalCellCount(D),mi.nz_used,mi.nz_used/(double)distributedGlobalCellCount(D),audit.internalFaces,gOutlet,(double)audit.coeffMin,(double)audit.coeffMean,(double)audit.coeffMax,
    (double)audit.anchorMin,(double)audit.anchorMean,(double)audit.anchorMax,(double)audit.exactDiagSum,(double)audit.compactDiagSumBeforeScale,
    (double)audit.traceScale,audit.positiveExactPair,(double)audit.exactSharedOffAbs,(double)audit.energyEdgeSum,(int)sym,audit.nonfinite));
  if(gNonfinite) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"Gate-9G FE-face Kp contained invalid conductances");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode updatePressureSchurFast(const Mesh& M,const Discrete& D,int rank,const GhostPlan& G,Vec rAU,const std::string& pPmatMode,double feFvP1Strength,double fvLsqStrength,double fvLsqTpfaFloor,PetscBool assembleFullSchur,PressureAssemblyPlan& P) {
  PetscFunctionBeginUser;
  PetscCall(VecGhostUpdateBegin(rAU,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecGhostUpdateEnd(rAU,INSERT_VALUES,SCATTER_FORWARD));
  Vec rloc=nullptr;
  const PetscScalar *ra=nullptr;
  PetscCall(VecGhostGetLocalForm(rAU,&rloc));
  PetscCall(VecGetArrayRead(rloc,&ra));

  // For the FV/LSQ surrogate, compute the exact Schur diagonal on owned cells
  // from the local FE rAU response, then exchange only face-neighbour pressure
  // diagonals.  This fixes the previous bug where exactCellSchurDiag(L) tried
  // to read all velocity DOFs of an off-rank neighbour L through G.
  const PetscScalar *fvDiagLocalArray=nullptr;
  Vec fvDiagLocalForm=nullptr;
  if(pPmatMode=="fv_lsq") {
    PetscScalar *da=nullptr;
    PetscCall(VecSet(P.fvCellDiag,0.0));
    PetscCall(VecGetArray(P.fvCellDiag,&da));
    for(PetscInt K=0;K<(PetscInt)M.tets.size();++K) if(D.cellOwner[K]==rank) {
      double diagK=0.0;
      for(int a=0;a<8;++a) {
        const PetscInt entity=(a<4)?M.tets[K][a]:((PetscInt)M.points.size()+M.oppFace[K][a-4]);
        const PetscInt gid=D.g2free[entity]; if(gid<0) continue;
        const PetscInt li=velocityLocalIndex(G,gid); // K is owned: guaranteed in G
        double b2=0.0;
        for(int d=0;d<3;++d) { const double b=P.Bcell[K][d*8+a]; b2+=b*b; }
        diagK += PetscRealPart(ra[li])*b2;
      }
      const PetscInt liP=D.pGid[K]-P.fvCellDiagStart;
      if(liP<0 || liP>=D.cellCount[rank])
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"owned pressure gid/layout mismatch in fv_lsq diagonal exchange");
      da[liP]=(PetscScalar)diagK;
    }
    PetscCall(VecRestoreArray(P.fvCellDiag,&da));
    PetscCall(VecGhostUpdateBegin(P.fvCellDiag,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecGhostUpdateEnd(P.fvCellDiag,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecGhostGetLocalForm(P.fvCellDiag,&fvDiagLocalForm));
    PetscCall(VecGetArrayRead(fvDiagLocalForm,&fvDiagLocalArray));
  }

  if(assembleFullSchur) {
    if(!P.S) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"explicit Schur assembly requested but expanded structure was not built");
    PetscCall(MatZeroEntries(P.S)); std::vector<PetscInt> cols; std::vector<PetscScalar> vals;
    if(pPmatMode=="full") {
      for(std::size_t r=0;r<P.flatRowGid.size();++r) { const PetscInt c0=P.flatRowColOffset[r],c1=P.flatRowColOffset[r+1],n=c1-c0; vals.resize((std::size_t)n);
        for(PetscInt jj=0;jj<n;++jj){const PetscInt cj=c0+jj,t0=P.flatColTermOffset[(std::size_t)cj],t1=P.flatColTermOffset[(std::size_t)cj+1];double v=0;for(PetscInt t=t0;t<t1;++t)v+=PetscRealPart(ra[P.flatTermLocalRau[(std::size_t)t]])*P.flatTermCoeff[(std::size_t)t];vals[(std::size_t)jj]=(PetscScalar)v;}
        if(n) PetscCall(MatSetValues(P.S,1,&P.flatRowGid[r],n,P.flatColGid.data()+c0,vals.data(),INSERT_VALUES)); }
    } else {
      for(const auto& rp:P.rows){const PetscInt n=(PetscInt)rp.columns.size();cols.resize(n);vals.resize(n);for(PetscInt j=0;j<n;++j){cols[j]=rp.columns[j].col;double v=0;for(const auto& term:rp.columns[j].terms)v+=PetscRealPart(ra[term.localRau])*term.coeff;vals[j]=(PetscScalar)v;}if(n)PetscCall(MatSetValues(P.S,1,&rp.row,n,cols.data(),vals.data(),INSERT_VALUES));}
    }
  }

  // Full Pmat mode does not use Pcompact at all.  In the factored-operator +
  // lagged-full-GAMG path this routine is called only on PC refresh steps, so
  // avoid rebuilding an otherwise unused 5-point surrogate.
  if(pPmatMode=="full") {
    PetscCall(VecRestoreArrayRead(rloc,&ra));
    PetscCall(VecGhostRestoreLocalForm(rAU,&rloc));
    if(assembleFullSchur) {
      PetscCall(MatAssemblyBegin(P.S,MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(P.S,MAT_FINAL_ASSEMBLY));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  // Build a compact GAMG P-matrix from the same rAU values.
  // compact_face: historical truncation: exact diagonal, BF3 face off-diagonal only.
  // fe_fv_face: independently constructed FE-informed FV-like surrogate.  The
  //   BF3 face contribution is retained exactly, while the P1 vertex diagonal
  //   energy is conservatively redistributed onto physical internal faces.
  //   For a cell K and one of its P1 vertices v,
  //       d_Kv = rAU_v |B_Kv|^2 .
  //   d_Kv is divided equally among K's internal faces that contain v.  On a
  //   shared face f=K|L, the symmetric P1 graph weight is
  //       w_f^P1 = strength * min(q_Kf,q_Lf) >= 0.
  //   Since sum_f q_Kf <= sum_v d_Kv, the redistributed P1 graph Laplacian
  //   cannot consume more than the exact P1 diagonal energy.  Any unmatched
  //   energy remains on the exact Schur diagonal.  Thus this is a compact,
  //   conservative SPD surrogate rather than a truncation of the dense P1
  //   vertex-star clique.
  PetscCall(MatZeroEntries(P.Pcompact));
  const PetscInt nv=(PetscInt)M.points.size(), ni=(PetscInt)M.neighbour.size(), nc=(PetscInt)M.tets.size();
  std::array<PetscInt,5> ccols{};
  std::array<PetscScalar,5> cvals{};
  double localP1Diag=0.0, localP1Degree=0.0, localBF3OffAbs=0.0;
  double localFvDiag=0.0,localFvDegree=0.0,localFvLsqGeom=0.0,localFvTpfaGeom=0.0,localFvCos=0.0;
  PetscInt localFvFaces=0; double localFvMinCos=1.0;

  auto vertexDiagContribution = [&](PetscInt C,int localVertex)->double {
    const PetscInt entity=M.tets[C][localVertex];
    const PetscInt gid=D.g2free[entity];
    if(gid<0) return 0.0;
    const PetscInt li=velocityLocalIndex(G,gid);
    double b2=0.0;
    for(int d=0;d<3;++d) { const double b=P.Bcell[C][d*8+localVertex]; b2+=b*b; }
    return PetscRealPart(ra[li])*b2;
  };
  auto internalFaceDegreeAtVertex = [&](PetscInt C,int localVertex)->int {
    int deg=0;
    // Local face i is opposite local vertex i, so it contains localVertex iff i!=localVertex.
    for(int i=0;i<4;++i) if(i!=localVertex && M.oppFace[C][i]<ni) ++deg;
    return deg;
  };
  auto p1DirectedFaceShare = [&](PetscInt C,PetscInt face)->double {
    int opp=-1;
    for(int i=0;i<4;++i) if(M.oppFace[C][i]==face) { opp=i; break; }
    if(opp<0) return 0.0;
    double q=0.0;
    for(int lv=0;lv<4;++lv) if(lv!=opp) {
      const int deg=internalFaceDegreeAtVertex(C,lv);
      if(deg>0) q += vertexDiagContribution(C,lv)/(double)deg;
    }
    return q;
  };

  auto exactCellSchurDiag = [&](PetscInt C)->double {
    if(pPmatMode=="fv_lsq") {
      const PetscInt pg=D.pGid[C];
      const auto it=P.fvCellDiagLocal.find(pg);
      if(it==P.fvCellDiagLocal.end())
        throw std::runtime_error("fv_lsq pressure diagonal ghost missing face-neighbour cell");
      return PetscRealPart(fvDiagLocalArray[it->second]);
    }
    double diagC=0.0;
    for(int a=0;a<8;++a) {
      const PetscInt entity=(a<4)?M.tets[C][a]:(nv+M.oppFace[C][a-4]);
      const PetscInt gid=D.g2free[entity]; if(gid<0) continue;
      const PetscInt li=velocityLocalIndex(G,gid);
      double b2=0.0; for(int d=0;d<3;++d) { const double b=P.Bcell[C][d*8+a]; b2+=b*b; }
      diagC += PetscRealPart(ra[li])*b2;
    }
    return diagC;
  };
  auto fvRawGeom = [&](PetscInt C,int lf)->double {
    return std::max(P.fvLsqGeom[C][lf],fvLsqTpfaFloor*P.fvTpfaGeom[C][lf]);
  };
  auto fvGeomDegree = [&](PetscInt C)->double {
    double z=0.0; for(int i=0;i<4;++i) if(M.oppFace[C][i]<ni) z+=fvRawGeom(C,i); return z;
  };
  auto fvFaceWeight = [&](PetscInt K,int i,PetscInt L,PetscInt f)->double {
    int j=-1; for(int q=0;q<4;++q) if(M.oppFace[L][q]==f) {j=q;break;}
    if(j<0) return 0.0;
    const double dK=exactCellSchurDiag(K), dL=exactCellSchurDiag(L);
    const double gK=fvGeomDegree(K), gL=fvGeomDegree(L);
    if(!(dK>0.0 && dL>0.0 && gK>0.0 && gL>0.0)) return 0.0;
    const double wK=fvLsqStrength*dK*fvRawGeom(K,i)/gK;
    const double wL=fvLsqStrength*dL*fvRawGeom(L,j)/gL;
    return std::max(0.0,std::min(wK,wL));
  };

  for(PetscInt K=0;K<nc;++K) if(D.cellOwner[K]==rank) {
    PetscInt ncol=1; ccols[0]=D.pGid[K]; cvals[0]=0.0;
    double p1DiagK=0.0;
    if(pPmatMode=="fv_lsq") {
      cvals[0]=(PetscScalar)exactCellSchurDiag(K);
      localFvDiag += PetscRealPart(cvals[0]);
    } else {
      // Exact full Schur diagonal: sum_a rAU_a |B_Ka|^2.
      for(int a=0;a<8;++a) {
        const PetscInt entity=(a<4)?M.tets[K][a]:(nv+M.oppFace[K][a-4]);
        const PetscInt gid=D.g2free[entity];
        if(gid<0) continue;
        const PetscInt li=velocityLocalIndex(G,gid);
        double b2=0.0; for(int d=0;d<3;++d) { const double b=P.Bcell[K][d*8+a]; b2+=b*b; }
        const double da=PetscRealPart(ra[li])*b2;
        cvals[0] += (PetscScalar)da;
        if(a<4) p1DiagK += da;
      }
      localP1Diag += p1DiagK;
    }

    // Face-neighbour off-diagonals: exact BF3 contribution, plus optional
    // conservative P1 vertex-energy redistribution for fe_fv_face.
    for(int i=0;i<4;++i) {
      const PetscInt f=M.oppFace[K][i];
      if(f>=ni) continue;
      const PetscInt entity=nv+f;
      const PetscInt gid=D.g2free[entity];
      if(gid<0) continue;
      const PetscInt L=(M.owner[f]==K)?M.neighbour[f]:M.owner[f];
      const int b=localBasisForEntity(M,L,entity);
      if(b<4) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"compact BF3 face/local-basis mismatch");
      const PetscInt li=velocityLocalIndex(G,gid);
      double dot=0.0;
      for(int d=0;d<3;++d) dot += P.Bcell[K][d*8+4+i]*P.Bcell[L][d*8+b];
      const double bf3off=PetscRealPart(ra[li])*dot;
      double off=bf3off;
      localBF3OffAbs += std::abs(bf3off);
      if(pPmatMode=="fe_fv_face" && feFvP1Strength>0.0) {
        const double qK=p1DirectedFaceShare(K,f);
        const double qL=p1DirectedFaceShare(L,f);
        const double w=feFvP1Strength*std::max(0.0,std::min(qK,qL));
        off -= w;                    // graph-Laplacian face coupling
        localP1Degree += w;          // row degree; each global face counts twice overall
      } else if(pPmatMode=="fv_lsq") {
        // A genuinely independent compact FV-like pressure operator: the
        // exact FE/BF3 off-diagonal is NOT used.  One-ring LSQ geometry sets
        // the relative face conductances; the FE rAU response enters through
        // exact-cell-Schur-diagonal normalization.  min(one-sided weights)
        // makes the assembled graph symmetric and guarantees row degree <=
        // fvLsqStrength*exact diagonal for 0<=strength<=1.
        const double w=fvFaceWeight(K,i,L,f);
        off=-w;
        localFvDegree+=w;
        localFvLsqGeom+=P.fvLsqGeom[K][i];
        localFvTpfaGeom+=P.fvTpfaGeom[K][i];
        localFvCos+=P.fvNonorthCos[K][i];
        localFvMinCos=std::min(localFvMinCos,P.fvNonorthCos[K][i]);
        ++localFvFaces;
      }
      ccols[ncol]=D.pGid[L]; cvals[ncol]=(PetscScalar)off; ++ncol;
    }
    PetscCall(MatSetValues(P.Pcompact,1,&ccols[0],ncol,ccols.data(),cvals.data(),INSERT_VALUES));
  }

  if(pPmatMode=="fe_fv_face") {
    static PetscInt reportCount=0;
    if(reportCount<3) {
      double gP1Diag=0.0,gP1Degree=0.0,gBF3OffAbs=0.0;
      PetscCallMPI(MPI_Allreduce(&localP1Diag,&gP1Diag,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
      PetscCallMPI(MPI_Allreduce(&localP1Degree,&gP1Degree,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
      PetscCallMPI(MPI_Allreduce(&localBF3OffAbs,&gBF3OffAbs,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "P1BF3_FE_FV_PMAT strength=%.6g p1DiagSum=%.12e redistributedRowDegree=%.12e redistributedFraction=%.6f bf3OffAbsRowSum=%.12e graph=cell_plus_face_neighbours\n",
        feFvP1Strength,gP1Diag,gP1Degree,gP1Diag>0.0?gP1Degree/gP1Diag:0.0,gBF3OffAbs));
      ++reportCount;
    }
  }

  if(pPmatMode=="fv_lsq") {
    static PetscInt fvReportCount=0;
    if(fvReportCount<3) {
      double gDiag=0.0,gDegree=0.0,gLsq=0.0,gTpfa=0.0,gCos=0.0,gMinCos=0.0;
      PetscInt gFaces=0;
      PetscCallMPI(MPI_Allreduce(&localFvDiag,&gDiag,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
      PetscCallMPI(MPI_Allreduce(&localFvDegree,&gDegree,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
      PetscCallMPI(MPI_Allreduce(&localFvLsqGeom,&gLsq,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
      PetscCallMPI(MPI_Allreduce(&localFvTpfaGeom,&gTpfa,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
      PetscCallMPI(MPI_Allreduce(&localFvCos,&gCos,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
      PetscCallMPI(MPI_Allreduce(&localFvFaces,&gFaces,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
      PetscCallMPI(MPI_Allreduce(&localFvMinCos,&gMinCos,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "P1BF3_LSQ_FV_PMAT strength=%.6g tpfaFloor=%.6g exactDiagSum=%.12e graphDegreeSum=%.12e degreeToDiag=%.6f directedFaces=%" PetscInt_FMT " meanLsqGeom=%.12e meanTpfaGeom=%.12e meanNonorthCos=%.6f minNonorthCos=%.6f graph=cell_plus_face_neighbours normalization=FE_exact_Schur_diag\n",
        fvLsqStrength,fvLsqTpfaFloor,gDiag,gDegree,gDiag>0.0?gDegree/gDiag:0.0,gFaces,
        gFaces?gLsq/(double)gFaces:0.0,gFaces?gTpfa/(double)gFaces:0.0,gFaces?gCos/(double)gFaces:0.0,gMinCos));
      ++fvReportCount;
    }
  }

  if(pPmatMode=="fv_lsq") {
    PetscCall(VecRestoreArrayRead(fvDiagLocalForm,&fvDiagLocalArray));
    PetscCall(VecGhostRestoreLocalForm(P.fvCellDiag,&fvDiagLocalForm));
  }
  PetscCall(VecRestoreArrayRead(rloc,&ra));
  PetscCall(VecGhostRestoreLocalForm(rAU,&rloc));
  if(assembleFullSchur) {
    PetscCall(MatAssemblyBegin(P.S,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(P.S,MAT_FINAL_ASSEMBLY));
  }
  PetscCall(MatAssemblyBegin(P.Pcompact,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P.Pcompact,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode destroyPressureAssemblyPlan(PressureAssemblyPlan& P) {
  PetscFunctionBeginUser;
  PetscCall(MatDestroy(&P.S));
  PetscCall(MatDestroy(&P.Pcompact));
  PetscCall(VecDestroy(&P.fvCellDiag));
  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode profileFillVector(Vec v);
static PetscErrorCode profileMaxSeconds(PetscLogDouble local,double *globalMax);

struct FactoredSchurContext {
  Mat B[3]={nullptr,nullptr,nullptr};
  Vec rAU=nullptr;
  Vec velocityWork=nullptr;
  Vec pressureWork=nullptr;
};

static PetscErrorCode factoredSchurMult(Mat A,Vec x,Vec y) {
  PetscFunctionBeginUser;
  FactoredSchurContext *C=nullptr;
  PetscCall(MatShellGetContext(A,&C));
  PetscCall(VecSet(y,0.0));
  for(int d=0;d<3;++d) {
    PetscCall(MatMultTranspose(C->B[d],x,C->velocityWork));
    PetscCall(VecPointwiseMult(C->velocityWork,C->velocityWork,C->rAU));
    PetscCall(MatMult(C->B[d],C->velocityWork,C->pressureWork));
    PetscCall(VecAXPY(y,1.0,C->pressureWork));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode createFactoredSchur(const Discrete& D,Vec rAU,Vec pressureTemplate,FactoredSchurContext& C,Mat *A) {
  PetscFunctionBeginUser;
  for(int d=0;d<3;++d) C.B[d]=D.B[d];
  C.rAU=rAU;
  PetscCall(VecDuplicate(D.rhs[0],&C.velocityWork));
  PetscCall(VecDuplicate(pressureTemplate,&C.pressureWork));
  PetscInt plocal=0,pglobal=0;
  PetscCall(VecGetLocalSize(pressureTemplate,&plocal));
  PetscCall(VecGetSize(pressureTemplate,&pglobal));
  PetscCall(MatCreateShell(PETSC_COMM_WORLD,plocal,plocal,pglobal,pglobal,&C,A));
  PetscCall(MatShellSetOperation(*A,MATOP_MULT,(void(*)(void))factoredSchurMult));
  PetscCall(MatSetOption(*A,MAT_SYMMETRIC,PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode destroyFactoredSchurContext(FactoredSchurContext& C) {
  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&C.velocityWork));
  PetscCall(VecDestroy(&C.pressureWork));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode profileOperatorMatMult(Mat A,Vec templateVec,PetscInt reps,double *secondsPerApply) {
  PetscFunctionBeginUser;
  Vec x=nullptr,y=nullptr;
  PetscCall(VecDuplicate(templateVec,&x));
  PetscCall(VecDuplicate(templateVec,&y));
  PetscCall(profileFillVector(x));
  for(int w=0;w<5;++w) PetscCall(MatMult(A,x,y));
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  PetscLogDouble t0,t1; PetscCall(PetscTime(&t0));
  for(PetscInt i=0;i<reps;++i) PetscCall(MatMult(A,x,y));
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  PetscCall(PetscTime(&t1));
  PetscCall(profileMaxSeconds((t1-t0)/(PetscLogDouble)reps,secondsPerApply));
  PetscCall(VecDestroy(&x)); PetscCall(VecDestroy(&y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode benchmarkFactoredSchur(Mat explicitS,Mat factoredS,const Discrete& D,Vec pressureTemplate,PetscInt expandedNnz,PetscInt reps) {
  PetscFunctionBeginUser;
  Vec x=nullptr,ye=nullptr,yf=nullptr,diff=nullptr;
  PetscCall(VecDuplicate(pressureTemplate,&x)); PetscCall(VecDuplicate(pressureTemplate,&ye));
  PetscCall(VecDuplicate(pressureTemplate,&yf)); PetscCall(VecDuplicate(pressureTemplate,&diff));
  PetscCall(profileFillVector(x));
  PetscCall(MatMult(explicitS,x,ye)); PetscCall(MatMult(factoredS,x,yf));
  PetscCall(VecWAXPY(diff,-1.0,ye,yf));
  PetscReal en=0.0,dn=0.0,di=0.0;
  PetscCall(VecNorm(ye,NORM_2,&en)); PetscCall(VecNorm(diff,NORM_2,&dn)); PetscCall(VecNorm(diff,NORM_INFINITY,&di));
  double explicitSec=0.0,factoredSec=0.0;
  PetscCall(profileOperatorMatMult(explicitS,pressureTemplate,reps,&explicitSec));
  PetscCall(profileOperatorMatMult(factoredS,pressureTemplate,reps,&factoredSec));
  double bNnz=0.0;
  for(int d=0;d<3;++d) if(D.B[d]) { MatInfo bi; PetscCall(MatGetInfo(D.B[d],MAT_GLOBAL_SUM,&bi)); bNnz+=bi.nz_used; }
  const double traversed=2.0*bNnz;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_FACTORED_SCHUR_CHECK relL2=%.12e absInf=%.12e explicitNorm=%.12e status=%s semantics=sum_d_Bd_rAU_BdT_no_extra_solve\n",
    (double)(en>0?dn/en:dn),(double)di,(double)en,(en>0?dn/en:dn)<5e-12?"PASS":"FAIL"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_FACTORED_SCHUR_PROFILE reps=%" PetscInt_FMT " explicitExpandedNnz=%" PetscInt_FMT " BxyzStoredNnz=%.0f factoredSparseEntriesTraversed=%.0f explicitMatMultMs=%.6f factoredMatMultMs=%.6f speedupExplicitOverFactored=%.6f factoredOverExplicit=%.6f\n",
    reps,expandedNnz,bNnz,traversed,1e3*explicitSec,1e3*factoredSec,explicitSec/factoredSec,factoredSec/explicitSec));
  PetscCall(VecDestroy(&x)); PetscCall(VecDestroy(&ye)); PetscCall(VecDestroy(&yf)); PetscCall(VecDestroy(&diff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode profileFillVector(Vec v) {
  PetscFunctionBeginUser;
  PetscInt lo=0,hi=0;
  PetscScalar *a=nullptr;
  PetscCall(VecGetOwnershipRange(v,&lo,&hi));
  PetscCall(VecGetArray(v,&a));
  for(PetscInt i=lo;i<hi;++i) {
    const double x=(double)(i+1);
    a[i-lo]=(PetscScalar)(std::sin(0.013*x)+0.37*std::cos(0.007*x));
  }
  PetscCall(VecRestoreArray(v,&a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode profileMaxSeconds(PetscLogDouble local,double *globalMax) {
  PetscFunctionBeginUser;
  double x=(double)local,mx=0.0;
  PetscCallMPI(MPI_Allreduce(&x,&mx,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  *globalMax=mx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode profileMatMult(Mat A,PetscInt reps,double *secondsPerApply) {
  PetscFunctionBeginUser;
  Vec x=nullptr,y=nullptr;
  PetscCall(MatCreateVecs(A,&x,&y));
  PetscCall(profileFillVector(x));
  for(int w=0;w<3;++w) PetscCall(MatMult(A,x,y));
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  PetscLogDouble t0,t1; PetscCall(PetscTime(&t0));
  for(PetscInt i=0;i<reps;++i) { PetscCall(MatMult(A,x,y)); std::swap(x,y); }
  PetscCall(PetscTime(&t1));
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  double mx=0; PetscCall(profileMaxSeconds(t1-t0,&mx));
  *secondsPerApply=(reps>0)?mx/(double)reps:0.0;
  PetscCall(VecDestroy(&x)); PetscCall(VecDestroy(&y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode profilePCApply(PC pc,Mat A,PetscInt reps,double *secondsPerApply) {
  PetscFunctionBeginUser;
  Vec x=nullptr,y=nullptr;
  PetscCall(MatCreateVecs(A,&x,&y));
  PetscCall(profileFillVector(x));
  for(int w=0;w<3;++w) PetscCall(PCApply(pc,x,y));
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  PetscLogDouble t0,t1; PetscCall(PetscTime(&t0));
  for(PetscInt i=0;i<reps;++i) { PetscCall(PCApply(pc,x,y)); std::swap(x,y); }
  PetscCall(PetscTime(&t1));
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  double mx=0; PetscCall(profileMaxSeconds(t1-t0,&mx));
  *secondsPerApply=(reps>0)?mx/(double)reps:0.0;
  PetscCall(VecDestroy(&x)); PetscCall(VecDestroy(&y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode profileKSPSolve(KSP ksp,Mat A,PetscInt reps,double *secondsPerSolve) {
  PetscFunctionBeginUser;
  Vec b=nullptr,x=nullptr;
  PetscCall(MatCreateVecs(A,&x,&b));
  PetscCall(profileFillVector(b));
  for(int w=0;w<2;++w) { PetscCall(VecSet(x,0)); PetscCall(KSPSetInitialGuessNonzero(ksp,PETSC_FALSE)); PetscCall(KSPSolve(ksp,b,x)); }
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  PetscLogDouble t0,t1; PetscCall(PetscTime(&t0));
  for(PetscInt i=0;i<reps;++i) { PetscCall(VecSet(x,0)); PetscCall(KSPSetInitialGuessNonzero(ksp,PETSC_FALSE)); PetscCall(KSPSolve(ksp,b,x)); }
  PetscCall(PetscTime(&t1));
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  double mx=0; PetscCall(profileMaxSeconds(t1-t0,&mx));
  *secondsPerSolve=(reps>0)?mx/(double)reps:0.0;
  PetscCall(VecDestroy(&b)); PetscCall(VecDestroy(&x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode pressureAMGProfile(KSP pksp,Mat S,Mat PmatFine,PetscInt fvCompactNnz,
                                          PetscInt fineReps,PetscInt pcReps,
                                          PetscInt cgIters,PetscInt cgReps,
                                          PetscInt levelMatReps,PetscInt levelSolveReps) {
  PetscFunctionBeginUser;
  PC pc=nullptr; const char *pct=nullptr,*kspt=nullptr;
  PetscCall(KSPGetPC(pksp,&pc)); PetscCall(PCGetType(pc,&pct)); PetscCall(KSPGetType(pksp,&kspt));
  PetscInt nFine=0,mFine=0,nlocFine=0; MatInfo finfo;
  PetscCall(MatGetSize(S,&nFine,&mFine)); PetscCall(MatGetLocalSize(S,&nlocFine,nullptr));
  PetscCall(MatGetInfo(S,MAT_GLOBAL_SUM,&finfo));
  MatInfo pminfo; PetscCall(MatGetInfo(PmatFine,MAT_GLOBAL_SUM,&pminfo));
  const double fineNnz=finfo.nz_used, pmatFineNnz=pminfo.nz_used;
  const double fvNnz=(double)fvCompactNnz;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_PRESSURE_PROFILE_FINE ksp=%s pc=%s rows=%" PetscInt_FMT " nnz=%.0f avgNnzPerRow=%.6f fvCompactFaceNnz=%" PetscInt_FMT " pressureNnzRatioToCompactFV=%.6f\n",
    kspt?kspt:"?",pct?pct:"?",nFine,fineNnz,nFine?fineNnz/(double)nFine:0.0,fvCompactNnz,fvNnz>0?fineNnz/fvNnz:0.0));

  PetscInt levels=0; PetscReal gc=0,oc=0;
  PetscCall(PCMGGetLevels(pc,&levels));
  PetscCall(PCMGGetGridComplexity(pc,&gc,&oc));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_PRESSURE_PROFILE_COMPLEXITY levels=%" PetscInt_FMT " gridComplexity=%.8f operatorComplexity=%.8f hierarchyFinePmatNnz=%.0f hierarchyTotalNnz=%.0f\n",
    levels,(double)gc,(double)oc,pmatFineNnz,(double)oc*pmatFineNnz));

  double fineMatSec=0; PetscCall(profileMatMult(S,fineReps,&fineMatSec));
  double pcSec=0; PetscCall(profilePCApply(pc,PmatFine,pcReps,&pcSec));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_PRESSURE_PROFILE_KERNEL fineMatMultReps=%" PetscInt_FMT " fineMatMultMs=%.6f fineMatMultNsPerNnz=%.6f pcApplyReps=%" PetscInt_FMT " gamgPcApplyMs=%.6f pcApplyOverFineMatMult=%.6f\n",
    fineReps,1e3*fineMatSec,fineNnz>0?1e9*fineMatSec/fineNnz:0.0,pcReps,1e3*pcSec,fineMatSec>0?pcSec/fineMatSec:0.0));

  double sumLevelNnz=0.0;
  for(PetscInt lev=0;lev<levels;++lev) {
    KSP lksp=nullptr;
    if(lev==0) PetscCall(PCMGGetCoarseSolve(pc,&lksp));
    else PetscCall(PCMGGetSmoother(pc,lev,&lksp));
    Mat A=nullptr,Pmat=nullptr;
    PetscCall(KSPGetOperators(lksp,&A,&Pmat));
    if(!Pmat) Pmat=A;
    PetscInt N=0,nloc=0,maxit=0; PetscReal rtol=0,atol=0,dtol=0; MatInfo info;
    PetscCall(MatGetSize(Pmat,&N,nullptr)); PetscCall(MatGetLocalSize(Pmat,&nloc,nullptr));
    PetscCall(MatGetInfo(Pmat,MAT_GLOBAL_SUM,&info)); sumLevelNnz += info.nz_used;
    PetscInt nlocMin=0,nlocMax=0;
    PetscCallMPI(MPI_Allreduce(&nloc,&nlocMin,1,MPIU_INT,MPI_MIN,PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(&nloc,&nlocMax,1,MPIU_INT,MPI_MAX,PETSC_COMM_WORLD));
    const char *lkt=nullptr,*lpct=nullptr; PC lpc=nullptr;
    PetscCall(KSPGetType(lksp,&lkt)); PetscCall(KSPGetPC(lksp,&lpc)); PetscCall(PCGetType(lpc,&lpct));
    PetscCall(KSPGetTolerances(lksp,&rtol,&atol,&dtol,&maxit));
    double mm=0,ss=0; PetscCall(profileMatMult(Pmat,levelMatReps,&mm));
    PetscCall(profileKSPSolve(lksp,Pmat,levelSolveReps,&ss));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_PRESSURE_PROFILE_LEVEL level=%" PetscInt_FMT " role=%s rows=%" PetscInt_FMT " localRowsMin=%" PetscInt_FMT " localRowsMax=%" PetscInt_FMT " nnz=%.0f avgNnzPerRow=%.6f ksp=%s pc=%s smootherMaxIt=%" PetscInt_FMT " matMultMs=%.6f levelSolveMs=%.6f\n",
      lev,lev==0?"coarse":"smooth",N,nlocMin,nlocMax,info.nz_used,N?info.nz_used/(double)N:0.0,lkt?lkt:"?",lpct?lpct:"?",maxit,1e3*mm,1e3*ss));
    if(lev>0) {
      Mat I=nullptr; PetscCall(PCMGGetInterpolation(pc,lev,&I));
      if(I) { MatInfo ii; PetscInt ir=0,ic=0; PetscCall(MatGetInfo(I,MAT_GLOBAL_SUM,&ii)); PetscCall(MatGetSize(I,&ir,&ic));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
          "P1BF3_PRESSURE_PROFILE_TRANSFER fineLevel=%" PetscInt_FMT " rows=%" PetscInt_FMT " cols=%" PetscInt_FMT " nnz=%.0f avgNnzPerRow=%.6f\n",
          lev,ir,ic,ii.nz_used,ir?ii.nz_used/(double)ir:0.0)); }
    }
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_PRESSURE_PROFILE_COMPLEXITY_CHECK summedLevelNnz=%.0f directOperatorComplexity=%.8f petscOperatorComplexity=%.8f\n",
    sumLevelNnz,pmatFineNnz>0?sumLevelNnz/pmatFineNnz:0.0,(double)oc));

  // Fixed-count CG using exactly the already-built/frozen GAMG PC.  This
  // measures fine SpMV + vector reductions + one PCApply per Krylov step.
  KSP probe=nullptr; PetscCall(KSPCreate(PETSC_COMM_WORLD,&probe));
  PetscCall(KSPSetOperators(probe,S,PmatFine)); PetscCall(KSPSetType(probe,KSPCG));
  PetscCall(KSPSetPC(probe,pc)); PetscCall(KSPSetReusePreconditioner(probe,PETSC_TRUE));
  PetscCall(KSPSetNormType(probe,KSP_NORM_NONE));
  PetscCall(KSPSetTolerances(probe,PETSC_CURRENT,PETSC_CURRENT,PETSC_CURRENT,cgIters));
  PetscCall(KSPSetConvergenceTest(probe,KSPConvergedSkip,nullptr,nullptr));
  PetscCall(KSPSetUp(probe));
  double cgSec=0; PetscCall(profileKSPSolve(probe,S,cgReps,&cgSec));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_PRESSURE_PROFILE_CG fixedIts=%" PetscInt_FMT " reps=%" PetscInt_FMT " solveMs=%.6f msPerCgIteration=%.6f impliedPcPlusKrylovOverheadMs=%.6f\n",
    cgIters,cgReps,1e3*cgSec,cgIters>0?1e3*cgSec/(double)cgIters:0.0,cgIters>0?1e3*cgSec/(double)cgIters-1e3*pcSec-1e3*fineMatSec:0.0));
  PetscCall(KSPDestroy(&probe));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_PRESSURE_PROFILE_DONE status=PASS\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

