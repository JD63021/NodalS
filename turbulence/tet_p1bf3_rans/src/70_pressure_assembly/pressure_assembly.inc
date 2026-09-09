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
  topology+=(unsigned long long)A.wallOwned.capacity()*sizeof(char);
  for(int d=0;d<3;++d){fields+=vr(A.fieldOwned[d])+vr(A.fieldGhost[d]);rhs+=vr(A.convRhs[d])+vr(A.supgRhs[d])+vr(A.mixlenRhs[d])+vr(A.wallRhs[d])+vr(A.inletRhs[d]);}
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

static PetscErrorCode buildPressureAssemblyPlan(const Mesh& M,const Discrete& D,const ProblemConfig& Pcfg,int rank,const GhostPlan& G,const std::string& pPmatMode,PetscBool buildExpandedSchur,PressureAssemblyPlan& P) {
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
      for(int i=0;i<4;++i) for(int d=0;d<3;++d) {
        P.Bcell[c][d*8+i]=customPressureEffectiveBForCell(M,D,Pcfg,c,vol,gl,d,i);
        P.Bcell[c][d*8+4+i]=customPressureEffectiveBForCell(M,D,Pcfg,c,vol,gl,d,4+i);
      }
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
          double BK[3]; for(int d=0;d<3;++d) BK[d]=customPressureEffectiveBForCell(M,D,Pcfg,K,kVol,kGrad,d,a);
          auto addSupport=[&](PetscInt L) {
            const int b=localBasisForEntity(M,L,entity[a]); if(b<0) throw std::runtime_error("M5A support/local-basis mismatch");
            const auto& q=getGeom(L); double lg[4][3]={{0}}; int z=1; for(int i=0;i<4;++i) for(int d=0;d<3;++d) lg[i][d]=q[(std::size_t)z++];
            double dot=0.0;
            for(int d=0;d<3;++d) dot+=BK[d]*customPressureEffectiveBForCell(M,D,Pcfg,L,q[0],lg,d,b);
            grouped[D.pGid[L]].push_back({li,dot});
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

  D.velCount.assign(size,0);
  if(D.g2free.size()!=(std::size_t)(nv+nf)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"compact Dirichlet gid map size mismatch");
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


// NodalS-1.00a distributed-memory Gate 0.
// This gate does NOT change the production solve path yet.  It constructs the
// exact rank-local cell support required by the current algebra from the frozen
// global indexing, packs compact local cell/face/point connectivity, and checks
// global topology/DOF/boundary/inlet-flux parity.  The full mesh is deliberately
// still replicated in Gate 0 so the distributed construction can be validated
// before root-only load/scatter is enabled in a later gate.
struct DistGate0LocalMesh {
  std::vector<PetscInt> cellGlobal, faceGlobal, pointGlobal;
  std::vector<std::array<int,4>> tets;
  std::vector<Face> faces;
  std::vector<std::array<int,4>> oppFace;
  PetscInt ownedCells=0, haloCells=0, velocityStarCells=0, pressureNeighbourCells=0;
};

static PetscErrorCode buildDistGate0LocalMesh(const Mesh& M,const Discrete& D,int rank,DistGate0LocalMesh& L) {
  PetscFunctionBeginUser;
  const PetscInt nv=(PetscInt)M.points.size(), nf=(PetscInt)M.faces.size(), ni=(PetscInt)M.neighbour.size(), nc=(PetscInt)M.tets.size();
  PetscInt vStart=0; for(int r=0;r<rank;++r) vStart+=D.velCount[(std::size_t)r];
  const PetscInt vEnd=vStart+D.velCount[(std::size_t)rank];
  std::vector<unsigned char> support((std::size_t)nc,0), fromVel((std::size_t)nc,0), fromPnbr((std::size_t)nc,0);

  // Pressure ownership support.
  for(PetscInt c=0;c<nc;++c) if(D.cellOwner[(std::size_t)c]==rank) support[(std::size_t)c]=1;

  // Momentum/B^T support: every cell touching an owned P1 or BF3 velocity DOF.
  for(PetscInt c=0;c<nc;++c) {
    bool touches=false;
    const auto& t=M.tets[(std::size_t)c];
    for(int i=0;i<4 && !touches;++i) {
      const PetscInt g=D.g2free[(std::size_t)t[(std::size_t)i]];
      if(g>=vStart && g<vEnd) touches=true;
    }
    for(int i=0;i<4 && !touches;++i) {
      const PetscInt e=nv+M.oppFace[(std::size_t)c][(std::size_t)i];
      const PetscInt g=D.g2free[(std::size_t)e];
      if(g>=vStart && g<vEnd) touches=true;
    }
    if(touches) { support[(std::size_t)c]=1; fromVel[(std::size_t)c]=1; }
  }

  // Compact pressure face-neighbour support for owned pressure cells.
  for(PetscInt c=0;c<nc;++c) if(D.cellOwner[(std::size_t)c]==rank) {
    for(int lf=0;lf<4;++lf) {
      const PetscInt f=M.oppFace[(std::size_t)c][(std::size_t)lf];
      if(f>=ni) continue;
      const PetscInt a=M.owner[(std::size_t)f], b=M.neighbour[(std::size_t)f];
      const PetscInt other=(a==c)?b:a;
      if(other>=0 && other<nc && D.cellOwner[(std::size_t)other]!=rank) {
        support[(std::size_t)other]=1; fromPnbr[(std::size_t)other]=1;
      }
    }
  }

  for(PetscInt c=0;c<nc;++c) if(support[(std::size_t)c]) {
    L.cellGlobal.push_back(c);
    if(D.cellOwner[(std::size_t)c]==rank) ++L.ownedCells; else ++L.haloCells;
    if(fromVel[(std::size_t)c]) ++L.velocityStarCells;
    if(fromPnbr[(std::size_t)c]) ++L.pressureNeighbourCells;
  }

  std::vector<int> pLut((std::size_t)nv,-1), fLut((std::size_t)nf,-1);
  std::vector<unsigned char> pMark((std::size_t)nv,0), fMark((std::size_t)nf,0);
  for(PetscInt gc:L.cellGlobal) {
    const auto& t=M.tets[(std::size_t)gc];
    for(int i=0;i<4;++i) pMark[(std::size_t)t[(std::size_t)i]]=1;
    for(int i=0;i<4;++i) fMark[(std::size_t)M.oppFace[(std::size_t)gc][(std::size_t)i]]=1;
  }
  for(PetscInt gp=0;gp<nv;++gp) if(pMark[(std::size_t)gp]) { pLut[(std::size_t)gp]=(int)L.pointGlobal.size(); L.pointGlobal.push_back(gp); }
  for(PetscInt gf=0;gf<nf;++gf) if(fMark[(std::size_t)gf]) { fLut[(std::size_t)gf]=(int)L.faceGlobal.size(); L.faceGlobal.push_back(gf); }

  L.tets.resize(L.cellGlobal.size()); L.oppFace.resize(L.cellGlobal.size());
  for(std::size_t lc=0;lc<L.cellGlobal.size();++lc) {
    const PetscInt gc=L.cellGlobal[lc];
    for(int i=0;i<4;++i) {
      const PetscInt gp=M.tets[(std::size_t)gc][(std::size_t)i];
      const PetscInt gf=M.oppFace[(std::size_t)gc][(std::size_t)i];
      if(pLut[(std::size_t)gp]<0 || fLut[(std::size_t)gf]<0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate0 local connectivity map incomplete");
      L.tets[lc][(std::size_t)i]=pLut[(std::size_t)gp];
      L.oppFace[lc][(std::size_t)i]=fLut[(std::size_t)gf];
    }
  }
  L.faces.resize(L.faceGlobal.size());
  for(std::size_t lf=0;lf<L.faceGlobal.size();++lf) {
    const PetscInt gf=L.faceGlobal[lf];
    for(int i=0;i<3;++i) {
      const PetscInt gp=M.faces[(std::size_t)gf].v[(std::size_t)i];
      if(pLut[(std::size_t)gp]<0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate0 local face references point outside support");
      L.faces[lf].v[(std::size_t)i]=pLut[(std::size_t)gp];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode runDistributedMeshGate0(const Mesh& M,const ProblemConfig& P,int rank,int size) {
  PetscFunctionBeginUser;
  const PetscInt nv=(PetscInt)M.points.size(), nf=(PetscInt)M.faces.size(), ni=(PetscInt)M.neighbour.size(), nc=(PetscInt)M.tets.size();
  Discrete D;
  PetscCall(buildOwnership(M,rank,size,P,D));
  DistGate0LocalMesh L;
  PetscCall(buildDistGate0LocalMesh(M,D,rank,L));

  // Storage ownership is independent of velocity Dirichlet status: each global
  // point/face is assigned to the minimum adjacent cell partition so unique
  // distributed coverage can be checked exactly.
  std::vector<int> pointStorageOwner((std::size_t)nv,-1), faceStorageOwner((std::size_t)nf,-1);
  for(PetscInt c=0;c<nc;++c) {
    const int r=D.cellOwner[(std::size_t)c];
    for(int i=0;i<4;++i) {
      const PetscInt gp=M.tets[(std::size_t)c][(std::size_t)i];
      int& o=pointStorageOwner[(std::size_t)gp]; o=(o<0)?r:std::min(o,r);
    }
  }
  for(PetscInt f=0;f<nf;++f) {
    const int r0=D.cellOwner[(std::size_t)M.owner[(std::size_t)f]];
    faceStorageOwner[(std::size_t)f]=(f<ni)?std::min(r0,D.cellOwner[(std::size_t)M.neighbour[(std::size_t)f]]):r0;
  }

  PetscInt localOwnedPoints=0,localOwnedFaces=0;
  for(PetscInt gp:L.pointGlobal) if(pointStorageOwner[(std::size_t)gp]==rank) ++localOwnedPoints;
  for(PetscInt gf:L.faceGlobal) if(faceStorageOwner[(std::size_t)gf]==rank) ++localOwnedFaces;

  PetscInt local[7]={L.ownedCells,localOwnedPoints,localOwnedFaces,(PetscInt)L.cellGlobal.size(),(PetscInt)L.pointGlobal.size(),(PetscInt)L.faceGlobal.size(),L.haloCells};
  PetscInt global[7]={0,0,0,0,0,0,0};
  PetscCallMPI(MPI_Allreduce(local,global,7,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  PetscInt pressureDofs=0,velocityDofs=0;
  PetscCallMPI(MPI_Allreduce(&D.cellCount[(std::size_t)rank],&pressureDofs,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&D.velCount[(std::size_t)rank],&velocityDofs,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));

  // Connectivity parity of the compact local pack against the frozen global mesh.
  PetscInt localMismatch=0;
  for(std::size_t lc=0;lc<L.cellGlobal.size();++lc) {
    const PetscInt gc=L.cellGlobal[lc];
    for(int i=0;i<4;++i) {
      const int lp=L.tets[lc][(std::size_t)i];
      const int lf=L.oppFace[lc][(std::size_t)i];
      if(lp<0 || lp>=(int)L.pointGlobal.size() || L.pointGlobal[(std::size_t)lp]!=M.tets[(std::size_t)gc][(std::size_t)i]) ++localMismatch;
      if(lf<0 || lf>=(int)L.faceGlobal.size() || L.faceGlobal[(std::size_t)lf]!=M.oppFace[(std::size_t)gc][(std::size_t)i]) ++localMismatch;
    }
  }
  for(std::size_t lf=0;lf<L.faceGlobal.size();++lf) {
    const PetscInt gf=L.faceGlobal[lf];
    for(int i=0;i<3;++i) {
      const int lp=L.faces[lf].v[(std::size_t)i];
      if(lp<0 || lp>=(int)L.pointGlobal.size() || L.pointGlobal[(std::size_t)lp]!=M.faces[(std::size_t)gf].v[(std::size_t)i]) ++localMismatch;
    }
  }
  PetscInt globalMismatch=0; PetscCallMPI(MPI_Allreduce(&localMismatch,&globalMismatch,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));

  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD,
    "P1BF3_DIST_GATE0_RANK rank=%d ownedCells=%" PetscInt_FMT " supportCells=%zu haloCells=%" PetscInt_FMT " velocityStarCells=%" PetscInt_FMT " pressureNeighbourCells=%" PetscInt_FMT " localPoints=%zu localFaces=%zu storageOwnedPoints=%" PetscInt_FMT " storageOwnedFaces=%" PetscInt_FMT "\n",
    rank,L.ownedCells,L.cellGlobal.size(),L.haloCells,L.velocityStarCells,L.pressureNeighbourCells,L.pointGlobal.size(),L.faceGlobal.size(),localOwnedPoints,localOwnedFaces));
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));

  bool ok=true;
  if(global[0]!=nc || global[1]!=nv || global[2]!=nf || pressureDofs!=nc || velocityDofs!=D.ns || globalMismatch!=0) ok=false;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_DIST_GATE0_GLOBAL_PARITY cellsRef=%" PetscInt_FMT " cellsDistributedOwned=%" PetscInt_FMT " pointsRef=%" PetscInt_FMT " pointsDistributedStorageOwned=%" PetscInt_FMT " facesRef=%" PetscInt_FMT " facesDistributedStorageOwned=%" PetscInt_FMT " internalFaces=%" PetscInt_FMT " boundaryFaces=%" PetscInt_FMT " packedConnectivityMismatches=%" PetscInt_FMT " status=%s\n",
    nc,global[0],nv,global[1],nf,global[2],ni,nf-ni,globalMismatch,ok?"PASS":"FAIL"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_DIST_GATE0_DOF_PARITY pressureDofs=%" PetscInt_FMT " expectedPressure=%" PetscInt_FMT " velocityDofs=%" PetscInt_FMT " expectedVelocity=%" PetscInt_FMT " freeVertices=%" PetscInt_FMT " freeFacesBF3=%" PetscInt_FMT " status=%s\n",
    pressureDofs,nc,velocityDofs,D.ns,D.freeVertices,D.freeFaces,(pressureDofs==nc && velocityDofs==D.ns)?"PASS":"FAIL"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_DIST_GATE0_SUPPORT aggregateSupportCells=%" PetscInt_FMT " aggregateHaloCells=%" PetscInt_FMT " supportDuplication=%.6f aggregateLocalPoints=%" PetscInt_FMT " aggregateLocalFaces=%" PetscInt_FMT " architecture=owned_cells_plus_owned_velocity_star_plus_owned_pressure_face_neighbours\n",
    global[3],global[6],nc?((double)global[3]/(double)nc):0.0,global[4],global[5]));

  // Boundary parity using an exactly-once owner-cell assignment of boundary faces.
  const int np=(int)M.patches.size();
  std::vector<double> loc((std::size_t)np*4,0.0), sum((std::size_t)np*4,0.0);
  std::vector<PetscInt> locFaces((std::size_t)np,0), sumFaces((std::size_t)np,0);
  for(PetscInt f=ni;f<nf;++f) if(D.cellOwner[(std::size_t)M.owner[(std::size_t)f]]==rank) {
    const int pi=M.facePatch[(std::size_t)f]; if(pi<0 || pi>=np) {ok=false; continue;}
    const Vec3 sf=faceOutwardAreaVector(M,f);
    loc[(std::size_t)4*pi+0]+=norm3(sf); loc[(std::size_t)4*pi+1]+=sf.x; loc[(std::size_t)4*pi+2]+=sf.y; loc[(std::size_t)4*pi+3]+=sf.z;
    ++locFaces[(std::size_t)pi];
  }
  PetscCallMPI(MPI_Allreduce(loc.data(),sum.data(),(PetscMPIInt)sum.size(),MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(locFaces.data(),sumFaces.data(),np,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  for(int pi=0;pi<np;++pi) {
    const PatchFrame ref=patchFrame(M,pi);
    const Vec3 av{sum[(std::size_t)4*pi+1],sum[(std::size_t)4*pi+2],sum[(std::size_t)4*pi+3]};
    const double proj=norm3(av); const Vec3 n=(proj>0)?scale3(av,1.0/proj):Vec3{};
    const double areaErr=std::abs(sum[(std::size_t)4*pi]-ref.area)/std::max(1.0,std::abs(ref.area));
    const double vecErr=norm3(sub3(av,ref.areaVector))/std::max(1.0,norm3(ref.areaVector));
    const bool pok=(sumFaces[(std::size_t)pi]==M.patches[(std::size_t)pi].nFaces && areaErr<5e-12 && vecErr<5e-12);
    ok=ok&&pok;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_DIST_GATE0_PATCH name=%s facesRef=%" PetscInt_FMT " facesDistributed=%" PetscInt_FMT " areaRef=%.12e areaDistributed=%.12e areaVectorDistributed=[%.12e,%.12e,%.12e] normalDistributed=[%.12e,%.12e,%.12e] areaRelErr=%.3e vectorRelErr=%.3e status=%s\n",
      M.patches[(std::size_t)pi].name.c_str(),M.patches[(std::size_t)pi].nFaces,sumFaces[(std::size_t)pi],ref.area,sum[(std::size_t)4*pi],av.x,av.y,av.z,n.x,n.y,n.z,areaErr,vecErr,pok?"PASS":"FAIL"));
  }

  if(P.mode==ProblemMode::Pipe && (P.inletBC==InletBCMode::PipeParabolic || P.inletBC==InletBCMode::PipeOneSeventh)) {
    double localFlux[2]={0.0,0.0}, flux[2]={0.0,0.0};
    const auto& pin=M.patches[(std::size_t)P.pipe.inlet];
    for(PetscInt f=pin.startFace;f<pin.startFace+pin.nFaces;++f) if(D.cellOwner[(std::size_t)M.owner[(std::size_t)f]]==rank) {
      const auto& F=M.faces[(std::size_t)f]; Vec3 X[3]={M.points[(std::size_t)F.v[0]],M.points[(std::size_t)F.v[1]],M.points[(std::size_t)F.v[2]]};
      const double A=triangleArea(X[0],X[1],X[2]), u=triangleAverageIdealPipeProfileUz(P.pipe,P.inletBC,X);
      localFlux[0]+=A*u; localFlux[1]+=A*(P.pipe.profileScale*u);
    }
    PetscCallMPI(MPI_Allreduce(localFlux,flux,2,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
    const double expected=P.pipe.bulkVelocity*P.pipe.inletArea;
    const double rel=std::abs(flux[1]-expected)/std::max(1e-300,std::abs(expected));
    const bool fok=rel<5e-12; ok=ok&&fok;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_DIST_GATE0_INLET_FLUX profile=%s rawDistributed=%.12e profileScale=%.12e normalizedDistributed=%.12e expectedUbulkArea=%.12e relErr=%.3e status=%s\n",
      P.inletBC==InletBCMode::PipeOneSeventh?"one_seventh":"parabolic",flux[0],P.pipe.profileScale,flux[1],expected,rel,fok?"PASS":"FAIL"));
  } else if(P.mode!=ProblemMode::MMS) {
    double localFlux=0.0,flux=0.0;
    const auto& pin=M.patches[(std::size_t)P.boundary.inlet];
    for(PetscInt f=pin.startFace;f<pin.startFace+pin.nFaces;++f) if(D.cellOwner[(std::size_t)M.owner[(std::size_t)f]]==rank) localFlux+=dot3(P.boundary.inletVelocity,faceOutwardAreaVector(M,f));
    PetscCallMPI(MPI_Allreduce(&localFlux,&flux,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
    const double expected=dot3(P.boundary.inletVelocity,P.boundary.inletAreaVector);
    const double rel=std::abs(flux-expected)/std::max(1e-300,std::abs(expected));
    const bool fok=rel<5e-12; ok=ok&&fok;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_DIST_GATE0_INLET_FLUX profile=fixed_normal distributed=%.12e expected=%.12e relErr=%.3e status=%s\n",flux,expected,rel,fok?"PASS":"FAIL"));
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_DIST_GATE0_RESULT status=%s ranks=%d fullMeshStillReplicated=1 solvePathUnchanged=1 localPackConstructed=1 next=root_only_load_pack_scatter_free_global_on_nonroot\n",
    ok?"PASS":"FAIL",size));
  if(!ok) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"distributed mesh Gate0 parity failed");
  PetscFunctionReturn(PETSC_SUCCESS);
}


// -----------------------------------------------------------------------------
// NodalS distributed production mesh (Gates 1-3)
// -----------------------------------------------------------------------------
// Rank 0 alone reads/retains the complete OpenFOAM mesh and global ownership
// oracle.  It packs each rank's owned-cell + velocity-star + owned-pressure
// face-neighbour support into a compact local Mesh/Discrete packet.  Production
// element/pressure/momentum loops then operate on local indices while PETSc and
// the custom peer halos retain the frozen global DOF numbering.

template<class T> static void distAppendPod(std::vector<char>& b,const T& v) {
  static_assert(std::is_trivially_copyable<T>::value,"POD required");
  const std::size_t o=b.size(); b.resize(o+sizeof(T)); std::memcpy(b.data()+o,&v,sizeof(T));
}
template<class T> static T distReadPod(const std::vector<char>& b,std::size_t& o) {
  static_assert(std::is_trivially_copyable<T>::value,"POD required");
  if(o+sizeof(T)>b.size()) throw std::runtime_error("distributed packet truncated");
  T v{}; std::memcpy(&v,b.data()+o,sizeof(T)); o+=sizeof(T); return v;
}
template<class T> static void distAppendVector(std::vector<char>& b,const std::vector<T>& v) {
  static_assert(std::is_trivially_copyable<T>::value,"POD vector required");
  const std::uint64_t n=(std::uint64_t)v.size(); distAppendPod(b,n);
  if(n){const std::size_t o=b.size(),nb=(std::size_t)n*sizeof(T);b.resize(o+nb);std::memcpy(b.data()+o,v.data(),nb);}
}
template<class T> static std::vector<T> distReadVector(const std::vector<char>& b,std::size_t& o) {
  static_assert(std::is_trivially_copyable<T>::value,"POD vector required");
  const std::uint64_t n=distReadPod<std::uint64_t>(b,o); const std::size_t nb=(std::size_t)n*sizeof(T);
  if(o+nb>b.size()) throw std::runtime_error("distributed vector truncated");
  std::vector<T> v((std::size_t)n); if(nb)std::memcpy(v.data(),b.data()+o,nb);o+=nb;return v;
}
static void distAppendString(std::vector<char>& b,const std::string& x){const std::uint64_t n=(std::uint64_t)x.size();distAppendPod(b,n);const std::size_t o=b.size();b.resize(o+(std::size_t)n);if(n)std::memcpy(b.data()+o,x.data(),(std::size_t)n);}
static std::string distReadString(const std::vector<char>& b,std::size_t& o){const std::uint64_t n=distReadPod<std::uint64_t>(b,o);if(o+(std::size_t)n>b.size())throw std::runtime_error("distributed string truncated");std::string x(b.data()+o,b.data()+o+(std::size_t)n);o+=(std::size_t)n;return x;}

static std::vector<char> packProblemConfigPipe(const ProblemConfig& P) {
  std::vector<char> b; const int mode=(int)P.mode,ibc=(int)P.inletBC,cc=P.centralConvection?1:0,ww=P.weakWallFunction?1:0;
  distAppendPod(b,mode);distAppendPod(b,ibc);distAppendPod(b,cc);distAppendPod(b,ww);distAppendPod(b,P.re);distAppendPod(b,P.nu);
  distAppendString(b,P.pipe.wallPatch);distAppendString(b,P.pipe.inletPatch);distAppendString(b,P.pipe.outletPatch);
  distAppendPod(b,P.pipe.wall);distAppendPod(b,P.pipe.inlet);distAppendPod(b,P.pipe.outlet);
  const double pd[]={P.pipe.cx,P.pipe.cy,P.pipe.zIn,P.pipe.zOut,P.pipe.R,P.pipe.D,P.pipe.L,P.pipe.inletArea,P.pipe.outletArea,P.pipe.circleArea,P.pipe.areaRatio,P.pipe.bulkVelocity,P.pipe.profileScale,P.pipe.nu,P.pipe.re,P.pipe.hpDrop,P.pipe.hpGradient};
  for(double x:pd)distAppendPod(b,x);
  const auto& B=P.boundary; const std::uint64_t nw=(std::uint64_t)B.wallPatches.size();distAppendPod(b,nw);for(const auto& x:B.wallPatches)distAppendString(b,x);distAppendVector(b,B.walls);
  distAppendString(b,B.inletPatch);distAppendString(b,B.outletPatch);distAppendPod(b,B.inlet);distAppendPod(b,B.outlet);
  const double bd[]={B.inletArea,B.outletArea,B.inletProjectedArea,B.outletProjectedArea,B.inletAreaVector.x,B.inletAreaVector.y,B.inletAreaVector.z,B.outletAreaVector.x,B.outletAreaVector.y,B.outletAreaVector.z,B.inletReferenceNormal.x,B.inletReferenceNormal.y,B.inletReferenceNormal.z,B.outletReferenceNormal.x,B.outletReferenceNormal.y,B.outletReferenceNormal.z,B.signedNormalSpeed,B.inletVelocity.x,B.inletVelocity.y,B.inletVelocity.z};
  for(double x:bd)distAppendPod(b,x); return b;
}
static ProblemConfig unpackProblemConfigPipe(const std::vector<char>& b) {
  std::size_t o=0; ProblemConfig P;P.mode=(ProblemMode)distReadPod<int>(b,o);P.inletBC=(InletBCMode)distReadPod<int>(b,o);P.centralConvection=distReadPod<int>(b,o)!=0;P.weakWallFunction=distReadPod<int>(b,o)!=0;P.re=distReadPod<double>(b,o);P.nu=distReadPod<double>(b,o);
  P.pipe.wallPatch=distReadString(b,o);P.pipe.inletPatch=distReadString(b,o);P.pipe.outletPatch=distReadString(b,o);P.pipe.wall=distReadPod<int>(b,o);P.pipe.inlet=distReadPod<int>(b,o);P.pipe.outlet=distReadPod<int>(b,o);
  double *pd[]={&P.pipe.cx,&P.pipe.cy,&P.pipe.zIn,&P.pipe.zOut,&P.pipe.R,&P.pipe.D,&P.pipe.L,&P.pipe.inletArea,&P.pipe.outletArea,&P.pipe.circleArea,&P.pipe.areaRatio,&P.pipe.bulkVelocity,&P.pipe.profileScale,&P.pipe.nu,&P.pipe.re,&P.pipe.hpDrop,&P.pipe.hpGradient};for(double* x:pd)*x=distReadPod<double>(b,o);
  auto& B=P.boundary;const std::uint64_t nw=distReadPod<std::uint64_t>(b,o);B.wallPatches.resize((std::size_t)nw);for(auto& x:B.wallPatches)x=distReadString(b,o);B.walls=distReadVector<int>(b,o);B.inletPatch=distReadString(b,o);B.outletPatch=distReadString(b,o);B.inlet=distReadPod<int>(b,o);B.outlet=distReadPod<int>(b,o);
  double *bd[]={&B.inletArea,&B.outletArea,&B.inletProjectedArea,&B.outletProjectedArea,&B.inletAreaVector.x,&B.inletAreaVector.y,&B.inletAreaVector.z,&B.outletAreaVector.x,&B.outletAreaVector.y,&B.outletAreaVector.z,&B.inletReferenceNormal.x,&B.inletReferenceNormal.y,&B.inletReferenceNormal.z,&B.outletReferenceNormal.x,&B.outletReferenceNormal.y,&B.outletReferenceNormal.z,&B.signedNormalSpeed,&B.inletVelocity.x,&B.inletVelocity.y,&B.inletVelocity.z};for(double* x:bd)*x=distReadPod<double>(b,o);
  if(o!=b.size())throw std::runtime_error("ProblemConfig packet trailing bytes");return P;
}

static PetscErrorCode distBcastBytes(std::vector<char>& b,int root=0) {
  PetscFunctionBeginUser; int rank=0;PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  unsigned long long n=(unsigned long long)b.size();PetscCallMPI(MPI_Bcast(&n,1,MPI_UNSIGNED_LONG_LONG,root,PETSC_COMM_WORLD));if(rank!=root)b.resize((std::size_t)n);
  std::size_t off=0;while(off<(std::size_t)n){const int chunk=(int)std::min<std::size_t>((std::size_t)INT_MAX/2,(std::size_t)n-off);PetscCallMPI(MPI_Bcast(b.data()+off,chunk,MPI_BYTE,root,PETSC_COMM_WORLD));off+=(std::size_t)chunk;}PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode distSendBytes(const std::vector<char>& b,int dst,int tag) {PetscFunctionBeginUser;unsigned long long n=(unsigned long long)b.size();PetscCallMPI(MPI_Send(&n,1,MPI_UNSIGNED_LONG_LONG,dst,tag,PETSC_COMM_WORLD));std::size_t o=0;while(o<b.size()){const int c=(int)std::min<std::size_t>((std::size_t)INT_MAX/2,b.size()-o);PetscCallMPI(MPI_Send((void*)(b.data()+o),c,MPI_BYTE,dst,tag+1,PETSC_COMM_WORLD));o+=(std::size_t)c;}PetscFunctionReturn(PETSC_SUCCESS);}
static PetscErrorCode distRecvBytes(std::vector<char>& b,int src,int tag) {PetscFunctionBeginUser;unsigned long long n=0;MPI_Status st;PetscCallMPI(MPI_Recv(&n,1,MPI_UNSIGNED_LONG_LONG,src,tag,PETSC_COMM_WORLD,&st));b.resize((std::size_t)n);std::size_t o=0;while(o<b.size()){const int c=(int)std::min<std::size_t>((std::size_t)INT_MAX/2,b.size()-o);PetscCallMPI(MPI_Recv(b.data()+o,c,MPI_BYTE,src,tag+1,PETSC_COMM_WORLD,&st));o+=(std::size_t)c;}PetscFunctionReturn(PETSC_SUCCESS);}

struct DistProductionPacket { Mesh M; Discrete D; PetscInt globalCells=0,globalPoints=0,globalFaces=0,globalInternalFaces=0; };

static DistProductionPacket buildDistProductionPacket(const Mesh& G,const Discrete& GD,int rank,PetscBool pressureVertexStarSupport=PETSC_FALSE) {
  DistProductionPacket X; X.globalCells=(PetscInt)G.tets.size();X.globalPoints=(PetscInt)G.points.size();X.globalFaces=(PetscInt)G.faces.size();X.globalInternalFaces=(PetscInt)G.neighbour.size();
  const PetscInt gnv=(PetscInt)G.points.size(),gnf=(PetscInt)G.faces.size(),gni=(PetscInt)G.neighbour.size(),gnc=(PetscInt)G.tets.size();
  PetscInt vStart=0;for(int r=0;r<rank;++r)vStart+=GD.velCount[(std::size_t)r];const PetscInt vEnd=vStart+GD.velCount[(std::size_t)rank];
  std::vector<unsigned char> support((std::size_t)gnc,0);
  for(PetscInt c=0;c<gnc;++c)if(GD.cellOwner[(std::size_t)c]==rank)support[(std::size_t)c]=1;
  for(PetscInt c=0;c<gnc;++c){bool hit=false;for(int i=0;i<4&&!hit;++i){const PetscInt q=GD.g2free[(std::size_t)G.tets[(std::size_t)c][i]];if(q>=vStart&&q<vEnd)hit=true;}for(int i=0;i<4&&!hit;++i){const PetscInt q=GD.g2free[(std::size_t)(gnv+G.oppFace[(std::size_t)c][i])];if(q>=vStart&&q<vEnd)hit=true;}if(hit)support[(std::size_t)c]=1;}
  for(PetscInt c=0;c<gnc;++c)if(GD.cellOwner[(std::size_t)c]==rank)for(int i=0;i<4;++i){const PetscInt f=G.oppFace[(std::size_t)c][i];if(f<gni){support[(std::size_t)G.owner[(std::size_t)f]]=1;support[(std::size_t)G.neighbour[(std::size_t)f]]=1;}}
  // FULLFAST-B0: a full Schur row couples pressure cells through every free P1
  // vertex shared by the owned pressure cell.  The compact production packet
  // previously carried only owned-velocity stars + pressure face neighbours.
  // For p_pmat=full, mark the complete vertex stars of OWNED PRESSURE rows as
  // additional support.  This is still rank-local support, not a replicated mesh.
  if(pressureVertexStarSupport) {
    std::vector<unsigned char> ownedPressureVertex((std::size_t)gnv,0);
    for(PetscInt c=0;c<gnc;++c) if(GD.cellOwner[(std::size_t)c]==rank)
      for(int i=0;i<4;++i) ownedPressureVertex[(std::size_t)G.tets[(std::size_t)c][i]]=1;
    for(PetscInt c=0;c<gnc;++c) {
      bool hit=false;
      for(int i=0;i<4 && !hit;++i) hit = ownedPressureVertex[(std::size_t)G.tets[(std::size_t)c][i]]!=0;
      if(hit) support[(std::size_t)c]=1;
    }
  }
  std::vector<PetscInt> cells;cells.reserve((std::size_t)GD.cellCount[(std::size_t)rank]*2);for(PetscInt c=0;c<gnc;++c)if(support[(std::size_t)c])cells.push_back(c);
  std::vector<int> cLut((std::size_t)gnc,-1);for(std::size_t i=0;i<cells.size();++i)cLut[(std::size_t)cells[i]]=(int)i;
  std::vector<unsigned char> pMark((std::size_t)gnv,0),fMark((std::size_t)gnf,0);for(PetscInt c:cells){for(int i=0;i<4;++i)pMark[(std::size_t)G.tets[(std::size_t)c][i]]=1;for(int i=0;i<4;++i)fMark[(std::size_t)G.oppFace[(std::size_t)c][i]]=1;}
  std::vector<PetscInt> points;for(PetscInt p=0;p<gnv;++p)if(pMark[(std::size_t)p])points.push_back(p);std::vector<int>pLut((std::size_t)gnv,-1);for(std::size_t i=0;i<points.size();++i)pLut[(std::size_t)points[i]]=(int)i;
  // Face ordering: complete internal faces first (preserves f<neighbour.size()),
  // then incomplete internal support faces, then true boundary faces grouped by patch.
  std::vector<PetscInt> fint,fincomplete,fbnd;for(PetscInt f=0;f<gnf;++f)if(fMark[(std::size_t)f]){if(f<gni){if(cLut[(std::size_t)G.owner[(std::size_t)f]]>=0&&cLut[(std::size_t)G.neighbour[(std::size_t)f]]>=0)fint.push_back(f);else fincomplete.push_back(f);}else fbnd.push_back(f);}
  std::vector<PetscInt> faces=fint;faces.insert(faces.end(),fincomplete.begin(),fincomplete.end());
  std::vector<Patch> patches;patches.reserve(G.patches.size());
  for(std::size_t pi=0;pi<G.patches.size();++pi){Patch q;q.name=G.patches[pi].name;q.startFace=(PetscInt)faces.size();for(PetscInt f:fbnd){if(G.facePatch[(std::size_t)f]==(int)pi){faces.push_back(f);++q.nFaces;}}patches.push_back(q);}
  std::vector<int> fLut((std::size_t)gnf,-1);for(std::size_t i=0;i<faces.size();++i)fLut[(std::size_t)faces[i]]=(int)i;
  Mesh& M=X.M;M.points.reserve(points.size());for(PetscInt gp:points)M.points.push_back(G.points[(std::size_t)gp]);M.faces.resize(faces.size());M.owner.resize(faces.size(),-1);M.facePatch.resize(faces.size(),-1);M.neighbour.resize(fint.size(),-1);M.patches=patches;
  for(std::size_t lf=0;lf<faces.size();++lf){const PetscInt gf=faces[lf];for(int j=0;j<3;++j)M.faces[lf].v[(std::size_t)j]=pLut[(std::size_t)G.faces[(std::size_t)gf].v[(std::size_t)j]];const int go=G.owner[(std::size_t)gf];int lo=(go>=0&&go<gnc)?cLut[(std::size_t)go]:-1;if(lo<0&&gf<gni)lo=cLut[(std::size_t)G.neighbour[(std::size_t)gf]];M.owner[lf]=lo;if(lf<fint.size())M.neighbour[lf]=cLut[(std::size_t)G.neighbour[(std::size_t)gf]];if(gf>=gni)M.facePatch[lf]=G.facePatch[(std::size_t)gf];}
  M.tets.resize(cells.size());M.oppFace.resize(cells.size());for(std::size_t lc=0;lc<cells.size();++lc){const PetscInt gc=cells[lc];for(int i=0;i<4;++i){M.tets[lc][(std::size_t)i]=pLut[(std::size_t)G.tets[(std::size_t)gc][i]];M.oppFace[lc][(std::size_t)i]=fLut[(std::size_t)G.oppFace[(std::size_t)gc][i]];}}
  Discrete& D=X.D;D.cellCount=GD.cellCount;D.velCount=GD.velCount;D.ns=GD.ns;D.freeVertices=GD.freeVertices;D.fixedVertices=GD.fixedVertices;D.freeFaces=GD.freeFaces;D.fixedFaces=GD.fixedFaces;D.cellOwner.resize(cells.size());D.pGid.resize(cells.size());for(std::size_t lc=0;lc<cells.size();++lc){D.cellOwner[lc]=GD.cellOwner[(std::size_t)cells[lc]];D.pGid[lc]=GD.pGid[(std::size_t)cells[lc]];}
  const std::size_t nent=M.points.size()+M.faces.size();D.fixedEntity.assign(nent,0);if(!GD.wallEntity.empty())D.wallEntity.assign(nent,0);else std::vector<char>().swap(D.wallEntity);D.g2free.assign(nent,-1);D.fixedDirValue.clear();D.fixedDirValue.reserve(nent/8+1);
  auto copyEntity=[&](std::size_t le,PetscInt ge){
    D.fixedEntity[le]=GD.fixedEntity[(std::size_t)ge];
    if(!D.wallEntity.empty()) D.wallEntity[le]=GD.wallEntity[(std::size_t)ge];
    const PetscInt gg=GD.g2free[(std::size_t)ge];
    if(gg>=0){D.g2free[le]=gg;}
    else{const PetscInt gs=fixedDirSlotFromEncodedGid(gg);const PetscInt ls=(PetscInt)D.fixedDirValue.size();D.fixedDirValue.push_back(GD.fixedDirValue[(std::size_t)gs]);D.g2free[le]=-(ls+2);}
  };
  for(std::size_t lp=0;lp<points.size();++lp)copyEntity(lp,points[lp]);for(std::size_t lf=0;lf<faces.size();++lf)copyEntity(M.points.size()+lf,gnv+faces[lf]);
  return X;
}

static std::vector<char> packDistProductionPacket(const DistProductionPacket& X) {
  std::vector<char>b;distAppendPod(b,X.globalCells);distAppendPod(b,X.globalPoints);distAppendPod(b,X.globalFaces);distAppendPod(b,X.globalInternalFaces);
  distAppendVector(b,X.M.points);distAppendVector(b,X.M.faces);distAppendVector(b,X.M.owner);distAppendVector(b,X.M.neighbour);distAppendVector(b,X.M.tets);distAppendVector(b,X.M.oppFace);distAppendVector(b,X.M.facePatch);const std::uint64_t np=(std::uint64_t)X.M.patches.size();distAppendPod(b,np);for(const auto&q:X.M.patches){distAppendString(b,q.name);distAppendPod(b,q.startFace);distAppendPod(b,q.nFaces);}
  distAppendVector(b,X.D.g2free);distAppendVector(b,X.D.pGid);distAppendVector(b,X.D.cellOwner);distAppendVector(b,X.D.velCount);distAppendVector(b,X.D.cellCount);distAppendVector(b,X.D.fixedEntity);distAppendVector(b,X.D.wallEntity);distAppendVector(b,X.D.fixedDirValue);distAppendPod(b,X.D.ns);distAppendPod(b,X.D.freeVertices);distAppendPod(b,X.D.freeFaces);distAppendPod(b,X.D.fixedVertices);distAppendPod(b,X.D.fixedFaces);return b;
}
static DistProductionPacket unpackDistProductionPacket(const std::vector<char>&b){std::size_t o=0;DistProductionPacket X;X.globalCells=distReadPod<PetscInt>(b,o);X.globalPoints=distReadPod<PetscInt>(b,o);X.globalFaces=distReadPod<PetscInt>(b,o);X.globalInternalFaces=distReadPod<PetscInt>(b,o);X.M.points=distReadVector<Vec3>(b,o);X.M.faces=distReadVector<Face>(b,o);X.M.owner=distReadVector<int>(b,o);X.M.neighbour=distReadVector<int>(b,o);X.M.tets=distReadVector<std::array<int,4>>(b,o);X.M.oppFace=distReadVector<std::array<int,4>>(b,o);X.M.facePatch=distReadVector<int>(b,o);const std::uint64_t np=distReadPod<std::uint64_t>(b,o);X.M.patches.resize((std::size_t)np);for(auto&q:X.M.patches){q.name=distReadString(b,o);q.startFace=distReadPod<PetscInt>(b,o);q.nFaces=distReadPod<PetscInt>(b,o);}X.D.g2free=distReadVector<PetscInt>(b,o);X.D.pGid=distReadVector<PetscInt>(b,o);X.D.cellOwner=distReadVector<int>(b,o);X.D.velCount=distReadVector<PetscInt>(b,o);X.D.cellCount=distReadVector<PetscInt>(b,o);X.D.fixedEntity=distReadVector<char>(b,o);X.D.wallEntity=distReadVector<char>(b,o);X.D.fixedDirValue=distReadVector<std::array<double,3>>(b,o);X.D.ns=distReadPod<PetscInt>(b,o);X.D.freeVertices=distReadPod<PetscInt>(b,o);X.D.freeFaces=distReadPod<PetscInt>(b,o);X.D.fixedVertices=distReadPod<PetscInt>(b,o);X.D.fixedFaces=distReadPod<PetscInt>(b,o);if(o!=b.size())throw std::runtime_error("local mesh packet trailing bytes");return X;}

static PetscErrorCode distributeProductionMeshRoot(const Mesh& G,const Discrete& GD,int rank,int size,DistProductionPacket& local,PetscBool pressureVertexStarSupport=PETSC_FALSE) {
  PetscFunctionBeginUser;
  if(rank==0){for(int r=0;r<size;++r){DistProductionPacket X=buildDistProductionPacket(G,GD,r,pressureVertexStarSupport);std::vector<char>b=packDistProductionPacket(X);if(r==0)local=std::move(X);else PetscCall(distSendBytes(b,r,17000+2*r));}}
  else{std::vector<char>b;PetscCall(distRecvBytes(b,0,17000+2*rank));local=unpackDistProductionPacket(b);}PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode assembleMPI(const Mesh& M, int rank, int size,const ProblemConfig& P, Discrete& D,
  PetscBool buildDiffusionReference=PETSC_FALSE,PetscBool buildBReference=PETSC_FALSE,PetscBool buildMomentumRhsReference=PETSC_FALSE) {
  PetscFunctionBeginUser;
  const PetscInt nv=(PetscInt)M.points.size(), nc=(PetscInt)M.tets.size();
  if(D.cellOwner.empty()) PetscCall(buildOwnership(M,rank,size,P,D));
  const PetscInt globalNc=distributedGlobalCellCount(D);
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
      PetscCall(MatCreateAIJ(PETSC_COMM_WORLD,nlp,nlv,globalNc,D.ns,0,bDnnz.data(),0,bOnnz.data(),&D.B[d]));
      PetscCall(MatSetOption(D.B[d],MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));
    }
    if(buildMomentumRhsReference){PetscCall(VecCreateMPI(PETSC_COMM_WORLD,nlv,D.ns,&D.rhs[d])); PetscCall(VecSet(D.rhs[d],0));}
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_M4B_B_STORAGE PETScB=%s physicalB=custom_FP64_MPI referenceDiagNnz=%" PetscInt_FMT " referenceOffdiagNnz=%" PetscInt_FMT "\n",
    buildBReference?"temporary_reference":"never_created_in_production",gbD,gbO));
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD,nlp,globalNc,&D.volumes)); PetscCall(VecSet(D.volumes,0));
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
    if(dgNumericalTraceInlet(P)) {
      int inletOpp=-1; Vec3 sf{};
      if(dgNumericalTraceFaceGeom(M,P,c,inletOpp,sf)) {
        const double sd[3]={sf.x,sf.y,sf.z};
        for(int d=0;d<3;++d) {
          for(int a=0;a<4;++a) if(a!=inletOpp) Bl[d][a]-=sd[d]/3.0;
          Bl[d][4+inletOpp]-=(9.0/20.0)*sd[d];
        }
        fixedDiv += dot3(P.boundary.inletVelocity,sf);
      }
    }
    for(int a=0;a<8;++a) {
      PetscInt ia=D.g2free[lg[a]];
      if(ia>=0) {
        for(int d=0;d<3;++d) {
          if(buildMomentumRhsReference) PetscCall(VecSetValue(D.rhs[d],ia,fl[d][a],ADD_VALUES));
          if(buildBReference && !(P.weakWallFunction && d<2 && !D.wallEntity.empty() && D.wallEntity[(std::size_t)lg[a]]))
            PetscCall(MatSetValue(D.B[d],pr,ia,Bl[d][a],ADD_VALUES));
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
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_FE cells=%" PetscInt_FMT " localSupportCells=%" PetscInt_FMT " scalarVelDofs=%" PetscInt_FMT " velocityDofs=%" PetscInt_FMT " pressureDofs=%" PetscInt_FMT " freeVertices=%" PetscInt_FMT " freeBF3Faces=%" PetscInt_FMT " totalVolume=%.16e minVol=%.6e maxVol=%.6e hEff=%.12e assembly=%s\n",globalNc,nc,D.ns,3*D.ns,globalNc,D.freeVertices,D.freeFaces,gsum,gmin,gmax,std::cbrt(gsum/globalNc),buildDiffusionReference?(buildBReference?"temporary_DA_plus_B_reference":"temporary_DA_reference_no_B"):(buildBReference?"temporary_B_reference_no_DA":"custom_pressure_B_no_PETSc_B_or_DA")));
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

static PetscErrorCode computeErrorsRoot(const Mesh& M,const Discrete& D,const std::array<std::vector<double>,3>& U,const std::vector<double>& p) {
  PetscFunctionBeginUser;
  const std::vector<double>* ua[3]; for(int d=0;d<3;++d) ua[d]=&U[(std::size_t)d]; auto Q=tetDuffy7(); PetscInt nv=M.points.size();
  double total=0,intdiff=0;
  for(PetscInt c=0;c<(PetscInt)M.tets.size();++c){auto t=M.tets[c];Vec3 X[4]={M.points[t[0]],M.points[t[1]],M.points[t[2]],M.points[t[3]]};double J[3][3]={{X[1].x-X[0].x,X[2].x-X[0].x,X[3].x-X[0].x},{X[1].y-X[0].y,X[2].y-X[0].y,X[3].y-X[0].y},{X[1].z-X[0].z,X[2].z-X[0].z,X[3].z-X[0].z}};double det=det3(J);for(auto&q:Q){double x=0,y=0,z=0;for(int i=0;i<4;++i){x+=q.lam[i]*X[i].x;y+=q.lam[i]*X[i].y;z+=q.lam[i]*X[i].z;}double w=q.w*det;total+=w;intdiff+=(p[(std::size_t)D.pGid[c]]-exactP(x,y,z))*w;}}
  double shift=intdiff/total, ue2=0,un2=0,pe2=0,pn2=0;
  for(PetscInt c=0;c<(PetscInt)M.tets.size();++c){auto t=M.tets[c];Vec3 X[4]={M.points[t[0]],M.points[t[1]],M.points[t[2]],M.points[t[3]]};double J[3][3]={{X[1].x-X[0].x,X[2].x-X[0].x,X[3].x-X[0].x},{X[1].y-X[0].y,X[2].y-X[0].y,X[3].y-X[0].y},{X[1].z-X[0].z,X[2].z-X[0].z,X[3].z-X[0].z}};double det=det3(J);PetscInt lg[8];for(int i=0;i<4;++i)lg[i]=t[i];for(int i=0;i<4;++i)lg[4+i]=nv+M.oppFace[c][i];double coeff[3][8]={{0}};for(int a=0;a<8;++a){PetscInt ia=D.g2free[lg[a]];if(ia>=0)for(int d=0;d<3;++d)coeff[d][a]=(*ua[d])[(std::size_t)ia];}
    for(auto&q:Q){double val[8],gr[8][3];basis(q.lam,val,gr);double x=0,y=0,z=0;for(int i=0;i<4;++i){x+=q.lam[i]*X[i].x;y+=q.lam[i]*X[i].y;z+=q.lam[i]*X[i].z;}double uh[3]={0},ue[3];for(int d=0;d<3;++d)for(int a=0;a<8;++a)uh[d]+=coeff[d][a]*val[a];exactU(x,y,z,ue);double pp=exactP(x,y,z),w=q.w*det;for(int d=0;d<3;++d){ue2+=(uh[d]-ue[d])*(uh[d]-ue[d])*w;un2+=ue[d]*ue[d]*w;}double de=p[(std::size_t)D.pGid[c]]-shift-pp;pe2+=de*de*w;pn2+=pp*pp*w;}}
PetscCall(PetscPrintf(PETSC_COMM_SELF,"P1BF3_MMS U_L2=%.12e U_relL2=%.12e P_shifted_L2=%.12e P_shifted_relL2=%.12e pressureShift=%.12e\n",std::sqrt(ue2),std::sqrt(ue2/un2),std::sqrt(pe2),std::sqrt(pe2/pn2),shift)); PetscFunctionReturn(PETSC_SUCCESS);
}

static double gatheredEntityValue(const Discrete& D,const std::vector<double>& ua,int d,PetscInt entity) {
  const PetscInt gid=D.g2free[entity]; return gid>=0?ua[(std::size_t)gid]:entityDirValue(D,d,entity);
}

static void fixedFaceMean(const Mesh& M,const Discrete& D,PetscInt f,double avg[3]) {
  const PetscInt nv=(PetscInt)M.points.size(); const auto& F=M.faces[f];
  for(int d=0;d<3;++d) {
    avg[d]=(entityDirValue(D,d,F.v[0])+entityDirValue(D,d,F.v[1])+entityDirValue(D,d,F.v[2]))/3.0;
    avg[d]+=(9.0/20.0)*entityDirValue(D,d,nv+f);
  }
}

static PetscErrorCode auditFixedNormalInletRoot(const Mesh& M,const Discrete& D,const ProblemConfig& P) {
  PetscFunctionBeginUser;
  if(P.inletBC!=InletBCMode::FixedNormalSpeed) PetscFunctionReturn(PETSC_SUCCESS);
  const auto& B=P.boundary; const auto& pin=M.patches[B.inlet];
  double flux=0.0,unMin=std::numeric_limits<double>::max(),unMax=-std::numeric_limits<double>::max();
  double meanVectorErrMax=0.0;
  for(PetscInt f=pin.startFace;f<pin.startFace+pin.nFaces;++f) {
    const Vec3 sf=faceOutwardAreaVector(M,f); const double area=norm3(sf); const Vec3 nf=scale3(sf,1.0/area);
    double avg[3]; fixedFaceMean(M,D,f,avg); const Vec3 u{avg[0],avg[1],avg[2]};
    flux += dot3(u,sf); const double un=dot3(u,nf); unMin=std::min(unMin,un); unMax=std::max(unMax,un);
    meanVectorErrMax=std::max(meanVectorErrMax,norm3(sub3(u,B.inletVelocity)));
  }
  const double target=B.signedNormalSpeed*B.inletProjectedArea;
  const double relErr=std::abs(flux-target)/std::max(std::abs(target),1e-300);
  PetscCall(PetscPrintf(PETSC_COMM_SELF,
    "P1BF3_INLET_BC_AUDIT patch=%s mode=fixed_normal_speed normalMode=average_patch_normal signedSpeed=%.12e referenceNormal=[%.12e,%.12e,%.12e] U=[%.12e,%.12e,%.12e] area=%.12e projectedArea=%.12e planarityRatio=%.12e targetFlux=%.12e imposedFlux=%.12e relFluxError=%.12e faceNormalVelocityMin=%.12e faceNormalVelocityMax=%.12e faceMeanVectorErrorMax=%.12e status=%s\n",
    B.inletPatch.c_str(),B.signedNormalSpeed,B.inletReferenceNormal.x,B.inletReferenceNormal.y,B.inletReferenceNormal.z,
    B.inletVelocity.x,B.inletVelocity.y,B.inletVelocity.z,B.inletArea,B.inletProjectedArea,B.inletProjectedArea/B.inletArea,
    target,flux,relErr,unMin,unMax,meanVectorErrMax,(relErr<1e-12 && meanVectorErrMax<1e-12)?"PASS":"CHECK"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

struct PatchSolutionStats {
  double area=0.0,flux=0.0,pOwnerMean=0.0,normalVelocityMin=std::numeric_limits<double>::max(),normalVelocityMax=-std::numeric_limits<double>::max();
};

static PatchSolutionStats patchSolutionStatsRoot(const Mesh& M,const Discrete& D,int pi,const std::vector<double>* ua[3],const double* pa) {
  PatchSolutionStats S; const PetscInt nv=(PetscInt)M.points.size(); const auto& P=M.patches[pi]; double psum=0.0;
  for(PetscInt f=P.startFace;f<P.startFace+P.nFaces;++f) {
    const auto& F=M.faces[f]; const Vec3 sf=faceOutwardAreaVector(M,f); const double area=norm3(sf); const Vec3 nf=scale3(sf,1.0/area);
    double avg[3]={0,0,0};
    for(int d=0;d<3;++d) {
      avg[d]=(gatheredEntityValue(D,*ua[d],d,F.v[0])+gatheredEntityValue(D,*ua[d],d,F.v[1])+gatheredEntityValue(D,*ua[d],d,F.v[2]))/3.0;
      avg[d]+=(9.0/20.0)*gatheredEntityValue(D,*ua[d],d,nv+f);
    }
    const Vec3 u{avg[0],avg[1],avg[2]}; const double un=dot3(u,nf);
    S.area+=area; S.flux+=dot3(u,sf); psum+=area*pa[D.pGid[M.owner[f]]];
    S.normalVelocityMin=std::min(S.normalVelocityMin,un); S.normalVelocityMax=std::max(S.normalVelocityMax,un);
  }
  S.pOwnerMean=psum/std::max(S.area,1e-300); return S;
}

static PetscErrorCode computeFlowDiagnosticsRoot(const Mesh& M,const Discrete& D,const ProblemConfig& P,const std::array<std::vector<double>,3>& U,const std::vector<double>& p) {
  PetscFunctionBeginUser;
  const std::vector<double>* ua[3]={nullptr,nullptr,nullptr};
  for(int d=0;d<3;++d) ua[d]=&U[(std::size_t)d];
  const auto in=patchSolutionStatsRoot(M,D,P.boundary.inlet,ua,p.data()),out=patchSolutionStatsRoot(M,D,P.boundary.outlet,ua,p.data());
  const double net=in.flux+out.flux,scale=std::max({std::abs(in.flux),std::abs(out.flux),1e-300});
  const double target=(P.inletBC==InletBCMode::FixedNormalSpeed)?P.boundary.signedNormalSpeed*P.boundary.inletProjectedArea:0.0;
  const double targetRel=(P.inletBC==InletBCMode::FixedNormalSpeed)?std::abs(in.flux-target)/std::max(std::abs(target),1e-300):0.0;
  PetscCall(PetscPrintf(PETSC_COMM_SELF,
    "P1BF3_FLOW_DIAGNOSTICS inlet=%s outlet=%s inletFlux=%.12e targetInletFlux=%.12e inletFluxRelError=%.12e outletFlux=%.12e netOutwardFlux=%.12e massRelative=%.12e inletArea=%.12e outletArea=%.12e inletFaceNormalVelocity=[%.12e,%.12e] outletFaceNormalVelocity=[%.12e,%.12e]\n",
    P.boundary.inletPatch.c_str(),P.boundary.outletPatch.c_str(),in.flux,target,targetRel,out.flux,net,std::abs(net)/scale,in.area,out.area,
    in.normalVelocityMin,in.normalVelocityMax,out.normalVelocityMin,out.normalVelocityMax));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,
    "P1BF3_FLOW_PRESSURE pInOwnerMean=%.12e pOutOwnerMean=%.12e dropInMinusOut=%.12e pressureGauge=physical_outlet_no_nullspace\n",
    in.pOwnerMean,out.pOwnerMean,in.pOwnerMean-out.pOwnerMean));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static Vec3 tetCellAverageVelocityRoot(const Mesh& M,const Discrete& D,PetscInt c,const std::vector<double>* ua[3]) {
  // Exact volume mean for P1+BF3 on an affine tetrahedron:
  // mean(lambda_i)=1/4 and mean(27*lambda_j*lambda_k*lambda_l)=9/40.
  const PetscInt nv=(PetscInt)M.points.size();
  PetscInt ent[8];
  for(int i=0;i<4;++i) ent[i]=M.tets[c][i];
  for(int i=0;i<4;++i) ent[4+i]=nv+M.oppFace[c][i];
  double u[3]={0.0,0.0,0.0};
  for(int d=0;d<3;++d) {
    for(int i=0;i<4;++i) u[d]+=0.25*gatheredEntityValue(D,*ua[d],d,ent[i]);
    for(int i=0;i<4;++i) u[d]+=(9.0/40.0)*gatheredEntityValue(D,*ua[d],d,ent[4+i]);
  }
  return Vec3{u[0],u[1],u[2]};
}

static double tetVolumeAbsRoot(const Mesh& M,PetscInt c) {
  const auto& t=M.tets[c];
  const Vec3& X0=M.points[t[0]]; const Vec3& X1=M.points[t[1]];
  const Vec3& X2=M.points[t[2]]; const Vec3& X3=M.points[t[3]];
  double J[3][3]={{X1.x-X0.x,X2.x-X0.x,X3.x-X0.x},
                  {X1.y-X0.y,X2.y-X0.y,X3.y-X0.y},
                  {X1.z-X0.z,X2.z-X0.z,X3.z-X0.z}};
  return std::abs(det3(J))/6.0;
}

static PetscErrorCode writeVtuRoot(const std::string& path,const Mesh& M,const Discrete& D,const std::array<std::vector<double>,3>& U,const std::vector<double>& p,PetscBool converged,PetscInt outerIts,const std::string& velocityMode) {
  PetscFunctionBeginUser;
  const bool writeLegacy=(velocityMode=="legacy" || velocityMode=="both");
  const bool writeU0=(velocityMode=="cell_average" || velocityMode=="u0" || velocityMode=="both");
  if(!writeLegacy && !writeU0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER_INPUT,"unknown VTU velocity mode %s",velocityMode.c_str());

  const std::vector<double>* ua[3]={nullptr,nullptr,nullptr};
  for(int d=0;d<3;++d) ua[d]=&U[(std::size_t)d];

  // U0 is the exact element-volume average of the solved P1+BF3 polynomial.
  // U0_stream is a volume-weighted vertex average of neighboring U0 cells.  It
  // exists only as a continuous visualization surrogate for ParaView streamlines.
  std::vector<Vec3> u0, u0Stream;
  if(writeU0) {
    u0.resize(M.tets.size());
    u0Stream.assign(M.points.size(),Vec3{0.0,0.0,0.0});
    std::vector<double> w(M.points.size(),0.0);
    for(PetscInt c=0;c<(PetscInt)M.tets.size();++c) {
      u0[c]=tetCellAverageVelocityRoot(M,D,c,ua);
      const double vol=tetVolumeAbsRoot(M,c);
      for(int i=0;i<4;++i) {
        const PetscInt v=M.tets[c][i];
        u0Stream[v].x+=vol*u0[c].x; u0Stream[v].y+=vol*u0[c].y; u0Stream[v].z+=vol*u0[c].z;
        w[v]+=vol;
      }
    }
    for(PetscInt v=0;v<(PetscInt)M.points.size();++v) if(w[v]>0.0) {
      u0Stream[v].x/=w[v]; u0Stream[v].y/=w[v]; u0Stream[v].z/=w[v];
    }
  }

  std::ofstream out(path); if(!out) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"cannot open VTU output %s",path.c_str());
  out<<std::setprecision(17)<<std::scientific;
  out<<"<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n<UnstructuredGrid>\n";
  out<<"<Piece NumberOfPoints=\""<<M.points.size()<<"\" NumberOfCells=\""<<M.tets.size()<<"\">\n";
  out<<"<Points><DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
  for(const auto& x:M.points) out<<x.x<<' '<<x.y<<' '<<x.z<<'\n'; out<<"</DataArray></Points>\n";
  out<<"<Cells>\n<DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
  for(const auto& t:M.tets) out<<t[0]<<' '<<t[1]<<' '<<t[2]<<' '<<t[3]<<'\n'; out<<"</DataArray>\n";
  out<<"<DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n"; for(size_t c=0;c<M.tets.size();++c) out<<4*(c+1)<<'\n'; out<<"</DataArray>\n";
  out<<"<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n"; for(size_t c=0;c<M.tets.size();++c) out<<10<<'\n'; out<<"</DataArray>\n</Cells>\n";

  out<<"<PointData";
  if(writeU0) out<<" Vectors=\"U0_stream\""; else if(writeLegacy) out<<" Vectors=\"U_P1\"";
  out<<">\n";
  if(writeLegacy) {
    out<<"<DataArray type=\"Float64\" Name=\"U_P1\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for(PetscInt v=0;v<(PetscInt)M.points.size();++v) out<<gatheredEntityValue(D,*ua[0],0,v)<<' '<<gatheredEntityValue(D,*ua[1],1,v)<<' '<<gatheredEntityValue(D,*ua[2],2,v)<<'\n';
    out<<"</DataArray>\n";
  }
  if(writeU0) {
    out<<"<DataArray type=\"Float64\" Name=\"U0_stream\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for(const auto& q:u0Stream) out<<q.x<<' '<<q.y<<' '<<q.z<<'\n';
    out<<"</DataArray>\n";
  }
  out<<"</PointData>\n";

  out<<"<CellData Scalars=\"p_P0\" Vectors=\""<<(writeU0?"U0":"U_P1BF3_centroid")<<"\">\n";
  out<<"<DataArray type=\"Float64\" Name=\"p_P0\" NumberOfComponents=\"1\" format=\"ascii\">\n";
  for(PetscInt c=0;c<(PetscInt)M.tets.size();++c) out<<p[(std::size_t)D.pGid[c]]<<'\n'; out<<"</DataArray>\n";

  if(writeU0) {
    out<<"<DataArray type=\"Float64\" Name=\"U0\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for(const auto& q:u0) out<<q.x<<' '<<q.y<<' '<<q.z<<'\n';
    out<<"</DataArray>\n";
  }

  if(writeLegacy) {
    out<<"<DataArray type=\"Float64\" Name=\"U_P1_centroid\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for(PetscInt c=0;c<(PetscInt)M.tets.size();++c) { double uc[3]={0,0,0}; for(int i=0;i<4;++i) for(int d=0;d<3;++d) uc[d]+=0.25*gatheredEntityValue(D,*ua[d],d,M.tets[c][i]); out<<uc[0]<<' '<<uc[1]<<' '<<uc[2]<<'\n'; }
    out<<"</DataArray>\n";
    out<<"<DataArray type=\"Float64\" Name=\"U_P1BF3_centroid\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    std::array<double,4> lam{{0.25,0.25,0.25,0.25}}; double val[8],gr[8][3]; basis(lam,val,gr); const PetscInt nv=(PetscInt)M.points.size();
    for(PetscInt c=0;c<(PetscInt)M.tets.size();++c) { PetscInt ent[8]; for(int i=0;i<4;++i)ent[i]=M.tets[c][i]; for(int i=0;i<4;++i)ent[4+i]=nv+M.oppFace[c][i]; double uc[3]={0,0,0}; for(int a=0;a<8;++a) for(int d=0;d<3;++d) uc[d]+=val[a]*gatheredEntityValue(D,*ua[d],d,ent[a]); out<<uc[0]<<' '<<uc[1]<<' '<<uc[2]<<'\n'; }
    out<<"</DataArray>\n";
  }

  out<<"<DataArray type=\"Int32\" Name=\"solve_converged\" NumberOfComponents=\"1\" format=\"ascii\">\n"; for(size_t c=0;c<M.tets.size();++c) out<<(converged?1:0)<<'\n'; out<<"</DataArray>\n";
  out<<"<DataArray type=\"Int32\" Name=\"outer_iterations\" NumberOfComponents=\"1\" format=\"ascii\">\n"; for(size_t c=0;c<M.tets.size();++c) out<<outerIts<<'\n'; out<<"</DataArray>\n";
  out<<"</CellData>\n</Piece>\n</UnstructuredGrid>\n</VTKFile>\n"; out.close();

  PetscCall(PetscPrintf(PETSC_COMM_SELF,
    "P1BF3_VTU path=%s mode=%s points=%zu cells=%zu pointVelocity=%s cellVelocity=%s cellPressure=p_P0 converged=%d outerIts=%" PetscInt_FMT " status=PASS\n",
    path.c_str(),velocityMode.c_str(),M.points.size(),M.tets.size(),writeU0?"U0_stream":"U_P1",writeU0?"U0":"U_P1BF3_centroid",converged?1:0,outerIts));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode computePipeDiagnosticsRoot(const Mesh& M,const Discrete& D,const ProblemConfig& P,const std::array<std::vector<double>,3>& U,const std::vector<double>& p) {
  PetscFunctionBeginUser;
  const auto& G=P.pipe;
  const std::vector<double>* ua[3]={nullptr,nullptr,nullptr};
  for(int d=0;d<3;++d) ua[d]=&U[(std::size_t)d];
  const PetscInt nv=(PetscInt)M.points.size();
  const auto Q=tetDuffy7();
  double ue2=0,un2=0,ueMesh2=0,trans2=0,pe2=0,pn2=0,total=0,pdiff=0;
  double sw=0,sz=0,szz=0,sp=0,szp=0;
  for(PetscInt c=0;c<(PetscInt)M.tets.size();++c) {
    const auto t=M.tets[c];Vec3 X[4]={M.points[t[0]],M.points[t[1]],M.points[t[2]],M.points[t[3]]};
    double J[3][3]={{X[1].x-X[0].x,X[2].x-X[0].x,X[3].x-X[0].x},{X[1].y-X[0].y,X[2].y-X[0].y,X[3].y-X[0].y},{X[1].z-X[0].z,X[2].z-X[0].z,X[3].z-X[0].z}};
    const double det=det3(J),vol=det/6.0;
    PetscInt entity[8];for(int i=0;i<4;++i)entity[i]=t[i];for(int i=0;i<4;++i)entity[4+i]=nv+M.oppFace[c][i];
    double coeff[3][8]={{0}};for(int a=0;a<8;++a)for(int d=0;d<3;++d)coeff[d][a]=gatheredEntityValue(D,*ua[d],d,entity[a]);
    const double pc=p[(std::size_t)D.pGid[c]];
    double zc=0;for(int i=0;i<4;++i)zc+=X[i].z;zc*=0.25;
    sw+=vol;sz+=vol*zc;szz+=vol*zc*zc;sp+=vol*pc;szp+=vol*zc*pc;
    for(const auto& q:Q) {
      double val[8],gr[8][3];basis(q.lam,val,gr);
      double x=0,y=0,z=0;for(int i=0;i<4;++i){x+=q.lam[i]*X[i].x;y+=q.lam[i]*X[i].y;z+=q.lam[i]*X[i].z;}
      double uh[3]={0,0,0};for(int d=0;d<3;++d)for(int a=0;a<8;++a)uh[d]+=coeff[d][a]*val[a];
      const double uz=pipeIdealUz(G,x,y),uzMesh=G.profileScale*uz,pex=pipeExactPressure(G,z),w=q.w*det;
      ue2+=(uh[0]*uh[0]+uh[1]*uh[1]+(uh[2]-uz)*(uh[2]-uz))*w;
      ueMesh2+=(uh[0]*uh[0]+uh[1]*uh[1]+(uh[2]-uzMesh)*(uh[2]-uzMesh))*w;
      un2+=uz*uz*w;trans2+=(uh[0]*uh[0]+uh[1]*uh[1])*w;
      const double dp=pc-pex;pe2+=dp*dp*w;pn2+=pex*pex*w;pdiff+=dp*w;total+=w;
    }
  }
  const double pshift=pdiff/total;
  double peShift2=0;
  for(PetscInt c=0;c<(PetscInt)M.tets.size();++c) {
    const auto t=M.tets[c];Vec3 X[4]={M.points[t[0]],M.points[t[1]],M.points[t[2]],M.points[t[3]]};
    double J[3][3]={{X[1].x-X[0].x,X[2].x-X[0].x,X[3].x-X[0].x},{X[1].y-X[0].y,X[2].y-X[0].y,X[3].y-X[0].y},{X[1].z-X[0].z,X[2].z-X[0].z,X[3].z-X[0].z}};
    const double det=det3(J),pc=p[(std::size_t)D.pGid[c]];
    for(const auto& q:Q){double z=0;for(int i=0;i<4;++i)z+=q.lam[i]*X[i].z;const double de=pc-pshift-pipeExactPressure(G,z);peShift2+=de*de*q.w*det;}
  }

  auto faceFluxAndPressure=[&](int pi,double& flux,double& pmean) {
    flux=0.0;pmean=0.0;double areaSum=0.0;const auto& pp=M.patches[pi];
    for(PetscInt f=pp.startFace;f<pp.startFace+pp.nFaces;++f) {
      const auto& F=M.faces[f];const Vec3& x0=M.points[F.v[0]];const Vec3& x1=M.points[F.v[1]];const Vec3& x2=M.points[F.v[2]];
      const double area=triangleArea(x0,x1,x2);double avg[3]={0,0,0};
      for(int d=0;d<3;++d) {
        avg[d]=(gatheredEntityValue(D,*ua[d],d,F.v[0])+gatheredEntityValue(D,*ua[d],d,F.v[1])+gatheredEntityValue(D,*ua[d],d,F.v[2]))/3.0;
        avg[d]+=(9.0/20.0)*gatheredEntityValue(D,*ua[d],d,nv+f);
      }
      flux+=area*avg[2];
      pmean+=area*p[(std::size_t)D.pGid[M.owner[f]]];areaSum+=area;
    }
    pmean/=areaSum;
  };
  double qIn=0,pIn=0,qOut=0,pOut=0;faceFluxAndPressure(G.inlet,qIn,pIn);faceFluxAndPressure(G.outlet,qOut,pOut);
  const double massRel=(qIn!=0)?(qOut-qIn)/qIn:0.0;
  const double denom=sw*szz-sz*sz;
  const double slope=(std::abs(denom)>1e-300)?(sw*szp-sz*sp)/denom:0.0;
  const double fitDrop=-slope*G.L,faceDrop=pIn-pOut;

  PetscCall(PetscPrintf(PETSC_COMM_SELF,
    "P1BF3_PIPE_ERROR U_L2=%.12e U_relL2=%.12e U_meshNormalized_L2=%.12e U_meshNormalized_relL2=%.12e transverse_L2=%.12e P_abs_L2=%.12e P_abs_relL2=%.12e P_shifted_L2=%.12e pressureShift=%.12e\n",
    std::sqrt(ue2),std::sqrt(ue2/std::max(un2,1e-300)),std::sqrt(ueMesh2),std::sqrt(ueMesh2/std::max(G.profileScale*G.profileScale*un2,1e-300)),std::sqrt(trans2),std::sqrt(pe2),std::sqrt(pe2/std::max(pn2,1e-300)),std::sqrt(peShift2),pshift));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,
    "P1BF3_PIPE_FLOW inletFlux=%.12e outletFlux=%.12e inletMean=%.12e outletMean=%.12e massRelative=%.12e inletArea=%.12e outletArea=%.12e\n",
    qIn,qOut,qIn/G.inletArea,qOut/G.outletArea,massRel,G.inletArea,G.outletArea));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,
    "P1BF3_PIPE_PRESSURE pInOwnerMean=%.12e pOutOwnerMean=%.12e dropFaceOwner=%.12e dropAxialFit=%.12e hpDrop=%.12e faceRelError=%.12e fitRelError=%.12e\n",
    pIn,pOut,faceDrop,fitDrop,G.hpDrop,(faceDrop-G.hpDrop)/G.hpDrop,(fitDrop-G.hpDrop)/G.hpDrop));

 
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode destroyDiscrete(Discrete& D){PetscFunctionBeginUser;PetscCall(MatDestroy(&D.A));for(int d=0;d<3;++d){PetscCall(MatDestroy(&D.B[d]));PetscCall(VecDestroy(&D.rhs[d]));}PetscCall(VecDestroy(&D.volumes));PetscCall(VecDestroy(&D.fixedDiv));PetscFunctionReturn(PETSC_SUCCESS);}

static PetscErrorCode buildPositiveRowL1Metric(Mat A, Vec metric) {
  PetscFunctionBeginUser;
  PetscInt rstart=0,rend=0,vstart=0,vend=0;
  PetscCall(MatGetOwnershipRange(A,&rstart,&rend));
  PetscCall(VecGetOwnershipRange(metric,&vstart,&vend));
  if(rstart!=vstart || rend!=vend) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,
    "row-L1 rAU metric ownership does not match momentum matrix ownership");

  PetscScalar *ma=nullptr;
  PetscCall(VecGetArray(metric,&ma));
  for(PetscInt row=rstart; row<rend; ++row) {
    PetscInt ncols=0;
    const PetscInt *cols=nullptr;
    const PetscScalar *vals=nullptr;
    PetscCall(MatGetRow(A,row,&ncols,&cols,&vals));
    PetscReal l1=0.0;
    for(PetscInt k=0;k<ncols;++k) l1 += PetscAbsScalar(vals[k]);
    PetscCall(MatRestoreRow(A,row,&ncols,&cols,&vals));
    ma[row-rstart]=(PetscScalar)l1;
  }
  PetscCall(VecRestoreArray(metric,&ma));
  PetscFunctionReturn(PETSC_SUCCESS);
}


struct SimplecMetricStats {
  PetscReal rawMin=0.0, rawMax=0.0;
  PetscReal blendedMin=0.0, blendedMax=0.0;
  PetscReal ratioMin=0.0, ratioMax=0.0;
  PetscInt nonPositiveRows=0, belowFloorRows=0, fallbackRows=0;
};

// Build the SIMPLEC correction denominator from the relaxed momentum matrix.
// In matrix sign convention the standard FV denominator a_P - sum(a_nb)
// becomes A_PP + sum_{N!=P} A_PN, i.e. the signed row sum A_r * 1.
// simplecBlend=1 gives full SIMPLEC; 0 recovers diagonal SIMPLE.  A positive
// floor/fallback is retained because CG requires the Schur metric rAU to stay
// positive even on strongly convective / constrained FE rows.
static PetscErrorCode buildSimplecMetric(Mat Ar, Vec relaxedDiag, Vec ones,
  PetscReal simplecBlend, PetscReal floorFraction, const std::string& fallback,
  Vec metric, SimplecMetricStats& stats) {
  PetscFunctionBeginUser;
  if(simplecBlend<0.0 || simplecBlend>1.0)
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"-simplec_blend must satisfy 0 <= blend <= 1");
  if(floorFraction<0.0)
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"-simplec_floor_fraction must be >= 0");
  if(fallback!="diag" && fallback!="floor" && fallback!="error")
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"-simplec_fallback must be diag, floor, or error");

  // metric <- signed row sum(Ar)
  PetscCall(MatMult(Ar,ones,metric));

  PetscInt mstart=0,mend=0,dstart=0,dend=0;
  PetscCall(VecGetOwnershipRange(metric,&mstart,&mend));
  PetscCall(VecGetOwnershipRange(relaxedDiag,&dstart,&dend));
  if(mstart!=dstart || mend!=dend)
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"SIMPLEC metric/diagonal ownership mismatch");

  PetscScalar *ma=nullptr;
  const PetscScalar *da=nullptr;
  PetscCall(VecGetArray(metric,&ma));
  PetscCall(VecGetArrayRead(relaxedDiag,&da));

  PetscReal rawMinLocal=PETSC_MAX_REAL,rawMaxLocal=-PETSC_MAX_REAL;
  PetscReal blendMinLocal=PETSC_MAX_REAL,blendMaxLocal=-PETSC_MAX_REAL;
  PetscReal ratioMinLocal=PETSC_MAX_REAL,ratioMaxLocal=-PETSC_MAX_REAL;
  PetscInt nonPositiveLocal=0,belowFloorLocal=0,fallbackLocal=0;
  PetscBool hardError=PETSC_FALSE;

  for(PetscInt i=0;i<mend-mstart;++i) {
    const PetscReal raw=PetscRealPart(ma[i]);
    const PetscReal d=PetscRealPart(da[i]);
    if(!(d>0.0) || PetscIsInfOrNanReal(d)) {
      hardError=PETSC_TRUE;
      continue;
    }
    const PetscReal blended=(1.0-simplecBlend)*d + simplecBlend*raw;
    const PetscReal threshold=floorFraction*d;
    const PetscReal ratio=raw/d;
    rawMinLocal=PetscMin(rawMinLocal,raw); rawMaxLocal=PetscMax(rawMaxLocal,raw);
    blendMinLocal=PetscMin(blendMinLocal,blended); blendMaxLocal=PetscMax(blendMaxLocal,blended);
    ratioMinLocal=PetscMin(ratioMinLocal,ratio); ratioMaxLocal=PetscMax(ratioMaxLocal,ratio);
    if(raw<=0.0) ++nonPositiveLocal;
    if(blended<=threshold) ++belowFloorLocal;

    PetscReal use=blended;
    if(blended<=threshold || PetscIsInfOrNanReal(blended)) {
      if(fallback=="error") {
        hardError=PETSC_TRUE;
      } else if(fallback=="diag") {
        use=d;
        ++fallbackLocal;
      } else {
        use=PetscMax(threshold,std::numeric_limits<PetscReal>::min());
        ++fallbackLocal;
      }
    }
    ma[i]=(PetscScalar)use;
  }

  PetscCall(VecRestoreArrayRead(relaxedDiag,&da));
  PetscCall(VecRestoreArray(metric,&ma));

  PetscInt hardLocal=hardError?1:0,hardGlobal=0;
  PetscCallMPI(MPI_Allreduce(&hardLocal,&hardGlobal,1,MPIU_INT,MPI_MAX,PETSC_COMM_WORLD));
  if(hardGlobal) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,
    "SIMPLEC produced a non-positive/invalid correction denominator; adjust -alpha_u, -simplec_blend, -simplec_floor_fraction, or -simplec_fallback");

  PetscCallMPI(MPI_Allreduce(&rawMinLocal,&stats.rawMin,1,MPIU_REAL,MPI_MIN,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&rawMaxLocal,&stats.rawMax,1,MPIU_REAL,MPI_MAX,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&blendMinLocal,&stats.blendedMin,1,MPIU_REAL,MPI_MIN,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&blendMaxLocal,&stats.blendedMax,1,MPIU_REAL,MPI_MAX,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&ratioMinLocal,&stats.ratioMin,1,MPIU_REAL,MPI_MIN,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&ratioMaxLocal,&stats.ratioMax,1,MPIU_REAL,MPI_MAX,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&nonPositiveLocal,&stats.nonPositiveRows,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&belowFloorLocal,&stats.belowFloorRows,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&fallbackLocal,&stats.fallbackRows,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace


// -----------------------------------------------------------------------------
// PCD Gate 3: P0 pressure-mass inverse, SHADOW ONLY.
//
// The live pressure solve remains Gate-1 PETSc FGMRES + the existing native-face
// GAMG preconditioner.  The Gate-3 shell applies only
//
//      y = M_p^{-1} x,       M_p = diag(cell volume)
//
// on the identical pressure RHS.  It immediately reconstructs M_p y and checks
// M_p(M_p^{-1}x)=x to roundoff.  No Kp, Fp, pressure BC or live-PC change exists
// in this gate.
// -----------------------------------------------------------------------------
struct Gate3MpShellCtx {
  Vec volumes=nullptr;        // borrowed; owned by Discrete
  Vec workIn=nullptr;
  Vec workBack=nullptr;
  Vec workDiff=nullptr;
  PetscInt setupCount=0;
  PetscInt applyCount=0;
  PetscReal volumeMin=0.0;
  PetscReal volumeMax=0.0;
  PetscReal invVolumeMin=0.0;
  PetscReal invVolumeMax=0.0;
  PetscReal lastAlgebraRel=0.0;
  PetscReal maxAlgebraRel=0.0;
};

static PetscErrorCode gate3MpShellSetUp(PC pc) {
  Gate3MpShellCtx *ctx=nullptr;
  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc,(void**)&ctx));
  if(!ctx) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_NULL,"Gate-3 PCShell context is null");
  if(!ctx->volumes) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_NULL,"Gate-3 P0 mass vector is null");
  if(!ctx->workIn || !ctx->workBack || !ctx->workDiff) {
    PetscCall(VecDuplicate(ctx->volumes,&ctx->workIn));
    PetscCall(VecDuplicate(ctx->volumes,&ctx->workBack));
    PetscCall(VecDuplicate(ctx->volumes,&ctx->workDiff));
  }
  PetscInt n=0,nloc=0,nv=0,nvloc=0;
  PetscCall(VecGetSize(ctx->workIn,&n));
  PetscCall(VecGetLocalSize(ctx->workIn,&nloc));
  PetscCall(VecGetSize(ctx->volumes,&nv));
  PetscCall(VecGetLocalSize(ctx->volumes,&nvloc));
  if(n!=nv || nloc!=nvloc) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_SIZ,"Gate-3 pressure/mass vector layout mismatch");
  PetscInt imin=-1,imax=-1;
  PetscCall(VecMin(ctx->volumes,&imin,&ctx->volumeMin));
  PetscCall(VecMax(ctx->volumes,&imax,&ctx->volumeMax));
  if(!(ctx->volumeMin>0.0) || !std::isfinite((double)ctx->volumeMin) || !std::isfinite((double)ctx->volumeMax))
    SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_FP,"Gate-3 P0 mass has non-positive or non-finite cell volume");
  ctx->invVolumeMin=1.0/ctx->volumeMax;
  ctx->invVolumeMax=1.0/ctx->volumeMin;
  ++ctx->setupCount;
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)pc),
    "P1BF3_GATE3_MP_SETUP setupCount=%" PetscInt_FMT " globalSize=%" PetscInt_FMT " localSizeRank0=%" PetscInt_FMT " volumeMin=%.12e volumeMax=%.12e invVolumeMin=%.12e invVolumeMax=%.12e mass=P0_cell_volume_diagonal action=Mp_inverse_shadow_only livePressurePC=UNCHANGED_GAMG\n",
    ctx->setupCount,n,nloc,(double)ctx->volumeMin,(double)ctx->volumeMax,(double)ctx->invVolumeMin,(double)ctx->invVolumeMax));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode gate3MpShellApply(PC pc,Vec x,Vec y) {
  Gate3MpShellCtx *ctx=nullptr;
  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc,(void**)&ctx));
  if(!ctx) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_NULL,"Gate-3 PCShell context is null");
  if(!ctx->workIn || !ctx->workBack || !ctx->workDiff) PetscCall(gate3MpShellSetUp(pc));

  // P0 pressure basis is one constant per tetrahedron, so the consistent pressure
  // mass is exactly diagonal with M_p(c,c)=|K_c|.  No lumping approximation is
  // being introduced here.
  PetscCall(VecCopy(x,ctx->workIn));
  PetscCall(VecPointwiseDivide(y,ctx->workIn,ctx->volumes));

  // Independent algebra check: M_p * (M_p^{-1} x) == x.
  PetscCall(VecPointwiseMult(ctx->workBack,y,ctx->volumes));
  PetscCall(VecCopy(ctx->workBack,ctx->workDiff));
  PetscCall(VecAXPY(ctx->workDiff,-1.0,x));
  PetscReal dn=0.0,xn=0.0;
  PetscCall(VecNorm(ctx->workDiff,NORM_2,&dn));
  PetscCall(VecNorm(x,NORM_2,&xn));
  const PetscReal rel=dn/PetscMax(xn,(PetscReal)1e-300);
  ctx->lastAlgebraRel=rel;
  ctx->maxAlgebraRel=PetscMax(ctx->maxAlgebraRel,rel);
  ++ctx->applyCount;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode gate3MpShellDestroy(PC pc) {
  Gate3MpShellCtx *ctx=nullptr;
  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc,(void**)&ctx));
  if(ctx) {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)pc),
      "P1BF3_GATE3_MP_DESTROY setupCount=%" PetscInt_FMT " applyCount=%" PetscInt_FMT " maxAlgebraRel=%.3e volumeMin=%.12e volumeMax=%.12e\n",
      ctx->setupCount,ctx->applyCount,(double)ctx->maxAlgebraRel,(double)ctx->volumeMin,(double)ctx->volumeMax));
    PetscCall(VecDestroy(&ctx->workIn));
    PetscCall(VecDestroy(&ctx->workBack));
    PetscCall(VecDestroy(&ctx->workDiff));
    ctx->volumes=nullptr;
    delete ctx;
    PetscCall(PCShellSetContext(pc,nullptr));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}


// -----------------------------------------------------------------------------
// PCD Gate 4: geometric compact pressure Laplacian K_p, SHADOW/SETUP ONLY.
// Internal coefficient is the proven P21 geometric FV coefficient
//   a_f = |S_f|^2 / |S_f . (x_N-x_P)| = |S_f|/h_n.
// Pipe pressure-space BCs in Kp: inlet/wall homogeneous Neumann, outlet p=0.
// No pressure nullspace is attached because the outlet Dirichlet rows anchor Kp.
// Kp is NOT attached to the live pressure solve in this gate.
// -----------------------------------------------------------------------------
struct Gate4KpCtx {
  Mat Kp=nullptr;
  Vec ones=nullptr,kpOnes=nullptr,test=nullptr,kpTest=nullptr,diag=nullptr;
  PetscInt globalSize=0,localSize=0,nnz=0,expectedNnz=0;
  PetscInt internalFaces=0,inletFaces=0,wallFaces=0,outletFaces=0;
  PetscReal coeffMin=PETSC_MAX_REAL,coeffMax=0.0;
  PetscReal outletCoeffMin=PETSC_MAX_REAL,outletCoeffMax=0.0;
  PetscReal diagMin=0.0,diagMax=0.0;
  PetscReal constantActionNorm=0.0,constantActionMin=0.0,constantActionMax=0.0;
  PetscReal testNorm=0.0,testEnergy=0.0,symmetryTol=1e-13;
  PetscBool symmetric=PETSC_FALSE,built=PETSC_FALSE;
};

static Vec3 gate4FaceCentre(const Mesh& M,PetscInt f) {
  Vec3 q{}; const auto& F=M.faces[(std::size_t)f];
  if(F.v.empty()) throw std::runtime_error("Gate-4 Kp encountered empty face");
  for(PetscInt v:F.v) { const auto& x=M.points[(std::size_t)v]; q.x+=x.x; q.y+=x.y; q.z+=x.z; }
  const double z=1.0/(double)F.v.size(); return {q.x*z,q.y*z,q.z*z};
}

static Vec3 gate4FaceAreaVector(const Mesh& M,PetscInt f) {
  const auto& F=M.faces[(std::size_t)f];
  if(F.v.size()!=3) throw std::runtime_error("Gate-4 Kp requires triangular tetrahedral faces");
  const Vec3& x0=M.points[(std::size_t)F.v[0]];
  const Vec3& x1=M.points[(std::size_t)F.v[1]];
  const Vec3& x2=M.points[(std::size_t)F.v[2]];
  const Vec3 cr=cross3(sub3(x1,x0),sub3(x2,x0));
  return {0.5*cr.x,0.5*cr.y,0.5*cr.z};
}

static double gate4GeomCoeff(const Vec3& S,const Vec3& d) {
  const double s2=S.x*S.x+S.y*S.y+S.z*S.z;
  const double sd=std::abs(S.x*d.x+S.y*d.y+S.z*d.z);
  if(!(s2>0.0) || !(sd>1.0e-30) || !std::isfinite(s2) || !std::isfinite(sd))
    throw std::runtime_error("Gate-4 geometric Kp encountered degenerate/nonorthogonal face metric");
  const double a=s2/sd;
  if(!(a>0.0) || !std::isfinite(a)) throw std::runtime_error("Gate-4 geometric Kp produced non-positive/non-finite coefficient");
  return a;
}


// -----------------------------------------------------------------------------
// Gate 9E: NGQI-style nodal-gradient auxiliary pressure Laplacian.
//
// 1) At each mesh vertex v, reconstruct grad(p)_v from a WIDE affine/P1 LSQ
//    fit to surrounding cell-centred P0 values.  The first support is the direct
//    vertex star plus all cells sharing any vertex with a star cell; if that is
//    rank deficient, expand by one additional cell-node-neighbour layer.
// 2) On a triangular face f, use the arithmetic mean of its three nodal
//    gradients, exactly as in the accepted tet NGQI reconstruction lineage.
// 3) Assemble the conservative raw FV/Gauss operator
//       L_raw p |_K = - sum_f S_{Kf} . grad_f(p)
//    on internal faces; inlet/wall fluxes are homogeneous Neumann and the
//    outlet gets the same p=0 geometric anchor as Gate 4.
// 4) L_raw is generally nonsymmetric.  PCG requires an SPD preconditioner, so
//    the live Kp is the symmetric M-matrix projection of L_raw:
//       s_ij = 0.5 (L_ij + L_ji),  w_ij=max(0,-s_ij),
//       K_ij=-w_ij, K_ii=sum_j w_ij + outlet_anchor_i.
//    This preserves the learned wide-stencil couplings that have Laplacian sign,
//    guarantees a symmetric graph Laplacian plus the outlet anchor, and keeps
//    the exact FE SIMPLE Schur completely unchanged.
// -----------------------------------------------------------------------------
struct Gate9eNodalKpAudit {
  Mat raw=nullptr;
  PetscInt rawNnz=0,kpNnz=0,nNodes=0;
  PetscInt supportMin=PETSC_MAX_INT,supportMax=0;
  PetscReal supportMean=0.0;
  PetscReal lsqConstDefect=0.0,lsqLinearDefect=0.0;
  PetscReal rawSymmetryDefect=0.0;
  PetscInt keptNegativePairs=0,discardedPositivePairs=0;
  PetscReal discardedPositiveAbs=0.0,keptNegativeAbs=0.0;
};

static bool gate9eInvert4(const double A[4][4],double AI[4][4]) {
  double q[4][8]={{0}};
  for(int i=0;i<4;++i) { for(int j=0;j<4;++j) q[i][j]=A[i][j]; q[i][4+i]=1.0; }
  for(int k=0;k<4;++k) {
    int piv=k; double best=std::abs(q[k][k]);
    for(int i=k+1;i<4;++i) if(std::abs(q[i][k])>best) {best=std::abs(q[i][k]);piv=i;}
    if(!(best>1e-14) || !std::isfinite(best)) return false;
    if(piv!=k) for(int j=0;j<8;++j) std::swap(q[k][j],q[piv][j]);
    const double d=q[k][k]; for(int j=0;j<8;++j) q[k][j]/=d;
    for(int i=0;i<4;++i) if(i!=k) { const double a=q[i][k]; for(int j=0;j<8;++j) q[i][j]-=a*q[k][j]; }
  }
  for(int i=0;i<4;++i) for(int j=0;j<4;++j) AI[i][j]=q[i][4+j];
  return true;
}

struct Gate9eNodeWeight { PetscInt cell=-1; Vec3 w{}; };

static PetscErrorCode gate9eBuildNodalKp(const Mesh& M,const Discrete& D,const ProblemConfig& P,int rank,Gate4KpCtx& G4,Gate9eNodalKpAudit& A9) {
  PetscFunctionBeginUser;
  if(P.mode==ProblemMode::MMS) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Gate-9E nodal Kp currently targets pipe/flow BCs");
  const PetscInt nc=(PetscInt)M.tets.size(),nv=(PetscInt)M.points.size(),ni=(PetscInt)M.neighbour.size();
  const PetscInt nlp=D.cellCount[(std::size_t)rank];
  PetscInt pStart=0; for(int r=0;r<rank;++r) pStart+=D.cellCount[(std::size_t)r];
  const auto cc=cellCentroids(M);

  std::vector<std::vector<PetscInt>> vertexCells((std::size_t)nv);
  for(PetscInt c=0;c<nc;++c) for(int a=0;a<4;++a) vertexCells[(std::size_t)M.tets[(std::size_t)c][a]].push_back(c);

  std::vector<std::vector<Gate9eNodeWeight>> W((std::size_t)nv);
  long long localSupportSum=0; PetscInt localSupportMin=PETSC_MAX_INT,localSupportMax=0;
  double localConstDef=0.0,localLinDef=0.0;
  for(PetscInt v=0;v<nv;++v) {
    std::set<PetscInt> supp;
    auto addCellNodeNeighbours=[&](const std::vector<PetscInt>& seed) {
      for(PetscInt c:seed) {
        supp.insert(c);
        for(int a=0;a<4;++a) {
          const PetscInt qv=M.tets[(std::size_t)c][a];
          for(PetscInt q:vertexCells[(std::size_t)qv]) supp.insert(q);
        }
      }
    };
    addCellNodeNeighbours(vertexCells[(std::size_t)v]);
    bool ok=false; double AI[4][4]={{0}},scale=1.0;
    for(int attempt=0;attempt<2 && !ok;++attempt) {
      if(attempt==1) { std::vector<PetscInt> seed(supp.begin(),supp.end()); addCellNodeNeighbours(seed); }
      scale=0.0; for(PetscInt c:supp) scale=std::max(scale,norm3(sub3(cc[(std::size_t)c],M.points[(std::size_t)v])));
      scale=std::max(scale,1e-30);
      double H[4][4]={{0}};
      for(PetscInt c:supp) {
        const Vec3 d=scale3(sub3(cc[(std::size_t)c],M.points[(std::size_t)v]),1.0/scale);
        const double a[4]={1.0,d.x,d.y,d.z};
        for(int i=0;i<4;++i) for(int j=0;j<4;++j) H[i][j]+=a[i]*a[j];
      }
      ok=gate9eInvert4(H,AI);
    }
    if(!ok) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate-9E nodal affine LSQ remained rank deficient after support expansion");
    auto& wv=W[(std::size_t)v]; wv.reserve(supp.size());
    Vec3 sumW{},linX{},linY{},linZ{};
    for(PetscInt c:supp) {
      const Vec3 d=scale3(sub3(cc[(std::size_t)c],M.points[(std::size_t)v]),1.0/scale);
      const double a[4]={1.0,d.x,d.y,d.z};
      Vec3 w{};
      for(int j=0;j<4;++j) { w.x+=AI[1][j]*a[j]/scale; w.y+=AI[2][j]*a[j]/scale; w.z+=AI[3][j]*a[j]/scale; }
      wv.push_back({c,w}); sumW=add3(sumW,w);
      const Vec3 dr=sub3(cc[(std::size_t)c],M.points[(std::size_t)v]);
      linX=add3(linX,scale3(w,dr.x)); linY=add3(linY,scale3(w,dr.y)); linZ=add3(linZ,scale3(w,dr.z));
    }
    const double cdef=norm3(sumW);
    const double ldef=std::max({std::abs(linX.x-1.0),std::abs(linX.y),std::abs(linX.z),
                                std::abs(linY.x),std::abs(linY.y-1.0),std::abs(linY.z),
                                std::abs(linZ.x),std::abs(linZ.y),std::abs(linZ.z-1.0)});
    localConstDef=std::max(localConstDef,cdef); localLinDef=std::max(localLinDef,ldef);
    const PetscInt ns=(PetscInt)supp.size(); localSupportMin=std::min(localSupportMin,ns); localSupportMax=std::max(localSupportMax,ns); localSupportSum+=ns;
  }
  A9.nNodes=nv;
  double gSupportSum=0.0,gConstDef=0.0,gLinDef=0.0; PetscInt gMin=0,gMax=0;
  const double ls=(double)localSupportSum;
  PetscCallMPI(MPI_Allreduce(&ls,&gSupportSum,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localSupportMin,&gMin,1,MPIU_INT,MPI_MIN,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localSupportMax,&gMax,1,MPIU_INT,MPI_MAX,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localConstDef,&gConstDef,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localLinDef,&gLinDef,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  // Mesh is replicated, so every rank sees every node; divide support sum by nranks below.
  int nranks=1; PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&nranks));
  A9.supportMin=gMin; A9.supportMax=gMax; A9.supportMean=(PetscReal)(gSupportSum/((double)nranks*(double)PetscMax(nv,(PetscInt)1)));
  A9.lsqConstDefect=(PetscReal)gConstDef; A9.lsqLinearDefect=(PetscReal)gLinDef;

  // Conservative raw Gauss FV Laplacian from face-averaged nodal gradients.
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD,nlp,nlp,nc,nc,32,nullptr,32,nullptr,&A9.raw));
  PetscCall(MatSetOption(A9.raw,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));
  for(PetscInt K=0;K<nc;++K) if(D.cellOwner[(std::size_t)K]==rank) {
    const PetscInt row=D.pGid[(std::size_t)K];
    std::map<PetscInt,double> rv;
    for(int lf=0;lf<4;++lf) {
      const PetscInt f=M.oppFace[(std::size_t)K][lf];
      if(f<ni) {
        Vec3 Sout=gate4FaceAreaVector(M,f);
        const Vec3 fc=gate4FaceCentre(M,f);
        if(Sout.x*(fc.x-cc[(std::size_t)K].x)+Sout.y*(fc.y-cc[(std::size_t)K].y)+Sout.z*(fc.z-cc[(std::size_t)K].z)<0.0) Sout=scale3(Sout,-1.0);
        const auto& F=M.faces[(std::size_t)f];
        for(PetscInt v:F.v) for(const auto& wc:W[(std::size_t)v]) {
          const double q=-(Sout.x*wc.w.x+Sout.y*wc.w.y+Sout.z*wc.w.z)/3.0;
          rv[D.pGid[(std::size_t)wc.cell]] += q;
        }
      } else {
        const int pi=M.facePatch[(std::size_t)f];
        if(pi==P.boundary.outlet) {
          const double a=gate4GeomCoeff(gate4FaceAreaVector(M,f),sub3(gate4FaceCentre(M,f),cc[(std::size_t)K]));
          rv[row]+=a;
        } else if(pi==P.boundary.inlet || isWallPatch(P.boundary,pi)) {
          // homogeneous Neumann: no flux
        } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate-9E encountered unclassified pressure boundary face");
      }
    }
    std::vector<PetscInt> cols; std::vector<PetscScalar> vals; cols.reserve(rv.size());vals.reserve(rv.size());
    for(const auto& kv:rv) if(std::abs(kv.second)>1e-18) {cols.push_back(kv.first);vals.push_back((PetscScalar)kv.second);}
    if(!cols.empty()) PetscCall(MatSetValues(A9.raw,1,&row,(PetscInt)cols.size(),cols.data(),vals.data(),INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A9.raw,MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(A9.raw,MAT_FINAL_ASSEMBLY));
  MatInfo rawInfo{}; PetscCall(MatGetInfo(A9.raw,MAT_GLOBAL_SUM,&rawInfo)); A9.rawNnz=(PetscInt)(rawInfo.nz_used+0.5);
  Mat RT=nullptr,SYM=nullptr; PetscCall(MatTranspose(A9.raw,MAT_INITIAL_MATRIX,&RT)); PetscCall(MatDuplicate(A9.raw,MAT_COPY_VALUES,&SYM));
  PetscCall(MatAXPY(SYM,1.0,RT,DIFFERENT_NONZERO_PATTERN)); PetscCall(MatScale(SYM,0.5)); PetscCall(MatDestroy(&RT));
  Mat rawDiff=nullptr; PetscCall(MatDuplicate(A9.raw,MAT_COPY_VALUES,&rawDiff)); Mat RT2=nullptr; PetscCall(MatTranspose(A9.raw,MAT_INITIAL_MATRIX,&RT2)); PetscCall(MatAXPY(rawDiff,-1.0,RT2,DIFFERENT_NONZERO_PATTERN));
  PetscReal rawNorm=0.0,diffNorm=0.0; PetscCall(MatNorm(A9.raw,NORM_FROBENIUS,&rawNorm)); PetscCall(MatNorm(rawDiff,NORM_FROBENIUS,&diffNorm));
  A9.rawSymmetryDefect=(PetscReal)(diffNorm/std::max((double)rawNorm,1e-300)); PetscCall(MatDestroy(&rawDiff)); PetscCall(MatDestroy(&RT2));

  // Symmetric M-matrix projection for PCG/GAMG.
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD,nlp,nlp,nc,nc,32,nullptr,32,nullptr,&G4.Kp));
  PetscCall(MatSetOption(G4.Kp,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE)); PetscCall(MatSetOption(G4.Kp,MAT_SYMMETRIC,PETSC_TRUE));
  PetscInt localKeep=0,localDiscard=0; double localKeepAbs=0.0,localDiscardAbs=0.0;
  for(PetscInt row=pStart;row<pStart+nlp;++row) {
    const PetscInt *cols=nullptr; const PetscScalar *vals=nullptr; PetscInt ncols=0;
    PetscCall(MatGetRow(SYM,row,&ncols,&cols,&vals));
    std::vector<PetscInt> outC; std::vector<PetscScalar> outV; double diag=0.0;
    for(PetscInt j=0;j<ncols;++j) if(cols[j]!=row) {
      const double sij=PetscRealPart(vals[j]);
      if(sij<0.0) { const double w=-sij; outC.push_back(cols[j]); outV.push_back((PetscScalar)(-w)); diag+=w; ++localKeep; localKeepAbs+=w; }
      else if(sij>1e-18) { ++localDiscard; localDiscardAbs+=sij; }
    }
    PetscCall(MatRestoreRow(SYM,row,&ncols,&cols,&vals));
    // Preserve physical outlet p=0 anchoring explicitly.
    PetscInt K=-1; for(PetscInt c=0;c<nc;++c) if(D.pGid[(std::size_t)c]==row) {K=c;break;}
    if(K<0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate-9E pressure gid->cell lookup failed");
    for(int lf=0;lf<4;++lf) { const PetscInt f=M.oppFace[(std::size_t)K][lf]; if(f>=ni && M.facePatch[(std::size_t)f]==P.boundary.outlet) diag+=gate4GeomCoeff(gate4FaceAreaVector(M,f),sub3(gate4FaceCentre(M,f),cc[(std::size_t)K])); }
    outC.push_back(row); outV.push_back((PetscScalar)diag);
    PetscCall(MatSetValues(G4.Kp,1,&row,(PetscInt)outC.size(),outC.data(),outV.data(),INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(G4.Kp,MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(G4.Kp,MAT_FINAL_ASSEMBLY)); PetscCall(MatDestroy(&SYM));
  PetscCallMPI(MPI_Allreduce(&localKeep,&A9.keptNegativePairs,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localDiscard,&A9.discardedPositivePairs,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localKeepAbs,&A9.keptNegativeAbs,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localDiscardAbs,&A9.discardedPositiveAbs,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));

  G4.globalSize=nc; G4.localSize=nlp; G4.internalFaces=ni; G4.inletFaces=0; G4.wallFaces=0; G4.outletFaces=0; G4.expectedNnz=0;
  for(PetscInt f=ni;f<(PetscInt)M.faces.size();++f) { const int pi=M.facePatch[(std::size_t)f]; if(pi==P.boundary.inlet)++G4.inletFaces; else if(pi==P.boundary.outlet)++G4.outletFaces; else if(isWallPatch(P.boundary,pi))++G4.wallFaces; }
  MatInfo info{}; PetscCall(MatGetInfo(G4.Kp,MAT_GLOBAL_SUM,&info)); G4.nnz=(PetscInt)(info.nz_used+0.5); A9.kpNnz=G4.nnz;
  G4.expectedNnz=G4.nnz; G4.coeffMin=1.0; G4.coeffMax=1.0;
  PetscCall(MatIsSymmetric(G4.Kp,1e-12,&G4.symmetric)); G4.symmetryTol=1e-12;
  PetscCall(MatCreateVecs(G4.Kp,&G4.ones,&G4.kpOnes)); PetscCall(VecDuplicate(G4.ones,&G4.test)); PetscCall(VecDuplicate(G4.ones,&G4.kpTest)); PetscCall(VecDuplicate(G4.ones,&G4.diag));
  PetscCall(VecSet(G4.ones,1.0)); PetscCall(MatMult(G4.Kp,G4.ones,G4.kpOnes)); PetscCall(VecNorm(G4.kpOnes,NORM_2,&G4.constantActionNorm)); PetscInt ii=-1; PetscCall(VecMin(G4.kpOnes,&ii,&G4.constantActionMin)); PetscCall(VecMax(G4.kpOnes,&ii,&G4.constantActionMax));
  PetscCall(MatGetDiagonal(G4.Kp,G4.diag)); PetscCall(VecMin(G4.diag,&ii,&G4.diagMin)); PetscCall(VecMax(G4.diag,&ii,&G4.diagMax));
  PetscInt xs=0,xe=0; PetscCall(VecGetOwnershipRange(G4.test,&xs,&xe)); PetscScalar *xa=nullptr; PetscCall(VecGetArray(G4.test,&xa)); for(PetscInt g=xs;g<xe;++g){const double z=(double)(g+1);xa[(std::size_t)(g-xs)]=(PetscScalar)(std::sin(.017*z)+.31*std::cos(.031*z));} PetscCall(VecRestoreArray(G4.test,&xa));
  PetscCall(MatMult(G4.Kp,G4.test,G4.kpTest)); PetscScalar dot=0.0; PetscCall(VecDot(G4.test,G4.kpTest,&dot)); G4.testEnergy=(PetscReal)PetscRealPart(dot); PetscCall(VecNorm(G4.test,NORM_2,&G4.testNorm)); G4.built=PETSC_TRUE;

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_GATE9E_NODAL_LSQ nodes=%" PetscInt_FMT " supportMin=%" PetscInt_FMT " supportMean=%.3f supportMax=%" PetscInt_FMT " constDefect=%.3e linearDefect=%.3e basis=affine_P1 wideSupport=pointCells_plus_cellNodeNeighbours runtimeUnknowns=cell_P0_only\n",
    nv,A9.supportMin,(double)A9.supportMean,A9.supportMax,(double)A9.lsqConstDefect,(double)A9.lsqLinearDefect));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_GATE9E_RAW_FV rows=%" PetscInt_FMT " nnz=%" PetscInt_FMT " avgNnzPerRow=%.3f faceGradient=mean_3_nodal_gradients cellGradient=mean_4_nodal_gradients_available rawSymmetryDefect=%.3e operator=-GaussDiv_faceNodalGradient outletAnchor=TPFA_p0 inletWall=Neumann\n",
    nc,A9.rawNnz,(double)A9.rawNnz/(double)PetscMax(nc,(PetscInt)1),(double)A9.rawSymmetryDefect));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_GATE9E_KP_PROJECTED rows=%" PetscInt_FMT " nnz=%" PetscInt_FMT " avgNnzPerRow=%.3f symmetric=%d diagMin=%.3e testEnergy=%.3e keptNegativeDirected=%" PetscInt_FMT " discardedPositiveDirected=%" PetscInt_FMT " discardedToKeptAbs=%.6e projection=symmetric_Mmatrix_of_raw_NGQI_FV purpose=PCG_GAMG\n",
    nc,G4.nnz,(double)G4.nnz/(double)PetscMax(nc,(PetscInt)1),(int)G4.symmetric,(double)G4.diagMin,(double)G4.testEnergy,A9.keptNegativePairs,A9.discardedPositivePairs,(double)A9.discardedPositiveAbs/std::max((double)A9.keptNegativeAbs,1e-300)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode gate4BuildKp(const Mesh& M,const Discrete& D,const ProblemConfig& P,int rank,Gate4KpCtx& G4) {
  PetscFunctionBeginUser;
  if(P.mode==ProblemMode::MMS) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Gate-4 pipe Kp probe requires -problem pipe or flow");
  const PetscInt nc=(PetscInt)M.tets.size(), ni=(PetscInt)M.neighbour.size();
  const PetscInt nlp=D.cellCount[(std::size_t)rank];
  PetscInt pStart=0; for(int r=0;r<rank;++r) pStart+=D.cellCount[(std::size_t)r];
  const PetscInt pEnd=pStart+nlp;
  G4.globalSize=nc; G4.localSize=nlp; G4.expectedNnz=nc+2*ni;

  std::vector<PetscInt> dnnz((std::size_t)nlp,1),onnz((std::size_t)nlp,0);
  for(PetscInt K=0;K<nc;++K) if(D.cellOwner[(std::size_t)K]==rank) {
    const PetscInt lr=D.pGid[(std::size_t)K]-pStart;
    if(lr<0 || lr>=nlp) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate-4 pressure ownership mismatch during Kp preallocation");
    for(int i=0;i<4;++i) {
      const PetscInt f=M.oppFace[(std::size_t)K][i]; if(f>=ni) continue;
      const PetscInt L=(M.owner[(std::size_t)f]==K)?M.neighbour[(std::size_t)f]:M.owner[(std::size_t)f];
      const PetscInt pg=D.pGid[(std::size_t)L];
      if(pg>=pStart && pg<pEnd) ++dnnz[(std::size_t)lr]; else ++onnz[(std::size_t)lr];
    }
  }
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD,nlp,nlp,nc,nc,0,dnnz.data(),0,onnz.data(),&G4.Kp));
  PetscCall(MatSetOption(G4.Kp,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));
  PetscCall(MatSetOption(G4.Kp,MAT_SYMMETRIC,PETSC_TRUE));

  const auto cc=cellCentroids(M);
  PetscInt localDirectedInternal=0,localInlet=0,localWall=0,localOutlet=0;
  double localCoeffMin=PETSC_MAX_REAL,localCoeffMax=0.0,localOutMin=PETSC_MAX_REAL,localOutMax=0.0;
  for(PetscInt K=0;K<nc;++K) if(D.cellOwner[(std::size_t)K]==rank) {
    const PetscInt row=D.pGid[(std::size_t)K];
    double diagK=0.0;
    std::array<PetscInt,5> cols{}; std::array<PetscScalar,5> vals{};
    PetscInt ncol=1; cols[0]=row; vals[0]=0.0;
    for(int i=0;i<4;++i) {
      const PetscInt f=M.oppFace[(std::size_t)K][i];
      const Vec3 S=gate4FaceAreaVector(M,f);
      if(f<ni) {
        const PetscInt L=(M.owner[(std::size_t)f]==K)?M.neighbour[(std::size_t)f]:M.owner[(std::size_t)f];
        const double a=gate4GeomCoeff(S,sub3(cc[(std::size_t)L],cc[(std::size_t)K]));
        diagK+=a; cols[(std::size_t)ncol]=D.pGid[(std::size_t)L]; vals[(std::size_t)ncol]=(PetscScalar)(-a); ++ncol;
        ++localDirectedInternal; localCoeffMin=std::min(localCoeffMin,a); localCoeffMax=std::max(localCoeffMax,a);
      } else {
        const int pi=M.facePatch[(std::size_t)f];
        if(pi==P.boundary.outlet) {
          const double a=gate4GeomCoeff(S,sub3(gate4FaceCentre(M,f),cc[(std::size_t)K]));
          diagK+=a; ++localOutlet; localOutMin=std::min(localOutMin,a); localOutMax=std::max(localOutMax,a);
          localCoeffMin=std::min(localCoeffMin,a); localCoeffMax=std::max(localCoeffMax,a);
        } else if(pi==P.boundary.inlet) ++localInlet;
        else if(isWallPatch(P.boundary,pi)) ++localWall;
        else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate-4 Kp encountered unclassified pressure boundary face");
      }
    }
    vals[0]=(PetscScalar)diagK;
    if(!(diagK>0.0) || !std::isfinite(diagK)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"Gate-4 Kp row has non-positive/non-finite diagonal");
    PetscCall(MatSetValues(G4.Kp,1,&row,ncol,cols.data(),vals.data(),INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(G4.Kp,MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(G4.Kp,MAT_FINAL_ASSEMBLY));

  PetscInt globalDirectedInternal=0;
  PetscCallMPI(MPI_Allreduce(&localDirectedInternal,&globalDirectedInternal,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localInlet,&G4.inletFaces,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localWall,&G4.wallFaces,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localOutlet,&G4.outletFaces,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  if(globalDirectedInternal%2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate-4 Kp directed internal-face count is odd");
  G4.internalFaces=globalDirectedInternal/2;
  double gCoeffMin=0.0,gCoeffMax=0.0,gOutMin=0.0,gOutMax=0.0;
  PetscCallMPI(MPI_Allreduce(&localCoeffMin,&gCoeffMin,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localCoeffMax,&gCoeffMax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localOutMin,&gOutMin,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localOutMax,&gOutMax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  G4.coeffMin=(PetscReal)gCoeffMin; G4.coeffMax=(PetscReal)gCoeffMax; G4.outletCoeffMin=(PetscReal)gOutMin; G4.outletCoeffMax=(PetscReal)gOutMax;

  MatInfo info{}; PetscCall(MatGetInfo(G4.Kp,MAT_GLOBAL_SUM,&info)); G4.nnz=(PetscInt)(info.nz_used+0.5);
  PetscCall(MatIsSymmetric(G4.Kp,G4.symmetryTol,&G4.symmetric));
  PetscCall(MatCreateVecs(G4.Kp,&G4.ones,&G4.kpOnes)); PetscCall(VecDuplicate(G4.ones,&G4.test));
  PetscCall(VecDuplicate(G4.ones,&G4.kpTest)); PetscCall(VecDuplicate(G4.ones,&G4.diag));
  PetscCall(VecSet(G4.ones,1.0)); PetscCall(MatMult(G4.Kp,G4.ones,G4.kpOnes)); PetscCall(VecNorm(G4.kpOnes,NORM_2,&G4.constantActionNorm));
  PetscInt imin=-1,imax=-1; PetscCall(VecMin(G4.kpOnes,&imin,&G4.constantActionMin)); PetscCall(VecMax(G4.kpOnes,&imax,&G4.constantActionMax));
  PetscCall(MatGetDiagonal(G4.Kp,G4.diag)); PetscCall(VecMin(G4.diag,&imin,&G4.diagMin)); PetscCall(VecMax(G4.diag,&imax,&G4.diagMax));

  PetscInt xs=0,xe=0; PetscCall(VecGetOwnershipRange(G4.test,&xs,&xe)); PetscScalar *xa=nullptr; PetscCall(VecGetArray(G4.test,&xa));
  for(PetscInt g=xs;g<xe;++g) { const double gd=(double)(g+1); xa[(std::size_t)(g-xs)]=(PetscScalar)(std::sin(0.017*gd)+0.31*std::cos(0.031*gd)); }
  PetscCall(VecRestoreArray(G4.test,&xa)); PetscCall(MatMult(G4.Kp,G4.test,G4.kpTest));
  PetscScalar dot=0.0; PetscCall(VecDot(G4.test,G4.kpTest,&dot)); G4.testEnergy=(PetscReal)PetscRealPart(dot); PetscCall(VecNorm(G4.test,NORM_2,&G4.testNorm));
  G4.built=PETSC_TRUE;

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_GATE4_KP_SETUP rows=%" PetscInt_FMT " localRowsRank0=%" PetscInt_FMT " nnz=%" PetscInt_FMT " expectedNnz=%" PetscInt_FMT " avgNnzPerRow=%.6f internalFaces=%" PetscInt_FMT " inletFaces=%" PetscInt_FMT " wallFaces=%" PetscInt_FMT " outletFaces=%" PetscInt_FMT " coeffMin=%.12e coeffMax=%.12e outletCoeffMin=%.12e outletCoeffMax=%.12e source=geometric_FV_laplacian_area2_over_absSdotd\n",
    G4.globalSize,G4.localSize,G4.nnz,G4.expectedNnz,(double)G4.nnz/(double)PetscMax(G4.globalSize,(PetscInt)1),G4.internalFaces,G4.inletFaces,G4.wallFaces,G4.outletFaces,(double)G4.coeffMin,(double)G4.coeffMax,(double)G4.outletCoeffMin,(double)G4.outletCoeffMax));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_GATE4_KP_BC inlet=homogeneous_Neumann wall=homogeneous_Neumann outlet=homogeneous_Dirichlet_p0 pressureNullspace=OFF outletAnchorFaces=%" PetscInt_FMT " constantActionNorm=%.12e constantActionMin=%.12e constantActionMax=%.12e\n",
    G4.outletFaces,(double)G4.constantActionNorm,(double)G4.constantActionMin,(double)G4.constantActionMax));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_GATE4_KP_AUDIT symmetric=%d symmetryTol=%.3e diagMin=%.12e diagMax=%.12e testNorm=%.12e testEnergy=%.12e liveSolveTouched=0 GAMG=NOT_ATTACHED_YET Fp=NONE\n",
    (int)G4.symmetric,(double)G4.symmetryTol,(double)G4.diagMin,(double)G4.diagMax,(double)G4.testNorm,(double)G4.testEnergy));
  PetscFunctionReturn(PETSC_SUCCESS);
}


// -----------------------------------------------------------------------------
// PCD Gate 5: standalone Kp^{-1} probe with PETSc CG+GAMG.
// SHADOW ONLY.  G4.test is manufactured p_* and G4.kpTest=Kp*p_*.
// After the tight CG+GAMG solve, reuse the SAME GAMG PC directly for four
// residual-correction V-cycles.  That no-outer-Krylov sequence is diagnostic
// only in Gate 5; it does not alter the live SIMPLE pressure solve.
// -----------------------------------------------------------------------------
struct Gate5KpGamgStats {
  PetscBool ran=PETSC_FALSE,cgPass=PETSC_FALSE,cyclesFinite=PETSC_FALSE;
  PetscInt cgIts=0;
  KSPConvergedReason reason=KSP_CONVERGED_ITERATING;
  PetscReal trueRel=PETSC_MAX_REAL,solutionRel=PETSC_MAX_REAL;
  PetscReal cycleRel[4]={PETSC_MAX_REAL,PETSC_MAX_REAL,PETSC_MAX_REAL,PETSC_MAX_REAL};
  PetscReal setupSeconds=0.0,solveSeconds=0.0;
};

static PetscErrorCode gate5StandaloneKpGamg(Gate4KpCtx& G4,Gate5KpGamgStats& G5) {
  PetscFunctionBeginUser;
  if(!G4.built || !G4.Kp) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate-5 requires validated Gate-4 Kp");
  KSP ksp=nullptr; PC pc=nullptr;
  Vec sol=nullptr,diff=nullptr,res=nullptr,kpSol=nullptr,cycleX=nullptr,cycleR=nullptr,cycleZ=nullptr;
  PetscLogDouble t0=0,t1=0;
  PetscCall(VecDuplicate(G4.test,&sol)); PetscCall(VecDuplicate(G4.test,&diff));
  PetscCall(VecDuplicate(G4.test,&res)); PetscCall(VecDuplicate(G4.test,&kpSol));
  PetscCall(VecDuplicate(G4.test,&cycleX)); PetscCall(VecDuplicate(G4.test,&cycleR)); PetscCall(VecDuplicate(G4.test,&cycleZ));
  PetscCall(VecSet(sol,0.0));

  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetOptionsPrefix(ksp,"gate5_kp_"));
  PetscCall(KSPSetOperators(ksp,G4.Kp,G4.Kp));
  PetscCall(KSPSetType(ksp,KSPCG));
  PetscCall(KSPSetTolerances(ksp,1.0e-10,0.0,PETSC_DEFAULT,200));
  PetscCall(KSPGetPC(ksp,&pc)); PetscCall(PCSetType(pc,PCGAMG));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(PetscTime(&t0)); PetscCall(KSPSetUp(ksp)); PetscCall(PetscTime(&t1)); G5.setupSeconds=(PetscReal)(t1-t0);

  PetscCall(PetscTime(&t0)); PetscCall(KSPSolve(ksp,G4.kpTest,sol)); PetscCall(PetscTime(&t1)); G5.solveSeconds=(PetscReal)(t1-t0);
  PetscCall(KSPGetIterationNumber(ksp,&G5.cgIts)); PetscCall(KSPGetConvergedReason(ksp,&G5.reason));
  PetscCall(MatMult(G4.Kp,sol,kpSol)); PetscCall(VecWAXPY(res,-1.0,kpSol,G4.kpTest));
  PetscReal bnorm=0.0,rnorm=0.0,xnorm=0.0,dnorm=0.0;
  PetscCall(VecNorm(G4.kpTest,NORM_2,&bnorm)); PetscCall(VecNorm(res,NORM_2,&rnorm));
  PetscCall(VecWAXPY(diff,-1.0,G4.test,sol)); PetscCall(VecNorm(G4.test,NORM_2,&xnorm)); PetscCall(VecNorm(diff,NORM_2,&dnorm));
  G5.trueRel=rnorm/PetscMax(bnorm,(PetscReal)1.0e-300);
  G5.solutionRel=dnorm/PetscMax(xnorm,(PetscReal)1.0e-300);
  G5.cgPass=(G5.reason>0 && G5.trueRel<=1.0e-8 && G5.solutionRel<=1.0e-7) ? PETSC_TRUE : PETSC_FALSE;

  // Direct GAMG-only experiment: x_{j+1}=x_j + PC_GAMG(r_j), r=b-Kp*x.
  PetscCall(VecSet(cycleX,0.0)); PetscCall(VecCopy(G4.kpTest,cycleR));
  PetscReal prevRel=1.0; G5.cyclesFinite=PETSC_TRUE;
  for(int j=0;j<4;++j) {
    PetscCall(PCApply(pc,cycleR,cycleZ));
    PetscCall(VecAXPY(cycleX,1.0,cycleZ));
    PetscCall(MatMult(G4.Kp,cycleX,kpSol)); PetscCall(VecWAXPY(cycleR,-1.0,kpSol,G4.kpTest));
    PetscCall(VecNorm(cycleR,NORM_2,&rnorm)); G5.cycleRel[j]=rnorm/PetscMax(bnorm,(PetscReal)1.0e-300);
    if(!std::isfinite((double)G5.cycleRel[j])) G5.cyclesFinite=PETSC_FALSE;
    const PetscReal contraction=(prevRel>0.0)?G5.cycleRel[j]/prevRel:PETSC_MAX_REAL;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_GATE5_GAMG_ONLY_CYCLE cycle=%d relResidual=%.12e contraction=%.12e semantics=direct_PCApply_one_GAMG_Vcycle_no_outer_Krylov diagnosticOnly=1\n",
      j+1,(double)G5.cycleRel[j],(double)contraction));
    prevRel=G5.cycleRel[j];
  }
  G5.ran=PETSC_TRUE;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_GATE5_KP_GAMG_SOLVE ksp=cg pc=gamg reason=%d its=%" PetscInt_FMT " trueRel=%.12e solutionRel=%.12e setupSeconds=%.6f solveSeconds=%.6f manufacturedRhs=Kp_times_pstar liveSolveTouched=0\n",
    (int)G5.reason,G5.cgIts,(double)G5.trueRel,(double)G5.solutionRel,(double)G5.setupSeconds,(double)G5.solveSeconds));

  PetscCall(VecDestroy(&sol)); PetscCall(VecDestroy(&diff)); PetscCall(VecDestroy(&res)); PetscCall(VecDestroy(&kpSol));
  PetscCall(VecDestroy(&cycleX)); PetscCall(VecDestroy(&cycleR)); PetscCall(VecDestroy(&cycleZ)); PetscCall(KSPDestroy(&ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}



// -----------------------------------------------------------------------------
// PCD Gate 7: pressure-space INTERNAL convection C_p(w), SHADOW ONLY.
//
// The current Picard/Oseen P1+BF3 velocity field is already available in the
// custom momentum owned/ghost halo.  On an internal triangular face the exact
// mean trace is
//
//   wbar_f = (w_v0+w_v1+w_v2)/3 + (9/20) w_BF3,f,
//
// because the BF3 trace is 27 lambda1 lambda2 lambda3 and its face mean is 9/20.
// Thus phi_f = S_f . wbar_f is the exact integrated normal flux of the current
// FE advecting velocity through the face.
//
// Gate 7 uses conservative CENTRAL pressure transport on INTERNAL faces only:
//   (Cp p)_P +=  phi_f (p_P+p_N)/2
//   (Cp p)_N += -phi_f (p_P+p_N)/2
// with S oriented owner P -> neighbour N.  This is +div(w p), equivalent to
// +w.grad(p) for divergence-free w.  Pressure-boundary convection/Robin terms
// are deliberately OFF until Gate 8.  Cp never enters the live pressure solve.
// -----------------------------------------------------------------------------
struct Gate7CpCtx {
  Mat Cp=nullptr;
  Vec ones=nullptr,cpOnes=nullptr,pMass=nullptr,cpMass=nullptr,diffMass=nullptr,fpInteriorMass=nullptr;
  PetscInt setupCount=0,updateCount=0,probeCount=0,internalFaces=0;
  PetscInt lastNonzeroFluxFaces=0,maxNonzeroFluxFaces=0;
  PetscReal lastFluxMin=0.0,lastFluxMax=0.0,lastFluxAbsMax=0.0,lastFluxL1=0.0,lastFluxL2=0.0,maxFluxAbs=0.0;
  PetscReal lastCpOneNorm=0.0,lastCpMassNorm=0.0,lastDiffMassNorm=0.0,lastFpInteriorNorm=0.0;
  PetscReal lastConvToDiff=0.0,maxConvToDiff=0.0,maxCpMassNorm=0.0;
  PetscBool lastSymmetric=PETSC_TRUE,sawNonsymmetric=PETSC_FALSE,allFinite=PETSC_TRUE;
};

static PetscErrorCode gate7EntityVelocity(const Discrete& D,const CustomMomentumCSR& A,PetscInt entity,double v[3]) {
  PetscFunctionBeginUser;
  if(entity<0 || entity>=(PetscInt)D.g2free.size()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Gate-7 velocity entity out of range");
  const PetscInt gid=D.g2free[(std::size_t)entity];
  if(gid>=0) {
    const PetscInt li=customMomentumLocalIndex(A,gid);
    if(li<0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate-7 Picard velocity entity absent from custom momentum owned/ghost halo");
    for(int d=0;d<3;++d) v[d]=customMomentumFieldValue(A,d,li);
  } else {
    for(int d=0;d<3;++d) v[d]=entityDirValue(D,d,entity);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode gate7FaceMeanVelocity(const Mesh& M,const Discrete& D,const CustomMomentumCSR& A,PetscInt f,double vbar[3]) {
  PetscFunctionBeginUser;
  const auto& F=M.faces[(std::size_t)f];
  if(F.v.size()!=3) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Gate-7 requires triangular tetrahedral faces");
  double vv[3][3]={{0}},vb[3]={0};
  for(int j=0;j<3;++j) PetscCall(gate7EntityVelocity(D,A,(PetscInt)F.v[(std::size_t)j],vv[j]));
  PetscCall(gate7EntityVelocity(D,A,(PetscInt)M.points.size()+f,vb));
  for(int d=0;d<3;++d) vbar[d]=(vv[0][d]+vv[1][d]+vv[2][d])/3.0+(9.0/20.0)*vb[d];
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode gate7CpSetUp(const Gate4KpCtx& G4,Gate7CpCtx& G7) {
  PetscFunctionBeginUser;
  if(!G4.built || !G4.Kp) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate-7 requires validated Gate-4 Kp");
  PetscCall(MatDuplicate(G4.Kp,MAT_DO_NOT_COPY_VALUES,&G7.Cp));
  PetscCall(MatSetOption(G7.Cp,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));
  PetscCall(MatCreateVecs(G7.Cp,&G7.pMass,&G7.cpMass));
  PetscCall(VecDuplicate(G7.pMass,&G7.diffMass)); PetscCall(VecDuplicate(G7.pMass,&G7.fpInteriorMass));
  PetscCall(VecDuplicate(G7.pMass,&G7.ones)); PetscCall(VecDuplicate(G7.pMass,&G7.cpOnes)); PetscCall(VecSet(G7.ones,1.0));
  G7.setupCount++;
  MatInfo info{}; PetscCall(MatGetInfo(G7.Cp,MAT_GLOBAL_SUM,&info));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_GATE7_CP_SETUP setupCount=%" PetscInt_FMT " rows=%" PetscInt_FMT " allocatedPatternNnz=%.0f pattern=diag_plus_internal_face_neighbours sourceVelocity=current_Picard_P1plusBF3 exactFaceMean=P1_vertex_mean_plus_9over20_BF3 boundaryConvection=OFF liveSolveTouched=0\n",
    G7.setupCount,G4.globalSize,(double)info.nz_allocated));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode gate7UpdateCpInterior(const Mesh& M,const Discrete& D,const ProblemConfig& P,int rank,
  const CustomMomentumCSR& A,Gate7CpCtx& G7,PetscInt simpleIt) {
  PetscFunctionBeginUser;
  if(!G7.Cp) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate-7 Cp not set up");
  if(P.mode==ProblemMode::MMS) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Gate-7 Cp probe requires pipe/flow mode");
  const PetscInt nc=(PetscInt)M.tets.size(),ni=(PetscInt)M.neighbour.size();
  const auto cc=cellCentroids(M);
  PetscCall(MatZeroEntries(G7.Cp));
  PetscInt localCanonicalFaces=0,localNonzero=0;
  double localMin=PETSC_MAX_REAL,localMax=-PETSC_MAX_REAL,localAbsMax=0.0,localL1=0.0,localL2sq=0.0;
  for(PetscInt K=0;K<nc;++K) if(D.cellOwner[(std::size_t)K]==rank) {
    const PetscInt row=D.pGid[(std::size_t)K];
    double diag=0.0; std::array<PetscInt,5> cols{}; std::array<PetscScalar,5> vals{}; PetscInt ncol=1;
    cols[0]=row; vals[0]=0.0;
    for(int i=0;i<4;++i) {
      const PetscInt f=M.oppFace[(std::size_t)K][i];
      if(f>=ni) continue; // Gate 7: ALL pressure boundary convection is OFF.
      const PetscInt Pcell=M.owner[(std::size_t)f],Ncell=M.neighbour[(std::size_t)f];
      const PetscInt L=(Pcell==K)?Ncell:Pcell;
      Vec3 S=gate4FaceAreaVector(M,f); const Vec3 dPN=sub3(cc[(std::size_t)Ncell],cc[(std::size_t)Pcell]);
      if(S.x*dPN.x+S.y*dPN.y+S.z*dPN.z<0.0) { S.x=-S.x; S.y=-S.y; S.z=-S.z; }
      double wbar[3]={0,0,0}; PetscCall(gate7FaceMeanVelocity(M,D,A,f,wbar));
      const double phi=S.x*wbar[0]+S.y*wbar[1]+S.z*wbar[2];
      if(!std::isfinite(phi)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"Gate-7 non-finite internal pressure convection flux");
      const double outward=(K==Pcell)?phi:-phi, c=0.5*outward;
      diag+=c; cols[(std::size_t)ncol]=D.pGid[(std::size_t)L]; vals[(std::size_t)ncol]=(PetscScalar)c; ++ncol;
      if(K==Pcell) {
        ++localCanonicalFaces; localMin=std::min(localMin,phi); localMax=std::max(localMax,phi);
        localAbsMax=std::max(localAbsMax,std::abs(phi)); localL1+=std::abs(phi); localL2sq+=phi*phi;
        if(std::abs(phi)>1.0e-14) ++localNonzero;
      }
    }
    vals[0]=(PetscScalar)diag;
    PetscCall(MatSetValues(G7.Cp,1,&row,ncol,cols.data(),vals.data(),INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(G7.Cp,MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(G7.Cp,MAT_FINAL_ASSEMBLY));
  PetscInt globalFaces=0,globalNonzero=0; double gMin=0.0,gMax=0.0,gAbsMax=0.0,gL1=0.0,gL2sq=0.0;
  PetscCallMPI(MPI_Allreduce(&localCanonicalFaces,&globalFaces,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localNonzero,&globalNonzero,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localMin,&gMin,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD)); PetscCallMPI(MPI_Allreduce(&localMax,&gMax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localAbsMax,&gAbsMax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD)); PetscCallMPI(MPI_Allreduce(&localL1,&gL1,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localL2sq,&gL2sq,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  if(globalFaces!=ni) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate-7 internal-face diagnostic count mismatch");
  G7.internalFaces=globalFaces; G7.lastNonzeroFluxFaces=globalNonzero; G7.maxNonzeroFluxFaces=PetscMax(G7.maxNonzeroFluxFaces,globalNonzero);
  G7.lastFluxMin=(PetscReal)gMin; G7.lastFluxMax=(PetscReal)gMax; G7.lastFluxAbsMax=(PetscReal)gAbsMax; G7.lastFluxL1=(PetscReal)gL1; G7.lastFluxL2=(PetscReal)std::sqrt(gL2sq);
  G7.maxFluxAbs=PetscMax(G7.maxFluxAbs,G7.lastFluxAbsMax);
  PetscCall(MatMult(G7.Cp,G7.ones,G7.cpOnes)); PetscCall(VecNorm(G7.cpOnes,NORM_2,&G7.lastCpOneNorm));
  PetscCall(MatIsSymmetric(G7.Cp,1.0e-13,&G7.lastSymmetric)); if(!G7.lastSymmetric) G7.sawNonsymmetric=PETSC_TRUE;
  if(!std::isfinite((double)G7.lastFluxAbsMax) || !std::isfinite((double)G7.lastCpOneNorm)) G7.allFinite=PETSC_FALSE;
  G7.updateCount++;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_GATE7_CP_UPDATE it=%" PetscInt_FMT " updateCount=%" PetscInt_FMT " internalFaces=%" PetscInt_FMT " nonzeroFluxFaces=%" PetscInt_FMT " fluxMin=%.12e fluxMax=%.12e fluxAbsMax=%.12e fluxL1=%.12e fluxL2=%.12e CpOneNorm=%.12e symmetric=%d centralFacePressure=arithmetic_mean boundaryConvection=OFF liveSolveTouched=0\n",
    simpleIt,G7.updateCount,G7.internalFaces,G7.lastNonzeroFluxFaces,(double)G7.lastFluxMin,(double)G7.lastFluxMax,(double)G7.lastFluxAbsMax,(double)G7.lastFluxL1,(double)G7.lastFluxL2,(double)G7.lastCpOneNorm,(int)G7.lastSymmetric));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode gate7ProbeActualPressureRhs(const Gate4KpCtx& G4,Gate7CpCtx& G7,Vec volumes,PetscReal nu,Vec rhs,PetscInt simpleIt) {
  PetscFunctionBeginUser;
  PetscCall(VecPointwiseDivide(G7.pMass,rhs,volumes)); PetscCall(MatMult(G7.Cp,G7.pMass,G7.cpMass));
  PetscCall(MatMult(G4.Kp,G7.pMass,G7.diffMass)); PetscCall(VecScale(G7.diffMass,nu));
  PetscCall(VecWAXPY(G7.fpInteriorMass,1.0,G7.cpMass,G7.diffMass));
  PetscCall(VecNorm(G7.cpMass,NORM_2,&G7.lastCpMassNorm)); PetscCall(VecNorm(G7.diffMass,NORM_2,&G7.lastDiffMassNorm)); PetscCall(VecNorm(G7.fpInteriorMass,NORM_2,&G7.lastFpInteriorNorm));
  G7.lastConvToDiff=G7.lastCpMassNorm/PetscMax(G7.lastDiffMassNorm,(PetscReal)1.0e-300); G7.maxConvToDiff=PetscMax(G7.maxConvToDiff,G7.lastConvToDiff); G7.maxCpMassNorm=PetscMax(G7.maxCpMassNorm,G7.lastCpMassNorm);
  if(!std::isfinite((double)G7.lastCpMassNorm) || !std::isfinite((double)G7.lastDiffMassNorm) || !std::isfinite((double)G7.lastFpInteriorNorm) || !std::isfinite((double)G7.lastConvToDiff)) G7.allFinite=PETSC_FALSE;
  G7.probeCount++;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_GATE7_CP_ACTION it=%" PetscInt_FMT " probeCount=%" PetscInt_FMT " CpMpInvRhsNorm=%.12e nuKpMpInvRhsNorm=%.12e FpInteriorMpInvRhsNorm=%.12e convToDiff=%.12e Fp=nuKp_plus_internal_Cp boundaryConvection=OFF KpInverse=NOT_APPLIED_GATE7 liveSolveTouched=0\n",
    simpleIt,G7.probeCount,(double)G7.lastCpMassNorm,(double)G7.lastDiffMassNorm,(double)G7.lastFpInteriorNorm,(double)G7.lastConvToDiff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode gate7CpDestroy(Gate7CpCtx& G7) {
  PetscFunctionBeginUser;
  PetscCall(MatDestroy(&G7.Cp)); PetscCall(VecDestroy(&G7.ones)); PetscCall(VecDestroy(&G7.cpOnes)); PetscCall(VecDestroy(&G7.pMass)); PetscCall(VecDestroy(&G7.cpMass)); PetscCall(VecDestroy(&G7.diffMass)); PetscCall(VecDestroy(&G7.fpInteriorMass));
  PetscFunctionReturn(PETSC_SUCCESS);
}


// -----------------------------------------------------------------------------
// PCD Gate 8: ESW pressure-space boundary-condition audit, SHADOW ONLY.
//
// The supplied ESW screenshot audit records the inflow condition
//
//   -nu * dp/dn + (w_h . n) p = 0.
//
// Gate 7 discretizes pressure convection in conservative flux form.  Therefore
// on an inlet face the explicit conservative boundary convection contribution
// is
//
//   C_in p = + phi_f p,      phi_f = integral_face (w_h . n) dS,
//
// while the Robin diffusion boundary contribution implied by the ESW condition
// is
//
//   R_in p = - phi_f p.
//
// Hence C_in + R_in must cancel exactly.  Gate 8 constructs BOTH pieces
// explicitly from the exact P1+BF3 face-mean advecting velocity and checks that
// cancellation on the actual SIMPLE pressure RHS after M_p^{-1}.  This makes
// the Robin condition explicit without double-counting an inlet term in F_p.
//
// Wall: w.n=0 (no-slip trace), homogeneous Neumann pressure diffusion.
// Outlet: p=0 Dirichlet for K_p/F_p, so the unknown pressure convection trace
//         contributes zero there; the Gate-4 K_p Dirichlet anchor remains.
//
// No Gate-8 object enters the live pressure solve.
// -----------------------------------------------------------------------------
struct Gate8EswBcCtx {
  Vec convDiag=nullptr,robinDiag=nullptr,boundaryDiag=nullptr;
  Vec pMass=nullptr,convAction=nullptr,robinAction=nullptr,boundaryAction=nullptr,fpFull=nullptr,fpDiff=nullptr;
  PetscInt setupCount=0,updateCount=0,probeCount=0;
  PetscInt inletFaces=0,wallFaces=0,outletFaces=0,lastInletNegativeFaces=0;
  PetscReal lastInletFluxMin=0.0,lastInletFluxMax=0.0,lastInletFluxAbsMax=0.0,lastInletFluxSum=0.0;
  PetscReal lastWallFluxAbsMax=0.0,lastOutletFluxAbsMax=0.0;
  PetscReal lastRobinCoeffMin=0.0,lastRobinCoeffMax=0.0;
  PetscReal lastConvNorm=0.0,lastRobinNorm=0.0,lastBoundaryNorm=0.0,lastCancelRel=0.0,maxCancelRel=0.0;
  PetscReal lastFullFpNorm=0.0,lastFullVsInteriorRel=0.0,maxFullVsInteriorRel=0.0;
  PetscReal maxWallFluxAbs=0.0;
  PetscBool inletFluxSignOkay=PETSC_TRUE,allFinite=PETSC_TRUE;
};

static PetscErrorCode gate8EswBcSetUp(const Gate4KpCtx& G4,Gate8EswBcCtx& G8) {
  PetscFunctionBeginUser;
  if(!G4.built || !G4.Kp) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate-8 requires validated Gate-4 Kp");
  PetscCall(MatCreateVecs(G4.Kp,&G8.convDiag,&G8.robinDiag));
  PetscCall(VecDuplicate(G8.convDiag,&G8.boundaryDiag));
  PetscCall(VecDuplicate(G8.convDiag,&G8.pMass));
  PetscCall(VecDuplicate(G8.convDiag,&G8.convAction));
  PetscCall(VecDuplicate(G8.convDiag,&G8.robinAction));
  PetscCall(VecDuplicate(G8.convDiag,&G8.boundaryAction));
  PetscCall(VecDuplicate(G8.convDiag,&G8.fpFull));
  PetscCall(VecDuplicate(G8.convDiag,&G8.fpDiff));
  G8.setupCount++;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_GATE8_ESW_BC_SETUP setupCount=%" PetscInt_FMT " rows=%" PetscInt_FMT " formula=-nu_dpdn_plus_wdotn_p_eq_0 inletImplementation=explicit_conservative_convection_plus_explicit_Robin_cancellation wall=homogeneous_Neumann_and_no_normal_velocity outlet=p0_Dirichlet unknownConvectionTrace=zero attachment=shadow_only liveSolveTouched=0\n",
    G8.setupCount,G4.globalSize));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode gate8UpdateEswBoundary(const Mesh& M,const Discrete& D,const ProblemConfig& P,int rank,
  const CustomMomentumCSR& A,Gate8EswBcCtx& G8,PetscInt simpleIt) {
  PetscFunctionBeginUser;
  const PetscInt nc=(PetscInt)M.tets.size(),ni=(PetscInt)M.neighbour.size();
  PetscCall(VecSet(G8.convDiag,0.0)); PetscCall(VecSet(G8.robinDiag,0.0));
  PetscInt localInlet=0,localWall=0,localOutlet=0,localNeg=0;
  double localInMin=PETSC_MAX_REAL,localInMax=-PETSC_MAX_REAL,localInAbs=0.0,localInSum=0.0;
  double localWallAbs=0.0,localOutAbs=0.0,localRobinMin=PETSC_MAX_REAL,localRobinMax=0.0;
  for(PetscInt K=0;K<nc;++K) if(D.cellOwner[(std::size_t)K]==rank) {
    const PetscInt row=D.pGid[(std::size_t)K];
    double convCell=0.0,robinCell=0.0;
    for(int i=0;i<4;++i) {
      const PetscInt f=M.oppFace[(std::size_t)K][i];
      if(f<ni) continue;
      const int pi=M.facePatch[(std::size_t)f];
      Vec3 S=faceOutwardAreaVector(M,f);
      double wbar[3]={0,0,0}; PetscCall(gate7FaceMeanVelocity(M,D,A,f,wbar));
      const double phi=S.x*wbar[0]+S.y*wbar[1]+S.z*wbar[2];
      if(!std::isfinite(phi)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"Gate-8 non-finite boundary advecting flux");
      if(pi==P.boundary.inlet) {
        // Conservative boundary convection: +phi*p.  ESW Robin contribution:
        // -phi*p from -nu dp/dn + (w.n)p=0.  They must cancel in Fp.
        convCell += phi; robinCell += -phi; ++localInlet;
        localInMin=std::min(localInMin,phi); localInMax=std::max(localInMax,phi);
        localInAbs=std::max(localInAbs,std::abs(phi)); localInSum+=phi;
        if(phi<0.0) ++localNeg;
        const double rc=-phi; localRobinMin=std::min(localRobinMin,rc); localRobinMax=std::max(localRobinMax,rc);
      } else if(isWallPatch(P.boundary,pi)) {
        ++localWall; localWallAbs=std::max(localWallAbs,std::abs(phi));
      } else if(pi==P.boundary.outlet) {
        ++localOutlet; localOutAbs=std::max(localOutAbs,std::abs(phi));
        // p=0: no unknown convection boundary coefficient is inserted.
      } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate-8 encountered unclassified boundary face");
    }
    PetscCall(VecSetValue(G8.convDiag,row,(PetscScalar)convCell,INSERT_VALUES));
    PetscCall(VecSetValue(G8.robinDiag,row,(PetscScalar)robinCell,INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(G8.convDiag)); PetscCall(VecAssemblyEnd(G8.convDiag));
  PetscCall(VecAssemblyBegin(G8.robinDiag)); PetscCall(VecAssemblyEnd(G8.robinDiag));
  PetscCall(VecWAXPY(G8.boundaryDiag,1.0,G8.convDiag,G8.robinDiag));

  PetscCallMPI(MPI_Allreduce(&localInlet,&G8.inletFaces,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localWall,&G8.wallFaces,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localOutlet,&G8.outletFaces,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localNeg,&G8.lastInletNegativeFaces,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  double gInMin=0,gInMax=0,gInAbs=0,gInSum=0,gWallAbs=0,gOutAbs=0,gRobinMin=0,gRobinMax=0;
  PetscCallMPI(MPI_Allreduce(&localInMin,&gInMin,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localInMax,&gInMax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localInAbs,&gInAbs,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localInSum,&gInSum,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localWallAbs,&gWallAbs,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localOutAbs,&gOutAbs,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localRobinMin,&gRobinMin,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localRobinMax,&gRobinMax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  G8.lastInletFluxMin=(PetscReal)gInMin; G8.lastInletFluxMax=(PetscReal)gInMax; G8.lastInletFluxAbsMax=(PetscReal)gInAbs; G8.lastInletFluxSum=(PetscReal)gInSum;
  G8.lastWallFluxAbsMax=(PetscReal)gWallAbs; G8.lastOutletFluxAbsMax=(PetscReal)gOutAbs; G8.maxWallFluxAbs=PetscMax(G8.maxWallFluxAbs,G8.lastWallFluxAbsMax);
  G8.lastRobinCoeffMin=(PetscReal)gRobinMin; G8.lastRobinCoeffMax=(PetscReal)gRobinMax;
  const PetscReal signTol=1.0e-13*PetscMax((PetscReal)1.0,G8.lastInletFluxAbsMax);
  if(G8.lastInletFluxSum>=0.0 || G8.lastInletFluxMax>signTol) G8.inletFluxSignOkay=PETSC_FALSE;
  if(!std::isfinite((double)G8.lastInletFluxSum) || !std::isfinite((double)G8.lastWallFluxAbsMax) || !std::isfinite((double)G8.lastOutletFluxAbsMax)) G8.allFinite=PETSC_FALSE;
  G8.updateCount++;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_GATE8_ESW_BC_UPDATE it=%" PetscInt_FMT " updateCount=%" PetscInt_FMT " inletFaces=%" PetscInt_FMT " inletNegativeFluxFaces=%" PetscInt_FMT " inletFluxMin=%.12e inletFluxMax=%.12e inletFluxAbsMax=%.12e inletFluxSum=%.12e robinCoeffMin=%.12e robinCoeffMax=%.12e wallFaces=%" PetscInt_FMT " wallFluxAbsMax=%.12e outletFaces=%" PetscInt_FMT " outletFluxAbsMax=%.12e inletRobinActive=1 sourceVelocity=current_Picard_P1plusBF3 exactFaceMean=1 liveSolveTouched=0\n",
    simpleIt,G8.updateCount,G8.inletFaces,G8.lastInletNegativeFaces,(double)G8.lastInletFluxMin,(double)G8.lastInletFluxMax,(double)G8.lastInletFluxAbsMax,(double)G8.lastInletFluxSum,(double)G8.lastRobinCoeffMin,(double)G8.lastRobinCoeffMax,G8.wallFaces,(double)G8.lastWallFluxAbsMax,G8.outletFaces,(double)G8.lastOutletFluxAbsMax));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode gate8ProbeActualPressureRhs(const Gate7CpCtx& G7,Gate8EswBcCtx& G8,Vec volumes,Vec rhs,PetscInt simpleIt) {
  PetscFunctionBeginUser;
  if(G7.probeCount<1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate-8 RHS probe requires Gate-7 interior Fp probe first");
  PetscCall(VecPointwiseDivide(G8.pMass,rhs,volumes));
  PetscCall(VecPointwiseMult(G8.convAction,G8.convDiag,G8.pMass));
  PetscCall(VecPointwiseMult(G8.robinAction,G8.robinDiag,G8.pMass));
  PetscCall(VecWAXPY(G8.boundaryAction,1.0,G8.convAction,G8.robinAction));
  PetscCall(VecNorm(G8.convAction,NORM_2,&G8.lastConvNorm)); PetscCall(VecNorm(G8.robinAction,NORM_2,&G8.lastRobinNorm)); PetscCall(VecNorm(G8.boundaryAction,NORM_2,&G8.lastBoundaryNorm));
  const PetscReal scale=PetscMax(PetscMax(G8.lastConvNorm,G8.lastRobinNorm),(PetscReal)1.0e-300);
  G8.lastCancelRel=G8.lastBoundaryNorm/scale; G8.maxCancelRel=PetscMax(G8.maxCancelRel,G8.lastCancelRel);
  PetscCall(VecWAXPY(G8.fpFull,1.0,G8.boundaryAction,G7.fpInteriorMass)); PetscCall(VecNorm(G8.fpFull,NORM_2,&G8.lastFullFpNorm));
  PetscCall(VecCopy(G8.fpFull,G8.fpDiff)); PetscCall(VecAXPY(G8.fpDiff,-1.0,G7.fpInteriorMass));
  PetscReal diffNorm=0.0,interiorNorm=0.0; PetscCall(VecNorm(G8.fpDiff,NORM_2,&diffNorm)); PetscCall(VecNorm(G7.fpInteriorMass,NORM_2,&interiorNorm));
  G8.lastFullVsInteriorRel=diffNorm/PetscMax(interiorNorm,(PetscReal)1.0e-300); G8.maxFullVsInteriorRel=PetscMax(G8.maxFullVsInteriorRel,G8.lastFullVsInteriorRel);
  if(!std::isfinite((double)G8.lastCancelRel) || !std::isfinite((double)G8.lastFullFpNorm) || !std::isfinite((double)G8.lastFullVsInteriorRel)) G8.allFinite=PETSC_FALSE;
  G8.probeCount++;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_GATE8_ESW_BC_ACTION it=%" PetscInt_FMT " probeCount=%" PetscInt_FMT " inletConvectionNorm=%.12e inletRobinNorm=%.12e inletCombinedNorm=%.12e inletCancellationRel=%.12e FpFullMpInvRhsNorm=%.12e fullVsGate7InteriorRel=%.12e semantics=conservative_inlet_convection_plus_ESW_Robin_cancel wall=no_normal_flux outlet=p0_Dirichlet KpInverse=NOT_APPLIED_GATE8 liveSolveTouched=0\n",
    simpleIt,G8.probeCount,(double)G8.lastConvNorm,(double)G8.lastRobinNorm,(double)G8.lastBoundaryNorm,(double)G8.lastCancelRel,(double)G8.lastFullFpNorm,(double)G8.lastFullVsInteriorRel));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode gate8EswBcDestroy(Gate8EswBcCtx& G8) {
  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&G8.convDiag)); PetscCall(VecDestroy(&G8.robinDiag)); PetscCall(VecDestroy(&G8.boundaryDiag));
  PetscCall(VecDestroy(&G8.pMass)); PetscCall(VecDestroy(&G8.convAction)); PetscCall(VecDestroy(&G8.robinAction)); PetscCall(VecDestroy(&G8.boundaryAction)); PetscCall(VecDestroy(&G8.fpFull)); PetscCall(VecDestroy(&G8.fpDiff));
  PetscFunctionReturn(PETSC_SUCCESS);
}


// -----------------------------------------------------------------------------
// PCD Gate 9: FIRST LIVE full PCD pressure preconditioner.
//
// The exact SIMPLE pressure-correction operator remains
//
//      S = B diag(rAU) B^T
//
// and PETSc FGMRES still solves S dp = r.  Only the preconditioner is changed
// from the old native-face GAMG surrogate to the ESW PCD action
//
//      y = K_p^{-1} F_p M_p^{-1} x,
//      F_p = nu K_p + C_p(w) + boundary terms.
//
// Gate 8 showed that, with the conservative pressure convection used here, the
// explicit inlet convection and ESW Robin diffusion terms cancel exactly.  We
// nevertheless apply the explicit Gate-8 boundaryDiag so the live algebra is
// exactly the audited full F_p, not a silently simplified variant.
//
// K_p^{-1} is deliberately tight CG+GAMG in Gate 9.  This is a correctness
// gate, not yet a performance-tuned production PCD implementation.
// -----------------------------------------------------------------------------
struct Gate9LivePcdCtx {
  Mat Kp=nullptr;                 // borrowed Gate-4 geometric pressure Laplacian
  Mat Cp=nullptr;                 // borrowed, retained only so legacy Gate-9 setup wiring stays unchanged
  Vec volumes=nullptr;            // borrowed, retained only for pressure-layout validation
  Vec boundaryDiag=nullptr;       // borrowed, not used by Gate 9C
  PetscReal nu=0.0;
  PC kpPc=nullptr;                // LIVE Gate-9C: one direct GAMG PCApply, no inner Krylov
  Vec kpCheck=nullptr,kpResidual=nullptr;
  PetscInt setupCount=0,applyCount=0,currentSimpleIt=0,outerSolveCount=0;
  PetscBool gamgOnlySolve=PETSC_FALSE; // Gate 9D: no KSPSolve; stationary exact-Schur residual correction only
  PetscReal lastKpCycleRel=PETSC_MAX_REAL,maxKpCycleRel=0.0;
  PetscReal lastInputNorm=0.0,lastOutputNorm=0.0;
  PetscReal lastApplySeconds=0.0,totalApplySeconds=0.0;
  PetscReal lastOuterTrueRel=PETSC_MAX_REAL,maxOuterTrueRel=0.0;
  PetscBool allFinite=PETSC_TRUE;
};

static PetscErrorCode gate9LivePcdSetUp(PC pc) {
  Gate9LivePcdCtx *ctx=nullptr;
  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc,(void**)&ctx));
  if(!ctx) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_NULL,"Gate-9C direct GAMG PCShell context is null");
  if(!ctx->Kp || !ctx->volumes)
    SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_NULL,"Gate-9C requires Kp and the pressure-layout vector");

  if(!ctx->kpCheck) {
    PetscCall(VecDuplicate(ctx->volumes,&ctx->kpCheck));
    PetscCall(VecDuplicate(ctx->volumes,&ctx->kpResidual));
  }
  if(!ctx->kpPc) {
    PetscCall(PCCreate(PetscObjectComm((PetscObject)ctx->Kp),&ctx->kpPc));
    PetscCall(PCSetOptionsPrefix(ctx->kpPc,"gate9c_kp_"));
    PetscCall(PCSetOperators(ctx->kpPc,ctx->Kp,ctx->Kp));
    PetscCall(PCSetType(ctx->kpPc,PCGAMG));
    PetscCall(PCSetFromOptions(ctx->kpPc));
    PetscCall(PCSetUp(ctx->kpPc));
  }

  PetscInt nkpr=0,nkpc=0,nv=0;
  PetscCall(MatGetSize(ctx->Kp,&nkpr,&nkpc)); PetscCall(VecGetSize(ctx->volumes,&nv));
  if(nkpr!=nv || nkpc!=nv)
    SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_SIZ,"Gate-9C Kp/pressure layout mismatch");
  ++ctx->setupCount;
  if(ctx->gamgOnlySolve) {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)pc),
      "P1BF3_GATE9D_GAMG_SETUP setupCount=%" PetscInt_FMT " rows=%" PetscInt_FMT " action=ONE_direct_PCApply_GAMG_on_geometric_Kp_per_stationary_cycle outerKrylov=NONE KSPSolve=NEVER_CALLED exactSchur=UNCHANGED_custom_FP64_B_rAU_Bt\n",
      ctx->setupCount,nv));
  } else {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)pc),
      "P1BF3_GATE9C_GAMG_SETUP setupCount=%" PetscInt_FMT " rows=%" PetscInt_FMT " action=ONE_direct_PCApply_GAMG_on_geometric_Kp innerKrylov=NONE attachment=LIVE_outer_FGMRES_PC exactSchur=UNCHANGED_custom_FP64_B_rAU_Bt\n",
      ctx->setupCount,nv));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode gate9LivePcdApply(PC pc,Vec x,Vec y) {
  Gate9LivePcdCtx *ctx=nullptr;
  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc,(void**)&ctx));
  if(!ctx) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_NULL,"Gate-9C direct GAMG context is null");
  if(!ctx->kpPc || !ctx->kpCheck) PetscCall(gate9LivePcdSetUp(pc));
  PetscLogDouble t0=0,t1=0; PetscCall(PetscTime(&t0));

  // Gate 9C: M23-style live preconditioner.  Apply exactly one GAMG V-cycle
  // built from the clean geometric Kp.  There is NO inner KSP/CG solve.
  PetscCall(PCApply(ctx->kpPc,x,y));

  // Diagnostic only: how well that one V-cycle solves Kp y = x.  This is not
  // a pass/fail condition because the outer FGMRES is responsible for Krylov
  // acceleration, exactly as in the M23-style architecture.
  PetscCall(MatMult(ctx->Kp,y,ctx->kpCheck));
  PetscCall(VecWAXPY(ctx->kpResidual,-1.0,ctx->kpCheck,x));
  PetscReal rhsNorm=0.0,resNorm=0.0;
  PetscCall(VecNorm(x,NORM_2,&rhsNorm)); PetscCall(VecNorm(ctx->kpResidual,NORM_2,&resNorm));
  ctx->lastKpCycleRel=resNorm/PetscMax(rhsNorm,(PetscReal)1.0e-300);
  ctx->maxKpCycleRel=PetscMax(ctx->maxKpCycleRel,ctx->lastKpCycleRel);
  ctx->lastInputNorm=rhsNorm; PetscCall(VecNorm(y,NORM_2,&ctx->lastOutputNorm));
  PetscCall(PetscTime(&t1)); ctx->lastApplySeconds=(PetscReal)(t1-t0); ctx->totalApplySeconds+=ctx->lastApplySeconds; ++ctx->applyCount;
  if(!std::isfinite((double)ctx->lastKpCycleRel) || !std::isfinite((double)ctx->lastOutputNorm) || !std::isfinite((double)ctx->lastApplySeconds)) ctx->allFinite=PETSC_FALSE;
  if(ctx->applyCount<=12 || ctx->applyCount%10==0) {
    if(ctx->gamgOnlySolve) PetscCall(PetscPrintf(PetscObjectComm((PetscObject)pc),
      "P1BF3_GATE9D_GAMG_APPLY simpleIt=%" PetscInt_FMT " applyCount=%" PetscInt_FMT " oneCycleKpRel=%.3e inputNorm=%.3e outputNorm=%.3e applySeconds=%.6e semantics=ONE_direct_PCApply_GAMG_no_inner_Krylov_no_outer_Krylov\n",
      ctx->currentSimpleIt,ctx->applyCount,(double)ctx->lastKpCycleRel,(double)ctx->lastInputNorm,(double)ctx->lastOutputNorm,(double)ctx->lastApplySeconds));
    else PetscCall(PetscPrintf(PetscObjectComm((PetscObject)pc),
      "P1BF3_GATE9C_GAMG_APPLY simpleIt=%" PetscInt_FMT " applyCount=%" PetscInt_FMT " oneCycleKpRel=%.3e inputNorm=%.3e outputNorm=%.3e applySeconds=%.6e semantics=ONE_direct_PCApply_GAMG_no_inner_Krylov\n",
      ctx->currentSimpleIt,ctx->applyCount,(double)ctx->lastKpCycleRel,(double)ctx->lastInputNorm,(double)ctx->lastOutputNorm,(double)ctx->lastApplySeconds));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode gate9LivePcdDestroy(PC pc) {
  Gate9LivePcdCtx *ctx=nullptr;
  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc,(void**)&ctx));
  if(ctx) {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)pc),
      ctx->gamgOnlySolve ?
      "P1BF3_GATE9D_GAMG_DESTROY setupCount=%" PetscInt_FMT " applyCount=%" PetscInt_FMT " pressureSolveCount=%" PetscInt_FMT " maxOneCycleKpRel=%.3e maxTrueRel=%.3e totalApplySeconds=%.6e\n" :
      "P1BF3_GATE9C_GAMG_DESTROY setupCount=%" PetscInt_FMT " applyCount=%" PetscInt_FMT " outerSolveCount=%" PetscInt_FMT " maxOneCycleKpRel=%.3e maxOuterTrueRel=%.3e totalApplySeconds=%.6e\n",
      ctx->setupCount,ctx->applyCount,ctx->outerSolveCount,(double)ctx->maxKpCycleRel,(double)ctx->maxOuterTrueRel,(double)ctx->totalApplySeconds));
    PetscCall(PCDestroy(&ctx->kpPc));
    PetscCall(VecDestroy(&ctx->kpCheck)); PetscCall(VecDestroy(&ctx->kpResidual));
    ctx->Kp=nullptr; ctx->Cp=nullptr; ctx->volumes=nullptr; ctx->boundaryDiag=nullptr;
    delete ctx; PetscCall(PCShellSetContext(pc,nullptr));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// -----------------------------------------------------------------------------
// PCD Gate 6: diffusion-only PCD chain, SHADOW ONLY.
//
// On the actual SIMPLE pressure RHS r, apply
//
//   pMass = M_p^{-1} r,       M_p = diag(cell volume)
//   pFp   = F_p pMass,        F_p = nu K_p   (diffusion-only Gate 6)
//   y     = K_p^{-1} pFp      using tight CG + PETSc GAMG
//
// Therefore the exact chain is y = nu M_p^{-1} r.  Gate 6 checks this identity
// and the explicit K_p solve residual.  The PCShell is shadow-only: its output
// never enters the live SIMPLE pressure solve.
// -----------------------------------------------------------------------------
struct Gate6DiffusionPcdCtx {
  Mat Kp=nullptr;          // borrowed from validated Gate-4 context
  Vec volumes=nullptr;     // borrowed from Discrete; exact P0 pressure mass diagonal
  PetscReal nu=0.0;
  KSP kpKsp=nullptr;
  Vec pMass=nullptr,pFp=nullptr,kpY=nullptr,kpResidual=nullptr,expected=nullptr,diff=nullptr;
  PetscInt setupCount=0,applyCount=0,lastKpIts=0,totalKpIts=0;
  KSPConvergedReason lastKpReason=KSP_CONVERGED_ITERATING;
  PetscReal lastChainRel=PETSC_MAX_REAL,maxChainRel=0.0;
  PetscReal lastKpTrueRel=PETSC_MAX_REAL,maxKpTrueRel=0.0;
  PetscReal lastRhsNorm=0.0,lastMassNorm=0.0,lastFpNorm=0.0,lastOutNorm=0.0;
  PetscBool allKpConverged=PETSC_TRUE;
};

static PetscErrorCode gate6DiffusionPcdSetUp(PC pc) {
  Gate6DiffusionPcdCtx *ctx=nullptr;
  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc,(void**)&ctx));
  if(!ctx) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_NULL,"Gate-6 PCShell context is null");
  if(!ctx->Kp || !ctx->volumes) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_NULL,"Gate-6 requires Kp and P0 mass volumes");
  if(!(ctx->nu>0.0) || !std::isfinite((double)ctx->nu)) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_FP,"Gate-6 viscosity must be positive and finite");

  if(!ctx->pMass) {
    PetscCall(VecDuplicate(ctx->volumes,&ctx->pMass));
    PetscCall(VecDuplicate(ctx->volumes,&ctx->pFp));
    PetscCall(VecDuplicate(ctx->volumes,&ctx->kpY));
    PetscCall(VecDuplicate(ctx->volumes,&ctx->kpResidual));
    PetscCall(VecDuplicate(ctx->volumes,&ctx->expected));
    PetscCall(VecDuplicate(ctx->volumes,&ctx->diff));
  }
  if(!ctx->kpKsp) {
    PetscCall(KSPCreate(PetscObjectComm((PetscObject)ctx->Kp),&ctx->kpKsp));
    PetscCall(KSPSetOptionsPrefix(ctx->kpKsp,"gate6_kp_"));
    PetscCall(KSPSetOperators(ctx->kpKsp,ctx->Kp,ctx->Kp));
    PetscCall(KSPSetType(ctx->kpKsp,KSPCG));
    PetscCall(KSPSetTolerances(ctx->kpKsp,1.0e-10,0.0,PETSC_DEFAULT,200));
    PC kpPc=nullptr; PetscCall(KSPGetPC(ctx->kpKsp,&kpPc));
    PetscCall(PCSetType(kpPc,PCGAMG));
    PetscCall(KSPSetFromOptions(ctx->kpKsp));
    PetscCall(KSPSetUp(ctx->kpKsp));
  }

  PetscInt n=0,nv=0,nkpr=0,nkpc=0;
  PetscCall(VecGetSize(ctx->volumes,&nv));
  PetscCall(MatGetSize(ctx->Kp,&nkpr,&nkpc));
  PetscCall(VecGetSize(ctx->pMass,&n));
  if(n!=nv || nkpr!=nv || nkpc!=nv) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_SIZ,"Gate-6 Kp/Mp pressure layout mismatch");

  ++ctx->setupCount;
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)pc),
    "P1BF3_GATE6_DIFFUSION_PCD_SETUP setupCount=%" PetscInt_FMT " rows=%" PetscInt_FMT " nu=%.12e chain=Kp_inverse_times_Fp_times_Mp_inverse Fp=nu_times_Kp KpInverse=CG_plus_GAMG attachment=shadow_only livePressurePC=UNCHANGED_GAMG\n",
    ctx->setupCount,n,(double)ctx->nu));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode gate6DiffusionPcdApply(PC pc,Vec x,Vec y) {
  Gate6DiffusionPcdCtx *ctx=nullptr;
  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc,(void**)&ctx));
  if(!ctx) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_NULL,"Gate-6 PCShell context is null");
  if(!ctx->kpKsp || !ctx->pMass) PetscCall(gate6DiffusionPcdSetUp(pc));

  // M_p^{-1} r, exact for P0 pressure since M_p(c,c)=cell volume.
  PetscCall(VecPointwiseDivide(ctx->pMass,x,ctx->volumes));

  // F_p M_p^{-1} r with the diffusion-only Gate-6 choice F_p = nu K_p.
  PetscCall(MatMult(ctx->Kp,ctx->pMass,ctx->pFp));
  PetscCall(VecScale(ctx->pFp,ctx->nu));

  // Tight standalone K_p^{-1}; this is still shadow-only and does not alter the
  // live pressure solve or its native-face GAMG hierarchy.
  PetscCall(VecSet(y,0.0));
  PetscCall(KSPSetInitialGuessNonzero(ctx->kpKsp,PETSC_FALSE));
  PetscCall(KSPSolve(ctx->kpKsp,ctx->pFp,y));
  PetscCall(KSPGetConvergedReason(ctx->kpKsp,&ctx->lastKpReason));
  PetscCall(KSPGetIterationNumber(ctx->kpKsp,&ctx->lastKpIts));
  ctx->totalKpIts += ctx->lastKpIts;
  if(ctx->lastKpReason<0) ctx->allKpConverged=PETSC_FALSE;

  // Explicit K_p solve residual: ||F_p M_p^-1 r - K_p y|| / ||F_p M_p^-1 r||.
  PetscCall(MatMult(ctx->Kp,y,ctx->kpY));
  PetscCall(VecCopy(ctx->pFp,ctx->kpResidual));
  PetscCall(VecAXPY(ctx->kpResidual,-1.0,ctx->kpY));
  PetscReal kr=0.0,fn=0.0;
  PetscCall(VecNorm(ctx->kpResidual,NORM_2,&kr));
  PetscCall(VecNorm(ctx->pFp,NORM_2,&fn));
  ctx->lastKpTrueRel=kr/PetscMax(fn,(PetscReal)1e-300);
  ctx->maxKpTrueRel=PetscMax(ctx->maxKpTrueRel,ctx->lastKpTrueRel);

  // Exact diffusion-PCD identity: K_p^-1 (nu K_p) M_p^-1 r = nu M_p^-1 r.
  PetscCall(VecCopy(ctx->pMass,ctx->expected));
  PetscCall(VecScale(ctx->expected,ctx->nu));
  PetscCall(VecCopy(y,ctx->diff));
  PetscCall(VecAXPY(ctx->diff,-1.0,ctx->expected));
  PetscReal dn=0.0,en=0.0;
  PetscCall(VecNorm(ctx->diff,NORM_2,&dn));
  PetscCall(VecNorm(ctx->expected,NORM_2,&en));
  ctx->lastChainRel=dn/PetscMax(en,(PetscReal)1e-300);
  ctx->maxChainRel=PetscMax(ctx->maxChainRel,ctx->lastChainRel);

  PetscCall(VecNorm(x,NORM_2,&ctx->lastRhsNorm));
  PetscCall(VecNorm(ctx->pMass,NORM_2,&ctx->lastMassNorm));
  ctx->lastFpNorm=fn;
  PetscCall(VecNorm(y,NORM_2,&ctx->lastOutNorm));
  ++ctx->applyCount;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode gate6DiffusionPcdDestroy(PC pc) {
  Gate6DiffusionPcdCtx *ctx=nullptr;
  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc,(void**)&ctx));
  if(ctx) {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)pc),
      "P1BF3_GATE6_DIFFUSION_PCD_DESTROY setupCount=%" PetscInt_FMT " applyCount=%" PetscInt_FMT " totalKpIts=%" PetscInt_FMT " maxChainRel=%.3e maxKpTrueRel=%.3e\n",
      ctx->setupCount,ctx->applyCount,ctx->totalKpIts,(double)ctx->maxChainRel,(double)ctx->maxKpTrueRel));
    PetscCall(KSPDestroy(&ctx->kpKsp));
    PetscCall(VecDestroy(&ctx->pMass));
    PetscCall(VecDestroy(&ctx->pFp));
    PetscCall(VecDestroy(&ctx->kpY));
    PetscCall(VecDestroy(&ctx->kpResidual));
    PetscCall(VecDestroy(&ctx->expected));
    PetscCall(VecDestroy(&ctx->diff));
    ctx->Kp=nullptr; ctx->volumes=nullptr;
    delete ctx;
    PetscCall(PCShellSetContext(pc,nullptr));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode gate4DestroyKp(Gate4KpCtx& G4) {
  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&G4.ones)); PetscCall(VecDestroy(&G4.kpOnes)); PetscCall(VecDestroy(&G4.test)); PetscCall(VecDestroy(&G4.kpTest)); PetscCall(VecDestroy(&G4.diag));
  PetscCall(MatDestroy(&G4.Kp)); G4.built=PETSC_FALSE; PetscFunctionReturn(PETSC_SUCCESS);
}

