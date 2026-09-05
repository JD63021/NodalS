// ============================================================================
// NodalS custom pressure aggregation AMG
// ============================================================================
//
// Purpose
// -------
// This file implements the native low-memory multigrid preconditioner used by
// the NodalS pressure-correction solve.  It was developed through roadmap Gates
// 1--5 and is now kept as a normal selectable pressure backend.  The numerical
// fine operator is the exact matrix-free pressure Schur action
//
//                     A_p = B diag(rAU) B^T,
//
// where B is the discontinuous P0 pressure / enriched-P1 velocity divergence
// coupling already assembled by NodalS.  The fine A_p CSR is intentionally
// never materialized by the custom backend.
//
// Architectural boundary
// ----------------------
// The AMG code is deliberately geometry- and boundary-condition agnostic.  It
// does NOT know about pipe radii, inlet profiles, wall patch names, outlet
// conventions, Reynolds number, or SIMPLE/SIMPLEC physics.  It consumes only:
//   * distributed pressure ownership and the pressure adjacency graph,
//   * the exact matrix-free A_p action,
//   * the exact A_p diagonal required by Jacobi/Chebyshev scaling, and
//   * the pressure gauge/nullspace policy supplied by the calling solver.
//
// This makes the module reusable for generic flow geometries while still using
// NodalS-native data structures and MPI exchange helpers.  It is a modular
// NodalS component, not a standalone external AMG library.
//
// Two transfer families
// ---------------------
// 1. Unsmoothed aggregation:
//      P = P_t
//    where every fine pressure cell injects into exactly one connected
//    rank-local aggregate.  This is the minimum-memory production mode.
//
// 2. Smoothed aggregation:
//      P = (I - omega D^{-1} A) P_t,     R = P^T,
//      A_c = P^T A P.
//    The first level is formed directly from the factored B diag(rAU) B^T
//    representation; later levels use explicit custom CSR operators.  No
//    generic PETSc fine-grid PtAP is used.
//
// V-cycle
// -------
// Every non-coarsest level uses a symmetric scaled Chebyshev/Jacobi correction
// before and after coarse correction.  The final small coarse problem is
// gathered and solved by a rank-0 dense LU.  Restriction is the transpose of
// prolongation so the preconditioner remains suitable for native FP64 PCG.
//
// Supported live outer iterations
// -------------------------------
//   * native FP64 PCG + unsmoothed AMG
//   * native FP64 PCG + smoothed AMG
//   * native FP64 Richardson + smoothed AMG
//
// The case-file layer intentionally does not advertise Richardson+unsmoothed:
// that stationary combination failed the large-mesh robustness comparison,
// although the low-level command-line pieces remain useful for diagnostics.
//
// Memory contract
// ---------------
// The live custom path must not allocate an explicit fine Schur CSR, the old
// flat PressureAssemblyPlan, a PETSc pressure Pmat, PETSc GAMG, or PETSc KSP.
// Those objects belong only to the separately selectable PETSc fallback.
// ============================================================================

enum class PressurePCBackendKind {
  PetscFullGAMG,
  CustomAggUnsmoothed,
  CustomAggSmoothed
};

static const char* pressurePCBackendName(PressurePCBackendKind k) {
  switch(k) {
    case PressurePCBackendKind::PetscFullGAMG: return "petsc_full_gamg";
    case PressurePCBackendKind::CustomAggUnsmoothed: return "custom_agg_unsmoothed";
    case PressurePCBackendKind::CustomAggSmoothed: return "custom_agg_smoothed";
  }
  return "unknown";
}

static PetscErrorCode pressurePCBackendParse(const char *name,PressurePCBackendKind *kind) {
  PetscFunctionBeginUser;
  const std::string s=name?std::string(name):std::string();
  if(s=="petsc_full_gamg" || s=="petsc_gamg" || s=="petsc") *kind=PressurePCBackendKind::PetscFullGAMG;
  else if(s=="custom_agg_unsmoothed" || s=="custom_unsmoothed") *kind=PressurePCBackendKind::CustomAggUnsmoothed;
  else if(s=="custom_agg_smoothed" || s=="custom_smoothed") *kind=PressurePCBackendKind::CustomAggSmoothed;
  else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,
    "-pressure_pc_backend must be petsc_full_gamg, custom_agg_unsmoothed, or custom_agg_smoothed");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscBool pressurePCBackendIsCustom(PressurePCBackendKind k) {
  return k==PressurePCBackendKind::PetscFullGAMG?PETSC_FALSE:PETSC_TRUE;
}

using CustomPressureAMGApplyFn = PetscErrorCode (*)(void*,const std::vector<double>&,std::vector<double>&);

struct CustomPressureAMGFineOperator {
  MPI_Comm comm=PETSC_COMM_WORLD;
  PetscInt globalRows=0;
  PetscInt localStart=0;
  PetscInt localEnd=0;
  PetscBool constantNullspace=PETSC_FALSE;
  void *ctx=nullptr;
  CustomPressureAMGApplyFn apply=nullptr;
};

// User-visible AMG controls.  The defaults below are the validated production
// values used in the large-mesh scaling campaign.  The public .case spelling
// lives in [pressure_amg] and is translated to these PETSc options by
// scripts/nodals_case.py.
struct CustomPressureAMGConfig {
  // Preferred number of pressure cells per rank-local connected aggregate.
  PetscInt targetAggregateSize=8;
  // Aggregates smaller than this are merged through a local face connection
  // when possible.  This is a preference/quality control, not a global graph
  // partitioning target.
  PetscInt minAggregateSize=6;
  // Diagnostic soft upper bound.  Connectedness is never sacrificed merely to
  // force this value.
  PetscInt softMaxAggregateSize=10;

  // Degree of each symmetric Chebyshev/Jacobi smoothing correction.
  PetscInt chebyshevDegree=2;
  // Power iterations used to estimate lambda_max(D^{-1}A) on each level.
  PetscInt powerIterations=16;
  // Multiplicative safety factor applied to the power-iteration estimate.
  PetscReal lambdaSafety=1.50;
  // Lower spectral endpoint is approximated as this fraction of lambda_max.
  PetscReal lambdaLowFraction=0.05;

  // Stop recursive aggregation once the global coarse row count is this small
  // and use the final dense coarse solve.
  PetscInt coarseTargetRows=1000;

  // Smoothed-aggregation-only controls.  interpolationMaxRowNnz bounds the
  // retained interpolation stencil after row normalization/pruning.
  PetscInt interpolationMaxRowNnz=8;
  // Jacobi smoothing omega = damping / lambda_max(D^{-1}A).
  PetscReal smoothedAggregationDamping=4.0/3.0;

  // Derived from the selected backend, not normally set directly by a case.
  PetscBool smoothedAggregation=PETSC_FALSE;
};

struct CustomPressureAMGCSR {
  PetscInt globalRows=0;
  PetscInt localStart=0;
  PetscInt localEnd=0;
  std::vector<PetscInt> rowPtr;
  std::vector<PetscInt> colGid;
  std::vector<PetscInt> colLocal;
  std::vector<double> values;
  std::vector<double> diagonal;
  CustomPeerHalo halo;
};

struct CustomPressureAMGTransfer {
  PetscInt coarseGlobalRows=0;
  PetscInt coarseLocalStart=0;
  PetscInt coarseLocalEnd=0;
  std::vector<PetscInt> rowPtr;
  std::vector<PetscInt> colGid;
  std::vector<PetscInt> colLocal;
  std::vector<double> values;
  CustomPeerHalo coarseHalo;
};

struct CustomPressureAMGLevel {
  PetscInt level=0;
  PetscInt globalRows=0;
  PetscInt localStart=0;
  PetscInt localEnd=0;
  PetscInt globalAggregates=0;
  PetscInt localAggregateStart=0;
  PetscInt localAggregateEnd=0;
  std::vector<PetscInt> aggregateId;
  CustomPressureAMGCSR A;
  CustomPressureAMGTransfer P; // transfer from this level to level+1 when smoothed aggregation is active
  std::vector<double> work0,work1,work2;
  double lambdaMax=0.0;
};

struct CustomPressureAMG {
  CustomPressureAMGConfig config;
  CustomPressureAMGFineOperator fine;
  std::vector<CustomPressureAMGLevel> levels;
  PetscBool topologyReady=PETSC_FALSE;
  PetscBool numericReady=PETSC_FALSE;
};

struct CustomPressureAMGExactSchurContext {
  CustomPressureBPlan *B=nullptr;
  const CustomMomentumCSR *M=nullptr;
};

static PetscErrorCode customPressureAMGExactSchurApply(void *raw,const std::vector<double>& x,std::vector<double>& y) {
  PetscFunctionBeginUser;
  auto *ctx=static_cast<CustomPressureAMGExactSchurContext*>(raw);
  if(!ctx || !ctx->B || !ctx->M) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_NULL,"custom AMG exact pressure operator context is incomplete");
  PetscCall(customPressureSchurApply(*ctx->B,*ctx->M,x,y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customPressureAMGInitFineOperator(CustomPressureBPlan& B,const CustomMomentumCSR& M,
  PetscInt globalRows,PetscBool constantNullspace,CustomPressureAMGExactSchurContext& ctx,CustomPressureAMGFineOperator& op) {
  PetscFunctionBeginUser;
  ctx.B=&B; ctx.M=&M;
  op.comm=PETSC_COMM_WORLD;
  op.globalRows=globalRows;
  op.localStart=B.pressureHalo.start;
  op.localEnd=B.pressureHalo.end;
  op.constantNullspace=constantNullspace;
  op.ctx=&ctx;
  op.apply=customPressureAMGExactSchurApply;
  if(op.localEnd-op.localStart!=B.pressureHalo.nOwned) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG pressure ownership descriptor mismatch");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customPressureAMGGate1Probe(CustomPressureBPlan& B,const CustomMomentumCSR& M,
  PetscInt globalRows,PetscBool constantNullspace) {
  PetscFunctionBeginUser;
  CustomPressureAMGExactSchurContext ctx;
  CustomPressureAMGFineOperator op;
  PetscCall(customPressureAMGInitFineOperator(B,M,globalRows,constantNullspace,ctx,op));
  if(!op.apply) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG Gate-1 exact-operator callback missing");

  std::vector<double> x((std::size_t)B.pressureHalo.nOwned,0.0),a,b;
  for(PetscInt i=0;i<B.pressureHalo.nOwned;++i) {
    const double g=(double)(B.pressureHalo.start+i+1);
    x[(std::size_t)i]=std::sin(0.000917*g)+0.31*std::cos(0.001331*g);
  }
  PetscCall(customPressureSchurApply(B,M,x,a));
  PetscCall(op.apply(op.ctx,x,b));
  if(a.size()!=b.size()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"custom AMG Gate-1 operator callback size mismatch");
  double ld2=0.0,lr2=0.0,lmax=0.0;
  for(std::size_t i=0;i<a.size();++i) {
    const double d=a[i]-b[i]; ld2+=d*d; lr2+=a[i]*a[i]; lmax=std::max(lmax,std::abs(d));
  }
  double gd2=0.0,gr2=0.0,gmax=0.0;
  PetscCallMPI(MPI_Allreduce(&ld2,&gd2,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&lr2,&gr2,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&lmax,&gmax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  const double rel=std::sqrt(gd2)/std::max(std::sqrt(gr2),1e-300);
  const PetscBool pass=(rel<=5e-15 && gmax<=5e-15)?PETSC_TRUE:PETSC_FALSE;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_CUSTOM_AMG_GATE1 status=%s backendSeam=READY hierarchy=NOT_BUILT exactOperatorCallback=PASS rel=%.3e maxAbs=%.3e globalRows=%" PetscInt_FMT " localRange=[%" PetscInt_FMT ",%" PetscInt_FMT ") constantNullspace=%d contract=distributed_pressure_range_plus_exact_operator_plus_gauge_only\n",
    pass?"PASS":"FAIL",rel,gmax,op.globalRows,op.localStart,op.localEnd,(int)op.constantNullspace));
  if(!pass) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG Gate-1 exact-operator callback parity failed");
  PetscFunctionReturn(PETSC_SUCCESS);
}


// -----------------------------------------------------------------------------
// Gate 2: rank-local connected tentative aggregation + exact fine diagonal.
// -----------------------------------------------------------------------------
// This remains a topology/operator layer.  No flow geometry, patch name, inlet
// profile, or outlet convention enters the aggregation.  The only mesh facts
// used are cell ownership and cell-to-cell face adjacency.

static PetscErrorCode customPressureAMGBuildFineAggregation(const Mesh& mesh,const Discrete& D,int rank,
  const CustomPressureAMGConfig& cfg,const CustomPressureAMGFineOperator& fine,CustomPressureAMGLevel& L) {
  PetscFunctionBeginUser;
  const PetscInt nOwned=fine.localEnd-fine.localStart;
  if(nOwned<0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG negative local pressure ownership");
  if((std::size_t)rank>=D.cellCount.size() || D.cellCount[(std::size_t)rank]!=nOwned)
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG pressure ownership/cell ownership mismatch");
  if(cfg.targetAggregateSize<2 || cfg.minAggregateSize<1 || cfg.softMaxAggregateSize<cfg.minAggregateSize)
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"custom AMG invalid aggregate-size configuration");

  std::vector<PetscInt> cellOfLocal((std::size_t)nOwned,-1);
  for(PetscInt c=0;c<(PetscInt)mesh.tets.size();++c) if(D.cellOwner[(std::size_t)c]==rank) {
    const PetscInt pl=D.pGid[(std::size_t)c]-fine.localStart;
    if(pl<0 || pl>=nOwned) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG owned cell has pressure GID outside local range");
    if(cellOfLocal[(std::size_t)pl]>=0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG duplicate local pressure cell");
    cellOfLocal[(std::size_t)pl]=c;
  }
  for(PetscInt i=0;i<nOwned;++i) if(cellOfLocal[(std::size_t)i]<0)
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG local pressure cell map is incomplete");

  // Tetrahedra have at most four face neighbours.  Keep this fixed-width and
  // compact instead of building a general graph package.
  std::vector<std::array<PetscInt,4>> nbr((std::size_t)nOwned);
  std::vector<std::uint8_t> degree((std::size_t)nOwned,0);
  for(auto& a:nbr) a={{-1,-1,-1,-1}};
  auto addNbr=[&](PetscInt i,PetscInt j)->PetscErrorCode {
    if(i<0 || i>=nOwned || j<0 || j>=nOwned) return PETSC_ERR_ARG_OUTOFRANGE;
    for(std::uint8_t k=0;k<degree[(std::size_t)i];++k) if(nbr[(std::size_t)i][k]==j) return PETSC_SUCCESS;
    if(degree[(std::size_t)i]>=4) return PETSC_ERR_PLIB;
    nbr[(std::size_t)i][degree[(std::size_t)i]++]=j;
    return PETSC_SUCCESS;
  };
  const PetscInt nInternal=(PetscInt)mesh.neighbour.size();
  for(PetscInt f=0;f<nInternal;++f) {
    const PetscInt c0=mesh.owner[(std::size_t)f],c1=mesh.neighbour[(std::size_t)f];
    if(D.cellOwner[(std::size_t)c0]!=rank || D.cellOwner[(std::size_t)c1]!=rank) continue;
    const PetscInt i=D.pGid[(std::size_t)c0]-fine.localStart;
    const PetscInt j=D.pGid[(std::size_t)c1]-fine.localStart;
    PetscCall(addNbr(i,j)); PetscCall(addNbr(j,i));
  }

  std::vector<PetscInt> aggOf((std::size_t)nOwned,-1);
  std::vector<std::vector<PetscInt>> members;
  members.reserve((std::size_t)((nOwned+cfg.targetAggregateSize-1)/cfg.targetAggregateSize));
  std::vector<PetscInt> queue;
  queue.reserve((std::size_t)std::max<PetscInt>(cfg.targetAggregateSize*2,16));

  // Deterministic rank-local BFS growth.  Seed order is local pressure GID
  // order; only face-connected cells can enter an aggregate.
  for(PetscInt seed=0;seed<nOwned;++seed) if(aggOf[(std::size_t)seed]<0) {
    const PetscInt a=(PetscInt)members.size();
    members.emplace_back();
    queue.clear(); queue.push_back(seed); aggOf[(std::size_t)seed]=a;
    for(std::size_t q=0;q<queue.size() && (PetscInt)members.back().size()<cfg.targetAggregateSize;++q) {
      const PetscInt i=queue[q];
      members.back().push_back(i);
      if((PetscInt)members.back().size()>=cfg.targetAggregateSize) break;
      std::array<PetscInt,4> cand=nbr[(std::size_t)i];
      std::sort(cand.begin(),cand.begin()+degree[(std::size_t)i]);
      for(std::uint8_t k=0;k<degree[(std::size_t)i] && (PetscInt)members.back().size()+(PetscInt)(queue.size()-q-1)<cfg.targetAggregateSize;++k) {
        const PetscInt j=cand[k];
        if(j>=0 && aggOf[(std::size_t)j]<0) { aggOf[(std::size_t)j]=a; queue.push_back(j); }
      }
    }
    // Every queued cell already belongs to this aggregate; append any queued
    // cells not yet consumed when the target was reached.
    for(std::size_t q=members.back().size();q<queue.size();++q) members.back().push_back(queue[q]);
  }

  // Merge undersized aggregates only through an existing local face edge.  A
  // soft maximum is preferred, not enforced, because partition-boundary pieces
  // can otherwise be left as poor singletons.
  for(PetscInt a=0;a<(PetscInt)members.size();++a) {
    if(members[(std::size_t)a].empty() || (PetscInt)members[(std::size_t)a].size()>=cfg.minAggregateSize) continue;
    PetscInt best=-1; PetscBool bestFits=PETSC_FALSE; PetscInt bestSize=std::numeric_limits<PetscInt>::max();
    for(PetscInt i:members[(std::size_t)a]) for(std::uint8_t k=0;k<degree[(std::size_t)i];++k) {
      const PetscInt j=nbr[(std::size_t)i][k],b=(j>=0)?aggOf[(std::size_t)j]:-1;
      if(b<0 || b==a || members[(std::size_t)b].empty()) continue;
      const PetscInt combined=(PetscInt)members[(std::size_t)a].size()+(PetscInt)members[(std::size_t)b].size();
      const PetscBool fits=combined<=cfg.softMaxAggregateSize?PETSC_TRUE:PETSC_FALSE;
      if(best<0 || (fits && !bestFits) || (fits==bestFits && (PetscInt)members[(std::size_t)b].size()<bestSize) ||
         (fits==bestFits && (PetscInt)members[(std::size_t)b].size()==bestSize && b<best)) {
        best=b; bestFits=fits; bestSize=(PetscInt)members[(std::size_t)b].size();
      }
    }
    if(best>=0) {
      for(PetscInt i:members[(std::size_t)a]) { aggOf[(std::size_t)i]=best; members[(std::size_t)best].push_back(i); }
      members[(std::size_t)a].clear();
    }
  }

  std::vector<PetscInt> oldToCompact(members.size(),-1);
  PetscInt localAgg=0;
  for(PetscInt a=0;a<(PetscInt)members.size();++a) if(!members[(std::size_t)a].empty()) oldToCompact[(std::size_t)a]=localAgg++;
  PetscInt aggStart=0;
  PetscCallMPI(MPI_Exscan(&localAgg,&aggStart,1,MPIU_INT,MPI_SUM,fine.comm));
  if(rank==0) aggStart=0;
  PetscInt globalAgg=0; PetscCallMPI(MPI_Allreduce(&localAgg,&globalAgg,1,MPIU_INT,MPI_SUM,fine.comm));
  if(fine.globalRows>0 && globalAgg<=0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG produced no aggregates");

  L.level=0; L.globalRows=fine.globalRows; L.localStart=fine.localStart; L.localEnd=fine.localEnd;
  L.globalAggregates=globalAgg; L.localAggregateStart=aggStart; L.localAggregateEnd=aggStart+localAgg;
  L.aggregateId.assign((std::size_t)nOwned,-1);
  for(PetscInt i=0;i<nOwned;++i) {
    const PetscInt a=aggOf[(std::size_t)i];
    if(a<0 || oldToCompact[(std::size_t)a]<0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG pressure cell is not in a live aggregate");
    L.aggregateId[(std::size_t)i]=aggStart+oldToCompact[(std::size_t)a];
  }

  // Explicitly verify connectedness of every rank-local aggregate.
  std::vector<PetscInt> mark((std::size_t)nOwned,-1),walk;
  PetscInt localDisconnected=0,localMin=std::numeric_limits<PetscInt>::max(),localMax=0,localUnder=0,localOver=0;
  unsigned long long localAssigned=0;
  for(PetscInt a=0;a<(PetscInt)members.size();++a) if(!members[(std::size_t)a].empty()) {
    const PetscInt sz=(PetscInt)members[(std::size_t)a].size(); localAssigned+=(unsigned long long)sz;
    localMin=std::min(localMin,sz); localMax=std::max(localMax,sz);
    if(sz<cfg.minAggregateSize) ++localUnder; if(sz>cfg.softMaxAggregateSize) ++localOver;
    walk.clear(); walk.push_back(members[(std::size_t)a][0]); mark[(std::size_t)walk[0]]=a;
    for(std::size_t q=0;q<walk.size();++q) {
      const PetscInt i=walk[q];
      for(std::uint8_t k=0;k<degree[(std::size_t)i];++k) {
        const PetscInt j=nbr[(std::size_t)i][k];
        if(j>=0 && aggOf[(std::size_t)j]==a && mark[(std::size_t)j]!=a) { mark[(std::size_t)j]=a; walk.push_back(j); }
      }
    }
    if((PetscInt)walk.size()!=sz) ++localDisconnected;
  }
  unsigned long long globalAssigned=0; PetscCallMPI(MPI_Allreduce(&localAssigned,&globalAssigned,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,fine.comm));
  PetscInt globalDisconnected=0,globalMin=0,globalMax=0,globalUnder=0,globalOver=0;
  PetscCallMPI(MPI_Allreduce(&localDisconnected,&globalDisconnected,1,MPIU_INT,MPI_SUM,fine.comm));
  PetscCallMPI(MPI_Allreduce(&localMin,&globalMin,1,MPIU_INT,MPI_MIN,fine.comm));
  PetscCallMPI(MPI_Allreduce(&localMax,&globalMax,1,MPIU_INT,MPI_MAX,fine.comm));
  PetscCallMPI(MPI_Allreduce(&localUnder,&globalUnder,1,MPIU_INT,MPI_SUM,fine.comm));
  PetscCallMPI(MPI_Allreduce(&localOver,&globalOver,1,MPIU_INT,MPI_SUM,fine.comm));
  if(globalAssigned!=(unsigned long long)fine.globalRows || globalDisconnected!=0)
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG aggregate validity/connectivity check failed");

  const double mean=globalAgg?((double)fine.globalRows/(double)globalAgg):0.0;
  const double ratio=globalAgg?((double)fine.globalRows/(double)globalAgg):0.0;
  PetscCall(PetscPrintf(fine.comm,
    "P1BF3_CUSTOM_AMG_GATE2_AGG status=PASS scope=rank_local_connected graph=cell_face target=%" PetscInt_FMT " preferred=[%" PetscInt_FMT ",%" PetscInt_FMT "] fineRows=%" PetscInt_FMT " aggregates=%" PetscInt_FMT " coarsening=%.6f size=[%" PetscInt_FMT ",%.6f,%" PetscInt_FMT "] undersized=%" PetscInt_FMT " softOversized=%" PetscInt_FMT " disconnected=%" PetscInt_FMT " assigned=%llu P0ConstantDefect=0.000e+00 crossRankAggregates=0\n",
    cfg.targetAggregateSize,cfg.minAggregateSize,cfg.softMaxAggregateSize,fine.globalRows,globalAgg,ratio,globalMin,mean,globalMax,globalUnder,globalOver,globalDisconnected,globalAssigned));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customPressureAMGComputeFineDiagonal(CustomPressureBPlan& B,const CustomMomentumCSR& M,
  std::vector<double>& diag) {
  PetscFunctionBeginUser;
  if((PetscInt)M.rAU.size()!=B.velocityHalo.nOwned)
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"custom AMG rAU ownership does not match B velocity ownership");
  PetscCall(customPeerExchange(B.velocityHalo,M.rAU,49321));
  diag.assign((std::size_t)B.pressureHalo.nOwned,0.0);
  for(const auto& cp:B.forwardCells) {
    if(cp.pLocal<0 || cp.pLocal>=B.pressureHalo.nOwned) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG fine diagonal pressure index invalid");
    double s=0.0;
    for(int a=0;a<8;++a) if(cp.velLocal[a]>=0) {
      const double rau=customPeerValue(B.velocityHalo,M.rAU,cp.velLocal[a]);
      if(!std::isfinite(rau) || rau<=0.0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"custom AMG fine diagonal encountered non-positive/non-finite rAU");
      for(int d=0;d<3;++d) { const double b=customPressureBCoeff(cp.vol,cp.gradLambda,d,a); s+=rau*b*b; }
    }
    diag[(std::size_t)cp.pLocal]=s;
  }
  double lmin=std::numeric_limits<double>::infinity(),lmax=0.0; PetscInt lbad=0;
  for(double v:diag) { if(!std::isfinite(v) || v<=0.0) ++lbad; else { lmin=std::min(lmin,v); lmax=std::max(lmax,v); } }
  double gmin=0.0,gmax=0.0; PetscInt gbad=0;
  PetscCallMPI(MPI_Allreduce(&lmin,&gmin,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&lmax,&gmax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&lbad,&gbad,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  if(gbad) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"custom AMG exact fine diagonal is not strictly positive/finite");
  PetscInt globalRows=0; const PetscInt localRows=B.pressureHalo.nOwned;
  PetscCallMPI(MPI_Allreduce(&localRows,&globalRows,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_CUSTOM_AMG_GATE2_DIAG status=PASS source=exact_B_rAU_Bt_formula storage=FP64_owned_rows rows=%" PetscInt_FMT " range=[%.12e,%.12e] bad=%" PetscInt_FMT " explicitFineSchurNeeded=0\n",
    globalRows,gmin,gmax,gbad));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customPressureAMGCompareFineDiagonal(Mat explicitSchur,const CustomPressureBPlan& B,
  const std::vector<double>& diag,double tol) {
  PetscFunctionBeginUser;
  if(!explicitSchur) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_NULL,"custom AMG Gate-2 diagonal parity requires the PETSc fallback Schur oracle");
  Vec dv=nullptr; PetscCall(MatCreateVecs(explicitSchur,nullptr,&dv)); PetscCall(MatGetDiagonal(explicitSchur,dv));
  PetscInt s=0,e=0; PetscCall(VecGetOwnershipRange(dv,&s,&e));
  if(s!=B.pressureHalo.start || e!=B.pressureHalo.end || (PetscInt)diag.size()!=e-s)
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"custom AMG Gate-2 diagonal oracle ownership mismatch");
  const PetscScalar *a=nullptr; PetscCall(VecGetArrayRead(dv,&a));
  double ld2=0.0,lr2=0.0,lmax=0.0,lrefmax=0.0;
  for(PetscInt i=0;i<e-s;++i) {
    const double r=(double)PetscRealPart(a[i]),d=diag[(std::size_t)i]-r;
    ld2+=d*d; lr2+=r*r; lmax=std::max(lmax,std::abs(d)); lrefmax=std::max(lrefmax,std::abs(r));
  }
  PetscCall(VecRestoreArrayRead(dv,&a)); PetscCall(VecDestroy(&dv));
  double gd2=0.0,gr2=0.0,gmax=0.0,grefmax=0.0;
  PetscCallMPI(MPI_Allreduce(&ld2,&gd2,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&lr2,&gr2,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&lmax,&gmax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&lrefmax,&grefmax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  const double rel=std::sqrt(gd2)/std::max(std::sqrt(gr2),1e-300);
  const double maxScaled=gmax/std::max(grefmax,1e-300);
  const PetscBool pass=(rel<=tol && maxScaled<=10.0*tol)?PETSC_TRUE:PETSC_FALSE;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_CUSTOM_AMG_GATE2_DIAG_PARITY status=%s oracle=PETSc_full_Schur_debug_only rel=%.3e maxAbs=%.3e maxScaled=%.3e tol=%.3e customConstructionDependsOnOracle=0\n",
    pass?"PASS":"FAIL",rel,gmax,maxScaled,tol));
  if(!pass) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG exact fine diagonal parity failed");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customPressureAMGGate2Probe(const Mesh& mesh,const Discrete& D,int rank,
  CustomPressureBPlan& B,const CustomMomentumCSR& M,Mat explicitSchur,PetscInt globalRows,PetscBool constantNullspace) {
  PetscFunctionBeginUser;
  CustomPressureAMGExactSchurContext ctx; CustomPressureAMG amg;
  PetscCall(customPressureAMGInitFineOperator(B,M,globalRows,constantNullspace,ctx,amg.fine));
  amg.levels.resize(1);
  PetscCall(customPressureAMGBuildFineAggregation(mesh,D,rank,amg.config,amg.fine,amg.levels[0]));
  PetscCall(customPressureAMGComputeFineDiagonal(B,M,amg.levels[0].A.diagonal));
  const double tol=(sizeof(PetscReal)<=4)?5e-5:5e-12;
  PetscCall(customPressureAMGCompareFineDiagonal(explicitSchur,B,amg.levels[0].A.diagonal,tol));
  amg.topologyReady=PETSC_TRUE; amg.numericReady=PETSC_TRUE;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_CUSTOM_AMG_GATE2 status=PASS roadmapGates=6 fineTopology=READY fineDiagonal=READY hierarchy=NOT_BUILT coarseOperators=NOT_BUILT smoother=NOT_BUILT pressureSolveBackend=PETSc_fallback generalFlowContract=cell_topology_plus_distributed_exact_operator_plus_gauge\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}


// -----------------------------------------------------------------------------
// Gate 3: first explicit Galerkin coarse operator A1 = P0^T A0 P0.
// -----------------------------------------------------------------------------
// A0 is NEVER materialized for this construction.  We exploit
//      A0 = B diag(rAU) B^T
// and the unsmoothed aggregate-ID transfer.  For each owned scalar velocity
// DOF q, pressure cells incident to q are first summed by aggregate:
//      c_I(q,d) = sum_{cell K in aggregate I} B[K,q,d].
// That velocity DOF contributes
//      rAU[q] * sum_d c_I(q,d)c_J(q,d)
// to coarse entry (I,J).  The q owner emits the contribution exactly once;
// contributions are routed to the owner of coarse row I and sort/merged there.
// This remains geometry/BC agnostic: only B, rAU, distributed ownership, and
// aggregate IDs enter the construction.

struct CustomPressureAMGStarEntry {
  PetscInt aggregate=-1;
  double b[3]={0.0,0.0,0.0};
};
struct CustomPressureAMGTriplet {
  PetscInt row=-1,col=-1;
  double value=0.0;
};
struct CustomPressureAMGPairKey {
  PetscInt row=-1,col=-1;
  bool operator==(const CustomPressureAMGPairKey& o) const noexcept { return row==o.row && col==o.col; }
};
struct CustomPressureAMGPairHash {
  std::size_t operator()(const CustomPressureAMGPairKey& k) const noexcept {
    const std::size_t a=std::hash<PetscInt>{}(k.row),b=std::hash<PetscInt>{}(k.col);
    return a^(b+0x9e3779b97f4a7c15ULL+(a<<6)+(a>>2));
  }
};

static PetscErrorCode customPressureAMGExchangeAggregateIds(const CustomPeerHalo& H,
  const std::vector<PetscInt>& owned,std::vector<PetscInt>& ghost,int tag) {
  PetscFunctionBeginUser;
  if((PetscInt)owned.size()!=H.nOwned) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"custom AMG aggregate-ID owned vector size mismatch");
  std::vector<PetscInt> send(H.reqRecvLocalOffset.size(),-1);
  ghost.assign(H.ghostGid.size(),-1);
  for(std::size_t k=0;k<H.reqRecvLocalOffset.size();++k) send[k]=owned[(std::size_t)H.reqRecvLocalOffset[k]];
  const int nr=(int)H.reqSendCounts.size();
  std::vector<MPI_Request> req; req.reserve((std::size_t)(2*nr));
  for(int r=0;r<nr;++r) if(H.reqSendCounts[(std::size_t)r]>0) {
    MPI_Request q=MPI_REQUEST_NULL;
    PetscCallMPI(MPI_Irecv(ghost.data()+H.reqSendDispls[(std::size_t)r],H.reqSendCounts[(std::size_t)r],MPIU_INT,r,tag,PETSC_COMM_WORLD,&q)); req.push_back(q);
  }
  for(int r=0;r<nr;++r) if(H.reqRecvCounts[(std::size_t)r]>0) {
    MPI_Request q=MPI_REQUEST_NULL;
    PetscCallMPI(MPI_Isend(send.data()+H.reqRecvDispls[(std::size_t)r],H.reqRecvCounts[(std::size_t)r],MPIU_INT,r,tag,PETSC_COMM_WORLD,&q)); req.push_back(q);
  }
  if(!req.empty()) PetscCallMPI(MPI_Waitall((int)req.size(),req.data(),MPI_STATUSES_IGNORE));
  for(PetscInt g:ghost) if(g<0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG aggregate-ID halo contains invalid GID");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customPressureAMGCoarseSpMV(CustomPressureAMGCSR& A,
  const std::vector<double>& x,std::vector<double>& y,int tag) {
  PetscFunctionBeginUser;
  const PetscInt n=A.localEnd-A.localStart;
  if((PetscInt)x.size()!=n || (PetscInt)A.rowPtr.size()!=n+1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"custom AMG coarse SpMV size mismatch");
  PetscCall(customPeerExchange(A.halo,x,tag));
  y.assign((std::size_t)n,0.0);
  for(PetscInt i=0;i<n;++i) {
    double s=0.0;
    for(PetscInt k=A.rowPtr[(std::size_t)i];k<A.rowPtr[(std::size_t)i+1];++k)
      s+=A.values[(std::size_t)k]*customPeerValue(A.halo,x,A.colLocal[(std::size_t)k]);
    y[(std::size_t)i]=s;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}



template<class Entry>
static PetscErrorCode customPressureAMGExchangeSparseRows(const CustomPeerHalo& H,
  const std::vector<PetscInt>& rowPtr,const std::vector<Entry>& entries,
  std::vector<PetscInt>& ghostPtr,std::vector<Entry>& ghostEntries,int tagBase) {
  PetscFunctionBeginUser;
  if((PetscInt)rowPtr.size()!=H.nOwned+1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"custom AMG sparse-row exchange rowPtr size mismatch");
  const int nr=(int)H.reqSendCounts.size();
  std::vector<PetscInt> sendLen(H.reqRecvLocalOffset.size(),0),recvLen(H.ghostGid.size(),0);
  for(std::size_t k=0;k<H.reqRecvLocalOffset.size();++k) {
    const PetscInt i=H.reqRecvLocalOffset[k];
    sendLen[k]=rowPtr[(std::size_t)i+1]-rowPtr[(std::size_t)i];
  }
  std::vector<MPI_Request> req; req.reserve((std::size_t)(2*nr));
  for(int r=0;r<nr;++r) if(H.reqSendCounts[(std::size_t)r]>0) {
    MPI_Request q=MPI_REQUEST_NULL;
    PetscCallMPI(MPI_Irecv(recvLen.data()+H.reqSendDispls[(std::size_t)r],H.reqSendCounts[(std::size_t)r],MPIU_INT,r,tagBase,PETSC_COMM_WORLD,&q)); req.push_back(q);
  }
  for(int r=0;r<nr;++r) if(H.reqRecvCounts[(std::size_t)r]>0) {
    MPI_Request q=MPI_REQUEST_NULL;
    PetscCallMPI(MPI_Isend(sendLen.data()+H.reqRecvDispls[(std::size_t)r],H.reqRecvCounts[(std::size_t)r],MPIU_INT,r,tagBase,PETSC_COMM_WORLD,&q)); req.push_back(q);
  }
  if(!req.empty()) PetscCallMPI(MPI_Waitall((int)req.size(),req.data(),MPI_STATUSES_IGNORE));
  ghostPtr.assign(H.ghostGid.size()+1,0);
  for(std::size_t k=0;k<recvLen.size();++k) {
    if(recvLen[k]<0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG sparse-row exchange negative row length");
    ghostPtr[k+1]=ghostPtr[k]+recvLen[k];
  }
  std::vector<int> sendCount((std::size_t)nr,0),recvCount((std::size_t)nr,0),sendDisp((std::size_t)nr,0),recvDisp((std::size_t)nr,0);
  for(int r=0;r<nr;++r) {
    long long a=0,b=0;
    const int rs=H.reqRecvDispls[(std::size_t)r],rn=H.reqRecvCounts[(std::size_t)r];
    for(int k=0;k<rn;++k) a+=(long long)sendLen[(std::size_t)(rs+k)];
    const int ss=H.reqSendDispls[(std::size_t)r],sn=H.reqSendCounts[(std::size_t)r];
    for(int k=0;k<sn;++k) b+=(long long)recvLen[(std::size_t)(ss+k)];
    if(a>std::numeric_limits<int>::max() || b>std::numeric_limits<int>::max()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"custom AMG sparse-row peer payload exceeds MPI int limit");
    sendCount[(std::size_t)r]=(int)a; recvCount[(std::size_t)r]=(int)b;
  }
  for(int r=1;r<nr;++r){sendDisp[(std::size_t)r]=sendDisp[(std::size_t)r-1]+sendCount[(std::size_t)r-1];recvDisp[(std::size_t)r]=recvDisp[(std::size_t)r-1]+recvCount[(std::size_t)r-1];}
  std::vector<Entry> sendEntries((std::size_t)(nr?sendDisp.back()+sendCount.back():0));
  for(int r=0;r<nr;++r) {
    std::size_t pos=(std::size_t)sendDisp[(std::size_t)r];
    const int st=H.reqRecvDispls[(std::size_t)r],nn=H.reqRecvCounts[(std::size_t)r];
    for(int kk=0;kk<nn;++kk) {
      const PetscInt i=H.reqRecvLocalOffset[(std::size_t)(st+kk)];
      for(PetscInt q=rowPtr[(std::size_t)i];q<rowPtr[(std::size_t)i+1];++q) sendEntries[pos++]=entries[(std::size_t)q];
    }
    if(pos!=(std::size_t)(sendDisp[(std::size_t)r]+sendCount[(std::size_t)r])) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG sparse-row send packing mismatch");
  }
  ghostEntries.resize((std::size_t)(nr?recvDisp.back()+recvCount.back():0));
  MPI_Datatype et=MPI_DATATYPE_NULL; PetscCallMPI(MPI_Type_contiguous((int)sizeof(Entry),MPI_BYTE,&et)); PetscCallMPI(MPI_Type_commit(&et));
  PetscCallMPI(MPI_Alltoallv(sendEntries.empty()?nullptr:sendEntries.data(),sendCount.data(),sendDisp.data(),et,
                             ghostEntries.empty()?nullptr:ghostEntries.data(),recvCount.data(),recvDisp.data(),et,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Type_free(&et));
  if((PetscInt)ghostEntries.size()!=ghostPtr.back()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG sparse-row receive payload mismatch");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static unsigned long long customPressureAMGTransferBytes(const CustomPressureAMGTransfer& P) {
  return (unsigned long long)(P.rowPtr.capacity()+P.colGid.capacity()+P.colLocal.capacity())*sizeof(PetscInt)
    +(unsigned long long)P.values.capacity()*sizeof(double)
    +(unsigned long long)(P.coarseHalo.ghostGid.capacity()+P.coarseHalo.reqRecvGid.capacity()+P.coarseHalo.reqRecvLocalOffset.capacity())*sizeof(PetscInt)
    +(unsigned long long)(P.coarseHalo.exchangeSend.capacity()+P.coarseHalo.ghostValues.capacity())*sizeof(double);
}

static PetscErrorCode customPressureAMGFinalizeTransferHalo(CustomPressureAMGLevel& L,CustomPressureAMGTransfer& P,const char *label) {
  PetscFunctionBeginUser;
  int rank=0,size=1; PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank)); PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  const PetscInt localAgg=L.localAggregateEnd-L.localAggregateStart;
  std::vector<PetscInt> counts((std::size_t)size,0),needed;
  PetscCallMPI(MPI_Allgather(&localAgg,1,MPIU_INT,counts.data(),1,MPIU_INT,PETSC_COMM_WORLD));
  needed.reserve(P.colGid.size());
  for(PetscInt g:P.colGid) if(g<L.localAggregateStart || g>=L.localAggregateEnd) needed.push_back(g);
  PetscCall(buildCustomPeerHalo(counts,rank,std::move(needed),P.coarseHalo,label));
  P.coarseGlobalRows=L.globalAggregates; P.coarseLocalStart=L.localAggregateStart; P.coarseLocalEnd=L.localAggregateEnd;
  P.colLocal.resize(P.colGid.size());
  for(std::size_t k=0;k<P.colGid.size();++k) {
    P.colLocal[k]=customPeerLocalIndex(P.coarseHalo,P.colGid[k]);
    if(P.colLocal[k]<0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG smoothed transfer halo missing coarse column");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customPressureAMGTransferProlongAdd(CustomPressureAMGTransfer& P,
  const std::vector<double>& ec,std::vector<double>& xf,int tag) {
  PetscFunctionBeginUser;
  const PetscInt nf=(PetscInt)P.rowPtr.size()-1;
  if((PetscInt)ec.size()!=P.coarseLocalEnd-P.coarseLocalStart || (PetscInt)xf.size()!=nf) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"custom AMG smoothed prolongation size mismatch");
  PetscCall(customPeerExchange(P.coarseHalo,ec,tag));
  for(PetscInt i=0;i<nf;++i) {
    double s=0.0;
    for(PetscInt k=P.rowPtr[(std::size_t)i];k<P.rowPtr[(std::size_t)i+1];++k) s+=P.values[(std::size_t)k]*customPeerValue(P.coarseHalo,ec,P.colLocal[(std::size_t)k]);
    xf[(std::size_t)i]+=s;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customPressureAMGTransferRestrict(CustomPressureAMGTransfer& P,
  const std::vector<double>& rf,std::vector<double>& rc,int tag) {
  PetscFunctionBeginUser;
  const PetscInt nf=(PetscInt)P.rowPtr.size()-1,nc=P.coarseLocalEnd-P.coarseLocalStart;
  if((PetscInt)rf.size()!=nf) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"custom AMG smoothed restriction size mismatch");
  rc.assign((std::size_t)nc,0.0);
  std::vector<double> ghostContrib(P.coarseHalo.ghostGid.size(),0.0);
  for(PetscInt i=0;i<nf;++i) for(PetscInt k=P.rowPtr[(std::size_t)i];k<P.rowPtr[(std::size_t)i+1];++k) {
    const PetscInt li=P.colLocal[(std::size_t)k]; const double v=P.values[(std::size_t)k]*rf[(std::size_t)i];
    if(li<P.coarseHalo.nOwned) rc[(std::size_t)li]+=v; else ghostContrib[(std::size_t)(li-P.coarseHalo.nOwned)]+=v;
  }
  const int nr=(int)P.coarseHalo.reqSendCounts.size();
  std::vector<double> recv(P.coarseHalo.reqRecvGid.size(),0.0); std::vector<MPI_Request> req; req.reserve((std::size_t)(2*nr));
  for(int r=0;r<nr;++r) if(P.coarseHalo.reqRecvCounts[(std::size_t)r]>0) {
    MPI_Request q=MPI_REQUEST_NULL;
    PetscCallMPI(MPI_Irecv(recv.data()+P.coarseHalo.reqRecvDispls[(std::size_t)r],P.coarseHalo.reqRecvCounts[(std::size_t)r],MPI_DOUBLE,r,tag,PETSC_COMM_WORLD,&q)); req.push_back(q);
  }
  for(int r=0;r<nr;++r) if(P.coarseHalo.reqSendCounts[(std::size_t)r]>0) {
    MPI_Request q=MPI_REQUEST_NULL;
    PetscCallMPI(MPI_Isend(ghostContrib.data()+P.coarseHalo.reqSendDispls[(std::size_t)r],P.coarseHalo.reqSendCounts[(std::size_t)r],MPI_DOUBLE,r,tag,PETSC_COMM_WORLD,&q)); req.push_back(q);
  }
  if(!req.empty()) PetscCallMPI(MPI_Waitall((int)req.size(),req.data(),MPI_STATUSES_IGNORE));
  for(std::size_t k=0;k<recv.size();++k) rc[(std::size_t)P.coarseHalo.reqRecvLocalOffset[k]]+=recv[k];
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void customPressureAMGNormalizePruneRow(std::vector<std::pair<PetscInt,double>>& row,
  PetscInt ownAgg,PetscInt maxNnz) {
  if(row.empty()) row.push_back({ownAgg,1.0});
  std::sort(row.begin(),row.end(),[](const auto& a,const auto& b){return a.first<b.first;});
  std::vector<std::pair<PetscInt,double>> merged; merged.reserve(row.size());
  for(const auto& e:row) { if(!merged.empty()&&merged.back().first==e.first) merged.back().second+=e.second; else merged.push_back(e); }
  row.swap(merged);
  if(maxNnz>0 && (PetscInt)row.size()>maxNnz) {
    std::vector<std::pair<PetscInt,double>> keep; keep.reserve((std::size_t)maxNnz);
    auto itOwn=std::find_if(row.begin(),row.end(),[&](const auto& e){return e.first==ownAgg;});
    if(itOwn!=row.end()) keep.push_back(*itOwn); else keep.push_back({ownAgg,0.0});
    std::vector<std::pair<PetscInt,double>> others;
    for(const auto& e:row) if(e.first!=ownAgg) others.push_back(e);
    std::sort(others.begin(),others.end(),[](const auto& a,const auto& b){const double aa=std::abs(a.second),bb=std::abs(b.second);return aa>bb || (aa==bb&&a.first<b.first);});
    for(const auto& e:others) if((PetscInt)keep.size()<maxNnz) keep.push_back(e);
    std::sort(keep.begin(),keep.end(),[](const auto& a,const auto& b){return a.first<b.first;}); row.swap(keep);
  }
  double sum=0.0; for(const auto& e:row) sum+=e.second;
  if(!std::isfinite(sum) || std::abs(sum)<1e-14) { row.clear(); row.push_back({ownAgg,1.0}); return; }
  for(auto& e:row) e.second/=sum;
}

struct CustomPressureAMGCoarseLU {
  PetscInt n=0;
  std::vector<int> counts,displs;
  std::vector<double> lu;          // rank 0 only
  std::vector<PetscInt> piv;       // rank 0 only
  PetscBool constantNullspace=PETSC_FALSE;
  PetscBool ready=PETSC_FALSE;
};

struct CustomPressureAMGRuntime {
  CustomPressureAMG amg;
  CustomPressureAMGExactSchurContext exactCtx;
  CustomPressureAMGCoarseLU coarse;
  PetscInt setupCount=0;
  PetscInt vcycleApplyCount=0;
  double retainedMiB=0.0;
};

// -----------------------------------------------------------------------------
// Smoothed-aggregation transfer construction
// -----------------------------------------------------------------------------
// The fine operator is matrix-free, so the first smoothed transfer cannot be
// built by handing a fine CSR matrix to a generic PtAP routine.  Instead the
// code expands the action of (I - omega D^{-1} B diag(rAU) B^T) P_t directly
// from the B-plan, merges duplicate coarse columns, normalizes each row to
// preserve constants, and prunes to interpolationMaxRowNnz.  A_c is then
// streamed from the same factored representation.  On coarser levels A is
// already a small explicit custom CSR, so the corresponding operations can use
// that CSR directly.
// -----------------------------------------------------------------------------
struct CustomPressureAMGQEntry { PetscInt coarse=-1; double b[3]={0.0,0.0,0.0}; };
struct CustomPressureAMGPEntry { PetscInt coarse=-1; double value=0.0; };

static PetscErrorCode customPressureAMGBuildFineSmoothedTransfer(CustomPressureAMGRuntime& R,
  CustomPressureBPlan& B,const CustomMomentumCSR& M,CustomPressureAMGLevel& L0) {
  PetscFunctionBeginUser;
  const PetscInt nVel=B.velocityHalo.nOwned,nP=B.pressureHalo.nOwned;
  if((PetscInt)L0.aggregateId.size()!=nP || (PetscInt)L0.A.diagonal.size()!=nP) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"custom AMG fine SA level size mismatch");
  if(!(L0.lambdaMax>0.0)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG fine SA requires spectral estimate before interpolation smoothing");
  const double omega=(double)R.amg.config.smoothedAggregationDamping/L0.lambdaMax;
  std::vector<PetscInt> starPtr((std::size_t)nVel+1,0);
  for(const auto& cp:B.transposeCells) for(PetscInt j=0;j<cp.nOwnedVel;++j) ++starPtr[(std::size_t)cp.velOwnedLocal[j]+1];
  for(PetscInt q=0;q<nVel;++q) starPtr[(std::size_t)q+1]+=starPtr[(std::size_t)q];
  std::vector<CustomPressureAMGStarEntry> star((std::size_t)starPtr.back()); std::vector<PetscInt> cur=starPtr;
  std::vector<PetscInt> ghostAgg; PetscCall(customPressureAMGExchangeAggregateIds(B.pressureHalo,L0.aggregateId,ghostAgg,50221));
  auto aggForP=[&](PetscInt pl)->PetscInt{return pl<B.pressureHalo.nOwned?L0.aggregateId[(std::size_t)pl]:ghostAgg[(std::size_t)(pl-B.pressureHalo.nOwned)];};
  for(const auto& cp:B.transposeCells) {
    const PetscInt ag=aggForP(cp.pLocal);
    for(PetscInt j=0;j<cp.nOwnedVel;++j){const PetscInt q=cp.velOwnedLocal[j],k=cur[(std::size_t)q]++;auto& e=star[(std::size_t)k];e.aggregate=ag;const int a=(int)cp.basis[j];for(int d=0;d<3;++d)e.b[d]=customPressureBCoeff(cp.vol,cp.gradLambda,d,a);}
  }
  std::vector<PetscInt> qPtr((std::size_t)nVel+1,0); std::vector<CustomPressureAMGQEntry> qEntries; qEntries.reserve(star.size());
  for(PetscInt q=0;q<nVel;++q){const PetscInt s=starPtr[(std::size_t)q],e=starPtr[(std::size_t)q+1];std::sort(star.begin()+s,star.begin()+e,[](const auto&a,const auto&b){return a.aggregate<b.aggregate;});for(PetscInt k=s;k<e;){CustomPressureAMGQEntry qe;qe.coarse=star[(std::size_t)k].aggregate;PetscInt j=k;for(;j<e&&star[(std::size_t)j].aggregate==qe.coarse;++j)for(int d=0;d<3;++d)qe.b[d]+=star[(std::size_t)j].b[d];qEntries.push_back(qe);k=j;}qPtr[(std::size_t)q+1]=(PetscInt)qEntries.size();}
  std::vector<CustomPressureAMGStarEntry>().swap(star); std::vector<PetscInt>().swap(cur);
  std::vector<PetscInt> ghostQPtr; std::vector<CustomPressureAMGQEntry> ghostQ; PetscCall(customPressureAMGExchangeSparseRows(B.velocityHalo,qPtr,qEntries,ghostQPtr,ghostQ,50231));
  PetscCall(customPeerExchange(B.velocityHalo,M.rAU,50241));
  auto qRange=[&](PetscInt li,const CustomPressureAMGQEntry*& begin,const CustomPressureAMGQEntry*& end){if(li<B.velocityHalo.nOwned){begin=qEntries.data()+qPtr[(std::size_t)li];end=qEntries.data()+qPtr[(std::size_t)li+1];}else{const PetscInt g=li-B.velocityHalo.nOwned;begin=ghostQ.data()+ghostQPtr[(std::size_t)g];end=ghostQ.data()+ghostQPtr[(std::size_t)g+1];}};
  auto& P=L0.P; P.rowPtr.assign((std::size_t)nP+1,0);P.colGid.clear();P.values.clear();
  std::vector<std::pair<PetscInt,double>> row;
  for(const auto& cp:B.forwardCells) {
    const PetscInt i=cp.pLocal; if(i<0||i>=nP)SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG fine SA pressure row invalid");
    row.clear(); row.push_back({L0.aggregateId[(std::size_t)i],1.0});
    const double invd=1.0/L0.A.diagonal[(std::size_t)i];
    for(int a=0;a<8;++a) if(cp.velLocal[a]>=0) {
      const PetscInt q=cp.velLocal[a]; const double rau=customPeerValue(B.velocityHalo,M.rAU,q);
      const CustomPressureAMGQEntry *qb=nullptr,*qe=nullptr; qRange(q,qb,qe);
      for(const auto* e=qb;e!=qe;++e){double dot=0.0;for(int d=0;d<3;++d)dot+=customPressureBCoeff(cp.vol,cp.gradLambda,d,a)*e->b[d];row.push_back({e->coarse,-omega*invd*rau*dot});}
    }
    customPressureAMGNormalizePruneRow(row,L0.aggregateId[(std::size_t)i],R.amg.config.interpolationMaxRowNnz);
    for(const auto& e:row){P.colGid.push_back(e.first);P.values.push_back(e.second);} P.rowPtr[(std::size_t)i+1]=(PetscInt)P.colGid.size();
  }
  PetscCall(customPressureAMGFinalizeTransferHalo(L0,P,"custom_sa_P0"));
  double ldef=0.0;PetscInt lmaxnnz=0; for(PetscInt i=0;i<nP;++i){double sum=0.0;const PetscInt nn=P.rowPtr[(std::size_t)i+1]-P.rowPtr[(std::size_t)i];lmaxnnz=std::max(lmaxnnz,nn);for(PetscInt k=P.rowPtr[(std::size_t)i];k<P.rowPtr[(std::size_t)i+1];++k)sum+=P.values[(std::size_t)k];ldef=std::max(ldef,std::abs(sum-1.0));}
  double gdef=0.0;PetscInt gmaxnnz=0;unsigned long long lnnz=(unsigned long long)P.values.size(),gnnz=0;PetscCallMPI(MPI_Allreduce(&ldef,&gdef,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));PetscCallMPI(MPI_Allreduce(&lmaxnnz,&gmaxnnz,1,MPIU_INT,MPI_MAX,PETSC_COMM_WORLD));PetscCallMPI(MPI_Allreduce(&lnnz,&gnnz,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_CUSTOM_AMG_SA_TRANSFER level=0 status=PASS construction=(I-omega_DinvA)Pt_factored_BDBt omega=%.12e damping=%.6f lambdaMax=%.12e rows=%" PetscInt_FMT " nnz=%llu avgNnz=%.6f maxRowNnz=%" PetscInt_FMT " rowSumDefect=%.3e fineSchurCSR=0 genericPtAP=0\n",omega,(double)R.amg.config.smoothedAggregationDamping,L0.lambdaMax,L0.globalRows,gnnz,L0.globalRows?((double)gnnz/(double)L0.globalRows):0.0,gmaxnnz,gdef));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customPressureAMGBuildFirstCoarseSmoothed(CustomPressureBPlan& B,const CustomMomentumCSR& M,
  CustomPressureAMGLevel& L0,CustomPressureAMGCSR& A1) {
  PetscFunctionBeginUser;
  int rank=0,size=1;PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  auto& P=L0.P; const PetscInt nVel=B.velocityHalo.nOwned;
  std::vector<PetscInt> ghostPPtr; std::vector<CustomPressureAMGPEntry> ownedPE(P.values.size()),ghostPE;
  for(std::size_t k=0;k<P.values.size();++k){ownedPE[k].coarse=P.colGid[k];ownedPE[k].value=P.values[k];}
  PetscCall(customPressureAMGExchangeSparseRows(B.pressureHalo,P.rowPtr,ownedPE,ghostPPtr,ghostPE,50301));
  auto pRange=[&](PetscInt pl,const CustomPressureAMGPEntry*& b,const CustomPressureAMGPEntry*& e){
    if(pl<B.pressureHalo.nOwned){b=ownedPE.data()+P.rowPtr[(std::size_t)pl];e=ownedPE.data()+P.rowPtr[(std::size_t)pl+1];}
    else{const PetscInt g=pl-B.pressureHalo.nOwned;b=ghostPE.data()+ghostPPtr[(std::size_t)g];e=ghostPE.data()+ghostPPtr[(std::size_t)g+1];}
  };
  struct Inc { PetscInt cell=-1; std::uint8_t basis=0; };
  std::vector<PetscInt> incPtr((std::size_t)nVel+1,0);
  for(PetscInt c=0;c<(PetscInt)B.transposeCells.size();++c){const auto& cp=B.transposeCells[(std::size_t)c];for(PetscInt j=0;j<cp.nOwnedVel;++j)++incPtr[(std::size_t)cp.velOwnedLocal[j]+1];}
  for(PetscInt q=0;q<nVel;++q)incPtr[(std::size_t)q+1]+=incPtr[(std::size_t)q];
  std::vector<Inc> inc((std::size_t)incPtr.back());std::vector<PetscInt> cur=incPtr;
  for(PetscInt c=0;c<(PetscInt)B.transposeCells.size();++c){const auto& cp=B.transposeCells[(std::size_t)c];for(PetscInt j=0;j<cp.nOwnedVel;++j){auto& z=inc[(std::size_t)cur[(std::size_t)cp.velOwnedLocal[j]]++];z.cell=c;z.basis=cp.basis[j];}}
  const PetscInt localAgg=L0.localAggregateEnd-L0.localAggregateStart;std::vector<PetscInt> counts((std::size_t)size,0),off((std::size_t)size+1,0);PetscCallMPI(MPI_Allgather(&localAgg,1,MPIU_INT,counts.data(),1,MPIU_INT,PETSC_COMM_WORLD));for(int r=0;r<size;++r)off[(std::size_t)r+1]=off[(std::size_t)r]+counts[(std::size_t)r];if(off.back()!=L0.globalAggregates)SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG SA coarse ownership mismatch");
  auto owner=[&](PetscInt g)->int{auto it=std::upper_bound(off.begin(),off.end(),g);int r=(int)(it-off.begin())-1;return(r>=0&&r<size&&g>=off[(std::size_t)r]&&g<off[(std::size_t)r+1])?r:-1;};
  std::vector<std::unordered_map<CustomPressureAMGPairKey,double,CustomPressureAMGPairHash>> by((std::size_t)size);
  std::vector<CustomPressureAMGStarEntry> tmp;std::vector<PetscInt> ga;std::vector<std::array<double,3>> gb;unsigned long long raw=0;
  for(PetscInt q=0;q<nVel;++q){
    tmp.clear();
    for(PetscInt kk=incPtr[(std::size_t)q];kk<incPtr[(std::size_t)q+1];++kk){const auto& z=inc[(std::size_t)kk];const auto& cp=B.transposeCells[(std::size_t)z.cell];const CustomPressureAMGPEntry *pb=nullptr,*pe=nullptr;pRange(cp.pLocal,pb,pe);for(const auto* pp=pb;pp!=pe;++pp){CustomPressureAMGStarEntry e;e.aggregate=pp->coarse;for(int d=0;d<3;++d)e.b[d]=pp->value*customPressureBCoeff(cp.vol,cp.gradLambda,d,(int)z.basis);tmp.push_back(e);}}
    if(tmp.empty())continue;
    std::sort(tmp.begin(),tmp.end(),[](const auto&a,const auto&b){return a.aggregate<b.aggregate;});ga.clear();gb.clear();
    for(std::size_t k=0;k<tmp.size();){const PetscInt ag=tmp[k].aggregate;std::array<double,3> c{{0,0,0}};std::size_t j=k;for(;j<tmp.size()&&tmp[j].aggregate==ag;++j)for(int d=0;d<3;++d)c[(std::size_t)d]+=tmp[j].b[d];ga.push_back(ag);gb.push_back(c);k=j;}
    const double rau=M.rAU[(std::size_t)q];if(!(rau>0.0)||!std::isfinite(rau))SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"custom AMG SA first coarse invalid rAU");
    for(std::size_t i=0;i<ga.size();++i){const int ro=owner(ga[i]);if(ro<0)SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG SA coarse row owner invalid");auto& map=by[(std::size_t)ro];for(std::size_t j=0;j<ga.size();++j){const double v=rau*(gb[i][0]*gb[j][0]+gb[i][1]*gb[j][1]+gb[i][2]*gb[j][2]);if(v!=0.0){map[{ga[i],ga[j]}]+=v;++raw;}}}
  }
  std::vector<int> sc((std::size_t)size,0),rc((std::size_t)size,0),sd((std::size_t)size,0),rd((std::size_t)size,0);std::size_t nts=0;
  for(int r=0;r<size;++r){if(by[(std::size_t)r].size()>(std::size_t)std::numeric_limits<int>::max())SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"custom AMG SA unique triplet peer count too large");sc[(std::size_t)r]=(int)by[(std::size_t)r].size();nts+=by[(std::size_t)r].size();}
  for(int r=1;r<size;++r)sd[(std::size_t)r]=sd[(std::size_t)r-1]+sc[(std::size_t)r-1];std::vector<CustomPressureAMGTriplet> send;send.reserve(nts);for(int r=0;r<size;++r)for(const auto& kv:by[(std::size_t)r])send.push_back({kv.first.row,kv.first.col,kv.second});
  PetscCallMPI(MPI_Alltoall(sc.data(),1,MPI_INT,rc.data(),1,MPI_INT,PETSC_COMM_WORLD));for(int r=1;r<size;++r)rd[(std::size_t)r]=rd[(std::size_t)r-1]+rc[(std::size_t)r-1];std::vector<CustomPressureAMGTriplet> recv((std::size_t)(size?rd.back()+rc.back():0));MPI_Datatype tt=MPI_DATATYPE_NULL;PetscCallMPI(MPI_Type_contiguous((int)sizeof(CustomPressureAMGTriplet),MPI_BYTE,&tt));PetscCallMPI(MPI_Type_commit(&tt));PetscCallMPI(MPI_Alltoallv(send.empty()?nullptr:send.data(),sc.data(),sd.data(),tt,recv.empty()?nullptr:recv.data(),rc.data(),rd.data(),tt,PETSC_COMM_WORLD));PetscCallMPI(MPI_Type_free(&tt));
  std::sort(recv.begin(),recv.end(),[](const auto&a,const auto&b){return a.row<b.row||(a.row==b.row&&a.col<b.col);});std::vector<CustomPressureAMGTriplet> merged;merged.reserve(recv.size());for(const auto&t:recv){if(!merged.empty()&&merged.back().row==t.row&&merged.back().col==t.col)merged.back().value+=t.value;else merged.push_back(t);}A1.globalRows=L0.globalAggregates;A1.localStart=L0.localAggregateStart;A1.localEnd=L0.localAggregateEnd;A1.rowPtr.assign((std::size_t)localAgg+1,0);A1.colGid.clear();A1.values.clear();A1.diagonal.assign((std::size_t)localAgg,0.0);std::size_t m=0;for(PetscInt i=0;i<localAgg;++i){const PetscInt rg=A1.localStart+i;while(m<merged.size()&&merged[m].row<rg)++m;while(m<merged.size()&&merged[m].row==rg){if(merged[m].value!=0.0){A1.colGid.push_back(merged[m].col);A1.values.push_back(merged[m].value);if(merged[m].col==rg)A1.diagonal[(std::size_t)i]=merged[m].value;}++m;}A1.rowPtr[(std::size_t)i+1]=(PetscInt)A1.colGid.size();}
  PetscInt lbad=0;double lmin=std::numeric_limits<double>::infinity(),lmax=0.0;for(double d:A1.diagonal){if(!(d>0.0)||!std::isfinite(d))++lbad;else{lmin=std::min(lmin,d);lmax=std::max(lmax,d);}}PetscInt gbad=0;double gmin=0,gmax=0;PetscCallMPI(MPI_Allreduce(&lbad,&gbad,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));PetscCallMPI(MPI_Allreduce(&lmin,&gmin,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD));PetscCallMPI(MPI_Allreduce(&lmax,&gmax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));if(gbad)SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"custom AMG SA first coarse diagonal invalid");std::vector<PetscInt> needed;for(PetscInt g:A1.colGid)if(g<A1.localStart||g>=A1.localEnd)needed.push_back(g);PetscCall(buildCustomPeerHalo(counts,rank,std::move(needed),A1.halo,"custom_sa_A1"));A1.colLocal.resize(A1.colGid.size());for(std::size_t k=0;k<A1.colGid.size();++k){A1.colLocal[k]=customPeerLocalIndex(A1.halo,A1.colGid[k]);if(A1.colLocal[k]<0)SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG SA first coarse halo missing column");}unsigned long long lnnz=(unsigned long long)A1.values.size(),gnnz=0,graw=0;PetscCallMPI(MPI_Allreduce(&lnnz,&gnnz,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));PetscCallMPI(MPI_Allreduce(&raw,&graw,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_CUSTOM_AMG_SA_COARSE level=1 status=PASS construction=Pt_BDBt_P_factored_streamed rows=%" PetscInt_FMT " nnz=%llu avgNnz=%.6f rawTerms=%llu diag=[%.12e,%.12e] fineSchurCSR=0 genericFinePtAP=0\n",A1.globalRows,gnnz,A1.globalRows?((double)gnnz/(double)A1.globalRows):0.0,graw,gmin,gmax));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customPressureAMGValidateFirstSmoothedGalerkin(CustomPressureAMGRuntime& R,
  CustomPressureAMGLevel& L0,CustomPressureAMGCSR& A1) {
  PetscFunctionBeginUser;
  const PetscInt nc=A1.localEnd-A1.localStart,nf=L0.localEnd-L0.localStart;
  std::vector<double> x((std::size_t)nc),fx((std::size_t)nf,0.0),afy,ref,y;
  for(PetscInt i=0;i<nc;++i){const double g=(double)(A1.localStart+i+1);x[(std::size_t)i]=std::sin(0.0137*g)+0.37*std::cos(0.0191*g);}
  PetscCall(customPressureAMGTransferProlongAdd(L0.P,x,fx,50801));
  PetscCall(R.amg.fine.apply(R.amg.fine.ctx,fx,afy));
  PetscCall(customPressureAMGTransferRestrict(L0.P,afy,ref,50802));
  PetscCall(customPressureAMGCoarseSpMV(A1,x,y,50803));
  double ld=0.0,lr=0.0,lmax=0.0;for(PetscInt i=0;i<nc;++i){const double q=y[(std::size_t)i]-ref[(std::size_t)i];ld+=q*q;lr+=ref[(std::size_t)i]*ref[(std::size_t)i];lmax=std::max(lmax,std::abs(q));}double gd=0.0,gr=0.0,gmax=0.0;PetscCallMPI(MPI_Allreduce(&ld,&gd,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));PetscCallMPI(MPI_Allreduce(&lr,&gr,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));PetscCallMPI(MPI_Allreduce(&lmax,&gmax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));const double rel=std::sqrt(gd)/std::max(std::sqrt(gr),1e-300);const double tol=(sizeof(PetscReal)<=4)?5e-4:5e-10;PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_CUSTOM_AMG_SA_GALERKIN level=1 status=%s check=A1x_vs_PtA0Px rel=%.3e maxAbs=%.3e tol=%.3e fineOperator=matrix_free_B_rAU_Bt explicitFineSchur=0\n",rel<=tol?"PASS":"FAIL",rel,gmax,tol));if(rel>tol)SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG smoothed first-level Galerkin validation failed");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customPressureAMGBuildFirstCoarse(CustomPressureBPlan& B,const CustomMomentumCSR& M,
  const CustomPressureAMGLevel& L0,CustomPressureAMGCSR& A1) {
  PetscFunctionBeginUser;
  int rank=0,size=1; PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank)); PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  const PetscInt localAgg=L0.localAggregateEnd-L0.localAggregateStart;
  if(localAgg<0 || L0.globalAggregates<=0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG invalid Gate-3 aggregate ownership");
  std::vector<PetscInt> aggCounts((std::size_t)size,0),aggOff((std::size_t)size+1,0);
  PetscCallMPI(MPI_Allgather(&localAgg,1,MPIU_INT,aggCounts.data(),1,MPIU_INT,PETSC_COMM_WORLD));
  for(int r=0;r<size;++r) aggOff[(std::size_t)r+1]=aggOff[(std::size_t)r]+aggCounts[(std::size_t)r];
  if(aggOff.back()!=L0.globalAggregates) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG aggregate ownership counts do not sum to global aggregate count");
  auto aggOwner=[&](PetscInt g)->int {
    auto it=std::upper_bound(aggOff.begin(),aggOff.end(),g);
    int r=(int)(it-aggOff.begin())-1;
    return (r>=0 && r<size && g>=aggOff[(std::size_t)r] && g<aggOff[(std::size_t)r+1])?r:-1;
  };

  // The pressure halo already contains every pressure cell needed by B^T for
  // the locally-owned velocity DOFs. Exchange only aggregate IDs on that same
  // graph; no extra fine-grid graph construction is required.
  std::vector<PetscInt> ghostAgg;
  PetscCall(customPressureAMGExchangeAggregateIds(B.pressureHalo,L0.aggregateId,ghostAgg,49501));
  auto aggregateForPressureLocal=[&](PetscInt pl)->PetscInt {
    return (pl<B.pressureHalo.nOwned)?L0.aggregateId[(std::size_t)pl]:ghostAgg[(std::size_t)(pl-B.pressureHalo.nOwned)];
  };

  const PetscInt nVel=(PetscInt)M.rAU.size();
  if(nVel!=B.velocityHalo.nOwned) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"custom AMG Gate-3 rAU/velocity ownership mismatch");
  std::vector<PetscInt> starPtr((std::size_t)nVel+1,0);
  unsigned long long localInc=0;
  for(const auto& cp:B.transposeCells) for(PetscInt j=0;j<cp.nOwnedVel;++j) {
    const PetscInt q=cp.velOwnedLocal[j]; if(q<0 || q>=nVel) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG Gate-3 velocity star index invalid");
    ++starPtr[(std::size_t)q+1]; ++localInc;
  }
  for(PetscInt q=0;q<nVel;++q) starPtr[(std::size_t)q+1]+=starPtr[(std::size_t)q];
  std::vector<CustomPressureAMGStarEntry> star((std::size_t)starPtr.back());
  std::vector<PetscInt> cursor=starPtr;
  for(const auto& cp:B.transposeCells) {
    const PetscInt ag=aggregateForPressureLocal(cp.pLocal);
    if(ag<0 || ag>=L0.globalAggregates) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG Gate-3 pressure aggregate GID invalid");
    for(PetscInt j=0;j<cp.nOwnedVel;++j) {
      const PetscInt q=cp.velOwnedLocal[j],k=cursor[(std::size_t)q]++;
      auto& e=star[(std::size_t)k]; e.aggregate=ag;
      const int a=(int)cp.basis[j]; for(int d=0;d<3;++d) e.b[d]=customPressureBCoeff(cp.vol,cp.gradLambda,d,a);
    }
  }

  std::vector<std::vector<CustomPressureAMGTriplet>> byOwner((std::size_t)size);
  std::vector<PetscInt> groupAgg; std::vector<std::array<double,3>> groupB;
  unsigned long long localRaw=0;
  for(PetscInt q=0;q<nVel;++q) {
    const PetscInt s=starPtr[(std::size_t)q],e=starPtr[(std::size_t)q+1];
    if(s==e) continue;
    std::sort(star.begin()+s,star.begin()+e,[](const auto& a,const auto& b){return a.aggregate<b.aggregate;});
    groupAgg.clear(); groupB.clear();
    for(PetscInt k=s;k<e;) {
      const PetscInt ag=star[(std::size_t)k].aggregate; std::array<double,3> c{{0.0,0.0,0.0}};
      PetscInt j=k; for(;j<e && star[(std::size_t)j].aggregate==ag;++j) for(int d=0;d<3;++d) c[(std::size_t)d]+=star[(std::size_t)j].b[d];
      groupAgg.push_back(ag); groupB.push_back(c); k=j;
    }
    const double rau=M.rAU[(std::size_t)q];
    if(!std::isfinite(rau) || rau<=0.0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"custom AMG Gate-3 encountered invalid rAU");
    for(std::size_t i=0;i<groupAgg.size();++i) {
      const int owner=aggOwner(groupAgg[i]); if(owner<0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG Gate-3 coarse row owner invalid");
      auto& v=byOwner[(std::size_t)owner];
      for(std::size_t j=0;j<groupAgg.size();++j) {
        const double aij=rau*(groupB[i][0]*groupB[j][0]+groupB[i][1]*groupB[j][1]+groupB[i][2]*groupB[j][2]);
        if(aij!=0.0) { v.push_back({groupAgg[i],groupAgg[j],aij}); ++localRaw; }
      }
    }
  }

  std::vector<int> sendCount((std::size_t)size,0),recvCount((std::size_t)size,0),sendDisp((std::size_t)size,0),recvDisp((std::size_t)size,0);
  std::size_t totalSend=0; for(int r=0;r<size;++r) { if(byOwner[(std::size_t)r].size()>(std::size_t)std::numeric_limits<int>::max()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"custom AMG Gate-3 per-peer triplet count exceeds MPI int limit"); sendCount[(std::size_t)r]=(int)byOwner[(std::size_t)r].size(); totalSend+=byOwner[(std::size_t)r].size(); }
  for(int r=1;r<size;++r) sendDisp[(std::size_t)r]=sendDisp[(std::size_t)r-1]+sendCount[(std::size_t)r-1];
  std::vector<CustomPressureAMGTriplet> send(totalSend); std::size_t pos=0;
  for(int r=0;r<size;++r) { std::copy(byOwner[(std::size_t)r].begin(),byOwner[(std::size_t)r].end(),send.begin()+pos); pos+=byOwner[(std::size_t)r].size(); std::vector<CustomPressureAMGTriplet>().swap(byOwner[(std::size_t)r]); }
  PetscCallMPI(MPI_Alltoall(sendCount.data(),1,MPI_INT,recvCount.data(),1,MPI_INT,PETSC_COMM_WORLD));
  std::size_t totalRecv=0; for(int r=0;r<size;++r) totalRecv+=(std::size_t)recvCount[(std::size_t)r];
  for(int r=1;r<size;++r) recvDisp[(std::size_t)r]=recvDisp[(std::size_t)r-1]+recvCount[(std::size_t)r-1];
  std::vector<CustomPressureAMGTriplet> recv(totalRecv);
  MPI_Datatype tripType=MPI_DATATYPE_NULL; PetscCallMPI(MPI_Type_contiguous((int)sizeof(CustomPressureAMGTriplet),MPI_BYTE,&tripType)); PetscCallMPI(MPI_Type_commit(&tripType));
  PetscCallMPI(MPI_Alltoallv(send.empty()?nullptr:send.data(),sendCount.data(),sendDisp.data(),tripType,
                             recv.empty()?nullptr:recv.data(),recvCount.data(),recvDisp.data(),tripType,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Type_free(&tripType));
  std::vector<CustomPressureAMGTriplet>().swap(send); std::vector<CustomPressureAMGStarEntry>().swap(star); std::vector<PetscInt>().swap(cursor);

  std::sort(recv.begin(),recv.end(),[](const auto& a,const auto& b){return (a.row<b.row)||(a.row==b.row && a.col<b.col);});
  std::vector<CustomPressureAMGTriplet> merged; merged.reserve(recv.size());
  for(const auto& t:recv) {
    if(t.row<L0.localAggregateStart || t.row>=L0.localAggregateEnd) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG Gate-3 received non-owned coarse row");
    if(!merged.empty() && merged.back().row==t.row && merged.back().col==t.col) merged.back().value+=t.value;
    else merged.push_back(t);
  }
  std::vector<CustomPressureAMGTriplet>().swap(recv);

  A1.globalRows=L0.globalAggregates; A1.localStart=L0.localAggregateStart; A1.localEnd=L0.localAggregateEnd;
  A1.rowPtr.assign((std::size_t)localAgg+1,0); A1.colGid.clear(); A1.values.clear(); A1.diagonal.assign((std::size_t)localAgg,0.0);
  A1.colGid.reserve(merged.size()); A1.values.reserve(merged.size());
  std::size_t m=0;
  for(PetscInt lr=0;lr<localAgg;++lr) {
    const PetscInt rg=A1.localStart+lr;
    while(m<merged.size() && merged[m].row<rg) ++m;
    while(m<merged.size() && merged[m].row==rg) {
      if(merged[m].value!=0.0) { A1.colGid.push_back(merged[m].col); A1.values.push_back(merged[m].value); if(merged[m].col==rg) A1.diagonal[(std::size_t)lr]=merged[m].value; }
      ++m;
    }
    A1.rowPtr[(std::size_t)lr+1]=(PetscInt)A1.colGid.size();
  }
  std::vector<CustomPressureAMGTriplet>().swap(merged);

  PetscInt localBadDiag=0; double lmin=std::numeric_limits<double>::infinity(),lmax=0.0;
  for(double d:A1.diagonal) { if(!std::isfinite(d) || d<=0.0) ++localBadDiag; else {lmin=std::min(lmin,d);lmax=std::max(lmax,d);} }
  PetscInt globalBadDiag=0; double gmin=0.0,gmax=0.0;
  PetscCallMPI(MPI_Allreduce(&localBadDiag,&globalBadDiag,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&lmin,&gmin,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD)); PetscCallMPI(MPI_Allreduce(&lmax,&gmax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  if(globalBadDiag) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"custom AMG Gate-3 coarse diagonal is not positive/finite");

  std::vector<PetscInt> needed; needed.reserve(A1.colGid.size());
  for(PetscInt g:A1.colGid) if(g<A1.localStart || g>=A1.localEnd) needed.push_back(g);
  PetscCall(buildCustomPeerHalo(aggCounts,rank,std::move(needed),A1.halo,"custom_amg_L1"));
  A1.colLocal.resize(A1.colGid.size());
  for(std::size_t k=0;k<A1.colGid.size();++k) { A1.colLocal[k]=customPeerLocalIndex(A1.halo,A1.colGid[k]); if(A1.colLocal[k]<0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG Gate-3 coarse halo missing column"); }

  unsigned long long localNnz=(unsigned long long)A1.values.size(),globalNnz=0,globalRaw=0;
  PetscCallMPI(MPI_Allreduce(&localNnz,&globalNnz,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localRaw,&globalRaw,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  const unsigned long long localBytes=(unsigned long long)(A1.rowPtr.capacity()+A1.colGid.capacity()+A1.colLocal.capacity())*sizeof(PetscInt)
    +(unsigned long long)(A1.values.capacity()+A1.diagonal.capacity())*sizeof(double);
  unsigned long long globalBytes=0; PetscCallMPI(MPI_Allreduce(&localBytes,&globalBytes,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  unsigned long long globalInc=0; PetscCallMPI(MPI_Allreduce(&localInc,&globalInc,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_CUSTOM_AMG_GATE3_A1 status=PASS construction=velocity_star_stream_B1_outer_product fineSchurCSRRequired=0 finePtAPRequired=0 rows=%" PetscInt_FMT " nnz=%llu avgNnz=%.6f rawTerms=%llu velocityStarIncidences=%llu diag=[%.12e,%.12e] retainedCSRMiB=%.3f\n",
    A1.globalRows,globalNnz,A1.globalRows?((double)globalNnz/(double)A1.globalRows):0.0,globalRaw,globalInc,gmin,gmax,(double)globalBytes/(1024.0*1024.0)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customPressureAMGGate3Validate(CustomPressureAMGFineOperator& fine,
  const CustomPressureAMGLevel& L0,CustomPressureAMGCSR& A1) {
  PetscFunctionBeginUser;
  const PetscInt nc=A1.localEnd-A1.localStart;
  std::vector<double> x((std::size_t)nc),z((std::size_t)nc),fineX((std::size_t)(fine.localEnd-fine.localStart),0.0),fineY,ref,y,az;
  for(PetscInt i=0;i<nc;++i) { const double g=(double)(A1.localStart+i+1); x[(std::size_t)i]=std::sin(0.017*g)+0.37*std::cos(0.011*g); z[(std::size_t)i]=std::cos(0.013*g)-0.21*std::sin(0.019*g); }
  for(PetscInt i=0;i<(PetscInt)fineX.size();++i) {
    const PetscInt ag=L0.aggregateId[(std::size_t)i];
    if(ag<A1.localStart || ag>=A1.localEnd) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG Gate-3 tentative prolongation references nonlocal aggregate");
    fineX[(std::size_t)i]=x[(std::size_t)(ag-A1.localStart)];
  }
  PetscCall(fine.apply(fine.ctx,fineX,fineY));
  ref.assign((std::size_t)nc,0.0);
  for(PetscInt i=0;i<(PetscInt)fineY.size();++i) ref[(std::size_t)(L0.aggregateId[(std::size_t)i]-A1.localStart)]+=fineY[(std::size_t)i];
  PetscCall(customPressureAMGCoarseSpMV(A1,x,y,49521));
  double ld2=0.0,lr2=0.0,lmax=0.0; for(PetscInt i=0;i<nc;++i){const double d=y[(std::size_t)i]-ref[(std::size_t)i];ld2+=d*d;lr2+=ref[(std::size_t)i]*ref[(std::size_t)i];lmax=std::max(lmax,std::abs(d));}
  double gd2=0.0,gr2=0.0,gmax=0.0; PetscCallMPI(MPI_Allreduce(&ld2,&gd2,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD)); PetscCallMPI(MPI_Allreduce(&lr2,&gr2,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD)); PetscCallMPI(MPI_Allreduce(&lmax,&gmax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  const double rel=std::sqrt(gd2)/std::max(std::sqrt(gr2),1e-300);

  PetscCall(customPressureAMGCoarseSpMV(A1,z,az,49522));
  double lxy=0.0,lyx=0.0; for(PetscInt i=0;i<nc;++i){lxy+=x[(std::size_t)i]*az[(std::size_t)i];lyx+=z[(std::size_t)i]*y[(std::size_t)i];}
  double gxy=0.0,gyx=0.0; PetscCallMPI(MPI_Allreduce(&lxy,&gxy,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD)); PetscCallMPI(MPI_Allreduce(&lyx,&gyx,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  const double sym=std::abs(gxy-gyx)/std::max({std::abs(gxy),std::abs(gyx),1e-300});
  double nullRel=0.0;
  if(fine.constantNullspace) {
    std::vector<double> one((std::size_t)nc,1.0),ao; PetscCall(customPressureAMGCoarseSpMV(A1,one,ao,49523));
    double ln=0.0,ld=0.0; for(PetscInt i=0;i<nc;++i){ln+=ao[(std::size_t)i]*ao[(std::size_t)i];ld+=A1.diagonal[(std::size_t)i]*A1.diagonal[(std::size_t)i];}
    double gn=0.0,gd=0.0; PetscCallMPI(MPI_Allreduce(&ln,&gn,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD)); PetscCallMPI(MPI_Allreduce(&ld,&gd,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD)); nullRel=std::sqrt(gn)/std::max(std::sqrt(gd),1e-300);
  }
  const double tol=(sizeof(PetscReal)<=4)?2e-4:2e-11;
  const PetscBool pass=(rel<=tol && sym<=20.0*tol && (!fine.constantNullspace || nullRel<=50.0*tol))?PETSC_TRUE:PETSC_FALSE;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_CUSTOM_AMG_GATE3_GALERKIN status=%s check=A1x_vs_PtA0Px exactFineOperator=matrix_free_B_rAU_Bt rel=%.3e maxAbs=%.3e symmetryRel=%.3e constantNullspace=%d constantActionRel=%.3e tol=%.3e explicitFineSchurUsed=0\n",
    pass?"PASS":"FAIL",rel,gmax,sym,(int)fine.constantNullspace,nullRel,tol));
  if(!pass) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG Gate-3 Galerkin/action validation failed");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customPressureAMGGate3Probe(const Mesh& mesh,const Discrete& D,int rank,
  CustomPressureBPlan& B,const CustomMomentumCSR& M,PetscInt globalRows,PetscBool constantNullspace) {
  PetscFunctionBeginUser;
  CustomPressureAMGExactSchurContext ctx; CustomPressureAMG amg;
  PetscCall(customPressureAMGInitFineOperator(B,M,globalRows,constantNullspace,ctx,amg.fine));
  amg.levels.resize(2);
  PetscCall(customPressureAMGBuildFineAggregation(mesh,D,rank,amg.config,amg.fine,amg.levels[0]));
  PetscCall(customPressureAMGComputeFineDiagonal(B,M,amg.levels[0].A.diagonal));
  PetscCall(customPressureAMGBuildFirstCoarse(B,M,amg.levels[0],amg.levels[1].A));
  amg.levels[1].level=1; amg.levels[1].globalRows=amg.levels[1].A.globalRows; amg.levels[1].localStart=amg.levels[1].A.localStart; amg.levels[1].localEnd=amg.levels[1].A.localEnd;
  PetscCall(customPressureAMGGate3Validate(amg.fine,amg.levels[0],amg.levels[1].A));
  amg.topologyReady=PETSC_TRUE; amg.numericReady=PETSC_TRUE;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_CUSTOM_AMG_GATE3 status=PASS roadmapGates=6 firstCoarseOperator=READY recursiveHierarchy=NOT_BUILT smoother=NOT_BUILT pressureSolveBackend=PETSc_fallback fineOperator=matrix_free generalFlowContract=distributed_B_rAU_Bt_plus_cell_topology_plus_gauge\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// -----------------------------------------------------------------------------
// Recursive hierarchy, level smoother and coarse solve
// -----------------------------------------------------------------------------
// Historically this block was Gate 4 of the bring-up.  It now serves both the
// unsmoothed and smoothed production hierarchies.  Every level owns its pressure
// rows contiguously by MPI rank.  Aggregation itself remains rank-local; coarse
// operator entries may of course couple aggregates owned by different ranks and
// are communicated through CustomPeerHalo.
//
// A V-cycle is deliberately fixed and linear: one symmetric pre-correction, one
// recursive coarse correction, and one symmetric post-correction.  That fixed
// linearity and R=P^T are what make the same V-cycle safe as a PCG
// preconditioner.
// -----------------------------------------------------------------------------


static PetscErrorCode customPressureAMGBuildExplicitAggregation(CustomPressureAMGCSR& A,
  const CustomPressureAMGConfig& cfg,PetscInt level,CustomPressureAMGLevel& L) {
  PetscFunctionBeginUser;
  int rank=0; PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  const PetscInt n=A.localEnd-A.localStart;
  if(n<0 || (PetscInt)A.rowPtr.size()!=n+1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG coarse aggregation CSR mismatch");
  std::vector<PetscInt> aggOf((std::size_t)n,-1);
  std::vector<std::vector<PetscInt>> members;
  members.reserve((std::size_t)std::max<PetscInt>(1,(n+cfg.targetAggregateSize-1)/cfg.targetAggregateSize));
  std::vector<PetscInt> queue; queue.reserve((std::size_t)std::max<PetscInt>(16,2*cfg.targetAggregateSize));

  auto forLocalNbr=[&](PetscInt i,const auto& f) {
    for(PetscInt k=A.rowPtr[(std::size_t)i];k<A.rowPtr[(std::size_t)i+1];++k) {
      const PetscInt g=A.colGid[(std::size_t)k];
      if(g>=A.localStart && g<A.localEnd && g!=A.localStart+i) f(g-A.localStart);
    }
  };

  for(PetscInt seed=0;seed<n;++seed) if(aggOf[(std::size_t)seed]<0) {
    const PetscInt a=(PetscInt)members.size(); members.emplace_back();
    queue.clear(); queue.push_back(seed); aggOf[(std::size_t)seed]=a;
    for(std::size_t q=0;q<queue.size() && (PetscInt)queue.size()<cfg.targetAggregateSize;++q) {
      std::vector<PetscInt> cand;
      forLocalNbr(queue[q],[&](PetscInt j){ if(aggOf[(std::size_t)j]<0) cand.push_back(j); });
      std::sort(cand.begin(),cand.end()); cand.erase(std::unique(cand.begin(),cand.end()),cand.end());
      for(PetscInt j:cand) {
        if((PetscInt)queue.size()>=cfg.targetAggregateSize) break;
        if(aggOf[(std::size_t)j]<0) { aggOf[(std::size_t)j]=a; queue.push_back(j); }
      }
    }
    members.back()=queue;
  }

  // Merge undersized pieces only over a real matrix edge.  Isolated local rows
  // are allowed to remain singleton aggregates; this is required near MPI
  // partition boundaries on very coarse levels.
  for(PetscInt a=0;a<(PetscInt)members.size();++a) {
    if(members[(std::size_t)a].empty() || (PetscInt)members[(std::size_t)a].size()>=cfg.minAggregateSize) continue;
    PetscInt best=-1,bestSize=std::numeric_limits<PetscInt>::max(); PetscBool bestFits=PETSC_FALSE;
    for(PetscInt i:members[(std::size_t)a]) forLocalNbr(i,[&](PetscInt j){
      const PetscInt b=aggOf[(std::size_t)j]; if(b<0 || b==a || members[(std::size_t)b].empty()) return;
      const PetscInt combined=(PetscInt)members[(std::size_t)a].size()+(PetscInt)members[(std::size_t)b].size();
      const PetscBool fits=combined<=cfg.softMaxAggregateSize?PETSC_TRUE:PETSC_FALSE;
      const PetscInt bs=(PetscInt)members[(std::size_t)b].size();
      if(best<0 || (fits&&!bestFits) || (fits==bestFits && bs<bestSize) || (fits==bestFits && bs==bestSize && b<best)) {
        best=b; bestSize=bs; bestFits=fits;
      }
    });
    if(best>=0) {
      for(PetscInt i:members[(std::size_t)a]) { aggOf[(std::size_t)i]=best; members[(std::size_t)best].push_back(i); }
      members[(std::size_t)a].clear();
    }
  }

  std::vector<PetscInt> oldToCompact(members.size(),-1); PetscInt localAgg=0;
  for(PetscInt a=0;a<(PetscInt)members.size();++a) if(!members[(std::size_t)a].empty()) oldToCompact[(std::size_t)a]=localAgg++;
  PetscInt aggStart=0; PetscCallMPI(MPI_Exscan(&localAgg,&aggStart,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD)); if(rank==0) aggStart=0;
  PetscInt globalAgg=0; PetscCallMPI(MPI_Allreduce(&localAgg,&globalAgg,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  if(A.globalRows>0 && globalAgg<=0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG coarse aggregation produced no aggregates");

  L.level=level; L.globalRows=A.globalRows; L.localStart=A.localStart; L.localEnd=A.localEnd;
  L.globalAggregates=globalAgg; L.localAggregateStart=aggStart; L.localAggregateEnd=aggStart+localAgg;
  L.aggregateId.assign((std::size_t)n,-1);
  for(PetscInt i=0;i<n;++i) {
    const PetscInt a=aggOf[(std::size_t)i];
    if(a<0 || oldToCompact[(std::size_t)a]<0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG coarse row unassigned");
    L.aggregateId[(std::size_t)i]=aggStart+oldToCompact[(std::size_t)a];
  }

  PetscInt lmin=std::numeric_limits<PetscInt>::max(),lmax=0,lunder=0,lover=0;
  unsigned long long lassigned=0;
  for(const auto& v:members) if(!v.empty()) { const PetscInt s=(PetscInt)v.size(); lmin=std::min(lmin,s); lmax=std::max(lmax,s); lassigned+=(unsigned long long)s; if(s<cfg.minAggregateSize)++lunder; if(s>cfg.softMaxAggregateSize)++lover; }
  if(localAgg==0) lmin=std::numeric_limits<PetscInt>::max();
  PetscInt gmin=0,gmax=0,gunder=0,gover=0; unsigned long long gassigned=0;
  PetscCallMPI(MPI_Allreduce(&lmin,&gmin,1,MPIU_INT,MPI_MIN,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&lmax,&gmax,1,MPIU_INT,MPI_MAX,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&lunder,&gunder,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&lover,&gover,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&lassigned,&gassigned,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  if(gassigned!=(unsigned long long)A.globalRows) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG coarse aggregate assignment incomplete");
  const double mean=globalAgg?((double)A.globalRows/(double)globalAgg):0.0;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_CUSTOM_AMG_GATE4_AGG level=%" PetscInt_FMT " status=PASS scope=rank_local_matrix_graph rows=%" PetscInt_FMT " aggregates=%" PetscInt_FMT " coarsening=%.6f size=[%" PetscInt_FMT ",%.6f,%" PetscInt_FMT "] undersized=%" PetscInt_FMT " softOversized=%" PetscInt_FMT " crossRankAggregates=0\n",
    level,A.globalRows,globalAgg,globalAgg?((double)A.globalRows/(double)globalAgg):0.0,gmin,mean,gmax,gunder,gover));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customPressureAMGBuildNextCoarse(CustomPressureAMGCSR& Af,
  CustomPressureAMGLevel& Lf,CustomPressureAMGCSR& Ac,PetscInt nextLevel) {
  PetscFunctionBeginUser;
  int rank=0,size=1; PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank)); PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  const PetscInt nf=Af.localEnd-Af.localStart, nc=Lf.localAggregateEnd-Lf.localAggregateStart;
  if((PetscInt)Lf.aggregateId.size()!=nf) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"custom AMG next-coarse aggregate map size mismatch");
  std::vector<PetscInt> ghostAgg;
  PetscCall(customPressureAMGExchangeAggregateIds(Af.halo,Lf.aggregateId,ghostAgg,49600+(int)nextLevel*7));
  auto colAgg=[&](PetscInt li)->PetscInt {
    return li<Af.halo.nOwned ? Lf.aggregateId[(std::size_t)li] : ghostAgg[(std::size_t)(li-Af.halo.nOwned)];
  };
  std::vector<CustomPressureAMGTriplet> t;
  t.reserve(Af.values.size());
  for(PetscInt i=0;i<nf;++i) {
    const PetscInt I=Lf.aggregateId[(std::size_t)i];
    if(I<Lf.localAggregateStart || I>=Lf.localAggregateEnd) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG next-coarse row aggregate not local");
    for(PetscInt k=Af.rowPtr[(std::size_t)i];k<Af.rowPtr[(std::size_t)i+1];++k) {
      const PetscInt J=colAgg(Af.colLocal[(std::size_t)k]);
      if(J<0 || J>=Lf.globalAggregates) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG next-coarse column aggregate invalid");
      t.push_back({I,J,Af.values[(std::size_t)k]});
    }
  }
  std::sort(t.begin(),t.end(),[](const auto& a,const auto& b){return (a.row<b.row)||(a.row==b.row&&a.col<b.col);});
  std::vector<CustomPressureAMGTriplet> m; m.reserve(t.size());
  for(const auto& q:t) { if(!m.empty()&&m.back().row==q.row&&m.back().col==q.col)m.back().value+=q.value; else m.push_back(q); }
  std::vector<CustomPressureAMGTriplet>().swap(t);

  Ac.globalRows=Lf.globalAggregates; Ac.localStart=Lf.localAggregateStart; Ac.localEnd=Lf.localAggregateEnd;
  Ac.rowPtr.assign((std::size_t)nc+1,0); Ac.colGid.clear(); Ac.colLocal.clear(); Ac.values.clear(); Ac.diagonal.assign((std::size_t)nc,0.0);
  Ac.colGid.reserve(m.size()); Ac.values.reserve(m.size()); std::size_t p=0;
  for(PetscInt lr=0;lr<nc;++lr) {
    const PetscInt rg=Ac.localStart+lr;
    while(p<m.size()&&m[p].row<rg)++p;
    while(p<m.size()&&m[p].row==rg) { if(m[p].value!=0.0){Ac.colGid.push_back(m[p].col);Ac.values.push_back(m[p].value);if(m[p].col==rg)Ac.diagonal[(std::size_t)lr]=m[p].value;} ++p; }
    Ac.rowPtr[(std::size_t)lr+1]=(PetscInt)Ac.colGid.size();
  }
  std::vector<CustomPressureAMGTriplet>().swap(m);
  PetscInt lbad=0; double lmin=std::numeric_limits<double>::infinity(),lmax=0.0;
  for(double d:Ac.diagonal){if(!std::isfinite(d)||d<=0.0)++lbad;else{lmin=std::min(lmin,d);lmax=std::max(lmax,d);}}
  PetscInt gbad=0; double gmin=0.0,gmax=0.0; PetscCallMPI(MPI_Allreduce(&lbad,&gbad,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&lmin,&gmin,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD)); PetscCallMPI(MPI_Allreduce(&lmax,&gmax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  if(gbad) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"custom AMG recursive coarse diagonal invalid");
  std::vector<PetscInt> counts((std::size_t)size,0),needed; PetscInt lc=nc;
  PetscCallMPI(MPI_Allgather(&lc,1,MPIU_INT,counts.data(),1,MPIU_INT,PETSC_COMM_WORLD));
  needed.reserve(Ac.colGid.size()); for(PetscInt g:Ac.colGid) if(g<Ac.localStart||g>=Ac.localEnd) needed.push_back(g);
  const std::string label="custom_amg_L"+std::to_string((long long)nextLevel);
  PetscCall(buildCustomPeerHalo(counts,rank,std::move(needed),Ac.halo,label.c_str()));
  Ac.colLocal.resize(Ac.colGid.size()); for(std::size_t k=0;k<Ac.colGid.size();++k){Ac.colLocal[k]=customPeerLocalIndex(Ac.halo,Ac.colGid[k]);if(Ac.colLocal[k]<0)SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG recursive coarse halo missing column");}
  unsigned long long lnnz=(unsigned long long)Ac.values.size(),gnnz=0; PetscCallMPI(MPI_Allreduce(&lnnz,&gnnz,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_CUSTOM_AMG_GATE4_COARSE level=%" PetscInt_FMT " status=PASS rows=%" PetscInt_FMT " nnz=%llu avgNnz=%.6f diag=[%.12e,%.12e] transfer=unsmoothed_PtAP_specialized explicitCSR=1\n",
    nextLevel,Ac.globalRows,gnnz,Ac.globalRows?((double)gnnz/(double)Ac.globalRows):0.0,gmin,gmax));
  PetscFunctionReturn(PETSC_SUCCESS);
}



static PetscErrorCode customPressureAMGBuildExplicitSmoothedTransfer(CustomPressureAMGRuntime& R,
  PetscInt lev,CustomPressureAMGLevel& L) {
  PetscFunctionBeginUser;
  auto& A=L.A; const PetscInt n=A.localEnd-A.localStart;
  if(!(L.lambdaMax>0.0) || (PetscInt)L.aggregateId.size()!=n) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG explicit SA transfer prerequisites missing");
  const double omega=(double)R.amg.config.smoothedAggregationDamping/L.lambdaMax;
  std::vector<PetscInt> ghostAgg; PetscCall(customPressureAMGExchangeAggregateIds(A.halo,L.aggregateId,ghostAgg,50400+(int)lev*13));
  auto aggOfLocal=[&](PetscInt li)->PetscInt{return li<A.halo.nOwned?L.aggregateId[(std::size_t)li]:ghostAgg[(std::size_t)(li-A.halo.nOwned)];};
  auto& P=L.P; P.rowPtr.assign((std::size_t)n+1,0);P.colGid.clear();P.values.clear();std::vector<std::pair<PetscInt,double>> row;
  for(PetscInt i=0;i<n;++i){row.clear();const PetscInt own=L.aggregateId[(std::size_t)i];row.push_back({own,1.0});const double invd=1.0/A.diagonal[(std::size_t)i];for(PetscInt k=A.rowPtr[(std::size_t)i];k<A.rowPtr[(std::size_t)i+1];++k)row.push_back({aggOfLocal(A.colLocal[(std::size_t)k]),-omega*invd*A.values[(std::size_t)k]});customPressureAMGNormalizePruneRow(row,own,R.amg.config.interpolationMaxRowNnz);for(const auto&e:row){P.colGid.push_back(e.first);P.values.push_back(e.second);}P.rowPtr[(std::size_t)i+1]=(PetscInt)P.colGid.size();}
  const std::string label="custom_sa_P"+std::to_string((long long)lev);PetscCall(customPressureAMGFinalizeTransferHalo(L,P,label.c_str()));
  double ldef=0.0;PetscInt lmaxnnz=0;for(PetscInt i=0;i<n;++i){double sum=0.0;const PetscInt nn=P.rowPtr[(std::size_t)i+1]-P.rowPtr[(std::size_t)i];lmaxnnz=std::max(lmaxnnz,nn);for(PetscInt k=P.rowPtr[(std::size_t)i];k<P.rowPtr[(std::size_t)i+1];++k)sum+=P.values[(std::size_t)k];ldef=std::max(ldef,std::abs(sum-1.0));}double gdef=0.0;PetscInt gmaxnnz=0;unsigned long long lnnz=(unsigned long long)P.values.size(),gnnz=0;PetscCallMPI(MPI_Allreduce(&ldef,&gdef,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));PetscCallMPI(MPI_Allreduce(&lmaxnnz,&gmaxnnz,1,MPIU_INT,MPI_MAX,PETSC_COMM_WORLD));PetscCallMPI(MPI_Allreduce(&lnnz,&gnnz,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_CUSTOM_AMG_SA_TRANSFER level=%" PetscInt_FMT " status=PASS construction=(I-omega_DinvA)Pt_explicit_coarse omega=%.12e damping=%.6f lambdaMax=%.12e rows=%" PetscInt_FMT " nnz=%llu avgNnz=%.6f maxRowNnz=%" PetscInt_FMT " rowSumDefect=%.3e\n",lev,omega,(double)R.amg.config.smoothedAggregationDamping,L.lambdaMax,L.globalRows,gnnz,L.globalRows?((double)gnnz/(double)L.globalRows):0.0,gmaxnnz,gdef));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customPressureAMGBuildNextCoarseSmoothed(CustomPressureAMGCSR& Af,
  CustomPressureAMGLevel& Lf,CustomPressureAMGCSR& Ac,PetscInt nextLevel) {
  PetscFunctionBeginUser;
  int rank=0,size=1;PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  auto& P=Lf.P;const PetscInt nf=Af.localEnd-Af.localStart,nc=Lf.localAggregateEnd-Lf.localAggregateStart;
  if((PetscInt)P.rowPtr.size()!=nf+1)SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"custom AMG SA next coarse transfer size mismatch");
  std::vector<CustomPressureAMGPEntry> owned(P.values.size()),ghost;for(std::size_t k=0;k<P.values.size();++k){owned[k].coarse=P.colGid[k];owned[k].value=P.values[k];}
  std::vector<PetscInt> ghostPtr;PetscCall(customPressureAMGExchangeSparseRows(Af.halo,P.rowPtr,owned,ghostPtr,ghost,50500+(int)nextLevel*17));
  auto pRange=[&](PetscInt li,const CustomPressureAMGPEntry*&b,const CustomPressureAMGPEntry*&e){if(li<Af.halo.nOwned){b=owned.data()+P.rowPtr[(std::size_t)li];e=owned.data()+P.rowPtr[(std::size_t)li+1];}else{const PetscInt g=li-Af.halo.nOwned;b=ghost.data()+ghostPtr[(std::size_t)g];e=ghost.data()+ghostPtr[(std::size_t)g+1];}};
  std::vector<PetscInt> counts((std::size_t)size,0),off((std::size_t)size+1,0);PetscCallMPI(MPI_Allgather(&nc,1,MPIU_INT,counts.data(),1,MPIU_INT,PETSC_COMM_WORLD));for(int r=0;r<size;++r)off[(std::size_t)r+1]=off[(std::size_t)r]+counts[(std::size_t)r];if(off.back()!=Lf.globalAggregates)SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG SA coarse ownership mismatch");auto owner=[&](PetscInt g)->int{auto it=std::upper_bound(off.begin(),off.end(),g);int r=(int)(it-off.begin())-1;return(r>=0&&r<size&&g>=off[(std::size_t)r]&&g<off[(std::size_t)r+1])?r:-1;};
  std::vector<std::unordered_map<CustomPressureAMGPairKey,double,CustomPressureAMGPairHash>> by((std::size_t)size);std::vector<std::pair<PetscInt,double>> qrow;unsigned long long localRaw=0;
  for(PetscInt i=0;i<nf;++i){
    qrow.clear();
    for(PetscInt k=Af.rowPtr[(std::size_t)i];k<Af.rowPtr[(std::size_t)i+1];++k){const double a=Af.values[(std::size_t)k];const CustomPressureAMGPEntry *pb=nullptr,*pe=nullptr;pRange(Af.colLocal[(std::size_t)k],pb,pe);for(const auto*pp=pb;pp!=pe;++pp)qrow.push_back({pp->coarse,a*pp->value});}
    std::sort(qrow.begin(),qrow.end(),[](const auto&a,const auto&b){return a.first<b.first;});std::size_t w=0;for(std::size_t k=0;k<qrow.size();){const PetscInt g=qrow[k].first;double v=0.0;std::size_t j=k;for(;j<qrow.size()&&qrow[j].first==g;++j)v+=qrow[j].second;qrow[w++]={g,v};k=j;}qrow.resize(w);
    for(PetscInt kp=P.rowPtr[(std::size_t)i];kp<P.rowPtr[(std::size_t)i+1];++kp){const PetscInt I=P.colGid[(std::size_t)kp];const double pi=P.values[(std::size_t)kp];const int ro=owner(I);if(ro<0)SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG SA recursive coarse row owner invalid");auto& map=by[(std::size_t)ro];for(const auto&q:qrow){const double aij=pi*q.second;if(aij!=0.0){map[{I,q.first}]+=aij;++localRaw;}}}
  }
  std::vector<int> sc((std::size_t)size,0),rc((std::size_t)size,0),sd((std::size_t)size,0),rd((std::size_t)size,0);std::size_t total=0;for(int r=0;r<size;++r){if(by[(std::size_t)r].size()>(std::size_t)std::numeric_limits<int>::max())SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"custom AMG SA recursive unique triplet peer count exceeds MPI int");sc[(std::size_t)r]=(int)by[(std::size_t)r].size();total+=by[(std::size_t)r].size();}for(int r=1;r<size;++r)sd[(std::size_t)r]=sd[(std::size_t)r-1]+sc[(std::size_t)r-1];std::vector<CustomPressureAMGTriplet> send;send.reserve(total);for(int r=0;r<size;++r)for(const auto& kv:by[(std::size_t)r])send.push_back({kv.first.row,kv.first.col,kv.second});PetscCallMPI(MPI_Alltoall(sc.data(),1,MPI_INT,rc.data(),1,MPI_INT,PETSC_COMM_WORLD));for(int r=1;r<size;++r)rd[(std::size_t)r]=rd[(std::size_t)r-1]+rc[(std::size_t)r-1];std::vector<CustomPressureAMGTriplet> recv((std::size_t)(size?rd.back()+rc.back():0));MPI_Datatype tt=MPI_DATATYPE_NULL;PetscCallMPI(MPI_Type_contiguous((int)sizeof(CustomPressureAMGTriplet),MPI_BYTE,&tt));PetscCallMPI(MPI_Type_commit(&tt));PetscCallMPI(MPI_Alltoallv(send.empty()?nullptr:send.data(),sc.data(),sd.data(),tt,recv.empty()?nullptr:recv.data(),rc.data(),rd.data(),tt,PETSC_COMM_WORLD));PetscCallMPI(MPI_Type_free(&tt));
  std::sort(recv.begin(),recv.end(),[](const auto&a,const auto&b){return a.row<b.row||(a.row==b.row&&a.col<b.col);});std::vector<CustomPressureAMGTriplet> merged;merged.reserve(recv.size());for(const auto&t:recv){if(t.row<Lf.localAggregateStart||t.row>=Lf.localAggregateEnd)SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG SA recursive received non-owned coarse row");if(!merged.empty()&&merged.back().row==t.row&&merged.back().col==t.col)merged.back().value+=t.value;else merged.push_back(t);}Ac.globalRows=Lf.globalAggregates;Ac.localStart=Lf.localAggregateStart;Ac.localEnd=Lf.localAggregateEnd;Ac.rowPtr.assign((std::size_t)nc+1,0);Ac.colGid.clear();Ac.values.clear();Ac.diagonal.assign((std::size_t)nc,0.0);std::size_t m=0;for(PetscInt i=0;i<nc;++i){const PetscInt rg=Ac.localStart+i;while(m<merged.size()&&merged[m].row<rg)++m;while(m<merged.size()&&merged[m].row==rg){if(merged[m].value!=0.0){Ac.colGid.push_back(merged[m].col);Ac.values.push_back(merged[m].value);if(merged[m].col==rg)Ac.diagonal[(std::size_t)i]=merged[m].value;}++m;}Ac.rowPtr[(std::size_t)i+1]=(PetscInt)Ac.colGid.size();}
  PetscInt lbad=0;double lmin=std::numeric_limits<double>::infinity(),lmax=0.0;for(double d:Ac.diagonal){if(!(d>0.0)||!std::isfinite(d))++lbad;else{lmin=std::min(lmin,d);lmax=std::max(lmax,d);}}PetscInt gbad=0;double gmin=0,gmax=0;PetscCallMPI(MPI_Allreduce(&lbad,&gbad,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));PetscCallMPI(MPI_Allreduce(&lmin,&gmin,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD));PetscCallMPI(MPI_Allreduce(&lmax,&gmax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));if(gbad)SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"custom AMG SA recursive coarse diagonal invalid");std::vector<PetscInt> needed;for(PetscInt g:Ac.colGid)if(g<Ac.localStart||g>=Ac.localEnd)needed.push_back(g);const std::string label="custom_sa_A"+std::to_string((long long)nextLevel);PetscCall(buildCustomPeerHalo(counts,rank,std::move(needed),Ac.halo,label.c_str()));Ac.colLocal.resize(Ac.colGid.size());for(std::size_t k=0;k<Ac.colGid.size();++k){Ac.colLocal[k]=customPeerLocalIndex(Ac.halo,Ac.colGid[k]);if(Ac.colLocal[k]<0)SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG SA recursive coarse halo missing column");}unsigned long long lnnz=(unsigned long long)Ac.values.size(),gnnz=0,graw=0;PetscCallMPI(MPI_Allreduce(&lnnz,&gnnz,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));PetscCallMPI(MPI_Allreduce(&localRaw,&graw,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_CUSTOM_AMG_SA_COARSE level=%" PetscInt_FMT " status=PASS construction=PtAP_specialized_smoothed_merged rows=%" PetscInt_FMT " nnz=%llu avgNnz=%.6f rawTerms=%llu diag=[%.12e,%.12e]\n",nextLevel,Ac.globalRows,gnnz,Ac.globalRows?((double)gnnz/(double)Ac.globalRows):0.0,graw,gmin,gmax));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customPressureAMGApplyLevel(CustomPressureAMGRuntime& R,PetscInt lev,
  const std::vector<double>& x,std::vector<double>& y,int tag) {
  PetscFunctionBeginUser;
  if(lev<0 || lev>=(PetscInt)R.amg.levels.size()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"custom AMG apply level out of range");
  if(lev==0) PetscCall(R.amg.fine.apply(R.amg.fine.ctx,x,y));
  else PetscCall(customPressureAMGCoarseSpMV(R.amg.levels[(std::size_t)lev].A,x,y,tag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customPressureAMGEstimateLambda(CustomPressureAMGRuntime& R,PetscInt lev) {
  PetscFunctionBeginUser;
  auto& L=R.amg.levels[(std::size_t)lev];
  const auto& d=L.A.diagonal; const PetscInt n=L.localEnd-L.localStart;
  if((PetscInt)d.size()!=n) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"custom AMG spectral diagonal size mismatch");
  std::vector<double> v((std::size_t)n),u((std::size_t)n),au,w;
  for(PetscInt i=0;i<n;++i){const double g=(double)(L.localStart+i+1);v[(std::size_t)i]=std::sin(0.731*g)+0.27*std::cos(1.117*g);}
  double vn=0.0; PetscCall(customPressureNorm2(v,&vn)); if(!(vn>0.0))SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"custom AMG power vector has zero norm");
  for(double& q:v)q/=vn;
  double ray=0.0;
  for(PetscInt it=0;it<R.amg.config.powerIterations;++it) {
    for(PetscInt i=0;i<n;++i)u[(std::size_t)i]=v[(std::size_t)i]/std::sqrt(d[(std::size_t)i]);
    PetscCall(customPressureAMGApplyLevel(R,lev,u,au,49700+(int)lev*17+(int)(it%7)));
    w.resize((std::size_t)n); for(PetscInt i=0;i<n;++i)w[(std::size_t)i]=au[(std::size_t)i]/std::sqrt(d[(std::size_t)i]);
    double num=0.0,den=0.0; PetscCall(customPressureDot(v,w,&num)); PetscCall(customPressureDot(v,v,&den)); ray=num/std::max(den,1e-300);
    double wn=0.0; PetscCall(customPressureNorm2(w,&wn)); if(!(wn>0.0)||!std::isfinite(wn))SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"custom AMG power iteration failed");
    for(PetscInt i=0;i<n;++i)v[(std::size_t)i]=w[(std::size_t)i]/wn;
  }
  if(!(ray>0.0)||!std::isfinite(ray)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"custom AMG lambda estimate invalid");
  L.lambdaMax=(double)R.amg.config.lambdaSafety*ray;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_CUSTOM_AMG_GATE4_SPECTRUM level=%" PetscInt_FMT " status=PASS powerIts=%" PetscInt_FMT " lambdaRayleigh=%.12e lambdaMaxUsed=%.12e safety=%.6f lambdaLowFraction=%.6f\n",
    lev,R.amg.config.powerIterations,ray,L.lambdaMax,(double)R.amg.config.lambdaSafety,(double)R.amg.config.lambdaLowFraction));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customPressureAMGChebyshevCorrection(CustomPressureAMGRuntime& R,PetscInt lev,
  const std::vector<double>& rhs,std::vector<double>& corr,int tagBase) {
  PetscFunctionBeginUser;
  auto& L=R.amg.levels[(std::size_t)lev]; const auto& d=L.A.diagonal; const PetscInt n=L.localEnd-L.localStart;
  if((PetscInt)rhs.size()!=n || (PetscInt)d.size()!=n) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"custom AMG Chebyshev vector size mismatch");
  const double hi=L.lambdaMax,lo=R.amg.config.lambdaLowFraction*hi;
  if(!(hi>lo&&lo>0.0))SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"custom AMG Chebyshev interval invalid");
  const PetscInt degree=R.amg.config.chebyshevDegree; if(degree<1)SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"custom AMG Chebyshev degree must be positive");
  std::vector<double> bs((std::size_t)n),y((std::size_t)n,0.0),u,au,hy((std::size_t)n,0.0);
  for(PetscInt i=0;i<n;++i)bs[(std::size_t)i]=rhs[(std::size_t)i]/std::sqrt(d[(std::size_t)i]);
  const double centre=0.5*(hi+lo),radius=0.5*(hi-lo),pi=3.141592653589793238462643383279502884;
  for(PetscInt k=0;k<degree;++k) {
    if(k>0) {
      u.resize((std::size_t)n); for(PetscInt i=0;i<n;++i)u[(std::size_t)i]=y[(std::size_t)i]/std::sqrt(d[(std::size_t)i]);
      PetscCall(customPressureAMGApplyLevel(R,lev,u,au,tagBase+(int)k));
      for(PetscInt i=0;i<n;++i)hy[(std::size_t)i]=au[(std::size_t)i]/std::sqrt(d[(std::size_t)i]);
    } else std::fill(hy.begin(),hy.end(),0.0);
    const double root=centre-radius*std::cos(pi*(2.0*(double)k+1.0)/(2.0*(double)degree));
    const double omega=1.0/root;
    for(PetscInt i=0;i<n;++i)y[(std::size_t)i]+=omega*(bs[(std::size_t)i]-hy[(std::size_t)i]);
  }
  corr.resize((std::size_t)n); for(PetscInt i=0;i<n;++i)corr[(std::size_t)i]=y[(std::size_t)i]/std::sqrt(d[(std::size_t)i]);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customPressureAMGBuildCoarseLU(CustomPressureAMGRuntime& R) {
  PetscFunctionBeginUser;
  auto& C=R.coarse; auto& A=R.amg.levels.back().A;
  int rank=0,size=1; PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank)); PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  const PetscInt nl=A.localEnd-A.localStart; C.n=A.globalRows; C.constantNullspace=R.amg.fine.constantNullspace;
  if(C.n<=0 || C.n>R.amg.config.coarseTargetRows) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"custom AMG final coarse system outside configured direct-solve target");
  C.counts.assign((std::size_t)size,0); int nli=(int)nl; PetscCallMPI(MPI_Allgather(&nli,1,MPI_INT,C.counts.data(),1,MPI_INT,PETSC_COMM_WORLD));
  C.displs.assign((std::size_t)size,0); for(int r=1;r<size;++r)C.displs[(std::size_t)r]=C.displs[(std::size_t)r-1]+C.counts[(std::size_t)r-1];
  struct T { PetscInt r,c; double v; };
  std::vector<T> loc; loc.reserve(A.values.size());
  for(PetscInt i=0;i<nl;++i)for(PetscInt k=A.rowPtr[(std::size_t)i];k<A.rowPtr[(std::size_t)i+1];++k)loc.push_back({A.localStart+i,A.colGid[(std::size_t)k],A.values[(std::size_t)k]});
  int lcount=(int)loc.size(); std::vector<int> bcnt((std::size_t)size,0),bdisp((std::size_t)size,0); PetscCallMPI(MPI_Gather(&lcount,1,MPI_INT,bcnt.data(),1,MPI_INT,0,PETSC_COMM_WORLD));
  int totalCount=0; if(rank==0){for(int r=1;r<size;++r)bdisp[(std::size_t)r]=bdisp[(std::size_t)r-1]+bcnt[(std::size_t)r-1];for(int v:bcnt)totalCount+=v;}
  std::vector<T> recv; if(rank==0)recv.resize((std::size_t)totalCount);
  MPI_Datatype tt=MPI_DATATYPE_NULL; PetscCallMPI(MPI_Type_contiguous((int)sizeof(T),MPI_BYTE,&tt)); PetscCallMPI(MPI_Type_commit(&tt));
  PetscCallMPI(MPI_Gatherv(loc.empty()?nullptr:loc.data(),lcount,tt,rank==0?recv.data():nullptr,bcnt.data(),bdisp.data(),tt,0,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Type_free(&tt));
  if(rank==0) {
    const std::size_t n=(std::size_t)C.n; if(n>0 && n>std::numeric_limits<std::size_t>::max()/n)SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"custom AMG coarse dense size overflow");
    C.lu.assign(n*n,0.0);
    for(const auto& a:recv) {
      if(a.r<0||a.r>=C.n||a.c<0||a.c>=C.n)SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"custom AMG coarse gather index invalid");
      C.lu[(std::size_t)a.r*n+(std::size_t)a.c]+=a.v;
    }
    if(C.constantNullspace && C.n>0){for(PetscInt j=0;j<C.n;++j){C.lu[(std::size_t)j]=0.0;C.lu[(std::size_t)j*n]=0.0;}C.lu[0]=1.0;}
    C.piv.assign(n,0);
    for(PetscInt k=0;k<C.n;++k){PetscInt p=k;double mx=std::abs(C.lu[(std::size_t)k*n+(std::size_t)k]);for(PetscInt i=k+1;i<C.n;++i){double q=std::abs(C.lu[(std::size_t)i*n+(std::size_t)k]);if(q>mx){mx=q;p=i;}}if(!(mx>1e-30)||!std::isfinite(mx))SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"custom AMG coarse LU zero pivot");C.piv[(std::size_t)k]=p;if(p!=k)for(PetscInt j=0;j<C.n;++j)std::swap(C.lu[(std::size_t)k*n+(std::size_t)j],C.lu[(std::size_t)p*n+(std::size_t)j]);const double piv=C.lu[(std::size_t)k*n+(std::size_t)k];for(PetscInt i=k+1;i<C.n;++i){double& lik=C.lu[(std::size_t)i*n+(std::size_t)k];lik/=piv;for(PetscInt j=k+1;j<C.n;++j)C.lu[(std::size_t)i*n+(std::size_t)j]-=lik*C.lu[(std::size_t)k*n+(std::size_t)j];}}
  }
  C.ready=PETSC_TRUE;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_CUSTOM_AMG_GATE4_COARSE_LU status=PASS rows=%" PetscInt_FMT " owner=rank0 factor=dense_partial_pivot_LU constantNullspace=%d gatheredOnceAtSetup=1\n",C.n,(int)C.constantNullspace));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customPressureAMGCoarseSolve(CustomPressureAMGRuntime& R,const std::vector<double>& rhs,std::vector<double>& x) {
  PetscFunctionBeginUser;
  auto& C=R.coarse; int rank=0; PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  if(!C.ready)SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG coarse solver not ready");
  const int nl=C.counts[(std::size_t)rank]; if((int)rhs.size()!=nl)SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"custom AMG coarse RHS size mismatch");
  std::vector<double> b; if(rank==0)b.resize((std::size_t)C.n);
  PetscCallMPI(MPI_Gatherv(rhs.empty()?nullptr:rhs.data(),nl,MPI_DOUBLE,rank==0?b.data():nullptr,C.counts.data(),C.displs.data(),MPI_DOUBLE,0,PETSC_COMM_WORLD));
  if(rank==0){const std::size_t n=(std::size_t)C.n;if(C.constantNullspace&&C.n>0)b[0]=0.0;for(PetscInt k=0;k<C.n;++k){const PetscInt p=C.piv[(std::size_t)k];if(p!=k)std::swap(b[(std::size_t)k],b[(std::size_t)p]);}for(PetscInt i=1;i<C.n;++i)for(PetscInt j=0;j<i;++j)b[(std::size_t)i]-=C.lu[(std::size_t)i*n+(std::size_t)j]*b[(std::size_t)j];for(PetscInt ii=C.n;ii-->0;){double q=b[(std::size_t)ii];for(PetscInt j=ii+1;j<C.n;++j)q-=C.lu[(std::size_t)ii*n+(std::size_t)j]*b[(std::size_t)j];b[(std::size_t)ii]=q/C.lu[(std::size_t)ii*n+(std::size_t)ii];}}
  x.assign((std::size_t)nl,0.0); PetscCallMPI(MPI_Scatterv(rank==0?b.data():nullptr,C.counts.data(),C.displs.data(),MPI_DOUBLE,x.empty()?nullptr:x.data(),nl,MPI_DOUBLE,0,PETSC_COMM_WORLD));
  if(C.constantNullspace)PetscCall(customPressureProjectConstant(x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customPressureAMGRestrict(CustomPressureAMGLevel& L,const CustomPressureAMGLevel& C,
  const std::vector<double>& rf,std::vector<double>& rc) {
  PetscFunctionBeginUser;
  if(!L.P.rowPtr.empty()) {
    if(L.P.coarseLocalStart!=C.localStart || L.P.coarseLocalEnd!=C.localEnd) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG smoothed restriction coarse ownership mismatch");
    PetscCall(customPressureAMGTransferRestrict(L.P,rf,rc,50600+(int)L.level*19));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if(rf.size()!=L.aggregateId.size())SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"custom AMG restriction size mismatch");
  rc.assign((std::size_t)(C.localEnd-C.localStart),0.0);
  for(std::size_t i=0;i<rf.size();++i){const PetscInt g=L.aggregateId[i];if(g<C.localStart||g>=C.localEnd)SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG restriction aggregate is not rank-local");rc[(std::size_t)(g-C.localStart)]+=rf[i];}
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customPressureAMGProlongAdd(CustomPressureAMGLevel& L,const CustomPressureAMGLevel& C,
  const std::vector<double>& ec,std::vector<double>& xf) {
  PetscFunctionBeginUser;
  if(!L.P.rowPtr.empty()) {
    if(L.P.coarseLocalStart!=C.localStart || L.P.coarseLocalEnd!=C.localEnd) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG smoothed prolongation coarse ownership mismatch");
    PetscCall(customPressureAMGTransferProlongAdd(L.P,ec,xf,50700+(int)L.level*19));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if(xf.size()!=L.aggregateId.size() || (PetscInt)ec.size()!=C.localEnd-C.localStart)SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"custom AMG prolongation size mismatch");
  for(std::size_t i=0;i<xf.size();++i){const PetscInt g=L.aggregateId[i];if(g<C.localStart||g>=C.localEnd)SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG prolongation aggregate is not rank-local");xf[i]+=ec[(std::size_t)(g-C.localStart)];}
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customPressureAMGVcycleLevel(CustomPressureAMGRuntime& R,PetscInt lev,
  const std::vector<double>& rhs,std::vector<double>& x) {
  PetscFunctionBeginUser;
  const PetscInt last=(PetscInt)R.amg.levels.size()-1;
  if(lev==last){PetscCall(customPressureAMGCoarseSolve(R,rhs,x));PetscFunctionReturn(PETSC_SUCCESS);}
  auto& L=R.amg.levels[(std::size_t)lev]; auto& C=R.amg.levels[(std::size_t)lev+1];
  const PetscInt n=L.localEnd-L.localStart; if((PetscInt)rhs.size()!=n)SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"custom AMG V-cycle RHS size mismatch");
  std::vector<double> corr,ax,res,rc,ec;
  x.assign((std::size_t)n,0.0);
  PetscCall(customPressureAMGChebyshevCorrection(R,lev,rhs,corr,49800+(int)lev*31)); x=corr;
  PetscCall(customPressureAMGApplyLevel(R,lev,x,ax,49900+(int)lev*31)); res.resize((std::size_t)n);for(PetscInt i=0;i<n;++i)res[(std::size_t)i]=rhs[(std::size_t)i]-ax[(std::size_t)i];
  PetscCall(customPressureAMGRestrict(L,C,res,rc)); PetscCall(customPressureAMGVcycleLevel(R,lev+1,rc,ec)); PetscCall(customPressureAMGProlongAdd(L,C,ec,x));
  PetscCall(customPressureAMGApplyLevel(R,lev,x,ax,50000+(int)lev*31)); for(PetscInt i=0;i<n;++i)res[(std::size_t)i]=rhs[(std::size_t)i]-ax[(std::size_t)i];
  PetscCall(customPressureAMGChebyshevCorrection(R,lev,res,corr,50100+(int)lev*31)); for(PetscInt i=0;i<n;++i)x[(std::size_t)i]+=corr[(std::size_t)i];
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customPressureAMGVcycle(CustomPressureAMGRuntime& R,const std::vector<double>& rhs,std::vector<double>& x) {
  PetscFunctionBeginUser;
  if(!R.amg.numericReady || !R.coarse.ready)SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG V-cycle called before setup");
  std::vector<double> r=rhs; if(R.amg.fine.constantNullspace)PetscCall(customPressureProjectConstant(r));
  PetscCall(customPressureAMGVcycleLevel(R,0,r,x)); if(R.amg.fine.constantNullspace)PetscCall(customPressureProjectConstant(x));
  ++R.vcycleApplyCount; PetscFunctionReturn(PETSC_SUCCESS);
}


// -----------------------------------------------------------------------------
// Live outer pressure iterations
// -----------------------------------------------------------------------------
// Native FP64 PCG and stationary Richardson both operate on the exact
// matrix-free Schur residual.  Each outer step applies exactly one selected
// custom V-cycle (unsmoothed or smoothed) as M^{-1}.  No PETSc Mat/KSP/PC,
// explicit fine Schur, or generic fine-grid PtAP participates in this live
// custom solve.  The workspace vectors below are retained across SIMPLE
// iterations to avoid allocator noise and repeated large temporary vectors.
struct CustomPressureAMGLivePCGWorkspace {
  std::vector<double> x,r,z,p,ap;
};

struct CustomPressureAMGLivePCGResult {
  PetscInt its=0;
  double trueRel=std::numeric_limits<double>::infinity();
  PetscBool converged=PETSC_FALSE;
  PetscBool finite=PETSC_TRUE;
};

static unsigned long long customPressureAMGLivePCGWorkspaceBytes(const CustomPressureAMGLivePCGWorkspace& W) {
  return (unsigned long long)(W.x.capacity()+W.r.capacity()+W.z.capacity()+W.p.capacity()+W.ap.capacity())*sizeof(double);
}

// Stationary Richardson around the exact matrix-free Schur.  This path was
// retained because it provides a clean AMG-to-AMG comparison against the
// FULLFAST PETSc Richardson semantics.  Production cases pair it with the
// smoothed hierarchy; unsmoothed Richardson is intentionally not a public case
// preset because it proved insufficiently robust on the larger meshes.
struct CustomPressureAMGLiveRichardsonWorkspace {
  std::vector<double> x,r,z,ax;
};

struct CustomPressureAMGLiveRichardsonResult {
  PetscInt its=0;
  double trueRel=std::numeric_limits<double>::infinity();
  PetscBool converged=PETSC_FALSE;
  PetscBool finite=PETSC_TRUE;
};

static unsigned long long customPressureAMGLiveRichardsonWorkspaceBytes(const CustomPressureAMGLiveRichardsonWorkspace& W) {
  return (unsigned long long)(W.x.capacity()+W.r.capacity()+W.z.capacity()+W.ax.capacity())*sizeof(double);
}

static PetscErrorCode customPressureAMGLiveRichardsonSolve(CustomPressureAMGRuntime& R,
  CustomPressureBPlan& B,const CustomMomentumCSR& M,const std::vector<double>& rhs,
  double rtol,double atol,double dtol,PetscInt maxIts,double omega,
  CustomPressureAMGLiveRichardsonWorkspace& W,CustomPressureAMGLiveRichardsonResult& out) {
  PetscFunctionBeginUser;
  if(!R.amg.topologyReady || !R.amg.numericReady || !R.coarse.ready) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG live Richardson called before hierarchy setup");
  if(rtol<0.0 || atol<0.0 || dtol<0.0 || maxIts<1 || !(omega>0.0) || !std::isfinite(omega)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"custom AMG live Richardson invalid parameters");
  const PetscInt n=B.pressureHalo.nOwned;
  if((PetscInt)rhs.size()!=n) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"custom AMG live Richardson RHS size mismatch");
  out=CustomPressureAMGLiveRichardsonResult{};
  W.x.assign((std::size_t)n,0.0);
  W.r=rhs;
  if(R.amg.fine.constantNullspace) PetscCall(customPressureProjectConstant(W.r));
  double bnorm=0.0; PetscCall(customPressureNorm2(W.r,&bnorm));
  const double target=std::max(atol,rtol*bnorm);
  if(bnorm<=target) { out.its=0; out.trueRel=0.0; out.converged=PETSC_TRUE; PetscFunctionReturn(PETSC_SUCCESS); }
  for(PetscInt k=0;k<maxIts;++k) {
    PetscCall(customPressureAMGVcycle(R,W.r,W.z));
    if(W.z.size()!=W.x.size()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"custom AMG live Richardson correction size mismatch");
    for(std::size_t i=0;i<W.x.size();++i) W.x[i]+=omega*W.z[i];
    PetscCall(customPressureSchurApply(B,M,W.x,W.ax));
    W.r=rhs;
    for(std::size_t i=0;i<W.r.size();++i) W.r[i]-=W.ax[i];
    if(R.amg.fine.constantNullspace) PetscCall(customPressureProjectConstant(W.r));
    double rn=0.0; PetscCall(customPressureNorm2(W.r,&rn));
    out.its=k+1; out.trueRel=rn/std::max(bnorm,1e-300);
    if(!std::isfinite(rn) || !std::isfinite(out.trueRel)) { out.finite=PETSC_FALSE; break; }
    if(rn<=target) { out.converged=PETSC_TRUE; break; }
    if(dtol>0.0 && out.trueRel>dtol) break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customPressureAMGLivePCGSolve(CustomPressureAMGRuntime& R,
  CustomPressureBPlan& B,const CustomMomentumCSR& M,const std::vector<double>& rhs,
  double rtol,double atol,double dtol,PetscInt maxIts,
  CustomPressureAMGLivePCGWorkspace& W,CustomPressureAMGLivePCGResult& out) {
  PetscFunctionBeginUser;
  if(!R.amg.topologyReady || !R.amg.numericReady || !R.coarse.ready) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG live PCG called before hierarchy setup");
  if(rtol<0.0 || atol<0.0 || dtol<0.0 || maxIts<1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"custom AMG live PCG invalid tolerances");
  const PetscInt n=B.pressureHalo.nOwned;
  if((PetscInt)rhs.size()!=n) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"custom AMG live PCG RHS size mismatch");
  out=CustomPressureAMGLivePCGResult{};
  W.x.assign((std::size_t)n,0.0);
  W.r=rhs;
  if(R.amg.fine.constantNullspace) PetscCall(customPressureProjectConstant(W.r));
  double bnorm=0.0; PetscCall(customPressureNorm2(W.r,&bnorm));
  const double target=std::max(atol,rtol*bnorm);
  if(bnorm<=target) { out.its=0; out.trueRel=0.0; out.converged=PETSC_TRUE; PetscFunctionReturn(PETSC_SUCCESS); }
  PetscCall(customPressureAMGVcycle(R,W.r,W.z));
  W.p=W.z;
  double rho=0.0; PetscCall(customPressureDot(W.r,W.z,&rho));
  if(!(rho>0.0) || !std::isfinite(rho)) { out.finite=PETSC_FALSE; PetscFunctionReturn(PETSC_SUCCESS); }
  for(PetscInt k=0;k<maxIts;++k) {
    PetscCall(customPressureSchurApply(B,M,W.p,W.ap));
    double pap=0.0; PetscCall(customPressureDot(W.p,W.ap,&pap));
    if(!(pap>0.0) || !std::isfinite(pap) || !std::isfinite(rho)) { out.finite=PETSC_FALSE; break; }
    const double alpha=rho/pap;
    if(!std::isfinite(alpha)) { out.finite=PETSC_FALSE; break; }
    for(std::size_t i=0;i<W.x.size();++i) { W.x[i]+=alpha*W.p[i]; W.r[i]-=alpha*W.ap[i]; }
    if(R.amg.fine.constantNullspace) PetscCall(customPressureProjectConstant(W.r));
    double rn=0.0; PetscCall(customPressureNorm2(W.r,&rn));
    out.its=k+1; out.trueRel=rn/std::max(bnorm,1e-300);
    if(!std::isfinite(rn) || !std::isfinite(out.trueRel)) { out.finite=PETSC_FALSE; break; }
    if(rn<=target) { out.converged=PETSC_TRUE; break; }
    if(dtol>0.0 && out.trueRel>dtol) break;
    PetscCall(customPressureAMGVcycle(R,W.r,W.z));
    double rhoNew=0.0; PetscCall(customPressureDot(W.r,W.z,&rhoNew));
    if(!(rhoNew>0.0) || !std::isfinite(rhoNew)) { out.finite=PETSC_FALSE; break; }
    const double beta=rhoNew/rho;
    if(!std::isfinite(beta)) { out.finite=PETSC_FALSE; break; }
    for(std::size_t i=0;i<W.p.size();++i) W.p[i]=W.z[i]+beta*W.p[i];
    rho=rhoNew;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static unsigned long long customPressureAMGCSRBytes(const CustomPressureAMGCSR& A) {
  return (unsigned long long)(A.rowPtr.capacity()+A.colGid.capacity()+A.colLocal.capacity())*sizeof(PetscInt)
    +(unsigned long long)(A.values.capacity()+A.diagonal.capacity())*sizeof(double)
    +(unsigned long long)(A.halo.ghostGid.capacity()+A.halo.reqRecvGid.capacity()+A.halo.reqRecvLocalOffset.capacity())*sizeof(PetscInt)
    +(unsigned long long)(A.halo.exchangeSend.capacity()+A.halo.ghostValues.capacity())*sizeof(double);
}

// Build/rebuild the complete numerical hierarchy for the current rAU field.
// Topological aggregation and numerical coarse operators are kept in one runtime
// object because SIMPLE periodically refreshes the pressure preconditioner as
// momentum diagonals change.  The refresh interval is controlled by
// -p_preconditioner_refresh / [pressure] refresh.  No physics-specific data is
// consulted here beyond the already prepared exact B/rAU operator inputs.
static PetscErrorCode customPressureAMGBuildHierarchy(const Mesh& mesh,const Discrete& D,int rank,
  CustomPressureBPlan& B,const CustomMomentumCSR& M,PetscInt globalRows,PetscBool constantNullspace,
  const CustomPressureAMGConfig& cfg,CustomPressureAMGRuntime& R) {
  PetscFunctionBeginUser;
  R=CustomPressureAMGRuntime{}; R.amg.config=cfg;
  PetscCall(customPressureAMGInitFineOperator(B,M,globalRows,constantNullspace,R.exactCtx,R.amg.fine));
  R.amg.levels.resize(2);
  PetscCall(customPressureAMGBuildFineAggregation(mesh,D,rank,R.amg.config,R.amg.fine,R.amg.levels[0]));
  PetscCall(customPressureAMGComputeFineDiagonal(B,M,R.amg.levels[0].A.diagonal));
  if(R.amg.config.smoothedAggregation) {
    PetscCall(customPressureAMGEstimateLambda(R,0));
    PetscCall(customPressureAMGBuildFineSmoothedTransfer(R,B,M,R.amg.levels[0]));
    PetscCall(customPressureAMGBuildFirstCoarseSmoothed(B,M,R.amg.levels[0],R.amg.levels[1].A));
    PetscCall(customPressureAMGValidateFirstSmoothedGalerkin(R,R.amg.levels[0],R.amg.levels[1].A));
  } else {
    PetscCall(customPressureAMGBuildFirstCoarse(B,M,R.amg.levels[0],R.amg.levels[1].A));
  }
  R.amg.levels[1].level=1;R.amg.levels[1].globalRows=R.amg.levels[1].A.globalRows;R.amg.levels[1].localStart=R.amg.levels[1].A.localStart;R.amg.levels[1].localEnd=R.amg.levels[1].A.localEnd;
  PetscInt lev=1;
  while(R.amg.levels[(std::size_t)lev].globalRows>R.amg.config.coarseTargetRows) {
    if(R.amg.config.smoothedAggregation && !(R.amg.levels[(std::size_t)lev].lambdaMax>0.0)) PetscCall(customPressureAMGEstimateLambda(R,lev));
    PetscCall(customPressureAMGBuildExplicitAggregation(R.amg.levels[(std::size_t)lev].A,R.amg.config,lev,R.amg.levels[(std::size_t)lev]));
    const PetscInt nextRows=R.amg.levels[(std::size_t)lev].globalAggregates;
    if(nextRows<=0 || nextRows>=R.amg.levels[(std::size_t)lev].globalRows) break;
    if(R.amg.config.smoothedAggregation) PetscCall(customPressureAMGBuildExplicitSmoothedTransfer(R,lev,R.amg.levels[(std::size_t)lev]));
    R.amg.levels.emplace_back(); auto& N=R.amg.levels.back(); N.level=lev+1;
    if(R.amg.config.smoothedAggregation) PetscCall(customPressureAMGBuildNextCoarseSmoothed(R.amg.levels[(std::size_t)lev].A,R.amg.levels[(std::size_t)lev],N.A,lev+1));
    else PetscCall(customPressureAMGBuildNextCoarse(R.amg.levels[(std::size_t)lev].A,R.amg.levels[(std::size_t)lev],N.A,lev+1));
    N.globalRows=N.A.globalRows;N.localStart=N.A.localStart;N.localEnd=N.A.localEnd; ++lev;
    if(lev>12)SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG hierarchy exceeded safety level count");
  }
  if(R.amg.levels.back().globalRows>R.amg.config.coarseTargetRows)
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"custom AMG rank-local coarsening stalled above direct coarse target; process agglomeration/cross-rank aggregation is a later optimization");
  for(PetscInt l=0;l<(PetscInt)R.amg.levels.size()-1;++l) if(!(R.amg.levels[(std::size_t)l].lambdaMax>0.0)) PetscCall(customPressureAMGEstimateLambda(R,l));
  PetscCall(customPressureAMGBuildCoarseLU(R));
  R.amg.topologyReady=PETSC_TRUE;R.amg.numericReady=PETSC_TRUE;R.setupCount=1;
  unsigned long long localBytes=0; for(const auto& L:R.amg.levels){localBytes+=(unsigned long long)L.aggregateId.capacity()*sizeof(PetscInt)+customPressureAMGCSRBytes(L.A)+customPressureAMGTransferBytes(L.P);} if(rank==0)localBytes+=(unsigned long long)R.coarse.lu.capacity()*sizeof(double)+(unsigned long long)R.coarse.piv.capacity()*sizeof(PetscInt);
  unsigned long long globalBytes=0;PetscCallMPI(MPI_Allreduce(&localBytes,&globalBytes,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));R.retainedMiB=(double)globalBytes/(1024.0*1024.0);
  unsigned long long coarseNnz=0,transferNnz=0;for(std::size_t l=1;l<R.amg.levels.size();++l){unsigned long long q=(unsigned long long)R.amg.levels[l].A.values.size(),g=0;PetscCallMPI(MPI_Allreduce(&q,&g,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));coarseNnz+=g;}for(std::size_t l=0;l+1<R.amg.levels.size();++l){unsigned long long q=(unsigned long long)R.amg.levels[l].P.values.size(),g=0;PetscCallMPI(MPI_Allreduce(&q,&g,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));transferNnz+=g;}
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_CUSTOM_AMG_HIERARCHY status=PASS levels=%zu fineRows=%" PetscInt_FMT " finalRows=%" PetscInt_FMT " coarseNnz=%llu transferNnz=%llu retainedEstimateMiB=%.3f transfer=%s rankLocalAggregates=1 fineOperator=matrix_free_B_rAU_Bt coarseOperators=custom_FP64_CSR smoother=symmetric_scaled_Chebyshev_Jacobi coarseSolve=rank0_LU fineSchurCSR=0\n",
    R.amg.levels.size(),globalRows,R.amg.levels.back().globalRows,coarseNnz,transferNnz,R.retainedMiB,R.amg.config.smoothedAggregation?"smoothed_jacobi_R_equals_Pt":"unsmoothed_injection_R_equals_Pt"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customPressureAMGGate4Validate(CustomPressureAMGRuntime& R,CustomPressureBPlan& B,const CustomMomentumCSR& M) {
  PetscFunctionBeginUser;
  const PetscInt n=R.amg.fine.localEnd-R.amg.fine.localStart;
  std::vector<double> x((std::size_t)n),y((std::size_t)n),mx,my,amx,rhs,sol;
  for(PetscInt i=0;i<n;++i){const double g=(double)(R.amg.fine.localStart+i+1);x[(std::size_t)i]=std::sin(0.0037*g)+0.23*std::cos(0.0071*g);y[(std::size_t)i]=std::cos(0.0053*g)-0.19*std::sin(0.0097*g);}
  if(R.amg.fine.constantNullspace){PetscCall(customPressureProjectConstant(x));PetscCall(customPressureProjectConstant(y));}
  PetscCall(customPressureAMGVcycle(R,x,mx)); PetscCall(customPressureAMGVcycle(R,y,my));
  double xmy=0.0,ymx=0.0,xmx=0.0;PetscCall(customPressureDot(x,my,&xmy));PetscCall(customPressureDot(y,mx,&ymx));PetscCall(customPressureDot(x,mx,&xmx));
  const double sym=std::abs(xmy-ymx)/std::max({std::abs(xmy),std::abs(ymx),1e-300});
  if(!(xmx>0.0)||!std::isfinite(xmx))SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"custom AMG V-cycle is not positive on validation vector");
  // M^{-1} is a preconditioner, not a stationary iteration with an implied unit
  // Richardson step.  Its overall positive scalar is immaterial to PCG.  Validate
  // the V-cycle in a scale-invariant way: symmetry + positivity, then minimize
  // ||x-alpha*A*M^{-1}x||_2 over the single scalar alpha.  Keep the raw alpha=1
  // ratio as a diagnostic only.
  PetscCall(customPressureSchurApply(B,M,mx,amx));
  rhs=x; for(std::size_t i=0;i<rhs.size();++i)rhs[i]-=amx[i];
  double r0=0.0,r1Raw=0.0;PetscCall(customPressureNorm2(x,&r0));PetscCall(customPressureNorm2(rhs,&r1Raw));
  double xA=0.0,AA=0.0;PetscCall(customPressureDot(x,amx,&xA));PetscCall(customPressureDot(amx,amx,&AA));
  if(!(AA>0.0)||!std::isfinite(AA)||!std::isfinite(xA))SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"custom AMG scale-invariant V-cycle validation failed");
  const double alphaOpt=xA/AA;
  rhs=x; for(std::size_t i=0;i<rhs.size();++i)rhs[i]-=alphaOpt*amx[i];
  double r1Opt=0.0;PetscCall(customPressureNorm2(rhs,&r1Opt));
  const double oneCycleRaw=r1Raw/std::max(r0,1e-300),oneCycleOpt=r1Opt/std::max(r0,1e-300);
  const double symTol=(sizeof(PetscReal)<=4)?2e-4:2e-10;
  const PetscBool pass=(sym<=symTol && alphaOpt>0.0 && std::isfinite(oneCycleOpt) && oneCycleOpt<1.0)?PETSC_TRUE:PETSC_FALSE;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_CUSTOM_AMG_GATE4_VCYCLE status=%s symmetryRel=%.3e positivity=%.12e oneCycleResidualRatioRaw=%.6e oneCycleResidualRatioOpt=%.6e optimalAlpha=%.12e chebDegree=%" PetscInt_FMT " prePost=1plus1 fixedLinear=1 R_equals_Pt=1 scaleInvariantValidation=1\n",
    pass?"PASS":"FAIL",sym,xmx,oneCycleRaw,oneCycleOpt,alphaOpt,R.amg.config.chebyshevDegree));
  if(!pass)SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom AMG Gate-4 V-cycle validation failed");

  // Native PCG around the exact matrix-free Schur, using one V-cycle as M^-1.
  std::vector<double> truth((std::size_t)n),r,z,p,ap; for(PetscInt i=0;i<n;++i){const double g=(double)(R.amg.fine.localStart+i+1);truth[(std::size_t)i]=std::sin(0.0021*g)+0.41*std::cos(0.0049*g);} if(R.amg.fine.constantNullspace)PetscCall(customPressureProjectConstant(truth));
  PetscCall(customPressureSchurApply(B,M,truth,rhs)); if(R.amg.fine.constantNullspace)PetscCall(customPressureProjectConstant(rhs));
  sol.assign((std::size_t)n,0.0);r=rhs;PetscCall(customPressureAMGVcycle(R,r,z));p=z;double rho=0.0;PetscCall(customPressureDot(r,z,&rho));double bnorm=0.0;PetscCall(customPressureNorm2(rhs,&bnorm));
  PetscInt its=0;double rel=1.0;PetscBool conv=PETSC_FALSE;const PetscInt maxIts=300;const double target=1e-8;
  for(PetscInt k=0;k<maxIts;++k){PetscCall(customPressureSchurApply(B,M,p,ap));double pap=0.0;PetscCall(customPressureDot(p,ap,&pap));if(!(pap>0.0)||!(rho>0.0)||!std::isfinite(pap)||!std::isfinite(rho))break;const double a=rho/pap;for(std::size_t i=0;i<sol.size();++i){sol[i]+=a*p[i];r[i]-=a*ap[i];}if(R.amg.fine.constantNullspace)PetscCall(customPressureProjectConstant(r));double rn=0.0;PetscCall(customPressureNorm2(r,&rn));its=k+1;rel=rn/std::max(bnorm,1e-300);if(rel<=target){conv=PETSC_TRUE;break;}PetscCall(customPressureAMGVcycle(R,r,z));double rnz=0.0;PetscCall(customPressureDot(r,z,&rnz));if(!(rnz>0.0)||!std::isfinite(rnz))break;const double beta=rnz/rho;for(std::size_t i=0;i<p.size();++i)p[i]=z[i]+beta*p[i];rho=rnz;}
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_CUSTOM_AMG_GATE4_PCG status=%s exactOperator=matrix_free_B_rAU_Bt preconditioner=ONE_custom_unsmoothed_Vcycle its=%" PetscInt_FMT " trueRel=%.3e target=1.000e-08 maxIts=%" PetscInt_FMT "\n",
    conv?"PASS":"FAIL",its,rel,maxIts));
  if(!conv)SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_NOT_CONVERGED,"custom AMG Gate-4 native PCG did not converge");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customPressureAMGGate4Probe(const Mesh& mesh,const Discrete& D,int rank,
  CustomPressureBPlan& B,const CustomMomentumCSR& M,PetscInt globalRows,PetscBool constantNullspace,
  const CustomPressureAMGConfig& cfg) {
  PetscFunctionBeginUser;
  CustomPressureAMGRuntime R; PetscCall(customPressureAMGBuildHierarchy(mesh,D,rank,B,M,globalRows,constantNullspace,cfg,R));
  PetscCall(customPressureAMGGate4Validate(R,B,M));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_CUSTOM_AMG_GATE4 status=PASS roadmapGates=6 unsmoothedHierarchy=READY recursiveCoarse=READY smoother=READY coarseSolve=READY nativePCG=PASS liveSimpleBackend=DEFERRED_TO_GATE5 petscFallback=UNCHANGED generalFlowContract=distributed_B_rAU_Bt_plus_pressure_graph_plus_gauge\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}
