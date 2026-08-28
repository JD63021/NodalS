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

