struct Discrete {
  Mat A=nullptr, B[3]={nullptr,nullptr,nullptr};
  Vec rhs[3]={nullptr,nullptr,nullptr}, volumes=nullptr, fixedDiv=nullptr;
  // M6A: physical pressure scalar data are assembled directly in native FP64.
  // PETSc pressure Vecs remain layout/preconditioner bridges only.
  std::vector<double> volumesOwnedFP64, fixedDivOwnedFP64;
  // M6B: all physical momentum source terms are native FP64 owned-row arrays.
  std::array<std::vector<double>,3> rhsOwnedFP64;
  std::vector<PetscInt> g2free, pGid;
  std::vector<int> cellOwner;
  std::vector<PetscInt> velCount, cellCount;
  std::vector<char> fixedEntity;
  // Stage-3 pipe wall bookkeeping.  wallEntity marks P1 vertices and BF3 face
  // entities lying on the cylindrical wall.  With the weak wall model these
  // scalar entities remain algebraic unknowns so Uz can slip; Ux/Uy are
  // clamped strongly in the custom scalar solves and omitted from Bx/By.
  std::vector<char> wallEntity;
  // A2 memory hygiene: fixed Dirichlet values are stored only for fixed entities.
  // g2free uses -1 for an as-yet-unassigned free entity during boundary setup and
  // -(slot+2) for a fixed entity; after ownership, free entities receive gid>=0.
  // This removes three dense FP64 arrays over all velocity entities without adding
  // another global entity->slot map.
  std::vector<std::array<double,3>> fixedDirValue;
  PetscInt ns=0, freeVertices=0, freeFaces=0, fixedVertices=0, fixedFaces=0;
};

static PetscInt distributedGlobalCellCount(const Discrete& D) {
  return std::accumulate(D.cellCount.begin(),D.cellCount.end(),(PetscInt)0);
}
static PetscInt distributedGlobalVelCount(const Discrete& D) { return D.ns; }

static inline PetscInt fixedDirSlotFromEncodedGid(PetscInt g) { return -g-2; }

static double entityDirValue(const Discrete& D,int d,PetscInt entity) {
  const PetscInt g=D.g2free[(std::size_t)entity];
  if(g>=0) throw std::runtime_error("entityDirValue called on free entity");
  const PetscInt slot=fixedDirSlotFromEncodedGid(g);
  if(slot<0 || slot>=(PetscInt)D.fixedDirValue.size()) throw std::runtime_error("invalid compact Dirichlet slot");
  return D.fixedDirValue[(std::size_t)slot][(std::size_t)d];
}

static void setEntityDirValue(Discrete& D,int d,PetscInt entity,double value) {
  const PetscInt g=D.g2free[(std::size_t)entity];
  const PetscInt slot=fixedDirSlotFromEncodedGid(g);
  if(slot<0 || slot>=(PetscInt)D.fixedDirValue.size()) throw std::runtime_error("invalid compact Dirichlet slot in setter");
  D.fixedDirValue[(std::size_t)slot][(std::size_t)d]=value;
}

static void finalizeCompactDirichletSlots(Discrete& D) {
  D.g2free.assign(D.fixedEntity.size(),-1);
  PetscInt nfixed=0;
  for(char f:D.fixedEntity) if(f) ++nfixed;
  D.fixedDirValue.assign((std::size_t)nfixed,std::array<double,3>{0.0,0.0,0.0});
  PetscInt slot=0;
  for(std::size_t e=0;e<D.fixedEntity.size();++e) if(D.fixedEntity[e]) D.g2free[e]=-(slot++ + 2);
}

static void prepareBoundaryData(const Mesh& M,const ProblemConfig& P,Discrete& D) {
  const PetscInt nv=(PetscInt)M.points.size(), nf=(PetscInt)M.faces.size(), ni=(PetscInt)M.neighbour.size();
  D.fixedEntity.assign(nv+nf,0);
  if(P.weakWallFunction) D.wallEntity.assign((std::size_t)(nv+nf),0);
  else std::vector<char>().swap(D.wallEntity);

  if(P.mode==ProblemMode::MMS) {
    for(PetscInt f=ni;f<nf;++f) {
      D.fixedEntity[nv+f]=1;
      for(int v:M.faces[f].v) D.fixedEntity[v]=1;
    }
    finalizeCompactDirichletSlots(D);
    return;
  }

  const auto& B=P.boundary;
  std::vector<char> onWall(nv,0),onInlet(nv,0);
  for(PetscInt f=ni;f<nf;++f) {
    const int pi=M.facePatch[f];
    const bool wall=isWallPatch(B,pi);
    if(wall) {
      if(P.weakWallFunction) D.wallEntity[(std::size_t)(nv+f)]=1;
      for(int v:M.faces[f].v) {
        onWall[(std::size_t)v]=1;
        if(P.weakWallFunction) D.wallEntity[(std::size_t)v]=1;
      }
    }
    // One scalar topology is shared by all three components.  For Stage 3 the
    // wall entities are released globally; Ux/Uy are subsequently clamped to
    // zero algebraically, while Uz remains active for the weak wall law.
    if((wall && !P.weakWallFunction) || (pi==B.inlet && !dgNumericalTraceInlet(P))) D.fixedEntity[(std::size_t)(nv+f)]=1;
    if(pi==B.inlet) for(int v:M.faces[f].v) onInlet[(std::size_t)v]=1;
  }
  for(PetscInt v=0;v<nv;++v)
    if((onWall[(std::size_t)v] && !P.weakWallFunction) || (onInlet[(std::size_t)v] && !dgNumericalTraceInlet(P)))
      D.fixedEntity[(std::size_t)v]=1;

  // Allocate compact slots only after the complete fixed-entity mask is known.
  finalizeCompactDirichletSlots(D);

  // dg_numerical_trace prescribes velocity only through the weak boundary forms.
  if(dgNumericalTraceInlet(P)) return;

  for(PetscInt v=0;v<nv;++v) {
    // Preserve the existing wall-priority convention at inlet/wall edge vertices.
    // BF3 below restores the requested inlet face mean exactly despite that corner incompatibility.
    if(onInlet[v] && !onWall[v]) {
      if(P.inletBC==InletBCMode::PipeParabolic || P.inletBC==InletBCMode::PipeOneSeventh) setEntityDirValue(D,2,v,pipeBoundaryUz(P,M.points[v].x,M.points[v].y));
      else { setEntityDirValue(D,0,v,B.inletVelocity.x); setEntityDirValue(D,1,v,B.inletVelocity.y); setEntityDirValue(D,2,v,B.inletVelocity.z); }
    }
  }

  // Integral_F b_F / |F| = 9/20 for b_F=27 lambda1 lambda2 lambda3.
  // Choose each boundary BF3 coefficient so the complete P1+BF3 trace has the
  // requested vector face mean exactly.  This is especially important at the
  // inlet/wall edge where conforming P1 vertices retain no-slip wall priority.
  const auto& pin=M.patches[B.inlet];
  for(PetscInt f=pin.startFace;f<pin.startFace+pin.nFaces;++f) {
    const auto& F=M.faces[f];
    double exactMean[3]={0,0,0};
    if(P.inletBC==InletBCMode::PipeParabolic || P.inletBC==InletBCMode::PipeOneSeventh) {
      Vec3 X[3]={M.points[F.v[0]],M.points[F.v[1]],M.points[F.v[2]]};
      exactMean[2]=P.pipe.profileScale*triangleAverageIdealPipeProfileUz(P.pipe,P.inletBC,X);
    } else { exactMean[0]=B.inletVelocity.x; exactMean[1]=B.inletVelocity.y; exactMean[2]=B.inletVelocity.z; }
    for(int d=0;d<3;++d) {
      const double vertexMean=(entityDirValue(D,d,F.v[0])+entityDirValue(D,d,F.v[1])+entityDirValue(D,d,F.v[2]))/3.0;
      setEntityDirValue(D,d,nv+f,(20.0/9.0)*(exactMean[d]-vertexMean));
    }
  }
}


struct CentralTensor {
  double t[8][8][8][3] = {};
};

static const CentralTensor& centralTensor() {
  static const CentralTensor T = []() {
    CentralTensor out;
    // phi_a * phi_m * grad(phi_b) has total polynomial degree <= 8 for
    // P1+BF3 on an affine tetrahedron.  The 5x5x5 collapsed rule is exact
    // through that degree, so the older 7^3 construction was unnecessary.
    // This tensor is still built only once per process.
    const auto Q=tetDuffy5();
    for(const auto& q:Q) {
      double val[8],gr[8][3];
      basis(q.lam,val,gr);
      for(int a=0;a<8;++a)
        for(int m=0;m<8;++m)
          for(int b=0;b<8;++b)
            for(int j=0;j<3;++j)
              out.t[a][m][b][j] += val[a]*val[m]*gr[b][j]*q.w;
    }
    return out;
  }();
  return T;
}

struct DiffusionTensor {
  double t[8][8][3][3] = {};
};

static const DiffusionTensor& diffusionTensor() {
  static const DiffusionTensor T = []() {
    DiffusionTensor out;
    // Reference contraction for int grad(phi_a).grad(phi_b) dV.  Build once
    // with the same exact collapsed 5^3 rule used by the accepted static K.
    const auto Q=tetDuffy5();
    for(const auto& q:Q) {
      double val[8],gr[8][3]; basis(q.lam,val,gr);
      for(int a=0;a<8;++a) for(int b=0;b<8;++b)
        for(int j=0;j<3;++j) for(int k=0;k<3;++k)
          out.t[a][b][j][k] += gr[a][j]*gr[b][k]*q.w;
    }
    return out;
  }();
  return T;
}

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
