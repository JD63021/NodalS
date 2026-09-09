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
  std::uint8_t wallBasis[8]={0,0,0,0,0,0,0,0};
  std::int8_t inletOpp=-1;
  double inletSf[3]={0.0,0.0,0.0};
  double vol=0.0,gradLambda[4][3]={{0}};
};
struct CustomPressureTransposeCell {
  PetscInt pLocal=-1;
  PetscInt nOwnedVel=0;
  std::uint8_t basis[8]={0};
  std::uint8_t wallOwned[8]={0};
  PetscInt velOwnedLocal[8]={0};
  std::int8_t inletOpp=-1;
  double inletSf[3]={0.0,0.0,0.0};
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

// Low-level raw DG trace correction. Boundary-condition interpretation stops
// here; pressure solvers and AMG consume only the effective-B accessors below.
static inline double customPressureBCoeffDG(double vol,const double gradLambda[4][3],
                                            int inletOpp,const double inletSf[3],int d,int a) {
  double v=customPressureBCoeff(vol,gradLambda,d,a);
  if(inletOpp>=0) {
    if(a<4 && a!=inletOpp) v-=inletSf[d]/3.0;
    else if(a==4+inletOpp) v-=(9.0/20.0)*inletSf[d];
  }
  return v;
}

// Canonical effective pressure coupling. The compact B plan stores only
// geometry plus boundary metadata; no 3x8 FP64 coefficient table is retained.
// All live consumers therefore share:
//   B_eff = B_volume - DG_inlet_trace, weak-wall scalar entities z-only.
static inline double customPressureEffectiveB(const CustomPressureForwardCell& cp,int d,int a) {
  if(d<2 && cp.wallBasis[a]) return 0.0;
  return customPressureBCoeffDG(cp.vol,cp.gradLambda,(int)cp.inletOpp,cp.inletSf,d,a);
}

static inline double customPressureEffectiveBt(const CustomPressureTransposeCell& cp,int d,PetscInt j) {
  if(d<2 && cp.wallOwned[j]) return 0.0;
  const int a=(int)cp.basis[j];
  return customPressureBCoeffDG(cp.vol,cp.gradLambda,(int)cp.inletOpp,cp.inletSf,d,a);
}

// The explicit full-Schur oracle uses the same coefficient factory. AMG does
// not call this mesh/BC helper; it sees only the canonical compact B plan.
static inline double customPressureEffectiveBForCell(const Mesh& M,const Discrete& D,
                                                      const ProblemConfig& P,PetscInt c,
                                                      double vol,const double gradLambda[4][3],
                                                      int d,int a) {
  int opp=-1; Vec3 sf{}; double sd[3]={0.0,0.0,0.0};
  if(dgNumericalTraceFaceGeom(M,P,c,opp,sf)) { sd[0]=sf.x; sd[1]=sf.y; sd[2]=sf.z; }
  const PetscInt nv=(PetscInt)M.points.size();
  const PetscInt entity=(a<4)?M.tets[(std::size_t)c][(std::size_t)a]
                              :nv+M.oppFace[(std::size_t)c][(std::size_t)(a-4)];
  const bool weakWallScalar=(!D.wallEntity.empty() && D.wallEntity[(std::size_t)entity]);
  if(weakWallScalar && d<2) return 0.0;
  return customPressureBCoeffDG(vol,gradLambda,opp,sd,d,a);
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

static PetscErrorCode buildCustomPressureBPlan(const Mesh& M,const Discrete& D,const ProblemConfig& problem,int rank,CustomPressureBPlan& P) {
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
    fillCustomPressureGeom(M,c,cp.vol,cp.gradLambda);
    { int io=-1; Vec3 sf{}; if(dgNumericalTraceFaceGeom(M,problem,c,io,sf)) { cp.inletOpp=(std::int8_t)io; cp.inletSf[0]=sf.x; cp.inletSf[1]=sf.y; cp.inletSf[2]=sf.z; } }
    P.forwardCells.push_back(cp);
  }
  PetscCall(buildCustomPeerHalo(D.velCount,rank,std::move(neededVel),P.velocityHalo,"velocity_for_B"));
  // Fill velocity local indices only after the halo is finalized.
  for(PetscInt c=0;c<nc;++c) if(D.cellOwner[(std::size_t)c]==rank) {
    auto& cp=P.forwardCells[(std::size_t)(D.pGid[(std::size_t)c]-pStart)];
    PetscInt ent[8]; for(int i=0;i<4;++i)ent[i]=M.tets[(std::size_t)c][i]; for(int i=0;i<4;++i)ent[4+i]=nv+M.oppFace[(std::size_t)c][i];
    for(int a=0;a<8;++a) {
      const PetscInt g=D.g2free[(std::size_t)ent[a]];
      cp.velLocal[a]=(g>=0)?customPeerLocalIndex(P.velocityHalo,g):-1;
      cp.wallBasis[a]=(!D.wallEntity.empty() && D.wallEntity[(std::size_t)ent[a]])?1:0;
      if(g>=0 && cp.velLocal[a]<0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M4A B forward halo missing velocity gid");
    }
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
    for(int a=0;a<8;++a) {
      gid[a]=D.g2free[(std::size_t)ent[a]];
      if(gid[a]>=vStart && gid[a]<vEnd) {
        cp.basis[cp.nOwnedVel]=(std::uint8_t)a;
        cp.wallOwned[cp.nOwnedVel]=(!D.wallEntity.empty() && D.wallEntity[(std::size_t)ent[a]])?1:0;
        cp.velOwnedLocal[cp.nOwnedVel]=gid[a]-vStart;
        ++cp.nOwnedVel;
      }
    }
    if(cp.nOwnedVel==0) continue;
    const PetscInt pg=D.pGid[(std::size_t)c]; if(pg<pStart || pg>=pEnd) neededP.push_back(pg);
    fillCustomPressureGeom(M,c,cp.vol,cp.gradLambda);
    { int io=-1; Vec3 sf{}; if(dgNumericalTraceFaceGeom(M,problem,c,io,sf)) { cp.inletOpp=(std::int8_t)io; cp.inletSf[0]=sf.x; cp.inletSf[1]=sf.y; cp.inletSf[2]=sf.z; } }
    P.transposeCells.push_back(cp);
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
  for(const auto& cp:P.forwardCells) {
    double s=0.0;
    for(int a=0;a<8;++a) if(cp.velLocal[a]>=0) {
      // Straight-pipe Stage 3: wall scalar entities are active only for the
      // axial z component.  Excluding them from Bx/By is the pressure-side
      // counterpart of the exact Ux/Uy wall clamp.
      s+=customPressureEffectiveB(cp,d,a)*customPeerValue(P.velocityHalo,xOwned,cp.velLocal[a]);
    }
    yOwned[(std::size_t)cp.pLocal]=s;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customPressureBtApply(CustomPressureBPlan& P,int d,const std::vector<double>& pOwned,std::vector<double>& yOwned) {
  PetscFunctionBeginUser; PetscCall(customPeerExchange(P.pressureHalo,pOwned,48211+d));
  yOwned.assign((std::size_t)P.velocityHalo.nOwned,0.0);
  for(const auto& cp:P.transposeCells) {
    const double pv=customPeerValue(P.pressureHalo,pOwned,cp.pLocal);
    for(PetscInt j=0;j<cp.nOwnedVel;++j) {
      yOwned[(std::size_t)cp.velOwnedLocal[j]]+=customPressureEffectiveBt(cp,d,j)*pv;
    }
  }
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
  for(int d=0;d<3;++d) { std::fill(A.convRhs[d].begin(),A.convRhs[d].end(),0.0); std::fill(A.supgRhs[d].begin(),A.supgRhs[d].end(),0.0); std::fill(A.mixlenRhs[d].begin(),A.mixlenRhs[d].end(),0.0); }
  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode buildCellBlockJacobiPlan(const Mesh& M,const Discrete& D,int rank,
  const CustomMomentumCSR& A,CellBlockJacobiPlan& P) {
  PetscFunctionBeginUser;
  (void)rank;
  if(P.built) PetscFunctionReturn(PETSC_SUCCESS);
  const PetscInt nv=(PetscInt)M.points.size();
  const int nr=(int)D.velCount.size();
  std::vector<PetscInt> off(D.velCount.size()+1,0);
  for(std::size_t r=0;r<D.velCount.size();++r) off[r+1]=off[r]+D.velCount[r];

  std::unordered_map<PetscInt,PetscInt> ghostIndex;
  ghostIndex.reserve(A.ghostGid.size()*2+1);
  for(PetscInt i=0;i<(PetscInt)A.ghostGid.size();++i) ghostIndex.emplace(A.ghostGid[(std::size_t)i],i);
  std::vector<std::vector<PetscInt>> rows(A.ghostGid.size());
  std::vector<std::vector<PetscInt>> ownedRows((std::size_t)A.nOwned);
  for(PetscInt c=0;c<(PetscInt)M.tets.size();++c) {
    PetscInt ent[8],gid[8];
    for(int i=0;i<4;++i) ent[i]=M.tets[(std::size_t)c][i];
    for(int i=0;i<4;++i) ent[4+i]=nv+M.oppFace[(std::size_t)c][i];
    for(int i=0;i<8;++i) gid[i]=D.g2free[(std::size_t)ent[i]];
    for(int a=0;a<8;++a) if(gid[a]>=0) {
      if(gid[a]>=A.rstart && gid[a]<A.rend) {
        auto& rr=ownedRows[(std::size_t)(gid[a]-A.rstart)];
        for(int b=0;b<8;++b) if(gid[b]>=0) rr.push_back(gid[b]);
      }
      auto it=ghostIndex.find(gid[a]);
      if(it==ghostIndex.end()) continue;
      auto& rr=rows[(std::size_t)it->second];
      for(int b=0;b<8;++b) if(gid[b]>=0) rr.push_back(gid[b]);
    }
  }

  // FIX2: materialize the sparse OWNED-row principal lookup before M1 releases
  // A.colGid.  Runtime then uses only these retained columns plus permanent
  // indices into A.aRel, preserving M1 memory hygiene without a dangling
  // dependency on the released full CSR column array.
  P.ownedRowPtr.assign((std::size_t)A.nOwned+1,0);
  for(PetscInt i=0;i<A.nOwned;++i) {
    auto& rr=ownedRows[(std::size_t)i];
    std::sort(rr.begin(),rr.end()); rr.erase(std::unique(rr.begin(),rr.end()),rr.end());
    const PetscInt row=A.rstart+i;
    if(!std::binary_search(rr.begin(),rr.end(),row))
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"cell-block Jacobi owned row topology is missing diagonal");
    P.ownedRowPtr[(std::size_t)i+1]=P.ownedRowPtr[(std::size_t)i]+(PetscInt)rr.size();
  }
  P.ownedRowColGid.resize((std::size_t)P.ownedRowPtr.back());
  P.ownedRowValuePos.resize((std::size_t)P.ownedRowPtr.back(),-1);
  for(PetscInt i=0;i<A.nOwned;++i) {
    PetscInt q=P.ownedRowPtr[(std::size_t)i];
    const PetscInt b=A.rowPtr[(std::size_t)i],e=A.rowPtr[(std::size_t)i+1];
    for(PetscInt col:ownedRows[(std::size_t)i]) {
      auto it=std::lower_bound(A.colGid.begin()+b,A.colGid.begin()+e,col);
      if(it==A.colGid.begin()+e || *it!=col)
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"cell-block Jacobi owned principal entry is absent from assembled CSR");
      P.ownedRowColGid[(std::size_t)q]=col;
      P.ownedRowValuePos[(std::size_t)q]=(PetscInt)(it-A.colGid.begin());
      ++q;
    }
  }
  std::vector<std::vector<PetscInt>>().swap(ownedRows);

  P.ghostRowPtr.assign(A.ghostGid.size()+1,0);
  for(PetscInt i=0;i<(PetscInt)A.ghostGid.size();++i) {
    auto& rr=rows[(std::size_t)i];
    std::sort(rr.begin(),rr.end()); rr.erase(std::unique(rr.begin(),rr.end()),rr.end());
    if(!std::binary_search(rr.begin(),rr.end(),A.ghostGid[(std::size_t)i]))
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"cell-block Jacobi ghost row topology is missing diagonal");
    P.ghostRowPtr[(std::size_t)i+1]=P.ghostRowPtr[(std::size_t)i]+(PetscInt)rr.size();
  }
  P.ghostRowColGid.resize((std::size_t)P.ghostRowPtr.back());
  for(PetscInt i=0;i<(PetscInt)A.ghostGid.size();++i) {
    PetscInt q=P.ghostRowPtr[(std::size_t)i];
    for(PetscInt g:rows[(std::size_t)i]) P.ghostRowColGid[(std::size_t)q++]=g;
  }
  P.ghostRowValues.assign(P.ghostRowColGid.size(),0.0);

  // -------------------------------------------------------------------------
  // Build an exact sparse value-request halo.
  //
  // FIX1: the original port counted only the principal-block entries needed on
  // the receiver, but packed each sender's COMPLETE CSR row.  Those counts are
  // different whenever an assembled row has couplings outside the requesting
  // tetrahedron, hence the setup-time "ghost row value-count mismatch".
  //
  // Instead exchange requested (rowGid,colGid) PAIRS once.  The row owner maps
  // each pair to an exact owned CSR slot.  Runtime refresh then communicates
  // one double per requested pair, so sender/receiver counts are identical by
  // construction and no unnecessary row entries cross MPI.
  // -------------------------------------------------------------------------
  P.recvCounts.assign((std::size_t)nr,0); // values this rank requests FROM each owner
  P.sendCounts.assign((std::size_t)nr,0); // values this rank must send TO each peer
  for(PetscInt gi=0;gi<(PetscInt)A.ghostGid.size();++gi) {
    const PetscInt row=A.ghostGid[(std::size_t)gi];
    const int owner=customMomentumOwnerOfGid(off,row);
    if(owner<0 || owner>=nr) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"cell-block Jacobi ghost row has invalid owner");
    const PetscInt len=P.ghostRowPtr[(std::size_t)gi+1]-P.ghostRowPtr[(std::size_t)gi];
    if(len>INT_MAX-P.recvCounts[(std::size_t)owner]) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"cell-block Jacobi recv count overflow");
    P.recvCounts[(std::size_t)owner]+=(int)len;
  }
  PetscCallMPI(MPI_Alltoall(P.recvCounts.data(),1,MPI_INT,P.sendCounts.data(),1,MPI_INT,PETSC_COMM_WORLD));

  P.sendDispls.assign((std::size_t)nr,0); P.recvDispls.assign((std::size_t)nr,0);
  for(int r=1;r<nr;++r) {
    P.sendDispls[(std::size_t)r]=P.sendDispls[(std::size_t)r-1]+P.sendCounts[(std::size_t)r-1];
    P.recvDispls[(std::size_t)r]=P.recvDispls[(std::size_t)r-1]+P.recvCounts[(std::size_t)r-1];
  }
  int nsend=0,nrecv=0; for(int v:P.sendCounts) nsend+=v; for(int v:P.recvCounts) nrecv+=v;
  if((std::size_t)nrecv!=P.ghostRowValues.size())
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"cell-block Jacobi sparse request count does not match ghost-row storage");

  // Requests leaving this rank are grouped by owner.  recvDest preserves where
  // the returned packed value belongs in ghostRowValues.
  std::vector<PetscInt> requestPairsOut((std::size_t)2*nrecv,-1);
  P.recvDest.assign((std::size_t)nrecv,-1);
  std::vector<int> cursor=P.recvDispls;
  for(PetscInt gi=0;gi<(PetscInt)A.ghostGid.size();++gi) {
    const PetscInt row=A.ghostGid[(std::size_t)gi];
    const int owner=customMomentumOwnerOfGid(off,row);
    for(PetscInt k=P.ghostRowPtr[(std::size_t)gi];k<P.ghostRowPtr[(std::size_t)gi+1];++k) {
      const int slot=cursor[(std::size_t)owner]++;
      requestPairsOut[(std::size_t)2*slot]=row;
      requestPairsOut[(std::size_t)2*slot+1]=P.ghostRowColGid[(std::size_t)k];
      P.recvDest[(std::size_t)slot]=k;
    }
  }
  for(int r=0;r<nr;++r) if(cursor[(std::size_t)r]!=P.recvDispls[(std::size_t)r]+P.recvCounts[(std::size_t)r])
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"cell-block Jacobi sparse request packing mismatch");

  std::vector<int> pairSendCounts((std::size_t)nr,0),pairSendDispls((std::size_t)nr,0),pairRecvCounts((std::size_t)nr,0),pairRecvDispls((std::size_t)nr,0);
  for(int r=0;r<nr;++r) {
    if(P.recvCounts[(std::size_t)r]>INT_MAX/2 || P.sendCounts[(std::size_t)r]>INT_MAX/2)
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"cell-block Jacobi request-pair count overflow");
    pairSendCounts[(std::size_t)r]=2*P.recvCounts[(std::size_t)r];
    pairRecvCounts[(std::size_t)r]=2*P.sendCounts[(std::size_t)r];
    pairSendDispls[(std::size_t)r]=2*P.recvDispls[(std::size_t)r];
    pairRecvDispls[(std::size_t)r]=2*P.sendDispls[(std::size_t)r];
  }
  std::vector<PetscInt> requestPairsIn((std::size_t)2*nsend,-1);
  PetscCallMPI(MPI_Alltoallv(requestPairsOut.empty()?nullptr:requestPairsOut.data(),pairSendCounts.data(),pairSendDispls.data(),MPIU_INT,
                             requestPairsIn.empty()?nullptr:requestPairsIn.data(),pairRecvCounts.data(),pairRecvDispls.data(),MPIU_INT,PETSC_COMM_WORLD));

  // Map every request received from a peer to one exact position in the owned
  // assembled relaxed CSR.  This mapping is topology-only and reused forever.
  P.sendValuePos.assign((std::size_t)nsend,-1);
  for(int q=0;q<nsend;++q) {
    const PetscInt row=requestPairsIn[(std::size_t)2*q],col=requestPairsIn[(std::size_t)2*q+1];
    if(row<A.rstart || row>=A.rend) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"cell-block Jacobi peer requested a non-owned row");
    const PetscInt i=row-A.rstart,b=A.rowPtr[(std::size_t)i],e=A.rowPtr[(std::size_t)i+1];
    auto it=std::lower_bound(A.colGid.begin()+b,A.colGid.begin()+e,col);
    if(it==A.colGid.begin()+e || *it!=col)
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"cell-block Jacobi requested principal entry is absent from owned CSR");
    P.sendValuePos[(std::size_t)q]=(PetscInt)(it-A.colGid.begin());
  }

  P.sendValues.assign((std::size_t)nsend,0.0);
  P.recvPackedValues.assign((std::size_t)nrecv,0.0);
  P.exchangedValuesPerRefresh=(unsigned long long)nsend+(unsigned long long)nrecv;

  P.ownedMultiplicity.assign((std::size_t)A.nOwned,0);
  for(PetscInt c=0;c<(PetscInt)M.tets.size();++c) {
    PetscInt ent[8],gid[8]; bool hit=false;
    for(int i=0;i<4;++i) ent[i]=M.tets[(std::size_t)c][i];
    for(int i=0;i<4;++i) ent[4+i]=nv+M.oppFace[(std::size_t)c][i];
    for(int i=0;i<8;++i) { gid[i]=D.g2free[(std::size_t)ent[i]]; if(gid[i]>=A.rstart && gid[i]<A.rend) hit=true; }
    if(!hit) continue;
    P.supportCells.push_back(c);
    for(int a=0;a<8;++a) if(gid[a]>=A.rstart && gid[a]<A.rend) ++P.ownedMultiplicity[(std::size_t)(gid[a]-A.rstart)];
  }
  for(PetscInt i=0;i<A.nOwned;++i) if(P.ownedMultiplicity[(std::size_t)i]<=0)
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"cell-block Jacobi owned row has zero incident-cell multiplicity");

  unsigned long long ls=(unsigned long long)P.supportCells.size(),gs=0,lgr=(unsigned long long)A.ghostGid.size(),ggr=0;
  unsigned long long lgn=(unsigned long long)P.ghostRowValues.size(),ggn=0,lex=P.exchangedValuesPerRefresh,gex=0;
  unsigned long long lop=(unsigned long long)P.ownedRowValuePos.size(),gop=0;
  PetscCallMPI(MPI_Allreduce(&ls,&gs,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&lgr,&ggr,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&lgn,&ggn,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&lex,&gex,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&lop,&gop,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  P.built=PETSC_TRUE;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_RAU_CELL_BLOCK_PLAN status=PASS mode=global_principal_tet_inverse_diag supportCellsSummed=%llu ownedPrincipalEntriesSummed=%llu ghostRowsSummed=%llu ghostPrincipalEntriesSummed=%llu exchangedValuesSummedPerRefresh=%llu exchangedMiBSummedPerRefresh=%.6f halo=sparse_requested_row_col_pairs ownedLookup=retained_sparse_col_and_aRel_position survives_M1_colGid_release=1 aggregation=mean_incident_cell_inverse_diagonal diagonalSanity=exact_for_global_diagonal_Arel\n",
    gs,gop,ggr,ggn,gex,(double)gex*sizeof(double)/(1024.0*1024.0)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode exchangeCellBlockGhostRows(const CustomMomentumCSR& A,CellBlockJacobiPlan& P) {
  PetscFunctionBeginUser;
  if(!P.built) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONGSTATE,"cell-block Jacobi plan is not built");
  if(P.sendValuePos.size()!=P.sendValues.size()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"cell-block Jacobi sparse send map size mismatch");
  if(P.recvDest.size()!=P.recvPackedValues.size()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"cell-block Jacobi sparse receive map size mismatch");
  for(std::size_t q=0;q<P.sendValuePos.size();++q) {
    const PetscInt k=P.sendValuePos[q];
    if(k<0 || (std::size_t)k>=A.aRel.size()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"cell-block Jacobi sparse send CSR position out of range");
    P.sendValues[q]=A.aRel[(std::size_t)k];
  }
  PetscCallMPI(MPI_Alltoallv(P.sendValues.empty()?nullptr:P.sendValues.data(),P.sendCounts.data(),P.sendDispls.data(),MPI_DOUBLE,
                              P.recvPackedValues.empty()?nullptr:P.recvPackedValues.data(),P.recvCounts.data(),P.recvDispls.data(),MPI_DOUBLE,PETSC_COMM_WORLD));
  for(std::size_t q=0;q<P.recvDest.size();++q) {
    const PetscInt k=P.recvDest[q];
    if(k<0 || (std::size_t)k>=P.ghostRowValues.size()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"cell-block Jacobi sparse receive destination out of range");
    P.ghostRowValues[(std::size_t)k]=P.recvPackedValues[q];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static double cellBlockJacobiEntry(const CustomMomentumCSR& A,const CellBlockJacobiPlan& P,PetscInt rowGid,PetscInt colGid) {
  if(rowGid>=A.rstart && rowGid<A.rend) {
    const PetscInt i=rowGid-A.rstart,b=P.ownedRowPtr[(std::size_t)i],e=P.ownedRowPtr[(std::size_t)i+1];
    auto it=std::lower_bound(P.ownedRowColGid.begin()+b,P.ownedRowColGid.begin()+e,colGid);
    if(it==P.ownedRowColGid.begin()+e || *it!=colGid) throw std::runtime_error("cell-block Jacobi owned principal entry missing from retained sparse topology");
    const PetscInt q=(PetscInt)(it-P.ownedRowColGid.begin());
    const PetscInt pos=P.ownedRowValuePos[(std::size_t)q];
    if(pos<0 || (std::size_t)pos>=A.aRel.size()) throw std::runtime_error("cell-block Jacobi owned principal CSR position out of range");
    return A.aRel[(std::size_t)pos];
  }
  auto git=std::lower_bound(A.ghostGid.begin(),A.ghostGid.end(),rowGid);
  if(git==A.ghostGid.end() || *git!=rowGid) throw std::runtime_error("cell-block Jacobi principal row is neither owned nor in ghost-row halo");
  const PetscInt gi=(PetscInt)(git-A.ghostGid.begin()),b=P.ghostRowPtr[(std::size_t)gi],e=P.ghostRowPtr[(std::size_t)gi+1];
  auto it=std::lower_bound(P.ghostRowColGid.begin()+b,P.ghostRowColGid.begin()+e,colGid);
  if(it==P.ghostRowColGid.begin()+e || *it!=colGid) throw std::runtime_error("cell-block Jacobi ghost principal entry missing from reconstructed CSR");
  return P.ghostRowValues[(std::size_t)(it-P.ghostRowColGid.begin())];
}

static bool invertDenseCellBlock(int n,const double *a,double *inv,double pivotTol,double& minPivotRel) {
  double m[64]={0.0};
  double scale=0.0;
  for(int i=0;i<n*n;++i) { m[i]=a[i]; scale=std::max(scale,std::abs(a[i])); inv[i]=0.0; }
  for(int i=0;i<n;++i) inv[i*n+i]=1.0;
  if(!(scale>0.0) || !std::isfinite(scale)) return false;
  minPivotRel=std::numeric_limits<double>::infinity();
  for(int k=0;k<n;++k) {
    int piv=k; double pv=std::abs(m[k*n+k]);
    for(int i=k+1;i<n;++i) if(std::abs(m[i*n+k])>pv) { pv=std::abs(m[i*n+k]); piv=i; }
    minPivotRel=std::min(minPivotRel,pv/scale);
    if(!std::isfinite(pv) || pv<=pivotTol*scale) return false;
    if(piv!=k) for(int j=0;j<n;++j) { std::swap(m[k*n+j],m[piv*n+j]); std::swap(inv[k*n+j],inv[piv*n+j]); }
    const double d=m[k*n+k];
    for(int j=0;j<n;++j) { m[k*n+j]/=d; inv[k*n+j]/=d; }
    for(int i=0;i<n;++i) if(i!=k) {
      const double f=m[i*n+k]; if(f==0.0) continue;
      for(int j=0;j<n;++j) { m[i*n+j]-=f*m[k*n+j]; inv[i*n+j]-=f*inv[k*n+j]; }
    }
  }
  return true;
}

static PetscErrorCode updateRauCellBlockDiag(const Mesh& M,const Discrete& D,CustomMomentumCSR& A,
  CellBlockJacobiPlan& P,double rauScale,double blockBlend,double ratioMin,double ratioMax,double pivotTol,
  const std::string& blockFallback,PetscInt simpleIt) {
  PetscFunctionBeginUser;
  PetscCall(exchangeCellBlockGhostRows(A,P));
  const PetscInt nv=(PetscInt)M.points.size();
  std::vector<double> sum((std::size_t)A.nOwned,0.0);
  std::vector<PetscInt> cnt((std::size_t)A.nOwned,0);
  unsigned long long localBlocks=0,localBlockFallbacks=0,localDiagFallbacks=0;
  double localPivotMin=1.0,localBaseMin=std::numeric_limits<double>::infinity(),localBaseMax=0.0,localBaseSum=0.0;
  double localRawRatioMin=std::numeric_limits<double>::infinity(),localRawRatioMax=0.0,localRawRatioSum=0.0;
  double localUsedRatioMin=std::numeric_limits<double>::infinity(),localUsedRatioMax=0.0,localUsedRatioSum=0.0;
  for(PetscInt c:P.supportCells) {
    PetscInt ent[8],gid[8]; int n=0;
    for(int i=0;i<4;++i) ent[i]=M.tets[(std::size_t)c][i];
    for(int i=0;i<4;++i) ent[4+i]=nv+M.oppFace[(std::size_t)c][i];
    for(int i=0;i<8;++i) { const PetscInt g=D.g2free[(std::size_t)ent[i]]; if(g>=0) gid[n++]=g; }
    if(n<=0) continue;
    double b[64]={0.0},bi[64]={0.0};
    try {
      for(int a=0;a<n;++a) for(int q=0;q<n;++q) b[a*n+q]=cellBlockJacobiEntry(A,P,gid[a],gid[q]);
    } catch(const std::exception& e) { SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,e.what()); }
    double piv=1.0; const bool ok=invertDenseCellBlock(n,b,bi,pivotTol,piv); ++localBlocks; localPivotMin=std::min(localPivotMin,piv);
    if(!ok) {
      ++localBlockFallbacks;
      if(blockFallback=="error") SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"cell-block Jacobi principal block failed pivot test with fallback=error");
    }
    for(int a=0;a<n;++a) if(gid[a]>=A.rstart && gid[a]<A.rend) {
      const PetscInt li=gid[a]-A.rstart;
      double v=ok?bi[a*n+a]:std::numeric_limits<double>::quiet_NaN();
      if(!(v>0.0) || !std::isfinite(v)) {
        if(blockFallback=="error") SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"cell-block Jacobi inverse diagonal is nonpositive/nonfinite with fallback=error");
        v=1.0/A.relaxedDiag[(std::size_t)li]; ++localDiagFallbacks;
      }
      sum[(std::size_t)li]+=v; ++cnt[(std::size_t)li];
    }
  }
  for(PetscInt i=0;i<A.nOwned;++i) {
    if(cnt[(std::size_t)i]!=P.ownedMultiplicity[(std::size_t)i] || cnt[(std::size_t)i]<=0)
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"cell-block Jacobi incident-cell accumulation mismatch");
    const double rawBase=sum[(std::size_t)i]/(double)cnt[(std::size_t)i];
    if(!(rawBase>0.0) || !std::isfinite(rawBase))
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"cell-block Jacobi produced invalid mean inverse diagonal");
    const double diagBase=1.0/A.relaxedDiag[(std::size_t)i];
    const double rawRatio=rawBase/diagBase;
    double clippedRatio=rawRatio;
    if(ratioMin>0.0) clippedRatio=std::max(clippedRatio,ratioMin);
    if(ratioMax>0.0) clippedRatio=std::min(clippedRatio,ratioMax);
    const double blockBase=clippedRatio*diagBase;
    const double base=(1.0-blockBlend)*diagBase+blockBlend*blockBase;
    if(!(base>0.0) || !std::isfinite(base))
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"cell-block Jacobi blend/clamp produced invalid inverse diagonal");
    const double usedRatio=base/diagBase;
    A.rAU[(std::size_t)i]=rauScale*base;
    A.metric[(std::size_t)i]=1.0/base;
    localBaseMin=std::min(localBaseMin,base); localBaseMax=std::max(localBaseMax,base); localBaseSum+=base;
    localRawRatioMin=std::min(localRawRatioMin,rawRatio); localRawRatioMax=std::max(localRawRatioMax,rawRatio); localRawRatioSum+=rawRatio;
    localUsedRatioMin=std::min(localUsedRatioMin,usedRatio); localUsedRatioMax=std::max(localUsedRatioMax,usedRatio); localUsedRatioSum+=usedRatio;
  }

  unsigned long long gBlocks=0,gBlockFallbacks=0,gDiagFallbacks=0;
  PetscCallMPI(MPI_Allreduce(&localBlocks,&gBlocks,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localBlockFallbacks,&gBlockFallbacks,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localDiagFallbacks,&gDiagFallbacks,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  double gPivotMin=0,gBaseMin=0,gBaseMax=0,gBaseSum=0;
  double gRawRatioMin=0,gRawRatioMax=0,gRawRatioSum=0,gUsedRatioMin=0,gUsedRatioMax=0,gUsedRatioSum=0;
  PetscCallMPI(MPI_Allreduce(&localPivotMin,&gPivotMin,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localBaseMin,&gBaseMin,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localBaseMax,&gBaseMax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localBaseSum,&gBaseSum,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localRawRatioMin,&gRawRatioMin,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localRawRatioMax,&gRawRatioMax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localRawRatioSum,&gRawRatioSum,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localUsedRatioMin,&gUsedRatioMin,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localUsedRatioMax,&gUsedRatioMax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localUsedRatioSum,&gUsedRatioSum,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  if(simpleIt<=10 || simpleIt%10==0) PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_RAU_CELL_BLOCK it=%" PetscInt_FMT " blocksSummed=%llu blockFallbacks=%llu diagFallbackContribs=%llu minPivotRel=%.3e pivotTol=%.3e baseInvDiag=[%.6e,%.6e,%.6e] rawRatioToDiagInv=[%.6e,%.6e,%.6e] usedRatioToDiagInv=[%.6e,%.6e,%.6e] blockBlend=%.6g ratioClamp=[%.6g,%.6g] fallback=%s rauScale=%.6g semantics=mean_diag_inverse_of_global_relaxed_tet_principal_block_then_optional_ratio_clamp_and_diag_blend\n",
    simpleIt,gBlocks,gBlockFallbacks,gDiagFallbacks,gPivotMin,pivotTol,gBaseMin,D.ns?gBaseSum/(double)D.ns:0.0,gBaseMax,
    gRawRatioMin,D.ns?gRawRatioSum/(double)D.ns:0.0,gRawRatioMax,
    gUsedRatioMin,D.ns?gUsedRatioSum/(double)D.ns:0.0,gUsedRatioMax,
    blockBlend,ratioMin,ratioMax,blockFallback.c_str(),rauScale));
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
    } else if(rauMode=="diag" || rauMode=="cell_block_diag") use=A.relaxedDiag[(std::size_t)i];
    else if(rauMode=="row_l1") { use=0.0; for(PetscInt k=A.rowPtr[(std::size_t)i];k<A.rowPtr[(std::size_t)i+1];++k) use+=std::abs(A.aRel[(std::size_t)k]); }
    else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"unknown custom rAU mode");
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


// NODALS_TET_MOMENTUM_BUDGET_20260908
// Exact algebraic axial-momentum budget on the final converged P1+BF3 state.
// The window test is P1-only: BF3 test coefficients are identically zero.
// This is intentional because the physical constant test is represented by
// the P1 partition of unity, not by summing the enrichment functions.
static double axialBudgetP1VertexWeight(const Vec3& x,const PipeGeometry& pipe,double z0D,double z1D) {
  const double zD=(x.z-pipe.zIn)/pipe.D;
  const double tol=1.0e-10*std::max(1.0,std::max(std::abs(z0D),std::abs(z1D)));
  if(zD<z0D-tol || zD>z1D+tol) return 0.0;
  // Half weight on an exactly aligned endpoint plane gives a symmetric
  // discrete box; the effective wall length is measured from this same test.
  if(std::abs(zD-z0D)<=tol || std::abs(zD-z1D)<=tol) return 0.5;
  return 1.0;
}

static PetscErrorCode buildAxialBudgetP1TestOwned(
    const Mesh& M,const Discrete& D,const PipeGeometry& pipe,
    const CustomMomentumCSR& A,double z0D,double z1D,std::vector<double>& wOwned) {
  PetscFunctionBeginUser;
  wOwned.assign((std::size_t)A.nOwned,0.0);
  for(PetscInt v=0;v<(PetscInt)M.points.size();++v) {
    if((std::size_t)v>=D.g2free.size()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"axial budget vertex/entity map out of range");
    const PetscInt gid=D.g2free[(std::size_t)v];
    if(gid>=A.rstart && gid<A.rend)
      wOwned[(std::size_t)(gid-A.rstart)]=axialBudgetP1VertexWeight(M.points[(std::size_t)v],pipe,z0D,z1D);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static double axialBudgetEffectiveWallLengthRoot(
    const Mesh& M,const ProblemConfig& cfg,double z0D,double z1D) {
  if(!(cfg.pipe.L>0.0)) return 0.0;
  double wallArea=0.0,chiArea=0.0;
  auto addPatch=[&](int pi) {
    if(pi<0 || pi>=(int)M.patches.size()) return;
    const auto& P=M.patches[(std::size_t)pi];
    for(PetscInt f=P.startFace;f<P.startFace+P.nFaces;++f) {
      const auto& F=M.faces[(std::size_t)f];
      const double area=triangleArea(M.points[(std::size_t)F.v[0]],M.points[(std::size_t)F.v[1]],M.points[(std::size_t)F.v[2]]);
      const double c=(axialBudgetP1VertexWeight(M.points[(std::size_t)F.v[0]],cfg.pipe,z0D,z1D)
                    +axialBudgetP1VertexWeight(M.points[(std::size_t)F.v[1]],cfg.pipe,z0D,z1D)
                    +axialBudgetP1VertexWeight(M.points[(std::size_t)F.v[2]],cfg.pipe,z0D,z1D))/3.0;
      wallArea+=area; chiArea+=area*c;
    }
  };
  if(cfg.pipe.wall>=0) addPatch(cfg.pipe.wall);
  else for(int pi:cfg.boundary.walls) addPatch(pi);
  if(!(wallArea>0.0)) return 0.0;
  const double perimeterEquivalent=wallArea/cfg.pipe.L;
  return chiArea/std::max(perimeterEquivalent,1.0e-300);
}

static PetscErrorCode printAxialMomentumBudget(
    const Mesh& M,const Mesh* rootGeometryMesh,const Discrete& D,const ProblemConfig& cfg,int rank,
    CustomMomentumCSR& A,const std::array<std::vector<double>,3>& U,
    const std::vector<double>& actionMolecular,
    const std::vector<double>& actionAfterMix,
    const std::vector<double>& actionAfterWall,
    const std::vector<double>& actionAfterConvection,
    const std::vector<double>& actionFinal,
    const std::vector<double>& pressureBtZ,
    PetscBool mixingLength,PetscBool weakWallFunction,PetscBool centralConvection,PetscBool useSupg) {
  PetscFunctionBeginUser;
  if((PetscInt)actionFinal.size()!=A.nOwned || (PetscInt)pressureBtZ.size()!=A.nOwned)
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"axial momentum budget vector size mismatch");
  const double windows[3][2]={{12.0,14.0},{14.0,16.0},{16.0,18.0}};
  std::vector<double> test;
  for(int wi=0;wi<3;++wi) {
    const double z0D=windows[wi][0],z1D=windows[wi][1];
    PetscCall(buildAxialBudgetP1TestOwned(M,D,cfg.pipe,A,z0D,z1D,test));
    double local[10]={0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
    for(PetscInt i=0;i<A.nOwned;++i) {
      const double w=test[(std::size_t)i];
      if(w==0.0) continue;
      const double molecular=actionMolecular[(std::size_t)i];
      const double mix=(actionAfterMix[(std::size_t)i]-actionMolecular[(std::size_t)i])
        -(mixingLength?A.mixlenRhs[2][(std::size_t)i]:0.0);
      const double wall=(actionAfterWall[(std::size_t)i]-actionAfterMix[(std::size_t)i])
        -(weakWallFunction?A.wallRhs[2][(std::size_t)i]:0.0);
      const double conv=(actionAfterConvection[(std::size_t)i]-actionAfterWall[(std::size_t)i])
        -(centralConvection?A.convRhs[2][(std::size_t)i]:0.0);
      const double supg=(actionFinal[(std::size_t)i]-actionAfterConvection[(std::size_t)i])
        -(useSupg?A.supgRhs[2][(std::size_t)i]:0.0);
      const double base=-D.rhsOwnedFP64[2][(std::size_t)i];
      const double pressure=-pressureBtZ[(std::size_t)i];
      const double sum=molecular+mix+wall+conv+supg+base+pressure;
      const double direct=actionFinal[(std::size_t)i]-D.rhsOwnedFP64[2][(std::size_t)i]
        -(mixingLength?A.mixlenRhs[2][(std::size_t)i]:0.0)
        -(weakWallFunction?A.wallRhs[2][(std::size_t)i]:0.0)
        -(centralConvection?A.convRhs[2][(std::size_t)i]:0.0)
        -(useSupg?A.supgRhs[2][(std::size_t)i]:0.0)
        -pressureBtZ[(std::size_t)i];
      local[0]+=w*molecular;
      local[1]+=w*mix;
      local[2]+=w*wall;
      local[3]+=w*conv;
      local[4]+=w*supg;
      local[5]+=w*base;
      local[6]+=w*pressure;
      local[7]+=w*sum;
      local[8]+=w*direct;
      local[9]+=w;
    }
    double global[10]={0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
    PetscCallMPI(MPI_Allreduce(local,global,10,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
    double Leff=0.0;
    if(rank==0) {
      if(!rootGeometryMesh) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"axial momentum budget missing root geometry mesh");
      Leff=axialBudgetEffectiveWallLengthRoot(*rootGeometryMesh,cfg,z0D,z1D);
    }
    PetscCallMPI(MPI_Bcast(&Leff,1,MPI_DOUBLE,0,PETSC_COMM_WORLD));
    const double Asec=cfg.pipe.inletArea;
    const double Ubulk=cfg.pipe.bulkVelocity;
    const double scale=(Leff>0.0 && Asec>0.0 && Ubulk>0.0)
      ? 2.0*cfg.pipe.D/(Leff*Asec*Ubulk*Ubulk) : 0.0;
    const double drive=-global[6];
    const double resistanceNoPressure=global[0]+global[1]+global[2]+global[3]+global[4]+global[5];
    const double closure=global[7]-global[8];
    const double denom=std::max(std::abs(drive),1.0e-300);
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_MOM_BUDGET window=%.0f-%.0fD test=P1_nodal_box_half_endpoints BF3test=zero nominalLengthD=%.6f effectiveLengthD=%.12f "
      "raw=[pressure=%.12e wall=%.12e molecular=%.12e eddy=%.12e convection=%.12e supg=%.12e baseSourceElim=%.12e resistanceNoPressure=%.12e residualSum=%.12e directResidual=%.12e closure=%.3e] "
      "fEq=[pressure=%.10f wall=%.10f molecular=%.10f eddy=%.10f convection=%.10f supg=%.10f baseSourceElim=%.10f residual=%.10f] "
      "shareOfDrive=[wall=%.6f molecular=%.6f eddy=%.6f convection=%.6f supg=%.6f baseSourceElim=%.6f resistance=%.6f] status=%s\n",
      z0D,z1D,z1D-z0D,Leff/cfg.pipe.D,
      global[6],global[2],global[0],global[1],global[3],global[4],global[5],resistanceNoPressure,global[7],global[8],closure,
      scale*global[6],scale*global[2],scale*global[0],scale*global[1],scale*global[3],scale*global[4],scale*global[5],scale*global[8],
      global[2]/denom,global[0]/denom,global[1]/denom,global[3]/denom,global[4]/denom,global[5]/denom,resistanceNoPressure/denom,
      std::abs(closure)<=1.0e-10*std::max(1.0,std::abs(global[8]))?"PASS":"CHECK"));
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
    } else if(rauMode=="diag" || rauMode=="cell_block_diag") use=A.relaxedDiag[(std::size_t)i];
    else if(rauMode=="row_l1") {
      use=0.0; for(PetscInt k=A.rowPtr[(std::size_t)i];k<A.rowPtr[(std::size_t)i+1];++k) use+=std::abs(A.aRel[(std::size_t)k]);
    } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"unknown custom shadow rAU mode");
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

static PetscErrorCode customMomentumSymmetricGS(CustomMomentumCSR& A,const std::vector<double>& b,std::vector<double>& x,double omega,PetscInt localSweeps,const std::vector<char>* clamp=nullptr) {
  PetscFunctionBeginUser;
  if((PetscInt)b.size()!=A.nOwned || (PetscInt)x.size()!=A.nOwned) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"custom SGS vector size mismatch");
  if(clamp && (PetscInt)clamp->size()!=A.nOwned) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"custom SGS clamp size mismatch");
  if(clamp) for(PetscInt i=0;i<A.nOwned;++i) if((*clamp)[(std::size_t)i]) x[(std::size_t)i]=0.0;
  auto& beff=A.workBeff;
  if((PetscInt)beff.size()!=A.nOwned) beff.assign((std::size_t)A.nOwned,0.0);
  // MPIAIJ local-SOR semantics: freeze the off-rank values for this outer SOR
  // application, form the effective local RHS once, then perform the requested
  // number of local symmetric sweeps on the owned-owned block.
  PetscCall(customMomentumExchange(A,x));
  for(PetscInt i=0;i<A.nOwned;++i) {
    if(clamp && (*clamp)[(std::size_t)i]) { beff[(std::size_t)i]=0.0; continue; }
    double v=b[(std::size_t)i];
    for(PetscInt k=A.rowPtr[(std::size_t)i];k<A.rowPtr[(std::size_t)i+1];++k) {
      const PetscInt li=A.colLocal[(std::size_t)k]; if(li>=A.nOwned) v-=A.aRel[(std::size_t)k]*A.ghostValues[(std::size_t)(li-A.nOwned)];
    }
    beff[(std::size_t)i]=v;
  }
  for(PetscInt sweep=0;sweep<localSweeps;++sweep) {
    for(PetscInt i=0;i<A.nOwned;++i) {
      if(clamp && (*clamp)[(std::size_t)i]) { x[(std::size_t)i]=0.0; continue; }
      const double old=x[(std::size_t)i],diag=A.aRel[(std::size_t)A.diagPos[(std::size_t)i]]; double rhs=beff[(std::size_t)i];
      for(PetscInt k=A.rowPtr[(std::size_t)i];k<A.rowPtr[(std::size_t)i+1];++k) {
        const PetscInt li=A.colLocal[(std::size_t)k]; if(li<A.nOwned && li!=i) rhs-=A.aRel[(std::size_t)k]*x[(std::size_t)li];
      }
      x[(std::size_t)i]=(1.0-omega)*old+omega*(rhs/diag);
    }
    for(PetscInt ii=A.nOwned;ii>0;--ii) {
      const PetscInt i=ii-1;
      if(clamp && (*clamp)[(std::size_t)i]) { x[(std::size_t)i]=0.0; continue; }
      const double old=x[(std::size_t)i],diag=A.aRel[(std::size_t)A.diagPos[(std::size_t)i]]; double rhs=beff[(std::size_t)i];
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

static PetscErrorCode customMomentumResidualNorm(CustomMomentumCSR& A,const std::vector<double>& b,const std::vector<double>& x,double& rn,const std::vector<char>* clamp=nullptr) {
  PetscFunctionBeginUser;
  if(clamp && (PetscInt)clamp->size()!=A.nOwned) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"custom residual clamp size mismatch");
  PetscCall(customMomentumMatVec(A,A.aRel,x,A.workY));
  double local=0.0;
  for(PetscInt i=0;i<A.nOwned;++i) {
    if(clamp && (*clamp)[(std::size_t)i]) continue;
    const double r=b[(std::size_t)i]-A.workY[(std::size_t)i]; local+=r*r;
  }
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
  PetscInt maxIts,PetscInt checkEvery,double omega,PetscInt localSweeps,PetscInt *parallelIts,double *relres,const std::vector<char>* clamp=nullptr) {
  PetscFunctionBeginUser;
  if((PetscInt)b.size()!=A.nOwned || (PetscInt)x.size()!=A.nOwned) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"M6B native momentum state size mismatch");
  if(clamp && (PetscInt)clamp->size()!=A.nOwned) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"M6B native clamp size mismatch");
  if(clamp) for(PetscInt i=0;i<A.nOwned;++i) if((*clamp)[(std::size_t)i]) x[(std::size_t)i]=0.0;
  double bn=0.0,rn=0.0,rnInitial=-1.0; PetscCall(customMomentumNorm2(b,bn));
  if(!std::isfinite(bn)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"M6B native momentum RHS norm NaN/Inf");
  if(bn==0.0) bn=1.0;
  PetscCall(customMomentumResidualNorm(A,b,x,rn,clamp)); rnInitial=rn; if(!std::isfinite(rnInitial)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"M6B native initial momentum residual NaN/Inf");
  const double target=std::max((double)atol,(double)rtol*bn);
  if(rnInitial<=target){*parallelIts=0;*relres=rnInitial/bn;PetscFunctionReturn(PETSC_SUCCESS);}
  PetscInt it=0; while(it<maxIts){ const PetscInt chunk=PetscMin(checkEvery,maxIts-it); for(PetscInt q=0;q<chunk;++q) PetscCall(customMomentumSymmetricGS(A,b,x,omega,localSweeps,clamp)); it+=chunk;
    PetscCall(customMomentumResidualNorm(A,b,x,rn,clamp)); if(!std::isfinite(rn)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"M6B native SGS NaN/Inf");
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
