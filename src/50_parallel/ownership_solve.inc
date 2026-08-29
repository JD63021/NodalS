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

  if(P.mode==ProblemMode::Pipe && P.inletBC==InletBCMode::PipeParabolic) {
    double localFlux[2]={0.0,0.0}, flux[2]={0.0,0.0};
    const auto& pin=M.patches[(std::size_t)P.pipe.inlet];
    for(PetscInt f=pin.startFace;f<pin.startFace+pin.nFaces;++f) if(D.cellOwner[(std::size_t)M.owner[(std::size_t)f]]==rank) {
      const auto& F=M.faces[(std::size_t)f]; Vec3 X[3]={M.points[(std::size_t)F.v[0]],M.points[(std::size_t)F.v[1]],M.points[(std::size_t)F.v[2]]};
      const double A=triangleArea(X[0],X[1],X[2]), u=triangleAverageIdealPipeUz(P.pipe,X);
      localFlux[0]+=A*u; localFlux[1]+=A*(P.pipe.profileScale*u);
    }
    PetscCallMPI(MPI_Allreduce(localFlux,flux,2,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
    const double expected=P.pipe.bulkVelocity*P.pipe.inletArea;
    const double rel=std::abs(flux[1]-expected)/std::max(1e-300,std::abs(expected));
    const bool fok=rel<5e-12; ok=ok&&fok;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_DIST_GATE0_INLET_FLUX profile=parabolic rawDistributed=%.12e profileScale=%.12e normalizedDistributed=%.12e expectedUbulkArea=%.12e relErr=%.3e status=%s\n",
      flux[0],P.pipe.profileScale,flux[1],expected,rel,fok?"PASS":"FAIL"));
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
  std::vector<char> b; const int mode=(int)P.mode,ibc=(int)P.inletBC,cc=P.centralConvection?1:0;
  distAppendPod(b,mode);distAppendPod(b,ibc);distAppendPod(b,cc);distAppendPod(b,P.re);distAppendPod(b,P.nu);
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
  std::size_t o=0; ProblemConfig P;P.mode=(ProblemMode)distReadPod<int>(b,o);P.inletBC=(InletBCMode)distReadPod<int>(b,o);P.centralConvection=distReadPod<int>(b,o)!=0;P.re=distReadPod<double>(b,o);P.nu=distReadPod<double>(b,o);
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
  const std::size_t nent=M.points.size()+M.faces.size();D.fixedEntity.assign(nent,0);D.g2free.assign(nent,-1);D.fixedDirValue.clear();D.fixedDirValue.reserve(nent/8+1);
  auto copyEntity=[&](std::size_t le,PetscInt ge){D.fixedEntity[le]=GD.fixedEntity[(std::size_t)ge];const PetscInt gg=GD.g2free[(std::size_t)ge];if(gg>=0){D.g2free[le]=gg;}else{const PetscInt gs=fixedDirSlotFromEncodedGid(gg);const PetscInt ls=(PetscInt)D.fixedDirValue.size();D.fixedDirValue.push_back(GD.fixedDirValue[(std::size_t)gs]);D.g2free[le]=-(ls+2);}};
  for(std::size_t lp=0;lp<points.size();++lp)copyEntity(lp,points[lp]);for(std::size_t lf=0;lf<faces.size();++lf)copyEntity(M.points.size()+lf,gnv+faces[lf]);
  return X;
}

static std::vector<char> packDistProductionPacket(const DistProductionPacket& X) {
  std::vector<char>b;distAppendPod(b,X.globalCells);distAppendPod(b,X.globalPoints);distAppendPod(b,X.globalFaces);distAppendPod(b,X.globalInternalFaces);
  distAppendVector(b,X.M.points);distAppendVector(b,X.M.faces);distAppendVector(b,X.M.owner);distAppendVector(b,X.M.neighbour);distAppendVector(b,X.M.tets);distAppendVector(b,X.M.oppFace);distAppendVector(b,X.M.facePatch);const std::uint64_t np=(std::uint64_t)X.M.patches.size();distAppendPod(b,np);for(const auto&q:X.M.patches){distAppendString(b,q.name);distAppendPod(b,q.startFace);distAppendPod(b,q.nFaces);}
  distAppendVector(b,X.D.g2free);distAppendVector(b,X.D.pGid);distAppendVector(b,X.D.cellOwner);distAppendVector(b,X.D.velCount);distAppendVector(b,X.D.cellCount);distAppendVector(b,X.D.fixedEntity);distAppendVector(b,X.D.fixedDirValue);distAppendPod(b,X.D.ns);distAppendPod(b,X.D.freeVertices);distAppendPod(b,X.D.freeFaces);distAppendPod(b,X.D.fixedVertices);distAppendPod(b,X.D.fixedFaces);return b;
}
static DistProductionPacket unpackDistProductionPacket(const std::vector<char>&b){std::size_t o=0;DistProductionPacket X;X.globalCells=distReadPod<PetscInt>(b,o);X.globalPoints=distReadPod<PetscInt>(b,o);X.globalFaces=distReadPod<PetscInt>(b,o);X.globalInternalFaces=distReadPod<PetscInt>(b,o);X.M.points=distReadVector<Vec3>(b,o);X.M.faces=distReadVector<Face>(b,o);X.M.owner=distReadVector<int>(b,o);X.M.neighbour=distReadVector<int>(b,o);X.M.tets=distReadVector<std::array<int,4>>(b,o);X.M.oppFace=distReadVector<std::array<int,4>>(b,o);X.M.facePatch=distReadVector<int>(b,o);const std::uint64_t np=distReadPod<std::uint64_t>(b,o);X.M.patches.resize((std::size_t)np);for(auto&q:X.M.patches){q.name=distReadString(b,o);q.startFace=distReadPod<PetscInt>(b,o);q.nFaces=distReadPod<PetscInt>(b,o);}X.D.g2free=distReadVector<PetscInt>(b,o);X.D.pGid=distReadVector<PetscInt>(b,o);X.D.cellOwner=distReadVector<int>(b,o);X.D.velCount=distReadVector<PetscInt>(b,o);X.D.cellCount=distReadVector<PetscInt>(b,o);X.D.fixedEntity=distReadVector<char>(b,o);X.D.fixedDirValue=distReadVector<std::array<double,3>>(b,o);X.D.ns=distReadPod<PetscInt>(b,o);X.D.freeVertices=distReadPod<PetscInt>(b,o);X.D.freeFaces=distReadPod<PetscInt>(b,o);X.D.fixedVertices=distReadPod<PetscInt>(b,o);X.D.fixedFaces=distReadPod<PetscInt>(b,o);if(o!=b.size())throw std::runtime_error("local mesh packet trailing bytes");return X;}

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

