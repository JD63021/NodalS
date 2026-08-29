enum class ProblemMode { MMS, Pipe, Flow };
enum class InletBCMode { PipeParabolic, FixedNormalSpeed };

struct PipeGeometry {
  std::string wallPatch="patch_0_0", inletPatch="patch_2_0", outletPatch="patch_1_0";
  int wall=-1,inlet=-1,outlet=-1;
  double cx=0.0,cy=0.0,zIn=0.0,zOut=0.0,R=0.0,D=0.0,L=0.0;
  double inletArea=0.0,outletArea=0.0,circleArea=0.0,areaRatio=0.0;
  double bulkVelocity=1.0,profileScale=1.0,nu=0.0,re=0.0,hpDrop=0.0,hpGradient=0.0;
};

struct BoundaryGeometry {
  std::vector<std::string> wallPatches;
  std::vector<int> walls;
  std::string inletPatch, outletPatch;
  int inlet=-1,outlet=-1;
  double inletArea=0.0,outletArea=0.0,inletProjectedArea=0.0,outletProjectedArea=0.0;
  Vec3 inletAreaVector{},outletAreaVector{},inletReferenceNormal{},outletReferenceNormal{};
  double signedNormalSpeed=-1.0;
  Vec3 inletVelocity{};
};

struct ProblemConfig {
  ProblemMode mode=ProblemMode::MMS;
  InletBCMode inletBC=InletBCMode::PipeParabolic;
  bool centralConvection=true;
  double re=1.0,nu=1.0;
  PipeGeometry pipe;
  BoundaryGeometry boundary;
};

static Vec3 cross3(const Vec3& a,const Vec3& b) {
  return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};
}
static Vec3 sub3(const Vec3& a,const Vec3& b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
static double norm3(const Vec3& a) { return std::sqrt(a.x*a.x+a.y*a.y+a.z*a.z); }
static double triangleArea(const Vec3& a,const Vec3& b,const Vec3& c) { return 0.5*norm3(cross3(sub3(b,a),sub3(c,a))); }
static Vec3 add3(const Vec3& a,const Vec3& b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
static Vec3 scale3(const Vec3& a,double s) { return {s*a.x,s*a.y,s*a.z}; }
static double dot3(const Vec3& a,const Vec3& b) { return a.x*b.x+a.y*b.y+a.z*b.z; }

static Vec3 cellCentroidOne(const Mesh& M,PetscInt c) {
  Vec3 q{}; for(int i=0;i<4;++i) q=add3(q,M.points[M.tets[c][i]]); return scale3(q,0.25);
}

static Vec3 faceOutwardAreaVector(const Mesh& M,PetscInt f) {
  const auto& F=M.faces[f];
  const Vec3& x0=M.points[F.v[0]]; const Vec3& x1=M.points[F.v[1]]; const Vec3& x2=M.points[F.v[2]];
  Vec3 sf=scale3(cross3(sub3(x1,x0),sub3(x2,x0)),0.5);
  const Vec3 fc=scale3(add3(add3(x0,x1),x2),1.0/3.0);
  const Vec3 cc=cellCentroidOne(M,M.owner[f]);
  if(dot3(sf,sub3(fc,cc))<0.0) sf=scale3(sf,-1.0);
  return sf;
}

struct PatchFrame { double area=0.0,projectedArea=0.0; Vec3 areaVector{},normal{}; };
static PatchFrame patchFrame(const Mesh& M,int pi) {
  PatchFrame F; const auto& P=M.patches[pi];
  for(PetscInt f=P.startFace;f<P.startFace+P.nFaces;++f) {
    const Vec3 sf=faceOutwardAreaVector(M,f); F.area += norm3(sf); F.areaVector=add3(F.areaVector,sf);
  }
  F.projectedArea=norm3(F.areaVector);
  if(!(F.area>0.0) || !(F.projectedArea>0.0)) throw std::runtime_error("degenerate boundary patch " + P.name);
  F.normal=scale3(F.areaVector,1.0/F.projectedArea);
  return F;
}

static std::vector<std::string> splitPatchNames(const std::string& text) {
  std::vector<std::string> out; std::string cur;
  auto flush=[&](){ if(!cur.empty()){out.push_back(cur);cur.clear();} };
  for(char ch:text) { if(ch==',' || ch==';' || ch==' ' || ch=='\t') flush(); else cur.push_back(ch); }
  flush(); return out;
}

static int patchIndex(const Mesh& M,const std::string& name) {
  for(int p=0;p<(int)M.patches.size();++p) if(M.patches[p].name==name) return p;
  return -1;
}

static BoundaryGeometry makeBoundaryGeometry(const Mesh& M,const std::vector<std::string>& wallNames,
                                             const std::string& inletName,const std::string& outletName,
                                             double signedNormalSpeed) {
  BoundaryGeometry B; B.wallPatches=wallNames; B.inletPatch=inletName; B.outletPatch=outletName; B.signedNormalSpeed=signedNormalSpeed;
  if(wallNames.empty()) throw std::runtime_error("at least one wall patch is required");
  for(const auto& w:wallNames) { int pi=patchIndex(M,w); if(pi<0) throw std::runtime_error("wall patch not found: "+w); B.walls.push_back(pi); }
  B.inlet=patchIndex(M,inletName); B.outlet=patchIndex(M,outletName);
  if(B.inlet<0) throw std::runtime_error("inlet patch not found: "+inletName);
  if(B.outlet<0) throw std::runtime_error("outlet patch not found: "+outletName);
  if(B.inlet==B.outlet) throw std::runtime_error("inlet and outlet patches must differ");
  for(int w:B.walls) if(w==B.inlet || w==B.outlet) throw std::runtime_error("wall patch overlaps inlet/outlet role");
  std::vector<int> role(M.patches.size(),0);
  for(int w:B.walls) role[w]++; role[B.inlet]++; role[B.outlet]++;
  std::ostringstream missing; bool anyMissing=false;
  for(int pi=0;pi<(int)M.patches.size();++pi) if(role[pi]==0) { if(anyMissing) missing<<','; missing<<M.patches[pi].name; anyMissing=true; }
  if(anyMissing) throw std::runtime_error("unclassified boundary patches (add them to -flow_wall_patches or choose inlet/outlet): "+missing.str());
  const PatchFrame fi=patchFrame(M,B.inlet),fo=patchFrame(M,B.outlet);
  B.inletArea=fi.area; B.outletArea=fo.area; B.inletProjectedArea=fi.projectedArea; B.outletProjectedArea=fo.projectedArea;
  B.inletAreaVector=fi.areaVector; B.outletAreaVector=fo.areaVector; B.inletReferenceNormal=fi.normal; B.outletReferenceNormal=fo.normal;
  B.inletVelocity=scale3(B.inletReferenceNormal,signedNormalSpeed);
  return B;
}

static bool isWallPatch(const BoundaryGeometry& B,int pi) {
  return std::find(B.walls.begin(),B.walls.end(),pi)!=B.walls.end();
}

static PetscErrorCode printPatchAuditRoot(const Mesh& M) {
  PetscFunctionBeginUser;
  for(int pi=0;pi<(int)M.patches.size();++pi) {
    const auto F=patchFrame(M,pi);
    PetscCall(PetscPrintf(PETSC_COMM_SELF,
      "P1BF3_PATCH_AUDIT name=%s startFace=%" PetscInt_FMT " nFaces=%" PetscInt_FMT " area=%.12e areaVector=[%.12e,%.12e,%.12e] averageOutwardNormal=[%.12e,%.12e,%.12e] projectedArea=%.12e planarityRatio=%.12e\n",
      M.patches[pi].name.c_str(),M.patches[pi].startFace,M.patches[pi].nFaces,F.area,F.areaVector.x,F.areaVector.y,F.areaVector.z,F.normal.x,F.normal.y,F.normal.z,F.projectedArea,F.projectedArea/F.area));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static double pipeIdealUz(const PipeGeometry& P,double x,double y) {
  const double rr=(x-P.cx)*(x-P.cx)+(y-P.cy)*(y-P.cy);
  return 2.0*P.bulkVelocity*(1.0-rr/(P.R*P.R));
}
static double pipeBoundaryUz(const PipeGeometry& P,double x,double y) { return P.profileScale*pipeIdealUz(P,x,y); }
static double pipeExactPressure(const PipeGeometry& P,double z) { return P.hpGradient*(P.zOut-z); }

static double triangleAverageIdealPipeUz(const PipeGeometry& P,const Vec3 X[3]) {
  double xx[3],yy[3];
  for(int i=0;i<3;++i){xx[i]=X[i].x-P.cx;yy[i]=X[i].y-P.cy;}
  const double avgx2=(xx[0]*xx[0]+xx[1]*xx[1]+xx[2]*xx[2]+xx[0]*xx[1]+xx[1]*xx[2]+xx[2]*xx[0])/6.0;
  const double avgy2=(yy[0]*yy[0]+yy[1]*yy[1]+yy[2]*yy[2]+yy[0]*yy[1]+yy[1]*yy[2]+yy[2]*yy[0])/6.0;
  return 2.0*P.bulkVelocity*(1.0-(avgx2+avgy2)/(P.R*P.R));
}

static PipeGeometry makePipeGeometry(const Mesh& M,double re,double bulkVelocity,
                                     const std::string& wallName,const std::string& inletName,const std::string& outletName) {
  PipeGeometry P; P.wallPatch=wallName;P.inletPatch=inletName;P.outletPatch=outletName;
  P.wall=patchIndex(M,wallName);P.inlet=patchIndex(M,inletName);P.outlet=patchIndex(M,outletName);
  if(P.wall<0||P.inlet<0||P.outlet<0) throw std::runtime_error("pipe patch names not found in boundary file");
  double xmin=1e300,xmax=-1e300,ymin=1e300,ymax=-1e300,zmin=1e300,zmax=-1e300;
  for(const auto& x:M.points){xmin=std::min(xmin,x.x);xmax=std::max(xmax,x.x);ymin=std::min(ymin,x.y);ymax=std::max(ymax,x.y);zmin=std::min(zmin,x.z);zmax=std::max(zmax,x.z);}
  P.cx=0.5*(xmin+xmax);P.cy=0.5*(ymin+ymax);P.zIn=zmin;P.zOut=zmax;P.L=zmax-zmin;
  P.R=0.5*std::max(xmax-xmin,ymax-ymin);P.D=2.0*P.R;P.bulkVelocity=bulkVelocity;P.re=re;
  if(!(P.R>0&&P.L>0&&re>0&&bulkVelocity>0)) throw std::runtime_error("invalid pipe geometry/Re/Umean");
  P.nu=bulkVelocity*P.D/re;
  P.circleArea=PI*P.R*P.R;
  auto patchArea=[&](int pi){double A=0;const auto& pp=M.patches[pi];for(PetscInt f=pp.startFace;f<pp.startFace+pp.nFaces;++f){const auto& F=M.faces[f];A+=triangleArea(M.points[F.v[0]],M.points[F.v[1]],M.points[F.v[2]]);}return A;};
  P.inletArea=patchArea(P.inlet);P.outletArea=patchArea(P.outlet);P.areaRatio=P.inletArea/P.circleArea;
  double rawFlux=0.0;const auto& pin=M.patches[P.inlet];
  for(PetscInt f=pin.startFace;f<pin.startFace+pin.nFaces;++f){const auto& F=M.faces[f];Vec3 X[3]={M.points[F.v[0]],M.points[F.v[1]],M.points[F.v[2]]};rawFlux+=triangleArea(X[0],X[1],X[2])*triangleAverageIdealPipeUz(P,X);}
  if(!(rawFlux>0)) throw std::runtime_error("non-positive raw parabolic inlet flux");
  P.profileScale=bulkVelocity*P.inletArea/rawFlux;
  P.hpGradient=32.0*P.nu*bulkVelocity/(P.D*P.D);
  P.hpDrop=P.hpGradient*P.L;
  return P;
}

static void problemForcing(const ProblemConfig& P,double x,double y,double z,double f[3]) {
  if(P.mode!=ProblemMode::MMS){f[0]=f[1]=f[2]=0.0;return;}
  forcing(x,y,z,P.nu,P.centralConvection,f);
}


static PetscErrorCode buildPlexAuditSelf(const Mesh& M) {
  PetscFunctionBeginUser;
  std::vector<PetscInt> cells(4*M.tets.size());
  for(size_t c=0;c<M.tets.size();++c) for(int i=0;i<4;++i) cells[4*c+i]=M.tets[c][i];
  std::vector<PetscReal> xyz(3*M.points.size());
  for(size_t i=0;i<M.points.size();++i){xyz[3*i]=M.points[i].x;xyz[3*i+1]=M.points[i].y;xyz[3*i+2]=M.points[i].z;}
  DM dm=nullptr;
  PetscCall(DMPlexCreateFromCellListPetsc(PETSC_COMM_SELF,3,(PetscInt)M.tets.size(),(PetscInt)M.points.size(),4,PETSC_TRUE,cells.data(),3,xyz.data(),&dm));
  PetscInt cs,ce,fs,fe,es,ee,vs,ve;
  PetscCall(DMPlexGetHeightStratum(dm,0,&cs,&ce)); PetscCall(DMPlexGetHeightStratum(dm,1,&fs,&fe));
  PetscCall(DMPlexGetDepthStratum(dm,1,&es,&ee)); PetscCall(DMPlexGetDepthStratum(dm,0,&vs,&ve));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"P1BF3_DMPLEX_AUDIT cells=%" PetscInt_FMT " faces=%" PetscInt_FMT " edges=%" PetscInt_FMT " vertices=%" PetscInt_FMT " status=%s scope=rank0_serial_audit\n",ce-cs,fe-fs,ee-es,ve-vs,((size_t)(ce-cs)==M.tets.size() && (size_t)(fe-fs)==M.faces.size() && (size_t)(ve-vs)==M.points.size())?"PASS":"CHECK"));
  PetscCall(DMDestroy(&dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

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
    if(isWallPatch(B,pi) || pi==B.inlet) D.fixedEntity[nv+f]=1;
    if(isWallPatch(B,pi)) for(int v:M.faces[f].v) onWall[v]=1;
    if(pi==B.inlet) for(int v:M.faces[f].v) onInlet[v]=1;
  }
  for(PetscInt v=0;v<nv;++v) if(onWall[v] || onInlet[v]) D.fixedEntity[v]=1;

  // Allocate compact slots only after the complete fixed-entity mask is known.
  finalizeCompactDirichletSlots(D);

  for(PetscInt v=0;v<nv;++v) {
    // Preserve the existing wall-priority convention at inlet/wall edge vertices.
    // BF3 below restores the requested inlet face mean exactly despite that corner incompatibility.
    if(onInlet[v] && !onWall[v]) {
      if(P.inletBC==InletBCMode::PipeParabolic) setEntityDirValue(D,2,v,pipeBoundaryUz(P.pipe,M.points[v].x,M.points[v].y));
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
    if(P.inletBC==InletBCMode::PipeParabolic) {
      Vec3 X[3]={M.points[F.v[0]],M.points[F.v[1]],M.points[F.v[2]]};
      exactMean[2]=P.pipe.profileScale*triangleAverageIdealPipeUz(P.pipe,X);
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

