struct ProblemConfig {
  ProblemMode mode=ProblemMode::MMS;
  InletBCMode inletBC=InletBCMode::PipeParabolic;
  bool centralConvection=true;
  bool weakWallFunction=false;
  double re=1.0,nu=1.0;
  PipeGeometry pipe;
  BoundaryGeometry boundary;
};

static inline bool dgNumericalTraceInlet(const ProblemConfig& P) {
  return P.inletBC==InletBCMode::DgNumericalTrace;
}

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

static bool dgNumericalTraceFaceGeom(const Mesh& M,const ProblemConfig& P,PetscInt c,int& opp,Vec3& sf) {
  opp=-1; sf={};
  if(!dgNumericalTraceInlet(P)) return false;
  for(int i=0;i<4;++i) {
    const PetscInt f=M.oppFace[(std::size_t)c][i];
    if(f<(PetscInt)M.neighbour.size()) continue;
    if(M.facePatch[(std::size_t)f]!=P.boundary.inlet) continue;
    if(opp>=0) throw std::runtime_error("tetrahedron has multiple weak-DG inlet faces");
    opp=i; sf=faceOutwardAreaVector(M,f);
  }
  return opp>=0;
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
static double pipeIdealOneSeventhUz(const PipeGeometry& P,double x,double y) {
  const double r=std::sqrt((x-P.cx)*(x-P.cx)+(y-P.cy)*(y-P.cy));
  const double q=std::max(0.0,1.0-r/P.R);
  return P.bulkVelocity*std::pow(q,1.0/7.0);
}
static double pipeBoundaryUz(const ProblemConfig& P,double x,double y) {
  const double raw=(P.inletBC==InletBCMode::PipeOneSeventh)?pipeIdealOneSeventhUz(P.pipe,x,y):pipeIdealUz(P.pipe,x,y);
  return P.pipe.profileScale*raw;
}
static double pipeExactPressure(const PipeGeometry& P,double z) { return P.hpGradient*(P.zOut-z); }

static double triangleAverageIdealPipeUz(const PipeGeometry& P,const Vec3 X[3]) {
  double xx[3],yy[3];
  for(int i=0;i<3;++i){xx[i]=X[i].x-P.cx;yy[i]=X[i].y-P.cy;}
  const double avgx2=(xx[0]*xx[0]+xx[1]*xx[1]+xx[2]*xx[2]+xx[0]*xx[1]+xx[1]*xx[2]+xx[2]*xx[0])/6.0;
  const double avgy2=(yy[0]*yy[0]+yy[1]*yy[1]+yy[2]*yy[2]+yy[0]*yy[1]+yy[1]*yy[2]+yy[2]*yy[0])/6.0;
  return 2.0*P.bulkVelocity*(1.0-(avgx2+avgy2)/(P.R*P.R));
}
static double triangleAverageIdealPipeOneSeventhUz(const PipeGeometry& P,const Vec3 X[3]) {
  // Dunavant degree-5 seven-point rule.  The 1/7 profile is non-polynomial,
  // so use the same robust face rule as the weak wall model rather than a
  // centroid sample.  Weights sum to one and therefore return a face average.
  static const double L[7][3]={
    {1.0/3.0,1.0/3.0,1.0/3.0},
    {0.059715871789770,0.470142064105115,0.470142064105115},
    {0.470142064105115,0.059715871789770,0.470142064105115},
    {0.470142064105115,0.470142064105115,0.059715871789770},
    {0.797426985353087,0.101286507323456,0.101286507323456},
    {0.101286507323456,0.797426985353087,0.101286507323456},
    {0.101286507323456,0.101286507323456,0.797426985353087}
  };
  static const double W[7]={0.225000000000000,0.132394152788506,0.132394152788506,0.132394152788506,0.125939180544827,0.125939180544827,0.125939180544827};
  double avg=0.0;
  for(int q=0;q<7;++q) {
    double x=0.0,y=0.0; for(int a=0;a<3;++a){x+=L[q][a]*X[a].x;y+=L[q][a]*X[a].y;}
    avg+=W[q]*pipeIdealOneSeventhUz(P,x,y);
  }
  return avg;
}
static double triangleAverageIdealPipeProfileUz(const PipeGeometry& P,InletBCMode mode,const Vec3 X[3]) {
  return (mode==InletBCMode::PipeOneSeventh)?triangleAverageIdealPipeOneSeventhUz(P,X):triangleAverageIdealPipeUz(P,X);
}

static PipeGeometry makePipeGeometry(const Mesh& M,double re,double bulkVelocity,InletBCMode inletBC,
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
  for(PetscInt f=pin.startFace;f<pin.startFace+pin.nFaces;++f){const auto& F=M.faces[f];Vec3 X[3]={M.points[F.v[0]],M.points[F.v[1]],M.points[F.v[2]]};rawFlux+=triangleArea(X[0],X[1],X[2])*triangleAverageIdealPipeProfileUz(P,inletBC,X);}
  if(!(rawFlux>0)) throw std::runtime_error("non-positive raw inlet profile flux");
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

