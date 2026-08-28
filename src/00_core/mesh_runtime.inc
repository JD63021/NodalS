namespace {
constexpr double PI = 3.141592653589793238462643383279502884;

struct Vec3 { double x=0, y=0, z=0; };
struct Face { std::vector<int> v; };
struct Patch { std::string name; PetscInt startFace=0,nFaces=0; };
struct Mesh {
  std::vector<Vec3> points;
  std::vector<Face> faces;
  std::vector<int> owner, neighbour;
  std::vector<std::array<int,4>> tets;
  std::vector<std::array<int,4>> oppFace; // local face opposite local tet vertex i
  std::vector<Patch> patches;
  std::vector<int> facePatch; // -1 internal, otherwise index into patches
};

static std::vector<Vec3> cellCentroids(const Mesh& M);

static std::string readFile(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::ostringstream ss; ss << in.rdbuf(); return ss.str();
}


struct ProcessMemoryKiB { long long rss=0, hwm=0; };

static ProcessMemoryKiB readProcessMemoryKiB() {
  ProcessMemoryKiB m;
  std::ifstream in("/proc/self/status");
  std::string key;
  while(in >> key) {
    if(key=="VmRSS:" || key=="VmHWM:") {
      long long v=0; std::string unit;
      in >> v >> unit;
      if(key=="VmRSS:") m.rss=v; else m.hwm=v;
    } else {
      std::string rest; std::getline(in,rest);
    }
  }
  return m;
}

static PetscErrorCode printSetupPhase(const char *name, PetscLogDouble localSeconds, PetscInt cells) {
  PetscFunctionBeginUser;
  double x=(double)localSeconds,mn=0,mx=0,sum=0;
  int size=1; PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Allreduce(&x,&mn,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&x,&mx,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&x,&sum,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_SETUP_PHASE name=%s seconds=%.6f minRankSeconds=%.6f meanRankSeconds=%.6f cells=%" PetscInt_FMT "\n",
    name,mx,mn,sum/(double)size,cells));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode printResourceMark(const char *label, PetscInt cells,
                                        PetscLogDouble phaseSeconds,
                                        PetscLogDouble profileOrigin) {
  PetscFunctionBeginUser;
  const ProcessMemoryKiB local=readProcessMemoryKiB();
  long long rssSum=0,rssMax=0,hwmSum=0,hwmMax=0;
  PetscCallMPI(MPI_Allreduce(&local.rss,&rssSum,1,MPI_LONG_LONG_INT,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&local.rss,&rssMax,1,MPI_LONG_LONG_INT,MPI_MAX,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&local.hwm,&hwmSum,1,MPI_LONG_LONG_INT,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&local.hwm,&hwmMax,1,MPI_LONG_LONG_INT,MPI_MAX,PETSC_COMM_WORLD));
  PetscLogDouble now=0; PetscCall(PetscTime(&now));
  const auto epochMs=std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
  const double rssSumMiB=(double)rssSum/1024.0;
  const double rssMaxMiB=(double)rssMax/1024.0;
  const double hwmSumMiB=(double)hwmSum/1024.0;
  const double hwmMaxMiB=(double)hwmMax/1024.0;
  const double rssBpc=cells>0 ? (double)rssSum*1024.0/(double)cells : 0.0;
  const double hwmBpc=cells>0 ? (double)hwmSum*1024.0/(double)cells : 0.0;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_RESOURCE_MARK label=%s epochMs=%lld elapsedSeconds=%.6f phaseSeconds=%.6f cells=%" PetscInt_FMT
    " rssSumMiB=%.3f rssMaxRankMiB=%.3f hwmSumMiB=%.3f hwmMaxRankMiB=%.3f rssBytesPerCell=%.3f hwmBytesPerCell=%.3f\n",
    label,(long long)epochMs,(double)(now-profileOrigin),(double)phaseSeconds,cells,
    rssSumMiB,rssMaxMiB,hwmSumMiB,hwmMaxMiB,rssBpc,hwmBpc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static std::string stripComments(std::string s) {
  s = std::regex_replace(s, std::regex(R"(/\*[\s\S]*?\*/)"), "");
  s = std::regex_replace(s, std::regex(R"(//[^\n\r]*)"), "");
  return s;
}

struct ListBody { int n=0; std::string body; };
static ListBody extractListBody(const std::string& path) {
  std::string s = stripComments(readFile(path));
  size_t foam = s.find("FoamFile");
  size_t afterHeader = 0;
  if (foam != std::string::npos) {
    size_t lb = s.find('{', foam);
    if (lb == std::string::npos) throw std::runtime_error("bad FoamFile header in " + path);
    int depth=1; size_t i=lb+1;
    for (; i<s.size() && depth; ++i) { if (s[i]=='{') ++depth; else if (s[i]=='}') --depth; }
    if (depth) throw std::runtime_error("unterminated FoamFile header in " + path);
    afterHeader = i;
  }
  std::string tail = s.substr(afterHeader);
  std::smatch m;
  std::regex re(R"((\d+)\s*\()") ;
  if (!std::regex_search(tail,m,re)) throw std::runtime_error("cannot find OpenFOAM list count in " + path);
  int n = std::stoi(m[1].str());
  size_t openPos = afterHeader + (size_t)m.position(0) + (size_t)m.length(0) - 1;
  int depth=1; size_t i=openPos+1;
  for (; i<s.size() && depth; ++i) { if (s[i]=='(') ++depth; else if (s[i]==')') --depth; }
  if (depth) throw std::runtime_error("unterminated OpenFOAM list in " + path);
  return {n, s.substr(openPos+1, (i-1)-(openPos+1))};
}

static std::vector<Vec3> readPoints(const std::string& path) {
  auto L=extractListBody(path); std::vector<Vec3> p; p.reserve(L.n);
  std::regex re(R"(\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\))");
  for (auto it=std::sregex_iterator(L.body.begin(),L.body.end(),re); it!=std::sregex_iterator(); ++it) {
    p.push_back({std::stod((*it)[1].str()),std::stod((*it)[2].str()),std::stod((*it)[3].str())});
  }
  if ((int)p.size()!=L.n) throw std::runtime_error("point count mismatch in " + path);
  return p;
}

static std::vector<Face> readFaces(const std::string& path) {
  auto L=extractListBody(path); std::vector<Face> F; F.reserve(L.n);
  std::regex re(R"((\d+)\s*\(([^()]*)\))");
  for (auto it=std::sregex_iterator(L.body.begin(),L.body.end(),re); it!=std::sregex_iterator(); ++it) {
    int k=std::stoi((*it)[1].str()); std::istringstream is((*it)[2].str()); Face f; int v;
    while (is>>v) f.v.push_back(v);
    if ((int)f.v.size()!=k) throw std::runtime_error("face arity mismatch in " + path);
    F.push_back(std::move(f));
  }
  if ((int)F.size()!=L.n) throw std::runtime_error("face count mismatch in " + path);
  return F;
}

static std::vector<int> readLabels(const std::string& path) {
  auto L=extractListBody(path); std::vector<int> a; a.reserve(L.n);
  std::istringstream is(L.body); long long v;
  while (is>>v) a.push_back((int)v);
  if ((int)a.size()!=L.n) throw std::runtime_error("label count mismatch in " + path + ": got " + std::to_string(a.size()) + " expected " + std::to_string(L.n));
  return a;
}

static std::vector<Patch> readBoundary(const std::string& path) {
  auto L=extractListBody(path);
  std::vector<Patch> out; out.reserve(L.n);
  std::regex re(R"(([A-Za-z0-9_.+\-]+)\s*\{([^{}]*)\})");
  for (auto it=std::sregex_iterator(L.body.begin(),L.body.end(),re); it!=std::sregex_iterator(); ++it) {
    const std::string name=(*it)[1].str(), body=(*it)[2].str();
    std::smatch mn,ms;
    if(!std::regex_search(body,mn,std::regex(R"(nFaces\s+(\d+)\s*;)")) ||
       !std::regex_search(body,ms,std::regex(R"(startFace\s+(\d+)\s*;)")))
      throw std::runtime_error("cannot parse boundary patch " + name + " in " + path);
    out.push_back({name,(PetscInt)std::stoll(ms[1].str()),(PetscInt)std::stoll(mn[1].str())});
  }
  if((int)out.size()!=L.n) throw std::runtime_error("boundary patch count mismatch in " + path);
  return out;
}

static double det3(const double J[3][3]) {
  return J[0][0]*(J[1][1]*J[2][2]-J[1][2]*J[2][1])
       - J[0][1]*(J[1][0]*J[2][2]-J[1][2]*J[2][0])
       + J[0][2]*(J[1][0]*J[2][1]-J[1][1]*J[2][0]);
}
static void inv3(const double J[3][3], double I[3][3]) {
  double d=det3(J); if (std::abs(d)<1e-18) throw std::runtime_error("singular tetrahedron");
  I[0][0]=(J[1][1]*J[2][2]-J[1][2]*J[2][1])/d;
  I[0][1]=(J[0][2]*J[2][1]-J[0][1]*J[2][2])/d;
  I[0][2]=(J[0][1]*J[1][2]-J[0][2]*J[1][1])/d;
  I[1][0]=(J[1][2]*J[2][0]-J[1][0]*J[2][2])/d;
  I[1][1]=(J[0][0]*J[2][2]-J[0][2]*J[2][0])/d;
  I[1][2]=(J[0][2]*J[1][0]-J[0][0]*J[1][2])/d;
  I[2][0]=(J[1][0]*J[2][1]-J[1][1]*J[2][0])/d;
  I[2][1]=(J[0][1]*J[2][0]-J[0][0]*J[2][1])/d;
  I[2][2]=(J[0][0]*J[1][1]-J[0][1]*J[1][0])/d;
}

static Mesh loadFoamTetMesh(const std::string& pm) {
  Mesh M; M.points=readPoints(pm+"/points"); M.faces=readFaces(pm+"/faces");
  M.owner=readLabels(pm+"/owner"); M.neighbour=readLabels(pm+"/neighbour");
  M.patches=readBoundary(pm+"/boundary");
  M.facePatch.assign(M.faces.size(),-1);
  for(int p=0;p<(int)M.patches.size();++p) {
    const auto& P=M.patches[p];
    for(PetscInt f=P.startFace; f<P.startFace+P.nFaces; ++f) {
      if(f<0 || f>=(PetscInt)M.faces.size()) throw std::runtime_error("boundary patch face range out of bounds");
      if(M.facePatch[f]>=0) throw std::runtime_error("boundary face appears in multiple patches");
      M.facePatch[f]=p;
    }
  }
  if (M.owner.size()!=M.faces.size()) throw std::runtime_error("owner size != faces size");
  for (auto& f:M.faces) if (f.v.size()!=3) throw std::runtime_error("non-triangular face found; this first solver requires tetrahedra");
  int maxc=-1; for(int x:M.owner) maxc=std::max(maxc,x); for(int x:M.neighbour) maxc=std::max(maxc,x);
  int nc=maxc+1; std::vector<std::vector<int>> cf(nc);
  for (int f=0; f<(int)M.faces.size(); ++f) cf[M.owner[f]].push_back(f);
  for (int f=0; f<(int)M.neighbour.size(); ++f) cf[M.neighbour[f]].push_back(f);
  M.tets.resize(nc); M.oppFace.resize(nc);
  for (int c=0;c<nc;++c) {
    if (cf[c].size()!=4) throw std::runtime_error("cell "+std::to_string(c)+" has "+std::to_string(cf[c].size())+" faces, not 4");
    std::set<int> vs; for(int f:cf[c]) for(int v:M.faces[f].v) vs.insert(v);
    if (vs.size()!=4) throw std::runtime_error("cell "+std::to_string(c)+" does not have 4 unique vertices");
    std::array<int,4> t; std::copy(vs.begin(),vs.end(),t.begin());
    auto X0=M.points[t[0]],X1=M.points[t[1]],X2=M.points[t[2]],X3=M.points[t[3]];
    double J[3][3]={{X1.x-X0.x,X2.x-X0.x,X3.x-X0.x},{X1.y-X0.y,X2.y-X0.y,X3.y-X0.y},{X1.z-X0.z,X2.z-X0.z,X3.z-X0.z}};
    if (det3(J)<0) std::swap(t[1],t[2]);
    M.tets[c]=t;
    for (int i=0;i<4;++i) {
      int found=-1;
      for(int f:cf[c]) {
        bool has=false; for(int v:M.faces[f].v) if(v==t[i]) {has=true;break;}
        if(!has) { if(found>=0) throw std::runtime_error("multiple opposite faces"); found=f; }
      }
      if(found<0) throw std::runtime_error("missing opposite face");
      M.oppFace[c][i]=found;
    }
  }
  return M;
}

