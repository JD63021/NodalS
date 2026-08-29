namespace {
constexpr double PI = 3.141592653589793238462643383279502884;

struct Vec3 { double x=0, y=0, z=0; };
struct Face { std::array<int,3> v{}; };
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

static bool parseIntegerToken(const std::string& tok,long long& value) {
  if(tok.empty()) return false;
  char *end=nullptr;
  errno=0;
  const long long v=std::strtoll(tok.c_str(),&end,10);
  if(errno!=0 || !end || *end!='\0') return false;
  value=v;
  return true;
}

class FoamTokenStream {
public:
  explicit FoamTokenStream(const std::string& p):in_(p),path_(p) {
    if(!in_) throw std::runtime_error("cannot open " + path_);
  }

  std::string next() {
    char c=0;
    while(in_.get(c)) {
      if(std::isspace(static_cast<unsigned char>(c))) continue;
      if(c=='/') {
        const int q=in_.peek();
        if(q=='/') { in_.get(); skipLineComment(); continue; }
        if(q=='*') { in_.get(); skipBlockComment(); continue; }
      }
      if(c=='(' || c==')' || c=='{' || c=='}' || c==';') return std::string(1,c);
      if(c=='\"') return readQuoted();

      std::string tok(1,c);
      while(true) {
        const int q=in_.peek();
        if(q==EOF) break;
        const char d=static_cast<char>(q);
        if(std::isspace(static_cast<unsigned char>(d)) || d=='(' || d==')' || d=='{' || d=='}' || d==';') break;
        if(d=='/') {
          // A slash beginning a comment terminates the current token. Plain slashes are
          // retained so quoted/unquoted paths remain harmless if encountered in headers.
          in_.get();
          const int r=in_.peek();
          in_.unget();
          if(r=='/' || r=='*') break;
        }
        in_.get(); tok.push_back(d);
      }
      return tok;
    }
    return {};
  }

  const std::string& path() const { return path_; }

private:
  std::ifstream in_;
  std::string path_;

  void skipLineComment() {
    char c=0; while(in_.get(c)) if(c=='\n' || c=='\r') break;
  }
  void skipBlockComment() {
    char prev=0,c=0;
    while(in_.get(c)) { if(prev=='*' && c=='/') return; prev=c; }
    throw std::runtime_error("unterminated block comment in " + path_);
  }
  std::string readQuoted() {
    std::string out; char c=0;
    while(in_.get(c)) {
      if(c=='\\') {
        char d=0; if(!in_.get(d)) break; out.push_back(d); continue;
      }
      if(c=='\"') return out;
      out.push_back(c);
    }
    throw std::runtime_error("unterminated quoted string in " + path_);
  }
};


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

static void expectFoamToken(FoamTokenStream& ts,const char *want) {
  const std::string got=ts.next();
  if(got!=want) throw std::runtime_error("expected token '"+std::string(want)+"' but got '"+got+"' in "+ts.path());
}

static long long requireIntegerToken(FoamTokenStream& ts,const std::string& tok,const char *what) {
  long long v=0;
  if(!parseIntegerToken(tok,v)) throw std::runtime_error("invalid "+std::string(what)+" token '"+tok+"' in "+ts.path());
  return v;
}

static double requireDoubleToken(FoamTokenStream& ts,const std::string& tok,const char *what) {
  if(tok.empty()) throw std::runtime_error("missing "+std::string(what)+" in "+ts.path());
  char *end=nullptr; errno=0;
  const double v=std::strtod(tok.c_str(),&end);
  if(errno!=0 || !end || *end!='\0') throw std::runtime_error("invalid "+std::string(what)+" token '"+tok+"' in "+ts.path());
  return v;
}

static int seekFoamListStart(FoamTokenStream& ts) {
  // polyMesh files have a FoamFile dictionary followed by: <count> ( ... ).
  // Scan tokens rather than materializing/regex-rewriting the complete file.
  for(std::string tok=ts.next(); !tok.empty(); tok=ts.next()) {
    long long n=0;
    if(!parseIntegerToken(tok,n) || n<0 || n>INT_MAX) continue;
    const std::string next=ts.next();
    if(next=="(") return static_cast<int>(n);
  }
  throw std::runtime_error("cannot find OpenFOAM list count in " + ts.path());
}

static std::vector<Vec3> readPoints(const std::string& path) {
  FoamTokenStream ts(path);
  const int n=seekFoamListStart(ts);
  std::vector<Vec3> p; p.reserve(static_cast<std::size_t>(n));
  for(int i=0;i<n;++i) {
    expectFoamToken(ts,"(");
    const double x=requireDoubleToken(ts,ts.next(),"point x");
    const double y=requireDoubleToken(ts,ts.next(),"point y");
    const double z=requireDoubleToken(ts,ts.next(),"point z");
    expectFoamToken(ts,")");
    p.push_back({x,y,z});
  }
  expectFoamToken(ts,")");
  return p;
}

static std::vector<Face> readFaces(const std::string& path) {
  FoamTokenStream ts(path);
  const int n=seekFoamListStart(ts);
  std::vector<Face> F; F.reserve(static_cast<std::size_t>(n));
  for(int i=0;i<n;++i) {
    const long long k=requireIntegerToken(ts,ts.next(),"face arity");
    if(k!=3) throw std::runtime_error("non-triangular face found in "+path+"; this tetrahedral solver requires face arity 3");
    expectFoamToken(ts,"(");
    Face f;
    for(int j=0;j<3;++j) {
      const long long v=requireIntegerToken(ts,ts.next(),"face vertex");
      if(v<0 || v>INT_MAX) throw std::runtime_error("face vertex label out of range in "+path);
      f.v[static_cast<std::size_t>(j)]=static_cast<int>(v);
    }
    expectFoamToken(ts,")");
    F.push_back(f);
  }
  expectFoamToken(ts,")");
  return F;
}

static std::vector<int> readLabels(const std::string& path) {
  FoamTokenStream ts(path);
  const int n=seekFoamListStart(ts);
  std::vector<int> a; a.reserve(static_cast<std::size_t>(n));
  for(int i=0;i<n;++i) {
    const long long v=requireIntegerToken(ts,ts.next(),"label");
    if(v<INT_MIN || v>INT_MAX) throw std::runtime_error("label out of int range in "+path);
    a.push_back(static_cast<int>(v));
  }
  expectFoamToken(ts,")");
  return a;
}

static std::vector<Patch> readBoundary(const std::string& path) {
  FoamTokenStream ts(path);
  const int n=seekFoamListStart(ts);
  std::vector<Patch> out; out.reserve(static_cast<std::size_t>(n));
  for(int pi=0;pi<n;++pi) {
    const std::string name=ts.next();
    if(name.empty() || name==")") throw std::runtime_error("missing boundary patch name in "+path);
    expectFoamToken(ts,"{");
    long long nFaces=-1,startFace=-1;
    int depth=1;
    while(depth>0) {
      const std::string tok=ts.next();
      if(tok.empty()) throw std::runtime_error("unterminated boundary patch "+name+" in "+path);
      if(tok=="{") { ++depth; continue; }
      if(tok=="}") { --depth; continue; }
      if(depth!=1) continue;
      if(tok=="nFaces") {
        nFaces=requireIntegerToken(ts,ts.next(),"nFaces");
        const std::string semi=ts.next(); if(semi!=";") throw std::runtime_error("missing ';' after nFaces for "+name+" in "+path);
      } else if(tok=="startFace") {
        startFace=requireIntegerToken(ts,ts.next(),"startFace");
        const std::string semi=ts.next(); if(semi!=";") throw std::runtime_error("missing ';' after startFace for "+name+" in "+path);
      }
    }
    if(nFaces<0 || startFace<0) throw std::runtime_error("cannot parse boundary patch "+name+" in "+path);
    out.push_back({name,(PetscInt)startFace,(PetscInt)nFaces});
  }
  expectFoamToken(ts,")");
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

