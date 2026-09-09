#include <petscksp.h>
#include <petscdmplex.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <climits>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <iostream>
#include <map>
#include <numeric>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>
#include <type_traits>

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

struct Quad { std::array<double,4> lam; double w; };
static std::vector<Quad> tetDuffy7() {
  const double x[7]={-0.94910791234275852453,-0.74153118559939443986,-0.40584515137739716691,0.0,0.40584515137739716691,0.74153118559939443986,0.94910791234275852453};
  const double w[7]={0.12948496616886969327,0.27970539148927666790,0.38183005050511894495,0.41795918367346938776,0.38183005050511894495,0.27970539148927666790,0.12948496616886969327};
  std::vector<Quad> q; q.reserve(343);
  for(int ia=0;ia<7;++ia) for(int ib=0;ib<7;++ib) for(int ic=0;ic<7;++ic) {
    double A=.5*(x[ia]+1),B=.5*(x[ib]+1),C=.5*(x[ic]+1);
    double wa=.5*w[ia],wb=.5*w[ib],wc=.5*w[ic];
    double r=A,s=(1-A)*B,t=(1-A)*(1-B)*C;
    double jac=(1-A)*(1-A)*(1-B);
    q.push_back({{1-r-s-t,r,s,t},wa*wb*wc*jac});
  }
  return q;
}


static std::vector<Quad> tetDuffy5() {
  // Exact 5x5x5 collapsed-coordinate Gauss-Jacobi rule retained from the
  // earlier P1+BF3/P0 SUPG implementation.  r weights include (1-r)^2,
  // s weights include (1-s), and t is ordinary Gauss-Legendre on [0,1].
  const double rn[5]={0.034578939918215090,0.17348032077169567,0.38988638706551931,0.63433347263088680,0.85105421294701644};
  const double rw[5]={0.081764784285771011,0.12619896189991137,0.089200161221590066,0.032055600722961895,0.0041138252030990035};
  const double sn[5]={0.039809857051468722,0.19801341787360821,0.43797481024738616,0.69546427335363614,0.90146491420117358};
  const double sw[5]={0.096781590226651476,0.16717463809436969,0.14638698708466985,0.073908870072616678,0.015747914521692299};
  const double tn[5]={0.046910077030668018,0.23076534494715845,0.50000000000000000,0.76923465505284150,0.95308992296933193};
  const double tw[5]={0.11846344252809449,0.23931433524968326,0.28444444444444450,0.23931433524968326,0.11846344252809449};
  std::vector<Quad> q; q.reserve(125);
  for(int ir=0;ir<5;++ir) for(int is=0;is<5;++is) for(int it=0;it<5;++it) {
    const double r=rn[ir], ss=sn[is], t=tn[it];
    const double omr=1.0-r, oms=1.0-ss;
    q.push_back({{omr*oms*(1.0-t),r,omr*ss,omr*oms*t},rw[ir]*sw[is]*tw[it]});
  }
  return q;
}


static std::vector<Quad> tetSupg64() {
  // Positive-weight 4x4x4 collapsed Gauss-Jacobi rule.  This is the compact
  // SUPG rule previously used in the P1+BF3 lineage for the non-polynomial
  // tau(u) integrand.  It is intentionally distinct from the degree-8 Keast
  // rule below: all weights are positive, which is attractive for a
  // stabilization term whose coefficient tau is positive but non-polynomial.
  const double rn[4]={0.0485005494469972764,0.238600737551862341,0.517047295104367421,0.795851417896772828};
  const double rw[4]={0.110888415611277741,0.143458789799214448,0.0686338871729230970,0.0103522407499180812};
  const double sn[4]={0.0571041961145177246,0.276843013638123803,0.583590432368916834,0.860240135656219485};
  const double sw[4]={0.135506913431488518,0.203464568010271102,0.129847547608232333,0.0311809709500080849};
  const double tn[4]={0.0694318442029737137,0.330009478207571871,0.669990521792428129,0.930568155797026231};
  const double tw[4]={0.173927422568726897,0.326072577431273103,0.326072577431273103,0.173927422568726897};
  std::vector<Quad> q; q.reserve(64);
  for(int ir=0;ir<4;++ir) for(int is=0;is<4;++is) for(int it=0;it<4;++it) {
    const double r=rn[ir], ss=sn[is], t=tn[it];
    const double omr=1.0-r, oms=1.0-ss;
    q.push_back({{omr*oms*(1.0-t),r,omr*ss,omr*oms*t},rw[ir]*sw[is]*tw[it]});
  }
  return q;
}

static std::vector<Quad> tetKeast45() {
  // Keast degree-8 fully symmetric tetrahedral cubature.  This exactly
  // integrates total-degree <= 8 polynomials on the affine reference tet, but
  // the centroid orbit has a negative weight.  It is therefore retained as an
  // experimental SUPG choice; the positive-weight 64-point rule is the fast
  // default for the non-polynomial tau(u) integrand.
  const std::array<std::array<double,4>,7> generator{{
    {{0.250000000000000000,0.250000000000000000,0.250000000000000000,0.250000000000000000}},
    {{0.617587190300082967,0.127470936566639015,0.127470936566639015,0.127470936566639015}},
    {{0.903763508822103123,0.0320788303926322960,0.0320788303926322960,0.0320788303926322960}},
    {{0.0497770956432810185,0.0497770956432810185,0.450222904356718978,0.450222904356718978}},
    {{0.183730447398549945,0.183730447398549945,0.316269552601450060,0.316269552601450060}},
    {{0.231901089397150906,0.231901089397150906,0.0229177878448171174,0.513280033360881072}},
    {{0.0379700484718286102,0.0379700484718286102,0.730313427807538396,0.193746475248804382}},
  }};
  const std::array<double,7> orbitWeight{{
    -0.0393270066412926145,
     0.00408131605934270525,
     0.000658086773304341943,
     0.00438425882512284693,
     0.0138300638425098166,
     0.00424043742468372453,
     0.00223873973961420164,
  }};
  std::vector<Quad> q; q.reserve(45);
  for(int orbit=0;orbit<7;++orbit) {
    auto lambda=generator[orbit];
    std::sort(lambda.begin(),lambda.end());
    do { q.push_back({lambda,orbitWeight[orbit]}); }
    while(std::next_permutation(lambda.begin(),lambda.end()));
  }
  if(q.size()!=45) throw std::runtime_error("Keast45 quadrature point count mismatch");
  return q;
}

static std::vector<Quad> supgQuadrature(PetscInt n) {
  if(n==125) return tetDuffy5();
  if(n==64) return tetSupg64();
  if(n==45) return tetKeast45();
  throw std::runtime_error("SUPG quadrature must be 125, 64, or 45 points");
}

static void referenceHessian(const std::array<double,4>& l, double hess[8][3][3]) {
  const double gl[4][3]={{-1,-1,-1},{1,0,0},{0,1,0},{0,0,1}};
  for(int a=0;a<8;++a) for(int d=0;d<3;++d) for(int e=0;e<3;++e) hess[a][d][e]=0.0;
  // Exact Hessian of b_i = 27 prod_{j != i} lambda_j.  This is the same
  // ordered-pair formula used by the earlier P1+BF3/P0 SUPG kernels.
  for(int i=0;i<4;++i) {
    const int a=4+i;
    for(int d=0;d<3;++d) for(int e=0;e<3;++e) {
      double value=0.0;
      for(int j=0;j<4;++j) {
        if(j==i) continue;
        for(int k=0;k<4;++k) {
          if(k==i || k==j) continue;
          double term=gl[j][d]*gl[k][e];
          for(int m=0;m<4;++m) if(m!=i && m!=j && m!=k) term*=l[m];
          value+=term;
        }
      }
      hess[a][d][e]=27.0*value;
    }
  }
}

static double tetDiameter(const Vec3 X[4]) {
  double h=0.0;
  for(int a=0;a<4;++a) for(int b=a+1;b<4;++b) {
    const double dx=X[a].x-X[b].x,dy=X[a].y-X[b].y,dz=X[a].z-X[b].z;
    h=std::max(h,std::sqrt(dx*dx+dy*dy+dz*dz));
  }
  return h;
}

static void basis(const std::array<double,4>& l, double val[8], double gr[8][3]) {
  const double gl[4][3]={{-1,-1,-1},{1,0,0},{0,1,0},{0,0,1}};
  for(int i=0;i<4;++i){ val[i]=l[i]; for(int d=0;d<3;++d) gr[i][d]=gl[i][d]; }
  for(int i=0;i<4;++i) {
    int js[3],k=0; for(int j=0;j<4;++j) if(j!=i) js[k++]=j;
    val[4+i]=27*l[js[0]]*l[js[1]]*l[js[2]];
    for(int d=0;d<3;++d) gr[4+i][d]=0;
    for(int a=0;a<3;++a) {
      int j=js[a],o1=js[(a+1)%3],o2=js[(a+2)%3];
      for(int d=0;d<3;++d) gr[4+i][d]+=27*l[o1]*l[o2]*gl[j][d];
    }
  }
}

static void exactPressureGradient(double x,double y,double z,double gp[3]) {
  const double sx=sin(PI*x), sy=sin(PI*y), sz=sin(PI*z);
  const double cx=cos(PI*x), cy=cos(PI*y), cz=cos(PI*z);
  gp[0]=-PI*sx*cy*cz;
  gp[1]=-PI*cx*sy*cz;
  gp[2]=-PI*cx*cy*sz;
}

static void stokesForcingNu1(double x,double y,double z,double f[3]) {
  const double sx=sin(PI*x),sy=sin(PI*y),sz=sin(PI*z),cx=cos(PI*x),cy=cos(PI*y),cz=cos(PI*z);
  f[0]=PI*(24*PI*PI*sx*sx*sy*sz*sz - 4*PI*PI*sx*sx*sy - sx*cz - 4*PI*PI*sy*sz*sz)*cy;
  f[1]=PI*(-24*PI*PI*sx*sy*sy*sz*sz + 4*PI*PI*sx*sy*sy + 4*PI*PI*sx*sz*sz - sy*cz)*cx;
  f[2]=-PI*sz*cx*cy;
}

static void exactConvection(double x,double y,double z,double c[3]) {
  const double sx=sin(PI*x), sy=sin(PI*y), sz=sin(PI*z);
  const double cx=cos(PI*x), cy=cos(PI*y);
  const double fac=4.0*PI*PI*PI*sz*sz*sz*sz;
  c[0]=fac*sx*sx*sx*sy*sy*cx;
  c[1]=fac*sx*sx*sy*sy*sy*cy;
  c[2]=0.0;
}

static void forcing(double x,double y,double z,double nu,bool centralConvection,double f[3]) {
  double fs[3],gp[3],cv[3]={0,0,0};
  stokesForcingNu1(x,y,z,fs);
  exactPressureGradient(x,y,z,gp);
  if(centralConvection) exactConvection(x,y,z,cv);
  // fs = -Delta(u_exact) + grad(p_exact). General viscosity gives
  // f = -nu Delta(u_exact) + (u_exact.grad)u_exact + grad(p_exact).
  for(int d=0; d<3; ++d) f[d]=nu*(fs[d]-gp[d]) + gp[d] + cv[d];
}

static void exactU(double x,double y,double z,double u[3]) {
  double sx=sin(PI*x),sy=sin(PI*y),sz=sin(PI*z);
  u[0]=PI*sx*sx*sin(2*PI*y)*sz*sz;
  u[1]=-PI*sin(2*PI*x)*sy*sy*sz*sz;
  u[2]=0;
}
static double exactP(double x,double y,double z) { return cos(PI*x)*cos(PI*y)*cos(PI*z); }

enum class ProblemMode { MMS, Pipe, Flow };
enum class InletBCMode { PipeParabolic, PipeOneSeventh, FixedNormalSpeed, DgNumericalTrace };

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

