#include <petscksp.h>
#include <petscdmplex.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
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
  std::array<std::vector<double>,3> dirValue;
  PetscInt ns=0, freeVertices=0, freeFaces=0, fixedVertices=0, fixedFaces=0;
};

static void prepareBoundaryData(const Mesh& M,const ProblemConfig& P,Discrete& D) {
  const PetscInt nv=(PetscInt)M.points.size(), nf=(PetscInt)M.faces.size(), ni=(PetscInt)M.neighbour.size();
  D.fixedEntity.assign(nv+nf,0);
  for(int d=0;d<3;++d) D.dirValue[d].assign(nv+nf,0.0);

  if(P.mode==ProblemMode::MMS) {
    for(PetscInt f=ni;f<nf;++f) {
      D.fixedEntity[nv+f]=1;
      for(int v:M.faces[f].v) D.fixedEntity[v]=1;
    }
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
  for(PetscInt v=0;v<nv;++v) {
    if(onWall[v] || onInlet[v]) D.fixedEntity[v]=1;
    // Preserve the existing wall-priority convention at inlet/wall edge vertices.
    // BF3 below restores the requested inlet face mean exactly despite that corner incompatibility.
    if(onInlet[v] && !onWall[v]) {
      if(P.inletBC==InletBCMode::PipeParabolic) D.dirValue[2][v]=pipeBoundaryUz(P.pipe,M.points[v].x,M.points[v].y);
      else { D.dirValue[0][v]=B.inletVelocity.x; D.dirValue[1][v]=B.inletVelocity.y; D.dirValue[2][v]=B.inletVelocity.z; }
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
      const double vertexMean=(D.dirValue[d][F.v[0]]+D.dirValue[d][F.v[1]]+D.dirValue[d][F.v[2]])/3.0;
      D.dirValue[d][nv+f]=(20.0/9.0)*(exactMean[d]-vertexMean);
    }
  }
}

static double entityDirValue(const Discrete& D,int d,PetscInt entity) {
  return D.dirValue[d][entity];
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
struct CustomMomentumCSR {
  PetscInt rstart=0,rend=0,nOwned=0;
  std::vector<PetscInt> rowPtr;
  std::vector<PetscInt> colGid;
  std::vector<PetscInt> colLocal; // [0,nOwned) owned, [nOwned,...) custom ghosts
  std::vector<PetscInt> diagPos;

  std::vector<PetscInt> ghostGid; // grouped by owning rank because global IDs are rank-contiguous
  std::vector<int> reqSendCounts,reqSendDispls,reqRecvCounts,reqRecvDispls;
  std::vector<PetscInt> reqRecvGid,reqRecvLocalOffset;
  std::vector<double> exchangeSend,ghostValues;
  std::vector<MPI_Request> exchangeRequests;

  // Final M2B momentum storage: one immutable static diffusion value array plus
  // one active numeric array. aRel contains the physical operator during direct
  // assembly and becomes the relaxed operator after finalizeRelaxation().
  std::vector<double> kNu,aRel;
  // Legacy M1/M2A shadow arrays are kept as empty fields so old diagnostic helpers
  // remain source-compatible; they consume no retained numeric payload in M2B.
  std::vector<double> convection,supg,aPhys;
  std::vector<double> physDiag,relaxDelta,relaxedDiag,metric,rAU;

  // Direct dynamic assembly gathers the three velocity components through the
  // same peer-only custom halo used by MatVec/SGS and accumulates owned-row RHS.
  std::array<std::vector<double>,3> fieldOwned,fieldGhost,convRhs,supgRhs;

  // Reused live-solver work buffers. Keep the custom path allocation-free.
  std::vector<double> workB,workX,workY,workBeff;
};

static int customMomentumOwnerOfGid(const std::vector<PetscInt>& off,PetscInt gid) {
  auto it=std::upper_bound(off.begin(),off.end(),gid);
  if(it==off.begin() || (it==off.end() && gid>=off.back())) return -1;
  return (int)((it-off.begin())-1);
}

static PetscErrorCode buildCustomMomentumCSR(const Mesh& M,const Discrete& D,int rank,CustomMomentumCSR& A) {
  PetscFunctionBeginUser;
  const PetscInt nv=(PetscInt)M.points.size();
  std::vector<PetscInt> off(D.velCount.size()+1,0);
  for(std::size_t r=0;r<D.velCount.size();++r) off[r+1]=off[r]+D.velCount[r];
  A.rstart=off[(std::size_t)rank]; A.rend=off[(std::size_t)rank+1]; A.nOwned=A.rend-A.rstart;

  std::vector<std::vector<PetscInt>> rows((std::size_t)A.nOwned);
  // Global mesh is still replicated in M1.  Evaluate row support from every tet
  // touching an owned row.  This exactly models the owned-row + halo assembly
  // that will remain after mesh localization.
  for(PetscInt c=0;c<(PetscInt)M.tets.size();++c) {
    PetscInt entity[8],gid[8];
    for(int i=0;i<4;++i) entity[i]=M.tets[(std::size_t)c][i];
    for(int i=0;i<4;++i) entity[4+i]=nv+M.oppFace[(std::size_t)c][i];
    for(int i=0;i<8;++i) gid[i]=D.g2free[(std::size_t)entity[i]];
    for(int a=0;a<8;++a) if(gid[a]>=A.rstart && gid[a]<A.rend) {
      auto& rr=rows[(std::size_t)(gid[a]-A.rstart)];
      for(int b=0;b<8;++b) if(gid[b]>=0) rr.push_back(gid[b]);
    }
  }

  A.rowPtr.assign((std::size_t)A.nOwned+1,0);
  for(PetscInt i=0;i<A.nOwned;++i) {
    auto& rr=rows[(std::size_t)i];
    std::sort(rr.begin(),rr.end()); rr.erase(std::unique(rr.begin(),rr.end()),rr.end());
    const PetscInt diag=A.rstart+i;
    if(!std::binary_search(rr.begin(),rr.end(),diag))
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom momentum CSR row is missing its diagonal");
    A.rowPtr[(std::size_t)i+1]=A.rowPtr[(std::size_t)i]+(PetscInt)rr.size();
  }
  A.colGid.resize((std::size_t)A.rowPtr.back());
  A.diagPos.assign((std::size_t)A.nOwned,-1);
  for(PetscInt i=0;i<A.nOwned;++i) {
    const auto& rr=rows[(std::size_t)i];
    PetscInt p=A.rowPtr[(std::size_t)i];
    for(PetscInt k=0;k<(PetscInt)rr.size();++k) {
      A.colGid[(std::size_t)(p+k)]=rr[(std::size_t)k];
      if(rr[(std::size_t)k]==A.rstart+i) A.diagPos[(std::size_t)i]=p+k;
    }
  }

  A.ghostGid.clear();
  for(PetscInt g:A.colGid) if(g<A.rstart || g>=A.rend) A.ghostGid.push_back(g);
  std::sort(A.ghostGid.begin(),A.ghostGid.end());
  A.ghostGid.erase(std::unique(A.ghostGid.begin(),A.ghostGid.end()),A.ghostGid.end());

  const int nr=(int)D.velCount.size();
  A.reqSendCounts.assign((std::size_t)nr,0); A.reqRecvCounts.assign((std::size_t)nr,0);
  for(PetscInt g:A.ghostGid) {
    const int owner=customMomentumOwnerOfGid(off,g);
    if(owner<0 || owner>=nr) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"invalid owner for custom momentum ghost gid");
    ++A.reqSendCounts[(std::size_t)owner];
  }
  A.reqSendDispls.assign((std::size_t)nr,0); A.reqRecvDispls.assign((std::size_t)nr,0);
  for(int r=1;r<nr;++r) A.reqSendDispls[(std::size_t)r]=A.reqSendDispls[(std::size_t)r-1]+A.reqSendCounts[(std::size_t)r-1];
  PetscCallMPI(MPI_Alltoall(A.reqSendCounts.data(),1,MPI_INT,A.reqRecvCounts.data(),1,MPI_INT,PETSC_COMM_WORLD));
  for(int r=1;r<nr;++r) A.reqRecvDispls[(std::size_t)r]=A.reqRecvDispls[(std::size_t)r-1]+A.reqRecvCounts[(std::size_t)r-1];
  int nRecvReq=0; for(int v:A.reqRecvCounts) nRecvReq+=v;
  A.reqRecvGid.assign((std::size_t)nRecvReq,-1);
  PetscCallMPI(MPI_Alltoallv(A.ghostGid.empty()?nullptr:A.ghostGid.data(),A.reqSendCounts.data(),A.reqSendDispls.data(),MPIU_INT,
                             A.reqRecvGid.empty()?nullptr:A.reqRecvGid.data(),A.reqRecvCounts.data(),A.reqRecvDispls.data(),MPIU_INT,PETSC_COMM_WORLD));
  A.reqRecvLocalOffset.resize(A.reqRecvGid.size());
  for(std::size_t k=0;k<A.reqRecvGid.size();++k) {
    const PetscInt g=A.reqRecvGid[k];
    if(g<A.rstart || g>=A.rend) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom momentum peer requested a non-owned gid");
    A.reqRecvLocalOffset[k]=g-A.rstart;
  }

  A.colLocal.resize(A.colGid.size());
  for(std::size_t k=0;k<A.colGid.size();++k) {
    const PetscInt g=A.colGid[k];
    if(g>=A.rstart && g<A.rend) A.colLocal[k]=g-A.rstart;
    else {
      auto it=std::lower_bound(A.ghostGid.begin(),A.ghostGid.end(),g);
      if(it==A.ghostGid.end() || *it!=g) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"custom momentum ghost lookup failed");
      A.colLocal[k]=A.nOwned+(PetscInt)(it-A.ghostGid.begin());
    }
  }

  const std::size_t nnz=A.colGid.size();
  A.kNu.assign(nnz,0.0); A.aRel.assign(nnz,0.0);
  A.convection.clear(); A.supg.clear(); A.aPhys.clear();
  A.physDiag.assign((std::size_t)A.nOwned,0.0); A.relaxDelta.assign((std::size_t)A.nOwned,0.0);
  A.relaxedDiag.assign((std::size_t)A.nOwned,0.0); A.metric.assign((std::size_t)A.nOwned,0.0);
  A.rAU.assign((std::size_t)A.nOwned,0.0);
  for(int d=0;d<3;++d) {
    A.fieldOwned[d].assign((std::size_t)A.nOwned,0.0);
    A.fieldGhost[d].assign(A.ghostGid.size(),0.0);
    A.convRhs[d].assign((std::size_t)A.nOwned,0.0);
    A.supgRhs[d].assign((std::size_t)A.nOwned,0.0);
  }
  A.workB.assign((std::size_t)A.nOwned,0.0); A.workX.assign((std::size_t)A.nOwned,0.0);
  A.workY.assign((std::size_t)A.nOwned,0.0); A.workBeff.assign((std::size_t)A.nOwned,0.0);
  A.exchangeSend.assign(A.reqRecvGid.size(),0.0); A.ghostValues.assign(A.ghostGid.size(),0.0);
  int peerOps=0; for(int r=0;r<nr;++r) { if(A.reqSendCounts[(std::size_t)r]>0) ++peerOps; if(A.reqRecvCounts[(std::size_t)r]>0) ++peerOps; }
  A.exchangeRequests.resize((std::size_t)peerOps);

  unsigned long long localNnz=(unsigned long long)nnz,globalNnz=0;
  PetscCallMPI(MPI_Allreduce(&localNnz,&globalNnz,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  PetscInt lg=(PetscInt)A.ghostGid.size(),gmin=0,gmax=0;
  PetscCallMPI(MPI_Allreduce(&lg,&gmin,1,MPIU_INT,MPI_MIN,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&lg,&gmax,1,MPIU_INT,MPI_MAX,PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_CUSTOM_MOM_CSR rows=%" PetscInt_FMT " globalNnz=%llu avgNnzPerRow=%.6f customGhostMin=%" PetscInt_FMT " customGhostMax=%" PetscInt_FMT " semantics=owned_rows_all_incident_tets_custom_MPI_exchange\n",
    D.ns,globalNnz,D.ns?((double)globalNnz/(double)D.ns):0.0,gmin,gmax));
  unsigned long long localBytes=0,globalBytes=0;
  localBytes += (unsigned long long)(A.rowPtr.capacity()+A.colGid.capacity()+A.colLocal.capacity()+A.diagPos.capacity()+A.ghostGid.capacity()+A.reqRecvGid.capacity()+A.reqRecvLocalOffset.capacity())*sizeof(PetscInt);
  localBytes += (unsigned long long)(A.reqSendCounts.capacity()+A.reqSendDispls.capacity()+A.reqRecvCounts.capacity()+A.reqRecvDispls.capacity())*sizeof(int);
  localBytes += (unsigned long long)(A.exchangeSend.capacity()+A.ghostValues.capacity()+A.kNu.capacity()+A.aRel.capacity()+A.physDiag.capacity()+A.relaxDelta.capacity()+A.relaxedDiag.capacity()+A.metric.capacity()+A.rAU.capacity()+A.workB.capacity()+A.workX.capacity()+A.workY.capacity()+A.workBeff.capacity())*sizeof(double);
  for(int d=0;d<3;++d) localBytes += (unsigned long long)(A.fieldOwned[d].capacity()+A.fieldGhost[d].capacity()+A.convRhs[d].capacity()+A.supgRhs[d].capacity())*sizeof(double);
  PetscCallMPI(MPI_Allreduce(&localBytes,&globalBytes,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_CUSTOM_MOM_MEMORY summedRetainedMiB=%.3f bytesPerGlobalVelDof=%.3f note=M3A_staticK_plus_activeA_direct_static_and_dynamic_no_PETSc_momentum_matrices\n",
    (double)globalBytes/(1024.0*1024.0),D.ns?((double)globalBytes/(double)D.ns):0.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}


// -----------------------------------------------------------------------------
// M4A custom FP64 pressure B/B^T shadow architecture
// -----------------------------------------------------------------------------
// The final mixed-precision design cannot leave the physical divergence/gradient
// operators in PETSc, because a single-precision PETSc build would then make B
// itself single precision.  M4A therefore builds genuine custom MPI B and B^T
// actions now, while the validated FP64 PETSc pressure path remains live.
//
// B x:       owned pressure rows, velocity owned+halo input.
// B^T p:     owned velocity rows, all incident pressure cells, pressure halo.
// S x:       B [ rAU .* (B^T x) ], entirely custom FP64 arithmetic.
//
// No global sparse insertion is used.  Runtime communication is peer-only; the
// one-time MPI_Alltoall[v] below is only for discovering which GIDs each peer
// needs, exactly as in the custom momentum halo.
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
  double vol=0.0,gradLambda[4][3]={{0}};
};
struct CustomPressureTransposeCell {
  PetscInt pLocal=-1;
  PetscInt nOwnedVel=0;
  std::uint8_t basis[8]={0};
  PetscInt velOwnedLocal[8]={0};
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

static PetscErrorCode buildCustomPressureBPlan(const Mesh& M,const Discrete& D,int rank,CustomPressureBPlan& P) {
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
    fillCustomPressureGeom(M,c,cp.vol,cp.gradLambda); P.forwardCells.push_back(cp);
  }
  PetscCall(buildCustomPeerHalo(D.velCount,rank,std::move(neededVel),P.velocityHalo,"velocity_for_B"));
  // Fill velocity local indices only after the halo is finalized.
  for(PetscInt c=0;c<nc;++c) if(D.cellOwner[(std::size_t)c]==rank) {
    auto& cp=P.forwardCells[(std::size_t)(D.pGid[(std::size_t)c]-pStart)];
    PetscInt ent[8]; for(int i=0;i<4;++i)ent[i]=M.tets[(std::size_t)c][i]; for(int i=0;i<4;++i)ent[4+i]=nv+M.oppFace[(std::size_t)c][i];
    for(int a=0;a<8;++a) { const PetscInt g=D.g2free[(std::size_t)ent[a]]; cp.velLocal[a]=(g>=0)?customPeerLocalIndex(P.velocityHalo,g):-1; if(g>=0 && cp.velLocal[a]<0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M4A B forward halo missing velocity gid"); }
  }

  P.transposeCells.clear();
  for(PetscInt c=0;c<nc;++c) {
    PetscInt ent[8],gid[8]; for(int i=0;i<4;++i)ent[i]=M.tets[(std::size_t)c][i]; for(int i=0;i<4;++i)ent[4+i]=nv+M.oppFace[(std::size_t)c][i];
    CustomPressureTransposeCell cp;
    for(int a=0;a<8;++a) { gid[a]=D.g2free[(std::size_t)ent[a]]; if(gid[a]>=vStart && gid[a]<vEnd) { cp.basis[cp.nOwnedVel]=(std::uint8_t)a; cp.velOwnedLocal[cp.nOwnedVel]=gid[a]-vStart; ++cp.nOwnedVel; } }
    if(cp.nOwnedVel==0) continue;
    const PetscInt pg=D.pGid[(std::size_t)c]; if(pg<pStart || pg>=pEnd) neededP.push_back(pg);
    fillCustomPressureGeom(M,c,cp.vol,cp.gradLambda); P.transposeCells.push_back(cp);
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
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_M4B_B_PLAN pressureRows=%llu transposeSupportCells=%llu transposeSupportReplication=%.6f retainedMiB=%.3f semantics=custom_FP64_owned_rows_peer_halo\n",gf,gt,M.tets.empty()?0.0:(double)gt/(double)M.tets.size(),(double)gb/(1024.0*1024.0)));
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
  for(const auto& cp:P.forwardCells) { double s=0.0; for(int a=0;a<8;++a) if(cp.velLocal[a]>=0) s+=customPressureBCoeff(cp.vol,cp.gradLambda,d,a)*customPeerValue(P.velocityHalo,xOwned,cp.velLocal[a]); yOwned[(std::size_t)cp.pLocal]=s; }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customPressureBtApply(CustomPressureBPlan& P,int d,const std::vector<double>& pOwned,std::vector<double>& yOwned) {
  PetscFunctionBeginUser; PetscCall(customPeerExchange(P.pressureHalo,pOwned,48211+d));
  yOwned.assign((std::size_t)P.velocityHalo.nOwned,0.0);
  for(const auto& cp:P.transposeCells) { const double pv=customPeerValue(P.pressureHalo,pOwned,cp.pLocal); for(PetscInt j=0;j<cp.nOwnedVel;++j) { const int a=(int)cp.basis[j]; yOwned[(std::size_t)cp.velOwnedLocal[j]]+=customPressureBCoeff(cp.vol,cp.gradLambda,d,a)*pv; } }
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
  const bool ok=n.rel<=tol; PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_M4A_SCHUR_PARITY SRel=%.3e SMaxAbs=%.3e tol=%.3e arithmetic=custom_FP64_B_rAU_Bt reference=PETSc_FP64_factored status=%s\n",n.rel,n.maxAbs,tol,ok?"PASS":"CHECK"));
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

static PetscErrorCode customMomentumResetPhysical(CustomMomentumCSR& A) {
  PetscFunctionBeginUser;
  A.aRel=A.kNu;
  for(int d=0;d<3;++d) { std::fill(A.convRhs[d].begin(),A.convRhs[d].end(),0.0); std::fill(A.supgRhs[d].begin(),A.supgRhs[d].end(),0.0); }
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
    } else if(rauMode=="diag") use=A.relaxedDiag[(std::size_t)i];
    else { use=0.0; for(PetscInt k=A.rowPtr[(std::size_t)i];k<A.rowPtr[(std::size_t)i+1];++k) use+=std::abs(A.aRel[(std::size_t)k]); }
    if(!(use>0.0) || !std::isfinite(use)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"invalid custom momentum correction metric");
    A.metric[(std::size_t)i]=use; A.rAU[(std::size_t)i]=rauScale/use;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customMomentumMatVec(CustomMomentumCSR& A,const std::vector<double>& vals,const std::vector<double>& xOwned,std::vector<double>& yOwned) {
  PetscFunctionBeginUser;
  if(vals.size()!=A.colGid.size()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"custom momentum matrix value size mismatch");
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

static PetscErrorCode customMomentumDerive(CustomMomentumCSR& A,bool centralConvection,bool implicitSupg,
  const std::string& uRelaxMode,double alphaU,const std::string& simpleVariant,const std::string& rauMode,
  double simplecBlend,double floorFraction,const std::string& fallback,double rauScale) {
  PetscFunctionBeginUser;
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
    } else if(rauMode=="diag") use=A.relaxedDiag[(std::size_t)i];
    else {
      use=0.0; for(PetscInt k=A.rowPtr[(std::size_t)i];k<A.rowPtr[(std::size_t)i+1];++k) use+=std::abs(A.aRel[(std::size_t)k]);
    }
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

static PetscErrorCode customMomentumSymmetricGS(CustomMomentumCSR& A,const std::vector<double>& b,std::vector<double>& x,double omega,PetscInt localSweeps) {
  PetscFunctionBeginUser;
  if((PetscInt)b.size()!=A.nOwned || (PetscInt)x.size()!=A.nOwned) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"custom SGS vector size mismatch");
  auto& beff=A.workBeff;
  if((PetscInt)beff.size()!=A.nOwned) beff.assign((std::size_t)A.nOwned,0.0);
  // MPIAIJ local-SOR semantics: freeze the off-rank values for this outer SOR
  // application, form the effective local RHS once, then perform the requested
  // number of local symmetric sweeps on the owned-owned block.
  PetscCall(customMomentumExchange(A,x));
  for(PetscInt i=0;i<A.nOwned;++i) {
    double v=b[(std::size_t)i];
    for(PetscInt k=A.rowPtr[(std::size_t)i];k<A.rowPtr[(std::size_t)i+1];++k) {
      const PetscInt li=A.colLocal[(std::size_t)k]; if(li>=A.nOwned) v-=A.aRel[(std::size_t)k]*A.ghostValues[(std::size_t)(li-A.nOwned)];
    }
    beff[(std::size_t)i]=v;
  }
  for(PetscInt sweep=0;sweep<localSweeps;++sweep) {
    for(PetscInt i=0;i<A.nOwned;++i) {
      const double old=x[(std::size_t)i],diag=A.aRel[(std::size_t)A.diagPos[(std::size_t)i]]; double rhs=beff[(std::size_t)i];
      for(PetscInt k=A.rowPtr[(std::size_t)i];k<A.rowPtr[(std::size_t)i+1];++k) {
        const PetscInt li=A.colLocal[(std::size_t)k]; if(li<A.nOwned && li!=i) rhs-=A.aRel[(std::size_t)k]*x[(std::size_t)li];
      }
      x[(std::size_t)i]=(1.0-omega)*old+omega*(rhs/diag);
    }
    for(PetscInt ii=A.nOwned;ii>0;--ii) {
      const PetscInt i=ii-1; const double old=x[(std::size_t)i],diag=A.aRel[(std::size_t)A.diagPos[(std::size_t)i]]; double rhs=beff[(std::size_t)i];
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

static PetscErrorCode customMomentumResidualNorm(CustomMomentumCSR& A,const std::vector<double>& b,const std::vector<double>& x,double& rn) {
  PetscFunctionBeginUser;
  PetscCall(customMomentumMatVec(A,A.aRel,x,A.workY));
  double local=0.0;
  for(PetscInt i=0;i<A.nOwned;++i) { const double r=b[(std::size_t)i]-A.workY[(std::size_t)i]; local+=r*r; }
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
  PetscInt maxIts,PetscInt checkEvery,double omega,PetscInt localSweeps,PetscInt *parallelIts,double *relres) {
  PetscFunctionBeginUser;
  if((PetscInt)b.size()!=A.nOwned || (PetscInt)x.size()!=A.nOwned) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"M6B native momentum state size mismatch");
  double bn=0.0,rn=0.0,rnInitial=-1.0; PetscCall(customMomentumNorm2(b,bn));
  if(!std::isfinite(bn)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"M6B native momentum RHS norm NaN/Inf");
  if(bn==0.0) bn=1.0;
  PetscCall(customMomentumResidualNorm(A,b,x,rn)); rnInitial=rn; if(!std::isfinite(rnInitial)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"M6B native initial momentum residual NaN/Inf");
  const double target=std::max((double)atol,(double)rtol*bn);
  if(rnInitial<=target){*parallelIts=0;*relres=rnInitial/bn;PetscFunctionReturn(PETSC_SUCCESS);}
  PetscInt it=0; while(it<maxIts){ const PetscInt chunk=PetscMin(checkEvery,maxIts-it); for(PetscInt q=0;q<chunk;++q) PetscCall(customMomentumSymmetricGS(A,b,x,omega,localSweeps)); it+=chunk;
    PetscCall(customMomentumResidualNorm(A,b,x,rn)); if(!std::isfinite(rn)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"M6B native SGS NaN/Inf");
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
struct CustomDynamicCellPlan {
  PetscInt cell=-1;
  PetscInt entity[8]={0,0,0,0,0,0,0,0};
  PetscInt gid[8]={-1,-1,-1,-1,-1,-1,-1,-1};
  PetscInt localIndex[8]={-1,-1,-1,-1,-1,-1,-1,-1};
  PetscInt ownedBasis[8]={0,0,0,0,0,0,0,0};
  PetscInt nOwnedRows=0;
  // Per-cell row-local CSR slots: 255 means fixed/inactive.  A P1+BF3 cell has
  // at most eight trial columns, so one byte is sufficient and removes all
  // sparse-column searches from the nonlinear assembly hot path.
  std::uint8_t rowSlot[8][8]={{0}};
  double det=0.0,invJ[3][3]={{0}},gradLambda[4][3]={{0}},h2=0.0;
};

struct CustomDynamicAssemblyPlan {
  std::vector<CustomDynamicCellPlan> cells;
  PetscInt nQ=0;
  std::vector<SupgReferencePoint> ref;
  std::vector<double> forcing;
  PetscBool forcingZero=PETSC_TRUE;
  double nu=0.0;
};

static PetscErrorCode buildCustomDynamicAssemblyPlan(const Mesh& M,const Discrete& D,const ProblemConfig& problem,
  const CustomMomentumCSR& A,PetscInt nQ,CustomDynamicAssemblyPlan& P) {
  PetscFunctionBeginUser;
  const PetscInt nv=(PetscInt)M.points.size();
  const auto Q=supgQuadrature(nQ);
  P.nQ=(PetscInt)Q.size(); P.nu=problem.nu; P.ref.resize(Q.size());
  double wsum=0.0,wabs=0.0,wmin=1e300,wmax=-1e300;
  for(std::size_t q=0;q<Q.size();++q) {
    P.ref[q].lam=Q[q].lam; P.ref[q].w=Q[q].w;
    basis(Q[q].lam,P.ref[q].phi,P.ref[q].gradRef); referenceHessian(Q[q].lam,P.ref[q].hessRef);
    wsum+=Q[q].w; wabs+=std::abs(Q[q].w); wmin=std::min(wmin,Q[q].w); wmax=std::max(wmax,Q[q].w);
  }
  P.cells.clear();
  // Row-support halo: include a tet on this rank iff at least one of its free
  // scalar velocity rows is owned here. Each global row is therefore assembled
  // exactly once, while partition-boundary cells may be evaluated by two ranks.
  for(PetscInt c=0;c<(PetscInt)M.tets.size();++c) {
    CustomDynamicCellPlan cp; cp.cell=c;
    for(int a=0;a<8;++a) for(int b=0;b<8;++b) cp.rowSlot[a][b]=255;
    const auto t=M.tets[(std::size_t)c];
    for(int i=0;i<4;++i) cp.entity[i]=t[i];
    for(int i=0;i<4;++i) cp.entity[4+i]=nv+M.oppFace[(std::size_t)c][i];
    for(int a=0;a<8;++a) {
      cp.gid[a]=D.g2free[(std::size_t)cp.entity[a]];
      if(cp.gid[a]>=A.rstart && cp.gid[a]<A.rend) cp.ownedBasis[cp.nOwnedRows++]=a;
    }
    if(cp.nOwnedRows==0) continue;
    for(int a=0;a<8;++a) if(cp.gid[a]>=0) {
      cp.localIndex[a]=customMomentumLocalIndex(A,cp.gid[a]);
      if(cp.localIndex[a]<0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M3A dynamic plan free gid absent from custom halo");
    }
    for(PetscInt ii=0;ii<cp.nOwnedRows;++ii) {
      const int a=(int)cp.ownedBasis[ii]; const PetscInt lr=cp.gid[a]-A.rstart;
      const PetscInt rb=A.rowPtr[(std::size_t)lr],re=A.rowPtr[(std::size_t)lr+1];
      if(re-rb>=255) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M3A one-byte cell CSR slot overflow");
      for(int b=0;b<8;++b) if(cp.gid[b]>=0) {
        auto it=std::lower_bound(A.colGid.begin()+rb,A.colGid.begin()+re,cp.gid[b]);
        if(it==A.colGid.begin()+re || *it!=cp.gid[b]) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M3A dynamic plan missing CSR column");
        cp.rowSlot[a][b]=(std::uint8_t)(it-(A.colGid.begin()+rb));
      }
    }
    const Vec3 X[4]={M.points[t[0]],M.points[t[1]],M.points[t[2]],M.points[t[3]]};
    double J[3][3]={{X[1].x-X[0].x,X[2].x-X[0].x,X[3].x-X[0].x},
                    {X[1].y-X[0].y,X[2].y-X[0].y,X[3].y-X[0].y},
                    {X[1].z-X[0].z,X[2].z-X[0].z,X[3].z-X[0].z}};
    cp.det=det3(J); if(cp.det<=0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"non-positive tet in M3A dynamic plan");
    inv3(J,cp.invJ);
    const double gradLambdaRef[4][3]={{-1,-1,-1},{1,0,0},{0,1,0},{0,0,1}};
    for(int i=0;i<4;++i) for(int d=0;d<3;++d) for(int j=0;j<3;++j) cp.gradLambda[i][d]+=gradLambdaRef[i][j]*cp.invJ[j][d];
    const double h=tetDiameter(X); cp.h2=h*h; if(!(cp.h2>0)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"zero tet diameter in M2B plan");
    P.cells.push_back(cp);
  }
  P.forcingZero=(problem.mode==ProblemMode::MMS)?PETSC_FALSE:PETSC_TRUE;
  if(!P.forcingZero) {
    P.forcing.assign(P.cells.size()*Q.size()*3,0.0);
    for(std::size_t ic=0;ic<P.cells.size();++ic) {
      const auto t=M.tets[(std::size_t)P.cells[ic].cell];
      const Vec3 X[4]={M.points[t[0]],M.points[t[1]],M.points[t[2]],M.points[t[3]]};
      for(std::size_t q=0;q<Q.size();++q) {
        double x=0,y=0,z=0; for(int i=0;i<4;++i){x+=P.ref[q].lam[i]*X[i].x;y+=P.ref[q].lam[i]*X[i].y;z+=P.ref[q].lam[i]*X[i].z;}
        double f[3]; problemForcing(problem,x,y,z,f); const std::size_t iq=(ic*Q.size()+q)*3;
        for(int d=0;d<3;++d) P.forcing[iq+d]=f[d];
      }
    }
  } else P.forcing.clear();
  unsigned long long lc=(unsigned long long)P.cells.size(),gc=0,lrows=0,grows=0;
  for(const auto& cp:P.cells) lrows+=(unsigned long long)cp.nOwnedRows;
  PetscCallMPI(MPI_Allreduce(&lc,&gc,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&lrows,&grows,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_M3A_DYNAMIC_PLAN summedRowSupportCells=%llu globalCells=%zu supportReplication=%.6f ownedRowIncidences=%llu quadraturePoints=%" PetscInt_FMT " matrixAssembly=direct_custom_owned_rows no_offrank_insertion=1 weightSum=%.16e weightAbsSum=%.16e weightRange=[%.6e,%.6e]\n",
    gc,M.tets.size(),M.tets.empty()?0.0:(double)gc/(double)M.tets.size(),grows,P.nQ,wsum,wabs,wmin,wmax));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode assembleStaticDiffusionCustom(const CustomDynamicAssemblyPlan& P,CustomMomentumCSR& A) {
  PetscFunctionBeginUser;
  std::fill(A.kNu.begin(),A.kNu.end(),0.0);
  // The old FE startup used the collapsed 5^3 rule for non-MMS flow.  Degree 4
  // is sufficient for all P1+BF3 diffusion products on affine tetrahedra, so
  // this is an exact one-time construction of the same scalar K topology.
  const auto Q=tetDuffy5();
  for(const auto& cp:P.cells) {
    double Kl[8][8]={{0}};
    for(const auto& q:Q) {
      double val[8],grr[8][3],gr[8][3]; basis(q.lam,val,grr);
      for(int a=0;a<8;++a) for(int d=0;d<3;++d) {
        gr[a][d]=0.0;
        for(int j=0;j<3;++j) gr[a][d]+=grr[a][j]*cp.invJ[j][d];
      }
      const double w=q.w*cp.det;
      for(int a=0;a<8;++a) for(int b=0;b<8;++b) {
        double dot=0.0; for(int d=0;d<3;++d) dot+=gr[a][d]*gr[b][d];
        Kl[a][b]+=dot*w;
      }
    }
    for(PetscInt ii=0;ii<cp.nOwnedRows;++ii) {
      const int a=(int)cp.ownedBasis[ii];
      const PetscInt lr=cp.gid[a]-A.rstart;
      for(int b=0;b<8;++b) if(cp.gid[b]>=0) {
        const std::uint8_t slot=cp.rowSlot[a][b];
        if(slot==255) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M3A static diffusion missing precomputed CSR slot");
        A.kNu[(std::size_t)(A.rowPtr[(std::size_t)lr]+slot)]+=Kl[a][b];
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode assembleStaticMomentumRhsNative(const Mesh& M,const Discrete& D,const ProblemConfig& problem,const CustomDynamicAssemblyPlan& P,CustomMomentumCSR& A,std::array<std::vector<double>,3>& rhs) {
  PetscFunctionBeginUser;
  for(int d=0;d<3;++d) rhs[(std::size_t)d].assign((std::size_t)A.nOwned,0.0);
  const auto Q=(problem.mode!=ProblemMode::MMS)?tetDuffy5():tetDuffy7();
  for(const auto& cp:P.cells){
    const auto t=M.tets[(std::size_t)cp.cell]; const Vec3 X[4]={M.points[t[0]],M.points[t[1]],M.points[t[2]],M.points[t[3]]};
    double Al[8][8]={{0}},fl[3][8]={{0}};
    for(const auto& q:Q){ double val[8],grr[8][3],gr[8][3]; basis(q.lam,val,grr);
      for(int a=0;a<8;++a) for(int d=0;d<3;++d){gr[a][d]=0.0;for(int j=0;j<3;++j)gr[a][d]+=grr[a][j]*cp.invJ[j][d];}
      double x=0,y=0,z=0;for(int i=0;i<4;++i){x+=q.lam[i]*X[i].x;y+=q.lam[i]*X[i].y;z+=q.lam[i]*X[i].z;} double ff[3];problemForcing(problem,x,y,z,ff);const double w=q.w*cp.det;
      for(int a=0;a<8;++a){for(int d=0;d<3;++d)fl[d][a]+=ff[d]*val[a]*w;for(int b=0;b<8;++b){double dot=0;for(int d=0;d<3;++d)dot+=gr[a][d]*gr[b][d];Al[a][b]+=dot*w;}}}
    for(PetscInt ii=0;ii<cp.nOwnedRows;++ii){const int a=(int)cp.ownedBasis[(std::size_t)ii];const PetscInt lr=cp.gid[a]-A.rstart;
      for(int d=0;d<3;++d) rhs[(std::size_t)d][(std::size_t)lr]+=fl[d][a];
      for(int b=0;b<8;++b) if(cp.gid[b]<0) for(int d=0;d<3;++d){const double ud=entityDirValue(D,d,cp.entity[b]);if(ud!=0.0)rhs[(std::size_t)d][(std::size_t)lr]-=problem.nu*Al[a][b]*ud;}
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode customMomentumAddEntry(CustomMomentumCSR& A,PetscInt rowGid,PetscInt colGid,double v) {
  PetscFunctionBeginUser;
  if(rowGid<A.rstart || rowGid>=A.rend) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"M3A attempted non-owned row insertion");
  const PetscInt i=rowGid-A.rstart,b=A.rowPtr[(std::size_t)i],e=A.rowPtr[(std::size_t)i+1];
  auto it=std::lower_bound(A.colGid.begin()+b,A.colGid.begin()+e,colGid);
  if(it==A.colGid.begin()+e || *it!=colGid) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M3A custom CSR missing element column");
  A.aRel[(std::size_t)(it-A.colGid.begin())]+=v;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode assembleCentralConvectionCustom(const Discrete& D,const CustomDynamicAssemblyPlan& P,CustomMomentumCSR& A) {
  PetscFunctionBeginUser;
  const auto& T=centralTensor();
  for(auto& r:A.convRhs) std::fill(r.begin(),r.end(),0.0);
  for(const auto& cp:P.cells) {
    double coeff[3][8]={{0}};
    for(int m=0;m<8;++m) {
      if(cp.gid[m]>=0) for(int d=0;d<3;++d) coeff[d][m]=customMomentumFieldValue(A,d,cp.localIndex[m]);
      else for(int d=0;d<3;++d) coeff[d][m]=entityDirValue(D,d,cp.entity[m]);
    }
    double uref[8][3]={{0}};
    for(int m=0;m<8;++m) for(int j=0;j<3;++j) for(int d=0;d<3;++d) uref[m][j]+=coeff[d][m]*cp.invJ[j][d];
    double Cl[8][8]={{0}};
    for(int a=0;a<8;++a) for(int b=0;b<8;++b) {
      double v=0.0; for(int m=0;m<8;++m) for(int j=0;j<3;++j) v+=uref[m][j]*T.t[a][m][b][j]; Cl[a][b]=cp.det*v;
    }
    for(PetscInt ii=0;ii<cp.nOwnedRows;++ii) {
      const int a=(int)cp.ownedBasis[ii]; const PetscInt row=cp.gid[a],lr=row-A.rstart;
      for(int b=0;b<8;++b) {
        if(cp.gid[b]>=0) { const std::uint8_t slot=cp.rowSlot[a][b]; if(slot==255) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M3A central missing precomputed slot"); A.aRel[(std::size_t)(A.rowPtr[(std::size_t)lr]+slot)]+=Cl[a][b]; }
        else for(int d=0;d<3;++d) A.convRhs[d][(std::size_t)lr]-=Cl[a][b]*entityDirValue(D,d,cp.entity[b]);
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode assembleSupgCustom(const Discrete& D,const CustomDynamicAssemblyPlan& P,CustomMomentumCSR& A,
  double tauScale,double supgMagic,const std::string& form,SupgStats& stats) {
  PetscFunctionBeginUser;
  const bool implicit=(form=="implicit"); if(!implicit && form!="explicit") SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"SUPG form must be implicit or explicit");
  for(auto& r:A.supgRhs) std::fill(r.begin(),r.end(),0.0);
  double localTauMin=1e300,localTauMax=0.0,localTauWeighted=0.0,localWeight=0.0;
  for(std::size_t ic=0;ic<P.cells.size();++ic) {
    const auto& cp=P.cells[ic];
    double coeff[3][8]={{0}};
    for(int m=0;m<8;++m) {
      if(cp.gid[m]>=0) for(int d=0;d<3;++d) coeff[d][m]=customMomentumFieldValue(A,d,cp.localIndex[m]);
      else for(int d=0;d<3;++d) coeff[d][m]=entityDirValue(D,d,cp.entity[m]);
    }
    double Sl[8][8]={{0}},Vr[3][8]={{0}},gradLambdaDot[4][4]={{0}};
    for(int i=0;i<4;++i) for(int j=i;j<4;++j) { double v=0; for(int d=0;d<3;++d)v+=cp.gradLambda[i][d]*cp.gradLambda[j][d]; gradLambdaDot[i][j]=gradLambdaDot[j][i]=v; }
    for(PetscInt q=0;q<P.nQ;++q) {
      const auto& rq=P.ref[(std::size_t)q]; const std::size_t iq=(ic*(std::size_t)P.nQ+(std::size_t)q)*3;
      double adv[3]={0,0,0}; for(int d=0;d<3;++d) for(int m=0;m<8;++m) adv[d]+=coeff[d][m]*rq.phi[m];
      const double speed2=adv[0]*adv[0]+adv[1]*adv[1]+adv[2]*adv[2];
      double gradBasis[8][3]={{0}},stream[8]={0},strongTrial[8]={0};
      for(int a=0;a<4;++a) for(int d=0;d<3;++d) gradBasis[a][d]=cp.gradLambda[a][d];
      for(int i=0;i<4;++i) { int js[3],kk=0; for(int j=0;j<4;++j) if(j!=i) js[kk++]=j; for(int d=0;d<3;++d)
        gradBasis[4+i][d]=27.0*(rq.lam[js[1]]*rq.lam[js[2]]*cp.gradLambda[js[0]][d]+rq.lam[js[0]]*rq.lam[js[2]]*cp.gradLambda[js[1]][d]+rq.lam[js[0]]*rq.lam[js[1]]*cp.gradLambda[js[2]][d]); }
      for(int a=0;a<8;++a) stream[a]=adv[0]*gradBasis[a][0]+adv[1]*gradBasis[a][1]+adv[2]*gradBasis[a][2];
      for(int a=0;a<4;++a) strongTrial[a]=stream[a];
      for(int i=0;i<4;++i) { int js[3],kk=0; for(int j=0;j<4;++j) if(j!=i) js[kk++]=j; const double lap=54.0*(rq.lam[js[2]]*gradLambdaDot[js[0]][js[1]]+rq.lam[js[1]]*gradLambdaDot[js[0]][js[2]]+rq.lam[js[0]]*gradLambdaDot[js[1]][js[2]]); strongTrial[4+i]=-P.nu*lap+stream[4+i]; }
      const double diff=4.0*P.nu/cp.h2,den=std::max(4.0*speed2/cp.h2+supgMagic*diff*diff,1e-30),tau=tauScale/std::sqrt(den),w=tau*rq.w*cp.det,volumeW=rq.w*cp.det;
      localTauMin=std::min(localTauMin,tau); localTauMax=std::max(localTauMax,tau); localTauWeighted+=tau*volumeW; localWeight+=volumeW;
      if(implicit) {
        for(int a=0;a<8;++a) { const double ta=w*stream[a]; for(int b=0;b<8;++b) Sl[a][b]+=ta*strongTrial[b]; if(!P.forcingZero) for(int d=0;d<3;++d) Vr[d][a]+=ta*P.forcing[iq+d]; }
      } else {
        double sr[3]={0,0,0}; for(int d=0;d<3;++d){for(int b=0;b<8;++b)sr[d]+=coeff[d][b]*strongTrial[b];if(!P.forcingZero)sr[d]-=P.forcing[iq+d];}
        for(int a=0;a<8;++a){const double ta=-w*stream[a];for(int d=0;d<3;++d)Vr[d][a]+=ta*sr[d];}
      }
    }
    for(PetscInt ii=0;ii<cp.nOwnedRows;++ii) {
      const int a=(int)cp.ownedBasis[ii]; const PetscInt row=cp.gid[a],lr=row-A.rstart;
      if(implicit) {
        for(int b=0;b<8;++b) {
          if(cp.gid[b]>=0) { const std::uint8_t slot=cp.rowSlot[a][b]; if(slot==255) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M3A SUPG missing precomputed slot"); A.aRel[(std::size_t)(A.rowPtr[(std::size_t)lr]+slot)]+=Sl[a][b]; }
          else for(int d=0;d<3;++d) A.supgRhs[d][(std::size_t)lr]-=Sl[a][b]*entityDirValue(D,d,cp.entity[b]);
        }
      }
      for(int d=0;d<3;++d) A.supgRhs[d][(std::size_t)lr]+=Vr[d][a];
    }
  }
  double globalMin=0,globalMax=0,globalTW=0,globalW=0;
  PetscCallMPI(MPI_Allreduce(&localTauMin,&globalMin,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localTauMax,&globalMax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localTauWeighted,&globalTW,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localWeight,&globalW,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  stats.tauMin=globalMin;stats.tauMax=globalMax;stats.tauMean=globalW?globalTW/globalW:0.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

struct SchurTerm {
  PetscInt localRau=-1;
  double coeff=0.0;
};

struct SchurColumnPlan {
  PetscInt col=-1;
  std::vector<SchurTerm> terms;
};

struct SchurRowPlan {
  PetscInt row=-1;
  std::vector<SchurColumnPlan> columns;
};

struct PressureAssemblyPlan {
  Mat S=nullptr;
  Mat Pcompact=nullptr; // compact face-neighbour GAMG surrogate
  PetscInt globalSchurNnz=0; // structural NNZ of explicitly expanded B diag(rAU) B^T
  std::vector<std::vector<PetscInt>> vertexCells;
  std::vector<std::array<double,24>> Bcell; // [d*8+a]
  // Geometry-only compact FV/LSQ data, indexed by cell/local tet face.
  // fvLsqGeom is |S_f dot c_Kf| where c_Kf is the direct-neighbour
  // coefficient in a one-ring weighted LSQ gradient.  fvTpfaGeom is a
  // positive nonorthogonality-capped two-point fallback.
  std::vector<std::array<double,4>> fvLsqGeom;
  std::vector<std::array<double,4>> fvTpfaGeom;
  std::vector<std::array<double,4>> fvNonorthCos;
  // Pressure-cell diagonal exchange used by the FV/LSQ surrogate.  The
  // velocity ghost plan contains the DOFs of owned elements only, so it is
  // deliberately NOT used to inspect a neighbouring off-rank cell.  Instead
  // each rank computes exact Schur diagonals for its owned pressure cells and
  // ghosts only the face-neighbour cell diagonals needed by the compact Pmat.
  Vec fvCellDiag=nullptr;
  PetscInt fvCellDiagStart=0;
  std::unordered_map<PetscInt,PetscInt> fvCellDiagLocal; // pressure gid -> local-form index
  std::vector<SchurRowPlan> rows;
  std::vector<PetscInt> flatRowGid,flatRowColOffset,flatColGid,flatColTermOffset,flatTermLocalRau;
  std::vector<double> flatTermCoeff;
};


static unsigned long long u64bytes(std::size_t n) {
  return (unsigned long long)n;
}

template<class T>
static unsigned long long vectorUsefulBytes(const std::vector<T>& v) {
  return u64bytes(v.size()) * u64bytes(sizeof(T));
}

template<class T>
static unsigned long long vectorRetainedBytes(const std::vector<T>& v) {
  return u64bytes(sizeof(v)) + u64bytes(v.capacity()) * u64bytes(sizeof(T));
}

template<class K,class V>
static unsigned long long unorderedMapRetainedEstimateBytes(const std::unordered_map<K,V>& m) {
  // Lower-bound-ish retained estimate: hash buckets plus one pair and two link/hash
  // words per node.  The allocator's own metadata is intentionally not guessed.
  return u64bytes(sizeof(m))
       + u64bytes(m.bucket_count()) * u64bytes(sizeof(void*))
       + u64bytes(m.size()) * (u64bytes(sizeof(std::pair<const K,V>)) + 2u*u64bytes(sizeof(void*)));
}

static PetscErrorCode printMemoryAuditBytes(const char *stage,const char *name,
                                            unsigned long long localUseful,
                                            unsigned long long localRetained,
                                            PetscInt cells,const char *scope,
                                            const char *note="") {
  PetscFunctionBeginUser;
  unsigned long long useful=0,retained=0,retMin=0,retMax=0;
  PetscCallMPI(MPI_Allreduce(&localUseful,&useful,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localRetained,&retained,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localRetained,&retMin,1,MPI_UNSIGNED_LONG_LONG,MPI_MIN,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localRetained,&retMax,1,MPI_UNSIGNED_LONG_LONG,MPI_MAX,PETSC_COMM_WORLD));
  const double mib=retained/(1024.0*1024.0);
  const double bpc=cells>0?(double)retained/(double)cells:0.0;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_MEMORY_AUDIT_OBJECT stage=%s name=%s scope=%s usefulBytes=%llu retainedEstimateBytes=%llu retainedMiB=%.3f retainedBytesPerCell=%.3f minRankRetainedBytes=%llu maxRankRetainedBytes=%llu note=%s\n",
    stage,name,scope,useful,retained,mib,bpc,retMin,retMax,note));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode printMemoryAuditMat(const char *stage,const char *name,Mat A,PetscInt cells) {
  PetscFunctionBeginUser;
  if(!A) PetscFunctionReturn(PETSC_SUCCESS);
  MatInfo info;
  PetscInt nr=0,nc=0;
  PetscCall(MatGetSize(A,&nr,&nc));
  PetscCall(MatGetInfo(A,MAT_GLOBAL_SUM,&info));
  const double ratio=info.nz_used>0.0?info.nz_allocated/info.nz_used:0.0;
  const double memMiB=info.memory/(1024.0*1024.0);
  const double memBpc=cells>0?info.memory/(double)cells:0.0;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_MEMORY_AUDIT_MAT stage=%s name=%s rows=%" PetscInt_FMT " cols=%" PetscInt_FMT
    " nzUsed=%.0f nzAllocated=%.0f nzUnneeded=%.0f allocOverUsed=%.6f mallocs=%.0f memoryBytes=%.0f memoryMiB=%.3f memoryBytesPerCell=%.3f\n",
    stage,name,nr,nc,info.nz_used,info.nz_allocated,info.nz_unneeded,ratio,info.mallocs,info.memory,memMiB,memBpc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode auditMeshMemory(const Mesh& M,PetscInt cells) {
  PetscFunctionBeginUser;
  unsigned long long useful=0,retained=0;
  useful += vectorUsefulBytes(M.points); retained += vectorRetainedBytes(M.points);
  PetscCall(printMemoryAuditBytes("after_mesh","mesh.points",vectorUsefulBytes(M.points),vectorRetainedBytes(M.points),cells,"replicated_per_rank"));
  unsigned long long fu=vectorUsefulBytes(M.faces),fr=u64bytes(sizeof(M.faces))+u64bytes(M.faces.capacity())*u64bytes(sizeof(Face));
  for(const auto& f:M.faces){fu+=vectorUsefulBytes(f.v);fr+=u64bytes(f.v.capacity())*u64bytes(sizeof(int));}
  useful+=fu;retained+=fr;
  PetscCall(printMemoryAuditBytes("after_mesh","mesh.faces_nested",fu,fr,cells,"replicated_per_rank","includes_outer_Face_vector_objects_and_inner_vertex_capacity"));
  auto addvec=[&](const char *name,const auto& v){
    const auto u=vectorUsefulBytes(v),r=vectorRetainedBytes(v); useful+=u;retained+=r;
    return printMemoryAuditBytes("after_mesh",name,u,r,cells,"replicated_per_rank");
  };
  PetscCall(addvec("mesh.owner",M.owner));
  PetscCall(addvec("mesh.neighbour",M.neighbour));
  PetscCall(addvec("mesh.tets",M.tets));
  PetscCall(addvec("mesh.oppFace",M.oppFace));
  PetscCall(addvec("mesh.facePatch",M.facePatch));
  unsigned long long pu=vectorUsefulBytes(M.patches),pr=u64bytes(sizeof(M.patches))+u64bytes(M.patches.capacity())*u64bytes(sizeof(Patch));
  for(const auto& p:M.patches){pu+=u64bytes(p.name.size()+1);pr+=u64bytes(p.name.capacity()+1);}
  useful+=pu;retained+=pr;
  PetscCall(printMemoryAuditBytes("after_mesh","mesh.patches",pu,pr,cells,"replicated_per_rank"));
  PetscCall(printMemoryAuditBytes("after_mesh","mesh.TOTAL_CPP",useful,retained,cells,"replicated_per_rank","excludes_allocator_metadata_and_temporary_parser_strings"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode auditDiscreteMemory(const Discrete& D,PetscInt cells) {
  PetscFunctionBeginUser;
  int rank=0; PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  unsigned long long useful=0,retained=0;
  auto emit=[&](const char *name,const auto& v,const char *scope="replicated_per_rank")->PetscErrorCode{
    const auto u=vectorUsefulBytes(v),r=vectorRetainedBytes(v); useful+=u;retained+=r;
    return printMemoryAuditBytes("after_fe_assembly",name,u,r,cells,scope);
  };
  PetscCall(emit("discrete.g2free",D.g2free));
  PetscCall(emit("discrete.pGid",D.pGid));
  PetscCall(emit("discrete.cellOwner",D.cellOwner));
  PetscCall(emit("discrete.velCount",D.velCount,"small_rank_metadata"));
  PetscCall(emit("discrete.cellCount",D.cellCount,"small_rank_metadata"));
  PetscCall(emit("discrete.fixedEntity",D.fixedEntity));
  for(int d=0;d<3;++d) {
    std::string n="discrete.dirValue"+std::to_string(d);
    PetscCall(emit(n.c_str(),D.dirValue[d]));
  }
  PetscCall(emit("discrete.volumesOwnedFP64",D.volumesOwnedFP64,"distributed_owned_pressure_rows"));
  PetscCall(emit("discrete.fixedDivOwnedFP64",D.fixedDivOwnedFP64,"distributed_owned_pressure_rows"));
  for(int d=0;d<3;++d){std::string n="discrete.rhsOwnedFP64_"+std::to_string(d);PetscCall(emit(n.c_str(),D.rhsOwnedFP64[(std::size_t)d],"distributed_owned_velocity_rows"));}
  PetscCall(printMemoryAuditBytes("after_fe_assembly","discrete.TOTAL_CPP",useful,retained,cells,"mixed","most_large_arrays_are_global_and_replicated_per_rank"));
  PetscCall(printMemoryAuditMat("after_fe_assembly","D.A_diffusion",D.A,cells));
  for(int d=0;d<3;++d){std::string n="D.B"+std::to_string(d);PetscCall(printMemoryAuditMat("after_fe_assembly",n.c_str(),D.B[d],cells));}
  unsigned long long rhsVecs=0;for(int d=0;d<3;++d)if(D.rhs[d])++rhsVecs;
  const unsigned long long vecPayload=(rhsVecs*(unsigned long long)D.velCount[rank] + 2ull*(unsigned long long)D.cellCount[rank])*sizeof(PetscScalar);
  PetscCall(printMemoryAuditBytes("after_fe_assembly","petsc.initial_vec_numeric_payload",vecPayload,vecPayload,cells,"distributed_owned_payload","M6B_pressure_layout_bridges_plus_optional_velocity_rhs_reference"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode auditPlanMemory(const GhostPlan& G,const PressureAssemblyPlan& P,
                                      const CentralAssemblyPlan& C,const SupgAssemblyPlan& S,
                                      PetscInt cells) {
  PetscFunctionBeginUser;
  // Velocity ghost plan
  unsigned long long gu=vectorUsefulBytes(G.ghosts),gr=vectorRetainedBytes(G.ghosts);
  gu += u64bytes(G.ghostLocal.size())*u64bytes(sizeof(std::pair<const PetscInt,PetscInt>));
  gr += unorderedMapRetainedEstimateBytes(G.ghostLocal);
  PetscCall(printMemoryAuditBytes("after_plans","ghost.velocity",gu,gr,cells,"distributed_owned_plus_halo","unordered_map_retained_is_lower_bound_estimate"));

  // Pressure plan: break out the globally replicated and local pieces.
  unsigned long long vu=u64bytes(sizeof(P.vertexCells))+u64bytes(P.vertexCells.size())*u64bytes(sizeof(std::vector<PetscInt>));
  unsigned long long vr=u64bytes(sizeof(P.vertexCells))+u64bytes(P.vertexCells.capacity())*u64bytes(sizeof(std::vector<PetscInt>));
  for(const auto& v:P.vertexCells){vu+=vectorUsefulBytes(v);vr+=u64bytes(v.capacity())*u64bytes(sizeof(PetscInt));}
  PetscCall(printMemoryAuditBytes("after_plans","pressure.vertexCells",vu,vr,cells,"replicated_per_rank","nested_support_vectors"));
  PetscCall(printMemoryAuditBytes("after_plans","pressure.Bcell",vectorUsefulBytes(P.Bcell),vectorRetainedBytes(P.Bcell),cells,"replicated_per_rank","24_doubles_per_global_cell_per_rank"));
  PetscCall(printMemoryAuditBytes("after_plans","pressure.fvLsqGeom",vectorUsefulBytes(P.fvLsqGeom),vectorRetainedBytes(P.fvLsqGeom),cells,"replicated_per_rank"));
  PetscCall(printMemoryAuditBytes("after_plans","pressure.fvTpfaGeom",vectorUsefulBytes(P.fvTpfaGeom),vectorRetainedBytes(P.fvTpfaGeom),cells,"replicated_per_rank"));
  PetscCall(printMemoryAuditBytes("after_plans","pressure.fvNonorthCos",vectorUsefulBytes(P.fvNonorthCos),vectorRetainedBytes(P.fvNonorthCos),cells,"replicated_per_rank"));

  unsigned long long ru=u64bytes(sizeof(P.rows))+u64bytes(P.rows.size())*u64bytes(sizeof(SchurRowPlan));
  unsigned long long rr=u64bytes(sizeof(P.rows))+u64bytes(P.rows.capacity())*u64bytes(sizeof(SchurRowPlan));
  unsigned long long rows=0,cols=0,terms=0;
  for(const auto& r:P.rows){
    ++rows;
    ru+=u64bytes(r.columns.size())*u64bytes(sizeof(SchurColumnPlan));
    rr+=u64bytes(r.columns.capacity())*u64bytes(sizeof(SchurColumnPlan));
    cols+=r.columns.size();
    for(const auto& c:r.columns){
      ru+=u64bytes(c.terms.size())*u64bytes(sizeof(SchurTerm));
      rr+=u64bytes(c.terms.capacity())*u64bytes(sizeof(SchurTerm));
      terms+=c.terms.size();
    }
  }
  PetscCall(printMemoryAuditBytes("after_plans","pressure.schurNestedRows",ru,rr,cells,"distributed_owned_rows","retained_includes_vector_objects_and_capacities_excludes_allocator_metadata"));
  const unsigned long long fu=vectorUsefulBytes(P.flatRowGid)+vectorUsefulBytes(P.flatRowColOffset)+vectorUsefulBytes(P.flatColGid)+vectorUsefulBytes(P.flatColTermOffset)+vectorUsefulBytes(P.flatTermLocalRau)+vectorUsefulBytes(P.flatTermCoeff);
  const unsigned long long fr=vectorRetainedBytes(P.flatRowGid)+vectorRetainedBytes(P.flatRowColOffset)+vectorRetainedBytes(P.flatColGid)+vectorRetainedBytes(P.flatColTermOffset)+vectorRetainedBytes(P.flatTermLocalRau)+vectorRetainedBytes(P.flatTermCoeff);
  PetscCall(printMemoryAuditBytes("after_plans","pressure.schurFlatPlan",fu,fr,cells,"distributed_owned_rows","flat_row_col_term_arrays_no_nested_vectors"));
  unsigned long long mapu=u64bytes(P.fvCellDiagLocal.size())*u64bytes(sizeof(std::pair<const PetscInt,PetscInt>));
  unsigned long long mapr=unorderedMapRetainedEstimateBytes(P.fvCellDiagLocal);
  PetscCall(printMemoryAuditBytes("after_plans","pressure.fvCellDiagLocal_map",mapu,mapr,cells,"distributed_owned_plus_halo","lower_bound_map_estimate"));
  const unsigned long long fvDiagVecLocal=u64bytes(P.fvCellDiagLocal.size())*u64bytes(sizeof(PetscScalar));
  PetscCall(printMemoryAuditBytes("after_plans","pressure.fvCellDiag_vec_numeric_payload",fvDiagVecLocal,fvDiagVecLocal,cells,"distributed_owned_plus_halo","PETSc_Vec_numeric_slots_only"));
  PetscCall(printMemoryAuditMat("after_plans","pressure.S_explicit_snapshot",P.S,cells));
  PetscCall(printMemoryAuditMat("after_plans","pressure.Pcompact",P.Pcompact,cells));
  if(!P.flatRowGid.empty()){rows=P.flatRowGid.size();cols=P.flatColGid.size();terms=P.flatTermLocalRau.size();}
  unsigned long long grow=rows,gcol=cols,gterm=terms;
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE,&grow,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE,&gcol,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE,&gterm,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_MEMORY_AUDIT_SCHUR_COUNTS ownedRows=%llu uniqueColumns=%llu algebraicTerms=%llu avgColumnsPerRow=%.3f avgTermsPerColumn=%.3f sizeofRow=%zu sizeofColumn=%zu sizeofTerm=%zu\n",
    grow,gcol,gterm,grow?(double)gcol/(double)grow:0.0,gcol?(double)gterm/(double)gcol:0.0,sizeof(SchurRowPlan),sizeof(SchurColumnPlan),sizeof(SchurTerm)));
  if(!P.flatRowGid.empty()) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_M5A_FLAT_SCHUR rows=%llu columns=%llu terms=%llu BcellBytes=%zu nestedRows=%zu\n",grow,gcol,gterm,P.Bcell.size()*sizeof(std::array<double,24>),P.rows.size()));

  // Central plan is owned-cell only.
  PetscCall(printMemoryAuditBytes("after_plans","central.cells",vectorUsefulBytes(C.cells),vectorRetainedBytes(C.cells),cells,"distributed_owned_cells"));

  // SUPG: reference table is replicated tiny data; cells/caches are owned-cell only.
  PetscCall(printMemoryAuditBytes("after_plans","supg.reference",vectorUsefulBytes(S.ref),vectorRetainedBytes(S.ref),cells,"replicated_per_rank"));
  PetscCall(printMemoryAuditBytes("after_plans","supg.cells",vectorUsefulBytes(S.cells),vectorRetainedBytes(S.cells),cells,"distributed_owned_cells"));
  PetscCall(printMemoryAuditBytes("after_plans","supg.grad_physical_q_basis_xyz",0,0,cells,"eliminated","reconstructed_from_12_gradLambda_doubles_per_owned_cell"));
  PetscCall(printMemoryAuditBytes("after_plans","supg.viscStrong_q_basis",0,0,cells,"eliminated","BF3_laplacian_reconstructed_from_gradLambda_dot_products"));
  PetscCall(printMemoryAuditBytes("after_plans","supg.forcing",vectorUsefulBytes(S.forcing),vectorRetainedBytes(S.forcing),cells,"distributed_owned_cells","empty_for_pipe_and_generic_flow"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode auditStateObjects(Mat C,Mat Sg,Mat Knu,Mat Aphys,Mat Ar,
                                        const Discrete& D,const GhostPlan& G,
                                        PetscInt cells) {
  PetscFunctionBeginUser;
  int rank=0; PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  if(C) PetscCall(printMemoryAuditMat("after_state_objects","C_convection",C,cells));
  if(Sg) PetscCall(printMemoryAuditMat("after_state_objects","Sg_supg",Sg,cells));
  if(Knu) PetscCall(printMemoryAuditMat("after_state_objects","Knu_static_diffusion",Knu,cells));
  if(Aphys) PetscCall(printMemoryAuditMat("after_state_objects","Aphys",Aphys,cells));
  if(Ar) PetscCall(printMemoryAuditMat("after_state_objects","Ar_relaxed",Ar,cells));
  const unsigned long long ghostLocal=(unsigned long long)G.ghosts.size();
  // Velocity-sized state/work vectors existing at this point.
  (void)ghostLocal;
  const unsigned long long pressureCount=2ull; // M6B: pcIn/pcOut only
  const unsigned long long localP=(unsigned long long)D.cellCount[rank];
  const unsigned long long localPayload=pressureCount*localP*sizeof(PetscScalar);
  PetscCall(printMemoryAuditBytes("after_state_objects","petsc.state_work_vec_numeric_payload",localPayload,localPayload,cells,"distributed_global_payload","M6B_pcIn_pcOut_only_excludes_PETSc_vec_headers"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode auditGAMGMemory(KSP pksp,PetscInt cells) {
  PetscFunctionBeginUser;
  if(!pksp) PetscFunctionReturn(PETSC_SUCCESS);
  PC pc=nullptr; const char *pct=nullptr;
  PetscCall(KSPGetPC(pksp,&pc)); PetscCall(PCGetType(pc,&pct));
  // M8 diagnostic-only generalization: PCMGGetLevels() is valid for GAMG/MG
  // but not for PCHYPRE/BoomerAMG.  Non-GAMG cases are still fully measured
  // by the common RSS/HWM resource marks; skip only the PETSc-MG internals audit.
  if(!pct || std::string(pct)!="gamg") {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_MEMORY_AUDIT_AMG_INTERNALS_SKIPPED pc=%s reason=non_GAMG use=RSS_HWM_resource_marks\n",pct?pct:"?"));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscInt levels=0;
  PetscCall(PCMGGetLevels(pc,&levels));
  double totalMatrixMemory=0.0,totalNnz=0.0,totalAllocated=0.0;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_MEMORY_AUDIT_GAMG_BEGIN pc=%s levels=%" PetscInt_FMT "\n",pct?pct:"?",levels));
  for(PetscInt lev=0;lev<levels;++lev) {
    KSP lksp=nullptr;
    if(lev==0) PetscCall(PCMGGetCoarseSolve(pc,&lksp));
    else PetscCall(PCMGGetSmoother(pc,lev,&lksp));
    Mat A=nullptr,Pm=nullptr; PetscCall(KSPGetOperators(lksp,&A,&Pm)); if(!Pm) Pm=A;
    if(Pm) {
      MatInfo i; PetscInt nr=0,nc=0; PetscCall(MatGetSize(Pm,&nr,&nc)); PetscCall(MatGetInfo(Pm,MAT_GLOBAL_SUM,&i));
      totalMatrixMemory+=i.memory; totalNnz+=i.nz_used; totalAllocated+=i.nz_allocated;
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "P1BF3_MEMORY_AUDIT_GAMG_LEVEL level=%" PetscInt_FMT " rows=%" PetscInt_FMT " cols=%" PetscInt_FMT
        " nzUsed=%.0f nzAllocated=%.0f allocOverUsed=%.6f memoryMiB=%.3f\n",
        lev,nr,nc,i.nz_used,i.nz_allocated,i.nz_used?i.nz_allocated/i.nz_used:0.0,i.memory/(1024.0*1024.0)));
    }
    if(lev>0) {
      Mat I=nullptr; PetscCall(PCMGGetInterpolation(pc,lev,&I));
      if(I) {
        MatInfo ii; PetscInt nr=0,nc=0; PetscCall(MatGetSize(I,&nr,&nc)); PetscCall(MatGetInfo(I,MAT_GLOBAL_SUM,&ii));
        totalMatrixMemory+=ii.memory; totalNnz+=ii.nz_used; totalAllocated+=ii.nz_allocated;
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
          "P1BF3_MEMORY_AUDIT_GAMG_TRANSFER fineLevel=%" PetscInt_FMT " rows=%" PetscInt_FMT " cols=%" PetscInt_FMT
          " nzUsed=%.0f nzAllocated=%.0f allocOverUsed=%.6f memoryMiB=%.3f\n",
          lev,nr,nc,ii.nz_used,ii.nz_allocated,ii.nz_used?ii.nz_allocated/ii.nz_used:0.0,ii.memory/(1024.0*1024.0)));
      }
    }
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_MEMORY_AUDIT_GAMG_TOTAL matrixAndTransferMemoryMiB=%.3f bytesPerCell=%.3f nzUsed=%.0f nzAllocated=%.0f allocOverUsed=%.6f note=excludes_KSP_PC_vectors_and_nonmatrix_metadata\n",
    totalMatrixMemory/(1024.0*1024.0),cells?totalMatrixMemory/(double)cells:0.0,totalNnz,totalAllocated,totalNnz?totalAllocated/totalNnz:0.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static int localBasisForEntity(const Mesh& M,PetscInt cell,PetscInt entity) {
  const PetscInt nv=(PetscInt)M.points.size();
  if(entity<nv) {
    for(int i=0;i<4;++i) if(M.tets[cell][i]==entity) return i;
  } else {
    const PetscInt f=entity-nv;
    for(int i=0;i<4;++i) if(M.oppFace[cell][i]==f) return 4+i;
  }
  return -1;
}

static PetscErrorCode buildPressureAssemblyPlan(const Mesh& M,const Discrete& D,int rank,const GhostPlan& G,const std::string& pPmatMode,PetscBool buildExpandedSchur,PressureAssemblyPlan& P) {
  PetscFunctionBeginUser;
  const PetscInt nv=(PetscInt)M.points.size(), ni=(PetscInt)M.neighbour.size(), nc=(PetscInt)M.tets.size();
  // Only full/legacy surrogate modes need the global vertex-star support.
  // native_face is intentionally lean: its compact face graph is built directly
  // from the custom live B geometry and never stores vertexCells or Bcell.
  P.vertexCells.clear();
  if(pPmatMode!="native_face") {
    P.vertexCells.assign(nv,{});
    for(PetscInt c=0;c<nc;++c) for(int i=0;i<4;++i) P.vertexCells[M.tets[c][i]].push_back(c);
  }

  // Production full Pmat reconstructs B analytically; legacy surrogate modes keep the old B cache.
  P.Bcell.clear();
  if(pPmatMode!="full" && pPmatMode!="native_face") {
    P.Bcell.resize(nc);
    const double glref[4][3]={{-1,-1,-1},{1,0,0},{0,1,0},{0,0,1}};
    for(PetscInt c=0;c<nc;++c) {
      const auto t=M.tets[c];
      const Vec3 X[4]={M.points[t[0]],M.points[t[1]],M.points[t[2]],M.points[t[3]]};
      double J[3][3]={{X[1].x-X[0].x,X[2].x-X[0].x,X[3].x-X[0].x},
                      {X[1].y-X[0].y,X[2].y-X[0].y,X[3].y-X[0].y},
                      {X[1].z-X[0].z,X[2].z-X[0].z,X[3].z-X[0].z}},invJ[3][3];
      const double det=det3(J),vol=det/6.0; inv3(J,invJ); double gl[4][3]={{0}};
      for(int i=0;i<4;++i) for(int d=0;d<3;++d) for(int j=0;j<3;++j) gl[i][d]+=glref[i][j]*invJ[j][d];
      for(int i=0;i<4;++i) for(int d=0;d<3;++d) { P.Bcell[c][d*8+i]=vol*gl[i][d]; P.Bcell[c][d*8+4+i]=-(27.0/20.0)*vol*gl[i][d]; }
    }
  }

  // Precompute a compact one-ring LSQ geometry for an FV-like pressure
  // surrogate.  The LSQ normal coefficient is used only to define the
  // face-neighbour PRECONDITIONING graph; the exact FE Schur remains the KSP
  // operator.  Boundary faces enter the LSQ metric through reflected
  // centroid pseudo-neighbours so wall/outlet cells retain a full-rank 3-D
  // geometry metric without adding pressure unknowns.
  if(pPmatMode=="fv_lsq") {
    const auto fvCellC = cellCentroids(M);
    P.fvLsqGeom.assign(nc,{});
    P.fvTpfaGeom.assign(nc,{});
    P.fvNonorthCos.assign(nc,{});
    auto faceCentre = [&](PetscInt f)->Vec3 {
      Vec3 q{}; const auto& F=M.faces[f];
      for(PetscInt v:F.v) { const auto& x=M.points[v]; q.x+=x.x; q.y+=x.y; q.z+=x.z; }
      const double z=1.0/(double)F.v.size(); return {q.x*z,q.y*z,q.z*z};
    };
    auto faceAreaVec = [&](PetscInt f)->Vec3 {
      const auto& F=M.faces[f];
      if(F.v.size()!=3) return {};
      const Vec3& x0=M.points[F.v[0]]; const Vec3& x1=M.points[F.v[1]]; const Vec3& x2=M.points[F.v[2]];
      const Vec3 cr=cross3(sub3(x1,x0),sub3(x2,x0)); return {0.5*cr.x,0.5*cr.y,0.5*cr.z};
    };
    for(PetscInt c=0;c<nc;++c) {
      double H[3][3]={{0}};
      std::array<Vec3,4> dFace{};
      std::array<double,4> wFace{};
      for(int i=0;i<4;++i) {
        const PetscInt f=M.oppFace[c][i];
        Vec3 d{};
        if(f<ni) {
          const PetscInt L=(M.owner[f]==c)?M.neighbour[f]:M.owner[f];
          d=sub3(fvCellC[L],fvCellC[c]);
        } else {
          const Vec3 fc=faceCentre(f);
          const Vec3 h=sub3(fc,fvCellC[c]); d={2.0*h.x,2.0*h.y,2.0*h.z};
        }
        const double d2=d.x*d.x+d.y*d.y+d.z*d.z;
        const double w=1.0/std::max(d2,1e-30);
        dFace[i]=d; wFace[i]=w;
        const double a[3]={d.x,d.y,d.z};
        for(int r=0;r<3;++r) for(int q=0;q<3;++q) H[r][q]+=w*a[r]*a[q];
      }
      const double tr=H[0][0]+H[1][1]+H[2][2];
      const double reg=std::max(1e-12,1e-10*tr/3.0);
      for(int d=0;d<3;++d) H[d][d]+=reg;
      double HI[3][3]; inv3(H,HI);
      for(int i=0;i<4;++i) {
        const PetscInt f=M.oppFace[c][i];
        if(f>=ni) { P.fvLsqGeom[c][i]=0.0; P.fvTpfaGeom[c][i]=0.0; P.fvNonorthCos[c][i]=0.0; continue; }
        const Vec3 d=dFace[i]; const double w=wFace[i];
        double cv[3]={0,0,0},a[3]={d.x,d.y,d.z};
        for(int r=0;r<3;++r) for(int q=0;q<3;++q) cv[r]+=HI[r][q]*w*a[q];
        const Vec3 S=faceAreaVec(f); const double A=norm3(S), dl=norm3(d);
        const double Sdotc=S.x*cv[0]+S.y*cv[1]+S.z*cv[2];
        const double Sdotd=std::abs(S.x*d.x+S.y*d.y+S.z*d.z);
        const double coso=(A>0.0 && dl>0.0)?Sdotd/(A*dl):0.0;
        P.fvNonorthCos[c][i]=coso;
        P.fvLsqGeom[c][i]=std::abs(Sdotc);
        // Positive TPFA fallback with a 0.1 cosine floor: it cannot blow up on
        // nearly tangential centre-to-centre lines and is used only as a small
        // floor under the LSQ coefficient.
        const double denom=std::max(Sdotd,0.1*A*dl);
        P.fvTpfaGeom[c][i]=(denom>0.0)?A*A/denom:0.0;
      }
    }

  } else {
    P.fvLsqGeom.clear(); P.fvTpfaGeom.clear(); P.fvNonorthCos.clear();
  }

  // M5A: flat full-Schur cache. No retained Bcell or nested vectors in production.
  P.rows.clear(); P.flatRowGid.clear(); P.flatRowColOffset.clear(); P.flatColGid.clear();
  P.flatColTermOffset.clear(); P.flatTermLocalRau.clear(); P.flatTermCoeff.clear();
  size_t localColumns=0,localTerms=0;
  if(buildExpandedSchur) {
    if(pPmatMode=="full") {
      P.flatRowGid.reserve((std::size_t)D.cellCount[rank]); P.flatRowColOffset.reserve((std::size_t)D.cellCount[rank]+1);
      P.flatRowColOffset.push_back(0); P.flatColTermOffset.push_back(0);
      for(PetscInt K=0;K<nc;++K) if(D.cellOwner[K]==rank) {
        PetscInt entity[8]; for(int i=0;i<4;++i) entity[i]=M.tets[K][i]; for(int i=0;i<4;++i) entity[4+i]=nv+M.oppFace[K][i];
        double kVol=0.0,kGrad[4][3]={{0}}; fillCustomPressureGeom(M,K,kVol,kGrad);
        std::unordered_map<PetscInt,std::array<double,13>> geom; geom.reserve(96);
        auto getGeom=[&](PetscInt C)->const std::array<double,13>& {
          auto it=geom.find(C); if(it!=geom.end()) return it->second;
          double v=0.0,g[4][3]={{0}}; fillCustomPressureGeom(M,C,v,g); std::array<double,13> q{}; q[0]=v; int z=1;
          for(int i=0;i<4;++i) for(int d=0;d<3;++d) q[(std::size_t)z++]=g[i][d]; return geom.emplace(C,q).first->second;
        };
        std::map<PetscInt,std::vector<SchurTerm>> grouped;
        for(int a=0;a<8;++a) {
          const PetscInt gid=D.g2free[entity[a]]; if(gid<0) continue; const PetscInt li=velocityLocalIndex(G,gid);
          double BK[3]; for(int d=0;d<3;++d) BK[d]=customPressureBCoeff(kVol,kGrad,d,a);
          auto addSupport=[&](PetscInt L) {
            const int b=localBasisForEntity(M,L,entity[a]); if(b<0) throw std::runtime_error("M5A support/local-basis mismatch");
            const auto& q=getGeom(L); double lg[4][3]={{0}}; int z=1; for(int i=0;i<4;++i) for(int d=0;d<3;++d) lg[i][d]=q[(std::size_t)z++];
            double dot=0.0; for(int d=0;d<3;++d) dot+=BK[d]*customPressureBCoeff(q[0],lg,d,b); grouped[D.pGid[L]].push_back({li,dot});
          };
          if(entity[a]<nv) for(PetscInt L:P.vertexCells[entity[a]]) addSupport(L);
          else { const PetscInt f=entity[a]-nv; addSupport(M.owner[f]); if(f<ni) addSupport(M.neighbour[f]); }
        }
        P.flatRowGid.push_back(D.pGid[K]);
        for(auto& kv:grouped) { P.flatColGid.push_back(kv.first); for(const auto& t:kv.second){P.flatTermLocalRau.push_back(t.localRau);P.flatTermCoeff.push_back(t.coeff);} P.flatColTermOffset.push_back((PetscInt)P.flatTermLocalRau.size()); }
        P.flatRowColOffset.push_back((PetscInt)P.flatColGid.size());
      }
      localColumns=P.flatColGid.size(); localTerms=P.flatTermLocalRau.size();
    } else {
      P.rows.reserve(D.cellCount[rank]);
      for(PetscInt K=0;K<nc;++K) if(D.cellOwner[K]==rank) {
        PetscInt entity[8]; for(int i=0;i<4;++i) entity[i]=M.tets[K][i]; for(int i=0;i<4;++i) entity[4+i]=nv+M.oppFace[K][i];
        std::map<PetscInt,std::vector<SchurTerm>> grouped;
        for(int a=0;a<8;++a) { const PetscInt gid=D.g2free[entity[a]]; if(gid<0) continue; const PetscInt li=velocityLocalIndex(G,gid); double BK[3]; for(int d=0;d<3;++d) BK[d]=P.Bcell[K][d*8+a];
          auto add=[&](PetscInt L){const int b=localBasisForEntity(M,L,entity[a]);double dot=0;for(int d=0;d<3;++d)dot+=BK[d]*P.Bcell[L][d*8+b];grouped[D.pGid[L]].push_back({li,dot});};
          if(entity[a]<nv) for(PetscInt L:P.vertexCells[entity[a]]) add(L); else {const PetscInt f=entity[a]-nv;add(M.owner[f]);if(f<ni)add(M.neighbour[f]);}
        }
        SchurRowPlan rp; rp.row=D.pGid[K]; rp.columns.reserve(grouped.size()); for(auto& kv:grouped){SchurColumnPlan cp;cp.col=kv.first;cp.terms=std::move(kv.second);localTerms+=cp.terms.size();rp.columns.push_back(std::move(cp));} localColumns+=rp.columns.size();P.rows.push_back(std::move(rp));
      }
    }
  }

  const PetscInt nlp=D.cellCount[rank];

  // Build FV pressure-side support only for the fv_lsq surrogate.  Production
  // p_pmat=full does not need any FV geometry, face-neighbour diagonal ghosts,
  // or compact Pmat storage.
  P.fvCellDiagStart=0;
  P.fvCellDiagLocal.clear();
  if(pPmatMode=="fv_lsq") {
    for(int r=0;r<rank;++r) P.fvCellDiagStart += D.cellCount[r];
    std::vector<PetscInt> fvDiagGhosts;
    for(PetscInt K=0;K<nc;++K) if(D.cellOwner[K]==rank) {
      for(int i=0;i<4;++i) {
        const PetscInt f=M.oppFace[K][i];
        if(f>=ni) continue;
        const PetscInt L=(M.owner[f]==K)?M.neighbour[f]:M.owner[f];
        if(D.cellOwner[L]!=rank) fvDiagGhosts.push_back(D.pGid[L]);
      }
    }
    std::sort(fvDiagGhosts.begin(),fvDiagGhosts.end());
    fvDiagGhosts.erase(std::unique(fvDiagGhosts.begin(),fvDiagGhosts.end()),fvDiagGhosts.end());
    PetscCall(VecCreateGhost(PETSC_COMM_WORLD,nlp,nc,(PetscInt)fvDiagGhosts.size(),
                             fvDiagGhosts.empty()?nullptr:fvDiagGhosts.data(),&P.fvCellDiag));
    for(PetscInt i=0;i<nlp;++i) P.fvCellDiagLocal[P.fvCellDiagStart+i]=i;
    for(PetscInt j=0;j<(PetscInt)fvDiagGhosts.size();++j)
      P.fvCellDiagLocal[fvDiagGhosts[j]]=nlp+j;
  }

  // Exact full-Schur preallocation from the already-built row topology.
  // This replaces the legacy blanket 128/128 allocation that was 3.81x used NNZ.
  if(buildExpandedSchur) {
    PetscInt pStart=0; for(int r=0;r<rank;++r) pStart+=D.cellCount[r];
    const PetscInt pEnd=pStart+nlp;
    std::vector<PetscInt> dnnz((size_t)nlp,0),onnz((size_t)nlp,0);
    if(pPmatMode=="full") {
      for(std::size_t r=0;r<P.flatRowGid.size();++r) { const PetscInt li=P.flatRowGid[r]-pStart; if(li<0 || li>=nlp) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"flat Schur row ownership mismatch");
        for(PetscInt j=P.flatRowColOffset[r];j<P.flatRowColOffset[r+1];++j){const PetscInt c=P.flatColGid[(std::size_t)j];if(c>=pStart&&c<pEnd)++dnnz[(size_t)li];else++onnz[(size_t)li];} }
    } else {
      for(const auto& rp:P.rows) { const PetscInt li=rp.row-pStart; if(li<0 || li>=nlp) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Schur row ownership mismatch during exact preallocation"); for(const auto& cp:rp.columns){if(cp.col>=pStart&&cp.col<pEnd)++dnnz[(size_t)li];else++onnz[(size_t)li];} }
    }
    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD,nlp,nlp,nc,nc,0,dnnz.data(),0,onnz.data(),&P.S));
    PetscCall(MatSetOption(P.S,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));
    PetscCall(MatSetOption(P.S,MAT_SYMMETRIC,PETSC_TRUE));
    PetscInt ld=0,lo=0; for(PetscInt i=0;i<nlp;++i){ld+=dnnz[(size_t)i];lo+=onnz[(size_t)i];}
    PetscInt gd=0,go=0; PetscCallMPI(MPI_Allreduce(&ld,&gd,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD)); PetscCallMPI(MPI_Allreduce(&lo,&go,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_M3B_PMAT_PREALLOC mode=exact_from_schur_topology diagNnz=%" PetscInt_FMT " offdiagNnz=%" PetscInt_FMT " totalNnz=%" PetscInt_FMT " allocationError=ON\n",gd,go,gd+go));
  }
  if(pPmatMode!="full") {
    // Compact pressure preconditioning matrix.  native_face uses exact row
    // preallocation for diagonal + physical face neighbours; legacy modes keep
    // their historical 5/5 upper bound.
    if(pPmatMode=="native_face") {
      PetscInt pStart=0; for(int r=0;r<rank;++r) pStart+=D.cellCount[r];
      const PetscInt pEnd=pStart+nlp;
      std::vector<PetscInt> dnnz((std::size_t)nlp,1),onnz((std::size_t)nlp,0);
      for(PetscInt K=0;K<nc;++K) if(D.cellOwner[(std::size_t)K]==rank) {
        const PetscInt lr=D.pGid[(std::size_t)K]-pStart;
        for(int i=0;i<4;++i) {
          const PetscInt f=M.oppFace[(std::size_t)K][i]; if(f>=ni) continue;
          const PetscInt L=(M.owner[(std::size_t)f]==K)?M.neighbour[(std::size_t)f]:M.owner[(std::size_t)f];
          const PetscInt pg=D.pGid[(std::size_t)L]; if(pg>=pStart && pg<pEnd) ++dnnz[(std::size_t)lr]; else ++onnz[(std::size_t)lr];
        }
      }
      PetscCall(MatCreateAIJ(PETSC_COMM_WORLD,nlp,nlp,nc,nc,0,dnnz.data(),0,onnz.data(),&P.Pcompact));
      PetscCall(MatSetOption(P.Pcompact,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));
      PetscCall(MatSetOption(P.Pcompact,MAT_SYMMETRIC,PETSC_TRUE));
      PetscInt ld=0,lo=0; for(PetscInt i=0;i<nlp;++i){ld+=dnnz[(std::size_t)i];lo+=onnz[(std::size_t)i];}
      PetscInt gd=0,go=0; PetscCallMPI(MPI_Allreduce(&ld,&gd,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD)); PetscCallMPI(MPI_Allreduce(&lo,&go,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "P1BF3_B1_COMPACT_PREALLOC mode=native_face diagNnz=%" PetscInt_FMT " offdiagNnz=%" PetscInt_FMT " totalNnz=%" PetscInt_FMT " avgNnzPerRow=%.6f allocationError=ON graph=cell_plus_face_neighbours\n",
        gd,go,gd+go,(double)(gd+go)/(double)nc));
    } else {
      PetscCall(MatCreateAIJ(PETSC_COMM_WORLD,nlp,nlp,nc,nc,5,nullptr,5,nullptr,&P.Pcompact));
      PetscCall(MatSetOption(P.Pcompact,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));
      PetscCall(MatSetOption(P.Pcompact,MAT_SYMMETRIC,PETSC_TRUE));
    }
  }
  if(pPmatMode=="full") PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_M5A_FULL_PMAT_STORAGE fvGeometry=ELIMINATED fvCellDiag=ELIMINATED Pcompact=ELIMINATED physicalBcell=ELIMINATED nestedSchur=ELIMINATED flatSchur=ACTIVE\n"));
  if(buildExpandedSchur) {
    unsigned long long lc=(unsigned long long)localColumns,lt=(unsigned long long)localTerms,gc=0,gt=0;
    PetscCallMPI(MPI_Allreduce(&lc,&gc,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(&lt,&gt,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
    P.globalSchurNnz=(PetscInt)gc;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_SCHUR_PLAN cachedTopology=1 storage=%s pressureRows=%" PetscInt_FMT " uniqueRowColumns=%llu algebraicTerms=%llu update=batched_row_MatSetValues\n",
      pPmatMode=="full"?"flat_CSR_terms":"nested_legacy",nc,gc,gt));
  } else {
    P.globalSchurNnz=-1;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_SCHUR_PLAN cachedTopology=0 pressureRows=%" PetscInt_FMT " explicitExpandedSchur=NOT_MATERIALIZED operator=factored_B_rAU_Bt\n",nc));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode reindexFlatSchurRauToCustom(PressureAssemblyPlan& P,const GhostPlan& G,const CustomMomentumCSR& A) {
  PetscFunctionBeginUser;
  for(auto& li:P.flatTermLocalRau){PetscInt gid=-1;if(li<G.nOwned)gid=G.rstart+li;else{const PetscInt q=li-G.nOwned;if(q<0 || q>=(PetscInt)G.ghosts.size())SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M6B flat Schur legacy rAU index out of range");gid=G.ghosts[(std::size_t)q];}const PetscInt cli=customMomentumLocalIndex(A,gid);if(cli<0)SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M6B flat Schur rAU gid missing custom halo");li=cli;}
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode updatePressureSchurFullNative(CustomMomentumCSR& A,PetscBool assembleFullSchur,PressureAssemblyPlan& P) {
  PetscFunctionBeginUser;
  if(!assembleFullSchur) PetscFunctionReturn(PETSC_SUCCESS);
  if(!P.S) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M6B explicit full Schur requested without matrix");
  PetscCall(customMomentumExchange(A,A.rAU)); PetscCall(MatZeroEntries(P.S)); std::vector<PetscScalar> vals;
  auto rauAt=[&](PetscInt li)->double{return li<A.nOwned?A.rAU[(std::size_t)li]:A.ghostValues[(std::size_t)(li-A.nOwned)];};
  for(std::size_t r=0;r<P.flatRowGid.size();++r){const PetscInt c0=P.flatRowColOffset[r],c1=P.flatRowColOffset[r+1],n=c1-c0;vals.resize((std::size_t)n);for(PetscInt jj=0;jj<n;++jj){const PetscInt cj=c0+jj,t0=P.flatColTermOffset[(std::size_t)cj],t1=P.flatColTermOffset[(std::size_t)cj+1];double v=0;for(PetscInt t=t0;t<t1;++t)v+=rauAt(P.flatTermLocalRau[(std::size_t)t])*P.flatTermCoeff[(std::size_t)t];vals[(std::size_t)jj]=(PetscScalar)v;}if(n)PetscCall(MatSetValues(P.S,1,&P.flatRowGid[r],n,P.flatColGid.data()+c0,vals.data(),INSERT_VALUES));}
  PetscCall(MatAssemblyBegin(P.S,MAT_FINAL_ASSEMBLY));PetscCall(MatAssemblyEnd(P.S,MAT_FINAL_ASSEMBLY));PetscFunctionReturn(PETSC_SUCCESS);
}


// M11 B1+B2: lean compact FE-informed face-neighbour Pmat for PETSc GAMG.
// The exact outer PCG operator remains custom FP64 B diag(rAU) B^T.
// Pnative keeps the exact Schur diagonal, exact BF3 internal-face coupling,
// and conservatively redistributes P1 vertex diagonal energy onto physical
// face-neighbour edges.  No full Schur topology, Bcell cache, or vertexCells
// cache is retained.
static PetscErrorCode updatePressureCompactNative(const Mesh& M,const Discrete& D,int rank,
                                                   CustomPressureBPlan& B,const CustomMomentumCSR& A,
                                                   double p1Strength,PressureAssemblyPlan& P) {
  PetscFunctionBeginUser;
  if(!P.Pcompact) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M11 native compact Pmat requested without matrix");
  if(p1Strength<0.0 || p1Strength>1.0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"M11 native compact P1 strength must satisfy 0 <= strength <= 1");
  PetscCall(customPeerExchange(B.velocityHalo,A.rAU,48331));
  PetscCall(MatZeroEntries(P.Pcompact));
  const PetscInt nv=(PetscInt)M.points.size(),ni=(PetscInt)M.neighbour.size(),nc=(PetscInt)M.tets.size();
  PetscInt pStart=0; for(int r=0;r<rank;++r) pStart+=D.cellCount[(std::size_t)r];
  std::array<PetscInt,5> cols{}; std::array<PetscScalar,5> vals{};
  double localDiag=0.0,localP1Diag=0.0,localRedistributed=0.0,localBF3Abs=0.0;

  auto rAtLocal=[&](PetscInt li)->double {
    if(li<0) return 0.0;
    return customPeerValue(B.velocityHalo,A.rAU,li);
  };
  auto internalFaceDegreeAtVertex=[&](PetscInt C,int lv)->int {
    int deg=0; for(int i=0;i<4;++i) if(i!=lv && M.oppFace[(std::size_t)C][i]<ni) ++deg; return deg;
  };
  auto vertexDiagWithR=[&](PetscInt C,int lv,double rauV)->double {
    if(!(rauV>0.0)) return 0.0;
    double vol=0.0,g[4][3]={{0}}; fillCustomPressureGeom(M,C,vol,g);
    double b2=0.0; for(int d=0;d<3;++d){const double q=customPressureBCoeff(vol,g,d,lv);b2+=q*q;} return rauV*b2;
  };

  for(PetscInt K=0;K<nc;++K) if(D.cellOwner[(std::size_t)K]==rank) {
    const PetscInt pl=D.pGid[(std::size_t)K]-pStart;
    if(pl<0 || pl>=(PetscInt)B.forwardCells.size()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M11 pressure forward-cell indexing mismatch");
    const auto& cp=B.forwardCells[(std::size_t)pl];
    PetscInt ncol=1; cols[0]=D.pGid[(std::size_t)K]; vals[0]=0.0;
    double p1DiagK=0.0;
    // Exact full Schur diagonal from all local P1+BF3 basis functions.
    for(int a=0;a<8;++a) if(cp.velLocal[a]>=0) {
      double b2=0.0; for(int d=0;d<3;++d){const double q=customPressureBCoeff(cp.vol,cp.gradLambda,d,a);b2+=q*q;}
      const double da=rAtLocal(cp.velLocal[a])*b2; vals[0]+=(PetscScalar)da; if(a<4) p1DiagK+=da;
    }
    localDiag += PetscRealPart(vals[0]); localP1Diag += p1DiagK;

    for(int i=0;i<4;++i) {
      const PetscInt f=M.oppFace[(std::size_t)K][i]; if(f>=ni) continue;
      const PetscInt L=(M.owner[(std::size_t)f]==K)?M.neighbour[(std::size_t)f]:M.owner[(std::size_t)f];
      int j=-1; for(int q=0;q<4;++q) if(M.oppFace[(std::size_t)L][q]==f){j=q;break;}
      if(j<0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M11 neighbour local face not found");
      const PetscInt liFace=cp.velLocal[4+i];
      if(liFace<0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M11 internal BF3 face unexpectedly constrained/missing from velocity halo");
      const double rauF=rAtLocal(liFace);
      double volL=0.0,gL[4][3]={{0}}; fillCustomPressureGeom(M,L,volL,gL);
      double dot=0.0; for(int d=0;d<3;++d) dot += customPressureBCoeff(cp.vol,cp.gradLambda,d,4+i)*customPressureBCoeff(volL,gL,d,4+j);
      const double bf3off=rauF*dot; localBF3Abs += std::abs(bf3off);

      double qK=0.0,qL=0.0;
      if(p1Strength>0.0) {
        // Only the three vertices on this shared physical face participate.
        for(int lvK=0;lvK<4;++lvK) if(lvK!=i) {
          const PetscInt v=M.tets[(std::size_t)K][lvK];
          const PetscInt lvL=localBasisForEntity(M,L,v);
          if(lvL<0 || lvL>=4) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M11 shared face vertex missing in neighbour tet");
          const PetscInt liV=cp.velLocal[lvK];
          if(liV<0) continue; // constrained Dirichlet velocity vertex: zero pressure response
          const double rauV=rAtLocal(liV);
          const int degK=internalFaceDegreeAtVertex(K,lvK),degL=internalFaceDegreeAtVertex(L,lvL);
          if(degK>0) qK += vertexDiagWithR(K,lvK,rauV)/(double)degK;
          if(degL>0) qL += vertexDiagWithR(L,lvL,rauV)/(double)degL;
        }
      }
      const double w=p1Strength*std::max(0.0,std::min(qK,qL));
      localRedistributed += w;
      cols[ncol]=D.pGid[(std::size_t)L]; vals[ncol]=(PetscScalar)(bf3off-w); ++ncol;
    }
    PetscCall(MatSetValues(P.Pcompact,1,&cols[0],ncol,cols.data(),vals.data(),INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(P.Pcompact,MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(P.Pcompact,MAT_FINAL_ASSEMBLY));

  double gDiag=0.0,gP1=0.0,gRed=0.0,gBF3=0.0;
  PetscCallMPI(MPI_Allreduce(&localDiag,&gDiag,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localP1Diag,&gP1,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localRedistributed,&gRed,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localBF3Abs,&gBF3,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  MatInfo mi; PetscCall(MatGetInfo(P.Pcompact,MAT_GLOBAL_SUM,&mi));
  PetscBool sym=PETSC_FALSE; PetscCall(MatIsSymmetric(P.Pcompact,1e-12,&sym));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_B1_COMPACT_VALUES mode=native_face p1Strength=%.6g pmatNnz=%.0f pmatAllocated=%.0f avgNnzPerRow=%.6f exactDiagSum=%.12e p1DiagSum=%.12e redistributedDirectedDegree=%.12e redistributedFraction=%.6f bf3OffAbsRowSum=%.12e symmetric=%d exactOperator=custom_FP64_B_rAU_Bt compactRole=GAMG_preconditioner_only\n",
    p1Strength,mi.nz_used,mi.nz_allocated,mi.nz_used/(double)nc,gDiag,gP1,gRed,gP1>0.0?gRed/gP1:0.0,gBF3,(int)sym));
  if(!sym) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M11 native compact pressure Pmat lost symmetry");
  PetscFunctionReturn(PETSC_SUCCESS);
}



// Gate 9G: compact FE/SIMPLE-energy face Laplacian.
// Keep exactly the native physical face-neighbour pressure graph, but choose
// each internal edge conductance from the actual B diag(rAU) B^T energy of a
// unit pressure jump across that face, restricted to velocity basis functions
// shared by the two adjacent tetrahedra:
//
//   a_f = 1/4 sum_{j shared(P,N)} rAU_j || B_Pj - B_Nj ||_2^2 .
//
// For q=[+1,-1] on the two cells, q^T [[a,-a],[-a,a]] q = 4 a, so the factor
// 1/4 matches the exact shared-DOF FE pressure-jump energy.  This is positive
// by construction, symmetric, uses the live SIMPLE/SIMPLEC mobility, and keeps
// only one owner-neighbour edge per physical face.  The outlet p=0 anchor is
// built on the same FE scale from velocity basis functions with nonzero trace
// on the outlet face.  Finally a single global trace match rescales the compact
// matrix to the exact Schur diagonal sum; this preserves relative face weights
// while removing a global scaling error that matters to Richardson/GAMG-only.
struct Gate9gFeFaceAudit {
  PetscReal coeffMin=PETSC_MAX_REAL,coeffMax=0.0,coeffMean=0.0;
  PetscReal anchorMin=PETSC_MAX_REAL,anchorMax=0.0,anchorMean=0.0;
  PetscReal exactDiagSum=0.0,compactDiagSumBeforeScale=0.0,traceScale=1.0;
  PetscReal exactSharedOffAbs=0.0,energyEdgeSum=0.0;
  PetscInt internalFaces=0,outletFaces=0,positiveExactPair=0,nonfinite=0;
};

static PetscErrorCode updatePressureCompactFeFaceEnergy(const Mesh& M,const Discrete& D,const ProblemConfig& Pcfg,
                                                         int rank,CustomPressureBPlan& B,const CustomMomentumCSR& A,
                                                         PressureAssemblyPlan& P,Gate9gFeFaceAudit& audit) {
  PetscFunctionBeginUser;
  if(!P.Pcompact) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate-9G FE-face Kp requested without compact Pmat");
  PetscCall(customPeerExchange(B.velocityHalo,A.rAU,59371));
  PetscCall(MatZeroEntries(P.Pcompact));
  PetscCall(MatSetOption(P.Pcompact,MAT_SYMMETRIC,PETSC_TRUE));
  const PetscInt nv=(PetscInt)M.points.size(),ni=(PetscInt)M.neighbour.size(),nc=(PetscInt)M.tets.size();
  PetscInt pStart=0; for(int r=0;r<rank;++r) pStart+=D.cellCount[(std::size_t)r];
  std::array<PetscInt,5> cols{}; std::array<PetscScalar,5> vals{};
  double localCoeffMin=PETSC_MAX_REAL,localCoeffMax=0.0,localCoeffSum=0.0;
  double localAnchorMin=PETSC_MAX_REAL,localAnchorMax=0.0,localAnchorSum=0.0;
  double localExactDiag=0.0,localCompactDiag=0.0,localExactOffAbs=0.0,localEnergyEdge=0.0;
  PetscInt localInternal=0,localOutlet=0,localPositivePair=0,localNonfinite=0;

  auto rAtLocal=[&](PetscInt li)->double { return li<0?0.0:customPeerValue(B.velocityHalo,A.rAU,li); };
  auto basisB2=[&](double vol,const double g[4][3],int a)->double {
    double q=0.0; for(int d=0;d<3;++d){const double b=customPressureBCoeff(vol,g,d,a);q+=b*b;} return q;
  };

  for(PetscInt K=0;K<nc;++K) if(D.cellOwner[(std::size_t)K]==rank) {
    const PetscInt pl=D.pGid[(std::size_t)K]-pStart;
    if(pl<0 || pl>=(PetscInt)B.forwardCells.size()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate-9G pressure forward-cell indexing mismatch");
    const auto& cp=B.forwardCells[(std::size_t)pl];
    PetscInt ncol=1; cols[0]=D.pGid[(std::size_t)K]; vals[0]=0.0;

    // Exact full Schur diagonal trace contribution for scaling/audit.
    double exactDiagK=0.0;
    for(int a=0;a<8;++a) if(cp.velLocal[a]>=0) exactDiagK += rAtLocal(cp.velLocal[a])*basisB2(cp.vol,cp.gradLambda,a);
    localExactDiag += exactDiagK;

    for(int i=0;i<4;++i) {
      const PetscInt f=M.oppFace[(std::size_t)K][i];
      if(f<ni) {
        const PetscInt L=(M.owner[(std::size_t)f]==K)?M.neighbour[(std::size_t)f]:M.owner[(std::size_t)f];
        int j=-1; for(int q=0;q<4;++q) if(M.oppFace[(std::size_t)L][q]==f){j=q;break;}
        if(j<0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate-9G neighbour local face not found");
        double volL=0.0,gL[4][3]={{0}}; fillCustomPressureGeom(M,L,volL,gL);
        double jumpEnergy=0.0,exactPair=0.0;

        // Three P1 vertices shared by the physical face.
        for(int lvK=0;lvK<4;++lvK) if(lvK!=i) {
          const PetscInt v=M.tets[(std::size_t)K][lvK];
          const PetscInt lvL=localBasisForEntity(M,L,v);
          if(lvL<0 || lvL>=4) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate-9G shared face vertex missing in neighbour tet");
          const PetscInt li=cp.velLocal[lvK];
          if(li<0) continue; // constrained velocity trace has no pressure response
          const double rr=rAtLocal(li);
          double d2=0.0,dot=0.0;
          for(int d=0;d<3;++d) {
            const double bK=customPressureBCoeff(cp.vol,cp.gradLambda,d,lvK);
            const double bL=customPressureBCoeff(volL,gL,d,lvL);
            const double db=bK-bL; d2+=db*db; dot+=bK*bL;
          }
          jumpEnergy += rr*d2; exactPair += rr*dot;
        }

        // Shared BF3 face bubble.
        const PetscInt liF=cp.velLocal[4+i];
        if(liF>=0) {
          const double rr=rAtLocal(liF); double d2=0.0,dot=0.0;
          for(int d=0;d<3;++d) {
            const double bK=customPressureBCoeff(cp.vol,cp.gradLambda,d,4+i);
            const double bL=customPressureBCoeff(volL,gL,d,4+j);
            const double db=bK-bL; d2+=db*db; dot+=bK*bL;
          }
          jumpEnergy += rr*d2; exactPair += rr*dot;
        }
        const double af=0.25*jumpEnergy;
        if(!std::isfinite(af) || af<0.0) {++localNonfinite; continue;}
        vals[0]+=(PetscScalar)af;
        cols[ncol]=D.pGid[(std::size_t)L]; vals[ncol]=(PetscScalar)(-af); ++ncol;
        localCoeffMin=std::min(localCoeffMin,af); localCoeffMax=std::max(localCoeffMax,af); localCoeffSum+=af;
        localExactOffAbs+=std::abs(exactPair); localEnergyEdge+=af; ++localInternal;
        if(exactPair>0.0) ++localPositivePair;
      } else {
        const int pi=M.facePatch[(std::size_t)f];
        if(pi==Pcfg.boundary.outlet) {
          // FE-scaled Dirichlet-to-zero edge.  Only P1 vertices on this face
          // and its BF3 trace bubble participate in the pressure boundary jump.
          double ab=0.0;
          for(int lv=0;lv<4;++lv) if(lv!=i && cp.velLocal[lv]>=0)
            ab += rAtLocal(cp.velLocal[lv])*basisB2(cp.vol,cp.gradLambda,lv);
          if(cp.velLocal[4+i]>=0) ab += rAtLocal(cp.velLocal[4+i])*basisB2(cp.vol,cp.gradLambda,4+i);
          if(!std::isfinite(ab) || ab<0.0) {++localNonfinite; ab=0.0;}
          vals[0]+=(PetscScalar)ab;
          localAnchorMin=std::min(localAnchorMin,ab); localAnchorMax=std::max(localAnchorMax,ab); localAnchorSum+=ab; ++localOutlet;
        } else if(pi==Pcfg.boundary.inlet || isWallPatch(Pcfg.boundary,pi)) {
          // homogeneous Neumann for the auxiliary pressure operator
        } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate-9G encountered unclassified pressure boundary face");
      }
    }
    localCompactDiag += PetscRealPart(vals[0]);
    PetscCall(MatSetValues(P.Pcompact,1,&cols[0],ncol,cols.data(),vals.data(),INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(P.Pcompact,MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(P.Pcompact,MAT_FINAL_ASSEMBLY));

  double gCoeffMin=0.0,gCoeffMax=0.0,gCoeffSum=0.0,gAnchorMin=0.0,gAnchorMax=0.0,gAnchorSum=0.0;
  double gExactDiag=0.0,gCompactDiag=0.0,gExactOffAbs=0.0,gEnergyEdge=0.0;
  PetscInt gInternal=0,gOutlet=0,gPositivePair=0,gNonfinite=0;
  PetscCallMPI(MPI_Allreduce(&localCoeffMin,&gCoeffMin,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localCoeffMax,&gCoeffMax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localCoeffSum,&gCoeffSum,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localAnchorMin,&gAnchorMin,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localAnchorMax,&gAnchorMax,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localAnchorSum,&gAnchorSum,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localExactDiag,&gExactDiag,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localCompactDiag,&gCompactDiag,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localExactOffAbs,&gExactOffAbs,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localEnergyEdge,&gEnergyEdge,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localInternal,&gInternal,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localOutlet,&gOutlet,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localPositivePair,&gPositivePair,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localNonfinite,&gNonfinite,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));

  const double traceScale=(gCompactDiag>0.0)?gExactDiag/gCompactDiag:1.0;
  if(!(traceScale>0.0) || !std::isfinite(traceScale)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"Gate-9G invalid FE-face trace scale");
  PetscCall(MatScale(P.Pcompact,traceScale));
  MatInfo mi{}; PetscCall(MatGetInfo(P.Pcompact,MAT_GLOBAL_SUM,&mi));
  PetscBool sym=PETSC_FALSE; PetscCall(MatIsSymmetric(P.Pcompact,1e-12,&sym));
  if(!sym) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate-9G FE-face compact Kp lost symmetry");

  audit.coeffMin=(PetscReal)(gInternal?gCoeffMin*traceScale:0.0);
  audit.coeffMax=(PetscReal)(gCoeffMax*traceScale);
  audit.coeffMean=(PetscReal)(gInternal?gCoeffSum*traceScale/(double)gInternal:0.0);
  audit.anchorMin=(PetscReal)(gOutlet?gAnchorMin*traceScale:0.0);
  audit.anchorMax=(PetscReal)(gAnchorMax*traceScale);
  audit.anchorMean=(PetscReal)(gOutlet?gAnchorSum*traceScale/(double)gOutlet:0.0);
  audit.exactDiagSum=(PetscReal)gExactDiag; audit.compactDiagSumBeforeScale=(PetscReal)gCompactDiag; audit.traceScale=(PetscReal)traceScale;
  audit.exactSharedOffAbs=(PetscReal)gExactOffAbs; audit.energyEdgeSum=(PetscReal)(gEnergyEdge*traceScale);
  audit.internalFaces=gInternal/2; audit.outletFaces=gOutlet; audit.positiveExactPair=gPositivePair/2; audit.nonfinite=gNonfinite;

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_GATE9G_FE_FACE_KP_VALUES rows=%" PetscInt_FMT " nnz=%.0f avgNnzPerRow=%.6f internalFaces=%" PetscInt_FMT " outletFaces=%" PetscInt_FMT " coeffMin=%.12e coeffMean=%.12e coeffMax=%.12e anchorMin=%.12e anchorMean=%.12e anchorMax=%.12e exactDiagSum=%.12e compactDiagBeforeScale=%.12e traceScale=%.12e positiveExactSharedPair=%" PetscInt_FMT " exactSharedOffAbs=%.12e energyEdgeSumScaled=%.12e symmetric=%d nonfinite=%" PetscInt_FMT " formula=quarter_shared_FE_jump_energy graph=cell_plus_face_neighbours exactOperator=UNCHANGED_custom_FP64_B_rAU_Bt\n",
    nc,mi.nz_used,mi.nz_used/(double)nc,audit.internalFaces,gOutlet,(double)audit.coeffMin,(double)audit.coeffMean,(double)audit.coeffMax,
    (double)audit.anchorMin,(double)audit.anchorMean,(double)audit.anchorMax,(double)audit.exactDiagSum,(double)audit.compactDiagSumBeforeScale,
    (double)audit.traceScale,audit.positiveExactPair,(double)audit.exactSharedOffAbs,(double)audit.energyEdgeSum,(int)sym,audit.nonfinite));
  if(gNonfinite) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"Gate-9G FE-face Kp contained invalid conductances");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode updatePressureSchurFast(const Mesh& M,const Discrete& D,int rank,const GhostPlan& G,Vec rAU,const std::string& pPmatMode,double feFvP1Strength,double fvLsqStrength,double fvLsqTpfaFloor,PetscBool assembleFullSchur,PressureAssemblyPlan& P) {
  PetscFunctionBeginUser;
  PetscCall(VecGhostUpdateBegin(rAU,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecGhostUpdateEnd(rAU,INSERT_VALUES,SCATTER_FORWARD));
  Vec rloc=nullptr;
  const PetscScalar *ra=nullptr;
  PetscCall(VecGhostGetLocalForm(rAU,&rloc));
  PetscCall(VecGetArrayRead(rloc,&ra));

  // For the FV/LSQ surrogate, compute the exact Schur diagonal on owned cells
  // from the local FE rAU response, then exchange only face-neighbour pressure
  // diagonals.  This fixes the previous bug where exactCellSchurDiag(L) tried
  // to read all velocity DOFs of an off-rank neighbour L through G.
  const PetscScalar *fvDiagLocalArray=nullptr;
  Vec fvDiagLocalForm=nullptr;
  if(pPmatMode=="fv_lsq") {
    PetscScalar *da=nullptr;
    PetscCall(VecSet(P.fvCellDiag,0.0));
    PetscCall(VecGetArray(P.fvCellDiag,&da));
    for(PetscInt K=0;K<(PetscInt)M.tets.size();++K) if(D.cellOwner[K]==rank) {
      double diagK=0.0;
      for(int a=0;a<8;++a) {
        const PetscInt entity=(a<4)?M.tets[K][a]:((PetscInt)M.points.size()+M.oppFace[K][a-4]);
        const PetscInt gid=D.g2free[entity]; if(gid<0) continue;
        const PetscInt li=velocityLocalIndex(G,gid); // K is owned: guaranteed in G
        double b2=0.0;
        for(int d=0;d<3;++d) { const double b=P.Bcell[K][d*8+a]; b2+=b*b; }
        diagK += PetscRealPart(ra[li])*b2;
      }
      const PetscInt liP=D.pGid[K]-P.fvCellDiagStart;
      if(liP<0 || liP>=D.cellCount[rank])
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"owned pressure gid/layout mismatch in fv_lsq diagonal exchange");
      da[liP]=(PetscScalar)diagK;
    }
    PetscCall(VecRestoreArray(P.fvCellDiag,&da));
    PetscCall(VecGhostUpdateBegin(P.fvCellDiag,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecGhostUpdateEnd(P.fvCellDiag,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecGhostGetLocalForm(P.fvCellDiag,&fvDiagLocalForm));
    PetscCall(VecGetArrayRead(fvDiagLocalForm,&fvDiagLocalArray));
  }

  if(assembleFullSchur) {
    if(!P.S) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"explicit Schur assembly requested but expanded structure was not built");
    PetscCall(MatZeroEntries(P.S)); std::vector<PetscInt> cols; std::vector<PetscScalar> vals;
    if(pPmatMode=="full") {
      for(std::size_t r=0;r<P.flatRowGid.size();++r) { const PetscInt c0=P.flatRowColOffset[r],c1=P.flatRowColOffset[r+1],n=c1-c0; vals.resize((std::size_t)n);
        for(PetscInt jj=0;jj<n;++jj){const PetscInt cj=c0+jj,t0=P.flatColTermOffset[(std::size_t)cj],t1=P.flatColTermOffset[(std::size_t)cj+1];double v=0;for(PetscInt t=t0;t<t1;++t)v+=PetscRealPart(ra[P.flatTermLocalRau[(std::size_t)t]])*P.flatTermCoeff[(std::size_t)t];vals[(std::size_t)jj]=(PetscScalar)v;}
        if(n) PetscCall(MatSetValues(P.S,1,&P.flatRowGid[r],n,P.flatColGid.data()+c0,vals.data(),INSERT_VALUES)); }
    } else {
      for(const auto& rp:P.rows){const PetscInt n=(PetscInt)rp.columns.size();cols.resize(n);vals.resize(n);for(PetscInt j=0;j<n;++j){cols[j]=rp.columns[j].col;double v=0;for(const auto& term:rp.columns[j].terms)v+=PetscRealPart(ra[term.localRau])*term.coeff;vals[j]=(PetscScalar)v;}if(n)PetscCall(MatSetValues(P.S,1,&rp.row,n,cols.data(),vals.data(),INSERT_VALUES));}
    }
  }

  // Full Pmat mode does not use Pcompact at all.  In the factored-operator +
  // lagged-full-GAMG path this routine is called only on PC refresh steps, so
  // avoid rebuilding an otherwise unused 5-point surrogate.
  if(pPmatMode=="full") {
    PetscCall(VecRestoreArrayRead(rloc,&ra));
    PetscCall(VecGhostRestoreLocalForm(rAU,&rloc));
    if(assembleFullSchur) {
      PetscCall(MatAssemblyBegin(P.S,MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(P.S,MAT_FINAL_ASSEMBLY));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  // Build a compact GAMG P-matrix from the same rAU values.
  // compact_face: historical truncation: exact diagonal, BF3 face off-diagonal only.
  // fe_fv_face: independently constructed FE-informed FV-like surrogate.  The
  //   BF3 face contribution is retained exactly, while the P1 vertex diagonal
  //   energy is conservatively redistributed onto physical internal faces.
  //   For a cell K and one of its P1 vertices v,
  //       d_Kv = rAU_v |B_Kv|^2 .
  //   d_Kv is divided equally among K's internal faces that contain v.  On a
  //   shared face f=K|L, the symmetric P1 graph weight is
  //       w_f^P1 = strength * min(q_Kf,q_Lf) >= 0.
  //   Since sum_f q_Kf <= sum_v d_Kv, the redistributed P1 graph Laplacian
  //   cannot consume more than the exact P1 diagonal energy.  Any unmatched
  //   energy remains on the exact Schur diagonal.  Thus this is a compact,
  //   conservative SPD surrogate rather than a truncation of the dense P1
  //   vertex-star clique.
  PetscCall(MatZeroEntries(P.Pcompact));
  const PetscInt nv=(PetscInt)M.points.size(), ni=(PetscInt)M.neighbour.size(), nc=(PetscInt)M.tets.size();
  std::array<PetscInt,5> ccols{};
  std::array<PetscScalar,5> cvals{};
  double localP1Diag=0.0, localP1Degree=0.0, localBF3OffAbs=0.0;
  double localFvDiag=0.0,localFvDegree=0.0,localFvLsqGeom=0.0,localFvTpfaGeom=0.0,localFvCos=0.0;
  PetscInt localFvFaces=0; double localFvMinCos=1.0;

  auto vertexDiagContribution = [&](PetscInt C,int localVertex)->double {
    const PetscInt entity=M.tets[C][localVertex];
    const PetscInt gid=D.g2free[entity];
    if(gid<0) return 0.0;
    const PetscInt li=velocityLocalIndex(G,gid);
    double b2=0.0;
    for(int d=0;d<3;++d) { const double b=P.Bcell[C][d*8+localVertex]; b2+=b*b; }
    return PetscRealPart(ra[li])*b2;
  };
  auto internalFaceDegreeAtVertex = [&](PetscInt C,int localVertex)->int {
    int deg=0;
    // Local face i is opposite local vertex i, so it contains localVertex iff i!=localVertex.
    for(int i=0;i<4;++i) if(i!=localVertex && M.oppFace[C][i]<ni) ++deg;
    return deg;
  };
  auto p1DirectedFaceShare = [&](PetscInt C,PetscInt face)->double {
    int opp=-1;
    for(int i=0;i<4;++i) if(M.oppFace[C][i]==face) { opp=i; break; }
    if(opp<0) return 0.0;
    double q=0.0;
    for(int lv=0;lv<4;++lv) if(lv!=opp) {
      const int deg=internalFaceDegreeAtVertex(C,lv);
      if(deg>0) q += vertexDiagContribution(C,lv)/(double)deg;
    }
    return q;
  };

  auto exactCellSchurDiag = [&](PetscInt C)->double {
    if(pPmatMode=="fv_lsq") {
      const PetscInt pg=D.pGid[C];
      const auto it=P.fvCellDiagLocal.find(pg);
      if(it==P.fvCellDiagLocal.end())
        throw std::runtime_error("fv_lsq pressure diagonal ghost missing face-neighbour cell");
      return PetscRealPart(fvDiagLocalArray[it->second]);
    }
    double diagC=0.0;
    for(int a=0;a<8;++a) {
      const PetscInt entity=(a<4)?M.tets[C][a]:(nv+M.oppFace[C][a-4]);
      const PetscInt gid=D.g2free[entity]; if(gid<0) continue;
      const PetscInt li=velocityLocalIndex(G,gid);
      double b2=0.0; for(int d=0;d<3;++d) { const double b=P.Bcell[C][d*8+a]; b2+=b*b; }
      diagC += PetscRealPart(ra[li])*b2;
    }
    return diagC;
  };
  auto fvRawGeom = [&](PetscInt C,int lf)->double {
    return std::max(P.fvLsqGeom[C][lf],fvLsqTpfaFloor*P.fvTpfaGeom[C][lf]);
  };
  auto fvGeomDegree = [&](PetscInt C)->double {
    double z=0.0; for(int i=0;i<4;++i) if(M.oppFace[C][i]<ni) z+=fvRawGeom(C,i); return z;
  };
  auto fvFaceWeight = [&](PetscInt K,int i,PetscInt L,PetscInt f)->double {
    int j=-1; for(int q=0;q<4;++q) if(M.oppFace[L][q]==f) {j=q;break;}
    if(j<0) return 0.0;
    const double dK=exactCellSchurDiag(K), dL=exactCellSchurDiag(L);
    const double gK=fvGeomDegree(K), gL=fvGeomDegree(L);
    if(!(dK>0.0 && dL>0.0 && gK>0.0 && gL>0.0)) return 0.0;
    const double wK=fvLsqStrength*dK*fvRawGeom(K,i)/gK;
    const double wL=fvLsqStrength*dL*fvRawGeom(L,j)/gL;
    return std::max(0.0,std::min(wK,wL));
  };

  for(PetscInt K=0;K<nc;++K) if(D.cellOwner[K]==rank) {
    PetscInt ncol=1; ccols[0]=D.pGid[K]; cvals[0]=0.0;
    double p1DiagK=0.0;
    if(pPmatMode=="fv_lsq") {
      cvals[0]=(PetscScalar)exactCellSchurDiag(K);
      localFvDiag += PetscRealPart(cvals[0]);
    } else {
      // Exact full Schur diagonal: sum_a rAU_a |B_Ka|^2.
      for(int a=0;a<8;++a) {
        const PetscInt entity=(a<4)?M.tets[K][a]:(nv+M.oppFace[K][a-4]);
        const PetscInt gid=D.g2free[entity];
        if(gid<0) continue;
        const PetscInt li=velocityLocalIndex(G,gid);
        double b2=0.0; for(int d=0;d<3;++d) { const double b=P.Bcell[K][d*8+a]; b2+=b*b; }
        const double da=PetscRealPart(ra[li])*b2;
        cvals[0] += (PetscScalar)da;
        if(a<4) p1DiagK += da;
      }
      localP1Diag += p1DiagK;
    }

    // Face-neighbour off-diagonals: exact BF3 contribution, plus optional
    // conservative P1 vertex-energy redistribution for fe_fv_face.
    for(int i=0;i<4;++i) {
      const PetscInt f=M.oppFace[K][i];
      if(f>=ni) continue;
      const PetscInt entity=nv+f;
      const PetscInt gid=D.g2free[entity];
      if(gid<0) continue;
      const PetscInt L=(M.owner[f]==K)?M.neighbour[f]:M.owner[f];
      const int b=localBasisForEntity(M,L,entity);
      if(b<4) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"compact BF3 face/local-basis mismatch");
      const PetscInt li=velocityLocalIndex(G,gid);
      double dot=0.0;
      for(int d=0;d<3;++d) dot += P.Bcell[K][d*8+4+i]*P.Bcell[L][d*8+b];
      const double bf3off=PetscRealPart(ra[li])*dot;
      double off=bf3off;
      localBF3OffAbs += std::abs(bf3off);
      if(pPmatMode=="fe_fv_face" && feFvP1Strength>0.0) {
        const double qK=p1DirectedFaceShare(K,f);
        const double qL=p1DirectedFaceShare(L,f);
        const double w=feFvP1Strength*std::max(0.0,std::min(qK,qL));
        off -= w;                    // graph-Laplacian face coupling
        localP1Degree += w;          // row degree; each global face counts twice overall
      } else if(pPmatMode=="fv_lsq") {
        // A genuinely independent compact FV-like pressure operator: the
        // exact FE/BF3 off-diagonal is NOT used.  One-ring LSQ geometry sets
        // the relative face conductances; the FE rAU response enters through
        // exact-cell-Schur-diagonal normalization.  min(one-sided weights)
        // makes the assembled graph symmetric and guarantees row degree <=
        // fvLsqStrength*exact diagonal for 0<=strength<=1.
        const double w=fvFaceWeight(K,i,L,f);
        off=-w;
        localFvDegree+=w;
        localFvLsqGeom+=P.fvLsqGeom[K][i];
        localFvTpfaGeom+=P.fvTpfaGeom[K][i];
        localFvCos+=P.fvNonorthCos[K][i];
        localFvMinCos=std::min(localFvMinCos,P.fvNonorthCos[K][i]);
        ++localFvFaces;
      }
      ccols[ncol]=D.pGid[L]; cvals[ncol]=(PetscScalar)off; ++ncol;
    }
    PetscCall(MatSetValues(P.Pcompact,1,&ccols[0],ncol,ccols.data(),cvals.data(),INSERT_VALUES));
  }

  if(pPmatMode=="fe_fv_face") {
    static PetscInt reportCount=0;
    if(reportCount<3) {
      double gP1Diag=0.0,gP1Degree=0.0,gBF3OffAbs=0.0;
      PetscCallMPI(MPI_Allreduce(&localP1Diag,&gP1Diag,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
      PetscCallMPI(MPI_Allreduce(&localP1Degree,&gP1Degree,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
      PetscCallMPI(MPI_Allreduce(&localBF3OffAbs,&gBF3OffAbs,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "P1BF3_FE_FV_PMAT strength=%.6g p1DiagSum=%.12e redistributedRowDegree=%.12e redistributedFraction=%.6f bf3OffAbsRowSum=%.12e graph=cell_plus_face_neighbours\n",
        feFvP1Strength,gP1Diag,gP1Degree,gP1Diag>0.0?gP1Degree/gP1Diag:0.0,gBF3OffAbs));
      ++reportCount;
    }
  }

  if(pPmatMode=="fv_lsq") {
    static PetscInt fvReportCount=0;
    if(fvReportCount<3) {
      double gDiag=0.0,gDegree=0.0,gLsq=0.0,gTpfa=0.0,gCos=0.0,gMinCos=0.0;
      PetscInt gFaces=0;
      PetscCallMPI(MPI_Allreduce(&localFvDiag,&gDiag,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
      PetscCallMPI(MPI_Allreduce(&localFvDegree,&gDegree,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
      PetscCallMPI(MPI_Allreduce(&localFvLsqGeom,&gLsq,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
      PetscCallMPI(MPI_Allreduce(&localFvTpfaGeom,&gTpfa,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
      PetscCallMPI(MPI_Allreduce(&localFvCos,&gCos,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
      PetscCallMPI(MPI_Allreduce(&localFvFaces,&gFaces,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
      PetscCallMPI(MPI_Allreduce(&localFvMinCos,&gMinCos,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "P1BF3_LSQ_FV_PMAT strength=%.6g tpfaFloor=%.6g exactDiagSum=%.12e graphDegreeSum=%.12e degreeToDiag=%.6f directedFaces=%" PetscInt_FMT " meanLsqGeom=%.12e meanTpfaGeom=%.12e meanNonorthCos=%.6f minNonorthCos=%.6f graph=cell_plus_face_neighbours normalization=FE_exact_Schur_diag\n",
        fvLsqStrength,fvLsqTpfaFloor,gDiag,gDegree,gDiag>0.0?gDegree/gDiag:0.0,gFaces,
        gFaces?gLsq/(double)gFaces:0.0,gFaces?gTpfa/(double)gFaces:0.0,gFaces?gCos/(double)gFaces:0.0,gMinCos));
      ++fvReportCount;
    }
  }

  if(pPmatMode=="fv_lsq") {
    PetscCall(VecRestoreArrayRead(fvDiagLocalForm,&fvDiagLocalArray));
    PetscCall(VecGhostRestoreLocalForm(P.fvCellDiag,&fvDiagLocalForm));
  }
  PetscCall(VecRestoreArrayRead(rloc,&ra));
  PetscCall(VecGhostRestoreLocalForm(rAU,&rloc));
  if(assembleFullSchur) {
    PetscCall(MatAssemblyBegin(P.S,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(P.S,MAT_FINAL_ASSEMBLY));
  }
  PetscCall(MatAssemblyBegin(P.Pcompact,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P.Pcompact,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode destroyPressureAssemblyPlan(PressureAssemblyPlan& P) {
  PetscFunctionBeginUser;
  PetscCall(MatDestroy(&P.S));
  PetscCall(MatDestroy(&P.Pcompact));
  PetscCall(VecDestroy(&P.fvCellDiag));
  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode profileFillVector(Vec v);
static PetscErrorCode profileMaxSeconds(PetscLogDouble local,double *globalMax);

struct FactoredSchurContext {
  Mat B[3]={nullptr,nullptr,nullptr};
  Vec rAU=nullptr;
  Vec velocityWork=nullptr;
  Vec pressureWork=nullptr;
};

static PetscErrorCode factoredSchurMult(Mat A,Vec x,Vec y) {
  PetscFunctionBeginUser;
  FactoredSchurContext *C=nullptr;
  PetscCall(MatShellGetContext(A,&C));
  PetscCall(VecSet(y,0.0));
  for(int d=0;d<3;++d) {
    PetscCall(MatMultTranspose(C->B[d],x,C->velocityWork));
    PetscCall(VecPointwiseMult(C->velocityWork,C->velocityWork,C->rAU));
    PetscCall(MatMult(C->B[d],C->velocityWork,C->pressureWork));
    PetscCall(VecAXPY(y,1.0,C->pressureWork));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode createFactoredSchur(const Discrete& D,Vec rAU,Vec pressureTemplate,FactoredSchurContext& C,Mat *A) {
  PetscFunctionBeginUser;
  for(int d=0;d<3;++d) C.B[d]=D.B[d];
  C.rAU=rAU;
  PetscCall(VecDuplicate(D.rhs[0],&C.velocityWork));
  PetscCall(VecDuplicate(pressureTemplate,&C.pressureWork));
  PetscInt plocal=0,pglobal=0;
  PetscCall(VecGetLocalSize(pressureTemplate,&plocal));
  PetscCall(VecGetSize(pressureTemplate,&pglobal));
  PetscCall(MatCreateShell(PETSC_COMM_WORLD,plocal,plocal,pglobal,pglobal,&C,A));
  PetscCall(MatShellSetOperation(*A,MATOP_MULT,(void(*)(void))factoredSchurMult));
  PetscCall(MatSetOption(*A,MAT_SYMMETRIC,PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode destroyFactoredSchurContext(FactoredSchurContext& C) {
  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&C.velocityWork));
  PetscCall(VecDestroy(&C.pressureWork));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode profileOperatorMatMult(Mat A,Vec templateVec,PetscInt reps,double *secondsPerApply) {
  PetscFunctionBeginUser;
  Vec x=nullptr,y=nullptr;
  PetscCall(VecDuplicate(templateVec,&x));
  PetscCall(VecDuplicate(templateVec,&y));
  PetscCall(profileFillVector(x));
  for(int w=0;w<5;++w) PetscCall(MatMult(A,x,y));
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  PetscLogDouble t0,t1; PetscCall(PetscTime(&t0));
  for(PetscInt i=0;i<reps;++i) PetscCall(MatMult(A,x,y));
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  PetscCall(PetscTime(&t1));
  PetscCall(profileMaxSeconds((t1-t0)/(PetscLogDouble)reps,secondsPerApply));
  PetscCall(VecDestroy(&x)); PetscCall(VecDestroy(&y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode benchmarkFactoredSchur(Mat explicitS,Mat factoredS,const Discrete& D,Vec pressureTemplate,PetscInt expandedNnz,PetscInt reps) {
  PetscFunctionBeginUser;
  Vec x=nullptr,ye=nullptr,yf=nullptr,diff=nullptr;
  PetscCall(VecDuplicate(pressureTemplate,&x)); PetscCall(VecDuplicate(pressureTemplate,&ye));
  PetscCall(VecDuplicate(pressureTemplate,&yf)); PetscCall(VecDuplicate(pressureTemplate,&diff));
  PetscCall(profileFillVector(x));
  PetscCall(MatMult(explicitS,x,ye)); PetscCall(MatMult(factoredS,x,yf));
  PetscCall(VecWAXPY(diff,-1.0,ye,yf));
  PetscReal en=0.0,dn=0.0,di=0.0;
  PetscCall(VecNorm(ye,NORM_2,&en)); PetscCall(VecNorm(diff,NORM_2,&dn)); PetscCall(VecNorm(diff,NORM_INFINITY,&di));
  double explicitSec=0.0,factoredSec=0.0;
  PetscCall(profileOperatorMatMult(explicitS,pressureTemplate,reps,&explicitSec));
  PetscCall(profileOperatorMatMult(factoredS,pressureTemplate,reps,&factoredSec));
  double bNnz=0.0;
  for(int d=0;d<3;++d) if(D.B[d]) { MatInfo bi; PetscCall(MatGetInfo(D.B[d],MAT_GLOBAL_SUM,&bi)); bNnz+=bi.nz_used; }
  const double traversed=2.0*bNnz;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_FACTORED_SCHUR_CHECK relL2=%.12e absInf=%.12e explicitNorm=%.12e status=%s semantics=sum_d_Bd_rAU_BdT_no_extra_solve\n",
    (double)(en>0?dn/en:dn),(double)di,(double)en,(en>0?dn/en:dn)<5e-12?"PASS":"FAIL"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_FACTORED_SCHUR_PROFILE reps=%" PetscInt_FMT " explicitExpandedNnz=%" PetscInt_FMT " BxyzStoredNnz=%.0f factoredSparseEntriesTraversed=%.0f explicitMatMultMs=%.6f factoredMatMultMs=%.6f speedupExplicitOverFactored=%.6f factoredOverExplicit=%.6f\n",
    reps,expandedNnz,bNnz,traversed,1e3*explicitSec,1e3*factoredSec,explicitSec/factoredSec,factoredSec/explicitSec));
  PetscCall(VecDestroy(&x)); PetscCall(VecDestroy(&ye)); PetscCall(VecDestroy(&yf)); PetscCall(VecDestroy(&diff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode profileFillVector(Vec v) {
  PetscFunctionBeginUser;
  PetscInt lo=0,hi=0;
  PetscScalar *a=nullptr;
  PetscCall(VecGetOwnershipRange(v,&lo,&hi));
  PetscCall(VecGetArray(v,&a));
  for(PetscInt i=lo;i<hi;++i) {
    const double x=(double)(i+1);
    a[i-lo]=(PetscScalar)(std::sin(0.013*x)+0.37*std::cos(0.007*x));
  }
  PetscCall(VecRestoreArray(v,&a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode profileMaxSeconds(PetscLogDouble local,double *globalMax) {
  PetscFunctionBeginUser;
  double x=(double)local,mx=0.0;
  PetscCallMPI(MPI_Allreduce(&x,&mx,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD));
  *globalMax=mx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode profileMatMult(Mat A,PetscInt reps,double *secondsPerApply) {
  PetscFunctionBeginUser;
  Vec x=nullptr,y=nullptr;
  PetscCall(MatCreateVecs(A,&x,&y));
  PetscCall(profileFillVector(x));
  for(int w=0;w<3;++w) PetscCall(MatMult(A,x,y));
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  PetscLogDouble t0,t1; PetscCall(PetscTime(&t0));
  for(PetscInt i=0;i<reps;++i) { PetscCall(MatMult(A,x,y)); std::swap(x,y); }
  PetscCall(PetscTime(&t1));
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  double mx=0; PetscCall(profileMaxSeconds(t1-t0,&mx));
  *secondsPerApply=(reps>0)?mx/(double)reps:0.0;
  PetscCall(VecDestroy(&x)); PetscCall(VecDestroy(&y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode profilePCApply(PC pc,Mat A,PetscInt reps,double *secondsPerApply) {
  PetscFunctionBeginUser;
  Vec x=nullptr,y=nullptr;
  PetscCall(MatCreateVecs(A,&x,&y));
  PetscCall(profileFillVector(x));
  for(int w=0;w<3;++w) PetscCall(PCApply(pc,x,y));
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  PetscLogDouble t0,t1; PetscCall(PetscTime(&t0));
  for(PetscInt i=0;i<reps;++i) { PetscCall(PCApply(pc,x,y)); std::swap(x,y); }
  PetscCall(PetscTime(&t1));
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  double mx=0; PetscCall(profileMaxSeconds(t1-t0,&mx));
  *secondsPerApply=(reps>0)?mx/(double)reps:0.0;
  PetscCall(VecDestroy(&x)); PetscCall(VecDestroy(&y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode profileKSPSolve(KSP ksp,Mat A,PetscInt reps,double *secondsPerSolve) {
  PetscFunctionBeginUser;
  Vec b=nullptr,x=nullptr;
  PetscCall(MatCreateVecs(A,&x,&b));
  PetscCall(profileFillVector(b));
  for(int w=0;w<2;++w) { PetscCall(VecSet(x,0)); PetscCall(KSPSetInitialGuessNonzero(ksp,PETSC_FALSE)); PetscCall(KSPSolve(ksp,b,x)); }
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  PetscLogDouble t0,t1; PetscCall(PetscTime(&t0));
  for(PetscInt i=0;i<reps;++i) { PetscCall(VecSet(x,0)); PetscCall(KSPSetInitialGuessNonzero(ksp,PETSC_FALSE)); PetscCall(KSPSolve(ksp,b,x)); }
  PetscCall(PetscTime(&t1));
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  double mx=0; PetscCall(profileMaxSeconds(t1-t0,&mx));
  *secondsPerSolve=(reps>0)?mx/(double)reps:0.0;
  PetscCall(VecDestroy(&b)); PetscCall(VecDestroy(&x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode pressureAMGProfile(KSP pksp,Mat S,Mat PmatFine,PetscInt fvCompactNnz,
                                          PetscInt fineReps,PetscInt pcReps,
                                          PetscInt cgIters,PetscInt cgReps,
                                          PetscInt levelMatReps,PetscInt levelSolveReps) {
  PetscFunctionBeginUser;
  PC pc=nullptr; const char *pct=nullptr,*kspt=nullptr;
  PetscCall(KSPGetPC(pksp,&pc)); PetscCall(PCGetType(pc,&pct)); PetscCall(KSPGetType(pksp,&kspt));
  PetscInt nFine=0,mFine=0,nlocFine=0; MatInfo finfo;
  PetscCall(MatGetSize(S,&nFine,&mFine)); PetscCall(MatGetLocalSize(S,&nlocFine,nullptr));
  PetscCall(MatGetInfo(S,MAT_GLOBAL_SUM,&finfo));
  MatInfo pminfo; PetscCall(MatGetInfo(PmatFine,MAT_GLOBAL_SUM,&pminfo));
  const double fineNnz=finfo.nz_used, pmatFineNnz=pminfo.nz_used;
  const double fvNnz=(double)fvCompactNnz;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_PRESSURE_PROFILE_FINE ksp=%s pc=%s rows=%" PetscInt_FMT " nnz=%.0f avgNnzPerRow=%.6f fvCompactFaceNnz=%" PetscInt_FMT " pressureNnzRatioToCompactFV=%.6f\n",
    kspt?kspt:"?",pct?pct:"?",nFine,fineNnz,nFine?fineNnz/(double)nFine:0.0,fvCompactNnz,fvNnz>0?fineNnz/fvNnz:0.0));

  PetscInt levels=0; PetscReal gc=0,oc=0;
  PetscCall(PCMGGetLevels(pc,&levels));
  PetscCall(PCMGGetGridComplexity(pc,&gc,&oc));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_PRESSURE_PROFILE_COMPLEXITY levels=%" PetscInt_FMT " gridComplexity=%.8f operatorComplexity=%.8f hierarchyFinePmatNnz=%.0f hierarchyTotalNnz=%.0f\n",
    levels,(double)gc,(double)oc,pmatFineNnz,(double)oc*pmatFineNnz));

  double fineMatSec=0; PetscCall(profileMatMult(S,fineReps,&fineMatSec));
  double pcSec=0; PetscCall(profilePCApply(pc,PmatFine,pcReps,&pcSec));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_PRESSURE_PROFILE_KERNEL fineMatMultReps=%" PetscInt_FMT " fineMatMultMs=%.6f fineMatMultNsPerNnz=%.6f pcApplyReps=%" PetscInt_FMT " gamgPcApplyMs=%.6f pcApplyOverFineMatMult=%.6f\n",
    fineReps,1e3*fineMatSec,fineNnz>0?1e9*fineMatSec/fineNnz:0.0,pcReps,1e3*pcSec,fineMatSec>0?pcSec/fineMatSec:0.0));

  double sumLevelNnz=0.0;
  for(PetscInt lev=0;lev<levels;++lev) {
    KSP lksp=nullptr;
    if(lev==0) PetscCall(PCMGGetCoarseSolve(pc,&lksp));
    else PetscCall(PCMGGetSmoother(pc,lev,&lksp));
    Mat A=nullptr,Pmat=nullptr;
    PetscCall(KSPGetOperators(lksp,&A,&Pmat));
    if(!Pmat) Pmat=A;
    PetscInt N=0,nloc=0,maxit=0; PetscReal rtol=0,atol=0,dtol=0; MatInfo info;
    PetscCall(MatGetSize(Pmat,&N,nullptr)); PetscCall(MatGetLocalSize(Pmat,&nloc,nullptr));
    PetscCall(MatGetInfo(Pmat,MAT_GLOBAL_SUM,&info)); sumLevelNnz += info.nz_used;
    PetscInt nlocMin=0,nlocMax=0;
    PetscCallMPI(MPI_Allreduce(&nloc,&nlocMin,1,MPIU_INT,MPI_MIN,PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(&nloc,&nlocMax,1,MPIU_INT,MPI_MAX,PETSC_COMM_WORLD));
    const char *lkt=nullptr,*lpct=nullptr; PC lpc=nullptr;
    PetscCall(KSPGetType(lksp,&lkt)); PetscCall(KSPGetPC(lksp,&lpc)); PetscCall(PCGetType(lpc,&lpct));
    PetscCall(KSPGetTolerances(lksp,&rtol,&atol,&dtol,&maxit));
    double mm=0,ss=0; PetscCall(profileMatMult(Pmat,levelMatReps,&mm));
    PetscCall(profileKSPSolve(lksp,Pmat,levelSolveReps,&ss));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_PRESSURE_PROFILE_LEVEL level=%" PetscInt_FMT " role=%s rows=%" PetscInt_FMT " localRowsMin=%" PetscInt_FMT " localRowsMax=%" PetscInt_FMT " nnz=%.0f avgNnzPerRow=%.6f ksp=%s pc=%s smootherMaxIt=%" PetscInt_FMT " matMultMs=%.6f levelSolveMs=%.6f\n",
      lev,lev==0?"coarse":"smooth",N,nlocMin,nlocMax,info.nz_used,N?info.nz_used/(double)N:0.0,lkt?lkt:"?",lpct?lpct:"?",maxit,1e3*mm,1e3*ss));
    if(lev>0) {
      Mat I=nullptr; PetscCall(PCMGGetInterpolation(pc,lev,&I));
      if(I) { MatInfo ii; PetscInt ir=0,ic=0; PetscCall(MatGetInfo(I,MAT_GLOBAL_SUM,&ii)); PetscCall(MatGetSize(I,&ir,&ic));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
          "P1BF3_PRESSURE_PROFILE_TRANSFER fineLevel=%" PetscInt_FMT " rows=%" PetscInt_FMT " cols=%" PetscInt_FMT " nnz=%.0f avgNnzPerRow=%.6f\n",
          lev,ir,ic,ii.nz_used,ir?ii.nz_used/(double)ir:0.0)); }
    }
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_PRESSURE_PROFILE_COMPLEXITY_CHECK summedLevelNnz=%.0f directOperatorComplexity=%.8f petscOperatorComplexity=%.8f\n",
    sumLevelNnz,pmatFineNnz>0?sumLevelNnz/pmatFineNnz:0.0,(double)oc));

  // Fixed-count CG using exactly the already-built/frozen GAMG PC.  This
  // measures fine SpMV + vector reductions + one PCApply per Krylov step.
  KSP probe=nullptr; PetscCall(KSPCreate(PETSC_COMM_WORLD,&probe));
  PetscCall(KSPSetOperators(probe,S,PmatFine)); PetscCall(KSPSetType(probe,KSPCG));
  PetscCall(KSPSetPC(probe,pc)); PetscCall(KSPSetReusePreconditioner(probe,PETSC_TRUE));
  PetscCall(KSPSetNormType(probe,KSP_NORM_NONE));
  PetscCall(KSPSetTolerances(probe,PETSC_CURRENT,PETSC_CURRENT,PETSC_CURRENT,cgIters));
  PetscCall(KSPSetConvergenceTest(probe,KSPConvergedSkip,nullptr,nullptr));
  PetscCall(KSPSetUp(probe));
  double cgSec=0; PetscCall(profileKSPSolve(probe,S,cgReps,&cgSec));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_PRESSURE_PROFILE_CG fixedIts=%" PetscInt_FMT " reps=%" PetscInt_FMT " solveMs=%.6f msPerCgIteration=%.6f impliedPcPlusKrylovOverheadMs=%.6f\n",
    cgIters,cgReps,1e3*cgSec,cgIters>0?1e3*cgSec/(double)cgIters:0.0,cgIters>0?1e3*cgSec/(double)cgIters-1e3*pcSec-1e3*fineMatSec:0.0));
  PetscCall(KSPDestroy(&probe));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_PRESSURE_PROFILE_DONE status=PASS\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

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

  D.velCount.assign(size,0); D.g2free.assign(nv+nf,-1);
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

static PetscErrorCode assembleMPI(const Mesh& M, int rank, int size,const ProblemConfig& P, Discrete& D,
  PetscBool buildDiffusionReference=PETSC_FALSE,PetscBool buildBReference=PETSC_FALSE,PetscBool buildMomentumRhsReference=PETSC_FALSE) {
  PetscFunctionBeginUser;
  const PetscInt nv=(PetscInt)M.points.size(), nc=(PetscInt)M.tets.size();
  PetscCall(buildOwnership(M,rank,size,P,D));
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
      PetscCall(MatCreateAIJ(PETSC_COMM_WORLD,nlp,nlv,nc,D.ns,0,bDnnz.data(),0,bOnnz.data(),&D.B[d]));
      PetscCall(MatSetOption(D.B[d],MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));
    }
    if(buildMomentumRhsReference){PetscCall(VecCreateMPI(PETSC_COMM_WORLD,nlv,D.ns,&D.rhs[d])); PetscCall(VecSet(D.rhs[d],0));}
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "P1BF3_M4B_B_STORAGE PETScB=%s physicalB=custom_FP64_MPI referenceDiagNnz=%" PetscInt_FMT " referenceOffdiagNnz=%" PetscInt_FMT "\n",
    buildBReference?"temporary_reference":"never_created_in_production",gbD,gbO));
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD,nlp,nc,&D.volumes)); PetscCall(VecSet(D.volumes,0));
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
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_FE cells=%" PetscInt_FMT " scalarVelDofs=%" PetscInt_FMT " velocityDofs=%" PetscInt_FMT " pressureDofs=%" PetscInt_FMT " freeVertices=%" PetscInt_FMT " freeBF3Faces=%" PetscInt_FMT " totalVolume=%.16e minVol=%.6e maxVol=%.6e hEff=%.12e assembly=%s\n",nc,D.ns,3*D.ns,nc,D.freeVertices,D.freeFaces,gsum,gmin,gmax,std::cbrt(gsum/nc),buildDiffusionReference?(buildBReference?"temporary_DA_plus_B_reference":"temporary_DA_reference_no_B"):(buildBReference?"temporary_B_reference_no_DA":"custom_pressure_B_no_PETSc_B_or_DA")));
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

static PetscErrorCode computeErrorsRoot(const Mesh& M,const Discrete& D,const std::array<std::vector<double>,3>& U,const std::vector<double>& p) {
  PetscFunctionBeginUser;
  const std::vector<double>* ua[3]; for(int d=0;d<3;++d) ua[d]=&U[(std::size_t)d]; auto Q=tetDuffy7(); PetscInt nv=M.points.size();
  double total=0,intdiff=0;
  for(PetscInt c=0;c<(PetscInt)M.tets.size();++c){auto t=M.tets[c];Vec3 X[4]={M.points[t[0]],M.points[t[1]],M.points[t[2]],M.points[t[3]]};double J[3][3]={{X[1].x-X[0].x,X[2].x-X[0].x,X[3].x-X[0].x},{X[1].y-X[0].y,X[2].y-X[0].y,X[3].y-X[0].y},{X[1].z-X[0].z,X[2].z-X[0].z,X[3].z-X[0].z}};double det=det3(J);for(auto&q:Q){double x=0,y=0,z=0;for(int i=0;i<4;++i){x+=q.lam[i]*X[i].x;y+=q.lam[i]*X[i].y;z+=q.lam[i]*X[i].z;}double w=q.w*det;total+=w;intdiff+=(p[(std::size_t)D.pGid[c]]-exactP(x,y,z))*w;}}
  double shift=intdiff/total, ue2=0,un2=0,pe2=0,pn2=0;
  for(PetscInt c=0;c<(PetscInt)M.tets.size();++c){auto t=M.tets[c];Vec3 X[4]={M.points[t[0]],M.points[t[1]],M.points[t[2]],M.points[t[3]]};double J[3][3]={{X[1].x-X[0].x,X[2].x-X[0].x,X[3].x-X[0].x},{X[1].y-X[0].y,X[2].y-X[0].y,X[3].y-X[0].y},{X[1].z-X[0].z,X[2].z-X[0].z,X[3].z-X[0].z}};double det=det3(J);PetscInt lg[8];for(int i=0;i<4;++i)lg[i]=t[i];for(int i=0;i<4;++i)lg[4+i]=nv+M.oppFace[c][i];double coeff[3][8]={{0}};for(int a=0;a<8;++a){PetscInt ia=D.g2free[lg[a]];if(ia>=0)for(int d=0;d<3;++d)coeff[d][a]=(*ua[d])[(std::size_t)ia];}
    for(auto&q:Q){double val[8],gr[8][3];basis(q.lam,val,gr);double x=0,y=0,z=0;for(int i=0;i<4;++i){x+=q.lam[i]*X[i].x;y+=q.lam[i]*X[i].y;z+=q.lam[i]*X[i].z;}double uh[3]={0},ue[3];for(int d=0;d<3;++d)for(int a=0;a<8;++a)uh[d]+=coeff[d][a]*val[a];exactU(x,y,z,ue);double pp=exactP(x,y,z),w=q.w*det;for(int d=0;d<3;++d){ue2+=(uh[d]-ue[d])*(uh[d]-ue[d])*w;un2+=ue[d]*ue[d]*w;}double de=p[(std::size_t)D.pGid[c]]-shift-pp;pe2+=de*de*w;pn2+=pp*pp*w;}}
PetscCall(PetscPrintf(PETSC_COMM_SELF,"P1BF3_MMS U_L2=%.12e U_relL2=%.12e P_shifted_L2=%.12e P_shifted_relL2=%.12e pressureShift=%.12e\n",std::sqrt(ue2),std::sqrt(ue2/un2),std::sqrt(pe2),std::sqrt(pe2/pn2),shift)); PetscFunctionReturn(PETSC_SUCCESS);
}

static double gatheredEntityValue(const Discrete& D,const std::vector<double>& ua,int d,PetscInt entity) {
  const PetscInt gid=D.g2free[entity]; return gid>=0?ua[(std::size_t)gid]:entityDirValue(D,d,entity);
}

static void fixedFaceMean(const Mesh& M,const Discrete& D,PetscInt f,double avg[3]) {
  const PetscInt nv=(PetscInt)M.points.size(); const auto& F=M.faces[f];
  for(int d=0;d<3;++d) {
    avg[d]=(entityDirValue(D,d,F.v[0])+entityDirValue(D,d,F.v[1])+entityDirValue(D,d,F.v[2]))/3.0;
    avg[d]+=(9.0/20.0)*entityDirValue(D,d,nv+f);
  }
}

static PetscErrorCode auditFixedNormalInletRoot(const Mesh& M,const Discrete& D,const ProblemConfig& P) {
  PetscFunctionBeginUser;
  if(P.inletBC!=InletBCMode::FixedNormalSpeed) PetscFunctionReturn(PETSC_SUCCESS);
  const auto& B=P.boundary; const auto& pin=M.patches[B.inlet];
  double flux=0.0,unMin=std::numeric_limits<double>::max(),unMax=-std::numeric_limits<double>::max();
  double meanVectorErrMax=0.0;
  for(PetscInt f=pin.startFace;f<pin.startFace+pin.nFaces;++f) {
    const Vec3 sf=faceOutwardAreaVector(M,f); const double area=norm3(sf); const Vec3 nf=scale3(sf,1.0/area);
    double avg[3]; fixedFaceMean(M,D,f,avg); const Vec3 u{avg[0],avg[1],avg[2]};
    flux += dot3(u,sf); const double un=dot3(u,nf); unMin=std::min(unMin,un); unMax=std::max(unMax,un);
    meanVectorErrMax=std::max(meanVectorErrMax,norm3(sub3(u,B.inletVelocity)));
  }
  const double target=B.signedNormalSpeed*B.inletProjectedArea;
  const double relErr=std::abs(flux-target)/std::max(std::abs(target),1e-300);
  PetscCall(PetscPrintf(PETSC_COMM_SELF,
    "P1BF3_INLET_BC_AUDIT patch=%s mode=fixed_normal_speed normalMode=average_patch_normal signedSpeed=%.12e referenceNormal=[%.12e,%.12e,%.12e] U=[%.12e,%.12e,%.12e] area=%.12e projectedArea=%.12e planarityRatio=%.12e targetFlux=%.12e imposedFlux=%.12e relFluxError=%.12e faceNormalVelocityMin=%.12e faceNormalVelocityMax=%.12e faceMeanVectorErrorMax=%.12e status=%s\n",
    B.inletPatch.c_str(),B.signedNormalSpeed,B.inletReferenceNormal.x,B.inletReferenceNormal.y,B.inletReferenceNormal.z,
    B.inletVelocity.x,B.inletVelocity.y,B.inletVelocity.z,B.inletArea,B.inletProjectedArea,B.inletProjectedArea/B.inletArea,
    target,flux,relErr,unMin,unMax,meanVectorErrMax,(relErr<1e-12 && meanVectorErrMax<1e-12)?"PASS":"CHECK"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

struct PatchSolutionStats {
  double area=0.0,flux=0.0,pOwnerMean=0.0,normalVelocityMin=std::numeric_limits<double>::max(),normalVelocityMax=-std::numeric_limits<double>::max();
};

static PatchSolutionStats patchSolutionStatsRoot(const Mesh& M,const Discrete& D,int pi,const std::vector<double>* ua[3],const double* pa) {
  PatchSolutionStats S; const PetscInt nv=(PetscInt)M.points.size(); const auto& P=M.patches[pi]; double psum=0.0;
  for(PetscInt f=P.startFace;f<P.startFace+P.nFaces;++f) {
    const auto& F=M.faces[f]; const Vec3 sf=faceOutwardAreaVector(M,f); const double area=norm3(sf); const Vec3 nf=scale3(sf,1.0/area);
    double avg[3]={0,0,0};
    for(int d=0;d<3;++d) {
      avg[d]=(gatheredEntityValue(D,*ua[d],d,F.v[0])+gatheredEntityValue(D,*ua[d],d,F.v[1])+gatheredEntityValue(D,*ua[d],d,F.v[2]))/3.0;
      avg[d]+=(9.0/20.0)*gatheredEntityValue(D,*ua[d],d,nv+f);
    }
    const Vec3 u{avg[0],avg[1],avg[2]}; const double un=dot3(u,nf);
    S.area+=area; S.flux+=dot3(u,sf); psum+=area*pa[D.pGid[M.owner[f]]];
    S.normalVelocityMin=std::min(S.normalVelocityMin,un); S.normalVelocityMax=std::max(S.normalVelocityMax,un);
  }
  S.pOwnerMean=psum/std::max(S.area,1e-300); return S;
}

static PetscErrorCode computeFlowDiagnosticsRoot(const Mesh& M,const Discrete& D,const ProblemConfig& P,const std::array<std::vector<double>,3>& U,const std::vector<double>& p) {
  PetscFunctionBeginUser;
  const std::vector<double>* ua[3]={nullptr,nullptr,nullptr};
  for(int d=0;d<3;++d) ua[d]=&U[(std::size_t)d];
  const auto in=patchSolutionStatsRoot(M,D,P.boundary.inlet,ua,p.data()),out=patchSolutionStatsRoot(M,D,P.boundary.outlet,ua,p.data());
  const double net=in.flux+out.flux,scale=std::max({std::abs(in.flux),std::abs(out.flux),1e-300});
  const double target=(P.inletBC==InletBCMode::FixedNormalSpeed)?P.boundary.signedNormalSpeed*P.boundary.inletProjectedArea:0.0;
  const double targetRel=(P.inletBC==InletBCMode::FixedNormalSpeed)?std::abs(in.flux-target)/std::max(std::abs(target),1e-300):0.0;
  PetscCall(PetscPrintf(PETSC_COMM_SELF,
    "P1BF3_FLOW_DIAGNOSTICS inlet=%s outlet=%s inletFlux=%.12e targetInletFlux=%.12e inletFluxRelError=%.12e outletFlux=%.12e netOutwardFlux=%.12e massRelative=%.12e inletArea=%.12e outletArea=%.12e inletFaceNormalVelocity=[%.12e,%.12e] outletFaceNormalVelocity=[%.12e,%.12e]\n",
    P.boundary.inletPatch.c_str(),P.boundary.outletPatch.c_str(),in.flux,target,targetRel,out.flux,net,std::abs(net)/scale,in.area,out.area,
    in.normalVelocityMin,in.normalVelocityMax,out.normalVelocityMin,out.normalVelocityMax));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,
    "P1BF3_FLOW_PRESSURE pInOwnerMean=%.12e pOutOwnerMean=%.12e dropInMinusOut=%.12e pressureGauge=physical_outlet_no_nullspace\n",
    in.pOwnerMean,out.pOwnerMean,in.pOwnerMean-out.pOwnerMean));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static Vec3 tetCellAverageVelocityRoot(const Mesh& M,const Discrete& D,PetscInt c,const std::vector<double>* ua[3]) {
  // Exact volume mean for P1+BF3 on an affine tetrahedron:
  // mean(lambda_i)=1/4 and mean(27*lambda_j*lambda_k*lambda_l)=9/40.
  const PetscInt nv=(PetscInt)M.points.size();
  PetscInt ent[8];
  for(int i=0;i<4;++i) ent[i]=M.tets[c][i];
  for(int i=0;i<4;++i) ent[4+i]=nv+M.oppFace[c][i];
  double u[3]={0.0,0.0,0.0};
  for(int d=0;d<3;++d) {
    for(int i=0;i<4;++i) u[d]+=0.25*gatheredEntityValue(D,*ua[d],d,ent[i]);
    for(int i=0;i<4;++i) u[d]+=(9.0/40.0)*gatheredEntityValue(D,*ua[d],d,ent[4+i]);
  }
  return Vec3{u[0],u[1],u[2]};
}

static double tetVolumeAbsRoot(const Mesh& M,PetscInt c) {
  const auto& t=M.tets[c];
  const Vec3& X0=M.points[t[0]]; const Vec3& X1=M.points[t[1]];
  const Vec3& X2=M.points[t[2]]; const Vec3& X3=M.points[t[3]];
  double J[3][3]={{X1.x-X0.x,X2.x-X0.x,X3.x-X0.x},
                  {X1.y-X0.y,X2.y-X0.y,X3.y-X0.y},
                  {X1.z-X0.z,X2.z-X0.z,X3.z-X0.z}};
  return std::abs(det3(J))/6.0;
}

static PetscErrorCode writeVtuRoot(const std::string& path,const Mesh& M,const Discrete& D,const std::array<std::vector<double>,3>& U,const std::vector<double>& p,PetscBool converged,PetscInt outerIts,const std::string& velocityMode) {
  PetscFunctionBeginUser;
  const bool writeLegacy=(velocityMode=="legacy" || velocityMode=="both");
  const bool writeU0=(velocityMode=="cell_average" || velocityMode=="u0" || velocityMode=="both");
  if(!writeLegacy && !writeU0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER_INPUT,"unknown VTU velocity mode %s",velocityMode.c_str());

  const std::vector<double>* ua[3]={nullptr,nullptr,nullptr};
  for(int d=0;d<3;++d) ua[d]=&U[(std::size_t)d];

  // U0 is the exact element-volume average of the solved P1+BF3 polynomial.
  // U0_stream is a volume-weighted vertex average of neighboring U0 cells.  It
  // exists only as a continuous visualization surrogate for ParaView streamlines.
  std::vector<Vec3> u0, u0Stream;
  if(writeU0) {
    u0.resize(M.tets.size());
    u0Stream.assign(M.points.size(),Vec3{0.0,0.0,0.0});
    std::vector<double> w(M.points.size(),0.0);
    for(PetscInt c=0;c<(PetscInt)M.tets.size();++c) {
      u0[c]=tetCellAverageVelocityRoot(M,D,c,ua);
      const double vol=tetVolumeAbsRoot(M,c);
      for(int i=0;i<4;++i) {
        const PetscInt v=M.tets[c][i];
        u0Stream[v].x+=vol*u0[c].x; u0Stream[v].y+=vol*u0[c].y; u0Stream[v].z+=vol*u0[c].z;
        w[v]+=vol;
      }
    }
    for(PetscInt v=0;v<(PetscInt)M.points.size();++v) if(w[v]>0.0) {
      u0Stream[v].x/=w[v]; u0Stream[v].y/=w[v]; u0Stream[v].z/=w[v];
    }
  }

  std::ofstream out(path); if(!out) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"cannot open VTU output %s",path.c_str());
  out<<std::setprecision(17)<<std::scientific;
  out<<"<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n<UnstructuredGrid>\n";
  out<<"<Piece NumberOfPoints=\""<<M.points.size()<<"\" NumberOfCells=\""<<M.tets.size()<<"\">\n";
  out<<"<Points><DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
  for(const auto& x:M.points) out<<x.x<<' '<<x.y<<' '<<x.z<<'\n'; out<<"</DataArray></Points>\n";
  out<<"<Cells>\n<DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
  for(const auto& t:M.tets) out<<t[0]<<' '<<t[1]<<' '<<t[2]<<' '<<t[3]<<'\n'; out<<"</DataArray>\n";
  out<<"<DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n"; for(size_t c=0;c<M.tets.size();++c) out<<4*(c+1)<<'\n'; out<<"</DataArray>\n";
  out<<"<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n"; for(size_t c=0;c<M.tets.size();++c) out<<10<<'\n'; out<<"</DataArray>\n</Cells>\n";

  out<<"<PointData";
  if(writeU0) out<<" Vectors=\"U0_stream\""; else if(writeLegacy) out<<" Vectors=\"U_P1\"";
  out<<">\n";
  if(writeLegacy) {
    out<<"<DataArray type=\"Float64\" Name=\"U_P1\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for(PetscInt v=0;v<(PetscInt)M.points.size();++v) out<<gatheredEntityValue(D,*ua[0],0,v)<<' '<<gatheredEntityValue(D,*ua[1],1,v)<<' '<<gatheredEntityValue(D,*ua[2],2,v)<<'\n';
    out<<"</DataArray>\n";
  }
  if(writeU0) {
    out<<"<DataArray type=\"Float64\" Name=\"U0_stream\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for(const auto& q:u0Stream) out<<q.x<<' '<<q.y<<' '<<q.z<<'\n';
    out<<"</DataArray>\n";
  }
  out<<"</PointData>\n";

  out<<"<CellData Scalars=\"p_P0\" Vectors=\""<<(writeU0?"U0":"U_P1BF3_centroid")<<"\">\n";
  out<<"<DataArray type=\"Float64\" Name=\"p_P0\" NumberOfComponents=\"1\" format=\"ascii\">\n";
  for(PetscInt c=0;c<(PetscInt)M.tets.size();++c) out<<p[(std::size_t)D.pGid[c]]<<'\n'; out<<"</DataArray>\n";

  if(writeU0) {
    out<<"<DataArray type=\"Float64\" Name=\"U0\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for(const auto& q:u0) out<<q.x<<' '<<q.y<<' '<<q.z<<'\n';
    out<<"</DataArray>\n";
  }

  if(writeLegacy) {
    out<<"<DataArray type=\"Float64\" Name=\"U_P1_centroid\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for(PetscInt c=0;c<(PetscInt)M.tets.size();++c) { double uc[3]={0,0,0}; for(int i=0;i<4;++i) for(int d=0;d<3;++d) uc[d]+=0.25*gatheredEntityValue(D,*ua[d],d,M.tets[c][i]); out<<uc[0]<<' '<<uc[1]<<' '<<uc[2]<<'\n'; }
    out<<"</DataArray>\n";
    out<<"<DataArray type=\"Float64\" Name=\"U_P1BF3_centroid\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    std::array<double,4> lam{{0.25,0.25,0.25,0.25}}; double val[8],gr[8][3]; basis(lam,val,gr); const PetscInt nv=(PetscInt)M.points.size();
    for(PetscInt c=0;c<(PetscInt)M.tets.size();++c) { PetscInt ent[8]; for(int i=0;i<4;++i)ent[i]=M.tets[c][i]; for(int i=0;i<4;++i)ent[4+i]=nv+M.oppFace[c][i]; double uc[3]={0,0,0}; for(int a=0;a<8;++a) for(int d=0;d<3;++d) uc[d]+=val[a]*gatheredEntityValue(D,*ua[d],d,ent[a]); out<<uc[0]<<' '<<uc[1]<<' '<<uc[2]<<'\n'; }
    out<<"</DataArray>\n";
  }

  out<<"<DataArray type=\"Int32\" Name=\"solve_converged\" NumberOfComponents=\"1\" format=\"ascii\">\n"; for(size_t c=0;c<M.tets.size();++c) out<<(converged?1:0)<<'\n'; out<<"</DataArray>\n";
  out<<"<DataArray type=\"Int32\" Name=\"outer_iterations\" NumberOfComponents=\"1\" format=\"ascii\">\n"; for(size_t c=0;c<M.tets.size();++c) out<<outerIts<<'\n'; out<<"</DataArray>\n";
  out<<"</CellData>\n</Piece>\n</UnstructuredGrid>\n</VTKFile>\n"; out.close();

  PetscCall(PetscPrintf(PETSC_COMM_SELF,
    "P1BF3_VTU path=%s mode=%s points=%zu cells=%zu pointVelocity=%s cellVelocity=%s cellPressure=p_P0 converged=%d outerIts=%" PetscInt_FMT " status=PASS\n",
    path.c_str(),velocityMode.c_str(),M.points.size(),M.tets.size(),writeU0?"U0_stream":"U_P1",writeU0?"U0":"U_P1BF3_centroid",converged?1:0,outerIts));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode computePipeDiagnosticsRoot(const Mesh& M,const Discrete& D,const ProblemConfig& P,const std::array<std::vector<double>,3>& U,const std::vector<double>& p) {
  PetscFunctionBeginUser;
  const auto& G=P.pipe;
  const std::vector<double>* ua[3]={nullptr,nullptr,nullptr};
  for(int d=0;d<3;++d) ua[d]=&U[(std::size_t)d];
  const PetscInt nv=(PetscInt)M.points.size();
  const auto Q=tetDuffy7();
  double ue2=0,un2=0,ueMesh2=0,trans2=0,pe2=0,pn2=0,total=0,pdiff=0;
  double sw=0,sz=0,szz=0,sp=0,szp=0;
  for(PetscInt c=0;c<(PetscInt)M.tets.size();++c) {
    const auto t=M.tets[c];Vec3 X[4]={M.points[t[0]],M.points[t[1]],M.points[t[2]],M.points[t[3]]};
    double J[3][3]={{X[1].x-X[0].x,X[2].x-X[0].x,X[3].x-X[0].x},{X[1].y-X[0].y,X[2].y-X[0].y,X[3].y-X[0].y},{X[1].z-X[0].z,X[2].z-X[0].z,X[3].z-X[0].z}};
    const double det=det3(J),vol=det/6.0;
    PetscInt entity[8];for(int i=0;i<4;++i)entity[i]=t[i];for(int i=0;i<4;++i)entity[4+i]=nv+M.oppFace[c][i];
    double coeff[3][8]={{0}};for(int a=0;a<8;++a)for(int d=0;d<3;++d)coeff[d][a]=gatheredEntityValue(D,*ua[d],d,entity[a]);
    const double pc=p[(std::size_t)D.pGid[c]];
    double zc=0;for(int i=0;i<4;++i)zc+=X[i].z;zc*=0.25;
    sw+=vol;sz+=vol*zc;szz+=vol*zc*zc;sp+=vol*pc;szp+=vol*zc*pc;
    for(const auto& q:Q) {
      double val[8],gr[8][3];basis(q.lam,val,gr);
      double x=0,y=0,z=0;for(int i=0;i<4;++i){x+=q.lam[i]*X[i].x;y+=q.lam[i]*X[i].y;z+=q.lam[i]*X[i].z;}
      double uh[3]={0,0,0};for(int d=0;d<3;++d)for(int a=0;a<8;++a)uh[d]+=coeff[d][a]*val[a];
      const double uz=pipeIdealUz(G,x,y),uzMesh=G.profileScale*uz,pex=pipeExactPressure(G,z),w=q.w*det;
      ue2+=(uh[0]*uh[0]+uh[1]*uh[1]+(uh[2]-uz)*(uh[2]-uz))*w;
      ueMesh2+=(uh[0]*uh[0]+uh[1]*uh[1]+(uh[2]-uzMesh)*(uh[2]-uzMesh))*w;
      un2+=uz*uz*w;trans2+=(uh[0]*uh[0]+uh[1]*uh[1])*w;
      const double dp=pc-pex;pe2+=dp*dp*w;pn2+=pex*pex*w;pdiff+=dp*w;total+=w;
    }
  }
  const double pshift=pdiff/total;
  double peShift2=0;
  for(PetscInt c=0;c<(PetscInt)M.tets.size();++c) {
    const auto t=M.tets[c];Vec3 X[4]={M.points[t[0]],M.points[t[1]],M.points[t[2]],M.points[t[3]]};
    double J[3][3]={{X[1].x-X[0].x,X[2].x-X[0].x,X[3].x-X[0].x},{X[1].y-X[0].y,X[2].y-X[0].y,X[3].y-X[0].y},{X[1].z-X[0].z,X[2].z-X[0].z,X[3].z-X[0].z}};
    const double det=det3(J),pc=p[(std::size_t)D.pGid[c]];
    for(const auto& q:Q){double z=0;for(int i=0;i<4;++i)z+=q.lam[i]*X[i].z;const double de=pc-pshift-pipeExactPressure(G,z);peShift2+=de*de*q.w*det;}
  }

  auto faceFluxAndPressure=[&](int pi,double& flux,double& pmean) {
    flux=0.0;pmean=0.0;double areaSum=0.0;const auto& pp=M.patches[pi];
    for(PetscInt f=pp.startFace;f<pp.startFace+pp.nFaces;++f) {
      const auto& F=M.faces[f];const Vec3& x0=M.points[F.v[0]];const Vec3& x1=M.points[F.v[1]];const Vec3& x2=M.points[F.v[2]];
      const double area=triangleArea(x0,x1,x2);double avg[3]={0,0,0};
      for(int d=0;d<3;++d) {
        avg[d]=(gatheredEntityValue(D,*ua[d],d,F.v[0])+gatheredEntityValue(D,*ua[d],d,F.v[1])+gatheredEntityValue(D,*ua[d],d,F.v[2]))/3.0;
        avg[d]+=(9.0/20.0)*gatheredEntityValue(D,*ua[d],d,nv+f);
      }
      flux+=area*avg[2];
      pmean+=area*p[(std::size_t)D.pGid[M.owner[f]]];areaSum+=area;
    }
    pmean/=areaSum;
  };
  double qIn=0,pIn=0,qOut=0,pOut=0;faceFluxAndPressure(G.inlet,qIn,pIn);faceFluxAndPressure(G.outlet,qOut,pOut);
  const double massRel=(qIn!=0)?(qOut-qIn)/qIn:0.0;
  const double denom=sw*szz-sz*sz;
  const double slope=(std::abs(denom)>1e-300)?(sw*szp-sz*sp)/denom:0.0;
  const double fitDrop=-slope*G.L,faceDrop=pIn-pOut;

  PetscCall(PetscPrintf(PETSC_COMM_SELF,
    "P1BF3_PIPE_ERROR U_L2=%.12e U_relL2=%.12e U_meshNormalized_L2=%.12e U_meshNormalized_relL2=%.12e transverse_L2=%.12e P_abs_L2=%.12e P_abs_relL2=%.12e P_shifted_L2=%.12e pressureShift=%.12e\n",
    std::sqrt(ue2),std::sqrt(ue2/std::max(un2,1e-300)),std::sqrt(ueMesh2),std::sqrt(ueMesh2/std::max(G.profileScale*G.profileScale*un2,1e-300)),std::sqrt(trans2),std::sqrt(pe2),std::sqrt(pe2/std::max(pn2,1e-300)),std::sqrt(peShift2),pshift));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,
    "P1BF3_PIPE_FLOW inletFlux=%.12e outletFlux=%.12e inletMean=%.12e outletMean=%.12e massRelative=%.12e inletArea=%.12e outletArea=%.12e\n",
    qIn,qOut,qIn/G.inletArea,qOut/G.outletArea,massRel,G.inletArea,G.outletArea));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,
    "P1BF3_PIPE_PRESSURE pInOwnerMean=%.12e pOutOwnerMean=%.12e dropFaceOwner=%.12e dropAxialFit=%.12e hpDrop=%.12e faceRelError=%.12e fitRelError=%.12e\n",
    pIn,pOut,faceDrop,fitDrop,G.hpDrop,(faceDrop-G.hpDrop)/G.hpDrop,(fitDrop-G.hpDrop)/G.hpDrop));

 
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode destroyDiscrete(Discrete& D){PetscFunctionBeginUser;PetscCall(MatDestroy(&D.A));for(int d=0;d<3;++d){PetscCall(MatDestroy(&D.B[d]));PetscCall(VecDestroy(&D.rhs[d]));}PetscCall(VecDestroy(&D.volumes));PetscCall(VecDestroy(&D.fixedDiv));PetscFunctionReturn(PETSC_SUCCESS);}

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

int main(int argc,char **argv) {
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,nullptr,"MPI P1+BF3/P0 SIMPLE steady Navier-Stokes MMS/pipe/generic-flow on OpenFOAM tetrahedral polyMesh\n"));
  int rank=0,size=1;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  MPI_Comm_size(PETSC_COMM_WORLD,&size);

  char meshPath[PETSC_MAX_PATH_LEN]="";
  PetscBool set=PETSC_FALSE;
  PetscCall(PetscOptionsGetString(nullptr,nullptr,"-mesh",meshPath,sizeof(meshPath),&set));
  if(!set) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Pass -mesh /path/to/constant/polyMesh");

  char problemName[32]="mms";
  PetscCall(PetscOptionsGetString(nullptr,nullptr,"-problem",problemName,sizeof(problemName),nullptr));
  const std::string problem(problemName);
  if(problem!="mms" && problem!="pipe" && problem!="flow") SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"-problem must be mms, pipe, or flow");

  char convectionName[32]="central";
  PetscCall(PetscOptionsGetString(nullptr,nullptr,"-convection",convectionName,sizeof(convectionName),nullptr));
  const std::string convection(convectionName);
  const bool centralConvection=(convection=="central");
  if(convection!="central" && convection!="none") SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"-convection must be central or none");

  PetscBool useSupg=PETSC_FALSE;
  PetscReal supgTauScale=0.05,supgMagic=9.0;
  PetscInt supgQuadPoints=64;
  char supgFormName[32]="implicit",supgKernelName[32]="fast";
  PetscCall(PetscOptionsGetBool(nullptr,nullptr,"-supg",&useSupg,nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-supg_tau_scale",&supgTauScale,nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-supg_magic",&supgMagic,nullptr));
  PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-supg_quad_points",&supgQuadPoints,nullptr));
  PetscCall(PetscOptionsGetString(nullptr,nullptr,"-supg_form",supgFormName,sizeof(supgFormName),nullptr));
  PetscCall(PetscOptionsGetString(nullptr,nullptr,"-supg_kernel",supgKernelName,sizeof(supgKernelName),nullptr));
  const std::string supgForm(supgFormName),supgKernel(supgKernelName);
  if(useSupg && !centralConvection) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"SUPG requires -convection central in this branch");
  if(supgTauScale<0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"supg_tau_scale >= 0 required");
  if(supgMagic<=0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"supg_magic > 0 required");
  if(supgQuadPoints!=125 && supgQuadPoints!=64 && supgQuadPoints!=45) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"-supg_quad_points must be 125, 64, or 45");
  if(supgForm!="implicit" && supgForm!="explicit") SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"-supg_form must be implicit or explicit");
  if(supgKernel!="fast" && supgKernel!="legacy") SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"-supg_kernel must be fast or legacy");
  if(useSupg && supgKernel=="legacy" && (supgQuadPoints!=125 || supgForm!="implicit")) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"legacy SUPG supports only -supg_quad_points 125 -supg_form implicit");

  PetscReal re=(problem=="pipe"?20.0:1.0),au=.5,ap=.5,rauScale=1.0,simpleTol=1e-6,uTol=1e-10,uAtol=0.0,uRelDrop=0.0,uOmega=1.0,pipeBulkVelocity=1.0;
  PetscInt maxOuter=2000,uMax=20000,uCheck=5,uLocalSweeps=1,pPreconditionerRefresh=0;
  PetscBool pressureProfile=PETSC_FALSE,factoredBenchmark=PETSC_FALSE,m10PcgProfile=PETSC_FALSE;
  PetscInt factoredBenchmarkAt=1,factoredBenchmarkReps=200;
  PetscInt pressureProfileAt=1,pressureProfileFineReps=200,pressureProfilePcReps=50,pressureProfileCgIts=10,pressureProfileCgReps=5,pressureProfileLevelMatReps=50,pressureProfileLevelSolveReps=10;
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-re",&re,nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-pipe_bulk_velocity",&pipeBulkVelocity,nullptr));
  PetscReal nuOption=1.0,inletNormalSpeed=-1.0; PetscBool nuWasSet=PETSC_FALSE,meshAuditOnly=PETSC_FALSE,writeVtu=PETSC_TRUE,resourceProfile=PETSC_FALSE,memoryAudit=PETSC_FALSE;
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-nu",&nuOption,&nuWasSet));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-inlet_normal_speed",&inletNormalSpeed,nullptr));
  PetscCall(PetscOptionsGetBool(nullptr,nullptr,"-mesh_audit_only",&meshAuditOnly,nullptr));
  PetscCall(PetscOptionsGetBool(nullptr,nullptr,"-write_vtu",&writeVtu,nullptr));
  PetscCall(PetscOptionsGetBool(nullptr,nullptr,"-resource_profile",&resourceProfile,nullptr));
  PetscCall(PetscOptionsGetBool(nullptr,nullptr,"-memory_audit",&memoryAudit,nullptr));
  PetscBool customMomentumLive=PETSC_FALSE,customMomentumShadow=PETSC_FALSE,customMomentumShadowStrict=PETSC_FALSE,customMomentumShadowSgs=PETSC_TRUE;
  PetscBool m2bDirectDynamic=PETSC_TRUE,m3StaticReference=PETSC_FALSE,m4bBReference=PETSC_FALSE,m5bPcgReference=PETSC_FALSE,m6bVelocityReference=PETSC_FALSE;
  PetscReal customMomentumShadowTol=5e-12,customPressureBShadowTol=5e-12,customPressurePcgReferenceTol=5e-10; PetscInt customMomentumShadowInterval=50;
  PetscCall(PetscOptionsGetBool(nullptr,nullptr,"-custom_momentum_live",&customMomentumLive,nullptr));
  PetscCall(PetscOptionsGetBool(nullptr,nullptr,"-custom_momentum_shadow",&customMomentumShadow,nullptr));
  PetscCall(PetscOptionsGetBool(nullptr,nullptr,"-custom_momentum_shadow_strict",&customMomentumShadowStrict,nullptr));
  PetscCall(PetscOptionsGetBool(nullptr,nullptr,"-custom_momentum_shadow_sgs",&customMomentumShadowSgs,nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-custom_momentum_shadow_tol",&customMomentumShadowTol,nullptr));
  PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-custom_momentum_shadow_interval",&customMomentumShadowInterval,nullptr));
  PetscCall(PetscOptionsGetBool(nullptr,nullptr,"-m2b_direct_dynamic",&m2bDirectDynamic,nullptr));
  PetscCall(PetscOptionsGetBool(nullptr,nullptr,"-m3_static_reference",&m3StaticReference,nullptr));
  PetscCall(PetscOptionsGetBool(nullptr,nullptr,"-m4b_b_reference",&m4bBReference,nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-custom_pressure_b_shadow_tol",&customPressureBShadowTol,nullptr));
  PetscCall(PetscOptionsGetBool(nullptr,nullptr,"-m5b_pcg_reference",&m5bPcgReference,nullptr));
  PetscCall(PetscOptionsGetBool(nullptr,nullptr,"-m6b_velocity_reference",&m6bVelocityReference,nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-custom_pressure_pcg_reference_tol",&customPressurePcgReferenceTol,nullptr));
  if(customPressureBShadowTol<=0.0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"-custom_pressure_b_shadow_tol must be > 0");
  if(customPressurePcgReferenceTol<=0.0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"-custom_pressure_pcg_reference_tol must be > 0");
  if(customMomentumShadowTol<=0.0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"-custom_momentum_shadow_tol must be > 0");
  if(customMomentumShadowInterval<0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"-custom_momentum_shadow_interval must be >= 0");
  char inletBcName[32]; std::snprintf(inletBcName,sizeof(inletBcName),"%s",problem=="flow"?"fixed_normal_speed":"parabolic");
  char inletNormalModeName[32]="average_patch_normal";
  char vtuOutput[PETSC_MAX_PATH_LEN]="p1bf3_solution.vtu";
  char vtuVelocityModeName[32]="legacy";
  PetscCall(PetscOptionsGetString(nullptr,nullptr,"-inlet_bc",inletBcName,sizeof(inletBcName),nullptr));
  PetscCall(PetscOptionsGetString(nullptr,nullptr,"-inlet_normal_mode",inletNormalModeName,sizeof(inletNormalModeName),nullptr));
  PetscCall(PetscOptionsGetString(nullptr,nullptr,"-vtu_output",vtuOutput,sizeof(vtuOutput),nullptr));
  PetscCall(PetscOptionsGetString(nullptr,nullptr,"-vtu_velocity_mode",vtuVelocityModeName,sizeof(vtuVelocityModeName),nullptr));
  const std::string inletBc(inletBcName),inletNormalMode(inletNormalModeName),vtuVelocityMode(vtuVelocityModeName);
  if(vtuVelocityMode!="legacy" && vtuVelocityMode!="cell_average" && vtuVelocityMode!="u0" && vtuVelocityMode!="both")
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"-vtu_velocity_mode must be legacy, cell_average (or u0), or both");
  if(inletBc!="parabolic" && inletBc!="fixed_normal_speed") SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"-inlet_bc must be parabolic or fixed_normal_speed");
  if(problem=="flow" && inletBc!="fixed_normal_speed") SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"generic -problem flow currently requires -inlet_bc fixed_normal_speed");
  if(inletBc=="fixed_normal_speed" && inletNormalMode!="average_patch_normal") SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"fixed_normal_speed currently supports only -inlet_normal_mode average_patch_normal");
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-alpha_u",&au,nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-alpha_p",&ap,nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-rau_scale",&rauScale,nullptr));

  // Separate pressure-velocity coupling selector.  This branch defaults to
  // SIMPLEC, while -simple_variant simple exactly restores the previous SIMPLE
  // correction metric and therefore provides an in-branch A/B control.
  char simpleVariantName[32]="simplec";
  PetscCall(PetscOptionsGetString(nullptr,nullptr,"-simple_variant",simpleVariantName,sizeof(simpleVariantName),nullptr));
  const std::string simpleVariant(simpleVariantName);
  if(simpleVariant!="simple" && simpleVariant!="simplec")
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"-simple_variant must be simple or simplec");
  PetscReal simplecBlend=1.0,simplecFloorFraction=1e-6;
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-simplec_blend",&simplecBlend,nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-simplec_floor_fraction",&simplecFloorFraction,nullptr));
  char simplecFallbackName[32]="diag";
  PetscCall(PetscOptionsGetString(nullptr,nullptr,"-simplec_fallback",simplecFallbackName,sizeof(simplecFallbackName),nullptr));
  const std::string simplecFallback(simplecFallbackName);
  if(simplecBlend<0.0 || simplecBlend>1.0)
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"-simplec_blend must satisfy 0 <= blend <= 1");
  if(simplecFloorFraction<0.0)
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"-simplec_floor_fraction must be >= 0");
  if(simplecFallback!="diag" && simplecFallback!="floor" && simplecFallback!="error")
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"-simplec_fallback must be diag, floor, or error");

  char rauModeName[32]="diag";
  PetscCall(PetscOptionsGetString(nullptr,nullptr,"-rau_mode",rauModeName,sizeof(rauModeName),nullptr));
  const std::string rauMode(rauModeName);
  if(rauMode!="diag" && rauMode!="row_l1") SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"-rau_mode must be diag or row_l1");

  char uRelaxModeName[32]="diag";
  PetscCall(PetscOptionsGetString(nullptr,nullptr,"-u_relax_mode",uRelaxModeName,sizeof(uRelaxModeName),nullptr));
  const std::string uRelaxMode(uRelaxModeName);
  if(uRelaxMode!="diag" && uRelaxMode!="row_l1") SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"-u_relax_mode must be diag or row_l1");
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-simple_rtol",&simpleTol,nullptr));
  PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-simple_max_it",&maxOuter,nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-u_rtol",&uTol,nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-u_atol",&uAtol,nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-u_rel_drop",&uRelDrop,nullptr));
  PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-u_max_it",&uMax,nullptr));
  PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-u_check_every",&uCheck,nullptr));
  PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-u_local_sweeps",&uLocalSweeps,nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-u_sor_omega",&uOmega,nullptr));
  PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-p_preconditioner_refresh",&pPreconditionerRefresh,nullptr));
  // PCD Gate 1: make the outer pressure Krylov selectable while leaving the
  // exact Schur operator and the existing PETSc pressure preconditioner intact.
  // custom_pcg is the untouched production path; petsc_fgmres is the Gate-1 path.
  char pressureSolveModeName[32]="custom_pcg";
  PetscCall(PetscOptionsGetString(nullptr,nullptr,"-pressure_solve_mode",pressureSolveModeName,sizeof(pressureSolveModeName),nullptr));
  const std::string pressureSolveMode(pressureSolveModeName);
  if(pressureSolveMode!="custom_pcg" && pressureSolveMode!="petsc_fgmres")
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"-pressure_solve_mode must be custom_pcg or petsc_fgmres");
  PetscBool gate1ComparePcg=PETSC_TRUE;
  PetscReal gate1SolutionTol=5e-8,gate1TrueResidualTol=1e-8;
  PetscCall(PetscOptionsGetBool(nullptr,nullptr,"-gate1_compare_pcg",&gate1ComparePcg,nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-gate1_solution_tol",&gate1SolutionTol,nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-gate1_true_residual_tol",&gate1TrueResidualTol,nullptr));
  if(gate1SolutionTol<=0.0 || gate1TrueResidualTol<=0.0)
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Gate-1 tolerances must be > 0");
  PetscBool gate3MpProbe=PETSC_FALSE;
  PetscReal gate3MpAlgebraTol=1e-14;
  PetscCall(PetscOptionsGetBool(nullptr,nullptr,"-gate3_mp_probe",&gate3MpProbe,nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-gate3_mp_algebra_tol",&gate3MpAlgebraTol,nullptr));
  if(gate3MpAlgebraTol<=0.0)
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"-gate3_mp_algebra_tol must be > 0");
  if(gate3MpProbe && pressureSolveMode!="petsc_fgmres")
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Gate-3 Mp probe requires -pressure_solve_mode petsc_fgmres");
  PetscBool gate4KpProbe=PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(nullptr,nullptr,"-gate4_kp_probe",&gate4KpProbe,nullptr));
  PetscBool gate5KpGamgProbe=PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(nullptr,nullptr,"-gate5_kp_gamg_probe",&gate5KpGamgProbe,nullptr));
  PetscBool gate6DiffusionPcdProbe=PETSC_FALSE;
  PetscReal gate6ChainTol=1e-7,gate6KpTrueResidualTol=1e-8;
  PetscCall(PetscOptionsGetBool(nullptr,nullptr,"-gate6_diffusion_pcd_probe",&gate6DiffusionPcdProbe,nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-gate6_chain_tol",&gate6ChainTol,nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-gate6_kp_true_residual_tol",&gate6KpTrueResidualTol,nullptr));
  if(gate6ChainTol<=0.0 || gate6KpTrueResidualTol<=0.0)
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Gate-6 tolerances must be > 0");
  PetscBool gate7CpProbe=PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(nullptr,nullptr,"-gate7_cp_probe",&gate7CpProbe,nullptr));
  PetscBool gate8EswBcProbe=PETSC_FALSE;
  PetscReal gate8CancelTol=1e-13,gate8WallFluxTol=1e-13;
  PetscCall(PetscOptionsGetBool(nullptr,nullptr,"-gate8_esw_bc_probe",&gate8EswBcProbe,nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-gate8_cancel_tol",&gate8CancelTol,nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-gate8_wall_flux_tol",&gate8WallFluxTol,nullptr));
  if(gate8CancelTol<=0.0 || gate8WallFluxTol<=0.0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Gate-8 tolerances must be > 0");
  PetscBool gate9LivePcd=PETSC_FALSE,gate9dGamgOnly=PETSC_FALSE,gate9eNgfv=PETSC_FALSE,gate9gFeFace=PETSC_FALSE,gate9gRichardson=PETSC_FALSE,gate9hChebyshev=PETSC_FALSE,gate9iAutoChebyshev=PETSC_FALSE;
  PetscReal gate9KpRtol=1e-10,gate9KpTrueResidualTol=1e-8,gate9OuterTrueResidualTol=1e-8;
  PetscReal gate9dOmega=1.0,gate9dDivergenceFactor=1e6;
  PetscReal gate9hLambdaMin=0.1,gate9hLambdaMax=4.5;
  PetscReal gate9iSafety=1.2,gate9iLambdaMinFraction=0.06,gate9iRtol=0.1,gate9iAtol=1e-10;
  PetscInt gate9KpMaxIts=200,gate9dMaxCycles=100,gate9hMaxSteps=60,gate9hCheckEvery=5;
  PetscInt gate9iPowerIts=8,gate9iInitialBlock=8,gate9iExtendBlock=4,gate9iMaxSteps=40;
  PetscInt gate9iSpectrumRefresh=0,gate9iFixedSteps=0;
  PetscBool gate9iRequireTarget=PETSC_TRUE;
  PetscCall(PetscOptionsGetBool(nullptr,nullptr,"-gate9_live_pcd",&gate9LivePcd,nullptr));
  PetscCall(PetscOptionsGetBool(nullptr,nullptr,"-gate9d_gamg_only",&gate9dGamgOnly,nullptr));
  PetscCall(PetscOptionsGetBool(nullptr,nullptr,"-gate9e_ngfv",&gate9eNgfv,nullptr));
  PetscCall(PetscOptionsGetBool(nullptr,nullptr,"-gate9g_fe_face_energy",&gate9gFeFace,nullptr));
  PetscCall(PetscOptionsGetBool(nullptr,nullptr,"-gate9g_richardson",&gate9gRichardson,nullptr));
  PetscCall(PetscOptionsGetBool(nullptr,nullptr,"-gate9h_chebyshev",&gate9hChebyshev,nullptr));
  PetscCall(PetscOptionsGetBool(nullptr,nullptr,"-gate9i_auto_chebyshev",&gate9iAutoChebyshev,nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-gate9h_lambda_min",&gate9hLambdaMin,nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-gate9h_lambda_max",&gate9hLambdaMax,nullptr));
  PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-gate9h_max_steps",&gate9hMaxSteps,nullptr));
  PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-gate9h_check_every",&gate9hCheckEvery,nullptr));
  PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-gate9i_power_its",&gate9iPowerIts,nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-gate9i_safety",&gate9iSafety,nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-gate9i_lambda_min_fraction",&gate9iLambdaMinFraction,nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-gate9i_rtol",&gate9iRtol,nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-gate9i_atol",&gate9iAtol,nullptr));
  PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-gate9i_initial_block",&gate9iInitialBlock,nullptr));
  PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-gate9i_extend_block",&gate9iExtendBlock,nullptr));
  PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-gate9i_max_steps",&gate9iMaxSteps,nullptr));
  PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-gate9i_spectrum_refresh",&gate9iSpectrumRefresh,nullptr));
  PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-gate9i_fixed_steps",&gate9iFixedSteps,nullptr));
  PetscCall(PetscOptionsGetBool(nullptr,nullptr,"-gate9i_require_target",&gate9iRequireTarget,nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-gate9d_omega",&gate9dOmega,nullptr));
  PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-gate9d_max_cycles",&gate9dMaxCycles,nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-gate9d_divergence_factor",&gate9dDivergenceFactor,nullptr));
  if(gate9dGamgOnly) gate9LivePcd=PETSC_TRUE;
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-gate9_kp_rtol",&gate9KpRtol,nullptr));
  PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-gate9_kp_max_it",&gate9KpMaxIts,nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-gate9_kp_true_residual_tol",&gate9KpTrueResidualTol,nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-gate9_outer_true_residual_tol",&gate9OuterTrueResidualTol,nullptr));
  if(gate9KpRtol<=0.0 || gate9KpMaxIts<1 || gate9KpTrueResidualTol<=0.0 || gate9OuterTrueResidualTol<=0.0)
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Gate-9 Kp/outer tolerances must be positive");
  if(gate9dOmega<=0.0 || gate9dMaxCycles<1 || gate9dDivergenceFactor<=1.0)
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Gate-9D requires omega>0, max_cycles>=1, divergence_factor>1");
  if(gate9gRichardson && !gate9gFeFace) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"-gate9g_richardson requires -gate9g_fe_face_energy 1");
  if(gate9hChebyshev && !gate9gFeFace) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"-gate9h_chebyshev requires -gate9g_fe_face_energy 1");
  if(gate9iAutoChebyshev && !gate9gFeFace) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"-gate9i_auto_chebyshev requires -gate9g_fe_face_energy 1");
  if((gate9hChebyshev?1:0)+(gate9iAutoChebyshev?1:0)+(gate9gRichardson?1:0)>1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Gate-9G Richardson, Gate-9H fixed Chebyshev, and Gate-9I auto Chebyshev are mutually exclusive");
  if(gate9hLambdaMin<=0.0 || gate9hLambdaMax<=gate9hLambdaMin || gate9hMaxSteps<1 || gate9hCheckEvery<1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Gate-9H requires 0<lambda_min<lambda_max, max_steps>=1, check_every>=1");
  if(gate9iPowerIts<2 || gate9iSafety<=1.0 || gate9iLambdaMinFraction<=0.0 || gate9iLambdaMinFraction>=1.0 || gate9iRtol<=0.0 || gate9iRtol>=1.0 || gate9iAtol<=0.0 || gate9iInitialBlock<1 || gate9iExtendBlock<1 || gate9iMaxSteps<gate9iInitialBlock || gate9iSpectrumRefresh<0 || gate9iFixedSteps<0)
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Gate-9I requires power_its>=2, safety>1, 0<lambda_min_fraction<1, 0<rtol<1, atol>0, valid block sizes, spectrum_refresh>=0, fixed_steps>=0");
  if(gate9gFeFace && pressureSolveMode!="custom_pcg") SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Gate-9G/H/I uses -pressure_solve_mode custom_pcg so PETSc KSP owns only the GAMG PC");
  // Gate 9C needs only Gate-4 geometric Kp; do not force Gate-7/8 PCD pieces.
  if(gate8EswBcProbe) gate7CpProbe=PETSC_TRUE;
  if(gate7CpProbe && !centralConvection) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Gate-7/8/9 Cp/ESW/PCD require -convection central");
  if(gate5KpGamgProbe || gate6DiffusionPcdProbe || gate7CpProbe || gate8EswBcProbe || gate9LivePcd || gate9eNgfv) gate4KpProbe=PETSC_TRUE;
  if(gate4KpProbe && pressureSolveMode!="petsc_fgmres" && !gate9eNgfv)
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Gate-4/5/6/7/8/9 Kp/PCD requires -pressure_solve_mode petsc_fgmres");
  if(gate9LivePcd && problem=="mms") SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Gate-9 live PCD is a pipe/flow gate, not MMS");
  char pPmatName[32]="full";
  PetscCall(PetscOptionsGetString(nullptr,nullptr,"-p_pmat",pPmatName,sizeof(pPmatName),nullptr));
  const std::string pPmatMode(pPmatName);
  if(pPmatMode!="full" && pPmatMode!="compact_face" && pPmatMode!="fe_fv_face" && pPmatMode!="fv_lsq" && pPmatMode!="native_face")
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"-p_pmat must be full, compact_face, fe_fv_face, fv_lsq, or native_face");
  char pOperatorName[32]="explicit";
  PetscCall(PetscOptionsGetString(nullptr,nullptr,"-p_operator",pOperatorName,sizeof(pOperatorName),nullptr));
  const std::string pOperatorMode(pOperatorName);
  if(pOperatorMode!="explicit" && pOperatorMode!="factored")
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"-p_operator must be explicit or factored");
  if(pPmatMode=="native_face" && pOperatorMode!="factored")
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"-p_pmat native_face requires -p_operator factored so the exact custom B rAU Bt remains the Krylov operator");
  PetscCall(PetscOptionsGetBool(nullptr,nullptr,"-pressure_factored_benchmark",&factoredBenchmark,nullptr));
  PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-pressure_factored_benchmark_at",&factoredBenchmarkAt,nullptr));
  if(pPmatMode=="native_face" && factoredBenchmark) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"-p_pmat native_face does not materialize the exact Schur benchmark matrix; disable -pressure_factored_benchmark");
  PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-pressure_factored_benchmark_reps",&factoredBenchmarkReps,nullptr));
  PetscReal feFvP1Strength=1.0;
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-p_fe_fv_p1_strength",&feFvP1Strength,nullptr));
  if(feFvP1Strength<0.0 || feFvP1Strength>3.0)
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"-p_fe_fv_p1_strength must satisfy 0 <= strength <= 3 (values >1 are diagnostic and need not remain diagonally dominant)");
  PetscReal fvLsqStrength=0.9,fvLsqTpfaFloor=0.10;
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-p_fv_lsq_strength",&fvLsqStrength,nullptr));
  PetscCall(PetscOptionsGetReal(nullptr,nullptr,"-p_fv_lsq_tpfa_floor",&fvLsqTpfaFloor,nullptr));
  if(fvLsqStrength<=0.0 || fvLsqStrength>1.0)
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"-p_fv_lsq_strength must satisfy 0 < strength <= 1");
  if(fvLsqTpfaFloor<0.0 || fvLsqTpfaFloor>1.0)
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"-p_fv_lsq_tpfa_floor must satisfy 0 <= floor <= 1");
  PetscCall(PetscOptionsGetBool(nullptr,nullptr,"-pressure_profile",&pressureProfile,nullptr));
  PetscCall(PetscOptionsGetBool(nullptr,nullptr,"-m10_pcg_profile",&m10PcgProfile,nullptr));
  PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-pressure_profile_at",&pressureProfileAt,nullptr));
  PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-pressure_profile_fine_reps",&pressureProfileFineReps,nullptr));
  PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-pressure_profile_pc_reps",&pressureProfilePcReps,nullptr));
  PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-pressure_profile_cg_its",&pressureProfileCgIts,nullptr));
  PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-pressure_profile_cg_reps",&pressureProfileCgReps,nullptr));
  PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-pressure_profile_level_mat_reps",&pressureProfileLevelMatReps,nullptr));
  PetscCall(PetscOptionsGetInt(nullptr,nullptr,"-pressure_profile_level_solve_reps",&pressureProfileLevelSolveReps,nullptr));

  char wallPatchName[64]="patch_0_0",inletPatchName[64]="patch_2_0",outletPatchName[64]="patch_1_0";
  PetscCall(PetscOptionsGetString(nullptr,nullptr,"-pipe_wall_patch",wallPatchName,sizeof(wallPatchName),nullptr));
  PetscCall(PetscOptionsGetString(nullptr,nullptr,"-pipe_inlet_patch",inletPatchName,sizeof(inletPatchName),nullptr));
  PetscCall(PetscOptionsGetString(nullptr,nullptr,"-pipe_outlet_patch",outletPatchName,sizeof(outletPatchName),nullptr));
  char flowWallPatches[512]="",flowInletPatch[128]="",flowOutletPatch[128]="";
  PetscCall(PetscOptionsGetString(nullptr,nullptr,"-flow_wall_patches",flowWallPatches,sizeof(flowWallPatches),nullptr));
  PetscCall(PetscOptionsGetString(nullptr,nullptr,"-flow_inlet_patch",flowInletPatch,sizeof(flowInletPatch),nullptr));
  PetscCall(PetscOptionsGetString(nullptr,nullptr,"-flow_outlet_patch",flowOutletPatch,sizeof(flowOutletPatch),nullptr));

  if(re<=0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Re > 0 required");
  if(pipeBulkVelocity<=0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"pipe_bulk_velocity > 0 required");
  if(nuWasSet && nuOption<=0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"-nu > 0 required");
  if(inletBc=="fixed_normal_speed" && inletNormalSpeed==0.0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"-inlet_normal_speed must be nonzero for fixed_normal_speed");
  if(au<=0||au>1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"0 < alpha_u <= 1 required");
  if(ap<=0||ap>1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"0 < alpha_p <= 1 required");
  if(rauScale<=0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"rau_scale > 0 required");
  if(uOmega<=0||uOmega>=2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"0 < u_sor_omega < 2 required");
  if(uTol<0 || uAtol<0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"u_rtol and u_atol must be >= 0");
  if(uRelDrop<0||uRelDrop>=1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"0 <= u_rel_drop < 1 required (0 disables initial-residual-drop stopping)");
  if(uLocalSweeps<1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"u_local_sweeps >= 1 required");
  if(pPreconditionerRefresh<0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"p_preconditioner_refresh >= 0 required (0 = PETSc default refresh behavior)");
  if(pressureProfileAt<1 || pressureProfileFineReps<1 || pressureProfilePcReps<1 || pressureProfileCgIts<1 || pressureProfileCgReps<1 || pressureProfileLevelMatReps<1 || pressureProfileLevelSolveReps<1)
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"pressure profile counts must all be >= 1");
  if(factoredBenchmarkAt<1 || factoredBenchmarkReps<1)
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"factored Schur benchmark counts must be >= 1");

  try {
    PetscLogDouble tTotal0,tAsm0,tAsm1,tSolve0,tSolve1;
    PetscLogDouble tMesh0,tMesh1,tAudit0,tAudit1,tProblem0,tProblem1,tGhost0,tGhost1,tPPlan0,tPPlan1,tCPlan0,tCPlan1,tSupgPlan0,tSupgPlan1,tObjects0,tObjects1;
    PetscCall(PetscTime(&tTotal0));
    if(resourceProfile) PetscCall(printResourceMark("startup",0,0.0,tTotal0));
    PetscCall(PetscTime(&tMesh0));
    Mesh M=loadFoamTetMesh(meshPath);
    PetscCall(PetscTime(&tMesh1));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_MESH path=%s points=%zu faces=%zu internalFaces=%zu boundaryFaces=%zu cells=%zu meshRead=replicated_per_rank\n",
      meshPath,M.points.size(),M.faces.size(),M.neighbour.size(),M.faces.size()-M.neighbour.size(),M.tets.size()));
    PetscCall(PetscTime(&tAudit0));
    if(rank==0) { PetscCall(buildPlexAuditSelf(M)); PetscCall(printPatchAuditRoot(M)); }
    PetscCall(PetscTime(&tAudit1));
    PetscCall(printSetupPhase("mesh_patch_audit_root",tAudit1-tAudit0,(PetscInt)M.tets.size()));
    if(resourceProfile) PetscCall(printResourceMark("after_mesh",(PetscInt)M.tets.size(),tMesh1-tMesh0,tTotal0));
    if(memoryAudit) PetscCall(auditMeshMemory(M,(PetscInt)M.tets.size()));
    PetscCall(printSetupPhase("mesh_load",tMesh1-tMesh0,(PetscInt)M.tets.size()));
    if(meshAuditOnly) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_MESH_AUDIT_ONLY status=PASS no_solve=1\n"));
      PetscCall(PetscFinalize()); return 0;
    }

    PetscCall(PetscTime(&tProblem0));
    ProblemConfig P;
    P.mode=(problem=="pipe")?ProblemMode::Pipe:((problem=="flow")?ProblemMode::Flow:ProblemMode::MMS);
    P.centralConvection=centralConvection; P.re=(double)re;
    P.inletBC=(inletBc=="fixed_normal_speed")?InletBCMode::FixedNormalSpeed:InletBCMode::PipeParabolic;
    if(P.mode==ProblemMode::Pipe) {
      const double inletCharacteristicSpeed=(P.inletBC==InletBCMode::FixedNormalSpeed)?std::abs((double)inletNormalSpeed):(double)pipeBulkVelocity;
      P.pipe=makePipeGeometry(M,(double)re,inletCharacteristicSpeed,wallPatchName,inletPatchName,outletPatchName);
      if(nuWasSet) {
        P.pipe.nu=(double)nuOption; P.pipe.re=inletCharacteristicSpeed*P.pipe.D/P.pipe.nu;
        P.pipe.hpGradient=32.0*P.pipe.nu*inletCharacteristicSpeed/(P.pipe.D*P.pipe.D); P.pipe.hpDrop=P.pipe.hpGradient*P.pipe.L;
      }
      P.re=P.pipe.re; P.nu=P.pipe.nu;
      P.boundary=makeBoundaryGeometry(M,{wallPatchName},inletPatchName,outletPatchName,(double)inletNormalSpeed);
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "P1BF3_PIPE_GEOMETRY axis=z zIn=%.12g zOut=%.12g L=%.12g R=%.12g D=%.12g Uchar=%.12g Re=%.12g nu=%.12g inletArea=%.12e circleArea=%.12e areaRatio=%.9f inletProfileScale=%.9f hpDrop=%.12g hpGrad=%.12g\n",
        P.pipe.zIn,P.pipe.zOut,P.pipe.L,P.pipe.R,P.pipe.D,P.pipe.bulkVelocity,P.pipe.re,P.pipe.nu,P.pipe.inletArea,P.pipe.circleArea,P.pipe.areaRatio,P.pipe.profileScale,P.pipe.hpDrop,P.pipe.hpGradient));
      if(P.inletBC==InletBCMode::PipeParabolic) PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "P1BF3_PIPE_BC wall=%s:noSlip inlet=%s:parabolic_faceMeanNormalized outlet=%s:natural_zero_traction pressureGauge=physical_outlet_no_nullspace\n",
        P.pipe.wallPatch.c_str(),P.pipe.inletPatch.c_str(),P.pipe.outletPatch.c_str()));
      else PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "P1BF3_PIPE_BC wall=%s:noSlip inlet=%s:fixed_normal_speed signedSpeed=%.12g normalMode=average_patch_normal U=[%.12e,%.12e,%.12e] outlet=%s:natural_zero_traction pressureGauge=physical_outlet_no_nullspace\n",
        wallPatchName,inletPatchName,(double)inletNormalSpeed,P.boundary.inletVelocity.x,P.boundary.inletVelocity.y,P.boundary.inletVelocity.z,outletPatchName));
    } else if(P.mode==ProblemMode::Flow) {
      const auto walls=splitPatchNames(flowWallPatches);
      if(walls.empty() || std::string(flowInletPatch).empty() || std::string(flowOutletPatch).empty())
        throw std::runtime_error("-problem flow requires -flow_wall_patches, -flow_inlet_patch, and -flow_outlet_patch (use -mesh_audit_only 1 first)");
      P.boundary=makeBoundaryGeometry(M,walls,flowInletPatch,flowOutletPatch,(double)inletNormalSpeed);
      P.nu=nuWasSet?(double)nuOption:1.0/(double)re; P.re=1.0/P.nu;
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "P1BF3_FLOW_BC walls=%s inlet=%s:fixed_normal_speed signedSpeed=%.12g normalMode=average_patch_normal U=[%.12e,%.12e,%.12e] inletArea=%.12e inletProjectedArea=%.12e outlet=%s:natural_zero_traction nu=%.12g unitScaleRe=%.12g pressureGauge=physical_outlet_no_nullspace\n",
        flowWallPatches,flowInletPatch,(double)inletNormalSpeed,P.boundary.inletVelocity.x,P.boundary.inletVelocity.y,P.boundary.inletVelocity.z,P.boundary.inletArea,P.boundary.inletProjectedArea,flowOutletPatch,P.nu,P.re));
    } else P.nu=1.0/(double)re;
    const double nu=P.nu;
    PetscCall(PetscTime(&tProblem1));
    PetscCall(printSetupPhase("problem_boundary_config",tProblem1-tProblem0,(PetscInt)M.tets.size()));
    if(resourceProfile) PetscCall(printResourceMark("after_problem_config",(PetscInt)M.tets.size(),tProblem1-tProblem0,tTotal0));

    Discrete D;
    PetscCall(PetscTime(&tAsm0));
    PetscCall(assembleMPI(M,rank,size,P,D,m3StaticReference,m4bBReference,m6bVelocityReference));
    PetscCall(PetscTime(&tAsm1));
    PetscCall(printSetupPhase("fe_assembly",tAsm1-tAsm0,(PetscInt)M.tets.size()));
    if(resourceProfile) PetscCall(printResourceMark("after_fe_assembly",(PetscInt)M.tets.size(),tAsm1-tAsm0,tTotal0));
    if(memoryAudit) PetscCall(auditDiscreteMemory(D,(PetscInt)M.tets.size()));
    if(rank==0 && P.inletBC==InletBCMode::FixedNormalSpeed) PetscCall(auditFixedNormalInletRoot(M,D,P));

    GhostPlan G;
    PetscCall(PetscTime(&tGhost0));
    PetscCall(buildVelocityGhostPlan(M,D,rank,G));
    PetscCall(PetscTime(&tGhost1));
    PetscCall(printSetupPhase("velocity_ghost_plan",tGhost1-tGhost0,(PetscInt)M.tets.size()));
    PressureAssemblyPlan PSchur;
    const PetscBool buildExpandedSchur = (pOperatorMode=="explicit" || pPmatMode=="full" || factoredBenchmark) ? PETSC_TRUE : PETSC_FALSE;
    PetscCall(PetscTime(&tPPlan0));
    PetscCall(buildPressureAssemblyPlan(M,D,rank,G,pPmatMode,buildExpandedSchur,PSchur));
    PetscCall(PetscTime(&tPPlan1));
    PetscCall(printSetupPhase("pressure_assembly_plan",tPPlan1-tPPlan0,(PetscInt)M.tets.size()));
    CentralAssemblyPlan CPlan; SupgAssemblyPlan SupgPlan; // empty legacy plans in M2B
    PetscCall(PetscTime(&tCPlan0)); PetscCall(PetscTime(&tCPlan1));
    PetscCall(printSetupPhase("central_assembly_plan",tCPlan1-tCPlan0,(PetscInt)M.tets.size()));
    PetscCall(PetscTime(&tSupgPlan0)); PetscCall(PetscTime(&tSupgPlan1));
    PetscCall(printSetupPhase("supg_assembly_plan",tSupgPlan1-tSupgPlan0,(PetscInt)M.tets.size()));
    if(resourceProfile) PetscCall(printResourceMark("after_plans",(PetscInt)M.tets.size(),(tGhost1-tGhost0)+(tPPlan1-tPPlan0),tTotal0));
    if(memoryAudit) PetscCall(auditPlanMemory(G,PSchur,CPlan,SupgPlan,(PetscInt)M.tets.size()));

    PetscCall(PetscTime(&tObjects0));
    std::array<std::vector<double>,3> U;
    for(int d=0;d<3;++d) U[(std::size_t)d].assign((std::size_t)D.velCount[rank],0.0);

    // M3A: the production path never constructs PETSc D.A.  Build the owned-row
    // custom topology directly from the mesh, build the row-support geometry
    // plan, then integrate the static diffusion K straight into custom FP64 CSR.
    Mat C=nullptr,Sg=nullptr;
    CustomMomentumCSR customMom;
    PetscLogDouble tc0=0,tc1=0; PetscCall(PetscTime(&tc0));
    PetscCall(buildCustomMomentumCSR(M,D,rank,customMom));
    CustomDynamicAssemblyPlan DynPlan;
    PetscLogDouble tdp0=0,tdp1=0; PetscCall(PetscTime(&tdp0));
    PetscCall(buildCustomDynamicAssemblyPlan(M,D,P,customMom,supgQuadPoints,DynPlan));
    PetscCall(PetscTime(&tdp1)); PetscCall(printSetupPhase("custom_dynamic_assembly_plan",tdp1-tdp0,(PetscInt)M.tets.size()));
    PetscCall(assembleStaticDiffusionCustom(DynPlan,customMom));
    PetscCall(assembleStaticMomentumRhsNative(M,D,P,DynPlan,customMom,D.rhsOwnedFP64));
    if(m6bVelocityReference){
      double worst=0.0,maxAbs=0.0;
      for(int d=0;d<3;++d){std::vector<double> ref;PetscCall(customMomentumVecOwned(D.rhs[d],customMom,ref));CustomParityNorm q;PetscCall(customMomentumCompareArrays(D.rhsOwnedFP64[(std::size_t)d],ref,q));worst=std::max(worst,q.rel);maxAbs=std::max(maxAbs,q.maxAbs);}
      const bool ok=worst<=customMomentumShadowTol;PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_M6B_VELOCITY_STATE_REFERENCE staticRhsRel=%.3e staticRhsMaxAbs=%.3e tol=%.3e status=%s action=destroy_PETSc_velocity_rhs_before_live_solve\n",worst,maxAbs,(double)customMomentumShadowTol,ok?"PASS":"CHECK"));
      if(!ok) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M6B native static momentum RHS parity failed");for(int d=0;d<3;++d)PetscCall(VecDestroy(&D.rhs[d]));
    } else PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_M6B_VELOCITY_STATE_REFERENCE status=SKIPPED_PRODUCTION PETScVelocityRhs=never_created\n"));
    PetscCall(reindexFlatSchurRauToCustom(PSchur,G,customMom));

    unsigned long long lnnz=(unsigned long long)customMom.colGid.size(),gnnz=0;
    PetscCallMPI(MPI_Allreduce(&lnnz,&gnnz,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
    double kactRel=0.0,kactMax=0.0; const char *referenceStatus="SKIPPED_PRODUCTION";
    if(D.A) {
      MatInfo kiRef; PetscCall(MatGetInfo(D.A,MAT_GLOBAL_SUM,&kiRef));
      CustomParityNorm kact; PetscCall(customMomentumActionParity(D.A,customMom,customMom.kNu,kact));
      kactRel=kact.rel; kactMax=kact.maxAbs;
      const bool ok=(std::llround(kiRef.nz_used)==(long long)gnnz && kact.rel<=customMomentumShadowTol);
      referenceStatus=ok?"PASS":"CHECK";
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "P1BF3_M3A_STATIC_REFERENCE petscDiffusionNnz=%.0f customNnz=%llu topologyStatus=%s KActionRel=%.3e KActionMaxAbs=%.3e tol=%.3e status=%s\n",
        kiRef.nz_used,gnnz,std::llround(kiRef.nz_used)==(long long)gnnz?"PASS":"CHECK",kact.rel,kact.maxAbs,(double)customMomentumShadowTol,referenceStatus));
      if(!ok) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M3A direct static diffusion parity failed");
      PetscCall(MatDestroy(&D.A));
    }
    for(double& v:customMom.kNu) v*=nu;
    PetscCall(PetscTime(&tc1)); PetscCall(printSetupPhase("custom_momentum_plan",tc1-tc0,(PetscInt)M.tets.size()));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_M3A_STATIC mode=direct_custom_owned_rows reference=%s globalNnz=%llu KActionRel=%.3e KActionMaxAbs=%.3e PETSc_DA_live=0 status=PASS\n",
      referenceStatus,gnnz,kactRel,kactMax));

    CustomPressureBPlan customPressureB;
    PetscLogDouble tbp0=0,tbp1=0; PetscCall(PetscTime(&tbp0));
    PetscCall(buildCustomPressureBPlan(M,D,rank,customPressureB));
    const char *bReferenceStatus="SKIPPED_PRODUCTION";
    if(m4bBReference) {
      PetscCall(customPressureBStaticParity(D,customPressureB,(double)customPressureBShadowTol));
      bReferenceStatus="PASS";
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_M4B_B_REFERENCE status=PASS action=destroy_PETSc_B_before_live_solve\n"));
      for(int d=0;d<3;++d) PetscCall(MatDestroy(&D.B[d]));
    } else {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_M4B_B_REFERENCE status=SKIPPED_PRODUCTION PETScB=never_created\n"));
    }
    PetscCall(PetscTime(&tbp1)); PetscCall(printSetupPhase("custom_pressure_B_live_plan",tbp1-tbp0,(PetscInt)M.tets.size()));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_M4B_CONFIG mode=live_custom_pressure physicalB=custom_FP64_MPI physicalBt=custom_FP64_MPI physicalSchur=custom_FP64_MPI PETScB=%s Pmat=%s reference=%s tol=%.3e\n",
      m4bBReference?"destroyed_after_reference":"never_created",pPmatMode=="native_face"?"PETSc_FP64_native_face_compact_GAMG":"PETSc_FP64_full_GAMG",bReferenceStatus,(double)customPressureBShadowTol));

    const double feScalarNnz=(double)gnnz;
    const double fvmScalarNnz=(double)M.tets.size()+2.0*(double)M.neighbour.size();
    const double feAvgNnz=feScalarNnz/(double)D.ns;
    const double fvmAvgNnz=fvmScalarNnz/(double)M.tets.size();
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_COST_MODEL feScalarDofs=%" PetscInt_FMT " fvmScalarDofs=%zu dofRatio=%.6f feScalarNnz=%.0f fvmCompactScalarNnz=%.0f feAvgNnzPerRow=%.6f fvmAvgNnzPerRow=%.6f stencilRowRatio=%.6f scalarMatrixNnzRatio=%.6f\n",
      D.ns,M.tets.size(),(double)D.ns/(double)M.tets.size(),feScalarNnz,fvmScalarNnz,feAvgNnz,fvmAvgNnz,feAvgNnz/fvmAvgNnz,feScalarNnz/fvmScalarNnz));
    if(memoryAudit) {
      unsigned long long lb=(unsigned long long)DynPlan.cells.capacity()*sizeof(CustomDynamicCellPlan)+(unsigned long long)DynPlan.ref.capacity()*sizeof(SupgReferencePoint)+(unsigned long long)DynPlan.forcing.capacity()*sizeof(double),gb=0;
      PetscCallMPI(MPI_Allreduce(&lb,&gb,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,PETSC_COMM_WORLD));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_MEMORY_AUDIT_OBJECT stage=after_state_objects name=custom.dynamicPlan scope=distributed_row_support_halo usefulBytes=%llu retainedEstimateBytes=%llu retainedMiB=%.3f retainedBytesPerCell=%.3f note=combined_central_SUPG_geometry_plan\n",gb,gb,(double)gb/(1024.0*1024.0),M.tets.empty()?0.0:(double)gb/(double)M.tets.size()));
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_M3A_MATRIX_LIFETIME D_A=never_created_in_production Knu=direct_custom_FP64 C=eliminated Sg=eliminated Aphys=eliminated Ar=custom_active_array liveMomentum=custom_FP64_MPI_SGS staticAssembly=direct_owned_rows dynamicAssembly=direct_owned_rows\n"));

    // M6B: velocity, momentum RHS/work, diagonal metrics and rAU are all native FP64 arrays.
    // PETSc state vectors are now pressure preconditioner bridges only.

    // M6A: pressure physical state is native C++ FP64.  PETSc pressure Vecs are
    // now only two bridge buffers used by GAMG PCApply (and the optional first
    // reference KSPSolve); no physical p/r/dp state lives in PetscScalar.
    Vec pcIn=nullptr,pcOut=nullptr;
    PetscCall(VecDuplicate(D.volumes,&pcIn)); PetscCall(VecSet(pcIn,0));
    PetscCall(VecDuplicate(D.volumes,&pcOut)); PetscCall(VecSet(pcOut,0));
    std::vector<double> pressureState((std::size_t)customPressureB.pressureHalo.nOwned,0.0);
    std::vector<double> pressureResidual((std::size_t)customPressureB.pressureHalo.nOwned,0.0);
    CustomPressurePCGWorkspace pressurePcgW;
    pressurePcgW.x.reserve((std::size_t)customPressureB.pressureHalo.nOwned);
    pressurePcgW.r.reserve((std::size_t)customPressureB.pressureHalo.nOwned);
    pressurePcgW.z.reserve((std::size_t)customPressureB.pressureHalo.nOwned);
    pressurePcgW.p.reserve((std::size_t)customPressureB.pressureHalo.nOwned);
    PetscBool pressurePcgReferenceDone=PETSC_FALSE;
    M10PressurePCGProfile m10PcgTotal,m10PcgWarm; m10PcgTotal.enabled=m10PcgProfile; m10PcgWarm.enabled=m10PcgProfile;

    CustomFactoredPressureContext factoredCtx;
    Mat factoredSchur=nullptr;
    PetscCall(createCustomFactoredPressure(customPressureB,customMom,D.volumes,factoredCtx,&factoredSchur));
    Mat pressureOperator = (pOperatorMode=="factored") ? factoredSchur : PSchur.S;
    PetscBool factoredBenchmarkDone=PETSC_FALSE,customPressureLiveSchurParityDone=PETSC_FALSE;
    PetscBool gate1FgmresParityDone=PETSC_FALSE;

    MatNullSpace nsp=nullptr;
    KSP pksp=nullptr;
    PC ppc=nullptr;
    PC gate3MpShell=nullptr;
    Gate3MpShellCtx *gate3MpCtx=nullptr;
    Gate4KpCtx gate4Kp;
    Gate9eNodalKpAudit gate9eAudit;
    Gate9gFeFaceAudit gate9gAudit;
    double gate9iLambdaHat=0.0,gate9iLambdaMinActive=0.0,gate9iLambdaMaxActive=0.0;
    PetscBool gate9iEstimatePending=gate9iAutoChebyshev?PETSC_TRUE:PETSC_FALSE;
    PetscInt gate9iEstimateCount=0,gate9iTotalPowerIts=0,gate9iTotalChebSteps=0,gate9iPressureSolves=0;
    double gate9iPowerSeconds=0.0,gate9iChebSeconds=0.0;
    Gate5KpGamgStats gate5KpGamg;
    PC gate6DiffusionPcdShell=nullptr;
    Gate6DiffusionPcdCtx *gate6DiffusionPcdCtx=nullptr;
    Gate7CpCtx gate7Cp;
    Gate8EswBcCtx gate8EswBc;
    Gate9LivePcdCtx *gate9LivePcdCtx=nullptr;
    KSPType pKspType=nullptr;
    PCType pPcType=nullptr;
    // The pressure preconditioning matrix object is fixed for the solve.  With
    // a factored exact operator and p_pmat=full, the explicit Schur is a lagged
    // GAMG snapshot: its values change only when the preconditioner is refreshed.
    Mat pressurePmat = (pPmatMode=="full") ? PSchur.S : PSchur.Pcompact;
    if(gate9gFeFace) PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_GATE9G_CONFIG enabled=1 Kp=compact_FE_face_jump_energy graph=cell_plus_face_neighbours coefficient=quarter_sum_shared_rAU_norm_Bp_minus_Bn_sq outletAnchor=FE_trace_energy traceMatch=exact_Schur_diagonal_sum exactOperator=custom_FP64_B_rAU_Bt tests=%s\n",
      gate9iAutoChebyshev?"AUTO_Chebyshev_power_estimate_plus_one_GAMG_no_outer_Krylov":(gate9hChebyshev?"Chebyshev_plus_one_GAMG_no_outer_Krylov":(gate9gRichardson?"Richardson_plus_one_GAMG_no_outer_Krylov":"custom_PCG_plus_one_GAMG"))));
    if(gate9iAutoChebyshev) PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_GATE9I_CONFIG enabled=1 pressureAlgorithm=auto_preconditioned_Chebyshev outerKrylov=NONE KSPSolve=NEVER_CALLED exactOperator=custom_FP64_B_rAU_Bt Kp=FE_face_jump_energy PC=ONE_GAMG_PCApply_per_stage spectrumEstimate=power_iteration_on_MinvS estimateCadence=%s spectrumRefresh=%" PetscInt_FMT " powerIts=%" PetscInt_FMT " lambdaMaxSafety=%.6e lambdaMinFraction=%.6e rtol=%.6e atol=%.6e mode=%s fixedSteps=%" PetscInt_FMT " requireTarget=%d initialBlock=%" PetscInt_FMT " extendBlock=%" PetscInt_FMT " maxSteps=%" PetscInt_FMT "\n",
      gate9iSpectrumRefresh>0?"independent_SIMPLE_cadence":"PC_refresh_only",gate9iSpectrumRefresh,gate9iPowerIts,(double)gate9iSafety,(double)gate9iLambdaMinFraction,(double)gate9iRtol,(double)gate9iAtol,gate9iFixedSteps>0?"fixed":"adaptive",gate9iFixedSteps,(int)gate9iRequireTarget,gate9iInitialBlock,gate9iExtendBlock,gate9iMaxSteps));
    if(gate9hChebyshev) PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_GATE9H_CONFIG enabled=1 pressureAlgorithm=preconditioned_Chebyshev_semiteration outerKrylov=NONE KSPSolve=NEVER_CALLED exactOperator=custom_FP64_B_rAU_Bt Kp=FE_face_jump_energy PC=ONE_GAMG_PCApply_per_step lambdaMin=%.12e lambdaMax=%.12e maxSteps=%" PetscInt_FMT " checkEvery=%" PetscInt_FMT " reductions=diagnostic_norm_checks_only\n",
      (double)gate9hLambdaMin,(double)gate9hLambdaMax,gate9hMaxSteps,gate9hCheckEvery));
    if(gate4KpProbe) {
      const char* gate4GamgRole=gate9LivePcd?(gate9dGamgOnly?"LIVE_GATE9D_GAMG_ONLY_STATIONARY":"LIVE_GATE9C_ONE_GAMG_PCApply"):(gate5KpGamgProbe?"STANDALONE_GATE5_ONLY":(gate6DiffusionPcdProbe?"GATE6_KP_INVERSE_SHADOW_ONLY":"NOT_YET"));
      const char* gate4FpRole=gate9LivePcd?(gate9dGamgOnly?"BYPASSED_GATE9D_GAMG_ONLY":"BYPASSED_GATE9C_DIRECT_KP_GAMG"):(gate8EswBcProbe?"nuKp_plus_internal_Cp_plus_ESW_BC_GATE8":(gate7CpProbe?"nuKp_plus_internal_Cp_GATE7":(gate6DiffusionPcdProbe?"nu_times_Kp_GATE6":"NONE")));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_GATE4_KP_CONFIG enabled=1 attachment=shadow_setup_only exactPressureOperator=custom_FP64_B_rAU_Bt liveOuter=PETSc_FGMRES livePC=existing_native_face_GAMG Mp=validated_GATE3_not_applied Kp=geometric_FV_laplacian Fp=%s GAMG_on_Kp=%s pressureBC_Kp=inlet_Neumann_wall_Neumann_outlet_p0\n",gate4FpRole,gate4GamgRole));
      if(gate9eNgfv) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_GATE9E_CONFIG enabled=1 nodalFit=wide_affine_P1 faceGradient=mean_3_nodes rawFV=-GaussDivGrad pcgSafeKp=symmetric_Mmatrix_projection exactSchur=UNCHANGED_custom_FP64_B_rAU_Bt tests=PCG_plus_one_GAMG_and_Richardson_plus_one_GAMG\n"));
        PetscCall(gate9eBuildNodalKp(M,D,P,rank,gate4Kp,gate9eAudit));
        if(!gate9LivePcd) pressurePmat=gate4Kp.Kp;
      } else PetscCall(gate4BuildKp(M,D,P,rank,gate4Kp));
      if(gate5KpGamgProbe) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_GATE5_CONFIG enabled=1 attachment=shadow_standalone_only Kp=validated_GATE4_geometric_FV_laplacian KpBC=inlet_Neumann_wall_Neumann_outlet_p0 pressureNullspace=OFF primarySolve=CG_plus_GAMG tightRtol=1e-10 gamgOnlyProbe=4_direct_PCApply_Vcycles_no_outer_Krylov Fp=NONE fullPCD=NOT_YET livePressureSolve=UNCHANGED\n"));
        PetscCall(gate5StandaloneKpGamg(gate4Kp,gate5KpGamg));
      }
      if(gate7CpProbe) {
        PetscCall(gate7CpSetUp(gate4Kp,gate7Cp));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_GATE7_CONFIG enabled=1 attachment=shadow_only Cp=internal_face_conservative_central sourceVelocity=current_Picard_P1plusBF3 exactFaceFlux=1 faceMean=P1_vertex_mean_plus_9over20_BF3 Fp=nuKp_plus_internal_Cp pressureBoundaryConvection=OFF robin=%s KpInverse=NOT_APPLIED_GATE7 liveOuter=PETSc_FGMRES livePC=existing_native_face_GAMG exactPressureOperator=custom_FP64_B_rAU_Bt fullPCD=NOT_YET\n",gate8EswBcProbe?"HANDLED_SEPARATELY_GATE8":"NOT_YET_GATE8"));
      }
      if(gate8EswBcProbe) {
        PetscCall(gate8EswBcSetUp(gate4Kp,gate8EswBc));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
          "P1BF3_GATE8_CONFIG enabled=1 attachment=shadow_only ESW918_chain=KpInv_Fp_MpInv Fp=nuKp_plus_internal_Cp_plus_ESW_boundary_semantics inletRobinFormula=-nu_dpdn_plus_wdotn_p_eq_0 inletRobinActive=1 inletDiscretization=explicit_conservative_convection_plus_explicit_Robin_cancellation sourceVelocity=current_Picard_P1plusBF3 exactFaceMean=1 wallBC=homogeneous_Neumann_wdotn0 outletBC=p0_Dirichlet pressureNullspace=OFF KpInverse=NOT_APPLIED_GATE8 liveOuter=PETSc_FGMRES livePC=existing_native_face_GAMG exactPressureOperator=custom_FP64_B_rAU_Bt cancelTol=%.3e wallFluxTol=%.3e fullPCD=NOT_LIVE_YET\n",
          (double)gate8CancelTol,(double)gate8WallFluxTol));
      }
      if(gate9LivePcd) {
        gate9LivePcdCtx=new Gate9LivePcdCtx();
        gate9LivePcdCtx->Kp=gate4Kp.Kp; gate9LivePcdCtx->volumes=D.volumes;
        gate9LivePcdCtx->gamgOnlySolve=gate9dGamgOnly;
        if(gate9dGamgOnly) PetscCall(PetscPrintf(PETSC_COMM_WORLD,
          "P1BF3_GATE9D_CONFIG enabled=1 pressureAlgorithm=stationary_exactSchur_residual_correction outerKrylov=NONE KSPSolve=NEVER_CALLED correction=omega_times_ONE_GAMG_PCApply_on_selected_Kp omega=%.6e maxCycles=%" PetscInt_FMT " trueResidualTol=%.3e divergenceFactor=%.3e exactOperator=custom_FP64_B_rAU_Bt Mp=BYPASSED Fp=BYPASSED innerKrylov=NONE\n",
          (double)gate9dOmega,gate9dMaxCycles,(double)gate9OuterTrueResidualTol,(double)gate9dDivergenceFactor));
        else PetscCall(PetscPrintf(PETSC_COMM_WORLD,
          "P1BF3_GATE9C_CONFIG enabled=1 attachment=LIVE_PCSHELL outer=FGMRES exactOperator=custom_FP64_B_rAU_Bt preconditioner=ONE_direct_GAMG_PCApply_on_geometric_Kp Mp=BYPASSED Fp=BYPASSED innerKrylov=NONE pcSide=RIGHT nativeFaceGAMG=NOT_LIVE_PC outerTrueResidualTol=%.3e\n",
          (double)gate9OuterTrueResidualTol));
      }
      if(gate6DiffusionPcdProbe) {
        gate6DiffusionPcdCtx=new Gate6DiffusionPcdCtx();
        gate6DiffusionPcdCtx->Kp=gate4Kp.Kp;
        gate6DiffusionPcdCtx->volumes=D.volumes;
        gate6DiffusionPcdCtx->nu=(PetscReal)P.nu;
        PetscCall(PCCreate(PETSC_COMM_WORLD,&gate6DiffusionPcdShell));
        PetscCall(PCSetType(gate6DiffusionPcdShell,PCSHELL));
        PetscCall(PCShellSetName(gate6DiffusionPcdShell,"P1BF3_PCD_GATE6_DIFFUSION_CHAIN_SHADOW"));
        PetscCall(PCShellSetContext(gate6DiffusionPcdShell,(void*)gate6DiffusionPcdCtx));
        PetscCall(PCShellSetSetUp(gate6DiffusionPcdShell,gate6DiffusionPcdSetUp));
        PetscCall(PCShellSetApply(gate6DiffusionPcdShell,gate6DiffusionPcdApply));
        PetscCall(PCShellSetDestroy(gate6DiffusionPcdShell,gate6DiffusionPcdDestroy));
        PetscCall(PCSetOperators(gate6DiffusionPcdShell,pressureOperator,pressurePmat));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
          "P1BF3_GATE6_CONFIG enabled=1 attachment=shadow_only chain=Kp_inverse_Fp_Mp_inverse Mp=P0_exact_cell_volume_diagonal Kp=validated_GATE4_geometric_FV_laplacian Fp=nu_times_Kp nu=%.12e KpBC=inlet_Neumann_wall_Neumann_outlet_p0 KpInverse=tight_CG_plus_GAMG liveOuter=PETSc_FGMRES livePC=existing_native_face_GAMG exactPressureOperator=custom_FP64_B_rAU_Bt chainTol=%.3e kpTrueResidualTol=%.3e fullPCD=NOT_YET\n",
          P.nu,(double)gate6ChainTol,(double)gate6KpTrueResidualTol));
      }
    }
    if(gate3MpProbe) {
      gate3MpCtx=new Gate3MpShellCtx();
      gate3MpCtx->volumes=D.volumes; // borrowed; exact P0 cell-volume mass diagonal
      PetscCall(PCCreate(PETSC_COMM_WORLD,&gate3MpShell));
      PetscCall(PCSetType(gate3MpShell,PCSHELL));
      PetscCall(PCShellSetName(gate3MpShell,"P1BF3_PCD_GATE3_MP_INVERSE_SHADOW"));
      PetscCall(PCShellSetContext(gate3MpShell,(void*)gate3MpCtx));
      PetscCall(PCShellSetSetUp(gate3MpShell,gate3MpShellSetUp));
      PetscCall(PCShellSetApply(gate3MpShell,gate3MpShellApply));
      PetscCall(PCShellSetDestroy(gate3MpShell,gate3MpShellDestroy));
      PetscCall(PCSetOperators(gate3MpShell,pressureOperator,pressurePmat));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "P1BF3_GATE3_MP_CONFIG enabled=1 attachment=shadow_only action=Mp_inverse mass=P0_exact_cell_volume_diagonal exactPressureOperator=custom_FP64_B_rAU_Bt liveOuter=PETSc_FGMRES livePC=existing_native_face_GAMG Kp=NONE Fp=NONE pressureBC=UNCHANGED_NOT_USED algebraTol=%.3e\n",
        (double)gate3MpAlgebraTol));
    }
    const PetscBool lagFullPmat = (pOperatorMode=="factored" && pPmatMode=="full" && pPreconditionerRefresh>0) ? PETSC_TRUE : PETSC_FALSE;
    const PetscBool lagNativeCompactPmat = (pOperatorMode=="factored" && pPmatMode=="native_face" && pPreconditionerRefresh>0) ? PETSC_TRUE : PETSC_FALSE;
    if(lagNativeCompactPmat) PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_B2_COMPACT_LAG_CONFIG enabled=1 interval=%" PetscInt_FMT " exactOperator=factored_B_rAU_Bt pmat=native_face_compact semantics=assemble_compact_only_on_GAMG_refresh\n",
      pPreconditionerRefresh));
    if(lagFullPmat) PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_FULL_PMAT_LAG_CONFIG enabled=1 interval=%" PetscInt_FMT " exactOperator=factored_B_rAU_Bt pmat=full_explicit_Schur_snapshot semantics=assemble_Pmat_only_on_GAMG_refresh\n",
      pPreconditionerRefresh));
    if(pPmatMode=="full") PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_M4B_PRESSURE_ARCH physicalB=custom_FP64_MPI physicalBt=custom_FP64_MPI exactOperator=custom_FP64_B_rAU_Bt PETScB=none_live Pmat=PETSc_FP64_full GAMG=PETSc_FP64 outerKrylov=custom_FP64_PCG PETScKSPRole=GAMG_PC_owner_only\n"));
    if(pPmatMode=="full") PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_M5A_CONFIG Bcell=eliminated schurStorage=flat_CSR_terms pmatRefresh=analytic_B_coefficients CustomOuterPCG=FP64 PETScGAMG=FP64\n"));
    if(pPmatMode=="full") PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_M5B_PCG_CONFIG outer=custom_FP64_MPI_PCG exactOperator=custom_FP64_B_rAU_Bt preconditioner=PETSc_FP64_GAMG_PCApply PETScKSPSolve=%s referenceTol=%.3e\n",
      m5bPcgReference?"first_solve_reference_only":"never_called_in_production",(double)customPressurePcgReferenceTol));
    if(pPmatMode=="full") PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_M6A_PRESSURE_STATE_CONFIG physicalP=native_CPP_FP64 continuity=native_CPP_FP64 pressureRhs=native_CPP_FP64 correction=custom_PCG_FP64 PETScPressureVecs=pcIn_pcOut_bridge_only staticVolumeFixedDiv=native_FP64_plus_layout_bridge eventualFP32SafePressureState=1\n"));
    if(pPmatMode=="full") PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_M6B_VELOCITY_STATE_CONFIG physicalU=native_CPP_FP64 staticMomentumRhs=native_CPP_FP64 dynamicMomentumRhs=native_CPP_FP64 momentumWork=native_CPP_FP64 diagMetricRau=custom_CPP_FP64 PETScVelocityVecs=none_live PETScPhysicalState=none_live eventualFP32SafeVelocityState=1 reference=%s\n",m6bVelocityReference?"PASS":"SKIPPED_PRODUCTION"));
    if(pPmatMode=="native_face") {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "P1BF3_B2_PRESSURE_ARCH exactOperator=custom_FP64_B_rAU_Bt outerKrylov=%s preconditioner=PETSc_FP64_GAMG pmat=native_face_compact graph=cell_plus_face_neighbours exactFullSchur=never_materialized flatSchurPlan=eliminated Bcell=eliminated vertexCells=eliminated p1RedistributionStrength=%.6g refresh=%" PetscInt_FMT "\n",
        pressureSolveMode=="petsc_fgmres"?"PETSc_FGMRES":(gate9iAutoChebyshev?"FP64_Chebyshev_no_outer_Krylov":"custom_FP64_PCG"),(double)feFvP1Strength,pPreconditionerRefresh));
      if(pressureSolveMode=="custom_pcg" && !gate9iAutoChebyshev && !gate9hChebyshev && !gate9gRichardson) PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "P1BF3_M5B_PCG_CONFIG outer=custom_FP64_MPI_PCG exactOperator=custom_FP64_B_rAU_Bt preconditioner=PETSc_FP64_GAMG_PCApply PETScKSPSolve=%s referenceTol=%.3e\n",
        m5bPcgReference?"first_solve_reference_only":"never_called_in_production",(double)customPressurePcgReferenceTol));
    }

    double localVolsum=std::accumulate(D.volumesOwnedFP64.begin(),D.volumesOwnedFP64.end(),0.0),volsum=0.0;
    PetscCallMPI(MPI_Allreduce(&localVolsum,&volsum,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));
    PetscCall(PetscTime(&tObjects1));
    PetscCall(printSetupPhase("state_matrix_vector_objects",tObjects1-tObjects0,(PetscInt)M.tets.size()));
    if(resourceProfile) PetscCall(printResourceMark("after_state_objects",(PetscInt)M.tets.size(),tObjects1-tObjects0,tTotal0));
    if(memoryAudit) {
      PetscCall(auditStateObjects(C,Sg,nullptr,nullptr,nullptr,D,G,(PetscInt)M.tets.size()));
      const unsigned long long cpb=customPressurePlanLocalBytes(customPressureB);
      PetscCall(printMemoryAuditBytes("after_state_objects","custom.pressureBPlan",cpb,cpb,(PetscInt)M.tets.size(),"distributed_owned_rows_plus_halo","live_FP64_B_Bt_geometry_and_peer_exchange_plan"));
      const unsigned long long cgb=customPressurePCGWorkspaceBytes(pressurePcgW);
      PetscCall(printMemoryAuditBytes("after_state_objects","custom.pressurePCGWorkspace",cgb,cgb,(PetscInt)M.tets.size(),"distributed_owned_pressure_rows","four_FP64_owned_vectors_x_r_z_p"));
      const unsigned long long psb=(unsigned long long)(pressureState.capacity()+pressureResidual.capacity())*sizeof(double);
      PetscCall(printMemoryAuditBytes("after_state_objects","custom.pressurePhysicalState",psb,psb,(PetscInt)M.tets.size(),"distributed_owned_pressure_rows","native_FP64_p_plus_continuity_rhs_no_PetscScalar"));
      unsigned long long vbytes=0;for(int d=0;d<3;++d)vbytes+=(unsigned long long)(U[(std::size_t)d].capacity()+D.rhsOwnedFP64[(std::size_t)d].capacity())*sizeof(double);
      PetscCall(printMemoryAuditBytes("after_state_objects","custom.velocityPhysicalState",vbytes,vbytes,(PetscInt)M.tets.size(),"distributed_owned_velocity_rows","native_FP64_U3_plus_static_rhs3; dynamic_rhs_and_metrics_already_in_custom_momentum_CSR"));
    }

    if(m10PcgProfile) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_M10_PROFILE_CONFIG enabled=1 scope=custom_pressure_PCG exactSchurBreakdown=Bt_rAU_B pcApplyBreakdown=bridge_kernel reductions=timed_MPI_allreduce vectorOps=timed_native_loops warmDefinition=exclude_first_pressure_solve\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_PHYSICS problem=%s equation=steady_navier_stokes Re=%.12g nu=%.12g ReConvention=%s convection=%s supg=%s tauScale=%.6g supgMagic=%.6g tauH=tet_max_edge strongResidual=-nu_laplacian_u_plus_a_dot_grad_u_minus_f pressureStrongGrad=P0_zero linearization=Picard_Oseen supgTauLinearization=lagged_previous_SIMPLE_state supgForm=%s supgKernel=%s supgQuad=%" PetscInt_FMT "\n",
      problem.c_str(),P.re,nu,P.mode==ProblemMode::Pipe?"Uchar_D_over_nu":"unit_scale_1_over_nu",centralConvection?"central":"none",useSupg?"ON":"OFF",(double)supgTauScale,(double)supgMagic,supgForm.c_str(),supgKernel.c_str(),supgQuadPoints));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_SIMPLE_CONFIG ranks=%d variant=%s alphaU=%.6g alphaP=%.6g uRelaxMode=%s rauScale=%.6g rauMode=%s "
      "simplecBlend=%.6g simplecFloorFraction=%.3e simplecFallback=%s rAU=rauScale/correction_metric simpleRtol=%.3e maxOuter=%" PetscInt_FMT
      " velocitySolver=processorBlockJacobi+localSGS precision=FP64 uRtol=%.3e uAtol=%.3e uRelDrop=%.3g uOmega=%.3g uLocalSweeps=%" PetscInt_FMT " uCheckEvery=%" PetscInt_FMT
      " outerGate=all_initial_residuals_Ux_Uy_Uz_plus_continuity pressureSolver=custom_FP64_PCG_plus_PETSc_PCApply pressureState=native_FP64 pPcRefresh=%" PetscInt_FMT "\n",
      size,simpleVariant.c_str(),(double)au,(double)ap,uRelaxMode.c_str(),(double)rauScale,rauMode.c_str(),
      (double)simplecBlend,(double)simplecFloorFraction,simplecFallback.c_str(),(double)simpleTol,maxOuter,
      (double)uTol,(double)uAtol,(double)uRelDrop,(double)uOmega,uLocalSweeps,uCheck,pPreconditionerRefresh));
    if(simpleVariant=="simplec") PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_SIMPLEC_CONFIG metric=relaxed_signed_row_sum formula=Dc=(1-blend)*diag(Ar)+blend*(Ar*1) standardBlend=1 "
      "pressureUnderRelaxation=alphaP userControlled note=alphaP_1_is_typical_SIMPLEC_test positiveGuard=%s floorFraction=%.3e\n",
      simplecFallback.c_str(),(double)simplecFloorFraction));
    if(useSupg) PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_SUPG_CONFIG tauLinearization=lagged_previous_SIMPLE_state form=%s kernel=%s quadraturePoints=%" PetscInt_FMT " tauScale=%.6g magic=%.6g affineGeometry=compact_gradLambda_cached physicalGrad=runtime_affine_reconstructed viscousStrong=runtime_affine_reconstructed assembly=outer_product explicitFormBuildsGlobalSUPGMatrix=%d\n",
      supgForm.c_str(),supgKernel.c_str(),supgQuadPoints,(double)supgTauScale,(double)supgMagic,supgForm=="explicit"?0:1));

    double r0=0.0,rel=std::numeric_limits<double>::max();
    PetscBool converged=PETSC_FALSE;
    PetscBool solveFailed=PETSC_FALSE;
    PetscInt pressureFailureReason=0;
    PetscInt finalIt=0;
    PetscReal finalUInitRel[3]={std::numeric_limits<PetscReal>::max(),std::numeric_limits<PetscReal>::max(),std::numeric_limits<PetscReal>::max()};
    PetscReal finalPInitRel=std::numeric_limits<PetscReal>::max();
    long long sumU[3]={0,0,0},sumP=0;
    PetscInt pSolves=0;
    double operatorUpdateSeconds=0.0;
    double convectionUpdateSeconds=0.0,supgUpdateSeconds=0.0,derivedUpdateSeconds=0.0,customMomentumLoadSeconds=0.0,schurUpdateSeconds=0.0,kspOperatorSeconds=0.0;
    double momentumSolveSeconds=0.0,pressureSolveSeconds=0.0,pressurePcRefreshSeconds=0.0;
    PetscInt pressurePcRefreshes=0,pressurePcReuses=0;
    PetscBool pressureProfileDone=PETSC_FALSE;
    const PetscInt fvCompactPressureNnz=(PetscInt)M.tets.size()+2*(PetscInt)M.neighbour.size();
    SupgStats supgStats;
    const bool staticPhysicalOperator=(!centralConvection && !useSupg);
    bool operatorReady=false;
    PetscReal minPhysDiag=0,maxPhysDiag=0,minDiag=0,maxDiag=0,minRauMetric=0,maxRauMetric=0;
    PetscInt minIdx=-1,maxIdx=-1;

    if(resourceProfile) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_RESOURCE_PROFILE enabled=1 hostMetric=/proc/self/status_rss_hwm aggregation=sum_and_max_across_ranks gpuMetric=external_nvidia_smi_runner timestamp=epochMs\n"));
    if(memoryAudit) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_MEMORY_AUDIT enabled=1 accounting=cpp_container_capacity_plus_PETSc_MatInfo note=unordered_map_and_allocator_overhead_are_estimates_or_unaccounted\n"));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "P1BF3_MEMORY_AUDIT_ABI sizeofPetscInt=%zu sizeofPetscScalar=%zu sizeofVoidPtr=%zu sizeofVector=%zu sizeofCentralCellPlan=%zu sizeofSupgCellPlan=%zu sizeofSupgReferencePoint=%zu\n",
        sizeof(PetscInt),sizeof(PetscScalar),sizeof(void*),sizeof(std::vector<int>),sizeof(CentralCellPlan),sizeof(SupgCellPlan),sizeof(SupgReferencePoint)));
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_OPTIMIZATION affineGeometryCached=1 centralTensorDegree=8 centralTensorRule=collapsed_5x5x5 staticDiffusion=nuK_cached schurTopologyCached=1 physicalOperatorMode=%s supgFastPath=%s supgAffineGradCached=%d supgViscStrongCached=%d supgOuterProduct=1 supgTauLagged=1 supgForm=%s supgQuad=%" PetscInt_FMT "\n",
      staticPhysicalOperator?"static_once":"convection_values_only",useSupg?supgKernel.c_str():"OFF_no_SUPG_work",
      0,0,supgForm.c_str(),supgQuadPoints));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_CUSTOM_MOM_CONFIG enabled=1 mode=M6B_native_FP64_velocity_state storage=custom_FP64_staticK_plus_activeA mpi=peer_only_nonblocking_value_exchange rowSupport=all_incident_tets liveSolver=custom_MPI_symmetric_GS PETScMomentum=none_static_or_dynamic staticAssembly=direct_custom_owned_rows dynamicAssembly=direct_custom_owned_rows tol=%.3e\n",
      (double)customMomentumShadowTol));

    PetscCall(PetscTime(&tSolve0));
    PetscCall(printSetupPhase("pre_solve_total",tSolve0-tTotal0,(PetscInt)M.tets.size()));
    if(resourceProfile) PetscCall(printResourceMark("solve_begin",(PetscInt)M.tets.size(),tSolve0-tTotal0,tTotal0));
    for(PetscInt it=1; it<=maxOuter; ++it) {
      // In the Stokes/no-convection case every object below is constant.  Build
      // Ar, rAU and S once and reuse them for all SIMPLE corrections.  For the
      // Oseen case only C(u^k) is physically dynamic; the derived relaxation
      // diagonal/rAU/Schur values are refreshed from that changed C.
      if(!operatorReady || !staticPhysicalOperator) {
        PetscLogDouble top0,top1,ts0,ts1;
        PetscCall(PetscTime(&top0));

        // M2B: gather current velocity once through the custom peer-only halo,
        // reset active values to static diffusion, then assemble every dynamic
        // element contribution directly into locally-owned CSR rows.
        PetscLogDouble tcm0=0,tcm1=0; PetscCall(PetscTime(&tcm0));
        PetscCall(customMomentumGatherVelocityNative(customMom,U));
        if(gate7CpProbe) PetscCall(gate7UpdateCpInterior(M,D,P,rank,customMom,gate7Cp,it));
        if(gate8EswBcProbe) PetscCall(gate8UpdateEswBoundary(M,D,P,rank,customMom,gate8EswBc,it));
        if(gate9LivePcd && gate9LivePcdCtx) gate9LivePcdCtx->currentSimpleIt=it;
        PetscCall(customMomentumResetPhysical(customMom));
        PetscCall(PetscTime(&tcm1)); customMomentumLoadSeconds += (double)(tcm1-tcm0);

        if(centralConvection) {
          PetscCall(PetscTime(&ts0));
          PetscCall(assembleCentralConvectionCustom(D,DynPlan,customMom));
          PetscCall(PetscTime(&ts1)); convectionUpdateSeconds += (double)(ts1-ts0);
        } else if(!operatorReady) for(auto& r:customMom.convRhs) std::fill(r.begin(),r.end(),0.0);

        if(useSupg) {
          if(supgKernel!="fast") SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"M2B direct dynamic assembly currently supports -supg_kernel fast only");
          PetscCall(PetscTime(&ts0));
          PetscCall(assembleSupgCustom(D,DynPlan,customMom,(double)supgTauScale,(double)supgMagic,supgForm,supgStats));
          PetscCall(PetscTime(&ts1)); supgUpdateSeconds += (double)(ts1-ts0);
        } else if(!operatorReady) { for(auto& r:customMom.supgRhs) std::fill(r.begin(),r.end(),0.0); supgStats={}; }

        PetscCall(PetscTime(&ts0));
        PetscCall(customMomentumFinalizeRelaxation(customMom,uRelaxMode,(double)au,simpleVariant,rauMode,
          (double)simplecBlend,(double)simplecFloorFraction,simplecFallback,(double)rauScale));

        double qmin=0,qmax=0;PetscCall(customGlobalMinMax(customMom.physDiag,qmin,qmax));minPhysDiag=(PetscReal)qmin;maxPhysDiag=(PetscReal)qmax;PetscCall(customGlobalMinMax(customMom.relaxedDiag,qmin,qmax));minDiag=(PetscReal)qmin;maxDiag=(PetscReal)qmax;PetscCall(customGlobalMinMax(customMom.metric,qmin,qmax));minRauMetric=(PetscReal)qmin;maxRauMetric=(PetscReal)qmax;
        if(it<=10 || it%10==0) PetscCall(PetscPrintf(PETSC_COMM_WORLD,
          "P1BF3_M2B_DERIVED it=%" PetscInt_FMT " aPhysDiag=[%.6e,%.6e] aRelDiag=[%.6e,%.6e] rauMetric=[%.6e,%.6e] source=direct_custom_dynamic derive=custom_FP64\n",
          it,(double)minPhysDiag,(double)maxPhysDiag,(double)minDiag,(double)maxDiag,(double)minRauMetric,(double)maxRauMetric));
        PetscCall(PetscTime(&ts1));
        derivedUpdateSeconds += (double)(ts1-ts0);

        PetscCall(PetscTime(&ts0));
        const PetscBool pcRefreshStep = (!pksp || (pPreconditionerRefresh>0 && (((it-1)%pPreconditionerRefresh)==0))) ? PETSC_TRUE : PETSC_FALSE;
        const PetscBool needExplicitSchur = (pOperatorMode=="explicit" ||
          (pPmatMode=="full" && (!lagFullPmat || pcRefreshStep)) ||
          (factoredBenchmark && !factoredBenchmarkDone && it==factoredBenchmarkAt)) ? PETSC_TRUE : PETSC_FALSE;
        const PetscBool needNativeCompact = (pPmatMode=="native_face" && (!lagNativeCompactPmat || pcRefreshStep)) ? PETSC_TRUE : PETSC_FALSE;
        if(needExplicitSchur) PetscCall(updatePressureSchurFullNative(customMom,PETSC_TRUE,PSchur));
        if(needNativeCompact) {
          if(gate9gFeFace) PetscCall(updatePressureCompactFeFaceEnergy(M,D,P,rank,customPressureB,customMom,PSchur,gate9gAudit));
          else PetscCall(updatePressureCompactNative(M,D,rank,customPressureB,customMom,(double)feFvP1Strength,PSchur));
        }
        if(pPmatMode!="full" && pPmatMode!="native_face")
          SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"M11 native velocity path supports production pressure Pmat modes full or native_face only");
        if(lagFullPmat && needExplicitSchur) PetscCall(PetscPrintf(PETSC_COMM_WORLD,
          "P1BF3_FULL_PMAT_VALUE_REFRESH it=%" PetscInt_FMT " interval=%" PetscInt_FMT " pmat=full_explicit_Schur_snapshot exactOperator=factored currentRauSnapshot=1\n",
          it,pPreconditionerRefresh));
        if(factoredBenchmark && !factoredBenchmarkDone && it==factoredBenchmarkAt) {
          PetscCall(benchmarkFactoredSchur(PSchur.S,factoredSchur,D,D.volumes,PSchur.globalSchurNnz,factoredBenchmarkReps));
          factoredBenchmarkDone=PETSC_TRUE;
        }
        PetscCall(PetscTime(&ts1));
        schurUpdateSeconds += (double)(ts1-ts0);
        if(!customPressureLiveSchurParityDone && needExplicitSchur) {
          PetscCall(customPressureLiveSchurParity(PSchur.S,D,customPressureB,customMom,(double)customPressureBShadowTol));
          customPressureLiveSchurParityDone=PETSC_TRUE;
        }
        if(pPmatMode=="native_face" && it==1) PetscCall(PetscPrintf(PETSC_COMM_WORLD,
          "P1BF3_B2_EXACT_OPERATOR_GUARD status=PASS physicalSchur=custom_FP64_B_rAU_Bt compactPmatNeverUsedForMatMult=1 previousExactParityValidated=1\n"));

        PetscCall(PetscTime(&ts0));
        if(!pksp) {
          if(P.mode==ProblemMode::MMS) {
            PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,nullptr,&nsp));
            PetscCall(MatSetNullSpace(pressureOperator,nsp));
            PetscCall(MatSetTransposeNullSpace(pressureOperator,nsp));
            PetscCall(MatSetNearNullSpace(pressureOperator,nsp));
            if(pressurePmat!=PSchur.S) {
              PetscCall(MatSetNullSpace(pressurePmat,nsp));
              PetscCall(MatSetTransposeNullSpace(pressurePmat,nsp));
              PetscCall(MatSetNearNullSpace(pressurePmat,nsp));
            }
            PetscBool nsok=PETSC_FALSE;
            PetscCall(MatNullSpaceTest(nsp,pressureOperator,&nsok));
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
              "P1BF3_PRESSURE_NULLSPACE constantMode=%s gauge=volume_weighted_mean_zero operator=dynamic_B_rAU_Bt\n",
              nsok?"PASS":"FAIL"));
            if(!nsok) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"pressure constant nullspace test failed");
          } else {
            Vec one=nullptr,act=nullptr;PetscReal an=0;
            PetscCall(VecDuplicate(D.volumes,&one));PetscCall(VecDuplicate(D.volumes,&act));PetscCall(VecSet(one,1.0));
            PetscCall(MatMult(pressureOperator,one,act));PetscCall(VecNorm(act,NORM_2,&an));
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
              "P1BF3_PRESSURE_GAUGE mode=physical_outlet_traction pressureNullspace=OFF constantActionNorm=%.12e operator=dynamic_B_rAU_Bt\n",(double)an));
            PetscCall(VecDestroy(&one));PetscCall(VecDestroy(&act));
            if(!(an>0.0)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"pipe pressure operator did not anchor constant mode");
          }

          PetscCall(KSPCreate(PETSC_COMM_WORLD,&pksp));
          PetscCall(KSPSetOperators(pksp,pressureOperator,pressurePmat));
          PetscCall(KSPSetType(pksp,pressureSolveMode=="petsc_fgmres" ? KSPFGMRES : KSPCG));
          PetscCall(KSPGetPC(pksp,&ppc));
          if(gate9LivePcd) {
            if(!gate9LivePcdCtx) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate-9 live PCD context was not constructed");
            PetscCall(PCSetType(ppc,PCSHELL));
            PetscCall(PCShellSetName(ppc,"P1BF3_GATE9_LIVE_PCD"));
            PetscCall(PCShellSetContext(ppc,(void*)gate9LivePcdCtx));
            PetscCall(PCShellSetSetUp(ppc,gate9LivePcdSetUp));
            PetscCall(PCShellSetApply(ppc,gate9LivePcdApply));
            PetscCall(PCShellSetDestroy(ppc,gate9LivePcdDestroy));
            PetscCall(KSPSetPCSide(pksp,PC_RIGHT));
          } else PetscCall(PCSetType(ppc,PCJACOBI));
          PetscCall(KSPSetTolerances(pksp,1e-10,PETSC_CURRENT,PETSC_CURRENT,20000));
          PetscCall(KSPSetOptionsPrefix(pksp,"p_"));
          PetscCall(KSPSetFromOptions(pksp));
          MatInfo pPInfo;
          PetscCall(MatGetInfo(pressurePmat,MAT_GLOBAL_SUM,&pPInfo));
          const double expandedNnz=(double)PSchur.globalSchurNnz;
          PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "P1BF3_PRESSURE_PMAT mode=%s pressureOperator=%s expandedEquivalentNnz=%.0f pmatNnz=%.0f pmatAllocated=%.0f allocOverUsed=%.12f pmatToExpanded=%.6f pmatAvgNnzPerRow=%.6f\n",
            pPmatMode.c_str(),pOperatorMode.c_str(),expandedNnz,pPInfo.nz_used,pPInfo.nz_allocated,pPInfo.nz_used?pPInfo.nz_allocated/pPInfo.nz_used:0.0,expandedNnz>0.0?pPInfo.nz_used/expandedNnz:-1.0,pPInfo.nz_used/(double)M.tets.size()));
          if(pPmatMode=="full") PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "P1BF3_M3B_PMAT_ALLOCATION used=%.0f allocated=%.0f allocOverUsed=%.12f status=%s\n",
            pPInfo.nz_used,pPInfo.nz_allocated,pPInfo.nz_used?pPInfo.nz_allocated/pPInfo.nz_used:0.0,(pPInfo.nz_used>0.0 && std::abs(pPInfo.nz_allocated-pPInfo.nz_used)<0.5)?"PASS":"CHECK"));
          if(pPmatMode=="native_face" && !gate9eNgfv) {
            const double compactExpected=(double)M.tets.size()+2.0*(double)M.neighbour.size();
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
              "P1BF3_B1_COMPACT_TOPOLOGY status=%s usedNnz=%.0f expectedFaceGraphNnz=%.0f avgNnzPerRow=%.6f allocationRatio=%.12f\n",
              std::abs(pPInfo.nz_used-compactExpected)<0.5?"PASS":"FAIL",pPInfo.nz_used,compactExpected,pPInfo.nz_used/(double)M.tets.size(),pPInfo.nz_used?pPInfo.nz_allocated/pPInfo.nz_used:0.0));
            if(std::abs(pPInfo.nz_used-compactExpected)>=0.5) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M11 compact face topology NNZ mismatch");
          }
          PetscLogDouble tpc0,tpc1; PetscCall(PetscTime(&tpc0));
          PetscCall(KSPSetUp(pksp));
          PetscCall(PetscTime(&tpc1)); pressurePcRefreshSeconds += (double)(tpc1-tpc0);
          PetscCall(printSetupPhase("initial_pressure_pc_gamg",tpc1-tpc0,(PetscInt)M.tets.size()));
          pressurePcRefreshes++;
          if(gate9iAutoChebyshev) gate9iEstimatePending=PETSC_TRUE;
          if(pPreconditionerRefresh>1) PetscCall(KSPSetReusePreconditioner(pksp,PETSC_TRUE));
          PetscCall(KSPGetType(pksp,&pKspType));
          PetscCall(PCGetType(ppc,&pPcType));
          if(pressureSolveMode=="petsc_fgmres" && std::string(pKspType)!=std::string(KSPFGMRES))
            SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Gate 1 requires PETSc FGMRES; remove any conflicting -p_ksp_type option");
          if(gate9LivePcd && std::string(pPcType)!=std::string(PCSHELL)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Gate-9 requires live PC type shell; remove conflicting -p_pc_type");
          if(gate9LivePcd) { PCSide side; PetscCall(KSPGetPCSide(pksp,&side)); if(side!=PC_RIGHT) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Gate-9 requires right preconditioning"); }
          PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "P1BF3_PRESSURE_SOLVER ksp=%s pc=%s pmat=%s operator=%s pressureOperatorUpdate=%s preconditionerRefresh=%" PetscInt_FMT " refreshSemantics=0_PETSc_default_else_initial_then_every_N_SIMPLE fullPmatLagged=%d nativeCompactLagged=%d\n",
            pKspType,pPcType,pPmatMode.c_str(),pOperatorMode.c_str(),staticPhysicalOperator?"static_once":"every_SIMPLE_iteration_values_only",pPreconditionerRefresh,(int)lagFullPmat,(int)lagNativeCompactPmat));
          PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "P1BF3_GATE1_PRESSURE_MODE mode=%s exactOperator=custom_FP64_B_rAU_Bt pmat=%s pc=%s compareCustomPCG=%d solutionTol=%.3e trueResidualTol=%.3e\n",
            pressureSolveMode.c_str(),pPmatMode.c_str(),pPcType,(int)gate1ComparePcg,(double)gate1SolutionTol,(double)gate1TrueResidualTol));
          PetscReal pcgRtolSetup=0,pcgAtolSetup=0,pcgDtolSetup=0; PetscInt pcgMaxItsSetup=0;
          PetscCall(KSPGetTolerances(pksp,&pcgRtolSetup,&pcgAtolSetup,&pcgDtolSetup,&pcgMaxItsSetup));
          PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "P1BF3_PRESSURE_OUTER_CONFIG mode=%s pc=%s configuredKsp=%s rtol=%.3e atol=%.3e dtol=%.3e maxIts=%" PetscInt_FMT "\n",
            pressureSolveMode.c_str(),pPcType,pKspType,(double)pcgRtolSetup,(double)pcgAtolSetup,(double)pcgDtolSetup,pcgMaxItsSetup));
          if(gate9dGamgOnly) PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "P1BF3_GATE9D_KSP_CONTAINER_NOTE configuredKsp=%s configuredPc=%s purpose=GAMG_hierarchy_setup_only KSPSolve=NEVER_CALLED pressureAlgorithm=NO_OUTER_KRYLOV\n",pKspType,pPcType));
        } else if(!staticPhysicalOperator) {
          // Same current pressure operator every SIMPLE step.  Optionally keep a
          // stale PC/GAMG built from an older Schur matrix and refresh only every N
          // SIMPLE iterations.  The Krylov operator is NEVER frozen.
          PetscCall(KSPSetOperators(pksp,pressureOperator,pressurePmat));
          if(pPreconditionerRefresh>0) {
            const PetscBool refreshNow = (((it-1)%pPreconditionerRefresh)==0) ? PETSC_TRUE : PETSC_FALSE;
            if(refreshNow) {
              PetscCall(KSPSetReusePreconditioner(pksp,PETSC_FALSE));
              PetscLogDouble tpc0,tpc1; PetscCall(PetscTime(&tpc0));
              PetscCall(KSPSetUp(pksp));
              PetscCall(PetscTime(&tpc1)); pressurePcRefreshSeconds += (double)(tpc1-tpc0);
              pressurePcRefreshes++;
              if(gate9iAutoChebyshev) gate9iEstimatePending=PETSC_TRUE;
              PetscCall(KSPSetReusePreconditioner(pksp,PETSC_TRUE));
              PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                "P1BF3_PRESSURE_PC_REFRESH it=%" PetscInt_FMT " interval=%" PetscInt_FMT " refreshCount=%" PetscInt_FMT " setupSeconds=%.6f\n",
                it,pPreconditionerRefresh,pressurePcRefreshes,(double)(tpc1-tpc0)));
            } else {
              PetscCall(KSPSetReusePreconditioner(pksp,PETSC_TRUE));
              pressurePcReuses++;
            }
          } else {
            // Baseline behavior: allow PETSc to refresh the PC as the matrix state changes.
            PetscCall(KSPSetReusePreconditioner(pksp,PETSC_FALSE));
          }
        }
        PetscCall(PetscTime(&ts1));
        kspOperatorSeconds += (double)(ts1-ts0);

        operatorReady=true;
        PetscCall(PetscTime(&top1));
        operatorUpdateSeconds += (double)(top1-top0);
        if(it==1) {
          PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "P1BF3_INITIAL_OPERATOR_SETUP seconds=%.6f convection=%.6f supg=%.6f derived=%.6f schur=%.6f kspAndPc=%.6f cells=%zu\n",
            (double)(top1-top0),convectionUpdateSeconds,supgUpdateSeconds,derivedUpdateSeconds,schurUpdateSeconds,kspOperatorSeconds,M.tets.size()));
          if(resourceProfile) PetscCall(printResourceMark("after_initial_operator_gamg",(PetscInt)M.tets.size(),top1-top0,tTotal0));
          if(memoryAudit) {
            PetscCall(printMemoryAuditMat("after_initial_operator_gamg","pressure.Pmat_active",pressurePmat,(PetscInt)M.tets.size()));
            PetscCall(auditGAMGMemory(pksp,(PetscInt)M.tets.size()));
            PetscCall(printResourceMark("memory_audit_complete",(PetscInt)M.tets.size(),0.0,tTotal0));
          }
        }
      }

      if(pressureProfile && !pressureProfileDone && it==pressureProfileAt) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
          "P1BF3_PRESSURE_PROFILE_BEGIN it=%" PetscInt_FMT " ranks=%d fvCompactPressureNnz=%" PetscInt_FMT " semantics=frozen_current_pressure_operator_and_existing_GAMG_hierarchy\n",
          it,size,fvCompactPressureNnz));
        if(pPmatMode!="full") {
          double pmatMs=0.0; PetscCall(profileMatMult(PSchur.Pcompact,pressureProfileFineReps,&pmatMs));
          MatInfo pmi; PetscCall(MatGetInfo(PSchur.Pcompact,MAT_GLOBAL_SUM,&pmi));
          PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "P1BF3_PRESSURE_PROFILE_PMAT mode=%s rows=%" PetscInt_FMT " nnz=%.0f avgNnzPerRow=%.6f matMultMs=%.6f\n",
            pPmatMode.c_str(),(PetscInt)M.tets.size(),pmi.nz_used,pmi.nz_used/(double)M.tets.size(),1e3*pmatMs));
        }
        PetscCall(pressureAMGProfile(pksp,pressureOperator,pressurePmat,fvCompactPressureNnz,pressureProfileFineReps,pressureProfilePcReps,
          pressureProfileCgIts,pressureProfileCgReps,pressureProfileLevelMatReps,pressureProfileLevelSolveReps));
        pressureProfileDone=PETSC_TRUE;
      }

      PetscInt uits[3]={0,0,0};
      PetscReal ur[3]={0,0,0},uInitRel[3]={0,0,0};
      // OpenFOAM-style outer residualControl audit: record each momentum equation
      // initial residual BEFORE its inner solve.  These are distinct from ur[],
      // which are the final inner-solve residuals.
      // M6B: all momentum RHS algebra and velocity iterates remain native FP64.
      for(int d=0;d<3;++d) {
        std::vector<double>& mb=customMom.workB; mb=D.rhsOwnedFP64[(std::size_t)d];
        if(centralConvection) for(std::size_t i=0;i<mb.size();++i) mb[i]+=customMom.convRhs[(std::size_t)d][i];
        if(useSupg) for(std::size_t i=0;i<mb.size();++i) mb[i]+=customMom.supgRhs[(std::size_t)d][i];
        PetscCall(customPressureBtApply(customPressureB,d,pressureState,customPressureB.velocityWork));
        if(customPressureB.velocityWork.size()!=mb.size()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"M6B B^T/velocity ownership mismatch");
        for(std::size_t i=0;i<mb.size();++i) mb[i]+=customPressureB.velocityWork[i]+customMom.relaxDelta[i]*U[(std::size_t)d][i];
        double ubn=0.0,urn0=0.0; PetscCall(customMomentumNorm2(mb,ubn)); if(ubn==0.0) ubn=1.0;
        PetscCall(customMomentumResidualNorm(customMom,mb,U[(std::size_t)d],urn0));
        uInitRel[d]=(PetscReal)(urn0/ubn); finalUInitRel[d]=uInitRel[d];
        PetscLogDouble tms0,tms1; PetscCall(PetscTime(&tms0)); double relu=0.0;
        PetscCall(smoothSolveCustomMomentumNative(customMom,mb,U[(std::size_t)d],(double)uTol,(double)uAtol,(double)uRelDrop,uMax,uCheck,(double)uOmega,uLocalSweeps,&uits[d],&relu)); ur[d]=(PetscReal)relu;
        PetscCall(PetscTime(&tms1)); momentumSolveSeconds += (double)(tms1-tms0); sumU[d]+=uits[d];
        if(uRelDrop<=0.0 && ((double)ur[d]*ubn)>1.2*std::max((double)uAtol,(double)uTol*ubn)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_NOT_CONVERGED,"momentum processor-block SGS failed at SIMPLE it %" PetscInt_FMT " comp %d relres %.3e",it,d,(double)ur[d]);
      }

      pressureResidual=D.fixedDivOwnedFP64;
      for(int d=0;d<3;++d) {
        PetscCall(customPressureBApply(customPressureB,d,U[(std::size_t)d],customPressureB.pressureWork));
        if(pressureResidual.size()!=customPressureB.pressureWork.size()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"M6A continuity pressure size mismatch");
        for(std::size_t i=0;i<pressureResidual.size();++i) pressureResidual[i]+=customPressureB.pressureWork[i];
      }
      double rn=0.0; PetscCall(customPressureNorm2(pressureResidual,&rn));
      if(it==1) r0=rn;
      rel=rn/std::max(r0,1e-300);
      finalIt=it;
      finalPInitRel=(PetscReal)rel;
      const PetscBool allInitialResidualsMet =
        (uInitRel[0]<(PetscReal)simpleTol && uInitRel[1]<(PetscReal)simpleTol && uInitRel[2]<(PetscReal)simpleTol && rel<(double)simpleTol)
        ? PETSC_TRUE : PETSC_FALSE;
      if(it<=10 || it%10==0 || allInitialResidualsMet) PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "P1BF3_SIMPLE_INITIAL_RESIDUALS it=%" PetscInt_FMT " Ux=%.12e Uy=%.12e Uz=%.12e pEquivalentContinuity=%.12e gate=%.3e allMet=%d semantics=momentum_initial_before_inner_solve_plus_continuity_before_pressure_correction\n",
        it,(double)uInitRel[0],(double)uInitRel[1],(double)uInitRel[2],rel,(double)simpleTol,(int)allInitialResidualsMet));

      for(double& v:pressureResidual) v=-v;
      if(nsp) PetscCall(customPressureProjectConstant(pressureResidual));
      double prhsNorm=0.0; PetscCall(customPressureNorm2(pressureResidual,&prhsNorm));
      if(!std::isfinite(prhsNorm)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"pressure-correction RHS became NaN/Inf at SIMPLE it %" PetscInt_FMT,it);
      PetscReal pcgRtol=0,pcgAtol=0,pcgDtol=0; PetscInt pcgMaxIts=0;
      PetscCall(KSPGetTolerances(pksp,&pcgRtol,&pcgAtol,&pcgDtol,&pcgMaxIts));

      PetscInt petscRefIts=-1; std::vector<double> petscRefSolution;
      if(m5bPcgReference && !pressurePcgReferenceDone) {
        PetscCall(customVecWriteOwnedRange(pcIn,customPressureB.pressureHalo.start,customPressureB.pressureHalo.end,pressureResidual));
        PetscCall(VecSet(pcOut,0)); PetscCall(KSPSetInitialGuessNonzero(pksp,PETSC_FALSE));
        PetscCall(KSPSolve(pksp,pcIn,pcOut));
        KSPConvergedReason refReason; PetscCall(KSPGetConvergedReason(pksp,&refReason)); PetscCall(KSPGetIterationNumber(pksp,&petscRefIts));
        if(refReason<0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_NOT_CONVERGED,"M6A PETSc reference CG failed");
        PetscCall(customVecOwnedRange(pcOut,customPressureB.pressureHalo.start,customPressureB.pressureHalo.end,petscRefSolution));
      }

      CustomPressurePCGResult pcgRes;
      M10PressurePCGProfile m10This; m10This.enabled=m10PcgProfile;
      PetscInt pits=0;
      std::vector<double> pressureCorrection;
      PetscLogDouble tps0,tps1; PetscCall(PetscTime(&tps0));
      if(gate9iAutoChebyshev) {
        // Gate 9I production pressure path:
        //   * Kp/GAMG is the compact FE-face-energy hierarchy from Gate 9G.
        //   * On each PC refresh only, estimate lambda_max(M^{-1}S_exact)
        //     with a short deterministic power iteration.
        //   * Freeze [lambda_min,lambda_max] until the next PC refresh, with
        //       lambda_max = safety * lambda_hat,
        //       lambda_min = lambdaMinFraction * lambda_max.
        //   * Each SIMPLE pressure correction uses reduction-free Chebyshev
        //     blocks: initialBlock stages, then extendBlock stages as needed.
        //   * Stop on the TRUE exact-Schur residual:
        //       ||r|| <= max(atol, rtol*||r0||).
        //
        // The PETSc KSP object is only a container for the GAMG PC. KSPSolve is
        // never called in this branch.
        // Gate 9M: optional spectrum cadence independent of GAMG rebuild cadence.
        // A value of 1 re-estimates on every SIMPLE pressure solve; 0 preserves
        // the legacy Gate-9I behavior (estimate only when GAMG is rebuilt).
        if(gate9iSpectrumRefresh>0 && (((it-1)%gate9iSpectrumRefresh)==0)) gate9iEstimatePending=PETSC_TRUE;
        if(gate9iEstimatePending || !(gate9iLambdaMaxActive>0.0 && gate9iLambdaMinActive>0.0)) {
          PetscLogDouble tp0,tp1; PetscCall(PetscTime(&tp0));
          std::vector<double> powQ(pressureResidual.size(),0.0),powZ;
          const PetscInt gstart=customPressureB.pressureHalo.start;
          for(std::size_t q=0;q<powQ.size();++q) {
            const double gi=(double)(gstart+(PetscInt)q+1);
            // Deterministic broadband seed: reproducible across MPI counts and
            // non-orthogonal to ordinary smooth/high-frequency pressure modes.
            powQ[q]=std::sin(0.371*gi)+0.5*std::cos(0.113*(gi+2.0))+0.25*std::sin(0.017*gi*gi);
          }
          if(nsp) PetscCall(customPressureProjectConstant(powQ));
          double qn=0.0; PetscCall(customPressureNorm2(powQ,&qn));
          if(!(qn>0.0) || !std::isfinite(qn)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"Gate-9I power seed norm invalid");
          for(double& v:powQ) v/=qn;

          double lambdaObservedMax=0.0,lambdaWindowMax=0.0,lastNorm=0.0,lastRayleigh=0.0;
          const PetscInt windowStart=PetscMax((PetscInt)1,gate9iPowerIts-2);
          for(PetscInt pit=1;pit<=gate9iPowerIts;++pit) {
            PetscCall(customPressureSchurApply(customPressureB,customMom,powQ,customPressureB.pressureWork));
            PetscCall(customVecWriteOwnedRange(pcIn,customPressureB.pressureHalo.start,customPressureB.pressureHalo.end,customPressureB.pressureWork));
            PetscCall(VecSet(pcOut,0.0)); PetscCall(PCApply(ppc,pcIn,pcOut));
            PetscCall(customVecOwnedRange(pcOut,customPressureB.pressureHalo.start,customPressureB.pressureHalo.end,powZ));
            double zn=0.0,rq=0.0;
            PetscCall(customPressureNorm2(powZ,&zn));
            PetscCall(customPressureDot(powQ,powZ,&rq));
            if(!(zn>0.0) || !std::isfinite(zn) || !std::isfinite(rq))
              SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"Gate-9I power iteration produced invalid spectral estimate");
            lastNorm=zn; lastRayleigh=std::abs(rq);
            const double candidate=std::max(lastNorm,lastRayleigh);
            lambdaObservedMax=std::max(lambdaObservedMax,candidate);
            if(pit>=windowStart) lambdaWindowMax=std::max(lambdaWindowMax,candidate);
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
              "P1BF3_GATE9I_POWER it=%" PetscInt_FMT " refresh=%" PetscInt_FMT " powerIt=%" PetscInt_FMT " normEstimate=%.12e rayleighAbs=%.12e observedMax=%.12e\n",
              it,gate9iEstimateCount+1,pit,lastNorm,lastRayleigh,lambdaObservedMax));
            for(std::size_t q=0;q<powQ.size();++q) powQ[q]=powZ[q]/zn;
            if(nsp) PetscCall(customPressureProjectConstant(powQ));
            double qrenorm=0.0; PetscCall(customPressureNorm2(powQ,&qrenorm));
            if(!(qrenorm>0.0) || !std::isfinite(qrenorm)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"Gate-9I power renormalization failed");
            for(double& v:powQ) v/=qrenorm;
          }
          gate9iLambdaHat=(lambdaWindowMax>0.0)?lambdaWindowMax:std::max(lastNorm,lastRayleigh);
          gate9iLambdaMaxActive=(double)gate9iSafety*gate9iLambdaHat;
          gate9iLambdaMinActive=(double)gate9iLambdaMinFraction*gate9iLambdaMaxActive;
          if(!(gate9iLambdaMinActive>0.0 && gate9iLambdaMaxActive>gate9iLambdaMinActive) ||
             !std::isfinite(gate9iLambdaMinActive) || !std::isfinite(gate9iLambdaMaxActive))
            SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FP,"Gate-9I auto Chebyshev interval invalid");
          gate9iEstimatePending=PETSC_FALSE;
          gate9iEstimateCount++;
          gate9iTotalPowerIts+=gate9iPowerIts;
          PetscCall(PetscTime(&tp1)); gate9iPowerSeconds+=(double)(tp1-tp0);
          PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "P1BF3_GATE9I_SPECTRUM_REFRESH it=%" PetscInt_FMT " refresh=%" PetscInt_FMT " powerIts=%" PetscInt_FMT " lambdaHat=%.12e lambdaMin=%.12e lambdaMax=%.12e safety=%.6e lambdaMinFraction=%.6e observedMax=%.12e estimateSeconds=%.6e spectrumCadence=%" PetscInt_FMT " pcRefresh=%" PetscInt_FMT "\n",
            it,gate9iEstimateCount,gate9iPowerIts,gate9iLambdaHat,gate9iLambdaMinActive,gate9iLambdaMaxActive,
            (double)gate9iSafety,(double)gate9iLambdaMinFraction,lambdaObservedMax,(double)(tp1-tp0),gate9iSpectrumRefresh,pPreconditionerRefresh));
        }

        PetscLogDouble tc0,tc1; PetscCall(PetscTime(&tc0));
        pressureCorrection.assign(pressureResidual.size(),0.0);
        std::vector<double> chebResidual=pressureResidual,chebZ,chebDir(pressureResidual.size(),0.0);
        const double rhsNorm0=std::max(prhsNorm,1e-300);
        const double targetAbs=std::max((double)gate9iAtol,(double)gate9iRtol*rhsNorm0);
        const double lmin=gate9iLambdaMinActive,lmax=gate9iLambdaMaxActive;
        const double theta=0.5*(lmax+lmin),delta=0.5*(lmax-lmin),sigma=theta/delta;
        double rhoPrev=1.0/sigma;
        const PetscBool chebFixed=(gate9iFixedSteps>0)?PETSC_TRUE:PETSC_FALSE;
        const PetscInt chebLimit=chebFixed?gate9iFixedSteps:gate9iMaxSteps;
        PetscBool chebConverged=(prhsNorm<=targetAbs)?PETSC_TRUE:PETSC_FALSE,chebFinite=PETSC_TRUE;
        PetscBool chebAccepted=PETSC_FALSE;
        PetscInt chebSteps=0; double chebAbs=prhsNorm,chebTrueRel=(rhsNorm0>0.0)?prhsNorm/rhsNorm0:0.0;
        PetscInt nextCheck=chebFixed?chebLimit:gate9iInitialBlock;

        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
          "P1BF3_GATE9I_CHEB_BEGIN it=%" PetscInt_FMT " rhsNorm=%.12e targetAbs=%.12e rtol=%.3e atol=%.3e lambdaMin=%.12e lambdaMax=%.12e mode=%s fixedSteps=%" PetscInt_FMT " requireTarget=%d initialBlock=%" PetscInt_FMT " extendBlock=%" PetscInt_FMT " maxSteps=%" PetscInt_FMT " spectrumRefresh=%" PetscInt_FMT " outerKrylov=NONE KSPSolve=NEVER_CALLED\n",
          it,prhsNorm,targetAbs,(double)gate9iRtol,(double)gate9iAtol,lmin,lmax,chebFixed?"fixed":"adaptive",gate9iFixedSteps,(int)gate9iRequireTarget,gate9iInitialBlock,gate9iExtendBlock,gate9iMaxSteps,gate9iEstimateCount));

        for(PetscInt k=0;k<chebLimit && (chebFixed || !chebConverged);++k) {
          PetscCall(customVecWriteOwnedRange(pcIn,customPressureB.pressureHalo.start,customPressureB.pressureHalo.end,chebResidual));
          PetscCall(VecSet(pcOut,0.0)); PetscCall(PCApply(ppc,pcIn,pcOut));
          PetscCall(customVecOwnedRange(pcOut,customPressureB.pressureHalo.start,customPressureB.pressureHalo.end,chebZ));
          if(chebZ.size()!=pressureCorrection.size()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Gate-9I Chebyshev correction size mismatch");
          if(k==0) {
            const double alpha0=1.0/theta;
            for(std::size_t q=0;q<chebDir.size();++q) chebDir[q]=alpha0*chebZ[q];
          } else {
            const double rho=1.0/(2.0*sigma-rhoPrev);
            const double beta=rho*rhoPrev;
            const double alpha=2.0*rho/delta;
            for(std::size_t q=0;q<chebDir.size();++q) chebDir[q]=beta*chebDir[q]+alpha*chebZ[q];
            rhoPrev=rho;
          }
          for(std::size_t q=0;q<pressureCorrection.size();++q) pressureCorrection[q]+=chebDir[q];
          PetscCall(customPressureSchurApply(customPressureB,customMom,pressureCorrection,customPressureB.pressureWork));
          for(std::size_t q=0;q<chebResidual.size();++q) chebResidual[q]=pressureResidual[q]-customPressureB.pressureWork[q];
          if(nsp) PetscCall(customPressureProjectConstant(chebResidual));
          chebSteps=k+1;

          const PetscBool doCheck=(chebSteps==nextCheck || chebSteps==chebLimit)?PETSC_TRUE:PETSC_FALSE;
          if(doCheck) {
            PetscCall(customPressureNorm2(chebResidual,&chebAbs));
            chebTrueRel=chebAbs/rhsNorm0;
            if(!std::isfinite(chebAbs) || !std::isfinite(chebTrueRel)) {chebFinite=PETSC_FALSE;break;}
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
              "P1BF3_GATE9I_CHEB_CHECK it=%" PetscInt_FMT " step=%" PetscInt_FMT " trueAbs=%.12e trueRel=%.12e targetAbs=%.12e targetRel=%.3e oneSchurPerStage=1 oneGamgPerStage=1 reductionsThisBlock=1 lambdaMin=%.6e lambdaMax=%.6e\n",
              it,chebSteps,chebAbs,chebTrueRel,targetAbs,(double)gate9iRtol,lmin,lmax));
            if(chebAbs<=targetAbs) chebConverged=PETSC_TRUE;
            if(chebTrueRel>=(double)gate9dDivergenceFactor) break;
            if(!chebFixed) nextCheck=PetscMin(chebLimit,chebSteps+gate9iExtendBlock);
          }
        }

        // A final norm is needed if an adaptive nonstandard maxSteps ended
        // between scheduled checks. Fixed mode always checks at chebLimit.
        if(!chebFixed && !chebConverged && chebFinite && chebSteps>0 &&
           chebSteps!=gate9iInitialBlock &&
           ((chebSteps-gate9iInitialBlock)%gate9iExtendBlock)!=0) {
          PetscCall(customPressureNorm2(chebResidual,&chebAbs));
          chebTrueRel=chebAbs/rhsNorm0;
          if(!std::isfinite(chebAbs) || !std::isfinite(chebTrueRel)) chebFinite=PETSC_FALSE;
          if(chebFinite && chebAbs<=targetAbs) chebConverged=PETSC_TRUE;
        }
        chebAccepted = chebFinite && (chebConverged || !gate9iRequireTarget);
        PetscCall(PetscTime(&tc1));
        gate9iChebSeconds+=(double)(tc1-tc0);
        gate9iTotalChebSteps+=chebSteps;
        gate9iPressureSolves++;
        pits=chebSteps;
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
          "P1BF3_GATE9I_CHEB_SOLVE it=%" PetscInt_FMT " accepted=%d targetMet=%d mode=%s steps=%" PetscInt_FMT " trueAbs=%.12e trueRel=%.12e targetAbs=%.12e rtol=%.3e atol=%.3e lambdaHat=%.12e lambdaMin=%.12e lambdaMax=%.12e solveSeconds=%.6e outerKrylov=NONE KSPSolve=NEVER_CALLED Kp=FE_face_jump_energy\n",
          it,(int)chebAccepted,(int)chebConverged,chebFixed?"fixed":"adaptive",chebSteps,chebAbs,chebTrueRel,targetAbs,(double)gate9iRtol,(double)gate9iAtol,
          gate9iLambdaHat,lmin,lmax,(double)(tc1-tc0)));
        if(!chebAccepted) {
          solveFailed=PETSC_TRUE; pressureFailureReason=-911;
          PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "P1BF3_SOLVE_FAILURE stage=gate9i_auto_chebyshev it=%" PetscInt_FMT " mode=%s steps=%" PetscInt_FMT " trueAbs=%.3e trueRel=%.3e targetAbs=%.3e lambdaMin=%.3e lambdaMax=%.3e requireTarget=%d action=stop_SIMPLE_and_write_last_iterate_VTU\n",
            it,chebFixed?"fixed":"adaptive",chebSteps,chebAbs,chebTrueRel,targetAbs,lmin,lmax,(int)gate9iRequireTarget));
          break;
        }
      } else if(gate9hChebyshev) {
        // Gate 9H: preconditioned Chebyshev semi-iteration on the unchanged
        // exact SIMPLE Schur.  No outer Krylov and no inner Krylov.  Each
        // stage uses exactly one exact-Schur action and one GAMG PCApply on
        // the compact FE-face-energy Kp.  The recurrence is the classical
        // minimax Chebyshev semi-iteration for eigenvalues in [lambdaMin,lambdaMax].
        pressureCorrection.assign(pressureResidual.size(),0.0);
        std::vector<double> chebResidual=pressureResidual,chebZ,chebDir(pressureResidual.size(),0.0);
        const double rhsNorm0=std::max(prhsNorm,1e-300);
        const double lmin=(double)gate9hLambdaMin,lmax=(double)gate9hLambdaMax;
        const double theta=0.5*(lmax+lmin),delta=0.5*(lmax-lmin),sigma=theta/delta;
        double rhoPrev=1.0/sigma;
        PetscBool chebConverged=PETSC_FALSE,chebFinite=PETSC_TRUE;
        PetscInt chebSteps=0; double chebTrueRel=1.0,prevCheckedRel=1.0;
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
          "P1BF3_GATE9H_CHEB_CYCLE it=%" PetscInt_FMT " step=0 trueRel=1.000000000000e+00 lambdaMin=%.6e lambdaMax=%.6e outerKrylov=NONE KSPSolve=NEVER_CALLED\n",
          it,lmin,lmax));
        for(PetscInt k=0;k<gate9hMaxSteps;++k) {
          PetscCall(customVecWriteOwnedRange(pcIn,customPressureB.pressureHalo.start,customPressureB.pressureHalo.end,chebResidual));
          PetscCall(VecSet(pcOut,0.0)); PetscCall(PCApply(ppc,pcIn,pcOut));
          PetscCall(customVecOwnedRange(pcOut,customPressureB.pressureHalo.start,customPressureB.pressureHalo.end,chebZ));
          if(chebZ.size()!=pressureCorrection.size()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Gate-9H Chebyshev correction size mismatch");
          if(k==0) {
            const double alpha0=1.0/theta;
            for(std::size_t q=0;q<chebDir.size();++q) chebDir[q]=alpha0*chebZ[q];
          } else {
            const double rho=1.0/(2.0*sigma-rhoPrev);
            const double beta=rho*rhoPrev;
            const double alpha=2.0*rho/delta;
            for(std::size_t q=0;q<chebDir.size();++q) chebDir[q]=beta*chebDir[q]+alpha*chebZ[q];
            rhoPrev=rho;
          }
          for(std::size_t q=0;q<pressureCorrection.size();++q) pressureCorrection[q]+=chebDir[q];
          PetscCall(customPressureSchurApply(customPressureB,customMom,pressureCorrection,customPressureB.pressureWork));
          for(std::size_t q=0;q<chebResidual.size();++q) chebResidual[q]=pressureResidual[q]-customPressureB.pressureWork[q];
          chebSteps=k+1;
          const PetscBool doCheck = (chebSteps<=5 || chebSteps%gate9hCheckEvery==0 || chebSteps==gate9hMaxSteps) ? PETSC_TRUE : PETSC_FALSE;
          if(doCheck) {
            double rr=0.0; PetscCall(customPressureNorm2(chebResidual,&rr)); chebTrueRel=rr/rhsNorm0;
            if(!std::isfinite(chebTrueRel)) {chebFinite=PETSC_FALSE;break;}
            const double ctr=chebTrueRel/std::max(prevCheckedRel,1e-300);
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
              "P1BF3_GATE9H_CHEB_CYCLE it=%" PetscInt_FMT " step=%" PetscInt_FMT " trueRel=%.12e checkedContraction=%.12e lambdaMin=%.6e lambdaMax=%.6e oneSchurApply=1 oneGamgPCApply=1 reductionsSinceLastCheck=1 outerKrylov=NONE\n",
              it,chebSteps,chebTrueRel,ctr,lmin,lmax));
            prevCheckedRel=chebTrueRel;
            if(chebTrueRel<=gate9OuterTrueResidualTol) {chebConverged=PETSC_TRUE;break;}
            if(chebTrueRel>=gate9dDivergenceFactor) break;
          }
        }
        // Ensure the reported final true residual is current if the final step
        // was not one of the periodic diagnostic checks.
        if(chebSteps%gate9hCheckEvery!=0 && chebSteps>5 && !chebConverged && chebFinite) {
          double rr=0.0; PetscCall(customPressureNorm2(chebResidual,&rr)); chebTrueRel=rr/rhsNorm0;
          if(!std::isfinite(chebTrueRel)) chebFinite=PETSC_FALSE;
          if(chebTrueRel<=gate9OuterTrueResidualTol) chebConverged=PETSC_TRUE;
        }
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
          "P1BF3_GATE9H_CHEB_SOLVE it=%" PetscInt_FMT " converged=%d steps=%" PetscInt_FMT " trueRel=%.12e trueResidualTol=%.3e lambdaMin=%.6e lambdaMax=%.6e outerKrylov=NONE KSPSolve=NEVER_CALLED oneGamgPerStep=1 oneSchurPerStep=1 Kp=FE_face_jump_energy\n",
          it,(int)chebConverged,chebSteps,chebTrueRel,(double)gate9OuterTrueResidualTol,lmin,lmax));
        pits=chebSteps;
        if(!chebFinite || !chebConverged) {
          solveFailed=PETSC_TRUE; pressureFailureReason=-910;
          PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "P1BF3_SOLVE_FAILURE stage=gate9h_chebyshev it=%" PetscInt_FMT " steps=%" PetscInt_FMT " trueRel=%.3e lambdaMin=%.3e lambdaMax=%.3e action=stop_SIMPLE_and_write_last_iterate_VTU\n",
            it,chebSteps,chebTrueRel,lmin,lmax));
          break;
        }
      } else if(gate9gRichardson) {
        // Gate 9G stationary pressure solve: no outer Krylov and no inner
        // Krylov.  Each correction is exactly one PCApply from the GAMG
        // hierarchy built on the FE-energy face Kp.
        pressureCorrection.assign(pressureResidual.size(),0.0);
        std::vector<double> statResidual=pressureResidual,statCorrection;
        const double rhsNorm0=std::max(prhsNorm,1e-300);
        PetscBool statConverged=PETSC_FALSE,statFinite=PETSC_TRUE;
        PetscInt statCycles=0;
        double statTrueRel=1.0;
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
          "P1BF3_GATE9G_RICHARDSON_CYCLE it=%" PetscInt_FMT " cycle=0 trueRel=1.000000000000e+00 contraction=1.000000000000e+00 omega=%.6e outerKrylov=NONE KSPSolve=NEVER_CALLED\n",it,(double)gate9dOmega));
        double prevRel=1.0;
        for(PetscInt cyc=1;cyc<=gate9dMaxCycles;++cyc) {
          PetscCall(customVecWriteOwnedRange(pcIn,customPressureB.pressureHalo.start,customPressureB.pressureHalo.end,statResidual));
          PetscCall(VecSet(pcOut,0.0)); PetscCall(PCApply(ppc,pcIn,pcOut));
          PetscCall(customVecOwnedRange(pcOut,customPressureB.pressureHalo.start,customPressureB.pressureHalo.end,statCorrection));
          if(statCorrection.size()!=pressureCorrection.size()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Gate-9G Richardson correction size mismatch");
          for(std::size_t q=0;q<pressureCorrection.size();++q) pressureCorrection[q]+=(double)gate9dOmega*statCorrection[q];
          PetscCall(customPressureSchurApply(customPressureB,customMom,pressureCorrection,customPressureB.pressureWork));
          for(std::size_t q=0;q<statResidual.size();++q) statResidual[q]=pressureResidual[q]-customPressureB.pressureWork[q];
          double rr=0.0; PetscCall(customPressureNorm2(statResidual,&rr)); statTrueRel=rr/rhsNorm0; statCycles=cyc;
          if(!std::isfinite(statTrueRel)) {statFinite=PETSC_FALSE;break;}
          const double ctr=statTrueRel/std::max(prevRel,1e-300);
          if(cyc<=20 || cyc%10==0 || statTrueRel<=gate9OuterTrueResidualTol) PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "P1BF3_GATE9G_RICHARDSON_CYCLE it=%" PetscInt_FMT " cycle=%" PetscInt_FMT " trueRel=%.12e contraction=%.12e omega=%.6e outerKrylov=NONE\n",
            it,cyc,statTrueRel,ctr,(double)gate9dOmega));
          prevRel=statTrueRel;
          if(statTrueRel<=gate9OuterTrueResidualTol) {statConverged=PETSC_TRUE;break;}
          if(statTrueRel>=gate9dDivergenceFactor) break;
        }
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
          "P1BF3_GATE9G_RICHARDSON_SOLVE it=%" PetscInt_FMT " converged=%d cycles=%" PetscInt_FMT " trueRel=%.12e trueResidualTol=%.3e omega=%.6e outerKrylov=NONE KSPSolve=NEVER_CALLED Kp=FE_face_jump_energy\n",
          it,(int)statConverged,statCycles,statTrueRel,(double)gate9OuterTrueResidualTol,(double)gate9dOmega));
        pits=statCycles;
        if(!statFinite || !statConverged) {
          solveFailed=PETSC_TRUE; pressureFailureReason=-909;
          PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "P1BF3_SOLVE_FAILURE stage=gate9g_richardson it=%" PetscInt_FMT " cycles=%" PetscInt_FMT " trueRel=%.3e action=stop_SIMPLE_and_write_last_iterate_VTU\n",it,statCycles,statTrueRel));
          break;
        }
      } else if(pressureSolveMode=="custom_pcg") {
        PetscCall(customPressurePCG(customPressureB,customMom,ppc,pcIn,pcOut,pressureResidual,
          (double)pcgRtol,(double)pcgAtol,(double)pcgDtol,pcgMaxIts,nsp?PETSC_TRUE:PETSC_FALSE,pressurePcgW,pcgRes,m10PcgProfile?&m10This:nullptr));
        if(!pcgRes.converged) {
          solveFailed=PETSC_TRUE; pressureFailureReason=-901;
          PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "P1BF3_SOLVE_FAILURE stage=custom_pressure_pcg it=%" PetscInt_FMT " its=%" PetscInt_FMT " relPrec=%.3e action=stop_SIMPLE_and_write_last_iterate_VTU\n",
            it,pcgRes.its,pcgRes.finalPreconditionedRel));
          break;
        }
        pits=pcgRes.its;
        pressureCorrection=pressurePcgW.x;
        if(gate9eNgfv) PetscCall(PetscPrintf(PETSC_COMM_WORLD,
          "P1BF3_GATE9E_PCG_SOLVE it=%" PetscInt_FMT " pcgIts=%" PetscInt_FMT " converged=%d finalPreconditionedRel=%.3e pc=ONE_GAMG_PCApply_on_nodal_Kp exactSchur=custom_FP64_B_rAU_Bt\n",
          it,pcgRes.its,(int)pcgRes.converged,pcgRes.finalPreconditionedRel));
        if(gate9gFeFace) PetscCall(PetscPrintf(PETSC_COMM_WORLD,
          "P1BF3_GATE9G_PCG_SOLVE it=%" PetscInt_FMT " pcgIts=%" PetscInt_FMT " converged=%d finalPreconditionedRel=%.3e pc=ONE_GAMG_PCApply_on_FE_face_energy_Kp exactSchur=custom_FP64_B_rAU_Bt\n",
          it,pcgRes.its,(int)pcgRes.converged,pcgRes.finalPreconditionedRel));
      } else {
        // Gate 1 live path: PETSc FGMRES sees the exact factored Schur as A and
        // exactly the same existing pressure Pmat/GAMG as the old custom PCG.
        PetscCall(customVecWriteOwnedRange(pcIn,customPressureB.pressureHalo.start,customPressureB.pressureHalo.end,pressureResidual));
        if(gate7CpProbe) PetscCall(gate7ProbeActualPressureRhs(gate4Kp,gate7Cp,D.volumes,(PetscReal)P.nu,pcIn,it));
        if(gate8EswBcProbe) PetscCall(gate8ProbeActualPressureRhs(gate7Cp,gate8EswBc,D.volumes,pcIn,it));
        if(gate6DiffusionPcdProbe) {
          if(gate6DiffusionPcdCtx->setupCount==0) PetscCall(PCSetUp(gate6DiffusionPcdShell));
          PetscCall(VecSet(pcOut,0.0));
          PetscCall(PCApply(gate6DiffusionPcdShell,pcIn,pcOut));
          PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "P1BF3_GATE6_DIFFUSION_PCD_APPLY it=%" PetscInt_FMT " applyCount=%" PetscInt_FMT " kpIts=%" PetscInt_FMT " kpReason=%d chainRel=%.3e maxChainRel=%.3e kpTrueRel=%.3e maxKpTrueRel=%.3e rhsNorm=%.3e mpInvRhsNorm=%.3e fpNorm=%.3e outNorm=%.3e liveSolveTouched=0\n",
            it,gate6DiffusionPcdCtx->applyCount,gate6DiffusionPcdCtx->lastKpIts,(int)gate6DiffusionPcdCtx->lastKpReason,
            (double)gate6DiffusionPcdCtx->lastChainRel,(double)gate6DiffusionPcdCtx->maxChainRel,
            (double)gate6DiffusionPcdCtx->lastKpTrueRel,(double)gate6DiffusionPcdCtx->maxKpTrueRel,
            (double)gate6DiffusionPcdCtx->lastRhsNorm,(double)gate6DiffusionPcdCtx->lastMassNorm,
            (double)gate6DiffusionPcdCtx->lastFpNorm,(double)gate6DiffusionPcdCtx->lastOutNorm));
        }
        if(gate3MpProbe) {
          if(gate3MpCtx->setupCount==0) PetscCall(PCSetUp(gate3MpShell));
          PetscCall(VecSet(pcOut,0.0));
          PetscCall(PCApply(gate3MpShell,pcIn,pcOut));
          PetscReal mpOutNorm=0.0,rhsNormShadow=0.0;
          PetscCall(VecNorm(pcOut,NORM_2,&mpOutNorm));
          PetscCall(VecNorm(pcIn,NORM_2,&rhsNormShadow));
          PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "P1BF3_GATE3_MP_APPLY it=%" PetscInt_FMT " applyCount=%" PetscInt_FMT " algebraRel=%.3e maxAlgebraRel=%.3e rhsNorm=%.3e mpInvRhsNorm=%.3e liveSolveTouched=0\n",
            it,gate3MpCtx->applyCount,(double)gate3MpCtx->lastAlgebraRel,(double)gate3MpCtx->maxAlgebraRel,(double)rhsNormShadow,(double)mpOutNorm));
        }
        const PetscInt gate9ApplyBefore=(gate9LivePcd && gate9LivePcdCtx)?gate9LivePcdCtx->applyCount:0;
        const PetscReal gate9TimeBefore=(gate9LivePcd && gate9LivePcdCtx)?gate9LivePcdCtx->totalApplySeconds:0.0;
        double trueRel=PETSC_MAX_REAL;
        if(gate9dGamgOnly) {
          // Gate 9D: NO outer Krylov.  Stationary residual correction on the
          // unchanged exact SIMPLE Schur, using exactly one geometric-Kp GAMG
          // PCApply per cycle:
          //   r_m = b - S_exact x_m
          //   x_{m+1} = x_m + omega * GAMG_Kp(r_m)
          // The PETSc KSP object exists only to own/setup the PC hierarchy;
          // KSPSolve is deliberately never called in this branch.
          pressureCorrection.assign(pressureResidual.size(),0.0);
          std::vector<double> statResidual(pressureResidual.size(),0.0),statCorrection;
          PetscBool statConverged=PETSC_FALSE,statFinite=PETSC_TRUE;
          PetscInt cyclesDone=0;
          double prevRel=1.0;
          for(PetscInt cyc=0;cyc<=gate9dMaxCycles;++cyc) {
            PetscCall(customPressureSchurApply(customPressureB,customMom,pressureCorrection,customPressureB.pressureWork));
            for(std::size_t i=0;i<statResidual.size();++i) statResidual[i]=pressureResidual[i]-customPressureB.pressureWork[i];
            if(nsp) PetscCall(customPressureProjectConstant(statResidual));
            double rr=0.0; PetscCall(customPressureNorm2(statResidual,&rr));
            trueRel=rr/std::max(prhsNorm,1e-300);
            const double contraction=(cyc==0)?1.0:trueRel/std::max(prevRel,1e-300);
            if(!std::isfinite(trueRel) || !std::isfinite(contraction)) statFinite=PETSC_FALSE;
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
              "P1BF3_GATE9D_CYCLE simpleIt=%" PetscInt_FMT " cycle=%" PetscInt_FMT " trueRel=%.12e contraction=%.12e omega=%.6e exactResidual=1 outerKrylov=NONE\n",
              it,cyc,trueRel,contraction,(double)gate9dOmega));
            if(statFinite && trueRel<=(double)gate9OuterTrueResidualTol) { statConverged=PETSC_TRUE; cyclesDone=cyc; break; }
            if(!statFinite || trueRel>(double)gate9dDivergenceFactor) { cyclesDone=cyc; break; }
            if(cyc==gate9dMaxCycles) { cyclesDone=cyc; break; }
            prevRel=trueRel;
            PetscCall(customVecWriteOwnedRange(pcIn,customPressureB.pressureHalo.start,customPressureB.pressureHalo.end,statResidual));
            PetscCall(VecSet(pcOut,0.0));
            PetscCall(PCApply(ppc,pcIn,pcOut));
            PetscCall(customVecOwnedRange(pcOut,customPressureB.pressureHalo.start,customPressureB.pressureHalo.end,statCorrection));
            if(statCorrection.size()!=pressureCorrection.size()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Gate-9D correction ownership mismatch");
            for(std::size_t i=0;i<pressureCorrection.size();++i) pressureCorrection[i]+=(double)gate9dOmega*statCorrection[i];
            cyclesDone=cyc+1;
          }
          pits=cyclesDone;
          if(gate9LivePcdCtx) {
            gate9LivePcdCtx->lastOuterTrueRel=(PetscReal)trueRel;
            gate9LivePcdCtx->maxOuterTrueRel=PetscMax(gate9LivePcdCtx->maxOuterTrueRel,(PetscReal)trueRel);
            if(statConverged) gate9LivePcdCtx->outerSolveCount++;
            if(!statFinite) gate9LivePcdCtx->allFinite=PETSC_FALSE;
          }
          const PetscInt appliesThis=(gate9LivePcdCtx?gate9LivePcdCtx->applyCount:0)-gate9ApplyBefore;
          const PetscReal gamgSecondsThis=(gate9LivePcdCtx?gate9LivePcdCtx->totalApplySeconds:0.0)-gate9TimeBefore;
          PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "P1BF3_GATE9D_SOLVE it=%" PetscInt_FMT " converged=%d cycles=%" PetscInt_FMT " trueRel=%.12e trueResidualTol=%.3e appliesThis=%" PetscInt_FMT " gamgApplySeconds=%.6e omega=%.6e outerKrylov=NONE KSPSolve=NEVER_CALLED exactSchur=UNCHANGED\n",
            it,(int)statConverged,cyclesDone,trueRel,(double)gate9OuterTrueResidualTol,appliesThis,(double)gamgSecondsThis,(double)gate9dOmega));
          if(!statConverged) {
            solveFailed=PETSC_TRUE; pressureFailureReason=-904;
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
              "P1BF3_SOLVE_FAILURE stage=gate9d_gamg_only it=%" PetscInt_FMT " cycles=%" PetscInt_FMT " trueRel=%.3e omega=%.6e action=stop_SIMPLE_and_write_last_iterate_VTU\n",
              it,cyclesDone,trueRel,(double)gate9dOmega));
            break;
          }
        } else {
          PetscCall(VecSet(pcOut,0.0));
          PetscCall(KSPSetInitialGuessNonzero(pksp,PETSC_FALSE));
          PetscCall(KSPSolve(pksp,pcIn,pcOut));
          KSPConvergedReason fReason; PetscReal fReported=0.0;
          PetscCall(KSPGetConvergedReason(pksp,&fReason));
          PetscCall(KSPGetIterationNumber(pksp,&pits));
          PetscCall(KSPGetResidualNorm(pksp,&fReported));
          if(fReason<0) {
            solveFailed=PETSC_TRUE; pressureFailureReason=(PetscInt)fReason;
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
              "P1BF3_SOLVE_FAILURE stage=gate1_petsc_fgmres it=%" PetscInt_FMT " its=%" PetscInt_FMT " reason=%d reportedResidual=%.3e action=stop_SIMPLE_and_write_last_iterate_VTU\n",
              it,pits,(int)fReason,(double)fReported));
            break;
          }
          PetscCall(customVecOwnedRange(pcOut,customPressureB.pressureHalo.start,customPressureB.pressureHalo.end,pressureCorrection));

          // Always report the TRUE residual of the unchanged custom exact Schur.
          PetscCall(customPressureSchurApply(customPressureB,customMom,pressureCorrection,customPressureB.pressureWork));
          std::vector<double> trueR(pressureResidual.size(),0.0);
          for(std::size_t i=0;i<trueR.size();++i) trueR[i]=pressureResidual[i]-customPressureB.pressureWork[i];
          if(nsp) PetscCall(customPressureProjectConstant(trueR));
          double trueRn=0.0; PetscCall(customPressureNorm2(trueR,&trueRn));
          trueRel=trueRn/std::max(prhsNorm,1e-300);
          PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "P1BF3_GATE1_FGMRES it=%" PetscInt_FMT " its=%" PetscInt_FMT " reason=%d reportedResidual=%.3e trueRel=%.3e rhsNorm=%.3e\n",
            it,pits,(int)fReason,(double)fReported,trueRel,prhsNorm));
          if(gate9LivePcd && gate9LivePcdCtx) {
            gate9LivePcdCtx->lastOuterTrueRel=(PetscReal)trueRel; gate9LivePcdCtx->maxOuterTrueRel=PetscMax(gate9LivePcdCtx->maxOuterTrueRel,(PetscReal)trueRel); gate9LivePcdCtx->outerSolveCount++;
            const PetscInt appliesThis=gate9LivePcdCtx->applyCount-gate9ApplyBefore; const PetscReal pcdSecondsThis=gate9LivePcdCtx->totalApplySeconds-gate9TimeBefore;
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
              "P1BF3_GATE9C_OUTER it=%" PetscInt_FMT " fgmresIts=%" PetscInt_FMT " reason=%d trueRel=%.3e appliesThis=%" PetscInt_FMT " gamgApplySeconds=%.6e maxOneCycleKpRel=%.3e livePC=ONE_direct_GAMG_PCApply_geometric_Kp innerKrylov=NONE exactSchur=UNCHANGED\n",
              it,pits,(int)fReason,trueRel,appliesThis,(double)pcdSecondsThis,(double)gate9LivePcdCtx->maxKpCycleRel));
            if(trueRel>(double)gate9OuterTrueResidualTol) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_NOT_CONVERGED,"Gate-9C direct Kp-GAMG outer FGMRES true residual exceeded gate tolerance");
          }
        }

        // First pressure solve only: solve the identical RHS with the untouched
        // custom PCG and compare solutions.  This is a shadow check, not part of
        // the pressure update.
        if(!gate9dGamgOnly && gate1ComparePcg && !gate1FgmresParityDone) {
          CustomPressurePCGResult shadow; CustomPressurePCGWorkspace shadowW;
          PetscCall(customPressurePCG(customPressureB,customMom,ppc,pcIn,pcOut,pressureResidual,
            (double)pcgRtol,(double)pcgAtol,(double)pcgDtol,pcgMaxIts,nsp?PETSC_TRUE:PETSC_FALSE,shadowW,shadow,nullptr));
          if(!shadow.converged) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_NOT_CONVERGED,"Gate-1 shadow custom PCG did not converge");
          if(shadowW.x.size()!=pressureCorrection.size()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Gate-1 FGMRES/PCG solution size mismatch");
          std::vector<double> diff(pressureCorrection.size(),0.0);
          for(std::size_t i=0;i<diff.size();++i) diff[i]=pressureCorrection[i]-shadowW.x[i];
          double dn=0.0,pn=0.0; PetscCall(customPressureNorm2(diff,&dn)); PetscCall(customPressureNorm2(shadowW.x,&pn));
          const double solRel=dn/std::max(pn,1e-300);
          const bool ok=(solRel<=(double)gate1SolutionTol && trueRel<=(double)gate1TrueResidualTol);
          PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "P1BF3_GATE1_PARITY it=%" PetscInt_FMT " fgmresIts=%" PetscInt_FMT " customPcgIts=%" PetscInt_FMT " solutionRel=%.3e trueRel=%.3e solutionTol=%.3e trueResidualTol=%.3e status=%s\n",
            it,pits,shadow.its,solRel,trueRel,(double)gate1SolutionTol,(double)gate1TrueResidualTol,ok?"PASS":"FAIL"));
          if(!ok) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate-1 FGMRES/custom-PCG parity failed");
          gate1FgmresParityDone=PETSC_TRUE;
        }
      }
      PetscCall(PetscTime(&tps1)); pressureSolveSeconds += (double)(tps1-tps0);
      if(pressureSolveMode=="custom_pcg" && m10PcgProfile) { m10This.solves=1; m10This.totalPcg=(double)(tps1-tps0); m10ProfileAdd(m10PcgTotal,m10This); if(it>1) m10ProfileAdd(m10PcgWarm,m10This); }
      if(pressureSolveMode=="custom_pcg" && m5bPcgReference && !pressurePcgReferenceDone) {
        if(petscRefSolution.size()!=pressurePcgW.x.size()) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"M6A reference solution size mismatch");
        std::vector<double> diff(pressurePcgW.x.size()); for(std::size_t i=0;i<diff.size();++i) diff[i]=pressurePcgW.x[i]-petscRefSolution[i];
        double dn=0.0,rnRef=0.0; PetscCall(customPressureNorm2(diff,&dn)); PetscCall(customPressureNorm2(petscRefSolution,&rnRef));
        const double solRel=dn/std::max(rnRef,1e-300); const PetscInt itDiff=(PetscInt)std::llabs((long long)pits-(long long)petscRefIts);
        const bool ok=(solRel<=(double)customPressurePcgReferenceTol && itDiff==0);
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
          "P1BF3_M6A_PRESSURE_STATE_REFERENCE it=%" PetscInt_FMT " customIts=%" PetscInt_FMT " petscIts=%" PetscInt_FMT " iterationDiff=%" PetscInt_FMT " solutionRel=%.3e customFinalPrecRel=%.3e tol=%.3e status=%s\n",
          it,pits,petscRefIts,itDiff,solRel,pcgRes.finalPreconditionedRel,(double)customPressurePcgReferenceTol,ok?"PASS":"CHECK"));
        if(!ok) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"M6A native pressure state reference parity failed");
        pressurePcgReferenceDone=PETSC_TRUE;
      }
      sumP+=pits; pSolves++;

      for(std::size_t i=0;i<pressureState.size();++i) pressureState[i]+=(double)ap*pressureCorrection[i];
      if(nsp) PetscCall(customPressureVolumeMeanShift(pressureState,D.volumesOwnedFP64,volsum));

      if(it<=10 || it%10==0) PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "P1BF3_SIMPLE variant=%s it=%" PetscInt_FMT " relCont=%.12e uInitRel=[%.2e,%.2e,%.2e] uIts=[%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT "] uRel=[%.2e,%.2e,%.2e] aPhysDiag=[%.6e,%.6e] aRelDiag=[%.6e,%.6e] rauMetric=[%.6e,%.6e] tau=[%.3e,%.3e,%.3e] pCG=%" PetscInt_FMT " allInitialMet=%d\n",
        simpleVariant.c_str(),it,(double)rel,(double)uInitRel[0],(double)uInitRel[1],(double)uInitRel[2],uits[0],uits[1],uits[2],(double)ur[0],(double)ur[1],(double)ur[2],(double)minPhysDiag,(double)maxPhysDiag,(double)minDiag,(double)maxDiag,(double)minRauMetric,(double)maxRauMetric,(double)supgStats.tauMin,(double)supgStats.tauMean,(double)supgStats.tauMax,pits,(int)allInitialResidualsMet));
      if(allInitialResidualsMet) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
          "P1BF3_SIMPLE_ALL_INITIAL_RESIDUALS_PASS it=%" PetscInt_FMT " Ux=%.12e Uy=%.12e Uz=%.12e pEquivalentContinuity=%.12e gate=%.3e pressureSolveCompletedThisIteration=1\n",
          it,(double)uInitRel[0],(double)uInitRel[1],(double)uInitRel[2],rel,(double)simpleTol));
        converged=PETSC_TRUE;
        break;
      }
    }
    PetscCall(PetscTime(&tSolve1));
    if(resourceProfile) PetscCall(printResourceMark("solve_end",(PetscInt)M.tets.size(),tSolve1-tSolve0,tTotal0));

    // Rebuild the physical nonlinear operator with the final velocity and report
    // the true steady Navier-Stokes momentum residual (no equation UR). M2B
    // evaluates this with the direct custom FP64 physical matrix action.
    PetscCall(customMomentumGatherVelocityNative(customMom,U));
    PetscCall(customMomentumResetPhysical(customMom));
    if(centralConvection) PetscCall(assembleCentralConvectionCustom(D,DynPlan,customMom));
    if(useSupg) PetscCall(assembleSupgCustom(D,DynPlan,customMom,(double)supgTauScale,(double)supgMagic,"implicit",supgStats));
    PetscReal momRelMax=0;
    for(int d=0;d<3;++d) {
      PetscCall(customMomentumMatVec(customMom,customMom.aRel,U[(std::size_t)d],customMom.workY));
      PetscCall(customPressureBtApply(customPressureB,d,pressureState,customPressureB.velocityWork));
      double lr2=0.0,lb2=0.0;
      for(std::size_t i=0;i<customMom.workY.size();++i){const double physRhs=D.rhsOwnedFP64[(std::size_t)d][i]+(centralConvection?customMom.convRhs[(std::size_t)d][i]:0.0)+(useSupg?customMom.supgRhs[(std::size_t)d][i]:0.0)+customPressureB.velocityWork[i];const double rr=customMom.workY[i]-physRhs;lr2+=rr*rr;const double bb=D.rhsOwnedFP64[(std::size_t)d][i]+(centralConvection?customMom.convRhs[(std::size_t)d][i]:0.0)+(useSupg?customMom.supgRhs[(std::size_t)d][i]:0.0);lb2+=bb*bb;}
      double gr2=0.0,gb2=0.0;PetscCallMPI(MPI_Allreduce(&lr2,&gr2,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));PetscCallMPI(MPI_Allreduce(&lb2,&gb2,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD));const double rr=std::sqrt(gr2),bb=std::sqrt(gb2);momRelMax=PetscMax(momRelMax,rr/(bb>0?bb:1.0));
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_NS_FINAL variant=%s momRelMax=%.12e contRelInitial=%.12e convection=%s Re=%.12g supg=%s tauScale=%.6g supgMagic=%.6g tau=[%.3e,%.3e,%.3e] pressureStrongGrad=P0_zero supgTauLinearization=lagged supgForm=%s supgKernel=%s supgQuad=%" PetscInt_FMT "\n",
      simpleVariant.c_str(),(double)momRelMax,(double)rel,centralConvection?"central":"none",P.re,useSupg?"ON":"OFF",(double)supgTauScale,(double)supgMagic,(double)supgStats.tauMin,(double)supgStats.tauMean,(double)supgStats.tauMax,supgForm.c_str(),supgKernel.c_str(),supgQuadPoints));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_WORK outerIts=%" PetscInt_FMT " avgUIts=[%.3f,%.3f,%.3f] avgPCG=%.3f pSolves=%" PetscInt_FMT " operatorUpdateSeconds=%.6f\n",
      finalIt,finalIt?double(sumU[0])/finalIt:0.0,finalIt?double(sumU[1])/finalIt:0.0,finalIt?double(sumU[2])/finalIt:0.0,
      pSolves?double(sumP)/pSolves:0.0,pSolves,operatorUpdateSeconds));
    if(gate9iAutoChebyshev) PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_GATE9I_SUMMARY pressureSolves=%" PetscInt_FMT " totalChebSteps=%" PetscInt_FMT " avgChebSteps=%.6f spectrumEstimates=%" PetscInt_FMT " totalPowerIts=%" PetscInt_FMT " finalLambdaHat=%.12e finalLambdaMin=%.12e finalLambdaMax=%.12e powerSeconds=%.6f chebSeconds=%.6f rtol=%.3e atol=%.3e pcRefresh=%" PetscInt_FMT " spectrumRefresh=%" PetscInt_FMT " fixedSteps=%" PetscInt_FMT " requireTarget=%d\n",
      gate9iPressureSolves,gate9iTotalChebSteps,gate9iPressureSolves?double(gate9iTotalChebSteps)/gate9iPressureSolves:0.0,
      gate9iEstimateCount,gate9iTotalPowerIts,gate9iLambdaHat,gate9iLambdaMinActive,gate9iLambdaMaxActive,
      gate9iPowerSeconds,gate9iChebSeconds,(double)gate9iRtol,(double)gate9iAtol,pPreconditionerRefresh,gate9iSpectrumRefresh,gate9iFixedSteps,(int)gate9iRequireTarget));
    const double simpleWall=(double)(tSolve1-tSolve0);
    const double measuredCore=operatorUpdateSeconds+momentumSolveSeconds+pressureSolveSeconds;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_TIMING_DETAIL convectionUpdateSeconds=%.6f supgUpdateSeconds=%.6f derivedUpdateSeconds=%.6f customMomentumFieldExchangeResetSeconds=%.6f schurUpdateSeconds=%.6f kspOperatorSeconds=%.6f operatorUpdateSeconds=%.6f momentumSolveSeconds=%.6f pressureSolveSeconds=%.6f pressurePcRefreshSeconds=%.6f pressurePcRefreshes=%" PetscInt_FMT " pressurePcReuses=%" PetscInt_FMT " simpleOtherSeconds=%.6f staticPhysicalOperator=%d supgForm=%s supgKernel=%s supgQuad=%" PetscInt_FMT "\n",
      convectionUpdateSeconds,supgUpdateSeconds,derivedUpdateSeconds,customMomentumLoadSeconds,schurUpdateSeconds,kspOperatorSeconds,operatorUpdateSeconds,
      momentumSolveSeconds,pressureSolveSeconds,pressurePcRefreshSeconds,pressurePcRefreshes,pressurePcReuses,PetscMax(0.0,simpleWall-measuredCore),staticPhysicalOperator?1:0,
      supgForm.c_str(),supgKernel.c_str(),supgQuadPoints));
    if(m10PcgProfile) {
      const PetscInt mixedDof=(PetscInt)M.tets.size()+3*D.ns;
      PetscCall(m10PrintPressurePCGProfile("all",m10PcgTotal,mixedDof));
      PetscCall(m10PrintPressurePCGProfile("warm_excluding_first_solve",m10PcgWarm,mixedDof));
    }
    if(resourceProfile) PetscCall(printResourceMark("before_root_gather",(PetscInt)M.tets.size(),0.0,tTotal0));

    std::array<std::vector<double>,3> Ug; PetscCall(customGatherOwnedVelocityToZero(U,D.velCount,Ug));
    std::vector<double> pGlobal; PetscCall(customGatherOwnedPressureToZero(pressureState,D.cellCount,pGlobal));
    if(rank==0) {
      if(P.mode==ProblemMode::Pipe && P.inletBC==InletBCMode::PipeParabolic) PetscCall(computePipeDiagnosticsRoot(M,D,P,Ug,pGlobal));
      else if(P.mode==ProblemMode::MMS) PetscCall(computeErrorsRoot(M,D,Ug,pGlobal));
      else PetscCall(computeFlowDiagnosticsRoot(M,D,P,Ug,pGlobal));
      if(writeVtu) PetscCall(writeVtuRoot(vtuOutput,M,D,Ug,pGlobal,converged,finalIt,vtuVelocityMode));
    }
    if(resourceProfile) PetscCall(printResourceMark("after_postprocess",(PetscInt)M.tets.size(),0.0,tTotal0));

    PetscLogDouble tTotal1;
    PetscCall(PetscTime(&tTotal1));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_TIMING ranks=%d assemblySeconds=%.6f operatorUpdateSeconds=%.6f simpleSolveSeconds=%.6f totalSeconds=%.6f\n",
      size,(double)(tAsm1-tAsm0),operatorUpdateSeconds,(double)(tSolve1-tSolve0),(double)(tTotal1-tTotal0)));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "P1BF3_RESULT status=%s variant=%s ranks=%d outerIts=%" PetscInt_FMT " finalRelCont=%.12e finalUInitRel=[%.12e,%.12e,%.12e] finalMomRel=%.12e gate=%.3e gateMode=all_initial_residuals_Ux_Uy_Uz_plus_continuity maxOuter=%" PetscInt_FMT " Re=%.12g convection=%s supg=%s tauScale=%.6g supgForm=%s supgKernel=%s supgQuad=%" PetscInt_FMT " solveFailed=%d pressureFailureReason=%" PetscInt_FMT "\n",
      converged?"PASS":"FAIL",simpleVariant.c_str(),size,finalIt,(double)finalPInitRel,(double)finalUInitRel[0],(double)finalUInitRel[1],(double)finalUInitRel[2],(double)momRelMax,(double)simpleTol,maxOuter,P.re,centralConvection?"central":"none",useSupg?"ON":"OFF",(double)supgTauScale,supgForm.c_str(),supgKernel.c_str(),supgQuadPoints,(int)solveFailed,pressureFailureReason));

    PetscBool gate3Pass=PETSC_TRUE;
    PetscInt gate3Setups=0,gate3Applies=0;
    PetscReal gate3MaxRel=0.0,gate3VolumeMin=0.0,gate3VolumeMax=0.0;
    if(gate3MpProbe) {
      gate3Setups=gate3MpCtx ? gate3MpCtx->setupCount : 0;
      gate3Applies=gate3MpCtx ? gate3MpCtx->applyCount : 0;
      gate3MaxRel=gate3MpCtx ? gate3MpCtx->maxAlgebraRel : PETSC_MAX_REAL;
      gate3VolumeMin=gate3MpCtx ? gate3MpCtx->volumeMin : 0.0;
      gate3VolumeMax=gate3MpCtx ? gate3MpCtx->volumeMax : 0.0;
      gate3Pass=(gate3Setups>=1 && gate3Applies>=1 && gate3MaxRel<=gate3MpAlgebraTol && gate3VolumeMin>0.0 && gate3VolumeMax>=gate3VolumeMin) ? PETSC_TRUE : PETSC_FALSE;
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "P1BF3_GATE3_RESULT=%s setupCount=%" PetscInt_FMT " applyCount=%" PetscInt_FMT " maxAlgebraRel=%.3e algebraTol=%.3e volumeMin=%.12e volumeMax=%.12e shellAttachment=shadow_only livePressurePC=UNCHANGED_GAMG exactOperator=UNCHANGED_custom_FP64_B_rAU_Bt mass=P0_exact_cell_volume_diagonal Kp=NONE Fp=NONE\n",
        gate3Pass?"PASS":"FAIL",gate3Setups,gate3Applies,(double)gate3MaxRel,(double)gate3MpAlgebraTol,(double)gate3VolumeMin,(double)gate3VolumeMax));
    }
    PetscBool gate4Pass=PETSC_TRUE;
    PetscBool gate6Pass=PETSC_TRUE;
    PetscBool gate7Pass=PETSC_TRUE;
    PetscBool gate8Pass=PETSC_TRUE;
    if(gate4KpProbe) {
      const PetscReal negTol=1e-10*PetscMax(gate4Kp.coeffMax,(PetscReal)1.0);
      gate4Pass=(gate4Kp.built && gate4Kp.symmetric && gate4Kp.nnz==gate4Kp.expectedNnz && gate4Kp.internalFaces==(PetscInt)M.neighbour.size() && gate4Kp.outletFaces>0 && gate4Kp.inletFaces>0 && gate4Kp.wallFaces>0 && gate4Kp.coeffMin>0.0 && gate4Kp.diagMin>0.0 && gate4Kp.constantActionNorm>0.0 && gate4Kp.constantActionMin>=-negTol && gate4Kp.testEnergy>0.0) ? PETSC_TRUE : PETSC_FALSE;
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_GATE4_RESULT=%s built=%d symmetric=%d nnz=%" PetscInt_FMT " expectedNnz=%" PetscInt_FMT " internalFaces=%" PetscInt_FMT " expectedInternalFaces=%" PetscInt_FMT " inletFaces=%" PetscInt_FMT " wallFaces=%" PetscInt_FMT " outletFaces=%" PetscInt_FMT " coeffMin=%.12e diagMin=%.12e constantActionNorm=%.12e constantActionMin=%.12e negativeRowSumTol=%.12e testEnergy=%.12e KpBC=inlet_Neumann_wall_Neumann_outlet_p0 pressureNullspace=OFF attachment=shadow_setup_only livePressurePC=UNCHANGED_GAMG exactOperator=UNCHANGED_custom_FP64_B_rAU_Bt Fp=%s GAMG_on_Kp=%s\n",gate4Pass?"PASS":"FAIL",(int)gate4Kp.built,(int)gate4Kp.symmetric,gate4Kp.nnz,gate4Kp.expectedNnz,gate4Kp.internalFaces,(PetscInt)M.neighbour.size(),gate4Kp.inletFaces,gate4Kp.wallFaces,gate4Kp.outletFaces,(double)gate4Kp.coeffMin,(double)gate4Kp.diagMin,(double)gate4Kp.constantActionNorm,(double)gate4Kp.constantActionMin,(double)negTol,(double)gate4Kp.testEnergy,gate9LivePcd?(gate9dGamgOnly?"BYPASSED_GATE9D_GAMG_ONLY":"BYPASSED_GATE9C_DIRECT_KP_GAMG"):(gate8EswBcProbe?"nuKp_plus_internal_Cp_plus_ESW_BC_GATE8":(gate7CpProbe?"nuKp_plus_internal_Cp_GATE7":(gate6DiffusionPcdProbe?"nu_times_Kp_GATE6":"NONE"))),gate9LivePcd?(gate9dGamgOnly?"LIVE_GATE9D_GAMG_ONLY_STATIONARY":"LIVE_GATE9C_ONE_GAMG_PCApply"):(gate5KpGamgProbe?"STANDALONE_GATE5_ONLY":(gate6DiffusionPcdProbe?"GATE6_KP_INVERSE_SHADOW_ONLY":"NOT_YET"))));
      if(gate5KpGamgProbe) {
        const PetscBool g5pass=(gate5KpGamg.ran && gate5KpGamg.cgPass && gate5KpGamg.cyclesFinite) ? PETSC_TRUE : PETSC_FALSE;
        const char* cycleTrend=(gate5KpGamg.cycleRel[3] < 1.0) ? "CONTRACTED_AFTER_4" : "NOT_CONTRACTED_AFTER_4";
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_GATE5_RESULT=%s cgIts=%" PetscInt_FMT " cgReason=%d trueRel=%.12e solutionRel=%.12e cgTrueRelTol=1.000e-08 cgSolutionRelTol=1.000e-07 gamgOnlyCycle1=%.12e gamgOnlyCycle2=%.12e gamgOnlyCycle3=%.12e gamgOnlyCycle4=%.12e gamgOnlyTrend=%s gamgOnlyIsDiagnostic=1 livePressurePC=UNCHANGED_GAMG exactOperator=UNCHANGED_custom_FP64_B_rAU_Bt Fp=NONE fullPCD=NOT_YET\n",
          g5pass?"PASS":"FAIL",gate5KpGamg.cgIts,(int)gate5KpGamg.reason,(double)gate5KpGamg.trueRel,(double)gate5KpGamg.solutionRel,(double)gate5KpGamg.cycleRel[0],(double)gate5KpGamg.cycleRel[1],(double)gate5KpGamg.cycleRel[2],(double)gate5KpGamg.cycleRel[3],cycleTrend));
      }
      if(gate7CpProbe) {
        gate7Pass=(gate7Cp.setupCount==1 && gate7Cp.updateCount>=1 && gate7Cp.probeCount>=1 && gate7Cp.internalFaces==(PetscInt)M.neighbour.size() && gate7Cp.allFinite && gate7Cp.maxFluxAbs>1.0e-14 && gate7Cp.maxNonzeroFluxFaces>0 && gate7Cp.sawNonsymmetric) ? PETSC_TRUE : PETSC_FALSE;
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"P1BF3_GATE7_RESULT=%s setupCount=%" PetscInt_FMT " updateCount=%" PetscInt_FMT " probeCount=%" PetscInt_FMT " internalFaces=%" PetscInt_FMT " expectedInternalFaces=%" PetscInt_FMT " maxNonzeroFluxFaces=%" PetscInt_FMT " maxFluxAbs=%.12e maxCpMassNorm=%.12e maxConvToDiff=%.12e sawNonsymmetric=%d allFinite=%d sourceVelocity=current_Picard_P1plusBF3 exactFaceMean=1 centralPressureInterpolation=1 boundaryConvection=internal_only_component robin=%s attachment=shadow_only livePressurePC=UNCHANGED_GAMG exactOperator=UNCHANGED_custom_FP64_B_rAU_Bt fullPCD=NOT_LIVE_YET\n",gate7Pass?"PASS":"FAIL",gate7Cp.setupCount,gate7Cp.updateCount,gate7Cp.probeCount,gate7Cp.internalFaces,(PetscInt)M.neighbour.size(),gate7Cp.maxNonzeroFluxFaces,(double)gate7Cp.maxFluxAbs,(double)gate7Cp.maxCpMassNorm,(double)gate7Cp.maxConvToDiff,(int)gate7Cp.sawNonsymmetric,(int)gate7Cp.allFinite,gate8EswBcProbe?"HANDLED_SEPARATELY_GATE8":"NOT_YET_GATE8"));
      }
      if(gate8EswBcProbe) {
        gate8Pass=(gate8EswBc.setupCount==1 && gate8EswBc.updateCount>=1 && gate8EswBc.probeCount>=1 && gate8EswBc.inletFaces==gate4Kp.inletFaces && gate8EswBc.wallFaces==gate4Kp.wallFaces && gate8EswBc.outletFaces==gate4Kp.outletFaces && gate8EswBc.inletFluxSignOkay && gate8EswBc.allFinite && gate8EswBc.maxCancelRel<=gate8CancelTol && gate8EswBc.maxFullVsInteriorRel<=gate8CancelTol && gate8EswBc.maxWallFluxAbs<=gate8WallFluxTol) ? PETSC_TRUE : PETSC_FALSE;
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
          "P1BF3_GATE8_RESULT=%s setupCount=%" PetscInt_FMT " updateCount=%" PetscInt_FMT " probeCount=%" PetscInt_FMT " inletFaces=%" PetscInt_FMT " wallFaces=%" PetscInt_FMT " outletFaces=%" PetscInt_FMT " inletFluxSignOkay=%d maxInletCancellationRel=%.12e cancelTol=%.12e maxFullVsGate7InteriorRel=%.12e maxWallFluxAbs=%.12e wallFluxTol=%.12e allFinite=%d inletRobinFormula=-nu_dpdn_plus_wdotn_p_eq_0 inletRobinActive=1 conservativeBoundaryConvectionAndRobinCancel=1 wallBC=homogeneous_Neumann_wdotn0 outletBC=p0_Dirichlet pressureNullspace=OFF attachment=shadow_only KpInverse=NOT_APPLIED_GATE8 livePressurePC=UNCHANGED_GAMG exactOperator=UNCHANGED_custom_FP64_B_rAU_Bt fullPCD=READY_FOR_GATE9_NOT_LIVE_YET\n",
          gate8Pass?"PASS":"FAIL",gate8EswBc.setupCount,gate8EswBc.updateCount,gate8EswBc.probeCount,gate8EswBc.inletFaces,gate8EswBc.wallFaces,gate8EswBc.outletFaces,(int)gate8EswBc.inletFluxSignOkay,(double)gate8EswBc.maxCancelRel,(double)gate8CancelTol,(double)gate8EswBc.maxFullVsInteriorRel,(double)gate8EswBc.maxWallFluxAbs,(double)gate8WallFluxTol,(int)gate8EswBc.allFinite));
      }
      if(gate6DiffusionPcdProbe) {
        gate6Pass=(gate6DiffusionPcdCtx && gate6DiffusionPcdCtx->setupCount>=1 && gate6DiffusionPcdCtx->applyCount>=1 && gate6DiffusionPcdCtx->allKpConverged && gate6DiffusionPcdCtx->maxChainRel<=gate6ChainTol && gate6DiffusionPcdCtx->maxKpTrueRel<=gate6KpTrueResidualTol) ? PETSC_TRUE : PETSC_FALSE;
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
          "P1BF3_GATE6_RESULT=%s setupCount=%" PetscInt_FMT " applyCount=%" PetscInt_FMT " totalKpIts=%" PetscInt_FMT " maxChainRel=%.3e chainTol=%.3e maxKpTrueRel=%.3e kpTrueResidualTol=%.3e allKpConverged=%d chain=Kp_inverse_times_nuKp_times_Mp_inverse expected=nu_times_Mp_inverse attachment=shadow_only livePressurePC=UNCHANGED_GAMG exactOperator=UNCHANGED_custom_FP64_B_rAU_Bt Fp=nu_times_Kp fullPCD=NOT_YET\n",
          gate6Pass?"PASS":"FAIL",gate6DiffusionPcdCtx->setupCount,gate6DiffusionPcdCtx->applyCount,gate6DiffusionPcdCtx->totalKpIts,
          (double)gate6DiffusionPcdCtx->maxChainRel,(double)gate6ChainTol,(double)gate6DiffusionPcdCtx->maxKpTrueRel,(double)gate6KpTrueResidualTol,(int)gate6DiffusionPcdCtx->allKpConverged));
      }
      if(gate9LivePcd) {
        const PetscBool gate9Pass=(gate9LivePcdCtx && gate9LivePcdCtx->setupCount>=1 && gate9LivePcdCtx->applyCount>=1 && gate9LivePcdCtx->outerSolveCount>=1 && gate9LivePcdCtx->allFinite && gate9LivePcdCtx->maxOuterTrueRel<=gate9OuterTrueResidualTol) ? PETSC_TRUE : PETSC_FALSE;
        if(gate9dGamgOnly) PetscCall(PetscPrintf(PETSC_COMM_WORLD,
          "P1BF3_GATE9D_RESULT=%s setupCount=%" PetscInt_FMT " pressureSolveCount=%" PetscInt_FMT " applyCount=%" PetscInt_FMT " maxOneCycleKpRel=%.3e finalTrueRel=%.3e trueResidualTol=%.3e totalGamgApplySeconds=%.6e allFinite=%d pressureAlgorithm=stationary_exactSchur_residual_correction outerKrylov=NONE KSPSolve=NEVER_CALLED omega=%.6e Kp=Gate4_geometric_FV_laplacian exactOperator=UNCHANGED_custom_FP64_B_rAU_Bt\n",
          gate9Pass?"PASS":"FAIL",gate9LivePcdCtx?gate9LivePcdCtx->setupCount:0,gate9LivePcdCtx?gate9LivePcdCtx->outerSolveCount:0,gate9LivePcdCtx?gate9LivePcdCtx->applyCount:0,gate9LivePcdCtx?(double)gate9LivePcdCtx->maxKpCycleRel:PETSC_MAX_REAL,gate9LivePcdCtx?(double)gate9LivePcdCtx->lastOuterTrueRel:PETSC_MAX_REAL,(double)gate9OuterTrueResidualTol,gate9LivePcdCtx?(double)gate9LivePcdCtx->totalApplySeconds:0.0,gate9LivePcdCtx?(int)gate9LivePcdCtx->allFinite:0,(double)gate9dOmega));
        else PetscCall(PetscPrintf(PETSC_COMM_WORLD,
          "P1BF3_GATE9C_RESULT=%s setupCount=%" PetscInt_FMT " outerSolveCount=%" PetscInt_FMT " applyCount=%" PetscInt_FMT " maxOneCycleKpRel=%.3e maxOuterTrueRel=%.3e outerTrueResidualTol=%.3e totalGamgApplySeconds=%.6e allFinite=%d liveOuter=FGMRES livePC=ONE_direct_GAMG_PCApply_geometric_Kp innerKrylov=NONE Fp=BYPASSED Mp=BYPASSED Kp=Gate4_geometric_FV_laplacian exactOperator=UNCHANGED_custom_FP64_B_rAU_Bt statusMeaning=M23_STYLE_GEOMETRIC_KP_GAMG\n",
          gate9Pass?"PASS":"FAIL",gate9LivePcdCtx?gate9LivePcdCtx->setupCount:0,gate9LivePcdCtx?gate9LivePcdCtx->outerSolveCount:0,gate9LivePcdCtx?gate9LivePcdCtx->applyCount:0,gate9LivePcdCtx?(double)gate9LivePcdCtx->maxKpCycleRel:PETSC_MAX_REAL,gate9LivePcdCtx?(double)gate9LivePcdCtx->maxOuterTrueRel:PETSC_MAX_REAL,(double)gate9OuterTrueResidualTol,gate9LivePcdCtx?(double)gate9LivePcdCtx->totalApplySeconds:0.0,gate9LivePcdCtx?(int)gate9LivePcdCtx->allFinite:0));
        if(!gate9Pass && !gate9dGamgOnly) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate-9C direct Kp-GAMG outer FGMRES did not converge");
        // Destroy outer KSP now so its PCShell releases the Gate-9 context while
        // Kp/Cp/Gate-8 vectors are still alive.
        PetscCall(KSPDestroy(&pksp)); ppc=nullptr; gate9LivePcdCtx=nullptr;
      }
      PetscCall(PCDestroy(&gate6DiffusionPcdShell));
      gate6DiffusionPcdCtx=nullptr;
      if(gate8EswBcProbe) PetscCall(gate8EswBcDestroy(gate8EswBc));
      if(gate7CpProbe) PetscCall(gate7CpDestroy(gate7Cp));
      PetscCall(gate4DestroyKp(gate4Kp));
      if(gate9eNgfv) PetscCall(MatDestroy(&gate9eAudit.raw));
    }
    PetscCall(PCDestroy(&gate3MpShell));
    gate3MpCtx=nullptr;
    PetscCall(VecDestroy(&pcIn));
    PetscCall(VecDestroy(&pcOut));
    PetscCall(KSPDestroy(&pksp));
    if(gate3MpProbe && !gate3Pass)
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate-3 P0 pressure-mass inverse algebra check failed");
    if(gate4KpProbe && !gate4Pass)
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate-4 geometric pressure Laplacian audit failed");
    if(gate5KpGamgProbe && !(gate5KpGamg.ran && gate5KpGamg.cgPass && gate5KpGamg.cyclesFinite))
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate-5 standalone Kp CG+GAMG solve failed");
    if(gate6DiffusionPcdProbe && !gate6Pass)
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate-6 diffusion-only PCD algebra check failed");
    if(gate7CpProbe && !gate7Pass)
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate-7 internal pressure convection audit failed");
    if(gate8EswBcProbe && !gate8Pass)
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Gate-8 ESW pressure boundary-condition audit failed");
    PetscCall(MatNullSpaceDestroy(&nsp));
    PetscCall(MatDestroy(&factoredSchur));
    PetscCall(destroyPressureAssemblyPlan(PSchur));
    PetscCall(MatDestroy(&C));
    PetscCall(MatDestroy(&Sg));
    PetscCall(destroyDiscrete(D));
  } catch(const std::exception& e) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"%s",e.what());
  }
  PetscCall(PetscFinalize());
  return 0;
}
