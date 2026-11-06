#pragma once
// Stage27 experimental P1 potential initialization for an arbitrary pure-tet mesh.
// Solves integral(grad phi . grad test) = integral(inlet test (Uin.n))
// with phi=0 at outlet vertices and impermeable Neumann walls.
// Then mass-lumped projects piecewise-constant grad(phi) into the conforming
// Q1 vertex velocity field and uses the existing BF3 boundary-face mode to
// impose EXACT face-mean inlet/wall normal velocities. Interior discrete
// continuity is not asserted by this projection: SIMPLE still corrects it.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <utility>
namespace nodals {namespace initial27 {
struct Result{int iterations=0;double potentialRel=0,inletFlux=0,wallFluxMax=0,wallFluxL1=0,
  outletFlux=0,fluxImbalanceRel=0,rawMeanWallFlux=0,maxInitialSpeed=0;};
using V=std::array<double,3>;
inline V minus(V a,V b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
inline V cross(V a,V b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
inline double dot(V a,V b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
template<class Point> V xyz(const Point&p){return {p.x,p.y,p.z};}
struct Element {std::array<int,4> v{};std::array<V,4> grad{};double volume=0;};
template<class Mesh> inline std::vector<Element> elements(const Mesh&M){
  std::vector<Element> out;out.reserve(M.tets.size());
  for(const auto&t:M.tets){
    Element e;for(int i=0;i<4;++i)e.v[i]=t.v[i];
    const V p0=xyz(M.points[e.v[0]]),p1=xyz(M.points[e.v[1]]),
            p2=xyz(M.points[e.v[2]]),p3=xyz(M.points[e.v[3]]);
    const V a=minus(p1,p0),b=minus(p2,p0),c=minus(p3,p0);
    const double det=dot(a,cross(b,c));
    if(!std::isfinite(det)||std::abs(det)<1e-300)throw std::runtime_error("STAGE27_POTENTIAL invalid tet determinant");
    e.volume=std::abs(det)/6.;
    V g1=cross(b,c),g2=cross(c,a),g3=cross(a,b);
    for(int j=0;j<3;++j){e.grad[1][j]=g1[j]/det;e.grad[2][j]=g2[j]/det;e.grad[3][j]=g3[j]/det;
      e.grad[0][j]=-e.grad[1][j]-e.grad[2][j]-e.grad[3][j];}
    out.push_back(e);
  }
  return out;
}
template<class Mesh,class Registry,class Inlet,class Initial>
Result apply(const Mesh&M,const Registry&R,const Inlet&inlet,Initial&init,
  double tolerance=1.e-10,int maxIterations=12000){
  if(M.h.nhex!=0||M.h.ntet==0||M.h.nv==0||M.tets.size()!=M.h.ntet)
    throw std::runtime_error("STAGE27_POTENTIAL requires pure-tet mesh");
  const int n=(int)M.h.nv;
  const auto elems=elements(M);
  std::vector<std::unordered_map<int,double>> rows((size_t)n);
  std::vector<double> rhs((size_t)n,0.);
  std::vector<unsigned char> outlet((size_t)n,0);
  for(const auto&f:R.faces)if(f.kind=="outlet"){
    constexpr int outletFaceIds[4][3]={{1,2,3},{0,3,2},{0,1,3},{0,2,1}};
    for(int j=0;j<3;++j)outlet[M.tets[f.cell].v[outletFaceIds[f.localFace][j]]]=1;
  }
  // Same local-face convention as Stage19 registry; opposed vertex = localFace.
  constexpr int faceIds[4][3]={{1,2,3},{0,3,2},{0,1,3},{0,2,1}};
  for(const auto&f:R.faces)if(f.kind=="inlet"){
    const double q=inlet.x*f.nx+inlet.y*f.ny+inlet.z*f.nz;
    for(int k=0;k<3;++k)rhs[M.tets[f.cell].v[faceIds[f.localFace][k]]]+=q*f.area/3.;
  }
  for(const auto&e:elems)for(int a=0;a<4;++a)for(int b=0;b<4;++b)
    rows[e.v[a]][e.v[b]]+=e.volume*dot(e.grad[a],e.grad[b]);
  std::vector<std::vector<std::pair<int,double>>> csr((size_t)n);
  std::vector<double> invDiag((size_t)n,0.);
  int fixedCount=0;
  for(int i=0;i<n;++i){
    if(outlet[i]){++fixedCount;csr[i].push_back({i,1.});invDiag[i]=1.;rhs[i]=0.;continue;}
    for(const auto&j:rows[i])if(!outlet[j.first])csr[i].push_back(j);
    std::sort(csr[i].begin(),csr[i].end(),[](auto a,auto b){return a.first<b.first;});
    for(const auto&j:csr[i])if(j.first==i)invDiag[i]=1./j.second;
    if(!(invDiag[i]>0.)||!std::isfinite(invDiag[i]))throw std::runtime_error("STAGE27_POTENTIAL nonpositive Laplace diagonal");
  }
  if(fixedCount==0)throw std::runtime_error("STAGE27_POTENTIAL no outlet Dirichlet vertices");
  auto mv=[&](const std::vector<double>&x,std::vector<double>&y){
    for(int i=0;i<n;++i){double s=0.;for(const auto&v:csr[i])s+=v.second*x[v.first];y[i]=s;}
  };
  auto dprod=[&](const std::vector<double>&x,const std::vector<double>&y){
    long double s=0;for(int i=0;i<n;++i)s+=(long double)x[i]*y[i];return (double)s;
  };
  std::vector<double> phi((size_t)n,0.),r=rhs,z((size_t)n),p((size_t)n),Ap((size_t)n);
  const double bnorm=std::sqrt(dprod(rhs,rhs));
  if(!(bnorm>0.)||!std::isfinite(bnorm))throw std::runtime_error("STAGE27_POTENTIAL zero/nonfinite inlet Neumann source");
  for(int i=0;i<n;++i)z[i]=invDiag[i]*r[i];
  p=z;
  double rho=dprod(r,z);Result report;bool converged=false;
  for(int it=0;it<maxIterations;++it){
    mv(p,Ap);const double pq=dprod(p,Ap);
    if(!(pq>0.)||!std::isfinite(pq))throw std::runtime_error("STAGE27_POTENTIAL CG pAp nonpositive");
    const double alpha=rho/pq;
    for(int i=0;i<n;++i){phi[i]+=alpha*p[i];r[i]-=alpha*Ap[i];}
    const double norm=std::sqrt(dprod(r,r));report.iterations=it+1;report.potentialRel=norm/bnorm;
    if(report.potentialRel<tolerance){converged=true;break;}
    for(int i=0;i<n;++i)z[i]=invDiag[i]*r[i];
    const double next=dprod(r,z);if(!(next>0.)||!std::isfinite(next))throw std::runtime_error("STAGE27_POTENTIAL CG breakdown");
    for(int i=0;i<n;++i)p[i]=z[i]+(next/rho)*p[i];
    rho=next;
  }
  std::printf("STAGE27_POTENTIAL_LAPLACE vertices=%d tets=%zu outletDirichlet=%d its=%d rel=%.12e tol=%.2e status=%s\n",
    n,elems.size(),fixedCount,report.iterations,report.potentialRel,tolerance,converged?"PASS":"FAIL");
  if(!converged)throw std::runtime_error("STAGE27_POTENTIAL Laplace solve not converged");
  // L2 mass-lumped projection from P1 tetrahedral potential gradient into the
  // continuous Q1 velocity vertex DOFs; initially zero BF3 coefficients.
  std::vector<V> u((size_t)n,V{0.,0.,0.});std::vector<double>mass((size_t)n,0.);
  for(const auto&e:elems){
    V gradPhi{0.,0.,0.};
    for(int a=0;a<4;++a)for(int k=0;k<3;++k)gradPhi[k]+=phi[e.v[a]]*e.grad[a][k];
    for(int a=0;a<4;++a){const int v=e.v[a];mass[v]+=e.volume/4.;
      for(int k=0;k<3;++k)u[v][k]+=gradPhi[k]*e.volume/4.;}
  }
  for(int i=0;i<n;++i){
    if(!(mass[i]>0.))throw std::runtime_error("STAGE27_POTENTIAL orphan vertex");
    for(int k=0;k<3;++k)u[i][k]/=mass[i];
    report.maxInitialSpeed=std::max(report.maxInitialSpeed,std::sqrt(dot(u[i],u[i])));
    init.u0[i]=u[i][0];init.u1[i]=u[i][1];init.u2[i]=u[i][2];
  }
  // Face bubble mean = 27/60=0.45. Its trace is zero on all other faces.
  // Apply the correction NORMAL to the face, matching the existing BF3
  // exact mean wall-flux constraint and inlet flux in the trace space.
  constexpr double bubbleFaceMean=0.45;
  // First evaluate the uncorrected outlet flux to distribute only its
  // global mass-balance defect. Outlet bubble coefficients are zero before
  // this point. This does not assert local cellwise divergence-freedom.
  double projectedOutletFlux=0.,outletArea=0.;
  for(const auto&f:R.faces)if(f.kind=="outlet"){
    V faceU{0.,0.,0.}; const auto&t=M.tets[f.cell];
    for(int k=0;k<3;++k){const int v=t.v[faceIds[f.localFace][k]];
      for(int d=0;d<3;++d)faceU[d]+=u[v][d]/3.;}
    projectedOutletFlux+=f.area*dot(faceU,V{f.nx,f.ny,f.nz});
    outletArea+=f.area;
  }
  if(!(outletArea>0.))throw std::runtime_error("STAGE27_POTENTIAL missing outlet area");
  const double prescribedInletFlux=[&](){double s=0.;for(const auto&f:R.faces)
    if(f.kind=="inlet")s+=f.area*(inlet.x*f.nx+inlet.y*f.ny+inlet.z*f.nz);
    return s;}();
  const double outletNormalCorrection=(-prescribedInletFlux-projectedOutletFlux)/outletArea;
  for(const auto&f:R.faces){
    const auto&t=M.tets[f.cell];const V normal{f.nx,f.ny,f.nz};
    V faceU{0.,0.,0.};
    for(int k=0;k<3;++k){const int v=t.v[faceIds[f.localFace][k]];
      for(int d=0;d<3;++d)faceU[d]+=u[v][d]/3.;}
    const double before=dot(faceU,normal);
    const double goal=f.kind=="inlet"?inlet.x*f.nx+inlet.y*f.ny+inlet.z*f.nz:
      f.kind=="outlet"?before+outletNormalCorrection:0.;
    const int bubble=M.tetVel[f.cell].g[4+f.localFace];
    if(bubble<n||bubble>=(int)init.u0.size())throw std::runtime_error("STAGE27_POTENTIAL BF3 face DOF invalid");
    // Shared boundary bubble may have a nonzero preexisting value only if
    // another patch has already assigned it: reject inconsistent topology.
    if(init.u0[bubble]!=0||init.u1[bubble]!=0||init.u2[bubble]!=0)
      throw std::runtime_error("STAGE27_POTENTIAL duplicate boundary BF3 DOF");
    const double corr=(goal-before)/bubbleFaceMean;
    init.u0[bubble]=corr*normal[0];init.u1[bubble]=corr*normal[1];init.u2[bubble]=corr*normal[2];
    const double after=before+bubbleFaceMean*corr;
    if(f.kind=="inlet")report.inletFlux+=after*f.area;
    else if(f.kind=="outlet")report.outletFlux+=after*f.area;
    else {report.wallFluxMax=std::max(report.wallFluxMax,std::abs(after));
      report.wallFluxL1+=f.area*std::abs(after);report.rawMeanWallFlux+=f.area*std::abs(before);}
  }
  report.fluxImbalanceRel=std::abs(report.inletFlux+report.outletFlux)/std::max(1e-300,std::abs(report.inletFlux));
  std::printf("STAGE27_POTENTIAL_PROJECT velocity=P1_GRAD_PHI_LUMPED_Q1_BF3 inletFlux=%.12e outletFlux=%.12e globalMismatchRel=%.12e wallFluxMax=%.12e wallFluxL1=%.12e wallRawLeakageL1=%.12e maxNodalSpeed=%.12e interiorDiscreteContinuity=REQUIRES_SIMPLE status=PASS\n",
    report.inletFlux,report.outletFlux,report.fluxImbalanceRel,report.wallFluxMax,report.wallFluxL1,report.rawMeanWallFlux,report.maxInitialSpeed);
  if(!(report.wallFluxMax<1e-10*std::max(report.maxInitialSpeed,1.))||!(report.inletFlux<0.)||!(report.fluxImbalanceRel<1.e-10))
    throw std::runtime_error("STAGE27_POTENTIAL invalid projected wall/inlet face mean flux");
  return report;
}
}}
