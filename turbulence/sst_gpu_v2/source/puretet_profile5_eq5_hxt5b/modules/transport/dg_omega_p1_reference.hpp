#pragma once
// NodalS omega-only DG-P1: portable C++17 reference operator.
// This is a correctness oracle for the CUDA implementation, NOT the coupled G2 solver.
// Unknown ordering: omega[4*tet + localVertex], no CG interelement sharing.
// Diffusion: symmetric interior penalty (SIPG), convection: advective-volume
// plus LF/upwind strong-form face correction. Consistent Nitsche physical boundary.
// Coefficients are frozen per element in this oracle; production GPU path must
// evaluate the real SST-2003m coefficients at quadrature points.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace nodals { namespace omega_dg {
struct Vec3 {
  double x=0, y=0, z=0;
  Vec3 operator+(Vec3 b) const {return {x+b.x,y+b.y,z+b.z};}
  Vec3 operator-(Vec3 b) const {return {x-b.x,y-b.y,z-b.z};}
  Vec3 operator*(double s) const {return {s*x,s*y,s*z};}
  Vec3 operator/(double s) const {return *this*(1./s);}
};
inline double dot(Vec3 a,Vec3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
inline Vec3 cross(Vec3 a,Vec3 b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
inline double norm(Vec3 a){return std::sqrt(dot(a,a));}
using Tet=std::array<int,4>;
using Triangle=std::array<int,3>;
struct Geometry {
  double volume=0;
  std::array<Vec3,4> grad{};
};
struct FaceSide {int tet=-1,opposite=-1;};
struct Face {Triangle vertices{};FaceSide left,right;Vec3 normal{};double area=0;};
enum class BoundaryKind { Natural, Inlet, Wall, Outlet };
struct BoundaryValue {
  BoundaryKind kind=BoundaryKind::Natural;
  // For Wall: externally computed trace_log omega target at this physical point.
  // For Inlet/Outlet backflow: value prescribed by current SST inlet policy.
  std::function<double(Vec3)> target;
};
struct Mesh {
  std::vector<Vec3> vertices;
  std::vector<Tet> tetra;
  std::vector<Geometry> cells;
  std::vector<Face> faces;
  std::map<Triangle,BoundaryValue> boundary;
};
struct Coeff {
  double molecularAndTurbulentDiffusion=0; // nu + sigmaW*nut
  double reaction=0; // beta*omega_old
  double source=0; // omega production + cross diffusion
  Vec3 velocity{}; // frozen carrier u in this portable oracle
};
struct Options {
  double penaltyC=10.;       // eta=C(p+1)^2 max(mu-/h-,mu+/h+)
  double convectionTheta=1.; // 1=upwind/LF, 0=central
  double boundaryTraceGamma=100000.; // benchmark's independent trace_log coefficient
  double wallNormalTolerance=1.e-8;
};
struct Operator {
  int ndof=0;
  std::vector<std::map<int,double>> rows;
  std::vector<double> rhs;
  explicit Operator(int n=0):ndof(n),rows((size_t)n),rhs((size_t)n,0){}
  void add(int i,int j,double v){
    if(i<0||i>=ndof||j<0||j>=ndof||!std::isfinite(v))
      throw std::runtime_error("DG omega: invalid sparse matrix entry");
    rows[(size_t)i][j]+=v;
  }
  void add_rhs(int i,double v){
    if(i<0||i>=ndof||!std::isfinite(v))
      throw std::runtime_error("DG omega: invalid rhs entry");
    rhs[(size_t)i]+=v;
  }
  std::vector<double> multiply(const std::vector<double>&x) const {
    if((int)x.size()!=ndof)throw std::runtime_error("DG omega: wrong x size");
    std::vector<double> out((size_t)ndof,0);
    for(int i=0;i<ndof;++i)for(auto [j,v]:rows[(size_t)i])out[(size_t)i]+=v*x[(size_t)j];
    return out;
  }
  double max_asymmetry() const {
    double worst=0;
    for(int i=0;i<ndof;++i)for(auto [j,v]:rows[(size_t)i]){
      auto it=rows[(size_t)j].find(i);
      worst=std::max(worst,std::abs(v-(it==rows[(size_t)j].end()?0:it->second)));
    }
    return worst;
  }
  std::size_t nnz() const {std::size_t n=0;for(const auto&r:rows)n+=r.size();return n;}
};
inline int dof(int t,int a){return 4*t+a;}
inline Geometry tetra_geometry(const std::vector<Vec3>&p,const Tet&t){
  for(int v:t)if(v<0||v>=(int)p.size())throw std::runtime_error("DG omega: invalid vertex");
  Vec3 e1=p[(size_t)t[1]]-p[(size_t)t[0]],e2=p[(size_t)t[2]]-p[(size_t)t[0]],e3=p[(size_t)t[3]]-p[(size_t)t[0]];
  const double det=dot(e1,cross(e2,e3));
  if(!std::isfinite(det)||std::abs(det)<=1.e-25)throw std::runtime_error("DG omega: invalid tetra volume");
  Geometry g;g.volume=std::abs(det)/6.;
  g.grad[1]=cross(e2,e3)/det;g.grad[2]=cross(e3,e1)/det;g.grad[3]=cross(e1,e2)/det;
  g.grad[0]=(g.grad[1]+g.grad[2]+g.grad[3])*(-1.);
  return g;
}
inline Triangle face_vertices(const Tet&t,int opposite){
  Triangle f{};int n=0;for(int a=0;a<4;++a)if(a!=opposite)f[(size_t)n++]=t[(size_t)a];
  std::sort(f.begin(),f.end());return f;
}
inline void initialize(Mesh&mesh){
  mesh.cells.clear();mesh.faces.clear();
  std::map<Triangle,int> owner;
  for(int t=0;t<(int)mesh.tetra.size();++t){
    const Tet&tet=mesh.tetra[(size_t)t];
    Tet sorted=tet;std::sort(sorted.begin(),sorted.end());
    if(std::adjacent_find(sorted.begin(),sorted.end())!=sorted.end())throw std::runtime_error("DG omega: duplicate tet vertex");
    mesh.cells.push_back(tetra_geometry(mesh.vertices,tet));
    for(int o=0;o<4;++o){
      Triangle key=face_vertices(tet,o);
      auto it=owner.find(key);
      if(it==owner.end()){
        Face f;f.vertices=key;f.left={t,o};
        owner[key]=(int)mesh.faces.size();mesh.faces.push_back(f);
      }else{
        Face&f=mesh.faces[(size_t)it->second];
        if(f.right.tet!=-1)throw std::runtime_error("DG omega: nonmanifold face");
        f.right={t,o};
      }
    }
  }
  for(Face&f:mesh.faces){
    Vec3 p0=mesh.vertices[(size_t)f.vertices[0]],p1=mesh.vertices[(size_t)f.vertices[1]],p2=mesh.vertices[(size_t)f.vertices[2]];
    Vec3 n=cross(p1-p0,p2-p0);
    const double nmag=norm(n);if(!(nmag>1.e-18))throw std::runtime_error("DG omega: degenerate face");
    Vec3 opp=mesh.vertices[(size_t)mesh.tetra[(size_t)f.left.tet][(size_t)f.left.opposite]];
    if(dot(n,opp-p0)>0)n=n*(-1.);
    f.normal=n/nmag;f.area=nmag*0.5;
    if(f.right.tet<0 && !mesh.boundary.count(f.vertices))mesh.boundary[f.vertices]=BoundaryValue{};
  }
}
inline std::array<double,4> face_phi(const Mesh&mesh,int tet,const Triangle&verts,const std::array<double,3>&triBary){
  std::array<double,4> phi{0,0,0,0};
  for(int a=0;a<4;++a)for(int f=0;f<3;++f)
    if(mesh.tetra[(size_t)tet][(size_t)a]==verts[(size_t)f])phi[(size_t)a]=triBary[(size_t)f];
  return phi;
}
inline Vec3 face_x(const Mesh&mesh,const Triangle&tri,const std::array<double,3>&b){
  Vec3 x{};for(int q=0;q<3;++q)x=x+mesh.vertices[(size_t)tri[(size_t)q]]*b[(size_t)q];return x;
}
inline constexpr std::array<std::array<double,3>,3> QTRI{{
  {{2./3.,1./6.,1./6.}},{{1./6.,2./3.,1./6.}},{{1./6.,1./6.,2./3.}}}};
inline double positive_target(const BoundaryValue&bc,Vec3 x){
  if(!bc.target)throw std::runtime_error("DG omega: missing exterior/wall target");
  double g=bc.target(x);
  if(!std::isfinite(g)||g<=0)throw std::runtime_error("DG omega: nonpositive exterior/wall omega target");
  return g;
}
inline Operator assemble(const Mesh&mesh,const std::vector<Coeff>&c,const Options&o){
  const int nt=(int)mesh.tetra.size();
  if(mesh.cells.size()!=mesh.tetra.size()||c.size()!=mesh.tetra.size()||!(o.penaltyC>0)||
     !(o.boundaryTraceGamma>0)||!(o.convectionTheta>=0&&o.convectionTheta<=1)||
     !(o.wallNormalTolerance>=0))throw std::runtime_error("DG omega: invalid assembly input");
  Operator A(4*nt);
  for(int t=0;t<nt;++t){
    const Geometry&g=mesh.cells[(size_t)t];const Coeff&z=c[(size_t)t];
    if(!std::isfinite(z.molecularAndTurbulentDiffusion)||z.molecularAndTurbulentDiffusion<0||
       !std::isfinite(z.reaction)||!std::isfinite(z.source))throw std::runtime_error("DG omega: invalid SST coefficient");
    for(int a=0;a<4;++a){
      A.add_rhs(dof(t,a),z.source*g.volume/4.);
      for(int b=0;b<4;++b){
        const double diffusion=z.molecularAndTurbulentDiffusion*g.volume*dot(g.grad[(size_t)a],g.grad[(size_t)b]);
        const double reaction=z.reaction*g.volume*(a==b?1./10.:1./20.);
        const double adv=g.volume/4.*dot(z.velocity,g.grad[(size_t)b]);
        A.add(dof(t,a),dof(t,b),diffusion+reaction+adv);
      }
    }
  }
  for(const Face&f:mesh.faces){
    const int L=f.left.tet, R=f.right.tet;
    const double muL=c[(size_t)L].molecularAndTurbulentDiffusion;
    const double hL=3.*mesh.cells[(size_t)L].volume/f.area;
    if(!(hL>0))throw std::runtime_error("DG omega: invalid left face height");
    if(R>=0){
      const double muR=c[(size_t)R].molecularAndTurbulentDiffusion;
      const double hR=3.*mesh.cells[(size_t)R].volume/f.area;
      if(!(hR>0))throw std::runtime_error("DG omega: invalid right face height");
      const double eta=o.penaltyC*4.*std::max(muL/hL,muR/hR);
      // Shared normal advection speed; when u is not perfectly H(div), use
      // the single arithmetic face speed and audit the two traces separately.
      const Vec3 u=(c[(size_t)L].velocity+c[(size_t)R].velocity)*0.5;
      const double un=dot(u,f.normal),alpha=o.convectionTheta*std::abs(un);
      const double cL=0.5*(alpha-un),cR=0.5*(alpha+un);
      for(auto q:QTRI){
        const double w=f.area/3.;
        const auto pl=face_phi(mesh,L,f.vertices,q),pr=face_phi(mesh,R,f.vertices,q);
        for(int sa=0;sa<2;++sa){
          const int tA=sa?R:L;const double sA=sa?-1.:1.;
          const auto&pa=sa?pr:pl;
          const double muA=sa?muR:muL;
          for(int a=0;a<4;++a){
            const double jv=sA*pa[(size_t)a];
            const double fluxV=0.5*muA*dot(mesh.cells[(size_t)tA].grad[(size_t)a],f.normal);
            for(int sb=0;sb<2;++sb){
              const int tB=sb?R:L;const double sB=sb?-1.:1.;
              const auto&pb=sb?pr:pl;
              const double muB=sb?muR:muL;
              for(int b=0;b<4;++b){
                const double jw=sB*pb[(size_t)b];
                const double fluxW=0.5*muB*dot(mesh.cells[(size_t)tB].grad[(size_t)b],f.normal);
                double entry=w*(-fluxW*jv-fluxV*jw+eta*jw*jv);
                // Advective-form DG flux correction on each element side:
                // n_L=n, n_R=-n; upwind theta=1 => only inflow side penalized.
                if(sa==0)entry+=w*cL*pa[(size_t)a]*(sb==0?pb[(size_t)b]:-pb[(size_t)b]);
                else entry+=w*cR*pa[(size_t)a]*(sb==1?pb[(size_t)b]:-pb[(size_t)b]);
                A.add(dof(tA,a),dof(tB,b),entry);
              }
            }
          }
        }
      }
    }else{
      auto it=mesh.boundary.find(f.vertices);
      if(it==mesh.boundary.end())throw std::runtime_error("DG omega: missing boundary classification");
      const BoundaryValue&bc=it->second;
      const double un=dot(c[(size_t)L].velocity,f.normal);
      if(bc.kind==BoundaryKind::Wall && std::abs(un)>o.wallNormalTolerance)
        throw std::runtime_error("DG omega: nonzero carrier normal velocity at wall; do not hide leakage");
      const bool dirichlet=bc.kind==BoundaryKind::Wall||bc.kind==BoundaryKind::Inlet;
      const bool backflow=bc.kind==BoundaryKind::Outlet&&un<0;
      for(auto q:QTRI){
        const double w=f.area/3.;const auto p=face_phi(mesh,L,f.vertices,q);
        const Vec3 x=face_x(mesh,f.vertices,q);
        if(dirichlet){
          const double target=positive_target(bc,x);
          const double eta=o.boundaryTraceGamma*muL/hL;
          for(int a=0;a<4;++a){
            const double gA=dot(mesh.cells[(size_t)L].grad[(size_t)a],f.normal);
            A.add_rhs(dof(L,a),w*(-muL*gA+eta*p[(size_t)a])*target);
            for(int b=0;b<4;++b){
              const double gB=dot(mesh.cells[(size_t)L].grad[(size_t)b],f.normal);
              A.add(dof(L,a),dof(L,b),w*(-muL*gB*p[(size_t)a]-muL*gA*p[(size_t)b]+eta*p[(size_t)a]*p[(size_t)b]));
            }
          }
        }
        if(un<0 && (bc.kind==BoundaryKind::Inlet||backflow)){
          const double target=positive_target(bc,x),weight=-un*w;
          for(int a=0;a<4;++a){
            A.add_rhs(dof(L,a),weight*target*p[(size_t)a]);
            for(int b=0;b<4;++b)A.add(dof(L,a),dof(L,b),weight*p[(size_t)a]*p[(size_t)b]);
          }
        }
      }
    }
  }
  return A;
}
// Zhang-Shu-style uniform scaling of the 4 Bernstein coefficients to a lower
// bound. Preserve cell mean (mean controls for affine simplex), fail explicitly
// when the mean is below the requested floor.
inline bool preserve_mean_floor(std::array<double,4>&w,double floor){
  const double m=(w[0]+w[1]+w[2]+w[3])/4.;
  if(!std::isfinite(m)||m<floor)return false;
  const double lo=*std::min_element(w.begin(),w.end());
  if(lo>=floor)return true;
  const double theta=std::max(0.,std::min(1.,(m-floor)/(m-lo)));
  for(double &v:w)v=m+theta*(v-m);
  return true;
}
inline double l2(const std::vector<double>&x){double v=0;for(double a:x)v+=a*a;return std::sqrt(v);}
// Dense 4x4 solves for exact element-block Jacobi, with pivoting. Reference only.
inline std::array<double,4> block_solve(std::array<std::array<double,4>,4> m,std::array<double,4> b){
  for(int k=0;k<4;++k){
    int p=k;for(int j=k+1;j<4;++j)if(std::abs(m[(size_t)j][(size_t)k])>std::abs(m[(size_t)p][(size_t)k]))p=j;
    if(!(std::abs(m[(size_t)p][(size_t)k])>1e-35))throw std::runtime_error("DG omega: singular 4x4 element block");
    std::swap(m[(size_t)p],m[(size_t)k]);std::swap(b[(size_t)p],b[(size_t)k]);
    const double d=m[(size_t)k][(size_t)k];
    for(int j=k;j<4;++j)m[(size_t)k][(size_t)j]/=d;
    b[(size_t)k]/=d;
    for(int i=0;i<4;++i)if(i!=k){
      const double s=m[(size_t)i][(size_t)k];
      for(int j=k;j<4;++j)m[(size_t)i][(size_t)j]-=s*m[(size_t)k][(size_t)j];
      b[(size_t)i]-=s*b[(size_t)k];
    }
  }
  return b;
}
inline std::vector<double> block_jacobi(const Operator&A,const std::vector<double>&rhs){
  if(A.ndof%4||rhs.size()!=(size_t)A.ndof)throw std::runtime_error("DG omega: bad preconditioner size");
  std::vector<double> y(rhs.size());
  for(int t=0;t<A.ndof/4;++t){
    std::array<std::array<double,4>,4> B{};std::array<double,4> b{};
    for(int a=0;a<4;++a){
      const int row=dof(t,a);b[(size_t)a]=rhs[(size_t)row];
      for(int j=0;j<4;++j){auto it=A.rows[(size_t)row].find(dof(t,j));if(it!=A.rows[(size_t)row].end())B[(size_t)a][(size_t)j]=it->second;}
    }
    auto x=block_solve(B,b);for(int a=0;a<4;++a)y[(size_t)dof(t,a)]=x[(size_t)a];
  }
  return y;
}
struct LinearResult {int iterations=0;double initial=0,final=0;bool converged=false;};
// Right-preconditioned flexible GMRES reference; true residual verified on exit.
// CUDA G2 must use its own GPU FGMRES/4x4 block-preconditioner implementation.
inline LinearResult fgmres(const Operator&A,std::vector<double>&x,int restart=20,int maxIts=200,double rtol=1.e-8){
  if((int)x.size()!=A.ndof||restart<1||maxIts<1||!(rtol>0))throw std::runtime_error("DG omega: bad FGMRES options");
  auto residual=[&](){auto ax=A.multiply(x);std::vector<double> r((size_t)A.ndof);for(int i=0;i<A.ndof;++i)r[(size_t)i]=A.rhs[(size_t)i]-ax[(size_t)i];return r;};
  std::vector<double> r=residual();LinearResult info;info.initial=l2(r);
  const double target=std::max(info.initial*rtol,1e-14);info.final=info.initial;
  if(info.final<=target){info.converged=true;return info;}
  while(info.iterations<maxIts){
    int m=std::min(restart,maxIts-info.iterations);
    std::vector<std::vector<double>> V,Z;
    std::vector<std::vector<double>> H((size_t)m+1,std::vector<double>((size_t)m,0.));
    std::vector<double> cs((size_t)m),sn((size_t)m),g((size_t)m+1,0.);
    const double beta=l2(r);g[0]=beta;
    for(double &v:r)v/=beta;
    V.push_back(r);
    int completed=0;
    for(int j=0;j<m;++j){
      auto z=block_jacobi(A,V[(size_t)j]);Z.push_back(z);
      auto w=A.multiply(z);
      for(int i=0;i<=j;++i){
        const double hij=std::inner_product(w.begin(),w.end(),V[(size_t)i].begin(),0.);
        H[(size_t)i][(size_t)j]=hij;
        for(int q=0;q<A.ndof;++q)w[(size_t)q]-=hij*V[(size_t)i][(size_t)q];
      }
      const double wn=l2(w);H[(size_t)j+1][(size_t)j]=wn;
      if(wn>0)for(double &v:w)v/=wn;
      V.push_back(w);
      for(int i=0;i<j;++i){
        const double h0=H[(size_t)i][(size_t)j],h1=H[(size_t)i+1][(size_t)j];
        H[(size_t)i][(size_t)j]=cs[(size_t)i]*h0+sn[(size_t)i]*h1;
        H[(size_t)i+1][(size_t)j]=-sn[(size_t)i]*h0+cs[(size_t)i]*h1;
      }
      const double a=H[(size_t)j][(size_t)j],b=H[(size_t)j+1][(size_t)j];
      const double den=std::hypot(a,b);
      cs[(size_t)j]=den? a/den:1.;sn[(size_t)j]=den? b/den:0.;
      H[(size_t)j][(size_t)j]=den;H[(size_t)j+1][(size_t)j]=0.;
      g[(size_t)j+1]=-sn[(size_t)j]*g[(size_t)j];g[(size_t)j]=cs[(size_t)j]*g[(size_t)j];
      completed=j+1;++info.iterations;
      if(std::abs(g[(size_t)j+1])<=target||wn==0)break;
    }
    std::vector<double> y((size_t)completed,0.);
    for(int i=completed-1;i>=0;--i){
      double b=g[(size_t)i];for(int j=i+1;j<completed;++j)b-=H[(size_t)i][(size_t)j]*y[(size_t)j];
      if(!(std::abs(H[(size_t)i][(size_t)i])>1e-35))throw std::runtime_error("DG omega: zero GMRES Hessenberg pivot");
      y[(size_t)i]=b/H[(size_t)i][(size_t)i];
    }
    for(int i=0;i<completed;++i)for(int q=0;q<A.ndof;++q)x[(size_t)q]+=y[(size_t)i]*Z[(size_t)i][(size_t)q];
    r=residual();info.final=l2(r);
    if(info.final<=target){info.converged=true;break;}
  }
  return info;
}
}} // namespace nodals::omega_dg