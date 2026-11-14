#include "sst_transport.hpp"
#include "quadrature24.hpp"
#include "sst2003m_distance_core.hpp" // exact GPU SST-2003m pointwise mathematics
#include "transport_core.hpp"          // exact GPU row-L1 and scalar source policies
#include "of_auto_core.hpp"            // exact GPU OF-auto blended wall state
#include "tet_ray_trace.hpp"           // exact GPU connected tetrahedral ray traversal
#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
namespace nodals::cpu_mc::m4 {
namespace {
using P=std::array<double,3>;
using Vec8=std::array<Vec3,8>;
struct Geo{double det=0;std::array<Vec3,4> dlam{};};
Geo geometry(const Mesh&m,int c){
 const auto&t=m.tets[c];auto a=m.points[t.v[0]],u=m.points[t.v[1]]-a,v=m.points[t.v[2]]-a,w=m.points[t.v[3]]-a;
 double det=dot(u,cross(v,w));if(!(det>0))throw std::runtime_error("invalid tet orientation in SST");
 Geo g;g.det=det;g.dlam[1]=(1.0/det)*cross(v,w);g.dlam[2]=(1.0/det)*cross(w,u);g.dlam[3]=(1.0/det)*cross(u,v);g.dlam[0]=-1.0*(g.dlam[1]+g.dlam[2]+g.dlam[3]);return g;
}
void shape(const Geo&g,const std::array<double,4>&l,double phi[8],Vec8&gr){
 for(int a=0;a<4;++a){phi[a]=l[a];gr[a]=g.dlam[a];}
 for(int a=0;a<4;++a){int j[3],q=0;for(int i=0;i<4;++i)if(i!=a)j[q++]=i;
  phi[4+a]=27*l[j[0]]*l[j[1]]*l[j[2]];gr[4+a]={};
  for(int i=0;i<3;++i)gr[4+a]=gr[4+a]+(27*l[j[(i+1)%3]]*l[j[(i+2)%3]])*g.dlam[j[i]];
 }
}
std::array<double,4> q24(int q){return {quad::l0[q],quad::l1[q],quad::l2[q],quad::l3[q]};}
constexpr double L0[12]={.063089014491502,.063089014491502,.873821971016996,.249286745170910,.249286745170910,.501426509658180,.053145049844816,.053145049844816,.310352451033785,.310352451033785,.636502499121399,.636502499121399};
constexpr double L1[12]={.063089014491502,.873821971016996,.063089014491502,.249286745170910,.501426509658180,.249286745170910,.310352451033785,.636502499121399,.053145049844816,.636502499121399,.053145049844816,.310352451033785};
constexpr double L2[12]={.873821971016996,.063089014491502,.063089014491502,.501426509658180,.249286745170910,.249286745170910,.636502499121399,.310352451033785,.636502499121399,.053145049844816,.310352451033785,.053145049844816};
constexpr double QW[12]={.050844906370207,.050844906370207,.050844906370207,.116786275726379,.116786275726379,.116786275726379,.082851075618374,.082851075618374,.082851075618374,.082851075618374,.082851075618374,.082851075618374};
struct Carrier {double gradU[9]{},gk[3]{},gw[3]{},k=0,omega=0,d=0;Vec3 vel{};};
Carrier carrier(const Mesh&m,int c,const std::array<double,4>&l,
              const Vec8&gp,const double phi[8],const std::vector<double>&u,
              const std::vector<double>&k,const std::vector<double>&w,
              const std::vector<double>&distance,const EnrichmentSpace*enrich){
 Carrier a;const auto&t=m.tets[c];const int nv=(int)m.points.size();
 for(int i=0;i<4;++i)a.d+=l[i]*distance[t.v[i]];
 for(int b=0;b<8;++b){const int ent=b<4?t.v[b]:nv+t.face[b-4];
  const double kb=b<4?k[t.v[b]]:0.0,wb=b<4?w[t.v[b]]:0.0;
  a.k+=phi[b]*kb;a.omega+=phi[b]*wb;
  const double ub[3]={u[3*ent],u[3*ent+1],u[3*ent+2]};
  for(int i=0;i<3;++i){
   const double gb[3]={gp[b].x,gp[b].y,gp[b].z};
   for(int d=0;d<3;++d)a.gradU[3*i+d]+=ub[i]*gb[d];
  }
  a.vel.x+=phi[b]*ub[0];a.vel.y+=phi[b]*ub[1];a.vel.z+=phi[b]*ub[2];
  a.gk[0]+=kb*gp[b].x;a.gk[1]+=kb*gp[b].y;a.gk[2]+=kb*gp[b].z;
  a.gw[0]+=wb*gp[b].x;a.gw[1]+=wb*gp[b].y;a.gw[2]+=wb*gp[b].z;
 }
 if(enrich&&enrich->enabled){
  double ephi[12];Vec3 eg[12];enrich->shape(c,l,ephi,eg);
  for(int b=8;b<12;++b){const int id=enrich->entity(c,b);if(id<0)continue;
    const double coeff[3]={u[3*(size_t)id],u[3*(size_t)id+1],u[3*(size_t)id+2]};
    const double gg[3]={eg[b].x,eg[b].y,eg[b].z};
    for(int i=0;i<3;++i)for(int d=0;d<3;++d)a.gradU[3*i+d]+=coeff[i]*gg[d];
    a.vel.x+=ephi[b]*coeff[0];a.vel.y+=ephi[b]*coeff[1];a.vel.z+=ephi[b]*coeff[2];
  }
 }
 return a;
}
using SST=nodals::sst2003m::LocalState<double>;
SST physics(const Carrier&a,const CaseConfig&cfg){
 return nodals::sst2003m::evaluate_at_distance<double>(a.d,cfg.diameter/2,cfg.nu,a.gradU,a.k,a.omega,a.gk,a.gw,cfg.kFloor,cfg.omegaFloor,1);
}
void add_update(std::vector<double>&field,const std::vector<double>&old,const Ownership&o,int rank){
 std::vector<double> difference(field.size(),0.0);
 for(int i=0;i<(int)field.size();++i)if(o.vertex[i]==rank)difference[i]=field[i]-old[i];
 MPI_Allreduce(MPI_IN_PLACE,difference.data(),(int)difference.size(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
 for(int i=0;i<(int)field.size();++i)field[i]=old[i]+difference[i];
}
// 125-point exact reference tetrahedron convection tensor, built ONCE on each CPU process.
// Indices a=active vertex test, m=8 carrier modes, b=8 trial modes, d=3.
struct ConvectionTensor{double val[4][8][8][3]{};};
const ConvectionTensor& conv_tensor(){
 static const ConvectionTensor t=[](){ConvectionTensor t{};
  constexpr double X[5]={.0469100770306680185,.23076534494715845,.5,.7692346550528415,.95308992296933193};
  constexpr double W[5]={.11846344252809449,.23931433524968326,.28444444444444450,.23931433524968326,.11846344252809449};
  Geo ref;ref.det=1;ref.dlam={Vec3{-1,-1,-1},Vec3{1,0,0},Vec3{0,1,0},Vec3{0,0,1}};
  for(int ir=0;ir<5;++ir)for(int is=0;is<5;++is)for(int it=0;it<5;++it){
   double r=X[ir],s=X[is],z=X[it],omr=1-r,oms=1-s;
   std::array<double,4> l{omr*oms*(1-z),r,omr*s,omr*oms*z};
   double ph[8];Vec8 gr;shape(ref,l,ph,gr);
   double wt=W[ir]*W[is]*W[it]*omr*omr*oms;
   for(int a=0;a<4;++a)for(int m=0;m<8;++m)for(int b=0;b<8;++b){
    t.val[a][m][b][0]+=wt*ph[a]*ph[m]*gr[b].x;
    t.val[a][m][b][1]+=wt*ph[a]*ph[m]*gr[b].y;
    t.val[a][m][b][2]+=wt*ph[a]*ph[m]*gr[b].z;
   }
  }return t;
 }();return t;
}
} // namespace

double sst_nut_at_point(const Mesh&m,int c,const std::array<double,4>&l,
 const std::vector<double>&u,const std::vector<double>&k,const std::vector<double>&omega,
 const std::vector<double>&distance,const CaseConfig&cfg,const EnrichmentSpace*enrich){
 const auto geom=geometry(m,c);double ph[8];Vec8 gp{};shape(geom,l,ph,gp);
 return physics(carrier(m,c,l,gp,ph,u,k,omega,distance,enrich),cfg).nut;
}
void momentum_turbulent_deviatoric(Element24&K,const Mesh&m,int c,const std::vector<double>&u,
 const std::vector<double>&k,const std::vector<double>&omega,
 const std::vector<double>&distance,const CaseConfig&cfg,const EnrichmentSpace*enrich){
 const auto geom=geometry(m,c);
 for(int q=0;q<24;++q){const auto l=q24(q);double ph[8];Vec8 gp{};shape(geom,l,ph,gp);
  const auto state=carrier(m,c,l,gp,ph,u,k,omega,distance,enrich);
  const auto ss=physics(state,cfg);const double nuT=ss.nut;
  if(!std::isfinite(nuT)||nuT<0)throw std::runtime_error("nonfinite turbulent viscosity in momentum");
  const double weight=geom.det*quad::w[q]*nuT;
  for(int a=0;a<8;++a)for(int b=0;b<8;++b){
   const double ga[3]={gp[a].x,gp[a].y,gp[a].z},gb[3]={gp[b].x,gp[b].y,gp[b].z};
   const double ab=dot(gp[a],gp[b]);
   for(int i=0;i<3;++i)for(int j=0;j<3;++j)
    K[3*a+i][3*b+j]+=weight*((i==j?ab:0)+ga[j]*gb[i]-(2.0/3.0)*ga[i]*gb[j]);
  }
 }
}
SSTTransport::SSTTransport(const Mesh&m,const Ownership&own,const CaseConfig&cfg,int rank,
 const std::vector<double>&wallDistance,const EnrichmentSpace*enrich)
:m_(m),owners_(own),cfg_(cfg),rank_(rank),nv_((int)m.points.size()),
 ne_((int)m.points.size()+(int)m.faces.size()),enrich_(enrich),distance_(wallDistance){
 if(distance_.size()!=m.points.size())throw std::runtime_error("SST: wall distance size mismatch");
 if(cfg.sstSpace!="q1"||cfg.compactP1)throw std::runtime_error("M4 requires original masked Q1 with BF3 enrichment in matrix");
 if(cfg.omegaMethod!="of_auto"||
    (cfg.omegaStage41!="trace_log"&&cfg.omegaStage41!="of_auto"&&cfg.omegaStage41!="sample_log")||
    cfg.omegaIndex!=3)
  throw std::runtime_error("M4 requires captured of_auto/trace_log sample point 3; refusing wall model replacement");
 if(cfg.supgK||cfg.supgOmega)throw std::runtime_error("M4 scalar SUPG implementation is not yet available");
 bulk_=std::sqrt(cfg.inlet[0]*cfg.inlet[0]+cfg.inlet[1]*cfg.inlet[1]+cfg.inlet[2]*cfg.inlet[2]);
 state_.kIn=1.5*std::pow(cfg.intensity*bulk_,2);
 state_.omegaIn=std::sqrt(state_.kIn)/(std::pow(0.09,0.25)*cfg.lengthRatio*cfg.diameter);
 if(!(state_.kIn>cfg.kFloor&&state_.omegaIn>cfg.omegaFloor))throw std::runtime_error("SST initial inlet k/omega under floors");
 state_.k.assign((size_t)nv_,state_.kIn);
 state_.omega.assign((size_t)nv_,state_.omegaIn);
 // One-time GPU portable connected ray tracing of the omega sample geometry.
 using nodals::wall::ray::TetRayTrace;
 std::vector<P> pts;pts.reserve(m.points.size());for(const auto&p:m.points)pts.push_back({p.x,p.y,p.z});
 std::vector<std::array<int,4>> tets;tets.reserve(m.tets.size());for(const auto&t:m.tets)tets.push_back(t.v);
 TetRayTrace trace(std::move(pts),std::move(tets));
 const double y=cfg.omegaY3;
 for(int f=0;f<(int)m.faces.size();++f)if(m.faces[f].patch==3&&own.cell[m.faces[f].cell0]==rank){
  const int c=m.faces[f].cell0;const auto&t=m.tets[c];int lf=-1;
  for(int a=0;a<4;++a)if(t.face[a]==f)lf=a;
  if(lf<0)throw std::runtime_error("SST omega wall face topology invalid");
  int j[3],idx=0;for(int a=0;a<4;++a)if(a!=lf)j[idx++]=a;
  const Vec3 af=outward_area(m,c,lf);const double ar=std::sqrt(dot(af,af));
  if(!(ar>0))throw std::runtime_error("SST omega wall face degenerate");
  P inward={-af.x/ar,-af.y/ar,-af.z/ar};
  for(int q=0;q<12;++q){
   Vec3 orig=L0[q]*m.points[t.v[j[0]]]+L1[q]*m.points[t.v[j[1]]]+L2[q]*m.points[t.v[j[2]]];
   const P origin{orig.x,orig.y,orig.z};
   auto hit=trace.trace(c,origin,inward,{y},cfg.maxCrossings,cfg.baryTolerance).front();
   wallPoints_.push_back({c,f,lf,q,hit.cell,hit.bary,y});
  }
 }
 long long local=(long long)wallPoints_.size(),total=0;
 MPI_Allreduce(&local,&total,1,MPI_LONG_LONG_INT,MPI_SUM,MPI_COMM_WORLD);
 if(rank==0)std::printf("NODALS_CPU_SST_M4_WALL_CACHE mode=%s wallQuadraturePoints=%lld omegaScale=%.8g sampleY=%.12e gamma=%.8g penaltyLength=%.8g source=GPU_TET_RAY_TRACE\n",
    cfg.omegaStage41=="trace_log"?"TRACE_LOG_NITSCHE":"OFFWALL_RANK_ONE",
    total,cfg.omegaScale,y,
    cfg.omegaStage41=="trace_log"?cfg.omegaTraceGamma:cfg.omegaPointGamma,
    cfg.omegaPenaltyLength);
}

// The scalar matrices are assembled from the original enriched P1+BF3 FEM
// test/trial basis; only 4 vertex rows participate in the Q1 linear solve.
// BF3 entries retain their contribution to the row-L1 fixed-point damping.
ScalarReport SSTTransport::advance_k(const std::vector<double>&velocity){
 return advance_impl(false,velocity,nullptr);
}
ScalarReport SSTTransport::advance_omega(const std::vector<double>&velocity,const nodals::cpu_mc::m3::WallFaceFit&fit){
 ++omegaAdvanceCount_;
 return advance_impl(true,velocity,&fit);
}
ScalarReport SSTTransport::advance_impl(bool omegaEq,const std::vector<double>&u,const nodals::cpu_mc::m3::WallFaceFit*fit){
 if((int)u.size()!=3*((enrich_&&enrich_->enabled)?enrich_->nent():ne_))
   throw std::runtime_error("SST scalar enriched velocity size mismatch");
 const int N=nv_;
 using Row=std::unordered_map<int,double>;
 std::vector<int> owned,localOf(N,-1);owned.reserve((size_t)N/16+20);
 for(int i=0;i<N;++i)if(owners_.vertex[i]==rank_){localOf[i]=(int)owned.size();owned.push_back(i);}
 std::vector<Row> mat(owned.size());std::vector<double> rhs(owned.size(),0.0);
 const std::vector<double>&k=state_.k,&omega=state_.omega;
 const double qIn=omegaEq?state_.omegaIn:state_.kIn;
 const auto&ct=conv_tensor();
 for(int c=0;c<(int)m_.tets.size();++c){
  const auto&t=m_.tets[c];bool any=false;
  for(int a=0;a<4;++a)if(localOf[t.v[a]]>=0)any=true;
  if(!any)continue;
  const Geo g=geometry(m_,c);
  int ent[8];for(int a=0;a<8;++a)ent[a]=a<4?t.v[a]:nv_+t.face[a-4];
  double M[4][8]{},B[4]{};
  // Molecular and SST turbulent diffusion/reaction/source at the 24 GPU points.
  for(int q=0;q<24;++q){auto l=q24(q);double ph[8];Vec8 gp{};shape(g,l,ph,gp);
   const Carrier a=carrier(m_,c,l,gp,ph,u,k,omega,distance_,enrich_);
   const auto ss=physics(a,cfg_);
   const double extra=(omegaEq?ss.sigmaW:ss.sigmaK)*ss.nut;
   const double diffusion=cfg_.nu+extra;
   const double reaction=omegaEq?ss.reactOmega:ss.reactK;
   const double source=omegaEq?ss.srcOmega:ss.pk;
   const double wt=g.det*quad::w[q];
   if(!(diffusion>0&&reaction>=0&&std::isfinite(source)))throw std::runtime_error("non-finite SST volume coefficients");
   for(int ia=0;ia<4;++ia){
    B[ia]+=wt*ph[ia]*source;
    for(int ib=0;ib<8;++ib)M[ia][ib]+=wt*(diffusion*dot(gp[ia],gp[ib])+reaction*ph[ia]*ph[ib]);
   }
  }
  // Original 5x5x5 Duffy-exact scalar advective convection (not an approximate 24pt rule).
  double uc[8][3]{};
  for(int m=0;m<8;++m){int e=ent[m];for(int d=0;d<3;++d)uc[m][d]=u[3*e+d];}
  for(int a=0;a<4;++a)for(int b=0;b<8;++b){double v=0;
   for(int mm=0;mm<8;++mm){
    // Reference gradient coordinate j maps to physical gradient via grad(lambda_{j+1}).
    for(int j=0;j<3;++j){
     const Vec3 gl=g.dlam[j+1];const double uref=dot({uc[mm][0],uc[mm][1],uc[mm][2]},gl);
     v+=uref*ct.val[a][mm][b][j];
    }
   }
   M[a][b]+=g.det*v;
  }
  // Existing pretabulated convection tensor covers only the P1+BF3 carrier.
  // Add the enriched velocity carrier against all masked P1 scalar rows.
  if(enrich_&&enrich_->enabled){
   constexpr double X[5]={.0469100770306680185,.23076534494715845,.5,.7692346550528415,.95308992296933193};
   constexpr double W[5]={.11846344252809449,.23931433524968326,.28444444444444450,.23931433524968326,.11846344252809449};
   for(int ir=0;ir<5;++ir)for(int is=0;is<5;++is)for(int iz=0;iz<5;++iz){
    const double a=X[ir],b=X[is],z=X[iz],oma=1-a,omb=1-b;
    std::array<double,4> l{oma*omb*(1-z),a,oma*b,oma*omb*z};
    double ep[12];Vec3 eg[12];enrich_->shape(c,l,ep,eg);
    Vec3 ve{};
    for(int j=8;j<12;++j){const int e=enrich_->entity(c,j);if(e<0)continue;
      const Vec3 ue{u[3*(size_t)e],u[3*(size_t)e+1],u[3*(size_t)e+2]};
      ve=ve+ep[j]*ue;
    }
    if(dot(ve,ve)==0.)continue;
    double ph[8];Vec8 gp{};shape(g,l,ph,gp);
    const double wt=g.det*W[ir]*W[is]*W[iz]*oma*oma*omb;
    for(int a2=0;a2<4;++a2)for(int b2=0;b2<8;++b2)
      M[a2][b2]+=wt*ph[a2]*dot(ve,gp[b2]);
   }
  }
  // Exact 12-point GPU DG inlet for scalar transport.
  for(int lf=0;lf<4;++lf){const auto&f=m_.faces[t.face[lf]];
   if(f.patch!=1)continue;
   Vec3 af=outward_area(m_,c,lf);double ar=std::sqrt(dot(af,af));Vec3 n=(1./ar)*af;
   const Vec3 inlet{cfg_.inlet[0],cfg_.inlet[1],cfg_.inlet[2]};
   const double qflux=std::max(0.0,-dot(inlet,n));
   int verts[3],j=0;for(int a=0;a<4;++a)if(a!=lf)verts[j++]=a;
   for(int q=0;q<12;++q){std::array<double,4> l{};l[verts[0]]=L0[q];l[verts[1]]=L1[q];l[verts[2]]=L2[q];
    double ph[8];Vec8 gp{};shape(g,l,ph,gp);
    const auto sc=physics(carrier(m_,c,l,gp,ph,u,k,omega,distance_,enrich_),cfg_);
    const double nuEff=cfg_.nu+(omegaEq?sc.sigmaW:sc.sigmaK)*sc.nut;
    const double wt=ar*QW[q];
    for(int a=0;a<4;++a){const double dn=dot(gp[a],n);
     B[a]+=wt*(qflux*ph[a]+nuEff*dn)*qIn;
     for(int b=0;b<8;++b)M[a][b]+=wt*(qflux*ph[a]*ph[b]+nuEff*(-ph[a]*dot(gp[b],n)+dn*ph[b]));
    }
   }
  }
  for(int a=0;a<4;++a){int ia=localOf[t.v[a]];if(ia<0)continue;
   rhs[ia]+=B[a];for(int b=0;b<8;++b)mat[ia][ent[b]]+=M[a][b];
  }
 }
 if(omegaEq && cfg_.omegaStage41=="trace_log"){
  if(!fit||fit->uTau.size()!=m_.faces.size())throw std::runtime_error("SST omega trace requires SAME Spalding wall state");
  for(const RayPoint&r:wallPoints_){
   const auto&t=m_.tets[r.wallCell];const Geo gw=geometry(m_,r.wallCell);
   auto&face=m_.faces[r.face];const Vec3 af=outward_area(m_,r.wallCell,r.localFace);
   const double area=std::sqrt(dot(af,af));const Vec3 n=(1.0/area)*af;
   const double h=3.*t.volume/area;
   const Geo gs=geometry(m_,r.hitCell);double phs[8];Vec8 gps{};shape(gs,r.bary,phs,gps);
   const Carrier a=carrier(m_,r.hitCell,r.bary,gps,phs,u,k,omega,distance_,enrich_);
   const auto ss=physics(a,cfg_);
   const double ut=fit->qUTau[(size_t)12*r.face+(size_t)r.q];const double target=ut/(std::sqrt(0.09)*0.4*r.y);
   // Pure logarithmic target, STAGE41_OMEGA_MODE=trace_log. GPU source uses
   // gamma=50 by default, NOT omega_wall.gamma=1e6 point penalty.
   const double nueff=cfg_.nu+ss.sigmaW*ss.nut;
   const double tau=cfg_.omegaTraceGamma*nueff/h;
   if(!(target>0&&std::isfinite(target)&&nueff>0&&h>0))throw std::runtime_error("SST trace-log invalid target or viscosity");
   int j[3],aidx=0;for(int a=0;a<4;++a)if(a!=r.localFace)j[aidx++]=a;
   std::array<double,4> l{};l[j[0]]=L0[r.q];l[j[1]]=L1[r.q];l[j[2]]=L2[r.q];
   double ph[8];Vec8 gp{};shape(gw,l,ph,gp);int ent[8];for(int a=0;a<8;++a)ent[a]=a<4?t.v[a]:nv_+t.face[a-4];
   const double wt=area*QW[r.q];
   for(int a=0;a<4;++a){int ia=localOf[t.v[a]];if(ia<0)continue;
    const double dna=dot(gp[a],n);
    rhs[ia]+=wt*(-nueff*dna+tau*ph[a])*target;
    for(int b=0;b<8;++b)mat[ia][ent[b]]+=wt*(-nueff*(dot(gp[b],n)*ph[a]+dna*ph[b])+tau*ph[a]*ph[b]);
   }
  }
 }
 // GPU stage41 off-wall omega point3: ONE rank-one penalty per wall-face
 // quadrature ray, assembled in the *interior hit cell*, NOT on wall nodes.
 // The masked BF3 scalar trial modes remain in the row L1 damping operator.
 if(omegaEq && cfg_.omegaStage41!="trace_log"){
  if(cfg_.expectRanks!=1)
    throw std::runtime_error("offwall omega point3 presently requires ONE MPI rank: hit-cell ownership not yet distributed");
  if(!fit||fit->qUTau.size()!=12*m_.faces.size())
    throw std::runtime_error("offwall omega requires Spalding friction velocity at all face quadrature rays");
  const double ell=cfg_.omegaPenaltyLength;
  const double gamma=cfg_.omegaPointGamma;
  double sumA=0,sumTarget=0,sumViscous=0,sumRey=0,smallestTarget=1e300,largestTarget=0;
  int nsamples=0;
  for(const RayPoint&r:wallPoints_){
   const auto&t=m_.tets[(size_t)r.hitCell];
   const Geo g=geometry(m_,r.hitCell);
   double ph[8];Vec8 gp{};shape(g,r.bary,ph,gp);
   const Carrier a=carrier(m_,r.hitCell,r.bary,gp,ph,u,k,omega,distance_,enrich_);
   const auto ss=physics(a,cfg_);
   const Vec3 areaVec=outward_area(m_,r.wallCell,r.localFace);
   const double area=std::sqrt(dot(areaVec,areaVec));
   if(!(area>0))throw std::runtime_error("offwall omega encountered degenerate wall face");
   const Vec3 n=(1.0/area)*areaVec;
   const double Un=dot(a.vel,n);
   const double Ut=std::sqrt(std::max(0.,dot(a.vel,a.vel)-Un*Un));
   const double ut=fit->qUTau[12*(size_t)r.face+(size_t)r.q];
   const auto oa=nodals::wall::evaluate_of_auto<double>(
       r.y,cfg_.nu,Ut,ut,a.k,a.omega,ss,
       cfg_.kFloor,cfg_.omegaFloor,0.4,9.025013499434122,
       0.075,cfg_.omegaOfReBlend,10.0,1);
   const double target=cfg_.omegaStage41=="sample_log"?oa.omegaLog:oa.omegaTarget;
   const double nuEff=cfg_.nu+ss.sigmaW*ss.nut;
   // GPU stage41 exact normalization:
   // coeff=gamma*(nuEff/ell)*(area*QW[q]*ell)/ell.
   const double coeff=gamma*nuEff*(area*QW[r.q])/ell;
   if(!(target>0 && nuEff>0 && coeff>0 && std::isfinite(target) &&
        std::isfinite(nuEff) && std::isfinite(coeff) && std::isfinite(oa.lamFrac)))
     throw std::runtime_error("invalid sampled omega target / effective viscosity / rank-one penalty");
   int ent[8];for(int b=0;b<8;++b)ent[b]=(b<4)?t.v[b]:(nv_+t.face[b-4]);
   for(int ia=0;ia<4;++ia){
    const int row=localOf[t.v[ia]];if(row<0)continue;
    rhs[(size_t)row]+=coeff*ph[ia]*target;
    for(int ib=0;ib<8;++ib)
      mat[(size_t)row][ent[ib]]+=coeff*ph[ia]*ph[ib];
   }
   const double w=area*QW[r.q];
   sumA+=w;sumTarget+=w*target;sumViscous+=w*oa.lamFrac;
   sumRey+=w*oa.rey;
   smallestTarget=std::min(smallestTarget,target);
   largestTarget=std::max(largestTarget,target);
   ++nsamples;
  }
  if(!(sumA>0))throw std::runtime_error("offwall omega received no wall rays");
  if(rank_==0 && (omegaAdvanceCount_==1||omegaAdvanceCount_%20==0))
   std::printf("NODALS_CPU_M4_OMEGA_POINT mode=%s gamma=%.8g penaltyLength=%.12e sampleY=%.12e samples=%d targetMean=%.12e targetMin=%.12e targetMax=%.12e viscousBlendMean=%.12e reyMean=%.12e status=PASS\n",
       cfg_.omegaStage41.c_str(),gamma,ell,cfg_.omegaY3,nsamples,
       sumTarget/sumA,smallestTarget,largestTarget,sumViscous/sumA,sumRey/sumA);
 }
 // Relaxed row-L1 sums include the constrained BF3 trial modes, exactly as in GPU Q1.
 std::vector<std::vector<std::pair<int,double>>> rows(owned.size());
 std::vector<double> inverse(owned.size(),0.0);
 std::vector<double>&current=omegaEq?state_.omega:state_.k;
 const std::vector<double>old=current;
 const double alpha=omegaEq?cfg_.urOmega:cfg_.urK;
 for(size_t i=0;i<owned.size();++i){
  int v=owned[i];auto&map=mat[i];
  double rowL1=0;for(const auto&entry:map)rowL1+=std::abs(entry.second);
  auto ip=map.find(v);if(ip==map.end())throw std::runtime_error("SST missing scalar matrix diagonal");
  auto relaxed=nodals::transport::row_l1_plan<double>(ip->second,rowL1,alpha);
  if(!(relaxed.diagonal>0)||!std::isfinite(relaxed.diagonal))throw std::runtime_error("SST relaxed diagonal not SPD-like");
  ip->second=relaxed.diagonal;
  rhs[i]+=relaxed.delta*old[v];
  inverse[i]=1.0/ip->second;
  rows[i].reserve(map.size());for(const auto&kv:map)rows[i].push_back(kv);
  std::sort(rows[i].begin(),rows[i].end());
 }
 const auto residual=[&](const std::vector<double>&x){double local=0,global=0;
  for(size_t i=0;i<owned.size();++i){double r=rhs[i];
   for(const auto&[j,v]:rows[i])r-=v*(j<nv_?x[j]:0.0);
   local+=r*r;
  }
  MPI_Allreduce(&local,&global,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  return std::sqrt(global);
 };
 ScalarReport report;report.before=residual(current);
 const int sweeps=omegaEq?cfg_.omegaSweeps:cfg_.kSweeps;
 for(int sweep=0;sweep<sweeps;++sweep){const auto previous=current;
  for(size_t i=0;i<owned.size();++i){double r=rhs[i];
   for(const auto&[j,v]:rows[i])r-=v*(j<nv_?current[j]:0.0);
   current[owned[i]]+=cfg_.scalarOmega*inverse[i]*r;
   if(!std::isfinite(current[owned[i]])){
    std::fprintf(stderr,"M4_DIAG_SCALAR_FAILURE rank=%d equation=%s sweep=%d vertex=%d rhs=%.12e diagonalInverse=%.12e residual=%.12e\n",
      rank_,omegaEq?"omega":"k",sweep,owned[i],rhs[i],inverse[i],r);
    throw std::runtime_error("nonfinite during masked scalar GS equation="+
      std::string(omegaEq?"omega":"k")+" sweep="+std::to_string(sweep)+
      " vertex="+std::to_string(owned[i]));
   }
  }
  add_update(current,previous,owners_,rank_);
 }
 report.after=residual(current);report.sweeps=sweeps;
 int localClips=0;
 for(int i:owned){double&value=current[i];
  const double floor=omegaEq?cfg_.omegaFloor:cfg_.kFloor;
  if(!std::isfinite(value))throw std::runtime_error("non-finite scalar after P1+BF3 masked GS");
  if(value<floor){value=floor;++localClips;}
 }
 std::vector<double> newOwned(current.size(),0.0);
 for(int i:owned)newOwned[i]=current[i];
 MPI_Allreduce(newOwned.data(),current.data(),nv_,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
 report.clips=localClips;int globalClips=0;
 MPI_Allreduce(&localClips,&globalClips,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);report.clips=globalClips;
 return report;
}

double SSTTransport::cell_nut(int c,const std::vector<double>&u)const{
 const Geo g=geometry(m_,c);double ph[8];Vec8 gr{};
 std::array<double,4> center{.25,.25,.25,.25};shape(g,center,ph,gr);
 return physics(carrier(m_,c,center,gr,ph,u,state_.k,state_.omega,distance_,enrich_),cfg_).nut;
}
void SSTTransport::refresh_wall_rays(){
 // Reuse the unchanged GPU geometric BVH distance; only re-trace the scalar wall points.
 SSTTransport refreshed(m_,owners_,cfg_,rank_,distance_,enrich_);
 wallPoints_.swap(refreshed.wallPoints_);
}
double SSTTransport::mean_nut(const std::vector<double>&u)const{
 double l=0,v=0;
 for(int c=0;c<(int)m_.tets.size();++c)if(owners_.cell[c]==rank_){
  const Geo g=geometry(m_,c);double ph[8];Vec8 gr{};std::array<double,4> center{.25,.25,.25,.25};shape(g,center,ph,gr);
  const auto ss=physics(carrier(m_,c,center,gr,ph,u,state_.k,state_.omega,distance_,enrich_),cfg_);
  l+=ss.nut*m_.tets[c].volume;v+=m_.tets[c].volume;
 }
 double a[2]={l,v},b[2]{};MPI_Allreduce(a,b,2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
 return b[0]/std::max(b[1],1e-300);
}
}
