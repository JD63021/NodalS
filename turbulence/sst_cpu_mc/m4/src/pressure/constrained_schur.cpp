#include "constrained_schur.hpp"
#include "pressure_amg.hpp"
#include <mpi.h>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>
#include <limits>
#include <cstdio>
namespace nodals::cpu_mc::m3 {
namespace {
inline double comp(Vec3 a,int i){return i==0?a.x:(i==1?a.y:a.z);}
inline double& component(Vec3&v,int i){return i==0?v.x:(i==1?v.y:v.z);}
inline double global_dot(const std::vector<double>&x,const std::vector<double>&y,int rank,const Ownership&own){
 double l=0;for(std::size_t c=0;c<x.size();++c)if(own.cell[c]==rank)l+=x[c]*y[c];
 double g=0;MPI_Allreduce(&l,&g,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);return g;
}
}
Schur::~Schur()=default;
Schur::Schur(const Mesh&m,const Ownership&o,const BF3Flux&w,int rank,std::vector<double> inverse,
    std::array<double,3> inletVelocity,const nodals::cpu_mc::m4::EnrichmentSpace*enrich)
:mesh_(m),own_(o),wall_(w),enrich_(enrich),rank_(rank),inv_(std::move(inverse)){
 const int nv=(int)m.points.size(),nc=(int)m.tets.size();
 const int ne=nv+(int)m.faces.size();
 const int neFull=(enrich_&&enrich_->enabled)?enrich_->nent():ne;
 if((int)inv_.size()!=3*neFull)throw std::runtime_error("Schur: inconsistent augmented rAU length");
 rows_.resize(nc);source_.assign(nc,0.0);diag_.assign(nc,0.0);
 // Exactly the GPU boundary construction: subtract integral_F phi_a n on
 // the inlet from Bvolume, and add the inlet flux to the source vector.
 // For an affine triangle: integral phi_vertex = Area/3 and
 // integral (27*l1*l2*l3) = (9/20)*Area.
 for(int c=0;c<nc;++c){
  const auto&t=m.tets[c];const auto raw=divergence_coefficients(m,c);
  Row& row=rows_[c];
  for(int a=0;a<8;++a){row.entity[a]=a<4?t.v[a]:nv+t.face[a-4];row.physical[a]=raw.b[a];}
  for(int a=8;a<12;++a)row.entity[a]=(enrich_&&enrich_->enabled)?enrich_->entity(c,a):-1;
  if(enrich_&&enrich_->enabled){
   std::array<Vec3,4> b{};enrich_->divergence(c,b);
   for(int a=0;a<4;++a)row.physical[8+a]=b[a];
  }
  for(int lf=0;lf<4;++lf){
   const int f=t.face[lf];if(m.faces[f].patch!=1)continue;
   const Vec3 sf=outward_area(m,c,lf);
   const double a=std::sqrt(dot(sf,sf));
   if(!(a>0))throw std::runtime_error("Schur: degenerate inlet area");
   for(int i=0;i<4;++i)if(i!=lf)row.physical[i]=row.physical[i]-(1.0/3.0)*sf;
   row.physical[4+lf]=row.physical[4+lf]-(9.0/20.0)*sf;
   // The original GPU uses the prescribed DG inlet flux, not a conforming
   // velocity Dirichlet condition at inlet-wall intersection vertices.
   if(enrich_&&enrich_->enabled){
    std::array<Vec3,4> be{};enrich_->face_flux(c,lf,be);
    for(int a=0;a<4;++a)row.physical[8+a]=row.physical[8+a]-be[a];
   }
   source_[c]+=dot(sf,{inletVelocity[0],inletVelocity[1],inletVelocity[2]});
  }
  row.reduced=row.physical;
  for(int lf=0;lf<4;++lf){
   const int f=t.face[lf];if(m.faces[f].patch!=3)continue;
   const Vec3 sf=outward_area(m,c,lf);
   const double a=std::sqrt(dot(sf,sf));
   if(!(a>0))throw std::runtime_error("Schur: degenerate wall area");
   const Vec3 n=(1.0/a)*sf;
   const double bn=dot(row.reduced[4+lf],n);
   row.reduced[4+lf]=row.reduced[4+lf]-bn*n;
   for(int j=0;j<4;++j)if(j!=lf)
     row.reduced[j]=row.reduced[j]-(20.0/27.0)*bn*n;
  }
  if(enrich_&&enrich_->enabled)for(int a=0;a<12;++a)
   if(row.entity[a]>=0 && enrich_->is_fixed(row.entity[a]))
    row.physical[a]=row.reduced[a]=Vec3{};
 }
 // The sparse pressure matrix is assembled from exactly B_reduced and the
 // frozen momentum inverse, with no pressure-diagonal approximation.
 using Pair=std::pair<int,double>;
 std::vector<std::vector<Pair>> inc((std::size_t)3*neFull);
 for(int c=0;c<nc;++c)for(int a=0;a<12;++a)for(int d=0;d<3;++d){
   if(rows_[c].entity[a]<0)continue;
   const double b=comp(rows_[c].reduced[a],d);
   if(std::abs(b)>1e-30)inc[(std::size_t)3*rows_[c].entity[a]+d].emplace_back(c,b);
 }
 const int ownCount=o.countCell[rank];ownedRows_.reserve((std::size_t)ownCount);
 rowPtr_.reserve((std::size_t)ownCount+1);rowPtr_.push_back(0);
 for(int c=0;c<nc;++c)if(o.cell[c]==rank){
  ownedRows_.push_back(c);
  std::unordered_map<int,double> tmp;
  for(int a=0;a<12;++a)for(int d=0;d<3;++d){
    if(rows_[c].entity[a]<0)continue;
    const int j=3*rows_[c].entity[a]+d;
    const double v=comp(rows_[c].reduced[a],d),inv=inv_[j];
    if(std::abs(v)<1e-30 || !(inv>0.0))continue;
    for(const auto& [col,bc]:inc[j])tmp[col]+=v*inv*bc;
  }
  std::vector<Pair> sorted;sorted.reserve(tmp.size());
  for(auto& kv:tmp){
   if(std::abs(kv.second)>1e-38)sorted.push_back(kv);}

  std::sort(sorted.begin(),sorted.end(),[](const Pair&a,const Pair&b){return a.first<b.first;});
  double diag=0.0;
  for(const auto&[j,v]:sorted){col_.push_back(j);val_.push_back(v);if(j==c)diag=v;}
  if(!(diag>1e-40) || !std::isfinite(diag))throw std::runtime_error("Schur: nonpositive exact Schur diagonal in cell "+std::to_string(c));
  diag_[c]=diag;rowPtr_.push_back((int)col_.size());
 }
 MPI_Allreduce(MPI_IN_PLACE,diag_.data(),nc,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
 // No AMG approximation is used in the fine operator: the aggregation
 // hierarchy is a preconditioner for the exact sparse Galerkin Schur matrix.
 amg_=std::make_unique<PressureAMG>(nc,ownedRows_,rowPtr_,col_,val_,diag_,o.cell,rank);
}

// Input is physical velocity u=T*x.
void Schur::B(const std::vector<double>&u,std::vector<double>&p)const{
 const int nc=(int)rows_.size();p.assign(nc,0.0);
 for(int c=0;c<nc;++c)if(own_.cell[c]==rank_){
  double s=0.0;const Row&r=rows_[c];
  for(int a=0;a<12;++a){if(r.entity[a]<0)continue;int e=3*r.entity[a];
   for(int d=0;d<3;++d)s+=comp(r.physical[a],d)*u[e+d];}
  p[c]=s;
 }
 MPI_Allreduce(MPI_IN_PLACE,p.data(),nc,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
}
void Schur::BT(const std::vector<double>&p,std::vector<double>&v)const{
 v.assign(inv_.size(),0.0);
 for(int c=0;c<(int)rows_.size();++c)if(own_.cell[c]==rank_){
  const Row&r=rows_[c];for(int a=0;a<12;++a){if(r.entity[a]<0)continue;const int e=3*r.entity[a];
   for(int d=0;d<3;++d)v[e+d]+=comp(r.physical[a],d)*p[c];}
 }
 MPI_Allreduce(MPI_IN_PLACE,v.data(),(int)v.size(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
}
void Schur::continuity(const std::vector<double>&u,std::vector<double>&p)const{
 B(u,p);
 for(std::size_t c=0;c<p.size();++c)p[c]+=source_[c];
}
void Schur::C(const std::vector<double>&raw,std::vector<double>&out)const{
 out=raw;wall_.Tt(out);
 for(std::size_t i=0;i<out.size();++i)out[i]*=inv_[i];
 wall_.Q(out);wall_.T(out);
}
void Schur::apply(const std::vector<double>&p,std::vector<double>&out)const{
 out.assign(rows_.size(),0.0);
 for(std::size_t i=0;i<ownedRows_.size();++i){const int c=ownedRows_[i];double sum=0.0;
  for(int k=rowPtr_[i];k<rowPtr_[i+1];++k)sum+=val_[k]*p[col_[k]];
  out[c]=sum;
 }
 MPI_Allreduce(MPI_IN_PLACE,out.data(),(int)out.size(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
}
Schur::Result Schur::solve(const std::vector<double>&b,std::vector<double>&x,double rtol,double atol,int limit)const{
 Result ans;const int n=(int)b.size();x.assign(n,0.0);
 std::vector<double> rhs=b;
 std::vector<double> res=rhs,z(n),dir(n),a;
 amg_->apply(res,z);dir=z;
 double rz=global_dot(res,z,rank_,own_),norm=std::sqrt(global_dot(res,res,rank_,own_));
 ans.r0=norm;ans.rFinal=norm;
 const double threshold=std::max(atol,rtol*norm);
 if(norm<=threshold){ans.converged=true;return ans;}
 for(int it=0;it<limit;++it){
  apply(dir,a);
  const double den=global_dot(dir,a,rank_,own_);
  if(!(den>0.0)||!std::isfinite(den))throw std::runtime_error("Schur PCG lost SPD");
  const double alpha=rz/den;
  for(int c=0;c<n;++c){x[c]+=alpha*dir[c];res[c]-=alpha*a[c];}
  ans.iterations=it+1;
  ans.rFinal=std::sqrt(global_dot(res,res,rank_,own_));
  if(!std::isfinite(ans.rFinal))throw std::runtime_error("nonfinite pressure PCG residual");
  if(ans.rFinal<=threshold){ans.converged=true;break;}
  amg_->apply(res,z);
  const double newRz=global_dot(res,z,rank_,own_);
  if(!(newRz>0.0)||!std::isfinite(newRz))throw std::runtime_error("AMG preconditioner lost SPD");
  const double beta=newRz/rz;rz=newRz;
  for(int c=0;c<n;++c)dir[c]=z[c]+beta*dir[c];
 }
 return ans;
}
double Schur::source_sum()const{
 double s=0.0;for(double v:source_)s+=v;return s;
}
}
