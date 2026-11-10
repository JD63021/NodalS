#include "constrained_schur.hpp"
#include <mpi.h>
#include <cmath>
#include <stdexcept>
#include <algorithm>
namespace nodals::cpu_mc::m3 {
namespace {
double comp(Vec3 a,int i){return i==0?a.x:(i==1?a.y:a.z);}
double global_dot(const std::vector<double>&x,const std::vector<double>&y,int rank,const Ownership&own){
 double l=0;for(int c=0;c<(int)x.size();++c)if(own.cell[c]==rank)l+=x[c]*y[c];
 double g=0;MPI_Allreduce(&l,&g,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);return g;
}
}
Schur::Schur(const Mesh&m,const Ownership&o,const BF3Flux&w,int rank,std::vector<double> inverse):mesh_(m),own_(o),wall_(w),rank_(rank),inv_(std::move(inverse)){
 const int ne=(int)m.points.size()+(int)m.faces.size();if((int)inv_.size()!=3*ne)throw std::runtime_error("Schur bad inverse size");
 diag_.assign(m.tets.size(),0.0);
 for(int c=0;c<(int)m.tets.size();++c)if(o.cell[c]==rank_){const auto b=divergence_coefficients(m,c);const auto&t=m.tets[c];
  for(int a=0;a<8;++a){int e=a<4?t.v[a]:(int)m.points.size()+t.face[a-4];for(int d=0;d<3;++d){double ba=comp(b.b[a],d);diag_[c]+=ba*ba*inv_[3*e+d];}}
  if(diag_[c]<1e-40)throw std::runtime_error("zero pressure diagonal");
 }
 MPI_Allreduce(MPI_IN_PLACE,diag_.data(),(int)diag_.size(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
 diag_[0]=1.0;
}
void Schur::B(const std::vector<double>&v,std::vector<double>&p)const{
 p.assign(mesh_.tets.size(),0);
 const int nv=(int)mesh_.points.size();
 for(int c=0;c<(int)mesh_.tets.size();++c)if(own_.cell[c]==rank_){const auto b=divergence_coefficients(mesh_,c);const auto&t=mesh_.tets[c];
  double sum=0;for(int a=0;a<8;++a){const int e=a<4?t.v[a]:nv+t.face[a-4];for(int d=0;d<3;++d)sum+=comp(b.b[a],d)*v[3*e+d];}p[c]=sum;
 }
 MPI_Allreduce(MPI_IN_PLACE,p.data(),(int)p.size(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
}
void Schur::BT(const std::vector<double>&p,std::vector<double>&v)const{
 v.assign(inv_.size(),0);const int nv=(int)mesh_.points.size();
 for(int c=0;c<(int)mesh_.tets.size();++c)if(own_.cell[c]==rank_){const auto b=divergence_coefficients(mesh_,c);const auto&t=mesh_.tets[c];
  for(int a=0;a<8;++a){const int e=a<4?t.v[a]:nv+t.face[a-4];for(int d=0;d<3;++d)v[3*e+d]+=comp(b.b[a],d)*p[c];}
 }
 MPI_Allreduce(MPI_IN_PLACE,v.data(),(int)v.size(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
}
void Schur::C(const std::vector<double>&raw,std::vector<double>&out)const{
 out=raw;wall_.Tt(out);
 for(std::size_t i=0;i<out.size();++i)out[i]*=inv_[i];
 wall_.Q(out);wall_.T(out);
}
void Schur::apply(const std::vector<double>&p,std::vector<double>&out)const{
 std::vector<double> bt,v;BT(p,bt);C(bt,v);B(v,out);out[0]=p[0];
}
Schur::Result Schur::solve(const std::vector<double>&b,std::vector<double>&x,double tol,int limit)const{
 Result r;const int n=(int)b.size();x.assign(n,0);
 std::vector<double> rhs=b;rhs[0]=0;std::vector<double> res=rhs, z(n), dir(n), a;
 for(int c=0;c<n;++c)z[c]=res[c]/diag_[c];dir=z;
 double rz=global_dot(res,z,rank_,own_), norm=std::sqrt(global_dot(res,res,rank_,own_));r.r0=norm;r.rFinal=norm;
 if(norm==0){r.converged=true;return r;}
 for(int it=0;it<limit;++it){
  apply(dir,a);const double den=global_dot(dir,a,rank_,own_);
  if(!(den>0) || !std::isfinite(den))throw std::runtime_error("pressure Schur lost SPD: PCG stopped");
  const double alpha=rz/den;for(int c=0;c<n;++c){x[c]+=alpha*dir[c];res[c]-=alpha*a[c];}
  res[0]=0;r.iterations=it+1;r.rFinal=std::sqrt(global_dot(res,res,rank_,own_));
  if(!std::isfinite(r.rFinal))throw std::runtime_error("nonfinite pressure residual");
  if(r.rFinal<=tol*r.r0){r.converged=true;break;}
  for(int c=0;c<n;++c)z[c]=res[c]/diag_[c];
  const double newRz=global_dot(res,z,rank_,own_);const double beta=newRz/rz;rz=newRz;
  for(int c=0;c<n;++c)dir[c]=z[c]+beta*dir[c];dir[0]=0;
 }
 return r;
}
}
