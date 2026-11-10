#include "m3_predictor.hpp"
#include <mpi.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstdio>
#include "spalding_core.hpp"
namespace nodals::cpu_mc::m3 {
namespace {
const double Q0[12]={.063089014491502,.063089014491502,.873821971016996,.249286745170910,.249286745170910,.501426509658180,.053145049844816,.053145049844816,.310352451033785,.310352451033785,.636502499121399,.636502499121399};
const double Q1[12]={.063089014491502,.873821971016996,.063089014491502,.249286745170910,.501426509658180,.249286745170910,.310352451033785,.636502499121399,.053145049844816,.636502499121399,.053145049844816,.310352451033785};
const double Q2[12]={.873821971016996,.063089014491502,.063089014491502,.501426509658180,.249286745170910,.249286745170910,.636502499121399,.310352451033785,.636502499121399,.053145049844816,.310352451033785,.053145049844816};
const double QW[12]={.050844906370207,.050844906370207,.050844906370207,.116786275726379,.116786275726379,.116786275726379,.082851075618374,.082851075618374,.082851075618374,.082851075618374,.082851075618374,.082851075618374};
const double X5[5]={.0469100770306680185,.23076534494715845,.5,.7692346550528415,.95308992296933193};
const double W5[5]={.11846344252809449,.23931433524968326,.28444444444444450,.23931433524968326,.11846344252809449};
using G=std::array<Vec3,8>;
struct Geom {std::array<Vec3,4> gl;double det;};
Geom geometry(const Mesh&m,int c){const auto &t=m.tets[c];const Vec3 a=m.points[t.v[0]],e1=m.points[t.v[1]]-a,e2=m.points[t.v[2]]-a,e3=m.points[t.v[3]]-a;const double det=dot(e1,cross(e2,e3));if(!(det>0))throw std::runtime_error("tet orientation invalid");Geom x{};x.det=det;x.gl[1]=(1.0/det)*cross(e2,e3);x.gl[2]=(1.0/det)*cross(e3,e1);x.gl[3]=(1.0/det)*cross(e1,e2);x.gl[0]=-1.0*(x.gl[1]+x.gl[2]+x.gl[3]);return x;}
void shape(const Geom&g,const std::array<double,4>&l,double val[8],G&gr){
 for(int i=0;i<4;++i){val[i]=l[i];gr[i]=g.gl[i];}
 for(int i=0;i<4;++i){int js[3],n=0;for(int j=0;j<4;++j)if(j!=i)js[n++]=j;
  val[4+i]=27*l[js[0]]*l[js[1]]*l[js[2]];gr[4+i]={};
  for(int j=0;j<3;++j)gr[4+i]=gr[4+i]+27.0*l[js[(j+1)%3]]*l[js[(j+2)%3]]*g.gl[js[j]];
 }
}
void face_basis(const Geom&geom,int lf,int q,double ph[8],G&gr){
 std::array<double,4> l{};int f=0;for(int i=0;i<4;++i)if(i!=lf){l[i]=(f==0?Q0[q]:(f==1?Q1[q]:Q2[q]));++f;}
 shape(geom,l,ph,gr);
}
void convection(Element24&K,const Geom&geom,const std::array<double,3>&carrier){
 for(int ir=0;ir<5;++ir)for(int is=0;is<5;++is)for(int it=0;it<5;++it){
  const double r=X5[ir],s=X5[is],t=X5[it],omr=1-r,oms=1-s;
  const double wt=W5[ir]*W5[is]*W5[it]*omr*omr*oms*geom.det;
  const std::array<double,4> l={omr*oms*(1-t),r,omr*s,omr*oms*t};double ph[8];G gr{};shape(geom,l,ph,gr);
  for(int a=0;a<8;++a)for(int b=0;b<8;++b){const double v=wt*ph[a]*(carrier[0]*gr[b].x+carrier[1]*gr[b].y+carrier[2]*gr[b].z);
    for(int d=0;d<3;++d)K[3*a+d][3*b+d]+=v;
  }
 }
}
}
M3Predictor::M3Predictor(const Mesh&m,const Ownership&o,const BF3Flux&w,const CaseConfig&config,int rank):mesh_(m),own_(o),wall_(w),case_(config),rank_(rank),n_(3*(m.points.size()+m.faces.size())){
 std::vector<double> localRhs(n_,0),localL1(n_,0);
 const int nv=(int)m.points.size();cells_.reserve(o.countCell[rank]);
 for(int c=0;c<(int)m.tets.size();++c)if(o.cell[c]==rank){
  const auto&tet=m.tets[c];CellBlock cb{};for(int a=0;a<8;++a)cb.ent[a]=a<4?tet.v[a]:nv+tet.face[a-4];
  const Geom geom=geometry(m,c);
  cb.k=deviatoric_element(m,c,config.nu);
  convection(cb.k,geom,config.inlet); // frozen Picard carrier; M4 will supply the actual iterate
  for(int lf=0;lf<4;++lf){
   const int f=tet.face[lf];const auto &face=m.faces[f];if(face.cell1>=0 || (face.patch!=1 && face.patch!=3))continue;
   const Vec3 area=outward_area(m,c,lf),n=(1.0/std::sqrt(dot(area,area)))*area;
   const double ar=std::sqrt(dot(area,area)),h=3.0*tet.volume/ar;
   const double inflow=std::max(0.0,-(config.inlet[0]*n.x+config.inlet[1]*n.y+config.inlet[2]*n.z));
   // Frozen zero-velocity initial state only: evaluate the ORIGINAL
   // five-point Spalding fitter, which returns the viscous beta=nu/y[2].
   // M4 must supply the actual five ray-interpolated velocities here.
   const double y[5]={0.4,0.7,1.0,1.3,1.6};
   const double yf[5]={y[0]*config.spaldingScale*h,y[1]*config.spaldingScale*h,
                       y[2]*config.spaldingScale*h,y[3]*config.spaldingScale*h,
                       y[4]*config.spaldingScale*h};
   const double zero[5]={0,0,0,0,0};
   double ut=0,yp=0,beta=0,umid=0,fit=0,span=0;
   if(!nodals::wall::fit_profile5_values(zero,zero,yf,config.nu,ut,yp,beta,umid,fit,span))
     throw std::runtime_error("five-point Spalding wall-law evaluation failed");
   const double nvect[3]={n.x,n.y,n.z};
   for(int q=0;q<12;++q){double ph[8];G g{};face_basis(geom,lf,q,ph,g);
    const double wt=ar*QW[q];
    for(int a=0;a<8;++a){const double dna=dot(g[a],n);if(face.patch==1){
       const double source=wt*(inflow*ph[a]+config.nu*dna);
       for(int i=0;i<3;++i)localRhs[3*cb.ent[a]+i]+=source*config.inlet[i];
    }
    for(int b=0;b<8;++b){const double dnb=dot(g[b],n);
     for(int i=0;i<3;++i)for(int j=0;j<3;++j){
       if(face.patch==1){
         if(i==j)cb.k[3*a+i][3*b+j]+=wt*(inflow*ph[a]*ph[b]+config.nu*(-ph[a]*dnb+dna*ph[b]));
       }else{
         // Vector tangential Robin + normal weak penalty; vector BF3 flux
         // is enforced separately by exact T / T^T elimination.
         const double nn=nvect[i]*nvect[j];
         const double v=beta*((i==j?1.0:0.0)-nn)+config.nitscheGamma*config.nu/h*nn;
         cb.k[3*a+i][3*b+j]+=wt*ph[a]*ph[b]*v;
       }
     }
    }
    }
   }
  }
  for(int a=0;a<8;++a)for(int i=0;i<3;++i){const int row=3*cb.ent[a]+i;
   for(int b=0;b<8;++b)for(int j=0;j<3;++j)localL1[row]+=std::abs(cb.k[3*a+i][3*b+j]);
  }
  cells_.push_back(std::move(cb));
 }
 rhs_=std::move(localRhs);rowL1_=std::move(localL1);
 MPI_Allreduce(MPI_IN_PLACE,rhs_.data(),(int)n_,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
 MPI_Allreduce(MPI_IN_PLACE,rowL1_.data(),(int)n_,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
 wall_.Tt(rhs_);
 inv_.resize(n_);for(std::size_t j=0;j<n_;++j)inv_[j]=(rowL1_[j]>1e-30)?config.alphaU/rowL1_[j]:0.0;
}
void M3Predictor::apply(const std::vector<double>&x,std::vector<double>&out)const{
 std::vector<double> phys=x;wall_.Q(phys);wall_.T(phys);
 out.assign(n_,0.0);
 for(const auto &cell:cells_){
   double xx[24],yy[24]{};for(int a=0;a<8;++a)for(int d=0;d<3;++d)xx[3*a+d]=phys[3*cell.ent[a]+d];
   for(int i=0;i<24;++i)for(int j=0;j<24;++j)yy[i]+=cell.k[i][j]*xx[j];
   for(int a=0;a<8;++a)for(int d=0;d<3;++d)out[3*cell.ent[a]+d]+=yy[3*a+d];
 }
 MPI_Allreduce(MPI_IN_PLACE,out.data(),(int)n_,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
 wall_.Tt(out);
}
void M3Predictor::solve(std::vector<double>&u,int sweeps,double omega,double*start,double*final)const{
 u.assign(n_,0);std::vector<double> Ku,res(n_),dir(n_),Ad;
 const auto norm=[&](const std::vector<double>&r){
   double l=0;for(double v:r)l+=v*v;return std::sqrt(l);
 };
 for(int it=0;it<=sweeps;++it){
  apply(u,Ku);for(std::size_t i=0;i<n_;++i)res[i]=rhs_[i]-Ku[i];
  const double rn=norm(res);if(it==0&&start)*start=rn;
  if(it==sweeps){if(final)*final=rn;break;}
  for(std::size_t i=0;i<n_;++i)dir[i]=inv_[i]*res[i];
  wall_.Q(dir);apply(dir,Ad);
  double numerator=0,denominator=0;
  for(std::size_t i=0;i<n_;++i){numerator+=res[i]*Ad[i];denominator+=Ad[i]*Ad[i];}
  if(denominator<1e-300)break;
  // Monotone one-dimensional residual minimisation, applied only as a
  // DEVELOPMENT substitute for the GPU/CPU SGS momentum preconditioner.
  const double length=omega*numerator/denominator;
  for(std::size_t i=0;i<n_;++i)u[i]+=length*dir[i];
  wall_.Q(u);
  for(double v:u)if(!std::isfinite(v))throw std::runtime_error("nonfinite M3 momentum sweep");
 }
 wall_.T(u);
}
}
