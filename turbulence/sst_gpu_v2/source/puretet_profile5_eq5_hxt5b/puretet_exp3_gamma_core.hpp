#pragma once
// Frozen EQ5 scale arithmetic. Standalone C++17 host testable.
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <limits>
namespace puretet_exp3_core {
struct System {
 int n=0,nv=0; double alpha=0.9,targetNorm=0;
 std::vector<int> row,col;std::vector<double> phys,wall,rhs,wallrhs,old;
 // 'phys' is complete physical system at gamma_0, without row-L1 relaxation.
 // 'rhs'  is physical RHS at gamma_0; gamma_0 wall RHS is separate.
 void validate() const {
  if(n<1||nv<1||nv>n||row.size()!=size_t(n+1)||col.size()!=phys.size()||wall.size()!=phys.size()||rhs.size()!=size_t(n)||wallrhs.size()!=size_t(n)||old.size()!=size_t(n)||row[0]!=0||row.back()!=int(col.size())||!(alpha>0&&alpha<=1))throw std::runtime_error("EXP3 system shape/alpha mismatch");
  for(int i=0;i<n;++i){bool diagonal=false;for(int a=row[i];a<row[i+1];++a){if(col[a]==i)diagonal=true;if(col[a]<0||col[a]>=n)throw std::runtime_error("EXP3 CSR out of bounds");}if(i<nv&&!diagonal)throw std::runtime_error("EXP3 missing diagonal");}
 }
 struct Scaled {std::vector<double> matrix,right;double minPhysicalDiag=std::numeric_limits<double>::infinity(),maxPhysicalDiag=0,diagRatio=0,maxRowL1OverDiag=0,maxRelaxShiftOverDiag=0;int invalidDiag=0;};
 Scaled scaled(double factor) const {
  if(!(factor>=0)||!std::isfinite(factor))throw std::runtime_error("EXP3 scale invalid");
  Scaled out;out.matrix.resize(phys.size());out.right=rhs;
  for(int i=0;i<n;++i){out.right[i]+=(factor-1)*wallrhs[i];double l1=0,d0=0;int d=-1;for(int a=row[i];a<row[i+1];++a){double z=phys[a]+(factor-1)*wall[a];out.matrix[a]=z;l1+=std::abs(z);if(col[a]==i){d=a;d0=z;}}
   if(i<nv){if(!(d0>0)||!std::isfinite(d0))out.invalidDiag++;out.minPhysicalDiag=std::min(out.minPhysicalDiag,d0);out.maxPhysicalDiag=std::max(out.maxPhysicalDiag,d0);if(d0>0){out.maxRowL1OverDiag=std::max(out.maxRowL1OverDiag,l1/d0);out.maxRelaxShiftOverDiag=std::max(out.maxRelaxShiftOverDiag,(1/alpha-1)*l1/d0);}}
   const double delta=(1/alpha-1)*l1;
   out.matrix[d]+=delta;out.right[i]+=delta*old[i];
  }
  out.diagRatio=out.minPhysicalDiag>0?out.maxPhysicalDiag/out.minPhysicalDiag:std::numeric_limits<double>::infinity();
  return out;
 }
 double wall_rel_mismatch(const std::vector<double>&x) const {
  if(x.size()!=size_t(n)||!(targetNorm>0))return std::numeric_limits<double>::quiet_NaN();
  // Quadratic LS residual = x^T W x -2 x^T b_wall + sum_m gamma*w_m*t_m^2.
  // All W, wallrhs and targetNorm are from baseline gamma0.
  long double q=0,bt=0;for(int i=0;i<nv;++i){long double r=0;for(int a=row[i];a<row[i+1];++a)r+=(long double)wall[a]*x[col[a]];q+=(long double)x[i]*r;bt+=(long double)x[i]*wallrhs[i];}
  const long double delta=q-2*bt+(long double)targetNorm;
  return std::sqrt((double)std::max((long double)0,delta/(long double)targetNorm));
 }
 double residual(const std::vector<double>&A,const std::vector<double>&b,const std::vector<double>&x)const {
  long double t=0;for(int i=0;i<nv;++i){long double r=-b[i];for(int a=row[i];a<row[i+1];++a)r+=(long double)A[a]*x[col[a]];t+=r*r;}return std::sqrt((double)t);
 }
};
}
