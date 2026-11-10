#include "pressure_amg.hpp"
#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <queue>
#include <stdexcept>
#include <numeric>
#include <limits>
namespace nodals::cpu_mc::m3 {
namespace {
void allsum(std::vector<double>&a){
 if(!a.empty())MPI_Allreduce(MPI_IN_PLACE,a.data(),(int)a.size(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
}
void allsum(std::vector<int>&a){
 if(!a.empty())MPI_Allreduce(MPI_IN_PLACE,a.data(),(int)a.size(),MPI_INT,MPI_SUM,MPI_COMM_WORLD);
}
}
PressureAMG::PressureAMG(int n,const std::vector<int>&owned,const std::vector<int>&rp,
 const std::vector<int>&ci,const std::vector<double>&a,const std::vector<double>&diag,
 const std::vector<int>&owner,int rank):rank_(rank),size_(1){
 MPI_Comm_size(MPI_COMM_WORLD,&size_);
 Level L;L.n=n;L.owned=owned;L.owner=owner;L.rowPtr=rp;L.col=ci;L.a=a;L.diag=diag;
 if((int)owner.size()!=n||(int)diag.size()!=n||rp.size()!=owned.size()+1)
   throw std::runtime_error("AMG fine CSR dimensions mismatch");
 levels_.push_back(std::move(L));
 // Exactly the CPU production strategy: rank-local connected aggregation.
 // Coarse P is constant injection, R=P^T, A_c=P^T A_f P.
 build_coarse(0,16);
 build_coarse(1,32);
 build_terminal_lu();
 if(rank_==0)std::printf("NODALS_CPU_SST_M3_AMG_SETUP status=PASS fine=%d mid=%d coarse=%d levels=%d method=RANK_CONNECTED_UNSMOOTHED_AGGREGATION_PCG_SGS\n",
 levels_[0].n,levels_[1].n,levels_[2].n,(int)levels_.size());
}
void PressureAMG::build_coarse(int lev,int groupSize){
 const Level& L=levels_[lev];
 // All ranks work with replicated coarse group IDs, but own disjoint rows.
 std::vector<int> group(L.n,-1),local(L.n,-1);
 std::vector<int> rowOf(L.n,-1);
 for(std::size_t i=0;i<L.owned.size();++i)rowOf[L.owned[i]]=(int)i;
 int ng=0;
 for(int seed:L.owned){
  if(local[seed]!=-1)continue;
  const int g=ng++;
  local[seed]=g;
  int count=1;
  std::queue<int>q;q.push(seed);
  while(!q.empty() && count<groupSize){
   const int c=q.front();q.pop();int i=rowOf[c];
   for(int k=L.rowPtr[i];k<L.rowPtr[i+1] && count<groupSize;++k){
    const int nb=L.col[k];
    if(nb==c||L.owner[nb]!=rank_||local[nb]>=0)continue;
    // Prefer genuinely connected pressure cells; never aggregate by mesh ID.
    if(!(std::abs(L.a[k])>1e-40))continue;
    local[nb]=g;++count;q.push(nb);
   }
  }
 }
 std::vector<int> counts(size_,0);counts[rank_]=ng;allsum(counts);
 int offset=0,total=0;for(int r=0;r<size_;++r){if(r<rank_)offset+=counts[r];total+=counts[r];}
 for(int c:L.owned)group[c]=local[c]+offset;
 // Unique owner for every fine cell means integer SUM provides exact global map.
 for(int&x:group)if(x<0)x=0;
 allsum(group);
 for(int c=0;c<L.n;++c)if(group[c]<0||group[c]>=total)
  throw std::runtime_error("AMG distributed aggregate ID mismatch");
 Level coarse;coarse.n=total;coarse.owner.resize(total);
 int off=0;for(int r=0;r<size_;++r){for(int i=0;i<counts[r];++i)coarse.owner[off+i]=r;off+=counts[r];}
 for(int j=offset;j<offset+ng;++j)coarse.owned.push_back(j);
 // Owned Galerkin rows; sums all fine rows belonging to the aggregate.
 std::vector<std::map<int,double>> rows((std::size_t)ng);
 for(std::size_t ri=0;ri<L.owned.size();++ri){
  const int c=L.owned[ri],gr=group[c]-offset;
  for(int k=L.rowPtr[ri];k<L.rowPtr[ri+1];++k)
    rows[gr][group[L.col[k]]]+=L.a[k];
 }
 coarse.rowPtr.push_back(0);coarse.diag.assign(total,0.0);
 for(int i=0;i<ng;++i){
  for(const auto &[j,v]:rows[i])if(std::abs(v)>1e-35){
   coarse.col.push_back(j);coarse.a.push_back(v);
   if(j==i+offset)coarse.diag[i+offset]=v;
  }
  coarse.rowPtr.push_back((int)coarse.a.size());
 }
 allsum(coarse.diag);
 for(int c:coarse.owned)if(!(coarse.diag[c]>1e-40))throw std::runtime_error("AMG nonpositive Galerkin diagonal");
 levels_[lev].coarseOf=std::move(group);
 levels_.push_back(std::move(coarse));
}
void PressureAMG::build_terminal_lu(){
 const Level&L=levels_.back();const int n=L.n;
 if(n>3000)throw std::runtime_error("AMG coarsest dense LU exceeded safety cap");
 lu_.assign((std::size_t)n*n,0.0);
 for(std::size_t ri=0;ri<L.owned.size();++ri){
  const int c=L.owned[ri];
  for(int j=L.rowPtr[ri];j<L.rowPtr[ri+1];++j)lu_[(std::size_t)c*n+L.col[j]]+=L.a[j];
 }
 allsum(lu_);piv_.resize(n);
 // Partial-pivot LU performed once. Factorization is local to each MPI rank.
 for(int k=0;k<n;++k){
  int piv=k;double biggest=std::abs(lu_[(std::size_t)k*n+k]);
  for(int i=k+1;i<n;++i){const double a=std::abs(lu_[(std::size_t)i*n+k]);if(a>biggest){biggest=a;piv=i;}}
  if(!(biggest>1e-35)||!std::isfinite(biggest))throw std::runtime_error("AMG terminal LU singular");
  piv_[k]=piv;
  if(piv!=k)for(int j=0;j<n;++j)std::swap(lu_[(std::size_t)k*n+j],lu_[(std::size_t)piv*n+j]);
  const double d=lu_[(std::size_t)k*n+k];
  for(int i=k+1;i<n;++i){double &l=lu_[(std::size_t)i*n+k];l/=d;
    for(int j=k+1;j<n;++j)lu_[(std::size_t)i*n+j]-=l*lu_[(std::size_t)k*n+j];}
 }
}
void PressureAMG::solve_terminal(const std::vector<double>&rhs,std::vector<double>&x)const{
 const int n=(int)rhs.size();x=rhs;
 for(int k=0;k<n;++k){
  if(piv_[k]!=k)std::swap(x[k],x[piv_[k]]);
  for(int i=k+1;i<n;++i)x[i]-=lu_[(std::size_t)i*n+k]*x[k];
 }
 for(int k=n-1;k>=0;--k){
  for(int j=k+1;j<n;++j)x[k]-=lu_[(std::size_t)k*n+j]*x[j];
  x[k]/=lu_[(std::size_t)k*n+k];
 }
}
void PressureAMG::multiply(int lev,const std::vector<double>&x,std::vector<double>&y)const{
 const Level&L=levels_[lev];y.assign(L.n,0.0);
 for(std::size_t i=0;i<L.owned.size();++i){
  double sum=0;for(int j=L.rowPtr[i];j<L.rowPtr[i+1];++j)sum+=L.a[j]*x[L.col[j]];
  y[L.owned[i]]=sum;
 }
 allsum(y);
}
void PressureAMG::sweep(int lev,const std::vector<double>&rhs,std::vector<double>&x,bool forward)const{
 const Level&L=levels_[lev];
 std::vector<double>changed(L.n,0.0);
 // Rank-local GS, processor-block Jacobi across MPI ranks, exactly as CPU AMG.
 for(std::size_t k=0;k<L.owned.size();++k){
  const std::size_t i=forward?k:L.owned.size()-k-1;
  const int row=L.owned[i];double s=rhs[row];
  for(int j=L.rowPtr[i];j<L.rowPtr[i+1];++j)if(L.col[j]!=row)s-=L.a[j]*x[L.col[j]];
  const double next=s/L.diag[row];
  changed[row]=next-x[row];x[row]=next;
 }
 // Each processor owns a disjoint row set. Apply only the changes, NOT the
 // full replicated x, to avoid summing stale off-rank values 16 times.
 allsum(changed);
 for(std::size_t i=0;i<x.size();++i)if(L.owner[i]!=rank_)x[i]+=changed[i];
}
void PressureAMG::vcycle(int lev,const std::vector<double>&rhs,std::vector<double>&x)const{
 if(lev==(int)levels_.size()-1){solve_terminal(rhs,x);return;}
 const Level&L=levels_[lev];x.assign(L.n,0.0);
 sweep(lev,rhs,x,true);
 std::vector<double>Ax; multiply(lev,x,Ax);
 std::vector<double>coarseR(levels_[lev+1].n,0.0);
 for(int c:L.owned)coarseR[L.coarseOf[c]]+=rhs[c]-Ax[c];
 allsum(coarseR);
 std::vector<double>coarseX;vcycle(lev+1,coarseR,coarseX);
 for(int c=0;c<L.n;++c)x[c]+=coarseX[L.coarseOf[c]];
 // Backward local SGS applies to the full Galerkin-corrected residual.
 sweep(lev,rhs,x,false);
}
void PressureAMG::apply(const std::vector<double>&r,std::vector<double>&z)const{
 vcycle(0,r,z);
 if(!z.empty())z[0]=0.0; // pinned pressure is excluded from the live solve.
}
}
