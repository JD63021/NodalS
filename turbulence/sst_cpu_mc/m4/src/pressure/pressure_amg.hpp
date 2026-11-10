#pragma once
#include <vector>
namespace nodals::cpu_mc::m3 {
// Adapted CPU-native design: rank-local connected unsmoothed aggregation,
// processor-block symmetric GS, Galerkin coarse operators and terminal LU.
// Unlike the original PETSc-dependent CPU module, this adapter accepts the
// M3 exact constrained Schur CSR directly and has no CUDA/PETSc dependencies.
class PressureAMG {
 struct Level {
  int n=0;
  std::vector<int> owned,owner,rowPtr,col,coarseOf;
  std::vector<double> a,diag;
 };
 int rank_,size_;
 std::vector<Level> levels_;
 std::vector<double> lu_;
 std::vector<int> piv_;
 void build_coarse(int fineLevel,int groupSize);
 void build_terminal_lu();
 void multiply(int lev,const std::vector<double>&x,std::vector<double>&y)const;
 void sweep(int lev,const std::vector<double>&rhs,std::vector<double>&x,bool forward)const;
 void vcycle(int lev,const std::vector<double>&rhs,std::vector<double>&x)const;
 void solve_terminal(const std::vector<double>&rhs,std::vector<double>&x)const;
public:
 PressureAMG(int n,const std::vector<int>&owned,const std::vector<int>&rp,
             const std::vector<int>&ci,const std::vector<double>&a,
             const std::vector<double>&diag,const std::vector<int>&owner,int rank);
 void apply(const std::vector<double>&r,std::vector<double>&z)const;
 int levels()const{return (int)levels_.size();}
 int coarse_size()const{return levels_.back().n;}
};
}
