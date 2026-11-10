#pragma once
#include "../mesh/tet_mesh.hpp"
#include "../parallel/rcb.hpp"
#include "../operators/p1bf3_p0.hpp"
#include "../wall/bf3_flux.hpp"
#include <array>
#include <memory>
#include <vector>
namespace nodals::cpu_mc::m3 {
using namespace nodals::cpu_mc::m2;
class PressureAMG;
// Exact constrained pressure operator:
//   S = (B_volume - B_DG_inlet) T Q diag(rAU) Q T^T (B_volume - B_DG_inlet)^T
//   continuity = (B_volume - B_DG_inlet) u + source_DG_inlet.
// The explicit sparse S is assembled only once per frozen Picard momentum
// metric; whenever the momentum row-L1 metric changes, create a new Schur.
class Schur {
 struct Row {std::array<int,8> entity{};std::array<Vec3,8> physical{},reduced{};};
 const Mesh& mesh_; const Ownership& own_; const BF3Flux& wall_;
 int rank_;
 std::vector<double> inv_,source_,diag_;
 std::vector<Row> rows_;
 // Only rows owned by this MPI rank are stored in CSR (global column IDs).
 std::vector<int> ownedRows_,rowPtr_,col_;
 std::vector<double> val_;
 std::unique_ptr<PressureAMG> amg_;
public:
 Schur(const Mesh&m,const Ownership&o,const BF3Flux&w,int rank,std::vector<double> inverse,std::array<double,3> inletVelocity);
 ~Schur();
 void B(const std::vector<double>&physicalVelocity,std::vector<double>&pressure)const;
 void BT(const std::vector<double>&pressure,std::vector<double>&velocity)const;
 void continuity(const std::vector<double>&physicalVelocity,std::vector<double>&residual)const;
 void C(const std::vector<double>&raw,std::vector<double>&out)const;
 void apply(const std::vector<double>&p,std::vector<double>&out)const;
 struct Result{int iterations=0;double r0=0,rFinal=0;bool converged=false;};
 Result solve(const std::vector<double>&b,std::vector<double>&x,double rtol,int maxits)const;
 std::size_t local_nnz()const{return val_.size();}
 double source_sum()const;
 const std::vector<double>& diagonals()const{return diag_;}
};
}
