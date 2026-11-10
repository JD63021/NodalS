#pragma once
#include "../mesh/tet_mesh.hpp"
#include "../parallel/rcb.hpp"
#include "../operators/p1bf3_p0.hpp"
#include "../wall/bf3_flux.hpp"
#include "../wall/wall_sampler.hpp"
#include "../case/case_config.hpp"
#include <vector>
#include <array>
namespace nodals::cpu_mc::m3 {
using namespace nodals::cpu_mc::m2;
// Owned-row 3x3 block CSR with processor-block symmetric GS across MPI ranks.
// Full P1+BF3 T^T A T is assembled from incident tets, including rank halo tets.
class M3Predictor {
 const Mesh&mesh_;const Ownership&own_;const BF3Flux&wall_;const CaseConfig&case_;const std::vector<double>&lag_;
 int rank_,nEntities_;
 std::vector<int> owned_,rowPtr_,cols_,ownerOfEntity_;
 std::vector<std::array<double,9>> a_,diagInverse_;
 std::vector<double> rhs_,rowL1_,inv_;
 std::vector<Vec3> wallBubbleNormal_; // GPU per-row tangential BF3 GS projection
 const std::vector<double>* k_=nullptr,*omega_=nullptr,*wallDistance_=nullptr;
 WallFaceFit wallFit_;
 std::size_t ownedCells_=0,incidentCells_=0;
 void sync_updates(std::vector<double>&x,const std::vector<double>&old)const;
public:
 M3Predictor(const Mesh&m,const Ownership&o,const BF3Flux&w,const CaseConfig&c,int rank,const std::vector<double>&lag,
   const std::vector<double>* k=nullptr,const std::vector<double>* omega=nullptr,
   const std::vector<double>* wallDistance=nullptr);
 // input and output in constrained BF3 PARAMETER coordinates (reduced operator)
 void apply(const std::vector<double>&x,std::vector<double>&out)const;
 // returns PHYSICAL velocity, u=T*x, for the pressure continuity calculation
 void solve(std::vector<double>&u,int sweeps,double omega,double*initialResidual=nullptr,double*finalResidual=nullptr,
   const std::vector<double>*pressureForce=nullptr,int*iterations=nullptr)const;
 const WallFaceFit&wall_fit()const{return wallFit_;}
 const std::vector<double>&inverse()const{return inv_;}
 std::size_t local_cells()const{return ownedCells_;}
 std::size_t incident_cells()const{return incidentCells_;}
 std::size_t local_nnz()const{return a_.size();}
};
}
