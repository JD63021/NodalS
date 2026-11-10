#pragma once
#include "../mesh/tet_mesh.hpp"
#include "../parallel/rcb.hpp"
#include "../operators/p1bf3_p0.hpp"
#include "../wall/bf3_flux.hpp"
#include "../case/case_config.hpp"
#include <vector>
#include <array>
namespace nodals::cpu_mc::m3 {
using namespace nodals::cpu_mc::m2;
class M3Predictor{
 const Mesh&mesh_;const Ownership&own_;const BF3Flux&wall_;const CaseConfig&case_;
 int rank_;std::size_t n_;
 struct CellBlock {std::array<int,8> ent{};Element24 k{};};
 std::vector<CellBlock> cells_;
 std::vector<double> rhs_,rowL1_,inv_;
public:
 M3Predictor(const Mesh&m,const Ownership&o,const BF3Flux&w,const CaseConfig&c,int rank);
 void apply(const std::vector<double>&x,std::vector<double>&out)const;
 void solve(std::vector<double>&u,int sweeps,double omega,double*initialResidual=nullptr,double*finalResidual=nullptr)const;
 const std::vector<double>& inverse()const{return inv_;}
 std::size_t local_cells()const{return cells_.size();}
};
}
