#pragma once
#include "../mesh/tet_mesh.hpp"
#include "../parallel/rcb.hpp"
#include "../operators/p1bf3_p0.hpp"
#include "../wall/bf3_flux.hpp"
#include <vector>
namespace nodals::cpu_mc::m3 {
using namespace nodals::cpu_mc::m2;
class Schur {
 const Mesh& mesh_;const Ownership& own_;const BF3Flux& wall_;
 int rank_;std::vector<double> inv_; std::vector<double> diag_;
public:
 Schur(const Mesh&m,const Ownership&o,const BF3Flux&w,int rank,std::vector<double> inverse);
 void B(const std::vector<double>&velocity,std::vector<double>&pressure)const;
 void BT(const std::vector<double>&pressure,std::vector<double>&velocity)const;
 void C(const std::vector<double>&raw,std::vector<double>&out)const;
 void apply(const std::vector<double>&p,std::vector<double>&out)const;
 struct Result{int iterations=0;double r0=0,rFinal=0;bool converged=false;};
 Result solve(const std::vector<double>&b,std::vector<double>&x,double rtol,int maxits)const;
};
}
