#include "../../../turbulence/sst_cpu_mc/m4/src/enrichment/grad_div_rank_one.hpp"
#include <array>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
using nodals::cpu_mc::m4::add_grad_div_rank_one;
template<std::size_t N>
void test(){
 constexpr std::size_t M=3*N;
 std::array<std::array<double,M>,M> K{};
 std::array<std::array<double,3>,N> g{};
 std::array<double,M> x{};
 for(std::size_t a=0;a<N;++a){
  g[a]={0.7+0.02*a,-0.3+0.06*a,0.2-0.01*a};
  for(int i=0;i<3;++i)x[3*a+i]=std::sin(0.5*a+0.3*i+0.1);
 }
 // Strict no-op with zero.
 add_grad_div_rank_one<N>(K,g,0);
 for(std::size_t i=0;i<M;++i)for(std::size_t j=0;j<M;++j)assert(K[i][j]==0);
 const double coeff=0.037;
 add_grad_div_rank_one<N>(K,g,coeff);
 double dx=0,q=0;
 for(std::size_t a=0;a<N;++a)for(int i=0;i<3;++i)
  dx+=x[3*a+i]*g[a][i];
 for(std::size_t i=0;i<M;++i)for(std::size_t j=0;j<M;++j){
  assert(std::abs(K[i][j]-K[j][i])<1e-15);
  q+=x[i]*K[i][j]*x[j];
 }
 assert(q>=-1e-12);
 assert(std::abs(q-coeff*dx*dx)<1e-12);
 // Manufactured grad-div-free vector is in the rank-one nullspace.
 x.fill(0.0);
 x[0]=g[1][0];x[3]=-g[0][0];
 dx=0;q=0;
 for(std::size_t a=0;a<N;++a)for(int i=0;i<3;++i)
  dx+=x[3*a+i]*g[a][i];
 for(std::size_t i=0;i<M;++i)for(std::size_t j=0;j<M;++j)
  q+=x[i]*K[i][j]*x[j];
 assert(std::abs(dx)<1e-14);
 assert(std::abs(q)<1e-12);
 if constexpr (N==12){
  // Essential: PP-PE, PE-PP, EE populated, no decoupled enrichment.
  const double ep=K[3*8+1][3*2+0];
  const double pe=K[3*2+0][3*8+1];
  assert(std::abs(ep)>1e-12 && ep==pe);
  assert(K[3*8][3*8]>0);
 }
 bool rejected=false;
 try{add_grad_div_rank_one<N>(K,g,-1.0);}
 catch(const std::runtime_error&){rejected=true;}
 assert(rejected);
 std::cout<<"NODALS_GRADDIV_TEST N="<<N<<" symmetry=PASS positive_semi_definite=PASS"
          <<" zero_coefficient_noop=PASS nullspace=PASS cross_blocks=PASS\n";
}
int main(){test<8>();test<12>();}
