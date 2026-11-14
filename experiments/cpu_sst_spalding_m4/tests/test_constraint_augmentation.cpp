#include "../../../turbulence/sst_cpu_mc/m4/src/enrichment/constraint_augmentation.hpp"
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <algorithm>
using nodals::cpu_mc::m4::add_constraint_augmentation;
template<std::size_t N> void test(){
 constexpr std::size_t M=3*N;
 std::array<std::array<double,M>,M> K{};
 std::array<double,M> rhs{},u{};
 std::array<std::array<double,3>,N> B{};
 for(std::size_t a=0;a<N;++a){
  B[a]={.18+.023*a,-.10+.005*a,.09-.008*a};
  for(int j=0;j<3;++j)u[3*a+j]=.6+.2*std::sin(a+.1*j);
 }
 const double gamma=1e6,vol=.00031,coeff=gamma/vol;
 double Bu=0;
 for(std::size_t a=0;a<N;++a)for(int j=0;j<3;++j)Bu+=B[a][j]*u[3*a+j];
 const double source=-Bu; // physical DG-inlet continuity residual B u+source=0
 add_constraint_augmentation<N>(K,rhs,B,source,0.0);
 for(std::size_t i=0;i<M;++i){
  assert(rhs[i]==0);
  for(std::size_t j=0;j<M;++j)assert(K[i][j]==0);
 }
 add_constraint_augmentation<N>(K,rhs,B,source,coeff);
 double quad=0,residualNorm=0,expectedNorm=0;
 for(std::size_t i=0;i<M;++i){
  double au=0;
  for(std::size_t j=0;j<M;++j){
   assert(std::abs(K[i][j]-K[j][i])<=1e-14*std::max(1.,std::abs(K[i][j])));
   au+=K[i][j]*u[j];
   quad+=u[i]*K[i][j]*u[j];
  }
  residualNorm+=std::pow(au-rhs[i],2);
  expectedNorm+=au*au;
 }
 assert(quad>0);
 assert(residualNorm<=1e-21*expectedNorm);
 assert(std::abs(quad-coeff*Bu*Bu)<1e-10*quad);
 for(std::size_t a=0;a<N;++a)for(int j=0;j<3;++j){
  const double bj=B[a][j];
  assert(std::abs(rhs[3*a+j]+coeff*bj*source)
        <=1e-14*std::max(1.,std::abs(rhs[3*a+j])));
 }
 // Test the general residual identity for a state not satisfying continuity:
 // (A_aug*u-b_aug)_i = gamma/vol * B_i*(Bu + source).
 u[0]+=0.33; Bu+=.33*B[0][0];
 double err=0,scale=0;
 for(std::size_t i=0;i<M;++i){
  double au=0;for(std::size_t j=0;j<M;++j)au+=K[i][j]*u[j];
  const double expected=coeff*B[i/3][i%3]*(Bu+source);
  err+=std::pow((au-rhs[i])-expected,2);
  scale+=expected*expected;
 }
 assert(err<=1e-20*scale);
 bool rejected=false;
 try{add_constraint_augmentation<N>(K,rhs,B,source,-1.0);}
 catch(const std::runtime_error&){rejected=true;}
 assert(rejected);
 std::cout<<"NODALS_CONSTRAINT_AUGMENTATION_TEST N="<<N<<" gamma=1e6"
 <<" nonzero_source_rhs=PASS exact_residual_identity=PASS"
 <<" positive_semidefinite=PASS zero_noop=PASS symmetry=PASS\n";
}
int main(){test<8>();test<12>();}
