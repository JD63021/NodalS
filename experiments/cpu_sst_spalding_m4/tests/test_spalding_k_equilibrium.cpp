#include "../../../turbulence/sst_cpu_mc/m4/src/transport/spalding_k_equilibrium.hpp"
#include <cassert>
#include <cmath>
#include <array>
#include <algorithm>
#include <iostream>
using nodals::cpu_mc::m4::spalding_log_equilibrium_k;
using nodals::cpu_mc::m4::spalding_offwall_k_penalty_coeff;
int main(){
 const double u=.5859;
 const double k=spalding_log_equilibrium_k(u,1e-8);
 assert(std::abs(k-u*u/.3)<1e-12);
 assert(std::abs(spalding_log_equilibrium_k(2*u,1e-8)-4*k)<1e-11);
 assert(spalding_log_equilibrium_k(0,1e-8)==1e-8);
 bool rejected=false;
 try{(void)spalding_log_equilibrium_k(-1,1e-8);}
 catch(...){rejected=true;}
 assert(rejected);
 // Offwall FE rank-one penalty is symmetric positive semidefinite.
 const std::array<double,8> phi{.13,.23,.35,.29,.01,.02,.03,.04};
 const std::array<double,8> trial{1.,-.3,.1,.6,-.2,.4,-.1,.2};
 const double nuEff=.025,area=.0003,qw=.08,ell=.0073442516;
 for(double gamma:{1000.,10000.}){
  const double coeff=spalding_offwall_k_penalty_coeff(gamma,nuEff,area,qw,ell);
  double quad=0,dot=0;
  for(int i=0;i<8;++i){
   dot+=phi[i]*trial[i];
   for(int j=0;j<8;++j)quad+=trial[i]*coeff*phi[i]*phi[j]*trial[j];
  }
  assert(coeff>0);
  assert(quad>=-1e-12);
  assert(std::abs(quad-coeff*dot*dot)<1e-12);
 }
 std::cout<<"K_EQ_TARGET_TEST=PASS kRef="<<k
          <<" targetMethod=SpaldingOffwall point=3 gammas=1000,10000\n";
}
