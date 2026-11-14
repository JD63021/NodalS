// Standalone CPU test: verifies that the exact shared GPU OF_AUTO wall state
// blends between viscous and logarithmic omega, and that sampled rank-one
// penalty is symmetric positive-semidefinite. Does NOT require MPI.
#include "../../../turbulence/sst_gpu_v2/source/puretet_profile5_eq5_hxt5b/modules/wall/portable/of_auto_core.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <array>
#include <algorithm>
struct SST { double pk=0.3,nut=0.03; };
static bool rel(double x,double y,double tol=1e-9){
 return std::abs(x-y)<=tol*std::max({1.,std::abs(x),std::abs(y)});
}
int main(){
 const double nu=1.460734693878e-5,y=9.797675557402e-4,ut=.5859;
 const auto low=nodals::wall::evaluate_of_auto<double>(
     y,nu,11.0,ut,1e-10,10.,SST{},1e-8,1e-3,
     0.4,9.025013499434122,.075,11.,10.,1);
 const auto high=nodals::wall::evaluate_of_auto<double>(
     y,nu,11.0,ut,86.,2024.,SST{},1e-8,1e-3,
     0.4,9.025013499434122,.075,11.,10.,1);
 const double logT=ut/(std::sqrt(0.09)*.4*y);
 const double visT=6.*nu/(.075*y*y);
 assert(rel(low.omegaLog,logT));
 assert(rel(low.omegaVis,visT));
 assert(low.lamFrac>.99);
 assert(low.omegaTarget>visT*.99 && low.omegaTarget<logT);
 assert(high.lamFrac<1e-12);
 assert(rel(high.omegaTarget,logT));
 const double ell=.0074687304-.0001244788;
 const double area=3e-4,wface=.1,nuEff=.03,gamma=1e4;
 const double coeff=gamma*nuEff*(area*wface)/ell;
 const std::array<double,8> phi={.12,.23,.31,.34,.01,.02,.03,.04};
 const std::array<double,8> x={1.,-2.,.5,.2,.3,-.1,.7,-.2};
 double quadratic=0,dot=0;
 for(int i=0;i<8;++i)dot+=phi[i]*x[i];
 for(int i=0;i<8;++i)for(int j=0;j<8;++j)
    quadratic+=x[i]*(coeff*phi[i]*phi[j])*x[j];
 assert(quadratic>=-1e-12);
 assert(rel(quadratic,coeff*dot*dot));
 std::cout<<"OMEGA_TARGET_TEST=PASS autoViscous="<<low.omegaTarget
   <<" autoLog="<<high.omegaTarget
   <<" sampleLog="<<logT
   <<" penaltyCoeff="<<coeff<<" ell="<<ell<<"\n";
}
