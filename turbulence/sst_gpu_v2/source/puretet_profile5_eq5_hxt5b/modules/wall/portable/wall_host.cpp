#include "wall_host.hpp"
namespace nodals { namespace wall {
HostOFState evaluate_of_auto_host(const HostOFInput& a){
 struct S{ double pk,nut; } ss{a.pk,a.nut};
 auto s=evaluate_of_auto<double>(a.y,a.nu,a.Ut,a.uTau,a.k,a.omega,ss,
      a.kFloor,a.omegaFloor,a.kappa,a.E,a.beta1,a.reBlend,a.prodLimit,a.sharedSpaldingWallState);
 return {s.omegaTarget,s.omegaVis,s.omegaLog,s.lamFrac,s.rey,s.yPlusK,s.pkAuto,s.nutSpalding};
}
HostProfile fit_profile5_host(const double U[5],const double Usigned[5],const double y[5],double nu){
 HostProfile x{};
 x.valid=fit_profile5_values(U,Usigned,y,nu,x.uTau,x.yPlus,x.beta,x.uMid,x.fitRms,x.pointUTauRelSpan);
 return x;
}
}}
