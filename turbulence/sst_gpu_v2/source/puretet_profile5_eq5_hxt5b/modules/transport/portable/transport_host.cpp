#include "transport_host.hpp"
#include "transport_core.hpp"
namespace nodals { namespace transport {
HostVolumeResult evaluate_volume_host(int equation,int kEquation,double wt,const HostVolumeState&s){
  const auto r=volume_coefficients<double>(equation,kEquation,wt,s);
  return {r.diffusion,r.reaction,r.source,r.omegaProductionAudit,r.omegaCrossAudit};
}
HostRelaxResult evaluate_relax_host(double d0,double l1,double alpha){
  const auto r=row_l1_plan<double>(d0,l1,alpha);return {r.delta,r.diagonal};
}
}}
