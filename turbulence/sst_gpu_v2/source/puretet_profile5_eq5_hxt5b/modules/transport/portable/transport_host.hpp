#pragma once
// Independently linkable host transport policy entry points.
namespace nodals { namespace transport {
struct HostVolumeState {
  double sigmaK,sigmaW,nut,reactK,reactOmega,pk,srcOmega;
  double omegaProductionAudit,omegaCrossAudit;
};
struct HostVolumeResult {double diffusion,reaction,source,omegaProductionAudit,omegaCrossAudit;};
struct HostRelaxResult {double delta,diagonal;};
HostVolumeResult evaluate_volume_host(int equation,int kEquation,double wt,const HostVolumeState&);
HostRelaxResult evaluate_relax_host(double unrelaxedDiagonal,double rowAbsSum,double alpha);
}} // nodals::transport
