#include "portable_probe.hpp"
// These are the original GPU solver's compiled, portable mathematical APIs.
// No SST equations or wall laws have been duplicated in the CPU project.
#include "sst2003m_host.hpp"
#include "wall_host.hpp"
#include "transport_host.hpp"
#include "coupling_host.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nodals::cpu_mc {
PortableProbeResult run_portable_probe(double nu) {
  using namespace nodals;
  PortableProbeResult result{};
  if (!(std::isfinite(nu) && nu>0.0)) return result;

  // Component-only probe. The existing evaluate_host helper is pipe-distance
  // based, so this does NOT assert general-geometry numerical parity.
  sst2003m::HostInput model{};
  model.x=0.0;model.y=0.0;model.radius=0.004;
  model.nu=nu;
  model.k=0.03;model.omega=120.0;
  model.gu[1]=75.0;model.gu[3]=-18.0;
  const auto phys=sst2003m::evaluate_host(model);
  result.nut=phys.nut;

  // Make synthetic Spalding-consistent five-point samples, to exercise the
  // *unchanged* portable five-point fit, rather than test arbitrary data.
  const double y[5]={8e-5,12e-5,16e-5,20e-5,24e-5};
  const double utRef=0.14;
  double u[5]{};
  for(int i=0;i<5;++i) {
    const double target=y[i]*utRef/nu;
    double left=0.0,right=80.0;
    for(int step=0;step<70;++step) {
      const double mid=(left+right)/2.0;
      if(wall::spy(mid,0.4,5.5)<target)left=mid;
      else right=mid;
    }
    u[i]=0.5*(left+right)*utRef;
  }
  const auto wp=wall::fit_profile5_host(u,u,y,nu);
  result.wall_utau=wp.uTau;

  wall::HostOFInput omegaWall{};
  omegaWall.y=y[2];omegaWall.nu=nu;omegaWall.Ut=u[2];
  omegaWall.uTau=wp.uTau;omegaWall.k=0.03;omegaWall.omega=120.0;
  omegaWall.kFloor=1e-8;omegaWall.omegaFloor=1e-3;
  omegaWall.kappa=0.4;omegaWall.E=9.025013499434122;
  omegaWall.beta1=0.075;omegaWall.reBlend=11.0;
  omegaWall.prodLimit=10.0;omegaWall.pk=phys.pk;omegaWall.nut=phys.nut;
  omegaWall.sharedSpaldingWallState=1;
  const auto ow=wall::evaluate_of_auto_host(omegaWall);
  result.omega_target=ow.omegaTarget;

  transport::HostVolumeState state{};
  state.sigmaK=phys.sigmaK;state.sigmaW=phys.sigmaW;
  state.nut=phys.nut;state.reactK=phys.reactK;state.reactOmega=phys.reactOmega;
  state.pk=phys.pk;state.srcOmega=phys.srcOmega;
  state.omegaProductionAudit=phys.omegaProductionAudit;
  state.omegaCrossAudit=phys.omegaCrossAudit;
  const auto vol=transport::evaluate_volume_host(0,0,2.5,state);
  const auto relax=transport::evaluate_relax_host(3.0,8.0,0.7);
  result.transport_diffusion=vol.diffusion;

  coupling::GateInputs gate{};
  gate.activeContinuityL2=1e-11; gate.activeContinuityMax=1e-10;
  gate.continuityAbsTolerance=1e-8;
  gate.turbulence={1e-5,1e-5,1e-5,0.0,0.0};
  gate.turbulenceRelTolerance=1e-3;
  gate.momentumAbs[0]=1e-7;gate.momentumAbs[1]=1e-7;
  gate.momentumAbs[2]=1e-7;gate.momentumAbsTolerance=1e-4;
  gate.requirePhysicalPlateau=false;
  gate.plateauCount=5;gate.plateauMinimum=5;
  gate.spanF=1e-5;gate.spanY=1e-5;gate.spanDP=1e-5;
  gate.plateauRelTolerance=1e-3;
  const auto status=coupling::evaluate_gate_host(gate);
  const auto callback=coupling::advance_turbulence(4,[](int i){return i+5;});

  result.passed =
      std::isfinite(result.nut) && result.nut>=0.0 &&
      wp.valid && std::isfinite(result.wall_utau) && result.wall_utau>0.0 &&
      std::fabs(result.wall_utau-utRef) < 0.05*utRef &&
      std::isfinite(result.omega_target) && result.omega_target>0.0 &&
      std::isfinite(vol.diffusion) &&
      std::fabs(vol.diffusion-2.5*state.sigmaK*state.nut)<1e-12 &&
      std::isfinite(relax.delta) && std::isfinite(relax.diagonal) &&
      std::fabs(relax.delta-(1.0/0.7-1.0)*8.0)<1e-12 &&
      status.converged && callback==9;
  return result;
}
} // namespace nodals::cpu_mc
