#pragma once
namespace nodals::cpu_mc {
struct PortableProbeResult {
  bool passed=false;
  double nut=0.0, wall_utau=0.0, omega_target=0.0, transport_diffusion=0.0;
};
PortableProbeResult run_portable_probe(double nu);
}
