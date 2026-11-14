#pragma once
// Pointwise high-Re logarithmic equilibrium k from the SAME Spalding
// friction velocity already fitted by the velocity/omega wall sampler.
// Independent of MPI, mesh geometry and CUDA; use only at off-wall samples.
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace nodals::cpu_mc::m4 {
inline double spalding_log_equilibrium_k(double uTau,double floorK){
 if(!(std::isfinite(uTau)&&uTau>=0&&std::isfinite(floorK)&&floorK>=0))
   throw std::runtime_error("invalid Spalding uTau / equilibrium k floor");
 return std::max(floorK,uTau*uTau/std::sqrt(0.09));
}
// Weak penalty on a scalar field at one interior quadrature point:
// gamma*(nu + sigma*nu_t)*wall_area*face_weight/(sample_y5-sample_y1).
inline double spalding_offwall_k_penalty_coeff(double gamma,double nuEff,
 double wallArea,double quadratureWeight,double sampleSpan){
 if(!(gamma>0&&nuEff>0&&wallArea>0&&quadratureWeight>0&&sampleSpan>0&&
      std::isfinite(gamma)&&std::isfinite(nuEff)&&std::isfinite(wallArea)&&
      std::isfinite(quadratureWeight)&&std::isfinite(sampleSpan)))
   throw std::runtime_error("invalid offwall equilibrium k penalty geometry");
 return gamma*nuEff*wallArea*quadratureWeight/sampleSpan;
}
} // namespace nodals::cpu_mc::m4
