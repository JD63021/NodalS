#pragma once
// NodalS Stage15: a CPU/CUDA-neutral host orchestration boundary.
// No CUDA calls, mesh ownership, pipe geometry or linear-algebra implementation.
#include <cmath>
#include "../../runtime/model_dispatch.hpp"
namespace nodals { namespace coupling {
struct SSTUpdates {
  double kRel, omegaRel, nutRel, kFloorFrac, omegaFloorFrac;
};
struct GateInputs {
  double activeContinuityL2, activeContinuityMax, continuityAbsTolerance;
  SSTUpdates turbulence;
  double turbulenceRelTolerance;
  double momentumAbs[3], momentumAbsTolerance;
  bool requirePhysicalPlateau;
  int plateauCount, plateauMinimum;
  double spanF, spanY, spanDP, plateauRelTolerance;
};
struct GateResult {
  bool massOk, turbulenceOk, momentumOk, plateauOk, physicalOk, converged;
};
bool mass_ok(double continuityL2,double continuityMax,double atol);
bool turbulence_ok(const SSTUpdates& s,double tol);
bool plateau_ok(int count,int minimum,double spanF,double spanY,double spanDP,double tol);
bool momentum_ok(double x,double y,double z,double tol);
GateResult evaluate_gate(const GateInputs& x);
// Host-side SST callback uses a separately compiled model dispatcher.
// The return type and invocation order are unchanged for the active SST path.
template<class AdvanceTurbulence>
inline auto advance_turbulence(int outer,AdvanceTurbulence&& call)
    -> decltype(call(outer)) {
  auto result=step_model(TurbulenceKind::SST2003m,outer,std::forward<AdvanceTurbulence>(call));
  return std::move(result.value());
}
}} // nodals::coupling
