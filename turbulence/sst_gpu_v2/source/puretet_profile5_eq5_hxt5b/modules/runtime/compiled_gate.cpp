// Stage16: moved original Stage15 gate arithmetic out of the application TU.
// Each function below is copied from the accepted Stage15 header with only
// its `inline` keyword removed. No threshold or operand is changed.
#include "../coupling/portable/coupling_core.hpp"
namespace nodals { namespace coupling {
bool mass_ok(double continuityL2,double continuityMax,double atol) {
  return continuityL2<=atol&&continuityMax<=atol;
}
bool turbulence_ok(const SSTUpdates& s,double tol) {
  return s.kFloorFrac<.01&&s.omegaFloorFrac<.01&&
    std::isfinite(s.kRel)&&std::isfinite(s.omegaRel)&&std::isfinite(s.nutRel)&&
    s.kRel<=tol&&s.omegaRel<=tol&&s.nutRel<=tol;
}
bool plateau_ok(int count,int minimum,double spanF,double spanY,double spanDP,double tol) {
  return count>=minimum&&spanF<=tol&&spanY<=tol&&spanDP<=tol;
}
bool momentum_ok(double x,double y,double z,double tol) {
  return tol<=0.0||(x<=tol&&y<=tol&&z<=tol);
}
GateResult evaluate_gate(const GateInputs& x) {
  GateResult r;
  r.massOk=mass_ok(x.activeContinuityL2,x.activeContinuityMax,x.continuityAbsTolerance);
  r.turbulenceOk=turbulence_ok(x.turbulence,x.turbulenceRelTolerance);
  r.momentumOk=momentum_ok(x.momentumAbs[0],x.momentumAbs[1],x.momentumAbs[2],x.momentumAbsTolerance);
  r.plateauOk=plateau_ok(x.plateauCount,x.plateauMinimum,x.spanF,x.spanY,x.spanDP,x.plateauRelTolerance);
  r.physicalOk=!x.requirePhysicalPlateau||r.plateauOk;
  r.converged=r.physicalOk&&r.massOk&&r.turbulenceOk&&r.momentumOk;
  return r;
}
}} // nodals::coupling
