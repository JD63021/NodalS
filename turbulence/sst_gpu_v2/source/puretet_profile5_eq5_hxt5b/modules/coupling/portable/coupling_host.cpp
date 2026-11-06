#include "coupling_host.hpp"
namespace nodals { namespace coupling {
GateResult evaluate_gate_host(const GateInputs& x) { return evaluate_gate(x); }
}} // nodals::coupling
