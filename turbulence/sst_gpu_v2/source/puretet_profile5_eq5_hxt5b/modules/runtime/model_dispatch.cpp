#include "model_dispatch.hpp"
#include <stdexcept>
namespace nodals { namespace coupling {
void dispatch_model_step(TurbulenceKind kind,void* context,void (*callback)(void*)) {
  switch (kind) {
    case TurbulenceKind::Laminar: return;
    case TurbulenceKind::SST2003m:
      if (!callback || !context) throw std::invalid_argument("missing turbulence callback");
      callback(context); return;
  }
  throw std::invalid_argument("unknown turbulence kind");
}
}} // nodals::coupling
