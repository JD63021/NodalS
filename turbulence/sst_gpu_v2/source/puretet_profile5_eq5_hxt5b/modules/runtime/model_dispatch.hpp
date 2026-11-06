#pragma once
// Stage16 independently linked turbulence kind dispatch. Host-side only.
// Both laminar and SST use the same model-call boundary; geometry is not owned here.
#include <optional>
#include <type_traits>
#include <utility>
namespace nodals { namespace coupling {
enum class TurbulenceKind : int { Laminar=0, SST2003m=1 };
// Synchronous call: callbacks and context never outlive this function.
void dispatch_model_step(TurbulenceKind kind, void* context, void (*callback)(void*));

template<class Fn>
inline auto step_model(TurbulenceKind kind,int iteration,Fn&& fn)
    -> std::optional<decltype(fn(iteration))> {
  using Result=decltype(fn(iteration));
  using Functor=std::remove_reference_t<Fn>;
  std::optional<Result> result;
  struct Context { Functor* functor; int iteration; std::optional<Result>* result; };
  Context context{&fn,iteration,&result};
  dispatch_model_step(kind,&context,+[](void* raw) {
    auto* ctx=static_cast<Context*>(raw);
    ctx->result->emplace((*ctx->functor)(ctx->iteration));
  });
  return result;
}
}} // nodals::coupling
