#pragma once
// POD GPU transfer type. std::array::operator[] is not guaranteed __device__
// on every nvcc / libstdc++ pair, so never use Tet10 (=std::array) in a kernel.
#include "bernstein_p2_tet.hpp"
#include <type_traits>
namespace nodals { namespace p2 {
struct Tet10Device { std::int32_t g[10]; };
static_assert(sizeof(Tet10Device)==10*sizeof(std::int32_t),"P2 packed tet layout");
static_assert(std::is_trivially_copyable<Tet10Device>::value,"P2 device connectivity POD");
inline std::vector<Tet10Device> pack_device_tetra(const Connectivity&C){
 std::vector<Tet10Device> out(C.tetra.size());
 for(std::size_t i=0;i<out.size();++i)
  for(int a=0;a<10;++a)out[i].g[a]=C.tetra[i][a];
 return out;
}
}} // namespace nodals::p2
