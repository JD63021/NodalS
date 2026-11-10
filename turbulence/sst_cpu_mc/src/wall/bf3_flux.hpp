#pragma once
#include "../mesh/tet_mesh.hpp"
#include <vector>
namespace nodals::cpu_mc::m3 {
using namespace nodals::cpu_mc::m2;
// GPU stage22d4i T, Q, T^T; acts on 3-vector coefficients, interleaved.
class BF3Flux {
 const Mesh& m_;
 struct WallFace{int bubble;int v[3];Vec3 n;double area;};
 std::vector<WallFace> wall_;
public:
 explicit BF3Flux(const Mesh& mesh);
 void Q(std::vector<double>&x)const;
 void T(std::vector<double>&x)const;
 void Tt(std::vector<double>&x)const;
 double max_mean_flux(const std::vector<double>&x)const;
 std::size_t count()const{return wall_.size();}
};
}
