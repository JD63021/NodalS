#pragma once
#include "../mesh/tet_mesh.hpp"
#include "../parallel/rcb.hpp"
#include "../case/case_config.hpp"
#include <vector>

namespace nodals::cpu_mc::m3 {
// Stage22D F2: one representative, area-averaged Spalding beta per wall face,
// evaluated from twelve certified inward wall-normal rays and the lagged
// P1+BF3 velocity. No reconstructed/zero-velocity wall surrogate is allowed.
struct WallFaceFit {
  std::vector<double> beta, uTau, yPlus;
  // Full GPU face-quadrature wall state: 12 Spalding fits per wall triangle.
  std::vector<double> qBeta, qUTau, qYPlus;
  std::size_t rays=0, crossed=0, shortened=0;
};
WallFaceFit sample_spalding_faces(const m2::Mesh &mesh, const m2::Ownership &own,
  int rank, const CaseConfig &cfg, const std::vector<double>&velocity);
}
