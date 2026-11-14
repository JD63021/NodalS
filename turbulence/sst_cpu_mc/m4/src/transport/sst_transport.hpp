#pragma once
#include "../mesh/tet_mesh.hpp"
#include "../parallel/rcb.hpp"
#include "../case/case_config.hpp"
#include "../operators/p1bf3_p0.hpp"
#include "../wall/wall_sampler.hpp"
#include "../enrichment/enrichment_space.hpp"
#include <array>
#include <vector>
namespace nodals::cpu_mc::m4 {
using namespace nodals::cpu_mc::m2;
using nodals::cpu_mc::m3::CaseConfig;
struct SSTFields {
 std::vector<double> k,omega;
 double kIn=0,omegaIn=0;
};
struct ScalarReport {double before=0,after=0;int sweeps=0;int clips=0;};
// Mutable CPU owner of the scalar state. P1+BF3 enriched rows are assembled
// but BF3 scalar coefficients are masked at zero, exactly as GPU SST_STATE_SPACE=q1.
class SSTTransport {
 struct RayPoint{int wallCell=-1,face=-1,localFace=-1,q=-1,hitCell=-1;std::array<double,4> bary{};double y=0;};
 const Mesh&m_; const Ownership&owners_;const CaseConfig&cfg_;int rank_,nv_,ne_;
 const EnrichmentSpace*enrich_=nullptr;
 std::vector<double> distance_;
 std::vector<RayPoint> wallPoints_;
 SSTFields state_;
 double bulk_=0;
 ScalarReport advance_impl(bool omegaEq,const std::vector<double>&u,const nodals::cpu_mc::m3::WallFaceFit*fit);
public:
 SSTTransport(const Mesh&m,const Ownership&o,const CaseConfig&cfg,int rank,
             const std::vector<double>&wallDistance,
             const EnrichmentSpace*enrich=nullptr);
 const SSTFields&state()const{return state_;}
 const std::vector<double>&wall_distance()const{return distance_;}
 // assembly uses real corrected velocity and both turbulence fields
 ScalarReport advance_k(const std::vector<double>&physicalVelocity);
 ScalarReport advance_omega(const std::vector<double>&physicalVelocity,const nodals::cpu_mc::m3::WallFaceFit&fit);
 double mean_nut(const std::vector<double>&velocity)const;
 double cell_nut(int c,const std::vector<double>&velocity)const;
 void refresh_wall_rays();
};
// GPU SST-2003m eddy viscosity at a tetrahedral barycentric quadrature point.
double sst_nut_at_point(const Mesh&m,int c,const std::array<double,4>&lambda,
    const std::vector<double>&u,const std::vector<double>&k,
    const std::vector<double>&omega,const std::vector<double>&distance,
    const CaseConfig&cfg,const EnrichmentSpace*enrich=nullptr);
// GPU exact Keast-24 momentum turbulent viscosity increment.
void momentum_turbulent_deviatoric(Element24&out,const Mesh&m,int c,
    const std::vector<double>&u,const std::vector<double>&k,
    const std::vector<double>&omega,const std::vector<double>&distance,
    const CaseConfig&cfg,const EnrichmentSpace*enrich=nullptr);
}
