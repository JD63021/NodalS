#pragma once
// Numerical audit of the ACTUAL full P1+BF3+Spalding velocity divergence.
// Uses the same 5^3 / 6x6xradial quadrature as the discrete grad-div forms.
// No changes to operators or solution. One-rank small-mesh diagnostic.
#include "../enrichment/enrichment_space.hpp"
#include "../case/case_config.hpp"
#include <array>
#include <vector>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <stdexcept>
#include <limits>
namespace nodals::cpu_mc::m4 {
inline void print_grad_div_audit(const m2::Mesh&m,const EnrichmentSpace&space,
 const std::vector<double>&u,const m3::CaseConfig&cfg,int it,int rank){
 if(rank!=0||cfg.expectRanks!=1)return;
 const m2::Vec3 Uin{cfg.inlet[0],cfg.inlet[1],cfg.inlet[2]};
 const double ub=std::sqrt(dot(Uin,Uin));
 const double D=cfg.diameter;
 if(!(ub>0&&D>0))throw std::runtime_error("grad-div audit invalid inlet or diameter");
 const int order=cfg.enrichRadialQuad;
 std::vector<double>x,w;
 std::vector<double>x6,w6;
 EnrichmentSpace::gauss(order,x,w);
 EnrichmentSpace::gauss(6,x6,w6);
 constexpr double X5[5]={.0469100770306680185,.23076534494715845,.5,.7692346550528415,.95308992296933193};
 constexpr double W5[5]={.11846344252809449,.23931433524968326,.28444444444444450,.23931433524868326,.11846344252809449};
 double integratedDiv2=0,vol=0,absMax=0;
 double partialDiv2[2]{},partialVol[2]{};
 double enrichedDiv2=0,enrichedVol=0;
 int peakCell=-1;
 for(std::size_t c=0;c<m.tets.size();++c){
  const auto&t=m.tets[c];
  const auto&p0=m.points[(size_t)t.v[0]];
  const double det=dot(m.points[(size_t)t.v[1]]-p0,
     cross(m.points[(size_t)t.v[2]]-p0,m.points[(size_t)t.v[3]]-p0));
  if(!(det>0))throw std::runtime_error("grad-div audit encountered invalid tetrahedron");
  bool enriched=false;
  if(space.enabled)for(int j=0;j<4;++j)
    enriched|=space.index[(size_t)t.v[j]]>=0;
  const int nr=enriched?6:5,ns=enriched?6:5,nz=enriched?order:5;
  for(int ir=0;ir<nr;++ir)for(int is=0;is<ns;++is)for(int iz=0;iz<nz;++iz){
   const double a=enriched?x6[(size_t)ir]:X5[ir];
   const double b=enriched?x6[(size_t)is]:X5[is];
   const double z=enriched?x[(size_t)iz]:X5[iz];
   const double ar=enriched?w6[(size_t)ir]:W5[ir];
   const double bs=enriched?w6[(size_t)is]:W5[is];
   const double cz=enriched?w[(size_t)iz]:W5[iz];
   const double wt=det*ar*bs*cz*(1-a)*(1-a)*(1-b);
   std::array<double,4> bary{(1-a)*(1-b)*(1-z),a,(1-a)*b,(1-a)*(1-b)*z};
   double phi[12];m2::Vec3 grad[12];space.shape((int)c,bary,phi,grad);
   double divU=0,y=0;
   for(int j=0;j<4;++j)y+=bary[j]*space.distance[(size_t)t.v[j]];
   for(int j=0;j<12;++j){
    const int e=space.entity((int)c,j);
    if(e<0)continue;
    const m2::Vec3 v{u[3*(size_t)e],u[3*(size_t)e+1],u[3*(size_t)e+2]};
    divU+=dot(v,grad[j]);
   }
   if(!(std::isfinite(divU)&&std::isfinite(wt)&&wt>0))
    throw std::runtime_error("grad-div audit nonfinite divergence or weight");
   const double integral=wt*divU*divU;
   integratedDiv2+=integral;vol+=wt;
   const int near=y<.1*D?0:1;
   partialDiv2[near]+=integral;partialVol[near]+=wt;
   if(enriched){enrichedDiv2+=integral;enrichedVol+=wt;}
   if(std::abs(divU)>absMax){absMax=std::abs(divU);peakCell=(int)c;}
  }
 }
 if(!(vol>0))throw std::runtime_error("grad-div audit volume zero");
 const double scale=D/ub;
 std::printf("NODALS_CPU_M4_GRADDIV_AUDIT it=%d gamma=%.12g divL2=%.12e divRMS=%.12e divRMS_D_over_U=%.12e nearWallRMS_D_over_U=%.12e coreRMS_D_over_U=%.12e enrichedRMS_D_over_U=%.12e maxAbsDiv_D_over_U=%.12e maxAbsDivCell=%d volume=%.12e status=DIAGNOSTIC\n",
  it,cfg.gradDivGamma,std::sqrt(integratedDiv2),std::sqrt(integratedDiv2/vol),
  scale*std::sqrt(integratedDiv2/vol),
  partialVol[0]>0?scale*std::sqrt(partialDiv2[0]/partialVol[0]):0,
  partialVol[1]>0?scale*std::sqrt(partialDiv2[1]/partialVol[1]):0,
  enrichedVol>0?scale*std::sqrt(enrichedDiv2/enrichedVol):0,
  scale*absMax,peakCell,vol);
}
}
