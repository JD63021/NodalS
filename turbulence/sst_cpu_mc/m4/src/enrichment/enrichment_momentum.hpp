#pragma once
// Additional Galerkin blocks PE, EP, EE for the conforming shared Spalding
// enrichment. The original PP blocks are assembled by existing M4 operators,
// using the total enriched SST gradient and advecting velocity.
// This file never adds an effective wall viscosity or Robin wall drag.
#include "enrichment_space.hpp"
#include "../transport/sst_transport.hpp"
#include <array>
#include <cmath>
#include <stdexcept>
namespace nodals::cpu_mc::m4 {
using K36=std::array<std::array<double,36>,36>;
using RHS36=std::array<double,36>;
inline void augment_momentum(const EnrichmentSpace&enrich,const Mesh&m,int cell,
    const std::vector<double>&lag,const std::vector<double>&k,
    const std::vector<double>&omega,const std::vector<double>&distance,
    const m3::CaseConfig&cfg,K36&EK,RHS36&ER){
  if(!enrich.enabled)return;
  const auto&t=m.tets[(size_t)cell];
  const Vec3 p0=m.points[(size_t)t.v[0]],p1=m.points[(size_t)t.v[1]],
             p2=m.points[(size_t)t.v[2]],p3=m.points[(size_t)t.v[3]];
  const double det=dot(p1-p0,cross(p2-p0,p3-p0));
  if(!(det>0))throw std::runtime_error("enrichment: negative oriented tetrahedron");
  // Selectively increase wall-normal quadrature ONLY on enriched cells.
  // Use 6x6xq Duffy integration; q=24 by default (configurable).
  const int order=cfg.enrichRadialQuad;
  std::vector<double>x,w;EnrichmentSpace::gauss(order,x,w);
  std::vector<double>xr,wr;EnrichmentSpace::gauss(6,xr,wr);
  for(int ir=0;ir<6;++ir)for(int is=0;is<6;++is)for(int iz=0;iz<order;++iz){
    const double a=xr[ir],b=xr[is],c=x[iz],oa=1-a,ob=1-b;
    const std::array<double,4> l{oa*ob*(1-c),a,oa*b,oa*ob*c};
    double phi[12];Vec3 gp[12];enrich.shape(cell,l,phi,gp);
    bool present=false;for(int i=8;i<12;++i)present|=enrich.entity(cell,i)>=0;
    if(!present)continue;
    const double nuT=sst_nut_at_point(m,cell,l,lag,k,omega,distance,cfg,&enrich);
    const double nueff=cfg.nu+nuT;
    if(!(nueff>0)||!std::isfinite(nueff))throw std::runtime_error("enrichment invalid SST viscosity");
    const Vec3 U=enrich.velocity(cell,l,lag);
    const double wt=det*wr[ir]*wr[is]*w[iz]*oa*oa*ob;
    for(int ia=0;ia<12;++ia)if(enrich.entity(cell,ia)>=0)
     for(int ib=0;ib<12;++ib)if(enrich.entity(cell,ib)>=0 && (ia>=8||ib>=8)){
      const Vec3 ga=gp[ia],gb=gp[ib];
      const double aa[3]={ga.x,ga.y,ga.z},bb[3]={gb.x,gb.y,gb.z};
      const double gradDot=dot(ga,gb),advection=wt*phi[ia]*dot(U,gb);
      for(int i=0;i<3;++i)for(int j=0;j<3;++j)
       EK[3*ia+i][3*ib+j]+=wt*nueff*((i==j?gradDot:0)+
         aa[j]*bb[i]-(2./3.)*aa[i]*bb[j])+(i==j?advection:0.0);
    }
  }
  // Original inlet is DG prescribed velocity and is not a no-slip wall.
  // Enrichment traces at inlet may be nonzero, so add matching PE/EE
  // inlet Nitsche and prescribed inflow terms at elevated quadrature.
  const int nv=(int)m.points.size();
  const double bulk[3]={cfg.inlet[0],cfg.inlet[1],cfg.inlet[2]};
  const Vec3 Uin{bulk[0],bulk[1],bulk[2]};
  for(int lf=0;lf<4;++lf){
    const auto&f=m.faces[(size_t)t.face[lf]];if(f.patch!=1)continue;
    const Vec3 area=outward_area(m,cell,lf);
    const double ar=std::sqrt(dot(area,area));
    const Vec3 n=(1./ar)*area;
    const double inflow=std::max(0.,-dot(Uin,n));
    int js[3],v=0;for(int a=0;a<4;++a)if(a!=lf)js[v++]=a;
    std::vector<double>xf,wf;EnrichmentSpace::gauss(12,xf,wf);
    for(int iq=0;iq<12;++iq)for(int jq=0;jq<12;++jq){
      const double a=xf[iq],b=xf[jq];
      std::array<double,4> l{};
      l[js[0]]=a;l[js[1]]=(1-a)*b;l[js[2]]=(1-a)*(1-b);
      double ph[12];Vec3 g[12];enrich.shape(cell,l,ph,g);
      const double nuT=sst_nut_at_point(m,cell,l,lag,k,omega,distance,cfg,&enrich);
      const double wt=2*ar*wf[iq]*wf[jq]*(1-a);
      const double nueff=cfg.nu+nuT;
      for(int i=0;i<12;++i)if(enrich.entity(cell,i)>=0){
       const double dni=dot(g[i],n);
       if(i>=8)for(int d=0;d<3;++d)ER[3*i+d]+=wt*(inflow*ph[i]+nueff*dni)*bulk[d];
       for(int j=0;j<12;++j)if(enrich.entity(cell,j)>=0&&(i>=8||j>=8)){
         const double val=wt*(inflow*ph[i]*ph[j]+nueff*(-ph[i]*dot(g[j],n)+dni*ph[j]));
         for(int d=0;d<3;++d)EK[3*i+d][3*j+d]+=val;
       }
      }
    }
  }
}
} // namespace nodals::cpu_mc::m4
