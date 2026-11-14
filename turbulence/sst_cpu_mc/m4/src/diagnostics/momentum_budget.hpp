#pragma once
// Passive diagnostic: surface-integrated RANS momentum budget on an MSH2
// tetrahedral domain. Never alters SIMPLE, turbulence, or wall coefficients.
// Integrates full enriched velocity gradients, not just vertex VTU values.
#include "../enrichment/enrichment_space.hpp"
#include "../transport/sst_transport.hpp"
#include "../case/case_config.hpp"
#include <array>
#include <vector>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <stdexcept>
namespace nodals::cpu_mc::m4 {
struct AxialBudget {
 double pforce=0, convOut=0, convInComputed=0,convInPrescribed=0;
 double viscAll=0, viscWall=0, viscEnd=0, viscMolecularWall=0;
 double wallArea=0,wallTauMolAbs=0,wallTauEffAbs=0,wallNutArea=0;
 double wallKArea=0,wallOmegaArea=0, wallNutMax=0;
 double massInComputed=0,massInPrescribed=0,massOut=0,massWall=0;
 int facesWall=0,facesIn=0,facesOut=0;
};
inline void print_momentum_budget(const m2::Mesh&m,const EnrichmentSpace&enrich,
 const SSTTransport&sst,const std::vector<double>&u,const std::vector<double>&p,
 const m3::CaseConfig&cfg,int it,int rank){
 if(rank!=0)return; // one-rank diagnostic path, user test is -np 1.
 if(cfg.expectRanks!=1)return;
 const double inletZ=cfg.inlet[2],inletU2=cfg.inlet[0]*cfg.inlet[0]+cfg.inlet[1]*cfg.inlet[1]+cfg.inlet[2]*cfg.inlet[2];
 if(!(inletU2>0))throw std::runtime_error("budget: zero inlet velocity");
 if((int)u.size()!=3*enrich.nent()||p.size()!=m.tets.size())
  throw std::runtime_error("budget: bad velocity/pressure sizes");
 std::vector<double> q,w;EnrichmentSpace::gauss(6,q,w);
 AxialBudget b;
 const auto&k=sst.state().k;
 const auto&omega=sst.state().omega;
 for(const auto&f:m.faces){
  if(f.patch!=1&&f.patch!=2&&f.patch!=3)continue;
  const int c=f.cell0;
  if(c<0)throw std::runtime_error("budget: no boundary cell");
  int lf=-1;
  const auto&t=m.tets[(size_t)c];
  for(int a=0;a<4;++a)if(t.face[a]==(int)(&f-&m.faces[0]))lf=a;
  if(lf<0)throw std::runtime_error("budget: cannot identify boundary tet face");
  const Vec3 arVec=outward_area(m,c,lf);
  const double ar=std::sqrt(dot(arVec,arVec));
  if(!(ar>0))throw std::runtime_error("budget: zero wall face");
  const Vec3 n=(1./ar)*arVec;
  const int patch=f.patch;
  if(patch==3)++b.facesWall;else if(patch==1)++b.facesIn;else ++b.facesOut;
  int v[3],nloc=0;for(int i=0;i<4;++i)if(i!=lf)v[nloc++]=i;
  for(int iq=0;iq<6;++iq)for(int jq=0;jq<6;++jq){
   const double a=q[iq],z=q[jq],wt=2.*ar*w[iq]*w[jq]*(1-a);
   std::array<double,4> l{};
   l[v[0]]=a;l[v[1]]=(1-a)*z;l[v[2]]=(1-a)*(1-z);
   double ph[12];Vec3 gg[12];enrich.shape(c,l,ph,gg);
   Vec3 velocity{};double grad[3][3]{};
   for(int mode=0;mode<12;++mode){
    const int entity=enrich.entity(c,mode);if(entity<0)continue;
    const double coeff[3]={u[3*(size_t)entity],u[3*(size_t)entity+1],u[3*(size_t)entity+2]};
    const double g[3]={gg[mode].x,gg[mode].y,gg[mode].z};
    velocity.x+=ph[mode]*coeff[0];velocity.y+=ph[mode]*coeff[1];velocity.z+=ph[mode]*coeff[2];
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)grad[i][j]+=coeff[i]*g[j];
   }
   const double nuT=sst_nut_at_point(m,c,l,u,k,omega,sst.wall_distance(),cfg,&enrich);
   if(!(nuT>=0&&std::isfinite(nuT)))throw std::runtime_error("budget: nonfinite eddy viscosity at boundary");
   const double div=grad[0][0]+grad[1][1]+grad[2][2];
   double Dn[3]{};
   const double nn[3]={n.x,n.y,n.z};
   for(int i=0;i<3;++i)for(int j=0;j<3;++j)
     Dn[i]+=(grad[i][j]+grad[j][i]-(i==j?2./3.*div:0.))*nn[j];
   const Vec3 vTang{Dn[0]-dot({Dn[0],Dn[1],Dn[2]},n)*n.x,
                    Dn[1]-dot({Dn[0],Dn[1],Dn[2]},n)*n.y,
                    Dn[2]-dot({Dn[0],Dn[1],Dn[2]},n)*n.z};
   const double shearShape=std::sqrt(dot(vTang,vTang));
   b.pforce-=wt*p[(size_t)c]*n.z;
   b.viscAll+=wt*(cfg.nu+nuT)*Dn[2];
   if(patch==3){
    b.wallArea+=wt;
    b.wallTauMolAbs+=wt*cfg.nu*shearShape;
    b.wallTauEffAbs+=wt*(cfg.nu+nuT)*shearShape;
    b.viscWall+=wt*(cfg.nu+nuT)*Dn[2];
    b.viscMolecularWall+=wt*cfg.nu*Dn[2];
    double kval=0,omval=0;for(int i=0;i<4;++i){kval+=l[i]*k[(size_t)t.v[i]];omval+=l[i]*omega[(size_t)t.v[i]];}
    b.wallKArea+=wt*kval;b.wallOmegaArea+=wt*omval;
    b.wallNutArea+=wt*nuT;b.wallNutMax=std::max(b.wallNutMax,nuT);
    b.massWall+=wt*dot(velocity,n);
   }else {
    b.viscEnd+=wt*(cfg.nu+nuT)*Dn[2];
    if(patch==1){
     b.massInComputed+=wt*dot(velocity,n);
     b.massInPrescribed+=wt*(cfg.inlet[0]*n.x+cfg.inlet[1]*n.y+cfg.inlet[2]*n.z);
     b.convInComputed+=wt*velocity.z*dot(velocity,n);
     b.convInPrescribed+=wt*inletZ*(cfg.inlet[0]*n.x+cfg.inlet[1]*n.y+cfg.inlet[2]*n.z);
    }else{
     b.massOut+=wt*dot(velocity,n);
     b.convOut+=wt*velocity.z*dot(velocity,n);
    }
   }
  }
 }
 if(!(b.wallArea>0))throw std::runtime_error("budget: no wall area");
 const double wallMol=b.wallTauMolAbs/b.wallArea, wallEff=b.wallTauEffAbs/b.wallArea;
 const double meanNut=b.wallNutArea/b.wallArea;
 const double rhsBudget=b.pforce+b.viscAll-(b.convOut+b.convInPrescribed);
 const double denom=std::max({1.,std::abs(b.pforce),std::abs(b.viscAll),std::abs(b.convOut+b.convInPrescribed)});
 const double fMol=8*wallMol/inletU2,fEff=8*wallEff/inletU2;
 std::printf("NODALS_CPU_M4_RANS_WALL_BUDGET it=%d meanTauMol=%.12e meanTauEff=%.12e tauEffToMol=%.9g fMol=%.9g fEff=%.9g meanNutNu=%.9g maxNutNu=%.9g wallK=%.9g wallOmega=%.9g wallArea=%.12e status=DIAGNOSTIC\n",
  it,wallMol,wallEff,wallEff/std::max(wallMol,1e-300),fMol,fEff,meanNut/cfg.nu,b.wallNutMax/cfg.nu,
  b.wallKArea/b.wallArea,b.wallOmegaArea/b.wallArea,b.wallArea);
 std::printf("NODALS_CPU_M4_AXIAL_MOMENTUM_BUDGET it=%d pressureForce=%.12e wallStressZ=%.12e wallStressMolZ=%.12e endStressZ=%.12e totalViscZ=%.12e advOut=%.12e advInPrescribed=%.12e advInComputed=%.12e surfaceMismatch=%.12e mismatchRel=%.9g status=APPROX_BOUNDARY_P0\n",
  it,b.pforce,b.viscWall,b.viscMolecularWall,b.viscEnd,b.viscAll,b.convOut,b.convInPrescribed,b.convInComputed,rhsBudget,std::abs(rhsBudget)/denom);
 std::printf("NODALS_CPU_M4_MASS_BOUNDARY it=%d fluxInPrescribed=%.12e fluxInComputed=%.12e fluxOut=%.12e fluxWall=%.12e netWithInletDG=%.12e netWithComputedInlet=%.12e facesWall=%d facesIn=%d facesOut=%d status=DIAGNOSTIC\n",
  it,b.massInPrescribed,b.massInComputed,b.massOut,b.massWall,
  b.massInPrescribed+b.massOut+b.massWall,b.massInComputed+b.massOut+b.massWall,
  b.facesWall,b.facesIn,b.facesOut);
}
} // namespace nodals::cpu_mc::m4
