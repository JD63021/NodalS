#pragma once
// Passive diagnostics to locate the onset of enriched high-Re pipe instability.
// Compute inlet trace errors and interior axial-bin peaks from FULL enriched
// P1+BF3+Spalding velocity; output VTU only contains vertex velocities.
// Intended for small one-rank experimental cases. No operator modifications.
#include "../mesh/tet_mesh.hpp"
#include "../enrichment/enrichment_space.hpp"
#include "../transport/sst_transport.hpp"
#include "../wall/wall_sampler.hpp"
#include "../case/case_config.hpp"
#include <array>
#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
namespace nodals::cpu_mc::m4 {
inline void print_inlet_localization(const m2::Mesh&m,const EnrichmentSpace&space,
 const SSTTransport&sst,const m3::WallFaceFit&fit,const std::vector<double>&u,
 const m3::CaseConfig&cfg,int it,int rank){
 if(rank!=0||cfg.expectRanks!=1||!space.enabled)return;
 const m2::Vec3 inlet{cfg.inlet[0],cfg.inlet[1],cfg.inlet[2]};
 const double speed=std::sqrt(dot(inlet,inlet));
 if(!(speed>0))throw std::runtime_error("inlet localization: inlet speed is zero");
 const m2::Vec3 axis=(1.0/speed)*inlet;
 double minAx=std::numeric_limits<double>::infinity();
 double maxAx=-std::numeric_limits<double>::infinity();
 for(const auto&p:m.points){
  const double x=dot(p,axis);
  minAx=std::min(minAx,x);
  maxAx=std::max(maxAx,x);
 }
 const double length=maxAx-minAx;
 if(!(length>0))throw std::runtime_error("inlet localization: axial span zero");
 const double D=cfg.diameter;
 // 0-.25D, .25-.5D, .5-1D, 1-2D, 2-4D, remaining.
 constexpr int NZ=6;
 const double cuts[NZ]={0.25,0.5,1.0,2.0,4.0,1.0e100};
 struct Peak{
  double maxSpeed=0,axialMin=std::numeric_limits<double>::infinity();
  double maxAxialError=0;
  double axialAtMax=0,wallDistanceAtMax=0;
  double p1AtMax=0,bf3AtMax=0,enrichAtMax=0,divAtMax=0;
  int cellAtMax=-1,nsamples=0,negativeAxial=0;
 };
 Peak bins[NZ]{};
 // Four vertex-focused barycentres + centre capture both interior and
 // unresolved near-inlet overshoots without assuming a structured mesh.
 constexpr double L[5][4]={
  {0.25,0.25,0.25,0.25},
  {0.70,0.10,0.10,0.10},
  {0.10,0.70,0.10,0.10},
  {0.10,0.10,0.70,0.10},
  {0.10,0.10,0.10,0.70}
 };
 const auto&ks=sst.state().k;
 const auto&ws=sst.state().omega;
 double peakK=0,peakOmega=0,peakKAx=0,peakOmegaAx=0;
 int peakKCell=-1,peakOmegaCell=-1;
 for(size_t c=0;c<m.tets.size();++c){
  const auto&t=m.tets[c];
  for(const auto&l4:L){
   std::array<double,4> l{l4[0],l4[1],l4[2],l4[3]};
   m2::Vec3 p{};double y=0,k=0,omega=0;
   for(int j=0;j<4;++j){
    p=p+l[j]*m.points[(size_t)t.v[j]];
    y+=l[j]*sst.wall_distance()[(size_t)t.v[j]];
    k+=l[j]*ks[(size_t)t.v[j]];
    omega+=l[j]*ws[(size_t)t.v[j]];
   }
   const double axial=dot(p,axis)-minAx;
   if(axial<0)continue;
   int bin=NZ-1;
   for(int j=0;j<NZ;++j)if(axial<=cuts[j]*D){bin=j;break;}
   const auto vel=space.velocity((int)c,l,u);
   const double us=std::sqrt(dot(vel,vel)),ua=dot(vel,axis);
   auto&b=bins[bin];
   ++b.nsamples;b.axialMin=std::min(b.axialMin,ua);
   if(ua<0)++b.negativeAxial;
   b.maxAxialError=std::max(b.maxAxialError,std::abs(ua-speed));
   if(us>b.maxSpeed){
    b.maxSpeed=us;b.axialAtMax=axial;b.wallDistanceAtMax=y;
    b.cellAtMax=(int)c;
    // Decompose actual physical u at this hot point into the three
    // finite-element function-space contributions, not coefficient norms.
    double ph[12];m2::Vec3 grad[12];space.shape((int)c,l,ph,grad);
    m2::Vec3 p1{},bf3{},spalding{};
    double div=0;
    for(int a=0;a<12;++a){
     const int id=space.entity((int)c,a);
     if(id<0)continue;
     const m2::Vec3 dof{u[3*(size_t)id],u[3*(size_t)id+1],u[3*(size_t)id+2]};
     const auto contribution=ph[a]*dof;
     if(a<4)p1=p1+contribution;
     else if(a<8)bf3=bf3+contribution;
     else spalding=spalding+contribution;
     div+=dot(grad[a],dof);
    }
    b.p1AtMax=std::sqrt(dot(p1,p1));
    b.bf3AtMax=std::sqrt(dot(bf3,bf3));
    b.enrichAtMax=std::sqrt(dot(spalding,spalding));
    b.divAtMax=div;
   }
   if(k>peakK){peakK=k;peakKAx=axial;peakKCell=(int)c;}
   if(omega>peakOmega){peakOmega=omega;peakOmegaAx=axial;peakOmegaCell=(int)c;}
  }
 }
 double area[2]{},square[2]{},squareAxial[2]{},actualAxial[2]{},prescribedFlux[2]{},computedFlux[2]{};
 double minAxial[2]={std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()};
 double maxAxial[2]={-std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()};
 int count[2]{};
 std::vector<double> q,w;EnrichmentSpace::gauss(6,q,w);
 for(const auto&f:m.faces){
  if(f.patch!=1)continue;
  const int c=f.cell0;if(c<0)continue;
  const auto&t=m.tets[(size_t)c];
  const int id=(int)(&f-m.faces.data());
  int lf=-1;
  for(int j=0;j<4;++j)if(t.face[j]==id)lf=j;
  if(lf<0)throw std::runtime_error("inlet localization: no local face");
  const m2::Vec3 av=outward_area(m,c,lf);
  const double a=std::sqrt(dot(av,av));const m2::Vec3 n=(1./a)*av;
  int v[3],k=0;for(int j=0;j<4;++j)if(j!=lf)v[k++]=j;
  for(int i=0;i<6;++i)for(int j=0;j<6;++j){
   const double x=q[i],z=q[j];
   std::array<double,4> l{};
   l[v[0]]=x;l[v[1]]=(1.-x)*z;l[v[2]]=(1.-x)*(1.-z);
   double y=0;for(int tloc=0;tloc<4;++tloc)
    y+=l[tloc]*sst.wall_distance()[(size_t)t.v[tloc]];
   const int band=y<=0.10*D?0:1;
   const double wt=2*a*w[i]*w[j]*(1-x);
   const auto vel=space.velocity(c,l,u);
   const auto du=vel-inlet;
   const double ua=dot(vel,axis);
   area[band]+=wt;square[band]+=wt*dot(du,du);
   squareAxial[band]+=wt*(ua-speed)*(ua-speed);
   actualAxial[band]+=wt*ua;
   prescribedFlux[band]+=wt*dot(inlet,n);
   computedFlux[band]+=wt*dot(vel,n);
   minAxial[band]=std::min(minAxial[band],ua);
   maxAxial[band]=std::max(maxAxial[band],ua);
   ++count[band];
  }
 }
 double worstWallUTau=-1,worstWallAxial=0;
 double wallUSum[NZ]{},wallUMax[NZ]{},wallUCount[NZ]{};
 int worstWallFace=-1,worstWallQ=-1;
 for(size_t i=0;i<m.faces.size();++i)if(m.faces[i].patch==3){
  const auto&f=m.faces[i];
  m2::Vec3 center{};
  for(int j=0;j<3;++j)center=center+(1./3.)*m.points[(size_t)f.v[j]];
  const double x=dot(center,axis)-minAx;
  int zone=NZ-1;
  for(int j=0;j<NZ;++j)if(x<=cuts[j]*D){zone=j;break;}
  for(int qid=0;qid<12;++qid){
   const double ut=fit.qUTau[12*i+(size_t)qid];
   wallUSum[zone]+=ut;wallUCount[zone]+=1.;
   wallUMax[zone]=std::max(wallUMax[zone],ut);
   if(ut>worstWallUTau){
    worstWallUTau=ut;worstWallAxial=x;
    worstWallFace=(int)i;worstWallQ=qid;
   }
  }
 }
 std::printf("NODALS_CPU_M4_INLET_LOCALIZATION it=%d Uin=%.9g L=%.9g D=%.9g wallFitLagged=1 peakWallUTau=%.9g peakWallUTauAxialD=%.9g peakWallFace=%d peakWallQ=%d peakK=%.9g peakKAxialD=%.9g peakKCell=%d peakOmega=%.9g peakOmegaAxialD=%.9g peakOmegaCell=%d status=DIAGNOSTIC\n",
   it,speed,length,D,worstWallUTau,worstWallAxial/D,worstWallFace,worstWallQ,
   peakK,peakKAx/D,peakKCell,peakOmega,peakOmegaAx/D,peakOmegaCell);
 for(int j=0;j<2;++j){
  if(!(area[j]>0))continue;
  std::printf("NODALS_CPU_M4_INLET_TRACE_BAND it=%d band=%s area=%.12e rmsVectorErrorOverU=%.9g rmsAxialErrorOverU=%.9g meanAxialOverU=%.9g minAxialOverU=%.9g maxAxialOverU=%.9g fluxPrescribed=%.12e fluxActual=%.12e points=%d status=DIAGNOSTIC\n",
   it,j==0?"WALL_EDGE_y_lt_0.1D":"CORE_y_ge_0.1D",area[j],
   std::sqrt(square[j]/area[j])/speed,
   std::sqrt(squareAxial[j]/area[j])/speed,
   actualAxial[j]/area[j]/speed,minAxial[j]/speed,maxAxial[j]/speed,
   prescribedFlux[j],computedFlux[j],count[j]);
 }
 // The natural outlet can admit backflow; check its signed flux directly.
 double outletArea=0,backArea=0,netOutlet=0,reverseOutlet=0,maxInflow=0;
 int backCount=0,totalCount=0;
 for(const auto&f:m.faces)if(f.patch==2){
  const int c=f.cell0,id=(int)(&f-m.faces.data());
  if(c<0)continue;
  const auto&t=m.tets[(size_t)c];
  int lf=-1;for(int j=0;j<4;++j)if(t.face[j]==id)lf=j;
  if(lf<0)throw std::runtime_error("outlet backflow: missing local face");
  const auto av=outward_area(m,c,lf);
  const double areaFace=std::sqrt(dot(av,av));
  if(!(areaFace>0))throw std::runtime_error("outlet backflow: zero face area");
  const auto n=(1./areaFace)*av;
  int ids[3],jv=0;for(int j=0;j<4;++j)if(j!=lf)ids[jv++]=j;
  for(int i=0;i<6;++i)for(int j=0;j<6;++j){
   const double x=q[i],z=q[j];
   std::array<double,4> l{};
   l[ids[0]]=x;l[ids[1]]=(1-x)*z;l[ids[2]]=(1-x)*(1-z);
   const double wt=2*areaFace*w[i]*w[j]*(1-x);
   const auto vel=space.velocity(c,l,u);
   const double vn=dot(vel,n);
   netOutlet+=wt*vn;outletArea+=wt;++totalCount;
   if(vn<0){reverseOutlet-=wt*vn;backArea+=wt;
    maxInflow=std::max(maxInflow,-vn);++backCount;}
  }
 }
 if(outletArea>0)
  std::printf("NODALS_CPU_M4_OUTLET_BACKFLOW it=%d netFlux=%.12e reverseFluxMagnitude=%.12e backflowAreaFraction=%.9g maxInwardNormalOverU=%.9g reverseQuadratureFraction=%.9g outletSamples=%d status=DIAGNOSTIC\n",
   it,netOutlet,reverseOutlet,backArea/outletArea,maxInflow/speed,
   totalCount>0?(double)backCount/totalCount:0,totalCount);
 for(int j=0;j<NZ;++j)if(wallUCount[j]>0){
  std::printf("NODALS_CPU_M4_WALL_FIT_BIN it=%d zone=%d xUpperD=%.9g meanLaggedUTau=%.9g maxLaggedUTau=%.9g fitSamples=%.0f status=DIAGNOSTIC\n",
   it,j,cuts[j],wallUSum[j]/wallUCount[j],wallUMax[j],wallUCount[j]);
 }
 for(int j=0;j<NZ;++j)if(bins[j].nsamples){
  const auto&b=bins[j];
  std::printf("NODALS_CPU_M4_VELOCITY_MODES it=%d zone=%d maxSpeedOverU=%.9g p1AtMaxOverU=%.9g bf3AtMaxOverU=%.9g spaldingAtMaxOverU=%.9g divAtMaxDOverU=%.9g maxSpeedCell=%d maxSpeedAxialD=%.9g maxSpeedWallDistanceD=%.9g status=DIAGNOSTIC\n",
   it,j,b.maxSpeed/speed,b.p1AtMax/speed,b.bf3AtMax/speed,
   b.enrichAtMax/speed,b.divAtMax*D/speed,
   b.cellAtMax,b.axialAtMax/D,b.wallDistanceAtMax/D);
  std::printf("NODALS_CPU_M4_AXIAL_SPIKE_BIN it=%d zone=%d xUpperD=%.9g maxSpeedOverU=%.9g maxAxialDeviationOverU=%.9g minAxialOverU=%.9g negativeSampleFraction=%.9g maxSpeedCell=%d maxSpeedAxialD=%.9g maxSpeedWallDistanceD=%.9g count=%d status=DIAGNOSTIC\n",
    it,j,cuts[j],b.maxSpeed/speed,b.maxAxialError/speed,
    b.axialMin/speed,(double)b.negativeAxial/b.nsamples,
    b.cellAtMax,b.axialAtMax/D,b.wallDistanceAtMax/D,b.nsamples);
 }
}
} // namespace nodals::cpu_mc::m4
