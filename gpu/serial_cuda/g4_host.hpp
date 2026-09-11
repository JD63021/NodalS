#pragma once
#include "g3_host.hpp"
#include "g2_host.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace nodals_gpu {

struct PipeHost {
  int wall=-1,inlet=-1,outlet=-1;
  double cx=0,cy=0,zIn=0,zOut=0,R=0,D=0,L=0;
  double bulk=1,profileScale=1,nu=0,re=20,inletArea=0,inletProjectedArea=0,circleArea=0,hpGradient=0,hpDrop=0;
  double inletNormal[3]={0.0,0.0,-1.0};
  double inletVelocity[3]={0.0,0.0,1.0};
};
inline Vec3d cross_g4(const Vec3d&a,const Vec3d&b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
inline Vec3d sub_g4(const Vec3d&a,const Vec3d&b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
inline double norm_g4(const Vec3d&a){return std::sqrt(a.x*a.x+a.y*a.y+a.z*a.z);}
inline double tri_area_g4(const Vec3d&a,const Vec3d&b,const Vec3d&c){return 0.5*norm_g4(cross_g4(sub_g4(b,a),sub_g4(c,a)));}
inline double pipe_ideal_uz_g4(const PipeHost&P,double x,double y){double rr=(x-P.cx)*(x-P.cx)+(y-P.cy)*(y-P.cy);return 2.0*P.bulk*(1.0-rr/(P.R*P.R));}
inline double triangle_avg_pipe_uz_g4(const PipeHost&P,const Vec3d X[3]){
  double xx[3],yy[3];for(int i=0;i<3;++i){xx[i]=X[i].x-P.cx;yy[i]=X[i].y-P.cy;}
  double ax=(xx[0]*xx[0]+xx[1]*xx[1]+xx[2]*xx[2]+xx[0]*xx[1]+xx[1]*xx[2]+xx[2]*xx[0])/6.0;
  double ay=(yy[0]*yy[0]+yy[1]*yy[1]+yy[2]*yy[2]+yy[0]*yy[1]+yy[1]*yy[2]+yy[2]*yy[0])/6.0;
  return 2.0*P.bulk*(1.0-(ax+ay)/(P.R*P.R));
}
inline PipeHost make_pipe_g4(const SerialTetMesh&M,double re,double bulk,const std::string&wall,const std::string&inlet,const std::string&outlet){
  PipeHost P;P.wall=patch_index(M,wall);P.inlet=patch_index(M,inlet);P.outlet=patch_index(M,outlet);P.re=re;P.bulk=bulk;
  if(P.wall<0||P.inlet<0||P.outlet<0)throw std::runtime_error("G4 pipe patch not found");
  double xmin=1e300,xmax=-1e300,ymin=1e300,ymax=-1e300,zmin=1e300,zmax=-1e300;
  for(const auto&x:M.points){xmin=std::min(xmin,x.x);xmax=std::max(xmax,x.x);ymin=std::min(ymin,x.y);ymax=std::max(ymax,x.y);zmin=std::min(zmin,x.z);zmax=std::max(zmax,x.z);}
  P.cx=.5*(xmin+xmax);P.cy=.5*(ymin+ymax);P.zIn=zmin;P.zOut=zmax;P.L=zmax-zmin;P.R=.5*std::max(xmax-xmin,ymax-ymin);P.D=2*P.R;
  if(!(P.R>0&&P.L>0&&re>0&&bulk>0))throw std::runtime_error("G4 invalid pipe geometry");
  P.nu=bulk*P.D/re;P.circleArea=3.14159265358979323846*P.R*P.R;
  auto pin=M.patches[(std::size_t)P.inlet];double raw=0;Vec3d inletSf{};
  for(int f=pin.start_face;f<pin.start_face+pin.n_faces;++f){const auto&F=M.faces[(std::size_t)f];Vec3d X[3]={M.points[(std::size_t)F.v[0]],M.points[(std::size_t)F.v[1]],M.points[(std::size_t)F.v[2]]};double a=tri_area_g4(X[0],X[1],X[2]);P.inletArea+=a;raw+=a*triangle_avg_pipe_uz_g4(P,X);const Vec3d sf=face_outward_area_vector_g2(M,f);inletSf.x+=sf.x;inletSf.y+=sf.y;inletSf.z+=sf.z;}
  P.inletProjectedArea=norm_g4(inletSf);if(!(P.inletProjectedArea>0.0))throw std::runtime_error("G4 degenerate inlet projected area");P.inletNormal[0]=inletSf.x/P.inletProjectedArea;P.inletNormal[1]=inletSf.y/P.inletProjectedArea;P.inletNormal[2]=inletSf.z/P.inletProjectedArea;for(int d=0;d<3;++d)P.inletVelocity[d]=-bulk*P.inletNormal[d];
  if(!(raw>0))throw std::runtime_error("G4 non-positive inlet flux");P.profileScale=bulk*P.inletArea/raw;P.hpGradient=32.0*P.nu*bulk/(P.D*P.D);P.hpDrop=P.hpGradient*P.L;return P;
}

struct G4CellPlanHost {
  std::int32_t ref[8]; // >=0 free velocity gid, <0 -(fixedSlot+1)
  std::uint8_t rowSlot[64];
  double det;
  double h2; // squared tetrahedron diameter; retained for SUPG tau(u)
  double invJ[9];
  std::int8_t inletOpp=-1;
  double inletSf[3]={0.0,0.0,0.0};
  std::uint8_t wallBasis[8]={0,0,0,0,0,0,0,0};
};
struct G4SetupHost {
  MomentumSetupHost momMask;
  MomentumCSRHost topo; // val is physical diffusion at nu only; alpha_u=1
  ColoringHost coloring;
  PressureSetupHost pressure;
  PipeHost pipe;
  bool dgInlet=false;
  bool weakWall=false;
  std::vector<unsigned char> wallEntity;
  std::vector<std::int32_t> fixedSlot;
  std::vector<std::array<double,3>> fixedValue;
  std::vector<G4CellPlanHost> cells;
  std::array<std::vector<double>,3> staticRhs;
  std::vector<double> fixedDiv,volumes;
};
inline int fixed_slot_from_ref_g4(std::int32_t r){return -r-1;}
inline double g4_effective_bcoeff_host(const G4CellPlanHost&cp,int a,int d){
  if(d<2 && cp.wallBasis[a])return 0.0;
  const double g1=cp.invJ[d],g2=cp.invJ[3+d],g3=cp.invJ[6+d];
  const int i=(a<4)?a:a-4;
  const double gl=i==0?-(g1+g2+g3):(i==1?g1:(i==2?g2:g3));
  const double base=(cp.det/6.0)*gl;
  double v=(a<4)?base:-(27.0/20.0)*base;
  if(cp.inletOpp>=0){if(a<4 && a!=(int)cp.inletOpp)v-=cp.inletSf[d]/3.0;else if(a==4+(int)cp.inletOpp)v-=(9.0/20.0)*cp.inletSf[d];}
  return v;
}
inline std::array<double,3> ref_value_host_g4(const G4SetupHost&S,std::int32_t r,const std::array<std::vector<double>,3>&U){
  if(r>=0)return {U[0][(std::size_t)r],U[1][(std::size_t)r],U[2][(std::size_t)r]};
  return S.fixedValue[(std::size_t)fixed_slot_from_ref_g4(r)];
}

struct CentralTensorG4{double t[8][8][8][3] = {};};
inline const CentralTensorG4& central_tensor_g4(){static const CentralTensorG4 T=[](){CentralTensorG4 o;auto Q=tet_duffy5_g3();for(const auto&q:Q){double gr[8][3],val[8];
    const double gl[4][3]={{-1,-1,-1},{1,0,0},{0,1,0},{0,0,1}};
    for(int i=0;i<4;++i){val[i]=q.lam[i];for(int d=0;d<3;++d)gr[i][d]=gl[i][d];}
    for(int i=0;i<4;++i){int js[3],kk=0;for(int j=0;j<4;++j)if(j!=i)js[kk++]=j;val[4+i]=27*q.lam[js[0]]*q.lam[js[1]]*q.lam[js[2]];for(int d=0;d<3;++d)gr[4+i][d]=0;for(int a=0;a<3;++a){int j=js[a],o1=js[(a+1)%3],o2=js[(a+2)%3];for(int d=0;d<3;++d)gr[4+i][d]+=27*q.lam[o1]*q.lam[o2]*gl[j][d];}}
    for(int a=0;a<8;++a)for(int m=0;m<8;++m)for(int b=0;b<8;++b)for(int j=0;j<3;++j)o.t[a][m][b][j]+=val[a]*val[m]*gr[b][j]*q.w;
  }return o;}();return T;}


// -----------------------------------------------------------------------------
// P5B: parallel G4CellPlanHost construction with exact current-main cp.h2.
// -----------------------------------------------------------------------------

inline int p5b_cell_threads(){
  int n=0;
  if(const char*e=std::getenv("NODALS_G4_CELL_THREADS"))n=std::atoi(e);
  if(n<=0){
    unsigned h=std::thread::hardware_concurrency();
    n=h?std::min((int)h,16):1;
  }
  return std::max(1,std::min(n,64));
}

inline void p5b_build_one_cell_plan(
    const SerialTetMesh&M,const G4SetupHost&S,std::size_t c,G4CellPlanHost&cp)
{
  const int nv=(int)M.points.size();
  std::fill(std::begin(cp.rowSlot),std::end(cp.rowSlot),(std::uint8_t)255);

  int ent[8];
  for(int i=0;i<4;++i)ent[i]=M.tets[c][i];
  for(int i=0;i<4;++i)ent[4+i]=nv+M.opp_face[c][i];

  for(int a=0;a<8;++a){
    int g=S.momMask.g2free[(std::size_t)ent[a]];
    cp.ref[a]=(g>=0)?g:-(S.fixedSlot[(std::size_t)ent[a]]+1);
    cp.wallBasis[a]=(!S.wallEntity.empty() && S.wallEntity[(std::size_t)ent[a]])?1:0;
  }

  auto t=M.tets[c];
  const Vec3d X[4]={
    M.points[(std::size_t)t[0]],M.points[(std::size_t)t[1]],
    M.points[(std::size_t)t[2]],M.points[(std::size_t)t[3]]
  };

  double J[3][3]={
    {X[1].x-X[0].x,X[2].x-X[0].x,X[3].x-X[0].x},
    {X[1].y-X[0].y,X[2].y-X[0].y,X[3].y-X[0].y},
    {X[1].z-X[0].z,X[2].z-X[0].z,X[3].z-X[0].z}
  },I[3][3];

  cp.det=det3(J);
  if(!(cp.det>0.0))throw std::runtime_error("P5B G4 non-positive tet");

  cp.h2=0.0;
  for(int aa=0;aa<4;++aa)
    for(int bb=aa+1;bb<4;++bb){
      const double dx=X[aa].x-X[bb].x;
      const double dy=X[aa].y-X[bb].y;
      const double dz=X[aa].z-X[bb].z;
      cp.h2=std::max(cp.h2,dx*dx+dy*dy+dz*dz);
    }
  if(!(cp.h2>0.0))throw std::runtime_error("P5B G4 zero tet diameter");

  inv3(J,I);
  for(int j=0;j<3;++j)
    for(int d=0;d<3;++d)
      cp.invJ[3*j+d]=I[j][d];
  cp.inletOpp=-1;cp.inletSf[0]=cp.inletSf[1]=cp.inletSf[2]=0.0;
  if(S.dgInlet){int io=-1;Vec3d sf{};if(cell_inlet_face_g2(M,S.pipe.inlet,(int)c,io,sf)){cp.inletOpp=(std::int8_t)io;cp.inletSf[0]=sf.x;cp.inletSf[1]=sf.y;cp.inletSf[2]=sf.z;}}

  for(int a=0;a<8;++a)if(cp.ref[a]>=0){
    int r=cp.ref[a];
    for(int b=0;b<8;++b)if(cp.ref[b]>=0){
      auto first=S.topo.col.begin()+S.topo.row[(std::size_t)r];
      auto last=S.topo.col.begin()+S.topo.row[(std::size_t)r+1];
      auto it=std::lower_bound(first,last,cp.ref[b]);
      if(it==last||*it!=cp.ref[b])throw std::runtime_error("P5B G4 row slot missing");
      auto slot=(std::int64_t)(it-first);
      if(slot>=255)throw std::runtime_error("P5B G4 row slot overflow");
      cp.rowSlot[8*a+b]=(std::uint8_t)slot;
    }
  }
}

inline std::vector<G4CellPlanHost> p5b_build_cell_plans_reference(
    const SerialTetMesh&M,const G4SetupHost&S)
{
  const auto t0=std::chrono::steady_clock::now();
  std::vector<G4CellPlanHost>C(M.tets.size());
  for(std::size_t c=0;c<M.tets.size();++c)
    p5b_build_one_cell_plan(M,S,c,C[c]);
  const auto t1=std::chrono::steady_clock::now();
  std::printf(
    "NODALS_GPU_P5B_CELLPLAN_REFERENCE_DONE status=PASS cells=%zu seconds=%.6f rssMiB=%.3f hwmMiB=%.3f\n",
    C.size(),std::chrono::duration<double>(t1-t0).count(),
    g4a_status_mib("VmRSS"),g4a_status_mib("VmHWM"));
  return C;
}

inline std::vector<G4CellPlanHost> p5b_build_cell_plans_parallel(
    const SerialTetMesh&M,const G4SetupHost&S)
{
  const auto t0=std::chrono::steady_clock::now();
  const int nth=p5b_cell_threads();
  std::vector<G4CellPlanHost>C(M.tets.size());

  g4a_parallel_chunks((int)M.tets.size(),nth,256,[&](int,int cb,int ce){
    for(int c=cb;c<ce;++c)
      p5b_build_one_cell_plan(M,S,(std::size_t)c,C[(std::size_t)c]);
  });

  const auto t1=std::chrono::steady_clock::now();
  std::printf(
    "NODALS_GPU_P5B_CELLPLAN_NEW status=PASS cells=%zu threads=%d seconds=%.6f bytes=%zu scratchExtraMiB=0.000 rssMiB=%.3f hwmMiB=%.3f\n",
    C.size(),nth,std::chrono::duration<double>(t1-t0).count(),
    C.size()*sizeof(G4CellPlanHost),
    g4a_status_mib("VmRSS"),g4a_status_mib("VmHWM"));
  return C;
}

inline void p5b_validate_cell_plans(
    const std::vector<G4CellPlanHost>&R,
    const std::vector<G4CellPlanHost>&N,
    double refSeconds,double newSeconds)
{
  const bool sizeExact=R.size()==N.size();
  bool refExact=sizeExact,rowExact=sizeExact,detExact=sizeExact,h2Exact=sizeExact,invExact=sizeExact;
  double detMaxAbs=0.0,h2MaxAbs=0.0,invMaxAbs=0.0;
  std::size_t firstBad=std::numeric_limits<std::size_t>::max();

  if(sizeExact){
    for(std::size_t c=0;c<R.size();++c){
      for(int a=0;a<8;++a)if(R[c].ref[a]!=N[c].ref[a]){
        refExact=false;
        if(firstBad==std::numeric_limits<std::size_t>::max())firstBad=c;
      }
      for(int k=0;k<64;++k)if(R[c].rowSlot[k]!=N[c].rowSlot[k]){
        rowExact=false;
        if(firstBad==std::numeric_limits<std::size_t>::max())firstBad=c;
      }
      if(R[c].det!=N[c].det){
        detExact=false; detMaxAbs=std::max(detMaxAbs,std::abs(R[c].det-N[c].det));
        if(firstBad==std::numeric_limits<std::size_t>::max())firstBad=c;
      }
      if(R[c].h2!=N[c].h2){
        h2Exact=false; h2MaxAbs=std::max(h2MaxAbs,std::abs(R[c].h2-N[c].h2));
        if(firstBad==std::numeric_limits<std::size_t>::max())firstBad=c;
      }
      for(int k=0;k<9;++k)if(R[c].invJ[k]!=N[c].invJ[k]){
        invExact=false; invMaxAbs=std::max(invMaxAbs,std::abs(R[c].invJ[k]-N[c].invJ[k]));
        if(firstBad==std::numeric_limits<std::size_t>::max())firstBad=c;
      }
    }
  }

  const bool pass=sizeExact&&refExact&&rowExact&&detExact&&h2Exact&&invExact;
  std::printf(
    "NODALS_GPU_P5B_CELLPLAN_PARITY status=%s sizeExact=%d refExact=%d rowSlotExact=%d "
    "detBitwiseExact=%d h2BitwiseExact=%d invJBitwiseExact=%d cellsRef=%zu cellsNew=%zu "
    "detMaxAbs=%.12e h2MaxAbs=%.12e invJMaxAbs=%.12e firstBadCell=%lld "
    "refSeconds=%.6f newSeconds=%.6f speedup=%.6f compareHoldsTwoCellPlans=1 productionPeakMustUseModeParallel=1\n",
    pass?"PASS":"FAIL",(int)sizeExact,(int)refExact,(int)rowExact,
    (int)detExact,(int)h2Exact,(int)invExact,R.size(),N.size(),
    detMaxAbs,h2MaxAbs,invMaxAbs,
    firstBad==std::numeric_limits<std::size_t>::max()?-1LL:(long long)firstBad,
    refSeconds,newSeconds,refSeconds/std::max(newSeconds,1e-300));

  if(!pass)throw std::runtime_error("P5B cell-plan parity failed");
}

inline std::vector<G4CellPlanHost> p5b_build_cell_plans_dispatch(
    const SerialTetMesh&M,const G4SetupHost&S)
{
  const char*e=std::getenv("NODALS_G4_CELL_MODE");
  const char*mode=(e&&*e)?e:"parallel";

  if(std::strcmp(mode,"reference")==0)
    return p5b_build_cell_plans_reference(M,S);
  if(std::strcmp(mode,"parallel")==0)
    return p5b_build_cell_plans_parallel(M,S);
  if(std::strcmp(mode,"compare")==0){
    const auto r0=std::chrono::steady_clock::now();
    auto R=p5b_build_cell_plans_reference(M,S);
    const auto r1=std::chrono::steady_clock::now();
    const auto n0=std::chrono::steady_clock::now();
    auto N=p5b_build_cell_plans_parallel(M,S);
    const auto n1=std::chrono::steady_clock::now();
    p5b_validate_cell_plans(
      R,N,
      std::chrono::duration<double>(r1-r0).count(),
      std::chrono::duration<double>(n1-n0).count());
    return N;
  }
  throw std::runtime_error("NODALS_G4_CELL_MODE must be reference, compare, or parallel");
}

inline G4SetupHost build_g4_setup(const SerialTetMesh&M,double re=20,double bulk=1.0,const std::string&wall="patch_0_0",const std::string&inlet="patch_2_0",const std::string&outlet="patch_1_0",bool dgInlet=false,bool weakWall=false){
  G4SetupHost S;S.dgInlet=dgInlet;S.weakWall=weakWall;S.pipe=make_pipe_g4(M,re,bulk,wall,inlet,outlet);S.momMask=build_momentum_setup(M,S.pipe.outlet,S.pipe.inlet,dgInlet,S.pipe.wall,weakWall);
  const int nv=(int)M.points.size(),nf=(int)M.faces.size(),ni=(int)M.neighbour.size();
  if(weakWall){
    S.wallEntity.assign((std::size_t)nv+nf,0);
    for(int f=ni;f<nf;++f)if(M.face_patch[(std::size_t)f]==S.pipe.wall){
      S.wallEntity[(std::size_t)nv+f]=1;
      for(int v:M.faces[(std::size_t)f].v)S.wallEntity[(std::size_t)v]=1;
    }
  }
  S.fixedSlot.assign((std::size_t)nv+nf,-1);int ns=0;for(std::size_t e=0;e<S.momMask.fixed.size();++e)if(S.momMask.fixed[e])S.fixedSlot[e]=ns++;
  S.fixedValue.assign((std::size_t)ns,{0.0,0.0,0.0});
  std::vector<unsigned char> onWall((std::size_t)nv,0),onInlet((std::size_t)nv,0);
  for(int f=ni;f<nf;++f){int p=M.face_patch[(std::size_t)f];if(p==S.pipe.wall)for(int v:M.faces[(std::size_t)f].v)onWall[(std::size_t)v]=1;if(p==S.pipe.inlet)for(int v:M.faces[(std::size_t)f].v)onInlet[(std::size_t)v]=1;}
  if(!dgInlet){
    for(int v=0;v<nv;++v)if(onInlet[(std::size_t)v]&&!onWall[(std::size_t)v]){int s=S.fixedSlot[(std::size_t)v];S.fixedValue[(std::size_t)s][2]=S.pipe.profileScale*pipe_ideal_uz_g4(S.pipe,M.points[(std::size_t)v].x,M.points[(std::size_t)v].y);}
    auto pin=M.patches[(std::size_t)S.pipe.inlet];for(int f=pin.start_face;f<pin.start_face+pin.n_faces;++f){const auto&F=M.faces[(std::size_t)f];Vec3d X[3]={M.points[(std::size_t)F.v[0]],M.points[(std::size_t)F.v[1]],M.points[(std::size_t)F.v[2]]};double exact=S.pipe.profileScale*triangle_avg_pipe_uz_g4(S.pipe,X);double vm=0;for(int j=0;j<3;++j){int sl=S.fixedSlot[(std::size_t)F.v[j]];vm+=S.fixedValue[(std::size_t)sl][2]/3.0;}int fs=S.fixedSlot[(std::size_t)nv+f];S.fixedValue[(std::size_t)fs][2]=(20.0/9.0)*(exact-vm);}
  }
  S.topo=build_static_relaxed_momentum_csr(M,S.momMask,S.pipe.nu,1.0);S.coloring=greedy_csr_coloring(S.topo);
  S.cells=p5b_build_cell_plans_dispatch(M,S);
  for(auto&r:S.staticRhs)r.assign((std::size_t)S.topo.n,0.0);const auto&D=diffusion_tensor_g3();
  for(const auto&cp:S.cells){double I[3][3];for(int j=0;j<3;++j)for(int d=0;d<3;++d)I[j][d]=cp.invJ[3*j+d];double metric[3][3]={{0}};for(int j=0;j<3;++j)for(int k=0;k<3;++k)for(int d=0;d<3;++d)metric[j][k]+=I[j][d]*I[k][d];double K[8][8]={{0}};for(int a=0;a<8;++a)for(int b=0;b<8;++b){double v=0;for(int j=0;j<3;++j)for(int k=0;k<3;++k)v+=D.t[a][b][j][k]*metric[j][k];K[a][b]=S.pipe.nu*cp.det*v;}for(int a=0;a<8;++a)if(cp.ref[a]>=0){int r=cp.ref[a];for(int b=0;b<8;++b)if(cp.ref[b]<0){auto fv=S.fixedValue[(std::size_t)fixed_slot_from_ref_g4(cp.ref[b])];for(int d=0;d<3;++d)S.staticRhs[(std::size_t)d][(std::size_t)r]-=K[a][b]*fv[(std::size_t)d];}}}
  S.pressure=build_pressure_setup(M,S.pipe.outlet,S.pipe.inlet,dgInlet,S.pipe.wall,weakWall);if(S.pressure.g2free!=S.momMask.g2free)throw std::runtime_error("G4 velocity numbering mismatch G2/G3");
  S.fixedDiv.assign(M.tets.size(),0.0);S.volumes.assign(M.tets.size(),0.0);
  for(std::size_t c=0;c<M.tets.size();++c){const auto&bp=S.pressure.cells[c];S.volumes[c]=S.cells[c].det/6.0;double q=0;if(dgInlet && bp.inletOpp>=0)q+=S.pipe.inletVelocity[0]*bp.inletSf[0]+S.pipe.inletVelocity[1]*bp.inletSf[1]+S.pipe.inletVelocity[2]*bp.inletSf[2];for(int a=0;a<8;++a)if(S.cells[c].ref[a]<0){auto fv=S.fixedValue[(std::size_t)fixed_slot_from_ref_g4(S.cells[c].ref[a])];for(int d=0;d<3;++d)q+=coeff(bp,a,d)*fv[(std::size_t)d];}S.fixedDiv[c]=q;}
  return S;
}

inline void host_assemble_central_g4(const G4SetupHost&S,const std::array<std::vector<double>,3>&U,std::vector<double>&aval,std::array<std::vector<double>,3>&conv){
  aval=S.topo.val;for(auto&r:conv)r.assign((std::size_t)S.topo.n,0.0);const auto&T=central_tensor_g4();
  for(const auto&cp:S.cells){double I[3][3];for(int j=0;j<3;++j)for(int d=0;d<3;++d)I[j][d]=cp.invJ[3*j+d];double cf[3][8]={{0}};for(int m=0;m<8;++m){auto v=ref_value_host_g4(S,cp.ref[m],U);for(int d=0;d<3;++d)cf[d][m]=v[(std::size_t)d];}double ur[8][3]={{0}};for(int m=0;m<8;++m)for(int j=0;j<3;++j)for(int d=0;d<3;++d)ur[m][j]+=cf[d][m]*I[j][d];double C[8][8]={{0}};for(int a=0;a<8;++a)for(int b=0;b<8;++b){double v=0;for(int m=0;m<8;++m)for(int j=0;j<3;++j)v+=ur[m][j]*T.t[a][m][b][j];C[a][b]=cp.det*v;}for(int a=0;a<8;++a)if(cp.ref[a]>=0){int r=cp.ref[a];for(int b=0;b<8;++b){if(cp.ref[b]>=0){int slot=cp.rowSlot[8*a+b];aval[(std::size_t)(S.topo.row[(std::size_t)r]+slot)]+=C[a][b];}else{auto fv=S.fixedValue[(std::size_t)fixed_slot_from_ref_g4(cp.ref[b])];for(int d=0;d<3;++d)conv[(std::size_t)d][(std::size_t)r]-=C[a][b]*fv[(std::size_t)d];}}}}
}
// Exact setup-side counterpart of the H8 CUDA SUPG64 kernel.
// Pipe forcing is zero, so only the implicit SUPG matrix and Dirichlet
// elimination RHS are formed. tau is lagged on the supplied velocity state.
inline void host_add_supg64_g4(
    const G4SetupHost&S,const std::array<std::vector<double>,3>&U,
    double tauScale,double supgMagic,
    std::vector<double>&aval,std::array<std::vector<double>,3>&rhs)
{
  const double rn[4]={0.0485005494469972764,0.238600737551862341,0.517047295104367421,0.795851417896772828};
  const double rw[4]={0.110888415611277741,0.143458789799214448,0.0686338871729230970,0.0103522407499180812};
  const double sn[4]={0.0571041961145177246,0.276843013638123803,0.583590432368916834,0.860240135656219485};
  const double sw[4]={0.135506913431488518,0.203464568010271102,0.129847547608232333,0.0311809709500080849};
  const double tn[4]={0.0694318442029737137,0.330009478207571871,0.669990521792428129,0.930568155797026231};
  const double tw[4]={0.173927422568726897,0.326072577431273103,0.326072577431273103,0.173927422568726897};

  for(const auto&cp:S.cells){
    double coeff[3][8]={{0}};
    for(int m=0;m<8;++m){
      const auto q=ref_value_host_g4(S,cp.ref[m],U);
      for(int d=0;d<3;++d)coeff[d][m]=q[(std::size_t)d];
    }

    double gl[4][3];
    for(int d=0;d<3;++d){
      gl[0][d]=-(cp.invJ[d]+cp.invJ[3+d]+cp.invJ[6+d]);
      gl[1][d]= cp.invJ[d];
      gl[2][d]= cp.invJ[3+d];
      gl[3][d]= cp.invJ[6+d];
    }
    double gd[4][4]={{0}};
    for(int i=0;i<4;++i)for(int j=i;j<4;++j){
      double q=0.0;for(int d=0;d<3;++d)q+=gl[i][d]*gl[j][d];
      gd[i][j]=gd[j][i]=q;
    }

    double Sl[8][8]={{0}};
    for(int ir=0;ir<4;++ir)for(int is=0;is<4;++is)for(int it=0;it<4;++it){
      const double r=rn[ir],ss=sn[is],tt=tn[it];
      const double omr=1.0-r,oms=1.0-ss;
      const double lam[4]={omr*oms*(1.0-tt),r,omr*ss,omr*oms*tt};
      const double qw=rw[ir]*sw[is]*tw[it];

      double phi[8]={lam[0],lam[1],lam[2],lam[3],0,0,0,0};
      for(int i=0;i<4;++i){
        int js[3],kk=0;for(int j=0;j<4;++j)if(j!=i)js[kk++]=j;
        phi[4+i]=27.0*lam[js[0]]*lam[js[1]]*lam[js[2]];
      }

      double adv[3]={0,0,0};
      for(int d=0;d<3;++d)for(int m=0;m<8;++m)adv[d]+=coeff[d][m]*phi[m];
      const double speed2=adv[0]*adv[0]+adv[1]*adv[1]+adv[2]*adv[2];

      double stream[8]={0},strong[8]={0};
      for(int a=0;a<4;++a)
        stream[a]=adv[0]*gl[a][0]+adv[1]*gl[a][1]+adv[2]*gl[a][2];
      for(int i=0;i<4;++i){
        int js[3],kk=0;for(int j=0;j<4;++j)if(j!=i)js[kk++]=j;
        double gb[3]={0,0,0};
        for(int d=0;d<3;++d)
          gb[d]=27.0*(lam[js[1]]*lam[js[2]]*gl[js[0]][d]
                    +lam[js[0]]*lam[js[2]]*gl[js[1]][d]
                    +lam[js[0]]*lam[js[1]]*gl[js[2]][d]);
        stream[4+i]=adv[0]*gb[0]+adv[1]*gb[1]+adv[2]*gb[2];
      }
      for(int a=0;a<4;++a)strong[a]=stream[a];
      for(int i=0;i<4;++i){
        int js[3],kk=0;for(int j=0;j<4;++j)if(j!=i)js[kk++]=j;
        const double lap=54.0*(lam[js[2]]*gd[js[0]][js[1]]
                             +lam[js[1]]*gd[js[0]][js[2]]
                             +lam[js[0]]*gd[js[1]][js[2]]);
        strong[4+i]=-S.pipe.nu*lap+stream[4+i];
      }

      const double diff=4.0*S.pipe.nu/cp.h2;
      const double den=std::max(4.0*speed2/cp.h2+supgMagic*diff*diff,1.0e-30);
      const double tau=tauScale/std::sqrt(den);
      const double w=tau*qw*cp.det;
      for(int a=0;a<8;++a){
        const double ta=w*stream[a];
        for(int b=0;b<8;++b)Sl[a][b]+=ta*strong[b];
      }
    }

    for(int a=0;a<8;++a){
      const int r=cp.ref[a];if(r<0)continue;
      for(int b=0;b<8;++b){
        const double z=Sl[a][b];
        if(cp.ref[b]>=0){
          const unsigned char sl=cp.rowSlot[8*a+b];
          if(sl==255)throw std::runtime_error("G4 host SUPG row slot missing");
          aval[(std::size_t)S.topo.row[(std::size_t)r]+sl]+=z;
        }else{
          const auto fv=S.fixedValue[(std::size_t)fixed_slot_from_ref_g4(cp.ref[b])];
          for(int d=0;d<3;++d)rhs[(std::size_t)d][(std::size_t)r]-=z*fv[(std::size_t)d];
        }
      }
    }
  }
}


inline std::vector<double> host_finalize_relax_g4(const G4SetupHost&S,std::vector<double>&a,double alphaU,std::vector<double>*deltaOut=nullptr){std::vector<double>rau((std::size_t)S.topo.n),delta((std::size_t)S.topo.n);double fac=1.0/alphaU-1.0;for(int i=0;i<S.topo.n;++i){auto first=S.topo.col.begin()+S.topo.row[(std::size_t)i],last=S.topo.col.begin()+S.topo.row[(std::size_t)i+1];auto it=std::lower_bound(first,last,i);if(it==last||*it!=i)throw std::runtime_error("G4 host diagonal absent");std::size_t dp=(std::size_t)(it-S.topo.col.begin());double m=0;for(std::int64_t k=S.topo.row[(std::size_t)i];k<S.topo.row[(std::size_t)i+1];++k)m+=std::abs(a[(std::size_t)k]);double de=fac*m;delta[(std::size_t)i]=de;a[dp]+=de;if(!(a[dp]>0))throw std::runtime_error("G4 host relaxed diag invalid");rau[(std::size_t)i]=1.0/a[dp];}if(deltaOut)*deltaOut=std::move(delta);return rau;}

inline double pressure_drop_fit_g4(const SerialTetMesh&M,const std::vector<double>&p){if(p.size()!=M.tets.size())return std::numeric_limits<double>::quiet_NaN();long double sz=0,sp=0,szz=0,szp=0;long double n=p.size();for(std::size_t c=0;c<M.tets.size();++c){double z=0;for(int i=0;i<4;++i)z+=M.points[(std::size_t)M.tets[c][i]].z*.25;sz+=z;sp+=p[c];szz+=z*z;szp+=z*p[c];}long double den=n*szz-sz*sz;if(std::abs((double)den)<1e-300)return 0;long double slope=(n*szp-sz*sp)/den;double zmin=1e300,zmax=-1e300;for(auto&x:M.points){zmin=std::min(zmin,x.z);zmax=std::max(zmax,x.z);}return (double)(-slope*(zmax-zmin));}

} // namespace nodals_gpu
