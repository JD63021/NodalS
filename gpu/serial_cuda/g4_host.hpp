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

namespace nodals_gpu {

struct PipeHost {
  int wall=-1,inlet=-1,outlet=-1;
  double cx=0,cy=0,zIn=0,zOut=0,R=0,D=0,L=0;
  double bulk=1,profileScale=1,nu=0,re=20,inletArea=0,circleArea=0,hpGradient=0,hpDrop=0;
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
  auto pin=M.patches[(std::size_t)P.inlet];double raw=0;
  for(int f=pin.start_face;f<pin.start_face+pin.n_faces;++f){const auto&F=M.faces[(std::size_t)f];Vec3d X[3]={M.points[(std::size_t)F.v[0]],M.points[(std::size_t)F.v[1]],M.points[(std::size_t)F.v[2]]};double a=tri_area_g4(X[0],X[1],X[2]);P.inletArea+=a;raw+=a*triangle_avg_pipe_uz_g4(P,X);}
  if(!(raw>0))throw std::runtime_error("G4 non-positive inlet flux");P.profileScale=bulk*P.inletArea/raw;P.hpGradient=32.0*P.nu*bulk/(P.D*P.D);P.hpDrop=P.hpGradient*P.L;return P;
}

struct G4CellPlanHost {
  std::int32_t ref[8]; // >=0 free velocity gid, <0 -(fixedSlot+1)
  std::uint8_t rowSlot[64];
  double det;
  double invJ[9];
};
struct G4SetupHost {
  MomentumSetupHost momMask;
  MomentumCSRHost topo; // val is physical diffusion at nu only; alpha_u=1
  ColoringHost coloring;
  PressureSetupHost pressure;
  PipeHost pipe;
  std::vector<std::int32_t> fixedSlot;
  std::vector<std::array<double,3>> fixedValue;
  std::vector<G4CellPlanHost> cells;
  std::array<std::vector<double>,3> staticRhs;
  std::vector<double> fixedDiv,volumes;
};
inline int fixed_slot_from_ref_g4(std::int32_t r){return -r-1;}
inline std::array<double,3> ref_value_host_g4(const G4SetupHost&S,std::int32_t r,const std::array<std::vector<double>,3>&U){
  if(r>=0)return {U[0][(std::size_t)r],U[1][(std::size_t)r],U[2][(std::size_t)r]};
  return S.fixedValue[(std::size_t)fixed_slot_from_ref_g4(r)];
}

struct CentralTensorG4{double t[8][8][8][3] = {};};
inline const CentralTensorG4& central_tensor_g4(){static const CentralTensorG4 T=[](){CentralTensorG4 o;auto Q=tet_duffy5_g3();for(const auto&q:Q){double gr[8][3],val[8];
    // current NodalS P1+BF3 values/gradients on reference tet
    const double gl[4][3]={{-1,-1,-1},{1,0,0},{0,1,0},{0,0,1}};
    for(int i=0;i<4;++i){val[i]=q.lam[i];for(int d=0;d<3;++d)gr[i][d]=gl[i][d];}
    for(int i=0;i<4;++i){int js[3],kk=0;for(int j=0;j<4;++j)if(j!=i)js[kk++]=j;val[4+i]=27*q.lam[js[0]]*q.lam[js[1]]*q.lam[js[2]];for(int d=0;d<3;++d)gr[4+i][d]=0;for(int a=0;a<3;++a){int j=js[a],o1=js[(a+1)%3],o2=js[(a+2)%3];for(int d=0;d<3;++d)gr[4+i][d]+=27*q.lam[o1]*q.lam[o2]*gl[j][d];}}
    for(int a=0;a<8;++a)for(int m=0;m<8;++m)for(int b=0;b<8;++b)for(int j=0;j<3;++j)o.t[a][m][b][j]+=val[a]*val[m]*gr[b][j]*q.w;
  }return o;}();return T;}


// -----------------------------------------------------------------------------
// Gate 4B: parallel construction of G4CellPlanHost.
//
// Every cell writes one independent G4CellPlanHost.  The parallel path executes
// the same per-cell operations as the original serial loop: ref[] construction,
// tet geometry, inverse Jacobian, and the same lower_bound row-slot lookup.
// -----------------------------------------------------------------------------

inline int g4b_cell_threads(){
  int n=0;
  if(const char*e=std::getenv("NODALS_G4_CELL_THREADS"))n=std::atoi(e);
  if(n<=0){unsigned h=std::thread::hardware_concurrency();n=h?std::min((int)h,16):1;}
  return std::max(1,std::min(n,64));
}

inline void g4b_build_one_cell_plan(
    const SerialTetMesh&M,const G4SetupHost&S,std::size_t c,
    G4CellPlanHost&cp){
  const int nv=(int)M.points.size();

  std::fill(std::begin(cp.rowSlot),std::end(cp.rowSlot),(std::uint8_t)255);

  int ent[8];
  for(int i=0;i<4;++i)ent[i]=M.tets[c][i];
  for(int i=0;i<4;++i)ent[4+i]=nv+M.opp_face[c][i];

  for(int a=0;a<8;++a){
    const int g=S.momMask.g2free[(std::size_t)ent[a]];
    cp.ref[a]=(g>=0)?g:-(S.fixedSlot[(std::size_t)ent[a]]+1);
  }

  auto t=M.tets[c];
  const Vec3d X[4]={
    M.points[(std::size_t)t[0]],
    M.points[(std::size_t)t[1]],
    M.points[(std::size_t)t[2]],
    M.points[(std::size_t)t[3]]
  };
  double J[3][3]={
    {X[1].x-X[0].x,X[2].x-X[0].x,X[3].x-X[0].x},
    {X[1].y-X[0].y,X[2].y-X[0].y,X[3].y-X[0].y},
    {X[1].z-X[0].z,X[2].z-X[0].z,X[3].z-X[0].z}
  },I[3][3];

  cp.det=det3(J);
  if(!(cp.det>0))throw std::runtime_error("G4 non-positive tet");
  inv3(J,I);
  for(int j=0;j<3;++j)
    for(int d=0;d<3;++d)
      cp.invJ[3*j+d]=I[j][d];

  for(int a=0;a<8;++a)if(cp.ref[a]>=0){
    const int r=cp.ref[a];
    for(int b=0;b<8;++b)if(cp.ref[b]>=0){
      auto first=S.topo.col.begin()+S.topo.row[(std::size_t)r];
      auto last =S.topo.col.begin()+S.topo.row[(std::size_t)r+1];
      auto it=std::lower_bound(first,last,cp.ref[b]);
      if(it==last||*it!=cp.ref[b])
        throw std::runtime_error("G4 row slot missing");
      const auto slot=(std::int64_t)(it-first);
      if(slot>=255)throw std::runtime_error("G4 row slot overflow");
      cp.rowSlot[8*a+b]=(std::uint8_t)slot;
    }
  }
}

inline std::vector<G4CellPlanHost> build_g4_cell_plans_reference(
    const SerialTetMesh&M,const G4SetupHost&S){
  const auto t0=std::chrono::steady_clock::now();
  std::vector<G4CellPlanHost>C(M.tets.size());
  for(std::size_t c=0;c<M.tets.size();++c)
    g4b_build_one_cell_plan(M,S,c,C[c]);
  const auto t1=std::chrono::steady_clock::now();

  std::printf(
    "NODALS_GPU_G4B_CELLPLAN_REFERENCE_DONE status=PASS cells=%zu seconds=%.6f rssMiB=%.3f hwmMiB=%.3f\n",
    C.size(),std::chrono::duration<double>(t1-t0).count(),
    g4a_status_mib("VmRSS"),g4a_status_mib("VmHWM"));
  return C;
}

inline std::vector<G4CellPlanHost> build_g4_cell_plans_parallel(
    const SerialTetMesh&M,const G4SetupHost&S){
  const auto t0=std::chrono::steady_clock::now();
  const int nth=g4b_cell_threads();
  std::vector<G4CellPlanHost>C(M.tets.size());

  g4a_parallel_chunks((int)M.tets.size(),nth,256,[&](int,int cb,int ce){
    for(int c=cb;c<ce;++c)
      g4b_build_one_cell_plan(M,S,(std::size_t)c,C[(std::size_t)c]);
  });

  const auto t1=std::chrono::steady_clock::now();
  std::printf(
    "NODALS_GPU_G4B_CELLPLAN_NEW status=PASS cells=%zu threads=%d seconds=%.6f bytes=%zu scratchExtraMiB=0.000 rssMiB=%.3f hwmMiB=%.3f\n",
    C.size(),nth,std::chrono::duration<double>(t1-t0).count(),
    C.size()*sizeof(G4CellPlanHost),
    g4a_status_mib("VmRSS"),g4a_status_mib("VmHWM"));
  return C;
}

inline void g4b_validate_cell_plans(
    const std::vector<G4CellPlanHost>&R,
    const std::vector<G4CellPlanHost>&N,
    double refSeconds,double newSeconds){
  const bool sizeExact=R.size()==N.size();
  bool refExact=sizeExact,rowExact=sizeExact,detExact=sizeExact,invExact=sizeExact;
  double detMaxAbs=0.0,invMaxAbs=0.0;
  std::size_t firstBad=std::numeric_limits<std::size_t>::max();

  if(sizeExact){
    for(std::size_t c=0;c<R.size();++c){
      for(int a=0;a<8;++a){
        if(R[c].ref[a]!=N[c].ref[a]){
          refExact=false;
          if(firstBad==std::numeric_limits<std::size_t>::max())firstBad=c;
        }
      }
      for(int k=0;k<64;++k){
        if(R[c].rowSlot[k]!=N[c].rowSlot[k]){
          rowExact=false;
          if(firstBad==std::numeric_limits<std::size_t>::max())firstBad=c;
        }
      }
      if(R[c].det!=N[c].det){
        detExact=false;
        detMaxAbs=std::max(detMaxAbs,std::abs(R[c].det-N[c].det));
        if(firstBad==std::numeric_limits<std::size_t>::max())firstBad=c;
      }
      for(int k=0;k<9;++k){
        if(R[c].invJ[k]!=N[c].invJ[k]){
          invExact=false;
          invMaxAbs=std::max(invMaxAbs,std::abs(R[c].invJ[k]-N[c].invJ[k]));
          if(firstBad==std::numeric_limits<std::size_t>::max())firstBad=c;
        }
      }
    }
  }

  const bool pass=sizeExact&&refExact&&rowExact&&detExact&&invExact;
  std::printf(
    "NODALS_GPU_G4B_CELLPLAN_PARITY status=%s sizeExact=%d refExact=%d rowSlotExact=%d detBitwiseExact=%d invJBitwiseExact=%d cellsRef=%zu cellsNew=%zu detMaxAbs=%.12e invJMaxAbs=%.12e firstBadCell=%lld refSeconds=%.6f newSeconds=%.6f speedup=%.6f compareHoldsTwoCellPlans=1 productionPeakMustUseModeParallel=1\n",
    pass?"PASS":"FAIL",(int)sizeExact,(int)refExact,(int)rowExact,
    (int)detExact,(int)invExact,R.size(),N.size(),detMaxAbs,invMaxAbs,
    firstBad==std::numeric_limits<std::size_t>::max()?
      -1LL:(long long)firstBad,
    refSeconds,newSeconds,refSeconds/std::max(newSeconds,1e-300));

  if(!pass)throw std::runtime_error("Gate4B cell-plan parity failed");
}

inline std::vector<G4CellPlanHost> build_g4_cell_plans_dispatch(
    const SerialTetMesh&M,const G4SetupHost&S){
  const char*e=std::getenv("NODALS_G4_CELL_MODE");
  const char*mode=(e&&*e)?e:"parallel";

  if(std::strcmp(mode,"reference")==0)
    return build_g4_cell_plans_reference(M,S);

  if(std::strcmp(mode,"parallel")==0)
    return build_g4_cell_plans_parallel(M,S);

  if(std::strcmp(mode,"compare")==0){
    const auto r0=std::chrono::steady_clock::now();
    auto R=build_g4_cell_plans_reference(M,S);
    const auto r1=std::chrono::steady_clock::now();

    const auto n0=std::chrono::steady_clock::now();
    auto N=build_g4_cell_plans_parallel(M,S);
    const auto n1=std::chrono::steady_clock::now();

    g4b_validate_cell_plans(
      R,N,
      std::chrono::duration<double>(r1-r0).count(),
      std::chrono::duration<double>(n1-n0).count());
    return N;
  }

  throw std::runtime_error(
    "NODALS_G4_CELL_MODE must be reference, compare, or parallel");
}

inline G4SetupHost build_g4_setup(const SerialTetMesh&M,double re=20,double bulk=1.0,const std::string&wall="patch_0_0",const std::string&inlet="patch_2_0",const std::string&outlet="patch_1_0"){
  using g4b_clock=std::chrono::steady_clock;
  const auto g4b_all0=g4b_clock::now();
  G4SetupHost S;
  const auto g4b_pipe0=g4b_clock::now();
  S.pipe=make_pipe_g4(M,re,bulk,wall,inlet,outlet);S.momMask=build_momentum_setup(M,S.pipe.outlet);
  const auto g4b_pipe1=g4b_clock::now();
  std::printf("NODALS_GPU_G4B_PROFILE stage=pipe_and_mask seconds=%.6f rssMiB=%.3f hwmMiB=%.3f status=PASS\n",
              std::chrono::duration<double>(g4b_pipe1-g4b_pipe0).count(),
              g4a_status_mib("VmRSS"),g4a_status_mib("VmHWM"));
  const int nv=(int)M.points.size(),nf=(int)M.faces.size(),ni=(int)M.neighbour.size();
  const auto g4b_bc0=g4b_clock::now();
  // Compact fixed slots in exact entity order: all vertices then all faces.
  S.fixedSlot.assign((std::size_t)nv+nf,-1);int ns=0;for(std::size_t e=0;e<S.momMask.fixed.size();++e)if(S.momMask.fixed[e])S.fixedSlot[e]=ns++;
  S.fixedValue.assign((std::size_t)ns,{0.0,0.0,0.0});
  std::vector<unsigned char> onWall((std::size_t)nv,0),onInlet((std::size_t)nv,0);
  for(int f=ni;f<nf;++f){int p=M.face_patch[(std::size_t)f];if(p==S.pipe.wall)for(int v:M.faces[(std::size_t)f].v)onWall[(std::size_t)v]=1;if(p==S.pipe.inlet)for(int v:M.faces[(std::size_t)f].v)onInlet[(std::size_t)v]=1;}
  for(int v=0;v<nv;++v)if(onInlet[(std::size_t)v]&&!onWall[(std::size_t)v]){int s=S.fixedSlot[(std::size_t)v];S.fixedValue[(std::size_t)s][2]=S.pipe.profileScale*pipe_ideal_uz_g4(S.pipe,M.points[(std::size_t)v].x,M.points[(std::size_t)v].y);}
  auto pin=M.patches[(std::size_t)S.pipe.inlet];for(int f=pin.start_face;f<pin.start_face+pin.n_faces;++f){const auto&F=M.faces[(std::size_t)f];Vec3d X[3]={M.points[(std::size_t)F.v[0]],M.points[(std::size_t)F.v[1]],M.points[(std::size_t)F.v[2]]};double exact=S.pipe.profileScale*triangle_avg_pipe_uz_g4(S.pipe,X);double vm=0;for(int j=0;j<3;++j){int sl=S.fixedSlot[(std::size_t)F.v[j]];vm+=S.fixedValue[(std::size_t)sl][2]/3.0;}int fs=S.fixedSlot[(std::size_t)nv+f];S.fixedValue[(std::size_t)fs][2]=(20.0/9.0)*(exact-vm);}
  const auto g4b_bc1=g4b_clock::now();
  std::printf("NODALS_GPU_G4B_PROFILE stage=boundary_values seconds=%.6f rssMiB=%.3f hwmMiB=%.3f status=PASS\n",
              std::chrono::duration<double>(g4b_bc1-g4b_bc0).count(),
              g4a_status_mib("VmRSS"),g4a_status_mib("VmHWM"));
  // Topology + physical diffusion using exact G3 static P1+BF3 tensor, no equation relaxation yet.
  const auto g4b_csr0=g4b_clock::now();
  S.topo=build_static_relaxed_momentum_csr(M,S.momMask,S.pipe.nu,1.0);
  const auto g4b_csr1=g4b_clock::now();
  std::printf("NODALS_GPU_G4B_PROFILE stage=momentum_csr seconds=%.6f rows=%d nnz=%zu rssMiB=%.3f hwmMiB=%.3f status=PASS\n",
              std::chrono::duration<double>(g4b_csr1-g4b_csr0).count(),
              S.topo.n,S.topo.val.size(),g4a_status_mib("VmRSS"),g4a_status_mib("VmHWM"));
  const auto g4b_col0=g4b_clock::now();
  S.coloring=greedy_csr_coloring(S.topo);
  const auto g4b_col1=g4b_clock::now();
  std::printf("NODALS_GPU_G4B_PROFILE stage=coloring seconds=%.6f colors=%d rssMiB=%.3f hwmMiB=%.3f status=PASS\n",
              std::chrono::duration<double>(g4b_col1-g4b_col0).count(),
              S.coloring.ncolors,g4a_status_mib("VmRSS"),g4a_status_mib("VmHWM"));
  const auto g4b_cell0=g4b_clock::now();
  S.cells=build_g4_cell_plans_dispatch(M,S);
  const auto g4b_cell1=g4b_clock::now();
  std::printf("NODALS_GPU_G4B_PROFILE stage=cell_geometry_and_rowslots seconds=%.6f cells=%zu rssMiB=%.3f hwmMiB=%.3f status=PASS\n",
              std::chrono::duration<double>(g4b_cell1-g4b_cell0).count(),
              S.cells.size(),g4a_status_mib("VmRSS"),g4a_status_mib("VmHWM"));
  // Static diffusion Dirichlet RHS (pipe volume forcing is zero).
  const auto g4b_rhs0=g4b_clock::now();
  for(auto&r:S.staticRhs)r.assign((std::size_t)S.topo.n,0.0);const auto&D=diffusion_tensor_g3();
  for(const auto&cp:S.cells){double I[3][3];for(int j=0;j<3;++j)for(int d=0;d<3;++d)I[j][d]=cp.invJ[3*j+d];double metric[3][3]={{0}};for(int j=0;j<3;++j)for(int k=0;k<3;++k)for(int d=0;d<3;++d)metric[j][k]+=I[j][d]*I[k][d];double K[8][8]={{0}};for(int a=0;a<8;++a)for(int b=0;b<8;++b){double v=0;for(int j=0;j<3;++j)for(int k=0;k<3;++k)v+=D.t[a][b][j][k]*metric[j][k];K[a][b]=S.pipe.nu*cp.det*v;}for(int a=0;a<8;++a)if(cp.ref[a]>=0){int r=cp.ref[a];for(int b=0;b<8;++b)if(cp.ref[b]<0){auto fv=S.fixedValue[(std::size_t)fixed_slot_from_ref_g4(cp.ref[b])];for(int d=0;d<3;++d)S.staticRhs[(std::size_t)d][(std::size_t)r]-=K[a][b]*fv[(std::size_t)d];}}}
  const auto g4b_rhs1=g4b_clock::now();
  std::printf("NODALS_GPU_G4B_PROFILE stage=static_rhs seconds=%.6f rssMiB=%.3f hwmMiB=%.3f status=PASS\n",
              std::chrono::duration<double>(g4b_rhs1-g4b_rhs0).count(),
              g4a_status_mib("VmRSS"),g4a_status_mib("VmHWM"));
  // Pressure B plan + fixed divergence + exact volumes. Reuse G2 setup topology, replace synthetic rAU later.
  const auto g4b_p0=g4b_clock::now();
  S.pressure=build_pressure_setup(M,S.pipe.outlet);if(S.pressure.g2free!=S.momMask.g2free)throw std::runtime_error("G4 velocity numbering mismatch G2/G3");
  const auto g4b_p1=g4b_clock::now();
  std::printf("NODALS_GPU_G4B_PROFILE stage=pressure_setup seconds=%.6f cells=%zu freeVel=%d rssMiB=%.3f hwmMiB=%.3f status=PASS\n",
              std::chrono::duration<double>(g4b_p1-g4b_p0).count(),
              S.pressure.cells.size(),S.pressure.free_vel,
              g4a_status_mib("VmRSS"),g4a_status_mib("VmHWM"));
  const auto g4b_fd0=g4b_clock::now();
  S.fixedDiv.assign(M.tets.size(),0.0);S.volumes.assign(M.tets.size(),0.0);
  for(std::size_t c=0;c<M.tets.size();++c){const auto&bp=S.pressure.cells[c];S.volumes[c]=S.cells[c].det/6.0;double q=0;for(int a=0;a<8;++a)if(S.cells[c].ref[a]<0){auto fv=S.fixedValue[(std::size_t)fixed_slot_from_ref_g4(S.cells[c].ref[a])];for(int d=0;d<3;++d)q+=coeff(bp,a,d)*fv[(std::size_t)d];}S.fixedDiv[c]=q;}
  const auto g4b_fd1=g4b_clock::now();
  const auto g4b_all1=g4b_clock::now();
  std::printf("NODALS_GPU_G4B_PROFILE stage=fixed_div_volume seconds=%.6f rssMiB=%.3f hwmMiB=%.3f status=PASS\n",
              std::chrono::duration<double>(g4b_fd1-g4b_fd0).count(),
              g4a_status_mib("VmRSS"),g4a_status_mib("VmHWM"));
  std::printf("NODALS_GPU_G4B_PROFILE stage=total seconds=%.6f status=PASS\n",
              std::chrono::duration<double>(g4b_all1-g4b_all0).count());
  return S;
}

inline void host_assemble_central_g4(const G4SetupHost&S,const std::array<std::vector<double>,3>&U,std::vector<double>&aval,std::array<std::vector<double>,3>&conv){
  aval=S.topo.val;for(auto&r:conv)r.assign((std::size_t)S.topo.n,0.0);const auto&T=central_tensor_g4();
  for(const auto&cp:S.cells){double I[3][3];for(int j=0;j<3;++j)for(int d=0;d<3;++d)I[j][d]=cp.invJ[3*j+d];double cf[3][8]={{0}};for(int m=0;m<8;++m){auto v=ref_value_host_g4(S,cp.ref[m],U);for(int d=0;d<3;++d)cf[d][m]=v[(std::size_t)d];}double ur[8][3]={{0}};for(int m=0;m<8;++m)for(int j=0;j<3;++j)for(int d=0;d<3;++d)ur[m][j]+=cf[d][m]*I[j][d];double C[8][8]={{0}};for(int a=0;a<8;++a)for(int b=0;b<8;++b){double v=0;for(int m=0;m<8;++m)for(int j=0;j<3;++j)v+=ur[m][j]*T.t[a][m][b][j];C[a][b]=cp.det*v;}for(int a=0;a<8;++a)if(cp.ref[a]>=0){int r=cp.ref[a];for(int b=0;b<8;++b){if(cp.ref[b]>=0){int slot=cp.rowSlot[8*a+b];aval[(std::size_t)(S.topo.row[(std::size_t)r]+slot)]+=C[a][b];}else{auto fv=S.fixedValue[(std::size_t)fixed_slot_from_ref_g4(cp.ref[b])];for(int d=0;d<3;++d)conv[(std::size_t)d][(std::size_t)r]-=C[a][b]*fv[(std::size_t)d];}}}}
}
inline std::vector<double> host_finalize_relax_g4(const G4SetupHost&S,std::vector<double>&a,double alphaU,std::vector<double>*deltaOut=nullptr){std::vector<double>rau((std::size_t)S.topo.n),delta((std::size_t)S.topo.n);double fac=1.0/alphaU-1.0;for(int i=0;i<S.topo.n;++i){auto first=S.topo.col.begin()+S.topo.row[(std::size_t)i],last=S.topo.col.begin()+S.topo.row[(std::size_t)i+1];auto it=std::lower_bound(first,last,i);if(it==last||*it!=i)throw std::runtime_error("G4 host diagonal absent");std::size_t dp=(std::size_t)(it-S.topo.col.begin());double m=0;for(std::int64_t k=S.topo.row[(std::size_t)i];k<S.topo.row[(std::size_t)i+1];++k)m+=std::abs(a[(std::size_t)k]);double de=fac*m;delta[(std::size_t)i]=de;a[dp]+=de;if(!(a[dp]>0))throw std::runtime_error("G4 host relaxed diag invalid");rau[(std::size_t)i]=1.0/a[dp];}if(deltaOut)*deltaOut=std::move(delta);return rau;}

inline double pressure_drop_fit_g4(const SerialTetMesh&M,const std::vector<double>&p){if(p.size()!=M.tets.size())return std::numeric_limits<double>::quiet_NaN();long double sz=0,sp=0,szz=0,szp=0;long double n=p.size();for(std::size_t c=0;c<M.tets.size();++c){double z=0;for(int i=0;i<4;++i)z+=M.points[(std::size_t)M.tets[c][i]].z*.25;sz+=z;sp+=p[c];szz+=z*z;szp+=z*p[c];}long double den=n*szz-sz*sz;if(std::abs((double)den)<1e-300)return 0;long double slope=(n*szp-sz*sp)/den;double zmin=1e300,zmax=-1e300;for(auto&x:M.points){zmin=std::min(zmin,x.z);zmax=std::max(zmax,x.z);}return (double)(-slope*(zmax-zmin));}

} // namespace nodals_gpu
