#pragma once
// Experimental, opt-in omega-only DG coupling to existing G2 SIMPLE.
// No changes to velocity/pressure/k discretization or CG default.
// The compact P1 output is ONLY a visualization/restart-UNSAFE projection.
#include "../transport/dg_omega_hxt1_topology.hpp"
#include "../transport/dg_omega_gpu_block_gs.cuh"
#include "../transport/dg_omega_sst_volume.cuh"
#include "../transport/dg_omega_gpu_limiter.cuh"
#include "../transport/dg_omega_wall_q12.cuh"
#include <array>
#include <map>
#include <memory>
#include <type_traits>
#include <cmath>

namespace nodals {namespace omega_dg_gpu {
struct CoupledOmegaState {
  BSR4Topology topo;
  nodals_hxt1::Dev<Cell> cells;
  nodals_hxt1::Dev<Face> faces;
  nodals_hxt1::Dev<int> row,col,diagonal,wallFaceIndex,cellVertices;
  nodals_hxt1::Dev<double> omega,previous,candidate,change;
  cublasHandle_t reductionHandle=nullptr;
  double lastUpdateRelative=0.;
  nodals_hxt1::Dev<double> matrix,rhs;
  nodals_hxt1::Dev<double> wallTarget,wallMu;
  nodals_hxt1::Dev<double> cgOutput;
  nodals_hxt1::Dev<unsigned int> cgCounts,error;
  nodals_hxt1::Dev<PhysicsAudit> physicsCounters;
  nodals_hxt1::Dev<LimiterCounters> limiterCounters;
  std::unique_ptr<DeviceColoring> colors;
  double inletOmega,traceGamma,convectionTheta=1.,sipgPenalty=10.;
  const double* wallDistanceDevice=nullptr;
  int nvertex=0,nwall=0;
  CoupledOmegaState(const nodals_hxt1::HostMesh&M,
                    const hxt5b::BoundaryHost&BH,
                    double omegaIn,double gamma,const double*distanceDevice)
      :topo(build_topology(M)),inletOmega(omegaIn),traceGamma(gamma),
       wallDistanceDevice(distanceDevice){
    if(!wallDistanceDevice)throw std::runtime_error("DG omega G2 requires geometric Stage20 wall distance");
    if(const char* raw=std::getenv("NODALS_DG_OMEGA_CONVECTION_THETA")){
      try {
        std::size_t end=0;
        const std::string value(raw);
        convectionTheta=std::stod(value,&end);
        if(end!=value.size())throw std::runtime_error("trailing characters");
      } catch(const std::exception&) {
        throw std::runtime_error("DG omega convection_theta must be a finite number in [0,1]");
      }
    }
    if(!std::isfinite(convectionTheta)||convectionTheta<0.||convectionTheta>1.)
      throw std::runtime_error("DG omega convection_theta must lie in [0,1]");
    if(const char* raw=std::getenv("NODALS_DG_OMEGA_SIPG_PENALTY")){
      try {
        std::size_t end=0;
        const std::string value(raw);
        sipgPenalty=std::stod(value,&end);
        if(end!=value.size())throw std::runtime_error("trailing characters");
      }catch(const std::exception&){
        throw std::runtime_error("DG omega sipg_penalty must be a finite number in (0,100000]");
      }
    }
    if(!std::isfinite(sipgPenalty)||sipgPenalty<=0.||sipgPenalty>100000.)
      throw std::runtime_error("DG omega sipg_penalty must lie in (0,100000]");
    if(sizeof(hxt5c::Real)!=sizeof(double))
      throw std::runtime_error("DG omega G2 requires FP64 NodalS build");
    if(!(omegaIn>0.)||!(gamma>=10.)||!(gamma<=1.e5))
      throw std::runtime_error("DG omega G2 inlet omega / trace gamma invalid");
    nvertex=(int)M.h.nv;
    nwall=(int)BH.wallT.size();
    if(nwall<=0||(std::size_t)nwall!=topo.wallFaces)
      throw std::runtime_error("DG omega G2 wall patch count mismatch");
    for(auto &f:topo.faces)if(f.right<0){
      if(f.boundary==Inlet||f.boundary==Outlet)
        for(double&t:f.target)t=omegaIn;
    }
    const int nt=(int)topo.cells.size();
    // Map physical face keys to the real native wall-plan order.
    std::map<std::array<int,3>,int> exterior;
    for(int fi=0;fi<(int)topo.faces.size();++fi){
      const auto &f=topo.faces[(std::size_t)fi];
      if(f.right>=0||f.boundary!=Wall)continue;
      std::array<int,3> key{};
      for(int j=0;j<3;++j)
        key[(std::size_t)j]=M.tets[(std::size_t)f.left].v[f.vertexL[j]];
      std::sort(key.begin(),key.end());
      if(!exterior.emplace(key,fi).second)
        throw std::runtime_error("DG omega duplicate wall face key");
    }
    std::vector<int> faceLookup((std::size_t)nwall);
    std::vector<unsigned char> seen(topo.faces.size(),0);
    for(int i=0;i<nwall;++i){
      auto plan=BH.wallT[(std::size_t)i];
      if(plan.cell<0||(std::size_t)plan.cell>=M.tets.size())
        throw std::runtime_error("DG omega invalid wall plan cell");
      // NodalS native outward tetra local face ordering.
      constexpr int localFace[4][3]={{1,2,3},{0,3,2},{0,1,3},{0,2,1}};
      if(plan.lf<0||plan.lf>3)throw std::runtime_error("DG omega invalid wall face id");
      std::array<int,3> key{};
      for(int j=0;j<3;++j)
        key[(std::size_t)j]=M.tets[(std::size_t)plan.cell].v[localFace[(int)plan.lf][j]];
      std::sort(key.begin(),key.end());
      const auto it=exterior.find(key);
      if(it==exterior.end())throw std::runtime_error("DG omega wall plan not in DG topology");
      const Face &f=topo.faces[(std::size_t)it->second];
      if(f.left!=plan.cell||seen[(std::size_t)it->second]++)
        throw std::runtime_error("DG omega wrong or duplicate wall plan");
      faceLookup[(std::size_t)i]=it->second;
    }
    cells.upload(topo.cells);
    faces.upload(topo.faces);
    row.upload(topo.rowPtr);
    col.upload(topo.colInd);
    diagonal.upload(topo.diagonalSlots);
    wallFaceIndex.upload(faceLookup);
    std::vector<int> con((std::size_t)4*nt);
    for(int t=0;t<nt;++t)
      for(int a=0;a<4;++a)
        con[(std::size_t)4*t+a]=M.tets[(std::size_t)t].v[a];
    cellVertices.upload(con);
    std::vector<double> initial((std::size_t)4*nt,omegaIn);
    omega.upload(initial);
    previous.upload(initial);
    candidate.upload(initial);
    change.allocate_zeros(initial.size());
    blas_ok(cublasCreate(&reductionHandle),"DG omega G2 reduction handle");
    matrix.allocate_zeros((std::size_t)16*topo.nblocks());
    rhs.allocate_zeros(initial.size());
    wallTarget.allocate_zeros((std::size_t)12*nwall);
    wallMu.allocate_zeros((std::size_t)12*nwall);
    cgOutput.allocate_zeros((std::size_t)nvertex);
    cgCounts.allocate_zeros((std::size_t)nvertex);
    error.allocate_zeros(1);
    physicsCounters.allocate_zeros(1);
    limiterCounters.allocate_zeros(1);
    colors.reset(new DeviceColoring(color_bsr4(nt,topo.rowPtr,topo.colInd)));
    std::printf("DG_OMEGA_G2_INIT tets=%d DGdofs=%d faceBlocks=%zu colors=%d "
       "wallFaces=%d inletOmega=%.9e traceGamma=%.9e convectionTheta=%.3f sipgPenalty=%.9g mode=TRUE_DG_P1\n",
       nt,4*nt,topo.nblocks(),colors->host.ncolors,nwall,omegaIn,traceGamma,convectionTheta,sipgPenalty);
  }
  ~CoupledOmegaState(){
    hxt5c::bind_dg_omega_wall_q12(nullptr,nullptr);
    hxt5c::bind_dg_omega_p1(nullptr);
    if(reductionHandle)cublasDestroy(reductionHandle);
  }
  void bind_constitutive()const{
    hxt5c::bind_dg_omega_p1(reinterpret_cast<const hxt5c::Real*>(omega.p));
  }
  void prepare_wall_capture(){
    HXT1_CUDA(cudaMemset(wallTarget.p,0,wallTarget.n*sizeof(double)));
    HXT1_CUDA(cudaMemset(wallMu.p,0,wallMu.n*sizeof(double)));
    hxt5c::bind_dg_omega_wall_q12(wallTarget.p,wallMu.p);
  }
  void finish_wall_capture(){
    hxt5c::bind_dg_omega_wall_q12(nullptr,nullptr);
  }
  void advance(int outer,const nodals_hxt1::Dev<nodals_hxt1::TetConn>&tc,
      const nodals_hxt1::Dev<nodals_hxt1::TetVel>&tv,
      const nodals_hxt1::Dev<nodals_hxt1::TetGeom>&tg,
      const hxt5b::TetFacePlan*wallPlans,
      const double*u0,const double*u1,const double*u2,
      const double*kCG,
      const hxt5c::Options&O,double radius,double nu,
      double*cgOmegaOutput,unsigned long long*oldClipCounter){
    const int nt=(int)topo.cells.size(), nf=(int)topo.faces.size();
    if(!wallPlans||!wallDistanceDevice||!kCG||!cgOmegaOutput||!oldClipCounter)
      throw std::runtime_error("DG omega G2 missing native wall/k/velocity buffers");
    HXT1_CUDA(cudaMemcpy(previous.p,omega.p,omega.n*sizeof(double),cudaMemcpyDeviceToDevice));
    HXT1_CUDA(cudaMemcpy(candidate.p,omega.p,omega.n*sizeof(double),cudaMemcpyDeviceToDevice));
    HXT1_CUDA(cudaMemset(matrix.p,0,matrix.n*sizeof(double)));
    HXT1_CUDA(cudaMemset(rhs.p,0,rhs.n*sizeof(double)));
    HXT1_CUDA(cudaMemset(error.p,0,sizeof(unsigned int)));
    HXT1_CUDA(cudaMemset(physicsCounters.p,0,sizeof(PhysicsAudit)));
    HXT1_CUDA(cudaMemset(limiterCounters.p,0,sizeof(LimiterCounters)));
    PhysicalSSTConfig physical{};
    physical.radius=radius;physical.nu=nu;
    physical.kFloor=O.kFloor;physical.omegaFloor=O.omegaFloor;
    physical.deviatoricProduction=(int)O.deviatoricProduction;
    sst_dg_omega_volume<<<(nt+127)/128,128>>>(
      nt,cells.p,tc.p,tv.p,u0,u1,u2,kCG,previous.p,
      wallDistanceDevice,diagonal.p,matrix.p,rhs.p,physical,physicsCounters.p);
    HXT1_CUDA(cudaGetLastError());
    Controls faceOptions;
    faceOptions.penaltyC=sipgPenalty;
    faceOptions.theta=convectionTheta;
    faceOptions.skipWall=true;
    // Upwind inflow and SIPG interior; wall handled by original Q12 closure.
    ::nodals::omega_dg_gpu::faces<<<(nf+127)/128,128>>>(
      nf,faces.p,cells.p,faceOptions,matrix.p,rhs.p,error.p);
    HXT1_CUDA(cudaGetLastError());
    physical_trace_log_wall_q12<<<(nwall+127)/128,128>>>(
      nwall,wallPlans,wallFaceIndex.p,tg.p,diagonal.p,
      wallTarget.p,wallMu.p,traceGamma,matrix.p,rhs.p,error.p);
    HXT1_CUDA(cudaGetLastError());
    HXT1_CUDA(cudaDeviceSynchronize());
    const auto ph=download_physics_audit();
    unsigned int bad=0;
    HXT1_CUDA(cudaMemcpy(&bad,error.p,sizeof(bad),cudaMemcpyDeviceToHost));
    if(bad||ph.invalid)
      throw std::runtime_error("DG omega G2 invalid volume or wall assembly: "+
        std::to_string(ph.invalid)+" / "+std::to_string(bad));
    MatrixView A{nt,row.p,col.p,diagonal.p,matrix.p};
    const auto linear=block_gs(A,*colors,rhs.p,candidate.p,2,false,1.);
    relax_and_limit<<<(nt+127)/128,128>>>(
      nt,previous.p,candidate.p,omega.p,O.alphaOmega,O.omegaFloor,limiterCounters.p);
    HXT1_CUDA(cudaGetLastError());
    HXT1_CUDA(cudaDeviceSynchronize());
    LimiterCounters limited{};
    HXT1_CUDA(cudaMemcpy(&limited,limiterCounters.p,sizeof(limited),cudaMemcpyDeviceToHost));
    if(limited.nonfiniteCells)
      throw std::runtime_error("DG omega G2 invalid input or nonfinite update: "+
        std::to_string(limited.nonfiniteCells));
    // Mean-deficit cells now trigger positivity-preserving backtracking,
    // not any mass-creating replacement with an artificial floor.
    if(outer<=5||outer%10==0)
      std::printf("DG_OMEGA_G2_POSITIVITY it=%d limited=%llu backtracked=%llu "
                  "heavilyBacktracked=%llu stalled=%llu negativeCandidates=%llu\n",
        outer,limited.limitedCells,limited.meanDeficitCells,
        limited.heavilyBacktrackedCells,limited.stalledCells,
        limited.negativeCandidateDofs);
    // True DG update norm, not the artificially smoothed vertex projection.
    HXT1_CUDA(cudaMemcpy(change.p,omega.p,omega.n*sizeof(double),cudaMemcpyDeviceToDevice));
    const double minusOne=-1.;
    blas_ok(cublasDaxpy(reductionHandle,(int)omega.n,&minusOne,
                       previous.p,1,change.p,1),"DG omega subtract previous");
    double updateNorm=0.,fieldNorm=0.;
    blas_ok(cublasDnrm2(reductionHandle,(int)omega.n,change.p,1,&updateNorm),
            "DG omega update norm");
    blas_ok(cublasDnrm2(reductionHandle,(int)omega.n,omega.p,1,&fieldNorm),
            "DG omega field norm");
    lastUpdateRelative=updateNorm/std::max(fieldNorm,1.e-300);
    if(!std::isfinite(lastUpdateRelative))
      throw std::runtime_error("DG omega nonfinite true field update norm");
    // Project to CG vertex values ONLY for existing VTU/checkpoint outputs.
    HXT1_CUDA(cudaMemset(cgOutput.p,0,cgOutput.n*sizeof(double)));
    HXT1_CUDA(cudaMemset(cgCounts.p,0,cgCounts.n*sizeof(unsigned int)));
    cg_vertex_visualization_projection<<<(nt+127)/128,128>>>(
      nt,cellVertices.p,nvertex,omega.p,cgOutput.p,cgCounts.p);
    cg_vertex_visualization_finalize<<<(nvertex+127)/128,128>>>(
      nvertex,cgOutput.p,cgCounts.p);
    HXT1_CUDA(cudaGetLastError());
    HXT1_CUDA(cudaMemcpy(cgOmegaOutput,cgOutput.p,
      cgOutput.n*sizeof(double),cudaMemcpyDeviceToDevice));
    // Existing old CG extra enrichment DOFs were initialized/fixed at zero.
    const unsigned long long clips=limited.limitedCells;
    HXT1_CUDA(cudaMemcpy(oldClipCounter,&clips,sizeof(clips),cudaMemcpyHostToDevice));
    if(outer<=5||outer%10==0)
      std::printf("DG_OMEGA_G2_STEP it=%d solver=BLOCK_FGS sweeps=%d colors=%d "
        "limited=%llu backtracked=%llu negativeCandidates=%llu omegaFloor=%g alpha=%g "
        "q12Wall=%d updateRel=%.9e convectionTheta=%.3f sipgPenalty=%.9g source=SST2003M_TRUE_DG feedback=ELEMENT_LOCAL\n",
        outer,linear.sweeps,linear.ncolors,limited.limitedCells,
        limited.meanDeficitCells,limited.negativeCandidateDofs,O.omegaFloor,O.alphaOmega,nwall,
        lastUpdateRelative,convectionTheta,sipgPenalty);
  }
  PhysicsAudit download_physics_audit()const{
    PhysicsAudit a{};
    HXT1_CUDA(cudaMemcpy(&a,physicsCounters.p,sizeof(a),cudaMemcpyDeviceToHost));
    return a;
  }
};
}} // nodals::omega_dg_gpu
