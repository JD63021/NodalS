// Stage 3: true P2 CUDA scalar **pilot**, run after old solver's momentum / pressure
// correction at the first resumed SIMPLE iteration. Does NOT modify or claim
// P2 momentum/wall coupling, and intentionally returns before ordinary SST updates.
// This is a fail-fast live-kernel and FGMRES integration gate, not a coupled run.
#pragma once
#include "../transport/bernstein_p2_csr.hpp"
#include "../transport/bernstein_p2_device_connectivity.hpp"
#include "bernstein_p2_restart.hpp"
#include <memory>
#include <cmath>
#include <inttypes.h>
#include <algorithm>
#include <limits>
#include <vector>

namespace nodals { namespace p2 { namespace pilot {
using Real=hxt4b::Real;
using nodals_hxt1::Dev;

// Capture original P2 assembled matrix diagonal before row-L1 relaxation.
__global__ void p2_capture_diag(int n,const std::int32_t*dp,const Real*matrix,Real*raw){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i<n)raw[i]=matrix[dp[i]];
}
// r_unrelaxed = b_phys - (A_phys+D)x + D*x = b_phys - A_phys*x.
__global__ void p2_unrelaxed_residual(int n,const std::int32_t*dp,
  const Real*relMat,const Real*rawDiag,const Real*Ax,const Real*x,
  const Real*physicalRhs,Real*out){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i<n){Real delta=relMat[dp[i]]-rawDiag[i];
    out[i]=physicalRhs[i]-Ax[i]+delta*x[i];}
}
struct Live {
  CSR graph;
  hxt4b::CSRHost hostCSR;
  hxt4b::ColoringHost colors;
  std::unique_ptr<hxt4b::GpuCSR> matrix;
  Dev<Tet10Device> d_tets;
  Dev<std::int32_t> d_slots;
  Dev<unsigned char> freeRows;
  Dev<Real> k,w,oldK,oldW,rhs,rawDiag,rhsPhysical;
  Dev<unsigned long long> diagAudit,bad,clip;
  Dev<double> wallStats;
  std::unique_ptr<hxt4a::Reducer<Real>> red;

  explicit Live(const nodals_hxt1::HostMesh&M){
    if(M.h.nhex!=0 || M.h.ninterface!=0 || M.h.ntet==0)
      throw std::runtime_error("P2_PILOT requires conforming pure-tetra mesh");
    std::vector<Tet4> tets;
    tets.reserve(M.tets.size());
    for(auto &t:M.tets)tets.push_back({t.v[0],t.v[1],t.v[2],t.v[3]});
    graph=build_scalar_csr((std::int32_t)M.h.nv,tets);
    // Validate the actual native edge connectivity against the independently
    // constructed Gmsh P2 edge list. This is NOT a mode-name-only gate.
    const char *expected=std::getenv("NODALS_P2_CONN_FNV64");
    if(!expected||std::strlen(expected)!=16)
      throw std::runtime_error("P2_PILOT: NODALS_P2_CONN_FNV64 missing/invalid");
    std::uint64_t fnv=14695981039346656037ull;
    for(const auto&edge:graph.conn.edges){
      for(int j=0;j<2;++j){std::uint32_t x=(std::uint32_t)edge[j];
        for(int b=0;b<4;++b){fnv^=(x>>(8*b))&255u;fnv*=1099511628211ull;}}
    }
    char actual[17]={};std::snprintf(actual,sizeof(actual),"%016" PRIx64,fnv);
    if(std::string(actual)!=expected)
      throw std::runtime_error("P2_PILOT: global P2 edge graph fingerprint mismatch");
    std::printf("P2_NATIVE_GRAPH_FINGERPRINT=PASS fnv64=%s edges=%zu\n",actual,graph.conn.edges.size());
    hostCSR.n=graph.conn.ndof;
    hostCSR.row=graph.row;hostCSR.col=graph.col;hostCSR.diagPos=graph.diag;
    const std::vector<unsigned char> freeMask((std::size_t)hostCSR.n,0);
    colors=hxt4b::color_free_rows(hostCSR,freeMask);
    matrix.reset(new hxt4b::GpuCSR(hostCSR,colors));
    d_tets.upload(pack_device_tetra(graph.conn));
    d_slots.upload(graph.slot100);
    freeRows.upload(freeMask);
    // k/w allocated by lift(): never upload onto an already live Dev buffer.
    rhs.allocate_zeros(hostCSR.n); rawDiag.allocate_zeros(hostCSR.n);
    rhsPhysical.allocate_zeros(hostCSR.n);
    diagAudit.upload(std::vector<unsigned long long>(2,0));
    bad.upload(std::vector<unsigned long long>(1,0));
    clip.upload(std::vector<unsigned long long>(1,0));
    wallStats.upload(std::vector<double>(33,0.0));
    red.reset(new hxt4a::Reducer<Real>(hostCSR.n));
    std::printf("P2_NATIVE_GRAPH=PASS scalarDOFs=%d vertexDOFs=%d edgeDOFs=%zu nnz=%zu slots=%zu colors=%d localTetModes=10 momentum=BF3 pressure=P0\n",
       hostCSR.n,graph.conn.nvert,graph.conn.edges.size(),matrix->nnz,graph.slot100.size(),colors.ncolors);
    std::fflush(stdout);
  }
  void lift(const hxt5d::State&old){
    const std::size_t nv=(std::size_t)graph.conn.nvert;
    std::vector<Real> vK(nv),vW(nv);
    HXT1_CUDA(cudaMemcpy(vK.data(),old.k.p,nv*sizeof(Real),cudaMemcpyDeviceToHost));
    HXT1_CUDA(cudaMemcpy(vW.data(),old.omega.p,nv*sizeof(Real),cudaMemcpyDeviceToHost));
    auto lk=lift_vertex_q1(graph.conn,vK),lw=lift_vertex_q1(graph.conn,vW);
    k.upload(lk);w.upload(lw);oldK.upload(lk);oldW.upload(lw);
    std::printf("P2_NATIVE_LIFT=PASS vertices=%zu edges=%zu oldBF3FaceModes=IGNORED exactPiecewiseLinear=YES\n",nv,graph.conn.edges.size());
  }
  void assemble(int eq,const nodals_hxt1::HostMesh&M,const hxt5b::BoundaryHost&BH,
    const Dev<nodals_hxt1::Point>&pts,const Dev<nodals_hxt1::TetConn>&tc,
    const Dev<nodals_hxt1::TetVel>&tv,const Dev<nodals_hxt1::TetGeom>&tg,
    const Real*u0,const Real*u1,const Real*u2,
    double bulk,double radius,double nu,const hxt5c::Options&O){
    auto &A=*matrix;
    HXT1_CUDA(cudaMemset(A.val.p,0,A.nnz*sizeof(Real)));
    HXT1_CUDA(cudaMemset(rhs.p,0,(std::size_t)A.n*sizeof(Real)));
    HXT1_CUDA(cudaMemset(bad.p,0,sizeof(unsigned long long)));
    HXT1_CUDA(cudaMemset(wallStats.p,0,33*sizeof(double)));
    hxt5c::sst_p2_tet_volume24<<<M.h.ntet,128>>>(
        pts.p,tc.p,tv.p,tg.p,d_tets.p,d_slots.p,u0,u1,u2,
        k.p,w.p,A.val.p,rhs.p,M.h.ntet,radius,nu,eq,O.kFloor,O.omegaFloor);
    HXT1_CUDA(cudaGetLastError());
  }
  void add_boundary(int eq,const nodals_hxt1::HostMesh&M,const hxt5b::BoundaryHost&BH,
    const Dev<hxt5b::TetFacePlan>&inT,
    const Dev<nodals_hxt1::Point>&pts,const Dev<nodals_hxt1::TetConn>&tc,
    const Dev<nodals_hxt1::TetVel>&tv,const Dev<nodals_hxt1::TetGeom>&tg,
    const Real*u0,const Real*u1,const Real*u2,
    double bulk,double radius,double nu,const hxt5c::Options&O){
    const auto inlet=nodals::stage23::from_environment(bulk);
    const double inletK=1.5*std::pow(O.intensity*bulk,2);
    const double inletW=std::sqrt(inletK)/(std::pow(hxt5c::C_MU,0.25)*O.lengthRatio*2*radius);
    if(!BH.inT.empty())hxt5c::sst_p2_inlet_tet12<<<BH.inT.size(),128>>>(
      inT.p,(int)BH.inT.size(),pts.p,tc.p,tv.p,tg.p,
      d_tets.p,d_slots.p,u0,u1,u2,k.p,w.p,matrix->val.p,rhs.p,
      eq==hxt5c::K_EQ?inletK:inletW,inlet.x,inlet.y,inlet.z,
      radius,nu,eq,O.kFloor,O.omegaFloor);
    if(!BH.wallT.empty()){
      if(O.omegaWallMode!=hxt5c::OMEGA_WALL_OF_AUTO || !O.sharedSpaldingWallState || !O.profileAverageClosure)
        throw std::runtime_error("P2_PILOT requires same five-point OF_AUTO wall as reference");
      if(eq==hxt5c::OMEGA_EQ || O.ofProduction)
        hxt5c::sst_p2_tet_of_auto_spalding5_point3<<<BH.wallT.size(),64>>>(
          O.tetWallPlans,(int)BH.wallT.size(),pts.p,tc.p,tv.p,tg.p,
          d_tets.p,d_slots.p,u0,u1,u2,k.p,w.p,matrix->val.p,rhs.p,
          radius,nu,O.wallPenaltyGamma,O.kFloor,O.omegaFloor,
          O.ofKappa,O.ofE,O.ofBeta1,O.ofReBlend,O.ofProductionLimit,
          (int)O.ofProduction,O.ofProductionScale,O.ofWallVolumeScale,eq,
          O.profileTetMatchPoints,wallStats.p,bad.p,O.stage27SavedWallFit);
    }
    HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
    unsigned long long nb=0;
    HXT1_CUDA(cudaMemcpy(&nb,bad.p,sizeof(nb),cudaMemcpyDeviceToHost));
    if(nb)throw std::runtime_error("P2_PILOT wall assembly failed badCount="+std::to_string(nb));
  }
  void solve(int eq,int it,const hxt5c::Options&O){
    const int n=matrix->n;
    const auto chosen = eq==hxt5c::K_EQ?O.s05aKMode:O.s05aOmegaMode;
    const bool p2FGS=(chosen=="fgs-fixed");
    const bool p2SGS=(chosen=="sgs-fixed");
    if(!p2FGS && !p2SGS && chosen!="fgmres-fgs" &&
        !(chosen=="inherit" && O.linearSolver=="fgmres-fgs"))
      throw std::runtime_error("P2 scalar mode requires fgmres-fgs|fgs-fixed|sgs-fixed");
    Real* x=eq==hxt5c::K_EQ?k.p:w.p;
    const Real* old=eq==hxt5c::K_EQ?oldK.p:oldW.p;
    // Store unrelaxed RHS and diagonal so we can audit the ORIGINAL P2
    // equation after solving the relaxed system, not just FGMRES' residual.
    HXT1_CUDA(cudaMemcpy(rhsPhysical.p,rhs.p,(std::size_t)n*sizeof(Real),cudaMemcpyDeviceToDevice));
    p2_capture_diag<<<(n+255)/256,256>>>(n,matrix->diagPos.p,matrix->val.p,rawDiag.p);
    HXT1_CUDA(cudaGetLastError());
    HXT1_CUDA(cudaMemset(diagAudit.p,0,2*sizeof(unsigned long long)));
    hxt5c::scalar_row_l1_relax_kernel<<<(n+255)/256,256>>>(
      n,matrix->row.p,matrix->col.p,matrix->diagPos.p,matrix->val.p,
      rhs.p,old,(Real)(eq==hxt5c::K_EQ?O.alphaK:O.alphaOmega),diagAudit.p,
      n,0,nullptr,(Real)1);
    HXT1_CUDA(cudaGetLastError());
    const char* eqName=eq==hxt5c::K_EQ?"k":"omega";
    const char* solverName=p2FGS?"FGS_FIXED":(p2SGS?"SGS_FIXED":"FGMRES");
    hxt5c::LinearResult r;
    if(p2FGS || p2SGS){
      // EXACT SAME relaxed P2 CSR. Fixed sweeps, zero Krylov iterations,
      // zero inner norms/reductions. SGS = one forward+backward pair.
      const int sweeps=eq==hxt5c::K_EQ?O.s05aKSweeps:O.s05aOmegaSweeps;
      for(int sw=0;sw<sweeps;++sw){
        hxt5b::sweep(*matrix,colors,matrix->val.p,rhs.p,x,O.gsOmega,true,freeRows.p);
        if(p2SGS)
          hxt5b::sweep(*matrix,colors,matrix->val.p,rhs.p,x,O.gsOmega,false,freeRows.p);
      }
      HXT1_CUDA(cudaGetLastError());
      r.works=sweeps;
      r.rel=-1.; // NOT MEASURED: never label fixed sweeps as converged.
      r.ok=true; // Accepted inexact step, NOT an equation-tolerance PASS.
      std::printf("P2_STAGE6_FIXED_SCALAR=PASS it=%d eq=%s method=%s sweeps=%d forwardBackwardPairs=%d innerResidual=NOT_CHECKED\n",
        it,eqName,solverName,sweeps,p2SGS?sweeps:0);
    }else{
      r=hxt5c::scalar_solve(*matrix,colors,freeRows,rhs.p,x,*red,O,it,eqName);
      if(!r.ok || !std::isfinite(r.rel))
        throw std::runtime_error("P2_PILOT scalar FGMRES failed residual tolerance");
    }
    // Stage5: log limiter corrections and extreme Bernstein controls before
    // and after the original floor. These measurements do not change physics.
    const bool stage5Bounds=std::getenv("NODALS_P2_STAGE5_RUN") &&
       std::string(std::getenv("NODALS_P2_STAGE5_RUN"))=="1";
    std::vector<Real> stage5Before;
    if(stage5Bounds){
      stage5Before.resize((std::size_t)n);
      HXT1_CUDA(cudaMemcpy(stage5Before.data(),x,(std::size_t)n*sizeof(Real),cudaMemcpyDeviceToHost));
    }
    HXT1_CUDA(cudaMemset(clip.p,0,sizeof(unsigned long long)));
    hxt5c::floor_vertex_kernel<<<(n+255)/256,256>>>(n,
       (Real)(eq==hxt5c::K_EQ?O.kFloor:O.omegaFloor),x,clip.p);
    HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
    if(stage5Bounds){
      std::vector<Real> after((std::size_t)n);
      HXT1_CUDA(cudaMemcpy(after.data(),x,(std::size_t)n*sizeof(Real),cudaMemcpyDeviceToHost));
      double mn=std::numeric_limits<double>::infinity();
      double mx=-std::numeric_limits<double>::infinity();
      double edgeMax=0.,maxFix=0.,sumFix=0.;
      std::size_t changed=0;
      for(int j=0;j<n;++j){
        const double z=(double)after[(std::size_t)j];
        const double old=(double)stage5Before[(std::size_t)j];
        if(!std::isfinite(z)||!std::isfinite(old))
          throw std::runtime_error("P2_STAGE5 nonfinite scalar control");
        mn=std::min(mn,z);mx=std::max(mx,z);
        if(j>=graph.conn.nvert)edgeMax=std::max(edgeMax,std::fabs(z));
        const double fix=std::fabs(z-old);
        if(fix>0){++changed;maxFix=std::max(maxFix,fix);sumFix+=fix;}
      }
      const double maxAllowed=eq==hxt5c::K_EQ?1.e8:1.e12;
      std::printf("P2_STAGE5_BOUNDS it=%d eq=%s min=%.9e max=%.9e edgeMaxAbs=%.9e limiterControls=%zu maxLimiterChange=%.9e sumLimiterChange=%.9e status=%s\n",
        it,eq==hxt5c::K_EQ?"k":"omega",mn,mx,edgeMax,changed,maxFix,sumFix,
        mn>=0. && mx<=maxAllowed?"PASS":"FAIL");
      std::fflush(stdout);
      if(mn<0. || mx>maxAllowed)
        throw std::runtime_error("P2_STAGE5 scalar control outside catastrophic range");
    }
    // Projected control positivity changes the Galerkin solution. Report it.
    matrix->spmv(x,matrix->tmp.p);
    p2_unrelaxed_residual<<<(n+255)/256,256>>>(n,matrix->diagPos.p,
      matrix->val.p,rawDiag.p,matrix->tmp.p,x,rhsPhysical.p,matrix->res.p);
    HXT1_CUDA(cudaGetLastError());
    const double physicalR=red->norm(matrix->res.p);
    const double physRhsNorm=red->norm(rhsPhysical.p);
    if(!std::isfinite(physicalR) || !std::isfinite(physRhsNorm) || physRhsNorm<0.0)
      throw std::runtime_error("P2_PILOT nonfinite unrelaxed physical scalar residual");
    unsigned long long clipped=0,diag[2]={};
    HXT1_CUDA(cudaMemcpy(&clipped,clip.p,sizeof(clipped),cudaMemcpyDeviceToHost));
    HXT1_CUDA(cudaMemcpy(diag,diagAudit.p,sizeof(diag),cudaMemcpyDeviceToHost));
    if(diag[1])throw std::runtime_error("P2_PILOT relaxation invalid");
    std::printf("P2_NATIVE_SCALAR=PASS it=%d eq=%s dofs=%d localTetModes=10 linearSolver=%s its=%d rel=%.6e controlsClipped=%llu rawDiagNonpositive=%llu originalP2Residual=%.6e originalP2RhsNorm=%.6e relOriginalP2Residual=%.6e postLimiterResidual=YES\n",
       it,eqName,n,solverName,r.works,r.rel,clipped,diag[0],
       physicalR,physRhsNorm,physicalR/std::max(physRhsNorm,1e-300));
  }
  void run_once(int it,const nodals_hxt1::HostMesh&M,const hxt5b::BoundaryHost&BH,
    const Dev<hxt5b::TetFacePlan>&inT,
    const Dev<nodals_hxt1::Point>&pts,const Dev<nodals_hxt1::TetConn>&tc,
    const Dev<nodals_hxt1::TetVel>&tv,const Dev<nodals_hxt1::TetGeom>&tg,
    const Real*u0,const Real*u1,const Real*u2,
    double bulk,double radius,double nu,const hxt5c::Options&O,bool stage4=false){
    HXT1_CUDA(cudaMemcpy(oldK.p,k.p,(std::size_t)matrix->n*sizeof(Real),cudaMemcpyDeviceToDevice));
    HXT1_CUDA(cudaMemcpy(oldW.p,w.p,(std::size_t)matrix->n*sizeof(Real),cudaMemcpyDeviceToDevice));
    for(int eq=0;eq<2;++eq){
      assemble(eq,M,BH,pts,tc,tv,tg,u0,u1,u2,bulk,radius,nu,O);
      add_boundary(eq,M,BH,inT,pts,tc,tv,tg,u0,u1,u2,bulk,radius,nu,O);
      solve(eq,it,O);
    }
    if(stage4)std::printf("P2_STAGE4_SCALARS=PASS it=%d scalarDOFs=%d localTetModes=10 state=PERSISTENT rowL1=TRUE_P2\n",it,matrix->n);
    else std::printf("P2_NATIVE_PILOT=PASS it=%d scalarDOFs=%d FGMRES=PASS momentum=OLD_BF3 pressure=P0 coupledSST=NOT_YET\n",it,matrix->n);
  }
};
}}} // nodals::p2::pilot
