// Full U-bend-mesh native-field SST-DG omega assembly and inexact block FGS smoke.
// The mesh is real; velocity,k,omega and wall distance are explicitly SYNTHETIC.
// Does NOT claim true U-bend physics or G2 SIMPLE-loop integration.
#include "../dg_omega_hxt1_topology.hpp"
#include "../dg_omega_gpu_block_gs.cuh"
#include "../dg_omega_sst_volume.cuh"
#include "../dg_omega_gpu_limiter.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
namespace dg=nodals::omega_dg_gpu;
template<class T>struct DevBuf{
 T*p=nullptr;size_t n=0;
 explicit DevBuf(size_t count):n(count){dg::cuda_ok(cudaMalloc((void**)&p,n*sizeof(T)),"SST smoke cudaMalloc");}
 ~DevBuf(){if(p)cudaFree(p);}
 DevBuf(const DevBuf&)=delete;DevBuf&operator=(const DevBuf&)=delete;
 void send(const std::vector<T>&a){
   if(a.size()!=n)throw std::runtime_error("host/device length mismatch");
   dg::cuda_ok(cudaMemcpy(p,a.data(),n*sizeof(T),cudaMemcpyHostToDevice),"SST smoke upload");
 }
 std::vector<T>recv() const{
   std::vector<T>a(n);dg::cuda_ok(cudaMemcpy(a.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost),"SST smoke download");
   return a;
 }
 void zero(){dg::cuda_ok(cudaMemset(p,0,n*sizeof(T)),"SST smoke memset");}
};
int main(int argc,char**argv){
 try{
   if(argc<2)throw std::runtime_error("usage: dg_omega_sst_smoke HXT1_mesh.bin [--sweeps 2] [--alpha 0.7] [--wall-distance 0.01]");
   std::string path=argv[1];int sweeps=2;
   double alpha=.7,wallDistance=.01;
   for(int i=2;i<argc;++i){
     std::string key=argv[i];
     if(i+1>=argc)throw std::runtime_error("missing value: "+key);
     const std::string arg=argv[++i];
     if(key=="--sweeps")sweeps=std::stoi(arg);
     else if(key=="--alpha")alpha=std::stod(arg);
     else if(key=="--wall-distance")wallDistance=std::stod(arg);
     else throw std::runtime_error("unknown option: "+key);
   }
   if(sweeps<0||sweeps>100||!(alpha>0&&alpha<=1)||!(wallDistance>0))
     throw std::runtime_error("invalid sweep or synthetic coefficients");
   auto M=nodals_hxt1::load(path);
   auto topology=dg::build_topology(M);
   const int nt=(int)topology.cells.size(),nd=4*nt,nv=(int)M.h.nv,ng=(int)M.h.nvel;
   const int nb=(int)topology.nblocks(),nf=(int)topology.faces.size();
   if(!(ng>=nv))throw std::runtime_error("invalid native velocity allocation");
   const double omegaOld=100.,kOld=0.01;
   std::vector<double> zeros((size_t)ng,0.),k((size_t)nv,kOld),
                       omega((size_t)nd,omegaOld),
                       distances((size_t)nv,wallDistance);
   for(auto&f:topology.faces)if(f.right<0)
      for(double&target:f.target)target=omegaOld;
   DevBuf<dg::Cell> cells((size_t)nt);cells.send(topology.cells);
   DevBuf<dg::Face> faces((size_t)nf);faces.send(topology.faces);
   DevBuf<nodals_hxt1::TetConn>tc((size_t)nt);tc.send(M.tets);
   DevBuf<nodals_hxt1::TetVel>tv((size_t)nt);tv.send(M.tetVel);
   DevBuf<int> row(topology.rowPtr.size());row.send(topology.rowPtr);
   DevBuf<int> col(topology.colInd.size());col.send(topology.colInd);
   DevBuf<int> diagonal(topology.diagonalSlots.size());diagonal.send(topology.diagonalSlots);
   DevBuf<double>u0((size_t)ng),u1((size_t)ng),u2((size_t)ng),kCG((size_t)nv);
   u0.send(zeros);u1.send(zeros);u2.send(zeros);kCG.send(k);
   DevBuf<double>distance((size_t)nv);distance.send(distances);
   DevBuf<double>old((size_t)nd),solution((size_t)nd),candidate((size_t)nd);
   old.send(omega);solution.send(omega);candidate.send(omega);
   DevBuf<double>matrix((size_t)nb*16),rhs((size_t)nd),ax((size_t)nd);
   DevBuf<unsigned int>bad(1);
   DevBuf<dg::PhysicsAudit> audit(1);
   DevBuf<dg::LimiterCounters> limit(1);
   matrix.zero();rhs.zero();bad.zero();audit.zero();limit.zero();
   dg::PhysicalSSTConfig opt;
   opt.radius=.04445; // required but geometric wall distance is passed independently
   opt.nu=1.460734693878e-5;opt.kFloor=1.e-8;
   opt.omegaFloor=1.e-3;opt.deviatoricProduction=1;
   auto t0=std::chrono::steady_clock::now();
   dg::sst_dg_omega_volume<<<(nt+127)/128,128>>>(
     nt,cells.p,tc.p,tv.p,u0.p,u1.p,u2.p,kCG.p,old.p,
     distance.p,diagonal.p,matrix.p,rhs.p,opt,audit.p);
   dg::cuda_ok(cudaGetLastError(),"DG SST volume launch");
   dg::Controls faceOpt;faceOpt.penaltyC=10.;faceOpt.traceGamma=100000.;
   dg::faces<<<(nf+127)/128,128>>>(
       nf,faces.p,cells.p,faceOpt,matrix.p,rhs.p,bad.p);
   dg::cuda_ok(cudaGetLastError(),"DG SST faces launch");
   dg::cuda_ok(cudaDeviceSynchronize(),"DG SST assembly synchronize");
   auto a=audit.recv()[0];
   auto errors=bad.recv()[0];
   if(a.invalid||errors)
     throw std::runtime_error("nonfinite SST volume / DG face coefficients: volume="+
       std::to_string(a.invalid)+" faces="+std::to_string(errors));
   const double assemblySeconds=std::chrono::duration<double>(
     std::chrono::steady_clock::now()-t0).count();
   auto hostCells=cells.recv();
   double nuTMean=0.,muMean=0.,reactionMean=0.,sourceMean=0.;
   for(const auto&cell:hostCells){
     if(!(cell.mu>=opt.nu)||!std::isfinite(cell.source))
       throw std::runtime_error("bad SST constitutive coefficients");
     muMean+=cell.mu;reactionMean+=cell.reaction;sourceMean+=cell.source;
     nuTMean+=cell.mu-opt.nu;
   }
   muMean/=nt;reactionMean/=nt;sourceMean/=nt;nuTMean/=nt;
   std::printf("DG_OMEGA_SST_PHYSICS_ASSEMBLY_PASS tets=%d dofs=%d "
     "assemblySeconds=%.4f muMean=%.10e nuTContributionMean=%.10e "
     "reactionMean=%.10e sourceMean=%.10e omegaFloored=%llu "
     "kFloored=%llu distanceMode=SYNTHETIC_CONSTANT\n",
     nt,nd,assemblySeconds,muMean,nuTMean,reactionMean,sourceMean,
     a.omegaFloored,a.kFloored);
   // Evaluate true residual with the warm initial condition (omega=100 everywhere).
   dg::MatrixView A{nt,row.p,col.p,diagonal.p,matrix.p};
   const auto hRhs=rhs.recv();
   auto true_rel=[&]()->double{
     dg::spmv(A,candidate.p,ax.p);
     const auto hAx=ax.recv();
     double x=0.,y=0.;
     for(int i=0;i<nd;++i){
       const double d=hRhs[(size_t)i]-hAx[(size_t)i];
       x+=d*d;y+=hRhs[(size_t)i]*hRhs[(size_t)i];
     }
     return std::sqrt(x)/std::max(std::sqrt(y),1e-300);
   };
   const double first=true_rel();
   auto colors=dg::color_bsr4(nt,topology.rowPtr,topology.colInd);
   dg::DeviceColoring deviceColors(std::move(colors));
   t0=std::chrono::steady_clock::now();
   auto step=dg::block_gs(A,deviceColors,rhs.p,candidate.p,sweeps,false,1.);
   const double after=true_rel();
   const double solveSeconds=std::chrono::duration<double>(
      std::chrono::steady_clock::now()-t0).count();
   dg::relax_and_limit<<<(nt+127)/128,128>>>(
     nt,old.p,candidate.p,solution.p,alpha,opt.omegaFloor,limit.p);
   dg::cuda_ok(cudaGetLastError(),"DG omega positivity limiter launch");
   dg::cuda_ok(cudaDeviceSynchronize(),"DG omega limiter synchronize");
   const auto lim=limit.recv()[0];
   const auto w=solution.recv();
   if(lim.nonfiniteCells)throw std::runtime_error("DG omega nonlinear update nonfinite");
   double minW=1e300,maxW=-1e300,meanW=0.;
   for(double x:w){
     if(!std::isfinite(x)||x<opt.omegaFloor*(1.-1e-12))
        throw std::runtime_error("DG omega updated field violates floor");
     minW=std::min(minW,x);maxW=std::max(maxW,x);meanW+=x;
   }
   meanW/=w.size();
   std::printf("DG_OMEGA_SST_WARM_FGS_PASS sweeps=%d colors=%d launches=%d "
     "initialRelative=%.10e finalRelative=%.10e solveSeconds=%.4f "
     "alpha=%.2f omegaMin=%.8e omegaMax=%.8e omegaMean=%.8e "
     "limitedCells=%llu meanDeficitCells=%llu negativeCandidateDofs=%llu\n",
      step.sweeps,step.ncolors,step.launches,first,after,solveSeconds,
      alpha,minW,maxW,meanW,lim.limitedCells,lim.meanDeficitCells,
      lim.negativeCandidateDofs);
   std::puts("DG_OMEGA_SST_STANDALONE_ALL_PASS");
   return 0;
 }catch(const std::exception&e){
   std::fprintf(stderr,"DG_OMEGA_SST_STANDALONE_FAIL reason=%s\n",e.what());
   return 1;
 }
}
