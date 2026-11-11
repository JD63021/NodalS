// Full native-HXT1-mesh DG-P1 GPU algebraic smoke.
// Frozen positive constant omega exact solution: A*omega==assembled boundary/volume RHS.
// This deliberately does NOT run nonlinear k-omega SST; it tests mesh, assembly,
// boundary terms, BSR4 matvec and GPU block-FGMRES on the actual U-bend topology.
#include "../dg_omega_hxt1_topology.hpp"
#include "../dg_omega_gpu_fgmres.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>
namespace dg=nodals::omega_dg_gpu;
template<class T>struct DevBuf{
 T*p=nullptr;size_t n=0;
 explicit DevBuf(size_t count):n(count){dg::cuda_ok(cudaMalloc((void**)&p,n*sizeof(T)),"fullmesh allocation");}
 ~DevBuf(){if(p)cudaFree(p);}
 DevBuf(const DevBuf&)=delete;DevBuf&operator=(const DevBuf&)=delete;
 void send(const std::vector<T>&x){
   if(x.size()!=n)throw std::runtime_error("fullmesh host/device vector dimension mismatch");
   dg::cuda_ok(cudaMemcpy(p,x.data(),n*sizeof(T),cudaMemcpyHostToDevice),"fullmesh host->device");
 }
 std::vector<T>recv()const{
   std::vector<T>x(n);dg::cuda_ok(cudaMemcpy(x.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost),"fullmesh device->host");
   return x;
 }
 void zero(){dg::cuda_ok(cudaMemset(p,0,n*sizeof(T)),"fullmesh zero");}
};
struct Args{
 std::string path;
 double mu=1.460734693878e-5,reaction=0.10,target=100.0,traceGamma=100000.,penaltyC=10.;
 int maxIterations=250,restart=20,printEvery=20;
 double rtol=1.e-6;
};
static Args parse(int argc,char**argv){
 if(argc<2)throw std::runtime_error("usage: dg_omega_fullmesh_smoke <HXT1_mesh.bin> [--rtol 1e-6] [--max 250] [--restart 20] [--trace-gamma 100000] [--penalty 10]");
 Args a;a.path=argv[1];
 for(int i=2;i<argc;++i){
   std::string k=argv[i];if(i+1>=argc)throw std::runtime_error("missing value for "+k);
   const std::string v=argv[++i];
   if(k=="--rtol")a.rtol=std::stod(v);
   else if(k=="--max")a.maxIterations=std::stoi(v);
   else if(k=="--restart")a.restart=std::stoi(v);
   else if(k=="--trace-gamma")a.traceGamma=std::stod(v);
   else if(k=="--penalty")a.penaltyC=std::stod(v);
   else if(k=="--mu")a.mu=std::stod(v);
   else if(k=="--reaction")a.reaction=std::stod(v);
   else if(k=="--target")a.target=std::stod(v);
   else if(k=="--print-every")a.printEvery=std::stoi(v);
   else throw std::runtime_error("unsupported argument "+k);
 }
 if(!(a.mu>0.)||!(a.reaction>0.)||!(a.target>0.)||!(a.traceGamma>0.)||
    !(a.penaltyC>0.)||!(a.rtol>0.))
   throw std::runtime_error("invalid physical or solver parameters");
 return a;
}
int main(int argc,char**argv){
 try{
   const Args a=parse(argc,argv);
   const auto before=std::chrono::steady_clock::now();
   auto M=nodals_hxt1::load(a.path);
   dg::BSR4Topology T=dg::build_topology(M);
   const int nt=(int)T.cells.size(),nd=4*nt,nb=(int)T.colInd.size();
   // Frozen constant SST field. Diffusion + reaction, no convection.
   // Exact constant solution with source = reaction*target, and physical
   // Dirichlet target = target on inlet and wall, natural zero-gradient outlet.
   for(auto&c:T.cells){
     c.mu=a.mu;c.reaction=a.reaction;c.source=a.reaction*a.target;
     c.u={0.,0.,0.};
   }
   for(auto&f:T.faces)if(f.right<0)
     for(double&x:f.target)x=a.target;
   std::size_t freeMem=0,totalMem=0;
   dg::cuda_ok(cudaMemGetInfo(&freeMem,&totalMem),"cudaMemGetInfo");
   std::printf("DG_OMEGA_FULLMESH_BEGIN tets=%d dofs=%d interiorFaces=%zu boundaryFaces=%zu "
     "blocks=%d estimateMiB=%.2f gpuFreeMiB=%.2f traceGamma=%.1f mu=%.9e reaction=%.6g\n",
     nt,nd,T.internalFaces,T.wallFaces+T.inletFaces+T.outletFaces,nb,
     T.byte_estimate()/1048576.,freeMem/1048576.,a.traceGamma,a.mu,a.reaction);
   std::fflush(stdout);
   if(freeMem < T.byte_estimate()+256ULL*1048576ULL)
     throw std::runtime_error("DG omega fullmesh insufficient GPU free memory");
   DevBuf<dg::Cell>cells(T.cells.size());cells.send(T.cells);
   DevBuf<dg::Face>faces(T.faces.size());faces.send(T.faces);
   DevBuf<int>row(T.rowPtr.size());row.send(T.rowPtr);
   DevBuf<int>col(T.colInd.size());col.send(T.colInd);
   DevBuf<int>diag(T.diagonalSlots.size());diag.send(T.diagonalSlots);
   DevBuf<double>blocks((size_t)16*nb);
   DevBuf<double>rhs((size_t)nd),x((size_t)nd),exact((size_t)nd),ax((size_t)nd);
   DevBuf<unsigned int>errors(1);
   blocks.zero();rhs.zero();errors.zero();
   const auto beginAssembly=std::chrono::steady_clock::now();
   dg::Controls opt;opt.penaltyC=a.penaltyC;opt.traceGamma=a.traceGamma;
   dg::volume<<<(nt+127)/128,128>>>(nt,cells.p,diag.p,blocks.p,rhs.p,errors.p);
   dg::cuda_ok(cudaGetLastError(),"DG fullmesh volume launch");
   const int nf=(int)T.faces.size();
   dg::faces<<<(nf+127)/128,128>>>(nf,faces.p,cells.p,opt,blocks.p,rhs.p,errors.p);
   dg::cuda_ok(cudaGetLastError(),"DG fullmesh face launch");
   dg::cuda_ok(cudaDeviceSynchronize(),"DG fullmesh assembly synchronize");
   const unsigned int bad=errors.recv()[0];
   if(bad)throw std::runtime_error("DG fullmesh assembly error counter "+std::to_string(bad));
   const double assemblySeconds=std::chrono::duration<double>(
     std::chrono::steady_clock::now()-beginAssembly).count();
   // Construct exact solution exclusively for verification. Solver sees only
   // matrix and independently assembled rhs, not the exact solution.
   std::vector<double>expected((size_t)nd,a.target);
   exact.send(expected);
   dg::MatrixView A{nt,row.p,col.p,diag.p,blocks.p};
   dg::spmv(A,exact.p,ax.p);
   auto hAx=ax.recv(),hRhs=rhs.recv();
   double err2=0,rhs2=0,errMax=0;
   for(int i=0;i<nd;++i){
     const double delta=hAx[(size_t)i]-hRhs[(size_t)i];
     if(!std::isfinite(delta))throw std::runtime_error("DG fullmesh nonfinite manufactured equation");
     err2+=delta*delta;rhs2+=hRhs[(size_t)i]*hRhs[(size_t)i];
     errMax=std::max(errMax,std::abs(delta));
   }
   const double equationRel=std::sqrt(err2)/std::max(std::sqrt(rhs2),1.e-300);
   std::printf("DG_OMEGA_FULLMESH_CONSISTENCY relative=%.12e maxAbsolute=%.12e "
     "assemblySeconds=%.3f status=%s\n",
     equationRel,errMax,assemblySeconds,equationRel<1e-8?"PASS":"FAIL");
   std::fflush(stdout);
   if(!(equationRel<1e-8))throw std::runtime_error(
     "DG omega assembled operator fails exact constant manufactured equation");
   x.zero();
   dg::FgmresOptions solver;
   solver.restart=a.restart;solver.maxIterations=a.maxIterations;
   solver.rtol=a.rtol;solver.printEvery=a.printEvery;
   const auto beginSolve=std::chrono::steady_clock::now();
   const auto result=dg::fgmres(A,rhs.p,x.p,solver);
   dg::cuda_ok(cudaDeviceSynchronize(),"DG fullmesh solve synchronize");
   const double solveSeconds=std::chrono::duration<double>(
     std::chrono::steady_clock::now()-beginSolve).count();
   auto got=x.recv();
   double maxErr=0,rmsErr2=0,minValue=1e300,maxValue=-1e300;
   for(int i=0;i<nd;++i){
     maxErr=std::max(maxErr,std::abs(got[(size_t)i]-a.target));
     rmsErr2+=std::pow(got[(size_t)i]-a.target,2);
     minValue=std::min(minValue,got[(size_t)i]);
     maxValue=std::max(maxValue,got[(size_t)i]);
   }
   const double rmsErr=std::sqrt(rmsErr2/nd);
   std::printf("DG_OMEGA_FULLMESH_FGMRES iterations=%d restarts=%d trueRelative=%.12e "
     "rmsError=%.9e maxError=%.9e minOmega=%.9g maxOmega=%.9g "
     "assemblySeconds=%.3f solveSeconds=%.3f converged=%d\n",
     result.iterations,result.restarts,result.relativeResidual,rmsErr,maxErr,
     minValue,maxValue,assemblySeconds,solveSeconds,int(result.converged));
   if(!result.converged)throw std::runtime_error("DG omega GPU fullmesh FGMRES failed to reach requested residual");
   if(maxErr/a.target>1.e-3)throw std::runtime_error("DG omega fullmesh wrong constant solution despite residual convergence");
   std::puts("DG_OMEGA_FULLMESH_ALL_PASS");
   (void)before;return 0;
 }catch(const std::exception&e){
   std::fprintf(stderr,"DG_OMEGA_FULLMESH_FAIL reason=%s\n",e.what());
   return 1;
 }
}
