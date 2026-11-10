// One genuine MPI momentum prediction / constrained SIMPLE pressure correction
// using original M2 mesh/RCB/B/Bt, vector deviatoric P1+BF3, GPU BF3 T/Q/Tt,
// physical DG velocity inlet, and conservative volume continuity.
// Explicitly NOT the production SST solve (no k/omega, ray wall, native AMG
// or colored M3 SGS yet). Numerical capabilities are printed, never hidden.
#include "../mesh/tet_mesh.hpp"
#include "../parallel/rcb.hpp"
#include "../wall/bf3_flux.hpp"
#include "../pressure/constrained_schur.hpp"
#include "../momentum/m3_predictor.hpp"
#include "../case/case_config.hpp"
#include <mpi.h>
#include <cstdio>
#include <cstring>
#include <exception>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
using namespace nodals::cpu_mc::m2;
using namespace nodals::cpu_mc::m3;
static double l2(const std::vector<double>&v){double s=0;for(double x:v)s+=x*x;return std::sqrt(s);}
int main(int argc,char**argv){
 int level=MPI_THREAD_SINGLE;MPI_Init_thread(&argc,&argv,MPI_THREAD_SINGLE,&level);
 int rank=0,size=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
 try{
  std::string mesh,filename;int expected=16;
  for(int i=1;i<argc;++i){if(i+1>=argc)throw std::runtime_error("missing argument");std::string k=argv[i],v=argv[++i];
   if(k=="--mesh")mesh=v;else if(k=="--case")filename=v;else if(k=="--expect-ranks")expected=std::stoi(v);
   else throw std::runtime_error("unexpected option: "+k);
  }
  if(size!=expected||filename.empty()||mesh.empty())throw std::runtime_error("required --mesh --case --expect-ranks mismatch");
  const auto cfg=read_case(filename,mesh,expected);const auto m=read_msh2(mesh);
  const auto own=partition_rcb(m,size);const BF3Flux constraint(m);
  const double t0=MPI_Wtime();
  if(rank==0)std::printf("NODALS_CPU_SST_M3_SETUP ranks=%d tets=%zu faces=%zu velocityEntities=%zu wallFaces=%zu convection=%s stress=%s muKinematic=%.12e case=%s\n",size,m.tets.size(),m.faces.size(),m.points.size()+m.faces.size(),constraint.count(),cfg.convection.c_str(),cfg.stress.c_str(),cfg.nu,filename.c_str());
  M3Predictor pred(m,own,constraint,cfg,rank);
  if(rank==0)std::printf("NODALS_CPU_SST_M3_MOMENTUM_ASSEMBLED localCellsRank0=%zu time=%.6f\n",pred.local_cells(),MPI_Wtime()-t0);
  std::vector<double> u;double m0=0,m1=0;pred.solve(u,cfg.momSweeps,cfg.momOmega,&m0,&m1);
  if(rank==0)std::printf("NODALS_CPU_SST_M3_MOMENTUM_PREDICTED sweeps=%d startingResidual=%.12e endingResidual=%.12e solver=COUPLED_RESIDUAL_MIN_JACOBI_DEVELOPMENT_NOT_M3_GS\n",cfg.momSweeps,m0,m1);
  const double preFlux=constraint.max_mean_flux(u);
  Schur schur(m,own,constraint,rank,pred.inverse());
  std::vector<double> b,dp,deltaU,bt,cont;
  schur.B(u,b);const double cont0=l2(b);
  const auto result=schur.solve(b,dp,cfg.pRtol,cfg.pMaxIts);
  schur.BT(dp,bt);schur.C(bt,deltaU);
  for(std::size_t i=0;i<u.size();++i)u[i]-=deltaU[i];
  schur.B(u,cont);const double cont1=l2(cont);
  const double postFlux=constraint.max_mean_flux(u);
  std::vector<double> sdp;schur.apply(dp,sdp);
  double error=0;for(std::size_t i=1;i<cont.size();++i){const double d=cont[i]-b[i]+sdp[i];error+=d*d;}
  error=std::sqrt(error)/std::max(cont0,1e-200);
  double normU=l2(u);
  if(!(std::isfinite(normU)&&std::isfinite(error)&&std::isfinite(cont1)))throw std::runtime_error("non-finite momentum/pressure fields");
  const bool accepted=(error<1e-8 && postFlux<1e-10 && normU>0 && result.converged);
  if(rank==0){
   std::printf("NODALS_CPU_SST_M3_PRESSURE pcgIterations=%d pcgConverged=%d initialResidual=%.12e finalResidual=%.12e preCorrectionContinuity=%.12e postCorrectionContinuity=%.12e\n",result.iterations,(int)result.converged,result.r0,result.rFinal,cont0,cont1);
   std::printf("NODALS_CPU_SST_M3_CONSTRAINT wallMeanFluxMaxBefore=%.12e wallMeanFluxMaxAfter=%.12e correctionIdentityRel=%.12e velocityNorm=%.12e\n",preFlux,postFlux,error,normU);
   std::printf("NODALS_CPU_SST_M3_SCOPE realMomentum=YES matrixFreeLocalElements=YES nontrivialInlet=YES devStress=YES advectiveConvection=FROZEN_INLET_CARRIER exactGPU_BF3_T_Q_Tt=YES pressureSchur=B_TQ_D_QTt_BT pressurePCG=YES pressureAMG=NOT_WIRED momentumSmoother=RESIDUAL_MIN_JACOBI_NOT_COLOR_GS wallSpalding=MOLECULAR_ZERO_CARRIER_ONLY sstTurbulence=NOT_YET numSimpleCorrections=1\n");
   std::printf("NODALS_CPU_SST_M3_RESULT status=%s elapsedSeconds=%.6f NOTE=DEVELOPMENT_PICARD_NOT_GPU_PARITY\n",accepted?"PASS":"CHECK",MPI_Wtime()-t0);
   std::fflush(stdout);
  }
  MPI_Finalize();return accepted?0:1;
 }catch(const std::exception&e){std::fprintf(stderr,"NODALS_CPU_SST_M3_FATAL rank=%d error=%s\n",rank,e.what());std::fflush(stderr);MPI_Abort(MPI_COMM_WORLD,2);return 2;}
}
