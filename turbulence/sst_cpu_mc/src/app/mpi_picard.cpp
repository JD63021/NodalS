// One genuine MPI momentum prediction / constrained SIMPLE pressure correction
// using M2 mesh/RCB and P1BF3/P0 operators, GPU BF3 T/Q/Tt,
// geometric DG velocity inlet, and GPU portable Stage27/F2 wall mathematics.
// This is a CPU SST momentum-pressure integration check with one correction.
// k/omega, turbulent effective viscosity, SUPG, checkpointing and a production
// SIMPLE loop remain outside its scope. Never report SST/GPU parity here.
#include "../mesh/tet_mesh.hpp"
#include "../parallel/rcb.hpp"
#include "../wall/bf3_flux.hpp"
#include "../pressure/constrained_schur.hpp"
#include "../momentum/m3_predictor.hpp"
#include "../initialization/potential_flow.hpp"
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
 setvbuf(stdout,nullptr,_IONBF,0);
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
  double flux=0;for(int f=0;f<(int)m.faces.size();++f)if(m.faces[f].patch==1){int c=m.faces[f].cell0;for(int lf=0;lf<4;++lf)if(m.tets[c].face[lf]==f){auto A=outward_area(m,c,lf);flux+=A.x*cfg.inlet[0]+A.y*cfg.inlet[1]+A.z*cfg.inlet[2];}}
  if(rank==0)std::printf("M3_INLET_PRESCRIBED_FLUX %.12g\n",flux);
  if(!(flux < 0.0) || !std::isfinite(flux))
    throw std::runtime_error("inlet velocity has non-inward flux for this mesh: CASE_MESH_MISMATCH; use matching .case and .msh, do not flip signs silently");
  const auto initial=potential_velocity(m,cfg.inlet,rank);
  M3Predictor pred(m,own,constraint,cfg,rank,initial);
  if(rank==0)std::printf("NODALS_CPU_SST_M3_MOMENTUM_ASSEMBLED localCellsRank0=%zu incidentTetRank0=%zu localBlockNNZ=%zu time=%.6f\n",pred.local_cells(),pred.incident_cells(),pred.local_nnz(),MPI_Wtime()-t0);
  std::vector<double> u;double m0=0,m1=0;pred.solve(u,cfg.momSweeps,cfg.momOmega,&m0,&m1);
  if(rank==0)std::printf("NODALS_CPU_SST_M3_MOMENTUM_PREDICTED sweeps=%d startingResidual=%.12e endingResidual=%.12e solver=PROCESSOR_BLOCK_3X3_FORWARD_GS\n",cfg.momSweeps,m0,m1);
  const double preFlux=constraint.max_mean_flux(u);
  Schur schur(m,own,constraint,rank,pred.inverse(),cfg.inlet);
  std::vector<double> b,dp,deltaU,bt,cont;
  schur.continuity(u,b);const double cont0=l2(b);
  const auto result=schur.solve(b,dp,cfg.pRtol,cfg.pMaxIts);
  schur.BT(dp,bt);schur.C(bt,deltaU);
  for(std::size_t i=0;i<u.size();++i)u[i]-=deltaU[i];
  schur.continuity(u,cont);const double cont1=l2(cont);
  const double postFlux=constraint.max_mean_flux(u);
  double activeContinuitySquared=0.0;
  for(std::size_t i=1;i<cont.size();++i)activeContinuitySquared+=cont[i]*cont[i];
  const double activeContinuity=std::sqrt(activeContinuitySquared);
  std::vector<double> sdp;schur.apply(dp,sdp);
  double error=0;for(std::size_t i=1;i<cont.size();++i){const double d=cont[i]-b[i]+sdp[i];error+=d*d;}
  error=std::sqrt(error)/std::max(cont0,1e-200);
  double normU=l2(u);
  if(!(std::isfinite(normU)&&std::isfinite(error)&&std::isfinite(cont1)))throw std::runtime_error("non-finite momentum/pressure fields");
  const bool accepted=(error<1e-8 && postFlux<1e-10 && normU>0 && result.converged);
  if(rank==0){
   std::printf("NODALS_CPU_SST_M3_PRESSURE pcgPreconditioner=AGG_SGS pcgIterations=%d pcgConverged=%d initialResidual=%.12e finalResidual=%.12e preCorrectionContinuity=%.12e postCorrectionContinuityFull=%.12e postCorrectionContinuityActive=%.12e pinPressureCell=0\n",result.iterations,(int)result.converged,result.r0,result.rFinal,cont0,cont1,activeContinuity);
   std::printf("NODALS_CPU_SST_M3_CONSTRAINT wallMeanFluxMaxBefore=%.12e wallMeanFluxMaxAfter=%.12e correctionIdentityRel=%.12e velocityNorm=%.12e\n",preFlux,postFlux,error,normU);
   std::printf("NODALS_CPU_SST_M3_SCOPE realMomentum=YES matrixFreeLocalElements=YES nontrivialInlet=YES devStress=YES advectiveConvection=SPATIALLY_VARYING_POTENTIAL_PICARD exactGPU_BF3_T_Q_Tt=YES pressureSchur=B_TQ_D_QTt_BT pressurePCG=YES pressureAMG=ADAPTED_CPU_AGG_SGS momentumSmoother=PROCESSOR_BLOCK_3X3_FORWARD_GS wallSpalding=GPU_F2_ACTUAL_RAYS sstTurbulence=NOT_YET numSimpleCorrections=1\n");
   std::printf("NODALS_CPU_SST_M3_RESULT status=%s elapsedSeconds=%.6f NOTE=ONE_PICARD_NOT_YET_FULL_SST_GPU_PARITY\n",accepted?"PASS":"CHECK",MPI_Wtime()-t0);
   std::fflush(stdout);
  }
  MPI_Finalize();return accepted?0:1;
 }catch(const std::exception&e){std::fprintf(stderr,"NODALS_CPU_SST_M3_FATAL rank=%d error=%s\n",rank,e.what());std::fflush(stderr);MPI_Abort(MPI_COMM_WORLD,2);return 2;}
}
