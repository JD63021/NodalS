// NodalS CPU SST M4: repeated MPI SIMPLE, coupled P1BF3 momentum,
// exact pressure Schur PCG/aggregation AMG, and masked Q1 SST transport.
// This executable is distinct from the untouched M3 numerical reference.
#include "../mesh/tet_mesh.hpp"
#include "../parallel/rcb.hpp"
#include "../wall/bf3_flux.hpp"
#include "../momentum/m3_predictor.hpp"
#include "../pressure/constrained_schur.hpp"
#include "../initialization/potential_flow.hpp"
#include "../geometry/wall_distance.hpp"
#include "../transport/sst_transport.hpp"
#include "../case/case_config.hpp"
#include "../output/flow_vtu.hpp"
#include <mpi.h>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <algorithm>
#include <numeric>
using namespace nodals::cpu_mc::m2;
using namespace nodals::cpu_mc::m3;
using namespace nodals::cpu_mc::m4;
namespace {
double norm(const std::vector<double>&v){double a=0;for(double q:v)a+=q*q;return std::sqrt(a);}
double sum(const std::vector<double>&v){return std::accumulate(v.begin(),v.end(),0.0);}
std::string expand_home(std::string p){if(p.rfind("~/",0)==0){const char*h=getenv("HOME");if(h)p=std::string(h)+p.substr(1);}return p;}
}
int main(int argc,char**argv){
 std::setvbuf(stdout,nullptr,_IONBF,0);
 int provided=0;MPI_Init_thread(&argc,&argv,MPI_THREAD_SINGLE,&provided);
 int rank=0,size=1;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
 try{
  std::string mesh,casePath,outputOverride;int expected=16,cap=0;
  for(int i=1;i<argc;++i){if(i+1>=argc)throw std::runtime_error("argument requires a value");
   const std::string k=argv[i],v=argv[++i];
   if(k=="--mesh")mesh=v;else if(k=="--case")casePath=v;
   else if(k=="--expect-ranks")expected=std::stoi(v);
   else if(k=="--max-outer")cap=std::stoi(v);
   else if(k=="--output-dir")outputOverride=v;
   else throw std::runtime_error("unknown option: "+k);
  }
  if(mesh.empty()||casePath.empty()||size!=expected||cap<0)throw std::runtime_error("need matching --mesh --case --expect-ranks and optional positive --max-outer");
  CaseConfig cfg=read_case(casePath,mesh,expected);
  if(cap>0)cfg.maxOuter=std::min(cfg.maxOuter,cap);
  const Mesh m=read_msh2(mesh);
  const auto own=partition_rcb(m,size);
  const BF3Flux wall(m);
  auto distance=geometric_wall_distance(m,rank);
  std::vector<double> u=potential_velocity(m,cfg.inlet,rank);
  SSTTransport sst(m,own,cfg,rank,distance);
  std::vector<double> p(m.tets.size(),0.0);
  std::string dir=expand_home(outputOverride.empty()?cfg.outputDirectory:outputOverride);
  if(rank==0)std::filesystem::create_directories(dir);
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank==0)std::printf("NODALS_CPU_SST_M4_SETUP status=PASS ranks=%d cells=%zu vertices=%zu faces=%zu maxOuter=%d transport=Q1_MASKED_P1BF3 matrix=ENRICHED momentum=M3_BLOCK_MPI pressure=PCG_AMG_SGS wall=SPALDING_PLUS_TRACE_LOG case=%s\n",size,m.tets.size(),m.points.size(),m.faces.size(),cfg.maxOuter,casePath.c_str());
  const double start=MPI_Wtime();
  bool allLinearOkay=true;int completed=0,refreshes=0;
  double lastFull=0,prevMeanNut=-1;
  for(int it=1;it<=cfg.maxOuter;++it){
   const double t0=MPI_Wtime();
   const auto oldU=u,oldK=sst.state().k,oldOmega=sst.state().omega;
   M3Predictor mom(m,own,wall,cfg,rank,oldU,&oldK,&oldOmega,&distance);
   Schur pressure(m,own,wall,rank,mom.inverse(),cfg.inlet);
   std::vector<double> gradP,uStar,cont,dp,bt,deltaU;
   pressure.BT(p,gradP);
   double mom0=0,mom1=0;
   mom.solve(uStar,cfg.momSweeps,cfg.momOmega,&mom0,&mom1,&gradP);
   pressure.continuity(uStar,cont);
   const double pre=norm(cont),preSum=sum(cont);
   const auto pr=pressure.solve(cont,dp,cfg.pRtol,cfg.pMaxIts);
   pressure.BT(dp,bt);pressure.C(bt,deltaU);
   u=std::move(uStar);
   for(size_t i=0;i<u.size();++i)u[i]-=deltaU[i];
   for(size_t c=0;c<p.size();++c)p[c]-=cfg.alphaP*dp[c];
   pressure.continuity(u,cont);
   const double post=norm(cont),postPin=cont[0],postSum=sum(cont);
   double active=0;for(size_t i=1;i<cont.size();++i)active+=cont[i]*cont[i];active=std::sqrt(active);
   const double fluxMax=wall.max_mean_flux(u);
   const auto kRes=sst.advance_k(u);
   const auto wRes=sst.advance_omega(u,mom.wall_fit());
   const double nut=sst.mean_nut(u);
   double deltaK=0,baseK=0,deltaW=0,baseW=0;
   for(size_t i=0;i<oldK.size();++i){deltaK+=std::pow(sst.state().k[i]-oldK[i],2);baseK+=oldK[i]*oldK[i];deltaW+=std::pow(sst.state().omega[i]-oldOmega[i],2);baseW+=oldOmega[i]*oldOmega[i];}
   const double kRel=std::sqrt(deltaK/std::max(baseK,1e-300));
   const double wRel=std::sqrt(deltaW/std::max(baseW,1e-300));
   const double nutRel=prevMeanNut>0?std::abs(nut-prevMeanNut)/prevMeanNut:1;
   prevMeanNut=nut;lastFull=post;completed=it;
   allLinearOkay=allLinearOkay&&pr.converged;
   if(rank==0 && (it<=3||it%cfg.printEvery==0||!pr.converged||it==cfg.maxOuter)){
    std::printf("NODALS_CPU_SST_M4_SIMPLE it=%d momStart=%.12e momEnd=%.12e pcg=%d pcgConverged=%d pressureR=%.12e contPre=%.12e contPostFull=%.12e contPostActive=%.12e pinnedCellResidual=%.12e sumContPre=%.12e sumContPost=%.12e wallMaxFlux=%.12e kRel=%.6e omegaRel=%.6e nutRel=%.6e meanNutOverNu=%.7e kRawRes=%.6e->%.6e omegaRawRes=%.6e->%.6e kClips=%d omegaClips=%d wallSeconds=%.6f\n",
     it,mom0,mom1,pr.iterations,(int)pr.converged,pr.rFinal,pre,post,active,postPin,preSum,postSum,fluxMax,kRel,wRel,nutRel,nut/cfg.nu,kRes.before,kRes.after,wRes.before,wRes.after,kRes.clips,wRes.clips,MPI_Wtime()-t0);
   }
   if(cfg.vtuEvery>0 && (it%cfg.vtuEvery==0 || it==cfg.maxOuter)){
    if(rank==0){const std::string fn=dir+"/flow_sst_cpu_it"+std::to_string(it)+".vtu";
     write_solution_vtu(m,sst,u,p,fn);std::printf("NODALS_CPU_SST_M4_VTU it=%d file=%s\n",it,fn.c_str());}
   }
   // Adaptive refresh: preserve the same first-refresh/cadence and scale bounds
   // from .case. The face-sampled mean y+ is a CPU representative estimate;
   // GPU per-quadrature target statistics remain the numerical comparison target.
   if(cfg.adaptiveSampling&&refreshes<cfg.adaptiveMax&&it>=cfg.adaptiveFirst&&
     (it-cfg.adaptiveFirst)%cfg.adaptiveEvery==0){
    double yPlusSum=0;int nFaces=0;
    for(size_t f=0;f<m.faces.size();++f)if(m.faces[f].patch==3){
     yPlusSum+=mom.wall_fit().yPlus[f];++nFaces;
    }
    const double meanY=nFaces>0?yPlusSum/nFaces:0;
    if(meanY>1e-10){
     const double newSp=std::clamp(cfg.spaldingScale*cfg.adaptiveTargetSp/meanY,cfg.adaptiveScaleMin,cfg.adaptiveScaleMax);
     const double newOm=std::clamp(cfg.omegaScale*cfg.adaptiveTargetOmega/meanY,cfg.adaptiveScaleMin,cfg.adaptiveScaleMax);
     const double factor=4.4084029504620439e-4*(cfg.diameter/.004);
     cfg.spaldingScale=newSp;cfg.omegaScale=newOm;cfg.spaldingY3=newSp*factor;cfg.omegaY3=newOm*factor;
     sst.refresh_wall_rays();
     if(rank==0)std::printf("NODALS_CPU_SST_M4_ADAPTIVE it=%d meanYPlus=%.7g spScale=%.7g omegaScale=%.7g provider=CPU_FACE_REPRESENTATIVE\n",it,meanY,newSp,newOm);
    }
    ++refreshes;
   }
   if(!(std::isfinite(post)&&std::isfinite(nut)&&std::isfinite(kRel)&&std::isfinite(wRel))||fluxMax>1e-8)
     throw std::runtime_error("nonfinite SIMPLE/SST field or wall impermeability failure");
   if(!pr.converged){
    if(rank==0)std::printf("NODALS_CPU_SST_M4_LINEAR_WARNING it=%d pressurePCG_NOT_CONVERGED=1 toleranceUnchanged=YES\n",it);
    break;
   }
   if(it>3&&post<cfg.continuityAtol&&kRel<cfg.plateauTol&&wRel<cfg.plateauTol){
    if(rank==0)std::printf("NODALS_CPU_SST_M4_CONVERGED it=%d status=PASS contFull=%.9e kRel=%.9e omegaRel=%.9e\n",it,post,kRel,wRel);
    break;
   }
  }
  if(rank==0)std::printf("NODALS_CPU_SST_M4_RESULT status=%s iterations=%d fullContinuity=%.12e elapsedSeconds=%.6f momentum=M3_COUPLED pressure=AMG_PCG scalar=MASKED_P1BF3_SCOPE_CASE spiralReference=CAPTURED gpuParity=NOT_YET_VERIFIED pinnedCell=MONITORED\n",allLinearOkay?"COMPLETED":"CHECK",completed,lastFull,MPI_Wtime()-start);
  MPI_Finalize();return allLinearOkay?0:1;
 }catch(const std::exception&e){std::fprintf(stderr,"NODALS_CPU_SST_M4_FATAL rank=%d error=%s\n",rank,e.what());std::fflush(stderr);MPI_Abort(MPI_COMM_WORLD,2);return 2;}
}
