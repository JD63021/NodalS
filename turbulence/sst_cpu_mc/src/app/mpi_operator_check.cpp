// MPI operator integration: real MSH2, RCB ownership, exact BF3/P0 B and BT,
// deviatoric momentum element check, and matrix-free synthetic Schur action.
// This executable is NOT a CFD solver: no boundary data, wall laws, nonlinear
// momentum assembly, SST transport, AMG or SIMPLE are executed.
#include "../mesh/tet_mesh.hpp"
#include "../parallel/rcb.hpp"
#include "../operators/p1bf3_p0.hpp"
#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nodals::cpu_mc::m2;
static double component(Vec3 a,int d){return d==0?a.x:(d==1?a.y:a.z);}
static double max_relative(double a,double b){return std::abs(a-b)/std::max({1.0,std::abs(a),std::abs(b)});}
static bool is_finite_double(double v){return std::isfinite(v);}

int main(int argc,char**argv){
  int provided=MPI_THREAD_SINGLE;
  if(MPI_Init_thread(&argc,&argv,MPI_THREAD_SINGLE,&provided)!=MPI_SUCCESS)return 5;
  int rank=0,size=1;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
  int result=0;
  try{
    std::string meshPath,casePath;int expected=16;
    for(int i=1;i<argc;++i){
      const std::string arg=argv[i];if(i+1>=argc)throw std::runtime_error("missing argument to "+arg);
      const std::string val=argv[++i];
      if(arg=="--mesh")meshPath=val;else if(arg=="--case")casePath=val;
      else if(arg=="--expect-ranks")expected=std::stoi(val);
      else throw std::runtime_error("unexpected argument "+arg);
    }
    if(meshPath.empty()||casePath.empty()||size!=expected)throw std::runtime_error("mesh/case/rank validation failed");
    const auto m=read_msh2(meshPath);
    const auto own=partition_rcb(m,size);
    const int nc=(int)m.tets.size(),nentity=(int)(m.points.size()+m.faces.size());
    if(nentity>3000000)throw std::runtime_error("mesh too large for M2 replicated-row algebra test");
    int myCells=own.countCell[rank],minimum=0,maximum=0;
    MPI_Allreduce(&myCells,&minimum,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
    MPI_Allreduce(&myCells,&maximum,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
    const double nu=1.460735e-5;
    double momMaxAsym=0,momConstMax=0,momEnergy=0;
    // The weak deviatoric stress kernel is checked on the first 8 cells,
    // assigned to their genuine RCB owners; does not assemble a flow solve.
    for(int c=0;c<std::min(nc,8);++c)if(own.cell[c]==rank){
      const auto K=deviatoric_element(m,c,nu);
      double trial[24],constant[24];
      for(int a=0;a<8;++a)for(int d=0;d<3;++d){
        trial[3*a+d]=std::sin(.23*(a+1)+.13*d+0.17*c);
        constant[3*a+d]=(a<4)?(0.5+d):0.0;
      }
      double denom=0,enum_=0;
      for(int i=0;i<24;++i)for(int j=0;j<24;++j){
        momMaxAsym=std::max(momMaxAsym,max_relative(K[i][j],K[j][i]));
        enum_+=trial[i]*K[i][j]*trial[j];
        if(i<12){} // branch retained to leave full 24x24 action explicit
        denom+=std::abs(K[i][j]);
      }
      momEnergy+=enum_;
      for(int i=0;i<24;++i){
        double sum=0;for(int j=0;j<24;++j)sum+=K[i][j]*constant[j];
        momConstMax=std::max(momConstMax,std::abs(sum)/std::max(denom,1e-300));
      }
    }
    double mm[3]={momMaxAsym,momConstMax,momEnergy},Mmx[3]={};
    MPI_Allreduce(mm,Mmx,2,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    MPI_Allreduce(mm+2,Mmx+2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

    // A deterministic *algebraic* velocity vector, NOT a CFD field.
    std::vector<double> u((std::size_t)3*nentity);
    for(int g=0;g<3*nentity;++g)u[g]=std::sin(0.01*(g+1))+0.1*std::cos(0.003*(g+7));
    std::vector<double> p(nc),localBt((std::size_t)3*nentity,0.0);
    for(int c=0;c<nc;++c)p[c]=std::cos(0.019*(c+1))+0.2*std::sin(0.007*(c+2));
    double lhs=0, maxConstDiv=0;
    for(int c=0;c<nc;++c)if(own.cell[c]==rank){
      const auto&tet=m.tets[c];const auto b=divergence_coefficients(m,c);
      double bu=0;Vec3 sumP1{};
      for(int a=0;a<8;++a){
        const int ent=a<4?tet.v[a]:(int)m.points.size()+tet.face[a-4];
        for(int d=0;d<3;++d){
          const double x=component(b.b[a],d);
          bu+=x*u[3*ent+d];
          localBt[3*ent+d]+=x*p[c];
        }
        if(a<4)sumP1=sumP1+b.b[a];
      }
      lhs+=p[c]*bu;
      maxConstDiv=std::max(maxConstDiv,std::abs(sumP1.x)+std::abs(sumP1.y)+std::abs(sumP1.z));
    }
    std::vector<double> bt(localBt.size());
    MPI_Allreduce(localBt.data(),bt.data(),(int)bt.size(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    double globalLhs=0,rhsLocal=0;
    MPI_Allreduce(&lhs,&globalLhs,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    for(int g=0;g<nentity;++g){
      const int owner=(g<(int)m.points.size())?own.vertex[g]:own.face[g-(int)m.points.size()];
      if(owner==rank)for(int d=0;d<3;++d)rhsLocal+=u[3*g+d]*bt[3*g+d];
    }
    double rhs=0;MPI_Allreduce(&rhsLocal,&rhs,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    double globalConst=0;MPI_Allreduce(&maxConstDiv,&globalConst,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    const double adjointError=max_relative(globalLhs,rhs);

    // One application of raw S=B rAU B^T using rAU=I for a positivity check.
    // Does NOT substitute for GPU's constrained rAU, PCG-AMG or corrector.
    double localEnergy=0,localNorm=0;
    for(int c=0;c<nc;++c)if(own.cell[c]==rank){
      const auto&tet=m.tets[c];const auto b=divergence_coefficients(m,c);
      double schur=0;
      for(int a=0;a<8;++a){
        const int ent=a<4?tet.v[a]:(int)m.points.size()+tet.face[a-4];
        for(int d=0;d<3;++d)schur+=component(b.b[a],d)*bt[3*ent+d];
      }
      localEnergy+=p[c]*schur;
      localNorm+=schur*schur;
    }
    double energy=0,norm=0;MPI_Allreduce(&localEnergy,&energy,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(&localNorm,&norm,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    bool ok=adjointError<1e-9 && globalConst<1e-10 && Mmx[0]<1e-9 && Mmx[1]<1e-9 &&
            Mmx[2]>=-1e-8 && energy>=-1e-9 && norm>0 && is_finite_double(energy) && is_finite_double(norm);
    int lok=ok?0:1,fail=0;MPI_Allreduce(&lok,&fail,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
    if(rank==0){
      std::printf("NODALS_CPU_SST_M2_MESH status=PASS tets=%d vertices=%zu faces=%zu inlet=%lld outlet=%lld wall=%lld volume=%.12e\n",
        nc,m.points.size(),m.faces.size(),(long long)m.boundary[1],(long long)m.boundary[2],(long long)m.boundary[3],m.totalVolume);
      std::printf("NODALS_CPU_SST_M2_RCB status=PASS mpiRanks=%d minRankCells=%d maxRankCells=%d velocityEntities=%d pressureCells=%d\n",
        size,minimum,maximum,nentity,nc);
      std::printf("NODALS_CPU_SST_M2_MOMENTUM_ELEMENT status=%s form=DEVIATORIC_P1BF3 maxAsym=%.4e maxConstantDefect=%.4e randomEnergy=%.9e cellsTested=%d\n",
        fail?"FAIL":"PASS",Mmx[0],Mmx[1],Mmx[2],std::min(nc,8));
      std::printf("NODALS_CPU_SST_M2_B_BT status=%s relativeAdjointError=%.4e constantVelocityDivergence=%.4e\n",
        fail?"FAIL":"PASS",adjointError,globalConst);
      std::printf("NODALS_CPU_SST_M2_SCHUR status=%s rawSchurEnergy=%.9e rawSchurL2=%.9e rAU=IDENTITY_DIAGNOSTIC_ONLY\n",
        fail?"FAIL":"PASS",energy,std::sqrt(norm));
      std::printf("NODALS_CPU_SST_M2_SCOPE case=%s mesh=%s realMeshRead=YES mpiOwnership=YES momentumElement=TEST_ONLY divergenceCoupling=YES pressureSchur=RAW_DIAGNOSTIC_ONLY constrainedBF3=NOT_YET_READIED cpuNativeAMG=NOT_YET_WIRED nonlinearMomentum=NOT_IMPLEMENTED simple=NOT_IMPLEMENTED turbulence=NOT_IMPLEMENTED gpuSource=UNCHANGED\n",
        casePath.c_str(),meshPath.c_str());
      std::printf("NODALS_CPU_SST_M2_RESULT status=%s ranks=%d\n",fail?"FAIL":"PASS",size);
      std::fflush(stdout);
    }
    result=fail;
  }catch(const std::exception&e){
    std::fprintf(stderr,"NODALS_CPU_SST_M2_ERROR rank=%d reason=%s\n",rank,e.what());
    result=2;
  }
  int finalResult=0;MPI_Allreduce(&result,&finalResult,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
  MPI_Finalize();return finalResult;
}
