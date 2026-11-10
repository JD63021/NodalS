// Message 1: multicore/MPI *foundation probe*, not a flow solver.
// Mesh and .case are passed through unchanged. The native CPU SST momentum,
// pressure, scalar transport assembly and SIMPLE driver are subsequent tasks.
#include "../bridge/portable_probe.hpp"
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>

int main(int argc,char**argv) {
  int provided=MPI_THREAD_SINGLE;
  if(MPI_Init_thread(&argc,&argv,MPI_THREAD_SINGLE,&provided)!=MPI_SUCCESS)return 3;
  int rank=0,size=1;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  std::string casePath,meshPath;
  double nu=1.460735e-5, re=0.0;
  int expected=16;
  int status=0;
  try {
    for(int i=1;i<argc;++i) {
      const std::string a(argv[i]);
      if(i+1>=argc)throw std::runtime_error("missing value for "+a);
      const std::string v(argv[++i]);
      if(a=="--case")casePath=v;
      else if(a=="--mesh")meshPath=v;
      else if(a=="--nu")nu=std::stod(v);
      else if(a=="--re")re=std::stod(v);
      else if(a=="--expect-ranks")expected=std::stoi(v);
      else throw std::runtime_error("unexpected argument "+a);
    }
    if(casePath.empty()||meshPath.empty()||expected<1 || !(nu>0.0))
      throw std::runtime_error("required --case --mesh --nu --expect-ranks");
    if(size!=expected)throw std::runtime_error("MPI size differs from --expect-ranks");
    const auto r=nodals::cpu_mc::run_portable_probe(nu);
    status=r.passed?0:1;
    int fail=0;
    MPI_Allreduce(&status,&fail,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
    if(rank==0) {
      std::printf("NODALS_CPU_SST_M1_PROBE status=%s ranks=%d mpiThread=%d "
                  "nu=%.12g re=%.12g portableNut=%.12g spaldingUTau=%.12g "
                  "omegaWallTarget=%.12g scalarDiffusion=%.12g\n",
                  fail?"FAIL":"PASS",size,provided,nu,re,r.nut,r.wall_utau,
                  r.omega_target,r.transport_diffusion);
      std::printf("NODALS_CPU_SST_M1_SCOPE mesh=%s case=%s meshRead=NO "
                  "assembly=NOT_IMPLEMENTED simple=NOT_IMPLEMENTED "
                  "solver=NOT_IMPLEMENTED originalGpu=UNCHANGED\n",
                  meshPath.c_str(),casePath.c_str());
      std::fflush(stdout);
    }
    status=fail;
  } catch (const std::exception& e) {
    std::fprintf(stderr,"NODALS_CPU_SST_M1_ERROR rank=%d reason=%s\n",rank,e.what());
    status=2;
  }
  int anyFailure=0;
  MPI_Allreduce(&status,&anyFailure,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
  MPI_Finalize();
  return anyFailure;
}
