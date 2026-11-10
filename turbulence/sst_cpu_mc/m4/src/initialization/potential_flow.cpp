#include "potential_flow.hpp"
#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include "stage27_potential_host.hpp"
namespace nodals::cpu_mc::m3 {
using namespace nodals::cpu_mc::m2;
namespace {
struct Header {int nhex=0;std::size_t ntet=0,nv=0;};
struct TetVel {std::array<int,8> g{};};
struct View {Header h;std::vector<Vec3> points;std::vector<Tet> tets;std::vector<TetVel> tetVel;};
struct BoundaryRec {int cell=0,localFace=0;std::string kind;double nx=0,ny=0,nz=0,area=0;};
struct Registry {std::vector<BoundaryRec> faces;};
struct Init {std::vector<double> u0,u1,u2;};
struct Inlet {double x,y,z;};
}
std::vector<double>potential_velocity(const Mesh&mesh,std::array<double,3>inlet,int rank){
 const int nv=(int)mesh.points.size(),ne=nv+(int)mesh.faces.size();
 std::vector<double>u((std::size_t)3*ne,0.0);
 if(rank==0){
  View M;M.h.ntet=mesh.tets.size();M.h.nv=mesh.points.size();
  M.points=mesh.points;M.tets=mesh.tets;
  M.tetVel.resize(M.tets.size());
  for(std::size_t c=0;c<M.tets.size();++c){
   for(int a=0;a<4;++a){M.tetVel[c].g[a]=M.tets[c].v[a];
    M.tetVel[c].g[4+a]=nv+M.tets[c].face[a];}
  }
  Registry R;
  for(int f=0;f<(int)mesh.faces.size();++f){
   const auto&F=mesh.faces[f];if(F.patch<1||F.patch>3)continue;
   int lf=-1;for(int a=0;a<4;++a)if(mesh.tets[F.cell0].face[a]==f)lf=a;
   if(lf<0)throw std::runtime_error("potential init missing face topology");
   const Vec3 av=outward_area(mesh,F.cell0,lf);const double area=std::sqrt(dot(av,av));
   if(!(area>0))throw std::runtime_error("potential init degenerate boundary face");
   R.faces.push_back({F.cell0,lf,F.patch==1?"inlet":(F.patch==2?"outlet":"wall"),
                      av.x/area,av.y/area,av.z/area,area});
  }
  Init s;for(auto*v:{&s.u0,&s.u1,&s.u2})v->assign(ne,0.0);
  nodals::initial27::apply(M,R,Inlet{inlet[0],inlet[1],inlet[2]},s);
  for(int e=0;e<ne;++e){u[3*e]=s.u0[e];u[3*e+1]=s.u1[e];u[3*e+2]=s.u2[e];}
 }
 MPI_Allreduce(MPI_IN_PLACE,u.data(),(int)u.size(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
 return u;
}
}
