#include "wall_distance.hpp"
#include "wall_distance_host.hpp" // unchanged NodalS GPU geometric wall-distance BVH
#include <mpi.h>
#include <cstdio>
#include <cmath>
#include <stdexcept>
namespace nodals::cpu_mc::m4 {
std::vector<double> geometric_wall_distance(const m2::Mesh&m,int rank){
 using namespace nodals::geometry;
 std::vector<Triangle> walls; walls.reserve((size_t)m.boundary[3]);
 std::vector<char> isWall(m.points.size(),0);
 for(const auto&f:m.faces)if(f.patch==3){
  const auto&a=m.points[f.v[0]],&b=m.points[f.v[1]],&c=m.points[f.v[2]];
  walls.push_back(make_triangle({a.x,a.y,a.z},{b.x,b.y,b.z},{c.x,c.y,c.z}));
  for(int v:f.v)isWall[(size_t)v]=1;
 }
 TriangleTree tree(std::move(walls));
 std::vector<double> dist(m.points.size(),0.0);
 double maxd=0,mean=0;size_t zero=0;
 for(size_t i=0;i<m.points.size();++i){
  const auto&p=m.points[i];
  dist[i]=isWall[i]?0.0:tree.distance({p.x,p.y,p.z});
  if(!std::isfinite(dist[i]) || dist[i]<0)throw std::runtime_error("non-finite exact geometric wall distance");
  maxd=std::max(maxd,dist[i]);mean+=dist[i];zero+=isWall[i]!=0;
 }
 if(rank==0)std::printf("NODALS_CPU_SST_M4_DISTANCE vertices=%zu wallVertices=%zu wallTriangles=%lld max=%.12e mean=%.12e provider=UNCHANGED_GPU_BVH\n",m.points.size(),zero,(long long)m.boundary[3],maxd,mean/dist.size());
 return dist;
}
}
