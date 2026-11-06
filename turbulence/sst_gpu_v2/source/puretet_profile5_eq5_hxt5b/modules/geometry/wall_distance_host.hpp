#pragma once
// Stage20: static nearest-solid-triangle wall distance, precomputed at mesh vertices.
// The GPU evaluates its continuous P1 interpolation at tetrahedral quadrature locations.
// Nothing here depends on SST, Spalding or a particular flow direction.
#include "../mesh/tet_patch_registry.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
namespace nodals { namespace geometry {
using P3=std::array<double,3>;
inline P3 sub(const P3&a,const P3&b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
inline P3 add(const P3&a,const P3&b){return {a[0]+b[0],a[1]+b[1],a[2]+b[2]};}
inline P3 mul(const P3&a,double s){return {a[0]*s,a[1]*s,a[2]*s};}
inline double dot(const P3&a,const P3&b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
inline P3 cross(const P3&a,const P3&b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
inline double dist_segment2(const P3&p,const P3&a,const P3&b){
  P3 ab=sub(b,a);const double z=dot(ab,ab);
  const double t=z>0?std::max(0.,std::min(1.,dot(sub(p,a),ab)/z)):0.;
  const P3 d=sub(p,add(a,mul(ab,t)));return dot(d,d);
}
struct Triangle{P3 a,b,c,lo,hi,center;};
inline Triangle make_triangle(P3 a,P3 b,P3 c){
  const auto ab=sub(b,a),ac=sub(c,a),n=cross(ab,ac);
  if(!(dot(n,n)>0)||!std::isfinite(dot(n,n)))throw std::runtime_error("STAGE20 degenerate physical wall triangle");
  Triangle t{};t.a=a;t.b=b;t.c=c;
  for(int d=0;d<3;++d){t.lo[d]=std::min({a[d],b[d],c[d]});t.hi[d]=std::max({a[d],b[d],c[d]});t.center[d]=(a[d]+b[d]+c[d])/3.;}
  return t;
}
inline double dist_triangle2(const P3&p,const Triangle&t){
  const P3 ab=sub(t.b,t.a),ac=sub(t.c,t.a),ap=sub(p,t.a),n=cross(ab,ac);
  const double nn=dot(n,n),npa=dot(n,ap);
  const P3 proj=sub(p,mul(n,npa/nn)),w=sub(proj,t.a);
  const double aa=dot(ab,ab),bb=dot(ab,ac),cc=dot(ac,ac),det=aa*cc-bb*bb;
  if(det>0){const double u=(dot(w,ab)*cc-dot(w,ac)*bb)/det,v=(dot(w,ac)*aa-dot(w,ab)*bb)/det;
    if(u>=-1e-12&&v>=-1e-12&&u+v<=1+1e-12)return (npa*npa)/nn;}
  return std::min({dist_segment2(p,t.a,t.b),dist_segment2(p,t.b,t.c),dist_segment2(p,t.c,t.a)});
}
inline double box_dist2(const P3&p,const P3&lo,const P3&hi){
  double v=0;for(int d=0;d<3;++d){const double e=std::max({lo[d]-p[d],0.,p[d]-hi[d]});v+=e*e;}return v;
}
class TriangleTree{
  struct Node{P3 lo{},hi{};int left=-1,right=-1,begin=0,end=0;};
  std::vector<Triangle> tri_;
  std::vector<int> order_;
  std::vector<Node> nodes_;
  int build(int begin,int end){
    const int id=(int)nodes_.size();nodes_.push_back({});
    P3 lo{1e300,1e300,1e300},hi{-1e300,-1e300,-1e300};
    for(int i=begin;i<end;++i)for(int k=0;k<3;++k){auto&t=tri_[(size_t)order_[(size_t)i]];lo[k]=std::min(lo[k],t.lo[k]);hi[k]=std::max(hi[k],t.hi[k]);}
    nodes_[(size_t)id].lo=lo;nodes_[(size_t)id].hi=hi;
    nodes_[(size_t)id].begin=begin;nodes_[(size_t)id].end=end;
    if(end-begin<=8)return id;
    int axis=0;for(int k=1;k<3;++k)if(hi[k]-lo[k]>hi[axis]-lo[axis])axis=k;
    int mid=(begin+end)/2;
    std::nth_element(order_.begin()+begin,order_.begin()+mid,order_.begin()+end,
      [&](int a,int b){return tri_[(size_t)a].center[axis]<tri_[(size_t)b].center[axis];});
    int l=build(begin,mid),r=build(mid,end);
    nodes_[(size_t)id].left=l;nodes_[(size_t)id].right=r;
    return id;
  }
  void visit(int i,const P3&p,double&best)const{
    const Node &n=nodes_[(size_t)i];if(box_dist2(p,n.lo,n.hi)>=best)return;
    if(n.left<0){for(int j=n.begin;j<n.end;++j)best=std::min(best,dist_triangle2(p,tri_[(size_t)order_[(size_t)j]]));return;}
    const auto &a=nodes_[(size_t)n.left],&b=nodes_[(size_t)n.right];
    const double da=box_dist2(p,a.lo,a.hi),db=box_dist2(p,b.lo,b.hi);
    if(da<=db){if(da<best)visit(n.left,p,best);if(db<best)visit(n.right,p,best);}
    else{if(db<best)visit(n.right,p,best);if(da<best)visit(n.left,p,best);}
  }
public:
  explicit TriangleTree(std::vector<Triangle> triangles):tri_(std::move(triangles)){
    if(tri_.empty())throw std::runtime_error("STAGE20 no wall triangles");
    order_.resize(tri_.size());std::iota(order_.begin(),order_.end(),0);nodes_.reserve(2*tri_.size());build(0,(int)tri_.size());
  }
  double distance(const P3&p)const{double q=std::numeric_limits<double>::infinity();visit(0,p,q);return std::sqrt(q);}
};
template<class Mesh> std::vector<double> vertex_wall_distances(const Mesh&M,const nodals::mesh::Registry&R){
  if(M.h.nhex||!M.h.ntet)throw std::runtime_error("STAGE20 geometric distance currently requires pure tetrahedral HXT1");
  std::vector<Triangle> faces;faces.reserve(R.wall);
  std::vector<bool> is_wall(M.points.size(),false);
  for(const auto&f:R.faces)if(f.kind=="wall"){
    const auto&t=M.tets[(size_t)f.cell];P3 v[3];
    for(int j=0;j<3;++j){int idx=t.v[nodals::mesh::FACE_LIDS[f.localFace][j]];
      const auto&p=M.points[(size_t)idx];v[j]={p.x,p.y,p.z};is_wall[(size_t)idx]=true;}
    faces.push_back(make_triangle(v[0],v[1],v[2]));
  }
  TriangleTree tree(std::move(faces));
  std::vector<double> result(M.points.size());double mx=0,sum=0;size_t zeros=0;
  for(size_t i=0;i<M.points.size();++i){const auto&p=M.points[i];
    // Explicitly construct the query point: CUDA's C++ front end cannot
    // reliably parse a braced-init-list in this conditional call site.
    P3 query{};
    query[0]=static_cast<double>(p.x);
    query[1]=static_cast<double>(p.y);
    query[2]=static_cast<double>(p.z);
    const double d=is_wall[i] ? 0.0 : tree.distance(query);
    if(!(d>=0)||!std::isfinite(d))throw std::runtime_error("STAGE20 non-finite geometric wall distance");
    result[i]=d;mx=std::max(mx,d);sum+=d;if(d==0)++zeros;
  }
  if(zeros==M.points.size())throw std::runtime_error("STAGE20 all mesh vertices lie on walls");
  std::printf("SST_WALL_DISTANCE_FIELD provider=GEOMETRIC_TRIANGLE_BVH interpolation=P1_TET vertices=%zu wallVertices=%zu wallFaces=%zu mean=%.12e max=%.12e status=PASS\n",M.points.size(),zeros,R.wall,sum/result.size(),mx);
  return result;
}
}} // nodals::geometry
