#pragma once
// Stage21: portable, mesh-only straight segment traversal. No CUDA, SST or wall law.
// The ray is traced through adjacent tetrahedra; reaching a boundary before the
// requested sample is an error, even if an endpoint in another disconnected fluid
// component would be locatable by a global point-location search.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nodals { namespace wall { namespace ray {
using P = std::array<double,3>;
using T = std::array<int,4>;
inline P sub(P a,P b) { return {a[0]-b[0],a[1]-b[1],a[2]-b[2]}; }
inline P add(P a,P b) { return {a[0]+b[0],a[1]+b[1],a[2]+b[2]}; }
inline P mul(double c,P a) {return {c*a[0],c*a[1],c*a[2]};}
inline double dot(P a,P b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
inline double det(P a,P b,P c){
  return a[0]*(b[1]*c[2]-b[2]*c[1])-b[0]*(a[1]*c[2]-a[2]*c[1])+c[0]*(a[1]*b[2]-a[2]*b[1]);
}
struct Hit {int cell=-1;std::array<double,4> bary{};int crossings=0;};
class TetRayTrace {
  std::vector<P> x_;
  std::vector<T> t_;
  std::vector<std::array<int,4>> adj_;
  struct Inv {P a,b,c;P o;double invdet;};
  std::vector<Inv> inv_;
  mutable std::size_t intervalRecoveries_=0;
public:
  TetRayTrace(std::vector<P> x,std::vector<T> t):x_(std::move(x)),t_(std::move(t)),adj_(t_.size()),inv_(t_.size()){
    if(x_.empty()||t_.empty())throw std::runtime_error("RAY_TRACE empty mesh");
    for(auto &a:adj_)a.fill(-1);
    using Key=std::array<int,3>;
    std::map<Key,std::pair<int,int>> faceOwners;
    static constexpr int face[4][3]={{1,2,3},{0,3,2},{0,1,3},{0,2,1}};
    for(std::size_t ci=0;ci<t_.size();++ci){
      const T&v=t_[ci];
      for(int j=0;j<4;++j)if(v[j]<0||std::size_t(v[j])>=x_.size())throw std::runtime_error("RAY_TRACE invalid vertex index");
      for(int a=0;a<4;++a)for(int b=0;b<a;++b)if(v[a]==v[b])throw std::runtime_error("RAY_TRACE degenerate connectivity");
      const P o=x_[v[0]],a=sub(x_[v[1]],o),b=sub(x_[v[2]],o),c=sub(x_[v[3]],o);
      const double D=det(a,b,c);
      const double scale=std::max({dot(a,a),dot(b,b),dot(c,c)});
      if(!(scale>0)||!std::isfinite(D)||std::abs(D)<=1e-13*scale*std::sqrt(scale))
        throw std::runtime_error("RAY_TRACE zero/degenerate tetrahedral volume");
      inv_[ci]={a,b,c,o,1.0/D};
      for(int f=0;f<4;++f){
        Key k={v[face[f][0]],v[face[f][1]],v[face[f][2]]};std::sort(k.begin(),k.end());
        auto it=faceOwners.find(k);
        if(it==faceOwners.end())faceOwners.emplace(k,std::make_pair(int(ci),f));
        else{
          const int c0=it->second.first,f0=it->second.second;
          if(adj_[c0][f0]!=-1)throw std::runtime_error("RAY_TRACE nonmanifold tetrahedral faces");
          adj_[ci][f]=c0;adj_[c0][f0]=int(ci);
        }
      }
    }
  }
  std::size_t ncell()const{return t_.size();}
  std::array<double,4> bary(int c,P p) const {
    if(c<0||std::size_t(c)>=t_.size())throw std::runtime_error("RAY_TRACE invalid cell");
    const Inv &g=inv_[c];P d=sub(p,g.o);
    const double l1=det(d,g.b,g.c)*g.invdet;
    const double l2=det(g.a,d,g.c)*g.invdet;
    const double l3=det(g.a,g.b,d)*g.invdet;
    return {1.0-l1-l2-l3,l1,l2,l3};
  }
  std::vector<Hit> trace_fast(int start,P origin,P inward,const std::vector<double>& distances,
                         int maxCrossings=4096,double tol=1e-9)const{
    if(start<0||std::size_t(start)>=t_.size()||maxCrossings<1||!std::isfinite(tol)||tol<=0||tol>1e-5)
      throw std::runtime_error("RAY_TRACE invalid ray policy");
    const double length=std::sqrt(dot(inward,inward));
    if(!(length>0)||!std::isfinite(length))throw std::runtime_error("RAY_TRACE invalid normal");
    const P d=mul(1.0/length,inward);
    const auto initial=bary(start,origin);
    if(*std::min_element(initial.begin(),initial.end())<-tol)
      throw std::runtime_error("RAY_TRACE origin outside owner tetrahedron");
    int cell=start,crossings=0;
    double position=0,previous=0;
    std::vector<Hit> hits;hits.reserve(distances.size());
    for(const double target:distances){
      if(!std::isfinite(target)||!(target>previous))
        throw std::runtime_error("RAY_TRACE sample distances must increase");
      previous=target;
      const P endpoint=add(origin,mul(target,d));
      while(true){
        // A tetrahedron is convex.  If the current point and the requested
        // endpoint are inside it, the ENTIRE remaining segment is inside.
        // Never accept an endpoint just because a crossing is <1e-13 m away:
        // in a thin tet, 1e-13 m can be many barycentric tolerances.
        const auto atEnd=bary(cell,endpoint);
        if(*std::min_element(atEnd.begin(),atEnd.end())>=-tol){
          hits.push_back({cell,atEnd,crossings});
          position=target;
          break;
        }
        const auto atStart=bary(cell,add(origin,mul(position,d)));
        if(*std::min_element(atStart.begin(),atStart.end()) < -4*tol)
          throw std::runtime_error("RAY_TRACE traversal position outside current tetrahedron");
        const auto& g=inv_[cell];
        const double cellScale=std::sqrt(std::max({dot(g.a,g.a),dot(g.b,g.b),dot(g.c,g.c)}));
        const double backwardsTol=tol*cellScale;
        const double remaining=target-position;
        if(!(remaining>0.0))throw std::runtime_error("RAY_TRACE invalid remaining distance");
        double step=std::numeric_limits<double>::infinity();
        // Only faces through which the TARGET lies outside are eligible.
        // This removes spurious zero-length crossings on nearly tangent faces.
        for(int j=0;j<4;++j){
          if(atEnd[j]>=-tol)continue;
          const double decay=atStart[j]-atEnd[j];
          if(!(decay>0.0))continue;
          const double candidate=remaining*atStart[j]/decay;
          if(candidate>=-backwardsTol && candidate<step)step=candidate;
        }
        if(!std::isfinite(step))
          throw std::runtime_error("RAY_TRACE cannot identify an exiting face");
        if(step < -backwardsTol)
          throw std::runtime_error("RAY_TRACE crossing behind current position");
        const double where=position+std::max(0.0,step);
        // The endpoint was *proved* outside the current tet, hence even a
        // crossing very close to the target must be followed, not rounded off.
        if(where>=target)
          throw std::runtime_error("RAY_TRACE crossing indistinguishable from sample");
        if(++crossings>maxCrossings)
          throw std::runtime_error("RAY_TRACE exceeded max tetra crossings");
        const P crossingPoint=add(origin,mul(where,d));
        const double remainingAfter=target-where;
        // Small forward probe: avoid jumping across a narrow tetra or exiting
        // a narrow channel. Reduce it again if no connected cell contains it.
        const double nominalProbe=std::min(0.5*remainingAfter,1e-9*cellScale);
        int next=-1;
        double acceptedPosition=where;
        for(int retry=0;retry<12 && next<0;++retry){
          const double probe=std::ldexp(nominalProbe,-retry);
          if(!(probe>0.0)||!(where+probe>where))break;
          const P after=add(origin,mul(where+probe,d));
          // Fast path: an ordinary face crossing has exactly one adjacent
          // tetrahedron, so it must NOT allocate a visited array proportional
          // to the entire mesh.  This is the common case for 2M+ samples.
          const auto faceB=bary(cell,crossingPoint);
          int best=-1;double bestMargin=-std::numeric_limits<double>::infinity();
          for(int j=0;j<4;++j){
            if(std::abs(faceB[j])>8*tol)continue;
            const int neigh=adj_[(std::size_t)cell][j];
            if(neigh<0)continue;
            const auto test=bary(neigh,after);
            const double margin=*std::min_element(test.begin(),test.end());
            if(margin>=-tol && margin>bestMargin){best=neigh;bestMargin=margin;}
          }
          // Only an edge/vertex ambiguity requires exploring the connected
          // face fan.  Bounded local vectors avoid O(mesh_cells) scratch.
          if(best<0){
            std::vector<int> queue(1,cell);
            for(std::size_t qi=0;qi<queue.size() && qi<256;++qi){
              const int c=queue[qi];
              const auto through=bary(c,crossingPoint);
              if(*std::min_element(through.begin(),through.end()) < -8*tol)continue;
              if(c!=cell){
                const auto test=bary(c,after);
                const double margin=*std::min_element(test.begin(),test.end());
                if(margin>=-tol && margin>bestMargin){best=c;bestMargin=margin;}
              }
              for(int j=0;j<4;++j){
                if(std::abs(through[j])>8*tol)continue;
                const int neigh=adj_[(std::size_t)c][j];
                if(neigh>=0 && std::find(queue.begin(),queue.end(),neigh)==queue.end())
                  queue.push_back(neigh);
              }
            }
          }
          if(best>=0){next=best;acceptedPosition=where+probe;}
        }
        if(next<0)
          throw std::runtime_error("RAY_TRACE wall-normal ray exited fluid or crossed ambiguous junction");
        if(!(acceptedPosition>position) || next==cell)
          throw std::runtime_error("RAY_TRACE unable to make forward topological progress");
        cell=next;position=acceptedPosition;
      }
    }
    return hits;
  }


  // Secondary geometric traversal, used if the fast probe-based walk loses
  // a skinny/edge-adjacent tetrahedron.  For a straight ray p(t)=o+t*d,
  // each tetrahedron occupies one interval [t_enter,t_exit], because its
  // barycentric coordinates are affine in t.  Consequently this method has
  // no finite-difference point probe or min-step heuristic.
  //
  // IMPORTANT: A globally located endpoint is never sufficient.  We only
  // enter cells connected through faces passing through the crossing point;
  // no endpoint search can bridge an exterior boundary or disconnected body.
  std::vector<Hit> trace_intervals(int start,P origin,P inward,
                       const std::vector<double>& distances,
                       int maxCrossings=4096,double tol=1e-9)const {
    if(start<0||std::size_t(start)>=t_.size()||maxCrossings<1||
       !std::isfinite(tol)||tol<=0||tol>1e-5)
      throw std::runtime_error("RAY_TRACE_INTERVAL invalid policy");
    const double len=std::sqrt(dot(inward,inward));
    if(!(len>0)||!std::isfinite(len))
      throw std::runtime_error("RAY_TRACE_INTERVAL invalid ray direction");
    const P d=mul(1.0/len,inward);
    const auto first=bary(start,origin);
    if(*std::min_element(first.begin(),first.end()) < -tol)
      throw std::runtime_error("RAY_TRACE_INTERVAL origin outside owner");
    struct Interval {double lo,hi;};
    auto segment=[&](int c)->Interval{
      const Inv&g=inv_[(std::size_t)c];
      const auto lam=bary(c,origin);
      const double r1=det(d,g.b,g.c)*g.invdet;
      const double r2=det(g.a,d,g.c)*g.invdet;
      const double r3=det(g.a,g.b,d)*g.invdet;
      const std::array<double,4> slope={-r1-r2-r3,r1,r2,r3};
      double lo=-std::numeric_limits<double>::infinity();
      double hi=std::numeric_limits<double>::infinity();
      for(int j=0;j<4;++j){
        // Allow only the existing barycentric roundoff tolerance, never a
        // geometrically significant gap between the two ray intervals.
        if(slope[(std::size_t)j]>0)
          lo=std::max(lo,(-tol-lam[(std::size_t)j])/slope[(std::size_t)j]);
        else if(slope[(std::size_t)j]<0)
          hi=std::min(hi,(-tol-lam[(std::size_t)j])/slope[(std::size_t)j]);
        else if(lam[(std::size_t)j]<-tol)
          return {std::numeric_limits<double>::infinity(),
                  -std::numeric_limits<double>::infinity()};
      }
      return {lo,hi};
    };
    int cell=start, crossingCount=0;
    double progress=0.0,previous=0.0;
    std::vector<Hit> results;
    results.reserve(distances.size());
    for(double target:distances){
      if(!std::isfinite(target)||!(target>previous))
        throw std::runtime_error("RAY_TRACE_INTERVAL non-increasing sample distances");
      previous=target;
      const P endpoint=add(origin,mul(target,d));
      for(;;){
        const auto b=bary(cell,endpoint);
        if(*std::min_element(b.begin(),b.end())>=-tol){
          results.push_back({cell,b,crossingCount});
          progress=target;
          break;
        }
        const Interval current=segment(cell);
        const double t=current.hi;
        const Inv&g=inv_[(std::size_t)cell];
        const double localScale=std::sqrt(std::max({dot(g.a,g.a),dot(g.b,g.b),dot(g.c,g.c)}));
        const double coordScale=std::max({std::abs(origin[0]),std::abs(origin[1]),std::abs(origin[2]),1e-3});
        const double eps=std::max(128.0*std::numeric_limits<double>::epsilon()*coordScale,
                                  4.0*tol*localScale);
        if(!std::isfinite(t)||t<progress-eps||t>=target+eps)
          throw std::runtime_error("RAY_TRACE_INTERVAL invalid connected-cell exit interval");
        if(++crossingCount>maxCrossings)
          throw std::runtime_error("RAY_TRACE_INTERVAL exceeded max tetra crossings");
        const P crossP=add(origin,mul(t,d));
        std::vector<int> queue{cell};
        int best=-1;
        double bestExit=t;
        double bestMargin=-std::numeric_limits<double>::infinity();
        // Only candidates in the face-connected fan at the EXACT ray
        // intersection are considered.  This bounded list is local to one
        // crossing, rather than O(all tetrahedra) allocation per sample.
        for(std::size_t qi=0;qi<queue.size() && qi<256;++qi){
          const int c=queue[qi];
          const auto through=bary(c,crossP);
          const double minB=*std::min_element(through.begin(),through.end());
          if(minB< -8*tol)continue;
          if(c!=cell){
            const Interval candidate=segment(c);
            const double cscale=std::sqrt(std::max({dot(inv_[(std::size_t)c].a,inv_[(std::size_t)c].a),
                                                  dot(inv_[(std::size_t)c].b,inv_[(std::size_t)c].b),
                                                  dot(inv_[(std::size_t)c].c,inv_[(std::size_t)c].c)}));
            const double ceps=std::max(eps,4.0*tol*cscale);
            if(candidate.lo<=t+ceps && candidate.hi>t+ceps*0.01){
              const double far=std::min(target,candidate.hi);
              const double witness=t+0.5*std::min(target-t,candidate.hi-t);
              const auto bw=bary(c,add(origin,mul(witness,d)));
              const double margin=*std::min_element(bw.begin(),bw.end());
              // A real positive-length overlap is required; touching an
              // isolated vertex or merely locating the final endpoint does
              // not establish a valid route through the fluid.
              if(margin>=-tol && (far>bestExit+ceps*0.01 ||
                   (std::abs(far-bestExit)<=ceps*0.01 && margin>bestMargin))){
                best=c;bestExit=far;bestMargin=margin;
              }
            }
          }
          for(int j=0;j<4;++j){
            if(std::abs(through[(std::size_t)j])>8*tol)continue;
            const int neigh=adj_[(std::size_t)c][(std::size_t)j];
            if(neigh>=0 && std::find(queue.begin(),queue.end(),neigh)==queue.end() &&
               queue.size()<256)queue.push_back(neigh);
          }
        }
        if(best<0){
          throw std::runtime_error("RAY_TRACE_INTERVAL no forward face-connected tetra; cell="+
               std::to_string(cell)+" sampleY="+std::to_string(target)+
               " crossingY="+std::to_string(t)+" fan="+std::to_string(queue.size()));
        }
        if(best==cell||bestExit<=t)
          throw std::runtime_error("RAY_TRACE_INTERVAL no forward progress");
        // progress records the certified boundary crossing (not a speculative
        // nudge that might skip a sliver cell).
        progress=std::max(progress,t);
        cell=best;
      }
    }
    return results;
  }

  std::size_t interval_recoveries()const{return intervalRecoveries_;}

  std::vector<Hit> trace(int start,P origin,P inward,
                         const std::vector<double>&distances,
                         int maxCrossings=4096,double tol=1e-9)const{
    try {return trace_fast(start,origin,inward,distances,maxCrossings,tol);}
    catch(const std::runtime_error&fastError){
      // Fall back on a distinct topologically constrained ray algorithm.
      // Both paths reject true fluid exits and insufficient ray clearance.
      const std::string cause=fastError.what();
      try{
        auto hits=trace_intervals(start,origin,inward,distances,maxCrossings,tol);
        ++intervalRecoveries_;
        return hits;
      }catch(const std::runtime_error&intervalError){
        throw std::runtime_error("RAY_TRACE fast="+cause+" interval="+intervalError.what());
      }
    }
  }

};
}}} // namespace nodals::wall::ray
