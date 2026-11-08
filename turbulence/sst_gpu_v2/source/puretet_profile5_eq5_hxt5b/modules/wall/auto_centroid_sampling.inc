// NODALS_AUTO_CENTROID_SAMPLING_V1
// Pure host-side geometry; included INSIDE namespace hxt5b after tet_wall_ray_cache.
// Preserve the existing Spalding/omega operators and the old fixed-scale path.
struct AutoFaceGeometry {
  double centroidHeight=0.;
  ::nodals::geometry::P3 faceCenter{};
  std::array<int,3> node{};
  bool sharpCorner=false;
  bool nearSharpCorner=false;
  // Three edge-neighbours of one physical triangular wall face.
  std::array<int,3> wallNeighbours{{-1,-1,-1}};
};
inline std::vector<AutoFaceGeometry> auto_face_geometry(
    const HostMesh&M,const std::vector<TetFacePlan>&faces,double cornerAngleDegrees){
  if(!(cornerAngleDegrees>0. && cornerAngleDegrees<180.))
    throw std::runtime_error("NODALS_AUTO invalid corner angle");
  std::vector<AutoFaceGeometry> g(faces.size());
  std::map<std::array<int,2>,std::vector<std::size_t>> edges;
  for(std::size_t fi=0;fi<faces.size();++fi){
    const auto&f=faces[fi];const auto&t=M.tets[(std::size_t)f.cell];
    if(!(f.area>0.))throw std::runtime_error("NODALS_AUTO invalid wall face area");
    ::nodals::geometry::P3 fc{0.,0.,0.},cc{0.,0.,0.};
    for(int j=0;j<3;++j){
      const int v=t.v[hxt2::TET_FACE[(int)f.lf][j]];
      g[fi].node[(std::size_t)j]=v;
      const auto&p=M.points[(std::size_t)v];
      fc[0]+=p.x/3.;fc[1]+=p.y/3.;fc[2]+=p.z/3.;
    }
    for(int j=0;j<4;++j){const auto&p=M.points[(std::size_t)t.v[j]];
      cc[0]+=p.x/4.;cc[1]+=p.y/4.;cc[2]+=p.z/4.;}
    // 3V/(4A), not 3V/A (which is the opposite-vertex altitude).
    const double h=-(cc[0]-fc[0])*f.n[0]-(cc[1]-fc[1])*f.n[1]-(cc[2]-fc[2])*f.n[2];
    if(!(std::isfinite(h)&&h>0.))
      throw std::runtime_error("NODALS_AUTO invalid centroid normal distance face="+std::to_string(fi));
    g[fi].centroidHeight=h;g[fi].faceCenter=fc;
    for(int j=0;j<3;++j){auto e=std::array<int,2>{g[fi].node[(std::size_t)j],g[fi].node[(std::size_t)((j+1)%3)]};
      if(e[0]>e[1])std::swap(e[0],e[1]);
      edges[e].push_back(fi);
    }
  }
  const double threshold=std::cos(cornerAngleDegrees*3.14159265358979323846/180.);
  for(const auto&entry:edges){const auto&adj=entry.second;
    if(adj.size()>2)throw std::runtime_error("NODALS_AUTO nonmanifold wall edge");
    if(adj.size()!=2)continue;
    for(int t=0;t<2;++t){
      const std::size_t mine=adj[(std::size_t)t];
      const int other=(int)adj[(std::size_t)(1-t)];
      auto& a=g[mine].wallNeighbours;
      bool set=false;
      for(int& slot:a)if(slot<0){slot=other;set=true;break;}
      if(!set)throw std::runtime_error("NODALS_AUTO wall face has >3 edge neighbours");
    }
    const auto&a=faces[adj[0]],&b=faces[adj[1]];
    const double d=a.n[0]*b.n[0]+a.n[1]*b.n[1]+a.n[2]*b.n[2];
    if(d<threshold){g[adj[0]].sharpCorner=true;g[adj[1]].sharpCorner=true;}
  }
  // Extend the *ray fallback eligibility* one surface edge beyond an actual
  // sharp edge.  This handles tetra face centroids lying just off that edge.
  for(std::size_t i=0;i<g.size();++i)if(g[i].sharpCorner){
    g[i].nearSharpCorner=true;
    for(int other:g[i].wallNeighbours)if(other>=0)
      g[(std::size_t)other].nearSharpCorner=true;
  }
  return g;
}
// NODALS_AUTO_CENTROID_CORNER_NEIGHBORHOOD_V11
// A close competing wall is the SAME CORNER only if the two wall faces are
// connected by <=4 mesh-surface edges AND the connecting path crosses a true
// >=corner_angle_degrees crease.  Mere Euclidean nearness or membership of
// the same wall patch is NOT sufficient: opposite channel walls still fail.
inline bool auto_same_sharp_corner(const std::vector<AutoFaceGeometry>&g,
                                   std::size_t origin,std::size_t other){
  if(origin>=g.size() || other>=g.size() || origin==other)return false;
  struct State{int face;bool crossedSharp;};
  std::vector<State> frontier{{(int)origin,g[origin].sharpCorner}};
  for(int depth=1;depth<=4;++depth){ // NODALS_AUTO_Q12_CORNER_V12 : four connected surface-edge hops

    std::vector<State> next;next.reserve(frontier.size()*3);
    for(const auto&state:frontier)
      for(int nb:g[(std::size_t)state.face].wallNeighbours)if(nb>=0){
        const bool sharp=state.crossedSharp || g[(std::size_t)nb].sharpCorner;
        if((std::size_t)nb==other && sharp)return true;
        if(next.size()<96)next.push_back({nb,sharp});
      }
    frontier.swap(next);
  }
  return false;
}
inline bool auto_share_corner_vertex(const AutoFaceGeometry&a,const AutoFaceGeometry&b){
  for(int i:a.node)for(int j:b.node)if(i==j)return true;
  return false;
}
// NODALS_AUTO_Q12_CORNER_V12
// Match the *actual* FEM wall quadrature and the *actual* inward sample rays.
// A wall face-centre alone is NOT a sufficient geometric preflight.
inline bool auto_same_local_surface(const std::vector<AutoFaceGeometry>&g,
                                    std::size_t origin,std::size_t other){
  if(origin>=g.size() || other>=g.size())return false;
  if(origin==other)return true;
  std::vector<int> frontier{(int)origin};
  std::vector<int> seen{(int)origin};
  for(int depth=1;depth<=4;++depth){
    std::vector<int> next;
    for(int f:frontier)for(int nb:g[(std::size_t)f].wallNeighbours)if(nb>=0){
      if((std::size_t)nb==other)return true;
      if(std::find(seen.begin(),seen.end(),nb)==seen.end()){
        seen.push_back(nb);next.push_back(nb);
      }
    }
    frontier.swap(next);
    if(frontier.empty())break;
  }
  return false;
}

inline std::vector<int> auto_preflight_samples(
    const HostMesh&M,const std::vector<TetFacePlan>&faces,
    const std::vector<AutoFaceGeometry>&geom,
    const std::vector<double>&wanted,
    const std::vector<double>&wallUtau,double nu,double targetYplus,
    const char*kind,bool refreshed){
  if(wanted.size()!=faces.size() || geom.size()!=faces.size() ||
     (!wallUtau.empty() && wallUtau.size()!=faces.size()))
    throw std::runtime_error("NODALS_AUTO sample/face count mismatch");
  std::vector<::nodals::geometry::Triangle> triangles;
  triangles.reserve(faces.size());
  for(std::size_t fi=0;fi<faces.size();++fi){
    ::nodals::geometry::P3 p[3];
    for(int j=0;j<3;++j){
      const auto &v=M.points[(std::size_t)geom[fi].node[(std::size_t)j]];
      p[j]={v.x,v.y,v.z};
    }
    triangles.push_back(::nodals::geometry::make_triangle(p[0],p[1],p[2]));
  }
  ::nodals::geometry::TriangleTree tree(std::move(triangles));
  // Every Q12 sample has an independent flag: do not mark an entire wall
  // face merely because one of its 12 quadrature rays needs corner fallback.
  std::vector<int> fallback(faces.size()*12u,0);
  // NODALS_AUTO_THIN_CENTROID_V13 : permitted thin-passage fallback, not a new wall law.
  const char* thinRaw=std::getenv("NODALS_AUTO_THIN_POLICY");
  const std::string thinPolicy=(thinRaw&&*thinRaw)?std::string(thinRaw):"error";
  if(thinPolicy!="error" && thinPolicy!="centroid_fallback_warn")
    throw std::runtime_error("NODALS_AUTO_THIN_POLICY must be error or centroid_fallback_warn");
  const bool thinFallback=(thinPolicy=="centroid_fallback_warn");
  std::size_t fallbackQuads=0,fallbackFaces=0,nearSmooth=0,thinQuads=0,thinFaces=0;
  double hmin=1e300,hmax=0.,hsum=0.;
  constexpr double fraction[8]={.125,.25,.375,.5,.625,.75,.875,1.};
  for(std::size_t fi=0;fi<faces.size();++fi){
    const auto &f=faces[fi];const double h=wanted[fi];
    if(!(std::isfinite(h)&&h>0.))
      throw std::runtime_error("NODALS_AUTO invalid target height face="+std::to_string(fi));
    const auto &t=M.tets[(std::size_t)f.cell];
    const int a0=hxt2::TET_FACE[(int)f.lf][0];
    const int a1=hxt2::TET_FACE[(int)f.lf][1];
    const int a2=hxt2::TET_FACE[(int)f.lf][2];
    const auto &v0=M.points[(std::size_t)t.v[a0]];
    const auto &v1=M.points[(std::size_t)t.v[a1]];
    const auto &v2=M.points[(std::size_t)t.v[a2]];
    int faceFallbacks=0,faceThinFallbacks=0;
    for(int q=0;q<12;++q){
      const ::nodals::geometry::P3 origin{
        HL0[q]*v0.x+HL1[q]*v1.x+HL2[q]*v2.x,
        HL0[q]*v0.y+HL1[q]*v1.y+HL2[q]*v2.y,
        HL0[q]*v0.z+HL1[q]*v1.z+HL2[q]*v2.z};
      bool smoothSeen=false;
      for(double frac:fraction){
        const double y=frac*h;
        const ::nodals::geometry::P3 x{
          origin[0]-y*f.n[0],origin[1]-y*f.n[1],origin[2]-y*f.n[2]};
        const auto near=tree.nearest(x);
        if(near.first>=.98*y || near.second==(int)fi)continue;
        const int other=near.second;
        if(other<0 || (std::size_t)other>=geom.size())
          throw std::runtime_error("NODALS_AUTO nearest-wall query failed");
        if(auto_same_sharp_corner(geom,fi,(std::size_t)other)){
          fallback[fi*12u+(std::size_t)q]=1;
          ++fallbackQuads;++faceFallbacks;
          if(fallbackQuads<=25){
            const double u=wallUtau.empty()?0.:wallUtau[fi];
            const double achieved=(u>0. && nu>0.)?geom[fi].centroidHeight*u/nu:-1.;
            std::printf("NODALS_AUTO_Q12_CORNER_FALLBACK field=%s face=%zu q=%d "
              "nearestFace=%d rayFraction=%.5g requestedY=%.9g centroidY=%.9g "
              "targetYplus=%.9g achievedYplus=%.9g status=FALLBACK\n",
              kind,fi,q,other,frac,h,geom[fi].centroidHeight,
              refreshed?targetYplus:-1.,achieved);
          }
          break;
        }
        // Closely connected smooth surface triangles can be a nearer
        // mathematical triangle without constituting another solid wall.
        // Unlike a crease, they do NOT invoke the centroid fallback.
        if(auto_same_local_surface(geom,fi,(std::size_t)other)){
          if(!smoothSeen){++nearSmooth;smoothSeen=true;}
          continue;
        }
        if(thinFallback){
          // Explicitly evaluate at the owner tetrahedron CENTROID. Never
          // pretend the target ray or requested y+ was attained.
          fallback[fi*12u+(std::size_t)q]=2; // code 1 = corner, 2 = thin
          ++thinQuads;++faceThinFallbacks;
          if(thinQuads<=25){
            const double u=wallUtau.empty()?0.:wallUtau[fi];
            const double yC=geom[fi].centroidHeight;
            const double achieved=(u>0.&&nu>0.)?yC*u/nu:-1.;
            std::printf("NODALS_AUTO_THIN_FALLBACK field=%s face=%zu q=%d "
              "otherWallFace=%d requestedY=%.9g obstructionY=%.9g "
              "nearestWallDistance=%.9g centroidY=%.9g "
              "targetYplus=%.9g achievedYplus=%.9g "
              "status=FALLBACK_WARNING\n",
              kind,fi,q,other,h,y,near.first,yC,
              refreshed?targetYplus:-1.,achieved);
          }
          break;
        }
        throw std::runtime_error(
          "NODALS_AUTO_THIN_CHANNEL field="+std::string(kind)+
          " face="+std::to_string(fi)+" q="+std::to_string(q)+
          " otherWallFace="+std::to_string(other)+
          " requestedY="+std::to_string(h)+
          " violationAtDistance="+std::to_string(y)+
          " nearestSolidDistance="+std::to_string(near.first)+
          " action=ERROR (separate wall region; set thin_channel_policy=centroid_fallback_warn to permit fallback)");
      }
    }
    fallbackFaces+=(faceFallbacks>0);
    thinFaces+=(faceThinFallbacks>0);
    hmin=std::min(hmin,h);hmax=std::max(hmax,h);hsum+=h;
  }
  std::printf("NODALS_AUTO_Q12_PREFLIGHT field=%s faces=%zu quads=%zu "
    "requestedYMin=%.12g requestedYMean=%.12g requestedYMax=%.12g "
    "cornersFallbackFaces=%zu cornersFallbackQuads=%zu smoothNeighbourQuads=%zu "
    "cornerLogTruncated=%d thinFallbackFaces=%zu thinFallbackQuads=%zu "
    "thinPolicy=%s thinLogTruncated=%d status=PASS\n",
    kind,faces.size(),faces.size()*12u,hmin,hsum/std::max<std::size_t>(1,faces.size()),
    hmax,fallbackFaces,fallbackQuads,nearSmooth,(int)(fallbackQuads>25),
    thinFaces,thinQuads,thinPolicy.c_str(),(int)(thinQuads>25));
  return fallback;
}
