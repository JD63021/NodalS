// Geometry-only setup cache for genuine tetrahedral wall faces.
// Normal points are located by barycentric adjacency walk, never clamped.
inline std::vector<TetWallMatchPoint> build_tet_wall_match_height_cache(
    const HostMesh&M,const std::vector<TetFacePlan>&faces,double y)
{
  if(!(y>0))throw std::runtime_error("tet wall y must be positive");
  const int nt=(int)M.h.ntet;
  std::vector<std::array<int,4>> adj((std::size_t)nt);
  for(auto &v:adj)v.fill(-1);
  using FK=std::array<int,3>;
  std::map<FK,std::pair<int,int>> own;
  for(int c=0;c<nt;++c)for(int lf=0;lf<4;++lf){
    FK k{};for(int j=0;j<3;++j)k[(std::size_t)j]=M.tets[(std::size_t)c].v[hxt2::TET_FACE[lf][j]];
    std::sort(k.begin(),k.end());auto it=own.find(k);
    if(it==own.end())own.emplace(k,std::make_pair(c,lf));
    else{int c0=it->second.first,l0=it->second.second;
      if(adj[(std::size_t)c0][(std::size_t)l0]>=0)throw std::runtime_error("non-manifold tet adjacency");
      adj[(std::size_t)c][(std::size_t)lf]=c0;
      adj[(std::size_t)c0][(std::size_t)l0]=c;}
  }
  auto bary=[&](int c,V3 p,double l[4]){
    const auto&t=M.tets[(std::size_t)c];V3 x0=vp(M.points[(std::size_t)t.v[0]]);
    V3 a=vsub(vp(M.points[(std::size_t)t.v[1]]),x0);
    V3 b=vsub(vp(M.points[(std::size_t)t.v[2]]),x0);
    V3 d=vsub(vp(M.points[(std::size_t)t.v[3]]),x0);
    double q[3];if(!wall_match_solve_cols(a,b,d,vsub(p,x0),q))return false;
    l[1]=q[0];l[2]=q[1];l[3]=q[2];l[0]=1-q[0]-q[1]-q[2];
    return std::isfinite(l[0])&&std::isfinite(l[1])&&std::isfinite(l[2])&&std::isfinite(l[3]);};
  std::vector<TetWallMatchPoint> out(faces.size()*12u);
  std::size_t cross=0,same=0;int maxWalk=0;
  std::vector<int> seen((std::size_t)nt,0); int stamp=0;
  for(std::size_t fi=0;fi<faces.size();++fi){
    const auto&f=faces[fi];const auto&t=M.tets[(std::size_t)f.cell];
    const int a0=hxt2::TET_FACE[f.lf][0],a1=hxt2::TET_FACE[f.lf][1],a2=hxt2::TET_FACE[f.lf][2];
    const V3 n{f.n[0],f.n[1],f.n[2]};
    for(int q=0;q<12;++q){
      V3 X=vadd(vadd(vmul(HL0[q],vp(M.points[(std::size_t)t.v[a0]])),vmul(HL1[q],vp(M.points[(std::size_t)t.v[a1]]))),vmul(HL2[q],vp(M.points[(std::size_t)t.v[a2]])));
      const V3 P=vsub(X,vmul(y,n));
      int c=f.cell,steps=0,found=-1;double l[4]={};
      ++stamp;
      while(c>=0 && steps<4096 && seen[(std::size_t)c]!=stamp){
        seen[(std::size_t)c]=stamp;++steps;
        if(!bary(c,P,l))break;
        int neg=0;for(int j=1;j<4;++j)if(l[j]<l[neg])neg=j;
        if(l[neg]>=-1.e-8){found=c;break;}
        c=adj[(std::size_t)c][(std::size_t)neg];
      }
      maxWalk=std::max(maxWalk,steps);
      // Fail visibly rather than invent a boundary-clamped sample.
      if(found<0)throw std::runtime_error("PURETET sample outside mesh or walk failed face="+std::to_string(fi)+" q="+std::to_string(q)+" y="+std::to_string(y)+" steps="+std::to_string(steps));
      auto&mp=out[fi*12u+(std::size_t)q];mp.cell=found;mp.y=y;
      tbasis(l,mp.b);
      for(int j=0;j<8;++j){mp.g[j]=M.tetVel[(std::size_t)found].g[j];
        if(mp.g[j]<0||mp.g[j]>=(int)M.h.nvel||!std::isfinite(mp.b[j]))
          throw std::runtime_error("PURETET invalid interpolation weight/DOF");}
      V3 xr{};
      for(int j=0;j<4;++j)
        xr=vadd(xr,vmul(l[j],vp(M.points[(std::size_t)M.tets[(std::size_t)found].v[j]])));
      double xyzError=vnorm(vsub(P,xr));
      if(xyzError>1e-8)throw std::runtime_error("PURETET point reconstruction audit failed");
      if(found==f.cell)++same;else ++cross;
    }
  }
  std::printf("PROFILE5_TET_CACHE wallFaces=%zu quadraturePoints=%zu samplesPerQ=1 totalSamples=%zu found=%zu notFound=0 sameWallTet=%zu crossedTet=%zu maxTetsCrossed=%d y=%.12e status=PASS\n",faces.size(),faces.size()*12u,out.size(),out.size(),same,cross,maxWalk,y);
  return out;
}

