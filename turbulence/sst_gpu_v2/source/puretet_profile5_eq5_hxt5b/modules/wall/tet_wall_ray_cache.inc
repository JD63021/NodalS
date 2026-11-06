// NODALS_FACE_YPLUS30_A_20261105
// Stage21 adapter: portable ray containment + original enriched P1+BF3 basis.
// Requires hxt5b namespace context and existing TetWallMatchPoint, tbasis, HL0..2.
inline std::vector<TetWallMatchPoint> build_tet_wall_ray_profile_cache(
    const HostMesh&M,const std::vector<TetFacePlan>&faces,const double y[5],
    int maxCrossings,double baryTolerance,unsigned activeMask=0x1fu, bool localAdapt=false,
    const std::vector<double>* faceY3=nullptr)
{
  using namespace ::nodals::wall::ray;
  std::vector<P> vertices;vertices.reserve(M.points.size());
  for(const auto& p:M.points)vertices.push_back(P{double(p.x),double(p.y),double(p.z)});
  std::vector<T> tet;tet.reserve(M.tets.size());
  for(const auto& c:M.tets)tet.push_back(T{int(c.v[0]),int(c.v[1]),int(c.v[2]),int(c.v[3])});
  const TetRayTrace locator(std::move(vertices),std::move(tet));
  // A five-slot host ABI is retained for Stage27 packed device consumers.
  // Only requested slots are ray-traced; inactive records are inert aliases of
  // the first contained active hit.  Five-point mode is the original path.
  if(!(activeMask&0x1fu) || (activeMask&~0x1fu))
    throw std::runtime_error("STAGE27_ACTIVE_RAY invalid active mask");
  std::vector<TetWallMatchPoint> result(faces.size()*12u*5u);
  std::size_t cross=0,same=0;int maxWalk=0;
  std::vector<int> activeSlots;
  std::vector<double> baseDistances;
  for(int m=0;m<5;++m)if(activeMask&(1u<<m)){
    activeSlots.push_back(m);baseDistances.push_back(y[m]);
  }
  // Local adaptation is an explicit opt-in restricted to one active sample.
  // Every accepted replacement endpoint is re-traced with the strict
  // face-connected algorithm. No global endpoint search is allowed.
  if(faceY3 && faceY3->size()!=faces.size())
    throw std::runtime_error("NODALS_FACE_YPLUS_RAY_COUNT_MISMATCH");
  if(faceY3 && !(y[2]>0.))
    throw std::runtime_error("NODALS_FACE_YPLUS_BAD_BASE_HEIGHT");
  if(localAdapt && activeSlots.size()!=1)
    throw std::runtime_error("STAGE27_LOCAL_RAY_ADAPT requires exactly one active physical slot");
  std::size_t shortened=0;
  double minFraction=1.0,maxFraction=0.0;
  const double floorFraction=0.25,safety=0.90;
  int ordinal[5]={-1,-1,-1,-1,-1};
  for(std::size_t k=0;k<activeSlots.size();++k)
    ordinal[activeSlots[k]]=(int)k;
  for(std::size_t fi=0;fi<faces.size();++fi){
    std::vector<double> distances=baseDistances;
    if(faceY3){
      const double selected=(*faceY3)[fi];
      if(!(std::isfinite(selected)&&selected>0.))
        throw std::runtime_error("NODALS_FACE_YPLUS_BAD_DISTANCE face="+std::to_string(fi));
      const double ratio=selected/y[2];
      for(double &v:distances)v*=ratio;
    }
    const auto&f=faces[fi];const auto&t=M.tets[(std::size_t)f.cell];
    const int a0=hxt2::TET_FACE[f.lf][0],a1=hxt2::TET_FACE[f.lf][1],a2=hxt2::TET_FACE[f.lf][2];
    const P inward={-f.n[0],-f.n[1],-f.n[2]};
    for(int q=0;q<12;++q){
      const auto&v0=M.points[(std::size_t)t.v[a0]],&v1=M.points[(std::size_t)t.v[a1]],&v2=M.points[(std::size_t)t.v[a2]];
      P start={HL0[q]*v0.x+HL1[q]*v1.x+HL2[q]*v2.x,
               HL0[q]*v0.y+HL1[q]*v1.y+HL2[q]*v2.y,
               HL0[q]*v0.z+HL1[q]*v1.z+HL2[q]*v2.z};
      std::vector<Hit> hits;
      std::vector<double> localHeights=distances;
      bool locallyShortened=false;
      try{hits=locator.trace(f.cell,start,inward,distances,maxCrossings,baryTolerance);}
      catch(const std::exception&e){
        if(localAdapt){
          const double requested=distances.front();
          double lo=0.,hi=requested;
          // A success certifies the complete segment from the wall to lo.
          // Failure narrows the search without accepting an exterior point.
          for(int it=0;it<28;++it){
            const double mid=0.5*(lo+hi);
            if(!(mid>0.))break;
            try{
              locator.trace(f.cell,start,inward,std::vector<double>{mid},
                            maxCrossings,baryTolerance);
              lo=mid;
            }catch(const std::exception&){hi=mid;}
          }
          const double chosen=std::min(requested,safety*lo);
          if(!(chosen>=floorFraction*requested && chosen<requested))
            throw std::runtime_error(
              "STAGE27_LOCAL_RAY_UNSAFE face="+std::to_string(fi)+
              " q="+std::to_string(q)+" ownerTet="+std::to_string(f.cell)+
              " physicalSlot="+std::to_string(activeSlots.front()+1)+
              " requestedY="+std::to_string(requested)+
              " maxCertifiedY="+std::to_string(lo)+
              " safeY="+std::to_string(chosen)+
              " minFraction=0.25 strictContainment=FAIL original="+e.what());
          // Mandatory final containment proof at the exact accepted distance.
          hits=locator.trace(f.cell,start,inward,std::vector<double>{chosen},
                             maxCrossings,baryTolerance);
          localHeights[0]=chosen;
          locallyShortened=true;
          ++shortened;
          const double frac=chosen/requested;
          minFraction=std::min(minFraction,frac);
          maxFraction=std::max(maxFraction,frac);
        }else{
        // Expensive prefix probes run only on failure to identify the exact
        // requested slot; never skip, shorten, or accept an invalid ray.
        int failedSlot=activeSlots.front();
        for(std::size_t k=0;k<activeSlots.size();++k){
          try{
            locator.trace(f.cell,start,inward,
              std::vector<double>(distances.begin(),distances.begin()+k+1),
              maxCrossings,baryTolerance);
          }catch(const std::exception&){failedSlot=activeSlots[k];break;}
        }
        throw std::runtime_error("STAGE27_ACTIVE_RAY_PRECHECK face="+std::to_string(fi)+
          " q="+std::to_string(q)+" ownerTet="+std::to_string(f.cell)+
          " sampleIndex="+std::to_string(failedSlot+1)+
          " requestedY="+std::to_string(distances[(std::size_t)ordinal[failedSlot]])+
          " strictContainment=FAIL "+e.what());
        }
      }
      (void)locallyShortened;
      for(int m=0;m<5;++m){
        // Inactive entries have valid, strictly contained *alias* geometry.
        // Their physical heights are retained, and no consumer may use their
        // interpolation as active data.  See STAGE24B_ACTIVE_SAMPLING.
        const Hit&hit=hits[(std::size_t)(ordinal[m]>=0?ordinal[m]:0)];
        auto&mp=result[(fi*12u+(std::size_t)q)*5u+(std::size_t)m];
        mp.cell=hit.cell;mp.y=(ordinal[m]>=0)?localHeights[(std::size_t)ordinal[m]]:y[m];
        double lam[4];for(int j=0;j<4;++j)lam[j]=hit.bary[(std::size_t)j];
        tbasis(lam,mp.b);
        for(int j=0;j<8;++j){
          mp.g[j]=M.tetVel[(std::size_t)hit.cell].g[j];
          if(mp.g[j]<0||mp.g[j]>=(int)M.h.nvel||!std::isfinite(mp.b[j]))
            throw std::runtime_error("STAGE21 invalid enriched interpolation coefficient");
        }
        if(ordinal[m]>=0){
          maxWalk=std::max(maxWalk,hit.crossings);
          if(hit.cell==f.cell)++same;else ++cross;
        }
      }
    }
  }
  const std::size_t actualRays=faces.size()*12u*activeSlots.size();
  std::printf("STAGE27_LOCAL_RAY_STATS enabled=%d wallQuads=%zu shortened=%zu "
    "fraction=%.9g minRetainedFrac=%.9g maxRetainedFrac=%.9g "
    "safety=0.90 minAllowed=0.25 strictContainment=YES physicalHeight=PER_QUAD status=PASS\n",
    (int)localAdapt,faces.size()*12u,shortened,
    actualRays?double(shortened)/double(actualRays):0.,
    shortened?minFraction:1.,shortened?maxFraction:1.);
  std::printf("SST_WALL_RAY_CACHE wallFaces=%zu quadraturePoints=%zu samplePoints=%zu "
    "contained=%zu sameTet=%zu crossedTet=%zu maxCrossings=%d intervalRecoveries=%zu profile=%s "
    "basis=P1_BF3 status=PASS\n",faces.size(),faces.size()*12u,result.size(),
    actualRays,same,cross,maxWalk,locator.interval_recoveries(),
    activeMask==0x1fu?"FIVE_POINT":"ACTIVE_ONLY");
  std::printf("STAGE27_ACTIVE_RAY_CACHE activeMask=0x%x actualRays=%zu "
    "legacyHostSlots=%zu strictContainment=YES inactiveRecords=INERT_ALIASES status=PASS\n",
    activeMask,actualRays,result.size());
  return result;
}
