// BF3_BERNSTEIN_TET_V1
// Exact degree-three tetrahedral Bernstein controls for the existing
// Q1 (4 vertex) + four cubic face-bubbles polynomial.
// N_i=lambda_i; F_i=27*prod_{j!=i}(lambda_j).
// Of the twenty P3 Bernstein controls, sixteen are vertex convex averages.
// Four special face controls are C_i=(v_j+v_k+v_l)/3+(9/2)*a_i.
// One global scalar face DOF means each face control is shared consistently.
static_assert(sizeof(BF3BernsteinFace)==3*sizeof(int),"unexpected BF3 face descriptor layout");

// Audit counters: 0 face lowered; 1 face upper-clipped; 2 vertex growth-clipped;
//                 3 invalid/nonfinite; 4 postprojection control failure;
//                 5 faces with nonzero coefficients after projection.
__global__ void bf3_bernstein_vertex_guard(int n, Real* x, const Real* previous,
    Real floorV,Real growth,unsigned long long* audit){
  const int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i>=n)return;
  const Real a=x[i],b=previous[i];
  if(!isfinite((double)a) || !isfinite((double)b)){
    atomicAdd(audit+3,1ULL);return;
  }
  if(a<floorV){ // Existing floor_vertex_kernel already enforces this.
    atomicAdd(audit+4,1ULL);return;
  }
  if(growth>Real(0)){
    const Real cap=growth*fmax((double)b,(double)floorV);
    if(!isfinite((double)cap)) {atomicAdd(audit+3,1ULL);return;}
    if(a>cap){x[i]=cap;atomicAdd(audit+2,1ULL);}
  }
}

__global__ void bf3_bernstein_face_projection(const BF3BernsteinFace* face,
    int nface,int nvert,Real* x,Real floorV,Real upperFactor,
    unsigned long long* audit){
  const int f=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(f>=nface)return;
  const BF3BernsteinFace g=face[f];
  const Real v0=x[g.v0],v1=x[g.v1],v2=x[g.v2];
  const Real a=x[nvert+f];
  if(!isfinite((double)v0)||!isfinite((double)v1)||
     !isfinite((double)v2)||!isfinite((double)a)){
      atomicAdd(audit+3,1ULL);return;
  }
  if(v0<floorV || v1<floorV || v2<floorV){
    atomicAdd(audit+4,1ULL);return;
  }
  const Real mean=(v0+v1+v2)/Real(3);
  const Real vmax=fmax((double)v0,fmax((double)v1,(double)v2));
  const Real upper=fmax((double)floorV,(double)(upperFactor*vmax));
  const Real control=mean+Real(4.5)*a;
  if(!isfinite((double)upper)||!isfinite((double)control)){
    atomicAdd(audit+3,1ULL);return;
  }
  Real bounded=control;
  if(control<floorV){bounded=floorV;atomicAdd(audit+0,1ULL);}
  if(control>upper){bounded=upper;atomicAdd(audit+1,1ULL);}
  if(bounded!=control)x[nvert+f]=(bounded-mean)/Real(4.5);
  const Real after=mean+Real(4.5)*x[nvert+f];
  const Real eps=Real(2e-12)*fmax(Real(1),fmax((double)fabs((double)upper),(double)fabs((double)floorV)));
  if(!isfinite((double)after)||after<floorV-eps||after>upper+eps)
    atomicAdd(audit+4,1ULL);
  if(x[nvert+f]!=Real(0))atomicAdd(audit+5,1ULL);
}

inline void bf3_bernstein_prepare(State& S,const nodals_hxt1::HostMesh&M){
  if(S.bernsteinReady)return;
  if(M.h.nhex!=0 || M.h.ntet==0)
    throw std::runtime_error("BF3_BERNSTEIN requires pure tetrahedral mesh");
  const int nvert=(int)M.h.nv;
  const int nface=S.ntotal-nvert;
  if(nface<=0 || (std::size_t)M.h.nvel!=(std::size_t)S.ntotal)
    throw std::runtime_error("BF3_BERNSTEIN inconsistent DOF counts");
  std::vector<BF3BernsteinFace> faces((std::size_t)nface,{-1,-1,-1});
  std::vector<int> incidence((std::size_t)nface,0);
  for(std::size_t c=0;c<M.tets.size();++c){
    for(int i=0;i<4;++i){
      const int id=M.tetVel[c].g[4+i]-nvert;
      if(id<0 || id>=nface)
        throw std::runtime_error("BF3_BERNSTEIN invalid global face bubble ID");
      int v[3],n=0;
      for(int j=0;j<4;++j)if(j!=i)v[n++]=M.tets[c].v[j];
      std::sort(v,v+3);
      if(v[0]<0||v[2]>=nvert||v[0]==v[1]||v[1]==v[2])
        throw std::runtime_error("BF3_BERNSTEIN invalid face vertices");
      const BF3BernsteinFace t{v[0],v[1],v[2]};
      const auto prev=faces[(std::size_t)id];
      if(incidence[(std::size_t)id]>0 &&
         (prev.v0!=t.v0||prev.v1!=t.v1||prev.v2!=t.v2))
        throw std::runtime_error("BF3_BERNSTEIN mismatched shared face topology");
      faces[(std::size_t)id]=t;
      ++incidence[(std::size_t)id];
    }
  }
  int internal=0,boundary=0;
  for(int n:incidence){
    if(n==1)++boundary;
    else if(n==2)++internal;
    else throw std::runtime_error("BF3_BERNSTEIN missing or non-manifold face");
  }
  S.bernsteinFaces.upload(faces);
  S.bernsteinReady=true;
  std::printf("BF3_BERNSTEIN_FACE_MAP faces=%d internal=%d boundary=%d "
      "globalFaceContinuity=YES degree=3 polynomialControls=20 status=PASS\n",
      nface,internal,boundary);
}

inline double bf3_bernstein_positive_env(const char*key,double fallback,double minimum){
  const char* raw=std::getenv(key);
  const double x=(raw&&*raw)?std::stod(raw):fallback;
  if(!std::isfinite(x)||x<minimum||x>1.e6)
    throw std::runtime_error(std::string("BF3_BERNSTEIN invalid ")+key);
  return x;
}

inline void bf3_bernstein_apply(State&S,const nodals_hxt1::HostMesh&M,
    Real* x,const Real* previous,Real floorV,int it,const char*label){
  bf3_bernstein_prepare(S,M);
  const double faceUpper=bf3_bernstein_positive_env("NODALS_BF3_FACE_UPPER_FACTOR",2.,1.);
  // >0: additional explicit vertex trust region (not part of Bernstein basis).
  // 0: disabled. Default 2 prevents catastrophic single-step growth of omega.
  const double growth=bf3_bernstein_positive_env("NODALS_BF3_VERTEX_GROWTH",2.,0.);
  HXT1_CUDA(cudaMemset(S.bernsteinCounters.p,0,6*sizeof(unsigned long long)));
  const int nvert=S.nvert,nface=S.ntotal-S.nvert;
  bf3_bernstein_vertex_guard<<<(nvert+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(
      nvert,x,previous,floorV,(Real)growth,S.bernsteinCounters.p);
  HXT1_CUDA(cudaGetLastError());
  bf3_bernstein_face_projection<<<(nface+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(
      S.bernsteinFaces.p,nface,nvert,x,floorV,(Real)faceUpper,S.bernsteinCounters.p);
  HXT1_CUDA(cudaGetLastError());
  unsigned long long counts[6]={};
  HXT1_CUDA(cudaMemcpy(counts,S.bernsteinCounters.p,sizeof(counts),cudaMemcpyDeviceToHost));
  const bool valid=(counts[3]==0 && counts[4]==0 && counts[5]>0);
  std::printf("BF3_BERNSTEIN_AUDIT it=%d equation=%s faceDofs=%d "
      "faceLowerClipped=%llu faceUpperClipped=%llu vertexGrowthClipped=%llu "
      "nonfinite=%llu postcheckFailed=%llu activeFaces=%llu "
      "faceUpperFactor=%.6g vertexGrowthFactor=%.6g lowerFloor=%.9e "
      "status=%s\n",it,label,nface,
      counts[0],counts[1],counts[2],counts[3],counts[4],counts[5],
      faceUpper,growth,(double)floorV,valid?"PASS":"FAIL");
  std::fflush(stdout);
  if(!valid)throw std::runtime_error("BF3_BERNSTEIN invalid projected scalar field");
}
