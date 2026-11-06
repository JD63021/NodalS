// Algebraic row-L1 under-relaxation for advection-dominated SST scalars.
// This mirrors the validated HXT5B momentum policy while preserving the
// fixed point of the physical scalar equation:
//   (A + Delta) q_new = b + Delta q_old,
//   Delta_i = (1/alpha - 1) * sum_j |A_ij|.
// For alpha <= 0.5 this also regularizes a finite raw central-advection
// diagonal for SGS without adding a term to the converged PDE.
__global__ void scalar_row_l1_relax_kernel(
    int n,const std::int64_t*row,const std::int32_t*col,const std::int32_t*dp,Real*val,Real*rhs,
    const Real*old,Real alpha,unsigned long long*audit,int vertexCount,int vertexOnlyL1,
    const Real*calibratedL1,const Real calibratedBoost)
{
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i>=n)return;
  const int d=dp[i];const Real d0=val[d];
  if(!(d0>Real(0))||!isfinite((double)d0))atomicAdd(audit+0,1ULL);
  Real l1=Real(0);bool finite=true;
  for(std::int64_t q=row[i];q<row[i+1];++q){
    if(vertexOnlyL1 && i<vertexCount && col[q]>=vertexCount)continue;
    Real a=val[q];if(!isfinite((double)a))finite=false;l1+=(Real)fabs((double)a);
  }
  // P1_L1_CALIB: retain original masked-row damping using a tiny per-vertex
  // calibration vector. No bubble scalar CSR or extra matrix is allocated here.
  // In the reference iteration, calibratedL1[i] = fullL1_i / vertexL1_i.
  // Multiplying only the damping norm preserves the physical P1 fixed point.
  if(calibratedL1 && i<vertexCount){
    const Real factor=calibratedL1[i]*calibratedBoost;
    if(!isfinite((double)factor)||!(factor>Real(0))) {atomicAdd(audit+1,1ULL);return;}
    l1*=factor;
  }
  const auto relax=nodals::transport::row_l1_plan<Real>(d0,l1,alpha);const Real delta=relax.delta;const Real dn=relax.diagonal;
  if(!finite||!(l1>Real(0))||!isfinite((double)delta)||!(dn>Real(0))||!isfinite((double)dn)){atomicAdd(audit+1,1ULL);return;}
  val[d]=dn;rhs[i]+=delta*old[i];
}

// P1_L1_CALIB_V1: one-time 39,208-entry per-equation calibration from the
// original masked P1 operator. Never allocate the enriched scalar CSR in the
// compact production path; reference is only needed for one short pilot run.
// Because the nonlinear coefficients change, this is an iteration-1 exact,
// subsequent-iteration FROZEN-FACTOR approximation, not exact dynamic parity.
// Versioned file: [8-byte magic][3 int64 fields: nvert, vvnnz, graphHash]
//                 [nvert IEEE754 float64 factors]. No free-form pickle.
static inline std::uint64_t p1_l1_hash_push(std::uint64_t h,std::uint64_t x){
  for(int k=0;k<8;++k){h^=(std::uint8_t)((x>>(8*k))&0xffU);h*=1099511628211ULL;}
  return h;
}
static inline std::uint64_t p1_l1_vertex_graph_hash(
  const std::vector<std::int64_t>&row,const std::vector<std::int32_t>&col,int nvert,
  std::int64_t &vvnnz){
  std::uint64_t h=1469598103934665603ULL;
  vvnnz=0;
  for(int i=0;i<nvert;++i){
    std::int64_t count=0;
    for(std::int64_t q=row[(size_t)i];q<row[(size_t)i+1];++q){
      const int j=col[(size_t)q];
      if(j<nvert){h=p1_l1_hash_push(h,(std::uint64_t)(unsigned int)j);++count;}
    }
    h=p1_l1_hash_push(h,(std::uint64_t)count);vvnnz+=count;
  }
  return h;
}
static inline void p1_l1_dump_reference(const std::string &dir,const char *eq,
 hxt4b::GpuCSR&A,int nvert){
  if(!(nvert>0&&A.n>nvert))throw std::runtime_error("P1_L1_CALIB reference requires masked enriched CSR");
  std::vector<std::int64_t> row((size_t)A.n+1);
  std::vector<std::int32_t> col(A.nnz);
  std::vector<Real> val(A.nnz);
  HXT1_CUDA(cudaMemcpy(row.data(),A.row.p,row.size()*sizeof(row[0]),cudaMemcpyDeviceToHost));
  HXT1_CUDA(cudaMemcpy(col.data(),A.col.p,col.size()*sizeof(col[0]),cudaMemcpyDeviceToHost));
  HXT1_CUDA(cudaMemcpy(val.data(),A.val.p,val.size()*sizeof(val[0]),cudaMemcpyDeviceToHost));
  std::vector<double> factor((size_t)nvert);
  double lo=1.e300,hi=0,avg=0;
  for(int i=0;i<nvert;++i){
    double full=0,vertex=0;
    for(std::int64_t q=row[(size_t)i];q<row[(size_t)i+1];++q){
      const double a=std::abs((double)val[(size_t)q]);
      if(!std::isfinite(a))throw std::runtime_error("P1_L1_CALIB non-finite reference matrix");
      full+=a;if(col[(size_t)q]<nvert)vertex+=a;
    }
    if(!(vertex>0))throw std::runtime_error("P1_L1_CALIB nonpositive vertex L1");
    const double f=full/vertex;
    if(!std::isfinite(f)||f<1-1.e-12||f>1.e10)throw std::runtime_error("P1_L1_CALIB invalid multiplier");
    factor[(size_t)i]=std::max(1.,f);lo=std::min(lo,f);hi=std::max(hi,f);avg+=f;
  }
  std::int64_t vvnnz=0;
  const std::uint64_t hash=p1_l1_vertex_graph_hash(row,col,nvert,vvnnz);
  const std::string path=dir+"/p1_l1_"+std::string(eq)+"_v1.bin";
  std::ofstream o(path,std::ios::binary);
  if(!o)throw std::runtime_error("P1_L1_CALIB cannot open reference file: "+path);
  const char magic[8]={'P','1','L','1','V','0','0','1'};
  const std::int64_t header[3]={(std::int64_t)nvert,vvnnz,(std::int64_t)hash};
  o.write(magic,8);o.write((const char*)header,sizeof(header));
  o.write((const char*)factor.data(),factor.size()*sizeof(double));
  if(!o)throw std::runtime_error("P1_L1_CALIB unable to write: "+path);
  std::printf("P1_L1_REFERENCE equation=%s vertices=%d vvnnz=%lld factorMin=%.8g factorMean=%.8g factorMax=%.8g path=%s status=PASS\n",
    eq,nvert,(long long)vvnnz,lo,avg/nvert,hi,path.c_str());
}
static inline std::vector<Real> p1_l1_load_calibration(const std::string&dir,
 const char *eq,hxt4b::GpuCSR&A,int nvert){
  if(A.n!=nvert)throw std::runtime_error("P1_L1_CALIB compact matrix shape mismatch");
  const std::string path=dir+"/p1_l1_"+std::string(eq)+"_v1.bin";
  std::ifstream in(path,std::ios::binary);
  if(!in)throw std::runtime_error("P1_L1_CALIB missing file: "+path);
  char magic[8];std::int64_t header[3];
  in.read(magic,8);in.read((char*)header,sizeof(header));
  if(!in||std::memcmp(magic,"P1L1V001",8)!=0||header[0]!=nvert||header[1]<1)
    throw std::runtime_error("P1_L1_CALIB header/mesh-size mismatch: "+path);
  std::vector<double> d((size_t)nvert);
  in.read((char*)d.data(),d.size()*sizeof(double));
  if(!in)throw std::runtime_error("P1_L1_CALIB truncated coefficients: "+path);
  char extra=0;in.read(&extra,1);
  if((size_t)in.gcount()!=0)throw std::runtime_error("P1_L1_CALIB trailing bytes: "+path);
  if(!in.eof())throw std::runtime_error("P1_L1_CALIB truncated coefficients: "+path);
  std::vector<std::int64_t> row((size_t)nvert+1);
  std::vector<std::int32_t> col(A.nnz);
  HXT1_CUDA(cudaMemcpy(row.data(),A.row.p,row.size()*sizeof(row[0]),cudaMemcpyDeviceToHost));
  HXT1_CUDA(cudaMemcpy(col.data(),A.col.p,col.size()*sizeof(col[0]),cudaMemcpyDeviceToHost));
  std::int64_t vvnnz=0;
  const auto hash=p1_l1_vertex_graph_hash(row,col,nvert,vvnnz);
  if(header[1]!=vvnnz||(std::uint64_t)header[2]!=hash)
    throw std::runtime_error("P1_L1_CALIB CSR graph mismatch (wrong mesh/numbering): "+path);
  std::vector<Real> out((size_t)nvert);
  double lo=1.e300,hi=0,avg=0;
  for(int i=0;i<nvert;++i){
    const double f=d[(size_t)i];
    if(!std::isfinite(f)||f<1||f>1.e10)throw std::runtime_error("P1_L1_CALIB bad factor: "+path);
    out[(size_t)i]=(Real)f;lo=std::min(lo,f);hi=std::max(hi,f);avg+=f;
  }
  std::printf("P1_L1_CALIBRATION equation=%s vertices=%d vvnnz=%lld factorMin=%.8g factorMean=%.8g factorMax=%.8g source=%s status=PASS\n",
    eq,nvert,(long long)vvnnz,lo,avg/nvert,hi,path.c_str());
  return out;
}

// P1_PARITY_DIAG: DEBUG-ONLY, raw and relaxed vertex blocks for both
// masked and compact P1. Binary format consumed by COMPARE_P1_OPERATORS.py.
// Not activated unless SST_P1_PARITY_DIR is set. Only outer iteration 1.
inline void p1_parity_dump_vertex_block(
    const char*dir,const char*mode,const char*equation,const char*phase,
    hxt4b::GpuCSR&A,const Dev<Real>&rhs,int nvert)
{
  if(!dir || !*dir)return;
  if(nvert>A.n || nvert<=0 || sizeof(Real)!=sizeof(double))
    throw std::runtime_error("P1 parity dump invalid vertex count or non-fp64 solver");
  std::vector<std::int64_t> row((std::size_t)A.n+1);
  std::vector<std::int32_t> col(A.nnz);
  std::vector<Real> val(A.nnz),b((std::size_t)nvert);
  HXT1_CUDA(cudaMemcpy(row.data(),A.row.p,row.size()*sizeof(row[0]),cudaMemcpyDeviceToHost));
  HXT1_CUDA(cudaMemcpy(col.data(),A.col.p,col.size()*sizeof(col[0]),cudaMemcpyDeviceToHost));
  HXT1_CUDA(cudaMemcpy(val.data(),A.val.p,val.size()*sizeof(val[0]),cudaMemcpyDeviceToHost));
  HXT1_CUDA(cudaMemcpy(b.data(),rhs.p,b.size()*sizeof(b[0]),cudaMemcpyDeviceToHost));
  std::vector<std::int64_t> rr((std::size_t)nvert+1,0);
  std::vector<std::int32_t> cc;cc.reserve((std::size_t)nvert*16);
  std::vector<double> vv;vv.reserve((std::size_t)nvert*16);
  for(int i=0;i<nvert;++i){
    for(std::int64_t q=row[(std::size_t)i];q<row[(std::size_t)i+1];++q){
      const int j=col[(std::size_t)q];
      if(j<nvert){cc.push_back(j);vv.push_back((double)val[(std::size_t)q]);}
    }
    rr[(std::size_t)i+1]=(std::int64_t)cc.size();
  }
  const std::string path=std::string(dir)+"/p1_"+mode+"_"+equation+"_"+phase+".bin";
  std::ofstream o(path,std::ios::binary);
  if(!o)throw std::runtime_error("P1 parity dump cannot open "+path);
  const char magic[8]={'P','1','D','I','A','G','0','1'};
  const std::int64_t count[2]={(std::int64_t)nvert,(std::int64_t)cc.size()};
  o.write(magic,8);o.write((const char*)count,sizeof(count));
  o.write((const char*)rr.data(),rr.size()*sizeof(rr[0]));
  o.write((const char*)cc.data(),cc.size()*sizeof(cc[0]));
  o.write((const char*)vv.data(),vv.size()*sizeof(vv[0]));
  o.write((const char*)b.data(),b.size()*sizeof(b[0]));
  if(!o)throw std::runtime_error("P1 parity dump write failed "+path);
  std::printf("P1_PARITY_DUMP mode=%s equation=%s phase=%s rows=%d nnz=%zu path=%s status=PASS\n",
      mode,equation,phase,nvert,cc.size(),path.c_str());
}

__global__ void field_diff_kernel(int n,const Real*a,const Real*b,Real*d){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)d[i]=a[i]-b[i];}
__global__ void floor_vertex_kernel(int n,Real floorv,Real*x,unsigned long long*count){int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n&&(!(x[i]>=floorv)||!isfinite((double)x[i]))){x[i]=floorv;atomicAdd(count,1ULL);}}

