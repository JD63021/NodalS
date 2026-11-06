namespace hxt5b_compact {

using Real = hxt4b::Real;

__device__ __forceinline__ void atomic_max_double(double* addr,double v)
{
  auto* p=(unsigned long long*)addr;
  unsigned long long old=*p,ass;
  do {
    ass=old;
    if(__longlong_as_double((long long)ass)>=v) break;
    old=atomicCAS(
      p,ass,
      (unsigned long long)__double_as_longlong(v));
  } while(ass!=old);
}

__global__ void continuity_stats_kernel(
    int n,const Real*x,double*out)
{
  __shared__ double ss[256];
  __shared__ double mm[256];

  const int t=threadIdx.x;
  const int i=(int)(blockIdx.x*blockDim.x+t);

  const double v=(i<n)?(double)x[i]:0.0;

  ss[t]=v;
  mm[t]=(i<n)?fabs(v):0.0;

  __syncthreads();

  for(int d=128;d>0;d>>=1){
    if(t<d){
      ss[t]+=ss[t+d];
      mm[t]=fmax(mm[t],mm[t+d]);
    }
    __syncthreads();
  }

  if(t==0){
    atomicAdd(out+0,ss[0]);
    atomic_max_double(out+1,mm[0]);
  }
}

struct ContinuityStats {
  double globalSigned=0.0;
  double globalAbs=0.0;
  double localMax=0.0;
};

inline ContinuityStats continuity_stats(
    int n,const Real*x,nodals_hxt1::Dev<double>&buf)
{
  if(buf.n<2)
    throw std::runtime_error(
      "compact continuity stats buffer too small");

  HXT1_CUDA(cudaMemset(buf.p,0,2*sizeof(double)));

  continuity_stats_kernel<<<(n+255)/256,256>>>(
    n,x,buf.p);

  HXT1_CUDA(cudaGetLastError());

  double h[2]={0.0,0.0};

  HXT1_CUDA(cudaMemcpy(
    h,buf.p,2*sizeof(double),
    cudaMemcpyDeviceToHost));

  ContinuityStats q;
  q.globalSigned=h[0];
  q.globalAbs=std::fabs(h[0]);
  q.localMax=h[1];
  return q;
}

 } // namespace hxt5b_compact
