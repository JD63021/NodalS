// Included inside namespace hxt5c, once, after `using Real`.
// Static GPU P1 vertex distance field. Null means Stage16 analytic pipe path.
__device__ const Real* stage20_wall_vertex_data = nullptr;
__device__ __forceinline__ Real stage20_distance(const nodals_hxt1::TetConn*tc,int cell,const double l[4]){
  const Real* values=stage20_wall_vertex_data;
  if(!values)return Real(-1); // keep accepted analytical kernel arithmetic
  Real d=0;
  for(int j=0;j<4;++j)d+=Real(l[j])*values[tc[cell].v[j]];
  return fmax(Real(0),d);
}
struct Stage20GpuDistanceOwner{
  Real* allocation=nullptr;
  Stage20GpuDistanceOwner()=default;
  Stage20GpuDistanceOwner(const Stage20GpuDistanceOwner&)=delete;
  Stage20GpuDistanceOwner& operator=(const Stage20GpuDistanceOwner&)=delete;
  void attach(const std::vector<double>&d){
    if(allocation)throw std::runtime_error("Stage20 distance already installed");
    std::vector<Real> v;v.reserve(d.size());for(double x:d)v.push_back((Real)x);
    HXT1_CUDA(cudaMalloc((void**)&allocation,v.size()*sizeof(Real)));
    HXT1_CUDA(cudaMemcpy(allocation,v.data(),v.size()*sizeof(Real),cudaMemcpyHostToDevice));
    const Real* ptr=allocation;
    HXT1_CUDA(cudaMemcpyToSymbol(stage20_wall_vertex_data,&ptr,sizeof(ptr)));
  }
  ~Stage20GpuDistanceOwner(){
    if(allocation){
      const Real* zero=nullptr;
      (void)cudaMemcpyToSymbol(stage20_wall_vertex_data,&zero,sizeof(zero));
      (void)cudaFree(allocation);
    }
  }
};
