// Stage 2.B.2: packed device record with *exact original* FP64
// barycentric values, cell and eight global velocity DOF indices.
// On demand reconstruct the four BF3 basis values by the EXACT host tbasis
// multiplication order (27 * three barycentrics).  Never use FP32.
// This is a value-returning accessor: callers bind a const reference to the
// temporary, extending its lifetime for the local scope.
__device__ __forceinline__ TetWallMatchPoint stage27_s02b_at(
    const TetWallPackedPoint* ptr,int fi,int q,int physical_slot){
  const int packed_slot=(::stage27_s02b_stride==5) ? physical_slot
                         : ::stage27_s02b_mapping[physical_slot];
  const int valid_slot=packed_slot>=0?packed_slot:0;
  const TetWallPackedPoint& r=ptr[
      ((std::size_t)fi*12u+(std::size_t)q)*
        (std::size_t)::stage27_s02b_stride+(std::size_t)valid_slot];
  TetWallMatchPoint result{};
  result.cell=r.cell;
  result.y=r.y; // Certified per-quad local physical sample height.
  #pragma unroll
  for(int j=0;j<8;++j)result.g[j]=r.g[j];
  #pragma unroll
  for(int j=0;j<4;++j)result.b[j]=r.l[j];
  #pragma unroll
  for(int i=0;i<4;++i){
    double v=27.0;
    #pragma unroll
    for(int j=0;j<4;++j)if(j!=i)v*=result.b[j];
    result.b[4+i]=v;
  }
  return result;
}
__device__ __forceinline__ double stage27_s02b_y(int physical_slot){
  return ::stage27_s02b_heights[physical_slot];
}
