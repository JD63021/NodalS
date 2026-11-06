// Stage13: portable mathematical objective, unchanged CUDA call signature.
__device__ __forceinline__ double spalding_profile5_objective(const double U[5],const double y[5],double nu,double ut){
 return nodals::wall::spalding_profile5_objective(U,y,nu,ut);
}
