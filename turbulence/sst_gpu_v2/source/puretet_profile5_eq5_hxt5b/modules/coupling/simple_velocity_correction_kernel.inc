// NODALS_TRUE_SIMPLE_CORRECTION_V1
// Classical SIMPLE velocity correction after solving
//     (B rAU B^T) dp = continuity
// with the sign convention used by this solver:
//     p <- p - alphaP dp
//     U <- U - rAU B^T dp
// The velocity correction uses the FULL pressure correction dp; pressure
// under-relaxation remains confined to the pressure update, as in classical
// SIMPLE. Prescribed velocity DOFs are left untouched.
template<class T>
__global__ void hxt5b_simple_velocity_correct(
    int n,
    const unsigned char*clamp,
    const T*rau,
    const T*btdp,
    T*u)
{
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=n) return;
  if(clamp && clamp[i]) return;
  u[i]-=rau[i]*btdp[i];
}


