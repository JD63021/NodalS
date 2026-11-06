// Stage13: CUDA adapters; numerical Spalding law is authored in portable core.
__device__ __forceinline__ double spy(double up,double k,double B){return nodals::wall::spy(up,k,B);}
__device__ __forceinline__ double spg(double up,double reY){return nodals::wall::spg(up,reY);}
__device__ __forceinline__ double spyd_newton(double up,double k,double B){return nodals::wall::spyd_newton(up,k,B);}
__device__ __forceinline__ double spgd_newton(double up,double reY){return nodals::wall::spgd_newton(up,reY);}
__device__ __forceinline__ bool spalding(double slip,double y,double nu,double&ut,double&yp,double&beta){return nodals::wall::spalding(slip,y,nu,ut,yp,beta);}
