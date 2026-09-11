#pragma once
#include "h10_dg_inlet.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <cmath>
#include <stdexcept>

namespace nodals_gpu {

// Gate 3: promoted weak Spalding wall used by the validated CPU RANS pipe:
// legacy face trace, y=0.25*(3V/A), kappa=0.4, B=5.5, betaScale=1,
// consistent tangent with blend=0.5, molecular/Nitsche consistency OFF.
// Wall velocity entities remain globally free because the scalar momentum
// topology is shared.  Ux/Uy are algebraically clamped to zero; Uz remains
// active and receives the nonlinear wall-law Robin operator.
struct H11WallFaceDevice {
  std::int32_t cell=0;
  std::uint8_t opp=0;
  OperatorReal area=0;
  OperatorReal y=0;
};

inline std::vector<H11WallFaceDevice> h11_build_wall_faces(
    const SerialTetMesh&M,const G4SetupHost&S,double distanceFactor=0.25)
{
  std::vector<H11WallFaceDevice> out;
  if(!S.weakWall)return out;
  const auto&P=M.patches[(std::size_t)S.pipe.wall];
  out.reserve((std::size_t)P.n_faces);
  for(int f=P.start_face;f<P.start_face+P.n_faces;++f){
    const int c=M.owner[(std::size_t)f];
    int opp=-1;for(int i=0;i<4;++i)if(M.opp_face[(std::size_t)c][i]==f){opp=i;break;}
    if(opp<0)throw std::runtime_error("Gate3 wall opposite face not found");
    const Vec3d sf=face_outward_area_vector_g2(M,f);
    const double area=std::sqrt(sf.x*sf.x+sf.y*sf.y+sf.z*sf.z);
    if(!(area>0.0))throw std::runtime_error("Gate3 degenerate wall face");
    const double hn=S.cells[(std::size_t)c].det/(2.0*area); // 3V/A = det/(2A)
    if(!(hn>0.0))throw std::runtime_error("Gate3 invalid wall normal length");
    H11WallFaceDevice q;q.cell=c;q.opp=(std::uint8_t)opp;
    q.area=OperatorReal(area);q.y=OperatorReal(distanceFactor*hn);out.push_back(q);
  }
  return out;
}

inline std::vector<std::uint8_t> h11_build_wall_row_mask(const G4SetupHost&S){
  std::vector<std::uint8_t> m((std::size_t)S.topo.n,0);
  if(!S.weakWall)return m;
  for(std::size_t e=0;e<S.wallEntity.size();++e)if(S.wallEntity[e]){
    const int g=S.momMask.g2free[e];if(g>=0)m[(std::size_t)g]=1;
  }
  return m;
}

__device__ __constant__ double h11_l0[7]={
  1.0/3.0,0.059715871789770,0.470142064105115,0.470142064105115,
  0.797426985353087,0.101286507323456,0.101286507323456};
__device__ __constant__ double h11_l1[7]={
  1.0/3.0,0.470142064105115,0.059715871789770,0.470142064105115,
  0.101286507323456,0.797426985353087,0.101286507323456};
__device__ __constant__ double h11_l2[7]={
  1.0/3.0,0.470142064105115,0.470142064105115,0.059715871789770,
  0.101286507323456,0.101286507323456,0.797426985353087};
__device__ __constant__ double h11_w[7]={
  0.225000000000000,0.132394152788506,0.132394152788506,0.132394152788506,
  0.125939180544827,0.125939180544827,0.125939180544827};

__device__ __forceinline__ double h11_yplus_from_uplus(double up,double kappa,double B){
  if(!(up>=0.0))return NAN;const double x=kappa*up;double rem=0.0;
  if(fabs(x)<1e-3){const double x2=x*x,x4=x2*x2;rem=x4*(1.0/24.0+x/120.0+x2/720.0+x2*x/5040.0);}
  else rem=expm1(x)-x-0.5*x*x-(x*x*x)/6.0;
  return up+exp(-kappa*B)*rem;
}
__device__ __forceinline__ double h11_yplus_deriv(double up,double kappa,double B){
  if(!(up>=0.0))return NAN;const double x=kappa*up;double rem=0.0;
  if(fabs(x)<1e-3){const double x2=x*x,x3=x2*x;rem=x3*(1.0/6.0+x/24.0+x2/120.0+x3/720.0);}
  else rem=expm1(x)-x-0.5*x*x;
  return 1.0+exp(-kappa*B)*kappa*rem;
}
__device__ __forceinline__ double h11_root_g(double up,double reY,double kappa,double B){
  return up*h11_yplus_from_uplus(up,kappa,B)-reY;
}
__device__ __forceinline__ bool h11_penalty(
    double slip,double y,double nu,double kappa,double B,
    double&uTau,double&yPlus,double&uPlus,double&beta)
{
  if(!(y>0.0)||!(nu>0.0)||!(kappa>0.0)||!(B>0.0)||!isfinite(slip))return false;
  const double U=fabs(slip),reY=y*U/nu;
  if(reY<=1e-14){uTau=0;yPlus=0;uPlus=0;beta=nu/y;return isfinite(beta);}
  double lo=0.0,hi=fmax(1.0,sqrt(reY)+1.0);
  int expand=0;while(h11_root_g(hi,reY,kappa,B)<0.0&&expand<40){hi*=2.0;++expand;}
  const double ghi=h11_root_g(hi,reY,kappa,B);if(!isfinite(ghi)||ghi<0.0)return false;
  for(int it=0;it<70;++it){const double mid=0.5*(lo+hi);if(h11_root_g(mid,reY,kappa,B)>0.0)hi=mid;else lo=mid;}
  uPlus=0.5*(lo+hi);if(!(uPlus>0.0)||!isfinite(uPlus))return false;
  uTau=U/uPlus;yPlus=y*uTau/nu;beta=uTau*uTau/U;
  return isfinite(uTau)&&isfinite(yPlus)&&isfinite(beta)&&beta>0.0;
}
__device__ __forceinline__ bool h11_tangent(
    double uPlus,double beta,double kappa,double B,double&tangent,double&ratio)
{
  if(!(beta>0.0)||!isfinite(beta)||!(uPlus>=0.0)||!isfinite(uPlus))return false;
  if(uPlus<=1e-14){tangent=beta;ratio=1.0;return true;}
  const double F=h11_yplus_from_uplus(uPlus,kappa,B),Fp=h11_yplus_deriv(uPlus,kappa,B),den=F+uPlus*Fp;
  if(!(F>0.0)||!(Fp>0.0)||!(den>0.0)||!isfinite(den))return false;
  ratio=2.0*uPlus*Fp/den;tangent=beta*ratio;
  return isfinite(tangent)&&tangent>0.0&&isfinite(ratio)&&ratio>0.0;
}

__device__ __forceinline__ void h11_atomic_max(double*addr,double v){
  auto*p=reinterpret_cast<unsigned long long*>(addr);unsigned long long old=*p,assumed;
  do{assumed=old;if(__longlong_as_double((long long)assumed)>=v)break;old=atomicCAS(p,assumed,(unsigned long long)__double_as_longlong(v));}while(assumed!=old);
}

__global__ void h11_spalding_wall_kernel(
    const H11WallFaceDevice* __restrict__ faces,int nf,
    const G4CellPlanDevice* __restrict__ cells,const std::int64_t* __restrict__ row,
    const StateReal* __restrict__ fixed,const StateReal* __restrict__ u2,
    double nu,double kappa,double B,double tangentBlend,
    OperatorReal* __restrict__ aval,StateReal* __restrict__ cr2,double* __restrict__ stats)
{
  const int fi=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(fi>=nf)return;
  const auto&F=faces[fi];const auto&cp=cells[F.cell];
  int fv[3],kk=0;for(int i=0;i<4;++i)if(i!=(int)F.opp)fv[kk++]=i;
  const int act[4]={fv[0],fv[1],fv[2],4+(int)F.opp};
  double uzc[4];for(int j=0;j<4;++j){const int r=cp.ref[act[j]];uzc[j]=(r>=0)?(double)u2[r]:(double)fixed[3*(std::size_t)(-r-1)+2];}
  double W[16]={0},R[4]={0};
  double areaW=0,ut2W=0,ypW=0,slipW=0,maxYp=0,maxUt=0,rootFail=0,tanFail=0;
  for(int q=0;q<7;++q){
    const double l0=h11_l0[q],l1=h11_l1[q],l2=h11_l2[q];
    const double phi[4]={l0,l1,l2,27.0*l0*l1*l2};
    double uz=0;for(int j=0;j<4;++j)uz+=uzc[j]*phi[j];
    const double slip=fabs(uz),y=(double)F.y;
    double ut=0,yp=0,up=0,beta=0;bool ok=h11_penalty(slip,y,nu,kappa,B,ut,yp,up,beta);
    if(!ok){beta=nu/y;ut=0;yp=0;up=0;rootFail+=1.0;}
    double tan=beta,ratio=1.0;if(ok&&!h11_tangent(up,beta,kappa,B,tan,ratio)){tan=beta;ratio=1.0;tanFail+=1.0;}
    const double jac=beta+tangentBlend*(tan-beta);
    const double rhs=(jac-beta)*uz;
    const double w=h11_w[q]*(double)F.area;
    for(int a=0;a<4;++a){R[a]+=rhs*phi[a]*w;for(int b=0;b<4;++b)W[4*a+b]+=jac*phi[a]*phi[b]*w;}
    areaW+=w;ut2W+=ut*ut*w;ypW+=yp*w;slipW+=slip*w;maxYp=fmax(maxYp,yp);maxUt=fmax(maxUt,ut);
  }
  for(int ia=0;ia<4;++ia){const int a=act[ia],rr=cp.ref[a];if(rr<0)continue;
    atomicAdd(cr2+rr,StateReal(R[ia]));
    for(int ib=0;ib<4;++ib){const int b=act[ib],cc=cp.ref[b];const OperatorReal z=OperatorReal(W[4*ia+ib]);
      if(cc>=0){const unsigned char sl=cp.rowSlot[8*a+b];if(sl!=255)atomicAdd(aval+row[rr]+sl,z);}
      else{const int fs=-cc-1;atomicAdd(cr2+rr,-StateReal(z)*fixed[3*(std::size_t)fs+2]);}
    }
  }
  if(stats){atomicAdd(stats+0,areaW);atomicAdd(stats+1,ut2W);atomicAdd(stats+2,ypW);atomicAdd(stats+3,slipW);h11_atomic_max(stats+4,maxYp);h11_atomic_max(stats+5,maxUt);atomicAdd(stats+6,rootFail);atomicAdd(stats+7,tanFail);}
}


// Gate 5I causal wall test: prescribe an exact reference wall shear while
// retaining the same weak wall entities and transverse clamps.  The target
// kinematic shear is tau_ref=u_tau_ref^2=f_D*Ubulk^2/8.  A lagged Picard
// Robin coefficient beta_ref=tau_ref/max(|U_trace|,slipFloor) is assembled.
// At a converged fixed point beta_ref*U_trace has exactly the prescribed
// traction magnitude.  This is intentionally NOT the Spalding law; it is a
// causal control experiment for wall-vs-bulk error.
__global__ void h11_reference_shear_wall_kernel(
    const H11WallFaceDevice* __restrict__ faces,int nf,
    const G4CellPlanDevice* __restrict__ cells,const std::int64_t* __restrict__ row,
    const StateReal* __restrict__ fixed,const StateReal* __restrict__ u2,
    double nu,double tauRef,double slipFloor,
    OperatorReal* __restrict__ aval,StateReal* __restrict__ cr2,double* __restrict__ stats)
{
  const int fi=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(fi>=nf)return;
  const auto&F=faces[fi];const auto&cp=cells[F.cell];
  int fv[3],kk=0;for(int i=0;i<4;++i)if(i!=(int)F.opp)fv[kk++]=i;
  const int act[4]={fv[0],fv[1],fv[2],4+(int)F.opp};
  double uzc[4];for(int j=0;j<4;++j){const int r=cp.ref[act[j]];uzc[j]=(r>=0)?(double)u2[r]:(double)fixed[3*(std::size_t)(-r-1)+2];}
  double W[16]={0};
  double areaW=0,ut2W=0,ypW=0,slipW=0,maxYp=0,maxUt=0;
  const double ut=sqrt(fmax(tauRef,0.0));
  for(int q=0;q<7;++q){
    const double l0=h11_l0[q],l1=h11_l1[q],l2=h11_l2[q];
    const double phi[4]={l0,l1,l2,27.0*l0*l1*l2};
    double uz=0;for(int j=0;j<4;++j)uz+=uzc[j]*phi[j];
    const double slip=fabs(uz),y=(double)F.y;
    const double beta=tauRef/fmax(slip,slipFloor);
    const double yp=y*ut/nu;
    const double w=h11_w[q]*(double)F.area;
    for(int a=0;a<4;++a)for(int b=0;b<4;++b)W[4*a+b]+=beta*phi[a]*phi[b]*w;
    areaW+=w;ut2W+=tauRef*w;ypW+=yp*w;slipW+=slip*w;maxYp=fmax(maxYp,yp);maxUt=fmax(maxUt,ut);
  }
  for(int ia=0;ia<4;++ia){const int a=act[ia],rr=cp.ref[a];if(rr<0)continue;
    for(int ib=0;ib<4;++ib){const int b=act[ib],cc=cp.ref[b];const OperatorReal z=OperatorReal(W[4*ia+ib]);
      if(cc>=0){const unsigned char sl=cp.rowSlot[8*a+b];if(sl!=255)atomicAdd(aval+row[rr]+sl,z);}
      else{const int fs=-cc-1;atomicAdd(cr2+rr,-StateReal(z)*fixed[3*(std::size_t)fs+2]);}
    }
  }
  if(stats){atomicAdd(stats+0,areaW);atomicAdd(stats+1,ut2W);atomicAdd(stats+2,ypW);atomicAdd(stats+3,slipW);h11_atomic_max(stats+4,maxYp);h11_atomic_max(stats+5,maxUt);}
}

__global__ void h11_clamp_transverse_rhs_state_kernel(
    int n,const std::uint8_t*wall,StateReal*b0,StateReal*b1,StateReal*u0,StateReal*u1){
  const int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n&&wall[i]){b0[i]=StateReal(0);b1[i]=StateReal(0);u0[i]=StateReal(0);u1[i]=StateReal(0);}
}

__global__ void h11_mcgs_color_kernel(
    int count,const std::int32_t*rows,const std::int64_t*rp,const std::int32_t*ci,
    const OperatorReal*av,const OperatorReal*diag,const StateReal*b0,const StateReal*b1,const StateReal*b2,
    StateReal*x0,StateReal*x1,StateReal*x2,OperatorReal omega,const std::uint8_t*wall)
{
  const int q=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(q>=count)return;const int i=rows[q];
  OperatorReal s0=0,s1=0,s2=0;for(std::int64_t k=rp[i];k<rp[i+1];++k){const int j=ci[k];if(j==i)continue;const OperatorReal a=av[k];s0+=a*x0[j];s1+=a*x1[j];s2+=a*x2[j];}
  const OperatorReal iv=OperatorReal(1)/diag[i];
  if(wall&&wall[i]){x0[i]=StateReal(0);x1[i]=StateReal(0);}else{
    const OperatorReal z0=(b0[i]-s0)*iv,z1=(b1[i]-s1)*iv;x0[i]=StateReal((OperatorReal(1)-omega)*x0[i]+omega*z0);x1[i]=StateReal((OperatorReal(1)-omega)*x1[i]+omega*z1);
  }
  const OperatorReal z2=(b2[i]-s2)*iv;x2[i]=StateReal((OperatorReal(1)-omega)*x2[i]+omega*z2);
}

__global__ void h11_mom_norms_kernel(
    int n,const std::int64_t*rp,const std::int32_t*ci,const OperatorReal*av,
    const StateReal*b0,const StateReal*b1,const StateReal*b2,
    const StateReal*x0,const StateReal*x1,const StateReal*x2,const std::uint8_t*wall,double*sums)
{
  const int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i>=n)return;OperatorReal ax0=0,ax1=0,ax2=0;
  for(std::int64_t k=rp[i];k<rp[i+1];++k){const int j=ci[k];const OperatorReal a=av[k];ax0+=a*x0[j];ax1+=a*x1[j];ax2+=a*x2[j];}
  if(!(wall&&wall[i])){const double db0=(double)b0[i],db1=(double)b1[i];const double r0=(double)(b0[i]-ax0),r1=(double)(b1[i]-ax1);atomicAdd(sums+0,db0*db0);atomicAdd(sums+1,db1*db1);atomicAdd(sums+3,r0*r0);atomicAdd(sums+4,r1*r1);}
  const double db2=(double)b2[i],r2=(double)(b2[i]-ax2);atomicAdd(sums+2,db2*db2);atomicAdd(sums+5,r2*r2);
}

} // namespace nodals_gpu
