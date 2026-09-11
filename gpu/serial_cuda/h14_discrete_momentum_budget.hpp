#pragma once
#include <array>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace nodals_gpu {

// Gate 5K: exact discrete z-momentum budget for a smooth interior axial test.
//
// For an axial window [z0,z1], define the P1 test
//   w(z) = sin(pi*(z-z0)/(z1-z0)) inside the window, 0 outside,
// with all BF3 test coefficients set to zero.  This is a valid global FE test
// (zero at window boundaries) and avoids an artificial discontinuous cutoff.
//
// Reported component signs:
//   molecular, nut, wall, dg, advective : LHS/resisting terms
//   pressure                            : RHS/driving B^T p term
//
// Therefore
//   physicalResidual = molecular + nut + wall + dg + advective - pressure.
//
// The "last-linearization" split reproduces the coefficients actually used in
// the final SIMPLE momentum sweep: lagged U before that sweep and the pressure
// that was present before the final pressure correction.  The diagnostic also
// reports the final-pressure B^T p term and the exact final relaxed-system
// residual after the one-sweep momentum work.

struct H14Snapshots {
  DeviceBuffer<StateReal> u0,u1,u2,pMom;
  bool enabled=false;
  void allocate(int nv,int nc){
    u0.allocate(nv);u1.allocate(nv);u2.allocate(nv);pMom.allocate(nc);enabled=true;
  }
  void capture(const G4Gpu&G){
    if(!enabled)return;
    NODALS_CUDA(cudaMemcpyAsync(u0.data(),G.u0.data(),G.u0.bytes(),cudaMemcpyDeviceToDevice));
    NODALS_CUDA(cudaMemcpyAsync(u1.data(),G.u1.data(),G.u1.bytes(),cudaMemcpyDeviceToDevice));
    NODALS_CUDA(cudaMemcpyAsync(u2.data(),G.u2.data(),G.u2.bytes(),cudaMemcpyDeviceToDevice));
    NODALS_CUDA(cudaMemcpyAsync(pMom.data(),G.p.data(),G.p.bytes(),cudaMemcpyDeviceToDevice));
  }
};

__global__ void h14_spmv_minus_rhs_kernel(
    int n,const std::int64_t*rp,const std::int32_t*ci,
    const OperatorReal*A,const StateReal*x,const StateReal*rhs,StateReal*y)
{
  const int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i>=n)return;
  OperatorReal s=0;
  for(std::int64_t k=rp[i];k<rp[i+1];++k)s+=A[k]*x[ci[k]];
  y[i]=StateReal(s)-(rhs?rhs[i]:StateReal(0));
}

__global__ void h14_relaxed_residual_kernel(
    int n,const std::int64_t*rp,const std::int32_t*ci,
    const OperatorReal*Arelaxed,const StateReal*uNew,
    const StateReal*s2,const StateReal*c2,const StateReal*bt,
    const OperatorReal*delta,const StateReal*uOld,StateReal*r)
{
  const int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i>=n)return;
  OperatorReal au=0;
  for(std::int64_t k=rp[i];k<rp[i+1];++k)au+=Arelaxed[k]*uNew[ci[k]];
  r[i]=StateReal(au)-s2[i]-c2[i]-bt[i]-StateReal(delta[i])*uOld[i];
}

__global__ void h14_relax_lag_kernel(
    int n,const OperatorReal*delta,const StateReal*uNew,const StateReal*uOld,
    StateReal*y)
{
  const int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i<n)y[i]=StateReal(delta[i])*(uNew[i]-uOld[i]);
}

inline std::vector<double> h14_download_state(const DeviceBuffer<StateReal>&d){
  std::vector<StateReal> q(d.size()); if(!q.empty())d.download(q.data(),q.size());
  std::vector<double> out(q.size());
  for(std::size_t i=0;i<q.size();++i)out[i]=(double)q[i];
  return out;
}

inline std::vector<double> h14_component_action(
    G4Gpu&G,const OperatorReal*A,const StateReal*rhs,const StateReal*u)
{
  DeviceBuffer<StateReal> r; r.allocate((std::size_t)G.nv);
  h14_spmv_minus_rhs_kernel<<<g4grid(G.nv),G4B>>>(
    G.nv,G.row.data(),G.col.data(),A,u,rhs,r.data());
  NODALS_CUDA(cudaGetLastError());
  return h14_download_state(r);
}

inline void h14_zero_dynamic_component(G4Gpu&G){
  NODALS_CUDA(cudaMemsetAsync(G.av.data(),0,G.av.bytes()));
  NODALS_CUDA(cudaMemsetAsync(G.c0.data(),0,G.c0.bytes()));
  NODALS_CUDA(cudaMemsetAsync(G.c1.data(),0,G.c1.bytes()));
  NODALS_CUDA(cudaMemsetAsync(G.c2.data(),0,G.c2.bytes()));
}

inline void h14_set_u(G4Gpu&G,
    const DeviceBuffer<StateReal>&u0,const DeviceBuffer<StateReal>&u1,
    const DeviceBuffer<StateReal>&u2)
{
  NODALS_CUDA(cudaMemcpyAsync(G.u0.data(),u0.data(),G.u0.bytes(),cudaMemcpyDeviceToDevice));
  NODALS_CUDA(cudaMemcpyAsync(G.u1.data(),u1.data(),G.u1.bytes(),cudaMemcpyDeviceToDevice));
  NODALS_CUDA(cudaMemcpyAsync(G.u2.data(),u2.data(),G.u2.bytes(),cudaMemcpyDeviceToDevice));
}

inline double h14_vertex_window_weight(
    double z,double z0,double z1,double D)
{
  const double x=z/D;
  if(!(x>z0 && x<z1))return 0.0;
  const double xi=(x-z0)/(z1-z0);
  return std::sin(3.141592653589793238462643383279502884*xi);
}

struct H14WindowTest {
  double z0=0,z1=0;
  std::vector<double>w;
  double volume=0.0;
  double wallArea=0.0;
  double norm=0.0;
};

inline H14WindowTest h14_build_window_test(
    const SerialTetMesh&M,const G4SetupHost&S,double z0,double z1)
{
  H14WindowTest T;T.z0=z0;T.z1=z1;T.w.assign((std::size_t)S.topo.n,0.0);
  const int nv=(int)M.points.size();

  std::vector<double>wv((std::size_t)nv,0.0);
  for(int v=0;v<nv;++v){
    const int g=S.momMask.g2free[(std::size_t)v];
    if(g<0)continue;
    const double q=h14_vertex_window_weight(M.points[(std::size_t)v].z,z0,z1,S.pipe.D);
    wv[(std::size_t)v]=q;
    T.w[(std::size_t)g]=q;
  }
  // BF3 coefficients intentionally remain zero: P1 alone represents constants
  // and the smooth axial test.
  for(std::size_t c=0;c<M.tets.size();++c){
    double s=0;for(int i=0;i<4;++i)s+=wv[(std::size_t)M.tets[c][i]];
    T.volume += S.volumes[c]*(s/4.0);
  }
  const auto&P=M.patches[(std::size_t)S.pipe.wall];
  for(int f=P.start_face;f<P.start_face+P.n_faces;++f){
    const auto&F=M.faces[(std::size_t)f];
    const double a=tri_area_g4(
      M.points[(std::size_t)F.v[0]],M.points[(std::size_t)F.v[1]],M.points[(std::size_t)F.v[2]]);
    const double s=wv[(std::size_t)F.v[0]]+wv[(std::size_t)F.v[1]]+wv[(std::size_t)F.v[2]];
    T.wallArea += a*(s/3.0);
  }
  if(!(T.volume>0.0))throw std::runtime_error("Gate5K test volume nonpositive");
  T.norm=2.0*S.pipe.D/(S.pipe.bulk*S.pipe.bulk*T.volume);
  return T;
}

inline double h14_dot_host(const std::vector<double>&w,const std::vector<double>&x){
  if(w.size()!=x.size())throw std::runtime_error("Gate5K host dot size mismatch");
  long double s=0;for(std::size_t i=0;i<w.size();++i)s+=(long double)w[i]*(long double)x[i];
  return (double)s;
}

inline std::vector<double> h14_sub(
    const std::vector<double>&a,const std::vector<double>&b)
{
  if(a.size()!=b.size())throw std::runtime_error("Gate5K vector size mismatch");
  std::vector<double>c(a.size());
  for(std::size_t i=0;i<a.size();++i)c[i]=a[i]-b[i];
  return c;
}

inline std::vector<double> h14_add5(
    const std::vector<double>&a,const std::vector<double>&b,
    const std::vector<double>&c,const std::vector<double>&d,
    const std::vector<double>&e)
{
  const std::size_t n=a.size();
  if(b.size()!=n||c.size()!=n||d.size()!=n||e.size()!=n)
    throw std::runtime_error("Gate5K add size mismatch");
  std::vector<double>r(n);
  for(std::size_t i=0;i<n;++i)r[i]=a[i]+b[i]+c[i]+d[i]+e[i];
  return r;
}

struct H14BudgetVectors {
  std::vector<double> molecular,nut,wall,dg,adv;
  std::vector<double> pressureMom,pressureFinal;
  std::vector<double> physicalLast,relaxLag,relaxedResidual;
  std::vector<double> componentPhysicalLast;
};

inline H14BudgetVectors h14_build_budget_vectors(
    G4Gpu&G,const G4SetupHost&S,const H14Snapshots&snap,
    bool mixingLength,bool dgInlet,bool weakWall,bool supg,
    double mixlenScale,int mixlenBlocks,
    int supgBlocks,double supgTauScale,double supgMagic)
{
  if(!snap.enabled)throw std::runtime_error("Gate5K snapshots disabled");
  if(supg)throw std::runtime_error("Gate5K current diagnostic expects SUPG OFF");

  H14BudgetVectors V;

  // First capture the exact residual of the FINAL RELAXED linear system before
  // mutating any assembly arrays.  The final matrix/rhs were assembled from
  // snap.u* and snap.pMom, while G.u* are the post-sweep solution.
  G.pf.bt_state(snap.pMom.data());
  DeviceBuffer<StateReal> rr; rr.allocate((std::size_t)G.nv);
  h14_relaxed_residual_kernel<<<g4grid(G.nv),G4B>>>(
    G.nv,G.row.data(),G.col.data(),G.av.data(),G.u2.data(),
    G.s2.data(),G.c2.data(),G.pf.v2.data(),G.delta.data(),snap.u2.data(),rr.data());
  NODALS_CUDA(cudaGetLastError());
  V.relaxedResidual=h14_download_state(rr);

  DeviceBuffer<StateReal> lag; lag.allocate((std::size_t)G.nv);
  h14_relax_lag_kernel<<<g4grid(G.nv),G4B>>>(
    G.nv,G.delta.data(),G.u2.data(),snap.u2.data(),lag.data());
  NODALS_CUDA(cudaGetLastError());
  V.relaxLag=h14_download_state(lag);
  V.physicalLast=h14_sub(V.relaxedResidual,V.relaxLag);

  // Keep post-sweep/final velocity safe while component kernels are assembled
  // on the lagged pre-sweep state.
  DeviceBuffer<StateReal> final0,final1,final2;
  final0.allocate((std::size_t)G.nv);final1.allocate((std::size_t)G.nv);final2.allocate((std::size_t)G.nv);
  NODALS_CUDA(cudaMemcpyAsync(final0.data(),G.u0.data(),G.u0.bytes(),cudaMemcpyDeviceToDevice));
  NODALS_CUDA(cudaMemcpyAsync(final1.data(),G.u1.data(),G.u1.bytes(),cudaMemcpyDeviceToDevice));
  NODALS_CUDA(cudaMemcpyAsync(final2.data(),G.u2.data(),G.u2.bytes(),cudaMemcpyDeviceToDevice));

  // Molecular diffusion is static and already separated.
  V.molecular=h14_component_action(G,G.diffusion.data(),G.s2.data(),final2.data());

  // Dynamic components: assemble coefficients/rhs on pre-sweep lagged U, then
  // apply their matrix to the post-sweep final U.
  h14_set_u(G,snap.u0,snap.u1,snap.u2);

  if(mixingLength){
    h14_zero_dynamic_component(G);
    h9_apply_mixlen(G,S.pipe,mixlenScale,mixlenBlocks);
    V.nut=h14_component_action(G,G.av.data(),G.c2.data(),final2.data());
  }else V.nut.assign((std::size_t)G.nv,0.0);

  if(weakWall){
    h14_zero_dynamic_component(G);
    h11_apply_wall(G,S.pipe);
    V.wall=h14_component_action(G,G.av.data(),G.c2.data(),final2.data());
  }else V.wall.assign((std::size_t)G.nv,0.0);

  if(dgInlet){
    h14_zero_dynamic_component(G);
    h10_apply_dg_inlet(G,S.pipe,mixingLength,mixlenScale);
    V.dg=h14_component_action(G,G.av.data(),G.c2.data(),final2.data());
  }else V.dg.assign((std::size_t)G.nv,0.0);

  h14_zero_dynamic_component(G);
  if(G.h7WarpConvection)h7_apply_convection_warp(G,G.h7ConvectionBlocks);
  else h7_apply_convection_scalar(G);
  V.adv=h14_component_action(G,G.av.data(),G.c2.data(),final2.data());

  // Restore final velocity.
  h14_set_u(G,final0,final1,final2);

  // Pressure actually used by final momentum sweep.
  G.pf.bt_state(snap.pMom.data());
  V.pressureMom=h14_download_state(G.pf.v2);

  // Final pressure after the final SIMPLE pressure correction.
  G.pf.bt_state(G.p.data());
  V.pressureFinal=h14_download_state(G.pf.v2);

  const auto resist=h14_add5(V.molecular,V.nut,V.wall,V.dg,V.adv);
  V.componentPhysicalLast=h14_sub(resist,V.pressureMom);
  return V;
}

inline void h14_write_discrete_budget(
    const std::string&path,const char*tag,
    const SerialTetMesh&M,const G4SetupHost&S,G4Gpu&G,
    const H14Snapshots&snap,
    bool mixingLength,bool dgInlet,bool weakWall,bool supg,
    double mixlenScale,int mixlenBlocks,
    int supgBlocks,double supgTauScale,double supgMagic)
{
  if(path.empty())return;
  const auto V=h14_build_budget_vectors(
    G,S,snap,mixingLength,dgInlet,weakWall,supg,
    mixlenScale,mixlenBlocks,supgBlocks,supgTauScale,supgMagic);

  struct W{double a,b;};
  const W win[]={{12,14},{14,16},{16,18},{18,19.8},{12,18}};

  std::ofstream os(path);
  if(!os)throw std::runtime_error("Gate5K cannot open budget CSV: "+path);
  os<<std::setprecision(16)
    <<"z0_over_D,z1_over_D,test_volume,test_wall_area,test_wall_area_over_volume,"
      "f_molecular,f_nut,f_wall,f_dg,f_advective,f_pressure_momentum,"
      "f_pressure_final,f_pressure_correction,f_resist_sum,"
      "f_component_physical_residual,f_direct_physical_residual,"
      "f_relaxation_lag,f_relaxed_linear_residual,component_direct_parity\n";

  for(const auto&q:win){
    const auto T=h14_build_window_test(M,S,q.a,q.b);
    auto F=[&](const std::vector<double>&x){return T.norm*h14_dot_host(T.w,x);};
    const double fm=F(V.molecular),fn=F(V.nut),fw=F(V.wall),fdg=F(V.dg),fa=F(V.adv);
    const double ppm=F(V.pressureMom),ppf=F(V.pressureFinal);
    const double resist=fm+fn+fw+fdg+fa;
    const double comp=resist-ppm;
    const double phys=F(V.physicalLast);
    const double rlag=F(V.relaxLag);
    const double rrel=F(V.relaxedResidual);
    const double parity=std::abs(comp-phys);
    const double pcor=ppf-ppm;
    const double wallGeom=T.wallArea/T.volume;
    const double closureFinal=resist-ppf;

    os<<q.a<<","<<q.b<<","<<T.volume<<","<<T.wallArea<<","<<wallGeom<<","
      <<fm<<","<<fn<<","<<fw<<","<<fdg<<","<<fa<<","<<ppm<<","<<ppf<<","<<pcor<<","
      <<resist<<","<<comp<<","<<phys<<","<<rlag<<","<<rrel<<","<<parity<<"\n";

    std::printf(
      "NODALS_DISCRETE_BUDGET_WINDOW tag=%s z0D=%.2f z1D=%.2f "
      "testVolume=%.12e wallAreaOverVolume=%.12e "
      "fMolecular=%.10f fNuT=%.10f fWall=%.10f fDG=%.10f fAdvective=%.10f "
      "fPressureMomentum=%.10f fPressureFinal=%.10f fPressureCorrection=%.10f "
      "fResistSum=%.10f fComponentPhysicalResidual=%.10f "
      "fDirectPhysicalResidual=%.10f fRelaxationLag=%.10f "
      "fRelaxedLinearResidual=%.10f componentDirectParity=%.3e "
      "fClosureUsingFinalPressure=%.10f status=%s\n",
      tag,q.a,q.b,T.volume,wallGeom,
      fm,fn,fw,fdg,fa,ppm,ppf,pcor,resist,comp,phys,rlag,rrel,parity,closureFinal,
      (std::isfinite(parity)&&parity<5e-6)?"PASS":"FAIL");
  }

  std::printf(
    "NODALS_DISCRETE_BUDGET_OUTPUT tag=%s csv=%s "
    "test=P1_SINE_WINDOW_BF3_ZERO "
    "componentState=FINAL_ITERATION_LAGGED_PRE_SWEEP "
    "matrixAction=POST_SWEEP_FINAL_U "
    "pressureMomentum=PRE_FINAL_PRESSURE_CORRECTION "
    "pressureFinal=POST_FINAL_PRESSURE_CORRECTION "
    "supg=%d status=PASS\n",
    tag,path.c_str(),supg?1:0);
}

} // namespace nodals_gpu
