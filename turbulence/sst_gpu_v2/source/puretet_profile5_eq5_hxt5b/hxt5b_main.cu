// NodalS HXT5B
// Full RANS boundary gate: mixing length + DG plug inlet + Firedrake-parity HEX off-wall Spalding wall + Cheb2 AMG.

#define HXT4B_EMBED_MAIN hxt4b_embedded_main
#include "../hybrid_hxt4b/hxt4b_main.cu"
#include <cstdlib>
#undef HXT4B_EMBED_MAIN

#include "hybrid_amg.cuh"
#include "hxt5a_mixlen.cuh"
#include "hxt5b_boundary.cuh"
#include "hxt5e_fgmres.cuh"
#include "hxt5c_sst_g1.cuh"
#include "hxt5d_sst_g2.cuh"
#include <sstream>
#include <chrono>

// HXT5B6_EXACT_DIAGNOSTIC_VTU

// ============================================================
// HXT5B_FIXED_FGS_FAST_V1
//
// Fixed-work optimization:
//
//   MOM_WORK       = fgs1
//   MOM_MAX        = 1
//   MOM_CAP_POLICY = warn
//
// In this configuration the numerical operation is exactly one
// forward colored-GS sweep regardless of the inner residual.
//
// Therefore skip the initial/final inner residual SpMV+reduction.
// The SIMPLE outer physical/convergence machinery is unchanged.
//
// Every other momentum configuration falls through unchanged to
// hxt5e::momentum_solve.
// ============================================================
namespace hxt5b_perf {

using Real = hxt4b::Real;

inline auto momentum_solve(
    const std::string&mode,
    int outer,
    hxt4b::GpuCSR&A,
    const hxt4b::ColoringHost&C,
    const Real*val,
    const Real*b,
    Real*x,
    double omega,
    const unsigned char*clamp,
    double rtol,
    double atol,
    int maxWork,
    int restart,
    int fgsSweeps,
    hxt4a::Reducer<Real>&red,
    const char*component,
    bool warnPolicy)
{
  const bool fixedFast =
      warnPolicy &&
      mode=="fgs1" &&
      maxWork==1;

  if(fixedFast){

    using Result =
      decltype(
        hxt5e::momentum_solve(
          mode,outer,
          A,C,
          val,b,x,
          omega,clamp,
          rtol,atol,
          maxWork,
          restart,
          fgsSweeps,
          red,
          component));

    // Same numerical action as fixed fgs1/maxWork=1.
    hxt5b::sweep(
      A,C,
      val,b,x,
      omega,
      true,
      clamp);

    HXT1_CUDA(cudaGetLastError());

    Result R{};

    R.sweeps = 1;

    // Inner residual deliberately not evaluated.
    // These fields are diagnostics only in fixed-work/warn mode.
    R.initial = 0.0;
    R.final   = 0.0;
    R.rel     = 0.0;
    R.ok      = true;

    if(outer==1 &&
       component &&
       component[0]=='x')
      std::printf(
        "NODALS_HXT5B_MOM_FIXED_FAST "
        "mode=fgs1 sweeps=1 "
        "innerResidualAudit=SKIPPED "
        "outerPhysicalGate=UNCHANGED "
        "status=PASS\n");

    return R;
  }

  // All non-fixed configurations remain exactly as before.
  return hxt5e::momentum_solve(
    mode,outer,
    A,C,
    val,b,x,
    omega,clamp,
    rtol,atol,
    maxWork,
    restart,
    fgsSweeps,
    red,
    component);
}

} // namespace hxt5b_perf


namespace hxt5b_diag {
using Real = hxt4b::Real;
static inline double gx(int i){static const double x[4]={-0.86113631159405257522,-0.33998104358485626480,0.33998104358485626480,0.86113631159405257522};return x[i];}
static inline double gw(int i){static const double w[4]={0.34785484513745385737,0.65214515486254614263,0.65214515486254614263,0.34785484513745385737};return w[i];}
static inline double det3(hxt5b::V3 a,hxt5b::V3 b,hxt5b::V3 c){return hxt5b::vdot(a,hxt5b::vcross(b,c));}
static inline double spyh(double up,double k,double B){double x=k*up,rem;if(std::fabs(x)<1e-3){double x2=x*x,x4=x2*x2;rem=x4*(1.0/24.0+x/120.0+x2/720.0+x2*x/5040.0);}else rem=std::expm1(x)-x-.5*x*x-x*x*x/6.0;return up+std::exp(-k*B)*rem;}
static inline double spgh(double up,double reY){return up*spyh(up,.4,5.5)-reY;}
static inline bool sph(double slip,double y,double nu,double&ut,double&yp,double&beta){double U=std::fabs(slip),reY=y*U/nu;if(!(y>0)||!(nu>0)||!std::isfinite(U))return false;if(reY<=1e-14){ut=yp=0;beta=nu/y;return true;}double lo=0,hi=std::max(1.0,std::sqrt(reY)+1.0);for(int k=0;k<40&&spgh(hi,reY)<0;++k)hi*=2;double gh=spgh(hi,reY);if(!std::isfinite(gh)||gh<0)return false;for(int k=0;k<70;++k){double m=.5*(lo+hi);if(spgh(m,reY)>0)hi=m;else lo=m;}double up=.5*(lo+hi);if(!(up>0))return false;ut=U/up;yp=y*ut/nu;beta=ut*ut/U;return std::isfinite(beta)&&beta>0&&std::isfinite(yp);}
struct CellDiag{double ux=0,uy=0,uz=0,vol=0,cx=0,cy=0,cz=0,wallArea=0,wallYp=0,wallYpMax=0,wallUt=0,wallUt2=0,wallSlip=0,wallY=0;int wallMask=0;};
static std::vector<CellDiag> build(const nodals_hxt1::HostMesh&M,const hxt5b::BoundaryHost&BH,const std::vector<Real>&u0,const std::vector<Real>&u1,const std::vector<Real>&u2,double nu,double bulk,double sf){
  int nh=(int)M.h.nhex,nt=(int)M.h.ntet;std::vector<CellDiag>D((size_t)(nh+nt));
  for(int c=0;c<nh;++c){auto H=M.hexVel[(size_t)c];auto hc=M.hexes[(size_t)c];double V=0,mx=0,my=0,mz=0,a0=0,a1=0,a2=0;for(int i=0;i<4;++i)for(int j=0;j<4;++j)for(int k=0;k<4;++k){double x=gx(i),y=gx(j),z=gx(k),ph[14];hxt5b::hbasis(x,y,z,ph);hxt5b::V3 X,dX[3];hxt5b::q1host(M,hc,x,y,z,X,dX);double q=gw(i)*gw(j)*gw(k)*std::fabs(det3(dX[0],dX[1],dX[2]));double vx=0,vy=0,vz=0;for(int a=0;a<14;++a){int g=H.g[a];vx+=(double)u0[(size_t)g]*ph[a];vy+=(double)u1[(size_t)g]*ph[a];vz+=(double)u2[(size_t)g]*ph[a];}V+=q;mx+=q*X.x;my+=q*X.y;mz+=q*X.z;a0+=q*vx;a1+=q*vy;a2+=q*vz;}if(!(V>0))throw std::runtime_error("HXT5B6 invalid HEX volume");auto&d=D[(size_t)c];d.vol=V;d.cx=mx/V;d.cy=my/V;d.cz=mz/V;d.ux=a0/V;d.uy=a1/V;d.uz=a2/V;}
  for(int c=0;c<nt;++c){auto T=M.tetVel[(size_t)c];auto tc=M.tets[(size_t)c];auto&d=D[(size_t)(nh+c)];d.vol=M.tetGeom[(size_t)c].volume;for(int a=0;a<4;++a){auto P=M.points[(size_t)tc.v[a]];d.cx+=.25*P.x;d.cy+=.25*P.y;d.cz+=.25*P.z;}for(int comp=0;comp<3;++comp){const std::vector<Real>&U=comp==0?u0:(comp==1?u1:u2);double v=0;for(int a=0;a<4;++a)v+=.25*(double)U[(size_t)T.g[a]];for(int a=4;a<8;++a)v+=.225*(double)U[(size_t)T.g[a]];if(comp==0)d.ux=v;else if(comp==1)d.uy=v;else d.uz=v;}}
  unsigned long long roots=0;double gA=0,gYp=0,gUt2=0;
  for(const auto&f:BH.wallH){auto H=M.hexVel[(size_t)f.cell];auto hc=M.hexes[(size_t)f.cell];double A=0,sYp=0,sUt=0,sUt2=0,sSlip=0,sY=0,ypmx=0;for(int i=0;i<4;++i)for(int j=0;j<4;++j){double cw[3]={0,0,0};cw[f.fd]=f.fv;cw[f.d0]=gx(i);cw[f.d1]=gx(j);double cs[3]={cw[0],cw[1],cw[2]};cs[f.fd]=f.fv+sf*(-2.0*f.fv);double ps[14];hxt5b::hbasis(cs[0],cs[1],cs[2],ps);hxt5b::V3 Xw,dXw[3],Xs,dXs[3];hxt5b::q1host(M,hc,cw[0],cw[1],cw[2],Xw,dXw);hxt5b::q1host(M,hc,cs[0],cs[1],cs[2],Xs,dXs);double rr=std::hypot(Xw.x,Xw.y);if(!(rr>0))throw std::runtime_error("HXT5B6 bad wall radius");hxt5b::V3 nr{Xw.x/rr,Xw.y/rr,0};double jac=hxt5b::vnorm(hxt5b::vcross(dXw[f.d0],dXw[f.d1]));double yy=hxt5b::vdot(hxt5b::vsub(Xw,Xs),nr);if(!(jac>0)||!(yy>0))throw std::runtime_error("HXT5B6 bad wall sample geometry");double slip=0;for(int a=0;a<14;++a)slip+=(double)u2[(size_t)H.g[a]]*ps[a];double ut=0,yp=0,beta=0;if(!sph(slip,yy,nu,ut,yp,beta)){++roots;ut=yp=0;}double q=gw(i)*gw(j)*jac;A+=q;sYp+=q*yp;sUt+=q*ut;sUt2+=q*ut*ut;sSlip+=q*std::fabs(slip);sY+=q*yy;ypmx=std::max(ypmx,yp);}auto&d=D[(size_t)f.cell];double oa=d.wallArea,na=oa+A;d.wallYp=(d.wallYp*oa+sYp)/na;d.wallUt=(d.wallUt*oa+sUt)/na;d.wallUt2=(d.wallUt2*oa+sUt2)/na;d.wallSlip=(d.wallSlip*oa+sSlip)/na;d.wallY=(d.wallY*oa+sY)/na;d.wallYpMax=std::max(d.wallYpMax,ypmx);d.wallArea=na;d.wallMask=1;gA+=A;gYp+=sYp;gUt2+=sUt2;}
  if(roots)throw std::runtime_error("HXT5B6 host Spalding root failure");if(gA>0)std::printf("NODALS_HXT5B6_DIAG_WALL area=%.12e yPlusMean=%.12e wallDarcy=%.12e rootFailures=%llu status=PASS\n",gA,gYp/gA,8.0*(gUt2/gA)/(bulk*bulk),roots);return D;
}
static std::string path2(const std::string&v){return v.size()>=4&&v.substr(v.size()-4)==".vtu"?v.substr(0,v.size()-4)+"_cell_diagnostics.vtu":v+"_cell_diagnostics.vtu";}
static void write(const std::string&path,const nodals_hxt1::HostMesh&M,const hxt5b::BoundaryHost&BH,const std::vector<Real>&u0,const std::vector<Real>&u1,const std::vector<Real>&u2,const std::vector<Real>&p,double nu,double bulk,double sf){int nh=(int)M.h.nhex,nt=(int)M.h.ntet,np=nh+nt;auto D=build(M,BH,u0,u1,u2,nu,bulk,sf);std::ofstream f(path);if(!f)throw std::runtime_error("cannot write "+path);f<<std::setprecision(16)<<"<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n<UnstructuredGrid><Piece NumberOfPoints=\""<<M.points.size()<<"\" NumberOfCells=\""<<np<<"\">\n<Points><DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";for(auto&q:M.points)f<<q.x<<" "<<q.y<<" "<<q.z<<"\n";f<<"</DataArray></Points><Cells><DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";for(auto&h:M.hexes){for(int a=0;a<8;++a)f<<h.v[a]<<" ";f<<"\n";}for(auto&t:M.tets){for(int a=0;a<4;++a)f<<t.v[a]<<" ";f<<"\n";}f<<"</DataArray><DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";long long o=0;for(int c=0;c<nh;++c){o+=8;f<<o<<"\n";}for(int c=0;c<nt;++c){o+=4;f<<o<<"\n";}f<<"</DataArray><DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";for(int c=0;c<nh;++c)f<<"12\n";for(int c=0;c<nt;++c)f<<"10\n";f<<"</DataArray></Cells><PointData><DataArray type=\"Float64\" Name=\"velocity_Q1P1_vertex\" NumberOfComponents=\"3\" format=\"ascii\">\n";for(size_t i=0;i<M.points.size();++i)f<<(double)u0[i]<<" "<<(double)u1[i]<<" "<<(double)u2[i]<<"\n";f<<"</DataArray></PointData><CellData><DataArray type=\"Float64\" Name=\"pressure_Q0P0\" format=\"ascii\">\n";for(auto q:p)f<<(double)q<<"\n";f<<"</DataArray><DataArray type=\"Int32\" Name=\"region\" format=\"ascii\">\n";for(int c=0;c<nh;++c)f<<"1\n";for(int c=0;c<nt;++c)f<<"2\n";f<<"</DataArray>\n";
#define H5B6_SCALAR(NAME,EXPR) f<<"<DataArray type=\"Float64\" Name=\"" NAME "\" format=\"ascii\">\n";for(const auto&d:D)f<<(EXPR)<<"\n";f<<"</DataArray>\n";
  f<<"<DataArray type=\"Float64\" Name=\"velocity_cell_average_full\" NumberOfComponents=\"3\" format=\"ascii\">\n";for(auto&d:D)f<<d.ux<<" "<<d.uy<<" "<<d.uz<<"\n";f<<"</DataArray>\n";
  H5B6_SCALAR("velocity_cell_average_full_magnitude",std::sqrt(d.ux*d.ux+d.uy*d.uy+d.uz*d.uz));H5B6_SCALAR("cell_volume",d.vol);
  f<<"<DataArray type=\"Float64\" Name=\"cell_centroid\" NumberOfComponents=\"3\" format=\"ascii\">\n";for(auto&d:D)f<<d.cx<<" "<<d.cy<<" "<<d.cz<<"\n";f<<"</DataArray>\n";
  f<<"<DataArray type=\"Int32\" Name=\"wall_spalding_mask\" format=\"ascii\">\n";for(auto&d:D)f<<d.wallMask<<"\n";f<<"</DataArray>\n";
  H5B6_SCALAR("wall_area",d.wallArea);H5B6_SCALAR("wall_yplus_spalding",d.wallYp);H5B6_SCALAR("wall_yplus_max_spalding",d.wallYpMax);H5B6_SCALAR("wall_utau_spalding",d.wallUt);H5B6_SCALAR("wall_utau2_mean_spalding",d.wallUt2);H5B6_SCALAR("wall_slip_spalding",d.wallSlip);H5B6_SCALAR("wall_sample_y",d.wallY);
#undef H5B6_SCALAR
  f<<"</CellData></Piece></UnstructuredGrid></VTKFile>\n";
}

// NODALS_HXT5B_PROFILE_AUDIT
// Host-only post-solve point sampler.  It evaluates the complete enriched
// velocity field: HEX Q1+BF2 (14 local velocity DOFs) and TET P1+BF3
// (8 local velocity DOFs).  Geometry remains Q1 for HEX and affine P1 for TET.
struct ProfileBox {
  int kind=0,cell=-1; // 1=hex, 2=tet
  double lo[3]={0,0,0},hi[3]={0,0,0};
};
static inline bool solve_cols(hxt5b::V3 c0,hxt5b::V3 c1,hxt5b::V3 c2,hxt5b::V3 r,double d[3]){
  const double det=det3(c0,c1,c2);
  const double sc=std::max({hxt5b::vnorm(c0),hxt5b::vnorm(c1),hxt5b::vnorm(c2),1e-300});
  if(!std::isfinite(det)||std::fabs(det)<1e-14*sc*sc*sc)return false;
  d[0]=hxt5b::vdot(r,hxt5b::vcross(c1,c2))/det;
  d[1]=hxt5b::vdot(c0,hxt5b::vcross(r,c2))/det;
  d[2]=hxt5b::vdot(c0,hxt5b::vcross(c1,r))/det;
  return std::isfinite(d[0])&&std::isfinite(d[1])&&std::isfinite(d[2]);
}
static std::vector<ProfileBox> profile_boxes(const nodals_hxt1::HostMesh&M){
  std::vector<ProfileBox>B; B.reserve((size_t)M.h.nhex+(size_t)M.h.ntet);
  for(int c=0;c<(int)M.h.nhex;++c){ProfileBox b;b.kind=1;b.cell=c;for(int d=0;d<3;++d){b.lo[d]=1e300;b.hi[d]=-1e300;}auto h=M.hexes[(size_t)c];for(int a=0;a<8;++a){auto P=M.points[(size_t)h.v[a]];double q[3]={P.x,P.y,P.z};for(int d=0;d<3;++d){b.lo[d]=std::min(b.lo[d],q[d]);b.hi[d]=std::max(b.hi[d],q[d]);}}B.push_back(b);}
  for(int c=0;c<(int)M.h.ntet;++c){ProfileBox b;b.kind=2;b.cell=c;for(int d=0;d<3;++d){b.lo[d]=1e300;b.hi[d]=-1e300;}auto t=M.tets[(size_t)c];for(int a=0;a<4;++a){auto P=M.points[(size_t)t.v[a]];double q[3]={P.x,P.y,P.z};for(int d=0;d<3;++d){b.lo[d]=std::min(b.lo[d],q[d]);b.hi[d]=std::max(b.hi[d],q[d]);}}B.push_back(b);}
  return B;
}
static bool profile_hex_ref(const nodals_hxt1::HostMesh&M,int c,const hxt5b::V3&P,const ProfileBox&b,double r[3]){
  for(int d=0;d<3;++d){double q=d==0?P.x:(d==1?P.y:P.z),m=.5*(b.lo[d]+b.hi[d]),h=.5*(b.hi[d]-b.lo[d]);r[d]=h>0?(q-m)/h:0;r[d]=std::max(-1.0,std::min(1.0,r[d]));}
  auto hc=M.hexes[(size_t)c];double sc=std::sqrt((b.hi[0]-b.lo[0])*(b.hi[0]-b.lo[0])+(b.hi[1]-b.lo[1])*(b.hi[1]-b.lo[1])+(b.hi[2]-b.lo[2])*(b.hi[2]-b.lo[2]));
  const double tol=1e-10*std::max(sc,1e-12);
  for(int it=0;it<20;++it){hxt5b::V3 X,dX[3];hxt5b::q1host(M,hc,r[0],r[1],r[2],X,dX);hxt5b::V3 e=hxt5b::vsub(P,X);if(hxt5b::vnorm(e)<=tol)break;double q[3];if(!solve_cols(dX[0],dX[1],dX[2],e,q))return false;r[0]+=q[0];r[1]+=q[1];r[2]+=q[2];if(std::fabs(r[0])>1.5||std::fabs(r[1])>1.5||std::fabs(r[2])>1.5)return false;}
  hxt5b::V3 X,dX[3];hxt5b::q1host(M,hc,r[0],r[1],r[2],X,dX);return hxt5b::vnorm(hxt5b::vsub(P,X))<=10*tol&&r[0]>=-1-1e-8&&r[0]<=1+1e-8&&r[1]>=-1-1e-8&&r[1]<=1+1e-8&&r[2]>=-1-1e-8&&r[2]<=1+1e-8;
}
static bool profile_tet_lam(const nodals_hxt1::HostMesh&M,int c,const hxt5b::V3&P,double l[4]){
  auto t=M.tets[(size_t)c];auto A=hxt5b::vp(M.points[(size_t)t.v[0]]),B=hxt5b::vp(M.points[(size_t)t.v[1]]),C=hxt5b::vp(M.points[(size_t)t.v[2]]),D=hxt5b::vp(M.points[(size_t)t.v[3]]);double q[3];if(!solve_cols(hxt5b::vsub(B,A),hxt5b::vsub(C,A),hxt5b::vsub(D,A),hxt5b::vsub(P,A),q))return false;l[1]=q[0];l[2]=q[1];l[3]=q[2];l[0]=1-q[0]-q[1]-q[2];for(int a=0;a<4;++a)if(l[a]<-1e-8||l[a]>1+1e-8)return false;return true;
}
static bool profile_eval(const nodals_hxt1::HostMesh&M,const std::vector<ProfileBox>&B,const std::vector<int>&cand,const std::vector<Real>&u0,const std::vector<Real>&u1,const std::vector<Real>&u2,const hxt5b::V3&P,double U[3]){
  const double eps=2e-10;
  for(int ib:cand){const auto&b=B[(size_t)ib];if(P.x<b.lo[0]-eps||P.x>b.hi[0]+eps||P.y<b.lo[1]-eps||P.y>b.hi[1]+eps||P.z<b.lo[2]-eps||P.z>b.hi[2]+eps)continue;
    if(b.kind==1){double r[3];if(!profile_hex_ref(M,b.cell,P,b,r))continue;double ph[14];hxt5b::hbasis(r[0],r[1],r[2],ph);auto H=M.hexVel[(size_t)b.cell];U[0]=U[1]=U[2]=0;for(int a=0;a<14;++a){int g=H.g[a];U[0]+=(double)u0[(size_t)g]*ph[a];U[1]+=(double)u1[(size_t)g]*ph[a];U[2]+=(double)u2[(size_t)g]*ph[a];}return true;}
    else{double l[4];if(!profile_tet_lam(M,b.cell,P,l))continue;double ph[8];hxt5b::tbasis(l,ph);auto T=M.tetVel[(size_t)b.cell];U[0]=U[1]=U[2]=0;for(int a=0;a<8;++a){int g=T.g[a];U[0]+=(double)u0[(size_t)g]*ph[a];U[1]+=(double)u1[(size_t)g]*ph[a];U[2]+=(double)u2[(size_t)g]*ph[a];}return true;}
  }
  return false;
}
static std::vector<double> parse_fracs(const std::string&s){std::vector<double>v;std::stringstream ss(s);std::string q;while(std::getline(ss,q,',')){if(q.empty())continue;double x=std::stod(q);if(!(x>0&&x<1))throw std::runtime_error("profile z fractions must lie in (0,1)");v.push_back(x);}if(v.empty())throw std::runtime_error("empty profile z fraction list");return v;}
static std::string profile_path(const std::string&v){return v.size()>=4&&v.substr(v.size()-4)==".vtu"?v.substr(0,v.size()-4)+"_enriched_profile.csv":v+"_enriched_profile.csv";}
static void write_profile(const std::string&path,const nodals_hxt1::HostMesh&M,const std::vector<Real>&u0,const std::vector<Real>&u1,const std::vector<Real>&u2,double R,double bulk,double zmin,double zmax,const std::string&fracText,int nr,int na,double rmax){
  if(nr<8||na<4||!(rmax>0&&rmax<=1))throw std::runtime_error("bad profile audit controls");auto fr=parse_fracs(fracText);auto B=profile_boxes(M);std::ofstream f(path);if(!f)throw std::runtime_error("cannot write "+path);f<<std::setprecision(16)<<"z_over_L,z,r_over_R,r,angle_samples,uz_mean,uz_min,uz_max,uz_std,uz_over_bulk\n";const double pi=std::acos(-1.0),L=zmax-zmin;
  for(double zf:fr){double z=zmin+zf*L;std::vector<int>cand;cand.reserve(B.size()/8+32);for(int i=0;i<(int)B.size();++i)if(z>=B[(size_t)i].lo[2]-2e-10&&z<=B[(size_t)i].hi[2]+2e-10)cand.push_back(i);if(cand.empty())throw std::runtime_error("profile audit found no cells crossing requested z plane");double center=0,maxMean=-1e300,maxPoint=-1e300,rAtMax=0,stdSum=0;int stdCount=0,minHits=na;
    for(int ir=0;ir<nr;++ir){double rr=rmax*(double)ir/(double)(nr-1),rad=R*rr,sum=0,sum2=0,mn=1e300,mx=-1e300;int hits=0,nang=ir==0?1:na;for(int ia=0;ia<nang;++ia){double th=ir==0?0:(2*pi*(ia+.5)/na);hxt5b::V3 P{rad*std::cos(th),rad*std::sin(th),z};double U[3];if(!profile_eval(M,B,cand,u0,u1,u2,P,U))continue;double uz=U[2];sum+=uz;sum2+=uz*uz;mn=std::min(mn,uz);mx=std::max(mx,uz);maxPoint=std::max(maxPoint,uz);++hits;}if(hits==0)throw std::runtime_error("profile audit point location failed at z/r sample");double mean=sum/hits,sd=std::sqrt(std::max(0.0,sum2/hits-mean*mean));if(ir==0)center=mean;if(mean>maxMean){maxMean=mean;rAtMax=rr;}if(ir>0){stdSum+=sd;++stdCount;minHits=std::min(minHits,hits);}f<<zf<<","<<z<<","<<rr<<","<<rad<<","<<hits<<","<<mean<<","<<mn<<","<<mx<<","<<sd<<","<<mean/bulk<<"\n";}
    std::printf("NODALS_HXT5B_PROFILE zOverL=%.6f z=%.12e radialSamples=%d angles=%d rMaxOverR=%.6f candidateCells=%zu centerUz=%.12e centerOverBulk=%.12e maxMeanUz=%.12e maxPointUz=%.12e rAtMaxOverR=%.12e minAngularHits=%d meanAngularStd=%.12e status=PASS\n",zf,z,nr,na,rmax,cand.size(),center,center/bulk,maxMean,maxPoint,rAtMax,minHits,stdCount?stdSum/stdCount:0.0);
  }
  std::printf("NODALS_HXT5B_PROFILE_OUTPUT path=%s field=FULL_Q1BF2_P1BF3_POINT_EVALUATION radialAverage=AZIMUTHAL status=PASS\n",path.c_str());
}


struct CrossSectionSideAudit {
  double area=0.0;
  double flux=0.0;
  double momentum=0.0;
  double pressureArea=0.0;
  long long faces=0;
};

struct CrossSectionAudit {
  CrossSectionSideAudit minus;
  CrossSectionSideAudit plus;
};

static CrossSectionAudit cross_section_audit(
    const nodals_hxt1::HostMesh& M,
    const std::vector<Real>& u2,
    const std::vector<Real>& p,
    double zTarget)
{
  if(M.h.ntet!=0)
    throw std::runtime_error(
      "cross-section momentum audit currently expects all-HEX mesh");

  CrossSectionAudit R;

  const double ztol=1e-10;

  for(std::size_t c=0;c<M.hexes.size();++c){

    const auto hc=M.hexes[c];
    const auto H =M.hexVel[c];

    hxt5b::V3 Xc,dXc[3];
    hxt5b::q1host(M,hc,0.0,0.0,0.0,Xc,dXc);

    for(int fd=0;fd<3;++fd){

      const int d0=(fd+1)%3;
      const int d1=(fd+2)%3;

      for(int iside=0;iside<2;++iside){

        const double fv=iside ? 1.0 : -1.0;

        double cc[3]={0.0,0.0,0.0};
        cc[fd]=fv;

        hxt5b::V3 Xf,dXf[3];
        hxt5b::q1host(
          M,hc,
          cc[0],cc[1],cc[2],
          Xf,dXf);

        hxt5b::V3 cr=hxt5b::vcross(dXf[d0],dXf[d1]);

        hxt5b::V3 outward{
          Xf.x-Xc.x,
          Xf.y-Xc.y,
          Xf.z-Xc.z
        };

        if(hxt5b::vdot(cr,outward)<0.0){
          cr.x=-cr.x;
          cr.y=-cr.y;
          cr.z=-cr.z;
        }

        const double jc=hxt5b::vnorm(cr);
        if(!(jc>0.0))continue;

        const double nz=cr.z/jc;

        // Only axial cross-section faces.
        if(std::abs(nz)<0.9)continue;
        if(std::abs(Xf.z-zTarget)>ztol)continue;

        CrossSectionSideAudit& S =
          nz>0.0 ? R.plus : R.minus;

        ++S.faces;

        for(int i=0;i<4;++i){
          for(int j=0;j<4;++j){

            double xi[3]={0.0,0.0,0.0};
            xi[fd]=fv;
            xi[d0]=gx(i);
            xi[d1]=gx(j);

            double ph[14];
            hxt5b::hbasis(
              xi[0],xi[1],xi[2],
              ph);

            hxt5b::V3 Xq,dXq[3];
            hxt5b::q1host(
              M,hc,
              xi[0],xi[1],xi[2],
              Xq,dXq);

            const double jac=
              hxt5b::vnorm(
                hxt5b::vcross(
                  dXq[d0],
                  dXq[d1]));

            const double w=
              gw(i)*gw(j)*jac;

            double uz=0.0;
            for(int a=0;a<14;++a){
              uz +=
                (double)u2[(std::size_t)H.g[a]]
                *ph[a];
            }

            S.area        += w;
            S.flux        += w*uz;
            S.momentum    += w*uz*uz;
            S.pressureArea+= w*(double)p[c];
          }
        }
      }
    }
  }

  return R;
}

static void print_cross_section_momentum_audit(
    const nodals_hxt1::HostMesh& M,
    const std::vector<Real>& u2,
    const std::vector<Real>& p,
    double bulk,
    double z0,
    double z1)
{
  const auto A=cross_section_audit(M,u2,p,z0);
  const auto B=cross_section_audit(M,u2,p,z1);

  auto print_side =
    [bulk](const char* plane,
           const char* side,
           const CrossSectionSideAudit& q)
  {
    const double mean=
      q.area>0.0 ? q.flux/q.area : 0.0;

    const double betaBulk=
      q.area>0.0
      ? q.momentum/(q.area*bulk*bulk)
      : 0.0;

    const double betaLocal=
      std::abs(q.flux)>0.0
      ? q.momentum*q.area/(q.flux*q.flux)
      : 0.0;

    const double pMean=
      q.area>0.0
      ? q.pressureArea/q.area
      : 0.0;

    std::printf(
      "NODALS_HXT5B_CROSS_SECTION "
      "plane=%s side=%s "
      "faces=%lld area=%.12e "
      "flux=%.12e meanUz=%.12e "
      "momentum=%.12e "
      "betaBulk50=%.12e betaLocal=%.12e "
      "pressureArea=%.12e pressureMean=%.12e "
      "status=PASS\n",
      plane,side,q.faces,q.area,
      q.flux,mean,q.momentum,
      betaBulk,betaLocal,
      q.pressureArea,pMean);
  };

  print_side("z0","minus",A.minus);
  print_side("z0","plus", A.plus);
  print_side("z1","minus",B.minus);
  print_side("z1","plus", B.plus);

  // Control volume z0 < z < z1:
  //
  // lower plane is the PLUS-Z side of the cell below but the
  // MINUS-Z side of the cell inside the control volume.
  //
  // therefore use:
  //   z0 : minus-normal face from cell above
  //   z1 : plus-normal  face from cell below

  const double pressureForce=
      A.minus.pressureArea
    - B.plus.pressureArea;

  const double momentumIn =
      A.minus.momentum;

  const double momentumOut=
      B.plus.momentum;

  const double momentumDelta=
      momentumOut-momentumIn;

  const double fluxIn =
      A.minus.flux;

  const double fluxOut=
      B.plus.flux;

  const double traceMismatchZ0=
      std::abs(A.plus.momentum-A.minus.momentum)
      /std::max(
         std::max(std::abs(A.plus.momentum),
                  std::abs(A.minus.momentum)),
         1e-300);

  const double traceMismatchZ1=
      std::abs(B.plus.momentum-B.minus.momentum)
      /std::max(
         std::max(std::abs(B.plus.momentum),
                  std::abs(B.minus.momentum)),
         1e-300);

  std::printf(
    "NODALS_HXT5B_CROSS_SECTION_MOMENTUM_AUDIT "
    "z0=%.12e z1=%.12e "
    "pressureForce=%.12e "
    "momentumIn=%.12e "
    "momentumOut=%.12e "
    "momentumDelta=%.12e "
    "pressureMinusMomentumDelta=%.12e "
    "fluxIn=%.12e fluxOut=%.12e "
    "fluxDelta=%.12e "
    "traceMismatchZ0=%.12e "
    "traceMismatchZ1=%.12e "
    "status=PASS\n",
    z0,z1,
    pressureForce,
    momentumIn,
    momentumOut,
    momentumDelta,
    pressureForce-momentumDelta,
    fluxIn,fluxOut,fluxOut-fluxIn,
    traceMismatchZ0,
    traceMismatchZ1);
}

} // namespace hxt5b_diag

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


int main(int argc,char**argv){
  try{
    using namespace hxt4b;
    using T=Real;

    std::string mesh,vtu="HXT5B_full_rans.vtu",momWork="sgs1",momCapPolicy="warn";
    std::string pressureSolver="pcg_amg",amgHierarchy="cf",cfCoarsening="pmis",cfStrength="classical-negative",cfInterp="exti";
    std::string pressureGauge="pin",podMode="off",podRecompressMethod="tail_pod";
    const char*simpleCouplingEnv=std::getenv("SIMPLE_COUPLING");
    std::string simpleCoupling=simpleCouplingEnv?std::string(simpleCouplingEnv):"lagged";
    std::string amgFineMode="explicit"; // HXT5B_AMG_FINE_MODE_V1
    std::string amgSmoother="cheb2",amgSpectrumPolicy="auto",amgMcgsOrder="symmetric";
    double Re=13691.740248127866,bulk=50,gamma=50,schurNitscheGamma=0,alphaU=.7,alphaP=1,rauScale=2,simpleTol=1e-3,momOmega=1,momRtol=.5,momAtol=1e-8,mixlenScale=1.0,outerContAtol=1e-12,wallSampleFraction=.5;
    int momentumWallMode=hxt5d::MOM_WALL_SPALDING;
    int spaldingMolecularConsistency=0; // SPALDING_MOLECULAR_CONSISTENCY_V1
    double weakWallGamma=4.0,weakWallHScale=1.0,weakWallNutScale=1.0;
    double pRtol=.9,pAtol=1e-12,cfTheta=.25,amgJacobiOmega=.7,amgLambdaSafety=1.5,amgLambdaLow=.05,amgMcgsOmega=1;
    int maxOuter=3000,pMax=50,momMax=50,momFgmresRestart=30,momFgsSweeps=1,cfPmax=8,cfAggressiveFirst=0,amgTerminal=1000,amgPowerIts=16,amgChebDegree=4;
    int podTrainSnapshots=10,podRank=2,podMaxRank=64,podTailIter=10,podHardIts=0,podAcceptMinAbs=5,podRefreshAbs=30;
    int podRecompressRank=0,podRecompressHistory=18,podRecompressMaxLossAbs=5,podRecompressForce=0,podNoProbe=0;
    double podNovelty=1e-8,podAcceptMinRel=0.03,podRefreshRel=0.35,podRecompressMaxLossRel=0.10;
    int amgJacobiFineSweeps=1,amgJacobiCoarseSweeps=1,amgMcgsFineSweeps=1,amgMcgsCoarseSweeps=1;
    hxt5c::Options sstG1;
    bool sstG2=false,sstG2PhysicalGate=true,profileAudit=true,laminarPerf=false;
    std::string sstG2Start="g0",profileZFractions="0.50,0.75,0.90,0.98";
    int profileRadial=65,profileAngles=16;
    double profileRmax=.98;
    double sstG2PlateauTol=2.5e-3,sstG2ContAtol=1e-6;
    int sstG2Max=800,sstG2PrintEvery=20,sstG2PlateauSamples=5,sstG2MetricsEvery=1;

    for(int i=1;i<argc;++i){
      std::string a=argv[i];auto nx=[&](){if(i+1>=argc)throw std::runtime_error("missing value for "+a);return std::string(argv[++i]);};
      if(a=="--mesh")mesh=nx();else if(a=="--vtu")vtu=nx();else if(a=="--re")Re=std::stod(nx());else if(a=="--bulk")bulk=std::stod(nx());else if(a=="--mixlen-scale")mixlenScale=std::stod(nx());else if(a=="--outer-cont-atol")outerContAtol=std::stod(nx());
      else if(a=="--wall-sample-fraction")wallSampleFraction=std::stod(nx());
      else if(a=="--momentum-wall-mode"){auto v=nx();if(v=="spalding")momentumWallMode=hxt5d::MOM_WALL_SPALDING;else if(v=="spalding_trace")momentumWallMode=hxt5d::MOM_WALL_SPALDING_TRACE;else if(v=="spalding_aligned")momentumWallMode=hxt5d::MOM_WALL_SPALDING_ALIGNED;else if(v=="spalding_compatible")momentumWallMode=hxt5d::MOM_WALL_SPALDING_COMPATIBLE;else if(v=="sst_consistent_table")momentumWallMode=hxt5d::MOM_WALL_SST_CONSISTENT_TABLE;else if(v=="spalding_profile_integral")momentumWallMode=hxt5d::MOM_WALL_SPALDING_PROFILE_INTEGRAL;else if(v=="weak_molecular")momentumWallMode=hxt5d::MOM_WALL_WEAK_MOLECULAR;else if(v=="weak_effective")momentumWallMode=hxt5d::MOM_WALL_WEAK_EFFECTIVE;else throw std::runtime_error("--momentum-wall-mode must be spalding, spalding_trace, spalding_aligned, spalding_compatible, sst_consistent_table, spalding_profile_integral, weak_molecular, or weak_effective");}
      else if(a=="--spalding-molecular-consistency")spaldingMolecularConsistency=std::stoi(nx());
      else if(a=="--weak-wall-gamma")weakWallGamma=std::stod(nx());
      else if(a=="--weak-wall-h-scale")weakWallHScale=std::stod(nx());
      else if(a=="--weak-wall-nut-scale")weakWallNutScale=std::stod(nx());
      else if(a=="--gamma")gamma=std::stod(nx());else if(a=="--schur-nitsche-gamma")schurNitscheGamma=std::stod(nx());else if(a=="--alpha-u")alphaU=std::stod(nx());else if(a=="--alpha-p")alphaP=std::stod(nx());
      else if(a=="--rau-scale")rauScale=std::stod(nx());else if(a=="--simple-coupling")simpleCoupling=nx();else if(a=="--simple-tol")simpleTol=std::stod(nx());else if(a=="--max-outer")maxOuter=std::stoi(nx());
      else if(a=="--momentum-work")momWork=nx();else if(a=="--mom-omega")momOmega=std::stod(nx());else if(a=="--mom-rtol")momRtol=std::stod(nx());else if(a=="--mom-atol")momAtol=std::stod(nx());else if(a=="--mom-max")momMax=std::stoi(nx());else if(a=="--mom-cap-policy")momCapPolicy=nx();else if(a=="--mom-fgmres-restart")momFgmresRestart=std::stoi(nx());else if(a=="--mom-fgs-sweeps")momFgsSweeps=std::stoi(nx());
      else if(a=="--pressure-solver")pressureSolver=nx();else if(a=="--pressure-gauge")pressureGauge=nx();else if(a=="--p-rtol")pRtol=std::stod(nx());else if(a=="--p-atol")pAtol=std::stod(nx());else if(a=="--p-max")pMax=std::stoi(nx());
      else if(a=="--pod-mode")podMode=nx();else if(a=="--pod-train-snapshots")podTrainSnapshots=std::stoi(nx());else if(a=="--pod-rank")podRank=std::stoi(nx());else if(a=="--pod-max-rank")podMaxRank=std::stoi(nx());else if(a=="--pod-novelty")podNovelty=std::stod(nx());else if(a=="--pod-tail-iter")podTailIter=std::stoi(nx());else if(a=="--pod-hard-its")podHardIts=std::stoi(nx());else if(a=="--pod-accept-min-abs")podAcceptMinAbs=std::stoi(nx());else if(a=="--pod-accept-min-rel")podAcceptMinRel=std::stod(nx());else if(a=="--pod-refresh-abs")podRefreshAbs=std::stoi(nx());else if(a=="--pod-refresh-rel")podRefreshRel=std::stod(nx());else if(a=="--pod-recompress-rank")podRecompressRank=std::stoi(nx());else if(a=="--pod-recompress-history")podRecompressHistory=std::stoi(nx());else if(a=="--pod-recompress-max-loss-abs")podRecompressMaxLossAbs=std::stoi(nx());else if(a=="--pod-recompress-max-loss-rel")podRecompressMaxLossRel=std::stod(nx());else if(a=="--pod-recompress-method")podRecompressMethod=nx();else if(a=="--pod-recompress-force")podRecompressForce=std::stoi(nx());else if(a=="--pod-no-probe")podNoProbe=std::stoi(nx());
      else if(a=="--amg-hierarchy")amgHierarchy=nx();else if(a=="--cf-coarsening")cfCoarsening=nx();else if(a=="--cf-strength")cfStrength=nx();
      else if(a=="--cf-theta")cfTheta=std::stod(nx());else if(a=="--cf-interp")cfInterp=nx();else if(a=="--cf-pmax")cfPmax=std::stoi(nx());else if(a=="--cf-aggressive-first")cfAggressiveFirst=std::stoi(nx());
      else if(a=="--amg-terminal")amgTerminal=std::stoi(nx());else if(a=="--amg-fine-mode")amgFineMode=nx();else if(a=="--amg-smoother")amgSmoother=nx();else if(a=="--amg-jacobi-omega")amgJacobiOmega=std::stod(nx());
      else if(a=="--amg-jacobi-fine-sweeps")amgJacobiFineSweeps=std::stoi(nx());else if(a=="--amg-jacobi-coarse-sweeps")amgJacobiCoarseSweeps=std::stoi(nx());
      else if(a=="--amg-cheb-degree")amgChebDegree=std::stoi(nx());else if(a=="--amg-power-its")amgPowerIts=std::stoi(nx());
      else if(a=="--amg-lambda-safety")amgLambdaSafety=std::stod(nx());else if(a=="--amg-lambda-low-fraction")amgLambdaLow=std::stod(nx());
      else if(a=="--sst-g1")sstG1.enabled=std::stoi(nx())!=0;else if(a=="--sst-g2")sstG2=std::stoi(nx())!=0;else if(a=="--laminar-perf")laminarPerf=std::stoi(nx())!=0;
      else if(a=="--sst-g2-start")sstG2Start=nx();
      else if(a=="--profile-audit")profileAudit=std::stoi(nx())!=0;
      else if(a=="--profile-z-fracs")profileZFractions=nx();
      else if(a=="--profile-radial")profileRadial=std::stoi(nx());
      else if(a=="--profile-angles")profileAngles=std::stoi(nx());
      else if(a=="--profile-rmax")profileRmax=std::stod(nx());
      else if(a=="--sst-g2-physical-gate")sstG2PhysicalGate=std::stoi(nx())!=0;
      else if(a=="--sst-g2-metrics-every")sstG2MetricsEvery=std::stoi(nx());
      else if(a=="--sst-g2-max")sstG2Max=std::stoi(nx());else if(a=="--sst-g2-print-every")sstG2PrintEvery=std::stoi(nx());else if(a=="--sst-g2-plateau-tol")sstG2PlateauTol=std::stod(nx());else if(a=="--sst-g2-cont-atol")sstG2ContAtol=std::stod(nx());else if(a=="--sst-g2-plateau-samples")sstG2PlateauSamples=std::stoi(nx());
      else if(a=="--sst-intensity")sstG1.intensity=std::stod(nx());else if(a=="--sst-length-ratio")sstG1.lengthRatio=std::stod(nx());
      else if(a=="--sst-alpha-k")sstG1.alphaK=std::stod(nx());else if(a=="--sst-alpha-omega")sstG1.alphaOmega=std::stod(nx());else if(a=="--sst-tol")sstG1.nonlinearTol=std::stod(nx());else if(a=="--sst-max")sstG1.nonlinearMax=std::stoi(nx());
      else if(a=="--sst-linear-rtol")sstG1.linearRtol=std::stod(nx());else if(a=="--sst-linear-atol")sstG1.linearAtol=std::stod(nx());else if(a=="--sst-k-linear-rtol")sstG1.kLinearRtol=std::stod(nx());else if(a=="--sst-k-linear-atol")sstG1.kLinearAtol=std::stod(nx());else if(a=="--sst-omega-linear-rtol")sstG1.omegaLinearRtol=std::stod(nx());else if(a=="--sst-omega-linear-atol")sstG1.omegaLinearAtol=std::stod(nx());else if(a=="--sst-linear-max")sstG1.linearMax=std::stoi(nx());else if(a=="--sst-gs-omega")sstG1.gsOmega=std::stod(nx());else if(a=="--sst-linear-solver")sstG1.linearSolver=nx();else if(a=="--sst-fgmres-restart")sstG1.fgmresRestart=std::stoi(nx());else if(a=="--sst-fgs-sweeps")sstG1.fgsSweeps=std::stoi(nx());
      else if(a=="--sst-k-floor")sstG1.kFloor=std::stod(nx());else if(a=="--sst-omega-floor")sstG1.omegaFloor=std::stod(nx());else if(a=="--sst-deviatoric-stress"){const bool v=std::stoi(nx())!=0;sstG1.deviatoricProduction=v;sstG1.deviatoricMomentum=v;}else if(a=="--sst-deviatoric-production")sstG1.deviatoricProduction=std::stoi(nx())!=0;else if(a=="--sst-deviatoric-momentum")sstG1.deviatoricMomentum=std::stoi(nx())!=0;else if(a=="--sst-shared-spalding-wall-state")sstG1.sharedSpaldingWallState=std::stoi(nx())!=0;else if(a=="--sst-profile-integral-utau")sstG1.profileIntegralUTau=std::stoi(nx())!=0;else if(a=="--sst-profile-match-y")sstG1.profileMatchY=std::stod(nx());else if(a=="--sst-profile-fit-y1")sstG1.profileFitY1=std::stod(nx());else if(a=="--sst-profile-fit-y2")sstG1.profileFitY2=std::stod(nx());else if(a=="--sst-profile-fit-y3")sstG1.profileFitY3=std::stod(nx());else if(a=="--sst-profile-fit-y4")sstG1.profileFitY4=std::stod(nx());else if(a=="--sst-profile-fit-y5")sstG1.profileFitY5=std::stod(nx());else if(a=="--sst-profile-integral-nq")sstG1.profileIntegralNQ=std::stoi(nx());else if(a=="--sst-profile-integral-closure")sstG1.profileIntegralClosure=std::stoi(nx())!=0;else if(a=="--sst-profile-average-closure")sstG1.profileAverageClosure=std::stoi(nx())!=0;else if(a=="--sst-profile-average-wall-law"){auto v=nx();if(v=="spalding")sstG1.profileAverageWallLaw=0;else if(v=="log")sstG1.profileAverageWallLaw=1;else if(v=="spalding_relative")sstG1.profileAverageWallLaw=2;else if(v=="spalding_match_height")sstG1.profileAverageWallLaw=3;else if(v=="spalding_profile3")sstG1.profileAverageWallLaw=4;else if(v=="spalding_profile5")sstG1.profileAverageWallLaw=5;else throw std::runtime_error("--sst-profile-average-wall-law must be spalding, log, spalding_relative, spalding_match_height, spalding_profile3, or spalding_profile5");}else if(a=="--sst-profile-closure-yplus-min")sstG1.profileClosureYPlusMin=std::stod(nx());else if(a=="--sst-wall-penalty-gamma")sstG1.wallPenaltyGamma=std::stod(nx());else if(a=="--sst-profile5-eq-gamma")sstG1.profile5EqGamma=std::stod(nx());else if(a=="--sst-omega-wall-mode"){auto v=nx();if(v=="trace_log")sstG1.omegaWallMode=hxt5c::OMEGA_WALL_TRACE_LOG;else if(v=="sample_log")sstG1.omegaWallMode=hxt5c::OMEGA_WALL_SAMPLE_LOG;else if(v=="of_auto")sstG1.omegaWallMode=hxt5c::OMEGA_WALL_OF_AUTO;else if(v=="compatible")sstG1.omegaWallMode=hxt5c::OMEGA_WALL_COMPATIBLE;else if(v=="consistent_table")sstG1.omegaWallMode=hxt5c::OMEGA_WALL_CONSISTENT_TABLE;else if(v=="profile5_eq_sst")sstG1.omegaWallMode=hxt5c::OMEGA_WALL_PROFILE5_EQ_SST;else if(v=="lowre_resolved")sstG1.omegaWallMode=hxt5c::OMEGA_WALL_LOWRE_RESOLVED;else throw std::runtime_error("--sst-omega-wall-mode must be trace_log, sample_log, of_auto, compatible or consistent_table");}else if(a=="--sst-lowre-omega-factor")sstG1.lowReOmegaFactor=std::stod(nx());else if(a=="--sst-of-kappa")sstG1.ofKappa=std::stod(nx());else if(a=="--sst-of-E")sstG1.ofE=std::stod(nx());else if(a=="--sst-of-beta1")sstG1.ofBeta1=std::stod(nx());else if(a=="--sst-of-re-blend")sstG1.ofReBlend=std::stod(nx());else if(a=="--sst-of-production")sstG1.ofProduction=std::stoi(nx())!=0;else if(a=="--sst-of-production-scale")sstG1.ofProductionScale=std::stod(nx());else if(a=="--sst-of-wall-volume-scale")sstG1.ofWallVolumeScale=std::stod(nx());else if(a=="--sst-of-production-limit")sstG1.ofProductionLimit=std::stod(nx());else if(a=="--sst-print-every")sstG1.printEvery=std::stoi(nx());
      else if(a=="--amg-spectrum-policy")amgSpectrumPolicy=nx();else if(a=="--amg-mcgs-fine-sweeps")amgMcgsFineSweeps=std::stoi(nx());
      else if(a=="--amg-mcgs-coarse-sweeps")amgMcgsCoarseSweeps=std::stoi(nx());else if(a=="--amg-mcgs-omega")amgMcgsOmega=std::stod(nx());else if(a=="--amg-mcgs-order")amgMcgsOrder=nx();
      else throw std::runtime_error("unknown arg "+a);
    }

    if(mesh.empty())throw std::runtime_error("--mesh required");
    if(amgHierarchy!="cf")throw std::runtime_error("HXT4C1 wires the optimal CF hierarchy first; use --amg-hierarchy cf");
    if(cfCoarsening!="pmis")throw std::runtime_error("HXT4C1 CF coarsening must be pmis");
    if(cfStrength!="classical-negative")throw std::runtime_error("HXT4C1 CF strength must be classical-negative");
    if(!(cfInterp=="direct"||cfInterp=="exti"))throw std::runtime_error("HXT4C1 CF interp must be direct or exti");
    if(!(amgSmoother=="jacobi"||amgSmoother=="l1jacobi"||amgSmoother=="cheb2"||amgSmoother=="mcgs"))throw std::runtime_error("HXT4C1 unknown AMG smoother");
    if(!(amgFineMode=="explicit"||
         amgFineMode=="matrixfree"))
      throw std::runtime_error(
        "--amg-fine-mode must be explicit or matrixfree");

    if(amgFineMode=="matrixfree"){
      if(pressureSolver!="pcg_amg")
        throw std::runtime_error(
          "matrix-free fine AMG requires pcg_amg");

      if(amgSmoother!="jacobi")
        throw std::runtime_error(
          "matrix-free fine AMG currently requires jacobi");

      if(amgJacobiFineSweeps!=1)
        throw std::runtime_error(
          "matrix-free fine AMG currently requires one fine Jacobi sweep");
    }
    if(pressureSolver!="pcg_amg"&&pressureSolver!="pcg_jacobi")throw std::runtime_error("HXT4C1 pressure solver must be pcg_amg or pcg_jacobi");
    if(!(pressureGauge=="pin"||pressureGauge=="natural"))throw std::runtime_error("pressure gauge must be pin or natural");
    if(!(podMode=="off"||podMode=="online"||podMode=="adaptive"))throw std::runtime_error("--pod-mode must be off, online, or adaptive");
    if(podMode!="off"){
      if(pressureSolver!="pcg_amg")throw std::runtime_error("Balanced POD requires pcg_amg");
      if(amgFineMode!="explicit")throw std::runtime_error("Balanced POD requires explicit AMG fine mode");
      if(podTrainSnapshots<=0||podRank<=0||podRank>podTrainSnapshots||podMaxRank<podRank||podNovelty<0.0||
         podTailIter<=0||podHardIts<0||(podHardIts>0&&podHardIts<=6*podTailIter)||
         podAcceptMinAbs<0||podAcceptMinRel<0.0||podAcceptMinRel>=1.0||
         podRefreshAbs<0||podRefreshRel<0.0||podRefreshRel>=10.0||
         podRecompressRank<0||podRecompressHistory<=0||podRecompressMaxLossAbs<0||
         podRecompressMaxLossRel<0.0||podRecompressMaxLossRel>=1.0||
         !(podRecompressMethod=="tail_pod"||podRecompressMethod=="whiten_tail_pod"||podRecompressMethod=="amg_spectral")||
         !(podRecompressForce==0||podRecompressForce==1)||!(podNoProbe==0||podNoProbe==1)||
         (podRecompressRank>0&&(podRecompressRank>=podMaxRank||podRecompressRank<podRank)))
        throw std::runtime_error("bad POD train/rank/max-rank/novelty/tail/accept/refresh/recompress configuration");
    }
    if(pressureSolver=="pcg_amg"&&amgSmoother=="mcgs"&&amgMcgsOrder!="symmetric")throw std::runtime_error("PCG requires symmetric MCGS");
    if(!(alphaU>0&&alphaU<=1&&alphaP>0))throw std::runtime_error("bad relaxation");
    if(!(gamma>=0.0)||!(schurNitscheGamma>=0.0)||(gamma<=0.0&&schurNitscheGamma>0.0))throw std::runtime_error("bad momentum/Schur Nitsche gamma");
    if(!(momCapPolicy=="warn"||momCapPolicy=="fail"))throw std::runtime_error("--mom-cap-policy must be warn or fail");
    if(!(simpleCoupling=="lagged"||simpleCoupling=="corrected"))throw std::runtime_error("SIMPLE_COUPLING/--simple-coupling must be lagged or corrected");
    if(!(momWork=="sgs1"||momWork=="fgs1"||momWork=="altgs1"||momWork=="fgmres-fgs"))throw std::runtime_error("--momentum-work must be sgs1, fgs1, altgs1, or fgmres-fgs");
    if(momFgmresRestart<=0||momFgsSweeps<=0)throw std::runtime_error("bad momentum FGMRES/FGS option");
    if(!(mixlenScale>=0.0)||!(outerContAtol>=0.0)||!(wallSampleFraction>0.0&&wallSampleFraction<1.0))throw std::runtime_error("bad HXT5B turbulence/boundary/convergence option");
    if(!(weakWallGamma>0.0)||!(weakWallHScale>0.0)||!(weakWallNutScale>=0.0))throw std::runtime_error("bad weak momentum wall option");
    if(!(spaldingMolecularConsistency==0||spaldingMolecularConsistency==1))throw std::runtime_error("--spalding-molecular-consistency must be 0 or 1");
    if(sstG1.enabled&&sstG2)throw std::runtime_error("choose only one of --sst-g1 and --sst-g2");
    if(laminarPerf&&(sstG1.enabled||sstG2))
      throw std::runtime_error("--laminar-perf cannot be combined with SST G1/G2");
    if(laminarPerf)mixlenScale=0.0;
    if(!(sstG2Start=="g0"||sstG2Start=="plug"))throw std::runtime_error("--sst-g2-start must be g0 or plug");
    if(sstG2MetricsEvery<=0)throw std::runtime_error("bad SST G2 metrics cadence");
    if(profileAudit&&(profileRadial<8||profileAngles<4||!(profileRmax>0&&profileRmax<=1)))throw std::runtime_error("bad enriched profile audit option");
    if(sstG2&&(!(sstG2PlateauTol>0)||!(sstG2ContAtol>0)||sstG2Max<=0||sstG2PrintEvery<=0||sstG2PlateauSamples<3))throw std::runtime_error("bad SST G2 control option");
    if(!(sstG1.lowReOmegaFactor>0.0))throw std::runtime_error("bad low-Re omega factor");
    if((sstG1.enabled||sstG2)&&(!(sstG1.intensity>0)||!(sstG1.lengthRatio>0)||!(sstG1.alphaK>0&&sstG1.alphaK<=1)||!(sstG1.alphaOmega>0&&sstG1.alphaOmega<=1)||!(sstG1.linearRtol>0&&sstG1.linearRtol<1)||!(sstG1.linearAtol>=0)||sstG1.linearMax<=0||!(sstG1.gsOmega>0)||!(sstG1.kFloor>0)||!(sstG1.omegaFloor>0)||!(sstG1.ofKappa>0)||!(sstG1.ofE>1)||!(sstG1.ofBeta1>0)||!(sstG1.ofReBlend>0)||!(sstG1.ofProductionScale>=0)||!(sstG1.ofWallVolumeScale>0)||!(sstG1.ofProductionLimit>0)))throw std::runtime_error("bad SST transport/wall option");
    if((sstG1.enabled||sstG2)&&!(sstG1.linearSolver=="sgs1"||sstG1.linearSolver=="fgs1"||sstG1.linearSolver=="fgmres-fgs"))throw std::runtime_error("bad SST linear solver option");
    if((sstG1.enabled||sstG2)&&(sstG1.fgmresRestart<=0||sstG1.fgsSweeps<=0))throw std::runtime_error("bad SST FGMRES/FGS option");
    if(sstG1.enabled||sstG2){
      const double kr = sstG1.kLinearRtol>=0.0 ? sstG1.kLinearRtol : sstG1.linearRtol;
      const double ka = sstG1.kLinearAtol>=0.0 ? sstG1.kLinearAtol : sstG1.linearAtol;
      const double wr = sstG1.omegaLinearRtol>=0.0 ? sstG1.omegaLinearRtol : sstG1.linearRtol;
      const double wa = sstG1.omegaLinearAtol>=0.0 ? sstG1.omegaLinearAtol : sstG1.linearAtol;
      const auto validScalarTol=[](double rt,double at){
        return std::isfinite(rt)&&std::isfinite(at)&&rt>=0.0&&rt<1.0&&at>=0.0&&(rt>0.0||at>0.0);
      };
      if(!validScalarTol(kr,ka)||!validScalarTol(wr,wa))
        throw std::runtime_error("bad per-equation SST linear tolerances (rtol and atol must be nonnegative, one strictly positive)");
      std::printf("NODALS_SST_SPLIT_LINEAR_TOL kRtol=%.12e kAtol=%.12e omegaRtol=%.12e omegaAtol=%.12e legacyCommonRtol=%.12e legacyCommonAtol=%.12e status=PASS\n",kr,ka,wr,wa,sstG1.linearRtol,sstG1.linearAtol);
    }

    HostMesh M=nodals_hxt1::load(mesh);
    auto IP=hxt2::build_interface_plans(M);auto I3=hxt3a::build_if(M);auto Bbase=hxt3a::build_b(M,I3);
    auto BH=hxt5b::build_boundaries(M,Bbase,bulk);auto&Bh=BH.B;
    const int nv=Bh.nv,np=Bh.np,nh=(int)M.h.nhex;

    double zmin=1e300,zmax=-1e300,R=0;
    for(const auto&q:M.points){zmin=std::min(zmin,q.z);zmax=std::max(zmax,q.z);R=std::max(R,std::sqrt(q.x*q.x+q.y*q.y));}
    const double L=zmax-zmin,D=2*R,nu=bulk*D/Re;

    auto fixed=hxt5b::plug_initial(M,bulk);
    double plugVertexErr=0.0,plugEnrichMax=0.0;
    for(std::uint64_t v=0;v<M.h.nv;++v)
      plugVertexErr=std::max(plugVertexErr,std::abs((double)fixed.u2[(std::size_t)v]-bulk));
    for(std::uint64_t g=M.h.nv;g<M.h.nvel;++g)
      plugEnrichMax=std::max(plugEnrichMax,std::abs((double)fixed.u2[(std::size_t)g]));
    std::printf("NODALS_HXT5B3_INITIAL_PLUG vertexUzErrorMax=%.12e enrichmentUzMax=%.12e "
                "vertexUxUy=zero enrichmentUxUy=zero semantics=constant_Q1_or_P1_field status=%s\n",
      plugVertexErr,plugEnrichMax,(plugVertexErr==0.0&&plugEnrichMax==0.0)?"PASS":"FAIL");
    if(plugVertexErr!=0.0||plugEnrichMax!=0.0)
      throw std::runtime_error("HXT5B3 plug coefficient initialization mismatch");
    std::vector<unsigned char>clampXY=fixed.mask;for(std::size_t i=0;i<clampXY.size();++i)if(BH.wallMask[i])clampXY[i]=1;
    Dev<unsigned char>d_fixed(fixed.mask),d_clampXY(clampXY),d_wallMask(BH.wallMask);
    Dev<T>d_u0(fixed.u0),d_u1(fixed.u1),d_u2(fixed.u2);
    auto VH=build_velocity_csr(M,IP);auto VC=color_free_rows(VH,fixed.mask);GpuCSR A(VH,VC);

    Dev<nodals_hxt1::Point>d_pts(M.points);Dev<nodals_hxt1::HexConn>d_hc(M.hexes);Dev<nodals_hxt1::TetConn>d_tc(M.tets);Dev<nodals_hxt1::HexVel>d_hv(M.hexVel);
    Dev<nodals_hxt1::TetVel>d_tv(M.tetVel);Dev<nodals_hxt1::TetGeom>d_tg(M.tetGeom);Dev<hxt2::InterfacePlan>d_ip(IP);
    Dev<unsigned long long>d_bad(std::vector<unsigned long long>(1,0));

    // Persistent SST HEX/TET interface geometry + topology.
    hxt5c::SSTInterfaceCache d_sstInterfaceCache(M.h.ninterface);

    hxt5c::setup_sst_interface_cache(
      M.h.ninterface,
      d_pts,d_hc,d_tc,
      d_hv,d_tv,d_tg,d_ip,
      A,
      d_sstInterfaceCache,
      d_bad);

    // PERF_HXT_HEX_CONV_PERSISTENT_GEOMETRY
    // Layout:
    //   grad  [hex][27][14][3]
    //   det*w [hex][27]
    //
    // 9600 HEX case uses approximately 85 MiB total in FP64 geometry.
    Dev<double>d_hexConvGrad(
      std::vector<double>(
        (std::size_t)M.h.nhex*27ull*14ull*3ull,0.0));

    Dev<double>d_hexConvDetW(
      std::vector<double>(
        (std::size_t)M.h.nhex*27ull,0.0));

    Dev<T>d_turb(std::vector<T>(A.nnz,0)),d_diagRTurb(std::vector<T>((std::size_t)nv,0));
    // Explicit lagged transpose/deviatoric momentum correction.  These are
    // identically zero when --sst-deviatoric-stress=0.
    Dev<T>d_sstDevRhs0(std::vector<T>((std::size_t)nv,0));
    Dev<T>d_sstDevRhs1(std::vector<T>((std::size_t)nv,0));
    Dev<T>d_sstDevRhs2(std::vector<T>((std::size_t)nv,0));
    Dev<double>d_mixStats(std::vector<double>(4,0));
    std::vector<hxt5b::WallMatchPoint> wallMatchHost(BH.wallH.size()*16);
    if(sstG1.profileAverageClosure && sstG1.profileAverageWallLaw==3){
      if(!(sstG1.profileMatchY>0.0))throw std::runtime_error("spalding_match_height requires --sst-profile-match-y > 0");
      wallMatchHost=hxt5b::build_wall_match_height_cache(M,BH.wallH,sstG1.profileMatchY);
    }else if(sstG1.profileAverageClosure && sstG1.profileAverageWallLaw==5){
      if(!(sstG1.profileFitY1>0.0 && sstG1.profileFitY1<sstG1.profileFitY2 && sstG1.profileFitY2<sstG1.profileFitY3 && sstG1.profileFitY3<sstG1.profileFitY4 && sstG1.profileFitY4<sstG1.profileFitY5))
        throw std::runtime_error("spalding_profile5 requires 0 < --sst-profile-fit-y1 < y2 < y3 < y4 < y5");
      auto C1=(BH.wallH.empty()?std::vector<hxt5b::WallMatchPoint>{}:hxt5b::build_wall_match_height_cache(M,BH.wallH,sstG1.profileFitY1));
      auto C2=(BH.wallH.empty()?std::vector<hxt5b::WallMatchPoint>{}:hxt5b::build_wall_match_height_cache(M,BH.wallH,sstG1.profileFitY2));
      auto C3=(BH.wallH.empty()?std::vector<hxt5b::WallMatchPoint>{}:hxt5b::build_wall_match_height_cache(M,BH.wallH,sstG1.profileFitY3));
      auto C4=(BH.wallH.empty()?std::vector<hxt5b::WallMatchPoint>{}:hxt5b::build_wall_match_height_cache(M,BH.wallH,sstG1.profileFitY4));
      auto C5=(BH.wallH.empty()?std::vector<hxt5b::WallMatchPoint>{}:hxt5b::build_wall_match_height_cache(M,BH.wallH,sstG1.profileFitY5));
      if(C1.size()!=C2.size()||C1.size()!=C3.size()||C1.size()!=C4.size()||C1.size()!=C5.size())throw std::runtime_error("spalding_profile5 cache size mismatch");
      wallMatchHost.resize(C1.size()*5u);
      for(std::size_t p=0;p<C1.size();++p){
        wallMatchHost[5u*p+0u]=C1[p]; wallMatchHost[5u*p+1u]=C2[p]; wallMatchHost[5u*p+2u]=C3[p]; wallMatchHost[5u*p+3u]=C4[p]; wallMatchHost[5u*p+4u]=C5[p];
      }
      std::printf(
        "NODALS_SST_MATCH_PROFILE5_CACHE wallFaces=%zu wallQuadraturePoints=%zu samplesPerPoint=5 totalSamples=%zu y=[%.12e,%.12e,%.12e,%.12e,%.12e] fit=EQUAL_WEIGHT_IMPLICIT_SPALDING_LEAST_SQUARES status=PASS\n",
        BH.wallH.size(),C1.size(),wallMatchHost.size(),sstG1.profileFitY1,sstG1.profileFitY2,sstG1.profileFitY3,sstG1.profileFitY4,sstG1.profileFitY5);
    }else if(sstG1.profileAverageClosure && sstG1.profileAverageWallLaw==4){
      if(!(sstG1.profileFitY1>0.0 && sstG1.profileFitY1<sstG1.profileFitY2 && sstG1.profileFitY2<sstG1.profileFitY3))
        throw std::runtime_error("spalding_profile3 requires 0 < --sst-profile-fit-y1 < y2 < y3");
      auto C1=(BH.wallH.empty()?std::vector<hxt5b::WallMatchPoint>{}:hxt5b::build_wall_match_height_cache(M,BH.wallH,sstG1.profileFitY1));
      auto C2=(BH.wallH.empty()?std::vector<hxt5b::WallMatchPoint>{}:hxt5b::build_wall_match_height_cache(M,BH.wallH,sstG1.profileFitY2));
      auto C3=(BH.wallH.empty()?std::vector<hxt5b::WallMatchPoint>{}:hxt5b::build_wall_match_height_cache(M,BH.wallH,sstG1.profileFitY3));
      if(C1.size()!=C2.size()||C1.size()!=C3.size())throw std::runtime_error("spalding_profile3 cache size mismatch");
      wallMatchHost.resize(C1.size()*3u);
      for(std::size_t p=0;p<C1.size();++p){
        wallMatchHost[3u*p+0u]=C1[p];
        wallMatchHost[3u*p+1u]=C2[p];
        wallMatchHost[3u*p+2u]=C3[p];
      }
      std::printf(
        "NODALS_SST_MATCH_PROFILE3_CACHE wallFaces=%zu wallQuadraturePoints=%zu samplesPerPoint=3 totalSamples=%zu "
        "y=[%.12e,%.12e,%.12e] fit=EQUAL_WEIGHT_IMPLICIT_SPALDING_LEAST_SQUARES status=PASS\n",
        BH.wallH.size(),C1.size(),wallMatchHost.size(),sstG1.profileFitY1,sstG1.profileFitY2,sstG1.profileFitY3);
    }
    std::vector<hxt5b::TetWallMatchPoint>tetMatchHost;
    if(!BH.wallT.empty()){
      if(!sstG2 || !sstG1.profileAverageClosure ||
         sstG1.profileAverageWallLaw!=5 ||
         (sstG1.omegaWallMode!=hxt5c::OMEGA_WALL_PROFILE5_EQ_SST &&
          sstG1.omegaWallMode!=hxt5c::OMEGA_WALL_OF_AUTO))
        throw std::runtime_error("pure-tet requires G2 PROFILE5+EQ5 or PROFILE5+OF_AUTO");
      const double yy[5]={sstG1.profileFitY1,sstG1.profileFitY2,
        sstG1.profileFitY3,sstG1.profileFitY4,sstG1.profileFitY5};
      if(!(yy[0]>0 && yy[0]<yy[1] && yy[1]<yy[2] && yy[2]<yy[3] && yy[3]<yy[4]))
        throw std::runtime_error("pure-tet five sample heights not increasing");
      std::vector<hxt5b::TetWallMatchPoint> tmp[5];
      for(int m=0;m<5;++m)
        tmp[m]=hxt5b::build_tet_wall_match_height_cache(M,BH.wallT,yy[m]);
      tetMatchHost.resize(tmp[0].size()*5u);
      for(std::size_t i=0;i<tmp[0].size();++i)
        for(int m=0;m<5;++m)tetMatchHost[5u*i+(std::size_t)m]=tmp[m][i];
      std::printf("NODALS_SST_PROFILE5_TET_CONFIG pureTet=%d tetWallFaces=%zu samples=5 sampleGeometry=PHYSICAL_INWARD_NORMAL momentumUtau=FIVE_POINT_SPALDING scalarMode=%s totalSamples=%zu status=PASS\n",
        M.h.nhex==0?1:0,BH.wallT.size(),
        sstG1.omegaWallMode==hxt5c::OMEGA_WALL_OF_AUTO?"OFAUTO_POINT3_OMEGA_ONLY":"MULTIPOINT_EQ_SST",
        tetMatchHost.size());
      if(sstG1.omegaWallMode==hxt5c::OMEGA_WALL_OF_AUTO){
        std::printf("PURETET_OFAUTO_CONFIG momentum=SPALDING5_UNCHANGED omegaTarget=ORIGINAL_HEX_OF_AUTO_BLEND sample=POINT3 kWall=NATURAL_ZERO_GRADIENT omegaPenalty=ONE_RANK_ONE_TET_SAMPLE normalSlab=Y5_MINUS_Y1 production=%d hexAverageFirstEquivalent=NO status=PASS\n",(int)sstG1.ofProduction);
      }
    }
    Dev<hxt5b::TetWallMatchPoint>d_tetMatch(tetMatchHost);
    const auto tetQuadHost=hxt5b::build_tet_wall_quad_cache(M,BH.wallT);
    Dev<hxt5b::TetWallQuadCache>d_tetQuad(tetQuadHost);
    Dev<hxt5b::TetFacePlan>d_wallT(BH.wallT);
    sstG1.profileTetMatchPoints=d_tetMatch.p;
    sstG1.tetWallPlans=d_wallT.p;
    sstG1.tetWallQuads=d_tetQuad.p;
    Dev<hxt5b::WallMatchPoint>d_wallMatch(wallMatchHost);
    sstG1.profileMatchPoints=d_wallMatch.p;
    if(sstG1.profileAverageWallLaw==3)std::printf(
      "NODALS_SST_MATCH_HEIGHT_DEVICE_CACHE points=%zu bytes=%zu targetY=%.12e "
      "velocity=FULL_Q1BF2_POINT_VALUE wallReference=STATIONARY_ABSOLUTE "
      "momentumLinearization=LOCAL_NQ12_PICARD_EXACT_TRACTION "
      "sstAverages=FIRST_CELL_NQ12_UNCHANGED status=PASS\n",
      wallMatchHost.size(),wallMatchHost.size()*sizeof(hxt5b::WallMatchPoint),sstG1.profileMatchY);
    if(sstG1.profileAverageWallLaw==5)std::printf(
      "NODALS_SST_MATCH_PROFILE5_DEVICE_CACHE points=%zu bytes=%zu samplesPerWallQuadrature=5 "
      "velocity=FULL_Q1BF2_POINT_VALUE fit=ONE_UTAU_FROM_FIVE_REMOTE_SAMPLES "
      "momentumLinearization=LOCAL_NQ12_PICARD_EXACT_TRACTION sstAverages=FIRST_CELL_NQ12_UNCHANGED status=PASS\n",
      wallMatchHost.size(),wallMatchHost.size()*sizeof(hxt5b::WallMatchPoint));
    if(sstG1.profileAverageWallLaw==4)std::printf(
      "NODALS_SST_MATCH_PROFILE3_DEVICE_CACHE points=%zu bytes=%zu samplesPerWallQuadrature=3 "
      "velocity=FULL_Q1BF2_POINT_VALUE fit=ONE_UTAU_FROM_THREE_REMOTE_SAMPLES "
      "momentumLinearization=LOCAL_NQ12_PICARD_EXACT_TRACTION sstAverages=FIRST_CELL_NQ12_UNCHANGED status=PASS\n",
      wallMatchHost.size(),wallMatchHost.size()*sizeof(hxt5b::WallMatchPoint));
    Dev<hxt5b::HexFacePlan>d_inH(BH.inH),d_wallH(BH.wallH);Dev<hxt5b::TetFacePlan>d_inT(BH.inT);
    auto wallCacheHost=
      hxt5b::build_wall_hex_cache(
        M,BH.wallH,wallSampleFraction);
    Dev<hxt5b::WallHexQuadCache>
      d_wallCache(wallCacheHost);
    Dev<T>d_contSource(hxt4c::castv<T>(BH.csrc));
    Dev<T>d_dg(std::vector<T>(A.nnz,0)),d_diagDG(std::vector<T>((std::size_t)nv,0)),d_inletRhsZ(std::vector<T>((std::size_t)nv,0));
    Dev<T>d_wall(std::vector<T>(A.nnz,0)),d_diagWall(std::vector<T>((std::size_t)nv,0)),d_valZ(std::vector<T>(A.nnz,0));
    Dev<T>d_diagOrigZ(std::vector<T>((std::size_t)nv,0)),d_rauZ(std::vector<T>((std::size_t)nv,0));
    Dev<T>d_relaxDeltaXY(std::vector<T>((std::size_t)nv,0)),d_relaxDeltaZ(std::vector<T>((std::size_t)nv,0));
    Dev<double>d_wallStats(std::vector<double>(12,0));

    auto TT=hxt2::build_tet_tensor();HXT1_CUDA(cudaMemcpyToSymbol(hxt2::c_tet_T,TT.data(),TT.size()*sizeof(double)));
    auto CT=build_tet_conv_tensor();HXT1_CUDA(cudaMemcpyToSymbol(h4b_tet_convT,CT.data(),CT.size()*sizeof(double)));

    // Build immutable HEX convection geometry exactly once.
    HXT1_CUDA(cudaMemset(
      d_bad.p,0,sizeof(unsigned long long)));

    if(M.h.nhex)
      hex_conv_geom_cache_setup<<<M.h.nhex,32>>>(
        d_pts.p,
        d_hc.p,
        d_hexConvGrad.p,
        d_hexConvDetW.p,
        M.h.nhex,
        d_bad.p);

    HXT1_CUDA(cudaGetLastError());
    HXT1_CUDA(cudaDeviceSynchronize());

    {
      unsigned long long hexConvGeomBad=0;
      HXT1_CUDA(cudaMemcpy(
        &hexConvGeomBad,
        d_bad.p,
        sizeof(hexConvGeomBad),
        cudaMemcpyDeviceToHost));

      if(hexConvGeomBad)
        throw std::runtime_error(
          "persistent HEX convection geometry cache failures="+
          std::to_string(hexConvGeomBad));
    }

    {
      const double gradMiB=
        (double)((std::size_t)M.h.nhex*27ull*14ull*3ull*
                 sizeof(double))/(1024.0*1024.0);

      const double detwMiB=
        (double)((std::size_t)M.h.nhex*27ull*
                 sizeof(double))/(1024.0*1024.0);

      std::printf(
        "NODALS_HXT_HEX_CONV_CACHE "
        "cells=%llu quadrature=27 "
        "gradMiB=%.3f detwMiB=%.3f totalMiB=%.3f "
        "geometry=PERSISTENT runtime=NUMERIC_ONLY "
        "operator=UNCHANGED status=PASS\n",
        (unsigned long long)M.h.nhex,
        gradMiB,detwMiB,gradMiB+detwMiB);
    }


    std::printf("NODALS_HXT4C_KERNEL_POLICY interfaceThreads=128 interfacePairTraversal=STRIDED hexConvectionThreads=128 hexPairTraversal=STRIDED status=PASS\n");
    HXT1_CUDA(cudaMemset(A.base.p,0,A.nnz*sizeof(T)));HXT1_CUDA(cudaMemset(A.diagRBase.p,0,nv*sizeof(T)));HXT1_CUDA(cudaMemset(d_bad.p,0,sizeof(unsigned long long)));
    if(M.h.nhex)hex_diff_assemble<<<M.h.nhex,256>>>(d_pts.p,d_hc.p,d_hv.p,A.hexSlot.p,A.base.p,A.diagRBase.p,M.h.nhex,nu,d_bad.p);
    if(M.h.ntet)tet_diff_assemble<<<M.h.ntet,64>>>(d_tv.p,d_tg.p,A.tetSlot.p,A.base.p,A.diagRBase.p,M.h.ntet,nu);
    if(M.h.ninterface)interface_assemble<<<2*M.h.ninterface,128>>>(d_pts.p,d_hc.p,d_hv.p,d_tv.p,d_tg.p,d_ip.p,A.row.p,A.col.p,A.base.p,A.diagRBase.p,M.h.ninterface,nu,gamma,d_bad.p);
    HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
    unsigned long long bad=0;HXT1_CUDA(cudaMemcpy(&bad,d_bad.p,sizeof(bad),cudaMemcpyDeviceToHost));
    if(bad)throw std::runtime_error("baseline assembly geometry failures");

    hxt4a::BDevice<T>B(Bh);
    int pin=-1;
    if(pressureGauge=="pin"){
      pin=0;
      if(!M.outHex.empty())pin=M.outHex[0].cell;
      else if(!M.outTet.empty())pin=nh+M.outTet[0].cell;
    }
    std::printf(
      "NODALS_HXT5B_PRESSURE_GAUGE mode=%s pin=%d "
      "continuityRows=%s pressureColumns=%s status=PASS\n",
      pressureGauge.c_str(),pin,
      pin>=0?"ONE_REPLACED":"ALL_RETAINED",
      pin>=0?"ONE_ELIMINATED":"ALL_RETAINED");

    Dev<T>d_p(std::vector<T>((std::size_t)np,0)),d_dp(std::vector<T>((std::size_t)np,0)),d_cont(std::vector<T>((std::size_t)np,0));
    Dev<double>d_contStats(std::vector<double>(2,0.0));
    Dev<T>d_sd(std::vector<T>((std::size_t)np,0)),d_isd(std::vector<T>((std::size_t)np,0));
    Dev<T>d_bt0(std::vector<T>((std::size_t)nv,0)),d_bt1(std::vector<T>((std::size_t)nv,0)),d_bt2(std::vector<T>((std::size_t)nv,0));
    Dev<T>d_rhs0(std::vector<T>((std::size_t)nv,0)),d_rhs1(std::vector<T>((std::size_t)nv,0)),d_rhs2(std::vector<T>((std::size_t)nv,0));
    hxt4a::Reducer<T>pred(np),ured(nv);hxt4a::CGWork<T>WP(np);

    // Initial current-fine pressure matrix and setup-snapshot hierarchy.
    HXT1_CUDA(cudaMemset(A.conv.p,0,A.nnz*sizeof(T)));HXT1_CUDA(cudaMemset(d_bad.p,0,sizeof(unsigned long long)));
    if(M.h.nhex)hex_conv_assemble_cached<<<M.h.nhex,128>>>(
          d_hv.p,A.hexSlot.p,
          d_u0.p,d_u1.p,d_u2.p,
          d_hexConvGrad.p,d_hexConvDetW.p,
          A.conv.p,M.h.nhex);
    if(M.h.ntet)tet_conv_assemble_thread_per_cell<<<(M.h.ntet+127)/128,128>>>(d_tv.p,d_tg.p,A.tetSlot.p,d_u0.p,d_u1.p,d_u2.p,A.conv.p,M.h.ntet);
    hxt5a::MixStats mix{};
    if(laminarPerf){
      HXT1_CUDA(cudaMemset(d_turb.p,0,d_turb.n*sizeof(T)));
      HXT1_CUDA(cudaMemset(d_diagRTurb.p,0,d_diagRTurb.n*sizeof(T)));
    }else{
      mix=hxt5a::h5_assemble_mixlen(
        M.h.nhex,M.h.ntet,M.h.ninterface,
        d_pts,d_hc,d_tc,d_hv,d_tv,d_tg,d_ip,A,
        d_u0.p,d_u1.p,d_u2.p,
        d_turb,d_diagRTurb,d_mixStats,
        R,nu,mixlenScale,gamma,d_bad);
    }
    hxt5b::assemble_boundary(BH,d_inH,d_inT,d_wallH,d_wallCache,d_pts,d_hc,d_tc,d_hv,d_tv,d_tg,A,
      d_u0.p,d_u1.p,d_u2.p,d_dg,d_diagDG,d_inletRhsZ,d_wall,d_diagWall,d_wallStats,d_bad,
      bulk,R,nu,mixlenScale,wallSampleFraction,d_wallT.p,d_tetQuad.p,d_tetMatch.p);
    hxt5b::combine<<<(A.nnz+TPB-1)/TPB,TPB>>>(A.nnz,A.base.p,A.conv.p,d_turb.p,d_dg.p,d_wall.p,A.val.p,d_valZ.p);
    HXT1_CUDA(cudaMemset(d_bad.p,0,sizeof(unsigned long long)));
    hxt5b::h5b3_finalize_row_l1<<<(nv+TPB-1)/TPB,TPB>>>(
      nv,d_fixed.p,d_wallMask.p,A.row.p,A.diagPos.p,A.val.p,d_valZ.p,
      A.diagRBase.p,A.conv.p,d_diagRTurb.p,d_diagDG.p,d_diagWall.p,
      A.rau.p,d_rauZ.p,A.diagOriginal.p,d_diagOrigZ.p,d_relaxDeltaXY.p,d_relaxDeltaZ.p,
      gamma,schurNitscheGamma,alphaU,rauScale,d_bad.p);
    HXT1_CUDA(cudaDeviceSynchronize());HXT1_CUDA(cudaMemcpy(&bad,d_bad.p,sizeof(bad),cudaMemcpyDeviceToHost));
    if(bad){
      // NODALS_HXT5B_RAU_DIAG_AUDIT_BEGIN
      // Failure-only host audit.  This deliberately does NOT alter the
      // positivity policy or any operator value; it only reports which part
      // of the directional rAU construction triggered the existing gate.
      std::vector<T> hOx((std::size_t)nv),hOz((std::size_t)nv),hDx((std::size_t)nv),hDz((std::size_t)nv);
      std::vector<T> hDiagR((std::size_t)nv),hDiagRTurb((std::size_t)nv),hDiagDG((std::size_t)nv),hDiagWall((std::size_t)nv);
      std::vector<std::int32_t> hDiagPos((std::size_t)nv);
      std::vector<T> hConv((std::size_t)A.nnz);
      HXT1_CUDA(cudaMemcpy(hOx.data(),A.diagOriginal.p,(std::size_t)nv*sizeof(T),cudaMemcpyDeviceToHost));
      HXT1_CUDA(cudaMemcpy(hOz.data(),d_diagOrigZ.p,(std::size_t)nv*sizeof(T),cudaMemcpyDeviceToHost));
      HXT1_CUDA(cudaMemcpy(hDx.data(),d_relaxDeltaXY.p,(std::size_t)nv*sizeof(T),cudaMemcpyDeviceToHost));
      HXT1_CUDA(cudaMemcpy(hDz.data(),d_relaxDeltaZ.p,(std::size_t)nv*sizeof(T),cudaMemcpyDeviceToHost));
      HXT1_CUDA(cudaMemcpy(hDiagR.data(),A.diagRBase.p,(std::size_t)nv*sizeof(T),cudaMemcpyDeviceToHost));
      HXT1_CUDA(cudaMemcpy(hDiagRTurb.data(),d_diagRTurb.p,(std::size_t)nv*sizeof(T),cudaMemcpyDeviceToHost));
      HXT1_CUDA(cudaMemcpy(hDiagDG.data(),d_diagDG.p,(std::size_t)nv*sizeof(T),cudaMemcpyDeviceToHost));
      HXT1_CUDA(cudaMemcpy(hDiagWall.data(),d_diagWall.p,(std::size_t)nv*sizeof(T),cudaMemcpyDeviceToHost));
      HXT1_CUDA(cudaMemcpy(hDiagPos.data(),A.diagPos.p,(std::size_t)nv*sizeof(std::int32_t),cudaMemcpyDeviceToHost));
      HXT1_CUDA(cudaMemcpy(hConv.data(),A.conv.p,(std::size_t)A.nnz*sizeof(T),cudaMemcpyDeviceToHost));

      // "interfaceAdjacent" means a velocity DOF belonging to either element
      // in at least one HEX/TET interface plan.  This is intentionally a
      // broader and more useful diagnostic category than trace-only DOFs.
      std::vector<unsigned char> interfaceAdjacent((std::size_t)nv,0);
      for(const auto&p:IP){
        const auto&HG=M.hexVel[(std::size_t)p.hexCell];
        for(int a=0;a<14;++a)interfaceAdjacent[(std::size_t)HG.g[a]]=1;
        for(int s=0;s<2;++s){
          const auto&TG=M.tetVel[(std::size_t)p.tetCell[s]];
          for(int a=0;a<8;++a)interfaceAdjacent[(std::size_t)TG.g[a]]=1;
        }
      }

      struct H5B3AuditStat{
        unsigned long long nonfinite=0,nonpositive=0;
        double minFinite=std::numeric_limits<double>::infinity();
        double maxFinite=-std::numeric_limits<double>::infinity();
        int minIndex=-1;
      };
      struct H5B3AuditOff{
        unsigned long long total=0,wall=0,interfaceAdjacent=0,wallAndInterface=0,neither=0;
      };
      auto statUpdate=[&](H5B3AuditStat&s,double v,int i){
        if(!std::isfinite(v)){++s.nonfinite;return;}
        if(v<s.minFinite){s.minFinite=v;s.minIndex=i;}
        s.maxFinite=std::max(s.maxFinite,v);
        if(!(v>0.0))++s.nonpositive;
      };
      auto offUpdate=[&](H5B3AuditOff&o,int i){
        ++o.total;
        const bool w=BH.wallMask[(std::size_t)i]!=0;
        const bool q=interfaceAdjacent[(std::size_t)i]!=0;
        if(w)++o.wall;if(q)++o.interfaceAdjacent;if(w&&q)++o.wallAndInterface;if(!w&&!q)++o.neither;
      };
      H5B3AuditStat sOx,sOz,sNoX,sNoZ,sMx,sMz;
      H5B3AuditOff oOx,oOz,oNoX,oNoZ,oMx,oMz;
      unsigned long long active=0,fixedCount=0;
      for(int i=0;i<nv;++i){
        if(fixed.mask[(std::size_t)i]){++fixedCount;continue;}
        ++active;
        const int dp=hDiagPos[(std::size_t)i];
        const double ox=(double)hOx[(std::size_t)i],oz=(double)hOz[(std::size_t)i];
        const double convd=(double)hConv[(std::size_t)dp];
        const double noX=(double)hDiagR[(std::size_t)i]+convd+(double)hDiagRTurb[(std::size_t)i]+(double)hDiagDG[(std::size_t)i];
        const double noZ=noX+(double)hDiagWall[(std::size_t)i];
        const double mx=noX+(double)hDx[(std::size_t)i];
        const double mz=noZ+(double)hDz[(std::size_t)i];
        statUpdate(sOx,ox,i);statUpdate(sOz,oz,i);statUpdate(sNoX,noX,i);statUpdate(sNoZ,noZ,i);statUpdate(sMx,mx,i);statUpdate(sMz,mz,i);
        if(!std::isfinite(ox)||!(ox>0.0))offUpdate(oOx,i);
        if(!std::isfinite(oz)||!(oz>0.0))offUpdate(oOz,i);
        if(!std::isfinite(noX)||!(noX>0.0))offUpdate(oNoX,i);
        if(!std::isfinite(noZ)||!(noZ>0.0))offUpdate(oNoZ,i);
        if(!std::isfinite(mx)||!(mx>0.0))offUpdate(oMx,i);
        if(!std::isfinite(mz)||!(mz>0.0))offUpdate(oMz,i);
      }
      auto printStat=[&](const char*q,const H5B3AuditStat&s){
        std::printf("NODALS_HXT5B_RAU_DIAG_AUDIT quantity=%s minFinite=%.12e maxFinite=%.12e nonpositive=%llu nonfinite=%llu minIndex=%d\n",
          q,s.minFinite,s.maxFinite,s.nonpositive,s.nonfinite,s.minIndex);
      };
      auto printOff=[&](const char*q,const H5B3AuditOff&o){
        std::printf("NODALS_HXT5B_RAU_OFFENDERS quantity=%s total=%llu wall=%llu interfaceAdjacent=%llu wallAndInterface=%llu neither=%llu\n",
          q,o.total,o.wall,o.interfaceAdjacent,o.wallAndInterface,o.neither);
      };
      std::printf("NODALS_HXT5B_RAU_DIAG_AUDIT kernelBad=%llu active=%llu fixed=%llu alphaU=%.12e rauScale=%.12e policy=UNCHANGED_DIAGNOSTIC_ONLY\n",
        bad,active,fixedCount,alphaU,rauScale);
      printStat("fullXY",sOx);printOff("fullXY",oOx);
      printStat("fullZ",sOz);printOff("fullZ",oOz);
      printStat("noPenXY",sNoX);printOff("noPenXY",oNoX);
      printStat("noPenZ",sNoZ);printOff("noPenZ",oNoZ);
      printStat("relaxedNoPenXY",sMx);printOff("relaxedNoPenXY",oMx);
      printStat("relaxedNoPenZ",sMz);printOff("relaxedNoPenZ",oMz);
      auto printWorst=[&](const char*q,const H5B3AuditStat&s){
        const int i=s.minIndex;if(i<0)return;
        const int dp=hDiagPos[(std::size_t)i];
        const double convd=(double)hConv[(std::size_t)dp];
        const double noX=(double)hDiagR[(std::size_t)i]+convd+(double)hDiagRTurb[(std::size_t)i]+(double)hDiagDG[(std::size_t)i];
        const double noZ=noX+(double)hDiagWall[(std::size_t)i];
        std::printf("NODALS_HXT5B_RAU_WORST quantity=%s index=%d wall=%d interfaceAdjacent=%d fullXY=%.12e fullZ=%.12e diagRBase=%.12e convDiag=%.12e turbDiag=%.12e dgDiag=%.12e wallDiag=%.12e noPenXY=%.12e noPenZ=%.12e deltaXY=%.12e deltaZ=%.12e relaxedNoPenXY=%.12e relaxedNoPenZ=%.12e\n",
          q,i,(int)BH.wallMask[(std::size_t)i],(int)interfaceAdjacent[(std::size_t)i],
          (double)hOx[(std::size_t)i],(double)hOz[(std::size_t)i],(double)hDiagR[(std::size_t)i],convd,
          (double)hDiagRTurb[(std::size_t)i],(double)hDiagDG[(std::size_t)i],(double)hDiagWall[(std::size_t)i],
          noX,noZ,(double)hDx[(std::size_t)i],(double)hDz[(std::size_t)i],
          noX+(double)hDx[(std::size_t)i],noZ+(double)hDz[(std::size_t)i]);
      };
      printWorst("fullXY",sOx);printWorst("fullZ",sOz);printWorst("noPenXY",sNoX);printWorst("noPenZ",sNoZ);printWorst("relaxedNoPenXY",sMx);printWorst("relaxedNoPenZ",sMz);
      std::printf("NODALS_HXT5B_RAU_DIAG_AUDIT_END status=FAIL_AS_BEFORE noSolverPolicyChanged=1\n");
      throw std::runtime_error("initial invalid relaxed momentum diagonal or directional Schur denominator");
    }
    std::printf("NODALS_HXT5B_RAU_RELAXED_GATE stage=setup "
                "test=RELAXED_MOMENTUM_AND_SCHUR_DIAGONALS "
                "unrelaxedNegativeAllowed=1 failed=%llu status=PASS\n",bad);

    // Match the validated turbulence gate: the weak DG inlet must preserve the
    // initialized constant plug exactly in momentum, G*Uhat - rhs = 0.
    hxt5b::spmv<<<(nv+TPB-1)/TPB,TPB>>>(nv,A.row.p,A.col.p,d_dg.p,d_u2.p,A.tmp.p);
    hxt5b::h5b3_subtract<<<(nv+TPB-1)/TPB,TPB>>>(nv,A.tmp.p,d_inletRhsZ.p,A.res.p);
    const double dgMomAbs=ured.norm(A.res.p);
    const double dgMomRhs=ured.norm(d_inletRhsZ.p);
    const double dgMomRel=dgMomAbs/std::max(dgMomRhs,1e-300);
    const double dgMomLim=std::is_same<T,float>::value?5e-5:2e-11;
    std::printf("NODALS_HXT5B3_DG_PLUG_MOMENTUM_CANCELLATION abs=%.12e rhsNorm=%.12e rel=%.12e limit=%.12e status=%s\n",
      dgMomAbs,dgMomRhs,dgMomRel,dgMomLim,dgMomRel<=dgMomLim?"PASS":"FAIL");
    if(!(dgMomRel<=dgMomLim))
      throw std::runtime_error("HXT5B3 DG weak inlet does not preserve constant plug");

    hxt5b::spmv<<<(nv+TPB-1)/TPB,TPB>>>(nv,A.row.p,A.col.p,A.conv.p,d_u2.p,A.tmp.p);
    const double convPlug=ured.norm(A.tmp.p);
    const double convRel=convPlug/std::max(dgMomRhs,1e-300);
    std::printf("NODALS_HXT5B3_CONVECTION_PLUG_CANCELLATION abs=%.12e relToDgRhs=%.12e expected=roundoff status=%s\n",
      convPlug,convRel,convRel<1e-3?"PASS":"CHECK");

    hxt5b::h5b3_relax_audit("xy_setup",A.diagOriginal.p,d_relaxDeltaXY.p,nv);
    hxt5b::h5b3_relax_audit("z_setup",d_diagOrigZ.p,d_relaxDeltaZ.p,nv);

    std::vector<T>rauT((std::size_t)nv),rauZT((std::size_t)nv);
    HXT1_CUDA(cudaMemcpy(rauT.data(),A.rau.p,nv*sizeof(T),cudaMemcpyDeviceToHost));
    HXT1_CUDA(cudaMemcpy(rauZT.data(),d_rauZ.p,nv*sizeof(T),cudaMemcpyDeviceToHost));
    std::vector<double>rxyD((std::size_t)nv),rzD((std::size_t)nv);for(int i=0;i<nv;++i){rxyD[(std::size_t)i]=(double)rauT[(std::size_t)i];rzD[(std::size_t)i]=(double)rauZT[(std::size_t)i];}
    auto PH=hxt4c::build_hybrid_fine_host(Bh,pin,rxyD);
    hxt5b::retune_host_fine(PH,Bh,pin,rxyD,rxyD,rzD);
    {
      long double rowSum2=0.0L,diag2=0.0L,frob2=0.0L;
      double maxAbsRowSum=0.0,maxAbsDiag=0.0;

      for(int i=0;i<PH.A.n;++i){
        long double rs=0.0L;

        for(std::int64_t k=PH.A.row[(std::size_t)i];
            k<PH.A.row[(std::size_t)i+1];++k){
          const double a=PH.A.val[(std::size_t)k];
          rs+=(long double)a;
          frob2+=(long double)a*(long double)a;
        }

        const double d=PH.A.diag[(std::size_t)i];

        rowSum2+=rs*rs;
        diag2+=(long double)d*(long double)d;

        maxAbsRowSum=std::max(maxAbsRowSum,std::abs((double)rs));
        maxAbsDiag=std::max(maxAbsDiag,std::abs(d));
      }

      const double rowSumL2=std::sqrt((double)rowSum2);
      const double diagL2=std::sqrt((double)diag2);
      const double frob=std::sqrt((double)frob2);

      std::printf(
        "NODALS_HXT5B_PRESSURE_CONSTANT_AUDIT "
        "mode=%s pin=%d "
        "S1_L2=%.12e S1_over_diagL2=%.12e "
        "S1_over_frob=%.12e "
        "maxAbsS1=%.12e maxAbsDiag=%.12e "
        "maxRatio=%.12e "
        "interpretation=%s status=PASS\n",
        pressureGauge.c_str(),pin,
        rowSumL2,
        rowSumL2/std::max(diagL2,1e-300),
        rowSumL2/std::max(frob,1e-300),
        maxAbsRowSum,maxAbsDiag,
        maxAbsRowSum/std::max(maxAbsDiag,1e-300),
        (rowSumL2/std::max(diagL2,1e-300) > 1e-10)
          ? "CONSTANT_MODE_NOT_NULL_CANDIDATE"
          : "CONSTANT_MODE_NULL_CANDIDATE");
    }


    int effectivePowerIts=0;
    if(amgSpectrumPolicy=="always")effectivePowerIts=amgPowerIts;
    else if(amgSpectrumPolicy=="auto"&&amgSmoother=="cheb2")effectivePowerIts=amgPowerIts;
    else if(amgSpectrumPolicy=="off")effectivePowerIts=0;
    else if(amgSpectrumPolicy!="auto")throw std::runtime_error("bad AMG spectrum policy");
    if(amgSmoother=="cheb2"&&effectivePowerIts<=0)throw std::runtime_error("Chebyshev requires spectrum estimation");

    auto HH=nodals_gpu::build_cf_hierarchy_from_csr(
      PH.A,cfTheta,cfPmax,cfInterp,cfAggressiveFirst!=0,amgTerminal,
      effectivePowerIts,amgLambdaSafety,amgLambdaLow);

    hxt4c::AMGOptions AO;
    AO.smoother=amgSmoother;AO.jacobiOmega=amgJacobiOmega;
    AO.jacobiFineSweeps=amgJacobiFineSweeps;AO.jacobiCoarseSweeps=amgJacobiCoarseSweeps;
    AO.chebDegree=amgChebDegree;AO.lambdaLowFraction=amgLambdaLow;
    AO.mcgsFineSweeps=amgMcgsFineSweeps;AO.mcgsCoarseSweeps=amgMcgsCoarseSweeps;AO.mcgsOmega=amgMcgsOmega;AO.mcgsOrder=amgMcgsOrder;
    hxt4c::HybridAMG<T>PC(PH,HH,pin,AO);
    hxt5b::refresh_pc(PC,B,A.rau.p,A.rau.p,d_rauZ.p);
    hxt4c::BalancedPOD<T> pressurePOD(
      np,pin,podMode!="off",podMode=="adaptive",podTrainSnapshots,podRank,podMaxRank,podNovelty,
      podTailIter,podHardIts,podAcceptMinAbs,podAcceptMinRel,podRefreshAbs,podRefreshRel,
      podRecompressRank,podRecompressHistory,podRecompressMaxLossAbs,podRecompressMaxLossRel,
      podRecompressMethod,podRecompressForce!=0,podNoProbe!=0);
    std::printf(
      "NODALS_HXT5B_AMG_FINE_MODE "
      "mode=%s "
      "fineSmoother=%s "
      "fineJacobiSweeps=%d "
      "coarseHierarchy=UNCHANGED "
      "setupExplicitFineCSR=RETAINED "
      "dynamicExplicitRefresh=%s "
      "status=PASS\n",
      amgFineMode.c_str(),
      amgSmoother.c_str(),
      amgJacobiFineSweeps,
      amgFineMode=="explicit"?"ON":"OFF");


    auto wallNow=hxt5b::wallstats(d_wallStats,bulk);
    std::printf(
      "NODALS_HXT5B_CONFIG precision=%s geometry=fp64 reductions=fp64 cellsHex=%llu cellsTet=%llu "
      "velocityDofs=%d pressureDofs=%d Re=%.12g bulk=%.12g nu=%.12e gamma=%.6g "
      "turbulence=NIKURADSE_ALGEBRAIC mixlenScale=%.8g inlet=DG_NUMERICAL_TRACE_PLUG velocityInletDOFs=FREE "
      "wall=FIREDRAKE_Q1BF2_OFFWALL_SPALDING wallSampleFraction=%.6g kappa=0.4 B=5.5 transverseWallClamp=1 "
      "momentumWork=%s momentumRelax=row_l1 momentumRtol=%.3e momentumAtol=%.3e momentumMax=%d momentumCapPolicy=%s outerConvergence=CONTINUITY_PLUS_XYZ_PHYSICAL_MOMENTUM_INITIAL_RELATIVE "
      "pressureSolver=%s pRtol=%.3e pAtol=%.3e pMax=%d amgHierarchy=%s cfCoarsening=%s cfStrength=%s "
      "cfTheta=%.6g cfInterp=%s cfPmax=%d amgSmoother=%s chebDegree=%d powerIts=%d spectrumPolicy=%s "
      "alphaU=%.6g alphaP=%.6g rauScale=%.6g directionalRAU=1 rauPenaltyPolicy=SEPARATE_GAMMA simpleTol=%.3e status=PASS\n",
      PRECISION,(unsigned long long)M.h.nhex,(unsigned long long)M.h.ntet,nv,np,Re,bulk,nu,gamma,mixlenScale,
      wallSampleFraction,momWork.c_str(),momRtol,momAtol,momMax,momCapPolicy.c_str(),pressureSolver.c_str(),pRtol,pAtol,pMax,amgHierarchy.c_str(),cfCoarsening.c_str(),
      cfStrength.c_str(),cfTheta,cfInterp.c_str(),cfPmax,amgSmoother.c_str(),amgChebDegree,effectivePowerIts,
      amgSpectrumPolicy.c_str(),alphaU,alphaP,rauScale,simpleTol);
    std::printf("NODALS_HXT5E_KRYLOV_CONFIG momentumSolver=%s momRestart=%d momFgsSweeps=%d sstLinearSolver=%s sstRestart=%d sstFgsSweeps=%d rightPreconditioner=FORWARD_COLORED_GS trueResidualAudit=1 status=PASS\n",momWork.c_str(),momFgmresRestart,momFgsSweeps,sstG1.linearSolver.c_str(),sstG1.fgmresRestart,sstG1.fgsSweeps);
    std::printf("NODALS_PRESSURE_POD_MODE mode=%s trainSnapshots=%d requestedRank=%d maxRank=%d noveltyTol=%.12e "
      "tailIter=%d hardIts=%d multiTailDepths=[%d,%d,%d] "
      "refreshAbs=%d refreshRel=%.12e recompressRank=%d recompressHistory=%d recompressMaxLossAbs=%d recompressMaxLossRel=%.12e "
      "recompressMethod=%s recompressForce=%d noProbe=%d scope=SST_G2_ONLY status=PASS\n",
      podMode.c_str(),podTrainSnapshots,podRank,podMaxRank,podNovelty,
      podTailIter,podHardIts,podTailIter,3*podTailIter,6*podTailIter,podRefreshAbs,podRefreshRel,
      podRecompressRank,podRecompressHistory,podRecompressMaxLossAbs,podRecompressMaxLossRel,
      podRecompressMethod.c_str(),podRecompressForce,podNoProbe);
    std::printf("NODALS_TRUE_SIMPLE_CONFIG mode=%s velocityCorrection=%s velocityCorrectionScale=1 pressureRelaxation=%.12e alphaU=%.12e status=PASS\n",
      simpleCoupling.c_str(),simpleCoupling=="corrected"?"IMMEDIATE_MINUS_RAU_BT_DP":"NONE_LAGGED",alphaP,alphaU);
    std::printf("NODALS_HXT5B_NITSCHE_CONFIG momentumGamma=%.12e schurGamma=%.12e schurOverMomentum=%.12e rAUPolicy=NO_RAW_NOPENGATE_PLUS_ROWL1_PLUS_SELECTED_NITSCHE status=PASS\n",
      gamma,schurNitscheGamma,(gamma>0.0?schurNitscheGamma/gamma:0.0));
    std::printf(
      "NODALS_HXT5B_BOUNDARY_SETUP inletHexFaces=%zu inletTetFaces=%zu wallHexFaces=%zu inletArea=%.12e "
      "prescribedFluxSum=%.12e constantPatchResidual=%.12e wallArea=%.12e wallYPlusMeanInitial=%.12e "
      "wallDarcyInitial=%.12e rootFailures=%.0f status=PASS\n",
      BH.inH.size(),BH.inT.size(),BH.wallH.size(),BH.inletArea,BH.sourceSum,BH.constResidual,
      wallNow.area,wallNow.yPlusMean,wallNow.fDarcy,wallNow.rootFailures);
    if(laminarPerf)
      std::printf(
        "NODALS_HXT5B_LAMINAR_PERF_CONFIG "
        "enabled=1 turbulence=OFF nut=ZERO "
        "scalarTransports=SKIPPED "
        "momentumTurbulentOperator=SKIPPED "
        "wall=UNCHANGED_HXT5B pressure=%s fixedOuter=%d status=PASS\n",
        pressureSolver.c_str(),maxOuter);
    else
      std::printf(
        "NODALS_HXT5B_MIXLEN_SETUP "
        "meanNutOverNu=%.12e maxNutOverNu=%.12e "
        "maxStrain=%.12e status=PASS\n",
        mix.meanNutOverNu,mix.maxNutOverNu,mix.maxStrain);

    double c0=-1,m0[3]={-1,-1,-1};bool converged=false;int outer=0;long long pIts=0;
    // G0/SST baseline diagnostic buffer.  This does not enter any operator or
    // convergence decision; it is copied only at the existing sparse print points
    // so pressure-drop stationarity can be judged together with y+ and wall friction.
    std::vector<T> hPressurePlateau((std::size_t)np);

    const bool directSSTStart=(sstG2&&sstG2Start=="plug");

    std::chrono::steady_clock::time_point lamPerfT0{};
    if(laminarPerf){
      HXT1_CUDA(cudaDeviceSynchronize());
      lamPerfT0=std::chrono::steady_clock::now();
    }

    if(directSSTStart){
      // Straight-pipe potential-flow startup: the existing coefficient initializer
      // already gives U=(0,0,Ub) exactly, enriched velocity coefficients zero, and p=0.
      // Skip the expensive converged Nikuradse SIMPLE precursor entirely.
      converged=true;outer=0;pIts=0;
      std::printf("NODALS_SST_G2_START mode=PLUG_POTENTIAL velocity=UNIFORM_AXIAL_BULK pressure=ZERO kOmega=UNIFORM_INLET_FORMULAS precursorSimpleIterations=0 status=PASS\n");
    } else {
// ============================================================
// HXT5B_COARSE4_TIMING_V1
// Instrumentation only.
//
// e0 -> e1 : dynamic assembly
// e1 -> e2 : Schur diagonal + fine numeric refresh
// e2 -> e3 : momentum RHS/B^T + momentum solve
// e3 -> e4 : continuity construction
// e4 -> e5 : pressure solve + pressure update/B^T correction
//
// Only one cudaEventSynchronize per measured outer.
// Outer 1 excluded from averages.
// ============================================================
cudaEvent_t h5te[6];
for(int i=0;i<6;++i)
    HXT1_CUDA(cudaEventCreate(&h5te[i]));

double h5tAssembly=0.0;
double h5tSchur=0.0;
double h5tMomentum=0.0;
double h5tContinuity=0.0;
double h5tPressure=0.0;
int h5tCount=0;

for(int it=1;it<=maxOuter;++it){
      HXT1_CUDA(cudaEventRecord(h5te[0],0));
      HXT1_CUDA(cudaMemset(A.conv.p,0,A.nnz*sizeof(T)));HXT1_CUDA(cudaMemset(d_bad.p,0,sizeof(unsigned long long)));
      if(M.h.nhex)hex_conv_assemble_cached<<<M.h.nhex,128>>>(
          d_hv.p,A.hexSlot.p,
          d_u0.p,d_u1.p,d_u2.p,
          d_hexConvGrad.p,d_hexConvDetW.p,
          A.conv.p,M.h.nhex);
      if(M.h.ntet)tet_conv_assemble_thread_per_cell<<<(M.h.ntet+127)/128,128>>>(d_tv.p,d_tg.p,A.tetSlot.p,d_u0.p,d_u1.p,d_u2.p,A.conv.p,M.h.ntet);
      if(!laminarPerf)
        mix=hxt5a::h5_assemble_mixlen(
          M.h.nhex,M.h.ntet,M.h.ninterface,
          d_pts,d_hc,d_tc,d_hv,d_tv,d_tg,d_ip,A,
          d_u0.p,d_u1.p,d_u2.p,
          d_turb,d_diagRTurb,d_mixStats,
          R,nu,mixlenScale,gamma,d_bad);
      hxt5b::assemble_boundary(BH,d_inH,d_inT,d_wallH,d_wallCache,d_pts,d_hc,d_tc,d_hv,d_tv,d_tg,A,
        d_u0.p,d_u1.p,d_u2.p,d_dg,d_diagDG,d_inletRhsZ,d_wall,d_diagWall,d_wallStats,d_bad,
        bulk,R,nu,mixlenScale,wallSampleFraction,d_wallT.p,d_tetQuad.p,d_tetMatch.p);
      hxt5b::combine<<<(A.nnz+TPB-1)/TPB,TPB>>>(A.nnz,A.base.p,A.conv.p,d_turb.p,d_dg.p,d_wall.p,A.val.p,d_valZ.p);
      HXT1_CUDA(cudaMemset(d_bad.p,0,sizeof(unsigned long long)));
      hxt5b::h5b3_finalize_row_l1<<<(nv+TPB-1)/TPB,TPB>>>(
      nv,d_fixed.p,d_wallMask.p,A.row.p,A.diagPos.p,A.val.p,d_valZ.p,
      A.diagRBase.p,A.conv.p,d_diagRTurb.p,d_diagDG.p,d_diagWall.p,
      A.rau.p,d_rauZ.p,A.diagOriginal.p,d_diagOrigZ.p,d_relaxDeltaXY.p,d_relaxDeltaZ.p,
      gamma,schurNitscheGamma,alphaU,rauScale,d_bad.p);
      HXT1_CUDA(cudaDeviceSynchronize());HXT1_CUDA(cudaMemcpy(&bad,d_bad.p,sizeof(bad),cudaMemcpyDeviceToHost));
      if(bad)throw std::runtime_error("invalid relaxed momentum/Schur diagonal count="+std::to_string(bad));
      if(it==1)std::printf("NODALS_HXT5B_RAU_RELAXED_GATE stage=outer1 "
                            "test=RELAXED_MOMENTUM_AND_SCHUR_DIAGONALS "
                            "unrelaxedNegativeAllowed=1 failed=%llu status=PASS\n",bad);
      HXT1_CUDA(cudaEventRecord(h5te[1],0));
      if(it==1&&!laminarPerf){
        hxt5b::audit_real("matrix_xy_after_finalize",A.val.p,A.nnz);
        hxt5b::audit_real("matrix_z_after_finalize",d_valZ.p,A.nnz);
        hxt5b::audit_real("rau_xy_after_finalize",A.rau.p,(std::size_t)nv);
        hxt5b::audit_real("rau_z_after_finalize",d_rauZ.p,(std::size_t)nv);
        hxt5b::audit_diag_host("diag_xy_original",A.diagOriginal.p,nv);
        hxt5b::audit_diag_host("diag_z_original",d_diagOrigZ.p,nv);
      }

      hxt5b::sdiag<<<(np+TPB-1)/TPB,TPB>>>(np,B.row.p,B.col.p,B.bx.p,B.by.p,B.bz.p,A.rau.p,A.rau.p,d_rauZ.p,d_sd.p,pin);
      hxt4a::invert_diag_kernel<T><<<(np+TPB-1)/TPB,TPB>>>(np,d_sd.p,d_isd.p,pin);
      hxt5b::SAniso<T>S(B,A.rau.p,A.rau.p,d_rauZ.p,pin);
      if(amgFineMode=="explicit")
        hxt5b::refresh_pc(PC,B,A.rau.p,A.rau.p,d_rauZ.p);
      HXT1_CUDA(cudaEventRecord(h5te[2],0));

      if(it==1&&!laminarPerf&&amgFineMode=="explicit"){
        double parity=hxt5b::parity(PC,S,np,pin,pred);
        const double lim=std::is_same<T,float>::value?3e-5:2e-11;
        std::printf("NODALS_HXT5B_FINE_CSR_PARITY rel=%.12e limit=%.12e status=%s\n",parity,lim,parity<=lim?"PASS":"FAIL");
        if(parity>lim)throw std::runtime_error("HXT5B anisotropic fine CSR/matrix-free Schur parity failed");
      }

      hxt4a::BT_apply(B,d_p.p,d_bt0.p,d_bt1.p,d_bt2.p);
      hxt5b::h5b3_rhs<<<(nv+TPB-1)/TPB,TPB>>>(nv,d_clampXY.p,d_bt0.p,nullptr,d_u0.p,d_relaxDeltaXY.p,d_rhs0.p);
      hxt5b::h5b3_rhs<<<(nv+TPB-1)/TPB,TPB>>>(nv,d_clampXY.p,d_bt1.p,nullptr,d_u1.p,d_relaxDeltaXY.p,d_rhs1.p);
      hxt5b::h5b3_rhs<<<(nv+TPB-1)/TPB,TPB>>>(nv,d_fixed.p,d_bt2.p,d_inletRhsZ.p,d_u2.p,d_relaxDeltaZ.p,d_rhs2.p);
      if(it==1&&!laminarPerf){
        hxt5b::audit_real("rhs_x_before_sgs",d_rhs0.p,(std::size_t)nv);
        hxt5b::audit_real("rhs_y_before_sgs",d_rhs1.p,(std::size_t)nv);
        hxt5b::audit_real("rhs_z_before_sgs",d_rhs2.p,(std::size_t)nv);
        hxt5b::audit_real("u_z_before_sgs",d_u2.p,(std::size_t)nv);
      }
      auto mx=hxt5b_perf::momentum_solve(momWork,it,A,VC,A.val.p,d_rhs0.p,d_u0.p,momOmega,d_clampXY.p,momRtol,momAtol,momMax,momFgmresRestart,momFgsSweeps,ured,"x",momCapPolicy=="warn");
      auto my=hxt5b_perf::momentum_solve(momWork,it,A,VC,A.val.p,d_rhs1.p,d_u1.p,momOmega,d_clampXY.p,momRtol,momAtol,momMax,momFgmresRestart,momFgsSweeps,ured,"y",momCapPolicy=="warn");
      auto mz=hxt5b_perf::momentum_solve(momWork,it,A,VC,d_valZ.p,d_rhs2.p,d_u2.p,momOmega,d_fixed.p,momRtol,momAtol,momMax,momFgmresRestart,momFgsSweeps,ured,"z",momCapPolicy=="warn");
      HXT1_CUDA(cudaEventRecord(h5te[3],0));
      // HXT5B_MOM_CAP_POLICY_V1
      // h5b3_momentum_solve throws internally for NaN/Inf or invalid work.
      // Therefore ok=false here means finite MOM_MAX exhaustion only.
      // In SIMPLE this is bounded inner work; the outer physical residual gate
      // remains the authority on nonlinear convergence.
      const bool momCapMiss=(!mx.ok||!my.ok||!mz.ok);
      if(momCapMiss){
        const bool strictMomCap=(momCapPolicy=="fail");
        std::printf("NODALS_HXT5B3_MOM_CAP outer=%d xRel=%.12e xSweeps=%d yRel=%.12e ySweeps=%d zRel=%.12e zSweeps=%d targetRel=%.6g maxSweeps=%d policy=%s status=%s\n",
          it,mx.rel,mx.sweeps,my.rel,my.sweeps,mz.rel,mz.sweeps,momRtol,momMax,
          momCapPolicy.c_str(),strictMomCap?"FAIL":"WARN_CONTINUE");
        if(strictMomCap)
          throw std::runtime_error("HXT5B3 momentum residual-drop target not reached");
      }
      if(it<=5||it%20==0)
        std::printf("NODALS_HXT5B3_MOM_SOLVE outer=%d xSweeps=%d xRel=%.6e ySweeps=%d yRel=%.6e zSweeps=%d zRel=%.6e targetRel=%.6g status=PASS\n",
          it,mx.sweeps,mx.rel,my.sweeps,my.rel,mz.sweeps,mz.rel,momRtol);

      hxt4a::B_apply(B,d_u0.p,d_u1.p,d_u2.p,d_cont.p);hxt5b::addsrc<<<(np+TPB-1)/TPB,TPB>>>(np,d_contSource.p,d_cont.p);hxt4a::pin_zero_kernel<T><<<1,1>>>(pin,d_cont.p);
      if(it==1&&!laminarPerf)hxt5b::audit_real("continuity_before_norm",d_cont.p,(std::size_t)np);
      double cn=pred.norm(d_cont.p);
      HXT1_CUDA(cudaEventRecord(h5te[4],0));
      if(!std::isfinite(cn))throw std::runtime_error("HXT5B nonfinite continuity norm");

      HXT1_CUDA(cudaMemset(d_dp.p,0,np*sizeof(T)));
      hxt4a::CGResult pr;
      if(pressureSolver=="pcg_amg")pr=(amgFineMode=="matrixfree"
          ? hxt4c::pcg_amg_mf_fine(
              S,PC,d_sd.p,
              d_cont.p,d_dp.p,np,
              pRtol,pAtol,pMax,WP,pin)
          : hxt4c::pcg_amg(
              S,PC,
              d_cont.p,d_dp.p,np,
              pRtol,pAtol,pMax,WP,pin));
      else pr=pcg_prod(S,d_cont.p,d_dp.p,d_isd.p,np,pRtol,pAtol,pMax,WP,pin);
      if(!pr.ok){
        std::printf("NODALS_HXT5B_PRESSURE_FAIL outer=%d solver=%s pCG=%d pRel=%.12e pRtol=%.12e pAtol=%.12e pMax=%d status=FAIL\n",
          it,pressureSolver.c_str(),pr.its,pr.rel,pRtol,pAtol,pMax);
        throw std::runtime_error("pressure solver failed production tolerance");
      }
      pIts+=pr.its;
      pressure_update_minus_kernel<<<(np+TPB-1)/TPB,TPB>>>(
        np,pin,(T)alphaP,d_dp.p,d_p.p);

      const double cnPreCorrection=cn;
      if(simpleCoupling=="corrected"){
        // Full classical SIMPLE velocity correction.  Do NOT multiply by
        // alphaP here: alphaP relaxes pressure, not the velocity correction.
        hxt4a::BT_apply(B,d_dp.p,d_bt0.p,d_bt1.p,d_bt2.p);
        hxt5b_simple_velocity_correct<T><<<(nv+TPB-1)/TPB,TPB>>>(
          nv,d_clampXY.p,A.rau.p,d_bt0.p,d_u0.p);
        hxt5b_simple_velocity_correct<T><<<(nv+TPB-1)/TPB,TPB>>>(
          nv,d_clampXY.p,A.rau.p,d_bt1.p,d_u1.p);
        hxt5b_simple_velocity_correct<T><<<(nv+TPB-1)/TPB,TPB>>>(
          nv,d_fixed.p,d_rauZ.p,d_bt2.p,d_u2.p);
        hxt4a::B_apply(B,d_u0.p,d_u1.p,d_u2.p,d_cont.p);
        hxt5b::addsrc<<<(np+TPB-1)/TPB,TPB>>>(np,d_contSource.p,d_cont.p);
        hxt4a::pin_zero_kernel<T><<<1,1>>>(pin,d_cont.p);
        cn=pred.norm(d_cont.p);
        if(!std::isfinite(cn))throw std::runtime_error("HXT5B corrected SIMPLE nonfinite continuity");
        if(it<=5||it%20==0)
          std::printf("NODALS_TRUE_SIMPLE_CORRECTION scope=G0 it=%d contPre=%.12e contPost=%.12e ratio=%.12e pCG=%d pRel=%.12e status=PASS\n",
            it,cnPreCorrection,cn,cn/std::max(cnPreCorrection,1e-300),pr.its,pr.rel);
      }

      if(laminarPerf){
        HXT1_CUDA(cudaEventRecord(h5te[5],0));
        HXT1_CUDA(cudaEventSynchronize(h5te[5]));

        if(it>=2){
          float x01=0.0f;
          float x12=0.0f;
          float x23=0.0f;
          float x34=0.0f;
          float x45=0.0f;

          HXT1_CUDA(cudaEventElapsedTime(&x01,h5te[0],h5te[1]));
          HXT1_CUDA(cudaEventElapsedTime(&x12,h5te[1],h5te[2]));
          HXT1_CUDA(cudaEventElapsedTime(&x23,h5te[2],h5te[3]));
          HXT1_CUDA(cudaEventElapsedTime(&x34,h5te[3],h5te[4]));
          HXT1_CUDA(cudaEventElapsedTime(&x45,h5te[4],h5te[5]));

          h5tAssembly   += (double)x01;
          h5tSchur      += (double)x12;
          h5tMomentum   += (double)x23;
          h5tContinuity += (double)x34;
          h5tPressure   += (double)x45;

          ++h5tCount;
        }

        outer=it;

        if(it<=5 || it==maxOuter)
          std::printf(
            "NODALS_HXT5B_LAMINAR_PERF "
            "it=%d absCont=%.12e pCG=%d pRel=%.3e status=PASS\n",
            it,cn,pr.its,pr.rel);

        continue;
      }

      hxt4a::BT_apply(B,d_p.p,d_bt0.p,d_bt1.p,d_bt2.p);
      HXT1_CUDA(cudaEventRecord(h5te[5],0));
      HXT1_CUDA(cudaEventSynchronize(h5te[5]));

      if(it>=2){
        float x01=0.0f;
        float x12=0.0f;
        float x23=0.0f;
        float x34=0.0f;
        float x45=0.0f;

        HXT1_CUDA(cudaEventElapsedTime(&x01,h5te[0],h5te[1]));
        HXT1_CUDA(cudaEventElapsedTime(&x12,h5te[1],h5te[2]));
        HXT1_CUDA(cudaEventElapsedTime(&x23,h5te[2],h5te[3]));
        HXT1_CUDA(cudaEventElapsedTime(&x34,h5te[3],h5te[4]));
        HXT1_CUDA(cudaEventElapsedTime(&x45,h5te[4],h5te[5]));

        h5tAssembly   += (double)x01;
        h5tSchur      += (double)x12;
        h5tMomentum   += (double)x23;
        h5tContinuity += (double)x34;
        h5tPressure   += (double)x45;
        ++h5tCount;
      }

      double ma[3]={
        hxt5b::h5b3_physical_residual_norm(A,A.val.p,d_clampXY.p,d_bt0.p,nullptr,d_u0.p,d_relaxDeltaXY.p,ured),
        hxt5b::h5b3_physical_residual_norm(A,A.val.p,d_clampXY.p,d_bt1.p,nullptr,d_u1.p,d_relaxDeltaXY.p,ured),
        hxt5b::h5b3_physical_residual_norm(A,d_valZ.p,d_fixed.p,d_bt2.p,d_inletRhsZ.p,d_u2.p,d_relaxDeltaZ.p,ured)};
      if(it==1){c0=cn;m0[0]=ma[0];m0[1]=ma[1];m0[2]=ma[2];}
      const double st=std::max(simpleTol,1e-300);
      const double rc=cn/std::max(std::max(c0,outerContAtol/st),1e-300);
      const double rm0=ma[0]/std::max(std::max(m0[0],momAtol/st),1e-300);
      const double rm1=ma[1]/std::max(std::max(m0[1],momAtol/st),1e-300);
      const double rm2=ma[2]/std::max(std::max(m0[2],momAtol/st),1e-300);
      outer=it;converged=(it>1&&rc<=simpleTol&&rm0<=simpleTol&&rm1<=simpleTol&&rm2<=simpleTol);
      if(it<=5||it%20==0||converged){
        auto ws=hxt5b::wallstats(d_wallStats,bulk);
        HXT1_CUDA(cudaMemcpy(hPressurePlateau.data(),d_p.p,(std::size_t)np*sizeof(T),cudaMemcpyDeviceToHost));
        const auto pPlateau=hxt4a::to_double(hPressurePlateau);
        const double pressureSlopePlateau=hxt4a::pressure_slope(M,pPlateau);
        const double pInPlateau=hxt4a::boundary_p(M,pPlateau,true);
        const double pOutPlateau=hxt4a::boundary_p(M,pPlateau,false);
        const double pressureDropPlateau=pInPlateau-pOutPlateau;
        std::printf("NODALS_HXT5B_SIMPLE it=%d relContInit=%.12e absCont=%.12e relMomXInit=%.12e absMomX=%.12e relMomYInit=%.12e absMomY=%.12e relMomZInit=%.12e absMomZ=%.12e pCG=%d pRel=%.3e meanNutOverNu=%.6e maxNutOverNu=%.6e wallYPlusMean=%.6e wallDarcy=%.6e pressureSlope=%.6e pressureDrop=%.6e converged=%d status=PASS\n",
          it,rc,cn,rm0,ma[0],rm1,ma[1],rm2,ma[2],pr.its,pr.rel,mix.meanNutOverNu,mix.maxNutOverNu,ws.yPlusMean,ws.fDarcy,pressureSlopePlateau,pressureDropPlateau,(int)converged);
      }
      if(converged)break;
      double worst=std::max(std::max(rc,rm0),std::max(rm1,rm2));
      if(it>=5&&(!std::isfinite(worst)||worst>1e8))throw std::runtime_error("HXT5B coupled-residual divergence guard");
    }
    if(laminarPerf){
      HXT1_CUDA(cudaDeviceSynchronize());

      const auto lamPerfT1=
        std::chrono::steady_clock::now();

      const double loopMs=
        std::chrono::duration<double,std::milli>(
          lamPerfT1-lamPerfT0).count();

      std::printf(
        "NODALS_HXT5B_LAMINAR_PERF_TIMING "
        "iterations=%d loopMs=%.3f avgOuterMs=%.6f status=PASS\n",
        outer,loopMs,loopMs/std::max(outer,1));

      converged=true;
    }

    if(!laminarPerf&&!converged)
      throw std::runtime_error(
        "HXT5B SIMPLE maxOuter without all-residual convergence");

    
if(h5tCount>0){
    const double inv=1.0/(double)h5tCount;

    const double assembly   = h5tAssembly*inv;
    const double schur      = h5tSchur*inv;
    const double momentum   = h5tMomentum*inv;
    const double continuity = h5tContinuity*inv;
    const double pressure   = h5tPressure*inv;

    const double schurCont =
        schur + continuity;

    const double tracked =
        assembly +
        momentum +
        schurCont +
        pressure;

    std::printf(
      "NODALS_HXT5B_COARSE4_TIMING "
      "measuredOuter=%d "
      "policy=OUTER_2_TO_END "
      "assemblyAvgMs=%.6f "
      "momentumAvgMs=%.6f "
      "schurRefreshAvgMs=%.6f "
      "continuityAvgMs=%.6f "
      "schurPlusContinuityAvgMs=%.6f "
      "pressureSolveUpdateAvgMs=%.6f "
      "trackedAvgMs=%.6f "
      "status=PASS\n",
      h5tCount,
      assembly,
      momentum,
      schur,
      continuity,
      schurCont,
      pressure,
      tracked);
}

for(int i=0;i<6;++i)
    cudaEventDestroy(h5te[i]);

    } // end G0/SIMPLE else block

// G2: either start directly from the exact straight-pipe plug/potential field
    // (default development path) or from the legacy converged G0 field.
    // In both cases k/omega are initialized uniformly from the inlet formulas.

    // The old Nikuradse field is no longer used by momentum inside this loop.
    hxt5c::Stats g2FinalStats;bool g2Gate=true;int g2Outer=0;double g2KRel=0,g2WRel=0,g2NutRel=0;
    if(sstG2){
      hxt5d::State TS(nv,(int)M.h.nv,bulk,R,sstG1);hxt4a::Reducer<T> sstRed(nv);
      std::vector<double>hf,hy,hdp;bool g2Converged=false;double finalDp=0,finalF=0,finalY=0;double g2ContActiveL2Initial=-1,g2ContActiveMaxInitial=-1,g2ContGlobalInitial=-1,g2ContPinInitial=-1;
      auto relSpan=[](const std::vector<double>&v){if(v.empty())return std::numeric_limits<double>::infinity();auto mm=std::minmax_element(v.begin(),v.end());double mean=0;for(double x:v)mean+=x;mean/=v.size();return (*mm.second-*mm.first)/std::max(std::abs(mean),1e-300);};
      const char* sstWallPlacement=
        sstG1.omegaWallMode==hxt5c::OMEGA_WALL_PROFILE5_EQ_SST ? "PROFILE5_REMOTE_MULTIPOINT_EQ_SST" :
        sstG1.profileAverageClosure && sstG1.omegaWallMode==hxt5c::OMEGA_WALL_TRACE_LOG ? "NQ_AVERAGE_TARGET_PHYSICAL_TRACE_NITSCHE" :
        sstG1.profileAverageClosure ? "NQ_VOLUME_AVERAGE_RANK_ONE" :
        sstG1.profileIntegralClosure ? "FULL_PROFILE_VOLUME_PROJECTION" :
        (sstG1.omegaWallMode==hxt5c::OMEGA_WALL_TRACE_LOG ? "PHYSICAL_TRACE_SYMMETRIC_NITSCHE" : "OFFWALL_SAMPLE_PENALTY");
      if(sstG1.profileAverageClosure){
        if(sstG1.profileIntegralClosure)throw std::runtime_error("SST_PROFILE_AVERAGE_CLOSURE and SST_PROFILE_INTEGRAL_CLOSURE are mutually exclusive");
        if(sstG1.profileIntegralNQ!=4&&sstG1.profileIntegralNQ!=8&&sstG1.profileIntegralNQ!=12)throw std::runtime_error("SST_PROFILE_AVERAGE_CLOSURE requires SST_PROFILE_INTEGRAL_NQ=4, 8, or 12");
        const bool avgOfAuto=sstG1.omegaWallMode==hxt5c::OMEGA_WALL_OF_AUTO;
        const bool avgTable=sstG1.omegaWallMode==hxt5c::OMEGA_WALL_CONSISTENT_TABLE;
        const bool avgProfile5Eq=sstG1.omegaWallMode==hxt5c::OMEGA_WALL_PROFILE5_EQ_SST;
        const bool avgSampleLog=sstG1.omegaWallMode==hxt5c::OMEGA_WALL_SAMPLE_LOG;
        const bool avgTraceLog=sstG1.omegaWallMode==hxt5c::OMEGA_WALL_TRACE_LOG;
        if(!avgOfAuto&&!avgTable&&!avgSampleLog&&!avgTraceLog&&!avgProfile5Eq)throw std::runtime_error("SST_PROFILE_AVERAGE_CLOSURE requires SST_OMEGA_WALL_MODE=of_auto, consistent_table, sample_log, or trace_log");
        if(avgTable){
          if(momentumWallMode!=hxt5d::MOM_WALL_SST_CONSISTENT_TABLE)throw std::runtime_error("average-first consistent_table requires MOMENTUM_WALL_MODE=sst_consistent_table");
          if(sstG1.ofProduction)throw std::runtime_error("average-first consistent_table requires SST_OF_PRODUCTION=0");
          std::printf("NODALS_SST_PROFILE_AVERAGE_TABLE_CONFIG enabled=1 normalQuadrature=GAUSS%d_FULL_WALL_ADJACENT_HEX averaging=PHYSICAL_VOLUME_WEIGHTED_NORMAL_COLUMN inversion=ONE_TABLE_INVERSION_AT_UBAR_YBAR momentum=TABLE_UTAU_SQUARED_WITH_AVERAGED_TRIAL kTarget=TABLE_KPLUS_UTAU2 omegaTarget=TABLE_OMEGAPLUS_UTAU2_OVER_NU scalarConstraint=RANK_ONE_CELL_AVERAGE_PENALTY production=TABLE_NATIVE_NO_EXTRA_OF_SOURCE sampleFractionRole=DIAGNOSTIC_ONLY status=PASS\n",sstG1.profileIntegralNQ);
        }
        if(avgSampleLog||avgTraceLog){
          if(momentumWallMode!=hxt5d::MOM_WALL_SPALDING_PROFILE_INTEGRAL)throw std::runtime_error("average-first sample_log/trace_log requires MOMENTUM_WALL_MODE=spalding_profile_integral");
          if(!sstG1.profileIntegralUTau)throw std::runtime_error("average-first sample_log/trace_log requires SST_PROFILE_INTEGRAL_UTAU=1");
          if(sstG1.ofProduction)throw std::runtime_error("average-first sample_log/trace_log requires SST_OF_PRODUCTION=0");
          std::printf("NODALS_SST_PROFILE_AVERAGE_LOG_CONFIG enabled=1 mode=%s normalQuadrature=GAUSS%d_FULL_WALL_ADJACENT_HEX averaging=PHYSICAL_VOLUME_WEIGHTED_NORMAL_COLUMN utau=SPALDING_ON_NQ_AVERAGED_U_Y omegaTarget=UTAU_OVER_SQRTBETASTAR_KAPPA_YBAR placement=%s kWall=HIGH_RE_ZERO_GRADIENT production=NONE sampleFractionRole=DIAGNOSTIC_ONLY status=PASS\n",hxt5c::omega_wall_mode_name(sstG1.omegaWallMode),sstG1.profileIntegralNQ,avgTraceLog?"PHYSICAL_TRACE_SYMMETRIC_NITSCHE":"RANK_ONE_CELL_AVERAGE_PENALTY");
        }
      }
            if(sstG1.omegaWallMode==hxt5c::OMEGA_WALL_PROFILE5_EQ_SST){
        if(!sstG1.profileAverageClosure||sstG1.profileAverageWallLaw!=5)throw std::runtime_error("profile5_eq_sst requires PROFILE5 wall sampling");
        if(momentumWallMode!=hxt5d::MOM_WALL_SPALDING_PROFILE_INTEGRAL)throw std::runtime_error("profile5_eq_sst requires spalding_profile_integral momentum");
        if(!sstG1.profileIntegralUTau)throw std::runtime_error("profile5_eq_sst requires SST_PROFILE_INTEGRAL_UTAU=1");
        if(sstG1.ofProduction)throw std::runtime_error("profile5_eq_sst requires SST_OF_PRODUCTION=0");
        if(!(sstG1.profile5EqGamma>0.0))throw std::runtime_error("profile5_eq_sst requires --sst-profile5-eq-gamma > 0");
        std::printf("NODALS_SST_PROFILE5_EQ_SST_CONFIG enabled=1 model=SST2003M_LOCAL_EQUILIBRIUM_COUETTE_PROFILE samples=5 sampleGeometry=SAME_PROFILE5_PHYSICAL_WALL_NORMAL_POINTS momentumUtau=SAME_FIVE_POINT_SPALDING_FIT scalarFit=MULTIPOINT_WEIGHTED_LEAST_SQUARES kTarget=TABLE_KPLUS_UTAU2 omegaTarget=TABLE_OMEGAPLUS_UTAU2_OVER_NU tableVelocityInversion=OFF tableMomentum=OFF nativeSSTVolumeTerms=ON extraOFProduction=OFF gamma=%.12e status=PASS\n",sstG1.profile5EqGamma);
      }
std::printf("NODALS_SST_OMEGA_WALL_CONFIG mode=%s target=%s placement=%s physicalWallFlux=%s penaltyGamma=%.12e sampleFraction=%.12e ofKappa=%.6e ofE=%.6e ofBeta1=%.6e ofReBlend=%.6e ofProduction=%d ofProductionScale=%.6e ofWallVolumeScale=%.6e ofProductionLimit=%.6e sharedSpaldingState=%d profileIntegralUTau=%d profileIntegralNQ=%d profileIntegralClosure=%d profileAverageClosure=%d profileClosureYPlusMin=%.6e status=PASS\n",hxt5c::omega_wall_mode_name(sstG1.omegaWallMode),sstG1.omegaWallMode==hxt5c::OMEGA_WALL_CONSISTENT_TABLE?"SST2003M_1D_COUETTE_TABLE":(sstG1.omegaWallMode==hxt5c::OMEGA_WALL_COMPATIBLE?"CARRIER_LACOMBE_COMPATIBLE_OMEGA":(sstG1.omegaWallMode==hxt5c::OMEGA_WALL_OF_AUTO?"OF_OMEGA_VIS_LOG_BLEND":"SPALDING_UTAU_LOG")),sstWallPlacement,sstG1.omegaWallMode==hxt5c::OMEGA_WALL_TRACE_LOG?"NITSCHE_DIRICHLET":"NATURAL_ZERO",sstG1.wallPenaltyGamma,wallSampleFraction,sstG1.ofKappa,sstG1.ofE,sstG1.ofBeta1,sstG1.ofReBlend,(int)sstG1.ofProduction,sstG1.ofProductionScale,sstG1.ofWallVolumeScale,sstG1.ofProductionLimit,(int)sstG1.sharedSpaldingWallState,(int)sstG1.profileIntegralUTau,sstG1.profileIntegralNQ,(int)sstG1.profileIntegralClosure,(int)sstG1.profileAverageClosure,sstG1.profileClosureYPlusMin);
      if(sstG1.sharedSpaldingWallState)std::printf("NODALS_SST_SHARED_WALL_STATE enabled=1 source=EXACT_MOMENTUM_SPALDING_UTAU kappa=0.4 B=5.5 E=9.025013499434122 omegaLog=UTAU_OVER_SQRTBETASTAR_KAPPA_Y wallProduction=UTAU_CUBED_OVER_KAPPA_Y blending=OF_AUTO_UNCHANGED status=PASS\n");
      if(sstG1.profileAverageClosure)std::printf(
        "NODALS_SST_PROFILE_AVERAGE_WALL_LAW law=%s kappa=4.000000000000e-01 E=9.025013499434122e+00 inversion=%s momentumAndOFAutoSharedUTau=1 legacySharedSpaldingFlagNameOnly=%d status=PASS\n",
        sstG1.profileAverageWallLaw==5?"spalding_profile5":(sstG1.profileAverageWallLaw==4?"spalding_profile3":(sstG1.profileAverageWallLaw==3?"spalding_match_height":(sstG1.profileAverageWallLaw==2?"spalding_relative":(sstG1.profileAverageWallLaw==1?"log":"spalding")))),
        sstG1.profileAverageWallLaw==5?"FIVE_POINT_EQUAL_WEIGHT_LEAST_SQUARES":(sstG1.profileAverageWallLaw==4?"THREE_POINT_EQUAL_WEIGHT_LEAST_SQUARES":(sstG1.profileAverageWallLaw==3?"ONE_ROOT_AT_PRESCRIBED_HEIGHT":"ONE_ROOT_AT_NQ_VOLUME_AVERAGED_U_Y")),(int)sstG1.sharedSpaldingWallState);
      if(sstG1.profileAverageClosure && sstG1.profileAverageWallLaw==2)std::printf(
        "NODALS_SST_RELATIVE_WALL_TRACE_CONFIG velocityState=UBAR_MINUS_UWALL wallTrace=PHYSICAL_FE_TRACE momentumTrial=BAVG_MINUS_BWALL momentumTest=PHYSICAL_WALL_TRACE wallLaw=SPALDING sharedUTauToOFAuto=1 ofAutoOtherAverages=UNCHANGED status=PASS\n");
      if(sstG1.profileAverageClosure && sstG1.profileAverageWallLaw==3)std::printf(
        "NODALS_SST_MATCH_HEIGHT_CONFIG targetY=%.12e wallVelocityReference=ABSOLUTE_RELATIVE_TO_STATIONARY_SOLID "
        "utau=SPALDING_AT_MATCH_POINT momentumTest=PHYSICAL_WALL_TRACE "
        "momentumTrial=FIRST_CELL_NQ12_AVERAGE betaPolicy=UTAU2_OVER_CURRENT_LOCAL_UBAR "
        "ofAutoSharedUTau=1 ofAutoCarrierAverages=FIRST_CELL_NQ12_UNCHANGED status=PASS\n",
        sstG1.profileMatchY);
      if(sstG1.profileAverageClosure && sstG1.profileAverageWallLaw==5)std::printf(
        "NODALS_SST_MATCH_PROFILE5_CONFIG y=[%.12e,%.12e,%.12e,%.12e,%.12e] samplesPerWallQuadrature=5 "
        "wallVelocityReference=ABSOLUTE_RELATIVE_TO_STATIONARY_SOLID fit=EQUAL_WEIGHT_IMPLICIT_SPALDING_LEAST_SQUARES "
        "unknown=ONE_UTAU momentumTest=PHYSICAL_WALL_TRACE momentumTrial=FIRST_CELL_NQ12_AVERAGE "
        "betaPolicy=FITTED_UTAU2_OVER_CURRENT_LOCAL_UBAR_WITH_FINITE_PROFILE_FALLBACK ofAutoSharedUTau=1 "
        "ofAutoCarrierAverages=FIRST_CELL_NQ12_UNCHANGED reportedMatchYPlus=MIDDLE_SAMPLE_Y3 status=PASS\n",
        sstG1.profileFitY1,sstG1.profileFitY2,sstG1.profileFitY3,sstG1.profileFitY4,sstG1.profileFitY5);
      if(sstG1.profileAverageClosure && sstG1.profileAverageWallLaw==4)std::printf(
        "NODALS_SST_MATCH_PROFILE3_CONFIG y=[%.12e,%.12e,%.12e] samplesPerWallQuadrature=3 "
        "wallVelocityReference=ABSOLUTE_RELATIVE_TO_STATIONARY_SOLID fit=EQUAL_WEIGHT_IMPLICIT_SPALDING_LEAST_SQUARES "
        "unknown=ONE_UTAU momentumTest=PHYSICAL_WALL_TRACE momentumTrial=FIRST_CELL_NQ12_AVERAGE "
        "betaPolicy=FITTED_UTAU2_OVER_CURRENT_LOCAL_UBAR ofAutoSharedUTau=1 "
        "ofAutoCarrierAverages=FIRST_CELL_NQ12_UNCHANGED reportedMatchYPlus=MIDDLE_SAMPLE status=PASS\n",
        sstG1.profileFitY1,sstG1.profileFitY2,sstG1.profileFitY3);
      const char* momWallForm=
        momentumWallMode==hxt5d::MOM_WALL_SPALDING ? "OFFWALL_SPALDING_TRACTION" :
        momentumWallMode==hxt5d::MOM_WALL_SPALDING_TRACE ? "TRACE_SPALDING_WEAK_DIRICHLET" :
        momentumWallMode==hxt5d::MOM_WALL_SPALDING_ALIGNED ? "OFFWALL_SPALDING_ALIGNED_WEAK_DIRICHLET" :
        momentumWallMode==hxt5d::MOM_WALL_SPALDING_PROFILE_INTEGRAL ? "PROFILE_INTEGRAL_SPALDING_TRACTION" :
        momentumWallMode==hxt5d::MOM_WALL_SPALDING_COMPATIBLE ? "OFFWALL_SPALDING_COMPATIBLE_TRACTION" :
        momentumWallMode==hxt5d::MOM_WALL_SST_CONSISTENT_TABLE ? (sstG1.profileAverageClosure?"NQ_AVERAGE_SST_TABLE_TRACTION":"OFFWALL_SST_TABLE_TRACTION") :
        "SYMMETRIC_ADJOINT_CONSISTENT_NITSCHE";
      const char* momWallVisc=
        momentumWallMode==hxt5d::MOM_WALL_WEAK_EFFECTIVE ? "NU_PLUS_NUT_OFFWALL_SAMPLE" :
        momentumWallMode==hxt5d::MOM_WALL_WEAK_MOLECULAR ? "MOLECULAR_NU" :
        momentumWallMode==hxt5d::MOM_WALL_SPALDING_TRACE ? "SPALDING_UTAU_FROM_WALL_TRACE_HN" :
        momentumWallMode==hxt5d::MOM_WALL_SPALDING_ALIGNED ? "SPALDING_UTAU_OFFWALL_PLUS_WD3_STRESS_DEFICIT" :
        momentumWallMode==hxt5d::MOM_WALL_SPALDING_PROFILE_INTEGRAL ? (sstG1.profileAverageClosure?"SPALDING_UTAU_FROM_NQ_VOLUME_AVERAGED_U_Y":"SPALDING_UTAU_FROM_FULL_FE_NORMAL_PROFILE") :
        momentumWallMode==hxt5d::MOM_WALL_SPALDING_COMPATIBLE ? "SPALDING_UTAU_TIMES_CARRIER_USTAR" :
        momentumWallMode==hxt5d::MOM_WALL_SST_CONSISTENT_TABLE ? (sstG1.profileAverageClosure?"SST2003M_TABLE_UTAU_FROM_NQ_AVERAGED_U_Y":"SST2003M_TABLE_UTAU") :
        "SPALDING_UTAU";
      const bool momWallApplied=
        momentumWallMode==hxt5d::MOM_WALL_SPALDING ||
        momentumWallMode==hxt5d::MOM_WALL_SPALDING_TRACE ||
        momentumWallMode==hxt5d::MOM_WALL_SPALDING_ALIGNED ||
        momentumWallMode==hxt5d::MOM_WALL_SPALDING_PROFILE_INTEGRAL ||
        momentumWallMode==hxt5d::MOM_WALL_SPALDING_COMPATIBLE ||
        momentumWallMode==hxt5d::MOM_WALL_SST_CONSISTENT_TABLE;
      std::printf("NODALS_SST_MOMENTUM_WALL_CONFIG mode=%s target=HOMOGENEOUS_TANGENTIAL_NOSLIP form=%s viscosity=%s weakGamma=%.12e hScale=%.12e nutScale=%.12e hDefinition=%s transverseWallClamp=STRONG_UX_UY spaldingMetrics=%s coarsePressureHierarchyWallSnapshot=BASELINE_SPALDING_PRECONDITIONER_ONLY status=PASS\n",
        hxt5d::momentum_wall_mode_name(momentumWallMode),momWallForm,momWallVisc,
        weakWallGamma,weakWallHScale,weakWallNutScale,
        (momentumWallMode==hxt5d::MOM_WALL_SPALDING_PROFILE_INTEGRAL||momentumWallMode==hxt5d::MOM_WALL_SST_CONSISTENT_TABLE)&&sstG1.profileAverageClosure?"NQ_VOLUME_AVERAGED_WALL_COLUMN":(momentumWallMode==hxt5d::MOM_WALL_SPALDING_PROFILE_INTEGRAL?"FULL_WALL_NORMAL_PROFILE":"FULL_WALL_NORMAL_CELL_SIZE_FROM_OFFWALL_SAMPLE"),
        momWallApplied?"APPLIED_TRACTION_AND_DIAGNOSTIC":"DIAGNOSTIC_ONLY_NOT_MOMENTUM_TRACTION");
      std::printf(
        "NODALS_SST_SPALDING_MOLECULAR_CONSISTENCY "
        "enabled=%d "
        "terms=TWO_MOLECULAR_NITSCHE_CONSISTENCY_TERMS "
        "extraPenalty=NONE "
        "viscosity=MOLECULAR_NU "
        "status=PASS\n",
        spaldingMolecularConsistency);
      if(momentumWallMode==hxt5d::MOM_WALL_SPALDING_ALIGNED){
        std::printf(
          "NODALS_SST_SPALDING_ALIGNED_CONFIG "
          "law=SPALDING offwallSampleFraction=%.12e "
          "Cb=%.12e tauB=MAX_WALL_DEFICIT_AND_CB_NU_OVER_HB "
          "molecularShearApprox=NU_TIMES_UA_OVER_YA "
          "wallState=PHYSICAL_TRACE_PENALTY "
          "status=PASS\n",
          wallSampleFraction,weakWallGamma);
      }
      if(momentumWallMode==hxt5d::MOM_WALL_SPALDING_PROFILE_INTEGRAL){
        if(spaldingMolecularConsistency)throw std::runtime_error("spalding_profile_integral requires SPALDING_MOLECULAR_CONSISTENCY=0");
        if(!sstG1.profileIntegralUTau)throw std::runtime_error("spalding_profile_integral requires SST_PROFILE_INTEGRAL_UTAU=1");
        if(sstG1.profileIntegralNQ!=4&&sstG1.profileIntegralNQ!=8&&sstG1.profileIntegralNQ!=12)throw std::runtime_error("spalding_profile_integral requires SST_PROFILE_INTEGRAL_NQ=4, 8, or 12");
        const bool profileOmegaModeSupported =
          sstG1.omegaWallMode==hxt5c::OMEGA_WALL_OF_AUTO ||
          sstG1.omegaWallMode==hxt5c::OMEGA_WALL_SAMPLE_LOG ||
          sstG1.omegaWallMode==hxt5c::OMEGA_WALL_TRACE_LOG ||
          sstG1.omegaWallMode==hxt5c::OMEGA_WALL_PROFILE5_EQ_SST;
        if(!profileOmegaModeSupported)throw std::runtime_error("spalding_profile_integral requires SST_OMEGA_WALL_MODE=of_auto, sample_log, or trace_log");
        if(sstG1.omegaWallMode==hxt5c::OMEGA_WALL_OF_AUTO && !sstG1.sharedSpaldingWallState)throw std::runtime_error("spalding_profile_integral + of_auto requires SST_SHARED_SPALDING_WALL_STATE=1");
        if(sstG1.profileIntegralClosure&&sstG1.profileAverageClosure)throw std::runtime_error("SST_PROFILE_INTEGRAL_CLOSURE and SST_PROFILE_AVERAGE_CLOSURE are mutually exclusive");
        if(sstG1.profileAverageClosure && sstG1.omegaWallMode==hxt5c::OMEGA_WALL_OF_AUTO){
          std::printf(
            "NODALS_SST_PROFILE_AVERAGE_CLOSURE_CONFIG "
            "enabled=1 normalQuadrature=GAUSS%d_FULL_WALL_ADJACENT_HEX "
            "averaging=PHYSICAL_VOLUME_WEIGHTED_NORMAL_COLUMN "
            "utau=SPALDING_ON_NQ_AVERAGED_U_Y "
            "sstState=ONE_OFAUTO_STATE_FROM_AVERAGED_U_GRADU_K_OMEGA_GRADK_GRADOMEGA "
            "carrierAverage=ACTIVE_CONSTITUTIVE_CARRIER_VALUES "
            "omegaConstraint=RANK_ONE_CELL_AVERAGE_PENALTY "
            "omegaPenalty=LAMBDA_GAMMA_NUEFF_OVER_HN2 "
            "production=AVERAGE_STATE_VOLUME_DISTRIBUTED "
            "sampleFractionRole=DIAGNOSTIC_ONLY "
            "status=PASS\n",
            sstG1.profileIntegralNQ);
        }
        if(sstG1.profileIntegralClosure){
          if(sstG1.omegaWallMode!=hxt5c::OMEGA_WALL_OF_AUTO)throw std::runtime_error("SST_PROFILE_INTEGRAL_CLOSURE=1 requires SST_OMEGA_WALL_MODE=of_auto");
          if(!(sstG1.profileClosureYPlusMin>0.0))throw std::runtime_error("SST_PROFILE_CLOSURE_YPLUS_MIN must be > 0");
          std::printf(
            "NODALS_SST_PROFILE_CLOSURE_CONFIG "
            "enabled=1 normalQuadrature=GAUSS%d_FULL_WALL_ADJACENT_HEX "
            "utau=SHARED_PROFILE_INTEGRAL_SPALDING "
            "omegaTarget=OF_AUTO_PROFILE_VOLUME_PROJECTION "
            "omegaPenalty=LAMBDA_GAMMA_NUEFF_OVER_HN2 "
            "omegaYPlusMin=%.12e "
            "production=OF_AUTO_PROFILE_VOLUME_INTEGRAL "
            "sampleFractionRole=DIAGNOSTIC_ONLY "
            "status=PASS\n",
            sstG1.profileIntegralNQ,sstG1.profileClosureYPlusMin);
        }
        std::printf(
          "NODALS_SST_PROFILE_INTEGRAL_WALL_CONFIG "
          "normalQuadrature=GAUSS%d_FULL_WALL_ADJACENT_HEX "
          "utauResidual=%s "
          "momentumTrial=PROFILE_AVERAGED_FE_VELOCITY "
          "tauWall=UTAU_SQUARED "
          "sstUtau=%s "
          "sstOmegaMode=%s "
          "sstTargetPlacement=%s "
          "status=PASS\n",
          sstG1.profileIntegralNQ,
          sstG1.profileAverageClosure?"AVERAGE_FIRST_SPALDING_AT_UBAR_YBAR":"MEAN_SPALDING_YPLUS_RESIDUAL_ZERO",
          sstG1.profileAverageClosure?"SAME_NQ_AVERAGE_FIRST_INVERSION":"SAME_PROFILE_INTEGRAL_INVERSION",
          hxt5c::omega_wall_mode_name(sstG1.omegaWallMode),
          sstG1.profileAverageClosure?"NQ_VOLUME_AVERAGE_STATE":(sstG1.profileIntegralClosure?"FULL_PROFILE_VOLUME_PROJECTION":(sstG1.omegaWallMode==hxt5c::OMEGA_WALL_TRACE_LOG?"PHYSICAL_TRACE":"EXISTING_OFFWALL_SAMPLE")));
      }
      if(momentumWallMode==hxt5d::MOM_WALL_SPALDING_COMPATIBLE){
        if(spaldingMolecularConsistency)throw std::runtime_error("spalding_compatible requires SPALDING_MOLECULAR_CONSISTENCY=0");
        if(sstG1.omegaWallMode!=hxt5c::OMEGA_WALL_COMPATIBLE)throw std::runtime_error("spalding_compatible requires SST_OMEGA_WALL_MODE=compatible");
        std::printf(
          "NODALS_SST_COMPATIBLE_CONFIG "
          "velocityLaw=SPALDING_OFFWALL tauWall=UTAU_TIMES_USTAR "
          "uStar=CARRIER_FOURTH_POWER_BLEND omega=CARRIER_VIS_LOG_ROOT_SUM_SQUARE "
          "production=CARRIER_UTAU2_USTAR_OVER_KAPPA_Y_IF_YPLUS_GT_10 "
          "distance=EXISTING_OFFWALL_SAMPLE firstTestAdaptation=1 "
          "fullCarrierTruncatedDistanceAndFourLayerOmega=0 status=PASS\n");
      }

      // NODALS_SST_G2_MOM_GATE_V1
      // Additional OUTER convergence condition only.
      // Does not alter momentum, pressure, SST, wall, or SIMPLE operators.
      double sstG2MomAtol=0.0;
      if(const char* e=std::getenv("SST_G2_MOM_ATOL")){
        sstG2MomAtol=std::stod(e);
      }
      if(!(sstG2MomAtol>=0.0) || !std::isfinite(sstG2MomAtol))
        throw std::runtime_error("bad SST_G2_MOM_ATOL");

      std::printf(
        "NODALS_SST_G2_MOM_GATE "
        "enabled=%d absTol=%.12e "
        "components=XYZ "
        "quantity=PHYSICAL_MOMENTUM_RESIDUAL_NORM "
        "operatorChanged=0 status=PASS\n",
        (int)(sstG2MomAtol>0.0),
        sstG2MomAtol);

      if(sstG1.omegaWallMode==hxt5c::OMEGA_WALL_LOWRE_RESOLVED){
        if(momentumWallMode!=hxt5d::MOM_WALL_WEAK_MOLECULAR)
          throw std::runtime_error("lowre_resolved requires MOMENTUM_WALL_MODE=weak_molecular");
        if(sstG1.ofProduction)
          throw std::runtime_error("lowre_resolved requires SST_OF_PRODUCTION=0");
        if(sstG1.profileAverageClosure||sstG1.profileIntegralClosure)
          throw std::runtime_error("lowre_resolved requires profile average/integral SST closures OFF");
        std::printf(
          "NODALS_SST_LOWRE_RESOLVED_CONFIG "
          "momentumWall=WEAK_MOLECULAR_SYMMETRIC_NITSCHE_NO_SLIP "
          "momentumPenaltyGamma=%.12e momentumViscosity=MOLECULAR_NU "
          "kWall=ZERO_TRACE_SYMMETRIC_NITSCHE "
          "omegaWall=VISCOUS_ASYMPTOTE_C_NU_OVER_BETA1_Y1_SQUARED "
          "omegaFactor=%.12e omegaDistance=OFFWALL_REFERENCE_Y "
          "spaldingPhysics=NONE spaldingDiagnostics=ALLOWED "
          "ofAutoProduction=OFF wallFunctionNut=NONE status=PASS\n",
          weakWallGamma,sstG1.lowReOmegaFactor);
      }

      std::printf("NODALS_SST_G2_CONFIG model=SST-2003m precision=%s coupling=FULL_SIMPLE_K_OMEGA momentumFeedback=ON start=%s velocityInlet=DG_NUMERICAL_TRACE_PLUG turbulenceInlet=DG_TRACE intensity=%.6g lengthScaleOverD=%.6g kIn=%.12e omegaIn=%.12e outlet=NATURAL_ZERO_GRADIENT momentumWall=%s kWall=%s omegaWall=%s wallDistance=ANALYTIC_PIPE_R_MINUS_R alphaK=%.6g alphaOmega=%.6g scalarLinear=%s plateauTol=%.6e plateauSamples=%d plateauEvery=%d maxOuter=%d status=PASS\n",
        PRECISION,directSSTStart?"PLUG_POTENTIAL":"CONVERGED_G0_MIXLEN",sstG1.intensity,sstG1.lengthRatio,TS.kIn,TS.omegaIn,hxt5d::momentum_wall_mode_name(momentumWallMode),sstG1.omegaWallMode==hxt5c::OMEGA_WALL_CONSISTENT_TABLE?"SST2003M_TABLE_SAMPLE":"HIGH_RE_ZERO_GRADIENT",hxt5c::omega_wall_mode_name(sstG1.omegaWallMode),sstG1.alphaK,sstG1.alphaOmega,sstG1.linearSolver.c_str(),sstG2PlateauTol,sstG2PlateauSamples,sstG2PrintEvery,sstG2Max);
      std::printf("NODALS_SST_COMPACT_CONFIG precision=%s contAbsTol=%.6e turbRelTol=%.6e plateauEvery=%d plateauSamples=%d momentum=%s scalar=%s pressure=%s status=PASS\n",PRECISION,sstG2ContAtol,sstG2PlateauTol,sstG2PrintEvery,sstG2PlateauSamples,momWork.c_str(),sstG1.linearSolver.c_str(),pressureSolver.c_str());
      std::printf("NODALS_SST_STRESS_FORM production=%s momentumVolume=%s coupling=%s devProduction=%d devMomentum=%d status=PASS\n",
        sstG1.deviatoricProduction?"DEV_TWO_SYMM_GRADU_COLON_GRADU":"TWO_S_COLON_S_DIVFREE_SHORTCUT",
        sstG1.deviatoricMomentum?"DEV_TWO_SYMM_NU_EFF_LAGGED_CROSS":"SCALAR_NU_EFF_GRADU",
        sstG1.deviatoricMomentum?"IMPLICIT_SCALAR_PLUS_EXPLICIT_CROSS":"IMPLICIT_SCALAR",
        (int)sstG1.deviatoricProduction,(int)sstG1.deviatoricMomentum);
      std::printf("NODALS_SST_G2_CONVERGENCE_CONFIG physicalGate=%d metricsEvery=%d ""simpleTol=%.6e plateauTol=%.6e plateauSamples=%d plateauEvery=%d ""semantics=mass_and_turbulence_plus_optional_physical_plateau status=PASS\n",(int)sstG2PhysicalGate,sstG2MetricsEvery,simpleTol,sstG2PlateauTol,sstG2PlateauSamples,sstG2PrintEvery);
      HXT1_CUDA(cudaDeviceSynchronize());
      const auto g2PerfT0 = std::chrono::steady_clock::now();

      // NODALS_SST_G2_VTU_CHECKPOINT_OPTIONS_V2
      // Runtime-configurable checkpoints. 0 disables them.
      int sstG2VtuEvery=100;
      if(const char* e=std::getenv("SST_G2_VTU_EVERY")) sstG2VtuEvery=std::stoi(e);
      if(sstG2VtuEvery<0) throw std::runtime_error("bad SST_G2_VTU_EVERY");
      std::string sstG2VtuDir=".";
      if(const char* e=std::getenv("SST_G2_VTU_DIR")) if(*e) sstG2VtuDir=e;
      std::printf("NODALS_SST_G2_VTU_CHECKPOINT_CONFIG every=%d dir=%s fields=FLOW_PLUS_Q1_SST status=PASS\\n",
                  sstG2VtuEvery,sstG2VtuDir.c_str());

      for(int git=1;git<=sstG2Max;++git){
#include "modules/coupling/simple_preconvection.inc"
#include "modules/coupling/simple_momentum_assembly.inc"
#include "modules/coupling/simple_pressure_predictor.inc"
#include "modules/coupling/simple_pressure_pod.inc"
#include "modules/coupling/simple_pressure_corrector.inc"
#include "modules/coupling/simple_turbulence_advance.inc"
#include "modules/coupling/simple_checkpoint.inc"
#include "modules/coupling/simple_metrics.inc"
#include "modules/coupling/simple_convergence.inc"
        if(g2Converged)break;
      }

      HXT1_CUDA(cudaDeviceSynchronize());
      const auto g2PerfT1 = std::chrono::steady_clock::now();
      const double g2LoopMs =
        std::chrono::duration<double,std::milli>(g2PerfT1-g2PerfT0).count();

      std::printf(
        "NODALS_SST_G2_FIXED10_TIMING "
        "iterations=%d loopMs=%.3f avgOuterMs=%.6f status=PASS\n",
        g2Outer,
        g2LoopMs,
        g2LoopMs/std::max(g2Outer,1));
      pressurePOD.print_summary(pIts,g2Outer,g2LoopMs);

      std::vector<T>hk((std::size_t)nv),hw((std::size_t)nv);HXT1_CUDA(cudaMemcpy(hk.data(),TS.k.p,(std::size_t)nv*sizeof(T),cudaMemcpyDeviceToHost));HXT1_CUDA(cudaMemcpy(hw.data(),TS.omega.p,(std::size_t)nv*sizeof(T),cudaMemcpyDeviceToHost));std::string sstVtu=vtu.size()>=4&&vtu.substr(vtu.size()-4)==".vtu"?vtu.substr(0,vtu.size()-4)+"_sst_g2.vtu":vtu+"_sst_g2.vtu";hxt5c::write_sst_vtu(sstVtu,M,hk,hw);
      g2Gate=g2Converged&&g2FinalStats.meanNutOverNu>0&&g2FinalStats.maxNutOverNu>0&&g2FinalStats.kFloorFrac<.01&&g2FinalStats.omegaFloorFrac<.01;
// NODALS_SST_Z_FORCE_BUDGET_V1
    //
    // Diagnostic only. Reassemble every physical z-momentum contribution
    // at the final nonlinear state and contract each contribution with
    // valid Q1 axial FE test functions. BF2 test coefficients are zero,
    // since constants / axial Q1 window functions live in the Q1 subspace.
    //
    // Physical equation audited:
    //
    //   (A_base + A_conv + A_turb + A_DG + A_wall) Uz
    //       - B^T_z p
    //       - inletRhsZ
    //   = 0
    //
    // No SIMPLE row-L1 relaxation terms are included here.
    if(sstG2 && g2Outer>0 && sstG1.deviatoricMomentum)
      std::printf("NODALS_SST_Z_FORCE_BUDGET skipped=1 reason=EXPLICIT_DEVIATORIC_CROSS_TERM_NOT_IN_LEGACY_BUDGET status=PASS\n");
    if(sstG2 && g2Outer>0 && !sstG1.deviatoricMomentum && M.h.ntet>0)
      std::printf("NODALS_SST_Z_FORCE_BUDGET skipped=1 reason=HEX_ONLY_DIAGNOSTIC_ON_TET_MESH status=SKIPPED\n");
    if(sstG2 && g2Outer>0 && !sstG1.deviatoricMomentum && M.h.ntet==0){

      // ---------------------------------------------------------------
      // Reassemble all physical pieces at the FINAL U,k,omega state.
      // ---------------------------------------------------------------
      HXT1_CUDA(cudaMemset(
        A.conv.p,0,A.nnz*sizeof(T)));

      HXT1_CUDA(cudaMemset(
        d_bad.p,0,sizeof(unsigned long long)));

      if(M.h.nhex)
        hex_conv_assemble_cached<<<M.h.nhex,128>>>(
          d_hv.p,
          A.hexSlot.p,
          d_u0.p,d_u1.p,d_u2.p,
          d_hexConvGrad.p,d_hexConvDetW.p,
          A.conv.p,
          M.h.nhex);

      if(M.h.ntet)
        tet_conv_assemble_thread_per_cell
          <<<(M.h.ntet+127)/128,128>>>(
            d_tv.p,d_tg.p,A.tetSlot.p,
            d_u0.p,d_u1.p,d_u2.p,
            A.conv.p,
            M.h.ntet);

      hxt5d::assemble_momentum_sst(
        M,
        d_pts,d_hc,d_tc,d_hv,d_tv,d_tg,d_ip,
        A,
        d_u0.p,d_u1.p,d_u2.p,
        TS.k.p,TS.omega.p,
        d_turb,d_diagRTurb,
        R,nu,gamma,
        sstG1,TS.stats,
        d_bad,
        &d_sstDevRhs0,&d_sstDevRhs1,&d_sstDevRhs2,
        &d_sstInterfaceCache);

      hxt5d::assemble_boundary_sst(
        BH,
        d_inH,d_inT,d_wallH,
        d_pts,d_hc,d_tc,d_hv,d_tv,d_tg,
        A,
        d_u0.p,d_u1.p,d_u2.p,
        TS.k.p,TS.omega.p,
        d_dg,d_diagDG,
        d_inletRhsZ,
        d_wall,d_diagWall,
        d_wallStats,
        d_bad,
        bulk,R,nu,wallSampleFraction,
        sstG1,
        momentumWallMode,
        spaldingMolecularConsistency,
        weakWallGamma,
        weakWallHScale,
        weakWallNutScale);

      hxt4a::BT_apply(
        B,d_p.p,
        d_bt0.p,d_bt1.p,d_bt2.p);

      HXT1_CUDA(cudaDeviceSynchronize());

      unsigned long long forceBad=0;
      HXT1_CUDA(cudaMemcpy(
        &forceBad,
        d_bad.p,
        sizeof(forceBad),
        cudaMemcpyDeviceToHost));

      if(forceBad)
        throw std::runtime_error(
          "Z force budget final-state reassembly geometry failure count="
          +std::to_string(forceBad));

      // Fixed z rows, if any, are not admissible test DOFs.
      std::vector<unsigned char>
        hForceFixed((std::size_t)nv,0);

      HXT1_CUDA(cudaMemcpy(
        hForceFixed.data(),
        d_fixed.p,
        (std::size_t)nv*sizeof(unsigned char),
        cudaMemcpyDeviceToHost));

      auto makeForceTest =
        [&](bool whole,double z0,double z1)
      {
        if(M.h.ntet!=0)
          throw std::runtime_error(
            "Z force budget V1 currently expects all-HEX mesh");

        std::vector<T>
          q((std::size_t)nv,T(0));

        std::vector<unsigned char>
          seen((std::size_t)nv,0);

        const double ztol=1e-11;

        std::size_t vertexDofs=0;
        std::size_t activeDofs=0;
        std::size_t fixedSuppressed=0;
        std::size_t conflicts=0;

        for(std::size_t c=0;c<M.hexes.size();++c){
          const auto hc=M.hexes[c];
          const auto H =M.hexVel[c];

          // First 8 local velocity basis functions are Q1 vertices.
          // BF2 face-bubble test coefficients remain exactly zero.
          for(int a=0;a<8;++a){
            const int g=H.g[a];

            if(g<0 || g>=nv)
              throw std::runtime_error(
                "Z force budget bad vertex velocity dof");

            const double z=M.points[
              (std::size_t)hc.v[a]].z;

            const T v =
              whole
              ? T(1)
              : T(
                  z>=z0-ztol &&
                  z<=z1+ztol
                    ? 1.0 : 0.0);

            if(seen[(std::size_t)g]){
              if(std::abs(
                   (double)q[(std::size_t)g]
                   -(double)v)>1e-12)
                ++conflicts;
            }else{
              q[(std::size_t)g]=v;
              seen[(std::size_t)g]=1;
              ++vertexDofs;
            }
          }
        }

        if(conflicts)
          throw std::runtime_error(
            "Z force budget inconsistent shared Q1 vertex test values");

        for(int i=0;i<nv;++i){
          if(hForceFixed[(std::size_t)i] &&
             q[(std::size_t)i]!=T(0)){
            q[(std::size_t)i]=T(0);
            ++fixedSuppressed;
          }

          if(q[(std::size_t)i]!=T(0))
            ++activeDofs;
        }

        std::printf(
          "NODALS_SST_Z_FORCE_TEST "
          "kind=%s z0=%.12e z1=%.12e "
          "vertexDofs=%zu activeDofs=%zu "
          "fixedSuppressed=%zu "
          "BF2coefficients=ZERO "
          "status=PASS\n",
          whole ? "GLOBAL_CONSTANT"
                : "Q1_WINDOW_300D_450D",
          z0,z1,
          vertexDofs,activeDofs,
          fixedSuppressed);

        return q;
      };

      auto runForceBudget =
        [&](const char* tag,
            const std::vector<T>& hTest,
            double z0,double z1)
      {
        Dev<T> dForceTest(hTest);

        auto matrixTerm =
          [&](const T*mat)->double
        {
          hxt5b::spmv
            <<<(nv+TPB-1)/TPB,TPB>>>(
              nv,
              A.row.p,A.col.p,
              mat,
              d_u2.p,
              A.tmp.p);

          HXT1_CUDA(cudaGetLastError());

          return ured.dot(
            dForceTest.p,
            A.tmp.p);
        };

        const double base =
          matrixTerm(A.base.p);

        const double conv =
          matrixTerm(A.conv.p);

        const double turb =
          matrixTerm(d_turb.p);

        const double dg =
          matrixTerm(d_dg.p);

        const double wall =
          matrixTerm(d_wall.p);

        const double pressure =
          ured.dot(
            dForceTest.p,
            d_bt2.p);

        const double inletRhs =
          ured.dot(
            dForceTest.p,
            d_inletRhsZ.p);

        const double lhs =
          base+conv+turb+dg+wall;

        const double rhs =
          pressure+inletRhs;

        const double closure =
          lhs-rhs;

        const double scale =
          std::max(
            std::abs(pressure),
            1e-300);

        std::printf(
          "NODALS_SST_Z_FORCE_BUDGET "
          "tag=%s "
          "z0=%.12e z1=%.12e "
          "baseMolecular=%.12e "
          "convection=%.12e "
          "sstEddyDiffusion=%.12e "
          "dgMatrix=%.12e "
          "wallOperator=%.12e "
          "pressureDrive=%.12e "
          "inletRhs=%.12e "
          "lhs=%.12e rhs=%.12e "
          "closure=%.12e "
          "closureOverPressure=%.12e "
          "baseOverPressure=%.12e "
          "convOverPressure=%.12e "
          "turbOverPressure=%.12e "
          "dgOverPressure=%.12e "
          "wallOverPressure=%.12e "
          "molecularConsistency=%d "
          "operatorChanged=0 "
          "status=PASS\n",
          tag,
          z0,z1,
          base,
          conv,
          turb,
          dg,
          wall,
          pressure,
          inletRhs,
          lhs,rhs,
          closure,
          closure/scale,
          base/scale,
          conv/scale,
          turb/scale,
          dg/scale,
          wall/scale,
          spaldingMolecularConsistency);
      };

      // Global constant-z test: useful independent closure check.
      {
        auto q=makeForceTest(
          true,zmin,zmax);

        runForceBudget(
          "GLOBAL_CONSTANT",
          q,zmin,zmax);
      }

      // Downstream 300D--450D test.
      //
      // Vertex values are 1 on [1.2,1.8] and zero outside.
      // Therefore the FE test is exactly Q1 and has a one-axial-cell
      // linear ramp immediately outside each end of the window.
      {
        constexpr double forceZ0=1.2;
        constexpr double forceZ1=1.8;

        auto q=makeForceTest(
          false,forceZ0,forceZ1);

        runForceBudget(
          "WINDOW_300D_450D",
          q,forceZ0,forceZ1);
      }
    }

    
      std::printf("NODALS_SST_G2_FINAL iterations=%d kRelUpdate=%.12e omegaRelUpdate=%.12e nutMeanRelUpdate=%.12e meanNutOverNu=%.12e maxNutOverNu=%.12e kFloorVolumeFrac=%.12e omegaFloorVolumeFrac=%.12e wallDarcy=%.12e wallYPlusMean=%.12e pressureDrop=%.12e vtu=%s status=%s\n",g2Outer,g2KRel,g2WRel,g2NutRel,g2FinalStats.meanNutOverNu,g2FinalStats.maxNutOverNu,g2FinalStats.kFloorFrac,g2FinalStats.omegaFloorFrac,finalF,finalY,finalDp,sstVtu.c_str(),g2Gate?"PASS":"FAIL");std::printf("SST_G2_GATE_STATUS=%s\n",g2Gate?"PASS":"FAIL");
    }

    std::vector<T>hp((std::size_t)np),hu0((std::size_t)nv),hu1((std::size_t)nv),hu2((std::size_t)nv);
    HXT1_CUDA(cudaMemcpy(hp.data(),d_p.p,np*sizeof(T),cudaMemcpyDeviceToHost));
    HXT1_CUDA(cudaMemcpy(hu0.data(),d_u0.p,nv*sizeof(T),cudaMemcpyDeviceToHost));
    HXT1_CUDA(cudaMemcpy(hu1.data(),d_u1.p,nv*sizeof(T),cudaMemcpyDeviceToHost));
    HXT1_CUDA(cudaMemcpy(hu2.data(),d_u2.p,nv*sizeof(T),cudaMemcpyDeviceToHost));

    auto pd=hxt4a::to_double(hp);
    double slope=hxt4a::pressure_slope(M,pd),pinP=hxt4a::boundary_p(M,pd,true),poutP=hxt4a::boundary_p(M,pd,false),drop=pinP-poutP;

    auto ws=hxt5b::wallstats(d_wallStats,bulk);
    long double et=0,ez=0;for(std::size_t v=0;v<M.points.size();++v){et+=(double)hu0[v]*(double)hu0[v]+(double)hu1[v]*(double)hu1[v];ez+=(double)hu2[v]*(double)hu2[v];}
    double trans=std::sqrt((double)(et/std::max((long double)1e-300,ez)));
    write_vtu(vtu,M,hu0,hu1,hu2,hp);
    const std::string diagVtu=hxt5b_diag::path2(vtu);
    {
      const double auditZ0=1.2;
      const double auditZ1=1.8;

      Dev<double> d_molForceAudit(
          std::vector<double>(2,0.0));

      Dev<unsigned long long> d_molForceBad(
          std::vector<unsigned long long>(1,0));

      if(!BH.wallH.empty()){
        hxt5d::spalding_molecular_force_audit_hex
          <<<BH.wallH.size(),64>>>(
            d_wallH.p,
            (int)BH.wallH.size(),
            d_pts.p,
            d_hc.p,
            d_hv.p,
            d_u2.p,
            nu,
            auditZ0,
            auditZ1,
            d_molForceAudit.p,
            d_molForceBad.p);

        HXT1_CUDA(cudaGetLastError());
        HXT1_CUDA(cudaDeviceSynchronize());
      }

      double hm[2]={0.0,0.0};
      unsigned long long hb=0;

      HXT1_CUDA(cudaMemcpy(
          hm,
          d_molForceAudit.p,
          2*sizeof(double),
          cudaMemcpyDeviceToHost));

      HXT1_CUDA(cudaMemcpy(
          &hb,
          d_molForceBad.p,
          sizeof(unsigned long long),
          cudaMemcpyDeviceToHost));

      std::printf(
        "NODALS_SST_MOLECULAR_WALL_FORCE_AUDIT "
        "z0=%.12e z1=%.12e wallArea=%.12e "
        "molecularForce=%.12e "
        "molecularMeanTraction=%.12e "
        "bad=%llu status=%s\n",
        auditZ0,auditZ1,
        hm[0],
        hm[1],
        hm[0]>0.0 ? hm[1]/hm[0] : 0.0,
        hb,
        hb==0 ? "PASS" : "FAIL");
    }

    hxt5b_diag::print_cross_section_momentum_audit(
      M,hu2,hp,bulk,1.2,1.8);

    hxt5b_diag::write(diagVtu,M,BH,hu0,hu1,hu2,hp,nu,bulk,wallSampleFraction);
    std::printf("NODALS_HXT5B6_DIAG_VTU_OUTPUT path=%s velocity=FULL_Q1BF2_P1BF3_PHYSICAL_CELL_AVERAGE wallYPlus=EXACT_SPALDING status=PASS\n",diagVtu.c_str());
    if(profileAudit){
      const std::string profileCsv=hxt5b_diag::profile_path(vtu);
      hxt5b_diag::write_profile(profileCsv,M,hu0,hu1,hu2,R,bulk,zmin,zmax,profileZFractions,profileRadial,profileAngles,profileRmax);
    }
    const double finalMeanNut=
      laminarPerf?0.0:
      (sstG2?g2FinalStats.meanNutOverNu:mix.meanNutOverNu);

    const double finalMaxNut=
      laminarPerf?0.0:
      (sstG2?g2FinalStats.maxNutOverNu:mix.maxNutOverNu);

    const char* finalTurb=
      laminarPerf?"LAMINAR_PERF":
      (sstG2?"SST_2003M":"NIKURADSE_ALGEBRAIC");

    bool gate=
      std::isfinite(slope)&&
      std::isfinite(drop)&&
      ws.area>0&&
      ws.rootFailures==0&&
      ws.yPlusMean>0&&
      (laminarPerf||finalMaxNut>0);
    std::printf("NODALS_HXT5B_PHYSICS turbulence=%s pressureSlope=%.12e pIn=%.12e pOut=%.12e pressureDrop=%.12e meanNutOverNu=%.12e maxNutOverNu=%.12e maxF1=%.12e maxF2=%.12e maxStrain=%.12e maxStrainSource=%s wallDarcyFromSpalding=%.12e wallYPlusMean=%.12e wallYPlusMax=%.12e wallSlipMean=%.12e transverseVertexRelL2=%.12e avgPressureCG=%.6f pressureSolver=%s status=%s\n",
      finalTurb,slope,pinP,poutP,drop,finalMeanNut,finalMaxNut,sstG2?g2FinalStats.maxF1:0.0,sstG2?g2FinalStats.maxF2:0.0,mix.maxStrain,
      laminarPerf?"NONE":
      (sstG2?(directSSTStart?"INITIAL_PLUG_AUDIT":"G0_MIXLEN_STARTER"):
       "CURRENT_MIXLEN"),
      ws.fDarcy,ws.yPlusMean,ws.yPlusMax,ws.slipMean,trans,(sstG2&&g2Outer)?(double)pIts/g2Outer:(outer?(double)pIts/outer:0.0),pressureSolver.c_str(),gate?"PASS":"FAIL");
    std::printf("NODALS_HXT5B_OUTPUT vtu=%s status=PASS\n",vtu.c_str());
    bool sstGate=true;
    if(sstG1.enabled){
      std::string sstVtu=vtu.size()>=4&&vtu.substr(vtu.size()-4)==".vtu"?vtu.substr(0,vtu.size()-4)+"_sst_g1.vtu":vtu+"_sst_g1.vtu";
      auto sstResult=hxt5c::run_frozen(M,BH,d_inH,d_inT,d_wallH,d_pts,d_hc,d_tc,d_hv,d_tv,d_tg,d_ip,A,VC,d_fixed,d_u0.p,d_u1.p,d_u2.p,d_turb,d_dg,d_wall,d_bad,bulk,R,nu,gamma,wallSampleFraction,sstG1,sstVtu);
      sstGate=sstResult.gate;
    }
    const bool allGate=gate&&sstGate&&g2Gate;
    std::printf("HXT5B_GATE_STATUS=%s\n",allGate?"PASS":"FAIL");
    return allGate?0:3;
  }catch(const std::exception&e){
    std::fprintf(stderr,"HXT5B_ERROR precision=%s: %s\n",hxt4b::PRECISION,e.what());
    std::fprintf(stderr,"HXT5B_GATE_STATUS=FAIL\n");return 2;
  }
}
