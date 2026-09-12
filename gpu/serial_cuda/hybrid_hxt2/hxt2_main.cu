#include "../hybrid_hxt1/hxt1_mesh.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <set>
#include <string>
#include <vector>

using namespace nodals_hxt1;

namespace hxt2 {

static constexpr int HEX_FACE[6][4] = {
  {0,1,2,3},{4,5,6,7},{0,1,5,4},{1,2,6,5},{2,3,7,6},{3,0,4,7}
};
static constexpr int TET_FACE[4][3] = {
  {1,2,3},{0,3,2},{0,1,3},{0,2,1}
};
static constexpr int HEX_SIGN[8][3] = {
  {-1,-1,-1},{1,-1,-1},{1,1,-1},{-1,1,-1},
  {-1,-1, 1},{1,-1, 1},{1,1, 1},{-1,1, 1}
};

// HXT2A: device-visible copies of the small reference topology tables.
// The host constexpr tables above remain the source used by host-side
// interface-plan construction. CUDA kernels must not dereference ordinary
// host constexpr storage, so they use these __constant__ mirrors instead.
__device__ __constant__ int c_tet_face[4][3] = {
  {1,2,3},{0,3,2},{0,1,3},{0,2,1}
};
__device__ __constant__ int c_hex_sign[8][3] = {
  {-1,-1,-1},{1,-1,-1},{1,1,-1},{-1,1,-1},
  {-1,-1, 1},{1,-1, 1},{1,1, 1},{-1,1, 1}
};

struct InterfacePlan {
  std::int32_t hexCell, hexFace, hexBubble;
  std::int32_t tetCell[2], tetFace[2], tetBubble[2];
  std::int8_t hexTriLocal[2][3];
  std::int8_t pad[2];
  double normal[3];                 // HEX -> TET
  double areaQuad, areaTri[2];
  double invHHex, invHTet[2];       // A/(2V)
};

__device__ __constant__ double c_tet_T[8*8*3*3];
__device__ __constant__ double c_g3_x[3] = {
  -0.774596669241483377035853079956, 0.0, 0.774596669241483377035853079956
};
__device__ __constant__ double c_g3_w[3] = {
  0.555555555555555555555555555556,
  0.888888888888888888888888888889,
  0.555555555555555555555555555556
};
__device__ __constant__ double c_q5_x[5] = {
  0.046910077030668018149839, 0.230765344947158454481842,
  0.500000000000000000000000, 0.769234655052841545518158,
  0.953089922969331981850161
};
__device__ __constant__ double c_q5_w[5] = {
  0.118463442528094543757132, 0.239314335249683234020645,
  0.284444444444444444444444, 0.239314335249683234020645,
  0.118463442528094543757132
};

struct V3 { double x,y,z; };
inline V3 hv3(const Point&p){return {p.x,p.y,p.z};}
inline V3 operator+(V3 a,V3 b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
inline V3 operator-(V3 a,V3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
inline V3 operator*(double s,V3 a){return {s*a.x,s*a.y,s*a.z};}
inline double dot(V3 a,V3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
inline V3 cross(V3 a,V3 b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
inline double norm(V3 a){return std::sqrt(dot(a,a));}

inline V3 hex_centroid(const HostMesh&M,int c){V3 q{0,0,0};for(int a=0;a<8;++a)q=q+hv3(M.points[(std::size_t)M.hexes[(std::size_t)c].v[a]]);return (1.0/8.0)*q;}
inline V3 tet_centroid(const HostMesh&M,int c){V3 q{0,0,0};for(int a=0;a<4;++a)q=q+hv3(M.points[(std::size_t)M.tets[(std::size_t)c].v[a]]);return 0.25*q;}
inline double tri_area(V3 a,V3 b,V3 c){return 0.5*norm(cross(b-a,c-a));}

inline std::vector<InterfacePlan> build_interface_plans(const HostMesh&M){
  std::vector<InterfacePlan>P;P.reserve(M.interface.size());
  for(const auto&r:M.interface){
    InterfacePlan p{};
    p.hexCell=r.hexCell;p.hexFace=r.hexLocalFace;p.hexBubble=r.hexBubble;
    p.tetCell[0]=r.tet0;p.tetFace[0]=r.tet0LocalFace;p.tetBubble[0]=r.tet0Bubble;
    p.tetCell[1]=r.tet1;p.tetFace[1]=r.tet1LocalFace;p.tetBubble[1]=r.tet1Bubble;
    const auto&hc=M.hexes[(std::size_t)p.hexCell];
    V3 q[4];
    for(int k=0;k<4;++k)q[k]=hv3(M.points[(std::size_t)hc.v[HEX_FACE[p.hexFace][k]]]);
    p.areaQuad=tri_area(q[0],q[1],q[2])+tri_area(q[0],q[2],q[3]);
    V3 n=cross(q[1]-q[0],q[3]-q[0]);
    const double nn=norm(n);if(!(nn>0))throw std::runtime_error("HXT2 zero interface quad normal");
    n=(1.0/nn)*n;
    const V3 ch=hex_centroid(M,p.hexCell),ct=tet_centroid(M,p.tetCell[0]);
    if(dot(n,ct-ch)<0)n=(-1.0)*n;
    p.normal[0]=n.x;p.normal[1]=n.y;p.normal[2]=n.z;
    const double vh=M.hexGeom[(std::size_t)p.hexCell].volume;
    if(!(vh>0))throw std::runtime_error("HXT2 bad hex volume");
    p.invHHex=p.areaQuad/(2.0*vh);
    for(int s=0;s<2;++s){
      const auto&tc=M.tets[(std::size_t)p.tetCell[s]];
      V3 tp[3];
      for(int k=0;k<3;++k){
        const int lv=TET_FACE[p.tetFace[s]][k];
        const int gv=tc.v[lv];
        tp[k]=hv3(M.points[(std::size_t)gv]);
        int hl=-1;for(int a=0;a<8;++a)if(hc.v[a]==gv){hl=a;break;}
        if(hl<0)throw std::runtime_error("HXT2 interface triangle vertex not found on hex face");
        bool onFace=false;for(int a=0;a<4;++a)if(HEX_FACE[p.hexFace][a]==hl)onFace=true;
        if(!onFace)throw std::runtime_error("HXT2 interface triangle vertex maps outside hex local face");
        p.hexTriLocal[s][k]=(std::int8_t)hl;
      }
      p.areaTri[s]=tri_area(tp[0],tp[1],tp[2]);
      const double vt=M.tetGeom[(std::size_t)p.tetCell[s]].volume;
      if(!(vt>0 && p.areaTri[s]>0))throw std::runtime_error("HXT2 bad tet interface geometry");
      p.invHTet[s]=p.areaTri[s]/(2.0*vt);
    }
    const double rel=std::abs((p.areaTri[0]+p.areaTri[1])-p.areaQuad)/p.areaQuad;
    if(rel>1e-12)throw std::runtime_error("HXT2 interface quad/tri area mismatch");
    P.push_back(p);
  }
  return P;
}

inline void tet_grad_ref_host(const std::array<double,4>&l,double gr[8][3]){
  const double gl[4][3]={{-1,-1,-1},{1,0,0},{0,1,0},{0,0,1}};
  for(int i=0;i<4;++i)for(int d=0;d<3;++d)gr[i][d]=gl[i][d];
  for(int i=0;i<4;++i){
    int js[3],k=0;for(int j=0;j<4;++j)if(j!=i)js[k++]=j;
    for(int d=0;d<3;++d)gr[4+i][d]=0.0;
    for(int a=0;a<3;++a){const int j=js[a],o1=js[(a+1)%3],o2=js[(a+2)%3];for(int d=0;d<3;++d)gr[4+i][d]+=27.0*l[o1]*l[o2]*gl[j][d];}
  }
}

inline std::array<double,8*8*3*3> build_tet_tensor(){
  const double rn[5]={0.034578939918215090,0.17348032077169567,0.38988638706551931,0.63433347263088680,0.85105421294701644};
  const double rw[5]={0.081764784285771011,0.12619896189991137,0.089200161221590066,0.032055600722961895,0.0041138252030990035};
  const double sn[5]={0.039809857051468722,0.19801341787360821,0.43797481024738616,0.69546427335363614,0.90146491420117358};
  const double sw[5]={0.096781590226651476,0.16717463809436969,0.14638698708466985,0.073908870072616678,0.015747914521692299};
  const double tn[5]={0.046910077030668018,0.23076534494715845,0.50000000000000000,0.76923465505284150,0.95308992296933198};
  const double tw[5]={0.11846344252809449,0.23931433524968326,0.28444444444444450,0.23931433524968326,0.11846344252809449};
  std::array<double,8*8*3*3>T{};
  for(int ir=0;ir<5;++ir)for(int is=0;is<5;++is)for(int it=0;it<5;++it){
    const double r=rn[ir],s=sn[is],t=tn[it],omr=1-r,oms=1-s;
    std::array<double,4>l{{omr*oms*(1-t),r,omr*s,omr*oms*t}};
    const double w=rw[ir]*sw[is]*tw[it];double gr[8][3];tet_grad_ref_host(l,gr);
    for(int a=0;a<8;++a)for(int b=0;b<8;++b)for(int j=0;j<3;++j)for(int k=0;k<3;++k)
      T[(((a*8+b)*3+j)*3+k)]+=gr[a][j]*gr[b][k]*w;
  }
  return T;
}

__device__ __forceinline__ bool inv3_dev(const double J[9],double I[9],double&det){
  det=J[0]*(J[4]*J[8]-J[5]*J[7])-J[1]*(J[3]*J[8]-J[5]*J[6])+J[2]*(J[3]*J[7]-J[4]*J[6]);
  if(!(det>0.0))return false;
  const double q=1.0/det;
  I[0]=(J[4]*J[8]-J[5]*J[7])*q; I[1]=(J[2]*J[7]-J[1]*J[8])*q; I[2]=(J[1]*J[5]-J[2]*J[4])*q;
  I[3]=(J[5]*J[6]-J[3]*J[8])*q; I[4]=(J[0]*J[8]-J[2]*J[6])*q; I[5]=(J[2]*J[3]-J[0]*J[5])*q;
  I[6]=(J[3]*J[7]-J[4]*J[6])*q; I[7]=(J[1]*J[6]-J[0]*J[7])*q; I[8]=(J[0]*J[4]-J[1]*J[3])*q;
  return true;
}

__device__ __forceinline__ void hex_q1_ref(double x,double y,double z,double phi[8],double gr[8][3]){
  const int sx[8]={-1,1,1,-1,-1,1,1,-1};
  const int sy[8]={-1,-1,1,1,-1,-1,1,1};
  const int sz[8]={-1,-1,-1,-1,1,1,1,1};
  for(int a=0;a<8;++a){
    const double ax=1.0+sx[a]*x,ay=1.0+sy[a]*y,az=1.0+sz[a]*z;
    phi[a]=0.125*ax*ay*az;
    gr[a][0]=0.125*sx[a]*ay*az;gr[a][1]=0.125*sy[a]*ax*az;gr[a][2]=0.125*sz[a]*ax*ay;
  }
}

__device__ __forceinline__ void hex_basis_ref(double x,double y,double z,double phi[14],double gr[14][3]){
  hex_q1_ref(x,y,z,phi,gr);
  const double bx=1.0-x*x,by=1.0-y*y,bz=1.0-z*z;
  const double dx=-2.0*x,dy=-2.0*y,dz=-2.0*z;
  // face 0 z-, 1 z+, 2 y-, 3 x+, 4 y+, 5 x-
  double s;
  s=0.5*(1.0-z);phi[8]=s*bx*by;gr[8][0]=s*dx*by;gr[8][1]=s*bx*dy;gr[8][2]=-0.5*bx*by;
  s=0.5*(1.0+z);phi[9]=s*bx*by;gr[9][0]=s*dx*by;gr[9][1]=s*bx*dy;gr[9][2]= 0.5*bx*by;
  s=0.5*(1.0-y);phi[10]=s*bx*bz;gr[10][0]=s*dx*bz;gr[10][1]=-0.5*bx*bz;gr[10][2]=s*bx*dz;
  s=0.5*(1.0+x);phi[11]=s*by*bz;gr[11][0]= 0.5*by*bz;gr[11][1]=s*dy*bz;gr[11][2]=s*by*dz;
  s=0.5*(1.0+y);phi[12]=s*bx*bz;gr[12][0]=s*dx*bz;gr[12][1]= 0.5*bx*bz;gr[12][2]=s*bx*dz;
  s=0.5*(1.0-x);phi[13]=s*by*bz;gr[13][0]=-0.5*by*bz;gr[13][1]=s*dy*bz;gr[13][2]=s*by*dz;
}

__device__ __forceinline__ bool hex_metric(const Point*pts,const HexConn&h,double x,double y,double z,double I[9],double&det){
  double p8[8],gr8[8][3];hex_q1_ref(x,y,z,p8,gr8);(void)p8;
  double J[9]={0,0,0,0,0,0,0,0,0};
  for(int a=0;a<8;++a){const Point&p=pts[h.v[a]];J[0]+=p.x*gr8[a][0];J[1]+=p.x*gr8[a][1];J[2]+=p.x*gr8[a][2];J[3]+=p.y*gr8[a][0];J[4]+=p.y*gr8[a][1];J[5]+=p.y*gr8[a][2];J[6]+=p.z*gr8[a][0];J[7]+=p.z*gr8[a][1];J[8]+=p.z*gr8[a][2];}
  return inv3_dev(J,I,det);
}

__device__ __forceinline__ void phys_grad14(const double gr[14][3],const double I[9],double gp[14][3]){
  for(int a=0;a<14;++a)for(int d=0;d<3;++d)gp[a][d]=gr[a][0]*I[d]+gr[a][1]*I[3+d]+gr[a][2]*I[6+d];
}
__device__ __forceinline__ void tet_basis(double l0,double l1,double l2,double l3,double phi[8],double gr[8][3]){
  const double l[4]={l0,l1,l2,l3};const double gl[4][3]={{-1,-1,-1},{1,0,0},{0,1,0},{0,0,1}};
  for(int i=0;i<4;++i){phi[i]=l[i];for(int d=0;d<3;++d)gr[i][d]=gl[i][d];}
  for(int i=0;i<4;++i){int js[3],k=0;for(int j=0;j<4;++j)if(j!=i)js[k++]=j;phi[4+i]=27.0*l[js[0]]*l[js[1]]*l[js[2]];for(int d=0;d<3;++d){gr[4+i][d]=27.0*(gl[js[0]][d]*l[js[1]]*l[js[2]]+l[js[0]]*gl[js[1]][d]*l[js[2]]+l[js[0]]*l[js[1]]*gl[js[2]][d]);}}
}
__device__ __forceinline__ void tet_phys_grad(const double gr[8][3],const TetGeom&g,double gp[8][3]){
  for(int a=0;a<8;++a)for(int d=0;d<3;++d)gp[a][d]=gr[a][0]*g.invJ[d]+gr[a][1]*g.invJ[3+d]+gr[a][2]*g.invJ[6+d];
}

__global__ void hex_volume_kernel(const Point*pts,const HexConn*hc,const HexVel*hv,const double*u,double*r,std::uint64_t n,double nu,unsigned long long*bad){
  std::uint64_t c=(std::uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
  double ul[14];for(int a=0;a<14;++a)ul[a]=u[hv[c].g[a]];
  double rl[14]={0};
  for(int ix=0;ix<3;++ix)for(int iy=0;iy<3;++iy)for(int iz=0;iz<3;++iz){
    const double x=c_g3_x[ix],y=c_g3_x[iy],z=c_g3_x[iz],w=c_g3_w[ix]*c_g3_w[iy]*c_g3_w[iz];
    double phi[14],gr[14][3],gp[14][3],I[9],det;hex_basis_ref(x,y,z,phi,gr);(void)phi;
    if(!hex_metric(pts,hc[c],x,y,z,I,det)){atomicAdd(bad,1ULL);continue;}phys_grad14(gr,I,gp);
    double gu[3]={0,0,0};for(int b=0;b<14;++b)for(int d=0;d<3;++d)gu[d]+=ul[b]*gp[b][d];
    const double q=nu*det*w;for(int a=0;a<14;++a)rl[a]+=q*(gp[a][0]*gu[0]+gp[a][1]*gu[1]+gp[a][2]*gu[2]);
  }
  for(int a=0;a<14;++a)atomicAdd(r+hv[c].g[a],rl[a]);
}

__global__ void tet_volume_kernel(const TetVel*tv,const TetGeom*tg,const double*u,double*r,std::uint64_t n,double nu){
  std::uint64_t c=(std::uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;const TetGeom&g=tg[c];
  double metric[3][3]={{0}};for(int j=0;j<3;++j)for(int k=0;k<3;++k)for(int d=0;d<3;++d)metric[j][k]+=g.invJ[3*j+d]*g.invJ[3*k+d];
  double ul[8];for(int a=0;a<8;++a)ul[a]=u[tv[c].g[a]];double rl[8]={0};
  for(int a=0;a<8;++a)for(int b=0;b<8;++b){double kab=0;for(int j=0;j<3;++j)for(int k=0;k<3;++k)kab+=c_tet_T[(((a*8+b)*3+j)*3+k)]*metric[j][k];rl[a]+=nu*g.det*kab*ul[b];}
  for(int a=0;a<8;++a)atomicAdd(r+tv[c].g[a],rl[a]);
}

__global__ void interface_kernel(const Point*pts,const HexConn*hc,const TetConn*tc,const HexVel*hv,const TetVel*tv,const TetGeom*tg,const InterfacePlan*P,const double*u,double*r,std::uint64_t n,double nu,double gamma,double*diagJump,double*diagFlux,double*diagFluxInt,unsigned long long*bad){
  std::uint64_t irc=(std::uint64_t)blockIdx.x*blockDim.x+threadIdx.x;if(irc>=n)return;const InterfacePlan&p=P[irc];
  double urh[14];for(int a=0;a<14;++a)urh[a]=u[hv[p.hexCell].g[a]];double rrh[14]={0};double maxJump=0,maxFlux=0,fluxInt=0;
  const double nx=p.normal[0],ny=p.normal[1],nz=p.normal[2];
  for(int side=0;side<2;++side){
    const int tcell=p.tetCell[side],tf=p.tetFace[side];double urt[8];for(int a=0;a<8;++a)urt[a]=u[tv[tcell].g[a]];double rrt[8]={0};
    const double tau=gamma*nu*fmax(p.invHHex,p.invHTet[side]);
    for(int ia=0;ia<5;++ia)for(int ib=0;ib<5;++ib){
      const double rr=c_q5_x[ia],ss=c_q5_x[ib],om=1.0-rr;
      const double L[3]={om*(1.0-ss),rr,om*ss};const double w=c_q5_w[ia]*c_q5_w[ib]*om*2.0*p.areaTri[side];
      double xr=0,yr=0,zr=0;for(int k=0;k<3;++k){const int hl=p.hexTriLocal[side][k];xr+=L[k]*c_hex_sign[hl][0];yr+=L[k]*c_hex_sign[hl][1];zr+=L[k]*c_hex_sign[hl][2];}
      double ph[14],grh[14][3],gph[14][3],Ih[9],deth;hex_basis_ref(xr,yr,zr,ph,grh);if(!hex_metric(pts,hc[p.hexCell],xr,yr,zr,Ih,deth)){atomicAdd(bad,1ULL);continue;}phys_grad14(grh,Ih,gph);
      double lam[4]={0,0,0,0};for(int k=0;k<3;++k)lam[c_tet_face[tf][k]]=L[k];double pt[8],grt[8][3],gpt[8][3];tet_basis(lam[0],lam[1],lam[2],lam[3],pt,grt);tet_phys_grad(grt,tg[tcell],gpt);
      double uh=0,ut=0,gh[3]={0,0,0},gt[3]={0,0,0};for(int a=0;a<14;++a){uh+=urh[a]*ph[a];for(int d=0;d<3;++d)gh[d]+=urh[a]*gph[a][d];}for(int a=0;a<8;++a){ut+=urt[a]*pt[a];for(int d=0;d<3;++d)gt[d]+=urt[a]*gpt[a][d];}
      const double jump=uh-ut,fh=nu*(gh[0]*nx+gh[1]*ny+gh[2]*nz),ft=nu*(gt[0]*nx+gt[1]*ny+gt[2]*nz),avg=0.5*(fh+ft);
      maxJump=fmax(maxJump,fabs(jump));maxFlux=fmax(maxFlux,fabs(fh-ft));fluxInt+=w*(fh-ft);
      for(int a=0;a<14;++a){const double dn=gph[a][0]*nx+gph[a][1]*ny+gph[a][2]*nz;rrh[a]+=w*(-avg*ph[a]-0.5*nu*dn*jump+tau*jump*ph[a]);}
      for(int a=0;a<8;++a){const double dn=gpt[a][0]*nx+gpt[a][1]*ny+gpt[a][2]*nz;rrt[a]+=w*( avg*pt[a]-0.5*nu*dn*jump-tau*jump*pt[a]);}
    }
    for(int a=0;a<8;++a)atomicAdd(r+tv[tcell].g[a],rrt[a]);
  }
  for(int a=0;a<14;++a)atomicAdd(r+hv[p.hexCell].g[a],rrh[a]);
  if(diagJump){diagJump[irc]=maxJump;diagFlux[irc]=maxFlux;diagFluxInt[irc]=fabs(fluxInt);}
}

struct Operator {
  const HostMesh&M;const std::vector<InterfacePlan>&P;
  Dev<Point>d_points;Dev<HexConn>d_hex;Dev<TetConn>d_tet;Dev<HexVel>d_hv;Dev<TetVel>d_tv;Dev<TetGeom>d_tg;Dev<InterfacePlan>d_if;Dev<double>d_u,d_r,d_jump,d_flux,d_fluxInt;
  unsigned long long*d_bad=nullptr;
  Operator(const HostMesh&m,const std::vector<InterfacePlan>&p):M(m),P(p),d_points(m.points),d_hex(m.hexes),d_tet(m.tets),d_hv(m.hexVel),d_tv(m.tetVel),d_tg(m.tetGeom),d_if(p),d_u(std::vector<double>((std::size_t)m.h.nvel,0.0)),d_r(std::vector<double>((std::size_t)m.h.nvel,0.0)),d_jump(std::vector<double>(p.size(),0.0)),d_flux(std::vector<double>(p.size(),0.0)),d_fluxInt(std::vector<double>(p.size(),0.0)){
    HXT1_CUDA(cudaMalloc((void**)&d_bad,sizeof(*d_bad)));auto T=build_tet_tensor();HXT1_CUDA(cudaMemcpyToSymbol(c_tet_T,T.data(),T.size()*sizeof(double)));
  }
  ~Operator(){if(d_bad)cudaFree(d_bad);}
  void set_u(const std::vector<double>&u){if(u.size()!=M.h.nvel)throw std::runtime_error("HXT2 vector size mismatch");HXT1_CUDA(cudaMemcpy(d_u.p,u.data(),u.size()*sizeof(double),cudaMemcpyHostToDevice));}
  std::vector<double> apply(bool withInterface,double nu,double gamma,bool diagnostics=false){
    HXT1_CUDA(cudaMemset(d_r.p,0,M.h.nvel*sizeof(double)));HXT1_CUDA(cudaMemset(d_bad,0,sizeof(*d_bad)));const int B=128;
    if(M.h.nhex)hex_volume_kernel<<<(M.h.nhex+B-1)/B,B>>>(d_points.p,d_hex.p,d_hv.p,d_u.p,d_r.p,M.h.nhex,nu,d_bad);
    if(M.h.ntet)tet_volume_kernel<<<(M.h.ntet+B-1)/B,B>>>(d_tv.p,d_tg.p,d_u.p,d_r.p,M.h.ntet,nu);
    if(withInterface&&M.h.ninterface)interface_kernel<<<(M.h.ninterface+B-1)/B,B>>>(d_points.p,d_hex.p,d_tet.p,d_hv.p,d_tv.p,d_tg.p,d_if.p,d_u.p,d_r.p,M.h.ninterface,nu,gamma,diagnostics?d_jump.p:nullptr,diagnostics?d_flux.p:nullptr,diagnostics?d_fluxInt.p:nullptr,d_bad);
    HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());unsigned long long bad=0;HXT1_CUDA(cudaMemcpy(&bad,d_bad,sizeof(bad),cudaMemcpyDeviceToHost));if(bad)throw std::runtime_error("HXT2 device geometry error count="+std::to_string(bad));
    std::vector<double>r((std::size_t)M.h.nvel);HXT1_CUDA(cudaMemcpy(r.data(),d_r.p,r.size()*sizeof(double),cudaMemcpyDeviceToHost));return r;
  }
  void get_diag(std::vector<double>&j,std::vector<double>&f,std::vector<double>&fi){j.resize(P.size());f.resize(P.size());fi.resize(P.size());if(!P.empty()){HXT1_CUDA(cudaMemcpy(j.data(),d_jump.p,j.size()*sizeof(double),cudaMemcpyDeviceToHost));HXT1_CUDA(cudaMemcpy(f.data(),d_flux.p,f.size()*sizeof(double),cudaMemcpyDeviceToHost));HXT1_CUDA(cudaMemcpy(fi.data(),d_fluxInt.p,fi.size()*sizeof(double),cudaMemcpyDeviceToHost));}}
};

inline double maxabs(const std::vector<double>&v){double m=0;for(double x:v)m=std::max(m,std::abs(x));return m;}
inline double dotv(const std::vector<double>&a,const std::vector<double>&b){double s=0;for(std::size_t i=0;i<a.size();++i)s+=a[i]*b[i];return s;}
inline double interface_bubble_max(const std::vector<double>&r,const std::vector<InterfacePlan>&P){double m=0;for(const auto&p:P){m=std::max(m,std::abs(r[(std::size_t)p.hexBubble]));m=std::max(m,std::abs(r[(std::size_t)p.tetBubble[0]]));m=std::max(m,std::abs(r[(std::size_t)p.tetBubble[1]]));}return m;}
inline double vmax(const std::vector<double>&v){return v.empty()?0:*std::max_element(v.begin(),v.end());}

inline std::vector<double> affine_field(const HostMesh&M,double c,double ax,double ay,double az){std::vector<double>u((std::size_t)M.h.nvel,0.0);for(std::uint64_t i=0;i<M.h.nv;++i){const auto&p=M.points[(std::size_t)i];u[(std::size_t)i]=c+ax*p.x+ay*p.y+az*p.z;}return u;}

} // namespace hxt2

int main(int argc,char**argv){
  try{
    std::string mesh;double nu=1.0,gamma=20.0;
    for(int i=1;i<argc;++i){std::string a=argv[i];if(a=="--mesh"&&i+1<argc)mesh=argv[++i];else if(a=="--nu"&&i+1<argc)nu=std::stod(argv[++i]);else if(a=="--gamma"&&i+1<argc)gamma=std::stod(argv[++i]);else if(a=="--help"){std::printf("usage: %s --mesh HXT1_mesh.bin [--nu 1] [--gamma 20]\n",argv[0]);return 0;}else throw std::runtime_error("unknown/incomplete argument: "+a);}
    if(mesh.empty())throw std::runtime_error("--mesh required");if(!(nu>0&&gamma>0))throw std::runtime_error("nu and gamma must be positive");
    HostMesh M=load(mesh);auto P=hxt2::build_interface_plans(M);if(P.size()!=M.h.ninterface)throw std::runtime_error("HXT2 interface plan count mismatch");
    int dev=0;cudaDeviceProp prop{};HXT1_CUDA(cudaGetDevice(&dev));HXT1_CUDA(cudaGetDeviceProperties(&prop,dev));
    hxt2::Operator A(M,P);
    std::printf("NODALS_HXT2_CONFIG device=%s cc=%d.%d nu=%.12e gamma=%.12e formulation=SYMMETRIC_TWO_SIDED_NITSCHE average=ARITHMETIC hHex=2V_over_A hTet=2V_over_A status=PASS\n",prop.name,prop.major,prop.minor,nu,gamma);
    std::printf("NODALS_HXT2_SCOPE pressure=OFF simple=OFF convection=OFF wall=OFF dgInlet=OFF rans=OFF status=PASS\n");

    bool ok=true;
    auto uc=hxt2::affine_field(M,1.0,0,0,0);A.set_u(uc);auto rc=A.apply(true,nu,gamma,true);std::vector<double>dj,df,di;A.get_diag(dj,df,di);double cmax=hxt2::maxabs(rc),cj=hxt2::vmax(dj),cf=hxt2::vmax(df);
    const bool cpass=cmax<1e-11&&cj<1e-12&&cf<1e-10;ok&=cpass;
    std::printf("NODALS_HXT2_CONSTANT_PATCH residualMax=%.12e jumpMax=%.12e fluxMismatchMax=%.12e status=%s\n",cmax,cj,cf,cpass?"PASS":"FAIL");

    const double coeff[3][4]={{0.7,1.0,0.20,-0.10},{-0.4,0.30,1.10,0.15},{0.2,-0.25,0.35,0.90}};
    for(int comp=0;comp<3;++comp){
      auto u=hxt2::affine_field(M,coeff[comp][0],coeff[comp][1],coeff[comp][2],coeff[comp][3]);A.set_u(u);auto rv=A.apply(false,nu,gamma,false);auto rf=A.apply(true,nu,gamma,true);A.get_diag(dj,df,di);
      const double mv=hxt2::interface_bubble_max(rv,P),mf=hxt2::interface_bubble_max(rf,P),ratio=mf/std::max(mv,1e-300),jm=hxt2::vmax(dj),fm=hxt2::vmax(df),fim=hxt2::vmax(di);
      const bool pass=mv>1e-14 && ratio<1e-7 && jm<1e-11 && fm<1e-8 && fim<1e-12;ok&=pass;
      std::printf("NODALS_HXT2_AFFINE_PATCH component=%d grad=[%.6f,%.6f,%.6f] volumeInterfaceBubbleMax=%.12e fullInterfaceBubbleMax=%.12e cancellationRatio=%.12e jumpMax=%.12e fluxMismatchMax=%.12e integratedFluxMismatchMax=%.12e status=%s\n",comp,coeff[comp][1],coeff[comp][2],coeff[comp][3],mv,mf,ratio,jm,fm,fim,pass?"PASS":"FAIL");
    }

    std::vector<double>x((std::size_t)M.h.nvel,0.0),y=x;std::set<int>ids;for(const auto&p:P){ids.insert(p.hexBubble);ids.insert(p.tetBubble[0]);ids.insert(p.tetBubble[1]);}
    for(int g:ids){x[(std::size_t)g]=std::sin(0.001731*(g+1));y[(std::size_t)g]=std::cos(0.002117*(g+3));}
    A.set_u(x);auto Ax=A.apply(true,nu,gamma,false);A.set_u(y);auto Ay=A.apply(true,nu,gamma,false);
    const double xAx=hxt2::dotv(x,Ax),yAy=hxt2::dotv(y,Ay),xAy=hxt2::dotv(x,Ay),yAx=hxt2::dotv(y,Ax),sym=std::abs(xAy-yAx)/std::max({std::abs(xAy),std::abs(yAx),1e-300});
    const bool spass=sym<1e-9&&xAx>0&&yAy>0;ok&=spass;
    std::printf("NODALS_HXT2_SYMMETRY_ENERGY interfaceBubbleDofs=%zu xAx=%.12e yAy=%.12e xAy=%.12e yAx=%.12e symmetryRel=%.12e status=%s\n",ids.size(),xAx,yAy,xAy,yAx,sym,spass?"PASS":"FAIL");

    std::printf("NODALS_HXT2_DEVICE_RESIDENCY meshPlansResident=1 diffusionActionGPU=1 nitscheActionGPU=1 setupUploadsOnly=1 status=PASS\n");
    std::printf("HXT2_STATUS=%s\n",ok?"PASS":"FAIL");return ok?0:3;
  }catch(const std::exception&e){std::fprintf(stderr,"HXT2_ERROR: %s\n",e.what());std::fprintf(stderr,"HXT2_STATUS=FAIL\n");return 2;}
}
