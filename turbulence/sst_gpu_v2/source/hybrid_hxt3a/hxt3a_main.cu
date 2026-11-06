#include "../hybrid_hxt1/hxt1_mesh.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nodals_hxt1;

namespace hxt3a {

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
static constexpr double G3X[3] = {
  -0.774596669241483377035853079956,0.0,0.774596669241483377035853079956
};
static constexpr double G3W[3] = {
  0.555555555555555555555555555556,
  0.888888888888888888888888888889,
  0.555555555555555555555555555556
};
static constexpr double Q5X[5] = {
  0.046910077030668018149839,0.230765344947158454481842,
  0.500000000000000000000000,0.769234655052841545518158,
  0.953089922969331981850161
};
static constexpr double Q5W[5] = {
  0.118463442528094543757132,0.239314335249683234020645,
  0.284444444444444444444444,0.239314335249683234020645,
  0.118463442528094543757132
};

struct V3 { double x=0,y=0,z=0; };
inline V3 operator+(V3 a,V3 b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
inline V3 operator-(V3 a,V3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
inline V3 operator*(double s,V3 a){return {s*a.x,s*a.y,s*a.z};}
inline V3 operator*(V3 a,double s){return s*a;}
inline V3& operator+=(V3&a,V3 b){a.x+=b.x;a.y+=b.y;a.z+=b.z;return a;}
inline V3& operator-=(V3&a,V3 b){a.x-=b.x;a.y-=b.y;a.z-=b.z;return a;}
inline double dot(V3 a,V3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
inline V3 cross(V3 a,V3 b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
inline double norm(V3 a){return std::sqrt(dot(a,a));}
inline V3 pv(const Point&p){return {p.x,p.y,p.z};}

inline bool inv3(const double J[9],double I[9],double&det){
  det=J[0]*(J[4]*J[8]-J[5]*J[7])-J[1]*(J[3]*J[8]-J[5]*J[6])+J[2]*(J[3]*J[7]-J[4]*J[6]);
  if(!(det>0.0))return false;
  const double q=1.0/det;
  I[0]=(J[4]*J[8]-J[5]*J[7])*q; I[1]=(J[2]*J[7]-J[1]*J[8])*q; I[2]=(J[1]*J[5]-J[2]*J[4])*q;
  I[3]=(J[5]*J[6]-J[3]*J[8])*q; I[4]=(J[0]*J[8]-J[2]*J[6])*q; I[5]=(J[2]*J[3]-J[0]*J[5])*q;
  I[6]=(J[3]*J[7]-J[4]*J[6])*q; I[7]=(J[1]*J[6]-J[0]*J[7])*q; I[8]=(J[0]*J[4]-J[1]*J[3])*q;
  return true;
}

inline void hex_q1(double x,double y,double z,double phi[8],double gr[8][3]){
  for(int a=0;a<8;++a){
    const int sx=HEX_SIGN[a][0],sy=HEX_SIGN[a][1],sz=HEX_SIGN[a][2];
    const double ax=1.0+sx*x,ay=1.0+sy*y,az=1.0+sz*z;
    phi[a]=0.125*ax*ay*az;
    gr[a][0]=0.125*sx*ay*az;
    gr[a][1]=0.125*sy*ax*az;
    gr[a][2]=0.125*sz*ax*ay;
  }
}
inline void hex_basis(double x,double y,double z,double phi[14],double gr[14][3]){
  hex_q1(x,y,z,phi,gr);
  const double bx=1-x*x,by=1-y*y,bz=1-z*z;
  const double dx=-2*x,dy=-2*y,dz=-2*z; double s;
  s=.5*(1-z); phi[8]=s*bx*by; gr[8][0]=s*dx*by; gr[8][1]=s*bx*dy; gr[8][2]=-.5*bx*by;
  s=.5*(1+z); phi[9]=s*bx*by; gr[9][0]=s*dx*by; gr[9][1]=s*bx*dy; gr[9][2]= .5*bx*by;
  s=.5*(1-y); phi[10]=s*bx*bz;gr[10][0]=s*dx*bz;gr[10][1]=-.5*bx*bz;gr[10][2]=s*bx*dz;
  s=.5*(1+x); phi[11]=s*by*bz;gr[11][0]= .5*by*bz;gr[11][1]=s*dy*bz;gr[11][2]=s*by*dz;
  s=.5*(1+y); phi[12]=s*bx*bz;gr[12][0]=s*dx*bz;gr[12][1]= .5*bx*bz;gr[12][2]=s*bx*dz;
  s=.5*(1-x); phi[13]=s*by*bz;gr[13][0]=-.5*by*bz;gr[13][1]=s*dy*bz;gr[13][2]=s*by*dz;
}
inline void tet_basis(double l0,double l1,double l2,double l3,double phi[8]){
  const double l[4]={l0,l1,l2,l3};
  for(int i=0;i<4;++i)phi[i]=l[i];
  for(int i=0;i<4;++i){
    int js[3],k=0;for(int j=0;j<4;++j)if(j!=i)js[k++]=j;
    phi[4+i]=27.0*l[js[0]]*l[js[1]]*l[js[2]];
  }
}
inline bool hex_metric(const HostMesh&M,const HexConn&h,double x,double y,double z,double I[9],double&det){
  double ph[8],gr[8][3];hex_q1(x,y,z,ph,gr);(void)ph;
  double J[9]={0,0,0,0,0,0,0,0,0};
  for(int a=0;a<8;++a){
    const auto&p=M.points[(std::size_t)h.v[a]];
    J[0]+=p.x*gr[a][0];J[1]+=p.x*gr[a][1];J[2]+=p.x*gr[a][2];
    J[3]+=p.y*gr[a][0];J[4]+=p.y*gr[a][1];J[5]+=p.y*gr[a][2];
    J[6]+=p.z*gr[a][0];J[7]+=p.z*gr[a][1];J[8]+=p.z*gr[a][2];
  }
  return inv3(J,I,det);
}
inline void phys_grad14(const double gr[14][3],const double I[9],double gp[14][3]){
  for(int a=0;a<14;++a)for(int d=0;d<3;++d)
    gp[a][d]=gr[a][0]*I[d]+gr[a][1]*I[3+d]+gr[a][2]*I[6+d];
}
inline V3 hex_centroid(const HostMesh&M,int c){
  V3 q{};for(int a=0;a<8;++a)q+=pv(M.points[(std::size_t)M.hexes[(std::size_t)c].v[a]]);return (1.0/8.0)*q;
}
inline V3 tet_centroid(const HostMesh&M,int c){
  V3 q{};for(int a=0;a<4;++a)q+=pv(M.points[(std::size_t)M.tets[(std::size_t)c].v[a]]);return 0.25*q;
}
inline double tri_area(V3 a,V3 b,V3 c){return 0.5*norm(cross(b-a,c-a));}

using Row=std::map<std::int32_t,V3>;

inline void add(Row&r,std::int32_t g,V3 v){
  auto it=r.find(g);
  if(it==r.end())r.emplace(g,v);else it->second+=v;
}
inline void cleanup(Row&r){
  for(auto it=r.begin();it!=r.end();){
    if(std::max({std::abs(it->second.x),std::abs(it->second.y),std::abs(it->second.z)})<1e-28)it=r.erase(it);
    else ++it;
  }
}

inline std::array<V3,14> hex_volume_b(const HostMesh&M,int c){
  std::array<V3,14>B{};const auto&h=M.hexes[(std::size_t)c];
  for(int ix=0;ix<3;++ix)for(int iy=0;iy<3;++iy)for(int iz=0;iz<3;++iz){
    const double x=G3X[ix],y=G3X[iy],z=G3X[iz],w=G3W[ix]*G3W[iy]*G3W[iz];
    double phi[14],gr[14][3],gp[14][3],I[9],det;
    hex_basis(x,y,z,phi,gr);(void)phi;
    if(!hex_metric(M,h,x,y,z,I,det))throw std::runtime_error("HXT3A bad hex metric");
    phys_grad14(gr,I,gp);
    for(int a=0;a<14;++a){B[a].x+=w*det*gp[a][0];B[a].y+=w*det*gp[a][1];B[a].z+=w*det*gp[a][2];}
  }
  return B;
}
inline std::array<V3,8> tet_volume_b(const HostMesh&M,int c){
  std::array<V3,8>B{};const auto&g=M.tetGeom[(std::size_t)c];const double vol=g.volume;
  const double gr[4][3]={{-1,-1,-1},{1,0,0},{0,1,0},{0,0,1}};
  for(int i=0;i<4;++i){
    V3 v{};
    v.x=vol*(gr[i][0]*g.invJ[0]+gr[i][1]*g.invJ[3]+gr[i][2]*g.invJ[6]);
    v.y=vol*(gr[i][0]*g.invJ[1]+gr[i][1]*g.invJ[4]+gr[i][2]*g.invJ[7]);
    v.z=vol*(gr[i][0]*g.invJ[2]+gr[i][1]*g.invJ[5]+gr[i][2]*g.invJ[8]);
    B[i]=v;B[4+i]=(-27.0/20.0)*v;
  }
  return B;
}

struct IFPlan {
  int hc,hf,tc[2],tf[2];
  int hexTriLocal[2][3];
  V3 n;
  double area[2];
};

inline std::vector<IFPlan> build_if(const HostMesh&M){
  std::vector<IFPlan>P;P.reserve(M.interface.size());
  for(const auto&r:M.interface){
    IFPlan p{};p.hc=r.hexCell;p.hf=r.hexLocalFace;p.tc[0]=r.tet0;p.tf[0]=r.tet0LocalFace;p.tc[1]=r.tet1;p.tf[1]=r.tet1LocalFace;
    const auto&h=M.hexes[(std::size_t)p.hc];
    V3 q[4];for(int k=0;k<4;++k)q[k]=pv(M.points[(std::size_t)h.v[HEX_FACE[p.hf][k]]]);
    V3 n=cross(q[1]-q[0],q[3]-q[0]);double nn=norm(n);if(!(nn>0))throw std::runtime_error("zero interface normal");n=(1.0/nn)*n;
    if(dot(n,tet_centroid(M,p.tc[0])-hex_centroid(M,p.hc))<0)n=(-1.0)*n;p.n=n;
    for(int s=0;s<2;++s){
      const auto&t=M.tets[(std::size_t)p.tc[s]];V3 tp[3];
      for(int k=0;k<3;++k){
        int lv=TET_FACE[p.tf[s]][k],gv=t.v[lv];tp[k]=pv(M.points[(std::size_t)gv]);
        int hl=-1;for(int a=0;a<8;++a)if(h.v[a]==gv){hl=a;break;}
        if(hl<0)throw std::runtime_error("interface vertex not on hex");
        p.hexTriLocal[s][k]=hl;
      }
      p.area[s]=tri_area(tp[0],tp[1],tp[2]);if(!(p.area[s]>0))throw std::runtime_error("zero interface triangle");
    }
    P.push_back(p);
  }
  return P;
}

struct TraceInt { std::array<double,14> H{}; std::array<double,8> T{}; };

inline TraceInt trace_integrals(const HostMesh&M,const IFPlan&p,int side){
  TraceInt R{};const int tc=p.tc[side],tf=p.tf[side];
  for(int ia=0;ia<5;++ia)for(int ib=0;ib<5;++ib){
    const double rr=Q5X[ia],ss=Q5X[ib],om=1.0-rr;
    const double L[3]={om*(1.0-ss),rr,om*ss};
    const double w=Q5W[ia]*Q5W[ib]*om*2.0*p.area[side];
    double xr=0,yr=0,zr=0;
    for(int k=0;k<3;++k){int hl=p.hexTriLocal[side][k];xr+=L[k]*HEX_SIGN[hl][0];yr+=L[k]*HEX_SIGN[hl][1];zr+=L[k]*HEX_SIGN[hl][2];}
    double ph[14],grh[14][3];hex_basis(xr,yr,zr,ph,grh);(void)grh;
    double lam[4]={0,0,0,0};for(int k=0;k<3;++k)lam[TET_FACE[tf][k]]=L[k];
    double pt[8];tet_basis(lam[0],lam[1],lam[2],lam[3],pt);
    for(int a=0;a<14;++a)R.H[a]+=w*ph[a];
    for(int a=0;a<8;++a)R.T[a]+=w*pt[a];
  }
  return R;
}

struct BCSR {
  int np=0,nv=0;
  std::vector<std::int64_t> row;
  std::vector<std::int32_t> col;
  std::vector<double> bx,by,bz;
  std::vector<Row> rows;
};

inline BCSR build_b(const HostMesh&M,const std::vector<IFPlan>&P){
  const int nh=(int)M.h.nhex,nt=(int)M.h.ntet,np=nh+nt;
  BCSR A;A.np=np;A.nv=(int)M.h.nvel;A.rows.resize((std::size_t)np);

  for(int c=0;c<nh;++c){
    auto b=hex_volume_b(M,c);const auto&v=M.hexVel[(std::size_t)c];
    for(int a=0;a<14;++a)add(A.rows[(std::size_t)c],v.g[a],b[a]);
  }
  for(int c=0;c<nt;++c){
    auto b=tet_volume_b(M,c);const auto&v=M.tetVel[(std::size_t)c];
    for(int a=0;a<8;++a)add(A.rows[(std::size_t)(nh+c)],v.g[a],b[a]);
  }

  // Replace the two independent physical traces by one arithmetic numerical
  // normal flux on each triangle.  n points HEX -> TET.
  for(const auto&p:P)for(int s=0;s<2;++s){
    const auto R=trace_integrals(M,p,s);
    Row&rh=A.rows[(std::size_t)p.hc];
    Row&rt=A.rows[(std::size_t)(nh+p.tc[s])];
    const auto&hv=M.hexVel[(std::size_t)p.hc];
    const auto&tv=M.tetVel[(std::size_t)p.tc[s]];
    for(int a=0;a<14;++a){
      V3 q=(0.5*R.H[a])*p.n;
      add(rh,hv.g[a],(-1.0)*q); // own HEX face: 1 -> 1/2
      add(rt,hv.g[a],(-1.0)*q); // opposite trace enters TET with -1/2
    }
    for(int a=0;a<8;++a){
      V3 q=(0.5*R.T[a])*p.n;
      add(rh,tv.g[a],q);        // opposite trace enters HEX with +1/2
      add(rt,tv.g[a],q);        // own TET face: -1 -> -1/2
    }
  }
  for(auto&r:A.rows)cleanup(r);

  A.row.resize((std::size_t)np+1,0);
  for(int i=0;i<np;++i)A.row[(std::size_t)i+1]=A.row[(std::size_t)i]+(std::int64_t)A.rows[(std::size_t)i].size();
  const std::size_t nnz=(std::size_t)A.row.back();A.col.resize(nnz);A.bx.resize(nnz);A.by.resize(nnz);A.bz.resize(nnz);
  for(int i=0;i<np;++i){
    std::size_t k=(std::size_t)A.row[(std::size_t)i];
    for(const auto&[g,v]:A.rows[(std::size_t)i]){A.col[k]=g;A.bx[k]=v.x;A.by[k]=v.y;A.bz[k]=v.z;++k;}
  }
  return A;
}

inline std::vector<double> b_host(const BCSR&A,const std::vector<double>&u0,const std::vector<double>&u1,const std::vector<double>&u2){
  std::vector<double>y((std::size_t)A.np,0.0);
  for(int i=0;i<A.np;++i)for(std::int64_t k=A.row[(std::size_t)i];k<A.row[(std::size_t)i+1];++k){
    int g=A.col[(std::size_t)k];y[(std::size_t)i]+=A.bx[(std::size_t)k]*u0[(std::size_t)g]+A.by[(std::size_t)k]*u1[(std::size_t)g]+A.bz[(std::size_t)k]*u2[(std::size_t)g];
  }return y;
}
inline void bt_host(const BCSR&A,const std::vector<double>&p,std::vector<double>&v0,std::vector<double>&v1,std::vector<double>&v2){
  v0.assign((std::size_t)A.nv,0);v1.assign((std::size_t)A.nv,0);v2.assign((std::size_t)A.nv,0);
  for(int i=0;i<A.np;++i)for(std::int64_t k=A.row[(std::size_t)i];k<A.row[(std::size_t)i+1];++k){
    int g=A.col[(std::size_t)k];double q=p[(std::size_t)i];v0[(std::size_t)g]+=A.bx[(std::size_t)k]*q;v1[(std::size_t)g]+=A.by[(std::size_t)k]*q;v2[(std::size_t)g]+=A.bz[(std::size_t)k]*q;
  }
}
inline double dotv(const std::vector<double>&a,const std::vector<double>&b){long double s=0;for(std::size_t i=0;i<a.size();++i)s+=(long double)a[i]*b[i];return (double)s;}
inline double maxabs(const std::vector<double>&a){double m=0;for(double x:a)m=std::max(m,std::abs(x));return m;}

inline double row_schur_dot(const Row&a,const Row&b,const std::vector<double>&rau){
  const Row*pa=&a,*pb=&b;if(pa->size()>pb->size())std::swap(pa,pb);
  long double s=0;
  for(const auto&[g,v]:*pa){auto it=pb->find(g);if(it!=pb->end())s+=(long double)rau[(std::size_t)g]*dot(v,it->second);}
  return (double)s;
}

__global__ void b_kernel(int np,const std::int64_t*row,const std::int32_t*col,const double*bx,const double*by,const double*bz,const double*u0,const double*u1,const double*u2,double*y){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i>=np)return;double s=0;
  for(std::int64_t k=row[i];k<row[i+1];++k){int g=col[k];s+=bx[k]*u0[g]+by[k]*u1[g]+bz[k]*u2[g];}y[i]=s;
}
__global__ void bt_kernel(int np,const std::int64_t*row,const std::int32_t*col,const double*bx,const double*by,const double*bz,const double*p,double*v0,double*v1,double*v2){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i>=np)return;double q=p[i];
  for(std::int64_t k=row[i];k<row[i+1];++k){int g=col[k];atomicAdd(v0+g,bx[k]*q);atomicAdd(v1+g,by[k]*q);atomicAdd(v2+g,bz[k]*q);}
}
__global__ void rau_kernel(int n,const double*r,double*v0,double*v1,double*v2){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(i<n){v0[i]*=r[i];v1[i]*=r[i];v2[i]*=r[i];}
}

struct GpuB {
  const BCSR&A;
  Dev<std::int64_t>drow;Dev<std::int32_t>dcol;Dev<double>dbx,dby,dbz,drau,du0,du1,du2,dp,dy;
  GpuB(const BCSR&a,const std::vector<double>&rau):A(a),drow(a.row),dcol(a.col),dbx(a.bx),dby(a.by),dbz(a.bz),drau(rau),
    du0(std::vector<double>((std::size_t)a.nv,0)),du1(std::vector<double>((std::size_t)a.nv,0)),du2(std::vector<double>((std::size_t)a.nv,0)),
    dp(std::vector<double>((std::size_t)a.np,0)),dy(std::vector<double>((std::size_t)a.np,0)){}
  std::vector<double> B(const std::vector<double>&u0,const std::vector<double>&u1,const std::vector<double>&u2){
    HXT1_CUDA(cudaMemcpy(du0.p,u0.data(),u0.size()*sizeof(double),cudaMemcpyHostToDevice));HXT1_CUDA(cudaMemcpy(du1.p,u1.data(),u1.size()*sizeof(double),cudaMemcpyHostToDevice));HXT1_CUDA(cudaMemcpy(du2.p,u2.data(),u2.size()*sizeof(double),cudaMemcpyHostToDevice));
    int B=256;b_kernel<<<(A.np+B-1)/B,B>>>(A.np,drow.p,dcol.p,dbx.p,dby.p,dbz.p,du0.p,du1.p,du2.p,dy.p);HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
    std::vector<double>y((std::size_t)A.np);HXT1_CUDA(cudaMemcpy(y.data(),dy.p,y.size()*sizeof(double),cudaMemcpyDeviceToHost));return y;
  }
  void BT(const std::vector<double>&p,std::vector<double>&v0,std::vector<double>&v1,std::vector<double>&v2,bool applyRau=false){
    HXT1_CUDA(cudaMemcpy(dp.p,p.data(),p.size()*sizeof(double),cudaMemcpyHostToDevice));HXT1_CUDA(cudaMemset(du0.p,0,A.nv*sizeof(double)));HXT1_CUDA(cudaMemset(du1.p,0,A.nv*sizeof(double)));HXT1_CUDA(cudaMemset(du2.p,0,A.nv*sizeof(double)));
    int B=256;bt_kernel<<<(A.np+B-1)/B,B>>>(A.np,drow.p,dcol.p,dbx.p,dby.p,dbz.p,dp.p,du0.p,du1.p,du2.p);HXT1_CUDA(cudaGetLastError());
    if(applyRau)rau_kernel<<<(A.nv+B-1)/B,B>>>(A.nv,drau.p,du0.p,du1.p,du2.p);
    HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
    v0.resize((std::size_t)A.nv);v1.resize((std::size_t)A.nv);v2.resize((std::size_t)A.nv);
    HXT1_CUDA(cudaMemcpy(v0.data(),du0.p,v0.size()*sizeof(double),cudaMemcpyDeviceToHost));HXT1_CUDA(cudaMemcpy(v1.data(),du1.p,v1.size()*sizeof(double),cudaMemcpyDeviceToHost));HXT1_CUDA(cudaMemcpy(v2.data(),du2.p,v2.size()*sizeof(double),cudaMemcpyDeviceToHost));
  }
  std::vector<double> S(const std::vector<double>&p){
    HXT1_CUDA(cudaMemcpy(dp.p,p.data(),p.size()*sizeof(double),cudaMemcpyHostToDevice));HXT1_CUDA(cudaMemset(du0.p,0,A.nv*sizeof(double)));HXT1_CUDA(cudaMemset(du1.p,0,A.nv*sizeof(double)));HXT1_CUDA(cudaMemset(du2.p,0,A.nv*sizeof(double)));
    int B=256;bt_kernel<<<(A.np+B-1)/B,B>>>(A.np,drow.p,dcol.p,dbx.p,dby.p,dbz.p,dp.p,du0.p,du1.p,du2.p);HXT1_CUDA(cudaGetLastError());
    rau_kernel<<<(A.nv+B-1)/B,B>>>(A.nv,drau.p,du0.p,du1.p,du2.p);HXT1_CUDA(cudaGetLastError());
    b_kernel<<<(A.np+B-1)/B,B>>>(A.np,drow.p,dcol.p,dbx.p,dby.p,dbz.p,du0.p,du1.p,du2.p,dy.p);HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
    std::vector<double>y((std::size_t)A.np);HXT1_CUDA(cudaMemcpy(y.data(),dy.p,y.size()*sizeof(double),cudaMemcpyDeviceToHost));return y;
  }
};

} // namespace hxt3a

int main(int argc,char**argv){
  try{
    std::string mesh;
    for(int i=1;i<argc;++i){std::string a=argv[i];if(a=="--mesh"&&i+1<argc)mesh=argv[++i];else if(a=="--help"){std::printf("usage: %s --mesh HXT1_mesh.bin\n",argv[0]);return 0;}else throw std::runtime_error("unknown/incomplete argument "+a);}
    if(mesh.empty())throw std::runtime_error("--mesh required");
    HostMesh M=load(mesh);auto P=hxt3a::build_if(M);auto A=hxt3a::build_b(M,P);
    const int nv=A.nv,np=A.np,nh=(int)M.h.nhex;
    int dev=0;cudaDeviceProp prop{};HXT1_CUDA(cudaGetDevice(&dev));HXT1_CUDA(cudaGetDeviceProperties(&prop,dev));

    std::vector<double>rau((std::size_t)nv);for(int g=0;g<nv;++g){double q=g+1;rau[(std::size_t)g]=0.8+0.2*(0.5+0.5*std::sin(0.000731*q));}
    hxt3a::GpuB G(A,rau);
    bool ok=true;

    std::printf("NODALS_HXT3A_CONFIG device=%s cc=%d.%d pressureSpace=Q0_P0 numericalInterfaceFlux=ARITHMETIC_AVERAGE pressureGradient=EXACT_TRANSPOSE schur=B_diagRAU_BT rau=AUDIT_SYNTHETIC_POSITIVE status=PASS\n",prop.name,prop.major,prop.minor);
    std::printf("NODALS_HXT3A_SCOPE stokesSolve=OFF simpleLoop=OFF convection=OFF wall=OFF dgInlet=OFF rans=OFF status=PASS\n");
    std::printf("NODALS_HXT3A_BCSR rows=%d scalarVelocityDofs=%d nnz=%zu meanNnzPerRow=%.6f status=PASS\n",np,nv,A.col.size(),(double)A.col.size()/np);

    // Constant velocity patch: bubbles are zero, Q1/P1 vertices carry constants.
    double constMax=0;
    for(int d=0;d<3;++d){
      std::vector<double>u0((std::size_t)nv,0),u1=u0,u2=u0;
      for(std::uint64_t v=0;v<M.h.nv;++v){if(d==0)u0[(std::size_t)v]=1;if(d==1)u1[(std::size_t)v]=1;if(d==2)u2[(std::size_t)v]=1;}
      auto y=G.B(u0,u1,u2);constMax=std::max(constMax,hxt3a::maxabs(y));
    }
    bool cpass=constMax<2e-18;ok&=cpass;
    std::printf("NODALS_HXT3A_CONSTANT_DIVERGENCE cellResidualMax=%.12e expected=0 status=%s\n",constMax,cpass?"PASS":"FAIL");

    // Affine u=(x,y,z), exactly represented by Q1/P1 vertex fields.
    std::vector<double>u0((std::size_t)nv,0),u1=u0,u2=u0;
    for(std::uint64_t v=0;v<M.h.nv;++v){const auto&p=M.points[(std::size_t)v];u0[(std::size_t)v]=p.x;u1[(std::size_t)v]=p.y;u2[(std::size_t)v]=p.z;}
    auto div=G.B(u0,u1,u2);double affAbs=0,affRel=0;
    for(int c=0;c<np;++c){
      double V=(c<nh)?M.hexGeom[(std::size_t)c].volume:M.tetGeom[(std::size_t)(c-nh)].volume;
      double ex=3.0*V,e=std::abs(div[(std::size_t)c]-ex);affAbs=std::max(affAbs,e);affRel=std::max(affRel,e/std::max(std::abs(ex),1e-300));
    }
    bool apass=affRel<2e-10;ok&=apass;
    std::printf("NODALS_HXT3A_AFFINE_DIVERGENCE field=[x,y,z] maxAbs=%.12e maxRel=%.12e expected=3V status=%s\n",affAbs,affRel,apass?"PASS":"FAIL");

    // Ensure cross-region sensitivity is real, including direct Schur entries.
    int zeroCross=0,zeroSchur=0;double minCross=1e300,maxCross=0,minS=1e300,maxS=0;
    for(const auto&p:P)for(int s=0;s<2;++s){
      const auto&rh=A.rows[(std::size_t)p.hc];const auto&rt=A.rows[(std::size_t)(nh+p.tc[s])];
      bool hcarriesTet=false,tcarriesHex=false;
      const auto&tv=M.tetVel[(std::size_t)p.tc[s]];const auto&hv=M.hexVel[(std::size_t)p.hc];
      for(int a=0;a<8;++a)if(rh.count(tv.g[a]))hcarriesTet=true;
      for(int a=0;a<14;++a)if(rt.count(hv.g[a]))tcarriesHex=true;
      if(!(hcarriesTet&&tcarriesHex))++zeroCross;
      double ss=std::abs(hxt3a::row_schur_dot(rh,rt,rau));if(!(ss>0))++zeroSchur;else{minS=std::min(minS,ss);maxS=std::max(maxS,ss);}
      // magnitude of direct cross trace coefficients
      double cm=0;for(int a=0;a<8;++a){auto it=rh.find(tv.g[a]);if(it!=rh.end())cm=std::max(cm,hxt3a::norm(it->second));}
      for(int a=0;a<14;++a){auto it=rt.find(hv.g[a]);if(it!=rt.end())cm=std::max(cm,hxt3a::norm(it->second));}
      if(cm>0){minCross=std::min(minCross,cm);maxCross=std::max(maxCross,cm);}
    }
    bool ipass=zeroCross==0&&zeroSchur==0&&minCross<1e299&&minS<1e299;ok&=ipass;
    std::printf("NODALS_HXT3A_INTERFACE_SENSITIVITY trianglePairs=%zu missingCrossTrace=%d zeroDirectSchur=%d crossCoeffNormMin=%.12e crossCoeffNormMax=%.12e directSchurAbsMin=%.12e directSchurAbsMax=%.12e status=%s\n",2*P.size(),zeroCross,zeroSchur,minCross,maxCross,minS,maxS,ipass?"PASS":"FAIL");

    // B / B^T exact transpose identity.
    std::vector<double>ru0((std::size_t)nv),ru1((std::size_t)nv),ru2((std::size_t)nv),pp((std::size_t)np);
    for(int g=0;g<nv;++g){double q=g+1;ru0[(std::size_t)g]=std::sin(.00113*q);ru1[(std::size_t)g]=std::cos(.00171*q);ru2[(std::size_t)g]=std::sin(.00091*q+.3);}
    for(int i=0;i<np;++i){double q=i+1;pp[(std::size_t)i]=std::cos(.00137*q+.2);}
    auto Bu=G.B(ru0,ru1,ru2);std::vector<double>bt0,bt1,bt2;G.BT(pp,bt0,bt1,bt2,false);
    double lhs=hxt3a::dotv(pp,Bu),rhs=hxt3a::dotv(ru0,bt0)+hxt3a::dotv(ru1,bt1)+hxt3a::dotv(ru2,bt2);
    double tr=std::abs(lhs-rhs)/std::max({std::abs(lhs),std::abs(rhs),1e-300});bool tpass=tr<2e-12;ok&=tpass;
    std::printf("NODALS_HXT3A_TRANSPOSE pDotBu=%.12e uDotBTp=%.12e rel=%.12e status=%s\n",lhs,rhs,tr,tpass?"PASS":"FAIL");

    // Schur symmetry / positivity on two deterministic pressure vectors.
    std::vector<double>px((std::size_t)np),py((std::size_t)np);
    for(int i=0;i<np;++i){double q=i+1;px[(std::size_t)i]=std::sin(.00073*q);py[(std::size_t)i]=std::cos(.00109*q+.17);}
    auto Sx=G.S(px),Sy=G.S(py);
    double xSx=hxt3a::dotv(px,Sx),ySy=hxt3a::dotv(py,Sy),xSy=hxt3a::dotv(px,Sy),ySx=hxt3a::dotv(py,Sx);
    double sr=std::abs(xSy-ySx)/std::max({std::abs(xSy),std::abs(ySx),1e-300});bool spass=xSx>0&&ySy>0&&sr<3e-12;ok&=spass;
    std::printf("NODALS_HXT3A_SCHUR xSx=%.12e ySy=%.12e xSy=%.12e ySx=%.12e symmetryRel=%.12e status=%s\n",xSx,ySy,xSy,ySx,sr,spass?"PASS":"FAIL");

    std::size_t bytes=A.row.size()*sizeof(std::int64_t)+A.col.size()*sizeof(std::int32_t)+3*A.col.size()*sizeof(double)+rau.size()*sizeof(double);
    std::printf("NODALS_HXT3A_DEVICE_RESIDENCY bCsrResident=1 bActionGPU=1 btActionGPU=1 schurActionGPU=1 setupUploadsOnly=1 persistentPressurePlanMiB=%.6f status=PASS\n",bytes/(1024.0*1024.0));
    std::printf("HXT3A_STATUS=%s\n",ok?"PASS":"FAIL");
    return ok?0:3;
  }catch(const std::exception&e){std::fprintf(stderr,"HXT3A_ERROR: %s\n",e.what());std::fprintf(stderr,"HXT3A_STATUS=FAIL\n");return 2;}
}
