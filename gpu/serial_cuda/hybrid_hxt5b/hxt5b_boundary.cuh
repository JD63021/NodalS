
#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace hxt5b {

using Real=hxt4b::Real;
using nodals_hxt1::Dev;
using nodals_hxt1::HostMesh;

static constexpr double GX4[4]={
  -0.8611363115940525752,-0.3399810435848562648,
   0.3399810435848562648, 0.8611363115940525752};
static constexpr double GW4[4]={
  0.3478548451374538574,0.6521451548625461426,
  0.6521451548625461426,0.3478548451374538574};

__device__ __constant__ double QL0[12]={
  0.063089014491502,0.063089014491502,0.873821971016996,
  0.249286745170910,0.249286745170910,0.501426509658180,
  0.053145049844816,0.053145049844816,0.310352451033785,
  0.310352451033785,0.636502499121399,0.636502499121399};
__device__ __constant__ double QL1[12]={
  0.063089014491502,0.873821971016996,0.063089014491502,
  0.249286745170910,0.501426509658180,0.249286745170910,
  0.310352451033785,0.636502499121399,0.053145049844816,
  0.636502499121399,0.053145049844816,0.310352451033785};
__device__ __constant__ double QL2[12]={
  0.873821971016996,0.063089014491502,0.063089014491502,
  0.501426509658180,0.249286745170910,0.249286745170910,
  0.636502499121399,0.310352451033785,0.636502499121399,
  0.053145049844816,0.310352451033785,0.053145049844816};
__device__ __constant__ double QW12[12]={
  0.050844906370207,0.050844906370207,0.050844906370207,
  0.116786275726379,0.116786275726379,0.116786275726379,
  0.082851075618374,0.082851075618374,0.082851075618374,
  0.082851075618374,0.082851075618374,0.082851075618374};

static constexpr double HL0[12]={
  0.063089014491502,0.063089014491502,0.873821971016996,
  0.249286745170910,0.249286745170910,0.501426509658180,
  0.053145049844816,0.053145049844816,0.310352451033785,
  0.310352451033785,0.636502499121399,0.636502499121399};
static constexpr double HL1[12]={
  0.063089014491502,0.873821971016996,0.063089014491502,
  0.249286745170910,0.501426509658180,0.249286745170910,
  0.310352451033785,0.636502499121399,0.053145049844816,
  0.636502499121399,0.053145049844816,0.310352451033785};
static constexpr double HL2[12]={
  0.873821971016996,0.063089014491502,0.063089014491502,
  0.501426509658180,0.249286745170910,0.249286745170910,
  0.636502499121399,0.310352451033785,0.636502499121399,
  0.053145049844816,0.310352451033785,0.053145049844816};
static constexpr double HW12[12]={
  0.050844906370207,0.050844906370207,0.050844906370207,
  0.116786275726379,0.116786275726379,0.116786275726379,
  0.082851075618374,0.082851075618374,0.082851075618374,
  0.082851075618374,0.082851075618374,0.082851075618374};

struct V3{double x=0,y=0,z=0;};
inline V3 vadd(V3 a,V3 b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
inline V3 vsub(V3 a,V3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
inline V3 vmul(double s,V3 a){return {s*a.x,s*a.y,s*a.z};}
inline V3 vcross(V3 a,V3 b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
inline double vdot(V3 a,V3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
inline double vnorm(V3 a){return std::sqrt(vdot(a,a));}
inline V3 vp(const nodals_hxt1::Point&p){return {p.x,p.y,p.z};}

struct HexFacePlan{
  std::int32_t cell=0;
  std::int8_t lf=0,fd=0,d0=1,d1=2;
  double fv=0;
  double n[3]={0,0,0};
};
struct TetFacePlan{
  std::int32_t cell=0;
  std::int8_t lf=0;
  double area=0,n[3]={0,0,0};
};

inline void href(int lf,int&fd,double&fv,int&d0,int&d1){
  fd=-1;
  for(int d=0;d<3;++d){
    int s=hxt2::HEX_SIGN[hxt2::HEX_FACE[lf][0]][d];bool same=true;
    for(int k=1;k<4;++k) if(hxt2::HEX_SIGN[hxt2::HEX_FACE[lf][k]][d]!=s) same=false;
    if(same){fd=d;fv=(double)s;break;}
  }
  if(fd<0)throw std::runtime_error("HXT5B HEX face reference identification failed");
  int q[2],m=0;for(int d=0;d<3;++d)if(d!=fd)q[m++]=d;d0=q[0];d1=q[1];
}
inline void hbasis(double x,double y,double z,double ph[14]){
  double c[3]={x,y,z};
  for(int a=0;a<8;++a){double q=1;for(int d=0;d<3;++d)q*=.5*(1+hxt2::HEX_SIGN[a][d]*c[d]);ph[a]=q;}
  for(int f=0;f<6;++f){int fd,d0,d1;double fv;href(f,fd,fv,d0,d1);ph[8+f]=.5*(1+fv*c[fd])*(1-c[d0]*c[d0])*(1-c[d1]*c[d1]);}
}
inline void tbasis(const double l[4],double ph[8]){
  for(int a=0;a<4;++a)ph[a]=l[a];
  for(int i=0;i<4;++i){double q=27;for(int j=0;j<4;++j)if(j!=i)q*=l[j];ph[4+i]=q;}
}
inline void q1host(const HostMesh&M,const nodals_hxt1::HexConn&h,double x,double y,double z,V3&X,V3 dX[3]){
  double c[3]={x,y,z};X={};dX[0]=dX[1]=dX[2]={};
  for(int a=0;a<8;++a){
    double f[3],df[3];for(int d=0;d<3;++d){double s=hxt2::HEX_SIGN[a][d];f[d]=.5*(1+s*c[d]);df[d]=.5*s;}
    double N=f[0]*f[1]*f[2],Nd[3]={df[0]*f[1]*f[2],f[0]*df[1]*f[2],f[0]*f[1]*df[2]};V3 P=vp(M.points[h.v[a]]);
    X=vadd(X,vmul(N,P));for(int d=0;d<3;++d)dX[d]=vadd(dX[d],vmul(Nd[d],P));
  }
}
inline HexFacePlan hexplan(const HostMesh&M,const nodals_hxt1::BoundaryRec&r){
  HexFacePlan P;P.cell=r.cell;
  int fd=0,d0=0,d1=0;double fv=0;
  href(r.localFace,fd,fv,d0,d1);
  if(fd<0||fd>2||d0<0||d0>2||d1<0||d1>2)
    throw std::runtime_error("HXT5B invalid HEX reference-face direction");
  P.lf=(std::int8_t)r.localFace;
  P.fd=(std::int8_t)fd;P.fv=fv;
  P.d0=(std::int8_t)d0;P.d1=(std::int8_t)d1;
  const auto&h=M.hexes[r.cell];V3 q[4];for(int k=0;k<4;++k)q[k]=vp(M.points[h.v[hxt2::HEX_FACE[r.localFace][k]]]);
  V3 av=vmul(.5,vadd(vcross(vsub(q[1],q[0]),vsub(q[2],q[0])),vcross(vsub(q[2],q[0]),vsub(q[3],q[0]))));
  V3 fc{};for(int k=0;k<4;++k)fc=vadd(fc,vmul(.25,q[k]));V3 cc{};for(int a=0;a<8;++a)cc=vadd(cc,vmul(.125,vp(M.points[h.v[a]])));
  if(vdot(av,vsub(fc,cc))<0)av=vmul(-1,av);double A=vnorm(av);if(!(A>0))throw std::runtime_error("HXT5B HEX boundary area invalid");
  av=vmul(1/A,av);P.n[0]=av.x;P.n[1]=av.y;P.n[2]=av.z;return P;
}
inline TetFacePlan tetplan(const HostMesh&M,const nodals_hxt1::BoundaryRec&r){
  TetFacePlan P;P.cell=r.cell;P.lf=r.localFace;const auto&t=M.tets[r.cell];V3 q[3];
  for(int k=0;k<3;++k)q[k]=vp(M.points[t.v[hxt2::TET_FACE[r.localFace][k]]]);
  V3 av=vmul(.5,vcross(vsub(q[1],q[0]),vsub(q[2],q[0]))),fc=vmul(1.0/3.0,vadd(vadd(q[0],q[1]),q[2])),cc{};
  for(int a=0;a<4;++a)cc=vadd(cc,vmul(.25,vp(M.points[t.v[a]])));if(vdot(av,vsub(fc,cc))<0)av=vmul(-1,av);
  P.area=vnorm(av);if(!(P.area>0))throw std::runtime_error("HXT5B TET boundary area invalid");av=vmul(1/P.area,av);P.n[0]=av.x;P.n[1]=av.y;P.n[2]=av.z;return P;
}

struct BoundaryHost{
  hxt3a::BCSR B;
  std::vector<double> csrc;
  std::vector<unsigned char> wallMask;
  std::vector<HexFacePlan> inH,wallH;
  std::vector<TetFacePlan> inT;
  double inletArea=0,sourceSum=0,constResidual=0;
};
inline void repack(hxt3a::BCSR&A){
  for(auto&r:A.rows)hxt3a::cleanup(r);A.row.assign((size_t)A.np+1,0);
  for(int i=0;i<A.np;++i)A.row[i+1]=A.row[i]+(std::int64_t)A.rows[i].size();
  size_t nnz=(size_t)A.row.back();A.col.resize(nnz);A.bx.resize(nnz);A.by.resize(nnz);A.bz.resize(nnz);
  for(int i=0;i<A.np;++i){size_t k=A.row[i];for(auto&kv:A.rows[i]){A.col[k]=kv.first;A.bx[k]=kv.second.x;A.by[k]=kv.second.y;A.bz[k]=kv.second.z;++k;}}
}
inline BoundaryHost build_boundaries(const HostMesh&M,const hxt3a::BCSR&B0,double bulk){
  BoundaryHost H;H.B=B0;H.csrc.assign(B0.np,0);H.wallMask.assign(B0.nv,0);int nh=M.h.nhex;
  for(auto&r:M.inHex){
    auto P=hexplan(M,r);H.inH.push_back(P);std::array<V3,14>I{};V3 Avec{};const auto&hc=M.hexes[r.cell];
    for(int i=0;i<4;++i)for(int j=0;j<4;++j){double c[3]={0,0,0};c[P.fd]=P.fv;c[P.d0]=GX4[i];c[P.d1]=GX4[j];V3 X,dX[3];q1host(M,hc,c[0],c[1],c[2],X,dX);
      V3 av=vcross(dX[P.d0],dX[P.d1]);V3 n{P.n[0],P.n[1],P.n[2]};if(vdot(av,n)<0)av=vmul(-1,av);V3 nds=vmul(GW4[i]*GW4[j],av);Avec=vadd(Avec,nds);
      double ph[14];hbasis(c[0],c[1],c[2],ph);for(int a=0;a<14;++a)I[a]=vadd(I[a],vmul(ph[a],nds));}
    auto&row=H.B.rows[r.cell];auto&v=M.hexVel[r.cell];for(int a=0;a<14;++a)hxt3a::add(row,v.g[a],{-I[a].x,-I[a].y,-I[a].z});
    H.csrc[r.cell]+=bulk*Avec.z;H.inletArea+=vnorm(Avec);H.sourceSum+=bulk*Avec.z;
  }
  for(auto&r:M.inTet){
    auto P=tetplan(M,r);H.inT.push_back(P);std::array<double,8>I{};int a0=hxt2::TET_FACE[r.localFace][0],a1=hxt2::TET_FACE[r.localFace][1],a2=hxt2::TET_FACE[r.localFace][2];
    for(int q=0;q<12;++q){double l[4]={0,0,0,0};l[a0]=HL0[q];l[a1]=HL1[q];l[a2]=HL2[q];double ph[8];tbasis(l,ph);for(int a=0;a<8;++a)I[a]+=HW12[q]*P.area*ph[a];}
    auto&row=H.B.rows[nh+r.cell];auto&v=M.tetVel[r.cell];for(int a=0;a<8;++a)hxt3a::add(row,v.g[a],{-I[a]*P.n[0],-I[a]*P.n[1],-I[a]*P.n[2]});
    H.csrc[nh+r.cell]+=bulk*P.area*P.n[2];H.inletArea+=P.area;H.sourceSum+=bulk*P.area*P.n[2];
  }
  for(auto&r:M.wallHex){H.wallH.push_back(hexplan(M,r));auto&h=M.hexes[r.cell];H.wallMask[r.bubble]=1;for(int k=0;k<4;++k)H.wallMask[h.v[hxt2::HEX_FACE[r.localFace][k]]]=1;}
  repack(H.B);
  std::vector<double>x(B0.nv,0),y=x,z=x;for(std::uint64_t v=0;v<M.h.nv;++v)z[v]=bulk;auto c=hxt3a::b_host(H.B,x,y,z);
  for(int i=0;i<H.B.np;++i)H.constResidual=std::max(H.constResidual,std::abs(c[i]+H.csrc[i]));
  if(H.constResidual>1e-10*std::max(1.0,std::abs(H.sourceSum)))throw std::runtime_error("HXT5B DG continuity trace/source audit failed");
  return H;
}
inline hxt4b::FixedData plug_initial(const HostMesh&M,double bulk){
  hxt4b::FixedData F;F.mask.assign(M.h.nvel,0);F.u0.assign(M.h.nvel,0);F.u1.assign(M.h.nvel,0);F.u2.assign(M.h.nvel,0);for(std::uint64_t v=0;v<M.h.nv;++v)F.u2[v]=(Real)bulk;return F;
}

__device__ __forceinline__ void q1dev(const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn&h,double x,double y,double z,double X[3],double dX[3][3]){
  double c[3]={x,y,z};X[0]=X[1]=X[2]=0;for(int d=0;d<3;++d)for(int j=0;j<3;++j)dX[d][j]=0;
  for(int a=0;a<8;++a){double f[3],df[3];for(int d=0;d<3;++d){double s=hxt4a::h4_hex_sign[a][d];f[d]=.5*(1+s*c[d]);df[d]=.5*s;}double N=f[0]*f[1]*f[2],Nd[3]={df[0]*f[1]*f[2],f[0]*df[1]*f[2],f[0]*f[1]*df[2]};auto P=pts[h.v[a]];X[0]+=N*P.x;X[1]+=N*P.y;X[2]+=N*P.z;for(int d=0;d<3;++d){dX[d][0]+=Nd[d]*P.x;dX[d][1]+=Nd[d]*P.y;dX[d][2]+=Nd[d]*P.z;}}
}
__device__ __forceinline__ double spy(double up,double k,double B){double x=k*up,rem;if(fabs(x)<1e-3){double x2=x*x,x4=x2*x2;rem=x4*(1.0/24+x/120+x2/720+x2*x/5040);}else rem=expm1(x)-x-.5*x*x-x*x*x/6;return up+exp(-k*B)*rem;}
__device__ __forceinline__ double spg(double up,double reY){return up*spy(up,.4,5.5)-reY;}
__device__ __forceinline__ bool spalding(double slip,double y,double nu,double&ut,double&yp,double&beta){
  double U=fabs(slip),reY=y*U/nu;if(!(y>0)||!(nu>0)||!isfinite(U))return false;if(reY<=1e-14){ut=yp=0;beta=nu/y;return true;}double lo=0,hi=fmax(1.0,sqrt(reY)+1);for(int k=0;k<40&&spg(hi,reY)<0;++k)hi*=2;double gh=spg(hi,reY);if(!isfinite(gh)||gh<0)return false;for(int k=0;k<70;++k){double m=.5*(lo+hi);if(spg(m,reY)>0)hi=m;else lo=m;}double up=.5*(lo+hi);if(!(up>0))return false;ut=U/up;yp=y*ut/nu;beta=ut*ut/U;return isfinite(beta)&&beta>0&&isfinite(yp);
}
__device__ __forceinline__ void amax(double*addr,double v){auto*p=(unsigned long long*)addr;unsigned long long old=*p,ass;do{ass=old;if(__longlong_as_double((long long)ass)>=v)break;old=atomicCAS(p,ass,(unsigned long long)__double_as_longlong(v));}while(ass!=old);}

__global__ void dg_hex(const HexFacePlan*F,int nf,const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,const nodals_hxt1::HexVel*hv,const std::int32_t*slot,const Real*u0,const Real*u1,const Real*u2,Real*mat,Real*diag,Real*rhsz,double bulk,double R,double nu,double ms,unsigned long long*bad){
  int fi=blockIdx.x;if(fi>=nf)return;auto f=F[fi];auto H=hv[f.cell];__shared__ Real U0[14],U1[14],U2[14],ph[16][14],gp[16][14][3],nue[16],wq[16];
  if(threadIdx.x<14){int g=H.g[threadIdx.x];U0[threadIdx.x]=u0[g];U1[threadIdx.x]=u1[g];U2[threadIdx.x]=u2[g];}__syncthreads();
  if(threadIdx.x<16){int q=threadIdx.x,i=q>>2,j=q&3;double c[3]={0,0,0};c[f.fd]=f.fv;c[f.d0]=hxt5a::h5_hex_x[i];c[f.d1]=hxt5a::h5_hex_x[j];double pp[14],gr[14][3],gg[14][3],I[9],det;hxt2::hex_basis_ref(c[0],c[1],c[2],pp,gr);if(!hxt2::hex_metric(pts,hc[f.cell],c[0],c[1],c[2],I,det)){atomicAdd(bad,1ULL);wq[q]=0;return;}hxt2::phys_grad14(gr,I,gg);double X[3],dX[3][3];q1dev(pts,hc[f.cell],c[0],c[1],c[2],X,dX);double cr[3]={dX[f.d0][1]*dX[f.d1][2]-dX[f.d0][2]*dX[f.d1][1],dX[f.d0][2]*dX[f.d1][0]-dX[f.d0][0]*dX[f.d1][2],dX[f.d0][0]*dX[f.d1][1]-dX[f.d0][1]*dX[f.d1][0]};if(cr[0]*f.n[0]+cr[1]*f.n[1]+cr[2]*f.n[2]<0){cr[0]*=-1;cr[1]*=-1;cr[2]*=-1;}double jac=sqrt(cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]);Real gu[9]={0};for(int a=0;a<14;++a){ph[q][a]=pp[a];for(int d=0;d<3;++d){Real gd=gg[a][d];gp[q][a][d]=gd;gu[d]+=U0[a]*gd;gu[3+d]+=U1[a]*gd;gu[6+d]+=U2[a]*gd;}}Real nut=hxt5a::h5_nikuradse_nut(X[0],X[1],R,ms,gu,nullptr);nue[q]=nu+nut;wq[q]=hxt5a::h5_hex_w[i]*hxt5a::h5_hex_w[j]*jac;}__syncthreads();
  Real inflow=fmax(Real(0),Real(-bulk*f.n[2]));
  for(int p=threadIdx.x;p<196;p+=blockDim.x){int a=p/14,b=p%14;Real z=0;for(int q=0;q<16;++q){Real dna=gp[q][a][0]*f.n[0]+gp[q][a][1]*f.n[1]+gp[q][a][2]*f.n[2],dnb=gp[q][b][0]*f.n[0]+gp[q][b][1]*f.n[1]+gp[q][b][2]*f.n[2];z+=wq[q]*(inflow*ph[q][a]*ph[q][b]+nue[q]*(-ph[q][a]*dnb+dna*ph[q][b]));}atomicAdd(mat+slot[(size_t)f.cell*196+p],z);if(a==b)atomicAdd(diag+H.g[a],z);}
  for(int a=threadIdx.x;a<14;a+=blockDim.x){Real z=0;for(int q=0;q<16;++q){Real dna=gp[q][a][0]*f.n[0]+gp[q][a][1]*f.n[1]+gp[q][a][2]*f.n[2];z+=wq[q]*(inflow*ph[q][a]*bulk+nue[q]*dna*bulk);}atomicAdd(rhsz+H.g[a],z);}
}
__global__ void dg_tet(const TetFacePlan*F,int nf,const nodals_hxt1::Point*pts,const nodals_hxt1::TetConn*tc,const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,const std::int32_t*slot,const Real*u0,const Real*u1,const Real*u2,Real*mat,Real*diag,Real*rhsz,double bulk,double R,double nu,double ms){
  int fi=blockIdx.x;if(fi>=nf)return;auto f=F[fi];auto T=tv[f.cell];__shared__ Real U0[8],U1[8],U2[8],ph[12][8],gp[12][8][3],nue[12],wq[12];if(threadIdx.x<8){int g=T.g[threadIdx.x];U0[threadIdx.x]=u0[g];U1[threadIdx.x]=u1[g];U2[threadIdx.x]=u2[g];}__syncthreads();
  if(threadIdx.x<12){int q=threadIdx.x,a0=hxt4a::h4_tet_face[f.lf][0],a1=hxt4a::h4_tet_face[f.lf][1],a2=hxt4a::h4_tet_face[f.lf][2];double l[4]={0,0,0,0};l[a0]=QL0[q];l[a1]=QL1[q];l[a2]=QL2[q];double pp[8],gr[8][3],gg[8][3];hxt2::tet_basis(l[0],l[1],l[2],l[3],pp,gr);hxt2::tet_phys_grad(gr,tg[f.cell],gg);Real gu[9]={0};double x=0,y=0;for(int a=0;a<8;++a){ph[q][a]=pp[a];for(int d=0;d<3;++d){Real gd=gg[a][d];gp[q][a][d]=gd;gu[d]+=U0[a]*gd;gu[3+d]+=U1[a]*gd;gu[6+d]+=U2[a]*gd;}}for(int a=0;a<4;++a){auto P=pts[tc[f.cell].v[a]];x+=l[a]*P.x;y+=l[a]*P.y;}Real nut=hxt5a::h5_nikuradse_nut(x,y,R,ms,gu,nullptr);nue[q]=nu+nut;wq[q]=QW12[q]*f.area;}__syncthreads();Real inflow=fmax(Real(0),Real(-bulk*f.n[2]));
  for(int p=threadIdx.x;p<64;p+=blockDim.x){int a=p/8,b=p%8;Real z=0;for(int q=0;q<12;++q){Real dna=gp[q][a][0]*f.n[0]+gp[q][a][1]*f.n[1]+gp[q][a][2]*f.n[2],dnb=gp[q][b][0]*f.n[0]+gp[q][b][1]*f.n[1]+gp[q][b][2]*f.n[2];z+=wq[q]*(inflow*ph[q][a]*ph[q][b]+nue[q]*(-ph[q][a]*dnb+dna*ph[q][b]));}atomicAdd(mat+slot[(size_t)f.cell*64+p],z);if(a==b)atomicAdd(diag+T.g[a],z);}
  for(int a=threadIdx.x;a<8;a+=blockDim.x){Real z=0;for(int q=0;q<12;++q){Real dna=gp[q][a][0]*f.n[0]+gp[q][a][1]*f.n[1]+gp[q][a][2]*f.n[2];z+=wq[q]*(inflow*ph[q][a]*bulk+nue[q]*dna*bulk);}atomicAdd(rhsz+T.g[a],z);}
}
__global__ void wall_hex(const HexFacePlan*F,int nf,const nodals_hxt1::Point*pts,const nodals_hxt1::HexConn*hc,const nodals_hxt1::HexVel*hv,const std::int32_t*slot,const Real*u2,Real*mat,Real*diag,double nu,double sf,double*stats,unsigned long long*bad){
  int fi=blockIdx.x;if(fi>=nf)return;auto f=F[fi];auto H=hv[f.cell];__shared__ Real U[14],bw[16][14],bs[16][14],betaw[16];__shared__ double ar[16],ut2[16],ypw[16],slw[16],my[16],mu[16];if(threadIdx.x<14)U[threadIdx.x]=u2[H.g[threadIdx.x]];__syncthreads();
  if(threadIdx.x<16){int q=threadIdx.x,i=q>>2,j=q&3;double cw[3]={0,0,0};cw[f.fd]=f.fv;cw[f.d0]=hxt5a::h5_hex_x[i];cw[f.d1]=hxt5a::h5_hex_x[j];double cs[3]={cw[0],cw[1],cw[2]};cs[f.fd]=f.fv+sf*(-2*f.fv);double pw[14],gw[14][3],ps[14],gs[14][3];hxt2::hex_basis_ref(cw[0],cw[1],cw[2],pw,gw);hxt2::hex_basis_ref(cs[0],cs[1],cs[2],ps,gs);double Xw[3],dXw[3][3],Xs[3],dXs[3][3];q1dev(pts,hc[f.cell],cw[0],cw[1],cw[2],Xw,dXw);q1dev(pts,hc[f.cell],cs[0],cs[1],cs[2],Xs,dXs);double rr=hypot(Xw[0],Xw[1]),jac=0,y=0,sl=0,ut=0,yp=0,beta=0;bool ok=rr>0;if(ok){double nr[3]={Xw[0]/rr,Xw[1]/rr,0};double cr[3]={dXw[f.d0][1]*dXw[f.d1][2]-dXw[f.d0][2]*dXw[f.d1][1],dXw[f.d0][2]*dXw[f.d1][0]-dXw[f.d0][0]*dXw[f.d1][2],dXw[f.d0][0]*dXw[f.d1][1]-dXw[f.d0][1]*dXw[f.d1][0]};if(cr[0]*nr[0]+cr[1]*nr[1]+cr[2]*nr[2]<0){cr[0]*=-1;cr[1]*=-1;cr[2]*=-1;}jac=sqrt(cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]);y=(Xw[0]-Xs[0])*nr[0]+(Xw[1]-Xs[1])*nr[1];ok=y>0&&jac>0;}for(int a=0;a<14;++a){bw[q][a]=ok?pw[a]:0;bs[q][a]=ok?ps[a]:0;if(ok)sl+=(double)U[a]*ps[a];}if(ok&&!spalding(sl,y,nu,ut,yp,beta)){beta=nu/y;ut=yp=0;atomicAdd(stats+6,1.0);}if(!ok){atomicAdd(bad,1ULL);beta=ut=yp=sl=jac=0;}double w=hxt5a::h5_hex_w[i]*hxt5a::h5_hex_w[j]*jac;betaw[q]=beta*w;ar[q]=w;ut2[q]=ut*ut*w;ypw[q]=yp*w;slw[q]=fabs(sl)*w;my[q]=yp;mu[q]=ut;}__syncthreads();
  for(int p=threadIdx.x;p<196;p+=blockDim.x){int a=p/14,b=p%14;Real z=0;for(int q=0;q<16;++q)z+=betaw[q]*bw[q][a]*bs[q][b];atomicAdd(mat+slot[(size_t)f.cell*196+p],z);if(a==b)atomicAdd(diag+H.g[a],z);}
  if(threadIdx.x==0){double A=0,U2=0,Y=0,S=0,MY=0,MU=0;for(int q=0;q<16;++q){A+=ar[q];U2+=ut2[q];Y+=ypw[q];S+=slw[q];MY=fmax(MY,my[q]);MU=fmax(MU,mu[q]);}atomicAdd(stats+0,A);atomicAdd(stats+1,U2);atomicAdd(stats+2,Y);atomicAdd(stats+3,S);amax(stats+4,MY);amax(stats+5,MU);}
}

struct WallStats{double area=0,fDarcy=0,yPlusMean=0,slipMean=0,yPlusMax=0,uTauMax=0,rootFailures=0;};
inline WallStats wallstats(const Dev<double>&d,double bulk){std::vector<double>h(d.n,0.0);if(d.n)HXT1_CUDA(cudaMemcpy(h.data(),d.p,d.n*sizeof(double),cudaMemcpyDeviceToHost));WallStats q;if(h.size()<7)return q;q.area=h[0];if(q.area>0){q.fDarcy=8*(h[1]/q.area)/(bulk*bulk);q.yPlusMean=h[2]/q.area;q.slipMean=h[3]/q.area;}q.yPlusMax=h[4];q.uTauMax=h[5];q.rootFailures=h[6];return q;}
inline void assemble_boundary(const BoundaryHost&H,const Dev<HexFacePlan>&ih,const Dev<TetFacePlan>&it,const Dev<HexFacePlan>&wh,const Dev<nodals_hxt1::Point>&pts,const Dev<nodals_hxt1::HexConn>&hc,const Dev<nodals_hxt1::TetConn>&tc,const Dev<nodals_hxt1::HexVel>&hv,const Dev<nodals_hxt1::TetVel>&tv,const Dev<nodals_hxt1::TetGeom>&tg,hxt4b::GpuCSR&A,const Real*u0,const Real*u1,const Real*u2,Dev<Real>&dg,Dev<Real>&ddg,Dev<Real>&rz,Dev<Real>&wall,Dev<Real>&dwall,Dev<double>&ws,Dev<unsigned long long>&bad,double bulk,double R,double nu,double ms,double sf){
  if(dg.n)HXT1_CUDA(cudaMemset(dg.p,0,dg.n*sizeof(Real)));if(ddg.n)HXT1_CUDA(cudaMemset(ddg.p,0,ddg.n*sizeof(Real)));if(rz.n)HXT1_CUDA(cudaMemset(rz.p,0,rz.n*sizeof(Real)));if(wall.n)HXT1_CUDA(cudaMemset(wall.p,0,wall.n*sizeof(Real)));if(dwall.n)HXT1_CUDA(cudaMemset(dwall.p,0,dwall.n*sizeof(Real)));if(ws.n)HXT1_CUDA(cudaMemset(ws.p,0,ws.n*sizeof(double)));if(bad.n)HXT1_CUDA(cudaMemset(bad.p,0,bad.n*sizeof(unsigned long long)));if(!H.inH.empty())dg_hex<<<H.inH.size(),64>>>(ih.p,H.inH.size(),pts.p,hc.p,hv.p,A.hexSlot.p,u0,u1,u2,dg.p,ddg.p,rz.p,bulk,R,nu,ms,bad.p);if(!H.inT.empty())dg_tet<<<H.inT.size(),64>>>(it.p,H.inT.size(),pts.p,tc.p,tv.p,tg.p,A.tetSlot.p,u0,u1,u2,dg.p,ddg.p,rz.p,bulk,R,nu,ms);if(!H.wallH.empty())wall_hex<<<H.wallH.size(),64>>>(wh.p,H.wallH.size(),pts.p,hc.p,hv.p,A.hexSlot.p,u2,wall.p,dwall.p,nu,sf,ws.p,bad.p);HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());unsigned long long q=0;HXT1_CUDA(cudaMemcpy(&q,bad.p,sizeof(q),cudaMemcpyDeviceToHost));if(q)throw std::runtime_error("HXT5B boundary geometry failure count="+std::to_string(q));
}

__global__ void combine(std::size_t n,const Real*b,const Real*c,const Real*t,const Real*dg,const Real*w,Real*xy,Real*z){size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n){Real q=b[i]+c[i]+t[i]+dg[i];xy[i]=q;z[i]=q+w[i];}}
__global__ void finalize(int n,const unsigned char*fixed,const unsigned char*wm,const std::int32_t*dp,Real*xy,Real*z,const Real*drb,const Real*conv,const Real*dt,const Real*ddg,const Real*dw,Real*rxy,Real*rz,Real*dxy,Real*dz,double au,double rs,unsigned long long*bad){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;int d=dp[i];Real ox=xy[d],oz=z[d];dxy[i]=ox;dz[i]=oz;if(fixed[i]){rxy[i]=rz[i]=0;return;}Real den=drb[i]+conv[d]+dt[i]+ddg[i],denz=den+dw[i];if(!(ox>0&&oz>0&&den>0&&denz>0)){atomicAdd(bad,1ULL);rxy[i]=rz[i]=0;return;}xy[d]=ox/au;z[d]=oz/au;rxy[i]=wm[i]?0:rs*au/den;rz[i]=rs*au/denz;
}
__global__ void rhs(int n,const unsigned char*cl,const Real*bt,const Real*extra,const Real*u,const Real*d,double au,Real*r){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)r[i]=cl[i]?0:bt[i]+(extra?extra[i]:0)+((1-au)/au)*d[i]*u[i];}
__global__ void mcgs(int begin,int count,const std::int32_t*rows,const std::int64_t*row,const std::int32_t*col,const Real*val,const std::int32_t*dp,const Real*b,Real*x,double om,const unsigned char*cl){int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=count)return;int i=rows[begin+q];if(cl[i]){x[i]=0;return;}Real off=0;for(auto k=row[i];k<row[i+1];++k)if(col[k]!=i)off+=val[k]*x[col[k]];Real gs=(b[i]-off)/val[dp[i]];x[i]+=om*(gs-x[i]);}
inline void sweep(hxt4b::GpuCSR&A,const hxt4b::ColoringHost&C,const Real*val,const Real*b,Real*x,double om,bool f,const unsigned char*cl){if(f){for(int c=0;c<C.ncolors;++c){int s=C.off[c],n=C.off[c+1]-s;mcgs<<<(n+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(s,n,A.colorRows.p,A.row.p,A.col.p,val,A.diagPos.p,b,x,om,cl);}}else{for(int c=C.ncolors-1;c>=0;--c){int s=C.off[c],n=C.off[c+1]-s;mcgs<<<(n+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(s,n,A.colorRows.p,A.row.p,A.col.p,val,A.diagPos.p,b,x,om,cl);}}}
inline void work(const std::string&m,int it,hxt4b::GpuCSR&A,const hxt4b::ColoringHost&C,const Real*val,const Real*b,Real*x,double om,const unsigned char*cl){if(m=="fgs1")sweep(A,C,val,b,x,om,true,cl);else if(m=="altgs1")sweep(A,C,val,b,x,om,(it&1)!=0,cl);else if(m=="sgs1"){sweep(A,C,val,b,x,om,true,cl);sweep(A,C,val,b,x,om,false,cl);}else throw std::runtime_error("bad momentum work");HXT1_CUDA(cudaGetLastError());}
__global__ void spmv(int n,const std::int64_t*r,const std::int32_t*c,const Real*v,const Real*x,Real*y){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){Real s=0;for(auto k=r[i];k<r[i+1];++k)s+=v[k]*x[c[k]];y[i]=s;}}
__global__ void pres(int n,const unsigned char*cl,const Real*bt,const Real*ex,const Real*u,const Real*d,double au,const Real*Ax,Real*r){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)r[i]=cl[i]?0:bt[i]+(ex?ex[i]:0)-(Ax[i]-((1-au)/au)*d[i]*u[i]);}
inline double pnorm(hxt4b::GpuCSR&A,const Real*val,const unsigned char*cl,const Real*bt,const Real*ex,const Real*u,const Real*d,double au,hxt4a::Reducer<Real>&red){spmv<<<(A.n+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(A.n,A.row.p,A.col.p,val,u,A.tmp.p);pres<<<(A.n+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(A.n,cl,bt,ex,u,d,au,A.tmp.p,A.res.p);return red.norm(A.res.p);}
__global__ void addsrc(int n,const Real*s,Real*x){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]+=s[i];}

__device__ __forceinline__ void h5b_atomic_max_double(double*addr,double v){
  auto*p=(unsigned long long*)addr;unsigned long long old=*p,ass;
  do{ass=old;if(__longlong_as_double((long long)ass)>=v)break;
     old=atomicCAS(p,ass,(unsigned long long)__double_as_longlong(v));}while(ass!=old);
}
__global__ void finite_audit_kernel(std::size_t n,const Real*x,unsigned long long*bad,double*maxabs){
  std::size_t i=(std::size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  double v=(double)x[i];
  if(!isfinite(v))atomicAdd(bad,1ULL);
  else h5b_atomic_max_double(maxabs,fabs(v));
}
struct FiniteAudit { unsigned long long nonfinite=0; double maxabs=0; };
inline FiniteAudit audit_real(const char*label,const Real*x,std::size_t n,bool failOnBad=true){
  Dev<unsigned long long>bad(std::vector<unsigned long long>(1,0));
  Dev<double>mx(std::vector<double>(1,0));
  if(n)finite_audit_kernel<<<(n+255)/256,256>>>(n,x,bad.p,mx.p);
  HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
  FiniteAudit A;
  HXT1_CUDA(cudaMemcpy(&A.nonfinite,bad.p,sizeof(A.nonfinite),cudaMemcpyDeviceToHost));
  HXT1_CUDA(cudaMemcpy(&A.maxabs,mx.p,sizeof(A.maxabs),cudaMemcpyDeviceToHost));
  std::printf("NODALS_HXT5B_FINITE_AUDIT label=%s n=%zu nonfinite=%llu maxAbs=%.12e status=%s\n",
    label,n,A.nonfinite,A.maxabs,A.nonfinite?"FAIL":"PASS");
  if(failOnBad&&A.nonfinite)throw std::runtime_error(std::string("HXT5B nonfinite array at ")+label);
  return A;
}
inline void audit_diag_host(const char*label,const Real*x,int n){
  std::vector<Real>h((std::size_t)n);
  HXT1_CUDA(cudaMemcpy(h.data(),x,(std::size_t)n*sizeof(Real),cudaMemcpyDeviceToHost));
  double mn=std::numeric_limits<double>::infinity(),mx=0;std::size_t bad=0,nonpos=0;
  for(Real q:h){double v=(double)q;if(!std::isfinite(v)){++bad;continue;}if(!(v>0)){++nonpos;continue;}mn=std::min(mn,v);mx=std::max(mx,v);}
  double ratio=(std::isfinite(mn)&&mn>0)?mx/mn:std::numeric_limits<double>::infinity();
  std::printf("NODALS_HXT5B_DIAG_AUDIT label=%s n=%d nonfinite=%zu nonpositive=%zu minPositive=%.12e maxPositive=%.12e maxOverMin=%.12e status=%s\n",
    label,n,bad,nonpos,mn,mx,ratio,(bad||nonpos)?"WARN":"PASS");
}

template<class T>__global__ void scale3(int n,const T*rx,const T*ry,const T*rz,T*x,T*y,T*z){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){x[i]*=rx[i];y[i]*=ry[i];z[i]*=rz[i];}}
template<class T>struct SAniso{hxt4a::BDevice<T>&B;const T*rx,*ry,*rz;int pin;Dev<T>x,y,z;SAniso(hxt4a::BDevice<T>&b,const T*a,const T*c,const T*d,int p):B(b),rx(a),ry(c),rz(d),pin(p),x(std::vector<T>(b.nv,0)),y(std::vector<T>(b.nv,0)),z(std::vector<T>(b.nv,0)){}void apply(const T*p,T*out){hxt4a::BT_apply(B,p,x.p,y.p,z.p);scale3<<<(B.nv+255)/256,256>>>(B.nv,rx,ry,rz,x.p,y.p,z.p);hxt4a::B_apply(B,x.p,y.p,z.p,out);hxt4a::pin_output_zero_kernel<T><<<1,1>>>(pin,out);}};
template<class T>__global__ void sdiag(int np,const std::int64_t*r,const std::int32_t*c,const T*bx,const T*by,const T*bz,const T*rx,const T*ry,const T*rz,T*d,int pin){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=np)return;if(i==pin){d[i]=1;return;}T s=0;for(auto k=r[i];k<r[i+1];++k){int g=c[k];s+=rx[g]*bx[k]*bx[k]+ry[g]*by[k]*by[k]+rz[g]*bz[k]*bz[k];}d[i]=s;}

inline double host_sentry(const hxt3a::BCSR&B,int i,int j,const std::vector<double>&rx,const std::vector<double>&ry,const std::vector<double>&rz){
  auto a=B.row[(std::size_t)i],ae=B.row[(std::size_t)i+1],b=B.row[(std::size_t)j],be=B.row[(std::size_t)j+1];long double q=0;
  while(a<ae&&b<be){int ga=B.col[(std::size_t)a],gb=B.col[(std::size_t)b];if(ga<gb){++a;continue;}if(gb<ga){++b;continue;}q+=(long double)rx[(std::size_t)ga]*B.bx[(std::size_t)a]*B.bx[(std::size_t)b]+(long double)ry[(std::size_t)ga]*B.by[(std::size_t)a]*B.by[(std::size_t)b]+(long double)rz[(std::size_t)ga]*B.bz[(std::size_t)a]*B.bz[(std::size_t)b];++a;++b;}return (double)q;
}
inline void retune_host_fine(hxt4c::HybridFineHost&H,const hxt3a::BCSR&B,int pin,const std::vector<double>&rx,const std::vector<double>&ry,const std::vector<double>&rz){
  if((int)rx.size()!=B.nv||(int)ry.size()!=B.nv||(int)rz.size()!=B.nv)throw std::runtime_error("HXT5B host directional rAU size mismatch");
  if(H.A.n!=B.np)throw std::runtime_error("HXT5B fine CSR row mismatch");
  for(int i=0;i<H.A.n;++i){
    bool haveDiag=false;
    for(auto k=H.A.row[(std::size_t)i];k<H.A.row[(std::size_t)i+1];++k){int j=H.A.col[(std::size_t)k];double v=(i==pin)?(j==pin?1.0:0.0):(j==pin?0.0:host_sentry(B,i,j,rx,ry,rz));H.A.val[(std::size_t)k]=v;if(j==i){H.A.diag[(std::size_t)i]=v;haveDiag=true;}}
    if(!haveDiag||!(H.A.diag[(std::size_t)i]>0.0)||!std::isfinite(H.A.diag[(std::size_t)i]))throw std::runtime_error("HXT5B exact directional setup Schur has invalid diagonal");
  }
  std::printf("NODALS_HXT5B_DIRECTIONAL_SETUP rows=%d nnz=%zu source=BX_RX_BXT_PLUS_BY_RY_BYT_PLUS_BZ_RZ_BZT status=PASS\\n",H.A.n,H.A.val.size());
}

template<class T>__global__ void refresh(std::size_t nnz,const std::int32_t*er,const std::int32_t*ac,const std::int64_t*br,const std::int32_t*bc,const T*bx,const T*by,const T*bz,const T*rx,const T*ry,const T*rz,T*av,int pin){size_t k=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(k>=nnz)return;int i=er[k],j=ac[k];if(i==pin){av[k]=(j==pin);return;}if(j==pin){av[k]=0;return;}auto a=br[i],ae=br[i+1],b=br[j],be=br[j+1];T s=0;while(a<ae&&b<be){int ga=bc[a],gb=bc[b];if(ga<gb){++a;continue;}if(gb<ga){++b;continue;}s+=rx[ga]*bx[a]*bx[b]+ry[ga]*by[a]*by[b]+rz[ga]*bz[a]*bz[b];++a;++b;}av[k]=s;}
template<class T>inline void refresh_pc(hxt4c::HybridAMG<T>&P,hxt4a::BDevice<T>&B,const T*rx,const T*ry,const T*rz){auto&F=P.fine;refresh<<<(F.val.n+255)/256,256>>>(F.val.n,F.entryRow.p,F.col.p,B.row.p,B.col.p,B.bx.p,B.by.p,B.bz.p,rx,ry,rz,F.val.p,F.pin);hxt4c::Buf<unsigned long long>bad(std::vector<unsigned long long>(1,0));hxt4c::diag_l1_kernel<<<(F.n+255)/256,256>>>(F.n,F.row.p,F.diagPos.p,F.val.p,F.diag.p,F.l1.p,bad.p);HXT1_CUDA(cudaDeviceSynchronize());unsigned long long q=0;HXT1_CUDA(cudaMemcpy(&q,bad.p,sizeof(q),cudaMemcpyDeviceToHost));if(q)throw std::runtime_error("HXT5B anisotropic fine Schur invalid");}
template<class T,class S>inline double parity(hxt4c::HybridAMG<T>&P,S&Sop,int n,int pin,hxt4a::Reducer<T>&red){std::vector<T>h(n);for(int i=0;i<n;++i)h[i]=(i==pin)?0:sin(.00131*(i+1))+.2*cos(.00077*(i+1));hxt4c::Buf<T>x(h),a(n),b(n),d(n);Sop.apply(x.p,a.p);P.fine.apply(x.p,b.p);hxt4c::diff_kernel<<<(n+255)/256,256>>>(n,a.p,b.p,d.p);return red.norm(d.p)/std::max(red.norm(a.p),1e-300);}


// HXT5B3_ROW_L1_IMPLEMENTATION
// Mirrors the validated turbulent NodalS momentum policy:
//   relaxMetric_i = sum_j |A_ij|
//   delta_i       = (1/alphaU - 1) * relaxMetric_i
//   Arel_ii       = Aphys_ii + delta_i
// while retaining the hybrid-specific pressure rule that the direct Nitsche
// penalty is excluded from the physical rAU denominator.

__global__ void h5b3_finalize_row_l1(
    int n,const unsigned char*fixed,const unsigned char*wallMask,
    const std::int64_t*row,const std::int32_t*diagPos,
    Real*xy,Real*z,
    const Real*diagRBase,const Real*conv,const Real*diagRTurb,
    const Real*diagDG,const Real*diagWall,
    Real*rxy,Real*rz,Real*diagOrigXY,Real*diagOrigZ,
    Real*relaxDeltaXY,Real*relaxDeltaZ,
    double alphaU,double rauScale,unsigned long long*bad)
{
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i>=n)return;
  const int d=diagPos[i];
  const Real ox=xy[d],oz=z[d];
  diagOrigXY[i]=ox;diagOrigZ[i]=oz;

  Real l1x=Real(0),l1z=Real(0);
  for(std::int64_t k=row[i];k<row[i+1];++k){
    l1x+=(Real)fabs((double)xy[k]);
    l1z+=(Real)fabs((double)z[k]);
  }
  const Real fac=(Real)(1.0/alphaU-1.0);
  const Real dx=fac*l1x,dz=fac*l1z;
  relaxDeltaXY[i]=dx;relaxDeltaZ[i]=dz;

  if(fixed[i]){
    rxy[i]=rz[i]=Real(0);
    return;
  }

  // Direct Nitsche penalty is absent from these physical diagonals.
  const Real noPenXY=diagRBase[i]+conv[d]+diagRTurb[i]+diagDG[i];
  const Real noPenZ =noPenXY+diagWall[i];

  if(!(ox>Real(0))||!(oz>Real(0))||!(l1x>Real(0))||!(l1z>Real(0))||
     !(noPenXY>Real(0))||!(noPenZ>Real(0))||
     !isfinite((double)ox)||!isfinite((double)oz)||
     !isfinite((double)dx)||!isfinite((double)dz)||
     !isfinite((double)noPenXY)||!isfinite((double)noPenZ)){
    atomicAdd(bad,1ULL);rxy[i]=rz[i]=Real(0);return;
  }

  xy[d]=ox+dx;
  z[d] =oz+dz;

  // The row-L1 delta is algorithmic under-relaxation and is retained in rAU.
  // The physical Nitsche penalty itself remains excluded.
  const Real mx=noPenXY+dx,mz=noPenZ+dz;
  if(!(mx>Real(0))||!(mz>Real(0))||
     !isfinite((double)mx)||!isfinite((double)mz)){
    atomicAdd(bad,1ULL);rxy[i]=rz[i]=Real(0);return;
  }
  rxy[i]=wallMask[i]?Real(0):(Real)(rauScale/(double)mx);
  rz[i]=(Real)(rauScale/(double)mz);
}

__global__ void h5b3_rhs(
    int n,const unsigned char*clamp,const Real*bt,const Real*extra,
    const Real*uold,const Real*relaxDelta,Real*rhs)
{
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i<n)rhs[i]=clamp[i]?Real(0):
    bt[i]+(extra?extra[i]:Real(0))+relaxDelta[i]*uold[i];
}

__global__ void h5b3_residual(
    int n,const unsigned char*clamp,const Real*b,const Real*Ax,Real*r)
{
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i<n)r[i]=clamp[i]?Real(0):(b[i]-Ax[i]);
}

inline double h5b3_relaxed_residual_norm(
    hxt4b::GpuCSR&A,const Real*val,const unsigned char*clamp,
    const Real*b,const Real*x,hxt4a::Reducer<Real>&red)
{
  hxt5b::spmv<<<(A.n+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(
    A.n,A.row.p,A.col.p,val,x,A.tmp.p);
  h5b3_residual<<<(A.n+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(
    A.n,clamp,b,A.tmp.p,A.res.p);
  HXT1_CUDA(cudaGetLastError());
  return red.norm(A.res.p);
}

struct H5B3MomentumResult {
  int sweeps=0;
  double initial=0.0;
  double final=0.0;
  double rel=0.0;
  bool ok=false;
};

inline H5B3MomentumResult h5b3_momentum_solve(
    const std::string&mode,int outer,hxt4b::GpuCSR&A,
    const hxt4b::ColoringHost&C,const Real*val,const Real*b,Real*x,
    double omega,const unsigned char*clamp,double rtol,double atol,int maxSweeps,
    hxt4a::Reducer<Real>&red,const char*component)
{
  H5B3MomentumResult R;
  R.initial=h5b3_relaxed_residual_norm(A,val,clamp,b,x,red);
  if(!std::isfinite(R.initial))
    throw std::runtime_error(std::string("HXT5B3 nonfinite initial momentum residual comp=")+component);
  if(R.initial<=atol){R.final=R.initial;R.rel=0.0;R.ok=true;return R;}
  const double target=std::max(atol,rtol*R.initial);

  for(int q=0;q<maxSweeps;++q){
    if(mode=="sgs1"){
      hxt5b::sweep(A,C,val,b,x,omega,true,clamp);
      hxt5b::sweep(A,C,val,b,x,omega,false,clamp);
    }else if(mode=="fgs1"){
      hxt5b::sweep(A,C,val,b,x,omega,true,clamp);
    }else if(mode=="altgs1"){
      hxt5b::sweep(A,C,val,b,x,omega,((outer+q)&1)!=0,clamp);
    }else throw std::runtime_error("HXT5B3 bad momentum work mode");

    R.sweeps=q+1;
    R.final=h5b3_relaxed_residual_norm(A,val,clamp,b,x,red);
    R.rel=R.final/std::max(R.initial,1e-300);
    if(!std::isfinite(R.final))
      throw std::runtime_error(std::string("HXT5B3 momentum SGS produced NaN/Inf comp=")+component);
    if(outer==1 && q<6)
      std::printf("NODALS_HXT5B3_MOM_SWEEP comp=%s sweep=%d initial=%.12e current=%.12e rel=%.12e targetRel=%.6g status=PASS\n",
        component,q+1,R.initial,R.final,R.rel,rtol);
    if(R.final<=target){R.ok=true;break;}
  }
  return R;
}

__global__ void h5b3_physical_residual(
    int n,const unsigned char*clamp,const Real*bt,const Real*extra,
    const Real*u,const Real*relaxDelta,const Real*ArelU,Real*r)
{
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i<n){
    if(clamp[i]){r[i]=Real(0);return;}
    r[i]=bt[i]+(extra?extra[i]:Real(0))-(ArelU[i]-relaxDelta[i]*u[i]);
  }
}

inline double h5b3_physical_residual_norm(
    hxt4b::GpuCSR&A,const Real*val,const unsigned char*clamp,
    const Real*bt,const Real*extra,const Real*u,const Real*relaxDelta,
    hxt4a::Reducer<Real>&red)
{
  hxt5b::spmv<<<(A.n+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(
    A.n,A.row.p,A.col.p,val,u,A.tmp.p);
  h5b3_physical_residual<<<(A.n+hxt4b::TPB-1)/hxt4b::TPB,hxt4b::TPB>>>(
    A.n,clamp,bt,extra,u,relaxDelta,A.tmp.p,A.res.p);
  HXT1_CUDA(cudaGetLastError());
  return red.norm(A.res.p);
}

__global__ void h5b3_subtract(int n,const Real*a,const Real*b,Real*r){
  int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i<n)r[i]=a[i]-b[i];
}

inline void h5b3_relax_audit(
    const char*label,const Real*diagOrig,const Real*delta,int n)
{
  std::vector<Real>d((std::size_t)n),q((std::size_t)n);
  HXT1_CUDA(cudaMemcpy(d.data(),diagOrig,(std::size_t)n*sizeof(Real),cudaMemcpyDeviceToHost));
  HXT1_CUDA(cudaMemcpy(q.data(),delta,(std::size_t)n*sizeof(Real),cudaMemcpyDeviceToHost));
  double minRatio=std::numeric_limits<double>::infinity(),maxRatio=0,mean=0;std::size_t used=0;
  for(int i=0;i<n;++i){
    const double a=(double)d[(std::size_t)i],b=(double)q[(std::size_t)i];
    if(a>0&&std::isfinite(a)&&std::isfinite(b)){
      const double r=b/a;minRatio=std::min(minRatio,r);maxRatio=std::max(maxRatio,r);mean+=r;++used;
    }
  }
  if(used)mean/=used;
  std::printf("NODALS_HXT5B3_ROWL1_AUDIT label=%s rows=%zu deltaOverPhysDiagMin=%.12e mean=%.12e max=%.12e status=PASS\n",
    label,used,minRatio,mean,maxRatio);
}

} // namespace hxt5b
