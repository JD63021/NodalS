#pragma once
#include "h9_rans_mixlen.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <cmath>
#include <stdexcept>

namespace nodals_gpu {

struct H10DgInletFaceDevice {
  std::int32_t cell=0;
  std::uint8_t opp=0;
  OperatorReal area=0;
  OperatorReal normal[3]={0,0,0};
};

inline std::vector<H10DgInletFaceDevice> h10_build_inlet_faces(
    const SerialTetMesh&M,const G4SetupHost&S)
{
  std::vector<H10DgInletFaceDevice> out;
  if(!S.dgInlet)return out;
  const auto&P=M.patches[(std::size_t)S.pipe.inlet];
  out.reserve((std::size_t)P.n_faces);
  for(int f=P.start_face;f<P.start_face+P.n_faces;++f){
    const int c=M.owner[(std::size_t)f];
    int opp=-1;
    for(int i=0;i<4;++i)if(M.opp_face[(std::size_t)c][i]==f){opp=i;break;}
    if(opp<0)throw std::runtime_error("Gate2 DG inlet opposite face not found");
    const Vec3d sf=face_outward_area_vector_g2(M,f);
    const double area=std::sqrt(sf.x*sf.x+sf.y*sf.y+sf.z*sf.z);
    if(!(area>0.0))throw std::runtime_error("Gate2 DG inlet degenerate face");
    H10DgInletFaceDevice q;
    q.cell=c;q.opp=(std::uint8_t)opp;q.area=OperatorReal(area);
    q.normal[0]=OperatorReal(sf.x/area);q.normal[1]=OperatorReal(sf.y/area);q.normal[2]=OperatorReal(sf.z/area);
    out.push_back(q);
  }
  return out;
}

__device__ __constant__ double h10_l0[12]={
  0.063089014491502,0.063089014491502,0.873821971016996,
  0.249286745170910,0.249286745170910,0.501426509658180,
  0.053145049844816,0.053145049844816,0.310352451033785,
  0.310352451033785,0.636502499121399,0.636502499121399};
__device__ __constant__ double h10_l1[12]={
  0.063089014491502,0.873821971016996,0.063089014491502,
  0.249286745170910,0.501426509658180,0.249286745170910,
  0.310352451033785,0.636502499121399,0.053145049844816,
  0.636502499121399,0.053145049844816,0.310352451033785};
__device__ __constant__ double h10_l2[12]={
  0.873821971016996,0.063089014491502,0.063089014491502,
  0.501426509658180,0.249286745170910,0.249286745170910,
  0.636502499121399,0.310352451033785,0.636502499121399,
  0.053145049844816,0.310352451033785,0.053145049844816};
__device__ __constant__ double h10_w[12]={
  0.050844906370207,0.050844906370207,0.050844906370207,
  0.116786275726379,0.116786275726379,0.116786275726379,
  0.082851075618374,0.082851075618374,0.082851075618374,
  0.082851075618374,0.082851075618374,0.082851075618374};

__device__ __forceinline__ OperatorReal h10_phi(int a,const OperatorReal lam[4]){
  if(a<4)return lam[a];
  const int i=a-4;OperatorReal p=OperatorReal(27);
  #pragma unroll
  for(int j=0;j<4;++j)if(j!=i)p*=lam[j];
  return p;
}

__device__ __forceinline__ OperatorReal h10_nut(
    const G4CellPlanDevice&cp,const OperatorReal coeff[3][8],const OperatorReal grad[8][3],
    const OperatorReal lam[4],const OperatorReal*xy,
    OperatorReal cx,OperatorReal cy,OperatorReal radius,OperatorReal scale)
{
  if(!(scale>OperatorReal(0)))return OperatorReal(0);
  OperatorReal gu[3][3]={{0}};
  #pragma unroll
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int a=0;a<8;++a)gu[i][j]+=coeff[i][a]*grad[a][j];
  OperatorReal ss=0;
  #pragma unroll
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){const OperatorReal sij=OperatorReal(0.5)*(gu[i][j]+gu[j][i]);ss+=sij*sij;}
  const OperatorReal strain=sqrt(fmax(OperatorReal(0),OperatorReal(2)*ss));
  OperatorReal xq=0,yq=0;
  #pragma unroll
  for(int i=0;i<4;++i){xq+=lam[i]*xy[2*i];yq+=lam[i]*xy[2*i+1];}
  const OperatorReal dx=xq-cx,dy=yq-cy;
  OperatorReal eta=sqrt(dx*dx+dy*dy)/radius;eta=fmin(OperatorReal(1),fmax(OperatorReal(0),eta));
  const OperatorReal e2=eta*eta,e4=e2*e2;
  const OperatorReal ell=radius*fmax(OperatorReal(0),OperatorReal(0.14)-OperatorReal(0.08)*e2-OperatorReal(0.06)*e4);
  return scale*ell*ell*strain;
}

// Gate 2 parity implementation.  There are only O(10^3) inlet faces on the
// VMFL003 development mesh, so one CUDA thread per face is intentionally used
// here to keep the weak-form port literal and auditable.  It remains fully
// device resident and is negligible next to the volume mixing-length kernel.
__global__ void h10_dg_inlet_kernel(
    const H10DgInletFaceDevice* __restrict__ faces,int nf,
    const G4CellPlanDevice* __restrict__ cells,const std::int64_t* __restrict__ row,
    const StateReal* __restrict__ fixed,const StateReal* __restrict__ u0,
    const StateReal* __restrict__ u1,const StateReal* __restrict__ u2,
    const OperatorReal* __restrict__ xy,int mixingLength,
    OperatorReal cx,OperatorReal cy,OperatorReal radius,OperatorReal nu,OperatorReal mixScale,
    OperatorReal uh0,OperatorReal uh1,OperatorReal uh2,
    OperatorReal* __restrict__ aval,StateReal* __restrict__ cr0,
    StateReal* __restrict__ cr1,StateReal* __restrict__ cr2)
{
  const int fi=(int)(blockIdx.x*blockDim.x+threadIdx.x);if(fi>=nf)return;
  const auto&F=faces[fi];const auto&cp=cells[F.cell];
  OperatorReal coeff[3][8];
  for(int a=0;a<8;++a){coeff[0][a]=OperatorReal(g4_ref_value(cp,a,0,u0,u1,u2,fixed));coeff[1][a]=OperatorReal(g4_ref_value(cp,a,1,u0,u1,u2,fixed));coeff[2][a]=OperatorReal(g4_ref_value(cp,a,2,u0,u1,u2,fixed));}
  OperatorReal G[64];OperatorReal R[24];for(int i=0;i<64;++i)G[i]=0;for(int i=0;i<24;++i)R[i]=0;
  const OperatorReal bn=uh0*F.normal[0]+uh1*F.normal[1]+uh2*F.normal[2];
  const OperatorReal inflow=fmax(OperatorReal(0),-bn);
  int fv[3],kk=0;for(int i=0;i<4;++i)if(i!=(int)F.opp)fv[kk++]=i;
  for(int q=0;q<12;++q){
    OperatorReal lam[4]={0,0,0,0};lam[fv[0]]=OperatorReal(h10_l0[q]);lam[fv[1]]=OperatorReal(h10_l1[q]);lam[fv[2]]=OperatorReal(h10_l2[q]);
    OperatorReal val[8],grad[8][3],dn[8];
    for(int a=0;a<8;++a){val[a]=h10_phi(a,lam);for(int d=0;d<3;++d)grad[a][d]=h9_basis_grad(cp,a,lam,d);dn[a]=grad[a][0]*F.normal[0]+grad[a][1]*F.normal[1]+grad[a][2]*F.normal[2];}
    OperatorReal nut=0;if(mixingLength)nut=h10_nut(cp,coeff,grad,lam,xy+(std::size_t)F.cell*8,cx,cy,radius,mixScale);
    const OperatorReal nue=nu+nut,w=OperatorReal(h10_w[q])*F.area;
    for(int a=0;a<8;++a){
      for(int b=0;b<8;++b)G[8*a+b]+=w*(inflow*val[a]*val[b]+nue*(-val[a]*dn[b]+dn[a]*val[b]));
      R[3*a+0]+=w*(inflow*val[a]*uh0+nue*dn[a]*uh0);
      R[3*a+1]+=w*(inflow*val[a]*uh1+nue*dn[a]*uh1);
      R[3*a+2]+=w*(inflow*val[a]*uh2+nue*dn[a]*uh2);
    }
  }
  for(int a=0;a<8;++a){const int rr=cp.ref[a];if(rr<0)continue;
    atomicAdd(cr0+rr,StateReal(R[3*a+0]));atomicAdd(cr1+rr,StateReal(R[3*a+1]));atomicAdd(cr2+rr,StateReal(R[3*a+2]));
    for(int b=0;b<8;++b){const OperatorReal gv=G[8*a+b];if(cp.ref[b]>=0){const unsigned char sl=cp.rowSlot[8*a+b];if(sl!=255)atomicAdd(aval+row[rr]+sl,gv);}else{const int fs=-cp.ref[b]-1;atomicAdd(cr0+rr,-StateReal(gv)*fixed[3*(std::size_t)fs+0]);atomicAdd(cr1+rr,-StateReal(gv)*fixed[3*(std::size_t)fs+1]);atomicAdd(cr2+rr,-StateReal(gv)*fixed[3*(std::size_t)fs+2]);}}
  }
}

} // namespace nodals_gpu
