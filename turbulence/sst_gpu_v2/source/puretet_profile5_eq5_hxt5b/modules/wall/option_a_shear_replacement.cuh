// OPTION_A_SHEAR_REPLACEMENT_V2
// Replace only the wall-normal/tangential DEVIATORIC SHEAR subspace in the
// wall-owner P1+BF3 tetrahedron, preserving all other SST momentum diffusion.
//
// The original full deviatoric volume form is
// a0(U,V) = int_T 2 nu_eff dev(DU):dev(DV) dV.
// With unit wall-face normal n define the orthogonal tangent projector
// P=I-n*n^T, g(U)=P [(grad U + grad U^T) n] = 2 P D(U)n.
// Then the shear subspace contributes exactly nu_eff*g(U).g(V) to a0.
// This kernel adds
//   delta_a = int_T (nu_wall - nu_eff(x)) g(U).g(V) dV
// which SUBSTITUTES only that shear contribution.
// If nu_wall==nu_eff pointwise, delta_a vanishes algebraically.
// Since the deviatoric shear subspace is orthogonal to remaining strain,
// replacing its positive coefficient preserves viscous positive semidefiniteness.
//
// Called ONLY for Option A, after usual scalar plus implicit-dev diffusion,
// before applying T/Tt wall BF3 transforms. The same quadrature/provider is
// used in its 9-block CSR and matrix-free operator variants.
// This is a cellwise shear-space approximation, not an exact OF face stencil.

__device__ __forceinline__ double option_a_g_component(
    const Real grad[3],double dn,const double n[3],int comp,int t)
{
  return (comp==t ? dn : 0.) + n[comp]*(double)grad[t]
         - 2.*n[comp]*n[t]*dn;
}

__device__ __forceinline__ double option_a_g_dot(
    const Real ga[3],const Real gb[3],const double n[3],int i,int j)
{
  const double dna=(double)ga[0]*n[0]+(double)ga[1]*n[1]+(double)ga[2]*n[2];
  const double dnb=(double)gb[0]*n[0]+(double)gb[1]*n[1]+(double)gb[2]*n[2];
  double sum=0.;
  for(int t=0;t<3;++t)
    sum+=option_a_g_component(ga,dna,n,i,t)
        *option_a_g_component(gb,dnb,n,j,t);
  return sum;
}

// Host evaluates the same F2 Spalding u_tau and constructs a wall viscosity.
// Each wall face has one unique owner tetrahedron (validated at startup).
// The gradient and SST state for this 24-point volume quadrature are frozen
// at the current SIMPLE iteration, just like the existing implicit-dev matrix.
__device__ __forceinline__ void option_a_shear_quad(
    const hxt5b::TetFacePlan face,
    const nodals_hxt1::Point* pts,const nodals_hxt1::TetConn* tc,
    const nodals_hxt1::TetVel* tv,const nodals_hxt1::TetGeom* tg,
    const Real* ux,const Real* uy,const Real* uz,
    const Real* k,const Real* omega,const Real* wallNu,
    double radius,double nu,double kFloor,double omegaFloor,
    Real GP[24][8][3],double DW[24])
{
  const int cell=face.cell;
  const int tid=(int)threadIdx.x;
  __shared__ Real U[3][8],K[8],W[8];
  if(tid<8){
    const int gid=tv[cell].g[tid];
    U[0][tid]=ux[gid];U[1][tid]=uy[gid];U[2][tid]=uz[gid];
    K[tid]=k[gid];W[tid]=omega[gid];
  }
  __syncthreads();
  if(tid<24){
    double l[4]={hxt5a::h5_tet24_l0[tid],hxt5a::h5_tet24_l1[tid],
      hxt5a::h5_tet24_l2[tid],hxt5a::h5_tet24_l3[tid]};
    double ph[8],gr[8][3],gp[8][3];
    hxt2::tet_basis(l[0],l[1],l[2],l[3],ph,gr);
    hxt2::tet_phys_grad(gr,tg[cell],gp);
    Real gu[9]={},gk[3]={},gw[3]={},kv=Real(0),wv=Real(0);
    for(int a=0;a<8;++a){
      const Real pa=(Real)ph[a];kv+=K[a]*pa;wv+=W[a]*pa;
      for(int d=0;d<3;++d){
        const Real g=(Real)gp[a][d]; GP[tid][a][d]=g;
        gu[d]+=U[0][a]*g; gu[3+d]+=U[1][a]*g; gu[6+d]+=U[2][a]*g;
        gk[d]+=K[a]*g;gw[d]+=W[a]*g;
      }
    }
    double x=0.,y=0.;
    for(int a=0;a<4;++a){
      const auto P=pts[tc[cell].v[a]];
      x+=l[a]*P.x; y+=l[a]*P.y;
    }
    const auto ss=hxt5c::local_sst((Real)x,(Real)y,(Real)radius,(Real)nu,
       gu,kv,wv,gk,gw,(Real)kFloor,(Real)omegaFloor);
    DW[tid]=((double)wallNu[cell]-nu-(double)ss.nut) *
            (double)(tg[cell].det*hxt5a::h5_tet24_w[tid]);
  }
  __syncthreads();
}

__global__ void option_a_shear_add_m3(
    const hxt5b::TetFacePlan*faces,int nf,
    const nodals_hxt1::Point* pts,const nodals_hxt1::TetConn* tc,
    const nodals_hxt1::TetVel* tv,const nodals_hxt1::TetGeom* tg,
    const std::int32_t*slot,
    const Real* ux,const Real* uy,const Real* uz,
    const Real* k,const Real* omega,const Real* wallNu,
    double radius,double nu,double kFloor,double omegaFloor,
    std::size_t nnz,Real*matrix)
{
  const int fi=(int)blockIdx.x;if(fi>=nf)return;
  const auto face=faces[fi];
  __shared__ Real GP[24][8][3];
  __shared__ double DW[24];
  option_a_shear_quad(face,pts,tc,tv,tg,ux,uy,uz,k,omega,wallNu,
        radius,nu,kFloor,omegaFloor,GP,DW);
  const double n[3]={face.n[0],face.n[1],face.n[2]};
  for(int p=(int)threadIdx.x;p<64;p+=(int)blockDim.x){
    const int a=p/8,b=p%8;
    const std::size_t loc=(std::size_t)slot[(std::size_t)face.cell*64u+(std::size_t)p];
    for(int i=0;i<3;++i)for(int j=0;j<3;++j){
      double val=0.;
      for(int q=0;q<24;++q)
        val+=DW[q]*option_a_g_dot(GP[q][a],GP[q][b],n,i,j);
      atomicAdd(matrix+((std::size_t)i*3u+(std::size_t)j)*nnz+loc,(Real)val);
    }
  }
}

__global__ void option_a_shear_apply(
    const hxt5b::TetFacePlan*faces,int nf,
    const nodals_hxt1::Point* pts,const nodals_hxt1::TetConn* tc,
    const nodals_hxt1::TetVel* tv,const nodals_hxt1::TetGeom* tg,
    const Real* ux,const Real* uy,const Real* uz,
    const Real* k,const Real* omega,const Real* wallNu,
    double radius,double nu,double kFloor,double omegaFloor,
    int nv,const Real*x,Real*y)
{
  const int fi=(int)blockIdx.x;if(fi>=nf)return;
  const auto face=faces[fi];
  __shared__ Real GP[24][8][3];
  __shared__ double DW[24];
  option_a_shear_quad(face,pts,tc,tv,tg,ux,uy,uz,k,omega,wallNu,
        radius,nu,kFloor,omegaFloor,GP,DW);
  const double n[3]={face.n[0],face.n[1],face.n[2]};
  for(int a=(int)threadIdx.x;a<8;a+=blockDim.x){
    double accum[3]={0.,0.,0.};
    for(int b=0;b<8;++b){
      const int ib=tv[face.cell].g[b];
      for(int i=0;i<3;++i)for(int j=0;j<3;++j){
        double val=0.;
        for(int q=0;q<24;++q)
          val+=DW[q]*option_a_g_dot(GP[q][a],GP[q][b],n,i,j);
        accum[i]+=val*(double)x[(std::size_t)j*nv+ib];
      }
    }
    const int ia=tv[face.cell].g[a];
    for(int i=0;i<3;++i)
      atomicAdd(y+(std::size_t)i*nv+ia,(Real)accum[i]);
  }
}
