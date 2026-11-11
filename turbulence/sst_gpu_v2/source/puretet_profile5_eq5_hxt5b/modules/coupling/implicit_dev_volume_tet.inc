// NODALS_IMPLICIT_DEVIATORIC_VOLUME_V1
// Pure tetrahedron P1+BF3 momentum volume constitutive correction:
//    A_ij(a,b) += integral nu_eff [d_j phi_a d_i phi_b
//                         -(2/3) d_i phi_a d_j phi_b] dV.
// Together with the EXISTING scalar diffusion int nu_eff grad(phi_a).grad(phi_b)
// on A_ii this is exactly int 2 nu_eff dev(sym grad U):grad(V) dV.
// nu_eff is frozen at the current outer nonlinear iterate (Picard), not
// at a previous momentum iteration. No lagged correction is added to RHS.
// Same 24-point quadrature / local_sst provider as sst_mom_tet.

__device__ __forceinline__ void implicit_dev_tet_quadrature(
    std::uint64_t cell, const nodals_hxt1::Point*pts,
    const nodals_hxt1::TetConn*tc,const nodals_hxt1::TetVel*tv,
    const nodals_hxt1::TetGeom*tg,
    const Real*u0,const Real*u1,const Real*u2,
    const Real*k,const Real*omega,double radius,double nu,
    double kFloor,double omegaFloor,
    Real U[3][8],Real K[8],Real W[8],Real GP[24][8][3],Real NUW[24]){
  const int tid=(int)threadIdx.x;
  if(tid<8){
    const int a=tv[cell].g[tid];
    U[0][tid]=u0[a];U[1][tid]=u1[a];U[2][tid]=u2[a];
    K[tid]=k[a];W[tid]=omega[a];
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
        const Real g=(Real)gp[a][d];GP[tid][a][d]=g;
        gk[d]+=K[a]*g;gw[d]+=W[a]*g;
        gu[d]+=U[0][a]*g;gu[3+d]+=U[1][a]*g;gu[6+d]+=U[2][a]*g;
      }
    }
    hxt5c::eval_dg_omega_p1((int)cell,l,tg[cell],wv,gw);
    double x=0.,y=0.;
    for(int a=0;a<4;++a){const auto p=pts[tc[cell].v[a]];x+=l[a]*p.x;y+=l[a]*p.y;}
    const auto ss=hxt5c::local_sst((Real)x,(Real)y,(Real)radius,(Real)nu,
      gu,kv,wv,gk,gw,(Real)kFloor,(Real)omegaFloor,hxt5c::dg_omega_p1_constitutive?hxt5c::stage20_distance(tc,(int)cell,l):Real(-1));
    NUW[tid]=((Real)nu+ss.nut)*(Real)(tg[cell].det*hxt5a::h5_tet24_w[tid]);
  }
  __syncthreads();
}

// Add the complete implicit cross-component contribution directly to the
// existing nine-block M3 CSR matrix (physical P1+BF3 basis BEFORE T/Tt).
__global__ void implicit_dev_tet_add_m3(
    const nodals_hxt1::Point*pts,const nodals_hxt1::TetConn*tc,
    const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,
    const std::int32_t*slot,const Real*u0,const Real*u1,const Real*u2,
    const Real*k,const Real*omega,std::uint64_t nc,
    double radius,double nu,double kFloor,double omegaFloor,
    std::size_t nnz,Real*mat){
  const std::uint64_t cell=(std::uint64_t)blockIdx.x;
  if(cell>=nc)return;
  __shared__ Real U[3][8],K[8],W[8],GP[24][8][3],NUW[24];
  implicit_dev_tet_quadrature(cell,pts,tc,tv,tg,u0,u1,u2,k,omega,
      radius,nu,kFloor,omegaFloor,U,K,W,GP,NUW);
  for(int p=(int)threadIdx.x;p<64;p+=(int)blockDim.x){
    const int a=p/8,b=p%8;
    const std::int32_t loc=slot[cell*64ull+(std::size_t)p];
    for(int i=0;i<3;++i)for(int j=0;j<3;++j){
      double v=0.0;
      for(int q=0;q<24;++q){
        v+=(double)NUW[q]*((double)GP[q][a][j]*GP[q][b][i]
          -(2.0/3.0)*(double)GP[q][a][i]*GP[q][b][j]);
      }
      atomicAdd(mat+((std::size_t)i*3+j)*nnz+(std::size_t)loc,(Real)v);
    }
  }
}

// Identical quadrature applied without CSR storage for coupled FGMRES and
// full-op diagnostics. x is the physical (T-projected) BF3 velocity vector;
// y is the physical residual before applying T^t.
__global__ void implicit_dev_tet_apply(
    const nodals_hxt1::Point*pts,const nodals_hxt1::TetConn*tc,
    const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,
    const Real*u0,const Real*u1,const Real*u2,
    const Real*k,const Real*omega,std::uint64_t nc,
    double radius,double nu,double kFloor,double omegaFloor,
    int nv,const Real*x,Real*y){
  const std::uint64_t cell=(std::uint64_t)blockIdx.x;
  if(cell>=nc)return;
  __shared__ Real U[3][8],K[8],W[8],GP[24][8][3],NUW[24];
  implicit_dev_tet_quadrature(cell,pts,tc,tv,tg,u0,u1,u2,k,omega,
      radius,nu,kFloor,omegaFloor,U,K,W,GP,NUW);
  for(int a=(int)threadIdx.x;a<8;a+=(int)blockDim.x){
    double s[3]={};
    for(int b=0;b<8;++b){
      const int ib=tv[cell].g[b];
      const double xb[3]={double(x[ib]),double(x[(std::size_t)nv+ib]),
                           double(x[2*(std::size_t)nv+ib])};
      for(int i=0;i<3;++i)for(int j=0;j<3;++j){
        double coeff=0.0;
        for(int q=0;q<24;++q){
          coeff+=(double)NUW[q]*((double)GP[q][a][j]*GP[q][b][i]
            -(2.0/3.0)*(double)GP[q][a][i]*GP[q][b][j]);
        }
        s[i]+=coeff*xb[j];
      }
    }
    const int ia=tv[cell].g[a];
    for(int i=0;i<3;++i)atomicAdd(y+(std::size_t)i*nv+ia,(Real)s[i]);
  }
}
