// NODALS_EMAC_PICARD_V1
// EMAC nonlinear convection: 2 D(u) u + (div u) u.
// Consistent single-carrier Picard linearization, with a = previous U:
// C_EMAC(a;u)_i = a_j d_j(u_i) + a_j d_i(u_j) + a_i d_j(u_j).
// Existing scalar A.conv provides FIRST term (a . grad) u_i.
// This file supplies the two EXTRA, genuinely cross-component terms in M3:
//    E_ij(test a, trial b) = int phi_a [ a_j d_i(phi_b)
//                                          + a_i d_j(phi_b) ] dV.
//
// For u==a, exactly recovers 2 D(a)a + (div a)a, including P1+BF3.
// The changed pressure variable is p_EMAC=p_physical - |u|^2/2 (rho=1).
// Existing boundary fluxes and pressure output remain unchanged: experimental
// VOLUME-ONLY implementation, pressure drops are NOT physical until corrected.
// Frozen a is the current Picard velocity; M3 CSR is Tt M T projected.

// Matrix add is one CTA per tetrahedron and one thread per local (a,b) pair.
// This kernel does not add the advective term: A.conv already did that.
__global__ void emac_picard_tet_add_m3(
    const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,
    const std::int32_t*slot,const Real*u0,const Real*u1,const Real*u2,
    std::uint64_t nc,std::size_t nnz,Real*blocks){
  const std::uint64_t cell=(std::uint64_t)blockIdx.x;
  if(cell>=nc)return;
  const int tid=(int)threadIdx.x;
  __shared__ double A[8][3];
  if(tid<8){
    const int ib=tv[cell].g[tid];
    A[tid][0]=(double)u0[ib];
    A[tid][1]=(double)u1[ib];
    A[tid][2]=(double)u2[ib];
  }
  __syncthreads();
  if(tid>=64)return;
  const int a=tid/8,b=tid%8;
  const auto& geom=tg[cell];
  // J_m,i = int phi_a phi_m d_i(phi_b) d(reference volume),
  // multiplied by physical cell det at final assembly.
  double J[8][3]={{0.0}};
  for(int m=0;m<8;++m)for(int r=0;r<3;++r){
    const double t=hxt4b::h4b_tet_convT[(((a*8+m)*8+b)*3+r)];
    for(int i=0;i<3;++i)J[m][i]+=t*geom.invJ[3*r+i];
  }
  const std::int32_t position=slot[cell*64ull+(std::size_t)tid];
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){
    double v=0.0;
    for(int m=0;m<8;++m)
      v+=A[m][j]*J[m][i] + A[m][i]*J[m][j];
    atomicAdd(blocks+((std::size_t)i*3+j)*nnz+(std::size_t)position,
              (Real)(geom.det*v));
  }
}

// Exact matrix-free action of the same extra cross-component Picard tensor.
// The existing S22D4DOperator::apply adds the advective diagonal A.conv term.
// Input is already converted from BF3-constrained reduced to physical basis;
// caller performs left BF3 Tt and fixed masks AFTER this kernel.
__global__ void emac_picard_tet_apply(
    const nodals_hxt1::TetVel*tv,const nodals_hxt1::TetGeom*tg,
    const Real*u0,const Real*u1,const Real*u2,
    std::uint64_t nc,int nv,const Real*x,Real*y){
  const std::uint64_t cell=(std::uint64_t)blockIdx.x;
  if(cell>=nc)return;
  const int tid=(int)threadIdx.x;
  __shared__ double A[8][3];
  if(tid<8){
    const int ib=tv[cell].g[tid];
    A[tid][0]=(double)u0[ib];
    A[tid][1]=(double)u1[ib];
    A[tid][2]=(double)u2[ib];
  }
  __syncthreads();
  if(tid>=8)return;
  const int a=tid;
  const auto& geom=tg[cell];
  double sum[3]={0.0,0.0,0.0};
  for(int b=0;b<8;++b){
    const int ib=tv[cell].g[b];
    const double xb[3]={double(x[ib]),double(x[(std::size_t)nv+ib]),
                         double(x[2*(std::size_t)nv+ib])};
    double J[8][3]={{0.0}};
    for(int m=0;m<8;++m)for(int r=0;r<3;++r){
      const double t=hxt4b::h4b_tet_convT[(((a*8+m)*8+b)*3+r)];
      for(int i=0;i<3;++i)J[m][i]+=t*geom.invJ[3*r+i];
    }
    for(int i=0;i<3;++i)for(int j=0;j<3;++j){
      double v=0.0;
      for(int m=0;m<8;++m)
        v+=A[m][j]*J[m][i]+A[m][i]*J[m][j];
      sum[i]+=v*xb[j];
    }
  }
  const int ia=tv[cell].g[a];
  for(int i=0;i<3;++i)
    atomicAdd(y+(std::size_t)i*nv+ia,(Real)(geom.det*sum[i]));
}
