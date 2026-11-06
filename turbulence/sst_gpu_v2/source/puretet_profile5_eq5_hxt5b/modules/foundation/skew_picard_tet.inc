// NODALS_SKEW_PICARD_V1
// Temam/skew Picard momentum volume operator for P1+BF3 tetrahedral elements.
// Carrier a = frozen, pressure-corrected u*; unknown = transported u.
// C_ab = integral phi_a [ (a.grad)phi_b + 0.5 phi_b div(a) ] dV.
// Existing h4b_tet_convT[ a, m, b, j ] = int phi_a phi_m d_j(phi_b).
// With a_m lagged, the divergence part is T[a,b,m,j]/2.
// No change to the separately assembled SST scalar-convection operator.
__global__ void tet_conv_assemble_skew_thread_per_cell(
    const nodals_hxt1::TetVel* tv,
    const nodals_hxt1::TetGeom* tg,
    const std::int32_t* slot,
    const Real* u0,const Real* u1,const Real* u2,
    Real* conv,std::uint64_t n)
{
  const std::uint64_t c=(std::uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(c>=n)return;
  const auto& V=tv[c];
  const auto& g=tg[c];
  double ur[8][3];
#pragma unroll
  for(int m=0;m<8;++m){
    const int gv=V.g[m];
    const double ux=(double)u0[gv];
    const double uy=(double)u1[gv];
    const double uz=(double)u2[gv];
#pragma unroll
    for(int j=0;j<3;++j)
      ur[m][j]=ux*g.invJ[3*j+0]+uy*g.invJ[3*j+1]+uz*g.invJ[3*j+2];
  }
#pragma unroll
  for(int a=0;a<8;++a){
#pragma unroll
    for(int b=0;b<8;++b){
      double q=0.0;
#pragma unroll
      for(int m=0;m<8;++m){
#pragma unroll
        for(int j=0;j<3;++j){
          q+=ur[m][j]*(
            h4b_tet_convT[(((a*8+m)*8+b)*3+j)] +
            0.5*h4b_tet_convT[(((a*8+b)*8+m)*3+j)]
          );
        }
      }
      q*=g.det;
      atomicAdd(conv+slot[c*64ull+(std::size_t)a*8ull+b],(Real)q);
    }
  }
}
