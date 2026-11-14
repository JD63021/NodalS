// NODALS_HYBRID_COUPLED_V1 — opt-in mixed-mesh block solver support.
//
// Exact frozen-momentum operator for CURRENT hybrid advective/vector-Laplacian
// physics: K = diag(Ax,Ay,Az); cross blocks zero because legacy HEX wall
// clamps transverse velocity and imposes axial Spalding only.
// This does NOT purport to be the more general pure-tet vector-wall operator.
//
// The M3 path reuses native 3x3 block Gauss-Seidel, with no BF3 T/Tt,
// because the hybrid mesh has NO tetrahedral wall faces to project.
// FGMRES reuses the native coupled 3Nv matrix-free block solver, with the
// same K. Both select current A/pressure/SST and are solver-only changes.
// No source changes to the user's production SST solver.
__global__ void hybrid_coupled_pack(
    int nv,const Real*u0,const Real*u1,const Real*u2,
    const Real*b0,const Real*b1,const Real*b2,
    const unsigned char*clampXY,const unsigned char*fixedZ,
    Real*x,Real*b)
{
  int i=(int)blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=nv)return;
  const Real*uu[3]={u0,u1,u2},*bb[3]={b0,b1,b2};
  for(int c=0;c<3;++c){
    const bool mask=(c==2 ? fixedZ[i]!=0 : clampXY[i]!=0);
    x[(std::size_t)c*nv+i]=mask?Real(0):uu[c][i];
    b[(std::size_t)c*nv+i]=mask?Real(0):bb[c][i];
  }
}

// Reuse exact HXT5D M3 9-block CSR assembly and 3x3-colour FGS.
// Off-blocks are genuinely zero for the selected existing physical operator,
// *not* hidden/lagged Nitsche interactions.
// Because there are no TET wall faces, faceByBubble[i]=-1 for every node.
// Return false if two sweeps increase the *full* frozen matrix residual.
inline S22D4DResult hybrid_coupled_m3_fgs(
    const S22D4DOperator&op,Real*x,const Real*b,
    Real*k,Real*backup,const int*faceByBubble,
    unsigned long long*bad,int sweeps,double omega,int outer,
    hxt4a::Reducer<Real>&red,Real*ax,Real*r)
{
  const int nv=op.A.n;
  if(nv<=0||op.nfaces!=0||op.bf3FluxMode||op.f1bFaceCoupling)
    throw std::runtime_error("HYBRID_M3 requires original unprojected HEX/TET momentum");
  if(sweeps<1||sweeps>60||!(omega>0. && omega<=1.))
    throw std::runtime_error("HYBRID_M3 invalid FGS sweeps or omega");
  const std::size_t bytes=(std::size_t)3*nv*sizeof(Real);
  HXT1_CUDA(cudaMemcpy(backup,x,bytes,cudaMemcpyDeviceToDevice));
  HXT1_CUDA(cudaMemset(bad,0,sizeof(unsigned long long)));
  m3_init_blocks<<<(op.A.nnz+255)/256,256>>>(
     op.A.nnz,op.v0,op.v1,op.v2,k);
  HXT1_CUDA(cudaGetLastError());

  auto residual=[&](){
    m3_spmv<<<(nv+255)/256,256>>>(nv,op.A.nnz,op.A.row.p,op.A.col.p,
      k,x,ax,op.xy,op.fixed);
    s22d4d_residual<<<(3*nv+255)/256,256>>>(nv,b,ax,op.xy,op.fixed,r);
    HXT1_CUDA(cudaGetLastError());
    return red.norm(r);
  };

  S22D4DResult q{};
  q.initial=residual();
  if(!std::isfinite(q.initial))
    throw std::runtime_error("HYBRID_M3 initial block residual nonfinite");

  // Exact assembled-CSR-versus-matrix-free operator parity, not just norm
  // comparison. This proves the M3 sweeps use the SAME frozen K as FGMRES.
  if(outer==1){
    const std::size_t bytes3=(std::size_t)3*nv*sizeof(Real);
    HXT1_CUDA(cudaMemcpy(r,ax,bytes3,cudaMemcpyDeviceToDevice));
    op.apply(x,ax);
    m3_sub<<<(3*nv+255)/256,256>>>(3*nv,r,ax,r);
    HXT1_CUDA(cudaGetLastError());
    const double discrepancy=red.norm(r);
    const double refNorm=red.norm(ax);
    const double rel=discrepancy/std::max(refNorm,1.e-300);
    std::printf("NODALS_HYBRID_M3_OPERATOR_PARITY it=1 "
      "difference=%.12e reference=%.12e relative=%.12e tol=1e-9 status=%s\n",
      discrepancy,refNorm,rel,
      (std::isfinite(rel)&&rel<=1.e-9)?"PASS":"FAIL");
    if(!std::isfinite(rel)||rel>1.e-9)
      throw std::runtime_error("Hybrid M3 CSR and matrix-free FGMRES operators disagree");
  }

  for(int pass=0;pass<sweeps;++pass){
    for(int cc=0;cc<op.C.ncolors;++cc){
      const int first=op.C.off[(std::size_t)cc];
      const int count=op.C.off[(std::size_t)cc+1]-first;
      if(count)
        m3_block_colour<<<(count+127)/128,128>>>(
          first,count,op.A.colorRows.p,op.A.row.p,op.A.col.p,
          op.A.diagPos.p,k,op.A.nnz,nv,b,x,
          op.xy,op.fixed,faceByBubble,op.faces,
          omega,bad);
    }
  }
  HXT1_CUDA(cudaGetLastError());
  HXT1_CUDA(cudaDeviceSynchronize());
  unsigned long long errors=0;
  HXT1_CUDA(cudaMemcpy(&errors,bad,sizeof(errors),cudaMemcpyDeviceToHost));
  if(errors)
    throw std::runtime_error("HYBRID_M3 3x3 block solve failed count="+std::to_string(errors));

  q.final=residual();
  q.rel=q.final/std::max(q.initial,1.e-300);
  q.its=sweeps;q.cycles=1;
  q.ok=std::isfinite(q.final) && q.final<=q.initial*(1.0+1.e-8);
  if(!q.ok){
    HXT1_CUDA(cudaMemcpy(x,backup,bytes,cudaMemcpyDeviceToDevice));
    std::printf("NODALS_HYBRID_M3_RESTORED it=%d "
      "reason=FROZEN_RESIDUAL_INCREASE status=WARN\n",outer);
  }
  std::printf("NODALS_HYBRID_M3_FGS it=%d sweeps=%d omega=%.6g "
    "initial=%.12e final=%.12e trueRel=%.12e accepted=%d "
    "operator=3NV_NINE_BLOCK_CSR_NO_TET_WALL_PROJECTION status=%s\n",
    outer,sweeps,omega,q.initial,q.final,q.rel,(int)q.ok,
    q.ok?"PASS":"WARN");
  return q;
}
