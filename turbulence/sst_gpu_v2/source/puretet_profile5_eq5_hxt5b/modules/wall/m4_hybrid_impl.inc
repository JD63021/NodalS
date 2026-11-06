// M4 pilot: algebraically identical M3 full CSR and frozen wall operator,
// but scalar FGS only where every off-component CSR row entry is exactly zero.
// The exact 3x3 block M3 solve remains for all coupled rows and wall bubbles.
// This v1 deliberately retains full 9-block CSR and its build; compact storage
// must be evaluated separately AFTER a measured FGS benefit.
__global__ void m4_classify_rows(int nv,std::size_t nnz,
    const std::int64_t*row,const Real*k,const int*faceByBubble,
    unsigned char*coupled){
  const int i=(int)blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=nv)return;
  bool c=(faceByBubble[i]>=0);
  for(auto p=row[i];p<row[i+1]&&!c;++p){
    // A zero test (not a threshold) guarantees no cross-component term is lost.
    if(k[1*nnz+p]!=Real(0)||k[2*nnz+p]!=Real(0)||
       k[3*nnz+p]!=Real(0)||k[5*nnz+p]!=Real(0)||
       k[6*nnz+p]!=Real(0)||k[7*nnz+p]!=Real(0))c=true;
  }
  coupled[i]=c?1:0;
}
__global__ void m4_hybrid_colour(int start,int count,const std::int32_t*rows,
    const std::int64_t*row,const std::int32_t*col,const std::int32_t*diag,
    const Real*k,std::size_t nnz,int nv,const Real*b,Real*x,
    const unsigned char*maskXY,const unsigned char*maskZ,
    const int*faceByBubble,const hxt5b::TetFacePlan*faces,
    const unsigned char*coupled,double omega,unsigned long long*bad){
  int q=(int)blockIdx.x*blockDim.x+threadIdx.x;if(q>=count)return;
  int i=rows[start+q];
  const bool fixed[3]={maskXY[i]!=0,maskXY[i]!=0,maskZ[i]!=0};
  if(fixed[0]&&fixed[1]&&fixed[2])return;
  if(!coupled[i]){
    // Strictly diagonal 3x3 row. M3's diagonal block solve simplifies exactly
    // to three scalar updates; do not read the six zero cross-component arrays.
    double rhs[3]={double(b[i]),double(b[(std::size_t)nv+i]),
                   double(b[2*(std::size_t)nv+i])};
    for(auto p=row[i];p<row[i+1];++p){
      const int j=col[p];
      rhs[0]-=(double)k[0*nnz+p]*double(x[j]);
      rhs[1]-=(double)k[4*nnz+p]*double(x[(std::size_t)nv+j]);
      rhs[2]-=(double)k[8*nnz+p]*double(x[2*(std::size_t)nv+j]);
    }
    const int d=diag[i];
    for(int c=0;c<3;++c){
      if(fixed[c])continue;
      const double a=(double)k[((std::size_t)c*3+c)*nnz+d];
      if(!(fabs(a)>1.e-27)||!isfinite(a)){
        atomicAdd(bad,1ULL);return;
      }
      x[(std::size_t)c*nv+i]+=(Real)(omega*rhs[c]/a);
    }
    return;
  }
  // Exact M3 local block procedure for every wall-influenced row.
  double rhs[3]={double(b[i]),double(b[(std::size_t)nv+i]),double(b[2*(std::size_t)nv+i])};
  for(auto p=row[i];p<row[i+1];++p){
    const int j=col[p];
    const double xx[3]={double(x[j]),double(x[(std::size_t)nv+j]),double(x[2*(std::size_t)nv+j])};
    for(int a=0;a<3;++a)for(int c=0;c<3;++c)
      rhs[a]-=(double)k[((std::size_t)a*3+c)*nnz+p]*xx[c];
  }
  double a[3][3];const int di=diag[i];
  for(int r=0;r<3;++r)for(int c=0;c<3;++c)a[r][c]=(double)k[((std::size_t)r*3+c)*nnz+di];
  const int fi=faceByBubble[i];
  double n[3]={0,0,0};
  if(fi>=0){
    const auto f=faces[fi];for(int c=0;c<3;++c)n[c]=f.n[c];
    const double scale=fmax(1.e-20,fmax(fabs(a[0][0]),fmax(fabs(a[1][1]),fabs(a[2][2]))));
    for(int r=0;r<3;++r)for(int c=0;c<3;++c)a[r][c]+=scale*n[r]*n[c];
    const double rn=n[0]*rhs[0]+n[1]*rhs[1]+n[2]*rhs[2];
    for(int c=0;c<3;++c)rhs[c]-=rn*n[c];
  }
  for(int c=0;c<3;++c)if(fixed[c]){
    for(int j=0;j<3;++j){a[c][j]=0;a[j][c]=0;}a[c][c]=1;rhs[c]=0;
  }
  double sol[3]={0,0,0};
  if(!m3_solve3(a,rhs,sol)){atomicAdd(bad,1ULL);return;}
  if(fi>=0){const double sn=n[0]*sol[0]+n[1]*sol[1]+n[2]*sol[2];
    for(int c=0;c<3;++c)sol[c]-=sn*n[c];}
  for(int c=0;c<3;++c)if(!fixed[c])x[(std::size_t)c*nv+i]+=(Real)(omega*sol[c]);
}
inline S22D4DResult m4_fixed_hybrid_csr(const S22D4DOperator&op,
    Real*x,const Real*b,Real*k,Real*tmp,int sweeps,double omega,
    const int*faceByBubble,unsigned long long*bad,int outer,
    unsigned char*coupled,Real*parityRef,Real*parityDiff,
    hxt4a::Reducer<Real>&red){
  if(!op.bf3FluxMode||!op.f1bFaceCoupling)
    throw std::runtime_error("M4 requires exact BF3 and F1B wall cache");
  const int n=op.A.n;
  S22D4DResult R{};
  s22d4d_zero_fixed<<<(3*n+255)/256,256>>>(n,op.xy,op.fixed,x);
  s22d4i_Q_apply<<<(op.nfaces+255)/256,256>>>(op.nfaces,op.faces,op.tvel,
    x,x+n,x+2*(std::size_t)n);
  const auto st=std::chrono::steady_clock::now();
  m3_build(op,k,tmp,bad);
  m4_classify_rows<<<(n+255)/256,256>>>(n,op.A.nnz,op.A.row.p,k,
    faceByBubble,coupled);
  HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
  const double buildMs=std::chrono::duration<double,std::milli>(
    std::chrono::steady_clock::now()-st).count();
  static bool auditFirst=true;
  const bool doAudit=auditFirst;
  if(doAudit){
    // First invocation of M4 only: no approximate matrix-free substitution.
    std::vector<unsigned char>h((std::size_t)n);
    HXT1_CUDA(cudaMemcpy(h.data(),coupled,h.size(),cudaMemcpyDeviceToHost));
    const std::size_t wc=(std::size_t)std::count(h.begin(),h.end(),(unsigned char)1);
    std::printf("NODALS_M4_COUPLED_ROWS it=%d coupled=%zu interior=%zu fraction=%.6f\n",
      outer,wc,(std::size_t)n-wc,double(wc)/double(n));
    HXT1_CUDA(cudaMemcpy(parityRef,x,3ull*(std::size_t)n*sizeof(Real),cudaMemcpyDeviceToDevice));
  }
  HXT1_CUDA(cudaMemset(bad,0,sizeof(unsigned long long)));
  const auto sw=std::chrono::steady_clock::now();
  auto sweep=[&](Real*xp,bool hybrid){
    for(int c=0;c<op.C.ncolors;++c){
      const int begin=op.C.off[(std::size_t)c];
      const int count=op.C.off[(std::size_t)c+1]-begin;
      if(!count)continue;
      if(hybrid)m4_hybrid_colour<<<(count+127)/128,128>>>(begin,count,
        op.A.colorRows.p,op.A.row.p,op.A.col.p,op.A.diagPos.p,
        k,op.A.nnz,n,b,xp,op.xy,op.fixed,faceByBubble,op.faces,
        coupled,omega,bad);
      else m3_block_colour<<<(count+127)/128,128>>>(begin,count,
        op.A.colorRows.p,op.A.row.p,op.A.col.p,op.A.diagPos.p,
        k,op.A.nnz,n,b,xp,op.xy,op.fixed,faceByBubble,op.faces,omega,bad);
    }
  };
  for(int s=0;s<sweeps;++s)sweep(x,true);
  HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
  const double sweepMs=std::chrono::duration<double,std::milli>(
    std::chrono::steady_clock::now()-sw).count();
  if(doAudit){
    for(int s=0;s<sweeps;++s)sweep(parityRef,false);
    HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
    m3_sub<<<(3*n+255)/256,256>>>(3*n,x,parityRef,parityDiff);
    const double nr=red.norm(parityRef);
    const double err=red.norm(parityDiff);
    const double rel=err/std::max(nr,1.e-300);
    const bool ok=std::isfinite(rel)&&rel<=1.e-9;
    std::printf("NODALS_M4_SWEEP_PARITY it=%d rel=%.12e tol=1e-9 status=%s\n",
      outer,rel,ok?"PASS":"FAIL");
    if(!ok)throw std::runtime_error("M4 vs M3 coupled FGS same-operator parity FAILED");
    auditFirst=false;
  }
  unsigned long long failures=0;
  HXT1_CUDA(cudaMemcpy(&failures,bad,sizeof(failures),cudaMemcpyDeviceToHost));
  if(failures)throw std::runtime_error("M4 local solve failure count="+std::to_string(failures));
  R.its=sweeps;R.cycles=1;R.ok=true;
  R.initial=R.final=R.rel=std::numeric_limits<double>::quiet_NaN();
  // Keep old machine-readable M3 timing key for the existing profiler.
  std::printf("NODALS_M3_FIXED it=%d method=m4 sweeps=%d omega=%.4f "
    "csrNnz=%zu ncolors=%d buildMs=%.3f sweepMs=%.3f "
    "variant=HYBRID_FULL_CSR_v1 scalarWallAudit=PASS krylov=OFF "
    "innerResiduals=NONE bf3=EXACT_TtMT status=PASS\n",
    outer,sweeps,omega,op.A.nnz,op.C.ncolors,buildMs,sweepMs);
  return R;
}
