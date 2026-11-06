#pragma once
// Included inside namespace hxt5d, after State. Frozen counterfactuals ONLY.
namespace puretet_frozen_exp2 {
using puretet_exp2_core::Snapshot;
inline void run(int it,const nodals_hxt1::HostMesh&M,hxt4b::GpuCSR&A,
                const hxt4b::ColoringHost&C,const Dev<unsigned char>&scalarFixed,
                const State&S,const Dev<Real>&vol,const Dev<Real>&dg,const Dev<Real>&wall,
                const hxt5c::LinearResult&baselineLinear,hxt4a::Reducer<Real>&red,const hxt5c::Options&O,double radius,
                const std::vector<Real>&baselineRaw)
{
  const int nv=S.nvert,n=A.n;
  if(hxt5c::puretet_source_exp1::lastOuter!=it ||
     hxt5c::puretet_source_exp1::lastCross.size()!=size_t(n) ||
     hxt5c::puretet_source_exp1::lastReaction.size()!=size_t(A.nnz))
    throw std::runtime_error("EXP2 source/reaction snapshot unavailable; check EXP1 installation");
  Snapshot snap;snap.n=n;snap.nv=nv;
  auto row=puretet_omega_diag::download(A.row,size_t(n+1));
  auto col=puretet_omega_diag::download(A.col,A.nnz);
  snap.row.assign(row.begin(),row.end());snap.col.assign(col.begin(),col.end());
  const auto a=puretet_omega_diag::download(A.val,A.nnz);
  const auto base=puretet_omega_diag::download(A.base,A.nnz);
  const auto conv=puretet_omega_diag::download(A.conv,A.nnz);
  const auto vv=puretet_omega_diag::download(vol,A.nnz);
  const auto inlet=puretet_omega_diag::download(dg,A.nnz);
  const auto wa=puretet_omega_diag::download(wall,A.nnz);
  const auto rhs=puretet_omega_diag::download(S.rhs,n);
  const auto old=puretet_omega_diag::download(S.oldW,n);
  auto put=[&](std::vector<double>&out,const auto&src){out.assign(src.begin(),src.end());};
  put(snap.relaxed,a);put(snap.base,base);put(snap.conv,conv);
  put(snap.volume,vv);put(snap.inlet,inlet);put(snap.wall,wa);
  put(snap.rhs,rhs);put(snap.old,old);
  put(snap.reaction,hxt5c::puretet_source_exp1::lastReaction);
  put(snap.cross,hxt5c::puretet_source_exp1::lastCross);
  snap.validate();
  // Matrix attribution: diffusion=total volume - linearized reaction.
  // Signed coefficient sums (NOT norms of a physical flux across annular boundaries).
  struct Region {int n=0;double positive[6]={},negative[6]={},offabs[6]={};int poscnt[6]={},negcnt[6]={};};
  Region q[6];
  std::vector<int> bins(nv);
  for(int i=0;i<nv;++i){const auto&P=M.points[size_t(i)];bins[i]=puretet_omega_diag::bin(std::hypot(P.x,P.y),radius);for(int b : {bins[i],5})q[b].n++;}
  for(int i=0;i<nv;++i){
    for(int p=snap.row[i];p<snap.row[i+1];++p){const int j=snap.col[p];if(j==i||j>=nv)continue;
      const double terms[6]={snap.base[p],snap.conv[p],snap.volume[p]-snap.reaction[p],snap.reaction[p],snap.inlet[p],snap.wall[p]};
      for(int b : {bins[i],5})for(int k=0;k<6;++k){auto &r=q[b];const double z=terms[k];if(z>0){r.poscnt[k]++;r.positive[k]+=z;}else if(z<0){r.negcnt[k]++;r.negative[k]+=z;}r.offabs[k]+=std::abs(z);}
    }
  }
  const char*component[6]={"base","convection","diffusion","reaction","inlet","eq5wall"};
  for(int b=0;b<6;++b)for(int k=0;k<6;++k){auto&r=q[b];std::printf("PURETET_EXP2_COMPONENT it=%d bin=%d component=%s rows=%d posOff=%d negOff=%d posOffSum=%.9e negOffSum=%.9e offL1=%.9e status=PASS\n",
    it,b,component[k],r.n,r.poscnt[k],r.negcnt[k],r.positive[k],r.negative[k],r.offabs[k]);}
  if(baselineRaw.size()!=size_t(nv))throw std::runtime_error("EXP2 baselineRaw size mismatch");
  const auto baselineFull=puretet_omega_diag::download(S.omega,n);
  auto describe=[&](const char*name,const std::vector<Real>&solution,const std::vector<double>&matrix,
                    const std::vector<double>&right,const hxt5c::LinearResult*lin){
    double min=std::numeric_limits<double>::infinity(),diff2=0,orig2=0;
    int neg=0,below=0,nonfinite=0;int regNeg[5]={0,0,0,0,0};int regBelow[5]={0,0,0,0,0};
    std::vector<double> hd(solution.begin(),solution.end());
    for(int i=0;i<nv;++i){const double v=hd[i],p=double(baselineRaw[size_t(i)]);
      if(!std::isfinite(v)){nonfinite++;continue;}
      min=std::min(min,v);if(v<0){neg++;regNeg[bins[i]]++;}if(v<double(O.omegaFloor)){below++;regBelow[bins[i]]++;}
      diff2+=(v-p)*(v-p);orig2+=p*p;
    }
    const double altRes=snap.residual_norm(matrix,right,hd);
    const double originalRes=snap.residual_norm(snap.relaxed,snap.rhs,hd);
    std::printf("PURETET_EXP2_SOLVE it=%d case=%s nVertex=%d minRaw=%.9e rawNegative=%d rawBelow=%d rawNonfinite=%d differenceRelL2=%.9e alternativeResidual=%.9e originalResidual=%.9e linearIts=%d linearRel=%.9e linearOK=%d status=%s\n",
      it,name,nv,min,neg,below,nonfinite,std::sqrt(diff2)/std::max(std::sqrt(orig2),1e-300),altRes,originalRes,lin?lin->works:0,lin?lin->rel:0.0,lin?int(lin->ok):1,(nonfinite==0 && (!lin||lin->ok))?"PASS":"WARNING");
    for(int b=0;b<5;++b)std::printf("PURETET_EXP2_RADIAL it=%d case=%s bin=%d vertices=%d rawNegative=%d rawBelow=%d status=PASS\n",it,name,b,q[b].n,regNeg[b],regBelow[b]);
  };
  // Baseline is exactly the previously computed raw omega; DO NOT recompute it.
  describe("BASELINE",baselineFull,snap.relaxed,snap.rhs,&baselineLinear);
  auto solveCase=[&](const char*label,const std::vector<double>&rhsModified,
                     const std::vector<double>&matrixModified,bool changeMatrix){
    std::vector<Real> rhsReal(n),xReal(n),matrixReal;
    for(int i=0;i<n;++i){rhsReal[i]=Real(rhsModified[size_t(i)]);xReal[i]=old[size_t(i)];}
    Dev<Real> rhsD,xD;rhsD.upload(rhsReal);xD.upload(xReal);
    if(changeMatrix){matrixReal.resize(A.nnz);for(size_t j=0;j<size_t(A.nnz);++j)matrixReal[j]=Real(matrixModified[j]);
      HXT1_CUDA(cudaMemcpy(A.val.p,matrixReal.data(),A.nnz*sizeof(Real),cudaMemcpyHostToDevice));}
    hxt5c::LinearResult lr;
    try{
      lr=hxt5c::scalar_solve(A,C,scalarFixed,rhsD.p,xD.p,red,O,it,"omega");
    }catch(...){
      if(changeMatrix)HXT1_CUDA(cudaMemcpy(A.val.p,a.data(),A.nnz*sizeof(Real),cudaMemcpyHostToDevice));
      throw;
    }
    if(changeMatrix)HXT1_CUDA(cudaMemcpy(A.val.p,a.data(),A.nnz*sizeof(Real),cudaMemcpyHostToDevice));
    const auto alt=puretet_omega_diag::download(xD,n);
    describe(label,alt,matrixModified,rhsModified,&lr);
  };
  solveCase("NONNEG_PHYS_RHS",snap.rhs_nonnegative_physical(),snap.relaxed,false);
  solveCase("NO_CROSS_RHS",snap.rhs_without_cross(),snap.relaxed,false);
  solveCase("REACTION_LUMP",snap.rhs,snap.lump_offdiag(Snapshot::REACTION),true);
  solveCase("EQ5_WALL_LUMP",snap.rhs,snap.lump_offdiag(Snapshot::WALL),true);
  std::printf("PURETET_EXP2_DONE it=%d frozen=1 productionStateUnchanged=1 productionMatrixRestored=1 status=PASS\n",it);
}
} // namespace puretet_frozen_exp2
