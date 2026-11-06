#pragma once
// Included in namespace hxt5d; read-only except transient overwrite/restoration of A.val.
#include "puretet_exp3_gamma_core.hpp"
#include <cstdlib>
#include <sstream>
#include <iomanip>
namespace puretet_gamma_exp3 {
inline std::vector<double> factors(){
  const char*e=std::getenv("PURETET_EXP3_GAMMA_FACTORS");
  const std::string s=e&&*e?e:"0.5,1,2,5,10,20,50,100";
  std::vector<double> out;std::stringstream ss(s);std::string t;
  while(std::getline(ss,t,',')){double x=std::stod(t);if(!(x>0)||!std::isfinite(x))throw std::runtime_error("EXP3 factors must be positive finite");out.push_back(x);}
  if(out.empty()||out.size()>16)throw std::runtime_error("EXP3 factors length must be 1..16");
  return out;
}
inline void run(int it,const nodals_hxt1::HostMesh&M,hxt4b::GpuCSR&A,const hxt4b::ColoringHost&C,
 const Dev<unsigned char>&fixed,const State&S,const Dev<Real>&vol,const Dev<Real>&dg,const Dev<Real>&wall,
 hxt4a::Reducer<Real>&red,const hxt5c::Options&O,double radius,const std::vector<Real>&baselineRaw)
{
 const int n=A.n,nv=S.nvert;
 if(hxt5c::puretet_source_exp1::lastOuter!=it||
    hxt5c::puretet_source_exp1::lastWallRhs.size()!=size_t(n))
    throw std::runtime_error("EXP3 wall RHS snapshot unavailable; EXP1 source audit not active");
 puretet_exp3_core::System X;X.n=n;X.nv=nv;X.alpha=O.alphaOmega;
 auto row=puretet_omega_diag::download(A.row,size_t(n+1));auto col=puretet_omega_diag::download(A.col,A.nnz);
 X.row.assign(row.begin(),row.end());X.col.assign(col.begin(),col.end());
 const auto val=puretet_omega_diag::download(A.val,A.nnz);
 const auto vwall=puretet_omega_diag::download(wall,A.nnz);
 const auto rhsrelaxed=puretet_omega_diag::download(S.rhs,n);
 const auto old=puretet_omega_diag::download(S.oldW,n);
 const auto sstats=puretet_omega_diag::download(S.wallStats,13);
 X.phys.resize(size_t(A.nnz));X.wall.resize(size_t(A.nnz));X.rhs.resize(n);X.wallrhs.resize(n);X.old.resize(n);
 for(int a=0;a<A.nnz;++a)X.wall[a]=double(vwall[a]);
 for(int i=0;i<n;++i){X.old[i]=double(old[i]);X.wallrhs[i]=double(hxt5c::puretet_source_exp1::lastWallRhs[i]);}
 // Reassemble the exact frozen *physical* P1 matrix from its original blocks.
 const auto bas=puretet_omega_diag::download(A.base,A.nnz);
 const auto conv=puretet_omega_diag::download(A.conv,A.nnz);
 const auto volh=puretet_omega_diag::download(vol,A.nnz);
 const auto dgh=puretet_omega_diag::download(dg,A.nnz);
 for(int a=0;a<A.nnz;++a)X.phys[a]=double(bas[a])+double(conv[a])+double(volh[a])+double(dgh[a])+X.wall[a];
 for(int i=0;i<n;++i){double diagdiff=0;for(int a=X.row[i];a<X.row[i+1];++a)if(X.col[a]==i){diagdiff=double(val[a])-X.phys[a];break;} X.rhs[i]=double(rhsrelaxed[i])-diagdiff*X.old[i];}
 X.targetNorm=double(sstats[6]); X.validate();
 if(!(X.targetNorm>0))throw std::runtime_error("EXP3 targetNorm missing: rebuild with EXP3 wallStats contribution");
 // Validate exact baseline reconstruction BEFORE the counterfactuals.
 auto norm= X.scaled(1.0);double aDiff=0,bDiff=0;
 for(int a=0;a<A.nnz;++a)aDiff=std::max(aDiff,std::abs(norm.matrix[a]-double(val[a])));
 for(int i=0;i<nv;++i)bDiff=std::max(bDiff,std::abs(norm.right[i]-double(rhsrelaxed[i])));
 std::printf("PURETET_EXP3_RECON it=%d gamma0=%.9e alphaOmega=%.6f maxMatrixAbsError=%.9e maxRhsAbsError=%.9e targetNorm=%.9e status=%s\n",
   it,O.profile5EqGamma,X.alpha,aDiff,bDiff,X.targetNorm,(aDiff<1e-6&&bDiff<1e-6)?"PASS":"WARNING");
 if(aDiff>1e-6||bDiff>1e-6)throw std::runtime_error("EXP3 assembly reconstruction mismatch");
 std::vector<double> raw(n);const auto post=puretet_omega_diag::download(S.omega,n);
 for(int i=0;i<n;++i)raw[i]=double(post[i]);
 if(baselineRaw.size()!=size_t(nv))throw std::runtime_error("EXP3 raw baseline size mismatch");
 for(int i=0;i<nv;++i)raw[i]=double(baselineRaw[i]);
 // FGS preconditioner in the scalar solve is rebuilt from the modified A by its
 // standard path; the momentum/pressure rAU is not part of this omega solve.
 std::vector<Real> restore(val.begin(),val.end());
 int done=0;const auto testFactors=factors();
 for(double fac : testFactors){
   auto scaled=X.scaled(fac);
   if(scaled.invalidDiag){
     std::printf("PURETET_EXP3_GAMMA it=%d factor=%.6g gamma=%.9e invalidPhysicalDiag=%d status=SKIP\n",it,fac,O.profile5EqGamma*fac,scaled.invalidDiag);continue;
   }
   std::vector<Real> aNew(size_t(A.nnz)),bNew(n),start(n);
   for(int a=0;a<A.nnz;++a)aNew[a]=Real(scaled.matrix[a]);
   for(int i=0;i<n;++i){bNew[i]=Real(scaled.right[i]);start[i]=Real(X.old[i]);}
   Dev<Real> dRhs,dX;dRhs.upload(bNew);dX.upload(start);
   hxt5c::LinearResult lr;bool threw=false;
   try {
     HXT1_CUDA(cudaMemcpy(A.val.p,aNew.data(),size_t(A.nnz)*sizeof(Real),cudaMemcpyHostToDevice));
     lr=hxt5c::scalar_solve(A,C,fixed,dRhs.p,dX.p,red,O,it,"omega");
   }catch(const std::exception&e){
     threw=true;
     std::printf("PURETET_EXP3_EXCEPTION it=%d factor=%.6g what=%s status=WARNING\n",it,fac,e.what());
   }
   HXT1_CUDA(cudaMemcpy(A.val.p,restore.data(),size_t(A.nnz)*sizeof(Real),cudaMemcpyHostToDevice));
   if(threw)continue;
   const auto alt=puretet_omega_diag::download(dX,n);
   std::vector<double> x(n);for(int i=0;i<n;++i)x[i]=alt[i];
   double mi=std::numeric_limits<double>::infinity(),num=0,den=0,rhsnorm=0,oldres;
   int neg=0,below=0,nonfinite=0;int radialNeg[5]={0,0,0,0,0};
   for(int i=0;i<nv;++i){const double v=x[i];if(!std::isfinite(v)){nonfinite++;continue;}
     mi=std::min(mi,v);neg+=(v<0);below+=(v<double(O.omegaFloor));
     if(v<0){const auto&P=M.points[size_t(i)];radialNeg[puretet_omega_diag::bin(std::hypot(P.x,P.y),radius)]++;}
     num+=(v-raw[i])*(v-raw[i]);den+=raw[i]*raw[i];rhsnorm+=scaled.right[i]*scaled.right[i];
   }
   const double res=X.residual(scaled.matrix,scaled.right,x);
   const double weak=X.wall_rel_mismatch(x);
   const double rawWeak=X.wall_rel_mismatch(raw);
   oldres=X.residual(norm.matrix,norm.right,x);
   std::printf("PURETET_EXP3_GAMMA it=%d factor=%.6g gamma=%.9e negative=%d belowFloor=%d minRaw=%.9e rawNonfinite=%d relativeChange=%.9e relResidual=%.9e originalResidual=%.9e wallTargetRelRMS=%.9e baselineWallTargetRelRMS=%.9e diagRatio=%.9e maxRowL1OverDiag=%.9e relaxShiftOverDiag=%.9e linearIts=%d linearRel=%.9e linearOK=%d status=%s\n",
     it,fac,O.profile5EqGamma*fac,neg,below,mi,nonfinite,std::sqrt(num/std::max(den,1e-300)),res/std::max(std::sqrt(rhsnorm),1e-300),oldres,weak,rawWeak,scaled.diagRatio,scaled.maxRowL1OverDiag,scaled.maxRelaxShiftOverDiag,lr.works,lr.rel,int(lr.ok),(!nonfinite&&lr.ok)?"PASS":"WARNING");
   std::printf("PURETET_EXP3_RADIAL it=%d factor=%.6g negativeByBin=[%d,%d,%d,%d,%d] status=PASS\n",it,fac,radialNeg[0],radialNeg[1],radialNeg[2],radialNeg[3],radialNeg[4]);done++;
 }
 // Production A is restored after *every* alternate solve.
 std::printf("PURETET_EXP3_DONE it=%d factorsTested=%d productionMatrixRestored=1 productionStateUnchanged=1 status=PASS\n",it,done);
}
} // namespace puretet_gamma_exp3
