// NODALS_BALANCED_POD_AMG_V1
// Online normalized snapshot POD + balanced AMG augmentation for repeated
// pressure-correction solves. Included from hybrid_amg.cuh inside namespace hxt4c.

#ifndef NODALS_BALANCED_POD_AMG_V1
#define NODALS_BALANCED_POD_AMG_V1

template<class T>
__global__ void pod_build_q_t_kernel(
    int n,int rank,const T*W,const T*AW,const T*a,const T*r,T*q,T*t)
{
  const int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i>=n)return;
  T qi=T(0),aqi=T(0);
  for(int k=0;k<rank;++k){
    const T c=a[k];
    qi +=W[(std::size_t)k*(std::size_t)n+(std::size_t)i]*c;
    aqi+=AW[(std::size_t)k*(std::size_t)n+(std::size_t)i]*c;
  }
  q[i]=qi;
  t[i]=r[i]-aqi;
}

template<class T>
__global__ void pod_finish_kernel(
    int n,int rank,const T*W,const T*b,const T*q,const T*g,T*z)
{
  const int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
  if(i>=n)return;
  T wb=T(0);
  for(int k=0;k<rank;++k)
    wb+=W[(std::size_t)k*(std::size_t)n+(std::size_t)i]*b[k];
  z[i]=q[i]+g[i]-wb;
}

inline void pod_jacobi_eigen(
    std::vector<double>A,int n,
    std::vector<double>&eval,
    std::vector<double>&V)
{
  V.assign((std::size_t)n*(std::size_t)n,0.0);
  for(int i=0;i<n;++i)V[(std::size_t)i*n+i]=1.0;
  if(n==1){eval.assign(1,A[0]);return;}

  for(int sweep=0;sweep<80*n*n;++sweep){
    int p=0,q=1;
    double mx=0.0,diagMax=0.0;
    for(int i=0;i<n;++i){
      diagMax=std::max(diagMax,std::abs(A[(std::size_t)i*n+i]));
      for(int j=i+1;j<n;++j){
        const double v=std::abs(A[(std::size_t)i*n+j]);
        if(v>mx){mx=v;p=i;q=j;}
      }
    }
    if(mx<=1e-14*std::max(diagMax,1.0))break;

    const double app=A[(std::size_t)p*n+p];
    const double aqq=A[(std::size_t)q*n+q];
    const double apq=A[(std::size_t)p*n+q];
    if(apq==0.0)continue;
    const double tau=(aqq-app)/(2.0*apq);
    const double t=(tau>=0.0?1.0:-1.0)/
      (std::abs(tau)+std::sqrt(1.0+tau*tau));
    const double c=1.0/std::sqrt(1.0+t*t);
    const double s=t*c;

    for(int k=0;k<n;++k){
      if(k==p||k==q)continue;
      const double akp=A[(std::size_t)k*n+p];
      const double akq=A[(std::size_t)k*n+q];
      const double nkp=c*akp-s*akq;
      const double nkq=s*akp+c*akq;
      A[(std::size_t)k*n+p]=A[(std::size_t)p*n+k]=nkp;
      A[(std::size_t)k*n+q]=A[(std::size_t)q*n+k]=nkq;
    }
    A[(std::size_t)p*n+p]=app-t*apq;
    A[(std::size_t)q*n+q]=aqq+t*apq;
    A[(std::size_t)p*n+q]=A[(std::size_t)q*n+p]=0.0;

    for(int k=0;k<n;++k){
      const double vkp=V[(std::size_t)k*n+p];
      const double vkq=V[(std::size_t)k*n+q];
      V[(std::size_t)k*n+p]=c*vkp-s*vkq;
      V[(std::size_t)k*n+q]=s*vkp+c*vkq;
    }
  }
  eval.resize((std::size_t)n);
  for(int i=0;i<n;++i)eval[(std::size_t)i]=A[(std::size_t)i*n+i];
}

inline std::vector<double> pod_spd_inverse(const std::vector<double>&A,int n)
{
  std::vector<double>L((std::size_t)n*(std::size_t)n,0.0);
  double maxDiag=0.0;
  for(int i=0;i<n;++i)maxDiag=std::max(maxDiag,std::abs(A[(std::size_t)i*n+i]));
  const double floor=1e-13*std::max(maxDiag,1e-300);
  for(int i=0;i<n;++i){
    for(int j=0;j<=i;++j){
      long double s=(long double)A[(std::size_t)i*n+j];
      for(int k=0;k<j;++k)
        s-=(long double)L[(std::size_t)i*n+k]*L[(std::size_t)j*n+k];
      if(i==j){
        if(!((double)s>floor)||!std::isfinite((double)s))
          throw std::runtime_error("Balanced POD reduced operator is not SPD");
        L[(std::size_t)i*n+j]=std::sqrt((double)s);
      }else{
        L[(std::size_t)i*n+j]=(double)s/L[(std::size_t)j*n+j];
      }
    }
  }

  std::vector<double>inv((std::size_t)n*(std::size_t)n,0.0),y((std::size_t)n),x((std::size_t)n);
  for(int col=0;col<n;++col){
    for(int i=0;i<n;++i){
      long double s=(i==col)?1.0L:0.0L;
      for(int k=0;k<i;++k)s-=(long double)L[(std::size_t)i*n+k]*y[(std::size_t)k];
      y[(std::size_t)i]=(double)s/L[(std::size_t)i*n+i];
    }
    for(int i=n-1;i>=0;--i){
      long double s=y[(std::size_t)i];
      for(int k=i+1;k<n;++k)s-=(long double)L[(std::size_t)k*n+i]*x[(std::size_t)k];
      x[(std::size_t)i]=(double)s/L[(std::size_t)i*n+i];
    }
    for(int i=0;i<n;++i)inv[(std::size_t)i*n+col]=x[(std::size_t)i];
  }
  return inv;
}

template<class T>
class BalancedPOD {
  int n_=0,pin_=-1,trainTarget_=10,targetRank_=2,maxRank_=2;
  int tailIter_=10,hardIts_=0;
  int seedCollected_=0,batchCollected_=0,rank_=0;
  int acceptedUpdates_=0,rejectedUpdates_=0,probeCount_=0;
  int currentOuter_=0;
  bool enabled_=false,adaptive_=false,ready_=false,prepared_=false;
  bool batchReady_=false,tentative_=false,tailIterateSaved_=false;
  bool multiTailSaved1_=false,multiTailSaved2_=false,multiTailSaved3_=false;
  bool tentativeIsRefresh_=false,refreshStale_=false;
  int tailSavedIter_=0,tentativeReplaceBlock_=-1;
  double noveltyTol_=1e-8;
  int acceptMinAbs_=5;
  double acceptMinRel_=0.03;

  // Fixed-rank moving-basis refresh.  Once maxRank is reached, pressure
  // solves are grouped in trainTarget-sized windows.  A sustained increase
  // relative to the best window in the current basis generation marks the
  // space stale.  The next tail-correction POD block tentatively replaces
  // one old targetRank block; same-system PCG decides whether to keep it.
  int refreshAbs_=30;
  double refreshRel_=0.35;

  // Periodic whole-space recompression.  Grow to maxRank, then form the
  // best current-A subspace of the accumulated W for recent normalized
  // x_final-x_tail corrections and compress back to recompressRank.
  int recompressRank_=0,recompressHistory_=18,recompressMaxLossAbs_=5;
  double recompressMaxLossRel_=0.10;
  std::string recompressMethod_="tail_pod";
  bool recompressForce_=false,noProbe_=false;
  bool recompressPending_=false,tentativeIsRecompress_=false;
  int recompressAccepted_=0,recompressRejected_=0,recompressCount_=0;
  std::vector<std::vector<double>>tailHistory_;

  int refreshWindows_=0,refreshCursor_=0,refreshGeneration_=0;
  int refreshAccepted_=0,refreshRejected_=0;
  double refreshBestMean_=std::numeric_limits<double>::infinity();
  std::vector<int>refreshWindowIts_;

  std::vector<double>snap_;     // normalized snapshot batch, column-major
  std::vector<double>Wh_;       // retained basis, column-major
  std::vector<double>Ginv_;
  std::vector<double>rollbackWh_;
  int rollbackRank_=0,tentativeAdded_=0;

  Buf<T>W_,AW_,q_,t_,g_,coeff_,tmpv_,tmpAv_,probeX_,tailX_,tailX2_,tailX3_;
  long long stockIts_=0,podIts_=0,probeIts_=0;
  int stockSolves_=0,podSolves_=0;
  double capturedNormMin_=std::numeric_limits<double>::infinity();
  double capturedNormMax_=0.0;
  double retainedEnergy_=0.0,lastBatchEnergy_=0.0;
  double aProjectMinRatio_=std::numeric_limits<double>::infinity();
  double aProjectMaxRatio_=0.0;

  static std::vector<double> matvec(
      const std::vector<double>&A,const std::vector<double>&x,int n)
  {
    std::vector<double>y((std::size_t)n,0.0);
    for(int i=0;i<n;++i){
      long double s=0.0L;
      for(int j=0;j<n;++j)s+=(long double)A[(std::size_t)i*n+j]*x[(std::size_t)j];
      y[(std::size_t)i]=(double)s;
    }
    return y;
  }

  static double norm2(const std::vector<double>&x){
    long double s=0.0L;for(double v:x)s+=(long double)v*v;
    return std::sqrt((double)s);
  }

  static long double host_dot(const double*a,const std::vector<T>&b,int n){
    long double s=0.0L;
    for(int i=0;i<n;++i)s+=(long double)a[i]*(long double)b[(std::size_t)i];
    return s;
  }

  void clear_batch(){
    snap_.clear();batchCollected_=0;batchReady_=false;
    snap_.reserve((std::size_t)n_*(std::size_t)trainTarget_);
  }

  void reset_refresh_monitor(){
    refreshWindowIts_.clear();
    refreshBestMean_=std::numeric_limits<double>::infinity();
    refreshWindows_=0;
    refreshStale_=false;
  }

  void upload_basis(){
    if(rank_<=0)throw std::runtime_error("Balanced POD upload with zero rank");
    W_.upload(castv<T>(Wh_));
    AW_.alloc((std::size_t)rank_*(std::size_t)n_);
    if(!q_.p){q_.alloc((std::size_t)n_);t_.alloc((std::size_t)n_);g_.alloc((std::size_t)n_);}
    if(!tmpv_.p){tmpv_.alloc((std::size_t)n_);tmpAv_.alloc((std::size_t)n_);probeX_.alloc((std::size_t)n_);}
    coeff_.alloc((std::size_t)rank_);
    prepared_=false;
  }

  std::vector<std::vector<double>> pod_modes_from_batch(int m,int wanted,double&energy)const{
    std::vector<std::vector<double>>out;
    energy=0.0;
    if(m<=0||wanted<=0)return out;
    std::vector<double>C((std::size_t)m*(std::size_t)m,0.0);
    for(int a=0;a<m;++a){
      for(int b=a;b<m;++b){
        long double ss=0.0L;
        const double*xa=snap_.data()+(std::size_t)a*n_;
        const double*xb=snap_.data()+(std::size_t)b*n_;
        for(int i=0;i<n_;++i)ss+=(long double)xa[i]*xb[i];
        C[(std::size_t)a*m+b]=C[(std::size_t)b*m+a]=(double)ss;
      }
    }
    std::vector<double>eval,V;
    pod_jacobi_eigen(C,m,eval,V);
    std::vector<int>idx((std::size_t)m);
    for(int i=0;i<m;++i)idx[(std::size_t)i]=i;
    std::sort(idx.begin(),idx.end(),[&](int a,int b){return eval[(std::size_t)a]>eval[(std::size_t)b];});
    const double lam0=std::max(eval[(std::size_t)idx[0]],0.0);
    double total=0.0;for(double v:eval)if(v>0.0)total+=v;
    double kept=0.0;
    const int take=std::min(wanted,m);
    for(int kk=0;kk<take;++kk){
      const int ec=idx[(std::size_t)kk];
      const double lam=eval[(std::size_t)ec];
      if(!(lam>1e-12*std::max(lam0,1e-300)))continue;
      std::vector<double>w((std::size_t)n_,0.0);
      const double invSig=1.0/std::sqrt(lam);
      for(int s=0;s<m;++s){
        const double a=V[(std::size_t)s*m+ec]*invSig;
        const double*e=snap_.data()+(std::size_t)s*n_;
        for(int i=0;i<n_;++i)w[(std::size_t)i]+=a*e[i];
      }
      const double wn=norm2(w);
      if(!(wn>1e-14)||!std::isfinite(wn))continue;
      for(double&v:w)v/=wn;
      out.push_back(std::move(w));
      kept+=lam;
    }
    energy=(total>0.0)?kept/total:0.0;
    return out;
  }

  void build_seed(){
    double energy=0.0;
    auto modes=pod_modes_from_batch(seedCollected_,std::min(targetRank_,maxRank_),energy);
    if(modes.empty())throw std::runtime_error("Balanced POD seed training produced zero usable modes");
    Wh_.clear();
    for(auto&w:modes)Wh_.insert(Wh_.end(),w.begin(),w.end());
    rank_=(int)modes.size();
    retainedEnergy_=energy;
    upload_basis();ready_=true;
    std::printf(
      "NODALS_BALANCED_POD_BUILD snapshots=%d requestedRank=%d rank=%d retainedEnergy=%.12e "
      "snapshotNormMin=%.12e snapshotNormMax=%.12e method=DETERMINISTIC_SNAPSHOT_POD status=PASS\n",
      seedCollected_,targetRank_,rank_,retainedEnergy_,capturedNormMin_,capturedNormMax_);
    snap_.clear();batchCollected_=0;batchReady_=false;
    snap_.reserve((std::size_t)n_*(std::size_t)trainTarget_);
  }

  void capture_seed(const T*x,int outer){
    std::vector<T>hx((std::size_t)n_);
    HXT1_CUDA(cudaMemcpy(hx.data(),x,(std::size_t)n_*sizeof(T),cudaMemcpyDeviceToHost));
    if(pin_>=0&&pin_<n_)hx[(std::size_t)pin_]=T(0);
    long double n2=0.0L;for(T v:hx){const double d=(double)v;n2+=(long double)d*d;}
    const double norm=std::sqrt((double)n2);
    if(!(norm>1e-300)||!std::isfinite(norm))
      throw std::runtime_error("Balanced POD pressure-correction snapshot has invalid norm");
    capturedNormMin_=std::min(capturedNormMin_,norm);
    capturedNormMax_=std::max(capturedNormMax_,norm);
    const double inv=1.0/norm;
    for(T v:hx)snap_.push_back((double)v*inv);
    ++seedCollected_;
    std::printf(
      "NODALS_BALANCED_POD_SNAPSHOT outer=%d snapshot=%d/%d rawNorm=%.12e kind=SEED_DP status=PASS\n",
      outer,seedCollected_,trainTarget_,norm);
    if(seedCollected_==trainTarget_)build_seed();
  }

  template<class Action>
  double apply_A_host(Action&Act,const std::vector<double>&v,std::vector<T>&hAv){
    if(!tmpv_.p){tmpv_.alloc((std::size_t)n_);tmpAv_.alloc((std::size_t)n_);}
    std::vector<T>hv((std::size_t)n_);
    for(int i=0;i<n_;++i)hv[(std::size_t)i]=(T)v[(std::size_t)i];
    HXT1_CUDA(cudaMemcpy(tmpv_.p,hv.data(),(std::size_t)n_*sizeof(T),cudaMemcpyHostToDevice));
    Act.apply(tmpv_.p,tmpAv_.p);
    hxt4a::pin_zero_kernel<T><<<1,1>>>(pin_,tmpAv_.p);
    HXT1_CUDA(cudaGetLastError());
    HXT1_CUDA(cudaMemcpy(hAv.data(),tmpAv_.p,(std::size_t)n_*sizeof(T),cudaMemcpyDeviceToHost));
    long double e=0.0L;
    for(int i=0;i<n_;++i)e+=(long double)v[(std::size_t)i]*(long double)hAv[(std::size_t)i];
    return (double)e;
  }

  template<class Action>
  bool a_orthogonalize_candidate(
      Action&Act,std::vector<double>&v,
      const std::vector<std::vector<double>>&newModes,
      const std::vector<double>&newAenergy,double&ratioOut)
  {
    const double before=norm2(v);
    if(!(before>0.0))return false;
    std::vector<T>hAv((std::size_t)n_);

    // Two current-A MGS passes.  The old space is projected jointly through
    // G^{-1}; newly accepted candidates in this tentative block are projected
    // individually using their A energies.
    for(int pass=0;pass<2;++pass){
      (void)apply_A_host(Act,v,hAv);
      if(rank_>0){
        if(!prepared_)throw std::runtime_error("A-orthogonalization requires current-A prepared old basis");
        std::vector<double>rhs((std::size_t)rank_,0.0);
        for(int j=0;j<rank_;++j){
          const double*wj=Wh_.data()+(std::size_t)j*n_;
          rhs[(std::size_t)j]=(double)host_dot(wj,hAv,n_);
        }
        auto c=matvec(Ginv_,rhs,rank_);
        for(int j=0;j<rank_;++j){
          const double*cwj=Wh_.data()+(std::size_t)j*n_;
          const double cj=c[(std::size_t)j];
          for(int i=0;i<n_;++i)v[(std::size_t)i]-=cj*cwj[i];
        }
      }

      if(!newModes.empty()){
        // Recompute A*v after removing the old block, then project against
        // the A-orthogonal tentative modes accepted earlier in this batch.
        (void)apply_A_host(Act,v,hAv);
        for(std::size_t j=0;j<newModes.size();++j){
          long double num=0.0L;
          const auto&u=newModes[j];
          for(int i=0;i<n_;++i)num+=(long double)u[(std::size_t)i]*(long double)hAv[(std::size_t)i];
          const double den=newAenergy[j];
          if(!(den>0.0))return false;
          const double a=(double)num/den;
          for(int i=0;i<n_;++i)v[(std::size_t)i]-=a*u[(std::size_t)i];
        }
      }
    }

    const double after=norm2(v);
    const double ratio=after/before;
    ratioOut=ratio;
    aProjectMinRatio_=std::min(aProjectMinRatio_,ratio);
    aProjectMaxRatio_=std::max(aProjectMaxRatio_,ratio);
    if(!(ratio>noveltyTol_)||!std::isfinite(ratio))return false;
    for(double&x:v)x/=after;
    const double ae=apply_A_host(Act,v,hAv);
    return ae>1e-300&&std::isfinite(ae);
  }

 public:
  BalancedPOD()=default;
  BalancedPOD(
      int n,int pin,bool enabled,bool adaptive,int trainTarget,int targetRank,int maxRank,double noveltyTol,
      int tailIter=10,int hardIts=0,int acceptMinAbs=5,double acceptMinRel=0.03,
      int refreshAbs=30,double refreshRel=0.35,
      int recompressRank=0,int recompressHistory=18,int recompressMaxLossAbs=5,double recompressMaxLossRel=0.10,
      std::string recompressMethod="tail_pod",bool recompressForce=false,bool noProbe=false):
    n_(n),pin_(pin),trainTarget_(trainTarget),targetRank_(targetRank),maxRank_(maxRank),
    tailIter_(tailIter),hardIts_(hardIts),enabled_(enabled),adaptive_(adaptive),noveltyTol_(noveltyTol),
    acceptMinAbs_(acceptMinAbs),acceptMinRel_(acceptMinRel),
    refreshAbs_(refreshAbs),refreshRel_(refreshRel),
    recompressRank_(recompressRank),recompressHistory_(recompressHistory),
    recompressMaxLossAbs_(recompressMaxLossAbs),recompressMaxLossRel_(recompressMaxLossRel),
    recompressMethod_(recompressMethod),recompressForce_(recompressForce),noProbe_(noProbe)
  {
    if(enabled_){
      if(n_<=0||trainTarget_<=0||targetRank_<=0||targetRank_>trainTarget_||maxRank_<targetRank_||
         !(noveltyTol_>=0.0)||tailIter_<=0||hardIts_<0||
         (hardIts_>0&&hardIts_<=6*tailIter_)||
         acceptMinAbs_<0||!(acceptMinRel_>=0.0&&acceptMinRel_<1.0)||
         refreshAbs_<0||!(refreshRel_>=0.0&&refreshRel_<10.0)||
         recompressRank_<0||recompressHistory_<=0||recompressMaxLossAbs_<0||
         !(recompressMaxLossRel_>=0.0&&recompressMaxLossRel_<1.0)||
         !(recompressMethod_=="tail_pod"||recompressMethod_=="whiten_tail_pod"||recompressMethod_=="amg_spectral")||
         (recompressRank_>0&&(recompressRank_>=maxRank_||recompressRank_<targetRank_)))
        throw std::runtime_error("bad Balanced POD configuration");
      snap_.reserve((std::size_t)n_*(std::size_t)trainTarget_);
      std::printf(
        "NODALS_BALANCED_POD_CONFIG enabled=1 mode=%s trainSnapshots=%d requestedRank=%d maxRank=%d noveltyTol=%.12e "
        "tailIter=%d hardIts=%d multiTailDepths=[%d,%d,%d] acceptMinAbs=%d acceptMinRel=%.12e refreshAbs=%d refreshRel=%.12e "
        "recompressRank=%d recompressHistory=%d recompressMaxLossAbs=%d recompressMaxLossRel=%.12e "
        "recompressMethod=%s recompressForce=%d noProbe=%d "
        "rankPolicy=GROW_UNTIL_MAXRANK postMaxPolicy=CONFIGURED_RECOMPRESS_OR_REFRESH "
        "seedSnapshot=CONVERGED_PRESSURE_CORRECTION_DP adaptiveSnapshot=%s "
        "candidateOrthogonalization=CURRENT_A_MGS acceptance=SAME_SYSTEM_PCG_PROBE "
        "normalization=L2 coarsePolicy=REBUILD_AW_G_EVERY_PRESSURE_SOLVE "
        "balance=Q_PLUS_IQA_B_IAQ status=PASS\n",
        adaptive_?"adaptive":"online",trainTarget_,targetRank_,maxRank_,noveltyTol_,
        tailIter_,hardIts_,tailIter_,3*tailIter_,6*tailIter_,acceptMinAbs_,acceptMinRel_,refreshAbs_,refreshRel_,
        recompressRank_,recompressHistory_,recompressMaxLossAbs_,recompressMaxLossRel_,
        recompressMethod_.c_str(),(int)recompressForce_,(int)noProbe_,
        hardIts_>0?"SINGLE_HARD_SOLVE_MULTI_TAIL_XFINAL_MINUS_XK":"KRYLOV_TAIL_CORRECTION_XFINAL_MINUS_XTAIL");
    }else std::printf("NODALS_BALANCED_POD_CONFIG enabled=0 status=PASS\n");
  }

  bool enabled()const{return enabled_;}
  bool adaptive()const{return adaptive_;}
  bool active()const{return enabled_&&ready_;}
  bool has_pending_batch()const{
    if(!(adaptive_&&!tentative_))return false;
    if(recompressPending_)return true;
    if(!batchReady_)return false;
    if(rank_<maxRank_)return true;
    return recompressRank_<=0&&rank_==maxRank_&&refreshStale_;
  }
  int probe_max(int baseIts,int pMax)const{
    if(tentativeIsRecompress_){
      if(recompressForce_)return pMax;
      const int loss=std::max(recompressMaxLossAbs_,(int)std::ceil(recompressMaxLossRel_*(double)baseIts));
      return std::max(1,std::min(pMax,baseIts+loss));
    }
    return std::max(1,std::min(pMax,baseIts));
  }
  bool tentative()const{return tentative_;}
  int rank()const{return rank_;}
  int collected()const{return seedCollected_;}
  int tail_iter()const{return tailIter_;}
  int hard_its()const{return hardIts_;}
  T* probe_x(){if(!probeX_.p)probeX_.alloc((std::size_t)n_);return probeX_.p;}
  bool no_probe()const{return noProbe_;}

  void accept_tentative_without_probe(int outer,int baseIts){
    if(!tentative_)return;
    const int candRank=rank_,oldRank=rollbackRank_;
    const bool refresh=tentativeIsRefresh_;
    const bool recompress=tentativeIsRecompress_;
    const int replaced=tentativeReplaceBlock_;

    if(recompress){
      ++recompressAccepted_;++acceptedUpdates_;
      rollbackWh_.clear();rollbackRank_=0;tentativeAdded_=0;tentative_=false;tentativeIsRecompress_=false;
      std::printf(
        "NODALS_BALANCED_POD_RECOMPRESS_DECISION outer=%d accepted=1 rankOld=%d rankNew=%d baseIts=%d "
        "probeIts=-1 loss=NA allowedLoss=NA method=%s forceAccept=%d noProbe=1 status=PASS\n",
        outer,oldRank,candRank,baseIts,recompressMethod_.c_str(),(int)recompressForce_);
      return;
    }

    ++acceptedUpdates_;
    if(refresh){
      ++refreshAccepted_;++refreshGeneration_;refreshCursor_=0;reset_refresh_monitor();
    }else if(candRank==maxRank_&&oldRank<maxRank_){
      if(recompressRank_>0){
        recompressPending_=(int)tailHistory_.size()>=recompressRank_;
        std::printf(
          "NODALS_BALANCED_POD_RECOMPRESS_ARM outer=%d rank=%d targetRank=%d history=%d requiredHistory=%d "
          "armed=%d status=PASS\n",
          outer,candRank,recompressRank_,(int)tailHistory_.size(),recompressRank_,(int)recompressPending_);
      }else{
        refreshGeneration_=1;refreshCursor_=0;reset_refresh_monitor();
      }
    }
    rollbackWh_.clear();rollbackRank_=0;tentativeAdded_=0;tentative_=false;
    tentativeIsRefresh_=false;tentativeReplaceBlock_=-1;
    std::printf(
      "NODALS_BALANCED_POD_AUGMENT_DECISION outer=%d accepted=1 mode=%s generation=%d replaceBlock=%d "
      "rankOld=%d rankNew=%d baseIts=%d probeIts=-1 improvement=NA required=NA "
      "acceptMinAbs=%d acceptMinRel=%.12e noProbe=1 status=PASS\n",
      outer,refresh?"REFRESH_REPLACE":"AUGMENT",refreshGeneration_,refresh?replaced:-1,
      oldRank,candRank,baseIts,acceptMinAbs_,acceptMinRel_);
  }

  void begin_solve(int outer){
    currentOuter_=outer;
    tailIterateSaved_=false;
    multiTailSaved1_=multiTailSaved2_=multiTailSaved3_=false;
    tailSavedIter_=0;
  }

  void note_solve(int outer,int its,bool usedPod){
    if(!enabled_)return;
    if(usedPod){++podSolves_;podIts_+=its;}else{++stockSolves_;stockIts_+=its;}
    std::printf(
      "NODALS_BALANCED_POD_SOLVE outer=%d rank=%d usedPOD=%d pCG=%d mode=%s status=PASS\n",
      outer,rank_,(int)usedPod,its,adaptive_?"adaptive":"online");

    // At the fixed rank cap, monitor non-overlapping windows aligned with
    // the tail-correction batches.  The best mean PCG count since the most
    // recent accepted refresh is the generation reference.  A window is
    // stale only when both relative and absolute degradation thresholds fire.
    if(adaptive_&&usedPod&&rank_==maxRank_&&!tentative_&&recompressRank_<=0){
      refreshWindowIts_.push_back(its);
      if((int)refreshWindowIts_.size()>=trainTarget_){
        long long ss=0;for(int v:refreshWindowIts_)ss+=v;
        const double mean=(double)ss/(double)refreshWindowIts_.size();
        ++refreshWindows_;
        if(!std::isfinite(refreshBestMean_)||mean<refreshBestMean_)refreshBestMean_=mean;
        const double absDeg=mean-refreshBestMean_;
        const double relDeg=(refreshBestMean_>0.0)?absDeg/refreshBestMean_:0.0;
        refreshStale_=refreshWindows_>=2&&absDeg>=(double)refreshAbs_&&relDeg>=refreshRel_;
        std::printf(
          "NODALS_BALANCED_POD_REFRESH_MONITOR outer=%d generation=%d window=%d meanPCG=%.6f "
          "bestMeanPCG=%.6f absDegrade=%.6f relDegrade=%.12e refreshAbs=%d refreshRel=%.12e "
          "stale=%d replaceCursor=%d status=PASS\n",
          outer,refreshGeneration_,refreshWindows_,mean,refreshBestMean_,absDeg,relDeg,
          refreshAbs_,refreshRel_,(int)refreshStale_,refreshCursor_);
        refreshWindowIts_.clear();

        // If this generation is still healthy, the just-completed three-tail
        // batch is deliberately forgotten.  This makes the next batch track
        // the current operator instead of keeping an old candidate around.
        if(batchReady_&&!refreshStale_){
          clear_batch();
          std::printf(
            "NODALS_BALANCED_POD_REFRESH_BATCH outer=%d generation=%d action=DISCARD_HEALTHY_WINDOW status=PASS\n",
            outer,refreshGeneration_);
        }
      }
    }
  }

  // Called by PCG immediately after the iterate/residual update.
  //
  // Legacy adaptive mode (hardIts_==0) saves one iterate at tailIter_ and,
  // after convergence, learns x_final-x_tailIter across trainTarget_ SIMPLE
  // solves.
  //
  // Single-hard-solve multi-tail mode (hardIts_>0) saves three iterates from
  // the SAME pressure solve at [tailIter_,3*tailIter_,6*tailIter_].  If that
  // solve finishes with at least hardIts_ PCG iterations, the three normalized
  // tails [x_f-x_k1, x_f-x_k2, x_f-x_k3] become one immediate POD batch.
  // Hence one hard solve can teach the next basis update without waiting for
  // two additional SIMPLE pressure solves.
  void save_krylov_tail_iterate(const T*x,int iter){
    if(!adaptive_||!ready_||tentative_||batchReady_)return;

    if(hardIts_>0){
      const int d1=tailIter_,d2=3*tailIter_,d3=6*tailIter_;
      Buf<T>*dst=nullptr;bool*flag=nullptr;int slot=0;
      if(iter==d1){dst=&tailX_;flag=&multiTailSaved1_;slot=1;}
      else if(iter==d2){dst=&tailX2_;flag=&multiTailSaved2_;slot=2;}
      else if(iter==d3){dst=&tailX3_;flag=&multiTailSaved3_;slot=3;}
      else return;
      if(!dst->p)dst->alloc((std::size_t)n_);
      HXT1_CUDA(cudaMemcpy(dst->p,x,(std::size_t)n_*sizeof(T),cudaMemcpyDeviceToDevice));
      hxt4a::pin_zero_kernel<T><<<1,1>>>(pin_,dst->p);
      HXT1_CUDA(cudaGetLastError());
      *flag=true;
      std::printf(
        "NODALS_BALANCED_POD_MULTI_TAIL_ITERATE outer=%d slot=%d pcgIter=%d "
        "depths=[%d,%d,%d] source=X_AT_FIXED_KRYLOV_DEPTH status=PASS\n",
        currentOuter_,slot,iter,d1,d2,d3);
      return;
    }

    if(iter!=tailIter_)return;
    if(!tailX_.p)tailX_.alloc((std::size_t)n_);
    HXT1_CUDA(cudaMemcpy(tailX_.p,x,(std::size_t)n_*sizeof(T),cudaMemcpyDeviceToDevice));
    hxt4a::pin_zero_kernel<T><<<1,1>>>(pin_,tailX_.p);
    HXT1_CUDA(cudaGetLastError());
    tailIterateSaved_=true;
    tailSavedIter_=iter;
    std::printf(
      "NODALS_BALANCED_POD_TAIL_ITERATE outer=%d pcgIter=%d source=X_AT_FIXED_KRYLOV_DEPTH status=PASS\n",
      currentOuter_,iter);
  }

  // End-of-PCG learning hook. Probe solves call PCG with podLearn=false and
  // therefore never enter this function.
  void capture_krylov_tail_correction(const T*xFinal,int finalIts,bool solveOK){
    if(!adaptive_||!ready_||tentative_||batchReady_)return;

    if(hardIts_>0){
      const int d1=tailIter_,d2=3*tailIter_,d3=6*tailIter_;
      const bool hard=solveOK&&finalIts>=hardIts_;
      if(!hard){
        std::printf(
          "NODALS_BALANCED_POD_MULTI_TAIL_DECISION outer=%d finalIts=%d hardThreshold=%d "
          "learn=0 reason=%s status=PASS\n",
          currentOuter_,finalIts,hardIts_,solveOK?"BELOW_HARD_THRESHOLD":"PRESSURE_SOLVE_NOT_CONVERGED");
        multiTailSaved1_=multiTailSaved2_=multiTailSaved3_=false;
        return;
      }
      if(!(multiTailSaved1_&&multiTailSaved2_&&multiTailSaved3_)){
        std::printf(
          "NODALS_BALANCED_POD_MULTI_TAIL_DECISION outer=%d finalIts=%d hardThreshold=%d learn=0 "
          "saved=[%d,%d,%d] reason=MISSING_KRYLOV_CHECKPOINT status=PASS\n",
          currentOuter_,finalIts,hardIts_,(int)multiTailSaved1_,(int)multiTailSaved2_,(int)multiTailSaved3_);
        multiTailSaved1_=multiTailSaved2_=multiTailSaved3_=false;
        return;
      }

      std::vector<T>hf((std::size_t)n_),h1((std::size_t)n_),h2((std::size_t)n_),h3((std::size_t)n_);
      HXT1_CUDA(cudaMemcpy(hf.data(),xFinal,(std::size_t)n_*sizeof(T),cudaMemcpyDeviceToHost));
      HXT1_CUDA(cudaMemcpy(h1.data(),tailX_.p,(std::size_t)n_*sizeof(T),cudaMemcpyDeviceToHost));
      HXT1_CUDA(cudaMemcpy(h2.data(),tailX2_.p,(std::size_t)n_*sizeof(T),cudaMemcpyDeviceToHost));
      HXT1_CUDA(cudaMemcpy(h3.data(),tailX3_.p,(std::size_t)n_*sizeof(T),cudaMemcpyDeviceToHost));
      if(pin_>=0&&pin_<n_){
        hf[(std::size_t)pin_]=T(0);h1[(std::size_t)pin_]=T(0);
        h2[(std::size_t)pin_]=T(0);h3[(std::size_t)pin_]=T(0);
      }

      const T*hs[3]={h1.data(),h2.data(),h3.data()};
      const int depths[3]={d1,d2,d3};
      int added=0;
      double rawNorm[3]={0.0,0.0,0.0};
      for(int q=0;q<3;++q){
        std::vector<double>e((std::size_t)n_,0.0);
        long double n2=0.0L;
        for(int i=0;i<n_;++i){
          const double d=(double)hf[(std::size_t)i]-(double)hs[q][(std::size_t)i];
          e[(std::size_t)i]=d;n2+=(long double)d*d;
        }
        const double en=std::sqrt((double)n2);rawNorm[q]=en;
        if(!(en>1e-300)||!std::isfinite(en))continue;
        const double inv=1.0/en;
        std::vector<double>enorm((std::size_t)n_);
        for(int i=0;i<n_;++i)enorm[(std::size_t)i]=e[(std::size_t)i]*inv;
        tailHistory_.push_back(enorm);
        if((int)tailHistory_.size()>recompressHistory_)tailHistory_.erase(tailHistory_.begin());
        for(double v:enorm)snap_.push_back(v);
        ++batchCollected_;++added;
        std::printf(
          "NODALS_BALANCED_POD_MULTI_TAIL_CORRECTION outer=%d depth=%d finalIts=%d remainingIts=%d "
          "snapshot=%d/%d rawNorm=%.12e source=X_FINAL_MINUS_XK_SAME_HARD_SOLVE status=PASS\n",
          currentOuter_,depths[q],finalIts,finalIts-depths[q],batchCollected_,trainTarget_,en);
      }
      multiTailSaved1_=multiTailSaved2_=multiTailSaved3_=false;
      if(added>=targetRank_&&batchCollected_>=trainTarget_){
        batchReady_=true;
        std::printf(
          "NODALS_BALANCED_POD_MULTI_TAIL_DECISION outer=%d finalIts=%d hardThreshold=%d learn=1 "
          "snapshots=%d depths=[%d,%d,%d] rawNorms=[%.12e,%.12e,%.12e] "
          "policy=IMMEDIATE_SAME_SOLVE_BATCH status=PASS\n",
          currentOuter_,finalIts,hardIts_,batchCollected_,d1,d2,d3,
          rawNorm[0],rawNorm[1],rawNorm[2]);
      }else{
        std::printf(
          "NODALS_BALANCED_POD_MULTI_TAIL_DECISION outer=%d finalIts=%d hardThreshold=%d learn=0 "
          "snapshots=%d requiredSnapshots=%d reason=INSUFFICIENT_USABLE_TAILS status=PASS\n",
          currentOuter_,finalIts,hardIts_,batchCollected_,trainTarget_);
        clear_batch();
      }
      return;
    }

    if(!tailIterateSaved_)return;
    const int savedIter=tailSavedIter_;
    tailIterateSaved_=false;
    tailSavedIter_=0;
    if(!solveOK||finalIts<=savedIter){
      std::printf(
        "NODALS_BALANCED_POD_TAIL_CORRECTION outer=%d pcgIterSaved=%d finalIts=%d accepted=0 "
        "reason=%s status=PASS\n",
        currentOuter_,savedIter,finalIts,solveOK?"NO_REMAINING_KRYLOV_DEPTH":"PRESSURE_SOLVE_NOT_CONVERGED");
      return;
    }

    std::vector<T>hf((std::size_t)n_),hk((std::size_t)n_);
    HXT1_CUDA(cudaMemcpy(hf.data(),xFinal,(std::size_t)n_*sizeof(T),cudaMemcpyDeviceToHost));
    HXT1_CUDA(cudaMemcpy(hk.data(),tailX_.p,(std::size_t)n_*sizeof(T),cudaMemcpyDeviceToHost));
    if(pin_>=0&&pin_<n_){hf[(std::size_t)pin_]=T(0);hk[(std::size_t)pin_]=T(0);}
    std::vector<double>e((std::size_t)n_,0.0);
    long double n2=0.0L;
    for(int i=0;i<n_;++i){
      const double d=(double)hf[(std::size_t)i]-(double)hk[(std::size_t)i];
      e[(std::size_t)i]=d;
      n2+=(long double)d*d;
    }
    const double en=std::sqrt((double)n2);
    if(!(en>1e-300)||!std::isfinite(en)){
      std::printf(
        "NODALS_BALANCED_POD_TAIL_CORRECTION outer=%d pcgIterSaved=%d finalIts=%d accepted=0 rawNorm=%.12e "
        "reason=ZERO_OR_NONFINITE status=PASS\n",
        currentOuter_,savedIter,finalIts,en);
      return;
    }
    const double inv=1.0/en;
    std::vector<double>enorm((std::size_t)n_);
    for(int i=0;i<n_;++i)enorm[(std::size_t)i]=e[(std::size_t)i]*inv;
    tailHistory_.push_back(enorm);
    if((int)tailHistory_.size()>recompressHistory_)tailHistory_.erase(tailHistory_.begin());
    for(double v:enorm)snap_.push_back(v);
    ++batchCollected_;
    std::printf(
      "NODALS_BALANCED_POD_TAIL_CORRECTION outer=%d pcgIterSaved=%d finalIts=%d remainingIts=%d "
      "snapshot=%d/%d rawNorm=%.12e source=X_FINAL_MINUS_X_TAILITER status=PASS\n",
      currentOuter_,savedIter,finalIts,finalIts-savedIter,batchCollected_,trainTarget_,en);
    if(batchCollected_>=trainTarget_)batchReady_=true;
  }

  void capture_after_solve(const T*dp,int outer){
    if(!enabled_)return;
    if(!ready_)capture_seed(dp,outer);
  }

  template<class Action>
  void prepare(Action&Act,hxt4a::Reducer<T>&red){
    if(!active())return;
    for(int k=0;k<rank_;++k){
      Act.apply(W_.p+(std::size_t)k*n_,AW_.p+(std::size_t)k*n_);
      hxt4a::pin_zero_kernel<T><<<1,1>>>(pin_,AW_.p+(std::size_t)k*n_);
    }
    HXT1_CUDA(cudaGetLastError());
    std::vector<double>G((std::size_t)rank_*(std::size_t)rank_,0.0);
    double asym=0.0,gscale=0.0,offdiag=0.0,diagscale=0.0;
    for(int i=0;i<rank_;++i)for(int j=0;j<rank_;++j)
      G[(std::size_t)i*rank_+j]=red.dot(W_.p+(std::size_t)i*n_,AW_.p+(std::size_t)j*n_);
    for(int i=0;i<rank_;++i){
      diagscale=std::max(diagscale,std::abs(G[(std::size_t)i*rank_+i]));
      for(int j=i+1;j<rank_;++j){
        asym=std::max(asym,std::abs(G[(std::size_t)i*rank_+j]-G[(std::size_t)j*rank_+i]));
        gscale=std::max(gscale,std::max(std::abs(G[(std::size_t)i*rank_+j]),std::abs(G[(std::size_t)j*rank_+i])));
        const double ss=.5*(G[(std::size_t)i*rank_+j]+G[(std::size_t)j*rank_+i]);
        G[(std::size_t)i*rank_+j]=G[(std::size_t)j*rank_+i]=ss;
        offdiag=std::max(offdiag,std::abs(ss));
      }
    }
    Ginv_=pod_spd_inverse(G,rank_);
    prepared_=true;
    double dmin=std::numeric_limits<double>::infinity(),dmax=0.0;
    for(int i=0;i<rank_;++i){dmin=std::min(dmin,G[(std::size_t)i*rank_+i]);dmax=std::max(dmax,G[(std::size_t)i*rank_+i]);}
    std::printf(
      "NODALS_BALANCED_POD_SETUP rank=%d tentative=%d GdiagMin=%.12e GdiagMax=%.12e GsymRel=%.12e GoffdiagRel=%.12e "
      "policy=CURRENT_A status=PASS\n",
      rank_,(int)tentative_,dmin,dmax,asym/std::max(gscale,1e-300),offdiag/std::max(diagscale,1e-300));
  }

  void apply(HybridAMG<T>&PC,const T*r,T*z,hxt4a::Reducer<T>&red){
    if(!prepared_)throw std::runtime_error("Balanced POD apply called before prepare");
    std::vector<double>rhs((std::size_t)rank_,0.0);
    for(int k=0;k<rank_;++k)rhs[(std::size_t)k]=red.dot(W_.p+(std::size_t)k*n_,r);
    auto a=matvec(Ginv_,rhs,rank_);
    std::vector<T>ha((std::size_t)rank_);for(int k=0;k<rank_;++k)ha[(std::size_t)k]=(T)a[(std::size_t)k];
    HXT1_CUDA(cudaMemcpy(coeff_.p,ha.data(),(std::size_t)rank_*sizeof(T),cudaMemcpyHostToDevice));
    pod_build_q_t_kernel<T><<<(n_+AMG_TPB-1)/AMG_TPB,AMG_TPB>>>(n_,rank_,W_.p,AW_.p,coeff_.p,r,q_.p,t_.p);
    hxt4a::pin_zero_kernel<T><<<1,1>>>(pin_,t_.p);
    PC.apply(t_.p,g_.p);hxt4a::pin_zero_kernel<T><<<1,1>>>(pin_,g_.p);
    std::vector<double>rhs2((std::size_t)rank_,0.0);
    for(int k=0;k<rank_;++k)rhs2[(std::size_t)k]=red.dot(AW_.p+(std::size_t)k*n_,g_.p);
    auto b=matvec(Ginv_,rhs2,rank_);
    for(int k=0;k<rank_;++k)ha[(std::size_t)k]=(T)b[(std::size_t)k];
    HXT1_CUDA(cudaMemcpy(coeff_.p,ha.data(),(std::size_t)rank_*sizeof(T),cudaMemcpyHostToDevice));
    pod_finish_kernel<T><<<(n_+AMG_TPB-1)/AMG_TPB,AMG_TPB>>>(n_,rank_,W_.p,coeff_.p,q_.p,g_.p,z);
    hxt4a::pin_zero_kernel<T><<<1,1>>>(pin_,z);HXT1_CUDA(cudaGetLastError());
  }


  static std::vector<double> symmetrized(const std::vector<double>&A,int n){
    std::vector<double>S=A;
    for(int i=0;i<n;++i)for(int j=i+1;j<n;++j){
      const double x=.5*(S[(std::size_t)i*n+j]+S[(std::size_t)j*n+i]);
      S[(std::size_t)i*n+j]=S[(std::size_t)j*n+i]=x;
    }
    return S;
  }

  static bool spd_invsqrt(
      const std::vector<double>&G,int n,std::vector<double>&M,
      double&lamMin,double&lamMax,double&cond)
  {
    std::vector<double>eval,V;
    pod_jacobi_eigen(G,n,eval,V);
    lamMin=std::numeric_limits<double>::infinity();lamMax=0.0;
    for(double x:eval){lamMin=std::min(lamMin,x);lamMax=std::max(lamMax,x);}
    if(!(lamMin>1e-13*std::max(lamMax,1e-300))||!std::isfinite(lamMin)||!std::isfinite(lamMax))
      return false;
    cond=lamMax/lamMin;
    M.assign((std::size_t)n*(std::size_t)n,0.0);
    for(int q=0;q<n;++q){
      const double s=1.0/std::sqrt(eval[(std::size_t)q]);
      for(int i=0;i<n;++i){
        const double vi=V[(std::size_t)i*n+q];
        for(int j=0;j<n;++j)
          M[(std::size_t)i*n+j]+=vi*s*V[(std::size_t)j*n+q];
      }
    }
    return true;
  }

  static std::vector<double> dense_mul(
      const std::vector<double>&A,const std::vector<double>&B,int n)
  {
    std::vector<double>C((std::size_t)n*(std::size_t)n,0.0);
    for(int i=0;i<n;++i)for(int k=0;k<n;++k){
      const double aik=A[(std::size_t)i*n+k];
      for(int j=0;j<n;++j)C[(std::size_t)i*n+j]+=aik*B[(std::size_t)k*n+j];
    }
    return C;
  }

  std::vector<double> combine_old_basis(
      const std::vector<double>&C,int oldRank,int newRank)const
  {
    std::vector<double>newW((std::size_t)newRank*(std::size_t)n_,0.0);
    for(int q=0;q<newRank;++q){
      double*wnew=newW.data()+(std::size_t)q*n_;
      for(int j=0;j<oldRank;++j){
        const double a=C[(std::size_t)j*newRank+q];
        const double*wold=rollbackWh_.data()+(std::size_t)j*n_;
        for(int i=0;i<n_;++i)wnew[i]+=a*wold[i];
      }
      long double ss=0.0L;
      for(int i=0;i<n_;++i)ss+=(long double)wnew[i]*wnew[i];
      const double un=std::sqrt((double)ss);
      if(!(un>1e-14)||!std::isfinite(un))
        throw std::runtime_error("recompression produced zero/nonfinite basis vector");
      for(int i=0;i<n_;++i)wnew[i]/=un;
    }
    return newW;
  }

  template<class Action>
  bool build_recompress_tentative(
      Action&Act,HybridAMG<T>&PC,hxt4a::Reducer<T>&red,int outer)
  {
    if(!(recompressPending_&&recompressRank_>0&&rank_==maxRank_))return false;
    if(!prepared_)prepare(Act,red);
    const int m=(int)tailHistory_.size();
    const int oldRank=rank_;
    const int want=std::min(recompressRank_,oldRank);
    if(want<=0||(recompressMethod_!="amg_spectral"&&m<=0)){
      recompressPending_=false;++recompressRejected_;
      std::printf(
        "NODALS_BALANCED_POD_RECOMPRESS_DECISION outer=%d accepted=0 rankOld=%d "
        "reason=INSUFFICIENT_HISTORY_OR_TARGET history=%d target=%d method=%s status=PASS\n",
        outer,rank_,m,recompressRank_,recompressMethod_.c_str());
      return false;
    }

    rollbackWh_=Wh_;rollbackRank_=rank_;
    tentativeIsRecompress_=true;tentativeIsRefresh_=false;tentativeReplaceBlock_=-1;

    std::vector<T>hAW((std::size_t)oldRank*(std::size_t)n_);
    HXT1_CUDA(cudaMemcpy(hAW.data(),AW_.p,hAW.size()*sizeof(T),cudaMemcpyDeviceToHost));

    std::vector<double>G((std::size_t)oldRank*(std::size_t)oldRank,0.0);
    for(int i=0;i<oldRank;++i){
      const double*wi=rollbackWh_.data()+(std::size_t)i*n_;
      for(int j=0;j<oldRank;++j){
        const T*awj=hAW.data()+(std::size_t)j*n_;
        long double ss=0.0L;
        for(int k=0;k<n_;++k)ss+=(long double)wi[k]*(long double)awj[k];
        G[(std::size_t)i*oldRank+j]=(double)ss;
      }
    }
    G=symmetrized(G,oldRank);

    std::vector<double>Gmhalf;
    double gmin=0.0,gmax=0.0,gcond=0.0;
    if(!spd_invsqrt(G,oldRank,Gmhalf,gmin,gmax,gcond)){
      Wh_=rollbackWh_;rank_=rollbackRank_;
      rollbackWh_.clear();rollbackRank_=0;tentativeIsRecompress_=false;recompressPending_=false;
      ++recompressRejected_;
      std::printf(
        "NODALS_BALANCED_POD_RECOMPRESS_DECISION outer=%d accepted=0 rankOld=%d reason=G_NOT_SPD "
        "method=%s status=PASS\n",outer,rank_,recompressMethod_.c_str());
      return false;
    }
    std::printf(
      "NODALS_BALANCED_POD_RECOMPRESS_G_SPECTRUM outer=%d rank=%d Gmin=%.12e Gmax=%.12e Gcond=%.12e status=PASS\n",
      outer,oldRank,gmin,gmax,gcond);

    std::vector<double>Csel;
    int keptModes=0;
    double energy=0.0;
    double muMin=std::numeric_limits<double>::quiet_NaN();
    double muSelMax=std::numeric_limits<double>::quiet_NaN();
    double muDropMin=std::numeric_limits<double>::quiet_NaN();
    double muMax=std::numeric_limits<double>::quiet_NaN();

    if(recompressMethod_=="tail_pod"){
      const int take=std::min(want,m);
      std::vector<double>B((std::size_t)m*(std::size_t)oldRank,0.0);
      std::vector<double>C((std::size_t)m*(std::size_t)oldRank,0.0);
      for(int q=0;q<m;++q){
        const auto&h=tailHistory_[(std::size_t)q];
        for(int j=0;j<oldRank;++j){
          long double ss=0.0L;
          const T*aw=hAW.data()+(std::size_t)j*n_;
          for(int i=0;i<n_;++i)ss+=(long double)aw[i]*(long double)h[(std::size_t)i];
          B[(std::size_t)q*oldRank+j]=(double)ss;
        }
        std::vector<double>b((std::size_t)oldRank);
        for(int j=0;j<oldRank;++j)b[(std::size_t)j]=B[(std::size_t)q*oldRank+j];
        auto c=matvec(Ginv_,b,oldRank);
        for(int j=0;j<oldRank;++j)C[(std::size_t)q*oldRank+j]=c[(std::size_t)j];
      }
      std::vector<double>K((std::size_t)m*(std::size_t)m,0.0);
      for(int q=0;q<m;++q)for(int t=q;t<m;++t){
        long double ss=0.0L;
        for(int j=0;j<oldRank;++j)
          ss+=(long double)B[(std::size_t)q*oldRank+j]*C[(std::size_t)t*oldRank+j];
        K[(std::size_t)q*m+t]=K[(std::size_t)t*m+q]=(double)ss;
      }
      std::vector<double>eval,V;
      pod_jacobi_eigen(K,m,eval,V);
      std::vector<int>order((std::size_t)m);for(int i=0;i<m;++i)order[(std::size_t)i]=i;
      std::sort(order.begin(),order.end(),[&](int a,int b){return eval[(std::size_t)a]>eval[(std::size_t)b];});
      const double lam0=std::max(eval[(std::size_t)order[0]],0.0);
      double total=0.0,kept=0.0;for(double x:eval)if(x>0.0)total+=x;
      Csel.assign((std::size_t)oldRank*(std::size_t)take,0.0);
      for(int kk=0;kk<take;++kk){
        const int ec=order[(std::size_t)kk];
        const double lam=eval[(std::size_t)ec];
        if(!(lam>1e-12*std::max(lam0,1e-300)))continue;
        const double invSig=1.0/std::sqrt(lam);
        for(int j=0;j<oldRank;++j){
          long double a=0.0L;
          for(int q=0;q<m;++q)
            a+=(long double)V[(std::size_t)q*m+ec]*invSig*C[(std::size_t)q*oldRank+j];
          Csel[(std::size_t)j*take+keptModes]=(double)a;
        }
        kept+=lam;++keptModes;
      }
      energy=(total>0.0)?kept/total:0.0;
      if(keptModes<take){
        std::vector<double>Tc((std::size_t)oldRank*(std::size_t)keptModes);
        for(int j=0;j<oldRank;++j)for(int q=0;q<keptModes;++q)
          Tc[(std::size_t)j*keptModes+q]=Csel[(std::size_t)j*take+q];
        Csel.swap(Tc);
      }
    }else if(recompressMethod_=="whiten_tail_pod"){
      std::vector<double>B((std::size_t)m*(std::size_t)oldRank,0.0);
      std::vector<double>D((std::size_t)m*(std::size_t)oldRank,0.0);
      for(int q=0;q<m;++q){
        const auto&h=tailHistory_[(std::size_t)q];
        for(int j=0;j<oldRank;++j){
          long double ss=0.0L;
          const T*aw=hAW.data()+(std::size_t)j*n_;
          for(int i=0;i<n_;++i)ss+=(long double)aw[i]*(long double)h[(std::size_t)i];
          B[(std::size_t)q*oldRank+j]=(double)ss;
        }
        for(int a=0;a<oldRank;++a){
          long double ss=0.0L;
          for(int j=0;j<oldRank;++j)
            ss+=(long double)B[(std::size_t)q*oldRank+j]*Gmhalf[(std::size_t)j*oldRank+a];
          D[(std::size_t)q*oldRank+a]=(double)ss;
        }
      }
      std::vector<double>S((std::size_t)oldRank*(std::size_t)oldRank,0.0);
      for(int a=0;a<oldRank;++a)for(int b=a;b<oldRank;++b){
        long double ss=0.0L;
        for(int q=0;q<m;++q)
          ss+=(long double)D[(std::size_t)q*oldRank+a]*D[(std::size_t)q*oldRank+b];
        S[(std::size_t)a*oldRank+b]=S[(std::size_t)b*oldRank+a]=(double)ss;
      }
      std::vector<double>eval,Y;
      pod_jacobi_eigen(S,oldRank,eval,Y);
      std::vector<int>order((std::size_t)oldRank);for(int i=0;i<oldRank;++i)order[(std::size_t)i]=i;
      std::sort(order.begin(),order.end(),[&](int a,int b){return eval[(std::size_t)a]>eval[(std::size_t)b];});
      const double lam0=std::max(eval[(std::size_t)order[0]],0.0);
      double total=0.0,kept=0.0;for(double x:eval)if(x>0.0)total+=x;
      Csel.assign((std::size_t)oldRank*(std::size_t)want,0.0);
      for(int kk=0;kk<want;++kk){
        const int ec=order[(std::size_t)kk];
        const double lam=eval[(std::size_t)ec];
        if(!(lam>1e-12*std::max(lam0,1e-300)))continue;
        for(int j=0;j<oldRank;++j){
          long double ss=0.0L;
          for(int a=0;a<oldRank;++a)
            ss+=(long double)Gmhalf[(std::size_t)j*oldRank+a]*Y[(std::size_t)a*oldRank+ec];
          Csel[(std::size_t)j*want+keptModes]=(double)ss;
        }
        kept+=lam;++keptModes;
      }
      energy=(total>0.0)?kept/total:0.0;
      if(keptModes<want){
        std::vector<double>Tc((std::size_t)oldRank*(std::size_t)keptModes);
        for(int j=0;j<oldRank;++j)for(int q=0;q<keptModes;++q)
          Tc[(std::size_t)j*keptModes+q]=Csel[(std::size_t)j*want+q];
        Csel.swap(Tc);
      }
    }else{
      std::vector<T>hBAW((std::size_t)oldRank*(std::size_t)n_);
      for(int j=0;j<oldRank;++j){
        PC.apply(AW_.p+(std::size_t)j*n_,tmpAv_.p);
        hxt4a::pin_zero_kernel<T><<<1,1>>>(pin_,tmpAv_.p);
        HXT1_CUDA(cudaGetLastError());
        HXT1_CUDA(cudaMemcpy(
          hBAW.data()+(std::size_t)j*n_,tmpAv_.p,(std::size_t)n_*sizeof(T),cudaMemcpyDeviceToHost));
      }
      std::vector<double>K((std::size_t)oldRank*(std::size_t)oldRank,0.0);
      for(int i=0;i<oldRank;++i)for(int j=0;j<oldRank;++j){
        long double ss=0.0L;
        const T*awi=hAW.data()+(std::size_t)i*n_;
        const T*bawj=hBAW.data()+(std::size_t)j*n_;
        for(int k=0;k<n_;++k)ss+=(long double)awi[k]*(long double)bawj[k];
        K[(std::size_t)i*oldRank+j]=(double)ss;
      }
      K=symmetrized(K,oldRank);
      auto MK=dense_mul(Gmhalf,K,oldRank);
      auto R=dense_mul(MK,Gmhalf,oldRank);
      R=symmetrized(R,oldRank);
      std::vector<double>mu,Y;
      pod_jacobi_eigen(R,oldRank,mu,Y);
      std::vector<int>order((std::size_t)oldRank);for(int i=0;i<oldRank;++i)order[(std::size_t)i]=i;
      std::sort(order.begin(),order.end(),[&](int a,int b){return mu[(std::size_t)a]<mu[(std::size_t)b];});
      muMin=mu[(std::size_t)order.front()];
      muMax=mu[(std::size_t)order.back()];
      muSelMax=mu[(std::size_t)order[(std::size_t)(want-1)]];
      muDropMin=(want<oldRank)?mu[(std::size_t)order[(std::size_t)want]]:muSelMax;
      Csel.assign((std::size_t)oldRank*(std::size_t)want,0.0);
      for(int kk=0;kk<want;++kk){
        const int ec=order[(std::size_t)kk];
        for(int j=0;j<oldRank;++j){
          long double ss=0.0L;
          for(int a=0;a<oldRank;++a)
            ss+=(long double)Gmhalf[(std::size_t)j*oldRank+a]*Y[(std::size_t)a*oldRank+ec];
          Csel[(std::size_t)j*want+kk]=(double)ss;
        }
      }
      keptModes=want;
      std::printf(
        "NODALS_BALANCED_POD_AMG_SPECTRUM outer=%d rank=%d select=%d muMin=%.12e muSelectedMax=%.12e "
        "muDroppedMin=%.12e muMax=%.12e spectralCondition=%.12e "
        "metric=GENERALIZED_RITZ_AWB0AW_OVER_WAW status=PASS\n",
        outer,oldRank,want,muMin,muSelMax,muDropMin,muMax,
        muMax/std::max(std::abs(muMin),1e-300));
    }

    if(keptModes<=0){
      Wh_=rollbackWh_;rank_=rollbackRank_;
      rollbackWh_.clear();rollbackRank_=0;tentativeIsRecompress_=false;recompressPending_=false;
      ++recompressRejected_;
      std::printf(
        "NODALS_BALANCED_POD_RECOMPRESS_DECISION outer=%d accepted=0 rankOld=%d reason=ZERO_USABLE_MODES "
        "method=%s status=PASS\n",outer,rank_,recompressMethod_.c_str());
      return false;
    }

    Wh_=combine_old_basis(Csel,oldRank,keptModes);
    rank_=keptModes;tentative_=true;recompressPending_=false;
    upload_basis();prepare(Act,red);
    ++recompressCount_;
    std::printf(
      "NODALS_BALANCED_POD_RECOMPRESS_TENTATIVE outer=%d cycle=%d rankOld=%d rankCandidate=%d targetRank=%d "
      "history=%d retainedProjectedTailEnergy=%.12e method=%s Gcond=%.12e "
      "muMin=%.12e muSelectedMax=%.12e muDroppedMin=%.12e muMax=%.12e "
      "probe=SAME_SYSTEM forceAccept=%d status=PASS\n",
      outer,recompressCount_,rollbackRank_,rank_,recompressRank_,m,energy,recompressMethod_.c_str(),gcond,
      muMin,muSelMax,muDropMin,muMax,(int)recompressForce_);
    return true;
  }

  // Build either a rank-growing augmentation (before maxRank) or a fixed-rank
  // moving-basis refresh (at maxRank).  Refresh removes one old targetRank
  // block, current-A orthogonalizes the new tail-correction block against the
  // survivors, appends it as the newest block, and probes the exact same A,b.
  // A rejected replacement restores the previous basis byte-for-byte.
  template<class Action>
  bool build_tentative(Action&Act,HybridAMG<T>&PC,hxt4a::Reducer<T>&red,int outer){
    if(!has_pending_batch())return false;
    if(recompressPending_)return build_recompress_tentative(Act,PC,red,outer);
    if(!prepared_)prepare(Act,red);

    const bool refresh=(rank_==maxRank_&&recompressRank_<=0);
    const int room=refresh?targetRank_:std::max(0,maxRank_-rank_);
    const int wanted=std::min(targetRank_,room);
    double energy=0.0;
    auto rawModes=pod_modes_from_batch(batchCollected_,wanted,energy);
    lastBatchEnergy_=energy;

    rollbackWh_=Wh_;rollbackRank_=rank_;
    tentativeIsRefresh_=refresh;
    tentativeReplaceBlock_=-1;

    if(refresh){
      if(targetRank_<=0||rank_%targetRank_!=0||wanted!=targetRank_){
        clear_batch();
        ++rejectedUpdates_;++refreshRejected_;
        std::printf(
          "NODALS_BALANCED_POD_REFRESH_DECISION outer=%d accepted=0 rank=%d reason=INCOMPATIBLE_BLOCK_LAYOUT "
          "blockSize=%d status=PASS\n",outer,rank_,targetRank_);
        rollbackWh_.clear();rollbackRank_=0;tentativeIsRefresh_=false;
        return false;
      }
      const int nBlocks=rank_/targetRank_;
      const int block=(nBlocks>0)?(refreshCursor_%nBlocks):0;
      const int k0=block*targetRank_,k1=k0+targetRank_;
      tentativeReplaceBlock_=block;

      // Keep all old blocks except the selected one.  Candidate modes are
      // then A-orthogonalized against this rank-(maxRank-targetRank) survivor
      // space, not against the block they are intended to replace.
      std::vector<double>survivors;
      survivors.reserve((std::size_t)(rank_-targetRank_)*(std::size_t)n_);
      for(int k=0;k<rank_;++k){
        if(k>=k0&&k<k1)continue;
        const double*w=rollbackWh_.data()+(std::size_t)k*n_;
        survivors.insert(survivors.end(),w,w+n_);
      }
      Wh_=std::move(survivors);rank_=rollbackRank_-targetRank_;
      upload_basis();prepare(Act,red);
    }

    std::vector<std::vector<double>>newModes;
    std::vector<double>newAenergy;
    for(auto v:rawModes){
      const double before=norm2(v);
      double projectRatio=0.0;
      if(!a_orthogonalize_candidate(Act,v,newModes,newAenergy,projectRatio))continue;
      std::vector<T>hAv((std::size_t)n_);
      const double ae=apply_A_host(Act,v,hAv);
      if(!(ae>1e-300)||!std::isfinite(ae))continue;
      const double after=norm2(v);
      std::printf(
        "NODALS_BALANCED_POD_CANDIDATE outer=%d candidate=%d rawEuclidNorm=%.12e projectedEuclidNorm=%.12e "
        "projectRatio=%.12e Aenergy=%.12e orthogonalization=CURRENT_A mode=%s status=PASS\n",
        outer,(int)newModes.size()+1,before,after,projectRatio,ae,refresh?"REFRESH":"AUGMENT");
      newModes.push_back(std::move(v));newAenergy.push_back(ae);
    }

    clear_batch();

    // A refresh must restore the full fixed rank; partial replacement would
    // silently change the application cost and is therefore rejected.
    if(newModes.empty()||(refresh&&(int)newModes.size()!=targetRank_)){
      if(refresh){
        Wh_=rollbackWh_;rank_=rollbackRank_;
        const int nBlocks=std::max(1,rank_/std::max(1,targetRank_));
        refreshCursor_=(refreshCursor_+1)%nBlocks;
        ++refreshRejected_;
      }
      ++rejectedUpdates_;
      const int old=rollbackRank_;
      rollbackWh_.clear();rollbackRank_=0;tentativeIsRefresh_=false;tentativeReplaceBlock_=-1;
      upload_basis();
      std::printf(
        "NODALS_BALANCED_POD_AUGMENT_DECISION outer=%d accepted=0 mode=%s rankOld=%d rankNew=%d "
        "reason=%s batchRetainedEnergy=%.12e status=PASS\n",
        outer,refresh?"REFRESH_REPLACE":"AUGMENT",old,rank_,
        refresh?"INCOMPLETE_CURRENT_A_REFRESH_BLOCK":"NO_CURRENT_A_INDEPENDENT_CANDIDATE",energy);
      return false;
    }

    for(auto&w:newModes)Wh_.insert(Wh_.end(),w.begin(),w.end());
    tentativeAdded_=(int)newModes.size();rank_+=tentativeAdded_;tentative_=true;
    upload_basis();prepare(Act,red);
    if(refresh){
      std::printf(
        "NODALS_BALANCED_POD_TENTATIVE outer=%d mode=REFRESH_REPLACE generation=%d replaceBlock=%d "
        "rankOld=%d rankCandidate=%d add=%d batchRetainedEnergy=%.12e "
        "orthogonalization=CURRENT_A probe=SAME_SYSTEM status=PASS\n",
        outer,refreshGeneration_,tentativeReplaceBlock_,rollbackRank_,rank_,tentativeAdded_,energy);
    }else{
      std::printf(
        "NODALS_BALANCED_POD_TENTATIVE outer=%d mode=AUGMENT rankOld=%d rankCandidate=%d add=%d "
        "batchRetainedEnergy=%.12e orthogonalization=CURRENT_A probe=SAME_SYSTEM status=PASS\n",
        outer,rollbackRank_,rank_,tentativeAdded_,energy);
    }
    return true;
  }

  void resolve_tentative(int outer,int baseIts,int probeIts,bool probeOK){
    if(!tentative_)return;
    ++probeCount_;probeIts_+=probeIts;
    const int required=std::max(acceptMinAbs_,(int)std::ceil(acceptMinRel_*(double)baseIts));
    const int improvement=probeOK?(baseIts-probeIts):std::numeric_limits<int>::min()/4;
    const bool accept=probeOK&&improvement>=required;
    const int candRank=rank_;
    const int oldRank=rollbackRank_;
    const bool refresh=tentativeIsRefresh_;
    const bool recompress=tentativeIsRecompress_;
    const int replaced=tentativeReplaceBlock_;

    if(recompress){
      const int allowedLoss=std::max(recompressMaxLossAbs_,(int)std::ceil(recompressMaxLossRel_*(double)baseIts));
      const bool keep=probeOK&&(recompressForce_||probeIts<=baseIts+allowedLoss);
      if(keep){
        ++recompressAccepted_;++acceptedUpdates_;
        rollbackWh_.clear();rollbackRank_=0;tentativeAdded_=0;tentative_=false;tentativeIsRecompress_=false;
        std::printf(
          "NODALS_BALANCED_POD_RECOMPRESS_DECISION outer=%d accepted=1 rankOld=%d rankNew=%d baseIts=%d probeIts=%d "
          "loss=%d allowedLoss=%d maxLossAbs=%d maxLossRel=%.12e method=%s forceAccept=%d status=PASS\n",
          outer,oldRank,candRank,baseIts,probeIts,probeIts-baseIts,allowedLoss,recompressMaxLossAbs_,recompressMaxLossRel_,
          recompressMethod_.c_str(),(int)recompressForce_);
      }else{
        Wh_=rollbackWh_;rank_=rollbackRank_;rollbackWh_.clear();rollbackRank_=0;tentativeAdded_=0;tentative_=false;
        tentativeIsRecompress_=false;++recompressRejected_;++rejectedUpdates_;upload_basis();
        std::printf(
          "NODALS_BALANCED_POD_RECOMPRESS_DECISION outer=%d accepted=0 rankOld=%d rankCandidate=%d rankNew=%d "
          "baseIts=%d probeIts=%d probeOK=%d loss=%d allowedLoss=%d method=%s forceAccept=%d "
          "reason=EXCESS_PCG_LOSS status=PASS\n",
          outer,oldRank,candRank,rank_,baseIts,probeIts,(int)probeOK,probeOK?probeIts-baseIts:-1,allowedLoss,
          recompressMethod_.c_str(),(int)recompressForce_);
      }
      return;
    }

    if(accept){
      ++acceptedUpdates_;
      if(refresh){
        ++refreshAccepted_;
        ++refreshGeneration_;
        refreshCursor_=0;
        reset_refresh_monitor();
      }else if(candRank==maxRank_&&oldRank<maxRank_){
        if(recompressRank_>0){
          recompressPending_=(int)tailHistory_.size()>=recompressRank_;
          std::printf(
            "NODALS_BALANCED_POD_RECOMPRESS_ARM outer=%d rank=%d targetRank=%d history=%d requiredHistory=%d armed=%d status=PASS\n",
            outer,candRank,recompressRank_,(int)tailHistory_.size(),recompressRank_,(int)recompressPending_);
        }else{
          // This is the first complete fixed-rank generation.
          refreshGeneration_=1;
          refreshCursor_=0;
          reset_refresh_monitor();
        }
      }
      rollbackWh_.clear();rollbackRank_=0;tentativeAdded_=0;tentative_=false;
      tentativeIsRefresh_=false;tentativeReplaceBlock_=-1;
      std::printf(
        "NODALS_BALANCED_POD_AUGMENT_DECISION outer=%d accepted=1 mode=%s generation=%d replaceBlock=%d "
        "rankOld=%d rankNew=%d baseIts=%d probeIts=%d improvement=%d required=%d "
        "acceptMinAbs=%d acceptMinRel=%.12e status=PASS\n",
        outer,refresh?"REFRESH_REPLACE":"AUGMENT",refreshGeneration_,refresh?replaced:-1,
        oldRank,candRank,baseIts,probeIts,improvement,required,acceptMinAbs_,acceptMinRel_);
    }else{
      Wh_=rollbackWh_;rank_=rollbackRank_;
      rollbackWh_.clear();rollbackRank_=0;tentativeAdded_=0;tentative_=false;
      if(refresh){
        ++refreshRejected_;
        const int nBlocks=std::max(1,rank_/std::max(1,targetRank_));
        refreshCursor_=(replaced+1)%nBlocks;
      }
      ++rejectedUpdates_;
      tentativeIsRefresh_=false;tentativeReplaceBlock_=-1;
      upload_basis();
      std::printf(
        "NODALS_BALANCED_POD_AUGMENT_DECISION outer=%d accepted=0 mode=%s generation=%d replaceBlock=%d "
        "rankOld=%d rankCandidate=%d rankNew=%d baseIts=%d probeIts=%d probeOK=%d improvement=%d "
        "required=%d acceptMinAbs=%d acceptMinRel=%.12e reason=INSUFFICIENT_PCG_BENEFIT nextReplaceCursor=%d status=PASS\n",
        outer,refresh?"REFRESH_REPLACE":"AUGMENT",refreshGeneration_,refresh?replaced:-1,
        oldRank,candRank,rank_,baseIts,probeIts,(int)probeOK,probeOK?baseIts-probeIts:-1,
        required,acceptMinAbs_,acceptMinRel_,refreshCursor_);
    }
  }

  void print_summary(long long totalIts,int outer,double loopMs)const{
    if(!enabled_)return;
    std::printf(
      "NODALS_BALANCED_POD_SUMMARY trainSnapshots=%d rank=%d stockSolves=%d stockPCGIts=%lld "
      "podSolves=%d podPCGIts=%lld probeSolves=%d probePCGIts=%lld totalActualPCGIts=%lld totalWorkPCGIts=%lld "
      "outer=%d loopMs=%.3f retainedEnergy=%.12e status=PASS\n",
      seedCollected_,rank_,stockSolves_,stockIts_,podSolves_,podIts_,probeCount_,probeIts_,
      totalIts,totalIts+probeIts_,outer,loopMs,retainedEnergy_);
    if(adaptive_){
      std::printf(
        "NODALS_BALANCED_POD_ADAPTIVE_SUMMARY seedSnapshots=%d tailBatch=%d tailIter=%d hardIts=%d multiTailDepths=[%d,%d,%d] addRank=%d finalRank=%d maxRank=%d "
        "acceptedUpdates=%d rejectedUpdates=%d probeSolves=%d probePCGIts=%lld lastBatchEnergy=%.12e "
        "refreshGeneration=%d refreshAccepted=%d refreshRejected=%d refreshBestMeanPCG=%.6f "
        "refreshAbs=%d refreshRel=%.12e replaceCursor=%d "
        "recompressCount=%d recompressAccepted=%d recompressRejected=%d recompressRank=%d recompressHistory=%d "
        "AprojectRatioMin=%.12e AprojectRatioMax=%.12e "
        "source=%s orthogonalization=CURRENT_A "
        "rankPolicy=GROW_UNTIL_MAXRANK postMaxPolicy=CONFIGURED_RECOMPRESS_OR_REFRESH acceptance=SAME_SYSTEM_PCG_PROBE status=PASS\n",
        seedCollected_,trainTarget_,tailIter_,hardIts_,tailIter_,3*tailIter_,6*tailIter_,targetRank_,rank_,maxRank_,
        acceptedUpdates_,rejectedUpdates_,probeCount_,probeIts_,
        lastBatchEnergy_,refreshGeneration_,refreshAccepted_,refreshRejected_,
        std::isfinite(refreshBestMean_)?refreshBestMean_:0.0,refreshAbs_,refreshRel_,refreshCursor_,
        recompressCount_,recompressAccepted_,recompressRejected_,recompressRank_,recompressHistory_,
        std::isfinite(aProjectMinRatio_)?aProjectMinRatio_:0.0,aProjectMaxRatio_,
        hardIts_>0?"SINGLE_HARD_SOLVE_MULTI_TAIL_XFINAL_MINUS_XK":"KRYLOV_TAIL_CORRECTION_XFINAL_MINUS_XTAIL");
    }
  }
};

#endif // NODALS_BALANCED_POD_AMG_V1
