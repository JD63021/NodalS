struct State {
  std::unique_ptr<::nodals::omega_dg_gpu::CoupledOmegaState> dgOmega;
  Dev<Real> k,omega,oldK,oldW,rhs;
  Dev<double> stats,wallStats;
  Dev<unsigned long long> clipK,clipW,diagAudit;
  Dev<unsigned char> q1Fixed;
  Dev<BF3BernsteinFace> bernsteinFaces;
  Dev<unsigned long long> bernsteinCounters;
  bool bernsteinReady=false;
  int nvert=0,ntotal=0;
  double kIn=0,omegaIn=0,prevMeanNut=-1;
  unsigned long long totalKClips=0,totalOmegaClips=0;
  State(int n,int nvertIn,double bulk,double radius,const hxt5c::Options&O){
    nvert=nvertIn;ntotal=n;
    const double D=2*radius,Lt=O.lengthRatio*D;
    kIn=1.5*std::pow(O.intensity*bulk,2);
    omegaIn=std::sqrt(kIn)/(std::pow(hxt5c::C_MU,0.25)*Lt);
    std::vector<Real> hk((std::size_t)n,Real(0)),hw((std::size_t)n,Real(0));
    for(int i=0;i<nvert;++i){hk[(std::size_t)i]=(Real)kIn;hw[(std::size_t)i]=(Real)omegaIn;}
    k.upload(hk);omega.upload(hw);oldK.upload(hk);oldW.upload(hw);
    rhs.upload(std::vector<Real>((std::size_t)n,Real(0)));
    stats.upload(std::vector<double>(7,0));wallStats.upload(std::vector<double>(33,0));
    clipK.upload(std::vector<unsigned long long>(1,0));clipW.upload(std::vector<unsigned long long>(1,0));diagAudit.upload(std::vector<unsigned long long>(2,0));
    std::vector<unsigned char> qmask((std::size_t)n,0);
    for(int i=nvert;i<n;++i)qmask[(std::size_t)i]=1;
    q1Fixed.upload(qmask);
    bernsteinCounters.upload(std::vector<unsigned long long>(6,0ULL));
    hxt5c::sync_sst_state_space_mode();
    hxt5c::sync_sst_deviatoric_production(O.deviatoricProduction);
  }
};

