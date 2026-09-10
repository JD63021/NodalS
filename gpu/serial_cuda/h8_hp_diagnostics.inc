struct H8HpErrors {
  double hEff=0.0;
  double uL2=0.0,uRelL2=0.0;
  double pShiftedL2=0.0,pShiftedRelL2=0.0;
  double pressureShift=0.0;
};

static inline double h8_final_velocity_coeff(
    const G4SetupHost&S,const G4CellPlanHost&cp,
    const std::array<std::vector<double>,3>&U,int a,int d)
{
  const int r=cp.ref[a];
  if(r>=0)return U[(std::size_t)d][(std::size_t)r];
  return S.fixedValue[(std::size_t)fixed_slot_from_ref_g4(r)][(std::size_t)d];
}

static H8HpErrors h8_compute_hp_errors(
    const SerialTetMesh&M,const G4SetupHost&S,
    const std::array<std::vector<double>,3>&U,
    const std::vector<double>&p)
{
  if(p.size()!=M.tets.size())
    throw std::runtime_error("H8 HP diagnostics pressure size mismatch");

  const auto Q=tet_duffy5_g3();
  long double total=0.0L,shiftIntegral=0.0L;

  for(std::size_t c=0;c<M.tets.size();++c){
    const auto&t=M.tets[c];
    const Vec3d X[4]={M.points[(std::size_t)t[0]],M.points[(std::size_t)t[1]],
                      M.points[(std::size_t)t[2]],M.points[(std::size_t)t[3]]};
    const double det=S.cells[c].det;
    for(const auto&q:Q){
      double z=0.0;
      for(int i=0;i<4;++i)z+=q.lam[i]*X[i].z;
      const double pExact=S.pipe.hpGradient*(S.pipe.zOut-z);
      const double w=q.w*det;
      total+=(long double)w;
      shiftIntegral+=(long double)(p[c]-pExact)*(long double)w;
    }
  }

  const double shift=(double)(shiftIntegral/std::max(total,(long double)1e-300L));

  long double ue2=0.0L,un2=0.0L,pe2=0.0L,pn2=0.0L;
  for(std::size_t c=0;c<M.tets.size();++c){
    const auto&t=M.tets[c];
    const Vec3d X[4]={M.points[(std::size_t)t[0]],M.points[(std::size_t)t[1]],
                      M.points[(std::size_t)t[2]],M.points[(std::size_t)t[3]]};
    const auto&cp=S.cells[c];
    double coeff[3][8]={{0}};
    for(int a=0;a<8;++a)for(int d=0;d<3;++d)
      coeff[d][a]=h8_final_velocity_coeff(S,cp,U,a,d);

    for(const auto&q:Q){
      double phi[8]={q.lam[0],q.lam[1],q.lam[2],q.lam[3],0,0,0,0};
      for(int i=0;i<4;++i){
        int js[3],kk=0;
        for(int j=0;j<4;++j)if(j!=i)js[kk++]=j;
        phi[4+i]=27.0*q.lam[js[0]]*q.lam[js[1]]*q.lam[js[2]];
      }

      double x=0.0,y=0.0,z=0.0;
      for(int i=0;i<4;++i){
        x+=q.lam[i]*X[i].x;
        y+=q.lam[i]*X[i].y;
        z+=q.lam[i]*X[i].z;
      }

      double uh[3]={0,0,0};
      for(int d=0;d<3;++d)for(int a=0;a<8;++a)uh[d]+=coeff[d][a]*phi[a];

      const double uzExact=pipe_ideal_uz_g4(S.pipe,x,y);
      const double pExact=S.pipe.hpGradient*(S.pipe.zOut-z);
      const double w=q.w*cp.det;

      const double du0=uh[0],du1=uh[1],du2=uh[2]-uzExact;
      ue2+=(long double)(du0*du0+du1*du1+du2*du2)*(long double)w;
      un2+=(long double)(uzExact*uzExact)*(long double)w;

      const double dp=p[c]-shift-pExact;
      pe2+=(long double)(dp*dp)*(long double)w;
      pn2+=(long double)(pExact*pExact)*(long double)w;
    }
  }

  H8HpErrors E;
  const double idealVol=3.141592653589793238462643383279502884*
                        S.pipe.R*S.pipe.R*S.pipe.L;
  const std::size_t ncSafe=M.tets.empty()?std::size_t(1):M.tets.size();
  E.hEff=std::cbrt(idealVol/(double)ncSafe);
  E.uL2=std::sqrt((double)ue2);
  E.uRelL2=std::sqrt((double)(ue2/std::max(un2,(long double)1e-300L)));
  E.pShiftedL2=std::sqrt((double)pe2);
  E.pShiftedRelL2=std::sqrt((double)(pe2/std::max(pn2,(long double)1e-300L)));
  E.pressureShift=shift;
  return E;
}
