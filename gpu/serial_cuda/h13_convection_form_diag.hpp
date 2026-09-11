#pragma once
#include <array>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <string>

namespace nodals_gpu {

// Gate 5J: evaluate convection forms on the FINAL converged FE field.
//
// Production operator:
//   C_adv,z = u . grad(Uz)
//
// Same-field conservative strong form:
//   C_cons,z = div(u Uz)
//            = u . grad(Uz) + Uz div(u)
//
// The weakAdv quantity below is the actual production central-convection
// element action, summed over the four P1 test functions. Since those four
// functions form a partition of unity, this is the constant-test action of
// the discrete advective operator. qAdv is the same quantity independently
// reconstructed with the 125-point tet quadrature already used by NodalS.
// Their agreement is a diagnostic parity gate.

inline double h13_coeff(
    const G4SetupHost&S,
    const std::array<std::vector<double>,3>&U,
    const G4CellPlanHost&cp,int a,int d)
{
  const int r=cp.ref[a];
  if(r>=0)return U[(std::size_t)d][(std::size_t)r];
  return S.fixedValue[(std::size_t)(-r-1)][(std::size_t)d];
}

inline double h13_gl(const G4CellPlanHost&cp,int i,int d)
{
  if(i==0)return -(cp.invJ[d]+cp.invJ[3+d]+cp.invJ[6+d]);
  return cp.invJ[3*(i-1)+d];
}

struct H13Agg {
  double vol=0.0;
  double weakAdv=0.0;
  double qAdv=0.0;
  double qCons=0.0;
  double qUzDiv=0.0;
  double qDiv=0.0;
  double qDiv2=0.0;
  long long cells=0;
};

inline H13Agg& h13_add(H13Agg&a,const H13Agg&b)
{
  a.vol+=b.vol;a.weakAdv+=b.weakAdv;a.qAdv+=b.qAdv;a.qCons+=b.qCons;
  a.qUzDiv+=b.qUzDiv;a.qDiv+=b.qDiv;a.qDiv2+=b.qDiv2;a.cells+=b.cells;
  return a;
}

inline void h13_cell_eval(
    const G4SetupHost&S,const G4CellPlanHost&cp,
    const std::array<std::vector<double>,3>&U,H13Agg&o)
{
  double cf[3][8];
  for(int d=0;d<3;++d)
    for(int a=0;a<8;++a)
      cf[d][a]=h13_coeff(S,U,cp,a,d);

  // Actual production weak-form advective constant-test action.
  const auto&T=central_tensor_g4();
  double ur[8][3]={{0}};
  for(int m=0;m<8;++m)
    for(int j=0;j<3;++j)
      for(int d=0;d<3;++d)
        ur[m][j]+=cf[d][m]*cp.invJ[3*j+d];

  double weak=0.0;
  for(int a=0;a<4;++a){ // sum P1 tests => constant test 1
    for(int b=0;b<8;++b){
      double cv=0.0;
      for(int m=0;m<8;++m)
        for(int j=0;j<3;++j)
          cv+=ur[m][j]*T.t[a][m][b][j];
      weak += cp.det*cv*cf[2][b];
    }
  }
  o.weakAdv=weak;

  // Independent strong-form quadrature.
  const auto Q=tet_duffy5_g3();
  for(const auto&q:Q){
    double phi[8],gr[8][3];
    for(int i=0;i<4;++i){
      phi[i]=q.lam[i];
      for(int d=0;d<3;++d)gr[i][d]=h13_gl(cp,i,d);
    }
    for(int i=0;i<4;++i){
      int js[3],kk=0;
      for(int j=0;j<4;++j)if(j!=i)js[kk++]=j;
      phi[4+i]=27.0*q.lam[js[0]]*q.lam[js[1]]*q.lam[js[2]];
      for(int d=0;d<3;++d){
        gr[4+i][d]=27.0*(
          q.lam[js[1]]*q.lam[js[2]]*h13_gl(cp,js[0],d)+
          q.lam[js[0]]*q.lam[js[2]]*h13_gl(cp,js[1],d)+
          q.lam[js[0]]*q.lam[js[1]]*h13_gl(cp,js[2],d));
      }
    }

    double uq[3]={0,0,0};
    double gu[3][3]={{0}};
    for(int a=0;a<8;++a){
      for(int c=0;c<3;++c){
        uq[c]+=cf[c][a]*phi[a];
        for(int d=0;d<3;++d)gu[c][d]+=cf[c][a]*gr[a][d];
      }
    }

    const double adv=uq[0]*gu[2][0]+uq[1]*gu[2][1]+uq[2]*gu[2][2];
    const double div=gu[0][0]+gu[1][1]+gu[2][2];
    const double uzdiv=uq[2]*div;
    const double cons=adv+uzdiv;
    const double w=q.w*cp.det;

    o.vol+=w;
    o.qAdv+=w*adv;
    o.qUzDiv+=w*uzdiv;
    o.qCons+=w*cons;
    o.qDiv+=w*div;
    o.qDiv2+=w*div*div;
  }
  o.cells=1;
}

inline double h13_f_from_integral(
    double integ,double vol,double D,double Ubulk)
{
  return vol>0.0 ? 2.0*D*(integ/vol)/(Ubulk*Ubulk) : 0.0;
}

inline void h13_print_window(
    const char*tag,double z0,double z1,const H13Agg&a,
    double D,double Ubulk)
{
  const double fw=h13_f_from_integral(a.weakAdv,a.vol,D,Ubulk);
  const double fa=h13_f_from_integral(a.qAdv,a.vol,D,Ubulk);
  const double fc=h13_f_from_integral(a.qCons,a.vol,D,Ubulk);
  const double fd=h13_f_from_integral(a.qUzDiv,a.vol,D,Ubulk);
  const double parity=std::abs(a.weakAdv-a.qAdv)/
    std::max({std::abs(a.weakAdv),std::abs(a.qAdv),1e-300});
  const double divMean=a.vol>0?a.qDiv/a.vol:0.0;
  const double divRms=a.vol>0?std::sqrt(std::max(0.0,a.qDiv2/a.vol)):0.0;
  std::printf(
    "NODALS_CONVECTION_FORM_WINDOW tag=%s z0D=%.2f z1D=%.2f cells=%lld "
    "fAdvWeak=%.10f fAdvStrong=%.10f fConservativeStrong=%.10f "
    "fUzDivU=%.10f weakStrongParityRel=%.12e divMean=%.12e divRms=%.12e "
    "identityErr=%.12e status=%s\n",
    tag,z0,z1,a.cells,fw,fa,fc,fd,parity,divMean,divRms,
    std::abs((fc-fa)-fd),
    (std::isfinite(fw)&&std::isfinite(fc)&&parity<1e-9)?"PASS":"FAIL");
}

inline void h13_write_convection_form_diag(
    const std::string&path,const char*tag,
    const SerialTetMesh&M,const G4SetupHost&S,
    const std::array<std::vector<double>,3>&U)
{
  if(path.empty())return;
  if(S.cells.size()!=M.tets.size())
    throw std::runtime_error("Gate5J cell-plan/mesh size mismatch");

  const double D=2.0*S.pipe.R,Ubulk=S.pipe.bulk;
  double slabD=0.2;
  if(const char*e=std::getenv("NODALS_CONVECTION_FORM_SLAB_D")){
    const double v=std::atof(e);if(v>0)slabD=v;
  }
  const int nslab=std::max(1,(int)std::ceil(S.pipe.L/(D*slabD)-1e-12));
  std::vector<H13Agg> slab((std::size_t)nslab);

  for(std::size_t c=0;c<M.tets.size();++c){
    H13Agg x;
    h13_cell_eval(S,S.cells[c],U,x);

    double zc=0.0;
    for(int j=0;j<4;++j)
      zc+=M.points[(std::size_t)M.tets[c][j]].z;
    zc*=0.25;
    int k=(int)std::floor((zc/D)/slabD);
    k=std::max(0,std::min(nslab-1,k));
    h13_add(slab[(std::size_t)k],x);
  }

  std::ofstream os(path);
  if(!os)throw std::runtime_error("Gate5J cannot open convection diagnostic CSV: "+path);
  os<<std::setprecision(16);
  os<<"slab,z0_over_D,z1_over_D,cells,volume,"
       "f_adv_weak,f_adv_strong,f_conservative_strong,f_uz_div_u,"
       "weak_strong_parity_rel,div_mean,div_rms\n";

  H13Agg total;
  for(int k=0;k<nslab;++k){
    const auto&a=slab[(std::size_t)k];
    const double fw=h13_f_from_integral(a.weakAdv,a.vol,D,Ubulk);
    const double fa=h13_f_from_integral(a.qAdv,a.vol,D,Ubulk);
    const double fc=h13_f_from_integral(a.qCons,a.vol,D,Ubulk);
    const double fd=h13_f_from_integral(a.qUzDiv,a.vol,D,Ubulk);
    const double parity=std::abs(a.weakAdv-a.qAdv)/
      std::max({std::abs(a.weakAdv),std::abs(a.qAdv),1e-300});
    const double dm=a.vol>0?a.qDiv/a.vol:0.0;
    const double dr=a.vol>0?std::sqrt(std::max(0.0,a.qDiv2/a.vol)):0.0;
    os<<k<<","<<k*slabD<<","<<(k+1)*slabD<<","<<a.cells<<","<<a.vol<<","
      <<fw<<","<<fa<<","<<fc<<","<<fd<<","<<parity<<","<<dm<<","<<dr<<"\n";
    h13_add(total,a);
  }

  auto window=[&](double z0,double z1){
    H13Agg a;
    for(int k=0;k<nslab;++k){
      const double c=(k+0.5)*slabD;
      if(c>=z0 && c<z1)h13_add(a,slab[(std::size_t)k]);
    }
    h13_print_window(tag,z0,z1,a,D,Ubulk);
  };

  window(12.0,14.0);
  window(14.0,16.0);
  window(16.0,18.0);
  window(18.0,19.8);
  window(12.0,18.0);

  const double pAll=std::abs(total.weakAdv-total.qAdv)/
    std::max({std::abs(total.weakAdv),std::abs(total.qAdv),1e-300});
  std::printf(
    "NODALS_CONVECTION_FORM_OUTPUT tag=%s csv=%s slabs=%d slabD=%.8g "
    "allWeakStrongParityRel=%.12e interpretation="
    "fAdvWeak_is_actual_discrete_advective_constant_test_action;"
    "fConservativeStrong=fAdvStrong+fUzDivU status=%s\n",
    tag,path.c_str(),nslab,slabD,pAll,pAll<1e-9?"PASS":"FAIL");
}

} // namespace nodals_gpu
