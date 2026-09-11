#pragma once
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace nodals_gpu {

inline double h11_host_yplus_from_uplus(double up,double kappa,double B){
  if(!(up>=0.0))return std::numeric_limits<double>::quiet_NaN();
  const double x=kappa*up;double rem=0.0;
  if(std::abs(x)<1e-3){const double x2=x*x,x4=x2*x2;rem=x4*(1.0/24.0+x/120.0+x2/720.0+x2*x/5040.0);}
  else rem=std::expm1(x)-x-0.5*x*x-(x*x*x)/6.0;
  return up+std::exp(-kappa*B)*rem;
}
inline double h11_host_root_g(double up,double reY,double kappa,double B){
  return up*h11_host_yplus_from_uplus(up,kappa,B)-reY;
}
inline bool h11_host_penalty(
    double slip,double y,double nu,double kappa,double B,
    double&uTau,double&yPlus,double&uPlus,double&beta)
{
  if(!(y>0.0)||!(nu>0.0)||!(kappa>0.0)||!(B>0.0)||!std::isfinite(slip))return false;
  const double U=std::abs(slip),reY=y*U/nu;
  if(reY<=1e-14){uTau=0;yPlus=0;uPlus=0;beta=nu/y;return std::isfinite(beta);}
  double lo=0.0,hi=std::max(1.0,std::sqrt(reY)+1.0);
  int expand=0;while(h11_host_root_g(hi,reY,kappa,B)<0.0&&expand<40){hi*=2.0;++expand;}
  const double ghi=h11_host_root_g(hi,reY,kappa,B);if(!std::isfinite(ghi)||ghi<0.0)return false;
  for(int it=0;it<70;++it){const double mid=0.5*(lo+hi);if(h11_host_root_g(mid,reY,kappa,B)>0.0)hi=mid;else lo=mid;}
  uPlus=0.5*(lo+hi);if(!(uPlus>0.0)||!std::isfinite(uPlus))return false;
  uTau=U/uPlus;yPlus=y*uTau/nu;beta=uTau*uTau/U;
  return std::isfinite(uTau)&&std::isfinite(yPlus)&&std::isfinite(beta)&&beta>0.0;
}

inline double h11_host_coeff(
    const G4SetupHost&S,const std::array<std::vector<double>,3>&U,int c,int a,int d)
{
  const int r=S.cells[(std::size_t)c].ref[a];
  if(r>=0)return U[(std::size_t)d][(std::size_t)r];
  return S.fixedValue[(std::size_t)(-r-1)][(std::size_t)d];
}

inline void h11_write_final_wall_csv(
    const std::string&path,const SerialTetMesh&M,const G4SetupHost&S,
    const std::array<std::vector<double>,3>&U,
    const std::string&mode,double referenceF)
{
  if(path.empty()||!S.weakWall)return;
  const auto faces=h11_build_wall_faces(M,S,0.25);
  std::ofstream os(path);
  if(!os)throw std::runtime_error("Gate5I cannot open wall diagnostic CSV: "+path);
  os<<std::setprecision(16);
  os<<"face_index,cell,opp,z_over_D,area,y,"
       "u_trace_signed,slip,"
       "u_tau_spalding,u_tau2_spalding,y_plus_spalding,beta_spalding,f_spalding,"
       "u_tau_applied,u_tau2_applied,beta_applied,f_applied,mode\n";

  const double l0[7]={1.0/3.0,0.059715871789770,0.470142064105115,0.470142064105115,0.797426985353087,0.101286507323456,0.101286507323456};
  const double l1[7]={1.0/3.0,0.470142064105115,0.059715871789770,0.470142064105115,0.101286507323456,0.797426985353087,0.101286507323456};
  const double l2[7]={1.0/3.0,0.470142064105115,0.470142064105115,0.059715871789770,0.101286507323456,0.101286507323456,0.797426985353087};
  const double qw[7]={0.225000000000000,0.132394152788506,0.132394152788506,0.132394152788506,0.125939180544827,0.125939180544827,0.125939180544827};

  const double utRef=S.pipe.bulk*std::sqrt(referenceF/8.0);
  const double tauRef=utRef*utRef;
  const double slipFloor=std::max(1e-12,1e-6*std::abs(S.pipe.bulk));

  double areaTot=0,spUt2A=0,appUt2A=0,spYpA=0,slipA=0;
  long long failures=0;

  for(std::size_t fi=0;fi<faces.size();++fi){
    const auto&F=faces[fi];const int c=F.cell,opp=(int)F.opp;
    int fv[3],kk=0;for(int i=0;i<4;++i)if(i!=opp)fv[kk++]=i;
    const int act[4]={fv[0],fv[1],fv[2],4+opp};

    double zc=0.0;
    const auto&t=M.tets[(std::size_t)c];
    for(int j=0;j<3;++j)zc+=M.points[(std::size_t)t[(std::size_t)fv[j]]].z;
    zc/=3.0;

    double uzc[4];for(int j=0;j<4;++j)uzc[j]=h11_host_coeff(S,U,c,act[j],2);
    double wsum=0,uzA=0,slA=0,utA=0,ut2A=0,ypA=0,betaA=0,appUtA=0,appUt2FaceA=0,appBetaA=0;
    for(int q=0;q<7;++q){
      const double phi[4]={l0[q],l1[q],l2[q],27.0*l0[q]*l1[q]*l2[q]};
      double uz=0;for(int j=0;j<4;++j)uz+=uzc[j]*phi[j];
      const double slip=std::abs(uz),w=qw[q]*(double)F.area;
      double ut=0,yp=0,up=0,beta=0;
      const bool ok=h11_host_penalty(slip,(double)F.y,S.pipe.nu,0.4,5.5,ut,yp,up,beta);
      if(!ok){++failures;ut=0;yp=0;beta=S.pipe.nu/(double)F.y;}
      const double appUt=(mode=="reference_shear")?utRef:ut;
      const double appUt2=appUt*appUt;
      const double appBeta=(mode=="reference_shear")?tauRef/std::max(slip,slipFloor):beta;
      wsum+=w;uzA+=uz*w;slA+=slip*w;utA+=ut*w;ut2A+=ut*ut*w;ypA+=yp*w;betaA+=beta*w;
      appUtA+=appUt*w;appUt2FaceA+=appUt2*w;appBetaA+=appBeta*w;
    }
    if(!(wsum>0.0))continue;
    const double uzm=uzA/wsum,slm=slA/wsum,utm=utA/wsum,ut2m=ut2A/wsum,ypm=ypA/wsum,betam=betaA/wsum;
    const double autm=appUtA/wsum,aut2m=appUt2FaceA/wsum,abetam=appBetaA/wsum;
    const double fsp=8.0*ut2m/(S.pipe.bulk*S.pipe.bulk);
    const double fap=8.0*aut2m/(S.pipe.bulk*S.pipe.bulk);
    os<<fi<<","<<c<<","<<opp<<","<<(zc/(2.0*S.pipe.R))<<","<<(double)F.area<<","<<(double)F.y<<","
      <<uzm<<","<<slm<<","<<utm<<","<<ut2m<<","<<ypm<<","<<betam<<","<<fsp<<","
      <<autm<<","<<aut2m<<","<<abetam<<","<<fap<<","<<mode<<"\n";
    areaTot+=(double)F.area;spUt2A+=ut2m*(double)F.area;appUt2A+=aut2m*(double)F.area;spYpA+=ypm*(double)F.area;slipA+=slm*(double)F.area;
  }
  const double fsp=areaTot>0?8.0*(spUt2A/areaTot)/(S.pipe.bulk*S.pipe.bulk):0.0;
  const double fap=areaTot>0?8.0*(appUt2A/areaTot)/(S.pipe.bulk*S.pipe.bulk):0.0;
  std::printf("NODALS_GPU_RANS_GATE5I_WALL_DIAG path=%s mode=%s faces=%zu area=%.12e fSpaldingFromFinalTrace=%.10f fApplied=%.10f referenceF=%.10f yPlusSpaldingMean=%.12e slipMean=%.12e rootFailures=%lld status=%s\n",
    path.c_str(),mode.c_str(),faces.size(),areaTot,fsp,fap,referenceF,
    areaTot>0?spYpA/areaTot:0.0,areaTot>0?slipA/areaTot:0.0,failures,failures==0?"PASS":"FAIL");
}

} // namespace nodals_gpu
