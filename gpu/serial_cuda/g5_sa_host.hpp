#pragma once
#include "g4_host.hpp"
#include "g2_host.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nodals_gpu {

struct SATransferHost {
  int nFine=0,nCoarse=0;
  std::vector<std::int64_t> row;
  std::vector<std::int32_t> col;
  std::vector<double> val;
};

struct SAIncidenceHost {
  std::vector<std::int64_t> row;
  std::vector<std::int32_t> cell;
  std::vector<std::uint8_t> basis;
};

struct SAHierarchyHost {
  std::vector<AggResult> agg;          // one aggregate map per transfer
  std::vector<SATransferHost> P;       // fine->A0, A0->A1, ...
  std::vector<CSRHost> csr;            // explicit levels; last is terminal
  std::vector<double> fineDiag;
  double fineLambda=0.0;
  std::vector<double> levelLambda;     // 0 for terminal
  std::vector<double> terminal_inv;
  int terminal_n=0;
  int powerIts=16;
  int interpMaxNnz=8;
  double lambdaSafety=1.5;
  double lambdaLowFraction=0.05;
  double saDamping=4.0/3.0;
};

inline double sa_elapsed_s(const std::chrono::steady_clock::time_point&a,
                           const std::chrono::steady_clock::time_point&b){
  return std::chrono::duration<double>(b-a).count();
}

inline SAIncidenceHost build_sa_incidence(const PressureSetupHost&S){
  SAIncidenceHost I;
  I.row.assign((std::size_t)S.free_vel+1,0);
  for(std::size_t c=0;c<S.cells.size();++c)
    for(int a=0;a<8;++a){
      int g=S.cells[c].vel[a];
      if(g>=0) ++I.row[(std::size_t)g+1];
    }
  for(int g=0;g<S.free_vel;++g) I.row[(std::size_t)g+1]+=I.row[(std::size_t)g];
  I.cell.resize((std::size_t)I.row.back());
  I.basis.resize((std::size_t)I.row.back());
  auto next=I.row;
  for(std::size_t c=0;c<S.cells.size();++c)
    for(int a=0;a<8;++a){
      int g=S.cells[c].vel[a];
      if(g<0) continue;
      std::size_t k=(std::size_t)next[(std::size_t)g]++;
      I.cell[k]=(std::int32_t)c;
      I.basis[k]=(std::uint8_t)a;
    }
  return I;
}

inline void sa_normalize_prune(std::vector<std::pair<int,double>>&row,int own,int maxNnz){
  if(row.empty()) row.push_back({own,1.0});
  std::sort(row.begin(),row.end(),[](const auto&a,const auto&b){return a.first<b.first;});
  std::size_t w=0;
  for(std::size_t k=0;k<row.size();){
    int c=row[k].first; double v=0.0; std::size_t j=k;
    for(;j<row.size()&&row[j].first==c;++j) v+=row[j].second;
    row[w++]={c,v}; k=j;
  }
  row.resize(w);
  if(maxNnz>0 && (int)row.size()>maxNnz){
    std::vector<std::pair<int,double>> keep;keep.reserve((std::size_t)maxNnz);
    auto it=std::find_if(row.begin(),row.end(),[&](const auto&e){return e.first==own;});
    if(it!=row.end())keep.push_back(*it);else keep.push_back({own,0.0});
    std::vector<std::pair<int,double>> others;others.reserve(row.size());
    for(const auto&e:row)if(e.first!=own)others.push_back(e);
    std::sort(others.begin(),others.end(),[](const auto&a,const auto&b){
      double aa=std::abs(a.second),bb=std::abs(b.second);
      return aa>bb || (aa==bb && a.first<b.first);
    });
    for(const auto&e:others)if((int)keep.size()<maxNnz)keep.push_back(e);
    std::sort(keep.begin(),keep.end(),[](const auto&a,const auto&b){return a.first<b.first;});
    row.swap(keep);
  }
  double sum=0.0;for(const auto&e:row)sum+=e.second;
  if(!std::isfinite(sum)||std::abs(sum)<1e-14){row.clear();row.push_back({own,1.0});return;}
  for(auto&e:row)e.second/=sum;
}

inline void sa_fine_apply_reuse(const PressureSetupHost&S,const std::vector<double>&x,
                                std::vector<double>&y,std::vector<double>&v0,
                                std::vector<double>&v1,std::vector<double>&v2){
  if(x.size()!=S.cells.size())throw std::runtime_error("SA fine apply x size");
  std::fill(v0.begin(),v0.end(),0.0);std::fill(v1.begin(),v1.end(),0.0);std::fill(v2.begin(),v2.end(),0.0);
  for(std::size_t c=0;c<S.cells.size();++c){
    const auto&cp=S.cells[c];const double p=x[c];
    for(int a=0;a<8;++a){int g=cp.vel[a];if(g<0)continue;
      v0[(std::size_t)g]+=coeff(cp,a,0)*p;
      v1[(std::size_t)g]+=coeff(cp,a,1)*p;
      v2[(std::size_t)g]+=coeff(cp,a,2)*p;
    }
  }
  for(int g=0;g<S.free_vel;++g){double r=S.rAU[(std::size_t)g];v0[(std::size_t)g]*=r;v1[(std::size_t)g]*=r;v2[(std::size_t)g]*=r;}
  y.assign(S.cells.size(),0.0);
  for(std::size_t c=0;c<S.cells.size();++c){
    const auto&cp=S.cells[c];double s=0.0;
    for(int a=0;a<8;++a){int g=cp.vel[a];if(g<0)continue;
      s+=coeff(cp,a,0)*v0[(std::size_t)g]+coeff(cp,a,1)*v1[(std::size_t)g]+coeff(cp,a,2)*v2[(std::size_t)g];
    }
    y[c]=s;
  }
}

inline double sa_power_fine(const PressureSetupHost&S,const std::vector<double>&diag,int its,double safety){
  const int n=(int)S.cells.size();
  std::vector<double> v((std::size_t)n),u((std::size_t)n),au,w;
  std::vector<double> t0((std::size_t)S.free_vel),t1((std::size_t)S.free_vel),t2((std::size_t)S.free_vel);
  long double ss=0.0L;
  for(int i=0;i<n;++i){double g=(double)(i+1);v[(std::size_t)i]=std::sin(.731*g)+.27*std::cos(1.117*g);ss+=(long double)v[(std::size_t)i]*v[(std::size_t)i];}
  double vn=std::sqrt((double)ss);for(auto&q:v)q/=vn;
  double ray=0.0;
  for(int it=0;it<its;++it){
    for(int i=0;i<n;++i)u[(std::size_t)i]=v[(std::size_t)i]/std::sqrt(diag[(std::size_t)i]);
    sa_fine_apply_reuse(S,u,au,t0,t1,t2);
    w.resize((std::size_t)n);long double num=0.0L,den=0.0L,nw=0.0L;
    for(int i=0;i<n;++i){w[(std::size_t)i]=au[(std::size_t)i]/std::sqrt(diag[(std::size_t)i]);num+=(long double)v[(std::size_t)i]*w[(std::size_t)i];den+=(long double)v[(std::size_t)i]*v[(std::size_t)i];nw+=(long double)w[(std::size_t)i]*w[(std::size_t)i];}
    ray=(double)(num/std::max(den,(long double)1e-300));double nrm=std::sqrt((double)nw);
    if(!(nrm>0.0)||!std::isfinite(nrm)||!(ray>0.0)||!std::isfinite(ray))throw std::runtime_error("SA fine power iteration failed");
    for(int i=0;i<n;++i)v[(std::size_t)i]=w[(std::size_t)i]/nrm;
  }
  return safety*ray;
}

inline void sa_csr_apply(const CSRHost&A,const std::vector<double>&x,std::vector<double>&y){
  y.assign((std::size_t)A.n,0.0);
  for(int i=0;i<A.n;++i){double s=0.0;for(std::int64_t k=A.row[(std::size_t)i];k<A.row[(std::size_t)i+1];++k)s+=A.val[(std::size_t)k]*x[(std::size_t)A.col[(std::size_t)k]];y[(std::size_t)i]=s;}
}
inline double sa_power_csr(const CSRHost&A,int its,double safety){
  std::vector<double>v((std::size_t)A.n),u((std::size_t)A.n),au,w((std::size_t)A.n);
  long double ss=0;for(int i=0;i<A.n;++i){double g=(double)(i+1);v[(std::size_t)i]=std::sin(.731*g)+.27*std::cos(1.117*g);ss+=(long double)v[(std::size_t)i]*v[(std::size_t)i];}
  double vn=std::sqrt((double)ss);for(auto&q:v)q/=vn;double ray=0.0;
  for(int it=0;it<its;++it){
    for(int i=0;i<A.n;++i)u[(std::size_t)i]=v[(std::size_t)i]/std::sqrt(A.diag[(std::size_t)i]);
    sa_csr_apply(A,u,au);long double num=0,den=0,nw=0;
    for(int i=0;i<A.n;++i){w[(std::size_t)i]=au[(std::size_t)i]/std::sqrt(A.diag[(std::size_t)i]);num+=(long double)v[(std::size_t)i]*w[(std::size_t)i];den+=(long double)v[(std::size_t)i]*v[(std::size_t)i];nw+=(long double)w[(std::size_t)i]*w[(std::size_t)i];}
    ray=(double)(num/std::max(den,(long double)1e-300));double nrm=std::sqrt((double)nw);
    if(!(nrm>0.0)||!std::isfinite(nrm)||!(ray>0.0)||!std::isfinite(ray))throw std::runtime_error("SA coarse power iteration failed");
    for(int i=0;i<A.n;++i)v[(std::size_t)i]=w[(std::size_t)i]/nrm;
  }
  return safety*ray;
}

inline SATransferHost build_sa_fine_transfer(const PressureSetupHost&S,const SAIncidenceHost&I,
                                              const AggResult&G,const std::vector<double>&diag,
                                              double lambdaMax,double damping,int maxNnz){
  const double omega=damping/lambdaMax;SATransferHost P;P.nFine=(int)S.cells.size();P.nCoarse=G.nagg;P.row.assign((std::size_t)P.nFine+1,0);
  P.col.reserve((std::size_t)P.nFine*3);P.val.reserve((std::size_t)P.nFine*3);
  std::vector<std::pair<int,double>> row;row.reserve(96);int maxrow=0;long double defect=0.0L;
  for(int i=0;i<P.nFine;++i){
    row.clear();const int own=G.id[(std::size_t)i];row.push_back({own,1.0});const auto&ci=S.cells[(std::size_t)i];const double invd=1.0/diag[(std::size_t)i];
    for(int a=0;a<8;++a){int g=ci.vel[a];if(g<0)continue;
      const double bi[3]={coeff(ci,a,0),coeff(ci,a,1),coeff(ci,a,2)};
      for(std::int64_t k=I.row[(std::size_t)g];k<I.row[(std::size_t)g+1];++k){
        int j=I.cell[(std::size_t)k],aj=(int)I.basis[(std::size_t)k];const auto&cj=S.cells[(std::size_t)j];
        double dot=0.0;for(int d=0;d<3;++d)dot+=bi[d]*coeff(cj,aj,d);
        row.push_back({G.id[(std::size_t)j],-omega*invd*S.rAU[(std::size_t)g]*dot});
      }
    }
    sa_normalize_prune(row,own,maxNnz);double sum=0.0;for(const auto&e:row){P.col.push_back(e.first);P.val.push_back(e.second);sum+=e.second;}P.row[(std::size_t)i+1]=(std::int64_t)P.col.size();maxrow=std::max(maxrow,(int)row.size());defect=std::max(defect,(long double)std::abs(sum-1.0));
  }
  std::printf("NODALS_GPU_SA_HOST_TRANSFER level=0 rows=%d coarse=%d nnz=%zu avgNnz=%.6f maxRowNnz=%d rowSumDefect=%.3Le omega=%.12e lambdaMax=%.12e status=PASS\n",
              P.nFine,P.nCoarse,P.val.size(),P.nFine?(double)P.val.size()/P.nFine:0.0,maxrow,defect,omega,lambdaMax);
  return P;
}

inline SATransferHost build_sa_explicit_transfer(const CSRHost&A,const AggResult&G,double lambdaMax,
                                                  double damping,int maxNnz,int level){
  const double omega=damping/lambdaMax;SATransferHost P;P.nFine=A.n;P.nCoarse=G.nagg;P.row.assign((std::size_t)A.n+1,0);P.col.reserve(A.n*3ull);P.val.reserve(A.n*3ull);
  std::vector<std::pair<int,double>>row;row.reserve(64);int maxrow=0;long double defect=0.0L;
  for(int i=0;i<A.n;++i){row.clear();int own=G.id[(std::size_t)i];row.push_back({own,1.0});double invd=1.0/A.diag[(std::size_t)i];
    for(std::int64_t k=A.row[(std::size_t)i];k<A.row[(std::size_t)i+1];++k)row.push_back({G.id[(std::size_t)A.col[(std::size_t)k]],-omega*invd*A.val[(std::size_t)k]});
    sa_normalize_prune(row,own,maxNnz);double sum=0.0;for(const auto&e:row){P.col.push_back(e.first);P.val.push_back(e.second);sum+=e.second;}P.row[(std::size_t)i+1]=(std::int64_t)P.col.size();maxrow=std::max(maxrow,(int)row.size());defect=std::max(defect,(long double)std::abs(sum-1.0));
  }
  std::printf("NODALS_GPU_SA_HOST_TRANSFER level=%d rows=%d coarse=%d nnz=%zu avgNnz=%.6f maxRowNnz=%d rowSumDefect=%.3Le omega=%.12e lambdaMax=%.12e status=PASS\n",
              level,P.nFine,P.nCoarse,P.val.size(),P.nFine?(double)P.val.size()/P.nFine:0.0,maxrow,defect,omega,lambdaMax);
  return P;
}

inline CSRHost sa_unordered_rows_to_csr(std::vector<std::unordered_map<int,double>>&rows){
  CSRHost A;A.n=(int)rows.size();A.row.assign((std::size_t)A.n+1,0);
  for(int i=0;i<A.n;++i)A.row[(std::size_t)i+1]=A.row[(std::size_t)i]+(std::int64_t)rows[(std::size_t)i].size();
  A.col.resize((std::size_t)A.row.back());A.val.resize((std::size_t)A.row.back());A.diag.assign((std::size_t)A.n,0.0);
  std::vector<std::pair<int,double>>tmp;
  for(int i=0;i<A.n;++i){tmp.clear();tmp.reserve(rows[(std::size_t)i].size());for(const auto&kv:rows[(std::size_t)i])if(kv.second!=0.0)tmp.push_back(kv);std::sort(tmp.begin(),tmp.end(),[](const auto&a,const auto&b){return a.first<b.first;});
    std::int64_t k=A.row[(std::size_t)i];for(const auto&e:tmp){A.col[(std::size_t)k]=e.first;A.val[(std::size_t)k]=e.second;if(e.first==i)A.diag[(std::size_t)i]=e.second;++k;}
    // rows may have exact-zero merged entries; compact if this occurs.
    A.row[(std::size_t)i+1]=k;
  }
  // Compact because exact zeros may have been skipped.
  std::int64_t write=0;
  std::vector<std::int64_t> nr((std::size_t)A.n+1,0);
  for(int i=0;i<A.n;++i){std::int64_t begin=A.row[(std::size_t)i],end=A.row[(std::size_t)i+1];nr[(std::size_t)i]=write;for(std::int64_t k=begin;k<end;++k){A.col[(std::size_t)write]=A.col[(std::size_t)k];A.val[(std::size_t)write]=A.val[(std::size_t)k];++write;}}
  nr[(std::size_t)A.n]=write;A.row.swap(nr);A.col.resize((std::size_t)write);A.val.resize((std::size_t)write);
  for(int i=0;i<A.n;++i)if(!(A.diag[(std::size_t)i]>0.0)||!std::isfinite(A.diag[(std::size_t)i]))throw std::runtime_error("SA coarse matrix non-positive diagonal");
  return A;
}

struct SAQEntry {int coarse=-1;double b[3]={0.0,0.0,0.0};};

inline CSRHost build_sa_first_coarse(const PressureSetupHost&S,const SAIncidenceHost&I,const SATransferHost&P){
  std::vector<std::unordered_map<int,double>> rows((std::size_t)P.nCoarse);
  std::vector<SAQEntry>q;q.reserve(64);unsigned long long raw=0;
  for(int g=0;g<S.free_vel;++g){
    q.clear();
    for(std::int64_t k=I.row[(std::size_t)g];k<I.row[(std::size_t)g+1];++k){
      int c=I.cell[(std::size_t)k],a=(int)I.basis[(std::size_t)k];const auto&cp=S.cells[(std::size_t)c];double bb[3]={coeff(cp,a,0),coeff(cp,a,1),coeff(cp,a,2)};
      for(std::int64_t kp=P.row[(std::size_t)c];kp<P.row[(std::size_t)c+1];++kp){double pv=P.val[(std::size_t)kp];SAQEntry e;e.coarse=P.col[(std::size_t)kp];for(int d=0;d<3;++d)e.b[d]=pv*bb[d];q.push_back(e);}
    }
    std::sort(q.begin(),q.end(),[](const auto&a,const auto&b){return a.coarse<b.coarse;});
    std::size_t w=0;
    for(std::size_t k=0;k<q.size();){int c=q[k].coarse;double bsum[3]={0,0,0};std::size_t j=k;for(;j<q.size()&&q[j].coarse==c;++j)for(int d=0;d<3;++d)bsum[d]+=q[j].b[d];q[w].coarse=c;for(int d=0;d<3;++d)q[w].b[d]=bsum[d];++w;k=j;}q.resize(w);
    double r=S.rAU[(std::size_t)g];
    for(std::size_t a=0;a<q.size();++a)for(std::size_t b=a;b<q.size();++b){double v=r*(q[a].b[0]*q[b].b[0]+q[a].b[1]*q[b].b[1]+q[a].b[2]*q[b].b[2]);if(v==0.0)continue;rows[(std::size_t)q[a].coarse][q[b].coarse]+=v;if(a!=b)rows[(std::size_t)q[b].coarse][q[a].coarse]+=v;raw+=(a==b)?1:2;}
  }
  auto A=sa_unordered_rows_to_csr(rows);std::printf("NODALS_GPU_SA_HOST_COARSE level=1 rows=%d nnz=%zu avgNnz=%.6f rawTerms=%llu symmetryRel=%.3e status=PASS\n",A.n,A.val.size(),A.n?(double)A.val.size()/A.n:0.0,raw,csr_symmetry_rel(A));return A;
}

inline CSRHost build_sa_ptap(const CSRHost&A,const SATransferHost&P,int nextLevel){
  if(P.nFine!=A.n)throw std::runtime_error("SA PtAP transfer size mismatch");
  std::vector<std::unordered_map<int,double>> rows((std::size_t)P.nCoarse);
  std::vector<std::pair<int,double>>q;q.reserve(128);unsigned long long raw=0;
  for(int i=0;i<A.n;++i){
    q.clear();
    for(std::int64_t k=A.row[(std::size_t)i];k<A.row[(std::size_t)i+1];++k){
      int j=A.col[(std::size_t)k];double a=A.val[(std::size_t)k];
      for(std::int64_t kp=P.row[(std::size_t)j];kp<P.row[(std::size_t)j+1];++kp)q.push_back({P.col[(std::size_t)kp],a*P.val[(std::size_t)kp]});
    }
    std::sort(q.begin(),q.end(),[](const auto&a,const auto&b){return a.first<b.first;});std::size_t w=0;
    for(std::size_t k=0;k<q.size();){int c=q[k].first;double v=0.0;std::size_t j=k;for(;j<q.size()&&q[j].first==c;++j)v+=q[j].second;q[w++]={c,v};k=j;}q.resize(w);
    for(std::int64_t kp=P.row[(std::size_t)i];kp<P.row[(std::size_t)i+1];++kp){int I=P.col[(std::size_t)kp];double pi=P.val[(std::size_t)kp];for(const auto&e:q){double v=pi*e.second;if(v!=0.0){rows[(std::size_t)I][e.first]+=v;++raw;}}}
  }
  auto C=sa_unordered_rows_to_csr(rows);std::printf("NODALS_GPU_SA_HOST_COARSE level=%d rows=%d nnz=%zu avgNnz=%.6f rawTerms=%llu symmetryRel=%.3e status=PASS\n",nextLevel,C.n,C.val.size(),C.n?(double)C.val.size()/C.n:0.0,raw,csr_symmetry_rel(C));return C;
}

inline SAHierarchyHost build_sa_hierarchy(const SerialTetMesh&M,PressureSetupHost&S,
                                           int target=16,int minsz=6,int softmax=18,int terminal=1000,
                                           int maxPnnz=8,int powerIts=16,double safety=1.5,
                                           double lowFrac=0.05,double damping=4.0/3.0){
  SAHierarchyHost H;H.powerIts=powerIts;H.interpMaxNnz=maxPnnz;H.lambdaSafety=safety;H.lambdaLowFraction=lowFrac;H.saDamping=damping;
  auto t0=std::chrono::steady_clock::now();auto I=build_sa_incidence(S);auto t1=std::chrono::steady_clock::now();
  std::printf("NODALS_GPU_SA_HOST_STAGE stage=incidence entries=%zu seconds=%.6f status=PASS\n",I.cell.size(),sa_elapsed_s(t0,t1));

  auto G0=aggregate_graph(face_graph(M),target,minsz,softmax);H.agg.push_back(G0);
  H.fineDiag=cpu_fine_diag(S);
  auto p0=std::chrono::steady_clock::now();H.fineLambda=sa_power_fine(S,H.fineDiag,powerIts,safety);auto p1=std::chrono::steady_clock::now();
  std::printf("NODALS_GPU_SA_HOST_SPECTRUM level=0 powerIts=%d lambdaMax=%.12e seconds=%.6f status=PASS\n",powerIts,H.fineLambda,sa_elapsed_s(p0,p1));
  auto tr0=std::chrono::steady_clock::now();H.P.push_back(build_sa_fine_transfer(S,I,G0,H.fineDiag,H.fineLambda,damping,maxPnnz));auto tr1=std::chrono::steady_clock::now();
  std::printf("NODALS_GPU_SA_HOST_STAGE stage=P0 seconds=%.6f status=PASS\n",sa_elapsed_s(tr0,tr1));
  auto c0=std::chrono::steady_clock::now();H.csr.push_back(build_sa_first_coarse(S,I,H.P[0]));auto c1=std::chrono::steady_clock::now();
  std::printf("NODALS_GPU_SA_HOST_STAGE stage=A1 seconds=%.6f status=PASS\n",sa_elapsed_s(c0,c1));

  // Incidence is no longer needed once P0 and A1 are formed.
  SAIncidenceHost().row.swap(I.row);SAIncidenceHost().cell.swap(I.cell);SAIncidenceHost().basis.swap(I.basis);

  while(H.csr.back().n>terminal){
    const int lev=(int)H.csr.size(); // explicit operator index lev-1; transfer level lev
    auto &A=H.csr.back();double lam=sa_power_csr(A,powerIts,safety);
    if(H.levelLambda.size()<H.csr.size())H.levelLambda.resize(H.csr.size(),0.0);
    H.levelLambda[H.csr.size()-1]=lam;
    std::printf("NODALS_GPU_SA_HOST_SPECTRUM level=%d powerIts=%d lambdaMax=%.12e status=PASS\n",lev,powerIts,lam);
    auto G=aggregate_graph(csr_graph(A),target,minsz,softmax);H.agg.push_back(G);
    auto P=build_sa_explicit_transfer(A,G,lam,damping,maxPnnz,lev);H.P.push_back(std::move(P));
    H.csr.push_back(build_sa_ptap(A,H.P.back(),lev+1));
  }
  if(H.levelLambda.size()<H.csr.size())H.levelLambda.resize(H.csr.size(),0.0);
  H.terminal_n=H.csr.back().n;H.terminal_inv=dense_inverse_cholesky(H.csr.back());
  std::printf("NODALS_GPU_SA_HOST_HIERARCHY levels=%zu transfers=%zu terminal=%d fineLambda=%.12e interpMaxNnz=%d powerIts=%d safety=%.6f lowFraction=%.6f damping=%.6f status=PASS\n",
              H.csr.size(),H.P.size(),H.terminal_n,H.fineLambda,maxPnnz,powerIts,safety,lowFrac,damping);
  return H;
}

} // namespace nodals_gpu
