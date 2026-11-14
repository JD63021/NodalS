#include "m3_predictor.hpp"
#include <mpi.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include "spalding_core.hpp"
#include "../transport/sst_transport.hpp"
#include "../enrichment/enrichment_momentum.hpp"
#include "../enrichment/grad_div_rank_one.hpp"
#include "../enrichment/constraint_augmentation.hpp"
#include "../pressure/discrete_constraint_row.hpp"
namespace nodals::cpu_mc::m3 {
namespace {
const double Q0[12]={.063089014491502,.063089014491502,.873821971016996,.249286745170910,.249286745170910,.501426509658180,.053145049844816,.053145049844816,.310352451033785,.310352451033785,.636502499121399,.636502499121399};
const double Q1[12]={.063089014491502,.873821971016996,.063089014491502,.249286745170910,.501426509658180,.249286745170910,.310352451033785,.636502499121399,.053145049844816,.636502499121399,.053145049844816,.310352451033785};
const double Q2[12]={.873821971016996,.063089014491502,.063089014491502,.501426509658180,.249286745170910,.249286745170910,.636502499121399,.310352451033785,.636502499121399,.053145049844816,.310352451033785,.053145049844816};
const double QW[12]={.050844906370207,.050844906370207,.050844906370207,.116786275726379,.116786275726379,.116786275726379,.082851075618374,.082851075618374,.082851075618374,.082851075618374,.082851075618374,.082851075618374};
const double X5[5]={.0469100770306680185,.23076534494715845,.5,.7692346550528415,.95308992296933193};
const double W5[5]={.11846344252809449,.23931433524968326,.28444444444444450,.23931433524968326,.11846344252809449};
using G=std::array<Vec3,8>;
struct Geom {std::array<Vec3,4> gl;double det;};
Geom geometry(const Mesh&m,int c){const auto &t=m.tets[c];const Vec3 a=m.points[t.v[0]],e1=m.points[t.v[1]]-a,e2=m.points[t.v[2]]-a,e3=m.points[t.v[3]]-a;const double det=dot(e1,cross(e2,e3));if(!(det>0))throw std::runtime_error("tet orientation invalid");Geom x{};x.det=det;x.gl[1]=(1.0/det)*cross(e2,e3);x.gl[2]=(1.0/det)*cross(e3,e1);x.gl[3]=(1.0/det)*cross(e1,e2);x.gl[0]=-1.0*(x.gl[1]+x.gl[2]+x.gl[3]);return x;}
void shape(const Geom&g,const std::array<double,4>&l,double val[8],G&gr){
 for(int i=0;i<4;++i){val[i]=l[i];gr[i]=g.gl[i];}
 for(int i=0;i<4;++i){int js[3],n=0;for(int j=0;j<4;++j)if(j!=i)js[n++]=j;
  val[4+i]=27*l[js[0]]*l[js[1]]*l[js[2]];gr[4+i]={};
  for(int j=0;j<3;++j)gr[4+i]=gr[4+i]+27.0*l[js[(j+1)%3]]*l[js[(j+2)%3]]*g.gl[js[j]];
 }
}
void face_basis(const Geom&geom,int lf,int q,double ph[8],G&gr){
 std::array<double,4> l{};int f=0;for(int i=0;i<4;++i)if(i!=lf){l[i]=(f==0?Q0[q]:(f==1?Q1[q]:Q2[q]));++f;}
 shape(geom,l,ph,gr);
}
void convection(Element24&K,const Geom&geom,const std::vector<double>&lag,const std::array<int,8>&entity,
  const nodals::cpu_mc::m4::EnrichmentSpace*enrich=nullptr,int cell=-1){
 for(int ir=0;ir<5;++ir)for(int is=0;is<5;++is)for(int it=0;it<5;++it){
  const double r=X5[ir],s=X5[is],t=X5[it],omr=1-r,oms=1-s;
  const double wt=W5[ir]*W5[is]*W5[it]*omr*omr*oms*geom.det;
  const std::array<double,4> l={omr*oms*(1-t),r,omr*s,omr*oms*t};double ph[8];G gr{};shape(geom,l,ph,gr);
  double carrier[3]={};for(int z=0;z<8;++z)for(int d=0;d<3;++d)carrier[d]+=ph[z]*lag[3*entity[z]+d];
   if(enrich&&enrich->enabled){
     const Vec3 whole=enrich->velocity(cell,l,lag);
     carrier[0]=whole.x;carrier[1]=whole.y;carrier[2]=whole.z;
   }
  for(int a=0;a<8;++a)for(int b=0;b<8;++b){const double v=wt*ph[a]*(carrier[0]*gr[b].x+carrier[1]*gr[b].y+carrier[2]*gr[b].z);
    for(int d=0;d<3;++d)K[3*a+d][3*b+d]+=v;
  }
 }
}
// On cells without active Spalding enrichment, integrate the same
// grad-div bilinear form over all 8 P1+BF3 modes.
void add_grad_div_unenriched(Element24&K,const Geom&g,double gamma){
 for(int ir=0;ir<5;++ir)for(int is=0;is<5;++is)for(int iz=0;iz<5;++iz){
  const double r=X5[ir],s=X5[is],t=X5[iz],omr=1-r,oms=1-s;
  const double wt=W5[ir]*W5[is]*W5[iz]*omr*omr*oms*g.det;
  const std::array<double,4> lam={omr*oms*(1-t),r,omr*s,omr*oms*t};
  double ph[8];G gp{};shape(g,lam,ph,gp);
  std::array<std::array<double,3>,8> gg{};
  for(int a=0;a<8;++a)gg[a]={gp[a].x,gp[a].y,gp[a].z};
  nodals::cpu_mc::m4::add_grad_div_rank_one<8>(K,gg,gamma*wt);
 }
}
}
namespace {
using B9=std::array<double,9>;
B9 inverse3(const B9&A){
 double inv[3][6]{};
 for(int i=0;i<3;++i){for(int j=0;j<3;++j)inv[i][j]=A[3*i+j];inv[i][3+i]=1.;}
 for(int k=0;k<3;++k){int piv=k;double big=std::abs(inv[k][k]);
  for(int i=k+1;i<3;++i)if(std::abs(inv[i][k])>big){big=std::abs(inv[i][k]);piv=i;}
  if(!(big>1e-35))throw std::runtime_error("M3 SGS: singular 3x3 block");
  if(piv!=k)for(int j=0;j<6;++j)std::swap(inv[piv][j],inv[k][j]);
  const double dia=inv[k][k];for(int j=0;j<6;++j)inv[k][j]/=dia;
  for(int i=0;i<3;++i)if(i!=k){const double v=inv[i][k];for(int j=0;j<6;++j)inv[i][j]-=v*inv[k][j];}
 }
 B9 out{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)out[3*i+j]=inv[i][j+3];return out;
}
// In-place T^T K T on the element, using the exact face BF3 projector.
// Gauge n*n on its redundant bubble coordinate makes local 3x3 SGS invertible.
void constrain_cell(Element24&K,std::array<double,24>&b,const Mesh&m,int c,bool addGauge=true){
 const auto&t=m.tets[c];
 for(int lf=0;lf<4;++lf){
  const auto&F=m.faces[t.face[lf]];
  if(F.patch!=3)continue;
  const Vec3 area=outward_area(m,c,lf);
  const double len=std::sqrt(dot(area,area));
  if(!(len>0))throw std::runtime_error("zero wall face area");
  const Vec3 nn=(1.0/len)*area;
  const double n[3]={nn.x,nn.y,nn.z};
  const int bubble=3*(4+lf);
  // Right transform: K <- K*T. Read the complete original bubble column first.
  for(int i=0;i<24;++i){
   double bn=0;for(int j=0;j<3;++j)bn+=K[i][bubble+j]*n[j];
   for(int j=0;j<3;++j)K[i][bubble+j]-=bn*n[j];
   for(int a=0;a<4;++a)if(a!=lf)for(int j=0;j<3;++j)K[i][3*a+j]-=(20.0/27.0)*bn*n[j];
  }
  // Left transform: K <- T^T*K, RHS <- T^T*RHS.
  for(int j=0;j<24;++j){
   double bn=0;for(int i=0;i<3;++i)bn+=K[bubble+i][j]*n[i];
   for(int i=0;i<3;++i)K[bubble+i][j]-=bn*n[i];
   for(int a=0;a<4;++a)if(a!=lf)for(int i=0;i<3;++i)K[3*a+i][j]-=(20.0/27.0)*bn*n[i];
  }
  double br=0;for(int j=0;j<3;++j)br+=b[bubble+j]*n[j];
  for(int j=0;j<3;++j)b[bubble+j]-=br*n[j];
  for(int a=0;a<4;++a)if(a!=lf)for(int j=0;j<3;++j)b[3*a+j]-=(20.0/27.0)*br*n[j];
  // Redundant normal-coordinate gauge, exactly decoupled from physical solution.
  if(addGauge){
    double s=0;for(int i=0;i<3;++i)s+=std::abs(K[bubble+i][bubble+i]);
    s=std::max(1e-14,s/3.0);
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)K[bubble+i][bubble+j]+=s*n[i]*n[j];
  }
 }
}
}
M3Predictor::M3Predictor(const Mesh&m,const Ownership&o,const BF3Flux&w,
                         const CaseConfig&config,int rank,const std::vector<double>&lag,
 const std::vector<double>*k,const std::vector<double>*omega,const std::vector<double>*wallDistance,
 const nodals::cpu_mc::m4::EnrichmentSpace*enrich)
:mesh_(m),own_(o),wall_(w),case_(config),lag_(lag),rank_(rank),nEntities_((enrich&&enrich->enabled)?enrich->nent():(int)m.points.size()+(int)m.faces.size()),
 k_(k),omega_(omega),wallDistance_(wallDistance),enrich_(enrich){
 const int nv=(int)m.points.size();
 wallFit_=sample_spalding_faces(m,o,rank,config,lag,enrich);
 ownerOfEntity_.resize(nEntities_);
 const int nbase=nv+(int)m.faces.size();
 for(int e=0;e<nbase;++e)ownerOfEntity_[e]=(e<nv?o.vertex[e]:o.face[e-nv]);
 if(enrich_&&enrich_->enabled)for(int v=0;v<nv;++v)if(enrich_->index[(size_t)v]>=0)
  ownerOfEntity_[(size_t)nbase+enrich_->index[(size_t)v]]=o.vertex[(size_t)v];
 // M4_WALL_BF3_SWEEP_PROJECTION_V1: GPU m3_block_colour projects every
 // wall-face bubble RHS and local update into the tangential plane.
 // Q at entry/exit alone is not equivalent to this per-sweep restriction.
 wallBubbleNormal_.assign((std::size_t)nEntities_,{});
 for(int f=0;f<(int)m.faces.size();++f){
  const auto&face=m.faces[(std::size_t)f];
  if(face.patch!=3)continue;
  int lf=-1;
  for(int i=0;i<4;++i)if(m.tets[(std::size_t)face.cell0].face[i]==f){lf=i;break;}
  if(lf<0)throw std::runtime_error("M3 wall bubble has no matching local face");
  const auto area=outward_area(m,face.cell0,lf);
  const double norm=std::sqrt(dot(area,area));
  if(!(norm>0))throw std::runtime_error("M3 wall bubble degenerate area");
  wallBubbleNormal_[(std::size_t)nv+(std::size_t)f]=(1.0/norm)*area;
 }
 std::vector<int>localIndex(nEntities_,-1);
 for(int e=0;e<nEntities_;++e)if(ownerOfEntity_[e]==rank){
  localIndex[e]=(int)owned_.size();owned_.push_back(e);
 }
 using Row=std::unordered_map<int,B9>;
 std::vector<Row> rows(owned_.size());
 rhs_.assign((std::size_t)3*nEntities_,0.0);
 std::vector<double> nitschePenaltyDiag((std::size_t)3*nEntities_,0.0);
 ownedCells_=(std::size_t)o.countCell[rank];
 for(int c=0;c<(int)m.tets.size();++c){
  const auto&t=m.tets[c];std::array<int,8>ent{};
  int augmented[12];for(int a=0;a<8;++a)augmented[a]=a<4?t.v[a]:nv+t.face[a-4];
  for(int a=8;a<12;++a)augmented[a]=(enrich_&&enrich_->enabled)?enrich_->entity(c,a):-1;
  bool relevant=false;
  for(int a=0;a<8;++a){ent[a]=a<4?t.v[a]:nv+t.face[a-4];
    if(ownerOfEntity_[ent[a]]==rank)relevant=true;}
  if(enrich_&&enrich_->enabled)for(int a=8;a<12;++a)
   if(augmented[a]>=0&&ownerOfEntity_[augmented[a]]==rank_)relevant=true;
  if(!relevant)continue;
  ++incidentCells_;
  const Geom geom=geometry(m,c);
  bool enrichedCell=false;
  if(enrich_&&enrich_->enabled)for(int j=0;j<4;++j)
     enrichedCell|=enrich_->index[(size_t)t.v[j]]>=0;
  Element24 K{};
  if(!enrichedCell){
   K=deviatoric_element(m,c,config.nu);
   if(k_&&omega_&&wallDistance_)
    nodals::cpu_mc::m4::momentum_turbulent_deviatoric(K,m,c,lag_,*k_,*omega_,*wallDistance_,config,enrich_);
   convection(K,geom,lag_,ent,enrich_,c);
   if(config.gradDivGamma>0&&config.gradDivForm=="volume")
     add_grad_div_unenriched(K,geom,config.gradDivGamma);
  }
  // For enriched cells, the complete PP/PE/EP/EE volume block is integrated
  // together with the same quadrature and the same frozen nu+nut.
  Element24 Kpenalty{}; // only Nitsche normal penalty, for GPU penalty-free rAU
  nodals::cpu_mc::m4::K36 Kextra{};
  nodals::cpu_mc::m4::RHS36 Bextra{};
  if(enrichedCell){
   if(!(k_&&omega_&&wallDistance_))throw std::runtime_error("enriched momentum requires SST fields");
   nodals::cpu_mc::m4::augment_momentum(*enrich_,m,c,lag_,*k_,*omega_,*wallDistance_,config,Kextra,Bextra);
  }
  std::array<double,24>br{};
  // Exact 12-point affine boundary quadrature, with DG vector prescribed inlet.
  for(int lf=0;lf<4;++lf){
   const auto&face=m.faces[t.face[lf]];
   if(face.cell1>=0||(face.patch!=1&&face.patch!=3))continue;
   const Vec3 area=outward_area(m,c,lf);
   const double ar=std::sqrt(dot(area,area));
   const Vec3 nn=(1.0/ar)*area;
   const double n[3]={nn.x,nn.y,nn.z};
   const double h=3.0*t.volume/ar;
   const double inflow=std::max(0.0,-(config.inlet[0]*n[0]+config.inlet[1]*n[1]+config.inlet[2]*n[2]));
   for(int q=0;q<12;++q){
    double ph[8];G g{};face_basis(geom,lf,q,ph,g);
    const double wt=ar*QW[q];
    const double beta=(face.patch==3 && !(enrich_&&enrich_->enabled))?
      wallFit_.qBeta[(size_t)12*t.face[lf]+(size_t)q]:0.0;
    // GPU full SST Nitsche and DG inlet use local nu_eff at face quadrature.
    const double nuEff=(k_&&omega_&&wallDistance_)?
      config.nu+nodals::cpu_mc::m4::sst_nut_at_point(m,c,
        [&](){std::array<double,4> l{};int n=0;for(int a=0;a<4;++a)if(a!=lf){
          l[a]=n==0?Q0[q]:(n==1?Q1[q]:Q2[q]);++n;}
          return l;}(),lag_,*k_,*omega_,*wallDistance_,config,enrich_):config.nu;
    for(int a=0;a<8;++a){
      const double dna=dot(g[a],nn);
      if(face.patch==1){
       const double f=wt*(inflow*ph[a]+nuEff*dna);
       for(int i=0;i<3;++i)br[3*a+i]+=f*config.inlet[i];
      }
      for(int b=0;b<8;++b){
       const double dnb=dot(g[b],nn);
       for(int i=0;i<3;++i)for(int j=0;j<3;++j){
        if(face.patch==1){
         if(i==j)K[3*a+i][3*b+j]+=wt*(inflow*ph[a]*ph[b]+nuEff*(-ph[a]*dnb+dna*ph[b]));
        }else{
         const double ij=n[i]*n[j];
         const double gamma=(enrich_&&enrich_->enabled)?0.0:config.nitscheGamma;
         const double v=beta*((i==j?1.0:0.0)-ij)+gamma*nuEff/h*ij;
         K[3*a+i][3*b+j]+=wt*ph[a]*ph[b]*v;
         Kpenalty[3*a+i][3*b+j]+=wt*ph[a]*ph[b]*
           ((enrich_&&enrich_->enabled)?0.0:config.nitscheGamma*nuEff/h*ij);
        }
       }
      }
    }
   }
  }
  // Exact algebraic P0 augmentation of the SAME B and DG source as
  // Schur::continuity(): B u + source = 0. The RHS term below is essential
  // at inlet cells with source != 0. Never mix this with volume grad-div.
  if(config.gradDivGamma>0&&config.gradDivForm=="constraint_p0"){
    if(!(enrich_&&enrich_->enabled))
      throw std::runtime_error("constraint_p0 requires enriched strong no-slip path");
    if(!(t.volume>0.0&&std::isfinite(t.volume)))
      throw std::runtime_error("constraint_p0 invalid cell pressure mass volume");
    const auto row=nodals::cpu_mc::m4::build_discrete_constraint_row(
      m,c,config.inlet,enrich_);
    std::array<std::array<double,3>,12> bvec{};
    for(int a=0;a<12;++a){
      if(row.entity[a]!=augmented[a])
        throw std::runtime_error("constraint_p0 B entity ordering mismatch");
      // In this strongly constrained enriched space, Schur's reduced
      // B and physical B must agree on every free velocity coefficient.
      if(row.entity[a]>=0&&!enrich_->is_fixed(row.entity[a])){
        const auto delta=row.physical[a]-row.reduced[a];
        const double dnorm=std::sqrt(dot(delta,delta));
        const double rnorm=std::sqrt(dot(row.physical[a],row.physical[a]));
        if(dnorm>1e-12*std::max(1.0,rnorm))
          throw std::runtime_error("constraint_p0 physical/reduced B mismatch on free velocity entity");
      }
      const auto v=row.physical[a];
      bvec[a]={v.x,v.y,v.z};
    }
    nodals::cpu_mc::m4::RHS36 rhsAug{};
    nodals::cpu_mc::m4::add_constraint_augmentation<12>(
      Kextra,rhsAug,bvec,row.source,config.gradDivGamma/t.volume);
    for(int a=0;a<12;++a)for(int i=0;i<3;++i){
      if(a<8)br[3*a+i]+=rhsAug[3*a+i];
      else Bextra[3*a+i]+=rhsAug[3*a+i];
    }
  }
  if(!(enrich_&&enrich_->enabled))constrain_cell(K,br,m,c);
  std::array<double,24>zeroPenaltyRhs{};
  if(!(enrich_&&enrich_->enabled))constrain_cell(Kpenalty,zeroPenaltyRhs,m,c,false);
  const int nloc=(enrich_&&enrich_->enabled)?12:8;
  for(int a=0;a<nloc;++a){
   const int rowEntity=augmented[a];
   if(rowEntity<0)continue;
   const int local=localIndex[rowEntity];
   if(local<0)continue;
   auto&row=rows[(size_t)local];
   for(int i=0;i<3;++i){
     rhs_[3*(size_t)rowEntity+i]+=a<8?br[3*a+i]:Bextra[3*a+i];
     if(a<8)nitschePenaltyDiag[3*(size_t)rowEntity+i]+=Kpenalty[3*a+i][3*a+i];
   }
   for(int b=0;b<nloc;++b){
    if(augmented[b]<0)continue;
    auto&block=row[augmented[b]];
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)
       block[3*i+j]+=(a<8&&b<8?K[3*a+i][3*b+j]:0.0)
         +((enrich_&&enrich_->enabled)?Kextra[3*a+i][3*b+j]:0.0);
   }
  }
 }
 rowPtr_.push_back(0);
 rowL1_.assign((std::size_t)3*nEntities_,0.0);
 diagInverse_.resize(owned_.size());
 inv_.assign((std::size_t)3*nEntities_,0.0);
 for(std::size_t ri=0;ri<owned_.size();++ri){
  const int entity=owned_[ri];
  // Strong no-slip: eliminate wall vertex and wall face-bubble rows/columns
  // before relaxation, rather than imposing a second Spalding Robin drag.
  if(enrich_&&enrich_->enabled){
   if(enrich_->is_fixed(entity)){
    rows[ri].clear();B9 identity{};
    for(int j=0;j<3;++j){identity[3*j+j]=1.;rhs_[3*(size_t)entity+j]=0.;}
    rows[ri][entity]=identity;
   }else{
    for(auto&kv:rows[ri])if(enrich_->is_fixed(kv.first))kv.second=B9{};
   }
  }
  std::vector<int>keys;keys.reserve(rows[ri].size());
  for(const auto&pair:rows[ri])keys.push_back(pair.first);
  std::sort(keys.begin(),keys.end());
  B9 diag{};double scalarRowL1[3]={};
  for(int col:keys){
   const B9 block=rows[ri].at(col);
   if(col==entity)diag=block;
   for(int i=0;i<3;++i){
    scalarRowL1[i]+=std::abs(block[3*i+i]);
    for(int j=0;j<3;++j)rowL1_[3*entity+i]+=std::abs(block[3*i+j]);
   }
  }
  // GPU h5b3_finalize_row_l1: delta=(alpha^-1-1)*sum_j |A_ii(row,j)|,
  // the same relaxation shift appears in the predictor operator and rAU.
  const double fac=1.0/config.alphaU-1.0;
  for(int i=0;i<3;++i){
   const double diagonalBeforeRelax=diag[3*i+i];
   const double shift=fac*scalarRowL1[i];
   rhs_[3*entity+i]+=shift*lag_[3*entity+i];
   diag[3*i+i]+=shift;
   rows[ri][entity][3*i+i]+=shift;
   const double momentumDiag=diag[3*i+i];
   const double pressureMetric=momentumDiag-nitschePenaltyDiag[(std::size_t)3*entity+i];
   // Match GPU h5b3_finalize_row_l1: Nitsche normal penalty contributes to
   // momentum, but NOT to pressure rAU when schurNitscheGamma=0.
   if(!(momentumDiag>0.0 && pressureMetric>0.0) ||
       !std::isfinite(momentumDiag) || !std::isfinite(pressureMetric)){
     const int nv=(int)mesh_.points.size();
     const int nbase=nv+(int)mesh_.faces.size();
     const char*kind=entity<nv?"P1_VERTEX":(entity<nbase?"BF3_FACE":"SPALDING_ENRICH");
     std::fprintf(stderr,
       "NODALS_CPU_M4_BAD_MOMENTUM_DIAG rank=%d entity=%d component=%d entityKind=%s "
       "relaxAlpha=%.17g originalDiag=%.17e scalarRowL1=%.17e relaxationShift=%.17e "
       "finalMomentumDiag=%.17e nitschePenaltyDiag=%.17e pressureMetric=%.17e "
       "rowEntries=%zu lagU=%.17e status=FAIL\n",
       rank_,entity,i,kind,config.alphaU,diagonalBeforeRelax,scalarRowL1[i],shift,
       momentumDiag,nitschePenaltyDiag[(std::size_t)3*entity+i],
       pressureMetric,rows[ri].size(),lag_[3*(size_t)entity+i]);
     std::fflush(stderr);
     throw std::runtime_error("M3: nonpositive relaxed momentum or penalty-free pressure metric");
   }
   inv_[3*entity+i]=(enrich_&&enrich_->is_fixed(entity))?0.0:1.0/pressureMetric;
  }
  for(int col:keys){cols_.push_back(col);a_.push_back(rows[ri].at(col));}
  diagInverse_[ri]=inverse3(diag);
  rowPtr_.push_back((int)a_.size());
 }
 MPI_Allreduce(MPI_IN_PLACE,inv_.data(),(int)inv_.size(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
 if(rank_==0 && std::getenv("M4_VERBOSE_SETUP"))std::printf("NODALS_CPU_SST_M3_SGS_SETUP status=PASS smoother=PROCESSOR_BLOCK_FORWARD_3X3_GS matrix=COUPLED_3X3_BCSR constraints=GALERKIN_TtAT schurRAU=PENALTY_FREE_ROW_L1\n");
}
void M3Predictor::apply(const std::vector<double>&x,std::vector<double>&out)const{
 out.assign((std::size_t)3*nEntities_,0.0);
 for(std::size_t i=0;i<owned_.size();++i){
  const int entity=owned_[i];
  for(int k=rowPtr_[i];k<rowPtr_[i+1];++k){
   const int col=cols_[k];const auto&b=a_[k];
   for(int r=0;r<3;++r)for(int s=0;s<3;++s)
     out[3*entity+r]+=b[3*r+s]*x[3*col+s];
  }
 }
 MPI_Allreduce(MPI_IN_PLACE,out.data(),(int)out.size(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
}
void M3Predictor::sync_updates(std::vector<double>&x,const std::vector<double>&old)const{
 std::vector<double>delta(x.size(),0.0);
 for(int e:owned_)for(int d=0;d<3;++d)delta[3*e+d]=x[3*e+d]-old[3*e+d];
 MPI_Allreduce(MPI_IN_PLACE,delta.data(),(int)delta.size(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
 for(std::size_t i=0;i<x.size();++i)x[i]=old[i]+delta[i];
}
void M3Predictor::solve(std::vector<double>&u,int sweeps,double omega,double*start,double*end,
 const std::vector<double>*pressureForce,int*iterations)const{
 std::vector<double>x=lag_,Ax,res;wall_.Q(x);
 if(enrich_&&enrich_->enabled)for(int e=0;e<enrich_->ne;++e)if(enrich_->is_fixed(e))
   for(int d=0;d<3;++d)x[3*(size_t)e+d]=0.;
 std::vector<double> rhsEffective=rhs_;
 if(pressureForce){
   if(pressureForce->size()!=rhs_.size())throw std::runtime_error("M3 pressure force dimension mismatch");
   std::vector<double> grad=*pressureForce;wall_.Tt(grad);
   for(int e:owned_)for(int d=0;d<3;++d)rhsEffective[3*e+d]+=grad[3*e+d];
 }

 const auto dot_owned=[&](const std::vector<double>&a,const std::vector<double>&b){
  double l=0.0,g=0.0;
  for(int e:owned_)for(int d=0;d<3;++d)l+=a[3*e+d]*b[3*e+d];
  MPI_Allreduce(&l,&g,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);return g;
 };
 const auto l2=[&](const std::vector<double>&v){double l=0.0;
  for(int e:owned_)for(int d=0;d<3;++d)l+=v[3*e+d]*v[3*e+d];
  double g=0;MPI_Allreduce(&l,&g,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  return std::sqrt(g);
 };
 apply(x,Ax);res.resize(x.size());
 for(std::size_t j=0;j<x.size();++j)res[j]=rhsEffective[j]-Ax[j];wall_.Q(res);
 // RHS is owned-row only. Collect it once, not once per smoother iteration.
 // Non-owning entries remain zero in every rank's local RHS.
 const double initial=l2(res);if(start)*start=initial;
 if(iterations)*iterations=0;
 if(case_.method=="fgmres"){
  const int restart=case_.momRestart,maxIts=case_.momMaxIts;
  const double tol=std::max(case_.momAtol,case_.momRtol*initial);
  const auto rn=[&](const std::vector<double>&v){return std::sqrt(std::max(0.,dot_owned(v,v)));};
  const auto prec=[&](const std::vector<double>&r,std::vector<double>&z){
    z.assign(r.size(),0.0);
    for(std::size_t ri=0;ri<owned_.size();++ri){
      const int entity=owned_[ri];
      double rr[3]={r[3*entity],r[3*entity+1],r[3*entity+2]};
      for(int k=rowPtr_[ri];k<rowPtr_[ri+1];++k){
       const int col=cols_[k];const B9&block=a_[k];
       for(int i=0;i<3;++i)for(int j=0;j<3;++j)rr[i]-=block[3*i+j]*z[3*col+j];
      }
      const Vec3 n=wallBubbleNormal_[(std::size_t)entity];
      const double nn[3]={n.x,n.y,n.z};
      const bool wallBubble=(n.x!=0||n.y!=0||n.z!=0);
      if(wallBubble){const double dr=nn[0]*rr[0]+nn[1]*rr[1]+nn[2]*rr[2];
       for(int i=0;i<3;++i)rr[i]-=dr*nn[i];}
      const B9&di=diagInverse_[ri];double dx[3]={};
      for(int i=0;i<3;++i)for(int j=0;j<3;++j)dx[i]+=di[3*i+j]*rr[j];
      if(wallBubble){const double dn=nn[0]*dx[0]+nn[1]*dx[1]+nn[2]*dx[2];
       for(int i=0;i<3;++i)dx[i]-=dn*nn[i];}
      for(int i=0;i<3;++i)z[3*entity+i]=dx[i];
    }
    // Sum owned increments so all MPI ranks have the same z.
    MPI_Allreduce(MPI_IN_PLACE,z.data(),(int)z.size(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    wall_.Q(z);
  };
  int total=0;double finalR=initial;bool converged=false;
  while(total<maxIts){
   apply(x,Ax);res.resize(x.size());
   for(std::size_t i=0;i<res.size();++i)res[i]=rhsEffective[i]-Ax[i];wall_.Q(res);
   const double beta=rn(res);finalR=beta;
   if(!std::isfinite(beta))break;
   if(beta<=tol){converged=true;break;}
   std::vector<std::vector<double>> V,Z;
   V.reserve(restart+1);Z.reserve(restart);
   for(double&v:res)v/=beta;V.push_back(std::move(res));
   std::vector<std::vector<double>> H((std::size_t)restart+1,std::vector<double>((std::size_t)restart,0.0));
   std::vector<double> cs(restart,0.0),sn(restart,0.0),g((std::size_t)restart+1,0.0);g[0]=beta;
   int nsteps=0;
   for(int j=0;j<restart&&total<maxIts;++j){
    std::vector<double> zj,w;
    prec(V[(std::size_t)j],zj);
    Z.push_back(std::move(zj));
    apply(Z.back(),w);wall_.Q(w);
    for(int i=0;i<=j;++i){
     const double hij=dot_owned(w,V[(std::size_t)i]);H[(std::size_t)i][(std::size_t)j]=hij;
     for(std::size_t q=0;q<w.size();++q)w[q]-=hij*V[(std::size_t)i][q];
    }
    const double hnext=rn(w);H[(std::size_t)j+1][(std::size_t)j]=hnext;
    if(hnext>0&&std::isfinite(hnext)){for(double&v:w)v/=hnext;V.push_back(std::move(w));}
    else V.push_back(std::vector<double>(x.size(),0.0));
    for(int i=0;i<j;++i){
     const double h0=H[(std::size_t)i][(std::size_t)j],h1=H[(std::size_t)i+1][(std::size_t)j];
     H[(std::size_t)i][(std::size_t)j]=cs[i]*h0+sn[i]*h1;
     H[(std::size_t)i+1][(std::size_t)j]=-sn[i]*h0+cs[i]*h1;
    }
    const double a=H[(std::size_t)j][(std::size_t)j],b=H[(std::size_t)j+1][(std::size_t)j];
    const double h=std::hypot(a,b);
    cs[j]=h>0?a/h:1;sn[j]=h>0?b/h:0;
    H[(std::size_t)j][(std::size_t)j]=cs[j]*a+sn[j]*b;
    H[(std::size_t)j+1][(std::size_t)j]=0;
    g[(std::size_t)j+1]=-sn[j]*g[(std::size_t)j];g[(std::size_t)j]=cs[j]*g[(std::size_t)j];
    ++total;nsteps=j+1;
    if(!std::isfinite(g[(std::size_t)j+1])||std::abs(g[(std::size_t)j+1])<=tol||hnext<1e-300)break;
   }
   if(nsteps==0)break;
   std::vector<double> y((std::size_t)nsteps,0.0);
   bool good=true;
   for(int i=nsteps-1;i>=0;--i){
    double v=g[(std::size_t)i];for(int j=i+1;j<nsteps;++j)v-=H[(std::size_t)i][(std::size_t)j]*y[(std::size_t)j];
    const double dia=H[(std::size_t)i][(std::size_t)i];
    if(!(std::abs(dia)>1e-300)||!std::isfinite(dia)){good=false;break;}
    y[(std::size_t)i]=v/dia;
   }
   if(!good)break;
   for(int j=0;j<nsteps;++j)for(std::size_t q=0;q<x.size();++q)x[q]+=y[(std::size_t)j]*Z[(std::size_t)j][q];
   wall_.Q(x);
  }
  apply(x,Ax);res.resize(x.size());
  for(std::size_t i=0;i<res.size();++i)res[i]=rhsEffective[i]-Ax[i];wall_.Q(res);
  finalR=rn(res);converged=converged||(std::isfinite(finalR)&&finalR<=tol);
  if(iterations)*iterations=total;
  if(!converged)throw std::runtime_error("momentum FGMRES failed case-selected rtol/atol within max_iterations="+std::to_string(maxIts)+"; final="+std::to_string(finalR)+" tolerance="+std::to_string(tol));
  if(end)*end=finalR;
 } else {
 // Fixed-sweep M3: exactly the case-selected sweep count, with no linear
 // rtol/atol gate and NO automatic FGMRES fallback. The nonlinear SIMPLE
 // loop, not this stationary iteration, governs physical convergence.
 for(int it=0;it<sweeps;++it){
  const auto old=x;
  for(int direction=0;direction<(case_.method=="m3-fgs"?1:2);++direction){
   for(std::size_t q=0;q<owned_.size();++q){
    const std::size_t ri=direction?owned_.size()-1-q:q;
    const int entity=owned_[ri];
    double rr[3]={rhsEffective[3*entity],rhsEffective[3*entity+1],rhsEffective[3*entity+2]};
    for(int k=rowPtr_[ri];k<rowPtr_[ri+1];++k){
     const int col=cols_[k];const B9&b=a_[k];
     for(int i=0;i<3;++i)for(int j=0;j<3;++j)rr[i]-=b[3*i+j]*x[3*col+j];
    }
    const Vec3 n=wallBubbleNormal_[(std::size_t)entity];
    const double nn[3]={n.x,n.y,n.z};
    const bool wallBubble=(n.x!=0.0||n.y!=0.0||n.z!=0.0);
    if(wallBubble){
     const double rn=nn[0]*rr[0]+nn[1]*rr[1]+nn[2]*rr[2];
     for(int i=0;i<3;++i)rr[i]-=rn*nn[i];
    }
    const B9&di=diagInverse_[ri];double dx[3]={};
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)dx[i]+=di[3*i+j]*rr[j];
    if(wallBubble){
     const double dn=nn[0]*dx[0]+nn[1]*dx[1]+nn[2]*dx[2];
     for(int i=0;i<3;++i)dx[i]-=dn*nn[i];
    }
    for(int i=0;i<3;++i)x[3*entity+i]+=omega*dx[i];
    if(wallBubble){
     const double xn=nn[0]*x[3*entity]+nn[1]*x[3*entity+1]+nn[2]*x[3*entity+2];
     for(int i=0;i<3;++i)x[3*entity+i]-=xn*nn[i];
    }
   }
  }
  sync_updates(x,old);
 }
 // Report the residual after the specified number of sweeps; never use
 // its magnitude to demand additional linear solves.
 apply(x,Ax);
 for(std::size_t j=0;j<x.size();++j)res[j]=rhsEffective[j]-Ax[j];
 wall_.Q(res);
 const double final=l2(res);
 if(end)*end=final;
 if(!std::isfinite(final))throw std::runtime_error("nonfinite fixed-sweep momentum residual");
 if(iterations)*iterations=sweeps;
 }
 u=std::move(x);wall_.Q(u);wall_.T(u);
 for(double v:u)if(!std::isfinite(v))throw std::runtime_error("nonfinite M3 SGS velocity");
}
}
