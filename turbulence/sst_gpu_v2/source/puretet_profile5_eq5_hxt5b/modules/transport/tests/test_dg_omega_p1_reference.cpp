#include "../dg_omega_p1_reference.hpp"
#include <cassert>
#include <iostream>
#include <iomanip>
using namespace nodals::omega_dg;
static void close(double got,double want,double tol=1.e-9){
  if(std::abs(got-want)>tol){std::cerr<<std::setprecision(17)<<"got="<<got<<" expected="<<want<<"\n";std::abort();}
}
static Mesh two_tets(){
  Mesh m;
  m.vertices={{0,0,0},{1,0,0},{0,1,0},{0,0,1},{0,0,-1}};
  m.tetra={{0,1,2,3},{0,2,1,4}};
  initialize(m);return m;
}
int main(){
  Mesh m=two_tets();
  assert(m.cells.size()==2&&m.faces.size()==7);
  int interior=0;for(auto&f:m.faces)if(f.right.tet>=0)++interior;
  assert(interior==1);
  for(const auto&g:m.cells){Vec3 s{};for(auto v:g.grad)s=s+v;close(norm(s),0,1e-13);}
  std::vector<Coeff> c(2);for(auto&z:c)z.molecularAndTurbulentDiffusion=0.1;
  Options o;
  auto A=assemble(m,c,o);
  close(A.max_asymmetry(),0,1e-10);
  close(l2(A.multiply(std::vector<double>(8,3.))),0,1e-9);
  auto exact=[](Vec3 x){return 2.+x.x+0.5*x.y+0.25*x.z;};
  for(auto&kv:m.boundary){kv.second.kind=BoundaryKind::Wall;kv.second.target=exact;}
  auto D=assemble(m,c,o);
  std::vector<double> solution(8);
  for(int t=0;t<2;++t)for(int j=0;j<4;++j)
    solution[(size_t)dof(t,j)]=exact(m.vertices[(size_t)m.tetra[(size_t)t][(size_t)j]]);
  auto residual=D.multiply(solution);for(int i=0;i<8;++i)residual[(size_t)i]-=D.rhs[(size_t)i];
  close(l2(residual),0,1e-8);
  close(D.max_asymmetry(),0,1e-8);
  std::vector<double> guess(8,0.);
  auto report=fgmres(D,guess,12,50,1e-11);
  if(!report.converged)return 2;
  for(int i=0;i<8;++i)close(guess[(size_t)i],solution[(size_t)i],1e-8);
  Mesh adv=two_tets();
  for(auto&kv:adv.boundary){kv.second.kind=BoundaryKind::Inlet;kv.second.target=[](Vec3){return 3.;};}
  for(auto&z:c){z.molecularAndTurbulentDiffusion=0;z.velocity={0.8,0.6,0.2};}
  auto C=assemble(adv,c,o);auto cr=C.multiply(std::vector<double>(8,3.));
  for(int i=0;i<8;++i)cr[(size_t)i]-=C.rhs[(size_t)i];
  close(l2(cr),0,1e-10);
  Mesh leak=two_tets();
  for(auto&kv:leak.boundary)kv.second.kind=BoundaryKind::Wall;
  bool rejected=false;try{(void)assemble(leak,c,o);}catch(const std::runtime_error&){rejected=true;}
  if(!rejected)return 3;
  std::array<double,4>w{-1,2,2,2};
  if(!preserve_mean_floor(w,0))return 4;
  close((w[0]+w[1]+w[2]+w[3])/4.,1.25,1e-14);
  if(*std::min_element(w.begin(),w.end())<-1e-14)return 5;
  std::array<double,4>bad{-1,-1,-1,-1};
  if(preserve_mean_floor(bad,0))return 6;
  std::cout<<"DG_OMEGA_P1_REFERENCE_PASS cells=2 faces=7 ndof=8 nnz="<<D.nnz()
           <<" iterations="<<report.iterations<<" residual="<<report.final<<"\n";
}
