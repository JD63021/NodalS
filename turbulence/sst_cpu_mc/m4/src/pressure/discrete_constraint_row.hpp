#pragma once
// Shared definition of the exact DG-inlet / constrained wall P0 continuity
// equation B u + source = 0. Used both by pressure and (optionally) by the
// algebraically augmented momentum operator, preventing a mismatched RHS.
// This is the SAME operator as the original M4 Schur builder.
#include "../mesh/tet_mesh.hpp"
#include "../operators/p1bf3_p0.hpp"
#include "../enrichment/enrichment_space.hpp"
#include <array>
#include <cmath>
#include <stdexcept>
namespace nodals::cpu_mc::m4 {
using namespace nodals::cpu_mc::m2;
struct DiscreteConstraintRow {
 std::array<int,12> entity{};
 std::array<Vec3,12> physical{},reduced{};
 double source=0.0;
};
inline DiscreteConstraintRow build_discrete_constraint_row(
 const Mesh&m,int c,const std::array<double,3>&inlet,
 const EnrichmentSpace*enrich){
 const auto&t=m.tets[(std::size_t)c];
 const int nv=(int)m.points.size();
 const auto raw=divergence_coefficients(m,c);
 DiscreteConstraintRow row;
 row.entity.fill(-1);
 for(int a=0;a<8;++a){
  row.entity[a]=a<4?t.v[a]:nv+t.face[a-4];
  row.physical[a]=raw.b[a];
 }
 for(int a=8;a<12;++a)
  row.entity[a]=(enrich&&enrich->enabled)?enrich->entity(c,a):-1;
 if(enrich&&enrich->enabled){
  std::array<Vec3,4> be{};
  enrich->divergence(c,be);
  for(int a=0;a<4;++a)row.physical[8+a]=be[a];
 }
 for(int lf=0;lf<4;++lf){
  if(m.faces[(std::size_t)t.face[lf]].patch!=1)continue;
  const Vec3 sf=outward_area(m,c,lf);
  const double area=std::sqrt(dot(sf,sf));
  if(!(area>0))throw std::runtime_error("continuity constraint: degenerate inlet area");
  for(int a=0;a<4;++a)if(a!=lf)
   row.physical[a]=row.physical[a]-(1.0/3.0)*sf;
  row.physical[4+lf]=row.physical[4+lf]-(9.0/20.0)*sf;
  if(enrich&&enrich->enabled){
   std::array<Vec3,4> be{};
   enrich->face_flux(c,lf,be);
   for(int a=0;a<4;++a)row.physical[8+a]=row.physical[8+a]-be[a];
  }
  row.source+=dot(sf,{inlet[0],inlet[1],inlet[2]});
 }
 row.reduced=row.physical;
 for(int lf=0;lf<4;++lf){
  if(m.faces[(std::size_t)t.face[lf]].patch!=3)continue;
  const Vec3 sf=outward_area(m,c,lf);
  const double area=std::sqrt(dot(sf,sf));
  if(!(area>0))throw std::runtime_error("continuity constraint: degenerate wall face");
  const Vec3 n=(1.0/area)*sf;
  const double bn=dot(row.reduced[4+lf],n);
  row.reduced[4+lf]=row.reduced[4+lf]-bn*n;
  for(int j=0;j<4;++j)if(j!=lf)
   row.reduced[j]=row.reduced[j]-(20.0/27.0)*bn*n;
 }
 if(enrich&&enrich->enabled)
  for(int a=0;a<12;++a)
   if(row.entity[a]>=0&&enrich->is_fixed(row.entity[a]))
    row.physical[a]=row.reduced[a]=Vec3{};
 return row;
}
}
