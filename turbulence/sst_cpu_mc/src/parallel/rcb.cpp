#include "rcb.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>

namespace nodals::cpu_mc::m2 {
Ownership partition_rcb(const Mesh&m,int nr){
  const int nc=(int)m.tets.size();if(nr<1||nr>nc)throw std::runtime_error("invalid rank count");
  std::vector<Vec3> cent(nc);
  for(int c=0;c<nc;++c){Vec3 p{};for(auto v:m.tets[c].v)p=p+m.points[v];cent[c]=0.25*p;}
  std::vector<int> ids(nc);std::iota(ids.begin(),ids.end(),0);
  Ownership own;own.cell.assign(nc,-1);
  std::function<void(int,int,int,int)> rec;
  rec=[&](int b,int e,int r0,int r1){
    if(r1-r0==1){for(int k=b;k<e;++k)own.cell[ids[k]]=r0;return;}
    double mn[3]={1e300,1e300,1e300},mx[3]={-1e300,-1e300,-1e300};
    for(int k=b;k<e;++k){auto q=cent[ids[k]];double val[3]={q.x,q.y,q.z};
      for(int d=0;d<3;++d){mn[d]=std::min(mn[d],val[d]);mx[d]=std::max(mx[d],val[d]);}}
    int axis=0;if(mx[1]-mn[1]>mx[axis]-mn[axis])axis=1;if(mx[2]-mn[2]>mx[axis]-mn[axis])axis=2;
    const int rm=(r0+r1)/2;
    const int nl=std::max(1,std::min(e-b-1,(int)((long long)(e-b)*(rm-r0)/(r1-r0))));
    auto coord=[&](int c){return axis==0?cent[c].x:(axis==1?cent[c].y:cent[c].z);};
    std::nth_element(ids.begin()+b,ids.begin()+b+nl,ids.begin()+e,[&](int a,int z){
      const double ca=coord(a),cz=coord(z);return ca<cz||(ca==cz&&a<z);
    });
    rec(b,b+nl,r0,rm);rec(b+nl,e,rm,r1);
  };
  rec(0,nc,0,nr);
  own.vertex.assign(m.points.size(),nr);
  own.face.assign(m.faces.size(),nr);
  own.countCell.assign(nr,0);own.countVertex.assign(nr,0);own.countFace.assign(nr,0);
  for(int c=0;c<nc;++c){const int r=own.cell[c];++own.countCell[r];
    for(int v:m.tets[c].v)own.vertex[v]=std::min(own.vertex[v],r);
    for(int f:m.tets[c].face)own.face[f]=std::min(own.face[f],r);
  }
  for(int r:own.vertex){if(r>=nr)throw std::runtime_error("unused vertex");++own.countVertex[r];}
  for(int r:own.face){if(r>=nr)throw std::runtime_error("unused face");++own.countFace[r];}
  return own;
}
}
