inline TetFacePlan tetplan(const HostMesh&M,const nodals_hxt1::BoundaryRec&r){
  TetFacePlan P;P.cell=r.cell;P.lf=r.localFace;const auto&t=M.tets[r.cell];V3 q[3];
  for(int k=0;k<3;++k)q[k]=vp(M.points[t.v[hxt2::TET_FACE[r.localFace][k]]]);
  V3 av=vmul(.5,vcross(vsub(q[1],q[0]),vsub(q[2],q[0]))),fc=vmul(1.0/3.0,vadd(vadd(q[0],q[1]),q[2])),cc{};
  for(int a=0;a<4;++a)cc=vadd(cc,vmul(.25,vp(M.points[t.v[a]])));if(vdot(av,vsub(fc,cc))<0)av=vmul(-1,av);
  P.area=vnorm(av);if(!(P.area>0))throw std::runtime_error("HXT5B TET boundary area invalid");av=vmul(1/P.area,av);P.n[0]=av.x;P.n[1]=av.y;P.n[2]=av.z;return P;
}

