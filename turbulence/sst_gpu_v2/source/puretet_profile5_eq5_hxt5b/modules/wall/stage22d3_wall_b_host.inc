// Stage22D3 experimental B_N matrix builder. The live solver's B has
// +int(div u) as its volume convention, so B_N = B - int_wall(n.u).
// Imported into hxt5b namespace in generated hxt5b_boundary.cuh.
inline void s22d3_patch_normal_wall_divergence(hxt3a::BCSR& B,
  const HostMesh& M,const std::vector<TetFacePlan>& wallT){
  if(M.h.nhex||M.h.ninterface)throw std::runtime_error("22D3 normal wall pressure patch requires pure tetrahedra");
  for(const auto&f:wallT){
    std::array<double,8>I{};
    const int i0=hxt2::TET_FACE[f.lf][0],i1=hxt2::TET_FACE[f.lf][1],i2=hxt2::TET_FACE[f.lf][2];
    for(int q=0;q<12;++q){
      double l[4]={0,0,0,0};l[i0]=HL0[q];l[i1]=HL1[q];l[i2]=HL2[q];
      double phi[8];tbasis(l,phi);
      for(int a=0;a<8;++a)I[a]+=HW12[q]*f.area*phi[a];
    }
    auto&row=B.rows[f.cell];const auto&T=M.tetVel[f.cell];
    for(int a=0;a<8;++a)
      hxt3a::add(row,T.g[a],{-I[a]*f.n[0],-I[a]*f.n[1],-I[a]*f.n[2]});
  }
  repack(B);
}
