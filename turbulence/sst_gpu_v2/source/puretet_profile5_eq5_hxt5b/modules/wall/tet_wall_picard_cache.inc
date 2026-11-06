inline std::vector<TetWallQuadCache> build_tet_wall_quad_cache(
    const HostMesh&M,const std::vector<TetFacePlan>&faces)
{
  std::vector<TetWallQuadCache> out(faces.size()*12u);
  for(std::size_t fi=0;fi<faces.size();++fi){
    const auto&f=faces[fi];const auto&T=M.tets[(std::size_t)f.cell];
    const int a0=hxt2::TET_FACE[f.lf][0],a1=hxt2::TET_FACE[f.lf][1],a2=hxt2::TET_FACE[f.lf][2];
    const V3 X0=vp(M.points[(std::size_t)T.v[0]]);
    const V3 A=vsub(vp(M.points[(std::size_t)T.v[1]]),X0);
    const V3 B=vsub(vp(M.points[(std::size_t)T.v[2]]),X0);
    const V3 C=vsub(vp(M.points[(std::size_t)T.v[3]]),X0);
    const V3 n{f.n[0],f.n[1],f.n[2]};
    double dl[3];if(!wall_match_solve_cols(A,B,C,vmul(-1,n),dl))throw std::runtime_error("tet wall normal ref transform failed");
    double dr[4]={-dl[0]-dl[1]-dl[2],dl[0],dl[1],dl[2]};
    for(int q=0;q<12;++q){
      double lw[4]={};lw[a0]=HL0[q];lw[a1]=HL1[q];lw[a2]=HL2[q];
      auto&x=out[fi*12u+(std::size_t)q];tbasis(lw,x.bw);
      double exit=1e300;
      for(int j=0;j<4;++j)if(dr[j]<-1e-15)exit=std::min(exit,-lw[j]/dr[j]);
      if(!(exit>0.0)||!std::isfinite(exit)||exit>1e100)
        throw std::runtime_error("tet Picard normal column exit failure");
      x.ybar=0.45*exit; // average over [0,0.9*exit]
      for(int m=0;m<4;++m){
        double s=0.45*exit*(1.0+GX4[m]);
        double lm[4];for(int j=0;j<4;++j)lm[j]=lw[j]+s*dr[j];
        double pb[8];tbasis(lm,pb);
        for(int j=0;j<8;++j)x.bp[j]+=0.5*GW4[m]*pb[j];
      }
      x.w=f.area*HW12[q];
      for(int j=0;j<8;++j)if(!std::isfinite(x.bw[j])||!std::isfinite(x.bp[j]))
        throw std::runtime_error("tet wall Picard cache nonfinite basis");
    }
  }
  std::printf("PROFILE5_TET_PICARD_CACHE wallFaces=%zu quadraturePoints=%zu method=FIRST_TET_NORMAL_COLUMN_GAUSS4_90PCT basis=P1_BF3 status=PASS\n",faces.size(),out.size());
  return out;
}

