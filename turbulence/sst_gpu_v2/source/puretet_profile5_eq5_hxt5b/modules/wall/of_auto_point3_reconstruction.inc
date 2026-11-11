  // Reconstruct the P1+BF3 state and physical gradient at the SAME point
  // used in the wall-normal sampling cache. In state-space q1, the enriched
  // coefficients are explicitly zeroed by the outer solver.
  const double l[4]={pm.b[0],pm.b[1],pm.b[2],pm.b[3]};
  double ph[8],gr[8][3],gp[8][3];
  hxt2::tet_basis(l[0],l[1],l[2],l[3],ph,gr);
  hxt2::tet_phys_grad(gr,tg[pm.cell],gp);
  Real gu[9]={},gk[3]={},gw[3]={};
  Real kv=0,wv=0,us[3]={};
  for(int a=0;a<8;++a){
    const int g=pm.g[a];
    if(g<0){atomicAdd(bad,1ULL);return;}
    const Real ka=k[g],wa=omega[g],ua=u0[g],va=u1[g],za=u2[g];
    const Real pa=(Real)ph[a];
    kv+=ka*pa;wv+=wa*pa;
    us[0]+=ua*pa;us[1]+=va*pa;us[2]+=za*pa;
    for(int d=0;d<3;++d){
      const Real gd=(Real)gp[a][d];
      gu[d]+=ua*gd;gu[3+d]+=va*gd;gu[6+d]+=za*gd;
      gk[d]+=ka*gd;gw[d]+=wa*gd;
    }
  }
  hxt5c::eval_dg_omega_p1(pm.cell,l,tg[pm.cell],wv,gw);
  double x=0.0,y=0.0;
  for(int a=0;a<4;++a){
    const auto X=pts[tc[pm.cell].v[a]];
    x+=l[a]*X.x;y+=l[a]*X.y;
  }
  const LocalSST ss=local_sst((Real)x,(Real)y,(Real)radius,(Real)nu,
      gu,kv,wv,gk,gw,(Real)kFloor,(Real)omegaFloor,stage20_distance(tc,pm.cell,l));
