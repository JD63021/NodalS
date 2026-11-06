  // Experimental TET trace-log omega closure: SAME sample #3 log target,
  // SAME full-vector five-sample Spalding u_tau, but enforce on the physical
  // wall trace with symmetric Nitsche (not with interior rank-one penalty).
  // ν_eff is evaluated at the existing interior carrier state (sample #3).
  // The wall-face tetrahedron defines the trace and its normal gradients.
  if(equation!=OMEGA_EQ)return;
  const auto wallT=tv[f.cell];
  const int face0=hxt4a::h4_tet_face[f.lf][0];
  const int face1=hxt4a::h4_tet_face[f.lf][1];
  const int face2=hxt4a::h4_tet_face[f.lf][2];
  double wl[4]={0.,0.,0.,0.};
  wl[face0]=hxt5b::QL0[q];
  wl[face1]=hxt5b::QL1[q];
  wl[face2]=hxt5b::QL2[q];
  double wph[8],wgr[8][3],wgp[8][3],normalGrad[8];
  hxt2::tet_basis(wl[0],wl[1],wl[2],wl[3],wph,wgr);
  hxt2::tet_phys_grad(wgr,tg[f.cell],wgp);
  const double hTet=(double)tg[f.cell].det/(2.0*f.area); // 3*volume/face-area
  const double mu=(double)nu+(double)ss.sigmaW*(double)ss.nut;
  const double tau=stage41_trace_gamma*mu/hTet;
  const double wallW=(double)hxt5b::QW12[q]*f.area;
  if(!(isfinite(hTet)&&hTet>0.&&isfinite(mu)&&mu>0.&&
       isfinite(tau)&&tau>0.&&isfinite(wallW)&&wallW>0.&&
       isfinite((double)stage41OmegaTarget)&&stage41OmegaTarget>Real(0))){
    atomicAdd(bad,1ULL);return;
  }
  double omegaAtTrace=0.;
  for(int a=0;a<8;++a){
    if(wallT.g[a]<0){atomicAdd(bad,1ULL);return;}
    omegaAtTrace+=(double)omega[wallT.g[a]]*wph[a];
    normalGrad[a]=wgp[a][0]*f.n[0]+wgp[a][1]*f.n[1]+wgp[a][2]*f.n[2];
  }
  // -<mu d_n w, v> - <mu d_n v, w> + <gamma mu/h w,v>
  // RHS = -<mu d_n v, g> + <gamma mu/h g,v>.
  // Each q thread contributes one 8x8 boundary-element matrix block.
  for(int a=0;a<8;++a){
    for(int b=0;b<8;++b){
      const double z=wallW*(
        -mu*normalGrad[b]*wph[a] - mu*normalGrad[a]*wph[b]
        +tau*wph[a]*wph[b]);
      atomicAdd(mat+slot[(std::size_t)f.cell*64ull+(std::size_t)(a*8+b)],(Real)z);
    }
    const double b=wallW*(-mu*normalGrad[a]+tau*wph[a])*(double)stage41OmegaTarget;
    atomicAdd(rhs+wallT.g[a],(Real)b);
  }
  if(wallStats){
    // Preserve the existing omega diagnostic normalization (wall area*slab).
    const double vweight=wallW*ell;
    atomicAdd(wallStats+0,vweight);
    atomicAdd(wallStats+1,(double)stage41OmegaTarget*vweight);
    atomicAdd(wallStats+2,pm.y*ut/nu*vweight);
    atomicAdd(wallStats+3,(double)wv*vweight);
    atomicAdd(wallStats+4,omegaAtTrace*vweight);
    atomicAdd(wallStats+5,(double)oa.rey*vweight);
    atomicAdd(wallStats+6,(double)oa.lamFrac*vweight);
    atomicAdd(wallStats+7,(double)oa.omegaVis*vweight);
    atomicAdd(wallStats+8,(double)oa.omegaLog*vweight);
    atomicAdd(wallStats+9,(double)ss.pk*vweight);
    atomicAdd(wallStats+10,(double)oa.pkAuto*vweight);
    atomicAdd(wallStats+11,(double)(oa.nutSpalding/(Real)nu)*vweight);
    atomicAdd(wallStats+12,(double)(ss.nut/(Real)nu)*vweight);
  }
