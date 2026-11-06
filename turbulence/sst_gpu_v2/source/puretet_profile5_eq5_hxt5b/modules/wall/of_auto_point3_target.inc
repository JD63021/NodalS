  const Real normal=us[0]*(Real)f.n[0]+us[1]*(Real)f.n[1]+us[2]*(Real)f.n[2];
  const Real Ut=sqrt(fmax(Real(0),us[0]*us[0]+us[1]*us[1]+us[2]*us[2]-normal*normal));
  // Reuse the original HEX OF_AUTO viscous/log blend EXACTLY, including the
  // shared five-point Spalding u_tau. No independent k-based u_tau inversion.
  const OFWallState oa=of_auto_wall_state((Real)pm.y,(Real)nu,Ut,(Real)ut,
     kv,wv,ss,(Real)kFloor,(Real)omegaFloor,
     (Real)kap,(Real)ee,(Real)beta1,(Real)reBlend,
     (Real)prodLimit,1);
