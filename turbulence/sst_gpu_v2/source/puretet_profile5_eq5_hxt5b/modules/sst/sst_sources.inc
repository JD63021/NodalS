  // SST-2003/OpenFOAM production uses dev(2S):grad(U).  For an exactly
  // divergence-free field this is identical to 2S:S.  Our FE velocity is
  // only weakly/cellwise divergence controlled, so retain the full metric
  // when requested instead of assuming div(U)=0 pointwise.
  const Real divU=gu[0]+gu[4]+gu[8];
  const Real prodMetric=c_sst_deviatoric_production
    ? fmax(Real(0),S2-Real(2.0/3.0)*divU*divU)
    : S2;
  const Real prodRaw=q.nut*prodMetric;
  const Real prodLim=Real(10*BETA_STAR)*k*om;
  q.pk=fmin(prodRaw,prodLim);
  q.reactK=Real(BETA_STAR)*om;
  q.reactOmega=q.beta*om; // beta*omega_old * omega_new
  Real omegaProd;
  const Real nutTiny=fmax(Real(1e-12)*nu,Real(1e-20));
  if(q.nut>nutTiny)omegaProd=q.gamma*q.pk/q.nut;
  else omegaProd=q.gamma*prodMetric;
  const Real cross=Real(2*SIGMA_W2)*(Real(1)-q.F1)*dotkw/om;
  q.omegaProductionAudit=omegaProd;q.omegaCrossAudit=cross;q.srcOmega=omegaProd+cross;
  return q;
