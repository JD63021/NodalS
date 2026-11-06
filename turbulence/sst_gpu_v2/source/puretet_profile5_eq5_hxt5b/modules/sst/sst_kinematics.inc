  LocalSST q;
  q.kFloored=!(kRaw>kFloor);q.omegaFloored=!(omegaRaw>omegaFloor);
  const Real k=fmax(kRaw,kFloor),om=fmax(omegaRaw,omegaFloor);
  Real ss=0;
  #pragma unroll
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){Real sij=Real(0.5)*(gu[3*i+j]+gu[3*j+i]);ss+=sij*sij;}
  const Real S2=fmax(Real(0),Real(2)*ss),S=sqrt(S2);
  q.strain=S;
