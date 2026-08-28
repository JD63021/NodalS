struct Quad { std::array<double,4> lam; double w; };
static std::vector<Quad> tetDuffy7() {
  const double x[7]={-0.94910791234275852453,-0.74153118559939443986,-0.40584515137739716691,0.0,0.40584515137739716691,0.74153118559939443986,0.94910791234275852453};
  const double w[7]={0.12948496616886969327,0.27970539148927666790,0.38183005050511894495,0.41795918367346938776,0.38183005050511894495,0.27970539148927666790,0.12948496616886969327};
  std::vector<Quad> q; q.reserve(343);
  for(int ia=0;ia<7;++ia) for(int ib=0;ib<7;++ib) for(int ic=0;ic<7;++ic) {
    double A=.5*(x[ia]+1),B=.5*(x[ib]+1),C=.5*(x[ic]+1);
    double wa=.5*w[ia],wb=.5*w[ib],wc=.5*w[ic];
    double r=A,s=(1-A)*B,t=(1-A)*(1-B)*C;
    double jac=(1-A)*(1-A)*(1-B);
    q.push_back({{1-r-s-t,r,s,t},wa*wb*wc*jac});
  }
  return q;
}


static std::vector<Quad> tetDuffy5() {
  // Exact 5x5x5 collapsed-coordinate Gauss-Jacobi rule retained from the
  // earlier P1+BF3/P0 SUPG implementation.  r weights include (1-r)^2,
  // s weights include (1-s), and t is ordinary Gauss-Legendre on [0,1].
  const double rn[5]={0.034578939918215090,0.17348032077169567,0.38988638706551931,0.63433347263088680,0.85105421294701644};
  const double rw[5]={0.081764784285771011,0.12619896189991137,0.089200161221590066,0.032055600722961895,0.0041138252030990035};
  const double sn[5]={0.039809857051468722,0.19801341787360821,0.43797481024738616,0.69546427335363614,0.90146491420117358};
  const double sw[5]={0.096781590226651476,0.16717463809436969,0.14638698708466985,0.073908870072616678,0.015747914521692299};
  const double tn[5]={0.046910077030668018,0.23076534494715845,0.50000000000000000,0.76923465505284150,0.95308992296933193};
  const double tw[5]={0.11846344252809449,0.23931433524968326,0.28444444444444450,0.23931433524968326,0.11846344252809449};
  std::vector<Quad> q; q.reserve(125);
  for(int ir=0;ir<5;++ir) for(int is=0;is<5;++is) for(int it=0;it<5;++it) {
    const double r=rn[ir], ss=sn[is], t=tn[it];
    const double omr=1.0-r, oms=1.0-ss;
    q.push_back({{omr*oms*(1.0-t),r,omr*ss,omr*oms*t},rw[ir]*sw[is]*tw[it]});
  }
  return q;
}


static std::vector<Quad> tetSupg64() {
  // Positive-weight 4x4x4 collapsed Gauss-Jacobi rule.  This is the compact
  // SUPG rule previously used in the P1+BF3 lineage for the non-polynomial
  // tau(u) integrand.  It is intentionally distinct from the degree-8 Keast
  // rule below: all weights are positive, which is attractive for a
  // stabilization term whose coefficient tau is positive but non-polynomial.
  const double rn[4]={0.0485005494469972764,0.238600737551862341,0.517047295104367421,0.795851417896772828};
  const double rw[4]={0.110888415611277741,0.143458789799214448,0.0686338871729230970,0.0103522407499180812};
  const double sn[4]={0.0571041961145177246,0.276843013638123803,0.583590432368916834,0.860240135656219485};
  const double sw[4]={0.135506913431488518,0.203464568010271102,0.129847547608232333,0.0311809709500080849};
  const double tn[4]={0.0694318442029737137,0.330009478207571871,0.669990521792428129,0.930568155797026231};
  const double tw[4]={0.173927422568726897,0.326072577431273103,0.326072577431273103,0.173927422568726897};
  std::vector<Quad> q; q.reserve(64);
  for(int ir=0;ir<4;++ir) for(int is=0;is<4;++is) for(int it=0;it<4;++it) {
    const double r=rn[ir], ss=sn[is], t=tn[it];
    const double omr=1.0-r, oms=1.0-ss;
    q.push_back({{omr*oms*(1.0-t),r,omr*ss,omr*oms*t},rw[ir]*sw[is]*tw[it]});
  }
  return q;
}

static std::vector<Quad> tetKeast45() {
  // Keast degree-8 fully symmetric tetrahedral cubature.  This exactly
  // integrates total-degree <= 8 polynomials on the affine reference tet, but
  // the centroid orbit has a negative weight.  It is therefore retained as an
  // experimental SUPG choice; the positive-weight 64-point rule is the fast
  // default for the non-polynomial tau(u) integrand.
  const std::array<std::array<double,4>,7> generator{{
    {{0.250000000000000000,0.250000000000000000,0.250000000000000000,0.250000000000000000}},
    {{0.617587190300082967,0.127470936566639015,0.127470936566639015,0.127470936566639015}},
    {{0.903763508822103123,0.0320788303926322960,0.0320788303926322960,0.0320788303926322960}},
    {{0.0497770956432810185,0.0497770956432810185,0.450222904356718978,0.450222904356718978}},
    {{0.183730447398549945,0.183730447398549945,0.316269552601450060,0.316269552601450060}},
    {{0.231901089397150906,0.231901089397150906,0.0229177878448171174,0.513280033360881072}},
    {{0.0379700484718286102,0.0379700484718286102,0.730313427807538396,0.193746475248804382}},
  }};
  const std::array<double,7> orbitWeight{{
    -0.0393270066412926145,
     0.00408131605934270525,
     0.000658086773304341943,
     0.00438425882512284693,
     0.0138300638425098166,
     0.00424043742468372453,
     0.00223873973961420164,
  }};
  std::vector<Quad> q; q.reserve(45);
  for(int orbit=0;orbit<7;++orbit) {
    auto lambda=generator[orbit];
    std::sort(lambda.begin(),lambda.end());
    do { q.push_back({lambda,orbitWeight[orbit]}); }
    while(std::next_permutation(lambda.begin(),lambda.end()));
  }
  if(q.size()!=45) throw std::runtime_error("Keast45 quadrature point count mismatch");
  return q;
}

static std::vector<Quad> supgQuadrature(PetscInt n) {
  if(n==125) return tetDuffy5();
  if(n==64) return tetSupg64();
  if(n==45) return tetKeast45();
  throw std::runtime_error("SUPG quadrature must be 125, 64, or 45 points");
}

static void referenceHessian(const std::array<double,4>& l, double hess[8][3][3]) {
  const double gl[4][3]={{-1,-1,-1},{1,0,0},{0,1,0},{0,0,1}};
  for(int a=0;a<8;++a) for(int d=0;d<3;++d) for(int e=0;e<3;++e) hess[a][d][e]=0.0;
  // Exact Hessian of b_i = 27 prod_{j != i} lambda_j.  This is the same
  // ordered-pair formula used by the earlier P1+BF3/P0 SUPG kernels.
  for(int i=0;i<4;++i) {
    const int a=4+i;
    for(int d=0;d<3;++d) for(int e=0;e<3;++e) {
      double value=0.0;
      for(int j=0;j<4;++j) {
        if(j==i) continue;
        for(int k=0;k<4;++k) {
          if(k==i || k==j) continue;
          double term=gl[j][d]*gl[k][e];
          for(int m=0;m<4;++m) if(m!=i && m!=j && m!=k) term*=l[m];
          value+=term;
        }
      }
      hess[a][d][e]=27.0*value;
    }
  }
}

static double tetDiameter(const Vec3 X[4]) {
  double h=0.0;
  for(int a=0;a<4;++a) for(int b=a+1;b<4;++b) {
    const double dx=X[a].x-X[b].x,dy=X[a].y-X[b].y,dz=X[a].z-X[b].z;
    h=std::max(h,std::sqrt(dx*dx+dy*dy+dz*dz));
  }
  return h;
}

static void basis(const std::array<double,4>& l, double val[8], double gr[8][3]) {
  const double gl[4][3]={{-1,-1,-1},{1,0,0},{0,1,0},{0,0,1}};
  for(int i=0;i<4;++i){ val[i]=l[i]; for(int d=0;d<3;++d) gr[i][d]=gl[i][d]; }
  for(int i=0;i<4;++i) {
    int js[3],k=0; for(int j=0;j<4;++j) if(j!=i) js[k++]=j;
    val[4+i]=27*l[js[0]]*l[js[1]]*l[js[2]];
    for(int d=0;d<3;++d) gr[4+i][d]=0;
    for(int a=0;a<3;++a) {
      int j=js[a],o1=js[(a+1)%3],o2=js[(a+2)%3];
      for(int d=0;d<3;++d) gr[4+i][d]+=27*l[o1]*l[o2]*gl[j][d];
    }
  }
}

static void exactPressureGradient(double x,double y,double z,double gp[3]) {
  const double sx=sin(PI*x), sy=sin(PI*y), sz=sin(PI*z);
  const double cx=cos(PI*x), cy=cos(PI*y), cz=cos(PI*z);
  gp[0]=-PI*sx*cy*cz;
  gp[1]=-PI*cx*sy*cz;
  gp[2]=-PI*cx*cy*sz;
}

static void stokesForcingNu1(double x,double y,double z,double f[3]) {
  const double sx=sin(PI*x),sy=sin(PI*y),sz=sin(PI*z),cx=cos(PI*x),cy=cos(PI*y),cz=cos(PI*z);
  f[0]=PI*(24*PI*PI*sx*sx*sy*sz*sz - 4*PI*PI*sx*sx*sy - sx*cz - 4*PI*PI*sy*sz*sz)*cy;
  f[1]=PI*(-24*PI*PI*sx*sy*sy*sz*sz + 4*PI*PI*sx*sy*sy + 4*PI*PI*sx*sz*sz - sy*cz)*cx;
  f[2]=-PI*sz*cx*cy;
}

static void exactConvection(double x,double y,double z,double c[3]) {
  const double sx=sin(PI*x), sy=sin(PI*y), sz=sin(PI*z);
  const double cx=cos(PI*x), cy=cos(PI*y);
  const double fac=4.0*PI*PI*PI*sz*sz*sz*sz;
  c[0]=fac*sx*sx*sx*sy*sy*cx;
  c[1]=fac*sx*sx*sy*sy*sy*cy;
  c[2]=0.0;
}

static void forcing(double x,double y,double z,double nu,bool centralConvection,double f[3]) {
  double fs[3],gp[3],cv[3]={0,0,0};
  stokesForcingNu1(x,y,z,fs);
  exactPressureGradient(x,y,z,gp);
  if(centralConvection) exactConvection(x,y,z,cv);
  // fs = -Delta(u_exact) + grad(p_exact). General viscosity gives
  // f = -nu Delta(u_exact) + (u_exact.grad)u_exact + grad(p_exact).
  for(int d=0; d<3; ++d) f[d]=nu*(fs[d]-gp[d]) + gp[d] + cv[d];
}

static void exactU(double x,double y,double z,double u[3]) {
  double sx=sin(PI*x),sy=sin(PI*y),sz=sin(PI*z);
  u[0]=PI*sx*sx*sin(2*PI*y)*sz*sz;
  u[1]=-PI*sin(2*PI*x)*sy*sy*sz*sz;
  u[2]=0;
}
static double exactP(double x,double y,double z) { return cos(PI*x)*cos(PI*y)*cos(PI*z); }

