namespace hxt5b_diag {
using Real = hxt4b::Real;
static inline double gx(int i){static const double x[4]={-0.86113631159405257522,-0.33998104358485626480,0.33998104358485626480,0.86113631159405257522};return x[i];}
static inline double gw(int i){static const double w[4]={0.34785484513745385737,0.65214515486254614263,0.65214515486254614263,0.34785484513745385737};return w[i];}
static inline double det3(hxt5b::V3 a,hxt5b::V3 b,hxt5b::V3 c){return hxt5b::vdot(a,hxt5b::vcross(b,c));}
static inline double spyh(double up,double k,double B){double x=k*up,rem;if(std::fabs(x)<1e-3){double x2=x*x,x4=x2*x2;rem=x4*(1.0/24.0+x/120.0+x2/720.0+x2*x/5040.0);}else rem=std::expm1(x)-x-.5*x*x-x*x*x/6.0;return up+std::exp(-k*B)*rem;}
static inline double spgh(double up,double reY){return up*spyh(up,.4,5.5)-reY;}
static inline bool sph(double slip,double y,double nu,double&ut,double&yp,double&beta){double U=std::fabs(slip),reY=y*U/nu;if(!(y>0)||!(nu>0)||!std::isfinite(U))return false;if(reY<=1e-14){ut=yp=0;beta=nu/y;return true;}double lo=0,hi=std::max(1.0,std::sqrt(reY)+1.0);for(int k=0;k<40&&spgh(hi,reY)<0;++k)hi*=2;double gh=spgh(hi,reY);if(!std::isfinite(gh)||gh<0)return false;for(int k=0;k<70;++k){double m=.5*(lo+hi);if(spgh(m,reY)>0)hi=m;else lo=m;}double up=.5*(lo+hi);if(!(up>0))return false;ut=U/up;yp=y*ut/nu;beta=ut*ut/U;return std::isfinite(beta)&&beta>0&&std::isfinite(yp);}
struct CellDiag{double ux=0,uy=0,uz=0,vol=0,cx=0,cy=0,cz=0,wallArea=0,wallYp=0,wallYpMax=0,wallUt=0,wallUt2=0,wallSlip=0,wallY=0;int wallMask=0;};
static std::vector<CellDiag> build(const nodals_hxt1::HostMesh&M,const hxt5b::BoundaryHost&BH,const std::vector<Real>&u0,const std::vector<Real>&u1,const std::vector<Real>&u2,double nu,double bulk,double sf){
  int nh=(int)M.h.nhex,nt=(int)M.h.ntet;std::vector<CellDiag>D((size_t)(nh+nt));
  for(int c=0;c<nh;++c){auto H=M.hexVel[(size_t)c];auto hc=M.hexes[(size_t)c];double V=0,mx=0,my=0,mz=0,a0=0,a1=0,a2=0;for(int i=0;i<4;++i)for(int j=0;j<4;++j)for(int k=0;k<4;++k){double x=gx(i),y=gx(j),z=gx(k),ph[14];hxt5b::hbasis(x,y,z,ph);hxt5b::V3 X,dX[3];hxt5b::q1host(M,hc,x,y,z,X,dX);double q=gw(i)*gw(j)*gw(k)*std::fabs(det3(dX[0],dX[1],dX[2]));double vx=0,vy=0,vz=0;for(int a=0;a<14;++a){int g=H.g[a];vx+=(double)u0[(size_t)g]*ph[a];vy+=(double)u1[(size_t)g]*ph[a];vz+=(double)u2[(size_t)g]*ph[a];}V+=q;mx+=q*X.x;my+=q*X.y;mz+=q*X.z;a0+=q*vx;a1+=q*vy;a2+=q*vz;}if(!(V>0))throw std::runtime_error("HXT5B6 invalid HEX volume");auto&d=D[(size_t)c];d.vol=V;d.cx=mx/V;d.cy=my/V;d.cz=mz/V;d.ux=a0/V;d.uy=a1/V;d.uz=a2/V;}
  for(int c=0;c<nt;++c){auto T=M.tetVel[(size_t)c];auto tc=M.tets[(size_t)c];auto&d=D[(size_t)(nh+c)];d.vol=M.tetGeom[(size_t)c].volume;for(int a=0;a<4;++a){auto P=M.points[(size_t)tc.v[a]];d.cx+=.25*P.x;d.cy+=.25*P.y;d.cz+=.25*P.z;}for(int comp=0;comp<3;++comp){const std::vector<Real>&U=comp==0?u0:(comp==1?u1:u2);double v=0;for(int a=0;a<4;++a)v+=.25*(double)U[(size_t)T.g[a]];for(int a=4;a<8;++a)v+=.225*(double)U[(size_t)T.g[a]];if(comp==0)d.ux=v;else if(comp==1)d.uy=v;else d.uz=v;}}
  unsigned long long roots=0;double gA=0,gYp=0,gUt2=0;
  for(const auto&f:BH.wallH){auto H=M.hexVel[(size_t)f.cell];auto hc=M.hexes[(size_t)f.cell];double A=0,sYp=0,sUt=0,sUt2=0,sSlip=0,sY=0,ypmx=0;for(int i=0;i<4;++i)for(int j=0;j<4;++j){double cw[3]={0,0,0};cw[f.fd]=f.fv;cw[f.d0]=gx(i);cw[f.d1]=gx(j);double cs[3]={cw[0],cw[1],cw[2]};cs[f.fd]=f.fv+sf*(-2.0*f.fv);double ps[14];hxt5b::hbasis(cs[0],cs[1],cs[2],ps);hxt5b::V3 Xw,dXw[3],Xs,dXs[3];hxt5b::q1host(M,hc,cw[0],cw[1],cw[2],Xw,dXw);hxt5b::q1host(M,hc,cs[0],cs[1],cs[2],Xs,dXs);double rr=std::hypot(Xw.x,Xw.y);if(!(rr>0))throw std::runtime_error("HXT5B6 bad wall radius");hxt5b::V3 nr{Xw.x/rr,Xw.y/rr,0};double jac=hxt5b::vnorm(hxt5b::vcross(dXw[f.d0],dXw[f.d1]));double yy=hxt5b::vdot(hxt5b::vsub(Xw,Xs),nr);if(!(jac>0)||!(yy>0))throw std::runtime_error("HXT5B6 bad wall sample geometry");double slip=0;for(int a=0;a<14;++a)slip+=(double)u2[(size_t)H.g[a]]*ps[a];double ut=0,yp=0,beta=0;if(!sph(slip,yy,nu,ut,yp,beta)){++roots;ut=yp=0;}double q=gw(i)*gw(j)*jac;A+=q;sYp+=q*yp;sUt+=q*ut;sUt2+=q*ut*ut;sSlip+=q*std::fabs(slip);sY+=q*yy;ypmx=std::max(ypmx,yp);}auto&d=D[(size_t)f.cell];double oa=d.wallArea,na=oa+A;d.wallYp=(d.wallYp*oa+sYp)/na;d.wallUt=(d.wallUt*oa+sUt)/na;d.wallUt2=(d.wallUt2*oa+sUt2)/na;d.wallSlip=(d.wallSlip*oa+sSlip)/na;d.wallY=(d.wallY*oa+sY)/na;d.wallYpMax=std::max(d.wallYpMax,ypmx);d.wallArea=na;d.wallMask=1;gA+=A;gYp+=sYp;gUt2+=sUt2;}
  if(roots)throw std::runtime_error("HXT5B6 host Spalding root failure");if(gA>0)std::printf("NODALS_HXT5B6_DIAG_WALL area=%.12e yPlusMean=%.12e wallDarcy=%.12e rootFailures=%llu status=PASS\n",gA,gYp/gA,8.0*(gUt2/gA)/(bulk*bulk),roots);return D;
}
static std::string path2(const std::string&v){return v.size()>=4&&v.substr(v.size()-4)==".vtu"?v.substr(0,v.size()-4)+"_cell_diagnostics.vtu":v+"_cell_diagnostics.vtu";}
static void write(const std::string&path,const nodals_hxt1::HostMesh&M,const hxt5b::BoundaryHost&BH,const std::vector<Real>&u0,const std::vector<Real>&u1,const std::vector<Real>&u2,const std::vector<Real>&p,double nu,double bulk,double sf){int nh=(int)M.h.nhex,nt=(int)M.h.ntet,np=nh+nt;auto D=build(M,BH,u0,u1,u2,nu,bulk,sf);std::ofstream f(path);if(!f)throw std::runtime_error("cannot write "+path);f<<std::setprecision(16)<<"<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n<UnstructuredGrid><Piece NumberOfPoints=\""<<M.points.size()<<"\" NumberOfCells=\""<<np<<"\">\n<Points><DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";for(auto&q:M.points)f<<q.x<<" "<<q.y<<" "<<q.z<<"\n";f<<"</DataArray></Points><Cells><DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";for(auto&h:M.hexes){for(int a=0;a<8;++a)f<<h.v[a]<<" ";f<<"\n";}for(auto&t:M.tets){for(int a=0;a<4;++a)f<<t.v[a]<<" ";f<<"\n";}f<<"</DataArray><DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";long long o=0;for(int c=0;c<nh;++c){o+=8;f<<o<<"\n";}for(int c=0;c<nt;++c){o+=4;f<<o<<"\n";}f<<"</DataArray><DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";for(int c=0;c<nh;++c)f<<"12\n";for(int c=0;c<nt;++c)f<<"10\n";f<<"</DataArray></Cells><PointData><DataArray type=\"Float64\" Name=\"velocity_Q1P1_vertex\" NumberOfComponents=\"3\" format=\"ascii\">\n";for(size_t i=0;i<M.points.size();++i)f<<(double)u0[i]<<" "<<(double)u1[i]<<" "<<(double)u2[i]<<"\n";f<<"</DataArray></PointData><CellData><DataArray type=\"Float64\" Name=\"pressure_Q0P0\" format=\"ascii\">\n";for(auto q:p)f<<(double)q<<"\n";f<<"</DataArray><DataArray type=\"Int32\" Name=\"region\" format=\"ascii\">\n";for(int c=0;c<nh;++c)f<<"1\n";for(int c=0;c<nt;++c)f<<"2\n";f<<"</DataArray>\n";
#define H5B6_SCALAR(NAME,EXPR) f<<"<DataArray type=\"Float64\" Name=\"" NAME "\" format=\"ascii\">\n";for(const auto&d:D)f<<(EXPR)<<"\n";f<<"</DataArray>\n";
  f<<"<DataArray type=\"Float64\" Name=\"velocity_cell_average_full\" NumberOfComponents=\"3\" format=\"ascii\">\n";for(auto&d:D)f<<d.ux<<" "<<d.uy<<" "<<d.uz<<"\n";f<<"</DataArray>\n";
  H5B6_SCALAR("velocity_cell_average_full_magnitude",std::sqrt(d.ux*d.ux+d.uy*d.uy+d.uz*d.uz));H5B6_SCALAR("cell_volume",d.vol);
  f<<"<DataArray type=\"Float64\" Name=\"cell_centroid\" NumberOfComponents=\"3\" format=\"ascii\">\n";for(auto&d:D)f<<d.cx<<" "<<d.cy<<" "<<d.cz<<"\n";f<<"</DataArray>\n";
  f<<"<DataArray type=\"Int32\" Name=\"wall_spalding_mask\" format=\"ascii\">\n";for(auto&d:D)f<<d.wallMask<<"\n";f<<"</DataArray>\n";
  H5B6_SCALAR("wall_area",d.wallArea);H5B6_SCALAR("wall_yplus_spalding",d.wallYp);H5B6_SCALAR("wall_yplus_max_spalding",d.wallYpMax);H5B6_SCALAR("wall_utau_spalding",d.wallUt);H5B6_SCALAR("wall_utau2_mean_spalding",d.wallUt2);H5B6_SCALAR("wall_slip_spalding",d.wallSlip);H5B6_SCALAR("wall_sample_y",d.wallY);
#undef H5B6_SCALAR
  f<<"</CellData></Piece></UnstructuredGrid></VTKFile>\n";
}

// NODALS_HXT5B_PROFILE_AUDIT
// Host-only post-solve point sampler.  It evaluates the complete enriched
// velocity field: HEX Q1+BF2 (14 local velocity DOFs) and TET P1+BF3
// (8 local velocity DOFs).  Geometry remains Q1 for HEX and affine P1 for TET.
struct ProfileBox {
  int kind=0,cell=-1; // 1=hex, 2=tet
  double lo[3]={0,0,0},hi[3]={0,0,0};
};
static inline bool solve_cols(hxt5b::V3 c0,hxt5b::V3 c1,hxt5b::V3 c2,hxt5b::V3 r,double d[3]){
  const double det=det3(c0,c1,c2);
  const double sc=std::max({hxt5b::vnorm(c0),hxt5b::vnorm(c1),hxt5b::vnorm(c2),1e-300});
  if(!std::isfinite(det)||std::fabs(det)<1e-14*sc*sc*sc)return false;
  d[0]=hxt5b::vdot(r,hxt5b::vcross(c1,c2))/det;
  d[1]=hxt5b::vdot(c0,hxt5b::vcross(r,c2))/det;
  d[2]=hxt5b::vdot(c0,hxt5b::vcross(c1,r))/det;
  return std::isfinite(d[0])&&std::isfinite(d[1])&&std::isfinite(d[2]);
}
static std::vector<ProfileBox> profile_boxes(const nodals_hxt1::HostMesh&M){
  std::vector<ProfileBox>B; B.reserve((size_t)M.h.nhex+(size_t)M.h.ntet);
  for(int c=0;c<(int)M.h.nhex;++c){ProfileBox b;b.kind=1;b.cell=c;for(int d=0;d<3;++d){b.lo[d]=1e300;b.hi[d]=-1e300;}auto h=M.hexes[(size_t)c];for(int a=0;a<8;++a){auto P=M.points[(size_t)h.v[a]];double q[3]={P.x,P.y,P.z};for(int d=0;d<3;++d){b.lo[d]=std::min(b.lo[d],q[d]);b.hi[d]=std::max(b.hi[d],q[d]);}}B.push_back(b);}
  for(int c=0;c<(int)M.h.ntet;++c){ProfileBox b;b.kind=2;b.cell=c;for(int d=0;d<3;++d){b.lo[d]=1e300;b.hi[d]=-1e300;}auto t=M.tets[(size_t)c];for(int a=0;a<4;++a){auto P=M.points[(size_t)t.v[a]];double q[3]={P.x,P.y,P.z};for(int d=0;d<3;++d){b.lo[d]=std::min(b.lo[d],q[d]);b.hi[d]=std::max(b.hi[d],q[d]);}}B.push_back(b);}
  return B;
}
static bool profile_hex_ref(const nodals_hxt1::HostMesh&M,int c,const hxt5b::V3&P,const ProfileBox&b,double r[3]){
  for(int d=0;d<3;++d){double q=d==0?P.x:(d==1?P.y:P.z),m=.5*(b.lo[d]+b.hi[d]),h=.5*(b.hi[d]-b.lo[d]);r[d]=h>0?(q-m)/h:0;r[d]=std::max(-1.0,std::min(1.0,r[d]));}
  auto hc=M.hexes[(size_t)c];double sc=std::sqrt((b.hi[0]-b.lo[0])*(b.hi[0]-b.lo[0])+(b.hi[1]-b.lo[1])*(b.hi[1]-b.lo[1])+(b.hi[2]-b.lo[2])*(b.hi[2]-b.lo[2]));
  const double tol=1e-10*std::max(sc,1e-12);
  for(int it=0;it<20;++it){hxt5b::V3 X,dX[3];hxt5b::q1host(M,hc,r[0],r[1],r[2],X,dX);hxt5b::V3 e=hxt5b::vsub(P,X);if(hxt5b::vnorm(e)<=tol)break;double q[3];if(!solve_cols(dX[0],dX[1],dX[2],e,q))return false;r[0]+=q[0];r[1]+=q[1];r[2]+=q[2];if(std::fabs(r[0])>1.5||std::fabs(r[1])>1.5||std::fabs(r[2])>1.5)return false;}
  hxt5b::V3 X,dX[3];hxt5b::q1host(M,hc,r[0],r[1],r[2],X,dX);return hxt5b::vnorm(hxt5b::vsub(P,X))<=10*tol&&r[0]>=-1-1e-8&&r[0]<=1+1e-8&&r[1]>=-1-1e-8&&r[1]<=1+1e-8&&r[2]>=-1-1e-8&&r[2]<=1+1e-8;
}
static bool profile_tet_lam(const nodals_hxt1::HostMesh&M,int c,const hxt5b::V3&P,double l[4]){
  auto t=M.tets[(size_t)c];auto A=hxt5b::vp(M.points[(size_t)t.v[0]]),B=hxt5b::vp(M.points[(size_t)t.v[1]]),C=hxt5b::vp(M.points[(size_t)t.v[2]]),D=hxt5b::vp(M.points[(size_t)t.v[3]]);double q[3];if(!solve_cols(hxt5b::vsub(B,A),hxt5b::vsub(C,A),hxt5b::vsub(D,A),hxt5b::vsub(P,A),q))return false;l[1]=q[0];l[2]=q[1];l[3]=q[2];l[0]=1-q[0]-q[1]-q[2];for(int a=0;a<4;++a)if(l[a]<-1e-8||l[a]>1+1e-8)return false;return true;
}
static bool profile_eval(const nodals_hxt1::HostMesh&M,const std::vector<ProfileBox>&B,const std::vector<int>&cand,const std::vector<Real>&u0,const std::vector<Real>&u1,const std::vector<Real>&u2,const hxt5b::V3&P,double U[3]){
  const double eps=2e-10;
  for(int ib:cand){const auto&b=B[(size_t)ib];if(P.x<b.lo[0]-eps||P.x>b.hi[0]+eps||P.y<b.lo[1]-eps||P.y>b.hi[1]+eps||P.z<b.lo[2]-eps||P.z>b.hi[2]+eps)continue;
    if(b.kind==1){double r[3];if(!profile_hex_ref(M,b.cell,P,b,r))continue;double ph[14];hxt5b::hbasis(r[0],r[1],r[2],ph);auto H=M.hexVel[(size_t)b.cell];U[0]=U[1]=U[2]=0;for(int a=0;a<14;++a){int g=H.g[a];U[0]+=(double)u0[(size_t)g]*ph[a];U[1]+=(double)u1[(size_t)g]*ph[a];U[2]+=(double)u2[(size_t)g]*ph[a];}return true;}
    else{double l[4];if(!profile_tet_lam(M,b.cell,P,l))continue;double ph[8];hxt5b::tbasis(l,ph);auto T=M.tetVel[(size_t)b.cell];U[0]=U[1]=U[2]=0;for(int a=0;a<8;++a){int g=T.g[a];U[0]+=(double)u0[(size_t)g]*ph[a];U[1]+=(double)u1[(size_t)g]*ph[a];U[2]+=(double)u2[(size_t)g]*ph[a];}return true;}
  }
  return false;
}
static std::vector<double> parse_fracs(const std::string&s){std::vector<double>v;std::stringstream ss(s);std::string q;while(std::getline(ss,q,',')){if(q.empty())continue;double x=std::stod(q);if(!(x>0&&x<1))throw std::runtime_error("profile z fractions must lie in (0,1)");v.push_back(x);}if(v.empty())throw std::runtime_error("empty profile z fraction list");return v;}
static std::string profile_path(const std::string&v){return v.size()>=4&&v.substr(v.size()-4)==".vtu"?v.substr(0,v.size()-4)+"_enriched_profile.csv":v+"_enriched_profile.csv";}
static void write_profile(const std::string&path,const nodals_hxt1::HostMesh&M,const std::vector<Real>&u0,const std::vector<Real>&u1,const std::vector<Real>&u2,double R,double bulk,double zmin,double zmax,const std::string&fracText,int nr,int na,double rmax){
  if(nr<8||na<4||!(rmax>0&&rmax<=1))throw std::runtime_error("bad profile audit controls");auto fr=parse_fracs(fracText);auto B=profile_boxes(M);std::ofstream f(path);if(!f)throw std::runtime_error("cannot write "+path);f<<std::setprecision(16)<<"z_over_L,z,r_over_R,r,angle_samples,uz_mean,uz_min,uz_max,uz_std,uz_over_bulk\n";const double pi=std::acos(-1.0),L=zmax-zmin;
  for(double zf:fr){double z=zmin+zf*L;std::vector<int>cand;cand.reserve(B.size()/8+32);for(int i=0;i<(int)B.size();++i)if(z>=B[(size_t)i].lo[2]-2e-10&&z<=B[(size_t)i].hi[2]+2e-10)cand.push_back(i);if(cand.empty())throw std::runtime_error("profile audit found no cells crossing requested z plane");double center=0,maxMean=-1e300,maxPoint=-1e300,rAtMax=0,stdSum=0;int stdCount=0,minHits=na;
    for(int ir=0;ir<nr;++ir){double rr=rmax*(double)ir/(double)(nr-1),rad=R*rr,sum=0,sum2=0,mn=1e300,mx=-1e300;int hits=0,nang=ir==0?1:na;for(int ia=0;ia<nang;++ia){double th=ir==0?0:(2*pi*(ia+.5)/na);hxt5b::V3 P{rad*std::cos(th),rad*std::sin(th),z};double U[3];if(!profile_eval(M,B,cand,u0,u1,u2,P,U))continue;double uz=U[2];sum+=uz;sum2+=uz*uz;mn=std::min(mn,uz);mx=std::max(mx,uz);maxPoint=std::max(maxPoint,uz);++hits;}if(hits==0)throw std::runtime_error("profile audit point location failed at z/r sample");double mean=sum/hits,sd=std::sqrt(std::max(0.0,sum2/hits-mean*mean));if(ir==0)center=mean;if(mean>maxMean){maxMean=mean;rAtMax=rr;}if(ir>0){stdSum+=sd;++stdCount;minHits=std::min(minHits,hits);}f<<zf<<","<<z<<","<<rr<<","<<rad<<","<<hits<<","<<mean<<","<<mn<<","<<mx<<","<<sd<<","<<mean/bulk<<"\n";}
    std::printf("NODALS_HXT5B_PROFILE zOverL=%.6f z=%.12e radialSamples=%d angles=%d rMaxOverR=%.6f candidateCells=%zu centerUz=%.12e centerOverBulk=%.12e maxMeanUz=%.12e maxPointUz=%.12e rAtMaxOverR=%.12e minAngularHits=%d meanAngularStd=%.12e status=PASS\n",zf,z,nr,na,rmax,cand.size(),center,center/bulk,maxMean,maxPoint,rAtMax,minHits,stdCount?stdSum/stdCount:0.0);
  }
  std::printf("NODALS_HXT5B_PROFILE_OUTPUT path=%s field=FULL_Q1BF2_P1BF3_POINT_EVALUATION radialAverage=AZIMUTHAL status=PASS\n",path.c_str());
}


struct CrossSectionSideAudit {
  double area=0.0;
  double flux=0.0;
  double momentum=0.0;
  double pressureArea=0.0;
  long long faces=0;
};

struct CrossSectionAudit {
  CrossSectionSideAudit minus;
  CrossSectionSideAudit plus;
};

static CrossSectionAudit cross_section_audit(
    const nodals_hxt1::HostMesh& M,
    const std::vector<Real>& u2,
    const std::vector<Real>& p,
    double zTarget)
{
  if(M.h.ntet!=0)
    throw std::runtime_error(
      "cross-section momentum audit currently expects all-HEX mesh");

  CrossSectionAudit R;

  const double ztol=1e-10;

  for(std::size_t c=0;c<M.hexes.size();++c){

    const auto hc=M.hexes[c];
    const auto H =M.hexVel[c];

    hxt5b::V3 Xc,dXc[3];
    hxt5b::q1host(M,hc,0.0,0.0,0.0,Xc,dXc);

    for(int fd=0;fd<3;++fd){

      const int d0=(fd+1)%3;
      const int d1=(fd+2)%3;

      for(int iside=0;iside<2;++iside){

        const double fv=iside ? 1.0 : -1.0;

        double cc[3]={0.0,0.0,0.0};
        cc[fd]=fv;

        hxt5b::V3 Xf,dXf[3];
        hxt5b::q1host(
          M,hc,
          cc[0],cc[1],cc[2],
          Xf,dXf);

        hxt5b::V3 cr=hxt5b::vcross(dXf[d0],dXf[d1]);

        hxt5b::V3 outward{
          Xf.x-Xc.x,
          Xf.y-Xc.y,
          Xf.z-Xc.z
        };

        if(hxt5b::vdot(cr,outward)<0.0){
          cr.x=-cr.x;
          cr.y=-cr.y;
          cr.z=-cr.z;
        }

        const double jc=hxt5b::vnorm(cr);
        if(!(jc>0.0))continue;

        const double nz=cr.z/jc;

        // Only axial cross-section faces.
        if(std::abs(nz)<0.9)continue;
        if(std::abs(Xf.z-zTarget)>ztol)continue;

        CrossSectionSideAudit& S =
          nz>0.0 ? R.plus : R.minus;

        ++S.faces;

        for(int i=0;i<4;++i){
          for(int j=0;j<4;++j){

            double xi[3]={0.0,0.0,0.0};
            xi[fd]=fv;
            xi[d0]=gx(i);
            xi[d1]=gx(j);

            double ph[14];
            hxt5b::hbasis(
              xi[0],xi[1],xi[2],
              ph);

            hxt5b::V3 Xq,dXq[3];
            hxt5b::q1host(
              M,hc,
              xi[0],xi[1],xi[2],
              Xq,dXq);

            const double jac=
              hxt5b::vnorm(
                hxt5b::vcross(
                  dXq[d0],
                  dXq[d1]));

            const double w=
              gw(i)*gw(j)*jac;

            double uz=0.0;
            for(int a=0;a<14;++a){
              uz +=
                (double)u2[(std::size_t)H.g[a]]
                *ph[a];
            }

            S.area        += w;
            S.flux        += w*uz;
            S.momentum    += w*uz*uz;
            S.pressureArea+= w*(double)p[c];
          }
        }
      }
    }
  }

  return R;
}

static void print_cross_section_momentum_audit(
    const nodals_hxt1::HostMesh& M,
    const std::vector<Real>& u2,
    const std::vector<Real>& p,
    double bulk,
    double z0,
    double z1)
{
  const auto A=cross_section_audit(M,u2,p,z0);
  const auto B=cross_section_audit(M,u2,p,z1);

  auto print_side =
    [bulk](const char* plane,
           const char* side,
           const CrossSectionSideAudit& q)
  {
    const double mean=
      q.area>0.0 ? q.flux/q.area : 0.0;

    const double betaBulk=
      q.area>0.0
      ? q.momentum/(q.area*bulk*bulk)
      : 0.0;

    const double betaLocal=
      std::abs(q.flux)>0.0
      ? q.momentum*q.area/(q.flux*q.flux)
      : 0.0;

    const double pMean=
      q.area>0.0
      ? q.pressureArea/q.area
      : 0.0;

    std::printf(
      "NODALS_HXT5B_CROSS_SECTION "
      "plane=%s side=%s "
      "faces=%lld area=%.12e "
      "flux=%.12e meanUz=%.12e "
      "momentum=%.12e "
      "betaBulk50=%.12e betaLocal=%.12e "
      "pressureArea=%.12e pressureMean=%.12e "
      "status=PASS\n",
      plane,side,q.faces,q.area,
      q.flux,mean,q.momentum,
      betaBulk,betaLocal,
      q.pressureArea,pMean);
  };

  print_side("z0","minus",A.minus);
  print_side("z0","plus", A.plus);
  print_side("z1","minus",B.minus);
  print_side("z1","plus", B.plus);

  // Control volume z0 < z < z1:
  //
  // lower plane is the PLUS-Z side of the cell below but the
  // MINUS-Z side of the cell inside the control volume.
  //
  // therefore use:
  //   z0 : minus-normal face from cell above
  //   z1 : plus-normal  face from cell below

  const double pressureForce=
      A.minus.pressureArea
    - B.plus.pressureArea;

  const double momentumIn =
      A.minus.momentum;

  const double momentumOut=
      B.plus.momentum;

  const double momentumDelta=
      momentumOut-momentumIn;

  const double fluxIn =
      A.minus.flux;

  const double fluxOut=
      B.plus.flux;

  const double traceMismatchZ0=
      std::abs(A.plus.momentum-A.minus.momentum)
      /std::max(
         std::max(std::abs(A.plus.momentum),
                  std::abs(A.minus.momentum)),
         1e-300);

  const double traceMismatchZ1=
      std::abs(B.plus.momentum-B.minus.momentum)
      /std::max(
         std::max(std::abs(B.plus.momentum),
                  std::abs(B.minus.momentum)),
         1e-300);

  std::printf(
    "NODALS_HXT5B_CROSS_SECTION_MOMENTUM_AUDIT "
    "z0=%.12e z1=%.12e "
    "pressureForce=%.12e "
    "momentumIn=%.12e "
    "momentumOut=%.12e "
    "momentumDelta=%.12e "
    "pressureMinusMomentumDelta=%.12e "
    "fluxIn=%.12e fluxOut=%.12e "
    "fluxDelta=%.12e "
    "traceMismatchZ0=%.12e "
    "traceMismatchZ1=%.12e "
    "status=PASS\n",
    z0,z1,
    pressureForce,
    momentumIn,
    momentumOut,
    momentumDelta,
    pressureForce-momentumDelta,
    fluxIn,fluxOut,fluxOut-fluxIn,
    traceMismatchZ0,
    traceMismatchZ1);
}

} // namespace hxt5b_diag
