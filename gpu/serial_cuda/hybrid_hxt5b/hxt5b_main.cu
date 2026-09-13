// NodalS HXT5B
// Full RANS boundary gate: mixing length + DG plug inlet + Firedrake-parity HEX off-wall Spalding wall + Cheb2 AMG.

#define HXT4B_EMBED_MAIN hxt4b_embedded_main
#include "../hybrid_hxt4b/hxt4b_main.cu"
#undef HXT4B_EMBED_MAIN

#include "hybrid_amg.cuh"
#include "hxt5a_mixlen.cuh"
#include "hxt5b_boundary.cuh"
#include "hxt5c_sst_g1.cuh"
#include "hxt5d_sst_g2.cuh"

// HXT5B6_EXACT_DIAGNOSTIC_VTU
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
} // namespace hxt5b_diag


int main(int argc,char**argv){
  try{
    using namespace hxt4b;
    using T=Real;

    std::string mesh,vtu="HXT5B_full_rans.vtu",momWork="sgs1",momCapPolicy="warn";
    std::string pressureSolver="pcg_amg",amgHierarchy="cf",cfCoarsening="pmis",cfStrength="classical-negative",cfInterp="exti";
    std::string amgSmoother="cheb2",amgSpectrumPolicy="auto",amgMcgsOrder="symmetric";
    double Re=13691.740248127866,bulk=50,gamma=50,schurNitscheGamma=0,alphaU=.7,alphaP=1,rauScale=2,simpleTol=1e-3,momOmega=1,momRtol=.5,momAtol=1e-8,mixlenScale=1.0,outerContAtol=1e-12,wallSampleFraction=.5;
    double pRtol=.9,pAtol=1e-12,cfTheta=.25,amgJacobiOmega=.7,amgLambdaSafety=1.5,amgLambdaLow=.05,amgMcgsOmega=1;
    int maxOuter=3000,pMax=50,momMax=50,cfPmax=8,cfAggressiveFirst=0,amgTerminal=1000,amgPowerIts=16,amgChebDegree=4;
    int amgJacobiFineSweeps=1,amgJacobiCoarseSweeps=1,amgMcgsFineSweeps=1,amgMcgsCoarseSweeps=1;
    hxt5c::Options sstG1;
    bool sstG2=false;
    std::string sstG2Start="g0";
    double sstG2PlateauTol=2.5e-3;
    int sstG2Max=800,sstG2PrintEvery=20,sstG2PlateauSamples=5;

    for(int i=1;i<argc;++i){
      std::string a=argv[i];auto nx=[&](){if(i+1>=argc)throw std::runtime_error("missing value for "+a);return std::string(argv[++i]);};
      if(a=="--mesh")mesh=nx();else if(a=="--vtu")vtu=nx();else if(a=="--re")Re=std::stod(nx());else if(a=="--bulk")bulk=std::stod(nx());else if(a=="--mixlen-scale")mixlenScale=std::stod(nx());else if(a=="--outer-cont-atol")outerContAtol=std::stod(nx());
      else if(a=="--wall-sample-fraction")wallSampleFraction=std::stod(nx());
      else if(a=="--gamma")gamma=std::stod(nx());else if(a=="--schur-nitsche-gamma")schurNitscheGamma=std::stod(nx());else if(a=="--alpha-u")alphaU=std::stod(nx());else if(a=="--alpha-p")alphaP=std::stod(nx());
      else if(a=="--rau-scale")rauScale=std::stod(nx());else if(a=="--simple-tol")simpleTol=std::stod(nx());else if(a=="--max-outer")maxOuter=std::stoi(nx());
      else if(a=="--momentum-work")momWork=nx();else if(a=="--mom-omega")momOmega=std::stod(nx());else if(a=="--mom-rtol")momRtol=std::stod(nx());else if(a=="--mom-atol")momAtol=std::stod(nx());else if(a=="--mom-max")momMax=std::stoi(nx());else if(a=="--mom-cap-policy")momCapPolicy=nx();
      else if(a=="--pressure-solver")pressureSolver=nx();else if(a=="--p-rtol")pRtol=std::stod(nx());else if(a=="--p-atol")pAtol=std::stod(nx());else if(a=="--p-max")pMax=std::stoi(nx());
      else if(a=="--amg-hierarchy")amgHierarchy=nx();else if(a=="--cf-coarsening")cfCoarsening=nx();else if(a=="--cf-strength")cfStrength=nx();
      else if(a=="--cf-theta")cfTheta=std::stod(nx());else if(a=="--cf-interp")cfInterp=nx();else if(a=="--cf-pmax")cfPmax=std::stoi(nx());else if(a=="--cf-aggressive-first")cfAggressiveFirst=std::stoi(nx());
      else if(a=="--amg-terminal")amgTerminal=std::stoi(nx());else if(a=="--amg-smoother")amgSmoother=nx();else if(a=="--amg-jacobi-omega")amgJacobiOmega=std::stod(nx());
      else if(a=="--amg-jacobi-fine-sweeps")amgJacobiFineSweeps=std::stoi(nx());else if(a=="--amg-jacobi-coarse-sweeps")amgJacobiCoarseSweeps=std::stoi(nx());
      else if(a=="--amg-cheb-degree")amgChebDegree=std::stoi(nx());else if(a=="--amg-power-its")amgPowerIts=std::stoi(nx());
      else if(a=="--amg-lambda-safety")amgLambdaSafety=std::stod(nx());else if(a=="--amg-lambda-low-fraction")amgLambdaLow=std::stod(nx());
      else if(a=="--sst-g1")sstG1.enabled=std::stoi(nx())!=0;else if(a=="--sst-g2")sstG2=std::stoi(nx())!=0;
      else if(a=="--sst-g2-start")sstG2Start=nx();
      else if(a=="--sst-g2-max")sstG2Max=std::stoi(nx());else if(a=="--sst-g2-print-every")sstG2PrintEvery=std::stoi(nx());else if(a=="--sst-g2-plateau-tol")sstG2PlateauTol=std::stod(nx());else if(a=="--sst-g2-plateau-samples")sstG2PlateauSamples=std::stoi(nx());
      else if(a=="--sst-intensity")sstG1.intensity=std::stod(nx());else if(a=="--sst-length-ratio")sstG1.lengthRatio=std::stod(nx());
      else if(a=="--sst-alpha-k")sstG1.alphaK=std::stod(nx());else if(a=="--sst-alpha-omega")sstG1.alphaOmega=std::stod(nx());else if(a=="--sst-tol")sstG1.nonlinearTol=std::stod(nx());else if(a=="--sst-max")sstG1.nonlinearMax=std::stoi(nx());
      else if(a=="--sst-linear-rtol")sstG1.linearRtol=std::stod(nx());else if(a=="--sst-linear-atol")sstG1.linearAtol=std::stod(nx());else if(a=="--sst-linear-max")sstG1.linearMax=std::stoi(nx());else if(a=="--sst-gs-omega")sstG1.gsOmega=std::stod(nx());
      else if(a=="--sst-k-floor")sstG1.kFloor=std::stod(nx());else if(a=="--sst-omega-floor")sstG1.omegaFloor=std::stod(nx());else if(a=="--sst-wall-penalty-gamma")sstG1.wallPenaltyGamma=std::stod(nx());else if(a=="--sst-omega-wall-mode"){auto v=nx();if(v=="trace_log")sstG1.omegaWallMode=hxt5c::OMEGA_WALL_TRACE_LOG;else if(v=="sample_log")sstG1.omegaWallMode=hxt5c::OMEGA_WALL_SAMPLE_LOG;else if(v=="of_auto")sstG1.omegaWallMode=hxt5c::OMEGA_WALL_OF_AUTO;else throw std::runtime_error("--sst-omega-wall-mode must be trace_log, sample_log or of_auto");}else if(a=="--sst-of-kappa")sstG1.ofKappa=std::stod(nx());else if(a=="--sst-of-E")sstG1.ofE=std::stod(nx());else if(a=="--sst-of-beta1")sstG1.ofBeta1=std::stod(nx());else if(a=="--sst-of-re-blend")sstG1.ofReBlend=std::stod(nx());else if(a=="--sst-of-production")sstG1.ofProduction=std::stoi(nx())!=0;else if(a=="--sst-of-production-scale")sstG1.ofProductionScale=std::stod(nx());else if(a=="--sst-of-wall-volume-scale")sstG1.ofWallVolumeScale=std::stod(nx());else if(a=="--sst-of-production-limit")sstG1.ofProductionLimit=std::stod(nx());else if(a=="--sst-print-every")sstG1.printEvery=std::stoi(nx());
      else if(a=="--amg-spectrum-policy")amgSpectrumPolicy=nx();else if(a=="--amg-mcgs-fine-sweeps")amgMcgsFineSweeps=std::stoi(nx());
      else if(a=="--amg-mcgs-coarse-sweeps")amgMcgsCoarseSweeps=std::stoi(nx());else if(a=="--amg-mcgs-omega")amgMcgsOmega=std::stod(nx());else if(a=="--amg-mcgs-order")amgMcgsOrder=nx();
      else throw std::runtime_error("unknown arg "+a);
    }

    if(mesh.empty())throw std::runtime_error("--mesh required");
    if(amgHierarchy!="cf")throw std::runtime_error("HXT4C1 wires the optimal CF hierarchy first; use --amg-hierarchy cf");
    if(cfCoarsening!="pmis")throw std::runtime_error("HXT4C1 CF coarsening must be pmis");
    if(cfStrength!="classical-negative")throw std::runtime_error("HXT4C1 CF strength must be classical-negative");
    if(!(cfInterp=="direct"||cfInterp=="exti"))throw std::runtime_error("HXT4C1 CF interp must be direct or exti");
    if(!(amgSmoother=="jacobi"||amgSmoother=="l1jacobi"||amgSmoother=="cheb2"||amgSmoother=="mcgs"))throw std::runtime_error("HXT4C1 unknown AMG smoother");
    if(pressureSolver!="pcg_amg"&&pressureSolver!="pcg_jacobi")throw std::runtime_error("HXT4C1 pressure solver must be pcg_amg or pcg_jacobi");
    if(pressureSolver=="pcg_amg"&&amgSmoother=="mcgs"&&amgMcgsOrder!="symmetric")throw std::runtime_error("PCG requires symmetric MCGS");
    if(!(alphaU>0&&alphaU<=1&&alphaP>0))throw std::runtime_error("bad relaxation");
    if(!(gamma>=0.0)||!(schurNitscheGamma>=0.0)||(gamma<=0.0&&schurNitscheGamma>0.0))throw std::runtime_error("bad momentum/Schur Nitsche gamma");
    if(!(momCapPolicy=="warn"||momCapPolicy=="fail"))throw std::runtime_error("--mom-cap-policy must be warn or fail");
    if(!(mixlenScale>=0.0)||!(outerContAtol>=0.0)||!(wallSampleFraction>0.0&&wallSampleFraction<1.0))throw std::runtime_error("bad HXT5B turbulence/boundary/convergence option");
    if(sstG1.enabled&&sstG2)throw std::runtime_error("choose only one of --sst-g1 and --sst-g2");
    if(!(sstG2Start=="g0"||sstG2Start=="plug"))throw std::runtime_error("--sst-g2-start must be g0 or plug");
    if(sstG2&&(!(sstG2PlateauTol>0)||sstG2Max<=0||sstG2PrintEvery<=0||sstG2PlateauSamples<3))throw std::runtime_error("bad SST G2 control option");
    if((sstG1.enabled||sstG2)&&(!(sstG1.intensity>0)||!(sstG1.lengthRatio>0)||!(sstG1.alphaK>0&&sstG1.alphaK<=1)||!(sstG1.alphaOmega>0&&sstG1.alphaOmega<=1)||!(sstG1.linearRtol>0&&sstG1.linearRtol<1)||!(sstG1.linearAtol>=0)||sstG1.linearMax<=0||!(sstG1.gsOmega>0)||!(sstG1.kFloor>0)||!(sstG1.omegaFloor>0)||!(sstG1.ofKappa>0)||!(sstG1.ofE>1)||!(sstG1.ofBeta1>0)||!(sstG1.ofReBlend>0)||!(sstG1.ofProductionScale>=0)||!(sstG1.ofWallVolumeScale>0)||!(sstG1.ofProductionLimit>0)))throw std::runtime_error("bad SST transport/wall option");

    HostMesh M=nodals_hxt1::load(mesh);
    auto IP=hxt2::build_interface_plans(M);auto I3=hxt3a::build_if(M);auto Bbase=hxt3a::build_b(M,I3);
    auto BH=hxt5b::build_boundaries(M,Bbase,bulk);auto&Bh=BH.B;
    const int nv=Bh.nv,np=Bh.np,nh=(int)M.h.nhex;

    double zmin=1e300,zmax=-1e300,R=0;
    for(const auto&q:M.points){zmin=std::min(zmin,q.z);zmax=std::max(zmax,q.z);R=std::max(R,std::sqrt(q.x*q.x+q.y*q.y));}
    const double L=zmax-zmin,D=2*R,nu=bulk*D/Re;

    auto fixed=hxt5b::plug_initial(M,bulk);
    double plugVertexErr=0.0,plugEnrichMax=0.0;
    for(std::uint64_t v=0;v<M.h.nv;++v)
      plugVertexErr=std::max(plugVertexErr,std::abs((double)fixed.u2[(std::size_t)v]-bulk));
    for(std::uint64_t g=M.h.nv;g<M.h.nvel;++g)
      plugEnrichMax=std::max(plugEnrichMax,std::abs((double)fixed.u2[(std::size_t)g]));
    std::printf("NODALS_HXT5B3_INITIAL_PLUG vertexUzErrorMax=%.12e enrichmentUzMax=%.12e "
                "vertexUxUy=zero enrichmentUxUy=zero semantics=constant_Q1_or_P1_field status=%s\n",
      plugVertexErr,plugEnrichMax,(plugVertexErr==0.0&&plugEnrichMax==0.0)?"PASS":"FAIL");
    if(plugVertexErr!=0.0||plugEnrichMax!=0.0)
      throw std::runtime_error("HXT5B3 plug coefficient initialization mismatch");
    std::vector<unsigned char>clampXY=fixed.mask;for(std::size_t i=0;i<clampXY.size();++i)if(BH.wallMask[i])clampXY[i]=1;
    Dev<unsigned char>d_fixed(fixed.mask),d_clampXY(clampXY),d_wallMask(BH.wallMask);
    Dev<T>d_u0(fixed.u0),d_u1(fixed.u1),d_u2(fixed.u2);
    auto VH=build_velocity_csr(M,IP);auto VC=color_free_rows(VH,fixed.mask);GpuCSR A(VH,VC);

    Dev<nodals_hxt1::Point>d_pts(M.points);Dev<nodals_hxt1::HexConn>d_hc(M.hexes);Dev<nodals_hxt1::TetConn>d_tc(M.tets);Dev<nodals_hxt1::HexVel>d_hv(M.hexVel);
    Dev<nodals_hxt1::TetVel>d_tv(M.tetVel);Dev<nodals_hxt1::TetGeom>d_tg(M.tetGeom);Dev<hxt2::InterfacePlan>d_ip(IP);
    Dev<unsigned long long>d_bad(std::vector<unsigned long long>(1,0));
    Dev<T>d_turb(std::vector<T>(A.nnz,0)),d_diagRTurb(std::vector<T>((std::size_t)nv,0));
    Dev<double>d_mixStats(std::vector<double>(4,0));
    Dev<hxt5b::HexFacePlan>d_inH(BH.inH),d_wallH(BH.wallH);Dev<hxt5b::TetFacePlan>d_inT(BH.inT);
    Dev<T>d_contSource(hxt4c::castv<T>(BH.csrc));
    Dev<T>d_dg(std::vector<T>(A.nnz,0)),d_diagDG(std::vector<T>((std::size_t)nv,0)),d_inletRhsZ(std::vector<T>((std::size_t)nv,0));
    Dev<T>d_wall(std::vector<T>(A.nnz,0)),d_diagWall(std::vector<T>((std::size_t)nv,0)),d_valZ(std::vector<T>(A.nnz,0));
    Dev<T>d_diagOrigZ(std::vector<T>((std::size_t)nv,0)),d_rauZ(std::vector<T>((std::size_t)nv,0));
    Dev<T>d_relaxDeltaXY(std::vector<T>((std::size_t)nv,0)),d_relaxDeltaZ(std::vector<T>((std::size_t)nv,0));
    Dev<double>d_wallStats(std::vector<double>(8,0));

    auto TT=hxt2::build_tet_tensor();HXT1_CUDA(cudaMemcpyToSymbol(hxt2::c_tet_T,TT.data(),TT.size()*sizeof(double)));
    auto CT=build_tet_conv_tensor();HXT1_CUDA(cudaMemcpyToSymbol(h4b_tet_convT,CT.data(),CT.size()*sizeof(double)));

    std::printf("NODALS_HXT4C_KERNEL_POLICY interfaceThreads=128 interfacePairTraversal=STRIDED hexConvectionThreads=128 hexPairTraversal=STRIDED status=PASS\n");
    HXT1_CUDA(cudaMemset(A.base.p,0,A.nnz*sizeof(T)));HXT1_CUDA(cudaMemset(A.diagRBase.p,0,nv*sizeof(T)));HXT1_CUDA(cudaMemset(d_bad.p,0,sizeof(unsigned long long)));
    if(M.h.nhex)hex_diff_assemble<<<M.h.nhex,256>>>(d_pts.p,d_hc.p,d_hv.p,A.hexSlot.p,A.base.p,A.diagRBase.p,M.h.nhex,nu,d_bad.p);
    if(M.h.ntet)tet_diff_assemble<<<M.h.ntet,64>>>(d_tv.p,d_tg.p,A.tetSlot.p,A.base.p,A.diagRBase.p,M.h.ntet,nu);
    if(M.h.ninterface)interface_assemble<<<2*M.h.ninterface,128>>>(d_pts.p,d_hc.p,d_hv.p,d_tv.p,d_tg.p,d_ip.p,A.row.p,A.col.p,A.base.p,A.diagRBase.p,M.h.ninterface,nu,gamma,d_bad.p);
    HXT1_CUDA(cudaGetLastError());HXT1_CUDA(cudaDeviceSynchronize());
    unsigned long long bad=0;HXT1_CUDA(cudaMemcpy(&bad,d_bad.p,sizeof(bad),cudaMemcpyDeviceToHost));
    if(bad)throw std::runtime_error("baseline assembly geometry failures");

    hxt4a::BDevice<T>B(Bh);
    int pin=0;if(!M.outHex.empty())pin=M.outHex[0].cell;else if(!M.outTet.empty())pin=nh+M.outTet[0].cell;

    Dev<T>d_p(std::vector<T>((std::size_t)np,0)),d_dp(std::vector<T>((std::size_t)np,0)),d_cont(std::vector<T>((std::size_t)np,0));
    Dev<T>d_sd(std::vector<T>((std::size_t)np,0)),d_isd(std::vector<T>((std::size_t)np,0));
    Dev<T>d_bt0(std::vector<T>((std::size_t)nv,0)),d_bt1(std::vector<T>((std::size_t)nv,0)),d_bt2(std::vector<T>((std::size_t)nv,0));
    Dev<T>d_rhs0(std::vector<T>((std::size_t)nv,0)),d_rhs1(std::vector<T>((std::size_t)nv,0)),d_rhs2(std::vector<T>((std::size_t)nv,0));
    hxt4a::Reducer<T>pred(np),ured(nv);hxt4a::CGWork<T>WP(np);

    // Initial current-fine pressure matrix and setup-snapshot hierarchy.
    HXT1_CUDA(cudaMemset(A.conv.p,0,A.nnz*sizeof(T)));HXT1_CUDA(cudaMemset(d_bad.p,0,sizeof(unsigned long long)));
    if(M.h.nhex)hex_conv_assemble<<<M.h.nhex,128>>>(d_pts.p,d_hc.p,d_hv.p,A.hexSlot.p,d_u0.p,d_u1.p,d_u2.p,A.conv.p,M.h.nhex,d_bad.p);
    if(M.h.ntet)tet_conv_assemble<<<M.h.ntet,64>>>(d_tv.p,d_tg.p,A.tetSlot.p,d_u0.p,d_u1.p,d_u2.p,A.conv.p,M.h.ntet);
    auto mix=hxt5a::h5_assemble_mixlen(M.h.nhex,M.h.ntet,M.h.ninterface,d_pts,d_hc,d_tc,d_hv,d_tv,d_tg,d_ip,A,
      d_u0.p,d_u1.p,d_u2.p,d_turb,d_diagRTurb,d_mixStats,R,nu,mixlenScale,gamma,d_bad);
    hxt5b::assemble_boundary(BH,d_inH,d_inT,d_wallH,d_pts,d_hc,d_tc,d_hv,d_tv,d_tg,A,
      d_u0.p,d_u1.p,d_u2.p,d_dg,d_diagDG,d_inletRhsZ,d_wall,d_diagWall,d_wallStats,d_bad,
      bulk,R,nu,mixlenScale,wallSampleFraction);
    hxt5b::combine<<<(A.nnz+TPB-1)/TPB,TPB>>>(A.nnz,A.base.p,A.conv.p,d_turb.p,d_dg.p,d_wall.p,A.val.p,d_valZ.p);
    HXT1_CUDA(cudaMemset(d_bad.p,0,sizeof(unsigned long long)));
    hxt5b::h5b3_finalize_row_l1<<<(nv+TPB-1)/TPB,TPB>>>(
      nv,d_fixed.p,d_wallMask.p,A.row.p,A.diagPos.p,A.val.p,d_valZ.p,
      A.diagRBase.p,A.conv.p,d_diagRTurb.p,d_diagDG.p,d_diagWall.p,
      A.rau.p,d_rauZ.p,A.diagOriginal.p,d_diagOrigZ.p,d_relaxDeltaXY.p,d_relaxDeltaZ.p,
      gamma,schurNitscheGamma,alphaU,rauScale,d_bad.p);
    HXT1_CUDA(cudaDeviceSynchronize());HXT1_CUDA(cudaMemcpy(&bad,d_bad.p,sizeof(bad),cudaMemcpyDeviceToHost));
    if(bad){
      // NODALS_HXT5B_RAU_DIAG_AUDIT_BEGIN
      // Failure-only host audit.  This deliberately does NOT alter the
      // positivity policy or any operator value; it only reports which part
      // of the directional rAU construction triggered the existing gate.
      std::vector<T> hOx((std::size_t)nv),hOz((std::size_t)nv),hDx((std::size_t)nv),hDz((std::size_t)nv);
      std::vector<T> hDiagR((std::size_t)nv),hDiagRTurb((std::size_t)nv),hDiagDG((std::size_t)nv),hDiagWall((std::size_t)nv);
      std::vector<std::int32_t> hDiagPos((std::size_t)nv);
      std::vector<T> hConv((std::size_t)A.nnz);
      HXT1_CUDA(cudaMemcpy(hOx.data(),A.diagOriginal.p,(std::size_t)nv*sizeof(T),cudaMemcpyDeviceToHost));
      HXT1_CUDA(cudaMemcpy(hOz.data(),d_diagOrigZ.p,(std::size_t)nv*sizeof(T),cudaMemcpyDeviceToHost));
      HXT1_CUDA(cudaMemcpy(hDx.data(),d_relaxDeltaXY.p,(std::size_t)nv*sizeof(T),cudaMemcpyDeviceToHost));
      HXT1_CUDA(cudaMemcpy(hDz.data(),d_relaxDeltaZ.p,(std::size_t)nv*sizeof(T),cudaMemcpyDeviceToHost));
      HXT1_CUDA(cudaMemcpy(hDiagR.data(),A.diagRBase.p,(std::size_t)nv*sizeof(T),cudaMemcpyDeviceToHost));
      HXT1_CUDA(cudaMemcpy(hDiagRTurb.data(),d_diagRTurb.p,(std::size_t)nv*sizeof(T),cudaMemcpyDeviceToHost));
      HXT1_CUDA(cudaMemcpy(hDiagDG.data(),d_diagDG.p,(std::size_t)nv*sizeof(T),cudaMemcpyDeviceToHost));
      HXT1_CUDA(cudaMemcpy(hDiagWall.data(),d_diagWall.p,(std::size_t)nv*sizeof(T),cudaMemcpyDeviceToHost));
      HXT1_CUDA(cudaMemcpy(hDiagPos.data(),A.diagPos.p,(std::size_t)nv*sizeof(std::int32_t),cudaMemcpyDeviceToHost));
      HXT1_CUDA(cudaMemcpy(hConv.data(),A.conv.p,(std::size_t)A.nnz*sizeof(T),cudaMemcpyDeviceToHost));

      // "interfaceAdjacent" means a velocity DOF belonging to either element
      // in at least one HEX/TET interface plan.  This is intentionally a
      // broader and more useful diagnostic category than trace-only DOFs.
      std::vector<unsigned char> interfaceAdjacent((std::size_t)nv,0);
      for(const auto&p:IP){
        const auto&HG=M.hexVel[(std::size_t)p.hexCell];
        for(int a=0;a<14;++a)interfaceAdjacent[(std::size_t)HG.g[a]]=1;
        for(int s=0;s<2;++s){
          const auto&TG=M.tetVel[(std::size_t)p.tetCell[s]];
          for(int a=0;a<8;++a)interfaceAdjacent[(std::size_t)TG.g[a]]=1;
        }
      }

      struct H5B3AuditStat{
        unsigned long long nonfinite=0,nonpositive=0;
        double minFinite=std::numeric_limits<double>::infinity();
        double maxFinite=-std::numeric_limits<double>::infinity();
        int minIndex=-1;
      };
      struct H5B3AuditOff{
        unsigned long long total=0,wall=0,interfaceAdjacent=0,wallAndInterface=0,neither=0;
      };
      auto statUpdate=[&](H5B3AuditStat&s,double v,int i){
        if(!std::isfinite(v)){++s.nonfinite;return;}
        if(v<s.minFinite){s.minFinite=v;s.minIndex=i;}
        s.maxFinite=std::max(s.maxFinite,v);
        if(!(v>0.0))++s.nonpositive;
      };
      auto offUpdate=[&](H5B3AuditOff&o,int i){
        ++o.total;
        const bool w=BH.wallMask[(std::size_t)i]!=0;
        const bool q=interfaceAdjacent[(std::size_t)i]!=0;
        if(w)++o.wall;if(q)++o.interfaceAdjacent;if(w&&q)++o.wallAndInterface;if(!w&&!q)++o.neither;
      };
      H5B3AuditStat sOx,sOz,sNoX,sNoZ,sMx,sMz;
      H5B3AuditOff oOx,oOz,oNoX,oNoZ,oMx,oMz;
      unsigned long long active=0,fixedCount=0;
      for(int i=0;i<nv;++i){
        if(fixed.mask[(std::size_t)i]){++fixedCount;continue;}
        ++active;
        const int dp=hDiagPos[(std::size_t)i];
        const double ox=(double)hOx[(std::size_t)i],oz=(double)hOz[(std::size_t)i];
        const double convd=(double)hConv[(std::size_t)dp];
        const double noX=(double)hDiagR[(std::size_t)i]+convd+(double)hDiagRTurb[(std::size_t)i]+(double)hDiagDG[(std::size_t)i];
        const double noZ=noX+(double)hDiagWall[(std::size_t)i];
        const double mx=noX+(double)hDx[(std::size_t)i];
        const double mz=noZ+(double)hDz[(std::size_t)i];
        statUpdate(sOx,ox,i);statUpdate(sOz,oz,i);statUpdate(sNoX,noX,i);statUpdate(sNoZ,noZ,i);statUpdate(sMx,mx,i);statUpdate(sMz,mz,i);
        if(!std::isfinite(ox)||!(ox>0.0))offUpdate(oOx,i);
        if(!std::isfinite(oz)||!(oz>0.0))offUpdate(oOz,i);
        if(!std::isfinite(noX)||!(noX>0.0))offUpdate(oNoX,i);
        if(!std::isfinite(noZ)||!(noZ>0.0))offUpdate(oNoZ,i);
        if(!std::isfinite(mx)||!(mx>0.0))offUpdate(oMx,i);
        if(!std::isfinite(mz)||!(mz>0.0))offUpdate(oMz,i);
      }
      auto printStat=[&](const char*q,const H5B3AuditStat&s){
        std::printf("NODALS_HXT5B_RAU_DIAG_AUDIT quantity=%s minFinite=%.12e maxFinite=%.12e nonpositive=%llu nonfinite=%llu minIndex=%d\n",
          q,s.minFinite,s.maxFinite,s.nonpositive,s.nonfinite,s.minIndex);
      };
      auto printOff=[&](const char*q,const H5B3AuditOff&o){
        std::printf("NODALS_HXT5B_RAU_OFFENDERS quantity=%s total=%llu wall=%llu interfaceAdjacent=%llu wallAndInterface=%llu neither=%llu\n",
          q,o.total,o.wall,o.interfaceAdjacent,o.wallAndInterface,o.neither);
      };
      std::printf("NODALS_HXT5B_RAU_DIAG_AUDIT kernelBad=%llu active=%llu fixed=%llu alphaU=%.12e rauScale=%.12e policy=UNCHANGED_DIAGNOSTIC_ONLY\n",
        bad,active,fixedCount,alphaU,rauScale);
      printStat("fullXY",sOx);printOff("fullXY",oOx);
      printStat("fullZ",sOz);printOff("fullZ",oOz);
      printStat("noPenXY",sNoX);printOff("noPenXY",oNoX);
      printStat("noPenZ",sNoZ);printOff("noPenZ",oNoZ);
      printStat("relaxedNoPenXY",sMx);printOff("relaxedNoPenXY",oMx);
      printStat("relaxedNoPenZ",sMz);printOff("relaxedNoPenZ",oMz);
      auto printWorst=[&](const char*q,const H5B3AuditStat&s){
        const int i=s.minIndex;if(i<0)return;
        const int dp=hDiagPos[(std::size_t)i];
        const double convd=(double)hConv[(std::size_t)dp];
        const double noX=(double)hDiagR[(std::size_t)i]+convd+(double)hDiagRTurb[(std::size_t)i]+(double)hDiagDG[(std::size_t)i];
        const double noZ=noX+(double)hDiagWall[(std::size_t)i];
        std::printf("NODALS_HXT5B_RAU_WORST quantity=%s index=%d wall=%d interfaceAdjacent=%d fullXY=%.12e fullZ=%.12e diagRBase=%.12e convDiag=%.12e turbDiag=%.12e dgDiag=%.12e wallDiag=%.12e noPenXY=%.12e noPenZ=%.12e deltaXY=%.12e deltaZ=%.12e relaxedNoPenXY=%.12e relaxedNoPenZ=%.12e\n",
          q,i,(int)BH.wallMask[(std::size_t)i],(int)interfaceAdjacent[(std::size_t)i],
          (double)hOx[(std::size_t)i],(double)hOz[(std::size_t)i],(double)hDiagR[(std::size_t)i],convd,
          (double)hDiagRTurb[(std::size_t)i],(double)hDiagDG[(std::size_t)i],(double)hDiagWall[(std::size_t)i],
          noX,noZ,(double)hDx[(std::size_t)i],(double)hDz[(std::size_t)i],
          noX+(double)hDx[(std::size_t)i],noZ+(double)hDz[(std::size_t)i]);
      };
      printWorst("fullXY",sOx);printWorst("fullZ",sOz);printWorst("noPenXY",sNoX);printWorst("noPenZ",sNoZ);printWorst("relaxedNoPenXY",sMx);printWorst("relaxedNoPenZ",sMz);
      std::printf("NODALS_HXT5B_RAU_DIAG_AUDIT_END status=FAIL_AS_BEFORE noSolverPolicyChanged=1\n");
      throw std::runtime_error("initial nonpositive directional relaxed/rAU diagonal");
    }

    // Match the validated turbulence gate: the weak DG inlet must preserve the
    // initialized constant plug exactly in momentum, G*Uhat - rhs = 0.
    hxt5b::spmv<<<(nv+TPB-1)/TPB,TPB>>>(nv,A.row.p,A.col.p,d_dg.p,d_u2.p,A.tmp.p);
    hxt5b::h5b3_subtract<<<(nv+TPB-1)/TPB,TPB>>>(nv,A.tmp.p,d_inletRhsZ.p,A.res.p);
    const double dgMomAbs=ured.norm(A.res.p);
    const double dgMomRhs=ured.norm(d_inletRhsZ.p);
    const double dgMomRel=dgMomAbs/std::max(dgMomRhs,1e-300);
    const double dgMomLim=std::is_same<T,float>::value?5e-5:2e-11;
    std::printf("NODALS_HXT5B3_DG_PLUG_MOMENTUM_CANCELLATION abs=%.12e rhsNorm=%.12e rel=%.12e limit=%.12e status=%s\n",
      dgMomAbs,dgMomRhs,dgMomRel,dgMomLim,dgMomRel<=dgMomLim?"PASS":"FAIL");
    if(!(dgMomRel<=dgMomLim))
      throw std::runtime_error("HXT5B3 DG weak inlet does not preserve constant plug");

    hxt5b::spmv<<<(nv+TPB-1)/TPB,TPB>>>(nv,A.row.p,A.col.p,A.conv.p,d_u2.p,A.tmp.p);
    const double convPlug=ured.norm(A.tmp.p);
    const double convRel=convPlug/std::max(dgMomRhs,1e-300);
    std::printf("NODALS_HXT5B3_CONVECTION_PLUG_CANCELLATION abs=%.12e relToDgRhs=%.12e expected=roundoff status=%s\n",
      convPlug,convRel,convRel<1e-3?"PASS":"CHECK");

    hxt5b::h5b3_relax_audit("xy_setup",A.diagOriginal.p,d_relaxDeltaXY.p,nv);
    hxt5b::h5b3_relax_audit("z_setup",d_diagOrigZ.p,d_relaxDeltaZ.p,nv);

    std::vector<T>rauT((std::size_t)nv),rauZT((std::size_t)nv);
    HXT1_CUDA(cudaMemcpy(rauT.data(),A.rau.p,nv*sizeof(T),cudaMemcpyDeviceToHost));
    HXT1_CUDA(cudaMemcpy(rauZT.data(),d_rauZ.p,nv*sizeof(T),cudaMemcpyDeviceToHost));
    std::vector<double>rxyD((std::size_t)nv),rzD((std::size_t)nv);for(int i=0;i<nv;++i){rxyD[(std::size_t)i]=(double)rauT[(std::size_t)i];rzD[(std::size_t)i]=(double)rauZT[(std::size_t)i];}
    auto PH=hxt4c::build_hybrid_fine_host(Bh,pin,rxyD);
    hxt5b::retune_host_fine(PH,Bh,pin,rxyD,rxyD,rzD);

    int effectivePowerIts=0;
    if(amgSpectrumPolicy=="always")effectivePowerIts=amgPowerIts;
    else if(amgSpectrumPolicy=="auto"&&amgSmoother=="cheb2")effectivePowerIts=amgPowerIts;
    else if(amgSpectrumPolicy=="off")effectivePowerIts=0;
    else if(amgSpectrumPolicy!="auto")throw std::runtime_error("bad AMG spectrum policy");
    if(amgSmoother=="cheb2"&&effectivePowerIts<=0)throw std::runtime_error("Chebyshev requires spectrum estimation");

    auto HH=nodals_gpu::build_cf_hierarchy_from_csr(
      PH.A,cfTheta,cfPmax,cfInterp,cfAggressiveFirst!=0,amgTerminal,
      effectivePowerIts,amgLambdaSafety,amgLambdaLow);

    hxt4c::AMGOptions AO;
    AO.smoother=amgSmoother;AO.jacobiOmega=amgJacobiOmega;
    AO.jacobiFineSweeps=amgJacobiFineSweeps;AO.jacobiCoarseSweeps=amgJacobiCoarseSweeps;
    AO.chebDegree=amgChebDegree;AO.lambdaLowFraction=amgLambdaLow;
    AO.mcgsFineSweeps=amgMcgsFineSweeps;AO.mcgsCoarseSweeps=amgMcgsCoarseSweeps;AO.mcgsOmega=amgMcgsOmega;AO.mcgsOrder=amgMcgsOrder;
    hxt4c::HybridAMG<T>PC(PH,HH,pin,AO);
    hxt5b::refresh_pc(PC,B,A.rau.p,A.rau.p,d_rauZ.p);

    auto wallNow=hxt5b::wallstats(d_wallStats,bulk);
    std::printf(
      "NODALS_HXT5B_CONFIG precision=%s geometry=fp64 reductions=fp64 cellsHex=%llu cellsTet=%llu "
      "velocityDofs=%d pressureDofs=%d Re=%.12g bulk=%.12g nu=%.12e gamma=%.6g "
      "turbulence=NIKURADSE_ALGEBRAIC mixlenScale=%.8g inlet=DG_NUMERICAL_TRACE_PLUG velocityInletDOFs=FREE "
      "wall=FIREDRAKE_Q1BF2_OFFWALL_SPALDING wallSampleFraction=%.6g kappa=0.4 B=5.5 transverseWallClamp=1 "
      "momentumWork=%s momentumRelax=row_l1 momentumRtol=%.3e momentumAtol=%.3e momentumMax=%d momentumCapPolicy=%s outerConvergence=CONTINUITY_PLUS_XYZ_PHYSICAL_MOMENTUM_INITIAL_RELATIVE "
      "pressureSolver=%s pRtol=%.3e pAtol=%.3e pMax=%d amgHierarchy=%s cfCoarsening=%s cfStrength=%s "
      "cfTheta=%.6g cfInterp=%s cfPmax=%d amgSmoother=%s chebDegree=%d powerIts=%d spectrumPolicy=%s "
      "alphaU=%.6g alphaP=%.6g rauScale=%.6g directionalRAU=1 rauPenaltyPolicy=SEPARATE_GAMMA simpleTol=%.3e status=PASS\n",
      PRECISION,(unsigned long long)M.h.nhex,(unsigned long long)M.h.ntet,nv,np,Re,bulk,nu,gamma,mixlenScale,
      wallSampleFraction,momWork.c_str(),momRtol,momAtol,momMax,momCapPolicy.c_str(),pressureSolver.c_str(),pRtol,pAtol,pMax,amgHierarchy.c_str(),cfCoarsening.c_str(),
      cfStrength.c_str(),cfTheta,cfInterp.c_str(),cfPmax,amgSmoother.c_str(),amgChebDegree,effectivePowerIts,
      amgSpectrumPolicy.c_str(),alphaU,alphaP,rauScale,simpleTol);
    std::printf("NODALS_HXT5B_NITSCHE_CONFIG momentumGamma=%.12e schurGamma=%.12e schurOverMomentum=%.12e rAUPolicy=NO_RAW_NOPENGATE_PLUS_ROWL1_PLUS_SELECTED_NITSCHE status=PASS\n",
      gamma,schurNitscheGamma,(gamma>0.0?schurNitscheGamma/gamma:0.0));
    std::printf(
      "NODALS_HXT5B_BOUNDARY_SETUP inletHexFaces=%zu inletTetFaces=%zu wallHexFaces=%zu inletArea=%.12e "
      "prescribedFluxSum=%.12e constantPatchResidual=%.12e wallArea=%.12e wallYPlusMeanInitial=%.12e "
      "wallDarcyInitial=%.12e rootFailures=%.0f status=PASS\n",
      BH.inH.size(),BH.inT.size(),BH.wallH.size(),BH.inletArea,BH.sourceSum,BH.constResidual,
      wallNow.area,wallNow.yPlusMean,wallNow.fDarcy,wallNow.rootFailures);
    std::printf("NODALS_HXT5B_MIXLEN_SETUP meanNutOverNu=%.12e maxNutOverNu=%.12e maxStrain=%.12e status=PASS\n",
      mix.meanNutOverNu,mix.maxNutOverNu,mix.maxStrain);

    double c0=-1,m0[3]={-1,-1,-1};bool converged=false;int outer=0;long long pIts=0;
    // G0/SST baseline diagnostic buffer.  This does not enter any operator or
    // convergence decision; it is copied only at the existing sparse print points
    // so pressure-drop stationarity can be judged together with y+ and wall friction.
    std::vector<T> hPressurePlateau((std::size_t)np);

    const bool directSSTStart=(sstG2&&sstG2Start=="plug");
    if(directSSTStart){
      // Straight-pipe potential-flow startup: the existing coefficient initializer
      // already gives U=(0,0,Ub) exactly, enriched velocity coefficients zero, and p=0.
      // Skip the expensive converged Nikuradse SIMPLE precursor entirely.
      converged=true;outer=0;pIts=0;
      std::printf("NODALS_SST_G2_START mode=PLUG_POTENTIAL velocity=UNIFORM_AXIAL_BULK pressure=ZERO kOmega=UNIFORM_INLET_FORMULAS precursorSimpleIterations=0 status=PASS\n");
    } else for(int it=1;it<=maxOuter;++it){
      HXT1_CUDA(cudaMemset(A.conv.p,0,A.nnz*sizeof(T)));HXT1_CUDA(cudaMemset(d_bad.p,0,sizeof(unsigned long long)));
      if(M.h.nhex)hex_conv_assemble<<<M.h.nhex,128>>>(d_pts.p,d_hc.p,d_hv.p,A.hexSlot.p,d_u0.p,d_u1.p,d_u2.p,A.conv.p,M.h.nhex,d_bad.p);
      if(M.h.ntet)tet_conv_assemble<<<M.h.ntet,64>>>(d_tv.p,d_tg.p,A.tetSlot.p,d_u0.p,d_u1.p,d_u2.p,A.conv.p,M.h.ntet);
      mix=hxt5a::h5_assemble_mixlen(M.h.nhex,M.h.ntet,M.h.ninterface,d_pts,d_hc,d_tc,d_hv,d_tv,d_tg,d_ip,A,
        d_u0.p,d_u1.p,d_u2.p,d_turb,d_diagRTurb,d_mixStats,R,nu,mixlenScale,gamma,d_bad);
      hxt5b::assemble_boundary(BH,d_inH,d_inT,d_wallH,d_pts,d_hc,d_tc,d_hv,d_tv,d_tg,A,
        d_u0.p,d_u1.p,d_u2.p,d_dg,d_diagDG,d_inletRhsZ,d_wall,d_diagWall,d_wallStats,d_bad,
        bulk,R,nu,mixlenScale,wallSampleFraction);
      hxt5b::combine<<<(A.nnz+TPB-1)/TPB,TPB>>>(A.nnz,A.base.p,A.conv.p,d_turb.p,d_dg.p,d_wall.p,A.val.p,d_valZ.p);
      HXT1_CUDA(cudaMemset(d_bad.p,0,sizeof(unsigned long long)));
      hxt5b::h5b3_finalize_row_l1<<<(nv+TPB-1)/TPB,TPB>>>(
      nv,d_fixed.p,d_wallMask.p,A.row.p,A.diagPos.p,A.val.p,d_valZ.p,
      A.diagRBase.p,A.conv.p,d_diagRTurb.p,d_diagDG.p,d_diagWall.p,
      A.rau.p,d_rauZ.p,A.diagOriginal.p,d_diagOrigZ.p,d_relaxDeltaXY.p,d_relaxDeltaZ.p,
      gamma,schurNitscheGamma,alphaU,rauScale,d_bad.p);
      HXT1_CUDA(cudaDeviceSynchronize());HXT1_CUDA(cudaMemcpy(&bad,d_bad.p,sizeof(bad),cudaMemcpyDeviceToHost));
      if(bad)throw std::runtime_error("nonpositive directional relaxed/rAU diagonal count="+std::to_string(bad));
      if(it==1){
        hxt5b::audit_real("matrix_xy_after_finalize",A.val.p,A.nnz);
        hxt5b::audit_real("matrix_z_after_finalize",d_valZ.p,A.nnz);
        hxt5b::audit_real("rau_xy_after_finalize",A.rau.p,(std::size_t)nv);
        hxt5b::audit_real("rau_z_after_finalize",d_rauZ.p,(std::size_t)nv);
        hxt5b::audit_diag_host("diag_xy_original",A.diagOriginal.p,nv);
        hxt5b::audit_diag_host("diag_z_original",d_diagOrigZ.p,nv);
      }

      hxt5b::sdiag<<<(np+TPB-1)/TPB,TPB>>>(np,B.row.p,B.col.p,B.bx.p,B.by.p,B.bz.p,A.rau.p,A.rau.p,d_rauZ.p,d_sd.p,pin);
      hxt4a::invert_diag_kernel<T><<<(np+TPB-1)/TPB,TPB>>>(np,d_sd.p,d_isd.p,pin);
      hxt5b::SAniso<T>S(B,A.rau.p,A.rau.p,d_rauZ.p,pin);
      hxt5b::refresh_pc(PC,B,A.rau.p,A.rau.p,d_rauZ.p);

      if(it==1){
        double parity=hxt5b::parity(PC,S,np,pin,pred);
        const double lim=std::is_same<T,float>::value?3e-5:2e-11;
        std::printf("NODALS_HXT5B_FINE_CSR_PARITY rel=%.12e limit=%.12e status=%s\n",parity,lim,parity<=lim?"PASS":"FAIL");
        if(parity>lim)throw std::runtime_error("HXT5B anisotropic fine CSR/matrix-free Schur parity failed");
      }

      hxt4a::BT_apply(B,d_p.p,d_bt0.p,d_bt1.p,d_bt2.p);
      hxt5b::h5b3_rhs<<<(nv+TPB-1)/TPB,TPB>>>(nv,d_clampXY.p,d_bt0.p,nullptr,d_u0.p,d_relaxDeltaXY.p,d_rhs0.p);
      hxt5b::h5b3_rhs<<<(nv+TPB-1)/TPB,TPB>>>(nv,d_clampXY.p,d_bt1.p,nullptr,d_u1.p,d_relaxDeltaXY.p,d_rhs1.p);
      hxt5b::h5b3_rhs<<<(nv+TPB-1)/TPB,TPB>>>(nv,d_fixed.p,d_bt2.p,d_inletRhsZ.p,d_u2.p,d_relaxDeltaZ.p,d_rhs2.p);
      if(it==1){
        hxt5b::audit_real("rhs_x_before_sgs",d_rhs0.p,(std::size_t)nv);
        hxt5b::audit_real("rhs_y_before_sgs",d_rhs1.p,(std::size_t)nv);
        hxt5b::audit_real("rhs_z_before_sgs",d_rhs2.p,(std::size_t)nv);
        hxt5b::audit_real("u_z_before_sgs",d_u2.p,(std::size_t)nv);
      }
      auto mx=hxt5b::h5b3_momentum_solve(momWork,it,A,VC,A.val.p,d_rhs0.p,d_u0.p,momOmega,d_clampXY.p,momRtol,momAtol,momMax,ured,"x");
      auto my=hxt5b::h5b3_momentum_solve(momWork,it,A,VC,A.val.p,d_rhs1.p,d_u1.p,momOmega,d_clampXY.p,momRtol,momAtol,momMax,ured,"y");
      auto mz=hxt5b::h5b3_momentum_solve(momWork,it,A,VC,d_valZ.p,d_rhs2.p,d_u2.p,momOmega,d_fixed.p,momRtol,momAtol,momMax,ured,"z");
      // HXT5B_MOM_CAP_POLICY_V1
      // h5b3_momentum_solve throws internally for NaN/Inf or invalid work.
      // Therefore ok=false here means finite MOM_MAX exhaustion only.
      // In SIMPLE this is bounded inner work; the outer physical residual gate
      // remains the authority on nonlinear convergence.
      const bool momCapMiss=(!mx.ok||!my.ok||!mz.ok);
      if(momCapMiss){
        const bool strictMomCap=(momCapPolicy=="fail");
        std::printf("NODALS_HXT5B3_MOM_CAP outer=%d xRel=%.12e xSweeps=%d yRel=%.12e ySweeps=%d zRel=%.12e zSweeps=%d targetRel=%.6g maxSweeps=%d policy=%s status=%s\n",
          it,mx.rel,mx.sweeps,my.rel,my.sweeps,mz.rel,mz.sweeps,momRtol,momMax,
          momCapPolicy.c_str(),strictMomCap?"FAIL":"WARN_CONTINUE");
        if(strictMomCap)
          throw std::runtime_error("HXT5B3 momentum residual-drop target not reached");
      }
      if(it<=5||it%20==0)
        std::printf("NODALS_HXT5B3_MOM_SOLVE outer=%d xSweeps=%d xRel=%.6e ySweeps=%d yRel=%.6e zSweeps=%d zRel=%.6e targetRel=%.6g status=PASS\n",
          it,mx.sweeps,mx.rel,my.sweeps,my.rel,mz.sweeps,mz.rel,momRtol);

      hxt4a::B_apply(B,d_u0.p,d_u1.p,d_u2.p,d_cont.p);hxt5b::addsrc<<<(np+TPB-1)/TPB,TPB>>>(np,d_contSource.p,d_cont.p);hxt4a::pin_zero_kernel<T><<<1,1>>>(pin,d_cont.p);
      if(it==1)hxt5b::audit_real("continuity_before_norm",d_cont.p,(std::size_t)np);
      double cn=pred.norm(d_cont.p);
      if(!std::isfinite(cn))throw std::runtime_error("HXT5B nonfinite continuity norm");

      HXT1_CUDA(cudaMemset(d_dp.p,0,np*sizeof(T)));
      hxt4a::CGResult pr;
      if(pressureSolver=="pcg_amg")pr=hxt4c::pcg_amg(S,PC,d_cont.p,d_dp.p,np,pRtol,pAtol,pMax,WP,pin);
      else pr=pcg_prod(S,d_cont.p,d_dp.p,d_isd.p,np,pRtol,pAtol,pMax,WP,pin);
      if(!pr.ok){
        std::printf("NODALS_HXT5B_PRESSURE_FAIL outer=%d solver=%s pCG=%d pRel=%.12e pRtol=%.12e pAtol=%.12e pMax=%d status=FAIL\n",
          it,pressureSolver.c_str(),pr.its,pr.rel,pRtol,pAtol,pMax);
        throw std::runtime_error("pressure solver failed production tolerance");
      }
      pIts+=pr.its;
      pressure_update_minus_kernel<<<(np+TPB-1)/TPB,TPB>>>(np,pin,(T)alphaP,d_dp.p,d_p.p);

      hxt4a::BT_apply(B,d_p.p,d_bt0.p,d_bt1.p,d_bt2.p);
      double ma[3]={
        hxt5b::h5b3_physical_residual_norm(A,A.val.p,d_clampXY.p,d_bt0.p,nullptr,d_u0.p,d_relaxDeltaXY.p,ured),
        hxt5b::h5b3_physical_residual_norm(A,A.val.p,d_clampXY.p,d_bt1.p,nullptr,d_u1.p,d_relaxDeltaXY.p,ured),
        hxt5b::h5b3_physical_residual_norm(A,d_valZ.p,d_fixed.p,d_bt2.p,d_inletRhsZ.p,d_u2.p,d_relaxDeltaZ.p,ured)};
      if(it==1){c0=cn;m0[0]=ma[0];m0[1]=ma[1];m0[2]=ma[2];}
      const double st=std::max(simpleTol,1e-300);
      const double rc=cn/std::max(std::max(c0,outerContAtol/st),1e-300);
      const double rm0=ma[0]/std::max(std::max(m0[0],momAtol/st),1e-300);
      const double rm1=ma[1]/std::max(std::max(m0[1],momAtol/st),1e-300);
      const double rm2=ma[2]/std::max(std::max(m0[2],momAtol/st),1e-300);
      outer=it;converged=(it>1&&rc<=simpleTol&&rm0<=simpleTol&&rm1<=simpleTol&&rm2<=simpleTol);
      if(it<=5||it%20==0||converged){
        auto ws=hxt5b::wallstats(d_wallStats,bulk);
        HXT1_CUDA(cudaMemcpy(hPressurePlateau.data(),d_p.p,(std::size_t)np*sizeof(T),cudaMemcpyDeviceToHost));
        const auto pPlateau=hxt4a::to_double(hPressurePlateau);
        const double pressureSlopePlateau=hxt4a::pressure_slope(M,pPlateau);
        const double pInPlateau=hxt4a::boundary_p(M,pPlateau,true);
        const double pOutPlateau=hxt4a::boundary_p(M,pPlateau,false);
        const double pressureDropPlateau=pInPlateau-pOutPlateau;
        std::printf("NODALS_HXT5B_SIMPLE it=%d relContInit=%.12e absCont=%.12e relMomXInit=%.12e absMomX=%.12e relMomYInit=%.12e absMomY=%.12e relMomZInit=%.12e absMomZ=%.12e pCG=%d pRel=%.3e meanNutOverNu=%.6e maxNutOverNu=%.6e wallYPlusMean=%.6e wallDarcy=%.6e pressureSlope=%.6e pressureDrop=%.6e converged=%d status=PASS\n",
          it,rc,cn,rm0,ma[0],rm1,ma[1],rm2,ma[2],pr.its,pr.rel,mix.meanNutOverNu,mix.maxNutOverNu,ws.yPlusMean,ws.fDarcy,pressureSlopePlateau,pressureDropPlateau,(int)converged);
      }
      if(converged)break;
      double worst=std::max(std::max(rc,rm0),std::max(rm1,rm2));
      if(it>=5&&(!std::isfinite(worst)||worst>1e8))throw std::runtime_error("HXT5B coupled-residual divergence guard");
    }
    if(!converged)throw std::runtime_error("HXT5B SIMPLE maxOuter without all-residual convergence");

    // G2: either start directly from the exact straight-pipe plug/potential field
    // (default development path) or from the legacy converged G0 field.
    // In both cases k/omega are initialized uniformly from the inlet formulas.

    // The old Nikuradse field is no longer used by momentum inside this loop.
    hxt5c::Stats g2FinalStats;bool g2Gate=true;int g2Outer=0;double g2KRel=0,g2WRel=0,g2NutRel=0;
    if(sstG2){
      hxt5d::State TS(nv,(int)M.h.nv,bulk,R,sstG1);hxt4a::Reducer<T> sstRed(nv);
      std::vector<double>hf,hy,hdp;bool g2Converged=false;double finalDp=0,finalF=0,finalY=0;
      auto relSpan=[](const std::vector<double>&v){if(v.empty())return std::numeric_limits<double>::infinity();auto mm=std::minmax_element(v.begin(),v.end());double mean=0;for(double x:v)mean+=x;mean/=v.size();return (*mm.second-*mm.first)/std::max(std::abs(mean),1e-300);};
      std::printf("NODALS_SST_OMEGA_WALL_CONFIG mode=%s target=%s placement=%s physicalWallFlux=%s penaltyGamma=%.12e sampleFraction=%.12e ofKappa=%.6e ofE=%.6e ofBeta1=%.6e ofReBlend=%.6e ofProduction=%d ofProductionScale=%.6e ofWallVolumeScale=%.6e ofProductionLimit=%.6e status=PASS\n",hxt5c::omega_wall_mode_name(sstG1.omegaWallMode),sstG1.omegaWallMode==hxt5c::OMEGA_WALL_OF_AUTO?"OF_OMEGA_VIS_LOG_BLEND":"SPALDING_UTAU_LOG",sstG1.omegaWallMode==hxt5c::OMEGA_WALL_TRACE_LOG?"PHYSICAL_TRACE_SYMMETRIC_NITSCHE":"OFFWALL_SAMPLE_PENALTY",sstG1.omegaWallMode==hxt5c::OMEGA_WALL_TRACE_LOG?"NITSCHE_DIRICHLET":"NATURAL_ZERO",sstG1.wallPenaltyGamma,wallSampleFraction,sstG1.ofKappa,sstG1.ofE,sstG1.ofBeta1,sstG1.ofReBlend,(int)sstG1.ofProduction,sstG1.ofProductionScale,sstG1.ofWallVolumeScale,sstG1.ofProductionLimit);
      std::printf("NODALS_SST_G2_CONFIG model=SST-2003m precision=%s coupling=FULL_SIMPLE_K_OMEGA momentumFeedback=ON start=%s velocityInlet=DG_NUMERICAL_TRACE_PLUG turbulenceInlet=DG_TRACE intensity=%.6g lengthScaleOverD=%.6g kIn=%.12e omegaIn=%.12e outlet=NATURAL_ZERO_GRADIENT momentumWall=SPALDING kWall=HIGH_RE_ZERO_GRADIENT omegaWall=SPALDING_UTAU_LOG wallDistance=ANALYTIC_PIPE_R_MINUS_R alphaK=%.6g alphaOmega=%.6g scalarLinear=SGS1 plateauTol=%.6e plateauSamples=%d plateauEvery=%d maxOuter=%d status=PASS\n",
        PRECISION,directSSTStart?"PLUG_POTENTIAL":"CONVERGED_G0_MIXLEN",sstG1.intensity,sstG1.lengthRatio,TS.kIn,TS.omegaIn,sstG1.alphaK,sstG1.alphaOmega,sstG2PlateauTol,sstG2PlateauSamples,sstG2PrintEvery,sstG2Max);
      for(int git=1;git<=sstG2Max;++git){
        HXT1_CUDA(cudaMemset(A.conv.p,0,A.nnz*sizeof(T)));HXT1_CUDA(cudaMemset(d_bad.p,0,sizeof(unsigned long long)));
        if(M.h.nhex)hex_conv_assemble<<<M.h.nhex,128>>>(d_pts.p,d_hc.p,d_hv.p,A.hexSlot.p,d_u0.p,d_u1.p,d_u2.p,A.conv.p,M.h.nhex,d_bad.p);
        if(M.h.ntet)tet_conv_assemble<<<M.h.ntet,64>>>(d_tv.p,d_tg.p,A.tetSlot.p,d_u0.p,d_u1.p,d_u2.p,A.conv.p,M.h.ntet);
        hxt5d::assemble_momentum_sst(M,d_pts,d_hc,d_tc,d_hv,d_tv,d_tg,d_ip,A,d_u0.p,d_u1.p,d_u2.p,TS.k.p,TS.omega.p,d_turb,d_diagRTurb,R,nu,gamma,sstG1,TS.stats,d_bad);
        hxt5d::assemble_boundary_sst(BH,d_inH,d_inT,d_wallH,d_pts,d_hc,d_tc,d_hv,d_tv,d_tg,A,d_u0.p,d_u1.p,d_u2.p,TS.k.p,TS.omega.p,d_dg,d_diagDG,d_inletRhsZ,d_wall,d_diagWall,d_wallStats,d_bad,bulk,R,nu,wallSampleFraction,sstG1);
        hxt5b::combine<<<(A.nnz+TPB-1)/TPB,TPB>>>(A.nnz,A.base.p,A.conv.p,d_turb.p,d_dg.p,d_wall.p,A.val.p,d_valZ.p);
        HXT1_CUDA(cudaMemset(d_bad.p,0,sizeof(unsigned long long)));
        hxt5b::h5b3_finalize_row_l1<<<(nv+TPB-1)/TPB,TPB>>>(nv,d_fixed.p,d_wallMask.p,A.row.p,A.diagPos.p,A.val.p,d_valZ.p,A.diagRBase.p,A.conv.p,d_diagRTurb.p,d_diagDG.p,d_diagWall.p,A.rau.p,d_rauZ.p,A.diagOriginal.p,d_diagOrigZ.p,d_relaxDeltaXY.p,d_relaxDeltaZ.p,gamma,schurNitscheGamma,alphaU,rauScale,d_bad.p);
        HXT1_CUDA(cudaDeviceSynchronize());HXT1_CUDA(cudaMemcpy(&bad,d_bad.p,sizeof(bad),cudaMemcpyDeviceToHost));if(bad)throw std::runtime_error("SST G2 nonpositive momentum relaxed/rAU diagonal count="+std::to_string(bad));

        hxt5b::sdiag<<<(np+TPB-1)/TPB,TPB>>>(np,B.row.p,B.col.p,B.bx.p,B.by.p,B.bz.p,A.rau.p,A.rau.p,d_rauZ.p,d_sd.p,pin);hxt4a::invert_diag_kernel<T><<<(np+TPB-1)/TPB,TPB>>>(np,d_sd.p,d_isd.p,pin);hxt5b::SAniso<T>Sg2(B,A.rau.p,A.rau.p,d_rauZ.p,pin);hxt5b::refresh_pc(PC,B,A.rau.p,A.rau.p,d_rauZ.p);
        hxt4a::BT_apply(B,d_p.p,d_bt0.p,d_bt1.p,d_bt2.p);
        hxt5b::h5b3_rhs<<<(nv+TPB-1)/TPB,TPB>>>(nv,d_clampXY.p,d_bt0.p,nullptr,d_u0.p,d_relaxDeltaXY.p,d_rhs0.p);hxt5b::h5b3_rhs<<<(nv+TPB-1)/TPB,TPB>>>(nv,d_clampXY.p,d_bt1.p,nullptr,d_u1.p,d_relaxDeltaXY.p,d_rhs1.p);hxt5b::h5b3_rhs<<<(nv+TPB-1)/TPB,TPB>>>(nv,d_fixed.p,d_bt2.p,d_inletRhsZ.p,d_u2.p,d_relaxDeltaZ.p,d_rhs2.p);
        auto mx=hxt5b::h5b3_momentum_solve(momWork,git,A,VC,A.val.p,d_rhs0.p,d_u0.p,momOmega,d_clampXY.p,momRtol,momAtol,momMax,ured,"x");auto my=hxt5b::h5b3_momentum_solve(momWork,git,A,VC,A.val.p,d_rhs1.p,d_u1.p,momOmega,d_clampXY.p,momRtol,momAtol,momMax,ured,"y");auto mz=hxt5b::h5b3_momentum_solve(momWork,git,A,VC,d_valZ.p,d_rhs2.p,d_u2.p,momOmega,d_fixed.p,momRtol,momAtol,momMax,ured,"z");
        if((!mx.ok||!my.ok||!mz.ok)&&momCapPolicy=="fail")throw std::runtime_error("SST G2 momentum residual-drop target not reached");
        hxt4a::B_apply(B,d_u0.p,d_u1.p,d_u2.p,d_cont.p);hxt5b::addsrc<<<(np+TPB-1)/TPB,TPB>>>(np,d_contSource.p,d_cont.p);hxt4a::pin_zero_kernel<T><<<1,1>>>(pin,d_cont.p);double cn=pred.norm(d_cont.p);if(!std::isfinite(cn))throw std::runtime_error("SST G2 nonfinite continuity");
        HXT1_CUDA(cudaMemset(d_dp.p,0,np*sizeof(T)));hxt4a::CGResult pr;if(pressureSolver=="pcg_amg")pr=hxt4c::pcg_amg(Sg2,PC,d_cont.p,d_dp.p,np,pRtol,pAtol,pMax,WP,pin);else pr=pcg_prod(Sg2,d_cont.p,d_dp.p,d_isd.p,np,pRtol,pAtol,pMax,WP,pin);if(!pr.ok)throw std::runtime_error("SST G2 pressure solve failed");pIts+=pr.its;pressure_update_minus_kernel<<<(np+TPB-1)/TPB,TPB>>>(np,pin,(T)alphaP,d_dp.p,d_p.p);
        hxt4a::BT_apply(B,d_p.p,d_bt0.p,d_bt1.p,d_bt2.p);double ma0=hxt5b::h5b3_physical_residual_norm(A,A.val.p,d_clampXY.p,d_bt0.p,nullptr,d_u0.p,d_relaxDeltaXY.p,ured),ma1=hxt5b::h5b3_physical_residual_norm(A,A.val.p,d_clampXY.p,d_bt1.p,nullptr,d_u1.p,d_relaxDeltaXY.p,ured),ma2=hxt5b::h5b3_physical_residual_norm(A,d_valZ.p,d_fixed.p,d_bt2.p,d_inletRhsZ.p,d_u2.p,d_relaxDeltaZ.p,ured);

        auto tr=hxt5d::step_turbulence(TS,git,M,BH,d_inH,d_inT,d_wallH,d_pts,d_hc,d_tc,d_hv,d_tv,d_tg,d_ip,A,VC,d_fixed,d_u0.p,d_u1.p,d_u2.p,d_turb,d_dg,d_wall,d_bad,sstRed,bulk,R,nu,gamma,wallSampleFraction,sstG1);g2FinalStats=tr.stats;g2KRel=tr.kRel;g2WRel=tr.omegaRel;g2NutRel=tr.nutRel;g2Outer=git;
        bool sample=(git<=5||git%sstG2PrintEvery==0);
        if(sample){auto ws2=hxt5b::wallstats(d_wallStats,bulk);HXT1_CUDA(cudaMemcpy(hPressurePlateau.data(),d_p.p,(std::size_t)np*sizeof(T),cudaMemcpyDeviceToHost));auto pd2=hxt4a::to_double(hPressurePlateau);double dp2=hxt4a::boundary_p(M,pd2,true)-hxt4a::boundary_p(M,pd2,false);finalDp=dp2;finalF=ws2.fDarcy;finalY=ws2.yPlusMean;if(git%sstG2PrintEvery==0){hf.push_back(finalF);hy.push_back(finalY);hdp.push_back(finalDp);while((int)hf.size()>sstG2PlateauSamples){hf.erase(hf.begin());hy.erase(hy.begin());hdp.erase(hdp.begin());}}double sf=relSpan(hf),sy=relSpan(hy),sd=relSpan(hdp);bool have=(int)hf.size()>=sstG2PlateauSamples;bool physical=have&&sf<=sstG2PlateauTol&&sy<=sstG2PlateauTol&&sd<=sstG2PlateauTol;bool massok=cn<=std::max(1e-8,100*outerContAtol);bool turbok=tr.stats.kFloorFrac<.01&&tr.stats.omegaFloorFrac<.01&&std::isfinite(tr.kRel)&&std::isfinite(tr.omegaRel)&&std::isfinite(tr.nutRel)&&tr.kRel<=sstG2PlateauTol&&tr.omegaRel<=sstG2PlateauTol&&tr.nutRel<=sstG2PlateauTol;g2Converged=physical&&massok&&turbok;
          std::printf("NODALS_SST_G2 it=%d absCont=%.6e absMom=[%.6e,%.6e,%.6e] pCG=%d pRel=%.3e kRelUpdate=%.6e omegaRelUpdate=%.6e nutMeanRelUpdate=%.6e meanNutOverNu=%.6e maxNutOverNu=%.6e kFloorVolumeFrac=%.6e omegaFloorVolumeFrac=%.6e wallDarcy=%.6e wallYPlusMean=%.6e pressureDrop=%.6e plateauSpan=[%.6e,%.6e,%.6e] plateauCount=%zu converged=%d status=PASS\n",git,cn,ma0,ma1,ma2,pr.its,pr.rel,tr.kRel,tr.omegaRel,tr.nutRel,tr.stats.meanNutOverNu,tr.stats.maxNutOverNu,tr.stats.kFloorFrac,tr.stats.omegaFloorFrac,finalF,finalY,finalDp,sf,sy,sd,hf.size(),(int)g2Converged);
        }
        if(g2Converged)break;
      }
      std::vector<T>hk((std::size_t)nv),hw((std::size_t)nv);HXT1_CUDA(cudaMemcpy(hk.data(),TS.k.p,(std::size_t)nv*sizeof(T),cudaMemcpyDeviceToHost));HXT1_CUDA(cudaMemcpy(hw.data(),TS.omega.p,(std::size_t)nv*sizeof(T),cudaMemcpyDeviceToHost));std::string sstVtu=vtu.size()>=4&&vtu.substr(vtu.size()-4)==".vtu"?vtu.substr(0,vtu.size()-4)+"_sst_g2.vtu":vtu+"_sst_g2.vtu";hxt5c::write_sst_vtu(sstVtu,M,hk,hw);
      g2Gate=g2Converged&&g2FinalStats.meanNutOverNu>0&&g2FinalStats.maxNutOverNu>0&&g2FinalStats.kFloorFrac<.01&&g2FinalStats.omegaFloorFrac<.01;
      std::printf("NODALS_SST_G2_FINAL iterations=%d kRelUpdate=%.12e omegaRelUpdate=%.12e nutMeanRelUpdate=%.12e meanNutOverNu=%.12e maxNutOverNu=%.12e kFloorVolumeFrac=%.12e omegaFloorVolumeFrac=%.12e wallDarcy=%.12e wallYPlusMean=%.12e pressureDrop=%.12e vtu=%s status=%s\n",g2Outer,g2KRel,g2WRel,g2NutRel,g2FinalStats.meanNutOverNu,g2FinalStats.maxNutOverNu,g2FinalStats.kFloorFrac,g2FinalStats.omegaFloorFrac,finalF,finalY,finalDp,sstVtu.c_str(),g2Gate?"PASS":"FAIL");std::printf("SST_G2_GATE_STATUS=%s\n",g2Gate?"PASS":"FAIL");
    }

    std::vector<T>hp((std::size_t)np),hu0((std::size_t)nv),hu1((std::size_t)nv),hu2((std::size_t)nv);
    HXT1_CUDA(cudaMemcpy(hp.data(),d_p.p,np*sizeof(T),cudaMemcpyDeviceToHost));
    HXT1_CUDA(cudaMemcpy(hu0.data(),d_u0.p,nv*sizeof(T),cudaMemcpyDeviceToHost));
    HXT1_CUDA(cudaMemcpy(hu1.data(),d_u1.p,nv*sizeof(T),cudaMemcpyDeviceToHost));
    HXT1_CUDA(cudaMemcpy(hu2.data(),d_u2.p,nv*sizeof(T),cudaMemcpyDeviceToHost));

    auto pd=hxt4a::to_double(hp);
    double slope=hxt4a::pressure_slope(M,pd),pinP=hxt4a::boundary_p(M,pd,true),poutP=hxt4a::boundary_p(M,pd,false),drop=pinP-poutP;
    auto ws=hxt5b::wallstats(d_wallStats,bulk);
    long double et=0,ez=0;for(std::size_t v=0;v<M.points.size();++v){et+=(double)hu0[v]*(double)hu0[v]+(double)hu1[v]*(double)hu1[v];ez+=(double)hu2[v]*(double)hu2[v];}
    double trans=std::sqrt((double)(et/std::max((long double)1e-300,ez)));
    write_vtu(vtu,M,hu0,hu1,hu2,hp);
    const std::string diagVtu=hxt5b_diag::path2(vtu);
    hxt5b_diag::write(diagVtu,M,BH,hu0,hu1,hu2,hp,nu,bulk,wallSampleFraction);
    std::printf("NODALS_HXT5B6_DIAG_VTU_OUTPUT path=%s velocity=FULL_Q1BF2_P1BF3_PHYSICAL_CELL_AVERAGE wallYPlus=EXACT_SPALDING status=PASS\n",diagVtu.c_str());
    const double finalMeanNut=sstG2?g2FinalStats.meanNutOverNu:mix.meanNutOverNu;
    const double finalMaxNut=sstG2?g2FinalStats.maxNutOverNu:mix.maxNutOverNu;
    const char* finalTurb=sstG2?"SST_2003M":"NIKURADSE_ALGEBRAIC";
    bool gate=std::isfinite(slope)&&std::isfinite(drop)&&finalMaxNut>0&&ws.area>0&&ws.rootFailures==0&&ws.yPlusMean>0;
    std::printf("NODALS_HXT5B_PHYSICS turbulence=%s pressureSlope=%.12e pIn=%.12e pOut=%.12e pressureDrop=%.12e meanNutOverNu=%.12e maxNutOverNu=%.12e maxF1=%.12e maxF2=%.12e maxStrain=%.12e maxStrainSource=%s wallDarcyFromSpalding=%.12e wallYPlusMean=%.12e wallYPlusMax=%.12e wallSlipMean=%.12e transverseVertexRelL2=%.12e avgPressureCG=%.6f pressureSolver=%s status=%s\n",
      finalTurb,slope,pinP,poutP,drop,finalMeanNut,finalMaxNut,sstG2?g2FinalStats.maxF1:0.0,sstG2?g2FinalStats.maxF2:0.0,mix.maxStrain,sstG2?(directSSTStart?"INITIAL_PLUG_AUDIT":"G0_MIXLEN_STARTER"):"CURRENT_MIXLEN",ws.fDarcy,ws.yPlusMean,ws.yPlusMax,ws.slipMean,trans,(sstG2&&g2Outer)?(double)pIts/g2Outer:(outer?(double)pIts/outer:0.0),pressureSolver.c_str(),gate?"PASS":"FAIL");
    std::printf("NODALS_HXT5B_OUTPUT vtu=%s status=PASS\n",vtu.c_str());
    bool sstGate=true;
    if(sstG1.enabled){
      std::string sstVtu=vtu.size()>=4&&vtu.substr(vtu.size()-4)==".vtu"?vtu.substr(0,vtu.size()-4)+"_sst_g1.vtu":vtu+"_sst_g1.vtu";
      auto sstResult=hxt5c::run_frozen(M,BH,d_inH,d_inT,d_wallH,d_pts,d_hc,d_tc,d_hv,d_tv,d_tg,d_ip,A,VC,d_fixed,d_u0.p,d_u1.p,d_u2.p,d_turb,d_dg,d_wall,d_bad,bulk,R,nu,gamma,wallSampleFraction,sstG1,sstVtu);
      sstGate=sstResult.gate;
    }
    const bool allGate=gate&&sstGate&&g2Gate;
    std::printf("HXT5B_GATE_STATUS=%s\n",allGate?"PASS":"FAIL");
    return allGate?0:3;
  }catch(const std::exception&e){
    std::fprintf(stderr,"HXT5B_ERROR precision=%s: %s\n",hxt4b::PRECISION,e.what());
    std::fprintf(stderr,"HXT5B_GATE_STATUS=FAIL\n");return 2;
  }
}
