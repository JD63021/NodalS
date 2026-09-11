// NodalS HXT5B
// Full RANS boundary gate: mixing length + DG plug inlet + Firedrake-parity HEX off-wall Spalding wall + Cheb2 AMG.

#define HXT4B_EMBED_MAIN hxt4b_embedded_main
#include "../hybrid_hxt4b/hxt4b_main.cu"
#undef HXT4B_EMBED_MAIN

#include "hybrid_amg.cuh"
#include "hxt5a_mixlen.cuh"
#include "hxt5b_boundary.cuh"

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
    double Re=13691.740248127866,bulk=50,gamma=50,alphaU=.7,alphaP=1,rauScale=2,simpleTol=1e-3,momOmega=1,momRtol=.5,momAtol=1e-8,mixlenScale=1.0,outerContAtol=1e-12,wallSampleFraction=.5;
    double pRtol=.9,pAtol=1e-12,cfTheta=.25,amgJacobiOmega=.7,amgLambdaSafety=1.5,amgLambdaLow=.05,amgMcgsOmega=1;
    int maxOuter=3000,pMax=50,momMax=50,cfPmax=8,cfAggressiveFirst=0,amgTerminal=1000,amgPowerIts=16,amgChebDegree=4;
    int amgJacobiFineSweeps=1,amgJacobiCoarseSweeps=1,amgMcgsFineSweeps=1,amgMcgsCoarseSweeps=1;

    for(int i=1;i<argc;++i){
      std::string a=argv[i];auto nx=[&](){if(i+1>=argc)throw std::runtime_error("missing value for "+a);return std::string(argv[++i]);};
      if(a=="--mesh")mesh=nx();else if(a=="--vtu")vtu=nx();else if(a=="--re")Re=std::stod(nx());else if(a=="--bulk")bulk=std::stod(nx());else if(a=="--mixlen-scale")mixlenScale=std::stod(nx());else if(a=="--outer-cont-atol")outerContAtol=std::stod(nx());
      else if(a=="--wall-sample-fraction")wallSampleFraction=std::stod(nx());
      else if(a=="--gamma")gamma=std::stod(nx());else if(a=="--alpha-u")alphaU=std::stod(nx());else if(a=="--alpha-p")alphaP=std::stod(nx());
      else if(a=="--rau-scale")rauScale=std::stod(nx());else if(a=="--simple-tol")simpleTol=std::stod(nx());else if(a=="--max-outer")maxOuter=std::stoi(nx());
      else if(a=="--momentum-work")momWork=nx();else if(a=="--mom-omega")momOmega=std::stod(nx());else if(a=="--mom-rtol")momRtol=std::stod(nx());else if(a=="--mom-atol")momAtol=std::stod(nx());else if(a=="--mom-max")momMax=std::stoi(nx());else if(a=="--mom-cap-policy")momCapPolicy=nx();
      else if(a=="--pressure-solver")pressureSolver=nx();else if(a=="--p-rtol")pRtol=std::stod(nx());else if(a=="--p-atol")pAtol=std::stod(nx());else if(a=="--p-max")pMax=std::stoi(nx());
      else if(a=="--amg-hierarchy")amgHierarchy=nx();else if(a=="--cf-coarsening")cfCoarsening=nx();else if(a=="--cf-strength")cfStrength=nx();
      else if(a=="--cf-theta")cfTheta=std::stod(nx());else if(a=="--cf-interp")cfInterp=nx();else if(a=="--cf-pmax")cfPmax=std::stoi(nx());else if(a=="--cf-aggressive-first")cfAggressiveFirst=std::stoi(nx());
      else if(a=="--amg-terminal")amgTerminal=std::stoi(nx());else if(a=="--amg-smoother")amgSmoother=nx();else if(a=="--amg-jacobi-omega")amgJacobiOmega=std::stod(nx());
      else if(a=="--amg-jacobi-fine-sweeps")amgJacobiFineSweeps=std::stoi(nx());else if(a=="--amg-jacobi-coarse-sweeps")amgJacobiCoarseSweeps=std::stoi(nx());
      else if(a=="--amg-cheb-degree")amgChebDegree=std::stoi(nx());else if(a=="--amg-power-its")amgPowerIts=std::stoi(nx());
      else if(a=="--amg-lambda-safety")amgLambdaSafety=std::stod(nx());else if(a=="--amg-lambda-low-fraction")amgLambdaLow=std::stod(nx());
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
    if(!(momCapPolicy=="warn"||momCapPolicy=="fail"))throw std::runtime_error("--mom-cap-policy must be warn or fail");
    if(!(mixlenScale>=0.0)||!(outerContAtol>=0.0)||!(wallSampleFraction>0.0&&wallSampleFraction<1.0))throw std::runtime_error("bad HXT5B turbulence/boundary/convergence option");

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
      alphaU,rauScale,d_bad.p);
    HXT1_CUDA(cudaDeviceSynchronize());HXT1_CUDA(cudaMemcpy(&bad,d_bad.p,sizeof(bad),cudaMemcpyDeviceToHost));
    if(bad)throw std::runtime_error("initial nonpositive directional relaxed/rAU diagonal");

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
      "alphaU=%.6g alphaP=%.6g rauScale=%.6g directionalRAU=1 rauPenaltyIncluded=0 simpleTol=%.3e status=PASS\n",
      PRECISION,(unsigned long long)M.h.nhex,(unsigned long long)M.h.ntet,nv,np,Re,bulk,nu,gamma,mixlenScale,
      wallSampleFraction,momWork.c_str(),momRtol,momAtol,momMax,momCapPolicy.c_str(),pressureSolver.c_str(),pRtol,pAtol,pMax,amgHierarchy.c_str(),cfCoarsening.c_str(),
      cfStrength.c_str(),cfTheta,cfInterp.c_str(),cfPmax,amgSmoother.c_str(),amgChebDegree,effectivePowerIts,
      amgSpectrumPolicy.c_str(),alphaU,alphaP,rauScale,simpleTol);
    std::printf(
      "NODALS_HXT5B_BOUNDARY_SETUP inletHexFaces=%zu inletTetFaces=%zu wallHexFaces=%zu inletArea=%.12e "
      "prescribedFluxSum=%.12e constantPatchResidual=%.12e wallArea=%.12e wallYPlusMeanInitial=%.12e "
      "wallDarcyInitial=%.12e rootFailures=%.0f status=PASS\n",
      BH.inH.size(),BH.inT.size(),BH.wallH.size(),BH.inletArea,BH.sourceSum,BH.constResidual,
      wallNow.area,wallNow.yPlusMean,wallNow.fDarcy,wallNow.rootFailures);
    std::printf("NODALS_HXT5B_MIXLEN_SETUP meanNutOverNu=%.12e maxNutOverNu=%.12e maxStrain=%.12e status=PASS\n",
      mix.meanNutOverNu,mix.maxNutOverNu,mix.maxStrain);

    double c0=-1,m0[3]={-1,-1,-1};bool converged=false;int outer=0;long long pIts=0;
    for(int it=1;it<=maxOuter;++it){
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
      alphaU,rauScale,d_bad.p);
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
        std::printf("NODALS_HXT5B_SIMPLE it=%d relContInit=%.12e absCont=%.12e relMomXInit=%.12e absMomX=%.12e relMomYInit=%.12e absMomY=%.12e relMomZInit=%.12e absMomZ=%.12e pCG=%d pRel=%.3e meanNutOverNu=%.6e maxNutOverNu=%.6e wallYPlusMean=%.6e wallDarcy=%.6e converged=%d status=PASS\n",
          it,rc,cn,rm0,ma[0],rm1,ma[1],rm2,ma[2],pr.its,pr.rel,mix.meanNutOverNu,mix.maxNutOverNu,ws.yPlusMean,ws.fDarcy,(int)converged);
      }
      if(converged)break;
      double worst=std::max(std::max(rc,rm0),std::max(rm1,rm2));
      if(it>=5&&(!std::isfinite(worst)||worst>1e8))throw std::runtime_error("HXT5B coupled-residual divergence guard");
    }
    if(!converged)throw std::runtime_error("HXT5B SIMPLE maxOuter without all-residual convergence");

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
    bool gate=std::isfinite(slope)&&std::isfinite(drop)&&mix.maxNutOverNu>0&&ws.area>0&&ws.rootFailures==0&&ws.yPlusMean>0;
    std::printf("NODALS_HXT5B_PHYSICS pressureSlope=%.12e pIn=%.12e pOut=%.12e pressureDrop=%.12e meanNutOverNu=%.12e maxNutOverNu=%.12e maxStrain=%.12e wallDarcyFromSpalding=%.12e wallYPlusMean=%.12e wallYPlusMax=%.12e wallSlipMean=%.12e transverseVertexRelL2=%.12e avgPressureCG=%.6f pressureSolver=%s status=%s\n",
      slope,pinP,poutP,drop,mix.meanNutOverNu,mix.maxNutOverNu,mix.maxStrain,ws.fDarcy,ws.yPlusMean,ws.yPlusMax,ws.slipMean,trans,outer?(double)pIts/outer:0.0,pressureSolver.c_str(),gate?"PASS":"FAIL");
    std::printf("NODALS_HXT5B_OUTPUT vtu=%s status=PASS\n",vtu.c_str());
    std::printf("HXT5B_GATE_STATUS=%s\n",gate?"PASS":"FAIL");
    return gate?0:3;
  }catch(const std::exception&e){
    std::fprintf(stderr,"HXT5B_ERROR precision=%s: %s\n",hxt4b::PRECISION,e.what());
    std::fprintf(stderr,"HXT5B_GATE_STATUS=FAIL\n");return 2;
  }
}
