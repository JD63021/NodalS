#include "case_config.hpp"
#include <fstream>
#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <cmath>
namespace nodals::cpu_mc::m3 {
static std::string trim(std::string s){const auto b=s.find_first_not_of(" \t\r\n");if(b==std::string::npos)return "";const auto e=s.find_last_not_of(" \t\r\n");return s.substr(b,e-b+1);}
CaseConfig read_case(const std::string& path,const std::string& meshPath,int n){
 std::ifstream in(path);if(!in)throw std::runtime_error("cannot open case: "+path);
 std::map<std::string,std::string> a;std::string line,section;
 while(std::getline(in,line)){
   auto comment=line.find('#');if(comment!=std::string::npos)line.resize(comment);
   line=trim(line);if(line.empty())continue;
   if(line.front()=='[' && line.back()==']'){section=trim(line.substr(1,line.size()-2));continue;}
   auto eq=line.find('=');if(eq!=std::string::npos)a[section+"."+trim(line.substr(0,eq))]=trim(line.substr(eq+1));
 }
 auto must=[&](std::string key)->std::string{if(!a.count(key))throw std::runtime_error("case missing required "+key);return a.at(key);};
 auto real=[&](std::string key)->double{return std::stod(must(key));};
 auto intval=[&](std::string key)->int{return std::stoi(must(key));};
 CaseConfig c;c.casePath=path;c.mesh=meshPath;c.expectRanks=n;
 c.nu=real("physics.kinematic_viscosity");c.diameter=real("mesh.diameter");c.alphaU=real("simple.ur_u");c.alphaP=real("simple.ur_p");
 c.convection=must("convection.form");c.stress=must("stress.form");c.method=must("solver.u.method");
 c.momSweeps=intval("solver.u.sweeps");c.momOmega=real("solver.u.omega");c.momRtol=real("solver.u.rtol");
 c.pRtol=real("solver.p.rtol");c.pMaxIts=intval("solver.p.max_iterations");c.spaldingScale=real("wall_function.spalding_sample_scale");c.nitscheGamma=real("wall_function.nitsche_gamma");
 c.wallSamples=intval("wall_function.samples");
 c.maxCrossings=std::stoi(a.count("wall_sampling.max_crossings")?a.at("wall_sampling.max_crossings"):"4096");
 c.baryTolerance=std::stod(a.count("wall_sampling.bary_tolerance")?a.at("wall_sampling.bary_tolerance"):"1e-9");
 c.zeroLimit=(a.count("wall_function.zero_limit")?a.at("wall_function.zero_limit"):"true")=="true";
 // Only permit local-shortening when it is explicitly requested in the case.
 c.localRayAdapt=(a.count("wall_sampling.local_adapt")?a.at("wall_sampling.local_adapt"):"false")=="true";
 c.spaldingY3=4.4084029504620439e-4*(c.diameter/0.004)*c.spaldingScale;
 if(a.count("wall_function.distances")){
   auto raw=a.at("wall_function.distances");for(char&ch:raw)if(ch==',')ch=' ';
   double y[5]{};std::istringstream yy(raw);for(int j=0;j<5;++j)if(!(yy>>y[j]))throw std::runtime_error("bad wall_function.distances");
   c.spaldingY3=y[2];
 }
 if(must("wall_sampling.mode")!="ray_trace")throw std::runtime_error("M3 supports only strict ray_trace, cannot change numerical sampling");
 c.momentumSupg=(must("supg.momentum")=="true");c.compactP1=(must("sst.compact_p1")=="true");
 std::string vel=must("inlet.main.velocity");for(char &q:vel)if(q=='('||q==')'||q==',')q=' ';std::istringstream vs(vel);
 if(!(vs>>c.inlet[0]>>c.inlet[1]>>c.inlet[2]))throw std::runtime_error("invalid inlet velocity vector");
 if(c.convection!="advective"||c.stress!="deviatoric"||c.method!="m3-fgs"||c.momentumSupg||c.compactP1)
   throw std::runtime_error("M3 does not support this case mode; refusing numerical substitution. Expected advective/deviatoric/m3-fgs, momentum SUPG off, masked P1.");
 if(!(c.nu>0 && c.alphaU>0 && c.alphaU<=1 && c.alphaP>0 && c.alphaP<=1 && c.momSweeps>0 && c.pMaxIts>0 && c.spaldingScale>0 && c.spaldingY3>0 && c.wallSamples==1 && c.maxCrossings>0 && c.baryTolerance>0))
   throw std::runtime_error("invalid physical or solver case settings");
 return c;
}
}
