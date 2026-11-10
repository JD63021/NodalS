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
 c.nu=real("physics.kinematic_viscosity");c.alphaU=real("simple.ur_u");c.alphaP=real("simple.ur_p");
 c.convection=must("convection.form");c.stress=must("stress.form");c.method=must("solver.u.method");
 c.momSweeps=intval("solver.u.sweeps");c.momOmega=real("solver.u.omega");c.momRtol=real("solver.u.rtol");
 c.pRtol=real("solver.p.rtol");c.pMaxIts=intval("solver.p.max_iterations");c.spaldingScale=real("wall_function.spalding_sample_scale");c.nitscheGamma=real("wall_function.nitsche_gamma");
 c.momentumSupg=(must("supg.momentum")=="true");c.compactP1=(must("sst.compact_p1")=="true");
 std::string vel=must("inlet.main.velocity");for(char &q:vel)if(q=='('||q==')'||q==',')q=' ';std::istringstream vs(vel);
 if(!(vs>>c.inlet[0]>>c.inlet[1]>>c.inlet[2]))throw std::runtime_error("invalid inlet velocity vector");
 if(c.convection!="advective"||c.stress!="deviatoric"||c.method!="m3-fgs"||c.momentumSupg||c.compactP1)
   throw std::runtime_error("M3 does not support this case mode; refusing numerical substitution. Expected advective/deviatoric/m3-fgs, momentum SUPG off, masked P1.");
 if(!(c.nu>0 && c.alphaU>0 && c.alphaU<=1 && c.alphaP>0 && c.alphaP<=1 && c.momSweeps>0 && c.pMaxIts>0 && c.spaldingScale>0))
   throw std::runtime_error("invalid physical or solver case settings");
 return c;
}
}
