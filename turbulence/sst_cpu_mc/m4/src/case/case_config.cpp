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
 c.enrichEnabled=a.count("enrichment.enabled")&&a.at("enrichment.enabled")=="true";
 c.enrichUTau=a.count("enrichment.utau_initial")?std::stod(a.at("enrichment.utau_initial")):0.5859;
 c.enrichCutoff=a.count("enrichment.yplus_cutoff")?std::stod(a.at("enrichment.yplus_cutoff")):80.;
 c.enrichRadialQuad=a.count("enrichment.quadrature_order")?std::stoi(a.at("enrichment.quadrature_order")):24;
 if(c.enrichEnabled&&(!(c.enrichUTau>0)||!(c.enrichCutoff>0)||
    c.enrichRadialQuad<4||c.enrichRadialQuad>48))
   throw std::runtime_error("invalid enrichment initial wall friction velocity / cutoff / quadrature");
 if(c.enrichEnabled && a.count("wall_function.momentum_bc") &&
    a.at("wall_function.momentum_bc")!="strong_noslip_enriched")
   throw std::runtime_error("enrichment requires wall_function.momentum_bc=strong_noslip_enriched");
 c.nu=real("physics.kinematic_viscosity");c.diameter=real("mesh.diameter");c.alphaU=real("simple.ur_u");c.alphaP=real("simple.ur_p");
 c.convection=must("convection.form");c.stress=must("stress.form");c.method=must("solver.u.method");
 c.momSweeps=a.count("solver.u.sweeps")?intval("solver.u.sweeps"):2;
 c.momOmega=a.count("solver.u.omega")?real("solver.u.omega"):0.7;
 c.momRtol=a.count("solver.u.rtol")?real("solver.u.rtol"):1e-4;
 c.momAtol=a.count("solver.u.atol")?real("solver.u.atol"):1e-10;
 c.momMaxIts=a.count("solver.u.max_iterations")?intval("solver.u.max_iterations"):400;
 c.momRestart=a.count("solver.u.restart")?intval("solver.u.restart"):30;
 c.pRtol=real("solver.p.rtol");c.pAtol=a.count("solver.p.atol")?real("solver.p.atol"):1e-12;c.pMaxIts=intval("solver.p.max_iterations");c.spaldingScale=real("wall_function.spalding_sample_scale");c.nitscheGamma=real("wall_function.nitsche_gamma");
 c.wallSamples=intval("wall_function.samples");
 if(must("wall_function.face_spalding")!="true")throw std::runtime_error("M4 CPU F2 supports only face_spalding=true for the captured GPU case; refusing quadrature-fit substitution");
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
   c.omegaPenaltyLength=y[4]-y[0]; // GPU profile5 y5-y1 thickness
 }
 if(must("wall_sampling.mode")!="ray_trace")throw std::runtime_error("M3 supports only strict ray_trace, cannot change numerical sampling");
 if(must("mesh.format")!="msh2")throw std::runtime_error("M3 CPU expects unchanged tetrahedral MSH2 mesh");
 if(must("initialization.mode")!="potential_flow")throw std::runtime_error("M3 Stage27 CPU path needs potential_flow, refusing numerical substitution");
 if(must("solver.p.method")!="pcg_amg")throw std::runtime_error("M3 CPU pressure backend needs pcg_amg");
 if(must("wall_function.law")!="spalding")throw std::runtime_error("M3 CPU wall model must be Spalding");
 if(std::stoi(must("wall_function.integral_nq"))!=12)throw std::runtime_error("M3 wall quadrature requires 12-point GPU rule");
 c.momentumSupg=(must("supg.momentum")=="true");c.compactP1=(must("sst.compact_p1")=="true");
 c.sstSpace=must("sst.state_space");c.intensity=real("sst.intensity");c.lengthRatio=real("sst.length_ratio");
 c.urK=real("sst.ur_k");c.urOmega=real("sst.ur_omega");
 c.kSolver=must("sst.k_solver");c.omegaSolver=must("sst.omega_solver");
 c.kSweeps=intval("sst.k_sweeps");c.omegaSweeps=intval("sst.omega_sweeps");
 c.supgK=(must("supg.k")=="true");c.supgOmega=(must("supg.omega")=="true");
 c.omegaMethod=must("omega_wall.method");c.omegaStage41=must("omega_wall.stage41_mode");
 c.omegaScale=real("omega_wall.sample_scale");c.omegaIndex=intval("omega_wall.sample_index");
 c.omegaProduction=(must("omega_wall.production")=="true");
 c.omegaTraceGamma=a.count("omega_wall.trace_gamma")?real("omega_wall.trace_gamma"):50.0;
 c.omegaPointGamma=a.count("omega_wall.penalty_gamma")?real("omega_wall.penalty_gamma"):10000.0;
 c.omegaOfReBlend=a.count("sst.of_re_blend")?real("sst.of_re_blend"):11.0;
 if(a.count("omega_wall.penalty_length"))
   c.omegaPenaltyLength=real("omega_wall.penalty_length");
 if(!(c.omegaPenaltyLength>0))c.omegaPenaltyLength=4.0*c.spaldingY3/3.0;
 c.omegaY3=4.4084029504620439e-4*(c.diameter/0.004)*c.omegaScale;
 c.maxOuter=intval("simple.max_iterations");c.simpleTol=real("simple.simple_tol");
 c.continuityAtol=real("simple.continuity_atol");c.continuityRtol=a.count("simple.continuity_rtol")?real("simple.continuity_rtol"):1e-6;
 c.plateauTol=real("simple.plateau_tol");
 // Pressure-drop plateau uses the face-area-weighted P0 pressure difference
 // over the last N SIMPLE iterates. The window RANGE must be <= tol*|mean|.
 c.pressureDropWindow=a.count("convergence.pressure_drop_window")?intval("convergence.pressure_drop_window"):20;
 c.pressureDropMinIterations=a.count("convergence.min_iterations")?intval("convergence.min_iterations"):30;
 c.pressureDropTol=a.count("convergence.pressure_drop_relative_tolerance")?real("convergence.pressure_drop_relative_tolerance"):0.01;
 if(a.count("convergence.method")&&must("convergence.method")!="pressure_drop_window")
  throw std::runtime_error("CPU M4 convergence.method requires pressure_drop_window");
 c.printEvery=intval("output.print_every");c.vtuEvery=intval("output.vtu_every");
 c.outputDirectory=must("output.directory");
 c.adaptiveSampling=(must("adaptive_sampling.enabled")=="true");
 c.adaptiveFirst=intval("adaptive_sampling.first_refresh");
 c.adaptiveEvery=intval("adaptive_sampling.refresh_every");
 c.adaptiveMax=intval("adaptive_sampling.max_refreshes");
 c.adaptiveTargetSp=real("adaptive_sampling.target_spalding_yplus");
 c.adaptiveTargetOmega=real("adaptive_sampling.target_omega_yplus");
 c.adaptiveScaleMin=real("adaptive_sampling.min_scale");c.adaptiveScaleMax=real("adaptive_sampling.max_scale");

 std::string vel=must("inlet.main.velocity");for(char &q:vel)if(q=='('||q==')'||q==',')q=' ';std::istringstream vs(vel);
 if(!(vs>>c.inlet[0]>>c.inlet[1]>>c.inlet[2]))throw std::runtime_error("invalid inlet velocity vector");
 if(c.convection!="advective"||c.stress!="deviatoric"||(c.method!="m3-fgs"&&c.method!="fgmres")||c.momentumSupg||c.compactP1)
   throw std::runtime_error("M3 does not support this case mode; refusing numerical substitution. Expected advective/deviatoric/fgmres or m3-fgs, momentum SUPG off, masked P1.");
 if(c.sstSpace!="q1"||c.kSolver!="fgs-fixed"||c.omegaSolver!="fgs-fixed"||
    c.omegaMethod!="of_auto"||
    (c.omegaStage41!="trace_log"&&c.omegaStage41!="of_auto"&&c.omegaStage41!="sample_log")||
    c.omegaIndex!=3||
    c.omegaProduction||c.supgK||c.supgOmega)
   throw std::runtime_error("M4 supports the captured spiral GPU SST q1/fgs-fixed/trace_log configuration; refusing to replace an unsupported option");
 if(!(c.intensity>0&&c.lengthRatio>0&&c.urK>0&&c.urK<=1&&c.urOmega>0&&c.urOmega<=1&&
      c.kSweeps>0&&c.omegaSweeps>0&&c.maxOuter>0&&c.omegaScale>0&&c.omegaTraceGamma>=10&&
      c.omegaPointGamma>=1&&c.omegaPointGamma<=1e8&&
      c.omegaPenaltyLength>0&&c.omegaOfReBlend>0))
   throw std::runtime_error("invalid SST config values");
 if(!(c.pRtol>=0&&c.pAtol>=0&&(c.pRtol>0||c.pAtol>0)&&c.continuityAtol>=0&&c.continuityRtol>=0&&
      c.pressureDropWindow>=2&&c.pressureDropMinIterations>=c.pressureDropWindow&&c.pressureDropTol>0&&c.pressureDropTol<1))
    throw std::runtime_error("invalid solver.p absolute/relative tolerance or convergence window settings");
 if(!(c.momRtol>=0 && c.momAtol>=0 && (c.momRtol>0 || c.momAtol>0) &&
       c.momMaxIts>0 && c.momRestart>=2 && c.momRestart<=c.momMaxIts))
   throw std::runtime_error("invalid solver.u FGMRES rtol/atol/restart/max_iterations");
 if(!(c.nu>0 && c.alphaU>0 && c.alphaU<=1 && c.alphaP>0 && c.alphaP<=1 && c.momSweeps>0 && c.pMaxIts>0 && c.spaldingScale>0 && c.spaldingY3>0 && c.wallSamples==1 && c.maxCrossings>0 && c.baryTolerance>0))
   throw std::runtime_error("invalid physical or solver case settings");
 return c;
}
}
