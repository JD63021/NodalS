#pragma once
// Optional rolling-iteration wall-Darcy / pressure-drop plateau gate.
// Pure host-side state. Legacy mode never changes native gate semantics.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdio>
namespace nodals { namespace coupling {
struct G2MetricPlateau {
  std::string mode="legacy";
  double tolerance=0.01;
  int window=25;
  bool requireResiduals=true;
  int lastIteration=-1;
  std::vector<double> wall;
  std::vector<double> drop;
  static std::string environment(const char* key,const char* otherwise){
    const char* val=std::getenv(key);return val&&*val?std::string(val):std::string(otherwise);
  }
  static G2MetricPlateau from_environment(){
    G2MetricPlateau g;
    g.mode=environment("NODALS_CONV_METRIC","legacy");
    g.tolerance=std::stod(environment("NODALS_CONV_RTOL","0.01"));
    g.window=std::stoi(environment("NODALS_CONV_WINDOW","25"));
    auto flag=environment("NODALS_CONV_REQUIRE_RESIDUALS","1");
    if(flag!="0" && flag!="1")throw std::runtime_error("NODALS_CONV_REQUIRE_RESIDUALS must be 0 or 1");
    g.requireResiduals=flag=="1";
    if(g.mode!="legacy" && g.mode!="pressure_drop" && g.mode!="wall_traction" && g.mode!="both")
      throw std::runtime_error("convergence.metric must be legacy, pressure_drop, wall_traction or both");
    if(!std::isfinite(g.tolerance) || g.tolerance<=0 || g.tolerance>=1)
      throw std::runtime_error("convergence.relative_tolerance must be in (0,1)");
    if(g.window<2 || g.window>10000)
      throw std::runtime_error("convergence.window_iterations must be between 2 and 10000");
    std::printf("NODALS_METRIC_GATE_CONFIG metric=%s tol=%.12g windowIterations=%d requireResiduals=%d status=PASS\n",
                g.mode.c_str(),g.tolerance,g.window,(int)g.requireResiduals);
    return g;
  }
  bool active()const{return mode!="legacy";}
  bool wantsWall()const{return mode=="wall_traction"||mode=="both";}
  bool wantsDrop()const{return mode=="pressure_drop"||mode=="both";}
  void observe(int it,double f,double dp,bool wallFresh){
    if(!active())return;
    if((lastIteration>=0 && it!=lastIteration+1) || !std::isfinite(dp) ||
       (wantsWall() && (!wallFresh || !std::isfinite(f)))){
      wall.clear();drop.clear();
    }
    lastIteration=it;
    if(!std::isfinite(dp) || (wantsWall() && (!wallFresh || !std::isfinite(f))))return;
    wall.push_back(f);drop.push_back(dp);
    if((int)wall.size()>window){wall.erase(wall.begin());drop.erase(drop.begin());}
  }
  double relativeSpan(const std::vector<double>& values)const{
    if((int)values.size()<window)return std::numeric_limits<double>::infinity();
    const auto extrema=std::minmax_element(values.begin(),values.end());
    double mean=0;for(double value:values)mean+=value;
    mean/=values.size();
    return (*extrema.second-*extrema.first)/std::max(std::abs(mean),1.0e-12);
  }
  double wallSpan()const{return relativeSpan(wall);}
  double dropSpan()const{return relativeSpan(drop);}
  bool passed()const{
    if(!active() || (int)drop.size()<window)return false;
    return (!wantsWall()||wallSpan()<=tolerance)&&(!wantsDrop()||dropSpan()<=tolerance);
  }
};
}} // namespace nodals::coupling
