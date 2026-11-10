#pragma once
#include <array>
#include <map>
#include <string>
namespace nodals::cpu_mc::m3 {
struct CaseConfig {
 std::string mesh,casePath,convection,stress,method;
 std::array<double,3> inlet{};
 double nu=0,alphaU=0.7,alphaP=0.7,momOmega=0.7,momRtol=1e-6,pRtol=1e-6,spaldingScale=0.5,nitscheGamma=1e-6;
 int momSweeps=40,pMaxIts=600,expectRanks=16;
 bool momentumSupg=false, compactP1=false;
};
CaseConfig read_case(const std::string& path,const std::string& meshPath,int expectedRanks);
}
