#pragma once
#include <array>
#include <map>
#include <string>
namespace nodals::cpu_mc::m3 {
struct CaseConfig {
 std::string mesh,casePath,convection,stress,method;
 std::array<double,3> inlet{};
 double diameter=0,spaldingY3=0;
 // Optional grad-div coefficient, with kinematic-viscosity units [m^2/s].
  // Zero reproduces the original momentum matrix (NO change to B or B^T).
  double gradDivGamma=0.0;
  double nu=0,alphaU=0.7,alphaP=0.7,momOmega=0.7,momRtol=1e-4,momAtol=1e-10,pRtol=1e-6,spaldingScale=0.5,nitscheGamma=1e-6;
 int momSweeps=40,momMaxIts=400,momRestart=30,pMaxIts=600,expectRanks=16,wallSamples=1,maxCrossings=4096;
 double baryTolerance=1e-9;
 bool zeroLimit=true,localRayAdapt=false;
 bool momentumSupg=false, compactP1=false;
 // M4: same fields and defaults as the captured GPU SST spiral .case.
 std::string sstSpace="q1",omegaMethod="of_auto",omegaStage41="trace_log";
 std::string kSolver="fgs-fixed",omegaSolver="fgs-fixed",outputDirectory=".";
 double intensity=.05,lengthRatio=.07,urK=.7,urOmega=.7;
 double kFloor=1e-8,omegaFloor=1e-3,omegaScale=.5,omegaY3=0,omegaTraceGamma=50;
  double omegaPointGamma=10000.0,omegaPenaltyLength=0.0,omegaOfReBlend=11.0;
  // Optional equilibrium k wall constraint at same traced interior point3.
  // no_flux retains the original natural boundary condition bit for bit.
  std::string kWallMethod="no_flux";
  double kWallGamma=1000.0,kWallPenaltyLength=0.0;
  int kWallSampleIndex=3;
 double pAtol=1e-12,continuityRtol=1e-6,pressureDropTol=0.01;
 int pressureDropWindow=20,pressureDropMinIterations=30;
 double scalarOmega=1.0,simpleTol=1e-4,continuityAtol=1e-8,plateauTol=.001;
 int kSweeps=40,omegaSweeps=40,omegaIndex=3,maxOuter=500,printEvery=10,vtuEvery=100;
 bool supgK=false,supgOmega=false,omegaProduction=false,adaptiveSampling=false;
 // Explicitly opt-in; baseline remains bit-for-bit unchanged when disabled.
 bool enrichEnabled=false;
 double enrichUTau=0.5859,enrichCutoff=80.0;
 int enrichRadialQuad=24;
 int adaptiveFirst=50,adaptiveEvery=500,adaptiveMax=2;
 double adaptiveTargetSp=30,adaptiveTargetOmega=30,adaptiveScaleMin=.05,adaptiveScaleMax=.9;

};
CaseConfig read_case(const std::string& path,const std::string& meshPath,int expectedRanks);
}
