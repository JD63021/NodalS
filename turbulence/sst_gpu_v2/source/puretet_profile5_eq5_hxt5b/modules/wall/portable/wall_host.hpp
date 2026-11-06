#pragma once
#include "spalding_core.hpp"
#include "of_auto_core.hpp"
namespace nodals { namespace wall {
struct HostOFInput {
  double y,nu,Ut,uTau,k,omega,kFloor,omegaFloor,kappa,E,beta1,reBlend,prodLimit,pk,nut;
  int sharedSpaldingWallState;
};
struct HostOFState { double omegaTarget,omegaVis,omegaLog,lamFrac,rey,yPlusK,pkAuto,nutSpalding; };
HostOFState evaluate_of_auto_host(const HostOFInput& in);
struct HostProfile { double uTau,yPlus,beta,uMid,fitRms,pointUTauRelSpan; bool valid; };
HostProfile fit_profile5_host(const double U[5], const double Usigned[5],const double y[5],double nu);
}}
