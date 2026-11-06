#pragma once
// Standalone compiled C++ API: requires no HXT or CUDA types.
#include "sst2003m_core.hpp"
namespace nodals { namespace sst2003m {
struct HostInput {
 double x=0,y=0,radius=0.002,nu=1.460734693878e-5;
 double gu[9]{};double k=0,omega=0;double gradK[3]{},gradOmega[3]{};
 double kFloor=1e-8,omegaFloor=1e-3;
 int deviatoricProduction=0;
};
LocalState<double> evaluate_host(const HostInput&in);
}}
