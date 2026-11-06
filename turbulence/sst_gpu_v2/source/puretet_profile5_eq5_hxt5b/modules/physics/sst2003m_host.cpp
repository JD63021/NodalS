#include "sst2003m_host.hpp"
namespace nodals { namespace sst2003m {
LocalState<double> evaluate_host(const HostInput&in){
 return evaluate_pipe<double>(in.x,in.y,in.radius,in.nu,in.gu,in.k,in.omega,
                             in.gradK,in.gradOmega,in.kFloor,in.omegaFloor,
                             in.deviatoricProduction);
}
}}
