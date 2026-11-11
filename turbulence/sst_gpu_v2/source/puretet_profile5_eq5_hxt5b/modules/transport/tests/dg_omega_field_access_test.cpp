#include "../dg_omega_field_access.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <algorithm>
using namespace nodals::omega_dg_field;
static void equal(double a,double b){if(std::fabs(a-b)>1e-12)throw std::runtime_error("DG omega field mismatch");}
int main(){
  // Two neighbors possess independent values on their shared geometric face.
  const double omega[8]={1,2,3,4,10,20,30,40};
  const double inverseJ[9]={1,0,0,0,1,0,0,0,1};
  const auto grad=from_inverse_jacobian(inverseJ);
  const double lambda[4]={.1,.2,.3,.4};
  auto a=evaluate_cell(omega,0,grad,lambda);
  equal(a.value,3.0);equal(a.grad[0],1);equal(a.grad[1],2);equal(a.grad[2],3);
  auto b=evaluate_cell(omega,1,grad,lambda);
  equal(b.value,30.0);
  const int faceVertices[3]={0,1,2};
  const double bary[3]={1./3,1./3,1./3};
  auto left=trace(omega,0,grad,faceVertices,bary);
  auto right=trace(omega,1,grad,faceVertices,bary);
  if(std::fabs(left.value-right.value)<1.0)throw std::runtime_error("DG trace was accidentally made continuous");
  std::printf("DG_OMEGA_FIELD_ACCESS_PASS leftTrace=%.10g rightTrace=%.10g jump=%.10g\n",left.value,right.value,right.value-left.value);
  return 0;
}
