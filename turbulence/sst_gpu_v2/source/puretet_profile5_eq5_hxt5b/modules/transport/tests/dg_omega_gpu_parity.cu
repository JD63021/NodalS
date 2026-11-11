// Compile with nvcc -std=c++17 -O2 dg_omega_gpu_parity.cu -o dg_omega_gpu_parity
// Compare CUDA BSR4 operators with the portable mathematical oracle.
#include "../dg_omega_p1_reference.hpp"
#include "../dg_omega_bsr4.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nodals::omega_dg;
namespace gpu=nodals::omega_dg_gpu;
#define CK(x) do {cudaError_t e=(x);if(e!=cudaSuccess)throw std::runtime_error(std::string(#x)+": "+cudaGetErrorString(e));}while(0)

template<class T> struct Buffer {
 T* p=nullptr;size_t n=0;
 Buffer(size_t count):n(count){CK(cudaMalloc((void**)&p,n*sizeof(T)));}
 ~Buffer(){if(p)cudaFree(p);}
 Buffer(const Buffer&)=delete;Buffer&operator=(const Buffer&)=delete;
 void send(const std::vector<T>&src){if(src.size()!=n)throw std::runtime_error("send size");CK(cudaMemcpy(p,src.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
 std::vector<T> recv()const{std::vector<T>dst(n);CK(cudaMemcpy(dst.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return dst;}
};
static gpu::Vec3 cv(Vec3 a){return {a.x,a.y,a.z};}
struct Packed {
 std::vector<gpu::Cell> cells;
 std::vector<gpu::Face> faces;
 std::vector<int> row,col,diag;
};
static Packed pack(const Mesh&m,const std::vector<Coeff>&coeff){
 const int n=(int)m.tetra.size();Packed p;
 std::vector<std::set<int>>adj((size_t)n);
 for(int t=0;t<n;++t)adj[(size_t)t].insert(t);
 for(auto &f:m.faces)if(f.right.tet>=0){
   adj[(size_t)f.left.tet].insert(f.right.tet);
   adj[(size_t)f.right.tet].insert(f.left.tet);
 }
 std::map<std::pair<int,int>,int>slots;
 p.row.push_back(0);
 for(int t=0;t<n;++t){
   for(int v:adj[(size_t)t]){
     slots[{t,v}]=(int)p.col.size();p.col.push_back(v);
   }
   p.row.push_back((int)p.col.size());
   p.diag.push_back(slots.at({t,t}));
 }
 for(int t=0;t<n;++t){
   gpu::Cell z{};auto&c=coeff[(size_t)t];auto&g=m.cells[(size_t)t];
   z.volume=g.volume;for(int a=0;a<4;++a)z.grad[a]=cv(g.grad[(size_t)a]);
   z.mu=c.molecularAndTurbulentDiffusion;z.reaction=c.reaction;
   z.source=c.source;z.u=cv(c.velocity);
   p.cells.push_back(z);
 }
 for(const auto &f:m.faces){
   gpu::Face out{};
   out.left=f.left.tet;out.right=f.right.tet;
   out.area=f.area;out.normal=cv(f.normal);
   out.slotLL=slots.at({out.left,out.left});
   out.slotLR=out.slotRL=out.slotRR=-1;
   if(out.right>=0){
     out.slotLR=slots.at({out.left,out.right});
     out.slotRL=slots.at({out.right,out.left});
     out.slotRR=slots.at({out.right,out.right});
   }
   for(int j=0;j<3;++j){
     const int v=f.vertices[(size_t)j];
     int il=-1,ir=-1;
     for(int a=0;a<4;++a){
       if(m.tetra[(size_t)out.left][(size_t)a]==v)il=a;
       if(out.right>=0&&m.tetra[(size_t)out.right][(size_t)a]==v)ir=a;
     }
     if(il<0||(out.right>=0&&ir<0))throw std::runtime_error("broken face mapping");
     out.vertexL[j]=il;out.vertexR[j]=ir;
   }
   if(out.right<0){
     const auto it=m.boundary.find(f.vertices);
     if(it==m.boundary.end())throw std::runtime_error("unclassified boundary");
     out.boundary=(int)it->second.kind;
     for(int q=0;q<3;++q)if(it->second.target)
       out.target[q]=it->second.target(face_x(m,f.vertices,QTRI[(size_t)q]));
   }
   p.faces.push_back(out);
 }
 return p;
}
static Mesh mesh(){
 Mesh m;
 m.vertices={{0,0,0},{1,0,0},{0,1,0},{0,0,1},{0,0,-1}};
 m.tetra={{0,1,2,3},{0,2,1,4}};
 initialize(m);
 return m;
}
static void check(double got,double expected,double tol,const char*name){
 if(!std::isfinite(got)||fabs(got-expected)>tol){
   std::fprintf(stderr,"%s mismatch got=%.17g reference=%.17g err=%.9g\n",name,got,expected,got-expected);
   throw std::runtime_error("GPU parity failed");
 }
}
static void exercise(const char*label,Mesh m,std::vector<Coeff> c,Options opt){
 const auto cpu=assemble(m,c,opt);
 const Packed p=pack(m,c);
 const int nt=(int)p.cells.size(),nd=4*nt,nb=(int)p.col.size();
 Buffer<gpu::Cell>gc(p.cells.size());gc.send(p.cells);
 Buffer<gpu::Face>gf(p.faces.size());gf.send(p.faces);
 Buffer<int>dr(p.row.size());dr.send(p.row);
 Buffer<int>di(p.col.size());di.send(p.col);
 Buffer<int>dd(p.diag.size());dd.send(p.diag);
 Buffer<double>mat((size_t)nb*16),rhs((size_t)nd);
 Buffer<unsigned int>bad(1);bad.send(std::vector<unsigned int>{0});
 CK(cudaMemset(mat.p,0,nb*16*sizeof(double)));
 CK(cudaMemset(rhs.p,0,nd*sizeof(double)));
 gpu::Controls o;o.penaltyC=opt.penaltyC;o.theta=opt.convectionTheta;
 o.traceGamma=opt.boundaryTraceGamma;o.wallNormalTolerance=opt.wallNormalTolerance;
 gpu::volume<<<(nt+127)/128,128>>>(nt,gc.p,dd.p,mat.p,rhs.p,bad.p);
 CK(cudaGetLastError());
 gpu::faces<<<((int)p.faces.size()+127)/128,128>>>(
   (int)p.faces.size(),gf.p,gc.p,o,mat.p,rhs.p,bad.p);
 CK(cudaGetLastError());CK(cudaDeviceSynchronize());
 const auto failures=bad.recv();
 if(failures[0])throw std::runtime_error("GPU assembly error counter="+std::to_string(failures[0]));
 auto a=mat.recv(),b=rhs.recv();
 double maxError=0;
 for(int t=0;t<nt;++t)for(int block=p.row[(size_t)t];block<p.row[(size_t)t+1];++block){
   const int s=p.col[(size_t)block];
   for(int i=0;i<4;++i)for(int j=0;j<4;++j){
     double actual=a[(size_t)16*block+4*i+j];
     auto it=cpu.rows[(size_t)4*t+i].find(4*s+j);
     double expected=it==cpu.rows[(size_t)4*t+i].end()?0.:it->second;
     maxError=std::max(maxError,std::abs(actual-expected));
     check(actual,expected,1.e-8,"matrix");
   }
 }
 for(int i=0;i<nd;++i)check(b[(size_t)i],cpu.rhs[(size_t)i],1.e-8,"rhs");
 std::vector<double> x((size_t)nd);
 for(int i=0;i<nd;++i)x[(size_t)i]=1.+0.2*i;
 Buffer<double>dx((size_t)nd),dy((size_t)nd),dz((size_t)nd);
 dx.send(x);
 gpu::bsr4_spmv<<<(nd+127)/128,128>>>(nt,dr.p,di.p,mat.p,dx.p,dy.p);
 CK(cudaGetLastError());CK(cudaDeviceSynchronize());
 auto gpuAx=dy.recv(),refAx=cpu.multiply(x);
 for(int i=0;i<nd;++i)check(gpuAx[(size_t)i],refAx[(size_t)i],1.e-8,"SpMV");
 std::vector<double> residual((size_t)nd);
 for(int i=0;i<nd;++i)residual[(size_t)i]=b[(size_t)i]-gpuAx[(size_t)i];
 auto refZ=block_jacobi(cpu,residual);
 dx.send(residual);
 gpu::bsr4_block_jacobi<<<(nt+127)/128,128>>>(nt,mat.p,dd.p,dx.p,dz.p,bad.p);
 CK(cudaGetLastError());CK(cudaDeviceSynchronize());
 auto gpuZ=dz.recv();
 if(bad.recv()[0])throw std::runtime_error("invalid GPU diagonal block");
 for(int i=0;i<nd;++i)check(gpuZ[(size_t)i],refZ[(size_t)i],1.e-8,"block Jacobi");
 std::printf("DG_OMEGA_GPU_PARITY_PASS scenario=%s cells=%d faces=%zu bsrBlocks=%d maxMatrixError=%.4e\n",
             label,nt,p.faces.size(),nb,maxError);
}
int main(){
 try{
   Mesh m=mesh();
   std::vector<Coeff> c(2);
   c[0].molecularAndTurbulentDiffusion=.1;c[1].molecularAndTurbulentDiffusion=.2;
   c[0].reaction=.5;c[1].reaction=.3;
   c[0].source=1.25;c[1].source=2.;
   for(auto&kv:m.boundary){
     kv.second.kind=BoundaryKind::Wall;
     kv.second.target=[](Vec3 x){return 2.+x.x+.5*x.y+.25*x.z;};
   }
   Options o;exercise("wall_SIPG",m,c,o);
   m=mesh();
   for(auto&kv:m.boundary){
     kv.second.kind=BoundaryKind::Inlet;
     kv.second.target=[](Vec3 x){return 2.+x.x+.5*x.y;};
   }
   c[0].velocity={.4,.6,.2};c[1].velocity={.6,.4,.2};
   exercise("advection_upwind",m,c,o);
   o.convectionTheta=.5;exercise("advection_theta_half",m,c,o);
   o.convectionTheta=0.;exercise("advection_central",m,c,o);
   std::puts("DG_OMEGA_GPU_PARITY_ALL_PASS");
   return 0;
 }catch(const std::exception&e){
   std::fprintf(stderr,"DG_OMEGA_GPU_PARITY_FAIL reason=%s\n",e.what());return 1;
 }
}
