#include "flow_vtu.hpp"
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include <cmath>
#include <array>
#include <utility>
namespace nodals::cpu_mc::m4 {
void write_solution_vtu(const m2::Mesh&m,const SSTTransport&sst,
 const std::vector<double>&u,const std::vector<double>&p,const std::string&path){
 std::ofstream f(path);if(!f)throw std::runtime_error("unable to open VTU: "+path);
 f<<std::setprecision(16);
 const int nv=(int)m.points.size(),nc=(int)m.tets.size();
 if((int)p.size()!=nc)throw std::runtime_error("invalid pressure for VTU");
 f<<"<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
 f<<"<UnstructuredGrid><Piece NumberOfPoints=\""<<nv<<"\" NumberOfCells=\""<<nc<<"\">\n";
 f<<"<Points><DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
 for(const auto&x:m.points)f<<x.x<<' '<<x.y<<' '<<x.z<<'\n';
 f<<"</DataArray></Points>\n<Cells><DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
 for(const auto&t:m.tets)f<<t.v[0]<<' '<<t.v[1]<<' '<<t.v[2]<<' '<<t.v[3]<<'\n';
 f<<"</DataArray><DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
 for(int c=1;c<=nc;++c)f<<4*c<<'\n';
 f<<"</DataArray><DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
 for(int c=0;c<nc;++c)f<<"10\n";
 f<<"</DataArray></Cells>\n<PointData Vectors=\"velocity\">\n";
 f<<"<DataArray type=\"Float64\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n";
 for(int i=0;i<nv;++i)f<<u[3*i]<<' '<<u[3*i+1]<<' '<<u[3*i+2]<<'\n';
 f<<"</DataArray>\n";
 for(const auto&field:std::array<std::pair<const char*,const std::vector<double>*>,2>{{
  {"k",&sst.state().k},{"omega",&sst.state().omega}}}){
  f<<"<DataArray type=\"Float64\" Name=\""<<field.first<<"\" format=\"ascii\">\n";
  for(double q:*field.second)f<<q<<'\n';
  f<<"</DataArray>\n";
 }
 f<<"</PointData>\n<CellData>\n<DataArray type=\"Float64\" Name=\"pressure\" format=\"ascii\">\n";
 for(double q:p)f<<q<<'\n';
 f<<"</DataArray>\n<DataArray type=\"Float64\" Name=\"nut\" format=\"ascii\">\n";
 for(int c=0;c<nc;++c)f<<sst.cell_nut(c,u)<<'\n';
 f<<"</DataArray></CellData></Piece></UnstructuredGrid></VTKFile>\n";
 if(!f)throw std::runtime_error("VTU write failed: "+path);
}
}
