#pragma once
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

struct H12VtuVec3 { double x=0.0,y=0.0,z=0.0; };

static inline H12VtuVec3 h12_cell_average_velocity(
    const G4SetupHost&S,std::size_t c,
    const std::array<std::vector<double>,3>&U)
{
  const auto&cp=S.cells[c];
  H12VtuVec3 q{};
  double* out[3]={&q.x,&q.y,&q.z};
  for(int d=0;d<3;++d){
    double v=0.0;
    for(int a=0;a<4;++a)v+=0.25*h8_final_velocity_coeff(S,cp,U,a,d);
    for(int a=4;a<8;++a)v+=(9.0/40.0)*h8_final_velocity_coeff(S,cp,U,a,d);
    *out[d]=v;
  }
  return q;
}

static void h12_write_vtu(
    const std::string&path,const SerialTetMesh&M,const G4SetupHost&S,
    const std::array<std::vector<double>,3>&U,const std::vector<double>&p,
    bool converged,int outerIts)
{
  if(p.size()!=M.tets.size())throw std::runtime_error("H12 VTU pressure size mismatch");

  std::vector<H12VtuVec3>u0(M.tets.size());
  std::vector<H12VtuVec3>uStream(M.points.size());
  std::vector<double>w(M.points.size(),0.0);
  for(std::size_t c=0;c<M.tets.size();++c){
    u0[c]=h12_cell_average_velocity(S,c,U);
    const double vol=S.cells[c].det/6.0;
    for(int i=0;i<4;++i){
      const std::size_t v=(std::size_t)M.tets[c][i];
      uStream[v].x+=vol*u0[c].x;
      uStream[v].y+=vol*u0[c].y;
      uStream[v].z+=vol*u0[c].z;
      w[v]+=vol;
    }
  }
  for(std::size_t v=0;v<M.points.size();++v)if(w[v]>0.0){
    uStream[v].x/=w[v];uStream[v].y/=w[v];uStream[v].z/=w[v];
  }

  std::ofstream out(path);
  if(!out)throw std::runtime_error("H12 cannot open VTU output: "+path);
  out<<std::setprecision(17)<<std::scientific;
  out<<"<?xml version=\"1.0\"?>\n"
     <<"<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
     <<"<UnstructuredGrid>\n"
     <<"<Piece NumberOfPoints=\""<<M.points.size()<<"\" NumberOfCells=\""<<M.tets.size()<<"\">\n";

  out<<"<Points><DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
  for(const auto&x:M.points)out<<x.x<<' '<<x.y<<' '<<x.z<<'\n';
  out<<"</DataArray></Points>\n";

  out<<"<Cells>\n<DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
  for(const auto&t:M.tets)out<<t[0]<<' '<<t[1]<<' '<<t[2]<<' '<<t[3]<<'\n';
  out<<"</DataArray>\n<DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
  for(std::size_t c=0;c<M.tets.size();++c)out<<4*(c+1)<<'\n';
  out<<"</DataArray>\n<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
  for(std::size_t c=0;c<M.tets.size();++c)out<<10<<'\n';
  out<<"</DataArray>\n</Cells>\n";

  out<<"<PointData Vectors=\"U0_stream\">\n"
     <<"<DataArray type=\"Float64\" Name=\"U0_stream\" NumberOfComponents=\"3\" format=\"ascii\">\n";
  for(const auto&q:uStream)out<<q.x<<' '<<q.y<<' '<<q.z<<'\n';
  out<<"</DataArray>\n</PointData>\n";

  out<<"<CellData Scalars=\"p_P0\" Vectors=\"U0\">\n"
     <<"<DataArray type=\"Float64\" Name=\"p_P0\" NumberOfComponents=\"1\" format=\"ascii\">\n";
  for(double q:p)out<<q<<'\n';
  out<<"</DataArray>\n"
     <<"<DataArray type=\"Float64\" Name=\"U0\" NumberOfComponents=\"3\" format=\"ascii\">\n";
  for(const auto&q:u0)out<<q.x<<' '<<q.y<<' '<<q.z<<'\n';
  out<<"</DataArray>\n"
     <<"<DataArray type=\"Int32\" Name=\"solve_converged\" NumberOfComponents=\"1\" format=\"ascii\">\n";
  for(std::size_t c=0;c<M.tets.size();++c)out<<(converged?1:0)<<'\n';
  out<<"</DataArray>\n"
     <<"<DataArray type=\"Int32\" Name=\"outer_iterations\" NumberOfComponents=\"1\" format=\"ascii\">\n";
  for(std::size_t c=0;c<M.tets.size();++c)out<<outerIts<<'\n';
  out<<"</DataArray>\n</CellData>\n</Piece>\n</UnstructuredGrid>\n</VTKFile>\n";
  out.close();
  if(!out)throw std::runtime_error("H12 VTU write failed: "+path);

  std::printf("NODALS_GPU_RANS_VTU path=%s points=%zu cells=%zu pointVelocity=U0_stream cellVelocity=U0 cellPressure=p_P0 pressureGauge=SOLVER_RAW outerIts=%d converged=%d status=PASS\n",
    path.c_str(),M.points.size(),M.tets.size(),outerIts,(int)converged);
}
