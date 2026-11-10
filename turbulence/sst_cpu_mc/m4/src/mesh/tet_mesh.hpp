#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace nodals::cpu_mc::m2 {
struct Vec3 { double x=0,y=0,z=0; };
inline Vec3 operator+(Vec3 a,Vec3 b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
inline Vec3 operator-(Vec3 a,Vec3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
inline Vec3 operator*(double s,Vec3 a){return {s*a.x,s*a.y,s*a.z};}
inline double dot(Vec3 a,Vec3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
inline Vec3 cross(Vec3 a,Vec3 b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
struct Tet {std::array<int,4> v{},face{};double volume=0;};
struct Face {std::array<int,3> v{}; int cell0=-1,cell1=-1,patch=0;};
struct Mesh {
  std::vector<Vec3> points;
  std::vector<Tet> tets;
  std::vector<Face> faces;
  std::array<std::int64_t,4> boundary{}; // inlet=1 outlet=2 wall=3
  double totalVolume=0;
};
Mesh read_msh2(const std::string& path);
Vec3 outward_area(const Mesh&mesh,int cell,int localFace);
}
