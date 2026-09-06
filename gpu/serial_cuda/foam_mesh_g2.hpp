#pragma once
#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace nodals_gpu {

struct Vec3d { double x=0.0,y=0.0,z=0.0; };
struct TriFace { std::array<std::int32_t,3> v{}; };
struct Patch { std::string name; std::int32_t start_face=0,n_faces=0; };
struct SerialTetMesh {
  std::vector<Vec3d> points;
  std::vector<TriFace> faces;
  std::vector<std::int32_t> owner,neighbour;
  std::vector<std::array<std::int32_t,4>> tets,opp_face;
  std::vector<Patch> patches;
  std::vector<std::int32_t> face_patch;
};

inline bool parse_integer_token(const std::string& tok,long long& value) {
  if(tok.empty()) return false; char* end=nullptr; errno=0;
  const long long v=std::strtoll(tok.c_str(),&end,10);
  if(errno || !end || *end!='\0') return false; value=v; return true;
}
class FoamTokenStream {
public:
  explicit FoamTokenStream(const std::string& p):in_(p),path_(p){if(!in_)throw std::runtime_error("cannot open "+path_);}  
  std::string next(){char c=0;while(in_.get(c)){if(std::isspace((unsigned char)c))continue;if(c=='/'){int q=in_.peek();if(q=='/'){in_.get();skip_line();continue;}if(q=='*'){in_.get();skip_block();continue;}}if(c=='('||c==')'||c=='{'||c=='}'||c==';')return std::string(1,c);if(c=='\"')return read_quoted();std::string tok(1,c);while(true){int q=in_.peek();if(q==EOF)break;char d=(char)q;if(std::isspace((unsigned char)d)||d=='('||d==')'||d=='{'||d=='}'||d==';')break;if(d=='/'){in_.get();int r=in_.peek();in_.unget();if(r=='/'||r=='*')break;}in_.get();tok.push_back(d);}return tok;}return {};}
  const std::string& path()const{return path_;}
private:
  void skip_line(){char c=0;while(in_.get(c))if(c=='\n'||c=='\r')break;}
  void skip_block(){char p=0,c=0;while(in_.get(c)){if(p=='*'&&c=='/')return;p=c;}throw std::runtime_error("unterminated block comment in "+path_);}
  std::string read_quoted(){std::string o;char c=0;while(in_.get(c)){if(c=='\\'){char d=0;if(!in_.get(d))break;o.push_back(d);continue;}if(c=='\"')return o;o.push_back(c);}throw std::runtime_error("unterminated quote in "+path_);}
  std::ifstream in_;std::string path_;
};
inline void expect_token(FoamTokenStream& ts,const char* w){auto g=ts.next();if(g!=w)throw std::runtime_error("expected token '"+std::string(w)+"' but got '"+g+"' in "+ts.path());}
inline long long require_int(FoamTokenStream& ts,const std::string& tok,const char* what){long long v=0;if(!parse_integer_token(tok,v))throw std::runtime_error("invalid "+std::string(what)+" in "+ts.path());return v;}
inline double require_double(FoamTokenStream& ts,const std::string& tok,const char* what){if(tok.empty())throw std::runtime_error("missing "+std::string(what)+" in "+ts.path());char* e=nullptr;errno=0;double v=std::strtod(tok.c_str(),&e);if(errno||!e||*e!='\0')throw std::runtime_error("invalid "+std::string(what)+" in "+ts.path());return v;}
inline int seek_list_start(FoamTokenStream& ts){for(std::string tok=ts.next();!tok.empty();tok=ts.next()){long long n=0;if(!parse_integer_token(tok,n)||n<0||n>INT_MAX)continue;if(ts.next()=="(")return (int)n;}throw std::runtime_error("cannot find OpenFOAM list count in "+ts.path());}
inline std::vector<Vec3d> read_points(const std::string& p){FoamTokenStream ts(p);int n=seek_list_start(ts);std::vector<Vec3d>a;a.reserve(n);for(int i=0;i<n;++i){expect_token(ts,"(");double x=require_double(ts,ts.next(),"x"),y=require_double(ts,ts.next(),"y"),z=require_double(ts,ts.next(),"z");expect_token(ts,")");a.push_back({x,y,z});}expect_token(ts,")");return a;}
inline std::vector<TriFace> read_faces(const std::string& p){FoamTokenStream ts(p);int n=seek_list_start(ts);std::vector<TriFace>a;a.reserve(n);for(int i=0;i<n;++i){long long k=require_int(ts,ts.next(),"face arity");if(k!=3)throw std::runtime_error("non-triangular face in "+p);expect_token(ts,"(");TriFace f;for(int j=0;j<3;++j){long long v=require_int(ts,ts.next(),"face vertex");if(v<0||v>INT_MAX)throw std::runtime_error("face vertex out of range");f.v[j]=(std::int32_t)v;}expect_token(ts,")");a.push_back(f);}expect_token(ts,")");return a;}
inline std::vector<std::int32_t> read_labels(const std::string& p){FoamTokenStream ts(p);int n=seek_list_start(ts);std::vector<std::int32_t>a;a.reserve(n);for(int i=0;i<n;++i){long long v=require_int(ts,ts.next(),"label");if(v<INT_MIN||v>INT_MAX)throw std::runtime_error("label out of range");a.push_back((std::int32_t)v);}expect_token(ts,")");return a;}
inline std::vector<Patch> read_boundary(const std::string& p){FoamTokenStream ts(p);int n=seek_list_start(ts);std::vector<Patch>a;a.reserve(n);for(int i=0;i<n;++i){std::string name=ts.next();if(name.empty()||name==")")throw std::runtime_error("missing patch name in "+p);expect_token(ts,"{");long long nf=-1,sf=-1;int depth=1;while(depth){std::string t=ts.next();if(t.empty())throw std::runtime_error("unterminated patch "+name);if(t=="{"){++depth;continue;}if(t=="}"){--depth;continue;}if(depth!=1)continue;if(t=="nFaces"){nf=require_int(ts,ts.next(),"nFaces");expect_token(ts,";");}else if(t=="startFace"){sf=require_int(ts,ts.next(),"startFace");expect_token(ts,";");}}if(nf<0||sf<0||nf>INT_MAX||sf>INT_MAX)throw std::runtime_error("invalid patch range "+name);a.push_back({name,(std::int32_t)sf,(std::int32_t)nf});}expect_token(ts,")");return a;}
inline double det3(const double J[3][3]){return J[0][0]*(J[1][1]*J[2][2]-J[1][2]*J[2][1])-J[0][1]*(J[1][0]*J[2][2]-J[1][2]*J[2][0])+J[0][2]*(J[1][0]*J[2][1]-J[1][1]*J[2][0]);}
inline void inv3(const double J[3][3],double I[3][3]){double d=det3(J);if(std::abs(d)<1e-30)throw std::runtime_error("singular tet");I[0][0]=(J[1][1]*J[2][2]-J[1][2]*J[2][1])/d;I[0][1]=(J[0][2]*J[2][1]-J[0][1]*J[2][2])/d;I[0][2]=(J[0][1]*J[1][2]-J[0][2]*J[1][1])/d;I[1][0]=(J[1][2]*J[2][0]-J[1][0]*J[2][2])/d;I[1][1]=(J[0][0]*J[2][2]-J[0][2]*J[2][0])/d;I[1][2]=(J[0][2]*J[1][0]-J[0][0]*J[1][2])/d;I[2][0]=(J[1][0]*J[2][1]-J[1][1]*J[2][0])/d;I[2][1]=(J[0][1]*J[2][0]-J[0][0]*J[2][1])/d;I[2][2]=(J[0][0]*J[1][1]-J[0][1]*J[1][0])/d;}
inline SerialTetMesh load_foam_tet_mesh(const std::string& pm){SerialTetMesh M;M.points=read_points(pm+"/points");M.faces=read_faces(pm+"/faces");M.owner=read_labels(pm+"/owner");M.neighbour=read_labels(pm+"/neighbour");M.patches=read_boundary(pm+"/boundary");if(M.owner.size()!=M.faces.size())throw std::runtime_error("owner size != faces size");M.face_patch.assign(M.faces.size(),-1);for(int p=0;p<(int)M.patches.size();++p){auto P=M.patches[p];for(int f=P.start_face;f<P.start_face+P.n_faces;++f){if(f<0||f>=(int)M.faces.size())throw std::runtime_error("patch range out of bounds");M.face_patch[f]=p;}}
  int maxc=-1;for(auto x:M.owner)maxc=std::max(maxc,(int)x);for(auto x:M.neighbour)maxc=std::max(maxc,(int)x);int nc=maxc+1;if(nc<=0)throw std::runtime_error("mesh has no cells");std::vector<std::vector<int>>cf(nc);for(int f=0;f<(int)M.faces.size();++f)cf[M.owner[f]].push_back(f);for(int f=0;f<(int)M.neighbour.size();++f)cf[M.neighbour[f]].push_back(f);M.tets.resize(nc);M.opp_face.resize(nc);for(int c=0;c<nc;++c){if(cf[c].size()!=4)throw std::runtime_error("non-tet cell "+std::to_string(c));std::set<int>vs;for(int f:cf[c])for(auto v:M.faces[f].v)vs.insert(v);if(vs.size()!=4)throw std::runtime_error("tet vertex count mismatch");std::array<std::int32_t,4>t{};int q=0;for(int v:vs)t[q++]=v;auto X0=M.points[t[0]],X1=M.points[t[1]],X2=M.points[t[2]],X3=M.points[t[3]];double J[3][3]={{X1.x-X0.x,X2.x-X0.x,X3.x-X0.x},{X1.y-X0.y,X2.y-X0.y,X3.y-X0.y},{X1.z-X0.z,X2.z-X0.z,X3.z-X0.z}};if(det3(J)<0)std::swap(t[1],t[2]);M.tets[c]=t;for(int i=0;i<4;++i){int found=-1;for(int f:cf[c]){bool has=false;for(auto v:M.faces[f].v)if(v==t[i]){has=true;break;}if(!has){if(found>=0)throw std::runtime_error("multiple opposite faces");found=f;}}if(found<0)throw std::runtime_error("missing opposite face");M.opp_face[c][i]=(std::int32_t)found;}}
  return M;}
inline int patch_index(const SerialTetMesh& M,const std::string& name){for(int i=0;i<(int)M.patches.size();++i)if(M.patches[i].name==name)return i;return -1;}
inline int auto_outlet_patch(const SerialTetMesh& M,const std::string& requested){if(requested!="auto"){int p=patch_index(M,requested);if(p<0)throw std::runtime_error("requested outlet patch not found: "+requested);return p;}for(const char* n:{"outlet","patch_1_0"}){int p=patch_index(M,n);if(p>=0)return p;}double best=-1e300;int bestp=-1;for(int p=0;p<(int)M.patches.size();++p){double z=0;long long n=0;auto P=M.patches[p];for(int f=P.start_face;f<P.start_face+P.n_faces;++f)for(auto v:M.faces[f].v){z+=M.points[v].z;++n;}if(n&&z/n>best){best=z/n;bestp=p;}}if(bestp<0)throw std::runtime_error("cannot auto-detect outlet patch");return bestp;}

} // namespace nodals_gpu
