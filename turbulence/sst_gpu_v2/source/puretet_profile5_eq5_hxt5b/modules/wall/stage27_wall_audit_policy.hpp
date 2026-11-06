#pragma once
// Stage 1: pure scheduling policy only. This module does NOT modify wall physics.
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <cstring>
#include <cerrno>
#include <climits>
namespace nodals { namespace stage27 {
struct WallAuditPolicy {
  enum class Mode { Full, Checkpoint, Off };
  Mode mode=Mode::Full;
  int period=25;
  bool predictor=true;

  static int positive_int_env(const char* name, int fallback){
    const char* raw=std::getenv(name);
    if(!raw||!*raw)return fallback;
    char* end=nullptr;
    errno=0;
    long val=std::strtol(raw,&end,10);
    if(errno!=0||end==raw||*end!='\0'||val<1||val>1000000)
      throw std::runtime_error(std::string("Stage27 invalid positive integer ")+name);
    return static_cast<int>(val);
  }
  static WallAuditPolicy from_env(){
    WallAuditPolicy p;
    const char* m=std::getenv("STAGE27_WALL_AUDIT_MODE");
    const std::string mode=m?m:"full";
    if(mode=="full")p.mode=Mode::Full;
    else if(mode=="checkpoint")p.mode=Mode::Checkpoint;
    else if(mode=="off")p.mode=Mode::Off;
    else throw std::runtime_error("STAGE27_WALL_AUDIT_MODE must be full or checkpoint");
    p.period=positive_int_env("STAGE27_WALL_AUDIT_EVERY",25);
    const char* pred=std::getenv("STAGE27_WALL_PREDICTOR_AUDIT");
    if(pred && std::strcmp(pred,"0")!=0 && std::strcmp(pred,"1")!=0)
      throw std::runtime_error("STAGE27_WALL_PREDICTOR_AUDIT must be 0 or 1");
    p.predictor=pred ? std::strcmp(pred,"1")==0 : p.mode==Mode::Full;
    if(p.mode==Mode::Off && p.predictor)
      throw std::runtime_error("Stage1A off mode cannot enable predictor audits");
    return p;
  }
  bool corrected_due(int it,int maxit)const{
    return mode==Mode::Full || (mode==Mode::Checkpoint && (it==1 || it==maxit || it%period==0));
  }
  bool due(int it,int maxit,const char* phase)const{
    if(mode==Mode::Full)return true;
    if(mode==Mode::Off)return false;
    if(std::strcmp(phase,"PREDICTOR")==0)return predictor && corrected_due(it,maxit);
    if(std::strcmp(phase,"CORRECTED")==0)return corrected_due(it,maxit);
    throw std::runtime_error("Stage27 unknown audit phase");
  }
  bool checkpoint()const{return mode==Mode::Checkpoint;}
  bool disabled()const{return mode==Mode::Off;}
  const char* name()const{return disabled()?"off":(checkpoint()?"checkpoint":"full");}
};
}}
