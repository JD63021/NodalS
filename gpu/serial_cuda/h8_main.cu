// H8 SUPG extension.
//
// The validated pre-SUPG H8 translation unit is retained verbatim as
// h8_main_base.cu. With SUPG disabled we strip the SUPG-only CLI flags and
// delegate directly to that historical entry point. With SUPG enabled we use
// the CPU-matching implicit 64-point stabilization path below.
#define main h8_base_main
#include "h8_main_base.cu"
#undef main

#include "h8_supg_support.inc"
#include "h8_hp_diagnostics.inc"
#include "h8_supg_main_cli.inc"
#include "h8_supg_main_setup.inc"
#include "h8_supg_main_loop.inc"
static bool h8_rans_cli_enabled(int argc,char**argv)
{
  for(int i=1;i<argc;++i){
    const std::string a=argv[i];
    if((a=="--mixing-length"||a=="--dg-inlet"||a=="--weak-wall")&&i+1<argc){
      if(std::atoi(argv[i+1])!=0)return true;
      ++i;
    }
  }
  return false;
}

int main(int argc,char**argv)
{
  const H8SupgCli q=h8_supg_scan(argc,argv);
  if(!q.enabled)return h8_base_without_supg_cli(argc,argv);
  // RANS SUPG must remain in the RANS operator path so mixing length,
  // DG inlet and weak Spalding wall are retained. Laminar SUPG continues
  // to use the historical independently validated SUPG entry point.
  if(h8_rans_cli_enabled(argc,argv))return h8_base_main(argc,argv);
  return h8_supg_main(argc,argv);
}
