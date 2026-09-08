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
#include "h8_supg_dispatch.inc"
