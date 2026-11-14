#!/usr/bin/env python3
"""Opt-in HEX-wall Stage41 trace_log wiring for mixed NodalS SST.

On the old hybrid code, STAGE41 trace_log activated the TET wall kernel only.
The native HEX kernel already has OMEGA_WALL_TRACE_LOG: symmetric Nitsche on
the physical face using the Spalding u_tau at a single off-wall sample.

This patch selects that HEX mode only for:
  equation omega, wall HEX present, SST_OMEGA_WALL_MODE=of_auto,
  STAGE41_OMEGA_MODE=trace_log.
No SUPG, momentum, pressure, interface, or pure-tet operator is changed.
It is safe to rerun, and limited to two already-compiled CUDA includes.
"""
import argparse
from pathlib import Path

def replace_once(file, old, new):
    s=file.read_text()
    if new in s:
        print("ALREADY_PATCHED",file)
        return
    n=s.count(old)
    if n!=1:
        raise RuntimeError(f"{file}: expected exactly one anchor, got {n}: {old[:120]!r}")
    file.write_text(s.replace(old,new,1))
    print("PATCHED",file)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app",required=True,type=Path)
    args=parser.parse_args()
    root=args.app.expanduser().resolve()/"source/puretet_profile5_eq5_hxt5b"
    face=root/"modules/transport/transport_face_wall.inc"
    kernel=root/"hxt5c_sst_g1.cuh"
    if not face.is_file() or not kernel.is_file():
        raise FileNotFoundError(f"Missing local November-14 SST source under {root}")

    # The existing HEX omega boundary kernel already supplies symmetric
    # consistency + adjoint terms and a log target; only its host dispatcher
    # was hard-wired to OF_AUTO's interior rank-one penalty.
    # hexWallGamma is the coefficient passed to TAU=(gamma/yd)*nu_eff.
    # y=sampleFraction*h, so gammaPass=traceGamma*sampleFraction gives exactly
    # traceGamma*nu_eff/h, matching the physical tet trace_log definition.
    old="  }else{\n    if((eq==OMEGA_EQ||"
    new="""  }else{
    const char* hybridS41=std::getenv("STAGE41_OMEGA_MODE");
    const bool hexOfAutoTraceLog =
      eq==OMEGA_EQ && !BH.wallH.empty() &&
      O.omegaWallMode==OMEGA_WALL_OF_AUTO &&
      hybridS41 && std::string(hybridS41)=="trace_log";
    const double hexTraceGamma=hexOfAutoTraceLog ?
      (std::getenv("STAGE41_TRACE_GAMMA") ?
        std::stod(std::getenv("STAGE41_TRACE_GAMMA")) : 50.0) : 0.0;
    const double hexWallGamma=hexOfAutoTraceLog ?
      hexTraceGamma*wallSampleFraction : O.wallPenaltyGamma;
    const int hexWallMode=hexOfAutoTraceLog ?
      OMEGA_WALL_TRACE_LOG : O.omegaWallMode;
    if(hexOfAutoTraceLog){
      static bool printedHexTraceLog=false;
      if(!printedHexTraceLog){
        std::printf("NODALS_HYBRID_HEX_TRACE_LOG enabled=1 omega=PURE_LOG_SPALDING_SINGLE_POINT "
          "placement=PHYSICAL_HEX_WALL_TRACE method=SYMMETRIC_NITSCHE "
          "gamma=%.12e sampleFraction=%.6f q1Wall=ON production=OFF status=PASS\\n",
          hexTraceGamma,wallSampleFraction);
        printedHexTraceLog=true;
      }
    }
    if((eq==OMEGA_EQ||"""
    replace_once(face,old,new)
    replace_once(face,
        "wallSampleFraction,O.wallPenaltyGamma,O.kFloor",
        "wallSampleFraction,hexWallGamma,O.kFloor")
    replace_once(face,
        "(int)eq,O.omegaWallMode,O.ofKappa",
        "(int)eq,hexWallMode,O.ofKappa")

    # Ensure OF_AUTO's kappa (case default 0.4) is also used in the
    # physically equivalent HEX log target, instead of the standalone
    # historical TRACE_LOG constant 0.41.
    replace_once(kernel,
        "ut/(sqrt(BETA_STAR)*KAPPA_OMEGA_WALL*yd)",
        "ut/(sqrt(BETA_STAR)*ofKappa*yd)")
    print("HYBRID_HEX_TRACE_LOG_PATCH=PASS sourceChange=CUDA_RECOMPILE_ONCE")

if __name__=="__main__":
    main()
