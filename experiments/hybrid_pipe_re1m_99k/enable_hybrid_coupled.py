#!/usr/bin/env python3
"""Opt-in native hybrid M3-FGS / coupled FGMRES port from November-14 SST.

Run AFTER enable_hybrid_local.py, fix_hybrid_startup.py and
enable_hex_trace_log.py on the isolated experimental application.

- Keeps the original HEX Q1+BF2/TET P1+BF3 momentum, DG/Nitsche interface,
  normal wall clamps, wall Spalding, Schur, turbulence and pressure unchanged.
- Adds solver selection independent of tetrahedral-only Stage22D4.
- M3 uses exactly two (editable) native 3x3 coloured FGS passes.
- Coupled FGMRES is the reference native 3Nv matrix-free solver.
- The current vector-Laplacian/legacy-wall operator is component-diagonal.
  This first port does not claim to restore the pure-tet vector-wall blocks.
"""
import argparse
from pathlib import Path

def change(path,old,new,label):
    s=path.read_text()
    if new in s:
        print("ALREADY",label,path)
        return
    n=s.count(old)
    if n!=1:
        raise RuntimeError(f"{label}: expected one anchor, found {n} in {path}: {old[:100]!r}")
    path.write_text(s.replace(old,new,1))
    print("PATCHED",label,path)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--app",required=True,type=Path)
    a=p.parse_args()
    app=a.app.expanduser().resolve()
    root=app/"source/puretet_profile5_eq5_hxt5b"
    native=root/"modules/app/sst_application.inc"
    predictor=root/"modules/coupling/simple_pressure_predictor.inc"
    wall=root/"modules/wall/stage22d4d_coupled_matrixfree.inc"
    g2=root/"hxt5d_sst_g2.cuh"
    py=app/"nodals.py"
    src=Path(__file__).resolve().parent/"src/hybrid_coupled_momentum.inc"
    include=root/"modules/coupling/hybrid_coupled_momentum.inc"
    if not src.is_file():raise FileNotFoundError(src)
    if not all(z.is_file() for z in (native,predictor,wall,g2,py)):
        raise FileNotFoundError("Incomplete experimental source under "+str(app))
    if not include.is_file():
        include.write_text(src.read_text())
        print("INSTALLED",include)
    elif include.read_text()!=src.read_text():
        raise RuntimeError("Refuse to overwrite manually edited hybrid coupled source: "+str(include))

    change(g2,
        '#include "modules/wall/stage22d4d_coupled_matrixfree.inc"',
        '#include "modules/wall/stage22d4d_coupled_matrixfree.inc"\n'
        '#include "modules/coupling/hybrid_coupled_momentum.inc"',
        "hybrid_solver_include")

    # S22D4DOperator is reusable on mixed meshes when it has no tetrahedral
    # wall faces. Never launch a CUDA kernel with a zero-block grid.
    change(wall,'    if(f1bFaceCoupling)\n      f1b_offblock_cached<<<nfaces,32>>>',
                '    if(nfaces>0 && f1bFaceCoupling)\n      f1b_offblock_cached<<<nfaces,32>>>',
                "guard_zero_face_cached")
    change(wall,'    else\n      s22d4d_offblock_action<<<nfaces,64>>>',
                '    else if(nfaces>0)\n      s22d4d_offblock_action<<<nfaces,64>>>',
                "guard_zero_face_uncached")

    change(native,
        '    const bool m3Active=m3Mode!="off";',
        '''    const bool m3Active=m3Mode!="off";
    const std::string hybridMomentumMode=std::getenv("NODALS_HYBRID_COUPLED_MODE")?
      std::getenv("NODALS_HYBRID_COUPLED_MODE"):"off";
    if(hybridMomentumMode!="off" && hybridMomentumMode!="m3-fgs" &&
       hybridMomentumMode!="coupled-fgmres")
      throw std::runtime_error("NODALS_HYBRID_COUPLED_MODE must be off|m3-fgs|coupled-fgmres");
    const bool hybridCoupled=hybridMomentumMode!="off";
    const bool hybridM3=hybridMomentumMode=="m3-fgs";
    if(hybridCoupled &&
       (!(M.h.nhex>0 && M.h.ntet>0 && M.h.ninterface>0) ||
        m3Active || stage22d4Live || stage22d4iEnable || stage22Vector ||
        momentumFullSupg || momentumEmac || implicitDevStress ||
        momentumConvectionForm!="advective" || stressForm!="vector_laplacian" ||
        !BH.wallT.empty() || !sstG2 || sstG2Start!="plug" ||
        simpleCoupling!="corrected"))
      throw std::runtime_error("Hybrid coupled V1 requires native mixed mesh, original vector-Laplacian/advective wall, no tetra BF3 wall projection");
    std::printf("NODALS_HYBRID_COUPLED_CONFIG mode=%s "
      "sweeps=%d block=3NV solverOnly=YES "
      "crossComponentPhysics=NONE_LEGACY_AXIAL_HEX_WALL "
      "tetraWallProjection=NOT_REQUIRED interface=ORIGINAL_NITSCHE status=PASS\\n",
      hybridMomentumMode.c_str(),
      s05b1Integer("NODALS_M3_SWEEPS",2,1,60));''',
        "hybrid_native_mode")

    change(native,
        'Dev<T>d_s22d4dX(std::vector<T>(stage22d4Live ? (std::size_t)3*nv : 1,0));',
        'Dev<T>d_s22d4dX(std::vector<T>((stage22d4Live||hybridCoupled) ? (std::size_t)3*nv : 1,0));',
        "hybrid_allocate_x")
    change(native,
        'Dev<T>d_s22d4dB(std::vector<T>(stage22d4Live ? (std::size_t)3*nv : 1,0));',
        'Dev<T>d_s22d4dB(std::vector<T>((stage22d4Live||hybridCoupled) ? (std::size_t)3*nv : 1,0));',
        "hybrid_allocate_b")
    change(native,
        'hxt4a::Reducer<T> stage22d4dRed(stage22d4Live ? 3*nv : 1);',
        'hxt4a::Reducer<T> stage22d4dRed((stage22d4Live||hybridCoupled) ? 3*nv : 1);',
        "hybrid_allocate_reducer")

    change(native,
        '    if(m3Active){\n      const std::size_t entries=9ull*A.nnz, bytes=entries*sizeof(T);',
        '    if(m3Active||hybridM3){\n      const std::size_t entries=9ull*A.nnz, bytes=entries*sizeof(T);',
        "hybrid_allocate_m3")

    change(native,
        '      d_m3K.n=entries;d_m3Temp.n=entries;\n'
        '      HXT1_CUDA(cudaMalloc((void**)&d_m3K.p,bytes));\n'
        '      HXT1_CUDA(cudaMalloc((void**)&d_m3Temp.p,bytes));',
        '      d_m3K.n=entries;d_m3Temp.n=hybridM3?(std::size_t)3*nv:entries;\n'
        '      HXT1_CUDA(cudaMalloc((void**)&d_m3K.p,bytes));\n'
        '      HXT1_CUDA(cudaMalloc((void**)&d_m3Temp.p,d_m3Temp.n*sizeof(T)));',
        "hybrid_m3_memory")

    change(predictor,
        '        double s22d4dR0=0.,s22d4dR1=0.;',
        '''        if(hybridCoupled){
          hxt5d::hybrid_coupled_pack<<<(nv+255)/256,256>>>(
            nv,d_u0.p,d_u1.p,d_u2.p,
            d_rhs0.p,d_rhs1.p,d_rhs2.p,
            d_clampXY.p,d_fixed.p,d_s22d4dX.p,d_s22d4dB.p);
          // The existing frozen HEX/TET A_x=A_y and A_z include the
          // symmetric interface terms, wall z-Spalding and relaxation.
          // There are NO tetrahedral *wall* faces on this hybrid mesh.
          hxt5d::S22D4DOperator hOp{
            A,VC,A.val.p,A.val.p,d_valZ.p,
            d_clampXY.p,d_fixed.p,
            sstG1.tetWallPlans,sstG1.tetWallQuads,
            d_tv.p,d_tg.p,d_s22d4dCoeff.p,
            0,0.0,d_s22d4iScratch.p,false,false,nullptr};
          hxt5d::S22D4DResult hResult{};
          if(hybridM3){
            hResult=hxt5d::hybrid_coupled_m3_fgs(hOp,
              d_s22d4dX.p,d_s22d4dB.p,d_m3K.p,d_m3Temp.p,
              d_m3BubbleFace.p,d_m3Bad.p,m3Sweeps,m3Omega,git,
              stage22d4dRed,d_m3Check.p,d_m3CheckDiff.p);
          }else{
            hResult=hxt5d::s22d4d_block_fgmres(hOp,
              d_s22d4dX.p,d_s22d4dB.p,
              stage22d4dRestart,stage22d4dMaxIts,momRtol,momAtol,
              momFgsSweeps,momOmega,stage22d4dRed,git);
          }
          hxt5d::s22d4d_unpack<<<(nv+255)/256,256>>>(
            nv,d_s22d4dX.p,d_u0.p,d_u1.p,d_u2.p);
          HXT1_CUDA(cudaGetLastError());
          if(!hResult.ok && momCapPolicy=="fail")
            throw std::runtime_error("Hybrid coupled momentum residual did not improve");
          if(git==1||git%sstG2PrintEvery==0)
            std::printf("NODALS_HYBRID_COUPLED_MOMENTUM it=%d "
              "solver=%s linearIts=%d rel=%.12e accepted=%d "
              "pressureSchur=ORIGINAL_RAU status=%s\\n",
              git,hybridMomentumMode.c_str(),hResult.its,hResult.rel,
              (int)hResult.ok,hResult.ok?"PASS":"WARN");
        }
        double s22d4dR0=0.,s22d4dR1=0.;''',
        "hybrid_predictor_solve")
    change(predictor,
        '        if(stage22d4dMode=="segregated"){',
        '        if(stage22d4dMode=="segregated" && !hybridCoupled){',
        "skip_legacy_segregated_when_hybrid")

    # The previous hybrid patch requires legacy-segregated. Permit only
    # the two new hybrid solver methods in addition; pure-tet is unchanged.
    change(py,
        "if start!='plug' or mode!='legacy-segregated':",
        "if start!='plug' or mode not in ('legacy-segregated','m3-fgs','coupled-fgmres'):",
        "frontend_allow_hybrid_coupled")
    change(py,
        "            'SST22_TRACTION_MODEL':'legacy_axial',",
        """            'NODALS_HYBRID_COUPLED_MODE':
                ('m3-fgs' if mode=='m3-fgs' else
                 'coupled-fgmres' if mode=='coupled-fgmres' else 'off'),
            # Native pure-tetra M3 must remain disabled: hybrid M3 has
            # its own exact unprojected operator and 3x3 CSR sweeps.
            'NODALS_M3_MODE':'off',
            'SST22_TRACTION_MODEL':'legacy_axial',""",
        "frontend_route_hybrid_modes")
    print("HYBRID_COUPLED_PORT=APPLIED_NATIVE_SOURCE_ONLY")
    print("NEXT_ACTION=INCREMENTAL_CUDA_BUILD_ONCE")

if __name__=="__main__":
    main()
