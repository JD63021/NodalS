#!/usr/bin/env python3
"""Opt-in one-compile HEX/TET SST patch for the November 14 local source snapshot.
Changes only nodals.py and hxt5c_sst_g1.cuh. No source hashes or installation gates.
"""
from pathlib import Path
import argparse

def replace_one(path, old, new):
    data=path.read_text()
    if new in data:
        print("ALREADY_PATCHED",path);return
    n=data.count(old)
    if n!=1:
        raise RuntimeError(f"{path}: expected one source anchor, found {n}: {old[:90]}")
    path.write_text(data.replace(old,new,1))
    print("PATCHED",path)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--app",type=Path,required=True)
    a=ap.parse_args()
    app=a.app.expanduser().resolve()
    py=app/"nodals.py"
    cu=app/"source/puretet_profile5_eq5_hxt5b/hxt5c_sst_g1.cuh"
    if not py.is_file() or not cu.is_file():
        raise FileNotFoundError("No GPU SST source in "+str(app))
    replace_one(py,
        "    start=get(cp,'initialization','mode','potential_flow').lower()",
        "    hybrid=flag(cp,'mesh','hybrid',False)\n    start=get(cp,'initialization','mode','potential_flow').lower()")
    replace_one(py,
        "'coupled-fgmres':'off'}",
        "'coupled-fgmres':'off','legacy-segregated':'off'}")
    replace_one(py,
        "    if mode not in modes:raise ValueError('momentum mode must be '+', '.join(modes))",
        "    if mode not in modes:raise ValueError('momentum mode must be '+', '.join(modes))\n    if mode=='legacy-segregated' and not hybrid:raise ValueError('legacy-segregated is reserved for the hybrid HEX/TET experiment')")
    replace_one(py,
        "choices=['m3-fgs','m3-sgs','m4-hybrid','coupled-fgmres']",
        "choices=['m3-fgs','m3-sgs','m4-hybrid','coupled-fgmres','legacy-segregated']")
    replace_one(py,
        "    ray_mode=get(cp,'wall_sampling','mode','ray_trace').strip().lower()",
        """    if hybrid:
        if get(cp,'mesh','format','').lower()!='hxt1':
            raise ValueError('hybrid SST requires a preflighted HXT1 .bin mesh')
        if start!='plug' or mode!='legacy-segregated':
            raise ValueError('hybrid experiment requires plug initialization and solver.u.method=legacy-segregated')
        if stress_form!='vector_laplacian' or convection_form!='advective':
            raise ValueError('hybrid momentum currently supports advective / vector_laplacian only')
        if any(flag(cp,'supg',v,False) for v in ('momentum','k','omega')):
            raise ValueError('hybrid SUPG not enabled: select supg.momentum=false')
        if samples!=1 or omega_sample!=3 or auto_enabled:
            raise ValueError('hybrid one-point run needs samples=1, omega index=3, initial_distance=legacy_scale')
        if couette_target not in ('sample_log','of_auto'):
            raise ValueError('hybrid HEX wall omega supports stage41_mode=sample_log or of_auto')
        if flag(cp,'adaptive_sampling','enabled',False):
            raise ValueError('hybrid HEX single-layer runs use fixed midpoint sampling, no adaptive tet rays')
    ray_mode=get(cp,'wall_sampling','mode','ray_trace').strip().lower()""")
    replace_one(py,
        "    env['SST_G2_VTU_DIR']=str(resolve_output_child(get(cp,'output','vtu_dir',''),output))",
        """    if hybrid:
        # Original hybrid axial-Spalding, Schur and HEX/TET Nitsche interface.
        # The current coupled BF3 tet wall and M3 workspaces are not hybrid-ready.
        env.update({
            'SST22_TRACTION_MODEL':'legacy_axial',
            'SST22D4_NITSCHE_ENABLE':'0',
            'SST22D4I_BF3':'0',
            'SST22D4I3A_SCHUR':'0',
            'NODALS_F1B_FROZEN_WALL':'0',
            'NODALS_F2_FACE_SPALDING':'0',
            'NODALS_M3_MODE':'off',
            'NODALS_AUTO_CENTROID_MODE':'0',
            'NODALS_AUTO_CENTROID_SP':'0',
            'NODALS_AUTO_CENTROID_OM':'0',
        })
    env['SST_G2_VTU_DIR']=str(resolve_output_child(get(cp,'output','vtu_dir',''),output))""")
    replace_one(py,
        "    if cp.has_section('native'):\n        for k,v in cp.items('native'):",
        """    if hybrid:
        # True one-point midpoint wall evaluation.  No PROFILE5 fit.
        for opt,value in {
            '--sst-profile-average-closure':'0',
            '--sst-profile-integral-closure':'0',
            '--sst-profile-integral-utau':'0',
            '--sst-shared-spalding-wall-state':'1',
        }.items():
            native[native.index(opt)+1]=value
        native.extend(['--wall-sample-fraction','0.5'])
    if cp.has_section('native'):
        for k,v in cp.items('native'):""")
    replace_one(cu,
        "__global__ void sst_omega_wall_hex(",
        "// Stage41 pure-log switch is defined later in the existing wall module.\nextern __device__ __constant__ int stage41_sample_log;\n__global__ void sst_omega_wall_hex(")
    replace_one(cu,
        "target=oa.omegaTarget;}",
        "target=stage41_sample_log?oa.omegaLog:oa.omegaTarget;}")
    print("HYBRID_EXPERIMENT_PATCH_APPLIED=PASS CUDA_REBUILD_NEEDED=ONCE")

if __name__=="__main__":main()
