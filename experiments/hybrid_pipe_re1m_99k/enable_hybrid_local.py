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
    # Startup patch adds an env.pop() between the hybrid block and this
    # original anchor; do not duplicate the entire hybrid configuration
    # when RUN.sh is called repeatedly after a successful first patch.
    if "env['SST_G2_VTU_DIR']" in old and "env.pop('SST22D4_NITSCHE_ENABLE', None)" in data:
        print("ALREADY_PATCHED_HYBRID_ENV",path);return
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
            'SST22D4D_MOM_MODE':'segregated',
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
    # CUDA has one *definition* of the global constant.  The old experiment
    # mistakenly inserted an extern before the later in-header definition;
    # NVCC treats that combination as a duplicate.  Repair existing copies too.
    wall = app/"source/puretet_profile5_eq5_hxt5b/modules/wall/of_auto_point3_kernel.inc"
    old_decl = chr(10).join((
        "// Stage41 pure-log switch is defined later in the existing wall module.",
        "extern __device__ __constant__ int stage41_sample_log;",
        ""
    ))
    new_def = chr(10).join((
        "// Shared HEX/TET Stage41 switch, exactly one CUDA definition.",
        "__device__ __constant__ int stage41_sample_log=0;",
        ""
    ))
    cs = cu.read_text()
    if old_decl in cs:
        cu.write_text(cs.replace(old_decl, new_def, 1))
        print("REPAIRED_EARLIER_CUDA_DECLARATION", cu)
    elif new_def not in cs:
        replace_one(cu,"__global__ void sst_omega_wall_hex(",
                    new_def+"__global__ void sst_omega_wall_hex(")
    else:
        print("CUDA_SYMBOL_ALREADY_CORRECT",cu)
    replace_one(wall,
        "__device__ __constant__ int stage41_sample_log=0;",
        "// stage41_sample_log defined once in hxt5c_sst_g1.cuh for HEX and TET.")
    replace_one(cu,
        "target=oa.omegaTarget;}",
        "target=stage41_sample_log?oa.omegaLog:oa.omegaTarget;}")
    # HXT1 binary meshes do not pass through mesh_prepare's .msh content
    # fingerprint. Stage31 nevertheless requires a SHA256 even on a fresh run.
    # Use the identical identity schema as .msh inputs, with the HXT1 bytes
    # as the mesh content. Restart safety remains enabled.
    replace_one(py,
        "    if args.restart and not mesh_sha and not get(cp,'restart','input_fingerprint'):",
        """    if hybrid and not mesh_sha:
        profile_path=get(cp,'inlet.main','profile_csv')
        profile_hash=sha256(absolute(profile_path,args.case.parent)) if profile_path else None
        fixed={'mesh_sha256':sha256(mesh),'profile_sha256':profile_hash,'re':reynolds,
               'nu':nu,'diameter':diameter,'density':1.0}
        env['STAGE31_INPUT_FINGERPRINT']=hashlib.sha256(
            json.dumps(fixed,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if args.restart and not mesh_sha and not get(cp,'restart','input_fingerprint'):""")
    print("HYBRID_EXPERIMENT_PATCH_APPLIED=PASS; STAGE31_BINARY_MESH_FINGERPRINT=ON")

if __name__=="__main__":main()
