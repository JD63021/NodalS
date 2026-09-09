#!/usr/bin/env python3
"""NodalS turbulence .case runner.

Initial supported skeleton:
  * tetrahedral P1+BF3/P0 steady SIMPLEC RANS
  * DG numerical-trace plug inlet
  * natural/pressure outlet
  * weak Spalding pipe wall
  * Nikuradse mixing-length eddy viscosity
  * native FP64 PCG pressure solve
  * custom unsmoothed+SGS or smoothed+Chebyshev AMG
  * PETSc full-GAMG PCG retained as fallback

This is intentionally narrow.  k-omega SST and broader BCs will extend this
interface without changing the current validated case semantics.
"""
from __future__ import annotations
import argparse,configparser,datetime as dt,os,shlex,subprocess,sys
from pathlib import Path

TRUE={"1","true","yes","on"}
FALSE={"0","false","no","off"}

def b(v,where):
    s=str(v).strip().lower()
    if s in TRUE:return True
    if s in FALSE:return False
    raise ValueError(f"{where}: expected boolean, got {v!r}")

def expand(v): return os.path.expandvars(os.path.expanduser(v.strip()))

def read_case(p):
    cp=configparser.ConfigParser(interpolation=None,inline_comment_prefixes=("#",";"))
    cp.optionxform=str.lower
    with p.open() as f: cp.read_file(f)
    return cp

def g(cp,s,k,d=None):
    return cp.get(s,k,fallback=d).strip() if cp.has_section(s) or d is not None else d

def req(cp,s,k):
    if not cp.has_option(s,k): raise ValueError(f"missing required [{s}] {k}")
    return cp.get(s,k).strip()

def opt(a,k,v): a += [f"-{k}",str(v)]

def build_options(cp,vtu):
    a=[]
    mesh=expand(req(cp,"solver","mesh"))
    opt(a,"mesh",mesh)
    opt(a,"problem","pipe")
    opt(a,"re",g(cp,"solver","re","13691.7402481"))
    opt(a,"nu",g(cp,"solver","nu","1.460734693878e-5"))
    conv=g(cp,"solver","convection","central").lower()
    if conv!="central": raise ValueError("turbulence skeleton currently requires [solver] convection=central")
    opt(a,"convection","central")

    # Pipe/DG boundary skeleton.
    wall=g(cp,"boundary","wall_patch","wall")
    inlet=g(cp,"boundary","inlet_patch","inlet")
    outlet=g(cp,"boundary","outlet_patch","outlet")
    bulk=g(cp,"boundary","bulk_velocity","50.0")
    inlet_type=g(cp,"boundary","inlet_type","dg_numerical_trace").lower()
    if inlet_type!="dg_numerical_trace":
        raise ValueError("turbulence skeleton currently requires [boundary] inlet_type=dg_numerical_trace")
    speed=abs(float(g(cp,"boundary","speed",bulk)))
    opt(a,"pipe_bulk_velocity",bulk)
    opt(a,"pipe_wall_patch",wall)
    opt(a,"pipe_inlet_patch",inlet)
    opt(a,"pipe_outlet_patch",outlet)
    opt(a,"inlet_bc","dg_numerical_trace")
    opt(a,"inlet_normal_mode","average_patch_normal")
    opt(a,"inlet_normal_speed",f"{-speed:.17g}")
    opt(a,"initial_pipe_velocity","plug")

    # Turbulence model skeleton.
    model=g(cp,"turbulence","model","nikuradse_mixing_length").lower()
    if model not in {"nikuradse_mixing_length","mixing_length","nikuradse_pipe"}:
        raise ValueError("[turbulence] model currently supports only nikuradse_mixing_length")
    opt(a,"mixing_length",1)
    opt(a,"mixlen_model","nikuradse_pipe")
    opt(a,"mixlen_scale",g(cp,"turbulence","scale","1.0"))
    opt(a,"mixlen_strain_mode",g(cp,"turbulence","strain_mode","raw"))
    opt(a,"mixlen_audit",1 if b(g(cp,"turbulence","audit","false"),"[turbulence] audit") else 0)

    # Weak wall.
    if not b(g(cp,"wall_function","enabled","true"),"[wall_function] enabled"):
        raise ValueError("current turbulence skeleton requires weak wall_function enabled=true")
    opt(a,"weak_wall_function",1)
    opt(a,"wall_law",g(cp,"wall_function","law","spalding"))
    opt(a,"wall_kappa",g(cp,"wall_function","kappa","0.4"))
    opt(a,"wall_B",g(cp,"wall_function","B","5.5"))
    opt(a,"wall_distance_factor",g(cp,"wall_function","distance_factor","0.25"))
    opt(a,"wall_beta_scale",g(cp,"wall_function","beta_scale","1.0"))
    opt(a,"wall_linearization",g(cp,"wall_function","linearization","consistent_tangent"))
    opt(a,"wall_tangent_blend",g(cp,"wall_function","tangent_blend","0.5"))

    opt(a,"supg",0)

    # SIMPLEC.
    opt(a,"simple_variant",g(cp,"simple","variant","simplec"))
    opt(a,"simplec_blend",g(cp,"simple","simplec_blend","1.0"))
    opt(a,"simplec_floor_fraction",g(cp,"simple","simplec_floor_fraction","1e-6"))
    opt(a,"simplec_fallback",g(cp,"simple","simplec_fallback","diag"))
    opt(a,"alpha_u",g(cp,"simple","alpha_u","0.70"))
    opt(a,"alpha_p",g(cp,"simple","alpha_p","0.30"))
    opt(a,"simple_rtol",g(cp,"simple","rtol","1e-3"))
    opt(a,"simple_max_it",g(cp,"simple","max_iterations","10000"))
    opt(a,"rau_mode",g(cp,"simple","rau_mode","diag"))
    opt(a,"rau_scale",g(cp,"simple","rau_scale","2.0"))

    # Momentum.
    opt(a,"u_relax_mode",g(cp,"momentum","relax_mode","row_l1"))
    opt(a,"u_rtol",g(cp,"momentum","rtol","1e-8"))
    opt(a,"u_atol",g(cp,"momentum","atol","0"))
    opt(a,"u_rel_drop",g(cp,"momentum","relative_drop","0.5"))
    opt(a,"u_check_every",g(cp,"momentum","check_every","1"))
    opt(a,"u_local_sweeps",g(cp,"momentum","local_sweeps","1"))
    opt(a,"u_sor_omega",g(cp,"momentum","sor_omega","1.0"))

    # Pressure.
    mode=g(cp,"pressure","mode","pcg_unsmoothed_sgs").lower()
    if mode=="pcg_unsmoothed_sgs":
        backend,smoother="custom_agg_unsmoothed","sgs"
    elif mode in {"pcg_smoothed_cheb","pcg_smoothed"}:
        backend,smoother="custom_agg_smoothed","chebyshev"
    elif mode in {"full_pcg_gamg","pcg_full_gamg"}:
        backend,smoother="petsc_full_gamg","chebyshev"
    else:
        raise ValueError("[pressure] mode must be pcg_unsmoothed_sgs, pcg_smoothed_cheb, or full_pcg_gamg")

    opt(a,"pressure_solve_mode","custom_pcg")
    opt(a,"pressure_pc_backend",backend)

    # Custom AMG controls are harmless for PETSc fallback and keep one case schema.
    opt(a,"custom_amg_smoother",smoother)
    opt(a,"custom_amg_target_aggregate",g(cp,"pressure_amg","target_aggregate","16"))
    opt(a,"custom_amg_min_aggregate",g(cp,"pressure_amg","min_aggregate","6"))
    opt(a,"custom_amg_soft_max_aggregate",g(cp,"pressure_amg","soft_max_aggregate","18"))
    opt(a,"custom_amg_coarse_target",g(cp,"pressure_amg","coarse_target","1000"))
    opt(a,"custom_amg_cheb_degree",g(cp,"pressure_amg","cheb_degree","2"))
    opt(a,"custom_amg_power_its",g(cp,"pressure_amg","power_iterations","16"))
    opt(a,"custom_amg_lambda_safety",g(cp,"pressure_amg","lambda_safety","1.50"))
    opt(a,"custom_amg_lambda_low_fraction",g(cp,"pressure_amg","lambda_low_fraction","0.05"))
    opt(a,"custom_amg_interp_max_nnz",g(cp,"pressure_amg","interpolation_max_nnz","8"))
    opt(a,"custom_amg_sa_damping",g(cp,"pressure_amg","sa_damping","1.3333333333333333"))

    opt(a,"gate1_compare_pcg",0)
    opt(a,"p_operator","factored")
    opt(a,"p_pmat","full")
    opt(a,"p_preconditioner_refresh",g(cp,"pressure","refresh","1"))
    opt(a,"p_ksp_type","cg")
    opt(a,"p_ksp_rtol",g(cp,"pressure","rtol","0.9"))
    opt(a,"p_ksp_atol",g(cp,"pressure","atol","1e-12"))
    opt(a,"p_ksp_divtol",g(cp,"pressure","divtol","1e8"))
    opt(a,"p_ksp_max_it",g(cp,"pressure","max_iterations","1500"))

    # PETSc full-GAMG oracle/fallback config retained exactly.
    opt(a,"p_pc_type","gamg")
    opt(a,"p_mg_levels_ksp_type","chebyshev")
    opt(a,"p_mg_levels_pc_type","jacobi")
    opt(a,"p_mg_levels_ksp_max_it",g(cp,"pressure","petsc_level_cheb_iterations","12"))

    opt(a,"write_vtu",1 if b(g(cp,"output","write_vtu","true"),"[output] write_vtu") else 0)
    opt(a,"vtu_output",str(vtu))
    opt(a,"vtu_velocity_mode",g(cp,"output","velocity_mode","both"))
    opt(a,"options_left","")
    # Remove empty value for PETSc flag.
    if a[-2:] == ["-options_left",""]: a.pop()
    return a

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("case",type=Path)
    ap.add_argument("--exe",type=Path)
    ap.add_argument("--np",type=int)
    ap.add_argument("--dry-run",action="store_true")
    ns,petsc=ap.parse_known_args()
    if petsc and petsc[0]=="--":
        petsc=petsc[1:]

    cp=read_case(ns.case)
    repo=Path(__file__).resolve().parents[1]
    exe=ns.exe or repo/"turbulence/tet_p1bf3_rans/nodals_turbulence_cpu_fp64"
    np=ns.np or int(g(cp,"run","np","16"))

    run_root=Path(expand(g(cp,"run","output_root","$HOME/Downloads/NodalS_TURBULENCE_RUNS")))
    name=g(cp,"run","name",ns.case.stem)
    stamp=dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir=run_root/f"{name}_{stamp}"
    outdir.mkdir(parents=True,exist_ok=True)
    log=outdir/f"{name}.log"
    vtu=outdir/f"{name}.vtu"

    args=build_options(cp,vtu)
    # Environment-only wall options used by the validated branch.
    env=os.environ.copy()
    env["PETSC_OPTIONS"]=(env.get("PETSC_OPTIONS","")+
        " -wall_sample_mode "+g(cp,"wall_function","sample_mode","legacy_face_trace")+
        " -wall_molecular_consistency "+("1" if b(g(cp,"wall_function","molecular_consistency","false"),"[wall_function] molecular_consistency") else "0")+
        " -momentum_budget 0").strip()

    cmd=["mpirun","-np",str(np),"--map-by","core","--bind-to","core",str(exe)]+args+petsc
    print("NODALS_TURB_CASE",ns.case)
    print("NODALS_TURB_EXE",exe)
    print("NODALS_TURB_LOG",log)
    print("NODALS_TURB_VTU",vtu)
    print("NODALS_TURB_CMD",shlex.join(cmd))
    if ns.dry_run:return 0
    if not exe.exists(): raise SystemExit(f"executable not found: {exe}")

    with log.open("w") as f:
        p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                           text=True,bufsize=1,env=env)
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line); f.write(line)
        rc=p.wait()
    print(f"NODALS_TURB_CASE_RESULT rc={rc} log={log} vtu={vtu}")
    return rc

if __name__=="__main__":
    raise SystemExit(main())
