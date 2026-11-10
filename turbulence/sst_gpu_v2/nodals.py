#!/usr/bin/env python3
"""NodalS SST: ordinary case-driven CUDA executor (NO stage install or binary hash gates).

The frozen numerical CUDA source is NOT altered by this Python front end.
Run `python3 nodals.py run --help` for controls. Requires numpy to convert MSH2.
"""
from __future__ import annotations
import argparse
import configparser
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
BOOL_TRUE={'1','yes','true','on'}
BOOL_FALSE={'0','no','false','off'}

def boolean(v):
    s=str(v).strip().lower()
    if s in BOOL_TRUE:return True
    if s in BOOL_FALSE:return False
    raise ValueError('Expected boolean true/false; got '+repr(v))

def absolute(p, parent=ROOT):
    q=Path(os.path.expandvars(os.path.expanduser(str(p))))
    return (q if q.is_absolute() else parent/q).resolve()

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):h.update(b)
    return h.hexdigest()

def read_case(path, overrides):
    cp=configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',';'), strict=True)
    if not cp.read(path):raise FileNotFoundError('Case file not found: '+str(path))
    for s in ('mesh','physics','inlet.main','simple','sst','solver.u'):
        if not cp.has_section(s):raise ValueError('Missing required case section ['+s+']')
    for item in overrides:
        try:k,v=item.split('=',1);s,k=k.rsplit('.',1)
        except ValueError:raise ValueError('--set expects SECTION.KEY=VALUE: '+item)
        if not cp.has_section(s):cp.add_section(s)
        cp.set(s,k,v)
    return cp

def get(cp,s,k,default=None):
    return cp.get(s,k,fallback=default)

def number(cp,s,k,default=None):
    v=get(cp,s,k,default)
    if v is None:raise ValueError('Required numeric key missing: '+s+'.'+k)
    n=float(v)
    if not math.isfinite(n):raise ValueError('Nonfinite '+s+'.'+k)
    return n

def integer(cp,s,k,default):
    n=int(get(cp,s,k,default))
    if n<0:raise ValueError('Negative value for '+s+'.'+k)
    return n

def flag(cp,s,k,default=False):
    return boolean(get(cp,s,k,str(default)))

def vec3(raw):
    vals=[float(x.strip()) for x in str(raw).strip().strip('()[]').split(',')]
    if len(vals)!=3 or not all(math.isfinite(v) for v in vals):
        raise ValueError('inlet.main.velocity must be a finite 3-component vector')
    return vals

def fmt(x):return format(float(x),'.17g')

def mesh_prepare(path, cache, prepare):
    if path.suffix.lower()=='.msh':
        # Content address prevents stale cache reuse without introducing binary hash gates.
        fingerprint=sha256(path)
        folder=cache/fingerprint[:20]
        converted=folder/'HXT1_mesh.bin'
        if prepare and not converted.is_file():
            print('NODALS_MESH_CONVERT='+str(path), flush=True)
            from tools.msh2_to_hxt1 import convert
            convert(path, folder, ROOT/'tools'/'prepare_hxt1.py')
        if prepare and not converted.is_file():
            raise RuntimeError('HXT1 conversion did not produce '+str(converted))
        return converted, fingerprint
    if path.suffix.lower() in ('.bin','.hxt1bin','.hxt1'):
        return path,None
    raise ValueError('Unsupported mesh format: expected .msh (Gmsh ASCII v2.2) or HXT1 binary')

def resolve_output_child(raw, output):
    # A blank directory means this run's output directory; relative values
    # name subdirectories of that run (not the source/working directory).
    if not raw or raw=='.':return output
    return absolute(raw,output)

def compact_line(line):
    # Keep full native output in terminal.log for troubleshooting, but show a
    # clean operator-facing console at the selected residual cadence.
    return line.startswith((
        'NODALS_SST_RESIDUALS ',
        'NODALS_SST_PLOT_METRICS ',
        'NODALS_SST_G2_VTU_CHECKPOINT ',
        'STAGE31_RESTART_SAVE=',
        'STAGE31_RESTART_LOAD=',
        'NODALS_SST_G2_FIXED10_TIMING ',
        'NODALS_SOFTWARE_COMPLETION=',
        'HXT5B_ERROR ',
        'NODALS_SOFTWARE_ERROR ',
        'CUDA_ERROR',
        'NODALS_SST_G2_FINAL ',
        'NODALS_METRIC_GATE ',
    ))

def make_invocation(cp, args, mesh, source_mesh, mesh_sha, output):
    if get(cp,'physics','density','1') and abs(number(cp,'physics','density',1)-1)>1e-12:
        raise ValueError('Current native solver uses kinematic pressure; physics.density must be 1')
    diameter=number(cp,'mesh','diameter')
    if diameter<=0:raise ValueError('mesh.diameter must be positive')
    reynolds=number(cp,'physics','reynolds')
    if reynolds<=0:raise ValueError('physics.reynolds must be positive')
    inlet_mode=get(cp,'inlet.main','mode','velocity').strip().lower()
    if inlet_mode=='velocity':
        vel=vec3(get(cp,'inlet.main','velocity'))
        bulk=math.sqrt(sum(x*x for x in vel))
    elif inlet_mode=='speed':
        bulk=number(cp,'inlet.main','speed')
        vel=vec3(get(cp,'inlet.main','direction','(1,0,0)'))
        n=math.sqrt(sum(x*x for x in vel))
        if not n:raise ValueError('Cannot use zero inlet direction')
        vel=[bulk*x/n for x in vel]
    else:raise ValueError('Supported inlet modes: velocity, speed (with direction)')
    if bulk<=0:raise ValueError('Inlet velocity must be nonzero')
    nu_raw=get(cp,'physics','kinematic_viscosity')
    nu=bulk*diameter/reynolds
    if nu_raw is not None:
        nu_value=number(cp,'physics','kinematic_viscosity')
        if nu_value<=0 or abs(nu_value/nu-1)>0.001:
            raise ValueError('inlet speed, diameter, Reynolds and kinematic_viscosity disagree (>0.1%)')
        nu=nu_value

    start=get(cp,'initialization','mode','potential_flow').lower()
    if start not in ('potential_flow','plug'):
        raise ValueError('initialization.mode must be potential_flow or plug')
    state=get(cp,'sst','state_space','q1').lower()
    if state not in ('q1','bernstein_q2','q1bf2_unlimited'):
        raise ValueError('sst.state_space must be q1, bernstein_q2 or q1bf2_unlimited')
    if state!='q1':
        print('WARNING: non-reference scalar discretization selected; numerical parity not established.',file=sys.stderr)
    mode=(args.momentum_solver or get(cp,'solver.u','method','m3-fgs')).lower()
    modes={'m3-fgs':'fgs','m3-sgs':'sgs','m4-hybrid':'m4','coupled-fgmres':'off'}
    if mode not in modes:raise ValueError('momentum mode must be '+', '.join(modes))
    # NODALS_CONSERVATIVE_PICARD_V1: independent momentum convection option.
    convection_form=get(cp,'convection','form','advective').strip().lower()
    # NODALS_SKEW_PICARD_V1: one-half lagged-divergence correction.
    # NODALS_EMAC_PICARD_V1: opt-in volume EMAC Picard (M3 FGS/SGS only).
    if convection_form not in ('advective','conservative','skew_symmetric','emac'):
        raise ValueError('convection.form must be advective, conservative, skew_symmetric, or emac')
    if convection_form in ('conservative','skew_symmetric'):
        if state!='q1':
            raise ValueError('conservative momentum convection requires sst.state_space=q1')
        # Pure compact P1 changes only scalar k/omega transport; skew and conservative momentum
        # operate on the unchanged P1+BF3 velocity space. Retain other model checks.
        if mode=='m4-hybrid':
            raise ValueError('conservative momentum convection not supported with experimental M4')
    if convection_form=='emac':
        if state!='q1':
            raise ValueError('EMAC momentum requires sst.state_space=q1')
        if mode not in ('m3-fgs','m3-sgs'):
            raise ValueError('EMAC momentum currently requires M3 FGS or SGS')
        if flag(cp,'supg','momentum',False):
            raise ValueError('EMAC momentum does not support momentum SUPG yet')
    stress_form=get(cp,'stress','form','vector_laplacian').strip().lower()
    if stress_form not in ('vector_laplacian','deviatoric'):
        raise ValueError('stress.form must be vector_laplacian or deviatoric')
    if stress_form=='deviatoric':
        if state!='q1':raise ValueError('deviatoric stress currently requires sst.state_space=q1')
        if mode=='m4-hybrid':raise ValueError('deviatoric stress not supported for experimental M4')
    if flag(cp,'supg','k',False) or flag(cp,'supg','omega',False):
        raise ValueError('Full strong-residual scalar SUPG is not verified for k or omega; both must be false')
    # Unified full strong-residual momentum SUPG. No streamline-only modes.
    if flag(cp,'supg','momentum',False):
        if stress_form!='deviatoric' or convection_form not in ('advective','skew_symmetric','conservative'):
            raise ValueError('Full momentum SUPG is only available with deviatoric stress and advective, skew_symmetric or conservative convection')
        if state!='q1':
            raise ValueError('Full momentum SUPG requires sst.state_space=q1')
        if mode not in ('m3-fgs','m3-sgs'):
            raise ValueError('Full momentum SUPG requires M3 FGS or SGS')
        fd=number(cp,'supg','nu_fd_rel',0.001)
        if not 0 < fd <= 0.01:
            raise ValueError('supg.nu_fd_rel must be in (0,0.01]')
        if not number(cp,'supg','tau_scale',0.05)>0:
            raise ValueError('supg.tau_scale must be positive')
    sweeps=integer(cp,'solver.u','sweeps',5)
    if not 1<=sweeps<=60:raise ValueError('solver.u.sweeps outside 1..60')

    max_iter=args.outer_iterations if args.outer_iterations is not None else integer(cp,'simple','max_iterations',10)
    if args.steps is not None:
        if args.restart:
            m=re.search(r'_it(\d+)\.ckpt$',args.restart.name)
            if not m:
                raise ValueError('For --steps and --restart, checkpoint name must contain _itNNN.ckpt; use --outer-iterations otherwise')
            max_iter=int(m.group(1))+args.steps
        else:max_iter=args.steps
    if max_iter<1:raise ValueError('Iteration limit must be positive')
    iterations=int(max_iter)
    print_every=integer(cp,'output','print_every',10)
    metrics_every=integer(cp,'output','metrics_every',10)
    check_every=integer(cp,'sst','check_every',10)
    conv_metric=get(cp,'convergence','metric','legacy').strip().lower()
    if conv_metric not in ('legacy','pressure_drop','wall_traction','both'):
        raise ValueError('convergence.metric must be legacy, pressure_drop, wall_traction or both')
    conv_tol=number(cp,'convergence','relative_tolerance',0.01)
    conv_window=integer(cp,'convergence','window_iterations',25)
    conv_require_residuals=flag(cp,'convergence','require_residuals',True)
    if not (0 < conv_tol < 1) or not (2 <= conv_window <= 10000):
        raise ValueError('convergence tolerance must be (0,1), window_iterations 2..10000')
    if conv_metric!='legacy' and metrics_every!=1:
        raise ValueError('convergence.metric requires output.metrics_every=1 for consecutive SIMPLE iterations')
    if print_every<1 or metrics_every<1 or print_every%metrics_every!=0:
        raise ValueError('output.print_every must be a positive multiple of output.metrics_every')
    if check_every>0 and print_every%check_every!=0:
        raise ValueError('sst.check_every must divide output.print_every (or be 0 to disable scalar true residuals)')
    iterations=int(max_iter)
    samples=integer(cp,'wall_function','samples',1)
    omega_sample=integer(cp,'omega_wall','sample_index',3)
    if not (1<=samples<=5 and 1<=omega_sample<=5):
        raise ValueError('wall sample indices must be between 1 and 5')
    # NODALS_AUTO_CENTROID_SAMPLING_V1: opt-in geometry-based initial heights.
    # A numeric scale overrides auto independently for each wall model.
    auto_mode=get(cp,'wall_sampling','initial_distance','legacy_scale').strip().lower()
    if auto_mode not in ('legacy_scale','first_cell_centroid'):
        raise ValueError('wall_sampling.initial_distance must be legacy_scale or first_cell_centroid')
    auto_enabled=(auto_mode=='first_cell_centroid')
    def sampling_value(sec,key,fallback):
        raw=get(cp,sec,key,fallback).strip().lower()
        if raw=='auto':
            if not auto_enabled:
                raise ValueError(f'{sec}.{key}=auto requires [wall_sampling] initial_distance=first_cell_centroid')
            return 0.5,True
        return float(raw),False
    scale,auto_sp=sampling_value('wall_function','spalding_sample_scale',
                                 'auto' if auto_enabled else '0.5')
    omega_scale,auto_om=sampling_value('omega_wall','sample_scale',
                                 'auto' if auto_enabled else str(scale))
    if not (0.001<=scale<=1 and 0.001<=omega_scale<=1):
        raise ValueError('Sampling scales must be between 0.001 and 1')
    if get(cp,'wall_function','law','spalding')!='spalding':
        raise ValueError('Frozen reference supports wall_function.law=spalding')
    if get(cp,'omega_wall','method','of_auto')!='of_auto':
        raise ValueError('Frozen reference supports omega_wall.method=of_auto')
    couette_target=get(cp,'omega_wall','stage41_mode','sample_log').strip().lower()
    if couette_target not in ('sample_log','of_auto','trace_log','couette_point3'):
        raise ValueError('omega_wall.stage41_mode must be sample_log, of_auto or couette_point3')
    if couette_target=='trace_log':
        # Any valid 1..5 active Spalding fitting points are supported. A
        # one-point fit uses physical PROFILE5 position #3 (active mask 0x4).
        # The omega target remains at point #3 in every case.
        if omega_sample!=3:
            raise ValueError('trace_log requires omega sample index 3')
        if flag(cp,'omega_wall','production',False):
            raise ValueError('trace_log requires production=false')
        if state!='q1':
            raise ValueError('trace_log currently supports masked-P1/Q1 SST')
        tg=number(cp,'omega_wall','trace_gamma',50.0)
        if not (10<=tg<=100000):
            raise ValueError('omega_wall.trace_gamma must be between 10 and 100000')
    if couette_target=='couette_point3':
        if samples!=5 or omega_sample!=3:
            raise ValueError('couette_point3 needs wall_function.samples=5 and omega_wall.sample_index=3')
        if flag(cp,'omega_wall','production',False):
            raise ValueError('couette_point3 is omega-only: set omega_wall.production=false')
        if state!='q1':
            raise ValueError('couette_point3 is supported only for masked-P1/Q1 SST')

    # The original case runner produced five *physical* Spalding sample heights.
    # This is the exact Stage33BC reference rule (diameter-scaled and then
    # scaled by the wall-sample fraction), not a new wall-model discretization.
    # Users of different geometries may specify physical distances explicitly.
    raw_distances=get(cp,'wall_function','distances')
    if raw_distances:
        try:fit_heights=[float(v.strip()) for v in raw_distances.split(',')]
        except ValueError as ex:raise ValueError('wall_function.distances must contain five numbers') from ex
    else:
        original=(1.4694676501540146e-4,2.9389353003080293e-4,
                  4.4084029504620439e-4,5.8778706006160585e-4,
                  7.3473382507700737e-4)
        fit_heights=[d*diameter/0.004*scale for d in original]
    if (len(fit_heights)!=5 or not all(math.isfinite(y) for y in fit_heights) or
        not (0<fit_heights[0]<fit_heights[1]<fit_heights[2]<fit_heights[3]<fit_heights[4])):
        raise ValueError('wall_function.distances must be five strictly increasing positive values')
    ray_mode=get(cp,'wall_sampling','mode','ray_trace').strip().lower()
    if ray_mode not in ('ray_trace','legacy_endpoint'):
        raise ValueError('wall_sampling.mode must be ray_trace or legacy_endpoint')
    if ray_mode=='legacy_endpoint':
        print('WARNING: legacy_endpoint uses different wall sampling than the M3 reference ray_trace',file=sys.stderr)
    ray_cross=integer(cp,'wall_sampling','max_crossings',4096)
    if ray_cross<1:raise ValueError('wall_sampling.max_crossings must be positive')
    ray_tol=number(cp,'wall_sampling','bary_tolerance',1e-9)
    if ray_tol<=0:raise ValueError('wall_sampling.bary_tolerance must be positive')

    # Bazilevs-B is a momentum-wall-only option. Never mutate omega_wall.method.
    bazb=get(cp,'wall_function','tangential_consistency','off').strip().lower()
    if bazb not in ('off','symmetric_nitsche'):
        raise ValueError('wall_function.tangential_consistency must be off or symmetric_nitsche')
    bazb_visc=get(cp,'wall_function','consistency_viscosity','nu').strip().lower()
    if bazb_visc not in ('nu','nu_plus_nut'):
        raise ValueError('wall_function.consistency_viscosity must be nu or nu_plus_nut')
    if bazb!='off':
        if mode not in ('m3-fgs','m3-sgs','coupled-fgmres') or stress_form!='deviatoric' or state!='q1':
            raise ValueError('symmetric_nitsche requires M3 FGS/SGS or coupled FGMRES, deviatoric momentum and q1 turbulence')
        if not flag(cp,'wall_function','face_spalding',True) or not flag(cp,'wall_function','frozen_wall_cache',True):
            raise ValueError('symmetric_nitsche requires existing F1B/F2 wall options')
        if flag(cp,'supg','momentum',False):
            raise ValueError('symmetric_nitsche first experiment requires momentum SUPG off')
    # Exact environment compatibility flags used by the existing, unmodified CUDA solver.
    env={
        'SST_STATE_SPACE':state,
        'SST_COMPACT_P1':str(int(flag(cp,'sst','compact_p1',False))),
        'SST_COMPACT_QUAD':get(cp,'sst','compact_quad','24'),
        'NODALS_P2_STAGE5_RUN':'0',
        'NODALS_P2_STAGE4_TWO_STEP':'0',
        'NODALS_P2_NATIVE_PILOT':'0',
        'SST26_SPIRAL_DIAMETER':fmt(diameter),
        'SST23_INLET_VELOCITY':','.join(map(fmt,vel)),
        'SST_WALL_DISTANCE_METHOD':get(cp,'wall_distance','method','geometric'),
        'SST_WALL_SAMPLING_MODE':ray_mode,
        'NODALS_AUTO_CENTROID_MODE':'1' if auto_enabled else '0',
        'NODALS_AUTO_CENTROID_SP':'1' if auto_sp else '0',
        'NODALS_AUTO_CENTROID_OM':'1' if auto_om else '0',
        'NODALS_AUTO_CORNER_DEGREES':get(cp,'wall_sampling','corner_angle_degrees','45'),
        # NODALS_AUTO_THIN_CENTROID_V13: optional warning-only thin-passage fallback.
        'NODALS_AUTO_THIN_POLICY':get(cp,'wall_sampling','thin_channel_policy','error').strip().lower(),
        'SST_WALL_RAY_MAX_CROSSINGS':str(ray_cross),
        'SST_WALL_RAY_BARY_TOLERANCE':fmt(ray_tol),
        'STAGE27_S02B_PACK':'1',
        'STAGE27_INITIAL_VELOCITY':'potential' if start=='potential_flow' else 'plug',
        'STAGE27_SPALDING_ZERO_LIMIT':str(int(flag(cp,'wall_function','zero_limit',True))),
        'STAGE27_LOCAL_RAY_ADAPT':'1' if samples==1 and omega_sample==3 else '0',
        'SST24B_ACTIVE_POINTS':str(samples),
        'SST24B_OMEGA_SAMPLE_INDEX':str(omega_sample),
        # Only request the separate omega ray when the physical sample
        # distances differ. Same-scale cases use the original shared ray path,
        # which supports all 1..5 Spalding fitting points unchanged.
        'STAGE32_SPALDING_SAMPLE_SCALE':fmt(scale),
        'STAGE32_OMEGA_SAMPLE_SCALE':fmt(omega_scale),
        'SST22_TRACTION_MODEL':'vector_projected',
        'SST22D4_NITSCHE_ENABLE':'1',
        'SST22D4_GAMMA_N':fmt(number(cp,'wall_function','nitsche_gamma',1e-6)),
        'SST22D4I_BF3':'1',
        'SST22D4I3A_SCHUR':'1',
        'SST22D4D_MOM_MODE':'coupled',
        'SST22D4C_AUDIT_EVERY':str(iterations),
        'NODALS_MOMENTUM_STRESS_FORM':stress_form,
        'NODALS_MOMENTUM_CONVECTION_FORM':convection_form,
        'NODALS_M3_MODE':modes[mode],
        'NODALS_M3_SWEEPS':str(sweeps),
        'NODALS_M3_OMEGA':fmt(number(cp,'solver.u','omega',0.7)),
        'NODALS_F1B_FROZEN_WALL':str(int(flag(cp,'wall_function','frozen_wall_cache',True))),
        'NODALS_F2_FACE_SPALDING':str(int(flag(cp,'wall_function','face_spalding',True))),
        'NODALS_F1_FACE_LOCAL':'0',
        'NODALS_MOM_WALL_OPERATOR':'original',
        'NODALS_BAZILEVS_B_VISCOSITY':bazb_visc if bazb=='symmetric_nitsche' else 'off',
        'NODALS_P_REFRESH_EVERY':'1',
        'NODALS_F1B_GPU_GATE':'0',
        'STAGE27_WALL_AUDIT_MODE':'off',
        'STAGE27_S05A_K_MODE':get(cp,'sst','k_solver','fgs-fixed'),
        'STAGE27_S05A_OMEGA_MODE':get(cp,'sst','omega_solver','fgs-fixed'),
        'STAGE27_S05A_K_SWEEPS':str(integer(cp,'sst','k_sweeps',8)),
        'STAGE27_S05A_OMEGA_SWEEPS':str(integer(cp,'sst','omega_sweeps',8)),
        'STAGE27_S05A_CHECK_EVERY':str(check_every),
        'STAGE27_S05B2_FAST_SCALAR_ACTION':'1',
        'STAGE27_S05B1_MOM_MODE':'inherit',
        'STAGE27_S05B3_MOM_MODE':'inherit',
        'STAGE33A_MOMENTUM_SUPG':'0',  # retired approximate residual
        'NODALS_MOMENTUM_FULL_SUPG':str(int(flag(cp,'supg','momentum',False))),
        'NODALS_MOMENTUM_SUPG_NU_FD_REL':fmt(number(cp,'supg','nu_fd_rel',0.001)),
        'STAGE33BC_K_SUPG':str(int(flag(cp,'supg','k',False))),
        'STAGE33BC_OMEGA_SUPG':str(int(flag(cp,'supg','omega',False))),
        'STAGE33A_TAU_SCALE':fmt(number(cp,'supg','tau_scale',0.05)),
        'STAGE33BC_K_TAU_SCALE':fmt(number(cp,'supg','k_tau_scale',0.05)),
        'STAGE33BC_OMEGA_TAU_SCALE':fmt(number(cp,'supg','omega_tau_scale',0.05)),
        'NODALS_G2_CHILD_PROFILE':str(int(flag(cp,'output','profile',False))),
        'NODALS_PREDICTOR_AUDIT':'0',
        'STAGE31_CHECKPOINT_EVERY':str(integer(cp,'output','checkpoint_every',100)),
        'STAGE31_RESTART_EVERY':str(integer(cp,'output','checkpoint_every',100)),
        'SST_G2_VTU_EVERY':str(integer(cp,'output','vtu_every',100)),
        'NODALS_SST_DIAGNOSTIC_EVERY':str(integer(cp,'output','diagnostics_every',0)),
        'NODALS_LEGACY_FINAL_DIAGNOSTICS':str(int(flag(cp,'output','legacy_final_diagnostics',False))),
        'P_AMG_SMOOTHER':get(cp,'amg','smoother','jacobi'),
        'P_AMG_FINE_SWEEPS':str(integer(cp,'amg','fine_sweeps',1)),
        'P_AMG_COARSE_SWEEPS':str(integer(cp,'amg','coarse_sweeps',1)),
        'STAGE41_OMEGA_MODE':couette_target,
        'STAGE41_TRACE_GAMMA':fmt(number(cp,'omega_wall','trace_gamma',50.0)),
        'STAGE27_S05B1_BLOCK_MAX':'3',
        'STAGE27_S05B1_COMPONENT_MAX':'6',
        'STAGE27_S05B1_PRECOND_FGS':'1',
        'SST22D4D_RESTART':str(integer(cp,'solver.u','restart',30)),
        'SST22D4D_MAXITS':str(integer(cp,'solver.u','max_iterations',800)),
    }
    env['SST_G2_VTU_DIR']=str(resolve_output_child(get(cp,'output','vtu_dir',''),output))
    env['STAGE31_CHECKPOINT_DIR']=str(resolve_output_child(get(cp,'output','checkpoint_dir',''),output))
    if mode=='m4-hybrid':print('WARNING: M4 is experimental and has not passed GPU parity.',file=sys.stderr)
    if mode=='coupled-fgmres':env['STAGE27_S05B1_MOM_MODE']='inherit'

    if args.restart:
        if not args.restart.is_file():raise FileNotFoundError('Restart checkpoint missing: '+str(args.restart))
        env['STAGE31_RESTART_FROM']=str(args.restart)
    # Preserve old Stage31 input identity semantics ONLY to protect restart state.
    # No installed binary or stage hash is ever checked for normal execution.
    if mesh_sha:
        profile_path=get(cp,'inlet.main','profile_csv')
        profile_hash=sha256(absolute(profile_path,args.case.parent)) if profile_path else None
        fixed={'mesh_sha256':mesh_sha,'profile_sha256':profile_hash,'re':reynolds,
               'nu':nu,'diameter':diameter,'density':1.0}
        env['STAGE31_INPUT_FINGERPRINT']=hashlib.sha256(
            json.dumps(fixed,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if args.restart and not mesh_sha and not get(cp,'restart','input_fingerprint'):
        raise ValueError('Restart from preconverted HXT1 requires restart.input_fingerprint (or original .msh)')
    if get(cp,'restart','input_fingerprint'):
        env['STAGE31_INPUT_FINGERPRINT']=get(cp,'restart','input_fingerprint')

    profile=get(cp,'inlet.main','profile_csv')
    if profile:
        profile=absolute(profile,args.case.parent)
        if not profile.is_file():raise FileNotFoundError('Inlet profile missing: '+str(profile))
        env['STAGE27_INLET_PROFILE_CSV']=str(profile)
    if get(cp,'mesh','patch_map'):
        patch=absolute(get(cp,'mesh','patch_map'),args.case.parent)
        if not patch.is_file():raise FileNotFoundError('Boundary patch map missing: '+str(patch))
        env['SST_PATCH_MAP_FILE']=str(patch)

    # Advanced compatibility with all existing native options while their APIs mature.
    if cp.has_section('environment'):
        for k,v in cp.items('environment'):
            key=k.upper()
            if not re.fullmatch(r'[A-Z_][A-Z_0-9]*',key):raise ValueError('Invalid environment key '+k)
            if key in {
                'NODALS_SKEW_STREAMLINE_DIFFUSION','NODALS_SKEW_SD_TAU_SCALE',
                'NODALS_SKEW_FULL_SUPG','NODALS_SKEW_FULL_NU_FD_REL',
                'NODALS_CONSERVATIVE_STREAMLINE_DIFFUSION','NODALS_CONSERVATIVE_SD_TAU_SCALE',
                'NODALS_CONSERVATIVE_FULL_SUPG','NODALS_CONSERVATIVE_FULL_NU_FD_REL',
                'STAGE33A_MOMENTUM_SUPG','NODALS_MOMENTUM_FULL_SUPG',
                'NODALS_MOMENTUM_SUPG_NU_FD_REL'}:
                raise ValueError(f'Deprecated/restricted momentum SUPG environment key {key}; use [supg] momentum, tau_scale, nu_fd_rel')
            env[key]=v

    # A separate optional physics-plateau gate. Environment is emitted after
    # user [environment] passthrough so parsed [convergence] is authoritative.
    env['NODALS_CONV_METRIC']=conv_metric
    env['NODALS_CONV_RTOL']=fmt(conv_tol)
    env['NODALS_CONV_WINDOW']=str(conv_window)
    env['NODALS_CONV_REQUIRE_RESIDUALS']='1' if conv_require_residuals else '0'
    native=[str(args.binary),'--mesh',str(mesh),'--vtu',str(output/'flow_sst.vtu'),
        '--re',fmt(reynolds),'--bulk',fmt(bulk),
        '--sst-g2','1','--sst-g2-start','plug',
        '--simple-coupling','corrected',
        '--max-outer',str(iterations),'--sst-g2-max',str(iterations),
        '--sst-g2-physical-gate',str(int(flag(cp,'simple','physical_gate',False))),
        '--sst-g2-print-every',str(print_every),
        '--sst-g2-metrics-every',str(metrics_every),
        '--sst-g2-cont-atol',fmt(number(cp,'simple','continuity_atol',1e-6)),
        '--sst-g2-plateau-tol',fmt(number(cp,'simple','plateau_tol',1e-2)),
        '--sst-g2-plateau-samples',str(integer(cp,'simple','plateau_samples',5)),
        # Puretet application rejects missing PROFILE5 closure and coordinates.
        # Pass them explicitly: RUN_HXT5B.sh otherwise defaults closure to 0.
        '--sst-shared-spalding-wall-state',str(int(flag(cp,'wall_function','shared_wall_state',True))),
        '--sst-profile-integral-utau',str(int(flag(cp,'wall_function','integral_utau',True))),
        '--sst-profile-integral-nq',str(integer(cp,'wall_function','integral_nq',12)),
        '--sst-profile-integral-closure','0',
        '--sst-profile-average-closure','1',
        '--sst-profile-average-wall-law','spalding_profile5',
        '--sst-profile-match-y','0',
        '--sst-profile-fit-y1',fmt(fit_heights[0]),
        '--sst-profile-fit-y2',fmt(fit_heights[1]),
        '--sst-profile-fit-y3',fmt(fit_heights[2]),
        '--sst-profile-fit-y4',fmt(fit_heights[3]),
        '--sst-profile-fit-y5',fmt(fit_heights[4]),
        '--sst-of-production',str(int(flag(cp,'omega_wall','production',False))),
        '--sst-of-kappa',fmt(number(cp,'sst','of_kappa',0.4)),
        '--sst-of-E',fmt(number(cp,'sst','of_e',9.025013499434122)),
        '--sst-of-beta1',fmt(number(cp,'sst','of_beta1',0.075)),
        '--sst-of-re-blend',fmt(number(cp,'sst','of_re_blend',11.0)),
        '--sst-intensity',fmt(number(cp,'sst','intensity',0.05)),
        '--sst-length-ratio',fmt(number(cp,'sst','length_ratio',0.07)),
        '--sst-alpha-k',fmt(number(cp,'sst','ur_k',0.7)),
        '--sst-alpha-omega',fmt(number(cp,'sst','ur_omega',0.7)),
        '--sst-k-solver',get(cp,'sst','k_solver','fgs-fixed'),
        '--sst-omega-solver',get(cp,'sst','omega_solver','fgs-fixed'),
        '--sst-k-fgs-sweeps',str(integer(cp,'sst','k_sweeps',8)),
        '--sst-omega-fgs-sweeps',str(integer(cp,'sst','omega_sweeps',8)),
        '--sst-fgs-check-every',str(check_every),
        '--sst-omega-wall-mode',get(cp,'omega_wall','method','of_auto'),
        '--sst-wall-penalty-gamma',fmt(number(cp,'omega_wall','gamma',1e6)),
        '--alpha-u',fmt(number(cp,'simple','ur_u',0.7)),
        '--alpha-p',fmt(number(cp,'simple','ur_p',0.7)),
        '--simple-tol',fmt(number(cp,'simple','simple_tol',1e-12)),
        '--gamma',fmt(number(cp,'wall_function','nitsche_gamma',1e-6)),
        '--momentum-work','fgmres-fgs',
        '--mom-max',str(integer(cp,'solver.u','max_iterations',800)),
        '--mom-rtol',fmt(number(cp,'solver.u','rtol',1e-6)),
        '--mom-atol',fmt(number(cp,'solver.u','atol',0.0)),
        '--mom-fgmres-restart',str(integer(cp,'solver.u','restart',30)),
        '--pressure-solver',get(cp,'solver.p','method','pcg_amg'),
        '--p-rtol',fmt(number(cp,'solver.p','rtol',0.5)),
        '--p-atol',fmt(number(cp,'solver.p','atol',1e-12)),
        '--p-max',str(integer(cp,'solver.p','max_iterations',600)),
        '--amg-hierarchy',get(cp,'amg','hierarchy','cf'),
        '--cf-coarsening',get(cp,'amg','coarsening','pmis'),
        '--cf-strength',get(cp,'amg','strength','classical-negative'),
        '--cf-interp',get(cp,'amg','interpolation','exti'),
        '--cf-theta',fmt(number(cp,'amg','theta',0.25)),
        '--amg-terminal',str(integer(cp,'amg','terminal',1000)),
        '--amg-smoother',get(cp,'amg','smoother','jacobi'),
        '--amg-fine-mode',get(cp,'amg','fine_mode','explicit'),
        '--amg-jacobi-fine-sweeps',str(integer(cp,'amg','fine_sweeps',1)),
        '--amg-jacobi-coarse-sweeps',str(integer(cp,'amg','coarse_sweeps',1)),
        '--profile-audit',str(int(flag(cp,'output','profile_audit',False)))
    ]
    if cp.has_section('native'):
        for k,v in cp.items('native'):
            if not re.fullmatch(r'[a-z][a-z0-9-]*',k):raise ValueError('Invalid native CLI key '+k)
            native.extend(['--'+k,v])
    for opt in args.native_option:
        try:k,v=opt.split('=',1)
        except ValueError:raise ValueError('--native-option expects flag=value')
        if not re.fullmatch(r'[a-z][a-z0-9-]*',k):raise ValueError('Invalid native option '+k)
        native.extend(['--'+k,v])
    return native,env,dict(reynolds=reynolds,bulk=bulk,diameter=diameter,nu=nu,
                           scalar_space=state,momentum=mode,outer_iteration_limit=iterations,
                           wall_sampling=ray_mode,profile_fit_heights=fit_heights)

# NODALS_SST_OPTIONAL_LOG_MONITOR_V1
# Plotting is an optional observer, never a solver dependency or convergence gate.
def maybe_launch_plot_monitor(cp, output):
    try:
        if not flag(cp, 'output', 'live_monitor', False):
            return
        import importlib.util
        import threading
        import time
        refresh = number(cp, 'output', 'monitor_refresh_seconds', 3.0)
        pressure_scale = number(cp, 'output', 'monitor_pressure_scale', 1.0)
        if not (0.2 <= refresh <= 3600.0 and pressure_scale > 0):
            raise ValueError('monitor refresh must be 0.2..3600 s and pressure scale >0')
        script = ROOT/'tools'/'nodals_live_monitor.py'
        if not script.is_file():
            raise FileNotFoundError('monitor script missing: '+str(script))
        if importlib.util.find_spec('matplotlib') is None:
            raise RuntimeError('matplotlib not installed; install python3-matplotlib')
        if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
            raise RuntimeError('GUI display missing (DISPLAY/WAYLAND_DISPLAY not set)')
        error_log=output/'nodals_monitor_errors.log'
        with error_log.open('a') as error_stream:
            proc = subprocess.Popen(
                [sys.executable, str(script), '--log', str(output/'terminal.log'),
                 '--interval', fmt(refresh), '--pressure-scale', fmt(pressure_scale)],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=error_stream, start_new_session=True)
        # Use an actual file (not a PIPE). The GUI can outlive nodals.py and
        # still report errors without keeping the solver's tee pipe open.
        def relay_errors():
            try:
                with error_log.open('r') as report:
                    while True:
                        line=report.readline()
                        if line:
                            if line.strip():
                                print('NODALS_MONITOR_ERROR '+line.rstrip()+' (solver continues)',
                                      file=sys.stderr,flush=True)
                        elif proc.poll() is not None:
                            break
                        else:
                            time.sleep(0.5)
            except Exception:
                pass
        threading.Thread(target=relay_errors, daemon=True).start()
        print('NODALS_MONITOR_STARTED pid='+str(proc.pid)+
              ' log='+str(output/'terminal.log')+
              ' errors='+str(error_log)+
              ' independent=1 solverGate=UNCHANGED',flush=True)
    except Exception as exc:
        print('NODALS_MONITOR_ERROR '+str(exc)+'; solver continues unchanged',
              file=sys.stderr,flush=True)

def run(args):
    args.case=absolute(args.case,Path.cwd())
    args.binary=absolute(args.binary,Path.cwd())
    if args.restart:args.restart=absolute(args.restart,Path.cwd())
    cp=read_case(args.case,args.set)
    if args.restart is None:
        restart_path=get(cp,'restart','path','').strip()
        if restart_path:
            args.restart=absolute(restart_path,args.case.parent)
    source=absolute(args.mesh or get(cp,'mesh','path'),args.case.parent)
    if not source.is_file():raise FileNotFoundError('Mesh not found: '+str(source))
    mesh_format=get(cp,'mesh','format','msh2').lower()
    if mesh_format not in ('msh2','hxt1'):raise ValueError('mesh.format must be msh2 or HXT1')
    cache=absolute(args.cache,Path.cwd())
    if args.output:
        out=absolute(args.output,Path.cwd())
    elif get(cp,'output','directory','').strip():
        out=absolute(get(cp,'output','directory'),args.case.parent)
    else:
        stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        out=ROOT/'runs'/f'{args.case.stem}_{stamp}'
    prep=not args.dry_run
    native_mesh,fingerprint=mesh_prepare(source,cache,prep)
    # Optional case-driven inlet interpolation for a different inlet mesh.
    # The pre-existing native loader still receives its strict vertex-matched CSV.
    raw_csv=get(cp,'inlet.main','profile_csv','').strip()
    mapping=get(cp,'inlet.main','profile_mapping','exact').strip().lower()
    flux_mode=get(cp,'inlet.main','profile_flux','preserve').strip().lower()
    if mapping not in ('exact','interpolate'):
        raise ValueError('inlet.main.profile_mapping must be exact or interpolate')
    if flux_mode not in ('preserve','match_bulk'):
        raise ValueError('inlet.main.profile_flux must be preserve or match_bulk')
    if mapping=='interpolate' and not raw_csv:
        raise ValueError('profile_mapping=interpolate requires inlet.main.profile_csv')
    if raw_csv:
        original_csv=absolute(raw_csv,args.case.parent)
        if not original_csv.is_file():
            raise FileNotFoundError('Inlet profile missing: '+str(original_csv))
        if mapping=='interpolate':
            if source.suffix.lower()!='.msh':
                raise ValueError('profile_mapping=interpolate requires mesh.path to be a Gmsh ASCII 2.2 .msh')
            from tools.profile_resample import prepare_profile
            inlet_mode=get(cp,'inlet.main','mode','velocity').strip().lower()
            if inlet_mode=='velocity':
                base_vel=vec3(get(cp,'inlet.main','velocity'))
            elif inlet_mode=='speed':
                base_vel=vec3(get(cp,'inlet.main','direction','(1,0,0)'))
                n=math.sqrt(sum(v*v for v in base_vel))
                if n==0:raise ValueError('Zero inlet direction')
                base_vel=[number(cp,'inlet.main','speed')*v/n for v in base_vel]
            else:raise ValueError('Unsupported inlet.main.mode for profile interpolation')
            corner_epsilon=number(cp,'inlet.main','profile_corner_epsilon',1e-9)
            cache_key=hashlib.sha256(json.dumps({
                'algorithm':'profile-interpolate-v1',
                'source':sha256(original_csv),
                'mesh':sha256(source),
                'flux_mode':flux_mode,
                'base_vel':base_vel,
                'corner_epsilon':corner_epsilon,
            },sort_keys=True).encode()).hexdigest()[:24]
            prepared=ROOT/'.cache'/'profiles'/f'{cache_key}_inlet.csv'
            if not prepared.is_file():
                report=prepare_profile(original_csv,source,prepared,base_vel,flux_mode,corner_epsilon)
                print('NODALS_INLET_PROFILE_PREP='+json.dumps(report,sort_keys=True),flush=True)
            elif prepared.is_file():
                print('NODALS_INLET_PROFILE_CACHED='+str(prepared),flush=True)
            cp.set('inlet.main','profile_csv',str(prepared))
        else:
            if flux_mode!='preserve':
                raise ValueError('profile_flux=match_bulk requires profile_mapping=interpolate')
            print('NODALS_INLET_PROFILE_EXACT='+str(original_csv),flush=True)
    argv,env,settings=make_invocation(cp,args,native_mesh,source,fingerprint,out)
    print('NODALS_SOFTWARE_RUN case='+str(args.case)+' mesh='+str(native_mesh))
    print('NODALS_SOFTWARE_MODE='+settings['momentum']+' scalar='+settings['scalar_space']+
          ' maxOuter='+str(settings['outer_iteration_limit']))
    print('NODALS_SOFTWARE_STRESS_FORM='+env['NODALS_MOMENTUM_STRESS_FORM'])
    print('NODALS_SOFTWARE_MOMENTUM_CONVECTION='+env['NODALS_MOMENTUM_CONVECTION_FORM'])
    print('NODALS_NATIVE_CMD='+shlex.join(argv))
    if args.dry_run:
        print('NODALS_DRY_RUN=YES; inlet profile may have been prepared in cache; no mesh conversion, GPU invocation or run directory created')
        return 0
    if args.prepare_only:
        print('NODALS_MESH_READY='+str(native_mesh)+'; GPU solver not invoked')
        return 0
    if not args.binary.is_file():
        raise FileNotFoundError('Executable not present: '+str(args.binary)+'; run make -j4 first')
    if not os.access(args.binary,os.X_OK):
        raise PermissionError('Executable is not executable: '+str(args.binary))
    out.mkdir(parents=True,exist_ok=True)
    # Native checkpoint and visualization destinations are explicit and writable.
    for directory in (env['SST_G2_VTU_DIR'],env['STAGE31_CHECKPOINT_DIR']):
        Path(directory).mkdir(parents=True,exist_ok=True)
    (out/'invocation.json').write_text(json.dumps({
        'case':str(args.case),'mesh_source':str(source),'compiled_mesh':str(native_mesh),
        'command':argv,'environment_overrides':env,'settings':settings,
        'source_mesh_sha256':fingerprint,
    },indent=2)+'\n')
    # Do not inherit stale stage-run flags from an earlier terminal session.
    # An advanced user can explicitly restore any required flags via [environment].
    clean_parent={k:v for k,v in os.environ.items()
                  if not re.match(r'^(SST|STAGE|NODALS|P_AMG|HXT)',k)}
    compact=get(cp,'output','console','compact').strip().lower()=='compact'
    if get(cp,'output','console','compact').strip().lower() not in ('compact','full'):
        raise ValueError('output.console must be compact or full')
    with (out/'terminal.log').open('w') as f:
        maybe_launch_plot_monitor(cp, out)
        proc=subprocess.Popen(argv,env={**clean_parent,**env},cwd=out,
                              stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,errors='replace',bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            f.write(line)
            if not compact or compact_line(line):print(line,end='',flush=True)
        rc=proc.wait()
    print('NODALS_NATIVE_EXIT_CODE='+str(rc)+' output='+str(out))
    return rc

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('run',help='Convert mesh if needed; execute frozen CUDA solver directly')
    p.add_argument('--case',type=Path,required=True)
    p.add_argument('--mesh',type=Path,help='Override case mesh')
    p.add_argument('--output',type=Path,help='Output directory (default: unique runs/case_timestamp)')
    p.add_argument('--binary',type=Path,default=ROOT/'build'/'nodals_sst')
    p.add_argument('--cache',type=Path,default=ROOT/'.cache'/'meshes')
    p.add_argument('--outer-iterations',type=int,help='Absolute native maximum iteration, including restart count')
    p.add_argument('--steps',type=int,help='Iterations from fresh start or from _itNNN.ckpt restart')
    p.add_argument('--restart',type=Path,help='Optional native Stage31 checkpoint, no mandatory checkpoint')
    p.add_argument('--momentum-solver',choices=['m3-fgs','m3-sgs','m4-hybrid','coupled-fgmres'])
    p.add_argument('--set',action='append',default=[],help='Override SECTION.KEY=VALUE, repeatable')
    p.add_argument('--native-option',action='append',default=[],help='Pass a native compiled option FLAG=VALUE')
    p.add_argument('--dry-run',action='store_true',help='Check input and show exact executable invocation')
    p.add_argument('--prepare-only',action='store_true',help='Build/preflight cached HXT1, no GPU run')
    a=parser.parse_args()
    try:return run(a)
    except (ValueError,RuntimeError,FileNotFoundError,PermissionError,subprocess.CalledProcessError) as e:
        print('NODALS_SOFTWARE_ERROR: '+str(e),file=sys.stderr)
        return 2

if __name__=='__main__':sys.exit(main())
