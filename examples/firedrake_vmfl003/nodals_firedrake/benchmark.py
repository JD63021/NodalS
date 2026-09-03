"""VMFL003 benchmark driver.

This file contains only the benchmark orchestration and post-processing. The
finite-element definition, operators, and wall/turbulence physics live in
separate modules so each mathematical component can be reviewed independently.
"""
from pathlib import Path
import csv
import time
import math
import numpy as np

from firedrake import (
    Constant, DirichletBC, FacetNormal, TestFunction, SpatialCoordinate,
    assemble, ds_b, ds_t,
)
from petsc4py import PETSc
from mpi4py import MPI

from .config import (
    Print, R, D, L, NU, RHO, U_BULK, RE,
    ALPHA_P, SIMPLE_TOL, MAX_OUTER, P_RTOL,
    WALL_KAPPA, WALL_B, WALL_SAMPLE_FRACTION,
    VMFL003_TARGET_DP_PA, SMOOTH_MOODY_FD, BENCHMARK_DP_TOL_PCT,
    DXQ,
)
from .element import build_mesh_and_spaces
from .operators import (
    assemble_B_components, assemble_dg_fixed_flux, assemble_scalar_diffusion,
    dg_inlet_bilinear, global_vec_sum, function_from_vec,
    finalise_simplec, build_schur, configure_ksp, local_bc_nodes,
    global_owned_count, set_owned_values, vec_copy_from_cofunction,
)
from .physics import (
    mpi_sum, mpi_max, audit_exact_bf2_basis, OffwallSpaldingPlan,
    assemble_rans_lagged, cell_average_stats, initialise_constant_q1_plug,
    wall_sample_stats, make_xy_clamped_matrix, true_momentum_solve,
)

comm = PETSc.COMM_WORLD.tompi4py()


def make_component_pressure_operators(Braw, wall_nodes):
    """Pressure coupling: Bx/By omit clamped wall DOFs; Bz retains axial wall DOFs."""
    mask = Braw[0].createVecRight()
    mask.set(1.0)
    if len(wall_nodes):
        mask.setValues(
            np.asarray(wall_nodes, dtype=PETSc.IntType),
            np.zeros(len(wall_nodes), dtype=float),
            addv=PETSc.InsertMode.INSERT_VALUES,
        )
    mask.assemble()

    bx = Braw[0].copy()
    by = Braw[1].copy()
    bx.diagonalScale(R=mask)
    by.diagonalScale(R=mask)
    bz = Braw[2]

    wallprobe = Braw[0].createVecRight()
    wallprobe.set(0.0)
    if len(wall_nodes):
        wallprobe.setValues(
            np.asarray(wall_nodes, dtype=PETSc.IntType),
            np.ones(len(wall_nodes), dtype=float),
            addv=PETSc.InsertMode.INSERT_VALUES,
        )
    wallprobe.assemble()

    qtmp = Braw[0].createVecLeft()
    bx.mult(wallprobe, qtmp); nx = qtmp.norm()
    by.mult(wallprobe, qtmp); ny = qtmp.norm()
    bz.mult(wallprobe, qtmp); nz = qtmp.norm()
    Print(
        f"VMFL003_PRESSURE_WALL_MASK_AUDIT normBxWallProbe={nx:.12e} "
        f"normByWallProbe={ny:.12e} normBzWallProbe={nz:.12e} "
        f"semantics=BxBy_skip_wall_Bz_retains_wall"
    )
    if nx > 1.0e-13 or ny > 1.0e-13 or nz <= 1.0e-13:
        raise RuntimeError(f"component pressure wall-mask audit failed Bx={nx} By={ny} Bz={nz}")
    wallprobe.destroy(); qtmp.destroy(); mask.destroy()
    Braw[0].destroy(); Braw[1].destroy()
    return [bx, by, bz]

def continuity_vector(S, Bs, U, fixed_flux):
    out = S.createVecRight(); out.set(0.0); out.axpy(1.0, fixed_flux)
    tmp = S.createVecRight()
    for d in range(3):
        Bs[d].mult(U[d], tmp); out.axpy(1.0, tmp)
    tmp.destroy()
    return out

def axial_pressure_planes(mesh, Q, pvec):
    """68 volume-weighted Q0 pressure planes."""
    q = TestFunction(Q); z = SpatialCoordinate(mesh)[2]
    vvec = vec_copy_from_cofunction(assemble(q*DXQ))
    zvec = vec_copy_from_cofunction(assemble(z*q*DXQ))
    va = np.asarray(vvec.getArray(readonly=True), dtype=float)
    za = np.asarray(zvec.getArray(readonly=True), dtype=float)
    pa = np.asarray(pvec.getArray(readonly=True), dtype=float)
    zc = za / np.maximum(va, 1.0e-300)
    centers = np.concatenate([
        (np.arange(20, dtype=float) + 0.5)*D,
        20.0*D + (np.arange(48, dtype=float) + 0.5)*(10.0*D),
    ])
    local_v=np.zeros(68); local_pv=np.zeros(68); local_n=np.zeros(68,dtype=np.int64)
    for i in range(len(zc)):
        k=int(np.argmin(np.abs(centers-zc[i])))
        if abs(centers[k]-zc[i]) > 1.0e-8*D:
            raise RuntimeError(f"cell centroid does not match known axial layer z={zc[i]:.17e}")
        local_v[k]+=va[i]; local_pv[k]+=va[i]*pa[i]; local_n[k]+=1
    gv=np.zeros_like(local_v); gpv=np.zeros_like(local_pv); gn=np.zeros_like(local_n)
    comm.Allreduce(local_v,gv,op=MPI.SUM); comm.Allreduce(local_pv,gpv,op=MPI.SUM); comm.Allreduce(local_n,gn,op=MPI.SUM)
    if np.any(gn != 192): raise RuntimeError("axial layer cell-count audit failed")
    pbar=gpv/gv; vvec.destroy(); zvec.destroy(); return centers,pbar,gn

def linear_fit(x, y):
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float)
    A=np.column_stack((x,np.ones_like(x))); coef,*_=np.linalg.lstsq(A,y,rcond=None)
    slope,intercept=float(coef[0]),float(coef[1]); pred=slope*x+intercept
    ssr=float(np.sum((y-pred)**2)); sst=float(np.sum((y-np.mean(y))**2)); r2=1.0-ssr/max(sst,1.0e-300)
    return slope,intercept,r2

def f_from_kinematic_slope(slope):
    return abs(-2.0*D*slope/(U_BULK*U_BULK))

def write_benchmark_output(outdir, zplanes, pbar, final_meta):
    outdir.mkdir(parents=True, exist_ok=True); zD=zplanes/D
    mask_dev=(zplanes>=0.60*L)&(zplanes<=0.90*L)
    dev_slope,_,dev_r2=linear_fit(zplanes[mask_dev],pbar[mask_dev]); dev_fd=f_from_kinematic_slope(dev_slope)
    center_span=float(zplanes[-1]-zplanes[0]); center_drop_kin=float(pbar[0]-pbar[-1])
    inlet_slope=float((pbar[1]-pbar[0])/(zplanes[1]-zplanes[0])); outlet_slope=float((pbar[-1]-pbar[-2])/(zplanes[-1]-zplanes[-2]))
    pin=float(pbar[0]-inlet_slope*zplanes[0]); pout=float(pbar[-1]+outlet_slope*(L-zplanes[-1])); full_drop_kin=pin-pout
    full_drop_pa=RHO*full_drop_kin; center_drop_pa=RHO*center_drop_kin
    full_fd=2.0*D*full_drop_kin/(L*U_BULK*U_BULK); center_fd=2.0*D*center_drop_kin/(center_span*U_BULK*U_BULK)
    target_err_pct=100.0*(full_drop_pa/VMFL003_TARGET_DP_PA-1.0); moody_err_pct=100.0*(dev_fd/SMOOTH_MOODY_FD-1.0)
    zones=[]
    for j in range(10):
        lo,hi=50.0*j,50.0*(j+1); mask=(zD>=lo)&(zD<=hi); slope,_,r2=linear_fit(zplanes[mask],pbar[mask])
        zones.append((j,lo,hi,0.5*(lo+hi),slope,r2,f_from_kinematic_slope(slope)))
    if comm.rank==0:
        with (outdir/"axial_pressure_planes.csv").open("w",newline="") as fh:
            w=csv.writer(fh); w.writerow(["plane","z_m","z_over_D","p_kinematic_m2_s2","p_physical_Pa","p_minus_last_Pa"])
            for i,(zz,pp) in enumerate(zip(zplanes,pbar)):
                w.writerow([i,f"{zz:.17e}",f"{zz/D:.12f}",f"{pp:.17e}",f"{RHO*pp:.17e}",f"{RHO*(pp-pbar[-1]):.17e}"])
        with (outdir/"friction_zones_50D.csv").open("w",newline="") as fh:
            w=csv.writer(fh); w.writerow(["zone","z_over_D_start","z_over_D_end","z_over_D_mid","dpdz_kinematic","fit_R2","Darcy_f"])
            for row in zones: w.writerow([row[0],f"{row[1]:.6f}",f"{row[2]:.6f}",f"{row[3]:.6f}",f"{row[4]:.17e}",f"{row[5]:.12e}",f"{row[6]:.12e}"])
        with (outdir/"benchmark_summary.txt").open("w") as fh:
            fh.write("VMFL003_FIREDRAKE_SUMMARY\n"+f"simple_iterations={final_meta['iterations']}\n"+f"simple_rel_cont={final_meta['rel_cont']:.17e}\n"+f"outlet_flux_rel_error={final_meta['flux_err']:.17e}\n"+f"yplus_mean={final_meta['yplus_mean']:.17e}\n"+f"fwall_spalding={final_meta['fwall']:.17e}\n"+f"developed_Darcy_f={dev_fd:.17e}\n"+f"full_drop_Pa={full_drop_pa:.17e}\n"+f"VMFL003_target_error_pct={target_err_pct:.12f}\n")
    Print(f"VMFL003_PRESSURE_FIT zOverD=[300,450] dpdz={dev_slope:.12e} fitR2={dev_r2:.10f} fDarcy={dev_fd:.10f} errorVsSmoothMoodyPct={moody_err_pct:.6f}")
    Print(f"VMFL003_TOTAL_PRESSURE_DROP centerSpanOverD={center_span/D:.6f} centerPlaneDropPa={center_drop_pa:.12e} fullPipeExtrapolatedDropPa={full_drop_pa:.12e} targetPa={VMFL003_TARGET_DP_PA:.12e} targetErrorPct={target_err_pct:.8f}")
    for row in zones: Print(f"VMFL003_FRICTION_ZONE zone={row[0]} zOverD=[{row[1]:.0f},{row[2]:.0f}] fDarcy={row[6]:.10f} fitR2={row[5]:.10f}")
    return {"dev_fd":dev_fd,"dev_slope":dev_slope,"dev_r2":dev_r2,"full_drop_pa":full_drop_pa,"full_fd":full_fd,"target_err_pct":target_err_pct,"zones":zones}

def run_benchmark():
    Print("=== FULL VMFL003 500D PRODUCTION SOLVE ===")
    Print(f"VMFL003_PHYSICS Ubulk={U_BULK:.12e} nu={NU:.12e} D={D:.12e} L={L:.12e} Re={RE:.12e} rho={RHO:.6f} convection=central mixlen=nikuradse_pipe wallLaw=spalding kappa={WALL_KAPPA} B={WALL_B} sampleMode=offwall_midplane sampleFraction={WALL_SAMPLE_FRACTION} DGinlet=1 turbulentNitsche=nu_plus_nuT SUPG=OFF")
    Print(f"VMFL003_PRODUCTION_GATE simpleRtol={SIMPLE_TOL:.3e} maxIts={MAX_OUTER} gate=all_initial_momentum_plus_continuity velocityCorrection=NONE_pressure_only pressureWallMask=BxBy_skip_wall_Bz_retains_wall")
    t0=time.perf_counter(); mesh,V,Q=build_mesh_and_spaces(); bf2_center_vals,bf2_center_err,q1_center_err=audit_exact_bf2_basis(V)
    Print(f"VMFL003_BF2_BASIS_AUDIT status=PASS contract=Q2mid_tangential_Q1linear_normal faceModes={len(bf2_center_vals)} cellCenterFaceValues=[{','.join(f'{x:.6f}' for x in bf2_center_vals)}] target=0.5 maxFaceErr={bf2_center_err:.3e} q1CenterErr={q1_center_err:.3e}")
    fixed=np.empty(0,dtype=PETSc.IntType); wall_bc=DirichletBC(V,0.0,1); wall_clamp=local_bc_nodes(wall_bc); wall_clamp_global=global_owned_count(wall_clamp)
    uhat=(Constant(0.0),Constant(0.0),Constant(U_BULK)); Braw=assemble_B_components(mesh,V,Q,dg_trace=True); Bs=make_component_pressure_operators(Braw,wall_clamp)
    fixed_flux=assemble_dg_fixed_flux(mesh,Q,uhat); target_outward=global_vec_sum(fixed_flux)
    probeA=assemble_scalar_diffusion(V,extra_bilinear=dg_inlet_bilinear(mesh,uhat=uhat)); pres0=probeA.createVecRight(); pres0.set(0.0)
    U=[probeA.createVecRight() for _ in range(3)]; [v.set(0.0) for v in U]; p=fixed_flux.duplicate(); p.set(0.0)
    wallplan=OffwallSpaldingPlan(mesh,V); q1_g,bf2_g,bf2_init_max=initialise_constant_q1_plug(mesh,V,U[2],wallplan.vkinds,U_BULK); set_owned_values(U[0],wall_clamp,0.0); set_owned_values(U[1],wall_clamp,0.0)
    plug_smin,plug_smean,plug_smax=wall_sample_stats(wallplan,U[2]); plug_err=max(abs(plug_smin-U_BULK),abs(plug_smean-U_BULK),abs(plug_smax-U_BULK))
    if plug_err>5.0e-11: raise RuntimeError("constant-Q1 plug sample audit failed")
    Print(f"VMFL003_INITIAL_PIPE_VELOCITY status=PASS mode=plug Q1OwnedGlobal={int(mpi_sum(len(q1_g)))} BF2OwnedGlobal={int(mpi_sum(len(bf2_g)))} bf2InitMax={bf2_init_max:.3e}")
    first_cont=None; last_rel=float("inf"); last_wall=None; max_nut_ratio=0.0; converged=False
    for it in range(1,MAX_OUTER+1):
        Aphys,brhs,strain,nut=assemble_rans_lagged(mesh,V,U,uhat); wstats=wallplan.add_wall_operator(Aphys,U[2]); nut_min,nut_mean,nut_max=cell_average_stats(mesh,Q,nut,scale=NU); max_nut_ratio=max(max_nut_ratio,nut_max); last_wall=wstats
        Arel,delta,rAU,rst=finalise_simplec(Aphys,fixed); S,ns=build_schur(Bs,rAU); Axy=make_xy_clamped_matrix(Arel,wall_clamp)
        km_xy=configure_ksp(Axy,"f8_mxy_","momentum"); km_z=configure_ksp(Arel,"f8_mz_","momentum"); kp=configure_ksp(S,"f8_p_","pressure")
        u_init,u_its,u_drop=true_momentum_solve(Arel,Axy,Bs,p,delta,U,wall_clamp,pres0,brhs,km_xy,km_z)
        divv=continuity_vector(S,Bs,U,fixed_flux); cn=divv.norm(); first_cont=max(cn,1.0e-300) if first_cont is None else first_cont; last_rel=cn/first_cont
        all_met=max(float(last_rel),float(u_init[0]),float(u_init[1]),float(u_init[2]))<SIMPLE_TOL
        prhs=divv.copy(); prhs.scale(-1.0); pcorr=S.createVecRight(); pcorr.set(0.0); kp.setInitialGuessNonzero(False); kp.solve(prhs,pcorr); reason=int(kp.getConvergedReason())
        tmpq=S.createVecRight(); S.mult(pcorr,tmpq); pres=prhs.copy(); pres.axpy(-1.0,tmpq); ptrue=pres.norm()/max(prhs.norm(),1.0e-300)
        if reason<=0 or not math.isfinite(ptrue) or ptrue>1.05*P_RTOL: raise RuntimeError(f"pressure failed it={it} reason={reason} trueRel={ptrue}")
        p.axpy(ALPHA_P,pcorr)
        if it<=10 or it%10==0 or all_met:
            Print(f"VMFL003_SIMPLE it={it} relCont={last_rel:.12e} uInitRel=[{u_init[0]:.3e},{u_init[1]:.3e},{u_init[2]:.3e}] uIts=[{u_its[0]},{u_its[1]},{u_its[2]}] uDrop=[{u_drop[0]:.3e},{u_drop[1]:.3e},{u_drop[2]:.3e}] pIts={kp.getIterationNumber()} pReason={reason} pTrueRel={ptrue:.3e} pressureUpdateOnly=1 rAU=[{rst['rau_min']:.3e},{rst['rau_max']:.3e}] fWall={wstats['fwall']:.6e} yPlusMean={wstats['yplus_mean']:.3e} nuTRatioMean={nut_mean:.3e} nuTRatioMax={nut_max:.3e}")
        pres.destroy(); tmpq.destroy(); pcorr.destroy(); prhs.destroy(); divv.destroy(); [vv.destroy() for vv in brhs]; km_xy.destroy(); km_z.destroy(); kp.destroy(); Axy.destroy(); ns.destroy(); S.destroy(); rAU.destroy(); Arel.destroy(); Aphys.destroy()
        if all_met: converged=True; break
    if not converged: raise RuntimeError(f"did not reach production gate {SIMPLE_TOL} in {MAX_OUTER} iterations")
    A_final_diag,brhs_final_diag,strain_final_diag,nut_final_diag=assemble_rans_lagged(mesh,V,U,uhat); final_wall=wallplan.add_wall_operator(A_final_diag,U[2]); final_nut_min,final_nut_mean,final_nut_max=cell_average_stats(mesh,Q,nut_final_diag,scale=NU)
    [vv.destroy() for vv in brhs_final_diag]; A_final_diag.destroy(); last_wall=final_wall; max_nut_ratio=max(max_nut_ratio,final_nut_max)
    pfun=function_from_vec(Q,p,"VMFL003_pressure"); uf=[function_from_vec(V,U[d],f"VMFL003_U{d}") for d in range(3)]; area_in=assemble(1.0*ds_b(domain=mesh)); flux_in=assemble(uf[2]*(-FacetNormal(mesh)[2])*ds_b); flux_out=assemble(uf[2]*(FacetNormal(mesh)[2])*ds_t); target_flux=U_BULK*area_in; flux_err=(flux_out-target_flux)/target_flux
    zplanes,pbar,plane_counts=axial_pressure_planes(mesh,Q,p); elapsed=time.perf_counter()-t0
    final_meta={"iterations":it,"rel_cont":last_rel,"flux_err":float(flux_err),"yplus_mean":float(last_wall["yplus_mean"]),"fwall":float(last_wall["fwall"])}; post=write_benchmark_output(Path("output/vmfl003"),zplanes,pbar,final_meta)
    Print(f"VMFL003_RESULT converged=1 iterations={it} simpleRtol={SIMPLE_TOL:.3e} relCont={last_rel:.12e} freeCgInletFlux={flux_in:.12e} outletFlux={flux_out:.12e} targetFlux={target_flux:.12e} outletFluxRelError={flux_err:.12e} fWallSpalding={last_wall['fwall']:.10e} yPlusMean={last_wall['yplus_mean']:.10e} maxNuTRatioCellAvg={max_nut_ratio:.10e} pressureUpdateOnly=1 velocityCorrection=NONE wallSeconds={elapsed:.6f}")
    Print(f"VMFL003_BENCHMARK officialTargetPa={VMFL003_TARGET_DP_PA:.12e} firedrakePa={post['full_drop_pa']:.12e} officialErrorPct={post['target_err_pct']:.8f} developedF={post['dev_fd']:.10f}")
    benchmark_ok=abs(post["target_err_pct"])<=BENCHMARK_DP_TOL_PCT; Print(f"VMFL003_BENCHMARK_STATUS={'PASS' if benchmark_ok else 'FAIL'}")
    if not benchmark_ok: raise RuntimeError("VMFL003 pressure-drop regression failed")
    Print("VMFL003_STATUS=PASS")
