"""PETSc assembly helpers and SIMPLEC algebra for the benchmark."""
import os
import numpy as np
from firedrake import (
    Function, TrialFunction, TestFunction, FacetNormal, Constant,
    assemble, grad, dot, max_value
)
from petsc4py import PETSc
from mpi4py import MPI

from .config import (
    Print, NU, U_BULK, DXQ, DSBQ,
    ALPHA_U, RAU_SCALE, SIMPLEC_BLEND, SIMPLEC_FLOOR_FRACTION,
)


def vec_copy_from_cofunction(f):
    with f.dat.vec_ro as v:
        out = v.duplicate()
        v.copy(out)
    return out

def function_from_vec(V, vec, name):
    f = Function(V, name=name)
    with f.dat.vec as fv:
        vec.copy(fv)
    return f

def _mpi_comm():
    return PETSc.COMM_WORLD.tompi4py()

def _new_space_vec(V):
    f = Function(V)
    with f.dat.vec_ro as v:
        out = v.duplicate()
    out.set(0.0)
    return out

def local_bc_nodes(*bcs):
    """Owned GLOBAL dof ids for BC nodes on this MPI rank."""
    if not bcs:
        return np.empty(0, dtype=PETSc.IntType)
    V = bcs[0].function_space()
    local = np.unique(np.concatenate([
        np.asarray(bc.nodes, dtype=PETSc.IntType) for bc in bcs
    ]))
    gids = np.asarray(V.dof_dset.lgmap.apply(local), dtype=PETSc.IntType)
    probe = _new_space_vec(V)
    rstart, rend = probe.getOwnershipRange()
    probe.destroy()
    gids = gids[(gids >= rstart) & (gids < rend)]
    return np.unique(gids).astype(PETSc.IntType)

def global_owned_count(local_ids):
    return int(_mpi_comm().allreduce(len(local_ids)))

def global_vec_sum(vec):
    """MPI-safe global sum of a distributed PETSc Vec."""
    local = float(np.asarray(vec.getArray(readonly=True)).sum())
    return float(_mpi_comm().allreduce(local, op=MPI.SUM))

def set_owned_values(vec, global_ids, value):
    if len(global_ids):
        vec.setValues(
            np.asarray(global_ids, dtype=PETSc.IntType),
            np.full(len(global_ids), float(value), dtype=float),
            addv=PETSc.InsertMode.INSERT_VALUES,
        )
    vec.assemble()

def assemble_scalar_diffusion(V, nu=NU, extra_bilinear=None):
    u = TrialFunction(V)
    v = TestFunction(V)
    a = nu*dot(grad(u), grad(v))*DXQ
    if extra_bilinear is not None:
        a = a + extra_bilinear(u, v)
    return assemble(a, mat_type="aij").petscmat

def assemble_B_components(mesh, V, Q, dg_trace=False):
    u = TrialFunction(V)
    q = TestFunction(Q)
    n = FacetNormal(mesh)
    Bs = []
    for d in range(3):
        form = q*u.dx(d)*DXQ
        if dg_trace:
            form = form - q*u*n[d]*DSBQ
        Bs.append(assemble(form, mat_type="aij").petscmat)
    return Bs

def assemble_dg_fixed_flux(mesh, Q, uhat):
    q = TestFunction(Q)
    n = FacetNormal(mesh)
    form = q*(uhat[0]*n[0] + uhat[1]*n[1] + uhat[2]*n[2])*DSBQ
    return vec_copy_from_cofunction(assemble(form))

def dg_inlet_bilinear(mesh, nu=NU, uhat=None):
    n = FacetNormal(mesh)
    if uhat is None:
        uhat = (Constant(0.0), Constant(0.0), Constant(U_BULK))
    bn = uhat[0]*n[0] + uhat[1]*n[1] + uhat[2]*n[2]
    inflow = max_value(0.0, -bn)

    def term(u, v):
        return (
            inflow*v*u
            + nu*(-v*dot(grad(u), n) + dot(grad(v), n)*u)
        )*DSBQ
    return term

def finalise_simplec(Aphys, fixed_nodes):
    """MPI free-velocity SIMPLEC operator with strong dofs eliminated."""
    Ared = Aphys.copy()
    Ared.zeroRowsColumns(np.asarray(fixed_nodes, dtype=PETSc.IntType), diag=1.0)
    Ared.assemble()

    ia, ja, av = Ared.getValuesCSR()
    rstart, rend = Ared.getOwnershipRange()
    nloc = rend - rstart
    fixed = np.zeros(nloc, dtype=bool)
    if len(fixed_nodes):
        loc = np.asarray(fixed_nodes, dtype=PETSc.IntType) - rstart
        loc = loc[(loc >= 0) & (loc < nloc)]
        fixed[loc] = True

    d0 = Ared.getDiagonal().getArray(readonly=True).copy()
    row_l1 = np.add.reduceat(np.abs(av), ia[:-1])
    free = ~fixed
    if np.any(~np.isfinite(row_l1[free])) or np.any(row_l1[free] <= 0.0):
        raise RuntimeError("nonfinite/nonpositive reduced physical row L1")

    rf = 1.0/ALPHA_U - 1.0
    delta = rf*row_l1
    delta[fixed] = 0.0

    Arel = Ared.copy()
    for iloc in np.nonzero(free)[0]:
        gid = rstart + int(iloc)
        Arel.setValue(gid, gid, float(delta[iloc]), addv=PETSc.InsertMode.ADD_VALUES)
    Arel.assemble()

    diag_rel = Arel.getDiagonal().getArray(readonly=True).copy()
    ia2, ja2, av2 = Arel.getValuesCSR()
    row_sum = np.add.reduceat(av2, ia2[:-1])
    threshold = SIMPLEC_FLOOR_FRACTION*diag_rel
    metric = (1.0-SIMPLEC_BLEND)*diag_rel + SIMPLEC_BLEND*row_sum
    bad = (~np.isfinite(metric)) | (metric <= threshold)
    fallback_local = int(np.count_nonzero(bad & free))
    metric[bad] = diag_rel[bad]
    if np.any((metric[free] <= 0.0) | (~np.isfinite(metric[free]))):
        raise RuntimeError("SIMPLEC metric positive guard failed")

    rAU = Ared.createVecRight()
    ra = rAU.getArray()
    ra[:] = 0.0
    ra[free] = RAU_SCALE/metric[free]
    del ra

    comm = _mpi_comm()
    gmin = lambda v: float(comm.allreduce(float(v), op=MPI.MIN))
    gmax = lambda v: float(comm.allreduce(float(v), op=MPI.MAX))
    rarr = rAU.getArray(readonly=True)
    stats = {
        "phys_diag_min": gmin(np.min(d0[free])),
        "phys_diag_max": gmax(np.max(d0[free])),
        "rel_diag_min": gmin(np.min(diag_rel[free])),
        "rel_diag_max": gmax(np.max(diag_rel[free])),
        "metric_min": gmin(np.min(metric[free])),
        "metric_max": gmax(np.max(metric[free])),
        "rau_min": gmin(np.min(rarr[free])),
        "rau_max": gmax(np.max(rarr[free])),
        "fallback_count": int(comm.allreduce(fallback_local)),
        "free_dofs": int(comm.allreduce(int(np.count_nonzero(free)))),
        "fixed_dofs": int(comm.allreduce(int(np.count_nonzero(fixed)))),
    }
    Ared.destroy()
    return Arel, delta, rAU, stats

def build_schur(Bs, rAU):
    S = None
    for B in Bs:
        BD = B.copy()
        BD.diagonalScale(R=rAU)
        Sd = BD.matTransposeMult(B)
        if S is None:
            S = Sd
        else:
            S.axpy(
                1.0, Sd,
                structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN
            )
            Sd.destroy()
        BD.destroy()
    S.assemble()
    try:
        S.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        S.setOption(PETSc.Mat.Option.SPD, True)
    except Exception:
        pass
    ns = PETSc.NullSpace().create(constant=True, comm=S.comm)
    # With a natural/open outlet this Schur operator is not singular.
    # Use the constant only as a GAMG near-nullspace hint.
    S.setNearNullSpace(ns)
    return S, ns

def configure_ksp(A, prefix, kind="momentum"):
    opts = PETSc.Options()
    mpi_size = PETSc.COMM_WORLD.getSize()

    if kind == "pressure":
        p_rtol = float(os.environ.get("P_RTOL", "0.5"))
        p_petsc_atol = float(os.environ.get("P_PETSC_ATOL", "1e-50"))
        p_max = int(os.environ.get("P_MAX_ITS", "300"))
        p_level_its = int(os.environ.get("P_MG_LEVEL_ITS", "6"))
        settings = {
            "ksp_type": "cg",
            "ksp_rtol": f"{p_rtol:.17g}",
            "ksp_atol": f"{p_petsc_atol:.17g}",
            "ksp_divtol": "1e8",
            "ksp_max_it": str(p_max),
            "ksp_norm_type": "unpreconditioned",
            "pc_type": "gamg",
            "pc_gamg_use_sa_esteig": "true",
            "mg_levels_ksp_type": "chebyshev",
            "mg_levels_ksp_max_it": str(p_level_its),
            "mg_levels_ksp_norm_type": "none",
            "mg_levels_pc_type": "jacobi",
            "mg_coarse_ksp_type": "preonly",
            "mg_coarse_pc_type": "lu",
        }
        pc_desc = (
            f"PCG+GAMG(Chebyshev{p_level_its}+Jacobi,coarseLU)"
            f"[rtol={p_rtol:g},atol={p_petsc_atol:g}]"
        )
    else:
        m_rtol = float(os.environ.get("M_RTOL", "0.5"))
        m_petsc_atol = float(os.environ.get("M_PETSC_ATOL", "1e-50"))
        m_max = int(os.environ.get("M_MAX_ITS", "20000"))
        if mpi_size == 1:
            settings = {
                "ksp_type": "fgmres",
                "ksp_rtol": f"{m_rtol:.17g}",
                "ksp_atol": f"{m_petsc_atol:.17g}",
                "ksp_divtol": "1e8",
                "ksp_max_it": str(m_max),
                "ksp_gmres_restart": "40",
                "ksp_pc_side": "right",
                "ksp_norm_type": "unpreconditioned",
                "ksp_converged_use_initial_residual_norm": "true",
                "pc_type": "gamg",
                "pc_gamg_use_sa_esteig": "true",
                "mg_levels_ksp_type": "chebyshev",
                "mg_levels_ksp_max_it": "2",
                "mg_levels_ksp_norm_type": "none",
                "mg_levels_pc_type": "jacobi",
            }
            pc_desc = (
                "FGMRES+GAMG(Chebyshev2+Jacobi)"
                f"[relativeDrop={m_rtol:g}]"
            )
        else:
            settings = {
                "ksp_type": "fgmres",
                "ksp_rtol": f"{m_rtol:.17g}",
                "ksp_atol": f"{m_petsc_atol:.17g}",
                "ksp_divtol": "1e8",
                "ksp_max_it": str(m_max),
                "ksp_gmres_restart": "40",
                "ksp_pc_side": "right",
                "ksp_norm_type": "unpreconditioned",
                "ksp_converged_use_initial_residual_norm": "true",
                "pc_type": "bjacobi",
                "sub_ksp_type": "preonly",
                "sub_pc_type": "ilu",
                "sub_pc_factor_levels": "0",
            }
            pc_desc = (
                "FGMRES+BJacobi(local ILU0)"
                f"[relativeDrop={m_rtol:g}]"
            )

    for k, val in settings.items():
        opts[prefix+k] = val

    ksp_setup_verbose = os.environ.get("KSP_SETUP_VERBOSE", "0") == "1"
    if ksp_setup_verbose:
        Print(
            f"KSP_SETUP_START prefix={prefix} kind={kind} mpiSize={mpi_size} "
            f"pc={pc_desc} size={A.getSize()[0]}x{A.getSize()[1]}"
        )
    ksp = PETSc.KSP().create(comm=A.comm)
    ksp.setOptionsPrefix(prefix)
    ksp.setOperators(A)
    ksp.setFromOptions()
    ksp.setUp()
    if ksp_setup_verbose:
        Print(
            f"KSP_SETUP_DONE prefix={prefix} kind={kind} "
            f"ksp={ksp.getType()} pc={ksp.getPC().getType()} "
            f"rtol={ksp.getTolerances()[0]:.3e} atol={ksp.getTolerances()[1]:.3e}"
        )
    return ksp

def set_fixed_rhs(rhs, fixed_nodes, prescribed):
    if len(fixed_nodes):
        idx = np.asarray(fixed_nodes, dtype=PETSc.IntType)
        vals = prescribed.getValues(idx)
        rhs.setValues(idx, vals, addv=PETSc.InsertMode.INSERT_VALUES)
    rhs.assemble()
