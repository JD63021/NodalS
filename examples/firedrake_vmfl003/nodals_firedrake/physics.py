"""Turbulence, wall function, exact BF2 trace evaluation, and momentum solve."""
import math
import numpy as np
from firedrake import (
    Constant, TrialFunction, TestFunction, FacetNormal, SpatialCoordinate,
    assemble, grad, dot, max_value, min_value, sqrt, inner, sym, as_vector,
)
from petsc4py import PETSc

from .config import (
    Print, R, NU, NQ, NZ, U_BULK, MIXLEN_SCALE,
    WALL_KAPPA, WALL_B, WALL_BETA_SCALE, WALL_SAMPLE_FRACTION,
    DXQ, DSBQ,
)
from .operators import (
    function_from_vec, vec_copy_from_cofunction, global_vec_sum,
    set_owned_values, set_fixed_rhs, _mpi_comm,
)

comm = PETSc.COMM_WORLD.tompi4py()

def mpi_sum(x):
    from mpi4py import MPI
    return comm.allreduce(x, op=MPI.SUM)

def mpi_min(x):
    from mpi4py import MPI
    return comm.allreduce(x, op=MPI.MIN)

def mpi_max(x):
    from mpi4py import MPI
    return comm.allreduce(x, op=MPI.MAX)


def _flatten_ints(obj):
    if isinstance(obj, (tuple, list, np.ndarray)):
        out = []
        for v in obj:
            out.extend(_flatten_ints(v))
        return out
    return [int(obj)]

def _scalar_finat_element(V, expected_dim):
    fe = V.finat_element
    if fe.space_dimension() == expected_dim:
        return fe
    if hasattr(fe, "base_element") and fe.base_element.space_dimension() == expected_dim:
        return fe.base_element
    if hasattr(fe, "_base_element") and fe._base_element.space_dimension() == expected_dim:
        return fe._base_element
    raise RuntimeError(
        f"cannot recover scalar FInAT element expectedDim={expected_dim} "
        f"got={fe.space_dimension()} type={type(fe).__name__}"
    )

def _reference_dof_metadata(fe, expected_dim):
    """Return reference-cell node centres and type for each scalar local basis."""
    if fe.space_dimension() != expected_dim:
        raise RuntimeError("reference metadata element dimension mismatch")

    cell = fe.cell
    verts = np.asarray(cell.get_vertices(), dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise RuntimeError(f"unexpected reference vertices shape={verts.shape}")

    topo = cell.get_topology()
    edofs = fe.entity_dofs()

    centres = np.full((expected_dim, 3), np.nan, dtype=float)
    kinds = np.full(expected_dim, -1, dtype=np.int32)
    entity_vertices = [None]*expected_dim

    for dim, ents in topo.items():
        dmap = edofs.get(dim, {})
        for ent, raw_vids in ents.items():
            vids = _flatten_ints(raw_vids)
            dofs = dmap.get(ent, [])
            if not dofs:
                continue
            centre = np.mean(verts[np.asarray(vids, dtype=int)], axis=0)
            nv = len(vids)
            if nv == 1:
                kind = 1
            elif nv == 4:
                kind = 2
            else:
                raise RuntimeError(
                    f"Q1BF2 unexpected dof-bearing entity with {nv} vertices "
                    f"dim={dim} ent={ent} dofs={dofs}"
                )
            for dof in dofs:
                dof = int(dof)
                centres[dof, :] = centre
                kinds[dof] = kind
                entity_vertices[dof] = tuple(vids)

    if np.any(~np.isfinite(centres)) or np.any(kinds < 0):
        bad = np.where(kinds < 0)[0].tolist()
        raise RuntimeError(f"reference metadata incomplete badDofs={bad}")

    nvert = int(np.count_nonzero(kinds == 1))
    nface = int(np.count_nonzero(kinds == 2))
    if expected_dim == 14 and (nvert, nface) != (8, 6):
        raise RuntimeError(f"Q1BF2 entity split changed vertex={nvert} face={nface}")
    if expected_dim == 8 and nvert != 8:
        raise RuntimeError(f"coordinate Q1 entity split changed vertex={nvert}")

    return verts, centres, kinds, entity_vertices, topo

def _lagrange1(x, node, lo, hi):
    tol = 1.0e-12*max(1.0, abs(lo), abs(hi))
    if abs(node-lo) <= tol:
        return (hi-x)/(hi-lo)
    if abs(node-hi) <= tol:
        return (x-lo)/(hi-lo)
    raise RuntimeError(f"Q1 node is not endpoint node={node} bounds=({lo},{hi})")

def _lagrange1_derivative(node, lo, hi):
    tol = 1.0e-12*max(1.0, abs(lo), abs(hi))
    if abs(node-lo) <= tol:
        return -1.0/(hi-lo)
    if abs(node-hi) <= tol:
        return 1.0/(hi-lo)
    raise RuntimeError(f"Q1 derivative node is not endpoint node={node}")

def _lagrange2(x, node, lo, hi):
    mid = 0.5*(lo + hi)
    nodes = (lo, mid, hi)
    idx = int(np.argmin([abs(node-v) for v in nodes]))
    if abs(node-nodes[idx]) > 1.0e-11*max(1.0, abs(lo), abs(hi)):
        raise RuntimeError(f"Q2 node not endpoint/midpoint node={node}")
    xi = nodes[idx]
    val = 1.0
    for j, xj in enumerate(nodes):
        if j != idx:
            val *= (x-xj)/(xi-xj)
    return val

def _basis_values(points, ref_nodes, kinds, bounds):
    """Exact Q1+BF2 reference basis values.

    Vertex modes are trilinear Q1. Face modes use Q1 in the face-normal
    coordinate and Q2 midpoint bubbles in both tangential coordinates.
    """
    points = np.asarray(points, dtype=float)
    out = np.ones((points.shape[0], len(kinds)), dtype=float)

    for a in range(len(kinds)):
        if kinds[a] == 1:
            for d in range(3):
                lo, hi = bounds[d]
                out[:, a] *= _lagrange1(points[:, d], ref_nodes[a, d], lo, hi)
            continue

        if kinds[a] != 2:
            raise RuntimeError(f"unexpected local basis kind={kinds[a]} dof={a}")

        normal_dims = []
        for d in range(3):
            lo, hi = bounds[d]
            mid = 0.5*(lo + hi)
            node = ref_nodes[a, d]
            tol = 1.0e-11*max(1.0, abs(lo), abs(hi))
            if abs(node-lo) <= tol or abs(node-hi) <= tol:
                normal_dims.append(d)
                out[:, a] *= _lagrange1(points[:, d], node, lo, hi)
            elif abs(node-mid) <= tol:
                out[:, a] *= _lagrange2(points[:, d], node, lo, hi)
            else:
                raise RuntimeError(
                    f"BF2 face-centre coordinate invalid dof={a} dim={d} "
                    f"node={node} bounds=({lo},{hi})"
                )

        if len(normal_dims) != 1:
            raise RuntimeError(
                f"BF2 face mode must have exactly one Q1-normal coordinate "
                f"dof={a} normalDims={normal_dims}"
            )

    return out

def audit_exact_bf2_basis(V):
    """Check that every BF2 mode equals 1/2 at the hex centre."""
    fe = _scalar_finat_element(V, 14)
    verts, ref_nodes, kinds, _, _ = _reference_dof_metadata(fe, 14)
    bounds = [(float(np.min(verts[:, d])), float(np.max(verts[:, d]))) for d in range(3)]
    centre = np.asarray([[0.5*(lo+hi) for lo, hi in bounds]], dtype=float)
    Bc = _basis_values(centre, ref_nodes, kinds, bounds)[0]
    face = Bc[kinds == 2]
    vertex = Bc[kinds == 1]
    face_err = float(np.max(np.abs(face - 0.5)))
    vertex_err = float(np.max(np.abs(vertex - 0.125)))
    if face_err > 1.0e-13 or vertex_err > 1.0e-13:
        raise RuntimeError(
            f"exact BF2 analytic contract failed faceCentreValues={face.tolist()} "
            f"faceErr={face_err} vertexErr={vertex_err}"
        )
    return face, face_err, vertex_err

def _q1_geometry_values(points, ref_nodes, bounds):
    points = np.asarray(points, dtype=float)
    phi = np.ones((points.shape[0], len(ref_nodes)), dtype=float)
    dphi = np.ones((points.shape[0], len(ref_nodes), 3), dtype=float)
    for a in range(len(ref_nodes)):
        one = []
        der = []
        for d in range(3):
            lo, hi = bounds[d]
            one.append(_lagrange1(points[:, d], ref_nodes[a, d], lo, hi))
            der.append(_lagrange1_derivative(ref_nodes[a, d], lo, hi))
        phi[:, a] = one[0]*one[1]*one[2]
        dphi[:, a, 0] = der[0]*one[1]*one[2]
        dphi[:, a, 1] = one[0]*der[1]*one[2]
        dphi[:, a, 2] = one[0]*one[1]*der[2]
    return phi, dphi

def _unique_faces_from_topology(ref_vertices, topo):
    faces = []
    seen = set()
    for dim, ents in topo.items():
        for ent, raw_vids in ents.items():
            vids = tuple(_flatten_ints(raw_vids))
            if len(vids) != 4:
                continue
            key = tuple(sorted(vids))
            if key in seen:
                continue
            seen.add(key)
            rc = ref_vertices[np.asarray(vids, dtype=int)]
            spans = np.ptp(rc, axis=0)
            fixed = np.where(spans < 1.0e-12)[0]
            if len(fixed) != 1:
                continue
            fd = int(fixed[0])
            fv = float(rc[0, fd])
            faces.append((key, fd, fv))
    if len(faces) != 6:
        raise RuntimeError(f"expected six reference faces, got {len(faces)}")
    return faces

class OffwallSpaldingPlan:
    """Q1+BF2 wall-test/offwall-sample cross-trace Spalding wall operator."""

    def __init__(self, mesh, V):
        self.mesh = mesh
        self.V = V
        vfe = _scalar_finat_element(V, 14)
        refverts, vref, vkinds, _, topo = _reference_dof_metadata(vfe, 14)
        self.refverts = refverts
        self.vref = vref
        self.vkinds = vkinds
        self.bounds = [(float(refverts[:, d].min()), float(refverts[:, d].max())) for d in range(3)]
        self.faces = _unique_faces_from_topology(refverts, topo)

        Vc = mesh.coordinates.function_space()
        cfe = _scalar_finat_element(Vc, 8)
        crefverts, cref, ckinds, _, _ = _reference_dof_metadata(cfe, 8)
        if np.max(np.abs(np.sort(crefverts, axis=0)-np.sort(refverts, axis=0))) > 1.0e-12:
            raise RuntimeError("velocity/coordinate reference cells disagree")
        self.cref = cref
        self.ckinds = ckinds
        self.coord_dof_to_ref_vertex = np.full(8, -1, dtype=np.int32)
        for a in range(8):
            dist = np.linalg.norm(refverts-cref[a], axis=1)
            vid = int(np.argmin(dist))
            if dist[vid] > 1.0e-12:
                raise RuntimeError("cannot associate coordinate Q1 dof with reference vertex")
            self.coord_dof_to_ref_vertex[a] = vid
        if len(set(self.coord_dof_to_ref_vertex.tolist())) != 8:
            raise RuntimeError("coordinate reference vertex association is not one-to-one")

        wall_subset = mesh.exterior_facets.subset(1)
        base_wall_all = np.asarray(wall_subset.indices, dtype=np.int64)
        base_wall = np.asarray(wall_subset.owned_indices, dtype=np.int64)
        gbase_visible = int(mpi_sum(len(base_wall_all)))
        gbase = int(mpi_sum(len(base_wall)))
        expected_base = 4*NQ
        if gbase != expected_base:
            raise RuntimeError(
                f"wall marker owned-topology mismatch baseWallFacetsOwned={gbase} "
                f"expected={expected_base} visibleWithHalo={gbase_visible}"
            )

        vmap = V.exterior_facet_node_map()
        cmap = Vc.exterior_facet_node_map()
        vvals = vmap.values_with_halo
        cvals = cmap.values_with_halo
        voff = np.asarray(vmap.offset, dtype=np.int64)
        coff = np.asarray(cmap.offset, dtype=np.int64)
        if voff.shape != (14,) or coff.shape != (8,):
            raise RuntimeError(f"unexpected extruded map offsets v={voff.shape} c={coff.shape}")
        coords = mesh.coordinates.dat.data_ro_with_halos
        lgmap = V.dof_dset.lgmap
        gx, gw = np.polynomial.legendre.leggauss(4)

        local_nodes=[]; global_nodes=[]; wall_basis=[]; sample_basis=[]
        qweights=[]; ydist=[]; geom_align=[]; geom_nz=[]
        local_face_count = 0
        for fi in base_wall:
            fi = int(fi)
            for layer in range(NZ):
                vnodes = np.asarray(vvals[fi], dtype=np.int64) + layer*voff
                cnodes = np.asarray(cvals[fi], dtype=np.int64) + layer*coff
                Xloc = np.asarray(coords[cnodes, :], dtype=float)
                Xref = np.empty((8, 3), dtype=float)
                for a in range(8):
                    Xref[self.coord_dof_to_ref_vertex[a], :] = Xloc[a, :]
                candidates=[]
                for vids, fd, fv in self.faces:
                    rr=np.hypot(Xref[np.asarray(vids),0],Xref[np.asarray(vids),1])
                    if np.max(np.abs(rr-R)) < 2.0e-10:
                        candidates.append((vids,fd,fv))
                if len(candidates)!=1:
                    raise RuntimeError(f"wall cell face identification failed layer={layer} candidates={len(candidates)}")
                vids, fd, fv = candidates[0]
                free_dims=[d for d in range(3) if d!=fd]
                qpts=[]; qwts_ref=[]; mapped=[]
                for d in free_dims:
                    lo,hi=self.bounds[d]
                    mapped.append((0.5*((hi-lo)*gx+(hi+lo)),0.5*(hi-lo)*gw))
                for i in range(4):
                    for j in range(4):
                        rp=np.zeros(3,dtype=float); rp[fd]=fv
                        rp[free_dims[0]]=mapped[0][0][i]; rp[free_dims[1]]=mapped[1][0][j]
                        qpts.append(rp); qwts_ref.append(mapped[0][1][i]*mapped[1][1][j])
                qpts=np.asarray(qpts,dtype=float); qwts_ref=np.asarray(qwts_ref,dtype=float)
                spts=qpts.copy(); lo,hi=self.bounds[fd]
                opposite=hi if abs(fv-lo)<abs(fv-hi) else lo
                spts[:,fd]=fv+WALL_SAMPLE_FRACTION*(opposite-fv)
                Bw=_basis_values(qpts,self.vref,self.vkinds,self.bounds)
                Bs=_basis_values(spts,self.vref,self.vkinds,self.bounds)
                q1mask=self.vkinds==1
                part_err=float(np.max(np.abs(np.sum(Bw[:,q1mask],axis=1)-1.0)))
                part_err=max(part_err,float(np.max(np.abs(np.sum(Bs[:,q1mask],axis=1)-1.0))))
                if part_err>5.0e-13:
                    raise RuntimeError(f"Q1 partition audit failed err={part_err}")
                gwall,dgw=_q1_geometry_values(qpts,self.cref,self.bounds)
                gsample,_=_q1_geometry_values(spts,self.cref,self.bounds)
                xw=gwall@Xloc; xs=gsample@Xloc
                J=np.einsum("qad,ai->qid",dgw,Xloc)
                t0=J[:,:,free_dims[0]]; t1=J[:,:,free_dims[1]]
                av=np.cross(t0,t1); amag=np.linalg.norm(av,axis=1)
                rr=np.hypot(xw[:,0],xw[:,1]); nr=np.column_stack((xw[:,0]/rr,xw[:,1]/rr,np.zeros_like(rr)))
                y=np.sum((xw-xs)*nr,axis=1)
                ng=av/amag[:,None]; align=np.abs(np.sum(ng*nr,axis=1)); nz=np.abs(ng[:,2])
                gids=np.asarray(lgmap.apply(vnodes.astype(PETSc.IntType)),dtype=PETSc.IntType)
                local_nodes.append(vnodes); global_nodes.append(gids); wall_basis.append(Bw); sample_basis.append(Bs)
                qweights.append(qwts_ref*amag); ydist.append(y); geom_align.append(align); geom_nz.append(nz)
                local_face_count += 1

        self.local_nodes=np.asarray(local_nodes,dtype=np.int64).reshape((-1,14))
        self.global_nodes=np.asarray(global_nodes,dtype=PETSc.IntType).reshape((-1,14))
        self.Bw=np.asarray(wall_basis,dtype=float).reshape((-1,16,14))
        self.Bs=np.asarray(sample_basis,dtype=float).reshape((-1,16,14))
        self.w=np.asarray(qweights,dtype=float).reshape((-1,16))
        self.y=np.asarray(ydist,dtype=float).reshape((-1,16))
        self.align=np.asarray(geom_align,dtype=float).reshape((-1,16))
        self.nz=np.asarray(geom_nz,dtype=float).reshape((-1,16))
        self.local_faces=local_face_count
        self.global_faces=int(mpi_sum(local_face_count))
        self.global_samples=int(mpi_sum(local_face_count*16))
        expected_faces=expected_base*NZ
        if self.global_faces!=expected_faces:
            raise RuntimeError(f"wall face count mismatch {self.global_faces} expected={expected_faces}")
        self.ymin=float(mpi_min(float(self.y.min()) if self.y.size else float("inf")))
        self.ymax=float(mpi_max(float(self.y.max()) if self.y.size else 0.0))
        local_area=float(self.w.sum()); self.area=float(mpi_sum(local_area))
        self.ymean=float(mpi_sum(float(np.sum(self.y*self.w)))/max(self.area,1.0e-300))
        self.min_align=float(mpi_min(float(self.align.min()) if self.align.size else 1.0))
        self.max_nz=float(mpi_max(float(self.nz.max()) if self.nz.size else 0.0))
        if self.min_align<0.95 or self.max_nz>1.0e-10:
            raise RuntimeError(f"wall geometry audit failed minAlign={self.min_align} maxNz={self.max_nz}")

    @staticmethod
    def _spalding_yplus(up):
        x=WALL_KAPPA*up
        with np.errstate(over="ignore",invalid="ignore"):
            rem=np.expm1(x)-x-0.5*x*x-(x*x*x)/6.0
        small=np.abs(x)<1.0e-3
        if np.any(small):
            xs=x[small]; x2=xs*xs; x4=x2*x2
            rem[small]=x4*(1.0/24.0+xs/120.0+x2/720.0+x2*xs/5040.0)
        return up+math.exp(-WALL_KAPPA*WALL_B)*rem

    @classmethod
    def _invert_spalding(cls,slip,y):
        U=np.abs(np.asarray(slip,dtype=float)); y=np.asarray(y,dtype=float); reY=y*U/NU
        uplus=np.zeros_like(U); utau=np.zeros_like(U); yplus=np.zeros_like(U); beta=NU/y
        fail=np.zeros_like(U,dtype=bool); active=reY>1.0e-14
        if np.any(active):
            r=reY[active]; lo=np.zeros_like(r); hi=np.maximum(1.0,np.sqrt(r)+1.0)
            def g(up): return up*cls._spalding_yplus(up)-r
            gh=g(hi)
            for _ in range(40):
                bad=(~np.isfinite(gh))|(gh<0.0)
                if not np.any(bad): break
                hi[bad]*=2.0; gh=g(hi)
            bad=(~np.isfinite(gh))|(gh<0.0)
            for _ in range(70):
                mid=0.5*(lo+hi); gm=g(mid); take_hi=gm>0.0
                hi[take_hi]=mid[take_hi]; lo[~take_hi]=mid[~take_hi]
            up=0.5*(lo+hi); Ua=U[active]; uta=Ua/np.maximum(up,1.0e-300)
            ypa=y[active]*uta/NU; be=uta*uta/np.maximum(Ua,1.0e-300)
            bad|=(~np.isfinite(up))|(up<=0.0); bad|=(~np.isfinite(uta))|(~np.isfinite(ypa)); bad|=(~np.isfinite(be))|(be<=0.0)
            idx=np.where(active); uplus[idx]=up; utau[idx]=uta; yplus[idx]=ypa; beta[idx]=be; fail[idx]=bad
        beta[fail]=NU/y[fail]; utau[fail]=0.0; yplus[fail]=0.0; uplus[fail]=0.0
        beta*=WALL_BETA_SCALE
        return utau,yplus,uplus,beta,fail

    def add_wall_operator(self,Aphys,Uz_vec):
        uzf=function_from_vec(self.V,Uz_vec,"wall_Ulag_axial"); udata=uzf.dat.data_ro_with_halos
        if self.local_faces:
            ucell=udata[self.local_nodes]; slip=np.einsum("fqa,fa->fq",self.Bs,ucell)
            utau,yplus,uplus,beta,fail=self._invert_spalding(slip,self.y)
            wallW=np.einsum("fq,fqa,fqb->fab",beta*self.w,self.Bw,self.Bs,optimize=True)
        else:
            slip=np.empty((0,16)); utau=yplus=uplus=beta=np.empty((0,16)); fail=np.empty((0,16),dtype=bool); wallW=np.empty((0,14,14))
        Awall=Aphys.copy(); Awall.zeroEntries()
        for f in range(self.local_faces):
            g=self.global_nodes[f]; Awall.setValues(g,g,wallW[f],addv=PETSc.InsertMode.ADD_VALUES)
        Awall.assemble(); wall_frob=float(Awall.norm(PETSc.NormType.FROBENIUS))
        Aphys.axpy(1.0,Awall,structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN); Aphys.assemble(); Awall.destroy()
        local_area=float(self.w.sum()); area=max(float(mpi_sum(local_area)),1.0e-300)
        def wmean(field): return float(mpi_sum(float(np.sum(field*self.w)))/area)
        def gmin(field): return float(mpi_min(float(field.min()) if field.size else float("inf")))
        def gmax(field): return float(mpi_max(float(field.max()) if field.size else 0.0))
        root_fail=int(mpi_sum(int(np.count_nonzero(fail))))
        return {"root_fail":root_fail,"beta_min":gmin(beta),"beta_mean":wmean(beta),"beta_max":gmax(beta),
                "yplus_min":gmin(yplus),"yplus_mean":wmean(yplus),"yplus_max":gmax(yplus),
                "slip_min":gmin(np.abs(slip)),"slip_mean":wmean(np.abs(slip)),"slip_max":gmax(np.abs(slip)),
                "utau_min":gmin(utau),"utau_mean":wmean(utau),"utau_max":gmax(utau),"utau2_mean":wmean(utau*utau),
                "fwall":8.0*wmean(utau*utau)/(U_BULK*U_BULK),"wall_frob":wall_frob}

def nikuradse_expressions(mesh,V,Uvecs):
    uf=[function_from_vec(V,Uvecs[d],f"Ulag{d}") for d in range(3)]
    U=as_vector(uf); S=sym(grad(U)); strain=sqrt(max_value(2.0*inner(S,S),0.0))
    x,y,z=SpatialCoordinate(mesh); rr=sqrt(x*x+y*y); eta=min_value(1.0,max_value(0.0,rr/R)); eta2=eta*eta
    ell=R*max_value(0.14-0.08*eta2-0.06*eta2*eta2,0.0); nut=Constant(MIXLEN_SCALE)*ell*ell*strain
    return uf,U,strain,ell,nut

def assemble_rans_lagged(mesh,V,Uvecs,uhat):
    uf,Uadv,strain,ell,nut=nikuradse_expressions(mesh,V,Uvecs); nu_eff=Constant(NU)+nut
    u=TrialFunction(V); v=TestFunction(V); n=FacetNormal(mesh)
    bn=uhat[0]*n[0]+uhat[1]*n[1]+uhat[2]*n[2]; inflow=max_value(0.0,-bn)
    a=(nu_eff*dot(grad(u),grad(v))+v*dot(Uadv,grad(u)))*DXQ + (inflow*v*u+nu_eff*(-v*dot(grad(u),n)+dot(grad(v),n)*u))*DSBQ
    A=assemble(a,mat_type="aij").petscmat
    brhs=[]
    for d in range(3):
        form=(inflow*v*uhat[d]+nu_eff*dot(grad(v),n)*uhat[d])*DSBQ
        brhs.append(vec_copy_from_cofunction(assemble(form)))
    return A,brhs,strain,nut

def cell_average_stats(mesh,Q,expr,scale=1.0):
    q=TestFunction(Q); vint=vec_copy_from_cofunction(assemble(q*DXQ)); eint=vec_copy_from_cofunction(assemble(expr*q*DXQ))
    va=vint.getArray(readonly=True); ea=eint.getArray(readonly=True); good=va>0.0; local=np.asarray(ea[good]/va[good]/scale,dtype=float)
    lmin=float(local.min()) if len(local) else float("inf"); lmax=float(local.max()) if len(local) else 0.0
    gmin=float(mpi_min(lmin)); gmax=float(mpi_max(lmax)); mean=global_vec_sum(eint)/(scale*max(global_vec_sum(vint),1.0e-300))
    vint.destroy(); eint.destroy(); return gmin,float(mean),gmax

def q1_bf2_owned_dofs(mesh,V,local_kinds):
    cmap=V.cell_node_map(); vals=np.asarray(cmap.values_with_halo,dtype=np.int64); off=np.asarray(cmap.offset,dtype=np.int64)
    q1pos=np.where(np.asarray(local_kinds)==1)[0]; bf2pos=np.where(np.asarray(local_kinds)==2)[0]
    owned_n=int(V.dof_dset.size); q1_local=[]; bf2_local=[]
    for layer in range(NZ):
        q1_local.append((vals[:,q1pos]+layer*off[q1pos]).ravel()); bf2_local.append((vals[:,bf2pos]+layer*off[bf2pos]).ravel())
    q1_local=np.unique(np.concatenate(q1_local)); bf2_local=np.unique(np.concatenate(bf2_local))
    q1_local=q1_local[(q1_local>=0)&(q1_local<owned_n)]; bf2_local=bf2_local[(bf2_local>=0)&(bf2_local<owned_n)]
    lg=V.dof_dset.lgmap; q1_g=np.unique(np.asarray(lg.apply(q1_local.astype(PETSc.IntType)),dtype=PETSc.IntType)); bf2_g=np.unique(np.asarray(lg.apply(bf2_local.astype(PETSc.IntType)),dtype=PETSc.IntType))
    if int(mpi_sum(len(q1_g)))!=14973 or int(mpi_sum(len(bf2_g)))!=40992:
        raise RuntimeError("Q1/BF2 owned topology mismatch")
    return q1_g,bf2_g

def initialise_constant_q1_plug(mesh,V,vec,local_kinds,value):
    q1_g,bf2_g=q1_bf2_owned_dofs(mesh,V,local_kinds); vec.set(0.0)
    if len(q1_g): vec.setValues(q1_g,np.full(len(q1_g),float(value),dtype=float),addv=PETSc.InsertMode.INSERT_VALUES)
    vec.assemble(); bf2_local_max=float(np.max(np.abs(vec.getValues(bf2_g)))) if len(bf2_g) else 0.0; bf2_max=float(mpi_max(bf2_local_max))
    if bf2_max!=0.0: raise RuntimeError(f"plug BF2 initialization is not zero max={bf2_max}")
    return q1_g,bf2_g,bf2_max

def wall_sample_stats(plan,Uz_vec):
    uzf=function_from_vec(plan.V,Uz_vec,"initial_plug_audit"); udata=uzf.dat.data_ro_with_halos
    if plan.local_faces:
        ucell=udata[plan.local_nodes]; slip=np.einsum("fqa,fa->fq",plan.Bs,ucell); local_min=float(slip.min()); local_max=float(slip.max()); local_w=float(np.sum(slip*plan.w))
    else:
        local_min=float("inf"); local_max=-float("inf"); local_w=0.0
    return float(mpi_min(local_min)),float(mpi_sum(local_w)/max(plan.area,1.0e-300)),float(mpi_max(local_max))

def make_xy_clamped_matrix(Arel,wall_nodes):
    Axy=Arel.copy(); Axy.zeroRowsColumns(np.asarray(wall_nodes,dtype=PETSc.IntType),diag=1.0); Axy.assemble(); return Axy

def true_momentum_solve(Arel,Axy,Bs,p,delta,U,wall_clamp,pres0,brhs,km_xy,km_z):
    u_init=[]; u_its=[]; u_drop=[]; Bt=Arel.createVecRight()
    for d in range(3):
        Acomp=Axy if d<2 else Arel; km=km_xy if d<2 else km_z; old=U[d].copy()
        if d<2: set_owned_values(old,wall_clamp,0.0); set_owned_values(U[d],wall_clamp,0.0)
        Bs[d].multTranspose(p,Bt); rhs=Acomp.createVecRight(); rhs.set(0.0); rhs.axpy(1.0,brhs[d]); rhs.axpy(1.0,Bt)
        ra=rhs.getArray(); oa=old.getArray(readonly=True); ra[:]+=delta*oa; del oa,ra
        if d<2: set_fixed_rhs(rhs,wall_clamp,pres0)
        else: rhs.assemble()
        rr=rhs.copy(); Acomp.mult(old,Bt); rr.axpy(-1.0,Bt); r0n=rr.norm(); bnorm=rhs.norm(); u_init.append(r0n/max(bnorm,1.0e-300)); rr.destroy()
        km.setInitialGuessNonzero(True); km.solve(rhs,U[d]); reason=int(km.getConvergedReason())
        if reason<=0: raise RuntimeError(f"momentum failed component={d} reason={reason} its={km.getIterationNumber()}")
        if d<2: set_owned_values(U[d],wall_clamp,0.0)
        r1=rhs.copy(); Acomp.mult(U[d],Bt); r1.axpy(-1.0,Bt); u_drop.append(r1.norm()/max(r0n,1.0e-300)); u_its.append(km.getIterationNumber())
        r1.destroy(); rhs.destroy(); old.destroy()
    Bt.destroy(); return u_init,u_its,u_drop
