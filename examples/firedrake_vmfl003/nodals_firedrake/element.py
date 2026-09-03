"""Exact Q1+BF2 scalar element used by the hexahedral benchmark.

The six face bubbles are NOT ordinary tensor-Q2 face-node functions.

For a face normal to n with tangential coordinates t1,t2, the basis is

    Q1_endpoint(n) * Q2_midpoint(t1) * Q2_midpoint(t2)

so each mode is 1 at its own face centre, zero on the other five faces, and
equals 1/2 at the hex centre. Four vertical-face modes are implemented as a
small custom FIAT quadrilateral edge element and extruded with a z-midpoint
factor. The two horizontal-face modes use the Q2 cell bubble times P1 in z.

This custom FIAT/FInAT bridge is necessary because generic FIAT restriction of
Serendipity S2 is not supported by Firedrake's current FIAT compatibility
layer.
"""
import numpy as np
from firedrake import Mesh, ExtrudedMesh, FunctionSpace
from finat.ufl import FiniteElement, TensorProductElement, RestrictedElement
from finat.ufl.restrictedelement import RestrictedElement as UFLRestrictedElement
import finat.element_factory as finat_element_factory
import finat.restricted as finat_restricted
from finat.fiat_elements import ScalarFiatElement
from FIAT.finite_element import FiniteElement as FIATFiniteElement
from FIAT.dual_set import DualSet as FIATDualSet
from FIAT.functional import PointEvaluation as FIATPointEvaluation
from FIAT.polynomial_set import mis as fiat_mis
from ufl import interval, quadrilateral

from .config import (
    DZ_FINE, DZ_LONG, NZ_FINE, NZ_LONG, NZ,
    EXPECTED_CELLS, EXPECTED_SCALAR_DOF,
)

BF2_BASIS_CONTRACT = "Q2_mid_tangent_x_Q2_mid_tangent_x_Q1_linear_normal"

class EntityOnlyRestrictedElement(UFLRestrictedElement):
    """UFL wrapper for FInAT restriction to facet entities, excluding closure."""
    def __repr__(self):
        return (
            f"EntityOnlyRestrictedElement({repr(self._element)}, "
            f"{repr(self._restriction_domain)})"
        )

    def reconstruct(self, **kwargs):
        element = self._element.reconstruct(**kwargs)
        return EntityOnlyRestrictedElement(element, self._restriction_domain)

    def shortstr(self):
        return (
            f"EntityOnly(<{self._element.shortstr()}>|_"
            f"{self._restriction_domain})"
        )



class _FaceBubbleEdgeFIAT(FIATFiniteElement):
    """Four exact edge-bubble modes on a FIAT quadrilateral.

    Each edge mode is nodal at its edge midpoint and has the exact form

        Q1_linear(edge-normal) * Q2_mid(edge-tangent).

    This deliberately bypasses FIAT.RestrictedElement, whose generic
    implementation requires get_nodal_basis() and therefore cannot restrict
    FIAT.Serendipity.

    The element carries DOFs only on the four 1-D edge entities.  Hence, after
    extrusion with the z-midpoint P2 factor, these become the four shared
    vertical face-centre BF2 modes used by the benchmark element.
    """

    def __init__(self, full_serendipity):
        ref = full_serendipity.get_reference_element()
        verts = np.asarray(ref.get_vertices(), dtype=float)
        topo = ref.get_topology()
        if verts.ndim != 2 or verts.shape[1] != 2:
            raise RuntimeError(
                f"edge BF2 expected 2-D quadrilateral, got vertices {verts.shape}"
            )
        if len(topo.get(1, {})) != 4:
            raise RuntimeError(
                f"edge BF2 expected four edges, got {len(topo.get(1, {}))}"
            )

        self._bounds = [
            (float(np.min(verts[:, d])), float(np.max(verts[:, d])))
            for d in range(2)
        ]
        self._edge_geom = []

        # Empty entity maps everywhere except one scalar DOF per edge.
        entity_ids = {
            dim: {ent: [] for ent in ents}
            for dim, ents in topo.items()
        }
        nodes = []

        for newdof, edge in enumerate(sorted(topo[1])):
            raw = topo[1][edge]
            vids = []
            def _flat(x):
                if isinstance(x, (tuple, list, np.ndarray)):
                    for xx in x:
                        _flat(xx)
                else:
                    vids.append(int(x))
            _flat(raw)
            vids = sorted(set(vids))
            if len(vids) != 2:
                raise RuntimeError(
                    f"edge={edge} expected two vertices, got {vids}"
                )
            ev = verts[np.asarray(vids, dtype=int)]
            span = np.ptp(ev, axis=0)
            normal = np.where(span < 1.0e-13)[0]
            tangent = np.where(span >= 1.0e-13)[0]
            if len(normal) != 1 or len(tangent) != 1:
                raise RuntimeError(
                    f"edge geometry invalid edge={edge} span={span.tolist()}"
                )
            n = int(normal[0])
            t = int(tangent[0])
            lo_n, hi_n = self._bounds[n]
            const = float(ev[0, n])
            tol = 1.0e-12*max(1.0, abs(lo_n), abs(hi_n))
            if abs(const-lo_n) <= tol:
                side = -1  # selector=(hi-x)/h
            elif abs(const-hi_n) <= tol:
                side = +1  # selector=(x-lo)/h
            else:
                raise RuntimeError(
                    f"edge normal coordinate not endpoint edge={edge} const={const}"
                )

            midpoint = tuple(np.mean(ev, axis=0).tolist())
            nodes.append(FIATPointEvaluation(ref, midpoint))
            entity_ids[1][edge] = [newdof]
            self._edge_geom.append((n, t, side))

        # Trivial scalar permutations: there is one DOF on each edge, so
        # reversing an edge leaves the one-element local ordering unchanged.
        # Empty entities likewise have empty permutations.
        entity_permutations = {}
        for dim, ents in topo.items():
            entity_permutations[dim] = {}
            for ent in ents:
                if dim == 0:
                    entity_permutations[dim][ent] = {0: []}
                elif dim == 1:
                    entity_permutations[dim][ent] = {0: [0], 1: [0]}
                elif dim == 2:
                    # UFC quadrilateral has eight cell orientations.
                    entity_permutations[dim][ent] = {k: [] for k in range(8)}
                else:
                    entity_permutations[dim][ent] = {0: []}

        dual = FIATDualSet(
            nodes, ref, entity_ids,
            entity_permutations=entity_permutations
        )
        super().__init__(
            ref_el=ref, dual=dual, order=2, formdegree=0, mapping="affine"
        )

    @staticmethod
    def is_nodal():
        return True

    def degree(self):
        # Q1(normal)*Q2(tangent) has total polynomial degree three.
        return 3

    def value_shape(self):
        return ()

    def tabulate(self, order, points, entity=None):
        ref = self.ref_el
        dim = ref.get_spatial_dimension()
        if dim != 2:
            raise RuntimeError(f"edge BF2 bad spatial dim={dim}")

        if entity is None:
            entity = (dim, 0)
        edim, eid = entity
        transform = ref.get_entity_transform(edim, eid)
        pts = np.asarray(transform(points), dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape((-1, dim))
        npts = pts.shape[0]

        out = {}
        for deriv_order in range(order + 1):
            for alpha in fiat_mis(dim, deriv_order):
                vals = np.zeros((4, npts), dtype=float)
                for i, (n, t, side) in enumerate(self._edge_geom):
                    lo_n, hi_n = self._bounds[n]
                    lo_t, hi_t = self._bounds[t]
                    hn = hi_n - lo_n
                    ht = hi_t - lo_t
                    an = int(alpha[n])
                    at = int(alpha[t])

                    # Q1 endpoint selector and its derivatives.
                    if an == 0:
                        if side < 0:
                            sn = (hi_n - pts[:, n]) / hn
                        else:
                            sn = (pts[:, n] - lo_n) / hn
                    elif an == 1:
                        sn = np.full(
                            npts, (-1.0 if side < 0 else 1.0) / hn,
                            dtype=float
                        )
                    else:
                        sn = np.zeros(npts, dtype=float)

                    # Nodal Q2 midpoint bubble: 1 at tangent midpoint.
                    if at == 0:
                        bt = (
                            4.0*(pts[:, t]-lo_t)*(hi_t-pts[:, t])
                            /(ht*ht)
                        )
                    elif at == 1:
                        bt = (
                            4.0*(lo_t + hi_t - 2.0*pts[:, t])
                            /(ht*ht)
                        )
                    elif at == 2:
                        bt = np.full(npts, -8.0/(ht*ht), dtype=float)
                    else:
                        bt = np.zeros(npts, dtype=float)

                    vals[i, :] = sn * bt
                out[tuple(alpha)] = vals
        return out


@finat_element_factory.convert.register(EntityOnlyRestrictedElement)
def _convert_entity_only_restriction(element, **kwargs):
    base, deps = finat_element_factory._create_element(element._element, **kwargs)

    # FIAT.RestrictedElement cannot restrict FIAT.Serendipity because
    # that non-Ciarlet element intentionally has no get_nodal_basis().
    # Replace exactly the S2 facet-only case by our four-function exact
    # edge element on the same FIAT quadrilateral reference cell.
    raw = getattr(base, "_element", None)
    if (
        element.restriction_domain() == "facet"
        and raw is not None
        and type(raw).__name__ == "Serendipity"
        and int(raw.get_order()) == 2
    ):
        out = ScalarFiatElement(_FaceBubbleEdgeFIAT(raw))
        if out.space_dimension() != 4:
            raise RuntimeError(
                f"custom edge element dimension {out.space_dimension()} != 4"
            )
        return out, deps

    out = finat_restricted.restrict(
        base, element.restriction_domain(), take_closure=False
    )
    if out is finat_restricted.null_element:
        raise ValueError(
            f"Empty entity-only restriction: {element.restriction_domain()}"
        )
    return out, deps


def build_mesh_and_spaces():
    base = Mesh("mesh/pipe_ogrid_firedrake.msh", name="vmfl003_cross_section")
    heights = np.asarray(
        [DZ_FINE]*NZ_FINE + [DZ_LONG]*NZ_LONG, dtype=float
    )
    mesh = ExtrudedMesh(
        base,
        layers=NZ,
        layer_height=heights,
        extrusion_type="uniform",
        name="vmfl003_500D"
    )

    # Exact Q1+BF2 polynomial space on the direct quadrilateral/extruded cells.
    # The horizontal S2 edge midpoint functions are Q1 in the edge-normal
    # coordinate and Q2-midpoint in the edge-tangent coordinate.
    Q1_xy = FiniteElement("Q", quadrilateral, 1, variant="equispaced")
    S2_xy = FiniteElement("S", quadrilateral, 2)
    Q2_xy = FiniteElement("Q", quadrilateral, 2, variant="equispaced")

    # Exactly four edge-midpoint serendipity modes, no vertex closure.
    BF_edge_xy = EntityOnlyRestrictedElement(S2_xy, "facet")

    # Exactly one Q2 cell-centre bubble.
    BF_cell_xy = RestrictedElement(Q2_xy, "interior")

    P1_z = FiniteElement("CG", interval, 1, variant="equispaced")
    P2_z = FiniteElement("CG", interval, 2, variant="equispaced")
    P2_z_mid = RestrictedElement(P2_z, "interior")

    Q1_hex = TensorProductElement(Q1_xy, P1_z)
    BF_vertical = TensorProductElement(BF_edge_xy, P2_z_mid)
    BF_endfaces = TensorProductElement(BF_cell_xy, P1_z)
    Q1BF2 = Q1_hex + BF_vertical + BF_endfaces

    V = FunctionSpace(mesh, Q1BF2, name="scalar_Q1BF2")
    Q = FunctionSpace(mesh, "DG", 0, name="pressure_Q0")

    if V.cell_node_map().arity != 14:
        raise RuntimeError(f"Q1BF2 local arity changed: {V.cell_node_map().arity}")
    if Q.cell_node_map().arity != 1:
        raise RuntimeError(f"Q0 local arity changed: {Q.cell_node_map().arity}")
    if Q.dim() != EXPECTED_CELLS:
        raise RuntimeError(f"Q0 dim {Q.dim()} != {EXPECTED_CELLS}")
    if V.dim() != EXPECTED_SCALAR_DOF:
        raise RuntimeError(f"Q1BF2 dim {V.dim()} != {EXPECTED_SCALAR_DOF}")

    return mesh, V, Q
