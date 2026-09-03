#!/usr/bin/env python3
from pathlib import Path
import sys

# Allow this script to be executed directly as:
#     python3 tools/smoke_bf2.py
# by adding the repository root (the parent of tools/) to sys.path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from petsc4py import PETSc
from nodals_firedrake.element import build_mesh_and_spaces
from nodals_firedrake.physics import audit_exact_bf2_basis

mesh, V, Q = build_mesh_and_spaces()
face, ferr, q1err = audit_exact_bf2_basis(V)

if V.cell_node_map().arity != 14:
    raise RuntimeError(f"local scalar arity={V.cell_node_map().arity}, expected 14")
if V.dim() != 55965:
    raise RuntimeError(f"global scalar dim={V.dim()}, expected 55965")
if Q.dim() != 13056:
    raise RuntimeError(f"pressure dim={Q.dim()}, expected 13056")

PETSc.Sys.Print(
    "BF2_SMOKE_STATUS=PASS "
    f"localScalar=14 globalScalar={V.dim()} pressure={Q.dim()} "
    f"cellCenterFaceValues=[{','.join(f'{x:.6f}' for x in face)}] "
    f"maxFaceError={ferr:.3e} q1CenterError={q1err:.3e}"
)
