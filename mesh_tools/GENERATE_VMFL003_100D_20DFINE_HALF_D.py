#!/usr/bin/env python3
import sys
from pathlib import Path
from collections import defaultdict

if len(sys.argv) != 3:
    raise SystemExit("usage: GENERATE_VMFL003_100D_20DFINE_HALF_D.py source20D.msh output100D.msh")

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

D = 0.004
L20 = 20.0 * D
N_SOURCE_SLABS = 360
N_DOWNSTREAM = 160
DZ_DOWNSTREAM = 0.5 * D
EXPECTED_TOTAL_SLABS = N_SOURCE_SLABS + N_DOWNSTREAM

lines = src.read_text().splitlines()

def section(name):
    begin = "$" + name
    end = "$End" + name
    try:
        i = lines.index(begin)
        j = lines.index(end, i+1)
    except ValueError:
        return None
    return lines[i+1:j]

fmt = section("MeshFormat")
if not fmt:
    raise RuntimeError("Missing $MeshFormat")
version = fmt[0].split()[0]
if not version.startswith("2."):
    raise RuntimeError(f"Source mesh is MSH version {version}; expected MSH2")

node_sec = section("Nodes")
if not node_sec:
    raise RuntimeError("Missing $Nodes")
nn = int(node_sec[0])
nodes = {}
for line in node_sec[1:1+nn]:
    p = line.split()
    nodes[int(p[0])] = (float(p[1]), float(p[2]), float(p[3]))

elem_sec = section("Elements")
if not elem_sec:
    raise RuntimeError("Missing $Elements")
ne = int(elem_sec[0])
tris, tets = [], []
for line in elem_sec[1:1+ne]:
    p = line.split()
    etype = int(p[1])
    ntags = int(p[2])
    tags = list(map(int,p[3:3+ntags]))
    conn = list(map(int,p[3+ntags:]))
    if etype == 2 and len(conn) == 3:
        tris.append((tags,conn))
    elif etype == 4 and len(conn) == 4:
        tets.append((tags,conn))

print(f"SOURCE nodes={len(nodes)} tris={len(tris)} tets={len(tets)}")

def r12(x):
    return round(x,12)

zkeys = sorted(set(r12(p[2]) for p in nodes.values()))
if len(zkeys) != 361:
    raise RuntimeError(f"Expected 361 axial planes in original 20D mesh, found {len(zkeys)}")

z_to_plane = {z:i for i,z in enumerate(zkeys)}
plane_nodes = defaultdict(list)
plane_z_actual = {}
for nid,(x,y,z) in nodes.items():
    k = z_to_plane[r12(z)]
    plane_nodes[k].append(nid)
    plane_z_actual.setdefault(k,z)
    if abs(plane_z_actual[k]-z) > 1e-14:
        raise RuntimeError(f"Plane {k} has inconsistent z")

source_z = [plane_z_actual[k] for k in range(len(zkeys))]
if abs(source_z[0]) > 1e-13:
    raise RuntimeError(f"Expected source inlet at z=0, got {source_z[0]}")
if abs(source_z[-1]-L20) > 1e-12:
    raise RuntimeError(f"Expected source outlet at 20D={L20}, got {source_z[-1]}")

nper = len(plane_nodes[0])
if nper != 451:
    raise RuntimeError(f"Expected 451 cross-section points, got {nper}")
for k in range(len(source_z)):
    if len(plane_nodes[k]) != nper:
        raise RuntimeError(f"Plane {k}: expected {nper} points, got {len(plane_nodes[k])}")

base = sorted(
    plane_nodes[0],
    key=lambda nid:(r12(nodes[nid][0]),r12(nodes[nid][1]))
)
base_xy = [(nodes[nid][0],nodes[nid][1]) for nid in base]
xy_to_local = {(r12(x),r12(y)):i for i,(x,y) in enumerate(base_xy)}

node_plane_local = {}
for k in range(len(source_z)):
    seen = set()
    for nid in plane_nodes[k]:
        x,y,z = nodes[nid]
        key = (r12(x),r12(y))
        if key not in xy_to_local:
            raise RuntimeError(f"Cross-section mismatch plane={k} node={nid} xy={key}")
        j = xy_to_local[key]
        if j in seen:
            raise RuntimeError(f"Duplicate radial point plane={k} local={j}")
        seen.add(j)
        node_plane_local[nid] = (k,j)
    if len(seen) != nper:
        raise RuntimeError(f"Plane {k}: incomplete cross-section")

print(f"CROSS_SECTION points={nper}")

tet_templates = []
for tags,conn in tets:
    pl = [node_plane_local[n][0] for n in conn]
    pmin,pmax = min(pl),max(pl)
    if pmax-pmin != 1:
        raise RuntimeError(f"Tet spans unexpected planes: {pl}")
    if pmin == 0:
        tet_templates.append(
            tuple((node_plane_local[n][0]-pmin,node_plane_local[n][1]) for n in conn)
        )
if len(tet_templates) != 2556:
    raise RuntimeError(f"Expected 2556 tets per slab, got {len(tet_templates)}")
print(f"TET_TEMPLATE tetsPerSlab={len(tet_templates)}")

inlet_templates, outlet_templates, wall_templates = [], [], []
last_source_plane = len(source_z)-1
for tags,conn in tris:
    pj = [node_plane_local[n] for n in conn]
    planes = [x[0] for x in pj]
    if all(p == 0 for p in planes):
        inlet_templates.append(tuple(j for p,j in pj))
    elif all(p == last_source_plane for p in planes):
        outlet_templates.append(tuple(j for p,j in pj))
    else:
        pmin,pmax = min(planes),max(planes)
        if pmax-pmin != 1:
            raise RuntimeError(f"Unexpected surface triangle plane span: {planes}")
        if pmin == 0:
            wall_templates.append(tuple((p-pmin,j) for p,j in pj))

if len(inlet_templates) != 852:
    raise RuntimeError(f"Expected 852 inlet triangles, got {len(inlet_templates)}")
if len(outlet_templates) != 852:
    raise RuntimeError(f"Expected 852 outlet triangles, got {len(outlet_templates)}")
if len(wall_templates) != 96:
    raise RuntimeError(f"Expected 96 wall triangles/slab, got {len(wall_templates)}")
print(
    f"SURFACE_TEMPLATE inletTris={len(inlet_templates)} "
    f"outletTris={len(outlet_templates)} wallTrisPerSlab={len(wall_templates)}"
)

# Keep 0-20D exactly from the original source, append 160 slabs of length 0.5D.
znew = list(source_z)
for i in range(1,N_DOWNSTREAM+1):
    znew.append(L20 + i*DZ_DOWNSTREAM)

nslab = len(znew)-1
if nslab != EXPECTED_TOTAL_SLABS:
    raise RuntimeError(f"Expected {EXPECTED_TOTAL_SLABS} slabs, got {nslab}")
if abs(znew[360]-20.0*D) > 1e-12:
    raise RuntimeError(f"20D interface mismatch: {znew[360]}")
if abs(znew[-1]-100.0*D) > 1e-12:
    raise RuntimeError(f"100D endpoint mismatch: {znew[-1]}")

source_dz = source_z[1]-source_z[0]
print("AXIAL_100D_20DFINE_HALF_D")
print(f"  D                         = {D:.16e}")
print(f"  source_0_20D_slabs        = {N_SOURCE_SLABS}")
print(f"  source_dz_over_D          = {source_dz/D:.12e}")
print(f"  downstream_20_100D_slabs  = {N_DOWNSTREAM}")
print(f"  downstream_dz_over_D      = {DZ_DOWNSTREAM/D:.12e}")
print(f"  spacing_jump_at_20D       = {DZ_DOWNSTREAM/source_dz:.12e}")
print(f"  total_slabs               = {nslab}")
print(f"  total_planes              = {len(znew)}")
print(f"  z_20D                     = {znew[360]:.16e}")
print(f"  z_100D                    = {znew[-1]:.16e}")

def nid(k,j):
    return k*nper + j + 1

new_nodes = []
for k,z in enumerate(znew):
    for j,(x,y) in enumerate(base_xy):
        new_nodes.append((nid(k,j),x,y,z))

expected_nodes = 451*521
if len(new_nodes) != expected_nodes:
    raise RuntimeError(f"Expected {expected_nodes} nodes, got {len(new_nodes)}")

TAG_INLET,TAG_WALL,TAG_OUTLET,TAG_FLUID = 1,2,3,4
surface_elements, volume_elements = [], []

for tri in inlet_templates:
    surface_elements.append((TAG_INLET,[nid(0,j) for j in tri]))
for k in range(nslab):
    for tri in wall_templates:
        surface_elements.append((TAG_WALL,[nid(k+off,j) for off,j in tri]))
for tri in outlet_templates:
    surface_elements.append((TAG_OUTLET,[nid(nslab,j) for j in tri]))
for k in range(nslab):
    for tet in tet_templates:
        volume_elements.append((TAG_FLUID,[nid(k+off,j) for off,j in tet]))

expected_tets = 2556*nslab
if len(volume_elements) != expected_tets:
    raise RuntimeError(f"Expected {expected_tets} tets, got {len(volume_elements)}")

print("NEW_MESH_COUNTS")
print(f"  points            = {len(new_nodes)}")
print(f"  inletTriangles    = {len(inlet_templates)}")
print(f"  wallTriangles     = {len(wall_templates)*nslab}")
print(f"  outletTriangles   = {len(outlet_templates)}")
print(f"  tetrahedra        = {len(volume_elements)}")
print(f"  expectedTetCount  = {expected_tets}")

with dst.open("w") as f:
    f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
    f.write("$PhysicalNames\n4\n")
    f.write(f'2 {TAG_INLET} "inlet"\n')
    f.write(f'2 {TAG_WALL} "wall"\n')
    f.write(f'2 {TAG_OUTLET} "outlet"\n')
    f.write(f'3 {TAG_FLUID} "fluid"\n')
    f.write("$EndPhysicalNames\n")
    f.write("$Nodes\n")
    f.write(f"{len(new_nodes)}\n")
    for i,x,y,z in new_nodes:
        f.write(f"{i} {x:.16e} {y:.16e} {z:.16e}\n")
    f.write("$EndNodes\n")
    total_elements = len(surface_elements)+len(volume_elements)
    f.write("$Elements\n")
    f.write(f"{total_elements}\n")
    eid = 1
    for physical,conn in surface_elements:
        f.write(f"{eid} 2 2 {physical} {physical} {conn[0]} {conn[1]} {conn[2]}\n")
        eid += 1
    for physical,conn in volume_elements:
        f.write(f"{eid} 4 2 {physical} {physical} {conn[0]} {conn[1]} {conn[2]} {conn[3]}\n")
        eid += 1
    f.write("$EndElements\n")

print(f"MSH_WRITTEN={dst}")
print(f"MSH_POINTS={len(new_nodes)}")
print(f"MSH_TETS={len(volume_elements)}")
print(f"L={znew[-1]:.16e}")
print(f"L_OVER_D={znew[-1]/D:.12f}")
print("VMFL003_100D_20DFINE_HALF_D_MESH_GENERATION=PASS")
