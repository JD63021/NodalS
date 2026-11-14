#!/usr/bin/env python3
"""Postprocess CPU M4 tetra VTU WITHOUT requiring enriched solver coefficients.
Area-weighted inlet/outlet P0 pressure and wall-adjacent cell nuT proxy.
This is NOT the true wall νt or the true enriched wall traction:
those require in-solver reconstruction (see NODALS_CPU_M4_RANS_WALL_BUDGET).
Usage:
python3 experiments/cpu_sst_spalding_m4/POSTPROCESS_VTU_PRESSURE_NUT.py /path/to/flow_sst_cpu_it500.vtu [mean_molecular_wall_tau]
"""
import sys, math, xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np

if len(sys.argv) not in (2,3):
    raise SystemExit("Usage: POSTPROCESS_VTU_PRESSURE_NUT.py flow.vtu [molecular_wall_tau]")
root=ET.parse(sys.argv[1]).getroot()
piece=root.find(".//Piece")
if piece is None: raise ValueError("No UnstructuredGrid/Piece")
def array(section,name):
    if name=="points":
        node=piece.find("Points/DataArray")
    else:
        node=piece.find(f"{section}/DataArray[@Name='{name}']")
    if node is None: raise ValueError(f"Missing {section}/{name}")
    return np.fromstring(node.text,sep=" ")
pts=array("","points").reshape(-1,3)
tet=array("Cells","connectivity").astype(int).reshape(-1,4)
p=array("CellData","pressure")
nut=array("CellData","nut")
k=array("PointData","k")
omega=array("PointData","omega")
if len(p)!=len(tet) or len(nut)!=len(tet) or len(k)!=len(pts):
    raise ValueError("VTU array dimensions inconsistent")
facelist=defaultdict(list)
for cid,ids in enumerate(tet):
    for omit in range(4):
        face=tuple(sorted(int(ids[j]) for j in range(4) if j!=omit))
        facelist[face].append(cid)

z0=float(np.min(pts[:,2]));z1=float(np.max(pts[:,2]))
R=float(np.max(np.hypot(pts[:,0],pts[:,1])))
TOL=1e-8*max(1.,abs(z1))
A={"inlet":0.,"outlet":0.,"wall":0.}
psum={"inlet":0.,"outlet":0.}
nutarea={"inlet":0.,"outlet":0.,"wall":0.}
karea={"wall":0.};oarea={"wall":0.}
wall_vertices=set()
count={"inlet":0,"outlet":0,"wall":0}
for face,cells in facelist.items():
    if len(cells)!=1: continue
    ids=np.array(face)
    xyz=pts[ids]
    a=.5*np.linalg.norm(np.cross(xyz[1]-xyz[0],xyz[2]-xyz[0]))
    if np.all(abs(xyz[:,2]-z0)<TOL): patch="inlet"
    elif np.all(abs(xyz[:,2]-z1)<TOL): patch="outlet"
    else:patch="wall"
    cid=cells[0];A[patch]+=a;count[patch]+=1;nutarea[patch]+=a*nut[cid]
    if patch=="wall":
        wall_vertices.update(face)
        karea["wall"]+=a*float(np.mean(k[ids]))
        oarea["wall"]+=a*float(np.mean(omega[ids]))
    else:psum[patch]+=a*p[cid]
if any(not A[x]>0 for x in A): raise ValueError(f"Bad boundary classification: {A}")
nu=1.460734693878e-5
U=11.9290594798
D=.0889
L=z1-z0
dp=psum["inlet"]/A["inlet"]-psum["outlet"]/A["outlet"]
fric_equiv=2*D*dp/(L*U*U)
tau_equiv=dp*A["inlet"]/A["wall"]
print(f"VTU_DIAG cells={len(tet)} vertices={len(pts)} wallVertices={len(wall_vertices)} L={L:.8g} D={D:.8g}")
print(f"VTU_BOUNDARIES areas={A} counts={count} wallAreaSum={A['wall']:.12g}")
print(f"VTU_PRESSURE pinlet={psum['inlet']/A['inlet']:.12g} poutlet={psum['outlet']/A['outlet']:.12g} dpKinematic={dp:.12g} apparentDarcy={fric_equiv:.10g}")
print(f"VTU_SST_WALL_PROXY kWallArea={karea['wall']/A['wall']:.12g} omegaWallArea={oarea['wall']/A['wall']:.12g} nutAdjCellArea={nutarea['wall']/A['wall']:.12g} nutAdjCellNuRatio={nutarea['wall']/A['wall']/nu:.10g}")
print(f"VTU_SST_ENDCELL_PROXY nutIn={nutarea['inlet']/A['inlet']:.12g} nutOut={nutarea['outlet']/A['outlet']:.12g} nutOutNuRatio={nutarea['outlet']/A['outlet']/nu:.10g}")
print(f"VTU_AXIAL_SHEAR_SCALE tauFromNetPressureIgnoringMomentumFlux={tau_equiv:.12g} fPressure={fric_equiv:.10g}")
if len(sys.argv)==3:
    mol=float(sys.argv[2])
    print(f"VTU_SHEAR_COMPARISON molWallTauFromLiveLog={mol:.12g} tauNeededOverMolApprox={tau_equiv/mol:.8g} nuTAdjOverNu={nutarea['wall']/A['wall']/nu:.8g}")
print("VTU_DIAG_NOTE wall-nut is a nearwall-CELL-CENTRE proxy, not exact wall-nut. Physical enriched wall stress and momentum flux require a native solver run.")
