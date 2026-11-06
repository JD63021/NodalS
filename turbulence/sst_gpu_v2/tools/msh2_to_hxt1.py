#!/usr/bin/env python3
"""General Gmsh ASCII MSH2.2 tetrahedra -> NodalS HXT1.
Required physical tags: inlet=1, outlet=2, wall=3, volume=4.
No assumptions about surface normals, duct shape, or global orientation.
"""
from __future__ import annotations
import argparse, collections, json, math, pathlib, shutil, subprocess, sys
import numpy as np

def read_msh2(mesh: pathlib.Path):
    with mesh.open('r', encoding='utf8', errors='strict') as f:
        def seek(mark):
            for line in f:
                if line.strip()==mark:return
            raise RuntimeError('Missing Gmsh block '+mark)
        seek('$MeshFormat');fmt=f.readline().split()
        if len(fmt)<2 or fmt[0]!='2.2' or fmt[1]!='0':
            raise RuntimeError('Expected ASCII MSH 2.2. Regenerate with gmsh -format msh2 -bin 0')
        seek('$Nodes');n=int(f.readline()); ids=np.empty(n,dtype=np.int64); pts=np.empty((n,3),dtype=np.float64)
        for k in range(n):
            x=f.readline().split()
            if len(x)!=4:raise RuntimeError('Bad Gmsh node record')
            ids[k]=int(x[0]);pts[k]=[float(v) for v in x[1:]]
        if not np.isfinite(pts).all() or len(set(ids))!=n:raise RuntimeError('Invalid/repeated node coordinates or IDs')
        id2local={int(s):i for i,s in enumerate(ids)}
        seek('$Elements');ne=int(f.readline()); tets=[]; tris={1:[],2:[],3:[]};groups=collections.Counter()
        for _ in range(ne):
            row=f.readline().split()
            if len(row)<4:raise RuntimeError('Bad element record')
            typ=int(row[1]);ntags=int(row[2]);tags=list(map(int,row[3:3+ntags]));vertices=row[3+ntags:]
            if ntags<1:raise RuntimeError('Missing physical tags on element')
            phys=tags[0];groups[(typ,phys)]+=1
            if typ==4:
                if phys!=4:raise RuntimeError('Unexpected volume physical tag '+str(phys))
                if len(vertices)!=4:raise RuntimeError('Tetrahedron must have exactly 4 nodes')
                tets.append([id2local[int(v)] for v in vertices])
            elif typ==2:
                if phys not in tris:raise RuntimeError('Unexpected triangle physical tag '+str(phys))
                if len(vertices)!=3:raise RuntimeError('Triangle must have exactly 3 nodes')
                tris[phys].append([id2local[int(v)] for v in vertices])
            elif typ in (1,15):pass  # line/point physical tagging irrelevant to 3D
            else:raise RuntimeError('Unexpected element type '+str(typ)+'; pure tet mesh required')
    tets=np.asarray(tets,dtype=np.int32).reshape(-1,4)
    if len(tets)<1 or any(not tris[k] for k in (1,2,3)):raise RuntimeError('Missing volume/inlet/outlet/wall elements')
    return pts,tets,{k:np.asarray(v,dtype=np.int32).reshape(-1,3) for k,v in tris.items()},dict((str(k),v) for k,v in groups.items())

def convert(mesh,out,writer):
    pts,tets,tris,groups=read_msh2(mesh)
    J=np.stack((pts[tets[:,1]]-pts[tets[:,0]],pts[tets[:,2]]-pts[tets[:,0]],pts[tets[:,3]]-pts[tets[:,0]]),axis=2)
    det=np.linalg.det(J)
    if not np.all(np.isfinite(det)) or np.any(np.abs(det)<1e-18):raise RuntimeError('Degenerate tetrahedra present')
    negative=det<0;tets[negative,2],tets[negative,3]=tets[negative,3].copy(),tets[negative,2].copy()
    det=np.abs(det)
    # Boundary completeness & non-manifold rejection
    face_loc=[(1,2,3),(0,3,2),(0,1,3),(0,2,1)]
    used=collections.Counter(tuple(sorted(int(v) for v in face)) for ix in face_loc for face in tets[:,ix])
    if any(c>2 for c in used.values()):raise RuntimeError('Non-manifold face (more than two cells)')
    exterior={face for face,c in used.items() if c==1}
    labeled=collections.Counter(tuple(sorted(int(v) for v in face)) for a in tris.values() for face in a)
    if any(c!=1 for c in labeled.values()):raise RuntimeError('Duplicate boundary triangle physical tag')
    if exterior!=set(labeled):
        raise RuntimeError(f'Missing or extraneous exterior faces: missing={len(exterior-set(labeled))}, extra={len(set(labeled)-exterior)}')
    if set(np.concatenate([tets.reshape(-1),*(v.reshape(-1) for v in tris.values())]))!=set(range(len(pts))):
        raise RuntimeError('Unused nodes; reject inconsistent mesh index set')
    # Geometrically general: inlet and outlet normals may point in ANY directions,
    # including the same direction for a 180-degree return bend. Validity and
    # completeness of physical boundary triangles have already been checked.
    def surface_area(tri):
        a=pts[tri[:,1]]-pts[tri[:,0]]
        b=pts[tri[:,2]]-pts[tri[:,0]]
        return float((np.linalg.norm(np.cross(a,b),axis=1)*0.5).sum())
    Ai=surface_area(tris[1]); Ao=surface_area(tris[2])
    if not (math.isfinite(Ai) and math.isfinite(Ao) and Ai>0 and Ao>0):
        raise RuntimeError('Invalid inlet/outlet surface area')
    out.mkdir(parents=True,exist_ok=True)
    hxt0=out/'HXT0.npz'
    np.savez_compressed(hxt0,points=pts,hexes=np.empty((0,8),dtype=np.int32),tets=tets,
        interface_quads=np.empty((0,4),dtype=np.int32),interface_triangles=np.empty((0,2,3),dtype=np.int32),
        wall_quads=np.empty((0,4),dtype=np.int32),inlet_hex_quads=np.empty((0,4),dtype=np.int32),
        inlet_tet_triangles=tris[1],outlet_hex_quads=np.empty((0,4),dtype=np.int32),
        outlet_tet_triangles=tris[2])
    print('NODALS_MSH_PREFLIGHT=PASS tets='+str(len(tets))+' points='+str(len(pts))+
          ' inletFaces='+str(len(tris[1]))+' outletFaces='+str(len(tris[2]))+
          ' wallFaces='+str(len(tris[3]))+
          ' minTetVolume='+str(float(det.min()/6.)),flush=True)
    subprocess.run([sys.executable,'-u',str(writer),'--input',str(hxt0),'--outdir',str(out)],check=True)
    d={'tetrahedra':len(tets),'vertices':len(pts),'inlet_faces':len(tris[1]),
       'outlet_faces':len(tris[2]),'wall_faces':len(tris[3]),'inlet_area':Ai,'outlet_area':Ao,
       'min_tet_volume':float(det.min()/6.),
       'tag_counts':groups,'hxt1_path':str((out/'HXT1_mesh.bin').resolve())}
    (out/'mesh_audit.json').write_text(json.dumps(d,indent=2)+'\n')
    print('NODALS_HXT1_CONVERSION=PASS path='+str(out/'HXT1_mesh.bin'),flush=True)
    return d

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--msh',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True)
    a=ap.parse_args();convert(a.msh.expanduser().resolve(),a.out.expanduser().resolve(),pathlib.Path(__file__).with_name('prepare_hxt1.py'))
if __name__=='__main__':main()
