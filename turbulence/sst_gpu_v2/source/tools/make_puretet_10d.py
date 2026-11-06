#!/usr/bin/env python3
"""Build a genuine unstructured 3-D 10D cylinder with Gmsh, HXT0 arrays and HXT1 binary.
Requires: numpy, gmsh Python module. No extrusion-based prism subdivision.
"""
import argparse
import pathlib
import subprocess
import sys

import numpy as np

D = 0.004
L = 10 * D
R = D / 2
FACES = np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])


def exterior_faces(tet):
    triangles=tet[:,FACES].reshape(-1,3)
    key=np.sort(triangles,axis=1)
    unique,first,count=np.unique(key,axis=0,return_index=True,return_counts=True)
    if np.any(count>2):
        raise RuntimeError('non-manifold tetrahedral volume faces')
    return triangles[first[count==1]]


def write_hxt0(points,tet,out):
    nv=len(points)
    J=np.stack((points[tet[:,1]]-points[tet[:,0]],points[tet[:,2]]-points[tet[:,0]],points[tet[:,3]]-points[tet[:,0]]),axis=2)
    det=np.linalg.det(J)
    if np.any(abs(det)<1e-20):
        raise RuntimeError(f'{sum(abs(det)<1e-20)} zero volume tetrahedra')
    inverted=det<0
    tet[inverted,1],tet[inverted,2] = tet[inverted,2].copy(),tet[inverted,1].copy()
    det[inverted]*=-1
    surface=exterior_faces(tet)
    z=points[surface,2]
    tol=1e-8
    i=np.max(abs(z),axis=1)<tol
    o=np.max(abs(z-L),axis=1)<tol
    wall=~(i|o)
    if not i.any() or not o.any() or not wall.any():
        raise RuntimeError('not all inlet, outlet and wall face sets were found')
    area=lambda f: np.linalg.norm(np.cross(points[f[:,1]]-points[f[:,0]],points[f[:,2]]-points[f[:,0]]),axis=1).sum()/2
    ai,ao,aw=map(area,(surface[i],surface[o],surface[wall]))
    print('PURETET_MESH_AUDIT nodes=%d tets=%d wallFaces=%d inletFaces=%d outletFaces=%d minDet=%.12e maxDet=%.12e inletArea=%.12e outletArea=%.12e wallArea=%.12e cylinderVolume=%.12e targetVolume=%.12e status=PASS' % (
        nv,len(tet),wall.sum(),i.sum(),o.sum(),det.min(),det.max(),ai,ao,aw,det.sum()/6,np.pi*R*R*L))
    if not np.isclose(ai,ao,rtol=.03):
        raise RuntimeError('inlet and outlet areas differ by over 3 percent')
    if np.max(np.linalg.norm(points[:,:2],axis=1))>R+1e-8:
        raise RuntimeError('mesh escapes the pipe radius')
    E=np.empty((0,8),np.int32); Q=np.empty((0,4),np.int32)
    np.savez_compressed(out,points=points,hexes=E,tets=tet,interface_quads=Q,interface_triangles=np.empty((0,2,3),np.int32),
        wall_quads=Q,inlet_hex_quads=Q,inlet_tet_triangles=surface[i],
        outlet_hex_quads=Q,outlet_tet_triangles=surface[o])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--outdir',required=True)
    ap.add_argument('--edge',type=float,default=0.00035,help='Target tetrahedral edge length in metres')
    ap.add_argument('--algorithm',type=int,default=1,choices=[1,4,7,10],help='Gmsh 3D algorithm: 1=Delaunay, 10=HXT')
    args=ap.parse_args()
    if not 1e-4 <= args.edge <= 1e-3: raise ValueError('edge size must be between 0.1 and 1 mm')
    try: import gmsh
    except ImportError as e: raise SystemExit('Install the gmsh Python module first, e.g. python3 -m pip install gmsh') from e
    out=pathlib.Path(args.outdir).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal',1)
        gmsh.model.add('nodals_puretet_10D')
        gmsh.model.occ.addCylinder(0.,0.,0.,0.,0.,L,R)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber('Mesh.Algorithm3D',args.algorithm)
        gmsh.option.setNumber('Mesh.CharacteristicLengthMin',args.edge)
        gmsh.option.setNumber('Mesh.CharacteristicLengthMax',args.edge*1.20)
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',0)
        gmsh.option.setNumber('Mesh.Optimize',1)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(1)
        tags,coords,_=gmsh.model.mesh.getNodes()
        tags=np.asarray(tags,dtype=np.int64)
        xyz=np.asarray(coords,dtype=np.float64).reshape(-1,3)
        order=np.argsort(tags); tags=tags[order]; xyz=xyz[order]
        types,_,conns=gmsh.model.mesh.getElements(3)
        elements=[]
        for kind,conn in zip(types,conns):
            if kind!=4: raise RuntimeError(f'unsupported 3D element type {kind}, expected linear tetrahedron (4)')
            conn=np.asarray(conn,dtype=np.int64).reshape(-1,4)
            ix=np.searchsorted(tags,conn)
            if not np.all(tags[ix]==conn): raise RuntimeError('missing Gmsh mesh node')
            elements.append(ix.astype(np.int32))
        if not elements: raise RuntimeError('Gmsh produced no tetrahedra')
        tet=np.vstack(elements)
        gmsh.write(str(out/'puretet_10D.msh'))
    finally:
        gmsh.finalize()
    inp=out/'HXT0_puretet_10D.npz'
    write_hxt0(xyz,tet,inp)
    converter=pathlib.Path(__file__).resolve().parents[1]/'hybrid_hxt1'/'prepare_hxt1.py'
    subprocess.run([sys.executable,str(converter),'--input',str(inp),'--outdir',str(out)],check=True)
    print('PURETET_HXT1_BINARY=%s'% (out/'HXT1_mesh.bin'))

if __name__=='__main__':main()
