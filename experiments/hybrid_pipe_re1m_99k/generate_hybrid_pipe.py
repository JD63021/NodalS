#!/usr/bin/env python3
"""Create the 99,216-cell single-HEX-ring 50D pipe as HXT0.npz.

The existing NodalS tools/prepare_hxt1.py converts this to HXT1_mesh.bin
and audits the HEX quad <-> two TET triangle interface. Requires numpy scipy.
"""
from __future__ import annotations
import argparse, json, math
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay


def disk_nodes(radius, ntheta, spacing):
    ang=2*np.pi*np.arange(ntheta)/ntheta
    ring=np.column_stack((radius*np.cos(ang),radius*np.sin(ang)))
    dy=spacing*np.sqrt(3)/2
    pts=[]
    for row in range(-math.ceil(radius/dy),math.ceil(radius/dy)+1):
        y=row*dy
        for col in range(-math.ceil(radius/spacing)-1,math.ceil(radius/spacing)+2):
            x=(col+0.5*(row & 1))*spacing
            if x*x+y*y<(radius-0.62*spacing)**2: pts.append((x,y))
    uv=np.concatenate((ring,np.asarray(pts,dtype=float).reshape(-1,2)))
    tri=Delaunay(uv).simplices.astype(np.int32)
    xy=uv[tri]
    cross=(xy[:,1,0]-xy[:,0,0])*(xy[:,2,1]-xy[:,0,1])-(xy[:,1,1]-xy[:,0,1])*(xy[:,2,0]-xy[:,0,0])
    tri[cross<0,1],tri[cross<0,2]=tri[cross<0,2].copy(),tri[cross<0,1].copy()
    return uv,tri


def choose_disk(radius,ntheta,n_z,target):
    best=None
    for spacing in np.linspace(.029,.061,105):
        uv,tri=disk_nodes(radius,ntheta,float(spacing))
        n=n_z*(len(tri)*3+ntheta)
        if best is None or abs(n-target)<abs(best[0]-target):
            best=(n,spacing,uv,tri)
    return best


def make_mesh(D,length_d,nu,re,ntheta,n_z,target,yplus):
    radius=D/2
    ubulk=re*nu/D
    f=.0117
    for _ in range(60):
        fnew=(-2*math.log10(2.51/(re*math.sqrt(f))))**-2
        if abs(fnew-f)<1e-14: break
        f=fnew
    utau=ubulk*math.sqrt(f/8)
    depth=2*yplus*nu/utau
    ri=radius-depth
    if not 0<ri<radius: raise ValueError('invalid wall ring depth')
    n,spacing,uv,tri=choose_disk(ri,ntheta,n_z,target)
    nc=len(uv)
    ang=2*np.pi*np.arange(ntheta)/ntheta
    outer=np.column_stack((radius*np.cos(ang),radius*np.sin(ang)))
    xy=np.concatenate((uv,outer))
    stride=len(xy)
    zlevels=np.linspace(0,D*length_d,n_z+1)
    points=np.concatenate([np.column_stack((xy,np.full(stride,z))) for z in zlevels]).astype(np.float64)
    def gid(v,k):return int(v)+int(k)*stride
    hexes=[];walls=[];inhex=[];outhex=[];interfaces=[]
    for k in range(n_z):
        for j in range(ntheta):
            jj=(j+1)%ntheta
            a,b=gid(j,k),gid(jj,k)
            c,d=gid(nc+j,k),gid(nc+jj,k)
            a1,b1,c1,d1=gid(j,k+1),gid(jj,k+1),gid(nc+j,k+1),gid(nc+jj,k+1)
            hexes.append((a,c,d,b,a1,c1,d1,b1))
            walls.append((c,d,d1,c1))
            interfaces.append((a,b,b1,a1))
            if k==0:inhex.append((a,c,d,b))
            if k==n_z-1:outhex.append((a1,c1,d1,b1))
    tets=[]
    for k in range(n_z):
        for t in tri:
            a,b,c=sorted(map(int,t))
            a0,b0,c0=gid(a,k),gid(b,k),gid(c,k)
            a1,b1,c1=gid(a,k+1),gid(b,k+1),gid(c,k+1)
            tets.extend(((a0,b0,c0,c1),(a0,b0,c1,b1),(a0,a1,b1,c1)))
    tets=np.asarray(tets,dtype=np.int32)
    det=np.linalg.det(np.stack((points[tets[:,1]]-points[tets[:,0]],points[tets[:,2]]-points[tets[:,0]],points[tets[:,3]]-points[tets[:,0]]),axis=2))
    if np.any(np.abs(det)<1e-14):raise RuntimeError('degenerate tetrahedron')
    neg=det<0
    tets[neg,1],tets[neg,2]=tets[neg,2].copy(),tets[neg,1].copy()
    if len(tets)!=len(tri)*n_z*3:raise AssertionError('incorrect prism count')
    faces={}
    local=((1,2,3),(0,3,2),(0,1,3),(0,2,1))
    for cell in tets:
        for ss in local:
            key=tuple(sorted(map(int,cell[list(ss)])))
            faces[key]=faces.get(key,0)+1
    seam=[]
    for q in interfaces:
        qset=set(q)
        pairs=[p for p in (tuple(sorted(x)) for x in combinations(q,3)) if faces.get(p,0)==1]
        if len(pairs)!=2 or set(pairs[0])|set(pairs[1])!=qset:
            raise RuntimeError(f'HEX/TET interface mismatch: {q}')
        seam.append(pairs)
    inlet=np.asarray([[gid(a,0),gid(b,0),gid(c,0)] for a,b,c in tri],dtype=np.int32)
    outlet=np.asarray([[gid(a,n_z),gid(b,n_z),gid(c,n_z)] for a,b,c in tri],dtype=np.int32)
    data=dict(points=points,hexes=np.asarray(hexes,dtype=np.int32),tets=tets,
        interface_quads=np.asarray(interfaces,dtype=np.int32),
        interface_triangles=np.asarray(seam,dtype=np.int32),
        wall_quads=np.asarray(walls,dtype=np.int32),
        inlet_hex_quads=np.asarray(inhex,dtype=np.int32),
        inlet_tet_triangles=inlet,outlet_hex_quads=np.asarray(outhex,dtype=np.int32),
        outlet_tet_triangles=outlet)
    meta=dict(re=re,diameter=D,length=D*length_d,bulk=ubulk,viscosity=nu,
        friction=f,utau=utau,wall_hex_depth=depth,nominal_midwall_yplus=yplus,
        ntheta=ntheta,n_z=n_z,core_spacing=spacing,n_2d_triangle=len(tri),
        n_hex=len(hexes),n_tet=len(tets),n_total=n)
    return data,meta


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--outdir',type=Path,required=True)
    ap.add_argument('--diameter',type=float,default=.889)
    ap.add_argument('--length-d',type=float,default=50)
    ap.add_argument('--re',type=float,default=1e6)
    ap.add_argument('--nu',type=float,default=1.460734693878e-5)
    ap.add_argument('--target-cells',type=int,default=99000)
    ap.add_argument('--ring-segments',type=int,default=96)
    ap.add_argument('--axial-segments',type=int,default=39)
    ap.add_argument('--sample-yplus',type=float,default=300)
    a=ap.parse_args()
    data,meta=make_mesh(a.diameter,a.length_d,a.nu,a.re,a.ring_segments,
                        a.axial_segments,a.target_cells,a.sample_yplus)
    a.outdir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.outdir/'HXT0.npz',**data)
    (a.outdir/'pipe_geometry.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))
    print('HXT0_READY',a.outdir/'HXT0.npz')

if __name__=='__main__':main()
