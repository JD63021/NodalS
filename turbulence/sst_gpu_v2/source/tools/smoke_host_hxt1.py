#!/usr/bin/env python3
"""Deterministic small pure-tet cylinder; tests HXT1 writer without Gmsh/CUDA."""
import importlib.util
import pathlib
import sys
import subprocess
import tempfile
import numpy as np
from scipy.spatial import Delaunay
ROOT=pathlib.Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location('make_tet',ROOT/'tools/make_puretet_10d.py')
m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
rg=np.random.default_rng(20261015)
pts=[]
# Circular polygon boundary with inlet, outlet and intermediate rings.
for z in np.linspace(0,.04,13):
    for theta in np.linspace(0,2*np.pi,18,endpoint=False):
        pts.append([.002*np.cos(theta),.002*np.sin(theta),z])
# Interior random points inside polygonal cylinder.
for _ in range(650):
    rr=.00175*np.sqrt(rg.uniform());th=rg.uniform(0,2*np.pi)
    pts.append([rr*np.cos(th),rr*np.sin(th),rg.uniform(0,.04)])
# Insert interior cap points, otherwise cap triangulation can be coarse.
for z in (0,.04):
    for j in range(90):
        rr=.0018*np.sqrt(rg.uniform());th=rg.uniform(0,2*np.pi)
        pts.append([rr*np.cos(th),rr*np.sin(th),z])
pts=np.asarray(pts,dtype=np.float64)
tet=Delaunay(pts).simplices.astype(np.int32)
J=np.stack((pts[tet[:,1]]-pts[tet[:,0]],pts[tet[:,2]]-pts[tet[:,0]],pts[tet[:,3]]-pts[tet[:,0]]),axis=2)
tet=tet[np.abs(np.linalg.det(J))>1e-20]
with tempfile.TemporaryDirectory(prefix='nodals_tet_smoke_') as d:
    d=pathlib.Path(d)
    m.write_hxt0(pts,tet,d/'HXT0.npz')
    z=np.load(d/'HXT0.npz')
    faces=m.exterior_faces(z['tets'])
    w=faces[np.max(np.abs(pts[faces][:,:,2]),axis=1)>1e-8]
    w=w[np.max(np.abs(pts[w][:,:,2]-.04),axis=1)>1e-8]
    surfpts=pts[w]
    norms=np.cross(surfpts[:,1]-surfpts[:,0],surfpts[:,2]-surfpts[:,0]);norms/=np.linalg.norm(norms,axis=1)[:,None]
    # Face table follows positive-tet orientation; check radial side normals.
    assert np.mean(np.sum(norms[:,:2]*surfpts[:,0,:2],axis=1))>0
    l0=np.array([.063089014491502,.063089014491502,.873821971016996,
        .249286745170910,.249286745170910,.501426509658180,
        .053145049844816,.053145049844816,.310352451033785,
        .310352451033785,.636502499121399,.636502499121399])
    l1=np.array([.063089014491502,.873821971016996,.063089014491502,
        .249286745170910,.501426509658180,.249286745170910,
        .310352451033785,.636502499121399,.053145049844816,
        .636502499121399,.053145049844816,.310352451033785])
    l2=1-l0-l1
    q=l0[None,:,None]*surfpts[:,0,None,:]+l1[None,:,None]*surfpts[:,1,None,:]+l2[None,:,None]*surfpts[:,2,None,:]
    xx=[]
    for y in (7.347338250770073e-05,1.163328556371928e-04,
              1.591923287666849e-04,2.020518018961770e-04,
              2.449112750256691e-04):
        xx.append((q-y*norms[:,None,:]).reshape(-1,3))
    xx=np.vstack(xx)
    found=np.sum(Delaunay(pts).find_simplex(xx)>=0)
    assert found==len(xx),(found,len(xx))
    print('PURETET_HOST_SAMPLING_AUDIT 5heights x 12quad x %dwallFaces = %d samples allInside=PASS' %(len(w),len(xx)))
    subprocess.run([sys.executable,str(ROOT/'hybrid_hxt1/prepare_hxt1.py'),'--input',str(d/'HXT0.npz'),'--outdir',str(d)],check=True)
    data=(d/'HXT1_mesh.bin').read_bytes()
    assert data[:8]==b'HXT1BIN1'
    import struct
    head=struct.unpack_from('<13Q',data,16)
    assert head[1]==0 and head[2]==len(tet) and head[10]>0 and head[12]>0
    print('PURETET_HOST_SMOKE status=PASS header_nhex=0 ntet=%d inletFaces=%d outletFaces=%d' %(head[2],head[10],head[12]))
