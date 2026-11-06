#!/usr/bin/env python3
"""Generate a conforming swept-hex -> 24 tet (face-center + cell-center) mesh.

Reads exact circular sections of Rhino STEP geometry through OpenCascade/OCP.
All positions in STEP are millimetres; output Gmsh msh2 uses metres.
No quad face uses a preferred diagonal: face centres are globally shared.

Requires Python with OCP and numpy. Pre-generated MSH is included in package.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_FACE
from OCP.TopoDS import TopoDS
from OCP.TopExp import TopExp_Explorer
from OCP.BRepAdaptor import BRepAdaptor_Surface

Q = 0.0005469756
NU = 0.01460735 / 1000.
RHO = 1000.
TWOPI = 2 * np.pi

def read_step(path):
    r=STEPControl_Reader()
    status=r.ReadFile(str(path))
    if int(status)!=1:
        raise RuntimeError(f'STEP read failed: {status}')
    if r.TransferRoots()<1:
        raise RuntimeError('STEP could not transfer solid')
    explorer=TopExp_Explorer(r.OneShape(), TopAbs_FACE)
    side=[]; caps=[]
    while explorer.More():
        surface=BRepAdaptor_Surface(TopoDS.Face_s(explorer.Current()), True)
        if 'BSplineSurface' in str(surface.GetType()): side.append(surface)
        elif 'Plane' in str(surface.GetType()): caps.append(surface)
        explorer.Next()
    if len(side)!=2 or len(caps)!=2:
        raise RuntimeError(f'Expected 2 BSpline tube halves and 2 planar caps; got {len(side)}, {len(caps)}')
    side.sort(key=lambda s: s.FirstVParameter())
    s0,s1=side
    if abs(s0.FirstUParameter()-s1.FirstUParameter())>1e-7 or abs(s0.LastUParameter()-s1.LastUParameter())>1e-7:
        raise RuntimeError('Tube surface longitudinal domains differ')
    return s0,s1

class Tube:
    def __init__(self,s0,s1):
        self.s0=s0; self.s1=s1
        self.umin=s0.FirstUParameter();self.umax=s0.LastUParameter()
        self.v0min=s0.FirstVParameter();self.v0max=s0.LastVParameter()
        self.v1min=s1.FirstVParameter();self.v1max=s1.LastVParameter()
    def wall(self,u,theta):
        theta=float(theta)%TWOPI
        if theta<=np.pi:
            v=self.v0min+(theta/np.pi)*(self.v0max-self.v0min)
            p=self.s0.Value(float(u),v)
        else:
            v=self.v1min+((theta-np.pi)/np.pi)*(self.v1max-self.v1min)
            p=self.s1.Value(float(u),v)
        return np.array((p.X(),p.Y(),p.Z()),dtype=float)
    def center(self,u):
        return (self.wall(u,0)+self.wall(u,np.pi))/2
    def point(self,u,xi,eta,c=None):
        x=xi*np.sqrt(max(0.,1.-eta*eta/2.))
        y=eta*np.sqrt(max(0.,1.-xi*xi/2.))
        rad=np.hypot(x,y)
        if c is None:c=self.center(u)
        if rad<1.e-12:return c.copy()
        return c+rad*(self.wall(u,np.arctan2(y,x))-c)

def build_u(tube, ns):
    dense_u=np.linspace(tube.umin,tube.umax,max(2501,ns*8))
    c=np.array([tube.center(u) for u in dense_u])
    ds=np.linalg.norm(np.diff(c,axis=0),axis=1)
    s=np.r_[0,np.cumsum(ds)]
    # reverse: k=0 corresponds to old OpenFOAM patch_1_0 at STEP u=umax
    u=np.interp(np.linspace(s[-1],0,ns+1),s,dense_u)
    um=(u[:-1]+u[1:])/2
    return u,um,float(s[-1])

def gen(step,out,ns=240,nxy=6):
    tic=time.time(); s0,s1=read_step(step);tube=Tube(s0,s1)
    u,um,length_mm=build_u(tube,ns)
    print(f'SPIRAL_STEP sweepLength_mm={length_mm:.5f} axial={ns} cross={nxy}x{nxy}',flush=True)
    coords=[]
    def new(x):
        coords.append(np.asarray(x,dtype=float)*.001)
        return len(coords)  # 1-based
    n=nxy
    xy=np.linspace(-1,1,n+1)
    g=np.empty((ns+1,n+1,n+1),dtype=np.int32)
    # record tangents for boundary normals
    centers=np.array([tube.center(t) for t in u]) * .001
    for k in range(ns+1):
        c=centers[k]*1000
        for j in range(n+1):
            for i in range(n+1):
                g[k,j,i]=new(tube.point(u[k],xy[i],xy[j],c))
    print(f'SPIRAL_NODES skeleton={len(coords)}',flush=True)
    # Cross sectional face centres (axial planes)
    fk=np.empty((ns+1,n,n),dtype=np.int32)
    for k in range(ns+1):
        for j in range(n):
            for i in range(n):
                ids=[g[k,j,i],g[k,j,i+1],g[k,j+1,i+1],g[k,j+1,i]]
                fk[k,j,i]=new(np.mean([coords[t-1] for t in ids],axis=0)*1000)
    # Faces of constant i (f_i), span j and k
    fi=np.empty((ns,n,n+1),dtype=np.int32)
    for k in range(ns):
        for j in range(n):
            for i in range(n+1):
                if i in (0,n):
                    x=xy[i]; yy=(xy[j]+xy[j+1])/2
                    p=tube.point(um[k],x,yy)
                else:
                    ids=[g[k,j,i],g[k,j+1,i],g[k+1,j+1,i],g[k+1,j,i]]
                    p=np.mean([coords[t-1] for t in ids],axis=0)*1000
                fi[k,j,i]=new(p)
    # Faces of constant j (f_j), span i and k
    fj=np.empty((ns,n+1,n),dtype=np.int32)
    for k in range(ns):
        for j in range(n+1):
            for i in range(n):
                if j in (0,n):
                    x=(xy[i]+xy[i+1])/2;yy=xy[j]
                    p=tube.point(um[k],x,yy)
                else:
                    ids=[g[k,j,i],g[k,j,i+1],g[k+1,j,i+1],g[k+1,j,i]]
                    p=np.mean([coords[t-1] for t in ids],axis=0)*1000
                fj[k,j,i]=new(p)
    # one cell-centre point per swept hex
    cc=np.empty((ns,n,n),dtype=np.int32)
    for k in range(ns):
        for j in range(n):
            for i in range(n):
                ids=[g[kk,jj,ii] for kk in (k,k+1) for jj in (j,j+1) for ii in (i,i+1)]
                cc[k,j,i]=new(np.mean([coords[t-1] for t in ids],axis=0)*1000)
    vertices=np.asarray(coords)
    print(f'SPIRAL_NODES complete={len(vertices)}',flush=True)
    # create exactly 24 tets for every swept hex; each internal face is conformed
    tets=np.empty((ns*n*n*24,4),dtype=np.int32)
    tri_in=[];tri_out=[];tri_wall=[]
    idx=0
    for k in range(ns):
        for j in range(n):
            for i in range(n):
                a,b,c,d=g[k,j,i],g[k,j,i+1],g[k,j+1,i+1],g[k,j+1,i]
                e,f,h,l=g[k+1,j,i],g[k+1,j,i+1],g[k+1,j+1,i+1],g[k+1,j+1,i]
                sides=(
                    ((a,b,c,d),fk[k,j,i],1 if k==0 else 0),
                    ((e,f,h,l),fk[k+1,j,i],2 if k==ns-1 else 0),
                    ((a,d,l,e),fi[k,j,i],3 if i==0 else 0),
                    ((b,c,h,f),fi[k,j,i+1],3 if i==n-1 else 0),
                    ((a,b,f,e),fj[k,j,i],3 if j==0 else 0),
                    ((d,c,h,l),fj[k,j+1,i],3 if j==n-1 else 0))
                cp=cc[k,j,i]
                for corners,facenode,tag in sides:
                    for q in range(4):
                        v0=corners[q];v1=corners[(q+1)%4]
                        tets[idx]=[cp,facenode,v0,v1];idx+=1
                        if tag:
                            if tag==1:tri_in.append((facenode,v0,v1))
                            elif tag==2:tri_out.append((facenode,v0,v1))
                            else:tri_wall.append((facenode,v0,v1))
    assert idx==len(tets)
    # Orient each element positively in Cartesian physical geometry
    v0,v1,v2,v3=vertices[tets[:,0]-1],vertices[tets[:,1]-1],vertices[tets[:,2]-1],vertices[tets[:,3]-1]
    vol6=np.einsum('ij,ij->i',np.cross(v1-v0,v2-v0),v3-v0)
    inverted=vol6<0
    tets[inverted,2:4]=tets[inverted,2:4][:,::-1]
    vol6=np.abs(vol6)
    if vol6.min()<=1.e-17:
        raise RuntimeError(f'Small or zero tet determinant {vol6.min()}')
    total_v=vol6.sum()/6.
    def oriented_tri(trilist, patch):
        arr=np.asarray(trilist,dtype=np.int32)
        pts=vertices[arr-1]
        cross=np.cross(pts[:,1]-pts[:,0],pts[:,2]-pts[:,0])
        cent=pts.mean(axis=1)
        if patch==1:
            want=centers[0]-centers[1]  # inlet outward = opposite axial forward
            dots=cross@want
        elif patch==2:
            want=centers[-1]-centers[-2] # outlet outward = axial forward
            dots=cross@want
        else:
            # radial from closest section's centre; use approximate axial nearest
            # by e.g KD-tree from centres of all axial stations
            from scipy.spatial import cKDTree
            ck=cKDTree(centers)
            ids=ck.query(cent)[1]
            dots=np.einsum('ij,ij->i',cross,cent-centers[ids])
        swaps=dots<0
        arr[swaps,1:3]=arr[swaps,1:3][:,::-1]
        cross[swaps]*=-1
        return arr,cross,swaps.sum()
    tin,nin,flipin=oriented_tri(tri_in,1)
    tout,nout,flipout=oriented_tri(tri_out,2)
    twall,nwall,flipwall=oriented_tri(tri_wall,3)
    ain=np.linalg.norm(nin.sum(axis=0))/2
    aout=np.linalg.norm(nout.sum(axis=0))/2
    normal_in=nin.sum(axis=0);normal_in/=np.linalg.norm(normal_in)
    normal_out=nout.sum(axis=0);normal_out/=np.linalg.norm(normal_out)
    veloc=Q/ain*(-normal_in)
    wetted_circ=np.sum(np.linalg.norm(np.diff(np.array([vertices[g[0,j,i]-1] for i in range(n+1) for j in [0]]),axis=0),axis=1)) # overridden below
    boundary_loop=[*(g[0,0,i] for i in range(n)),*(g[0,j,n] for j in range(n)),*(g[0,n,i] for i in range(n,0,-1)),*(g[0,j,0] for j in range(n,0,-1))]
    perimeter=np.linalg.norm(np.diff(np.asarray([vertices[x-1] for x in boundary_loop+[boundary_loop[0]]]),axis=0),axis=1).sum()
    diameter=4*ain/perimeter
    inlet_speed=Q/ain; re=inlet_speed*diameter/NU
    capcent=vertices[tin[:,0]-1].mean(axis=0)
    stats={
      'generator':'OCP STEP exact-sections swept square-to-circle hex-to-face-center 24tet',
      'source_step':step.name,'source_units':'mm','mesh_units':'m',
      'cross_cells_per_side':n,'axial_segments':ns,'hex_equivalents':ns*n*n,
      'points':len(vertices),'tetrahedra':len(tets),'faces_inlet':len(tin),'faces_outlet':len(tout),'faces_wall':len(twall),
      'tets_ever_negative_reoriented':int(inverted.sum()),
      'min_tet_volume_m3':float(vol6.min()/6), 'max_tet_volume_m3':float(vol6.max()/6),
      'volume_total_m3':float(total_v),'step_volume_m3':19950.17703247872*1.e-9,
      'volume_rel_error_to_cad':float(total_v/(19950.17703247872*1.e-9)-1),
      'step_sweep_centerline_length_m':length_mm/1000,
      'bbox_min_m':vertices.min(axis=0).tolist(),'bbox_max_m':vertices.max(axis=0).tolist(),
      'inlet_area_m2':float(ain),'outlet_area_m2':float(aout),
      'inlet_normal_outward':normal_in.tolist(),'outlet_normal_outward':normal_out.tolist(),
      'inlet_velocity_mps':veloc.tolist(), 'inlet_bulk_speed_mps':float(inlet_speed),
      'inlet_perimeter_m':float(perimeter),'inlet_hydraulic_diameter_m':float(diameter),
      'volumetric_flow_m3ps':Q,'density_kgpm3':RHO,'dynamic_viscosity_Pas':NU*RHO,'kinematic_viscosity_m2ps':NU,
      'reynolds_number':float(re),'physical_pressure_pa_factor':RHO,
      'orientation_swaps_boundary':{'inlet':int(flipin),'outlet':int(flipout),'wall':int(flipwall)},
      'elapsed_seconds':time.time()-tic,
    }
    print('SPIRAL_QUALITY',json.dumps({k:stats[k] for k in ['tetrahedra','points','min_tet_volume_m3','volume_rel_error_to_cad','inlet_area_m2','inlet_bulk_speed_mps','reynolds_number']}),flush=True)
    assert abs(stats['volume_rel_error_to_cad'])<0.035, 'Mesh volume differs too much from STEP CAD'
    assert abs(ain/aout-1)<0.015, 'Inlet/outlet areas differ too much'
    print(f'SPIRAL_WRITING {out}',flush=True)
    with out.open('w') as fp:
        fp.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n4\n2 1 "inlet"\n2 2 "outlet"\n2 3 "wall"\n3 4 "fluid"\n$EndPhysicalNames\n')
        fp.write(f'$Nodes\n{len(vertices)}\n')
        for idx, (x,y,z) in enumerate(vertices,1):fp.write(f'{idx} {x:.15g} {y:.15g} {z:.15g}\n')
        fp.write('$EndNodes\n')
        total=len(tets)+len(tin)+len(tout)+len(twall)
        fp.write(f'$Elements\n{total}\n')
        elt=1
        for tag, arr in [(1,tin),(2,tout),(3,twall)]:
            for a,b,c in arr:
                fp.write(f'{elt} 2 2 {tag} {tag} {a} {b} {c}\n');elt+=1
        for a,b,c,d in tets:
            fp.write(f'{elt} 4 2 4 4 {a} {b} {c} {d}\n');elt+=1
        fp.write('$EndElements\n')
    (out.parent/(out.stem+'_stats.json')).write_text(json.dumps(stats,indent=2)+'\n')
    print('SPIRAL_GENERATION=PASS', 'path=',out,'runtime_s=',round(time.time()-tic,2),flush=True)
    return stats

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--step',type=Path,default=Path('~/Downloads/spiral.stp').expanduser())
    ap.add_argument('--out',type=Path,default=Path('spiral_swept_207360_24tet.msh'))
    ap.add_argument('--axial',type=int,default=240)
    ap.add_argument('--cross',type=int,default=6)
    a=ap.parse_args()
    if a.axial<5 or a.cross<2:ap.error('Use --axial>=5, --cross>=2')
    gen(a.step,a.out,a.axial,a.cross)
