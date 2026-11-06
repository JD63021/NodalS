"""Case-only developed inlet CSV preparation for the existing pure-tet native loader.

No CUDA code or boundary operators are modified. Input CSV retains its exact
8-column Stage27 schema; derived CSV contains one exact xyz row per inlet node.
"""
from __future__ import annotations

import csv
from pathlib import Path
import math
import numpy as np

COLUMNS=('x_m','y_m','z_m','Ux_m_s','Uy_m_s','Uz_m_s','k_m2_s2','omega_1_s')


def read_profile(path):
    path=Path(path)
    with path.open(newline='') as f:
        reader=csv.reader(f)
        header=tuple(next(reader,()))
        if header != COLUMNS:
            raise ValueError('Inlet CSV header must be exactly '+','.join(COLUMNS))
        rows=[]
        for row in reader:
            if not row:continue
            if len(row)!=8:raise ValueError('Expected 8 columns in '+str(path))
            rows.append([float(v) for v in row])
    a=np.asarray(rows,dtype=np.float64)
    if a.ndim!=2 or a.shape[0]<3 or a.shape[1]!=8 or not np.isfinite(a).all():
        raise ValueError('Inlet CSV needs >=3 finite, 8-column points')
    if len(np.unique(a[:,1:3],axis=0))!=len(a):
        raise ValueError('Duplicate y,z coordinates in inlet CSV')
    return a


def inlet_nodes(msh, physical_tag=1):
    """Read Gmsh 2.2 ASCII inlet boundary triangles and their exact vertices."""
    msh=Path(msh)
    nodes={}
    faces=[]
    with msh.open() as f:
        if f.readline().strip()!='$MeshFormat' or not f.readline().startswith('2.2 0'):
            raise ValueError('Profile interpolation needs Gmsh ASCII v2.2 mesh')
        section=None
        for line in f:
            s=line.strip()
            if s=='$Nodes':
                n=int(next(f))
                for _ in range(n):
                    row=next(f).split()
                    nodes[int(row[0])]=np.asarray([float(x) for x in row[1:4]],dtype=np.float64)
                continue
            if s=='$Elements':
                n=int(next(f))
                for _ in range(n):
                    row=list(map(int,next(f).split()))
                    kind,ntags=row[1:3]
                    if kind==2 and ntags and row[3]==physical_tag:
                        ids=row[3+ntags:]
                        if len(ids)!=3:raise ValueError('Expected triangle inlet face')
                        faces.append(ids)
                break
    if not faces:raise ValueError('No inlet triangles tagged physical '+str(physical_tag))
    ids=sorted({j for tri in faces for j in tri})
    coords=np.stack([nodes[k] for k in ids])
    if not np.isfinite(coords).all():raise ValueError('Nonfinite inlet mesh coordinates')
    return ids,coords,np.asarray(faces,dtype=np.int64)


def face_flux(coords,faces,ids,vel,base_velocity):
    lookup={v:i for i,v in enumerate(ids)}
    i=np.asarray([[lookup[j] for j in tri] for tri in faces],dtype=int)
    xyz=coords[i]
    area_normal=np.cross(xyz[:,1,:]-xyz[:,0,:],xyz[:,2,:]-xyz[:,0,:])*0.5
    # The existing native solver requires the prescribed inlet vector to be
    # inward on every face. Use it to orient the planar inlet face normals.
    dn=area_normal @ np.asarray(base_velocity,dtype=np.float64)
    if np.any(abs(dn)<1e-16):
        raise ValueError('Some inlet faces nearly tangent to the reference velocity')
    area_normal[dn>0]*=-1
    flow=-np.einsum('ij,ij->i',area_normal,np.mean(vel[i],axis=1))
    area=np.linalg.norm(area_normal,axis=1)
    if np.any(area<=0):raise ValueError('Degenerate inlet triangle')
    return float(flow.sum()),float(np.min(flow/area)),float(area.sum()),area_normal


def prepare_profile(source_csv, msh, dest, base_velocity, normalization='preserve', corner_epsilon=1e-9):
    """Interpolate source y,z to inlet vertices, optionally match constant inlet flux.

    Importantly: the native solver uses ONLY Ux,Uy,Uz; k,omega are retained in
    the derived CSV for provenance but still use uniform native inlet treatment.
    """
    if normalization not in ('preserve','match_bulk'):
        raise ValueError('profile_flux must be preserve or match_bulk')
    if not math.isfinite(corner_epsilon) or corner_epsilon<0 or corner_epsilon>1e-3:
        raise ValueError('profile_corner_epsilon must be between 0 and 1e-3 m/s')
    source=read_profile(source_csv)
    ids,xyz,faces=inlet_nodes(msh)
    target=xyz[:,1:3]
    try:
        from scipy.interpolate import LinearNDInterpolator
        from scipy.spatial import cKDTree
    except ImportError as ex:
        raise RuntimeError('Install scipy for profile_mapping=interpolate: python3 -m pip install scipy') from ex
    interp=LinearNDInterpolator(source[:,1:3],source[:,3:8],fill_value=np.nan)
    sampled=np.asarray(interp(target),dtype=float)
    missing=~np.isfinite(sampled).all(axis=1)
    if missing.any():
        # Geometric inlet endpoints can fall microscopically outside the source
        # convex hull due to roundoff; no unrestricted extrapolation.
        dist,indx=cKDTree(source[:,1:3]).query(target[missing])
        tol=1e-8  # metres, far smaller than ~0.089 m duct width
        if np.any(dist>tol):
            raise ValueError(f'{int(np.count_nonzero(dist>tol))} inlet vertices outside profile domain '
                             f'(max closest distance {max(dist):.5g} m). Use an inlet-spanning CSV.')
        sampled[missing]=source[indx,3:8]
    flux0,min_speed,area,area_normals=face_flux(xyz,faces,ids,sampled[:,:3],base_velocity)
    # Some unstructured corner triangles have ALL THREE vertices on the square
    # duct's two wall edges. A true no-slip precursor is exactly zero there.
    # Native Stage27 currently requires strictly positive inward speed on *every*
    # inlet triangle; an explicit tiny epsilon lifts only those zero-flux faces.
    # This is a documented CSV preprocessing tolerance, not a CUDA code change.
    lifted_faces=0
    if min_speed<=0 or (corner_epsilon>0 and min_speed<corner_epsilon):
        lookup={vid:i for i,vid in enumerate(ids)}
        ii=np.asarray([[lookup[j] for j in tri] for tri in faces],dtype=int)
        face_avg=np.mean(sampled[ii,:3],axis=1)
        inward=-np.einsum('ij,ij->i',area_normals,face_avg) / np.linalg.norm(area_normals,axis=1)
        if np.any(inward<-1e-8):
            raise ValueError('Developed inlet has actual reverse-flow faces; refuse to mask physical backflow')
        zero_faces=np.flatnonzero(inward<=max(0.0,corner_epsilon*0.5))
        lifted_faces=len(zero_faces)
        if lifted_faces and corner_epsilon==0:
            raise ValueError('Zero-flux inlet corner triangles require profile_corner_epsilon > 0 '
                             '(native loader currently requires strict inward flux)')
        if lifted_faces:
            direction=np.asarray(base_velocity,dtype=float)
            direction/=np.linalg.norm(direction)
            sampled[np.unique(ii[zero_faces].ravel()),:3]+=corner_epsilon*direction
            flux0,min_speed,area,_=face_flux(xyz,faces,ids,sampled[:,:3],base_velocity)
    if flux0<=0 or min_speed<=0:
        raise ValueError(f'Interpolated inlet has non-inward face(s): flux={flux0:.9g}, minInward={min_speed:.9g}')
    base=np.repeat(np.asarray(base_velocity,dtype=float)[None,:],len(ids),axis=0)
    bulk_flux,_,_,_=face_flux(xyz,faces,ids,base,base_velocity)
    factor=1.0
    if normalization=='match_bulk':
        if not bulk_flux>0:raise ValueError('Reference bulk inlet flux is not positive')
        factor=bulk_flux/flux0
        sampled[:,:3]*=factor
    flux,min_speed,_,_=face_flux(xyz,faces,ids,sampled[:,:3],base_velocity)
    data=np.column_stack([xyz,sampled])
    dest=Path(dest)
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix(dest.suffix+'.tmp')
    with tmp.open('w',newline='') as f:
        writer=csv.writer(f,lineterminator='\n')
        writer.writerow(COLUMNS)
        for row in data:
            writer.writerow([format(float(x),'.17g') for x in row])
    tmp.replace(dest)
    return dict(input_rows=int(source.shape[0]),inlet_vertices=len(ids),inlet_faces=len(faces),
                area_m2=area,source_flux_m3_s=flux0,bulk_flux_m3_s=bulk_flux,
                exported_flux_m3_s=flux,velocity_scale=factor,min_inward_m_s=min_speed,
                output=str(dest),normalization=normalization,corner_epsilon=corner_epsilon,lifted_zero_faces=lifted_faces)
