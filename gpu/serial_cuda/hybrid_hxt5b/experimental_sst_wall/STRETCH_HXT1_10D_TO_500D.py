#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math, struct
from pathlib import Path
import numpy as np

MAGIC=b"HXT1BIN1"
VERSION=1
HEADER_FMT="<8sII13Q"
HEADER_SIZE=struct.calcsize(HEADER_FMT)
HEX_SIGNS=np.asarray([
    (-1.,-1.,-1.),(1.,-1.,-1.),(1.,1.,-1.),(-1.,1.,-1.),
    (-1.,-1., 1.),(1.,-1., 1.),(1.,1., 1.),(-1.,1., 1.)
],dtype=np.float64)

def sha_bytes(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes(order="C")).hexdigest()

def hex_geom(points,h):
    p=points[np.asarray(h,dtype=np.int64)]
    h2=0.0
    for a in range(8):
        for b in range(a+1,8):
            h2=max(h2,float(np.dot(p[a]-p[b],p[a]-p[b])))
    g=1.0/math.sqrt(3.0)
    dets=[]; invs=[]; vol=0.0
    sx,sy,sz=HEX_SIGNS[:,0],HEX_SIGNS[:,1],HEX_SIGNS[:,2]
    for xi in (-g,g):
        for eta in (-g,g):
            for zeta in (-g,g):
                dxi  =0.125*sx*(1.0+sy*eta)*(1.0+sz*zeta)
                deta =0.125*sy*(1.0+sx*xi )*(1.0+sz*zeta)
                dzeta=0.125*sz*(1.0+sx*xi )*(1.0+sy*eta)
                J=np.empty((3,3),dtype=np.float64)
                J[:,0]=dxi@p; J[:,1]=deta@p; J[:,2]=dzeta@p
                d=float(np.linalg.det(J))
                dets.append(d)
                invs.extend(np.linalg.inv(J).reshape(-1).tolist())
                vol += d
    return [vol,h2,*dets,*invs]

def tet_geom(points,t):
    p=points[np.asarray(t,dtype=np.int64)]
    J=np.column_stack((p[1]-p[0],p[2]-p[0],p[3]-p[0]))
    det=float(np.linalg.det(J)); vol=det/6.0
    h2=max(float(np.dot(p[a]-p[b],p[a]-p[b])) for a in range(4) for b in range(a+1,4))
    return [vol,h2,det,*np.linalg.inv(J).reshape(-1).tolist()]

def take(buf,off,count,dtype,shape):
    dt=np.dtype(dtype)
    nbytes=count*dt.itemsize
    if off+nbytes>len(buf): raise RuntimeError("short HXT1 binary")
    a=np.frombuffer(buf,dtype=dt,count=count,offset=off).copy().reshape(shape)
    return a,off+nbytes

def load(path: Path):
    b=path.read_bytes()
    if len(b)<HEADER_SIZE: raise RuntimeError("file too short")
    u=struct.unpack_from(HEADER_FMT,b,0)
    magic,version,reserved=u[:3]; q=u[3:]
    if magic!=MAGIC: raise RuntimeError(f"bad magic {magic!r}")
    if version!=VERSION: raise RuntimeError(f"unsupported version {version}")
    nv,nh,nt,nhf,ntf,nvel,npres,nif,nwall,ninh,nint,nouth,noutt=map(int,q)
    off=HEADER_SIZE
    A={}
    A['points'],off=take(b,off,nv*3,'<f8',(nv,3))
    A['hexes'],off=take(b,off,nh*8,'<i4',(nh,8))
    A['tets'],off=take(b,off,nt*4,'<i4',(nt,4))
    A['hvel'],off=take(b,off,nh*14,'<i4',(nh,14))
    A['tvel'],off=take(b,off,nt*8,'<i4',(nt,8))
    A['hp'],off=take(b,off,nh,'<i4',(nh,))
    A['tp'],off=take(b,off,nt,'<i4',(nt,))
    A['hgeom'],off=take(b,off,nh*82,'<f8',(nh,82))
    A['tgeom'],off=take(b,off,nt*12,'<f8',(nt,12))
    A['irec'],off=take(b,off,nif*9,'<i4',(nif,9))
    A['wall'],off=take(b,off,nwall*3,'<i4',(nwall,3))
    A['inh'],off=take(b,off,ninh*3,'<i4',(ninh,3))
    A['int'],off=take(b,off,nint*3,'<i4',(nint,3))
    A['outh'],off=take(b,off,nouth*3,'<i4',(nouth,3))
    A['outt'],off=take(b,off,noutt*3,'<i4',(noutt,3))
    if off!=len(b): raise RuntimeError(f"trailing bytes: parsed={off}, file={len(b)}")
    return (version,reserved,q),A

def write(path: Path, header, A):
    version,reserved,q=header
    with path.open('wb') as f:
        f.write(struct.pack(HEADER_FMT,MAGIC,version,reserved,*q))
        for k in ('points','hexes','tets','hvel','tvel','hp','tp','hgeom','tgeom','irec','wall','inh','int','outh','outt'):
            f.write(np.ascontiguousarray(A[k]).tobytes(order='C'))

def main():
    ap=argparse.ArgumentParser(description='Stretch an HXT1 binary mesh axially and rebuild its precomputed geometry plans.')
    ap.add_argument('--input',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--scale',type=float,default=50.0)
    ap.add_argument('--axis',choices=('x','y','z'),default='z')
    ap.add_argument('--manifest',default=None)
    args=ap.parse_args()
    if not (args.scale>0): raise SystemExit('scale must be positive')
    src=Path(args.input).expanduser().resolve(); dst=Path(args.output).expanduser().resolve()
    dst.parent.mkdir(parents=True,exist_ok=True)
    header,A=load(src)
    old={k:v.copy() for k,v in A.items()}
    ai={'x':0,'y':1,'z':2}[args.axis]
    p=A['points']; amin=float(p[:,ai].min()); amax=float(p[:,ai].max()); L0=amax-amin
    if not (L0>0): raise RuntimeError('zero axial extent')
    ranges0=np.ptp(p,axis=0)
    p[:,ai]=amin+args.scale*(p[:,ai]-amin)
    ranges1=np.ptp(p,axis=0)

    # HXT1 geometry is consumed directly by CUDA operators, so it must be rebuilt.
    A['hgeom']=np.asarray([hex_geom(p,h) for h in A['hexes']],dtype=np.float64).reshape((-1,82))
    A['tgeom']=np.asarray([tet_geom(p,t) for t in A['tets']],dtype=np.float64).reshape((-1,12))

    if len(A['hgeom']) and np.min(A['hgeom'][:,2:10])<=0: raise RuntimeError('non-positive HEX Gauss detJ after stretch')
    if len(A['tgeom']) and np.min(A['tgeom'][:,2])<=0: raise RuntimeError('non-positive TET determinant after stretch')
    if len(A['hgeom']) and np.min(A['hgeom'][:,0])<=0: raise RuntimeError('non-positive HEX volume after stretch')
    if len(A['tgeom']) and np.min(A['tgeom'][:,0])<=0: raise RuntimeError('non-positive TET volume after stretch')

    invariant_keys=('hexes','tets','hvel','tvel','hp','tp','irec','wall','inh','int','outh','outt')
    inv={k:(sha_bytes(old[k])==sha_bytes(A[k])) for k in invariant_keys}
    other=[j for j in range(3) if j!=ai]
    transverse_exact=bool(np.array_equal(old['points'][:,other],A['points'][:,other]))
    axial_ok=bool(np.allclose(A['points'][:,ai],amin+args.scale*(old['points'][:,ai]-amin),rtol=0,atol=1e-14))
    L1=float(A['points'][:,ai].max()-A['points'][:,ai].min())
    vold0=float(old['hgeom'][:,0].sum()+old['tgeom'][:,0].sum())
    vold1=float(A['hgeom'][:,0].sum()+A['tgeom'][:,0].sum())
    volume_ratio=vold1/vold0

    checks={
      'connectivity_dof_boundary_records_unchanged': all(inv.values()),
      'transverse_coordinates_bitwise_unchanged': transverse_exact,
      'axial_coordinates_scaled': axial_ok,
      'length_ratio_matches_scale': abs(L1/L0-args.scale)<=1e-11*max(1.0,args.scale),
      'total_volume_ratio_matches_scale': abs(volume_ratio-args.scale)<=2e-10*max(1.0,args.scale),
      'positive_hex_geometry': (len(A['hgeom'])==0 or (float(np.min(A['hgeom'][:,0]))>0 and float(np.min(A['hgeom'][:,2:10]))>0)),
      'positive_tet_geometry': (len(A['tgeom'])==0 or (float(np.min(A['tgeom'][:,0]))>0 and float(np.min(A['tgeom'][:,2]))>0)),
    }
    status='PASS' if all(checks.values()) else 'FAIL'
    write(dst,header,A)
    if dst.stat().st_size!=src.stat().st_size:
        checks['binary_size_unchanged']=False; status='FAIL'
    else: checks['binary_size_unchanged']=True

    q=header[2]
    manifest={
      'status':status,'source':str(src),'output':str(dst),'axis':args.axis,'scale':args.scale,
      'counts':{'vertices':q[0],'hex_cells':q[1],'tet_cells':q[2],'velocity_dofs':q[5],'pressure_dofs':q[6]},
      'source_extent':{'min':amin,'max':amax,'length':L0,'ranges_xyz':ranges0.tolist()},
      'stretched_extent':{'min':float(A['points'][:,ai].min()),'max':float(A['points'][:,ai].max()),'length':L1,'ranges_xyz':ranges1.tolist()},
      'total_volume_source':vold0,'total_volume_stretched':vold1,'volume_ratio':volume_ratio,
      'invariant_sha256':{k:{'source':sha_bytes(old[k]),'stretched':sha_bytes(A[k])} for k in invariant_keys},
      'checks':checks,
      'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
      'output_sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),
    }
    mf=Path(args.manifest).expanduser().resolve() if args.manifest else dst.with_suffix('.manifest.json')
    mf.write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'HXT1_STRETCH_SOURCE path={src} vertices={q[0]} hex={q[1]} tet={q[2]} velocityDofs={q[5]} pressureDofs={q[6]} axis={args.axis} L={L0:.12e}')
    print(f'HXT1_STRETCH_OUTPUT path={dst} axis={args.axis} scale={args.scale:.12g} L={L1:.12e} lengthRatio={L1/L0:.12e} volumeRatio={volume_ratio:.12e}')
    print('HXT1_STRETCH_INVARIANTS connectivity=UNCHANGED dofs=UNCHANGED boundaries=UNCHANGED transverseCoordinates=UNCHANGED geometryPlans=REBUILT')
    print(f'HXT1_STRETCH_MANIFEST path={mf}')
    print(f'HXT1_STRETCH_STATUS={status}')
    if status!='PASS': raise SystemExit(2)

if __name__=='__main__': main()
