#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math, struct
from pathlib import Path
import numpy as np

MAGIC=b"HXT1BIN1"
VERSION=1
HEADER_FMT="<8sII13Q"
HEADER_SIZE=struct.calcsize(HEADER_FMT)
HEX_FACES=(
    (0,1,2,3),(4,5,6,7),(0,1,5,4),
    (1,2,6,5),(2,3,7,6),(3,0,4,7),
)
HEX_SIGNS=np.asarray([
    (-1.,-1.,-1.),(1.,-1.,-1.),(1.,1.,-1.),(-1.,1.,-1.),
    (-1.,-1., 1.),(1.,-1., 1.),(1.,1., 1.),(-1.,1., 1.)
],dtype=np.float64)

def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes(order='C')).hexdigest()

def take(buf,off,count,dtype,shape):
    dt=np.dtype(dtype); nb=count*dt.itemsize
    if off+nb>len(buf): raise RuntimeError('short HXT1 binary')
    a=np.frombuffer(buf,dtype=dt,count=count,offset=off).copy().reshape(shape)
    return a,off+nb

def load(path: Path):
    b=path.read_bytes()
    if len(b)<HEADER_SIZE: raise RuntimeError('file too short')
    u=struct.unpack_from(HEADER_FMT,b,0)
    magic,version,reserved=u[:3]; q=u[3:]
    if magic!=MAGIC: raise RuntimeError(f'bad magic {magic!r}')
    if version!=VERSION: raise RuntimeError(f'unsupported version {version}')
    nv,nh,nt,nhf,ntf,nvel,npres,nif,nwall,ninh,nint,nouth,noutt=map(int,q)
    off=HEADER_SIZE; A={}
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
    if off!=len(b): raise RuntimeError(f'trailing bytes: parsed={off}, file={len(b)}')
    return (version,reserved,q),A

def write(path: Path,header,A):
    version,reserved,q=header
    with path.open('wb') as f:
        f.write(struct.pack(HEADER_FMT,MAGIC,version,reserved,*q))
        for k in ('points','hexes','tets','hvel','tvel','hp','tp','hgeom','tgeom','irec','wall','inh','int','outh','outt'):
            f.write(np.ascontiguousarray(A[k]).tobytes(order='C'))

def hex_geom(points,h):
    p=points[np.asarray(h,dtype=np.int64)]
    h2=max(float(np.dot(p[a]-p[b],p[a]-p[b])) for a in range(8) for b in range(a+1,8))
    g=1.0/math.sqrt(3.0); dets=[]; invs=[]; vol=0.0
    sx,sy,sz=HEX_SIGNS[:,0],HEX_SIGNS[:,1],HEX_SIGNS[:,2]
    for xi in (-g,g):
        for eta in (-g,g):
            for zeta in (-g,g):
                dxi  =0.125*sx*(1.0+sy*eta)*(1.0+sz*zeta)
                deta =0.125*sy*(1.0+sx*xi )*(1.0+sz*zeta)
                dzeta=0.125*sz*(1.0+sx*xi )*(1.0+sy*eta)
                J=np.empty((3,3),dtype=np.float64)
                J[:,0]=dxi@p; J[:,1]=deta@p; J[:,2]=dzeta@p
                d=float(np.linalg.det(J)); dets.append(d)
                invs.extend(np.linalg.inv(J).reshape(-1).tolist()); vol+=d
    return [vol,h2,*dets,*invs]

def tet_geom(points,t):
    p=points[np.asarray(t,dtype=np.int64)]
    J=np.column_stack((p[1]-p[0],p[2]-p[0],p[3]-p[0]))
    det=float(np.linalg.det(J)); vol=det/6.0
    h2=max(float(np.dot(p[a]-p[b],p[a]-p[b])) for a in range(4) for b in range(a+1,4))
    return [vol,h2,det,*np.linalg.inv(J).reshape(-1).tolist()]

def face_vertices(hexes,records):
    out=set()
    for row in np.asarray(records,dtype=np.int64):
        c=int(row[0]); lf=int(row[1])
        if c<0 or c>=len(hexes) or lf<0 or lf>=6: raise RuntimeError('bad hex boundary record')
        h=hexes[c]
        out.update(int(h[j]) for j in HEX_FACES[lf])
    return np.asarray(sorted(out),dtype=np.int64)

def interface_vertices(hexes,irec):
    # irec columns 0,1 identify the HEX cell/local face at the HEX/TET interface.
    out=set()
    for row in np.asarray(irec,dtype=np.int64):
        c=int(row[0]); lf=int(row[1])
        if c<0 or c>=len(hexes) or lf<0 or lf>=6: raise RuntimeError('bad interface record')
        h=hexes[c]
        out.update(int(h[j]) for j in HEX_FACES[lf])
    return np.asarray(sorted(out),dtype=np.int64)

def main():
    ap=argparse.ArgumentParser(description='Create same-connectivity 500D low-y+ HXT1 mesh by radial wall-layer compression plus axial stretch.')
    ap.add_argument('--input',required=True,help='original 10D HXT1_mesh.bin')
    ap.add_argument('--output',required=True)
    ap.add_argument('--axial-scale',type=float,default=50.0)
    ap.add_argument('--wall-gap-scale',type=float,default=0.32,help='new interface-to-wall radial gap divided by old gap')
    ap.add_argument('--ref-yplus-mean',type=float,default=83.44770399615)
    ap.add_argument('--ref-yplus-max',type=float,default=101.7253636439)
    ap.add_argument('--manifest',default=None)
    args=ap.parse_args()
    if args.axial_scale<=0: raise SystemExit('axial-scale must be >0')
    if not (0<args.wall_gap_scale<1): raise SystemExit('wall-gap-scale must be in (0,1)')

    src=Path(args.input).expanduser().resolve(); dst=Path(args.output).expanduser().resolve()
    dst.parent.mkdir(parents=True,exist_ok=True)
    header,A=load(src); old={k:v.copy() for k,v in A.items()}
    q=header[2]

    p=A['points']; nv=len(p)
    wallv=face_vertices(A['hexes'],A['wall'])
    intv=interface_vertices(A['hexes'],A['irec'])
    if len(wallv)==0 or len(intv)==0: raise RuntimeError('could not identify wall/interface vertices')

    # Pipe axis is z; infer transverse center and radius from unique wall vertices.
    center=np.mean(p[wallv,:2],axis=0)
    rw=np.linalg.norm(p[wallv,:2]-center[None,:],axis=1)
    R=float(np.mean(rw))
    wall_roundness=float((np.max(rw)-np.min(rw))/R)
    if wall_roundness>1e-8:
        raise RuntimeError(f'wall is not circular enough for radial map: relative radius span={wall_roundness:.3e}')

    ri=np.linalg.norm(p[intv,:2]-center[None,:],axis=1)
    r0=float(np.min(ri))
    if not (0<r0<R): raise RuntimeError(f'bad interface anchor r0={r0} R={R}')
    qgap=float(args.wall_gap_scale)
    r0new=R-qgap*(R-r0)
    core_scale=r0new/r0

    # Continuous monotone radial map:
    #   [0,r0] -> uniform enlargement to [0,r0new]
    #   [r0,R] -> shrink every remaining wall gap by exactly qgap.
    xy=p[:,:2]-center[None,:]
    rr=np.linalg.norm(xy,axis=1)
    rrnew=np.empty_like(rr)
    lo=rr<=r0
    rrnew[lo]=core_scale*rr[lo]
    rrnew[~lo]=R-qgap*(R-rr[~lo])
    # Exact wall preservation avoids roundoff moving boundary coordinates.
    rrnew[wallv]=rr[wallv]
    fac=np.ones_like(rr)
    nz=rr>0
    fac[nz]=rrnew[nz]/rr[nz]
    p[:,:2]=center[None,:]+xy*fac[:,None]
    p[wallv,:2]=old['points'][wallv,:2]

    # Stretch original 10D mesh directly to 500D.
    zmin=float(p[:,2].min()); zmax=float(p[:,2].max()); L0=zmax-zmin
    p[:,2]=zmin+args.axial_scale*(p[:,2]-zmin)
    L1=float(p[:,2].max()-p[:,2].min())

    # Rebuild geometry consumed directly by CUDA kernels.
    A['hgeom']=np.asarray([hex_geom(p,h) for h in A['hexes']],dtype=np.float64).reshape((-1,82))
    A['tgeom']=np.asarray([tet_geom(p,t) for t in A['tets']],dtype=np.float64).reshape((-1,12))

    min_hdet=float(np.min(A['hgeom'][:,2:10])) if len(A['hgeom']) else math.inf
    min_tdet=float(np.min(A['tgeom'][:,2])) if len(A['tgeom']) else math.inf
    min_hvol=float(np.min(A['hgeom'][:,0])) if len(A['hgeom']) else math.inf
    min_tvol=float(np.min(A['tgeom'][:,0])) if len(A['tgeom']) else math.inf

    # Geometry diagnostics on the interface wall gap.
    ri_new=np.linalg.norm(p[intv,:2]-center[None,:],axis=1)
    gap0=R-ri; gap1=R-ri_new
    valid=gap0>1e-14*R
    ratios=gap1[valid]/gap0[valid]

    invariant_keys=('hexes','tets','hvel','tvel','hp','tp','irec','wall','inh','int','outh','outt')
    inv={k:(sha(old[k])==sha(A[k])) for k in invariant_keys}
    wall_exact=bool(np.array_equal(p[wallv,:2],old['points'][wallv,:2]))
    z_ok=bool(np.allclose(p[:,2],zmin+args.axial_scale*(old['points'][:,2]-zmin),rtol=0,atol=5e-14))
    checks={
        'connectivity_dof_boundary_records_unchanged':all(inv.values()),
        'wall_xy_bitwise_unchanged':wall_exact,
        'axial_coordinates_scaled':z_ok,
        'length_ratio_matches':abs(L1/L0-args.axial_scale)<=1e-11*max(1.0,args.axial_scale),
        'wall_gap_ratio_matches':bool(len(ratios)>0 and np.max(np.abs(ratios-qgap))<2e-12),
        'positive_hex_gauss_detJ':min_hdet>0,
        'positive_tet_det':min_tdet>0,
        'positive_hex_volume':min_hvol>0,
        'positive_tet_volume':min_tvol>0,
    }
    status='PASS' if all(checks.values()) else 'FAIL'
    write(dst,header,A)
    checks['binary_size_unchanged']=(dst.stat().st_size==src.stat().st_size)
    status='PASS' if all(checks.values()) else 'FAIL'

    pred_mean=args.ref_yplus_mean*qgap
    pred_max=args.ref_yplus_max*qgap
    manifest={
        'status':status,'source':str(src),'output':str(dst),
        'axial_scale':args.axial_scale,'wall_gap_scale':qgap,
        'counts':{'vertices':q[0],'hex_cells':q[1],'tet_cells':q[2],'velocity_dofs':q[5],'pressure_dofs':q[6]},
        'pipe':{'center_xy':center.tolist(),'radius':R,'wall_roundness_rel':wall_roundness,'source_length':L0,'new_length':L1},
        'radial_map':{'interface_anchor_radius':r0,'interface_anchor_radius_new':r0new,'core_uniform_scale':core_scale,
                      'old_gap_min':float(np.min(gap0)),'old_gap_max':float(np.max(gap0)),'old_gap_mean':float(np.mean(gap0)),
                      'new_gap_min':float(np.min(gap1)),'new_gap_max':float(np.max(gap1)),'new_gap_mean':float(np.mean(gap1)),
                      'gap_ratio_min':float(np.min(ratios)),'gap_ratio_max':float(np.max(ratios)),'gap_ratio_mean':float(np.mean(ratios))},
        'rough_yplus_prediction':{'reference_mean':args.ref_yplus_mean,'reference_max':args.ref_yplus_max,
                                  'predicted_mean_if_utau_unchanged':pred_mean,'predicted_max_if_utau_unchanged':pred_max},
        'geometry':{'min_hex_gauss_detJ':min_hdet,'min_tet_det':min_tdet,'min_hex_volume':min_hvol,'min_tet_volume':min_tvol},
        'checks':checks,
        'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
        'output_sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),
    }
    mf=Path(args.manifest).expanduser().resolve() if args.manifest else dst.with_suffix('.manifest.json')
    mf.write_text(json.dumps(manifest,indent=2)+'\n')

    print(f'HXT1_LOWY_SOURCE path={src} vertices={q[0]} hex={q[1]} tet={q[2]} velocityDofs={q[5]} pressureDofs={q[6]} L={L0:.12e} R={R:.12e}')
    print(f'HXT1_LOWY_RADIAL wallGapScale={qgap:.12e} anchorR={r0:.12e} anchorRnew={r0new:.12e} coreScale={core_scale:.12e}')
    print(f'HXT1_LOWY_GAP oldMin={np.min(gap0):.12e} oldMean={np.mean(gap0):.12e} oldMax={np.max(gap0):.12e} newMin={np.min(gap1):.12e} newMean={np.mean(gap1):.12e} newMax={np.max(gap1):.12e} ratioMin={np.min(ratios):.12e} ratioMax={np.max(ratios):.12e}')
    print(f'HXT1_LOWY_YPLUS_ROUGH referenceMean={args.ref_yplus_mean:.9e} referenceMax={args.ref_yplus_max:.9e} predictedMean={pred_mean:.9e} predictedMax={pred_max:.9e} assumption=UNCHANGED_UTAU')
    print(f'HXT1_LOWY_AXIAL scale={args.axial_scale:.12g} Lnew={L1:.12e} L_over_D={L1/(2*R):.9f}')
    print(f'HXT1_LOWY_GEOMETRY minHexGaussDetJ={min_hdet:.12e} minTetDet={min_tdet:.12e} minHexVol={min_hvol:.12e} minTetVol={min_tvol:.12e}')
    print('HXT1_LOWY_INVARIANTS connectivity=UNCHANGED dofs=UNCHANGED boundaries=UNCHANGED wallCoordinates=UNCHANGED geometryPlans=REBUILT')
    print(f'HXT1_LOWY_MANIFEST path={mf}')
    print(f'HXT1_LOWY_STATUS={status}')
    if status!='PASS': raise SystemExit(2)

if __name__=='__main__': main()
