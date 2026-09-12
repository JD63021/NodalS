#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, struct
from collections import defaultdict
from pathlib import Path
import numpy as np

MAGIC=b"HXT1BIN1"
VERSION=1

HEX_FACES=(
    (0,1,2,3),       # z-
    (4,5,6,7),       # z+
    (0,1,5,4),
    (1,2,6,5),
    (2,3,7,6),
    (3,0,4,7),
)
TET_FACES=(
    (1,2,3),         # opposite vertex 0
    (0,3,2),         # opposite vertex 1
    (0,1,3),         # opposite vertex 2
    (0,2,1),         # opposite vertex 3
)
HEX_SIGNS=np.asarray([
    (-1.,-1.,-1.),(1.,-1.,-1.),(1.,1.,-1.),(-1.,1.,-1.),
    (-1.,-1., 1.),(1.,-1., 1.),(1.,1., 1.),(-1.,1., 1.)
],dtype=np.float64)

def key(conn):
    return tuple(sorted(map(int,conn)))

def inv3(J):
    return np.linalg.inv(J)

def hex_geom(points, h):
    p=points[np.asarray(h,dtype=np.int64)]
    h2=0.0
    for a in range(8):
        for b in range(a+1,8):
            h2=max(h2,float(np.dot(p[a]-p[b],p[a]-p[b])))
    g=1.0/math.sqrt(3.0)
    dets=[]
    invs=[]
    vol=0.0
    sx,sy,sz=HEX_SIGNS[:,0],HEX_SIGNS[:,1],HEX_SIGNS[:,2]
    for xi in (-g,g):
        for eta in (-g,g):
            for zeta in (-g,g):
                dxi  =0.125*sx*(1.0+sy*eta)*(1.0+sz*zeta)
                deta =0.125*sy*(1.0+sx*xi )*(1.0+sz*zeta)
                dzeta=0.125*sz*(1.0+sx*xi )*(1.0+sy*eta)
                J=np.empty((3,3),dtype=np.float64)
                J[:,0]=dxi@p
                J[:,1]=deta@p
                J[:,2]=dzeta@p
                d=float(np.linalg.det(J))
                dets.append(d)
                invs.extend(inv3(J).reshape(-1).tolist())
                vol += d
    return [vol,h2,*dets,*invs]

def tet_geom(points,t):
    p=points[np.asarray(t,dtype=np.int64)]
    J=np.column_stack((p[1]-p[0],p[2]-p[0],p[3]-p[0]))
    det=float(np.linalg.det(J))
    vol=det/6.0
    h2=max(float(np.dot(p[a]-p[b],p[a]-p[b]))
           for a in range(4) for b in range(a+1,4))
    I=np.linalg.inv(J)
    return [vol,h2,det,*I.reshape(-1).tolist()]

def bf2_value(lf,xi,eta,zeta):
    b=lambda s: 1.0-s*s
    if lf==0: return 0.5*(1.0-zeta)*b(xi)*b(eta)
    if lf==1: return 0.5*(1.0+zeta)*b(xi)*b(eta)
    if lf==2: return 0.5*(1.0-eta)*b(xi)*b(zeta)
    if lf==3: return 0.5*(1.0+xi)*b(eta)*b(zeta)
    if lf==4: return 0.5*(1.0+eta)*b(xi)*b(zeta)
    if lf==5: return 0.5*(1.0-xi)*b(eta)*b(zeta)
    raise ValueError(lf)

def bf2_audit():
    centers=[
        (0,0,-1),(0,0,1),(0,-1,0),(1,0,0),(0,1,0),(-1,0,0)
    ]
    M=np.asarray([[bf2_value(i,*x) for i in range(6)] for x in centers])
    cc=np.asarray([bf2_value(i,0,0,0) for i in range(6)])
    return float(np.max(np.abs(M-np.eye(6)))), float(np.max(np.abs(cc-0.5)))

def build_face_owners(cells,faces):
    owners=defaultdict(list)
    for c,conn in enumerate(cells):
        for lf,loc in enumerate(faces):
            k=key(conn[list(loc)])
            owners[k].append((c,lf))
    return owners

def map_boundary(records,owners,face_gid):
    out=[]
    for rec in records:
        k=key(rec)
        oo=owners.get(k,[])
        if len(oo)!=1:
            raise RuntimeError(f"boundary face {k} owner count {len(oo)} != 1")
        c,lf=oo[0]
        out.append((c,lf,face_gid[k]))
    return np.asarray(out,dtype=np.int32).reshape((-1,3))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True,help="HXT0_mesh.npz")
    ap.add_argument("--outdir",required=True)
    args=ap.parse_args()
    inp=Path(args.input)
    outdir=Path(args.outdir)
    outdir.mkdir(parents=True,exist_ok=True)

    z=np.load(inp)
    points=np.asarray(z["points"],dtype=np.float64)
    hexes=np.asarray(z["hexes"],dtype=np.int32)
    tets=np.asarray(z["tets"],dtype=np.int32)
    iq=np.asarray(z["interface_quads"],dtype=np.int32)
    it=np.asarray(z["interface_triangles"],dtype=np.int32)
    wall=np.asarray(z["wall_quads"],dtype=np.int32)
    in_h=np.asarray(z["inlet_hex_quads"],dtype=np.int32)
    in_t=np.asarray(z["inlet_tet_triangles"],dtype=np.int32)
    out_h=np.asarray(z["outlet_hex_quads"],dtype=np.int32)
    out_t=np.asarray(z["outlet_tet_triangles"],dtype=np.int32)

    nv=len(points); nh=len(hexes); nt=len(tets)

    howners=build_face_owners(hexes,HEX_FACES)
    towners=build_face_owners(tets,TET_FACES)

    if any(len(v)>2 for v in howners.values()):
        raise RuntimeError("non-manifold HEX face")
    if any(len(v)>2 for v in towners.values()):
        raise RuntimeError("non-manifold TET face")

    hkeys=sorted(howners)
    tkeys=sorted(towners)
    hgid={k:nv+i for i,k in enumerate(hkeys)}
    tgid={k:nv+len(hkeys)+i for i,k in enumerate(tkeys)}
    nvel=nv+len(hkeys)+len(tkeys)
    npressure=nh+nt

    hvel=np.empty((nh,14),dtype=np.int32)
    hvel[:,:8]=hexes
    for c,h in enumerate(hexes):
        for lf,loc in enumerate(HEX_FACES):
            hvel[c,8+lf]=hgid[key(h[list(loc)])]

    tvel=np.empty((nt,8),dtype=np.int32)
    tvel[:,:4]=tets
    for c,t in enumerate(tets):
        for lf,loc in enumerate(TET_FACES):
            tvel[c,4+lf]=tgid[key(t[list(loc)])]

    hp=np.arange(nh,dtype=np.int32)
    tp=np.arange(nh,nh+nt,dtype=np.int32)

    # HXT0 interface -> exact cell/local-face/bubble map.
    if len(iq)!=len(it):
        raise RuntimeError("interface quad/triangle record mismatch")
    irec=np.empty((len(iq),9),dtype=np.int32)
    max_vertex_set_diff=0
    for r in range(len(iq)):
        qk=key(iq[r])
        ho=howners.get(qk,[])
        if len(ho)!=1:
            raise RuntimeError(f"interface HEX face owner count {len(ho)} != 1 at {r}")
        hc,hlf=ho[0]
        row=[hc,hlf,hgid[qk]]
        qset=set(map(int,iq[r]))
        union=set()
        for side in range(2):
            tk=key(it[r,side])
            to=towners.get(tk,[])
            if len(to)!=1:
                raise RuntimeError(f"interface TET face owner count {len(to)} != 1 at {r}/{side}")
            tc,tlf=to[0]
            row.extend([tc,tlf,tgid[tk]])
            union.update(map(int,it[r,side]))
        max_vertex_set_diff=max(max_vertex_set_diff,len(qset.symmetric_difference(union)))
        irec[r,:]=row

    wallrec=map_boundary(wall,howners,hgid)
    inhexrec=map_boundary(in_h,howners,hgid)
    intetrec=map_boundary(in_t,towners,tgid)
    outhexrec=map_boundary(out_h,howners,hgid)
    outtetrec=map_boundary(out_t,towners,tgid)

    # Geometry plans. Layout is deliberately POD-friendly for CUDA upload.
    hgeom=np.asarray([hex_geom(points,h) for h in hexes],dtype=np.float64)
    tgeom=np.asarray([tet_geom(points,t) for t in tets],dtype=np.float64)
    assert hgeom.shape==(nh,82)
    assert tgeom.shape==(nt,12)

    bf2_face_err,bf2_center_err=bf2_audit()

    hinc=np.asarray([len(howners[k]) for k in hkeys],dtype=np.int32)
    tinc=np.asarray([len(towners[k]) for k in tkeys],dtype=np.int32)
    interface_hex_bubbles=set(map(int,irec[:,2]))
    interface_tet_bubbles=set(map(int,irec[:,5]))|set(map(int,irec[:,8]))

    checks={
        "hex_local_arity_14": hvel.shape[1]==14,
        "tet_local_arity_8": tvel.shape[1]==8,
        "pressure_one_per_cell": npressure==nh+nt,
        "velocity_gid_contiguous": int(max(hvel.max(),tvel.max()))==nvel-1 and int(min(hvel.min(),tvel.min()))==0,
        "hex_face_incidence_1_or_2": bool(np.all((hinc==1)|(hinc==2))),
        "tet_face_incidence_1_or_2": bool(np.all((tinc==1)|(tinc==2))),
        "interface_vertex_sets_exact": max_vertex_set_diff==0,
        "interface_hex_bubble_unique": len(interface_hex_bubbles)==len(iq),
        "interface_tet_bubble_unique": len(interface_tet_bubbles)==2*len(iq),
        "hex_tet_face_dof_ranges_disjoint": (nv+len(hkeys)-1)<(nv+len(hkeys)),
        "hex_detJ_positive": float(np.min(hgeom[:,2:10]))>0.0,
        "tet_det_positive": float(np.min(tgeom[:,2]))>0.0,
        "hex_volume_positive": float(np.min(hgeom[:,0]))>0.0,
        "tet_volume_positive": float(np.min(tgeom[:,0]))>0.0,
        "bf2_face_kronecker": bf2_face_err<1e-15,
        "bf2_center_half": bf2_center_err<1e-15,
        "wall_faces_mapped": len(wallrec)==len(wall),
        "inlet_faces_mapped": len(inhexrec)==len(in_h) and len(intetrec)==len(in_t),
        "outlet_faces_mapped": len(outhexrec)==len(out_h) and len(outtetrec)==len(out_t),
    }
    status="PASS" if all(checks.values()) else "FAIL"

    header=(nv,nh,nt,len(hkeys),len(tkeys),nvel,npressure,len(irec),
            len(wallrec),len(inhexrec),len(intetrec),len(outhexrec),len(outtetrec))
    binpath=outdir/"HXT1_mesh.bin"
    with binpath.open("wb") as f:
        f.write(struct.pack("<8sII13Q",MAGIC,VERSION,0,*header))
        for a in (points,hexes,tets,hvel,tvel,hp,tp,hgeom,tgeom,irec,
                  wallrec,inhexrec,intetrec,outhexrec,outtetrec):
            f.write(np.ascontiguousarray(a).tobytes(order="C"))

    meta={
        "status":status,
        "input":str(inp),
        "counts":{
            "vertices":nv,"hex_cells":nh,"tet_cells":nt,
            "hex_face_bubbles":len(hkeys),"tet_face_bubbles":len(tkeys),
            "scalar_velocity_dofs":nvel,"pressure_dofs":npressure,
            "interface_records":len(irec),
            "wall_hex_faces":len(wallrec),
            "inlet_hex_faces":len(inhexrec),"inlet_tet_faces":len(intetrec),
            "outlet_hex_faces":len(outhexrec),"outlet_tet_faces":len(outtetrec),
        },
        "geometry":{
            "min_hex_gauss_detJ":float(np.min(hgeom[:,2:10])),
            "min_tet_det":float(np.min(tgeom[:,2])),
            "hex_volume_sum":float(np.sum(hgeom[:,0])),
            "tet_volume_sum":float(np.sum(tgeom[:,0])),
        },
        "bf2":{
            "contract":"Q1 endpoint in face-normal coordinate times Q2 midpoint bubbles in the two tangential coordinates",
            "face_center_kronecker_max_abs":bf2_face_err,
            "cell_center_value_minus_half_max_abs":bf2_center_err,
        },
        "checks":checks,
    }
    (outdir/"HXT1_manifest.json").write_text(json.dumps(meta,indent=2)+"\n")

    lines=[]
    def p(s=""):
        print(s); lines.append(s)
    p("HXT1 NODALS MIXED HEX/TET GEOMETRY + DOF AUDIT")
    p("================================================")
    p(f"HXT1_HOST_STATUS={status}")
    p(f"N_VERTICES={nv}")
    p(f"N_HEX={nh}")
    p(f"N_TET={nt}")
    p(f"N_HEX_FACE_BUBBLES={len(hkeys)}")
    p(f"N_TET_FACE_BUBBLES={len(tkeys)}")
    p(f"N_SCALAR_VELOCITY_DOFS={nvel}")
    p(f"N_PRESSURE_DOFS={npressure}")
    p("HEX_LOCAL_SCALAR_ARITY=14")
    p("TET_LOCAL_SCALAR_ARITY=8")
    p(f"N_INTERFACE_RECORDS={len(irec)}")
    p(f"N_INTERFACE_HEX_BUBBLES={len(interface_hex_bubbles)}")
    p(f"N_INTERFACE_TET_BUBBLES={len(interface_tet_bubbles)}")
    p(f"N_WALL_HEX_FACES={len(wallrec)}")
    p(f"N_INLET_HEX_FACES={len(inhexrec)}")
    p(f"N_INLET_TET_FACES={len(intetrec)}")
    p(f"N_OUTLET_HEX_FACES={len(outhexrec)}")
    p(f"N_OUTLET_TET_FACES={len(outtetrec)}")
    p(f"MIN_HEX_GAUSS_DETJ={float(np.min(hgeom[:,2:10])):.16e}")
    p(f"MIN_TET_DET={float(np.min(tgeom[:,2])):.16e}")
    p(f"HEX_VOLUME_SUM={float(np.sum(hgeom[:,0])):.16e}")
    p(f"TET_VOLUME_SUM={float(np.sum(tgeom[:,0])):.16e}")
    p(f"TOTAL_VOLUME={float(np.sum(hgeom[:,0])+np.sum(tgeom[:,0])):.16e}")
    p(f"BF2_FACE_CENTER_KRONECKER_MAXABS={bf2_face_err:.16e}")
    p(f"BF2_CELL_CENTER_HALF_MAXABS={bf2_center_err:.16e}")
    for name,ok in checks.items():
        p(f"HXT1_CHECK {name}={'PASS' if ok else 'FAIL'}")
    p(f"HXT1_HOST_STATUS={status}")
    (outdir/"HXT1_AUDIT_HOST.txt").write_text("\n".join(lines)+"\n")
    if status!="PASS":
        raise SystemExit(3)

if __name__=="__main__":
    main()
