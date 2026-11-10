#!/usr/bin/env python3
"""Lossless-topology conversion: ASCII OpenFOAM tetrahedral polyMesh to Gmsh MSH2.2.

Copies the same point coordinates, the same 4-node cells and the original
boundary triangles. Refuses unsupported non-tetrahedra or unknown patches.
It is a mesh-conversion helper, NOT a CFD solver or remesher.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import re
import sys


def read_foam_list(path: Path):
    text = path.read_text(encoding="utf-8")
    match = re.search(r"(?m)^\s*(\d+)\s*\n\s*\(\s*\n", text)
    if not match:
        raise ValueError(f"no ASCII Foam list in {path}")
    count = int(match.group(1))
    body = text[match.end():]
    close = re.search(r"(?m)^\s*\)\s*$", body)
    if not close:
        raise ValueError(f"unclosed Foam list in {path}")
    return count, [line.strip() for line in body[:close.start()].splitlines() if line.strip()]


def parse_mesh(poly: Path):
    pn, points_raw = read_foam_list(poly / "points")
    fn, faces_raw = read_foam_list(poly / "faces")
    on, owners_raw = read_foam_list(poly / "owner")
    nn, nbrs_raw = read_foam_list(poly / "neighbour")
    if any((len(lines) != n) for lines,n in
           ((points_raw,pn),(faces_raw,fn),(owners_raw,on),(nbrs_raw,nn))):
        raise ValueError("Foam header and data counts disagree")
    if on != fn or nn >= on:
        raise ValueError("invalid face/owner/neighbour cardinality")
    points = []
    for line in points_raw:
        x = re.fullmatch(r"\(\s*([^\s()]+)\s+([^\s()]+)\s+([^\s()]+)\s*\)", line)
        if not x:
            raise ValueError("unsupported point record: " + line)
        points.append(tuple(float(v) for v in x.groups()))
    faces = []
    for line in faces_raw:
        t = re.fullmatch(r"(\d+)\(\s*([^()]*)\s*\)", line)
        if not t:
            raise ValueError("unsupported face record: " + line)
        ids = tuple(int(v) for v in t.group(2).split())
        if len(ids) != int(t.group(1)) or len(ids) != 3:
            raise ValueError("only pure tetrahedral triangular-face meshes are supported")
        if any(v < 0 or v >= pn for v in ids):
            raise ValueError("face points out of range")
        faces.append(ids)
    owner = [int(v) for v in owners_raw]
    neighbour = [int(v) for v in nbrs_raw]
    nc = max(owner + neighbour) + 1
    boundary_text = (poly / "boundary").read_text(encoding="utf-8")
    patch_count_match = re.search(r"(?m)^\s*(\d+)\s*\n\s*\(\s*\n", boundary_text)
    if not patch_count_match:
        raise ValueError("no boundary patch list")
    rest = boundary_text[patch_count_match.end():]
    matches = re.findall(r"(?m)^\s*([\w.:-]+)\s*\n\s*\{([^}]*)\}",rest)
    patches = {}
    for name, body in matches:
        s = re.search(r"\bstartFace\s+(\d+)\s*;",body)
        n = re.search(r"\bnFaces\s+(\d+)\s*;",body)
        if s and n:
            patches[name] = (int(s.group(1)), int(n.group(1)))
    if len(patches) != int(patch_count_match.group(1)):
        raise ValueError("boundary patch count differs from Foam header")
    return points, faces, owner, neighbour, nc, patches


def signed_six_vol(a,b,c,d):
    ab = [b[i]-a[i] for i in range(3)]
    ac = [c[i]-a[i] for i in range(3)]
    ad = [d[i]-a[i] for i in range(3)]
    return (ab[0]*(ac[1]*ad[2]-ac[2]*ad[1])
            -ab[1]*(ac[0]*ad[2]-ac[2]*ad[0])
            +ab[2]*(ac[0]*ad[1]-ac[1]*ad[0]))


def convert(poly: Path, dest: Path, mapping: dict[str,int]):
    points,faces,owner,neighbour,nc,patches=parse_mesh(poly)
    if set(patches) != set(mapping):
        raise ValueError(f"boundary mismatch: found {list(patches)}, expected {list(mapping)}")
    face_to_tag = {}
    for name,(start,n) in patches.items():
        if start < len(neighbour) or start+n > len(faces):
            raise ValueError(f"invalid boundary range for {name}")
        for f in range(start,start+n):
            if f in face_to_tag:
                raise ValueError("overlapping boundary ranges")
            face_to_tag[f] = mapping[name]
    if len(face_to_tag) + len(neighbour) != len(faces):
        raise ValueError("boundary ranges do not cover all external faces")

    cell_vertices=[set() for _ in range(nc)]
    cell_faces=[0]*nc
    for i,face in enumerate(faces):
        co=owner[i]
        cell_vertices[co].update(face); cell_faces[co]+=1
        if i<len(neighbour):
            cn=neighbour[i]
            if cn==co:
                raise ValueError("self-neighbour cell")
            cell_vertices[cn].update(face); cell_faces[cn]+=1
    if any(len(v)!=4 or k!=4 for v,k in zip(cell_vertices,cell_faces)):
        samples=[(j,len(v),cell_faces[j]) for j,v in enumerate(cell_vertices)
                 if len(v)!=4 or cell_faces[j]!=4][:8]
        raise ValueError(f"not pure tet mesh; examples: {samples}")

    dest.parent.mkdir(parents=True,exist_ok=True)
    nneg=0
    volsum=0.0
    with dest.open("w",encoding="utf-8") as f:
        f.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n')
        f.write('$PhysicalNames\n4\n2 1 "inlet"\n2 2 "outlet"\n2 3 "wall"\n3 4 "fluid"\n$EndPhysicalNames\n')
        f.write(f'$Nodes\n{len(points)}\n')
        for i,(x,y,z) in enumerate(points,1):
            f.write(f'{i} {x:.17g} {y:.17g} {z:.17g}\n')
        f.write('$EndNodes\n')
        f.write(f'$Elements\n{len(face_to_tag)+nc}\n')
        eid=0
        for fi in sorted(face_to_tag):
            eid+=1
            tri=faces[fi]
            p=face_to_tag[fi]
            f.write(f'{eid} 2 2 {p} {p} {tri[0]+1} {tri[1]+1} {tri[2]+1}\n')
        for vset in cell_vertices:
            a,b,c,d=sorted(vset)
            vol=signed_six_vol(points[a],points[b],points[c],points[d])
            if vol==0.0:
                raise ValueError(f"zero-volume tetra near cell {eid}")
            if vol<0.0:
                b,c=c,b; vol=-vol;nneg+=1
            volsum+=vol/6.0
            eid+=1
            f.write(f'{eid} 4 2 4 4 {a+1} {b+1} {c+1} {d+1}\n')
        f.write('$EndElements\n')
    c=Counter(face_to_tag.values())
    print(f'FOAM_TO_MSH2 status=PASS nodes={len(points)} cells={nc} '
          f'internalFaces={len(neighbour)} boundaryFaces={len(face_to_tag)} '
          f'tag1_inlet={c[1]} tag2_outlet={c[2]} tag3_wall={c[3]} '
          f'reorientedTets={nneg} positiveTotalVolume={volsum:.12g} output={dest}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mesh',type=Path,help='OpenFOAM polyMesh directory or case/constant/polyMesh')
    p.add_argument('-o','--output',type=Path,required=True)
    p.add_argument('--wall',default='patch_0_0')
    p.add_argument('--inlet',default='patch_2_0')
    p.add_argument('--outlet',default='patch_1_0')
    args=p.parse_args()
    poly=args.mesh
    if (poly/'constant/polyMesh').is_dir():poly=poly/'constant/polyMesh'
    mapping={args.wall:3,args.inlet:1,args.outlet:2}
    if len(mapping)!=3:raise SystemExit('patch names must be distinct')
    convert(poly,args.output,mapping)
if __name__=='__main__':
    try:main()
    except (OSError,ValueError) as exc:
        print(f'FOAM_TO_MSH2 status=FAIL reason={exc}',file=sys.stderr)
        raise SystemExit(2)
