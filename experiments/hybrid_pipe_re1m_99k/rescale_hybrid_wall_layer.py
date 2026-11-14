#!/usr/bin/env python3
"""Create y+150 wall-HEX mesh from the original y+300 HXT0 mesh.

Preserves *all* HEX/TET connectivity and 99,216 cell count. Only transforms
interior radial (x,y) vertex locations: new inner wall radius = R-depth150.
All outer wall points, z positions, boundary labels, and face connectivity
remain exactly unchanged. HXT1 is subsequently rebuilt and audited.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",required=True,type=Path)
    ap.add_argument("--outdir",required=True,type=Path)
    ap.add_argument("--target-yplus",type=float,default=150.0)
    a=ap.parse_args()
    src=a.input.expanduser().resolve()
    meta_path=src.with_name("pipe_geometry.json")
    if not src.is_file() or not meta_path.is_file():
        raise FileNotFoundError("Need previous hybrid HXT0.npz and pipe_geometry.json beside it")
    meta=json.loads(meta_path.read_text())
    D=float(meta["diameter"]);R=D/2
    old_depth=float(meta["wall_hex_depth"])
    old_yplus=float(meta["nominal_midwall_yplus"])
    if old_yplus<=0 or a.target_yplus<=0:raise ValueError("invalid nominal y+")
    new_depth=old_depth*a.target_yplus/old_yplus
    r_old=R-old_depth; r_new=R-new_depth
    if not (0<r_old<R and 0<r_new<R):
        raise ValueError("bad inner interface radius")
    factor=r_new/r_old

    with np.load(src,allow_pickle=False) as npz:
        data={k:npz[k].copy() for k in npz.files}
    pts=data["points"]
    if pts.ndim!=2 or pts.shape[1]!=3:raise ValueError("bad point shape")
    nr=np.linalg.norm(pts[:,:2],axis=1)
    wall=np.abs(nr-R)<1e-8*D
    core=~wall
    if np.any(nr[core]>r_old+1e-8*D):
        raise ValueError("Unexpected interior point outside old HEX/TET interface")
    if not np.any(core) or not np.any(wall):
        raise ValueError("both core and wall point sets required")
    before_wall=pts[wall].copy()
    pts[core,:2]*=factor
    if not np.array_equal(pts[wall],before_wall):
        raise RuntimeError("outer wall vertices altered")
    data["points"]=pts

    nhex=len(data["hexes"]);ntet=len(data["tets"])
    if nhex<=0 or ntet<=0:raise ValueError("not a mixed mesh")
    out=a.outdir.expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    dst=out/"HXT0.npz"
    if src==dst:raise ValueError("will not overwrite the original y+300 mesh")
    np.savez_compressed(dst,**data)
    # Repaired geometry/face orientation is validated by the original HXT1
    # converter; never bypass its positive-Jacobian or matching-face audits.
    digest=hashlib.sha256(src.read_bytes()).hexdigest()
    meta.update(wall_hex_depth=new_depth,
        nominal_midwall_yplus=float(a.target_yplus),
        radial_scale_core=factor,
        parent_hxt0=str(src),
        parent_hxt0_sha256=digest,
        n_hex=nhex,n_tet=ntet,n_total=nhex+ntet)
    (out/"pipe_geometry.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps({
        "old_depth_m":old_depth,"new_depth_m":new_depth,
        "old_sample_distance_m":old_depth/2,
        "new_sample_distance_m":new_depth/2,
        "nominal_yplus":a.target_yplus,"core_radial_scale":factor,
        "unchanged_cell_connectivity":True,
        "wall_points_unchanged":int(wall.sum()),
        "hex":nhex,"tet":ntet,"total":nhex+ntet,
        "hxt0":str(dst)
    },indent=2))

if __name__=="__main__":main()
