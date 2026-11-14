#!/usr/bin/env python3
"""Offline 50D hybrid HEX/TET NodalS P0-pressure Moody comparison.

Adapted from NODALS_PIPE50_POSTPROCESS_20261111, preserving geometrical
cross-section integration. Requires numpy; matplotlib is optional.
Never changes the solver, its checkpoints, or the input VTU.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

HEX_EDGES=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
           (0,4),(1,5),(2,6),(3,7))
TET_EDGES=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))
WINDOWS=((20,50),(20,30),(30,40),(40,50),(40,49))

def colebrook(Re,rr=0.0):
    if Re<=4000 or rr<0: raise ValueError("Invalid turbulent Re/roughness")
    f=0.02
    for _ in range(500):
        fn=(-2*math.log10(rr/3.7+2.51/(Re*math.sqrt(f))))**-2
        if abs(fn-f)<1e-14:return fn
        f=fn
    raise RuntimeError("Colebrook did not converge")

def array(piece,section,names):
    el=piece.find(section)
    if el is None: raise ValueError("Missing VTU "+section)
    aa={a.get("Name"):a for a in el.findall("DataArray")}
    for name in names:
        if name in aa:
            a=aa[name]
            if a.get("format","ascii").lower()!="ascii":
                raise ValueError("Only ASCII VTU arrays supported: "+str(name))
            return np.fromstring(a.text or "",sep=" "),name
    raise ValueError(f"Missing {section}/{names}; available {list(aa)}")

def read_vtu(path):
    piece=ET.parse(path).getroot().find(".//Piece")
    if piece is None:raise ValueError("No VTU Piece")
    n=int(piece.get("NumberOfCells"))
    pts,_=array(piece,"Points",(None,))
    if pts.size%3:raise ValueError("Invalid point shape")
    pts=pts.reshape(-1,3)
    conn,_=array(piece,"Cells",("connectivity",))
    offsets,_=array(piece,"Cells",("offsets",))
    types,_=array(piece,"Cells",("types",))
    p,field=array(piece,"CellData",("pressure_Q0P0","p_P0","pressure"))
    conn=conn.astype(np.int64);offsets=offsets.astype(np.int64)
    types=types.astype(np.uint8)
    if len(offsets)!=n or len(types)!=n or len(p)!=n:
        raise ValueError("VTU cell arrays have different lengths")
    if not np.all(np.isin(types,(10,12))):
        raise ValueError("Only HEX(12) and TET(10) VTK cells supported")
    starts=np.r_[0,offsets[:-1]]
    if not np.all(offsets-starts==np.where(types==12,8,4)):
        raise ValueError("Connectivity offsets disagree with VTK types")
    if offsets[-1]!=len(conn) or np.any(conn<0) or np.any(conn>=len(pts)):
        raise ValueError("Bad VTU connectivity")
    if not np.isfinite(pts).all() or not np.isfinite(p).all():
        raise ValueError("Nonfinite VTU points or P0 pressure")
    groups=[]
    for t,nv,edges in ((10,4,TET_EDGES),(12,8,HEX_EDGES)):
        ix=np.where(types==t)[0]
        groups.append((conn[starts[ix,None]+np.arange(nv)[None,:]],p[ix],edges))
    print(f"PIPE50_HYBRID_VTU pressure=CellData/{field} TET={len(groups[0][0])} HEX={len(groups[1][0])}")
    return pts,groups

def area_polygon(xy):
    xy=np.asarray(xy)
    c=xy.mean(axis=0)
    xy=xy[np.argsort(np.arctan2(xy[:,1]-c[1],xy[:,0]-c[0]))]
    return 0.5*abs(float(np.dot(xy[:,0],np.roll(xy[:,1],-1))-
                         np.dot(xy[:,1],np.roll(xy[:,0],-1))))

def cut_area(xyz,z,edges):
    xy=[]
    for i,j in edges:
        a,b=xyz[i,2]-z,xyz[j,2]-z
        if a*b<0:
            t=-a/(b-a)
            xy.append(xyz[i,:2]+t*(xyz[j,:2]-xyz[i,:2]))
    if len(xy)<3:
        raise ValueError(f"Degenerate HEX/TET plane intersection at z={z}")
    return area_polygon(xy)

class Slices:
    def __init__(self,pts,groups,D):
        self.pts=pts;self.groups=groups;self.D=D
        self.range=[(pts[cells,2].min(axis=1),pts[cells,2].max(axis=1))
                    for cells,_,_ in groups]
        base=pts[np.abs(pts[:,2]-pts[:,2].min())<1.e-9*D,:2]
        rad=np.linalg.norm(base,axis=1)
        ring=base[np.abs(rad-rad.max())<1.e-8*D]
        if len(ring)<8:raise ValueError("Missing circular wall ring")
        ring=ring[np.argsort(np.arctan2(ring[:,1],ring[:,0]))]
        self.area=area_polygon(ring)
        self.perimeter=float(np.linalg.norm(ring-np.roll(ring,-1,axis=0),axis=1).sum())
        if abs(self.area/(math.pi*(D/2)**2)-1)>0.01:
            raise ValueError("Mesh section area differs more than 1% from circular pipe")
        self.cache={}
    def get(self,z):
        key=round(float(z/self.D),12)
        if key in self.cache:return self.cache[key]
        weighted=0.;area=0.;nt=0;nh=0
        for (cells,p,edges),(lo,hi),idgroup in zip(self.groups,self.range,range(2)):
            ix=np.flatnonzero((lo<z)&(hi>z))
            areas=np.asarray([cut_area(self.pts[cells[i]],z,edges) for i in ix])
            weighted+=float(np.dot(areas,p[ix]))
            area+=float(areas.sum())
            if idgroup==0:nt=len(ix)
            else:nh=len(ix)
        if area<=0 or abs(area/self.area-1)>0.005:
            raise ValueError(f"Section area mismatch at z/D={z/self.D}: {area} vs {self.area}")
        val=(weighted/area,nt,nh)
        self.cache[key]=val
        return val

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("vtu",type=Path)
    ap.add_argument("--diameter",type=float,default=0.889)
    ap.add_argument("--bulk",type=float,default=16.431211404701912)
    ap.add_argument("--re",type=float,default=1e6)
    ap.add_argument("--relative-roughness",type=float,default=0.)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--no-plot",action="store_true")
    a=ap.parse_args()
    D,U,Re=a.diameter,a.bulk,a.re
    if D<=0 or U<=0 or Re<=4000:ap.error("Invalid D/U/Re")
    if not a.vtu.is_file():ap.error("VTU file does not exist: "+str(a.vtu))
    pts,groups=read_vtu(a.vtu)
    if abs(pts[:,2].min())>D*1e-8 or abs(pts[:,2].max()-50*D)>D*1e-8:
        raise ValueError("Expect pipe from z=0 to z=50D")
    ss=Slices(pts,groups,D)
    zgrid=np.unique(pts[:,2]); mid=(zgrid[:-1]+zgrid[1:])/2
    if len(mid)<15:raise ValueError("Too few axial mesh slabs")
    p=np.array([ss.get(z)[0] for z in mid])
    x=mid/D
    def at(zd):
        z=zd*D
        i=int(np.clip(np.searchsorted(mid,z),1,len(mid)-1))
        za,zb=mid[i-1:i+1];pa,pb=p[i-1:i+1]
        return float(pa+(pb-pa)*(z-za)/(zb-za))
    fref=colebrook(Re,a.relative_roughness)
    rows=[]
    for low,high in WINDOWS:
        pa,pb=at(low),at(high)
        f_end=2*(pa-pb)/((high-low)*U*U)
        mask=(x>=low)&(x<=high)
        if mask.sum()<4:raise ValueError(f"Too few mids in {low}-{high}D")
        slope,intercept=np.polyfit(x[mask],p[mask],1)
        predicted=slope*x[mask]+intercept
        den=float(np.sum((p[mask]-p[mask].mean())**2))
        r2=1-float(np.sum((p[mask]-predicted)**2))/den if den else float("nan")
        f_fit=-2*float(slope)/(U*U)
        rows.append(dict(window=f"{low}-{high}D",start_D=low,end_D=high,
            P0_pressure_start=pa,P0_pressure_end=pb,
            delta_p_kinematic=pa-pb,
            delta_p_Moody_kinematic=fref*(high-low)*U*U/2,
            Darcy_f_endpoint=f_end,
            error_endpoint_pct=100*(f_end/fref-1),
            Darcy_f_fitted=f_fit,error_fitted_pct=100*(f_fit/fref-1),
            fit_R2=r2,fit_sections=int(mask.sum())))
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    with (out/"local_pressure_friction.csv").open("w",newline="") as fp:
        w=csv.DictWriter(fp,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    with (out/"axial_pressure_20to50D.csv").open("w",newline="") as fp:
        w=csv.writer(fp)
        w.writerow(("z_over_D","section_P0_pressure","intersected_tets","intersected_hexes"))
        for zd,zz,pp in zip(x,mid,p):
            if 19.5<=zd<=50:
                _,nt,nh=ss.get(zz)
                w.writerow((zd,pp,nt,nh))
    obj=dict(vtu=str(a.vtu),cells_tet=len(groups[0][0]),cells_hex=len(groups[1][0]),
        Re=Re,D_m=D,U_bulk=U,Moody_Darcy_f=fref,
        mesh_area=ss.area,mesh_hydraulic_D=4*ss.area/ss.perimeter,
        windows=rows,method="P0 geometric mixed-cell sections; slab-midpoint interpolation and separate local linear fits")
    (out/"result.json").write_text(json.dumps(obj,indent=2)+"\n")
    print("MOODY_REFERENCE_DARCY",f"{fref:.9f}")
    print("ZONE             f_endpoint  error_end%     f_fit    error_fit%    R2")
    for r in rows:
        print(f"{r['window']:>8}  {r['Darcy_f_endpoint']:.9f} {r['error_endpoint_pct']:+9.3f}%  "
              f"{r['Darcy_f_fitted']:.9f} {r['error_fitted_pct']:+9.3f}%  {r['fit_R2']:.7f}")
    print("POSTPROCESS_OUTPUT",out)
    if not a.no_plot:
        try:import matplotlib.pyplot as plt
        except ImportError:print("PLOT_SKIPPED: matplotlib unavailable")
        else:
            fig,ax=plt.subplots(figsize=(10,5))
            mm=(x>=20)&(x<=50)
            ax.plot(x[mm],p[mm],"o-",label="P0 area-mean section pressure")
            for r in rows[1:4]:
                lo,hi=r["start_D"],r["end_D"]
                ix=(x>=lo)&(x<=hi)
                slope,bias=np.polyfit(x[ix],p[ix],1)
                xx=np.array((lo,hi))
                ax.plot(xx,slope*xx+bias,"--",
                        label=f"{lo}-{hi}D fit ({r['error_fitted_pct']:+.2f}%)")
            ax.grid(alpha=.3);ax.legend()
            ax.set(xlabel="z/D",ylabel="Kinematic pressure (m²/s²)",
                   title="Hybrid HEX/TET: Moody friction from local pressure gradients")
            fig.tight_layout();fig.savefig(out/"pressure_20to50D.png",dpi=160)
            plt.close(fig)

if __name__=="__main__":main()
