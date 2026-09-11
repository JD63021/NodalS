#!/usr/bin/env python3
import argparse, csv, math
from pathlib import Path
import numpy as np

def read_wall(path):
    rows=[]
    with open(path,newline="") as f:
        for r in csv.DictReader(f):
            q={k:v for k,v in r.items()}
            for k in ["z_over_D","area","u_trace_signed","slip","u_tau_spalding","u_tau2_spalding",
                      "y_plus_spalding","beta_spalding","f_spalding","u_tau_applied",
                      "u_tau2_applied","beta_applied","f_applied"]:
                q[k]=float(q[k])
            rows.append(q)
    return rows

def read_pairs(path):
    a=np.genfromtxt(path,delimiter=",",names=True)
    return a

def aw(rows,key,m):
    vals=np.array([r[key] for r in rows],float)
    area=np.array([r["area"] for r in rows],float)
    if not np.any(m): return math.nan
    return float(np.sum(vals[m]*area[m])/np.sum(area[m]))

def summarize(case, wall, pair, z0, z1, ref):
    zw=np.array([r["z_over_D"] for r in wall],float)
    wm=(zw>=z0)&(zw<z1)
    zp=np.asarray(pair["z_mid_over_D"],float)
    pm=(zp>=z0)&(zp<z1)
    fp=float(np.mean(pair["f_Darcy_local"][pm])) if np.any(pm) else math.nan
    fpm=float(np.median(pair["f_Darcy_local"][pm])) if np.any(pm) else math.nan
    fsp=aw(wall,"f_spalding",wm)
    fap=aw(wall,"f_applied",wm)
    slip=aw(wall,"slip",wm)
    yp=aw(wall,"y_plus_spalding",wm)
    return dict(case=case,z0D=z0,z1D=z1,
                pressurePairs=int(np.count_nonzero(pm)),wallFaces=int(np.count_nonzero(wm)),
                fPressure=fp,fPressureMedian=fpm,fWallSpalding=fsp,fWallApplied=fap,
                pressureErrPct=100*(fp/ref-1),wallSpaldingErrPct=100*(fsp/ref-1),
                wallAppliedErrPct=100*(fap/ref-1),
                slipMean=slip,yPlusSpaldingMean=yp)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--baseline-wall",required=True)
    ap.add_argument("--forced-wall",required=True)
    ap.add_argument("--baseline-pairs",required=True)
    ap.add_argument("--forced-pairs",required=True)
    ap.add_argument("--reference-f",type=float,default=0.0284003)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    bw=read_wall(a.baseline_wall); fw=read_wall(a.forced_wall)
    bp=read_pairs(a.baseline_pairs); fp=read_pairs(a.forced_pairs)
    windows=[(12,14),(14,16),(16,18),(18,19.8)]
    rows=[]
    for z0,z1 in windows:
        rb=summarize("baseline_spalding",bw,bp,z0,z1,a.reference_f)
        rf=summarize("reference_shear",fw,fp,z0,z1,a.reference_f)
        rows += [rb,rf]
        print("NODALS_WALL_CAUSAL_WINDOW "
              f"case=baseline z0D={z0:.2f} z1D={z1:.2f} "
              f"fPressure={rb['fPressure']:.10f} fWallSpalding={rb['fWallSpalding']:.10f} "
              f"fWallApplied={rb['fWallApplied']:.10f} pressureErrPct={rb['pressureErrPct']:.6f} "
              f"slipMean={rb['slipMean']:.8f} yPlusSpaldingMean={rb['yPlusSpaldingMean']:.8f} status=PASS")
        print("NODALS_WALL_CAUSAL_WINDOW "
              f"case=reference_shear z0D={z0:.2f} z1D={z1:.2f} "
              f"fPressure={rf['fPressure']:.10f} fWallSpaldingFromTrace={rf['fWallSpalding']:.10f} "
              f"fWallApplied={rf['fWallApplied']:.10f} pressureErrPct={rf['pressureErrPct']:.6f} "
              f"slipMean={rf['slipMean']:.8f} yPlusSpaldingMean={rf['yPlusSpaldingMean']:.8f} status=PASS")
        print("NODALS_WALL_CAUSAL_DELTA "
              f"z0D={z0:.2f} z1D={z1:.2f} "
              f"deltaPressureF={rf['fPressure']-rb['fPressure']:.10e} "
              f"deltaPressureErrPct={rf['pressureErrPct']-rb['pressureErrPct']:.6f} "
              f"deltaSlip={rf['slipMean']-rb['slipMean']:.10e} "
              f"forcedPressureMinusAppliedF={rf['fPressure']-rf['fWallApplied']:.10e} status=PASS")
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
    print(f"NODALS_WALL_CAUSAL_OUTPUT csv={out} status=PASS")

if __name__=="__main__":
    main()
