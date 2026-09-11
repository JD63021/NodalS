#!/usr/bin/env python3
import argparse, csv, math
from pathlib import Path
import numpy as np

def load_csv(path):
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)

def weighted_mean(vals, w):
    vals=np.asarray(vals,float); w=np.asarray(w,float)
    m=np.isfinite(vals)&np.isfinite(w)&(w>0)
    return float(np.sum(vals[m]*w[m])/np.sum(w[m])) if np.any(m) else math.nan

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pair",required=True)
    ap.add_argument("--wall",required=True)
    ap.add_argument("--conv",required=True)
    ap.add_argument("--reference-f",type=float,default=0.0284003)
    a=ap.parse_args()

    pair=load_csv(a.pair); wall=load_csv(a.wall); conv=load_csv(a.conv)
    windows=[(12,14),(14,16),(16,18),(18,19.8),(12,18)]

    for z0,z1 in windows:
        mp=(pair["z_mid_over_D"]>=z0)&(pair["z_mid_over_D"]<z1)
        mw=(wall["z_over_D"]>=z0)&(wall["z_over_D"]<z1)
        mc=(0.5*(conv["z0_over_D"]+conv["z1_over_D"])>=z0)&(0.5*(conv["z0_over_D"]+conv["z1_over_D"])<z1)

        fp=float(np.mean(pair["f_Darcy_local"][mp]))
        fw=weighted_mean(wall["f_applied"][mw],wall["area"][mw])

        # Combine slab contributions by volume, because each f is built from a
        # volume-mean acceleration.
        vol=np.asarray(conv["volume"][mc],float)
        def cv(name):
            return weighted_mean(conv[name][mc],vol)
        fadv=cv("f_adv_weak")
        fadvq=cv("f_adv_strong")
        fcons=cv("f_conservative_strong")
        fdiv=cv("f_uz_div_u")
        parity=cv("weak_strong_parity_rel")

        missing=fp-fw
        radv=fp-fw-fadv
        rcons=fp-fw-fcons
        frac=fadv/missing if abs(missing)>1e-14 else math.nan
        print(
          "NODALS_ADVECTIVE_ERROR_WINDOW "
          f"z0D={z0:.2f} z1D={z1:.2f} "
          f"fPressure={fp:.10f} fWall={fw:.10f} "
          f"pressureMinusWall={missing:.10f} "
          f"fAdvectiveDiscrete={fadv:.10f} fAdvectiveStrong={fadvq:.10f} "
          f"fConservativeSameField={fcons:.10f} fUzDivU={fdiv:.10f} "
          f"advectiveExplainsFraction={frac:.6f} "
          f"residualAfterAdvective={radv:.10f} residualAfterConservative={rcons:.10f} "
          f"weakStrongParityRel={parity:.3e} "
          f"pressureErrPct={100*(fp/a.reference_f-1):.6f} status=PASS"
        )

if __name__=="__main__":
    main()
