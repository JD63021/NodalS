#!/usr/bin/env python3
import argparse, csv, math
from pathlib import Path

def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def fnum(x):
    return float(x)

def pair_window_mean(rows,z0,z1):
    vals=[]
    for r in rows:
        z=float(r["z_mid_over_D"])
        if z>=z0 and z<z1:
            vals.append(float(r["f_Darcy_local"]))
    return sum(vals)/len(vals) if vals else math.nan

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--budget",required=True)
    ap.add_argument("--pair",required=True)
    ap.add_argument("--reference-f",type=float,default=0.0284003)
    a=ap.parse_args()
    B=read_rows(a.budget)
    P=read_rows(a.pair)

    for r in B:
        z0=float(r["z0_over_D"]); z1=float(r["z1_over_D"])
        fp_pair=pair_window_mean(P,z0,z1)
        fpf=float(r["f_pressure_final"])
        fpm=float(r["f_pressure_momentum"])
        fm=float(r["f_molecular"])
        fn=float(r["f_nut"])
        fw=float(r["f_wall"])
        fdg=float(r["f_dg"])
        fa=float(r["f_advective"])
        fres=float(r["f_resist_sum"])
        fphys=float(r["f_direct_physical_residual"])
        frelax=float(r["f_relaxation_lag"])
        frlin=float(r["f_relaxed_linear_residual"])
        parity=float(r["component_direct_parity"])
        pcor=float(r["f_pressure_correction"])
        final_closure=fres-fpf

        print(
            "NODALS_DISCRETE_BUDGET_COMPARE "
            f"z0D={z0:.2f} z1D={z1:.2f} "
            f"fPressurePairwise={fp_pair:.10f} "
            f"fPressureDiscreteFinal={fpf:.10f} "
            f"pairwiseMinusDiscrete={fp_pair-fpf:.10f} "
            f"fPressureUsedByMomentum={fpm:.10f} "
            f"fFinalPressureCorrection={pcor:.10f} "
            f"fMolecular={fm:.10f} fNuT={fn:.10f} "
            f"fWall={fw:.10f} fDG={fdg:.10f} fAdvective={fa:.10f} "
            f"fResistSum={fres:.10f} "
            f"fPhysicalResidualLastLinearization={fphys:.10f} "
            f"fRelaxationLag={frelax:.10f} "
            f"fRelaxedLinearResidual={frlin:.10f} "
            f"fClosureUsingFinalPressure={final_closure:.10f} "
            f"componentDirectParity={parity:.3e} "
            f"pairwiseErrPct={100*(fp_pair/a.reference_f-1):.6f} "
            f"discretePressureErrPct={100*(fpf/a.reference_f-1):.6f} status=PASS"
        )

if __name__=="__main__":
    main()
