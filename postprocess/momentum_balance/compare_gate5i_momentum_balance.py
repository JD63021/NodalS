#!/usr/bin/env python3
"""
Gate 5I axial momentum-balance diagnostic.

For an incompressible straight pipe, using kinematic pressure p:

    -A dp/dz = P * tau_w + d/dz int_A Uz^2 dA

With Darcy-friction normalization,

    f_pressure = f_wall + f_development

where

    f_pressure    = 2 D (-dp/dz) / Ubulk^2
    f_wall        = 8 <u_tau^2>_wall / Ubulk^2
    beta(z)       = [int_A Uz^2 dA] / [A Ubulk^2]
    f_development = 2 d beta / d(z/D)

The VTU contains exact cell-average U0, not the full P1+BF3 polynomial.
Therefore beta is approximated as

    beta_VTU = sum_cells V_c (U0_z,c)^2 / (V_slab Ubulk^2)

This is sufficient for a strong diagnostic, especially for comparing the
baseline and reference-shear cases on the identical mesh, but it is not an
exact FE quadrature of Uz^2.
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np

try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
except Exception as exc:
    raise SystemExit(
        "ERROR: Python VTK is required.\n"
        "Install with: python -m pip install vtk\n"
        f"Original import error: {exc}"
    )

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_vtu(path, D, Ubulk):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    g = reader.GetOutput()

    nc = g.GetNumberOfCells()
    npnt = g.GetNumberOfPoints()
    if nc <= 0 or npnt <= 0:
        raise RuntimeError(f"{path}: empty VTU")

    pts = vtk_to_numpy(g.GetPoints().GetData()).astype(np.float64, copy=False)

    cells = g.GetCells()
    offs = vtk_to_numpy(cells.GetOffsetsArray()).astype(np.int64, copy=False)
    conn = vtk_to_numpy(cells.GetConnectivityArray()).astype(np.int64, copy=False)
    if len(offs) != nc + 1 or np.any(np.diff(offs) != 4):
        raise RuntimeError(f"{path}: expected tetrahedra only")
    tet = conn.reshape(nc, 4)

    cdata = g.GetCellData()
    pa = cdata.GetArray("p_P0")
    ua = cdata.GetArray("U0")
    if pa is None or ua is None:
        names = [cdata.GetArrayName(i) for i in range(cdata.GetNumberOfArrays())]
        raise RuntimeError(f"{path}: need CellData p_P0 and U0; available={names}")

    p = vtk_to_numpy(pa).astype(np.float64, copy=False).reshape(-1)
    U = vtk_to_numpy(ua).astype(np.float64, copy=False)
    if U.ndim != 2 or U.shape[1] < 3:
        raise RuntimeError(f"{path}: U0 is not a 3-vector")
    uz = U[:, 2]

    vv = pts[tet]
    a = vv[:, 1] - vv[:, 0]
    b = vv[:, 2] - vv[:, 0]
    c = vv[:, 3] - vv[:, 0]
    vol = np.abs(np.einsum("ij,ij->i", np.cross(a, b), c)) / 6.0

    # Recover exact extruded axial planes and assign every tet to one slab.
    zround = np.round(pts[:, 2], 12)
    planes = np.unique(zround)
    planes.sort()
    z_to_i = {float(z): i for i, z in enumerate(planes)}
    vp = np.empty((nc, 4), dtype=np.int64)
    for j in range(4):
        vp[:, j] = np.array([z_to_i[float(z)] for z in zround[tet[:, j]]], dtype=np.int64)
    smin = np.min(vp, axis=1)
    smax = np.max(vp, axis=1)
    if np.any(smax - smin != 1):
        bad = int(np.flatnonzero(smax-smin != 1)[0])
        raise RuntimeError(
            f"{path}: cell {bad} does not span exactly one axial slab "
            f"(planes {smin[bad]}..{smax[bad]})"
        )

    ns = len(planes) - 1
    if ns <= 2:
        raise RuntimeError(f"{path}: too few slabs ({ns})")

    slabV = np.bincount(smin, weights=vol, minlength=ns)
    slabP = np.bincount(smin, weights=vol*p, minlength=ns) / slabV
    slabUz = np.bincount(smin, weights=vol*uz, minlength=ns) / slabV
    slabUz2 = np.bincount(smin, weights=vol*uz*uz, minlength=ns) / slabV

    z0 = planes[:-1]
    z1 = planes[1:]
    dz = z1 - z0
    zmid = 0.5*(z0+z1)
    zmidD = zmid / D
    dzD = dz / D
    area = slabV / dz

    beta = slabUz2/(Ubulk*Ubulk)
    bulk_ratio = slabUz/Ubulk

    # Differential forms. np.gradient uses centered differences inside.
    dp_dz = np.gradient(slabP, zmid, edge_order=2)
    dbeta_dzD = np.gradient(beta, zmidD, edge_order=2)

    f_pressure = 2.0*D*(-dp_dz)/(Ubulk*Ubulk)
    f_dev = 2.0*dbeta_dzD

    return dict(
        path=str(path),
        nc=nc,
        npnt=npnt,
        planes=planes,
        z0=z0,
        z1=z1,
        zmid=zmid,
        zmidD=zmidD,
        dz=dz,
        dzD=dzD,
        area=area,
        volume=slabV,
        p=slabP,
        Uz=slabUz,
        Uz2=slabUz2,
        beta=beta,
        bulkRatio=bulk_ratio,
        fPressure=f_pressure,
        fDev=f_dev,
    )


def read_wall(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            q = dict(r)
            for k in ["z_over_D","area","f_spalding","f_applied",
                      "u_tau2_spalding","u_tau2_applied","slip","y_plus_spalding"]:
                q[k] = float(q[k])
            rows.append(q)
    if not rows:
        raise RuntimeError(f"{path}: no wall rows")
    return rows


def wall_to_slabs(rows, planesD):
    ns = len(planesD)-1
    z = np.array([r["z_over_D"] for r in rows], float)
    area = np.array([r["area"] for r in rows], float)
    fap = np.array([r["f_applied"] for r in rows], float)
    fsp = np.array([r["f_spalding"] for r in rows], float)
    slip = np.array([r["slip"] for r in rows], float)
    yp = np.array([r["y_plus_spalding"] for r in rows], float)

    sid = np.searchsorted(planesD, z, side="right") - 1
    sid = np.clip(sid, 0, ns-1)

    A = np.bincount(sid, weights=area, minlength=ns)
    def avg(x):
        num = np.bincount(sid, weights=area*x, minlength=ns)
        out = np.full(ns, np.nan)
        good = A > 0
        out[good] = num[good]/A[good]
        return out

    return dict(area=A, fApplied=avg(fap), fSpalding=avg(fsp),
                slip=avg(slip), yPlus=avg(yp))


def read_pairwise_optional(path):
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    a = np.genfromtxt(p, delimiter=",", names=True)
    names = a.dtype.names or ()
    if "z_mid_over_D" not in names or "f_Darcy_local" not in names:
        return None
    return a


def linear_slope(x, y):
    good = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(good) < 2:
        return math.nan, math.nan
    coef = np.polyfit(x[good], y[good], 1)
    pred = np.polyval(coef, x[good])
    ssr = float(np.sum((y[good]-pred)**2))
    sst = float(np.sum((y[good]-np.mean(y[good]))**2))
    r2 = 1.0 - ssr/sst if sst > 0 else 1.0
    return float(coef[0]), r2


def area_weight_wall_window(rows, key, z0, z1):
    vals = []
    weights = []
    for r in rows:
        z = r["z_over_D"]
        if z0 <= z < z1:
            vals.append(r[key]); weights.append(r["area"])
    if not vals:
        return math.nan
    vals = np.asarray(vals, float); weights = np.asarray(weights, float)
    return float(np.sum(vals*weights)/np.sum(weights))


def pair_mean(pair, z0, z1):
    if pair is None:
        return math.nan
    z = np.asarray(pair["z_mid_over_D"], float)
    f = np.asarray(pair["f_Darcy_local"], float)
    m = (z >= z0) & (z < z1)
    return float(np.mean(f[m])) if np.any(m) else math.nan


def summarize_window(case_name, dat, wall_rows, pair, z0, z1, Ubulk, fref):
    m = (dat["zmidD"] >= z0) & (dat["zmidD"] < z1)
    if np.count_nonzero(m) < 2:
        raise RuntimeError(f"{case_name}: not enough slabs in {z0}-{z1}D")

    # Fit p against z/D directly:
    # dp/dz = (1/D) dp/d(z/D)
    # f_p = -2 [dp/d(z/D)] / Ubulk^2.
    sp, p_r2 = linear_slope(dat["zmidD"][m], dat["p"][m])
    sb, b_r2 = linear_slope(dat["zmidD"][m], dat["beta"][m])
    fp_fit = -2.0*sp/(Ubulk*Ubulk)
    fd_fit = 2.0*sb

    fw = area_weight_wall_window(wall_rows, "f_applied", z0, z1)
    fws = area_weight_wall_window(wall_rows, "f_spalding", z0, z1)
    slip = area_weight_wall_window(wall_rows, "slip", z0, z1)
    yp = area_weight_wall_window(wall_rows, "y_plus_spalding", z0, z1)

    residual = fp_fit - fw - fd_fit
    fp_pair = pair_mean(pair, z0, z1)

    return dict(
        case=case_name, z0D=z0, z1D=z1,
        slabs=int(np.count_nonzero(m)),
        fPressureFit=fp_fit,
        fPressurePairwise=fp_pair,
        fWallApplied=fw,
        fWallSpalding=fws,
        fDevelopmentFit=fd_fit,
        fWallPlusDevelopment=fw+fd_fit,
        closureResidual=residual,
        closureResidualPctOfReference=100.0*residual/fref,
        pressureErrorPct=100.0*(fp_fit/fref-1.0),
        appliedWallErrorPct=100.0*(fw/fref-1.0),
        developmentPctOfReference=100.0*fd_fit/fref,
        betaSlopePerD=sb,
        betaMean=float(np.mean(dat["beta"][m])),
        betaStart=float(dat["beta"][np.flatnonzero(m)[0]]),
        betaEnd=float(dat["beta"][np.flatnonzero(m)[-1]]),
        resolvedBulkRatioMean=float(np.mean(dat["bulkRatio"][m])),
        pressureFitR2=p_r2,
        betaFitR2=b_r2,
        slipMean=slip,
        yPlusMean=yp,
    )


def write_rows(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_slab_csv(path, case_name, dat, ws):
    rows=[]
    for i in range(len(dat["zmidD"])):
        fw = ws["fApplied"][i]
        fd = dat["fDev"][i]
        fp = dat["fPressure"][i]
        rows.append(dict(
            case=case_name,
            slab=i,
            z0D=dat["z0"][i]/D_GLOBAL,
            z1D=dat["z1"][i]/D_GLOBAL,
            zMidD=dat["zmidD"][i],
            dzD=dat["dzD"][i],
            areaFromVolume=dat["area"][i],
            pMean=dat["p"][i],
            UzMean=dat["Uz"][i],
            bulkRatio=dat["bulkRatio"][i],
            betaVTU=dat["beta"][i],
            fPressureDifferential=fp,
            fWallApplied=fw,
            fWallSpalding=ws["fSpalding"][i],
            fDevelopmentDifferential=fd,
            fWallPlusDevelopment=(fw+fd) if np.isfinite(fw) else math.nan,
            closureResidual=(fp-fw-fd) if np.isfinite(fw) else math.nan,
            slipMean=ws["slip"][i],
            yPlusMean=ws["yPlus"][i],
        ))
    write_rows(path, rows)


def plot_case(outdir, case_name, dat, ws, zlo, zhi, fref):
    m = (dat["zmidD"] >= zlo) & (dat["zmidD"] <= zhi)
    x = dat["zmidD"][m]

    fig = plt.figure(figsize=(8.0,5.0))
    ax = fig.add_subplot(111)
    ax.plot(x, dat["fPressure"][m], label="f pressure")
    ax.plot(x, ws["fApplied"][m], label="f wall applied")
    ax.plot(x, dat["fDev"][m], label="f development")
    ax.plot(x, ws["fApplied"][m]+dat["fDev"][m], label="f wall + development")
    ax.axhline(fref, linestyle="--", label="reference f")
    ax.set_xlabel("z / D")
    ax.set_ylabel("Darcy-friction equivalent")
    ax.set_title(f"{case_name}: axial momentum balance")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir/f"{case_name}_momentum_balance.png", dpi=170)
    plt.close(fig)

    fig = plt.figure(figsize=(8.0,5.0))
    ax = fig.add_subplot(111)
    ax.plot(x, dat["beta"][m])
    ax.set_xlabel("z / D")
    ax.set_ylabel("beta_VTU = <Uz^2> / Ubulk^2")
    ax.set_title(f"{case_name}: resolved momentum shape factor")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir/f"{case_name}_beta.png", dpi=170)
    plt.close(fig)

    fig = plt.figure(figsize=(8.0,5.0))
    ax = fig.add_subplot(111)
    residual = dat["fPressure"][m]-ws["fApplied"][m]-dat["fDev"][m]
    ax.plot(x, residual)
    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel("z / D")
    ax.set_ylabel("f_pressure - f_wall - f_development")
    ax.set_title(f"{case_name}: momentum-closure residual")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir/f"{case_name}_closure_residual.png", dpi=170)
    plt.close(fig)


def main():
    global D_GLOBAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-vtu", required=True)
    ap.add_argument("--forced-vtu", required=True)
    ap.add_argument("--baseline-wall", required=True)
    ap.add_argument("--forced-wall", required=True)
    ap.add_argument("--baseline-pairs")
    ap.add_argument("--forced-pairs")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--diameter", type=float, default=0.004)
    ap.add_argument("--bulk", type=float, default=50.0)
    ap.add_argument("--reference-f", type=float, default=0.0284003)
    ap.add_argument("--plot-z0-D", type=float, default=10.0)
    ap.add_argument("--plot-z1-D", type=float, default=19.6)
    args = ap.parse_args()

    D_GLOBAL = args.diameter
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    print("NODALS_MOMENTUM_BALANCE_READ case=baseline", flush=True)
    b = read_vtu(args.baseline_vtu, args.diameter, args.bulk)
    print("NODALS_MOMENTUM_BALANCE_READ case=reference_shear", flush=True)
    f = read_vtu(args.forced_vtu, args.diameter, args.bulk)
    bw = read_wall(args.baseline_wall)
    fw = read_wall(args.forced_wall)

    if len(b["planes"]) != len(f["planes"]) or not np.allclose(b["planes"], f["planes"], rtol=0, atol=1e-12):
        raise RuntimeError("Baseline and forced VTUs do not use the same axial planes")

    planesD = b["planes"]/args.diameter
    bws = wall_to_slabs(bw, planesD)
    fws = wall_to_slabs(fw, planesD)

    bp = read_pairwise_optional(args.baseline_pairs)
    fp = read_pairwise_optional(args.forced_pairs)

    # Geometry audit.
    Aexact = math.pi*(0.5*args.diameter)**2
    Amean = float(np.mean(b["area"]))
    Arel = (Amean/Aexact)-1.0
    print(
        "NODALS_MOMENTUM_BALANCE_GEOMETRY "
        f"slabs={len(b['zmidD'])} meanDzD={np.mean(b['dzD']):.12e} "
        f"areaFromVolume={Amean:.12e} exactArea={Aexact:.12e} "
        f"areaRelError={Arel:.12e} status=PASS"
    )

    write_slab_csv(out/"baseline_slab_balance.csv", "baseline", b, bws)
    write_slab_csv(out/"reference_shear_slab_balance.csv", "reference_shear", f, fws)

    windows=[(12.0,14.0),(14.0,16.0),(16.0,18.0),(12.0,18.0),(18.0,19.8)]
    rows=[]
    for z0,z1 in windows:
        rb=summarize_window("baseline",b,bw,bp,z0,z1,args.bulk,args.reference_f)
        rf=summarize_window("reference_shear",f,fw,fp,z0,z1,args.bulk,args.reference_f)
        rows.extend([rb,rf])
        for r in [rb,rf]:
            print(
                "NODALS_MOMENTUM_BALANCE_WINDOW "
                f"case={r['case']} z0D={z0:.2f} z1D={z1:.2f} slabs={r['slabs']} "
                f"fPressureFit={r['fPressureFit']:.10f} "
                f"fPressurePairwise={r['fPressurePairwise']:.10f} "
                f"fWallApplied={r['fWallApplied']:.10f} "
                f"fWallSpalding={r['fWallSpalding']:.10f} "
                f"fDevelopmentFit={r['fDevelopmentFit']:.10f} "
                f"fWallPlusDevelopment={r['fWallPlusDevelopment']:.10f} "
                f"closureResidual={r['closureResidual']:.10e} "
                f"closureResidualPctRef={r['closureResidualPctOfReference']:.6f} "
                f"betaSlopePerD={r['betaSlopePerD']:.10e} "
                f"betaMean={r['betaMean']:.10f} "
                f"bulkRatio={r['resolvedBulkRatioMean']:.10f} "
                f"pFitR2={r['pressureFitR2']:.8f} betaFitR2={r['betaFitR2']:.8f} status=PASS"
            )
        print(
            "NODALS_MOMENTUM_BALANCE_DELTA "
            f"z0D={z0:.2f} z1D={z1:.2f} "
            f"deltaPressure={rf['fPressureFit']-rb['fPressureFit']:.10e} "
            f"deltaWall={rf['fWallApplied']-rb['fWallApplied']:.10e} "
            f"deltaDevelopment={rf['fDevelopmentFit']-rb['fDevelopmentFit']:.10e} "
            f"deltaClosureResidual={rf['closureResidual']-rb['closureResidual']:.10e} status=PASS"
        )

    write_rows(out/"window_momentum_balance.csv", rows)

    plot_case(out, "baseline", b, bws, args.plot_z0_D, args.plot_z1_D, args.reference_f)
    plot_case(out, "reference_shear", f, fws, args.plot_z0_D, args.plot_z1_D, args.reference_f)

    # Compare beta directly.
    m = (b["zmidD"] >= args.plot_z0_D) & (b["zmidD"] <= args.plot_z1_D)
    fig = plt.figure(figsize=(8.0,5.0))
    ax = fig.add_subplot(111)
    ax.plot(b["zmidD"][m], b["beta"][m], label="baseline")
    ax.plot(f["zmidD"][m], f["beta"][m], label="reference shear")
    ax.set_xlabel("z / D")
    ax.set_ylabel("beta_VTU")
    ax.set_title("Resolved momentum-shape-factor comparison")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/"beta_compare.png", dpi=170)
    plt.close(fig)

    print(
        "NODALS_MOMENTUM_BALANCE_LIMITATION "
        "betaMoment=SUM_V_OF_CELL_AVERAGE_UZ_SQUARED "
        "exactP1BF3UzSquaredQuadrature=NOT_AVAILABLE_IN_VTU "
        "interpretation=STRONG_DIAGNOSTIC_NOT_EXACT_FE_MOMENT status=PASS"
    )
    print(
        "NODALS_MOMENTUM_BALANCE_OUTPUT "
        f"outdir={out} windowCsv={out/'window_momentum_balance.csv'} "
        f"baselineSlabCsv={out/'baseline_slab_balance.csv'} "
        f"forcedSlabCsv={out/'reference_shear_slab_balance.csv'} "
        f"baselinePlot={out/'baseline_momentum_balance.png'} "
        f"forcedPlot={out/'reference_shear_momentum_balance.png'} "
        f"betaCompare={out/'beta_compare.png'} status=PASS"
    )


if __name__ == "__main__":
    main()
