#!/usr/bin/env python3
"""
Compare the 10-slabs/D and 5-slabs/D VMFL003 VTUs.

This script answers "where does the discrepancy appear?" using only fields
already present in the VTU:
  - CellData U0
  - CellData p_P0
  - mesh geometry/connectivity

It does NOT claim to identify the causal origin of the error, because the VTU
does not contain the exact weak-Spalding wall trace, u_tau, beta, or y+ on each
wall face.

Outputs:
  radial_compare.csv
  developed_region_summary.csv
  axial_f_by_region_5slabD.csv
  axial_f_by_region_10slabD.csv
  radial_f_compare.png
  radial_uz_compare.png
  radial_delta_uz.png
  axial_f_regions_5slabD.png
  axial_f_regions_10slabD.png
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
except Exception as exc:
    raise SystemExit(
        "ERROR: Python VTK is required for this diagnostic.\n"
        "Try one of:\n"
        "  python3 -m pip install --user vtk\n"
        "or, on Ubuntu/Debian:\n"
        "  sudo apt install python3-vtk9\n"
        f"Original import error: {exc}"
    )

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def weighted_mean(x, w):
    x = np.asarray(x)
    w = np.asarray(w)
    good = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(good):
        return float("nan")
    return float(np.sum(x[good] * w[good]) / np.sum(w[good]))


def read_case(path, D, R, cx, cy, wall_vertex_tol_rel):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    grid = reader.GetOutput()

    nc = grid.GetNumberOfCells()
    npnt = grid.GetNumberOfPoints()
    if nc <= 0 or npnt <= 0:
        raise RuntimeError(f"{path}: empty grid")

    # Require tetrahedra only.
    cell_types = vtk_to_numpy(grid.GetCellTypesArray())
    unique_types = np.unique(cell_types)
    if not (len(unique_types) == 1 and int(unique_types[0]) == vtk.VTK_TETRA):
        raise RuntimeError(
            f"{path}: expected all tetrahedra (VTK type {vtk.VTK_TETRA}), "
            f"found cell types {unique_types.tolist()}"
        )

    pts = vtk_to_numpy(grid.GetPoints().GetData()).astype(np.float64, copy=False)

    cells = grid.GetCells()
    offsets = vtk_to_numpy(cells.GetOffsetsArray()).astype(np.int64, copy=False)
    conn = vtk_to_numpy(cells.GetConnectivityArray()).astype(np.int64, copy=False)
    if len(offsets) != nc + 1 or np.any(np.diff(offsets) != 4):
        raise RuntimeError(f"{path}: tetra connectivity offsets are not 4-wide")
    tet = conn.reshape(nc, 4)

    cdata = grid.GetCellData()
    p_arr = cdata.GetArray("p_P0")
    u_arr = cdata.GetArray("U0")
    if p_arr is None:
        names = [cdata.GetArrayName(i) for i in range(cdata.GetNumberOfArrays())]
        raise RuntimeError(f"{path}: CellData p_P0 missing. Available={names}")
    if u_arr is None:
        names = [cdata.GetArrayName(i) for i in range(cdata.GetNumberOfArrays())]
        raise RuntimeError(f"{path}: CellData U0 missing. Available={names}")

    p = vtk_to_numpy(p_arr).astype(np.float64, copy=False).reshape(-1)
    U = vtk_to_numpy(u_arr).astype(np.float64, copy=False)
    if U.ndim != 2 or U.shape[1] < 3:
        raise RuntimeError(f"{path}: U0 is not a 3-component vector")
    U = U[:, :3]

    v = pts[tet]                           # (nc,4,3)
    cc = np.mean(v, axis=1)
    a = v[:, 1, :] - v[:, 0, :]
    b = v[:, 2, :] - v[:, 0, :]
    c = v[:, 3, :] - v[:, 0, :]
    vol = np.abs(np.einsum("ij,ij->i", np.cross(a, b), c)) / 6.0

    rv = np.sqrt((v[:, :, 0] - cx)**2 + (v[:, :, 1] - cy)**2)
    wall_touch = np.max(rv, axis=1) >= R * (1.0 - wall_vertex_tol_rel)

    r = np.sqrt((cc[:, 0] - cx)**2 + (cc[:, 1] - cy)**2)
    rr = r / R
    zD = cc[:, 2] / D

    return {
        "path": str(path),
        "grid": grid,
        "ncell": nc,
        "npoint": npnt,
        "points": pts,
        "tet": tet,
        "cc": cc,
        "vol": vol,
        "p": p,
        "U": U,
        "rR": rr,
        "zD": zD,
        "wall_touch": wall_touch,
    }


def build_pairs(case, D, Ubulk, rho, xy_tol_D):
    cc = case["cc"]
    p = case["p"]
    rr = case["rR"]
    wall_touch = case["wall_touch"]

    tol = xy_tol_D * D
    kx = np.rint(cc[:, 0] / tol).astype(np.int64)
    ky = np.rint(cc[:, 1] / tol).astype(np.int64)
    z = cc[:, 2]

    order = np.lexsort((z, ky, kx))
    kx_s = kx[order]
    ky_s = ky[order]
    z_s = z[order]
    p_s = p[order]
    rr_s = rr[order]
    wt_s = wall_touch[order]

    same = (kx_s[1:] == kx_s[:-1]) & (ky_s[1:] == ky_s[:-1])
    dz = z_s[1:] - z_s[:-1]
    good = same & (dz > 0.0)

    i0 = np.nonzero(good)[0]
    i1 = i0 + 1

    dzg = dz[good]
    # p_P0 is treated exactly as in the existing friction postprocessor:
    # rho defaults to 1, so this works for the current kinematic-pressure convention.
    dpdz = (p_s[i0] - p_s[i1]) / dzg
    f = 2.0 * D * dpdz / (rho * Ubulk * Ubulk)

    return {
        "zD": 0.5 * (z_s[i0] + z_s[i1]) / D,
        "rR": 0.5 * (rr_s[i0] + rr_s[i1]),
        "f": f,
        "dpdz": dpdz,
        "wall_touch": wt_s[i0] & wt_s[i1],
        "dzD": dzg / D,
    }


def region_name(rr, wall_touch):
    out = np.full(rr.shape, "nearwall", dtype=object)
    out[rr < 0.50] = "core"
    out[(rr >= 0.50) & (rr < 0.80)] = "mid"
    out[rr >= 0.80] = "nearwall"
    out[wall_touch] = "wall_touch"
    return out


def radial_profiles(case, pairs, z0D, z1D, nbins):
    edges = np.linspace(0.0, 1.0, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Velocity profile, volume weighted.
    cm = (case["zD"] >= z0D) & (case["zD"] < z1D)
    bid = np.digitize(case["rR"], edges) - 1
    uz = case["U"][:, 2]
    vmean = np.full(nbins, np.nan)
    vcount = np.zeros(nbins, dtype=np.int64)
    for b in range(nbins):
        m = cm & (bid == b)
        vcount[b] = int(np.count_nonzero(m))
        if vcount[b]:
            vmean[b] = weighted_mean(uz[m], case["vol"][m])

    # Pressure-gradient/friction profile.
    pm = (pairs["zD"] >= z0D) & (pairs["zD"] < z1D)
    pbid = np.digitize(pairs["rR"], edges) - 1
    fmean = np.full(nbins, np.nan)
    fmed = np.full(nbins, np.nan)
    pcount = np.zeros(nbins, dtype=np.int64)
    for b in range(nbins):
        m = pm & (pbid == b)
        pcount[b] = int(np.count_nonzero(m))
        if pcount[b]:
            fmean[b] = float(np.mean(pairs["f"][m]))
            fmed[b] = float(np.median(pairs["f"][m]))

    return {
        "edges": edges,
        "rR": centers,
        "Uz": vmean,
        "Ucount": vcount,
        "fMean": fmean,
        "fMedian": fmed,
        "fCount": pcount,
    }


def region_summary(case_name, case, pairs, z0D, z1D, Ubulk, fref):
    rows = []
    specs = [
        ("core", lambda rr, wt: rr < 0.50),
        ("mid", lambda rr, wt: (rr >= 0.50) & (rr < 0.80)),
        ("nearwall", lambda rr, wt: rr >= 0.80),
        ("wall_touch", lambda rr, wt: wt),
        ("non_wall_touch", lambda rr, wt: ~wt),
        ("all", lambda rr, wt: np.ones(rr.shape, dtype=bool)),
    ]

    cmz = (case["zD"] >= z0D) & (case["zD"] < z1D)
    pmz = (pairs["zD"] >= z0D) & (pairs["zD"] < z1D)

    for name, selector in specs:
        cm = cmz & selector(case["rR"], case["wall_touch"])
        pm = pmz & selector(pairs["rR"], pairs["wall_touch"])

        uz = weighted_mean(case["U"][cm, 2], case["vol"][cm]) if np.any(cm) else float("nan")
        urms = math.sqrt(
            weighted_mean(case["U"][cm, 2]**2, case["vol"][cm])
        ) if np.any(cm) else float("nan")

        if np.any(pm):
            fm = float(np.mean(pairs["f"][pm]))
            fmed = float(np.median(pairs["f"][pm]))
            fstd = float(np.std(pairs["f"][pm]))
        else:
            fm = fmed = fstd = float("nan")

        rows.append({
            "case": case_name,
            "region": name,
            "z0D": z0D,
            "z1D": z1D,
            "cellCount": int(np.count_nonzero(cm)),
            "pairCount": int(np.count_nonzero(pm)),
            "UzVolMean": uz,
            "UzVolMeanOverUbulk": uz/Ubulk if np.isfinite(uz) else float("nan"),
            "UzRms": urms,
            "fMean": fm,
            "fMedian": fmed,
            "fStd": fstd,
            "errorVsReferencePercent": 100.0*(fm/fref - 1.0) if np.isfinite(fm) else float("nan"),
        })
    return rows


def axial_region_profile(case_name, pairs, z0D, z1D, binD):
    edges = np.arange(z0D, z1D + 0.5*binD, binD)
    if edges[-1] < z1D:
        edges = np.append(edges, z1D)
    regions = [
        ("core", lambda rr, wt: rr < 0.50),
        ("mid", lambda rr, wt: (rr >= 0.50) & (rr < 0.80)),
        ("nearwall", lambda rr, wt: rr >= 0.80),
        ("wall_touch", lambda rr, wt: wt),
        ("all", lambda rr, wt: np.ones(rr.shape, dtype=bool)),
    ]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        zm = (pairs["zD"] >= a) & (pairs["zD"] < b)
        for name, selector in regions:
            m = zm & selector(pairs["rR"], pairs["wall_touch"])
            if np.any(m):
                out.append({
                    "case": case_name,
                    "region": name,
                    "z0D": float(a),
                    "z1D": float(b),
                    "zMidD": float(0.5*(a+b)),
                    "count": int(np.count_nonzero(m)),
                    "fMean": float(np.mean(pairs["f"][m])),
                    "fMedian": float(np.median(pairs["f"][m])),
                })
    return out


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_radial_f(out, p10, p5, fref):
    fig = plt.figure(figsize=(7.2, 4.8))
    ax = fig.add_subplot(111)
    ax.plot(p10["rR"], p10["fMean"], marker="o", label="10 slabs/D")
    ax.plot(p5["rR"], p5["fMean"], marker="s", label="5 slabs/D")
    ax.axhline(fref, linestyle="--", label=f"reference {fref:.7f}")
    ax.set_xlabel("r / R")
    ax.set_ylabel("local Darcy f from pairwise dp/dz")
    ax.set_title("Developed radial pressure-gradient profile")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_radial_uz(out, p10, p5, Ubulk):
    fig = plt.figure(figsize=(7.2, 4.8))
    ax = fig.add_subplot(111)
    ax.plot(p10["rR"], p10["Uz"]/Ubulk, marker="o", label="10 slabs/D")
    ax.plot(p5["rR"], p5["Uz"]/Ubulk, marker="s", label="5 slabs/D")
    ax.set_xlabel("r / R")
    ax.set_ylabel("Uz / Ubulk")
    ax.set_title("Developed radial velocity profile")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_radial_du(out, p10, p5, Ubulk):
    fig = plt.figure(figsize=(7.2, 4.8))
    ax = fig.add_subplot(111)
    d = (p5["Uz"] - p10["Uz"]) / Ubulk
    ax.plot(p10["rR"], d, marker="o")
    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel("r / R")
    ax.set_ylabel("(Uz_5 - Uz_10) / Ubulk")
    ax.set_title("Velocity-profile change caused by axial coarsening")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_axial_regions(out, rows, fref, title):
    fig = plt.figure(figsize=(7.6, 4.9))
    ax = fig.add_subplot(111)
    for region in ["core", "mid", "nearwall", "wall_touch", "all"]:
        rr = [r for r in rows if r["region"] == region]
        if rr:
            ax.plot(
                [r["zMidD"] for r in rr],
                [r["fMean"] for r in rr],
                marker="o",
                label=region,
            )
    ax.axhline(fref, linestyle="--", label=f"reference {fref:.7f}")
    ax.set_xlabel("z / D")
    ax.set_ylabel("local Darcy f from pairwise dp/dz")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vtu10", required=True, help="10-slabs/D VTU")
    ap.add_argument("--vtu5", required=True, help="5-slabs/D VTU")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--diameter", type=float, default=0.004)
    ap.add_argument("--bulk", type=float, default=50.0)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--reference-f", type=float, default=0.0284003)
    ap.add_argument("--z0-D", type=float, default=14.0)
    ap.add_argument("--z1-D", type=float, default=18.0)
    ap.add_argument("--radial-bins", type=int, default=20)
    ap.add_argument("--axial-bin-D", type=float, default=0.5)
    ap.add_argument("--xy-tol-D", type=float, default=1e-6)
    ap.add_argument("--center-x", type=float, default=0.0)
    ap.add_argument("--center-y", type=float, default=0.0)
    ap.add_argument("--wall-vertex-tol-rel", type=float, default=1e-6)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    D = args.diameter
    R = 0.5*D

    print("NODALS_RADIAL_DIAG_READ case=10slabD", flush=True)
    c10 = read_case(args.vtu10, D, R, args.center_x, args.center_y, args.wall_vertex_tol_rel)
    print("NODALS_RADIAL_DIAG_READ case=5slabD", flush=True)
    c5 = read_case(args.vtu5, D, R, args.center_x, args.center_y, args.wall_vertex_tol_rel)

    print("NODALS_RADIAL_DIAG_PAIR case=10slabD", flush=True)
    q10 = build_pairs(c10, D, args.bulk, args.rho, args.xy_tol_D)
    print("NODALS_RADIAL_DIAG_PAIR case=5slabD", flush=True)
    q5 = build_pairs(c5, D, args.bulk, args.rho, args.xy_tol_D)

    p10 = radial_profiles(c10, q10, args.z0_D, args.z1_D, args.radial_bins)
    p5 = radial_profiles(c5, q5, args.z0_D, args.z1_D, args.radial_bins)

    radial_rows = []
    for i in range(args.radial_bins):
        radial_rows.append({
            "rOverR": p10["rR"][i],
            "Uz10": p10["Uz"][i],
            "Uz5": p5["Uz"][i],
            "Uz10OverUbulk": p10["Uz"][i]/args.bulk,
            "Uz5OverUbulk": p5["Uz"][i]/args.bulk,
            "deltaUz5minus10": p5["Uz"][i]-p10["Uz"][i],
            "deltaUzOverUbulk": (p5["Uz"][i]-p10["Uz"][i])/args.bulk,
            "f10Mean": p10["fMean"][i],
            "f5Mean": p5["fMean"][i],
            "deltaF5minus10": p5["fMean"][i]-p10["fMean"][i],
            "f10Median": p10["fMedian"][i],
            "f5Median": p5["fMedian"][i],
            "velocityCellCount10": int(p10["Ucount"][i]),
            "velocityCellCount5": int(p5["Ucount"][i]),
            "pressurePairCount10": int(p10["fCount"][i]),
            "pressurePairCount5": int(p5["fCount"][i]),
        })
    write_csv(out/"radial_compare.csv", radial_rows)

    s10 = region_summary("10slabD", c10, q10, args.z0_D, args.z1_D, args.bulk, args.reference_f)
    s5 = region_summary("5slabD", c5, q5, args.z0_D, args.z1_D, args.bulk, args.reference_f)
    write_csv(out/"developed_region_summary.csv", s10+s5)

    a10 = axial_region_profile("10slabD", q10, 12.0, 19.5, args.axial_bin_D)
    a5 = axial_region_profile("5slabD", q5, 12.0, 19.5, args.axial_bin_D)
    write_csv(out/"axial_f_by_region_10slabD.csv", a10)
    write_csv(out/"axial_f_by_region_5slabD.csv", a5)

    plot_radial_f(out/"radial_f_compare.png", p10, p5, args.reference_f)
    plot_radial_uz(out/"radial_uz_compare.png", p10, p5, args.bulk)
    plot_radial_du(out/"radial_delta_uz.png", p10, p5, args.bulk)
    plot_axial_regions(out/"axial_f_regions_10slabD.png", a10, args.reference_f,
                       "10 slabs/D: axial friction by radial region")
    plot_axial_regions(out/"axial_f_regions_5slabD.png", a5, args.reference_f,
                       "5 slabs/D: axial friction by radial region")

    # Compact terminal summary.
    by10 = {r["region"]: r for r in s10}
    by5 = {r["region"]: r for r in s5}
    print(
        "NODALS_RADIAL_DIAG_CONFIG "
        f"zWindowD=[{args.z0_D:g},{args.z1_D:g}] "
        f"radialBins={args.radial_bins} fReference={args.reference_f:.10f} "
        f"case10Cells={c10['ncell']} case5Cells={c5['ncell']} status=PASS"
    )
    for reg in ["core", "mid", "nearwall", "wall_touch", "all"]:
        r10 = by10[reg]
        r5 = by5[reg]
        print(
            "NODALS_RADIAL_DIAG_REGION "
            f"region={reg} "
            f"f10={r10['fMean']:.10f} f5={r5['fMean']:.10f} "
            f"deltaF5minus10={r5['fMean']-r10['fMean']:.10e} "
            f"Uz10OverUbulk={r10['UzVolMeanOverUbulk']:.10f} "
            f"Uz5OverUbulk={r5['UzVolMeanOverUbulk']:.10f} "
            f"deltaUzOverUbulk={(r5['UzVolMean']-r10['UzVolMean'])/args.bulk:.10e} "
            f"err10Pct={r10['errorVsReferencePercent']:.6f} "
            f"err5Pct={r5['errorVsReferencePercent']:.6f} status=PASS"
        )

    # RMS profile change in three radial bands is a useful "where it manifests" metric.
    rr = p10["rR"]
    du = (p5["Uz"] - p10["Uz"]) / args.bulk
    for name, mask in [
        ("core", rr < 0.5),
        ("mid", (rr >= 0.5) & (rr < 0.8)),
        ("nearwall", rr >= 0.8),
    ]:
        vals = du[mask & np.isfinite(du)]
        rms = float(np.sqrt(np.mean(vals*vals))) if vals.size else float("nan")
        mx = float(np.max(np.abs(vals))) if vals.size else float("nan")
        print(
            "NODALS_RADIAL_DIAG_PROFILE_CHANGE "
            f"region={name} rmsDeltaUzOverUbulk={rms:.10e} "
            f"maxAbsDeltaUzOverUbulk={mx:.10e} status=PASS"
        )

    print(
        "NODALS_RADIAL_DIAG_OUTPUT "
        f"outdir={out} radialCsv={out/'radial_compare.csv'} "
        f"summaryCsv={out/'developed_region_summary.csv'} "
        f"radialFPlot={out/'radial_f_compare.png'} "
        f"radialUzPlot={out/'radial_uz_compare.png'} "
        f"deltaUzPlot={out/'radial_delta_uz.png'} status=PASS"
    )


if __name__ == "__main__":
    main()
