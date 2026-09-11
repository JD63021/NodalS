#!/usr/bin/env python3
"""
Cell-center axial pressure-gradient / Darcy-friction postprocessor for NodalS VTU.

Designed for the serial GPU RANS VTU writer, but intentionally independent of
NodalS and CUDA.  It reads an ASCII VTU containing tetrahedra and cell field
p_P0, groups aligned cell centroids into axial rows, and evaluates adjacent
cell-center pressure gradients without fitting a global pressure line.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"numpy is required: {exc}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute row-wise cell-center dp/dz and local Darcy friction from NodalS VTU"
    )
    p.add_argument("vtu", type=Path)
    p.add_argument("--outdir", type=Path, default=None,
                   help="output directory; default: <vtu_stem>_axial_friction beside VTU")
    p.add_argument("--diameter", type=float, default=0.004, help="pipe diameter D")
    p.add_argument("--bulk", type=float, default=50.0, help="bulk velocity U")
    p.add_argument("--rho", type=float, default=1.0,
                   help="density factor in f=2D(-dp/dz)/(rho U^2); use 1 for NodalS kinematic pressure")
    p.add_argument("--re", type=float, default=13691.7402481, help="Reynolds number for Moody/Colebrook")
    p.add_argument("--rel-roughness", type=float, default=0.0, help="epsilon/D for Colebrook")
    p.add_argument("--reference-f", type=float, default=0.0284003,
                   help="optional fixed benchmark Darcy f; set <=0 to disable")
    p.add_argument("--xy-tol-D", type=float, default=1.0e-6,
                   help="transverse row-key tolerance divided by D")
    p.add_argument("--bin-width-D", type=float, default=0.25,
                   help="axial averaging bin width divided by D")
    p.add_argument("--tail-start-D", type=float, default=12.0,
                   help="start of printed downstream window summaries")
    p.add_argument("--window-D", type=float, default=2.0,
                   help="width of printed downstream windows")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def smooth_moody_friction(reynolds: float, rel_roughness: float) -> float:
    if not math.isfinite(reynolds) or reynolds <= 0:
        raise ValueError("Re must be positive and finite")
    if rel_roughness < 0 or not math.isfinite(rel_roughness):
        raise ValueError("relative roughness must be non-negative and finite")
    if reynolds < 2300.0:
        return 64.0 / reynolds
    f = 0.03
    for _ in range(100):
        arg = rel_roughness / 3.7 + 2.51 / (reynolds * math.sqrt(f))
        invsqrt = -2.0 * math.log10(arg)
        f_new = 1.0 / (invsqrt * invsqrt)
        if abs(f_new - f) <= 1e-14 * max(1.0, abs(f)):
            return f_new
        f = f_new
    return f


def _attrs(line: str) -> dict[str, str]:
    return dict(re.findall(r'(\w+)="([^"]*)"', line))


def read_nodals_ascii_vtu(path: Path):
    """Read only Points, connectivity and p_P0 without constructing an XML tree."""
    npts = ncells = None
    points = conn = pressure = None
    ip = ic = ipp = 0
    section = None
    target = None
    buf: list[str] = []

    def flush_buffer():
        nonlocal ip, ic, ipp, buf
        if not buf or target in (None, "skip"):
            buf = []
            return
        text = " ".join(buf)
        buf = []
        if target == "points":
            q = np.fromstring(text, sep=" ", dtype=np.float64)
            if q.size % 3:
                raise RuntimeError("malformed Points DataArray")
            q = q.reshape((-1, 3))
            points[ip:ip + q.shape[0], :] = q
            ip += q.shape[0]
        elif target == "connectivity":
            q = np.fromstring(text, sep=" ", dtype=np.int64)
            if q.size % 4:
                raise RuntimeError("expected tetrahedral connectivity")
            q = q.reshape((-1, 4))
            conn[ic:ic + q.shape[0], :] = q
            ic += q.shape[0]
        elif target == "pressure":
            q = np.fromstring(text, sep=" ", dtype=np.float64)
            pressure[ipp:ipp + q.size] = q
            ipp += q.size

    with path.open("r", encoding="utf-8", errors="strict") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue

            if "<Piece " in line:
                a = _attrs(line)
                npts = int(a["NumberOfPoints"])
                ncells = int(a["NumberOfCells"])
                points = np.empty((npts, 3), dtype=np.float64)
                conn = np.empty((ncells, 4), dtype=np.int64)
                pressure = np.empty(ncells, dtype=np.float64)

            if "<Points>" in line:
                section = "points"
            elif line.startswith("<Cells>"):
                section = "cells"
            elif line.startswith("<PointData"):
                section = "pointdata"
            elif line.startswith("<CellData"):
                section = "celldata"

            if "<DataArray" in line:
                flush_buffer()
                if points is None:
                    raise RuntimeError("VTU Piece header missing before DataArray")
                a = _attrs(line)
                name = a.get("Name", "")
                fmt = a.get("format", "")
                if fmt.lower() != "ascii":
                    raise RuntimeError("postprocessor currently requires ASCII VTU arrays")
                if section == "points" and not name:
                    target = "points"
                elif section == "cells" and name == "connectivity":
                    target = "connectivity"
                elif section == "celldata" and name == "p_P0":
                    target = "pressure"
                else:
                    target = "skip"
                continue

            if "</DataArray>" in line:
                flush_buffer()
                target = None
                if "</Points>" in line or "</Cells>" in line or \
                   "</PointData>" in line or "</CellData>" in line:
                    section = None
                continue

            if target not in (None, "skip"):
                buf.append(line)
                if len(buf) >= 8192:
                    flush_buffer()

            if line.startswith("</Points>") or line.startswith("</Cells>") or \
               line.startswith("</PointData>") or line.startswith("</CellData>"):
                section = None

    flush_buffer()
    if npts is None or ncells is None:
        raise RuntimeError("could not find VTU Piece metadata")
    if ip != npts:
        raise RuntimeError(f"points count mismatch: read {ip}, expected {npts}")
    if ic != ncells:
        raise RuntimeError(f"connectivity count mismatch: read {ic}, expected {ncells}")
    if ipp != ncells:
        raise RuntimeError(f"p_P0 count mismatch: read {ipp}, expected {ncells}")
    if conn.min(initial=0) < 0 or conn.max(initial=0) >= npts:
        raise RuntimeError("connectivity index outside point range")
    if not np.isfinite(points).all() or not np.isfinite(pressure).all():
        raise RuntimeError("VTU contains non-finite point coordinates or pressure")
    return points, conn, pressure


def cell_centroids(points: np.ndarray, conn: np.ndarray) -> np.ndarray:
    # Avoid points[conn] to keep peak memory small on million-cell meshes.
    c = points[conn[:, 0]].copy()
    c += points[conn[:, 1]]
    c += points[conn[:, 2]]
    c += points[conn[:, 3]]
    c *= 0.25
    return c


def aligned_row_pairs(centres: np.ndarray, pressure: np.ndarray, diameter: float, xy_tol_D: float):
    tol = diameter * xy_tol_D
    if not (tol > 0.0 and math.isfinite(tol)):
        raise ValueError("xy row tolerance must be positive")

    # Integer transverse row keys; safe because pipe coordinates are tiny compared with int64 range.
    kx = np.rint(centres[:, 0] / tol).astype(np.int64)
    ky = np.rint(centres[:, 1] / tol).astype(np.int64)
    order = np.lexsort((centres[:, 2], ky, kx))
    sx, sy, sz = kx[order], ky[order], centres[order, 2]

    new_group = np.ones(order.size, dtype=bool)
    new_group[1:] = (sx[1:] != sx[:-1]) | (sy[1:] != sy[:-1])
    group_id = np.cumsum(new_group, dtype=np.int64) - 1
    counts = np.bincount(group_id)
    rows_with_pairs = int(np.count_nonzero(counts >= 2))

    same = (sx[1:] == sx[:-1]) & (sy[1:] == sy[:-1])
    up = order[:-1][same]
    dn = order[1:][same]
    gid = group_id[:-1][same]
    if up.size == 0:
        raise RuntimeError(
            f"no aligned cell-centre pairs found with xy_tol_D={xy_tol_D:g}; increase --xy-tol-D"
        )

    dz = centres[dn, 2] - centres[up, 2]
    dx = centres[dn, 0] - centres[up, 0]
    dy = centres[dn, 1] - centres[up, 1]
    dr = np.hypot(dx, dy)
    valid = (dz > max(1e-15, diameter * 1e-12)) & (dr <= 2.0 * tol)
    up, dn, gid, dz, dr = up[valid], dn[valid], gid[valid], dz[valid], dr[valid]
    if up.size == 0:
        raise RuntimeError("row groups exist but no valid positive-dz aligned pairs remain")

    zmid = 0.5 * (centres[up, 2] + centres[dn, 2])
    xmid = 0.5 * (centres[up, 0] + centres[dn, 0])
    ymid = 0.5 * (centres[up, 1] + centres[dn, 1])
    dpdz_loss = (pressure[up] - pressure[dn]) / dz
    return {
        "up": up, "dn": dn, "row": gid, "dz": dz, "dr": dr,
        "xmid": xmid, "ymid": ymid, "zmid": zmid, "dpdz": dpdz_loss,
        "rows_with_pairs": rows_with_pairs, "total_row_keys": int(counts.size),
    }


def save_pair_csv(path: Path, pairs: dict, p: np.ndarray, D: float, f: np.ndarray,
                  f_moody: float, f_ref: float | None):
    n = pairs["up"].size
    ref = f_ref if f_ref is not None else math.nan
    data = np.empty((n, 15), dtype=np.float64)
    data[:, 0] = pairs["row"]
    data[:, 1] = pairs["up"]
    data[:, 2] = pairs["dn"]
    data[:, 3] = pairs["xmid"] / D
    data[:, 4] = pairs["ymid"] / D
    data[:, 5] = pairs["zmid"] / D
    data[:, 6] = pairs["dz"] / D
    data[:, 7] = pairs["dr"] / D
    data[:, 8] = p[pairs["up"]]
    data[:, 9] = p[pairs["dn"]]
    data[:, 10] = pairs["dpdz"]
    data[:, 11] = f
    data[:, 12] = 100.0 * (f / f_moody - 1.0)
    data[:, 13] = ref
    data[:, 14] = 100.0 * (f / ref - 1.0) if f_ref is not None else math.nan
    header = (
        "row_id,cell_up,cell_down,x_mid_over_D,y_mid_over_D,z_mid_over_D,dz_over_D,"
        "transverse_offset_over_D,p_up,p_down,pressure_loss_gradient,f_Darcy_local,"
        "error_vs_Moody_percent,f_reference,error_vs_reference_percent"
    )
    fmts = ["%.0f", "%.0f", "%.0f"] + ["%.12e"] * 12
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt=fmts)


def binned_profile(zD: np.ndarray, dpdz: np.ndarray, f: np.ndarray, width: float,
                   f_moody: float, f_ref: float | None):
    z0 = math.floor(float(np.min(zD)) / width) * width
    bid = np.floor((zD - z0) / width).astype(np.int64)
    nb = int(bid.max()) + 1
    count = np.bincount(bid, minlength=nb)
    sum_g = np.bincount(bid, weights=dpdz, minlength=nb)
    sum_f = np.bincount(bid, weights=f, minlength=nb)
    sum_f2 = np.bincount(bid, weights=f * f, minlength=nb)
    mean_g = np.divide(sum_g, count, out=np.full(nb, np.nan), where=count > 0)
    mean_f = np.divide(sum_f, count, out=np.full(nb, np.nan), where=count > 0)
    var = np.divide(sum_f2, count, out=np.full(nb, np.nan), where=count > 0) - mean_f * mean_f
    std_f = np.sqrt(np.maximum(var, 0.0))
    med_f = np.full(nb, np.nan)
    for b in range(nb):
        if count[b]:
            med_f[b] = float(np.median(f[bid == b]))
    zc = z0 + (np.arange(nb) + 0.5) * width
    ref = f_ref if f_ref is not None else math.nan
    return np.column_stack([
        zc, count, mean_g, mean_f, med_f, std_f,
        100.0 * (mean_f / f_moody - 1.0),
        np.full(nb, f_moody),
        np.full(nb, ref),
        100.0 * (mean_f / ref - 1.0) if f_ref is not None else np.full(nb, np.nan),
    ])


def save_profile_csv(path: Path, profile: np.ndarray):
    header = (
        "z_bin_center_over_D,count,pressure_loss_gradient_mean,f_Darcy_mean,f_Darcy_median,"
        "f_Darcy_std,error_mean_vs_Moody_percent,f_Moody,f_reference,error_mean_vs_reference_percent"
    )
    np.savetxt(path, profile, delimiter=",", header=header, comments="",
               fmt=["%.12e", "%.0f"] + ["%.12e"] * 8)


def maybe_plot(path: Path, profile: np.ndarray, f_moody: float, f_ref: float | None):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"NODALS_VTU_FRICTION_PLOT status=SKIP reason={type(exc).__name__}")
        return
    ok = profile[:, 1] > 0
    z = profile[ok, 0]
    fm = profile[ok, 3]
    fs = profile[ok, 5]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(z, fm, label="cell-pair binned mean")
    ax.fill_between(z, fm - fs, fm + fs, alpha=0.18, label="±1 std across rows")
    ax.axhline(f_moody, linestyle="--", label=f"Moody/Colebrook {f_moody:.6f}")
    if f_ref is not None:
        ax.axhline(f_ref, linestyle=":", label=f"reference {f_ref:.6f}")
    ax.set_xlabel("z / D")
    ax.set_ylabel("Darcy friction factor")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"NODALS_VTU_FRICTION_PLOT path={path} status=PASS")


def main() -> int:
    a = parse_args()
    if not a.vtu.is_file():
        raise SystemExit(f"VTU not found: {a.vtu}")
    if not (a.diameter > 0 and a.bulk > 0 and a.rho > 0 and a.bin_width_D > 0 and a.window_D > 0):
        raise SystemExit("diameter, bulk, rho, bin width and window width must be positive")

    outdir = a.outdir or a.vtu.with_name(a.vtu.stem + "_axial_friction")
    outdir.mkdir(parents=True, exist_ok=True)
    pair_csv = outdir / "cell_center_pairwise_dp_dz.csv"
    profile_csv = outdir / "axial_friction_profile.csv"
    plot_png = outdir / "axial_friction_profile.png"

    f_moody = smooth_moody_friction(a.re, a.rel_roughness)
    f_ref = a.reference_f if a.reference_f > 0 else None
    points, conn, p = read_nodals_ascii_vtu(a.vtu)
    centres = cell_centroids(points, conn)
    pairs = aligned_row_pairs(centres, p, a.diameter, a.xy_tol_D)
    f_local = 2.0 * a.diameter * pairs["dpdz"] / (a.rho * a.bulk * a.bulk)
    finite = np.isfinite(f_local) & np.isfinite(pairs["dpdz"])
    if not finite.all():
        for key in ("up", "dn", "row", "dz", "dr", "xmid", "ymid", "zmid", "dpdz"):
            pairs[key] = pairs[key][finite]
        f_local = f_local[finite]

    zD = pairs["zmid"] / a.diameter
    profile = binned_profile(zD, pairs["dpdz"], f_local, a.bin_width_D, f_moody, f_ref)
    save_pair_csv(pair_csv, pairs, p, a.diameter, f_local, f_moody, f_ref)
    save_profile_csv(profile_csv, profile)
    if not a.no_plot:
        maybe_plot(plot_png, profile, f_moody, f_ref)

    domain_min_D = float(np.min(centres[:, 2]) / a.diameter)
    domain_max_D = float(np.max(centres[:, 2]) / a.diameter)
    neg_frac = float(np.mean(pairs["dpdz"] < 0.0))
    print(
        "NODALS_VTU_FRICTION_CONFIG "
        f"vtu={a.vtu} cells={conn.shape[0]} points={points.shape[0]} D={a.diameter:.12e} "
        f"Ubulk={a.bulk:.12e} rho={a.rho:.12e} Re={a.re:.12e} relRoughness={a.rel_roughness:.12e} "
        f"fMoody={f_moody:.10f} fReference={(f_ref if f_ref is not None else float('nan')):.10f} "
        f"xyTolD={a.xy_tol_D:.3e} binWidthD={a.bin_width_D:.6g} status=PASS"
    )
    print(
        "NODALS_VTU_FRICTION_ROWS "
        f"totalRowKeys={pairs['total_row_keys']} rowsWithPairs={pairs['rows_with_pairs']} "
        f"alignedPairs={pairs['up'].size} medianDzD={np.median(pairs['dz'])/a.diameter:.12e} "
        f"maxTransverseOffsetD={np.max(pairs['dr'])/a.diameter:.12e} negativeGradientFraction={neg_frac:.12e} "
        f"domainCellCenterZ=[{domain_min_D:.6f},{domain_max_D:.6f}] status=PASS"
    )

    z = max(a.tail_start_D, float(np.min(zD)))
    zmax = float(np.max(zD))
    while z < zmax - 1e-12:
        z1 = min(z + a.window_D, zmax + 1e-12)
        m = (zD >= z) & (zD < z1)
        if np.any(m):
            fm = float(np.mean(f_local[m]))
            med = float(np.median(f_local[m]))
            gg = float(np.mean(pairs["dpdz"][m]))
            em = 100.0 * (fm / f_moody - 1.0)
            er = 100.0 * (fm / f_ref - 1.0) if f_ref is not None else math.nan
            print(
                "NODALS_VTU_FRICTION_WINDOW "
                f"z0D={z:.6f} z1D={z1:.6f} pairs={int(np.count_nonzero(m))} "
                f"dpdzMean={gg:.12e} fDarcyMean={fm:.10f} fDarcyMedian={med:.10f} "
                f"fMoody={f_moody:.10f} errorVsMoodyPercent={em:.6f} "
                f"fReference={(f_ref if f_ref is not None else float('nan')):.10f} errorVsReferencePercent={er:.6f} status=PASS"
            )
        z += a.window_D

    print(
        "NODALS_VTU_FRICTION_OUTPUT "
        f"pairCsv={pair_csv} profileCsv={profile_csv} "
        f"plot={(plot_png if not a.no_plot else 'DISABLED')} status=PASS"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"NODALS_VTU_FRICTION_EXCEPTION type={type(exc).__name__} what={exc}", file=sys.stderr)
        raise SystemExit(90)
