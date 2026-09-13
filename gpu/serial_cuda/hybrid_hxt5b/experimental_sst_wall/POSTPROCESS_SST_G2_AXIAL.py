#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, math, subprocess, sys, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np


def local_name(tag: str) -> str:
    return tag.split('}', 1)[-1]


def child(parent, name):
    if parent is None:
        return None
    for c in parent:
        if local_name(c.tag) == name:
            return c
    return None


def data_array(parent, name=None):
    if parent is None:
        return None
    for a in parent.iter():
        if local_name(a.tag) != 'DataArray':
            continue
        if name is None or a.attrib.get('Name') == name:
            return a
    return None


def parse_array(a, dtype=float):
    if a is None:
        return None
    if a.attrib.get('format', 'ascii').lower() != 'ascii':
        raise RuntimeError('Only ASCII VTU DataArrays are supported by this postprocessor')
    return np.fromstring(a.text or '', sep=' ', dtype=dtype)


def weighted_mean(x, w):
    good = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(good):
        return float('nan')
    sw = float(np.sum(w[good]))
    return float(np.sum(x[good] * w[good]) / sw) if sw > 0 else float('nan')


def weighted_fit(x, y, w):
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[good], y[good], w[good]
    if x.size < 3:
        return None
    sw = float(np.sum(w))
    xb = float(np.sum(w * x) / sw)
    yb = float(np.sum(w * y) / sw)
    dx = x - xb
    den = float(np.sum(w * dx * dx))
    if den <= 0:
        return None
    m = float(np.sum(w * dx * (y - yb)) / den)
    b = yb - m * xb
    yp = m * x + b
    ssr = float(np.sum(w * (y - yp) ** 2))
    sst = float(np.sum(w * (y - yb) ** 2))
    r2 = 1.0 - ssr / sst if sst > 0 else float('nan')
    return m, b, r2, int(x.size)


def relative_span(values):
    q = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    if q.size < 2:
        return float('nan')
    den = float(np.mean(np.abs(q)))
    return float((np.max(q) - np.min(q)) / den) if den > 0 else float('nan')


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def load_vtu(path: Path):
    root = ET.parse(path).getroot()
    piece = next(e for e in root.iter() if local_name(e.tag) == 'Piece')
    points_node = child(piece, 'Points')
    point_data = child(piece, 'PointData')
    cell_data = child(piece, 'CellData')
    cells = child(piece, 'Cells')
    pts = parse_array(data_array(points_node), float)
    pts = pts.reshape((-1, 3))
    return piece, pts, point_data, cell_data, cells


def named_cell_array(cell_data, ncell, name, ncomp=1, dtype=float, required=True):
    a = parse_array(data_array(cell_data, name), dtype)
    if a is None:
        if required:
            raise RuntimeError(f'missing CellData array: {name}')
        return None
    if ncomp > 1:
        a = a.reshape((-1, ncomp))
    if len(a) != ncell:
        raise RuntimeError(f'{name}: expected {ncell} cells, got {len(a)}')
    return a


def named_point_array(point_data, npoint, name, ncomp=1, dtype=float, required=True):
    a = parse_array(data_array(point_data, name), dtype)
    if a is None:
        if required:
            raise RuntimeError(f'missing PointData array: {name}')
        return None
    if ncomp > 1:
        a = a.reshape((-1, ncomp))
    if len(a) != npoint:
        raise RuntimeError(f'{name}: expected {npoint} points, got {len(a)}')
    return a


def cell_average_from_point_field(cells, ncell, qpoint):
    conn = parse_array(data_array(cells, 'connectivity'), np.int64)
    offs = parse_array(data_array(cells, 'offsets'), np.int64)
    if conn is None or offs is None or len(offs) != ncell:
        raise RuntimeError('SST VTU connectivity/offsets missing or inconsistent')
    out = np.empty(ncell, dtype=float)
    start = 0
    for c, end in enumerate(offs):
        ids = conn[start:int(end)]
        out[c] = float(np.mean(qpoint[ids]))
        start = int(end)
    return out


def maybe_pyplot(mode):
    if mode == 'off':
        return None
    probe = subprocess.run([sys.executable, '-c', 'import matplotlib.pyplot as plt'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if probe.returncode != 0:
        if mode == 'on':
            raise RuntimeError('matplotlib requested but import failed: ' + (probe.stderr or '').splitlines()[-1])
        print('SST_AXIAL_PLOTS status=SKIPPED reason=matplotlib_unavailable')
        return None
    import matplotlib.pyplot as plt
    return plt


def radial_profile(r_over_R, values, weights, nbins):
    edges = np.linspace(0.0, 1.0, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(r_over_R, edges) - 1, 0, nbins - 1)
    out = np.full(nbins, np.nan)
    for j in range(nbins):
        m = idx == j
        if np.any(m):
            out[j] = weighted_mean(values[m], weights[m])
    return centers, out


def main():
    ap = argparse.ArgumentParser(description='Axial postprocess for NodalS SST G2 hybrid pipe VTUs')
    ap.add_argument('--diag-vtu', required=True, type=Path,
                    help='HXT5B_full_rans_fp32_cell_diagnostics.vtu')
    ap.add_argument('--sst-vtu', type=Path, default=None,
                    help='HXT5B_full_rans_fp32_sst_g2.vtu (optional; adds k/omega profiles)')
    ap.add_argument('--outdir', required=True, type=Path)
    ap.add_argument('--D', type=float, default=0.004)
    ap.add_argument('--ub', type=float, default=50.0)
    ap.add_argument('--re', type=float, default=13691.7402481)
    ap.add_argument('--axis', choices=('x','y','z'), default='z')
    ap.add_argument('--pressure-bin-D', type=float, default=0.25)
    ap.add_argument('--pressure-window-D', type=float, default=2.0)
    ap.add_argument('--pressure-step-D', type=float, default=0.25)
    ap.add_argument('--wall-zone-D', type=float, default=0.5)
    ap.add_argument('--radial-bins', type=int, default=30)
    ap.add_argument('--profile-halfwidth-D', type=float, default=0.25)
    ap.add_argument('--developed-start-frac', type=float, default=0.70)
    ap.add_argument('--developed-end-frac', type=float, default=0.90)
    ap.add_argument('--plots', choices=('auto','on','off'), default='auto')
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    piece, pts, point_data, cell_data, cells = load_vtu(args.diag_vtu)
    ncell = int(piece.attrib['NumberOfCells'])
    npoint = int(piece.attrib['NumberOfPoints'])
    if len(pts) != npoint:
        raise RuntimeError('point count mismatch')

    p = named_cell_array(cell_data, ncell, 'pressure_Q0P0')
    U = named_cell_array(cell_data, ncell, 'velocity_cell_average_full', 3)
    V = named_cell_array(cell_data, ncell, 'cell_volume')
    C = named_cell_array(cell_data, ncell, 'cell_centroid', 3)
    wall = named_cell_array(cell_data, ncell, 'wall_spalding_mask', dtype=np.int64) == 1
    wall_area = named_cell_array(cell_data, ncell, 'wall_area')
    yplus = named_cell_array(cell_data, ncell, 'wall_yplus_spalding')
    yplus_max = named_cell_array(cell_data, ncell, 'wall_yplus_max_spalding')
    utau = named_cell_array(cell_data, ncell, 'wall_utau_spalding')
    utau2 = named_cell_array(cell_data, ncell, 'wall_utau2_mean_spalding')
    slip = named_cell_array(cell_data, ncell, 'wall_slip_spalding')
    sample_y = named_cell_array(cell_data, ncell, 'wall_sample_y')

    iax = {'x':0,'y':1,'z':2}[args.axis]
    transverse = [i for i in range(3) if i != iax]
    amin = float(np.min(pts[:, iax]))
    amax = float(np.max(pts[:, iax]))
    length = amax - amin
    LD = length / args.D
    xD = (C[:, iax] - amin) / args.D
    r = np.sqrt(C[:, transverse[0]]**2 + C[:, transverse[1]]**2)
    R = 0.5 * args.D
    rR = r / R

    f_blasius = 0.3164 / (args.re ** 0.25)
    print(f'SST_AXIAL_GEOMETRY axis={args.axis} min={amin:.12e} max={amax:.12e} L={length:.12e} D={args.D:.12e} L_over_D={LD:.9f}')
    print(f'SST_AXIAL_REFERENCE Re={args.re:.12e} Ub={args.ub:.12e} fBlasius={f_blasius:.12e}')

    # Pressure section bins.
    binw = args.pressure_bin_D
    pedges = np.arange(0.0, LD + binw * 1.0001, binw)
    if pedges[-1] < LD:
        pedges = np.append(pedges, LD)
    pidx = np.clip(np.digitize(xD, pedges) - 1, 0, len(pedges)-2)
    pressure_rows = []
    for j in range(len(pedges)-1):
        m = pidx == j
        if not np.any(m):
            continue
        zmid = weighted_mean(xD[m], V[m])
        pressure_rows.append(dict(
            xD=zmid,
            pMean=weighted_mean(p[m], V[m]),
            UzMean=weighted_mean(U[m, iax], V[m]),
            volume=float(np.sum(V[m])),
            cells=int(np.count_nonzero(m))))
    write_csv(args.outdir/'01_axial_section_means.csv', pressure_rows)

    px = np.array([q['xD'] for q in pressure_rows])
    pp = np.array([q['pMean'] for q in pressure_rows])
    pw = np.array([q['volume'] for q in pressure_rows])

    # Sliding pressure-gradient friction factor.
    sliding_rows = []
    half = 0.5 * args.pressure_window_D
    centers = np.arange(half, LD-half + 0.5*args.pressure_step_D, args.pressure_step_D)
    for c in centers:
        m = (px >= c-half) & (px <= c+half)
        fit = weighted_fit(px[m]*args.D + amin, pp[m], pw[m])
        if fit is None:
            continue
        slope, intercept, r2, n = fit
        fd = -2.0 * args.D * slope / (args.ub**2)
        sliding_rows.append(dict(xCenter_D=c, windowD=args.pressure_window_D,
                                 dpdxKinematic=slope, fDarcyPressure=fd,
                                 errorVsBlasiusPct=100.0*(fd/f_blasius-1.0), R2=r2, bins=n))
    write_csv(args.outdir/'02_pressure_sliding_friction.csv', sliding_rows)

    # Axial wall diagnostics.
    wall_rows = []
    wz = args.wall_zone_D
    z0 = 0.0
    while z0 < LD - 1e-12:
        z1 = min(LD, z0+wz)
        m = wall & (xD >= z0) & (xD < z1 + (1e-12 if z1 == LD else 0.0)) & (wall_area > 0)
        if np.any(m):
            aa = wall_area[m]
            fdw = 8.0 * weighted_mean(utau2[m], aa) / (args.ub**2)
            wall_rows.append(dict(
                xStart_D=z0, xEnd_D=z1, xMid_D=0.5*(z0+z1),
                wallDarcy=fdw, errorVsBlasiusPct=100.0*(fdw/f_blasius-1.0),
                yPlusMean=weighted_mean(yplus[m], aa),
                yPlusMax=float(np.nanmax(yplus_max[m])),
                uTauMean=weighted_mean(utau[m], aa),
                slipMean=weighted_mean(slip[m], aa),
                sampleYMean=weighted_mean(sample_y[m], aa),
                wallArea=float(np.sum(aa)), cells=int(np.count_nonzero(m))))
        z0 = z1
    write_csv(args.outdir/'03_wall_axial.csv', wall_rows)

    # Optional SST fields from point VTU -> simple nodal cell averages.
    kcell = omegacell = None
    if args.sst_vtu is not None and args.sst_vtu.exists():
        spiece, spts, spd, scd, scells = load_vtu(args.sst_vtu)
        if int(spiece.attrib['NumberOfCells']) != ncell or int(spiece.attrib['NumberOfPoints']) != npoint:
            raise RuntimeError('SST VTU topology count differs from diagnostic VTU')
        if np.max(np.abs(spts - pts)) > 1e-12:
            raise RuntimeError('SST VTU point coordinates differ from diagnostic VTU')
        kp = named_point_array(spd, npoint, 'k_vertex')
        op = named_point_array(spd, npoint, 'omega_vertex')
        kcell = cell_average_from_point_field(scells, ncell, kp)
        omegacell = cell_average_from_point_field(scells, ncell, op)
        sst_rows = []
        for j in range(len(pedges)-1):
            m = pidx == j
            if not np.any(m):
                continue
            sst_rows.append(dict(xD=weighted_mean(xD[m], V[m]),
                                 kMean=weighted_mean(kcell[m], V[m]),
                                 omegaMean=weighted_mean(omegacell[m], V[m]),
                                 volume=float(np.sum(V[m]))))
        write_csv(args.outdir/'04_sst_axial_means.csv', sst_rows)
        print('SST_AXIAL_SST_FIELDS status=PASS fields=k_vertex,omega_vertex')
    else:
        print('SST_AXIAL_SST_FIELDS status=SKIPPED reason=no_sst_vtu')

    # Radial profiles at useful stations; same policy automatically scales to 100D.
    auto_stations = [0.5, 1.0, 0.25*LD, 0.50*LD, 0.70*LD, 0.80*LD, 0.90*LD, 0.95*LD]
    stations = []
    for s in auto_stations:
        s = max(0.0, min(LD, s))
        if all(abs(s-t) > 1e-8 for t in stations):
            stations.append(s)
    profile_rows = []
    profile_curves = []
    for s in stations:
        m = np.abs(xD-s) <= args.profile_halfwidth_D
        if not np.any(m):
            continue
        Ub_sec = weighted_mean(U[m, iax], V[m])
        rr, uz = radial_profile(rR[m], U[m, iax], V[m], args.radial_bins)
        kval = oval = None
        if kcell is not None:
            _, kval = radial_profile(rR[m], kcell[m], V[m], args.radial_bins)
            _, oval = radial_profile(rR[m], omegacell[m], V[m], args.radial_bins)
        profile_curves.append((s, rr, uz, Ub_sec, kval, oval))
        for j in range(len(rr)):
            profile_rows.append(dict(
                station_D=s, rOverR=rr[j], Uz=uz[j], UzOverUb=(uz[j]/args.ub if math.isfinite(uz[j]) else float('nan')),
                UzOverSectionMean=(uz[j]/Ub_sec if math.isfinite(uz[j]) and Ub_sec != 0 else float('nan')),
                k=(kval[j] if kval is not None else float('nan')),
                omega=(oval[j] if oval is not None else float('nan'))))
    write_csv(args.outdir/'05_radial_profiles.csv', profile_rows)

    # Developed-candidate window (default 70%-90%, i.e. 7D-9D for 10D).
    d0 = args.developed_start_frac * LD
    d1 = args.developed_end_frac * LD
    if d1 <= d0:
        raise RuntimeError('developed window is empty')
    mp = (px >= d0) & (px <= d1)
    fit = weighted_fit(px[mp]*args.D + amin, pp[mp], pw[mp])
    if fit is None:
        raise RuntimeError('not enough pressure bins in developed candidate window')
    dslope, db, dr2, dn = fit
    fd_dev_p = -2.0 * args.D * dslope / (args.ub**2)
    mw = np.array([(q['xMid_D'] >= d0 and q['xMid_D'] <= d1) for q in wall_rows], dtype=bool)
    fdw_vals = [wall_rows[i]['wallDarcy'] for i in range(len(wall_rows)) if mw[i]]
    y_vals = [wall_rows[i]['yPlusMean'] for i in range(len(wall_rows)) if mw[i]]
    wa_vals = np.array([wall_rows[i]['wallArea'] for i in range(len(wall_rows)) if mw[i]], dtype=float)
    if fdw_vals:
        fdw_dev = weighted_mean(np.array(fdw_vals), wa_vals)
        y_dev = weighted_mean(np.array(y_vals), wa_vals)
        fdw_span = relative_span(fdw_vals)
    else:
        fdw_dev = y_dev = fdw_span = float('nan')

    # Last wall zone and pressure friction near outlet for outlet-effect visibility.
    last_wall = wall_rows[-1] if wall_rows else None
    downstream_sliding = [q for q in sliding_rows if q['xCenter_D'] >= d0 and q['xCenter_D'] <= d1]
    pressure_dev_span = relative_span([q['fDarcyPressure'] for q in downstream_sliding])

    summary_lines = [
        'NODALS SST G2 AXIAL POSTPROCESS',
        f'geometry: axis={args.axis} L/D={LD:.9f} D={args.D:.9e} Ub={args.ub:.9e} Re={args.re:.9e}',
        f'reference: Blasius Darcy f={f_blasius:.9e}',
        f'developed_candidate_window: {d0:.6g}D to {d1:.6g}D',
        f'developed_pressure_fit: dpdx={dslope:.12e} fDarcy={fd_dev_p:.12e} R2={dr2:.9f} bins={dn}',
        f'developed_wall: fDarcy={fdw_dev:.12e} yPlusMean={y_dev:.9e} wallFrictionRelSpan={fdw_span:.9e}',
        f'developed_pressure_sliding_rel_span={pressure_dev_span:.9e}',
        f'pressure_vs_wall_friction_difference_pct={100.0*(fd_dev_p/fdw_dev-1.0) if math.isfinite(fdw_dev) and fdw_dev!=0 else float("nan"):.9e}',
        f'wall_vs_blasius_pct={100.0*(fdw_dev/f_blasius-1.0) if math.isfinite(fdw_dev) else float("nan"):.9e}',
        f'pressure_vs_blasius_pct={100.0*(fd_dev_p/f_blasius-1.0):.9e}',
    ]
    if last_wall is not None:
        summary_lines.append(f'last_wall_zone: {last_wall["xStart_D"]:.6g}D-{last_wall["xEnd_D"]:.6g}D fDarcy={last_wall["wallDarcy"]:.12e} yPlus={last_wall["yPlusMean"]:.9e}')
    (args.outdir/'SUMMARY.txt').write_text('\n'.join(summary_lines) + '\n')

    for line in summary_lines:
        print('SST_AXIAL_SUMMARY ' + line)

    # Plots.
    plt = maybe_pyplot(args.plots)
    if plt is not None:
        if pressure_rows:
            fig, ax = plt.subplots(figsize=(10,5.5))
            ax.plot(px, pp, marker='o', ms=2)
            ax.axvspan(d0, d1, alpha=0.12)
            ax.set_xlabel(f'{args.axis}/D')
            ax.set_ylabel('mean kinematic pressure')
            ax.grid(True, alpha=0.25)
            fig.tight_layout(); fig.savefig(args.outdir/'P01_pressure_vs_xD.png', dpi=220); plt.close(fig)
        if sliding_rows or wall_rows:
            fig, ax = plt.subplots(figsize=(10,5.5))
            if sliding_rows:
                ax.plot([q['xCenter_D'] for q in sliding_rows], [q['fDarcyPressure'] for q in sliding_rows], label='pressure-gradient f_D')
            if wall_rows:
                ax.plot([q['xMid_D'] for q in wall_rows], [q['wallDarcy'] for q in wall_rows], label='wall-shear f_D')
            ax.axhline(f_blasius, ls='--', label='Blasius')
            ax.axvspan(d0, d1, alpha=0.12)
            ax.set_xlabel(f'{args.axis}/D'); ax.set_ylabel('Darcy friction factor'); ax.grid(True, alpha=0.25); ax.legend()
            fig.tight_layout(); fig.savefig(args.outdir/'P02_friction_vs_xD.png', dpi=220); plt.close(fig)
        if wall_rows:
            fig, ax = plt.subplots(figsize=(10,5.5))
            ax.plot([q['xMid_D'] for q in wall_rows], [q['yPlusMean'] for q in wall_rows], label='mean y+')
            ax.plot([q['xMid_D'] for q in wall_rows], [q['yPlusMax'] for q in wall_rows], label='max y+')
            ax.axvspan(d0, d1, alpha=0.12)
            ax.set_xlabel(f'{args.axis}/D'); ax.set_ylabel('y+'); ax.grid(True, alpha=0.25); ax.legend()
            fig.tight_layout(); fig.savefig(args.outdir/'P03_yplus_vs_xD.png', dpi=220); plt.close(fig)
        if profile_curves:
            fig, ax = plt.subplots(figsize=(8,6))
            for s, rr, uz, Ubsec, kval, oval in profile_curves:
                ax.plot(rr, uz/args.ub, label=f'{s:g}D')
            ax.set_xlabel('r/R'); ax.set_ylabel('Uz / global Ub'); ax.grid(True, alpha=0.25); ax.legend(fontsize=8)
            fig.tight_layout(); fig.savefig(args.outdir/'P04_velocity_profiles.png', dpi=220); plt.close(fig)
            if kcell is not None:
                fig, ax = plt.subplots(figsize=(8,6))
                for s, rr, uz, Ubsec, kval, oval in profile_curves:
                    ax.plot(rr, kval, label=f'{s:g}D')
                ax.set_xlabel('r/R'); ax.set_ylabel('k'); ax.grid(True, alpha=0.25); ax.legend(fontsize=8)
                fig.tight_layout(); fig.savefig(args.outdir/'P05_k_profiles.png', dpi=220); plt.close(fig)
                fig, ax = plt.subplots(figsize=(8,6))
                for s, rr, uz, Ubsec, kval, oval in profile_curves:
                    ax.plot(rr, oval, label=f'{s:g}D')
                ax.set_xlabel('r/R'); ax.set_ylabel('omega'); ax.grid(True, alpha=0.25); ax.legend(fontsize=8)
                fig.tight_layout(); fig.savefig(args.outdir/'P06_omega_profiles.png', dpi=220); plt.close(fig)
        print('SST_AXIAL_PLOTS status=PASS')

    print(f'SST_AXIAL_OUTPUT outdir={args.outdir}')
    print('SST_AXIAL_POSTPROCESS status=PASS')


if __name__ == '__main__':
    main()
