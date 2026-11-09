#!/usr/bin/env python3
"""Standalone, log-only NodalS SST live monitor (no numerical solver coupling).

Reads optional residual/physics diagnostics from terminal.log. Missing data are
simply not plotted. Only matplotlib is needed for GUI/snapshot modes. The
solver never reads anything from this program.
"""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
import re
import sys
import time

KEYVAL = re.compile(r'([A-Za-z][A-Za-z0-9_]*)=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?|nan|inf|-inf)', re.I)
ABS_LABELS = ('UxAbs', 'UyAbs', 'UzAbs', 'pAbs', 'contAbs', 'kAbs', 'omegaAbs')
REL_LABELS = ('UxRel', 'UyRel', 'UzRel', 'pRel', 'contRel', 'kRel', 'omegaRel', 'kUpdateRel', 'omegaUpdateRel', 'nutUpdateRel')
PHYS_LABELS = ('pressureDrop', 'wallDarcy')
PREFIXES = (
    'NODALS_SST_RESIDUALS ', 'NODALS_SST_G2 ', 'NODALS_SST_G2_FINAL ',
    'NODALS_SST_PLOT_METRICS ', 'NODALS_SST_COMPACT_PHYS ',
    'NODALS_SST_COMPACT_ABS ', 'NODALS_SST_COMPACT_REL ',
)

def _safe_float(txt):
    try:
        val = float(txt)
        return val if math.isfinite(val) else None
    except (ValueError, TypeError):
        return None

class SeriesStore:
    def __init__(self, max_points=5000):
        self.samples = {}
        self.max_points = max(50, int(max_points))
        self.version = 0

    def consume(self, line):
        line=line.strip()
        prefix=next((k for k in PREFIXES if line.startswith(k)), None)
        if prefix is None:
            return False
        vals={k: _safe_float(v) for k,v in KEYVAL.findall(line)}
        raw_it=vals.get('it',vals.get('iterations'))
        if raw_it is None or raw_it < 0 or int(raw_it) != raw_it:
            return False
        it=int(raw_it)
        row=self.samples.setdefault(it,{})
        updated=False
        def put(k, val, fallback=False):
            nonlocal updated
            if val is None:
                return
            if fallback and k in row:
                return
            if row.get(k)!=val:
                row[k]=val
                updated=True

        if prefix.startswith('NODALS_SST_RESIDUALS'):
            for key in ABS_LABELS+REL_LABELS:
                put(key,vals.get(key))
            # Some logs use longer update labels.
            put('nutUpdateRel',vals.get('nutMeanRelUpdate'))
        elif prefix.startswith('NODALS_SST_G2 '):
            put('contAbs',vals.get('absCont'),fallback=True)
            m=re.search(r'absMom=\[([^]]+)\]',line)
            if m:
                parts=m.group(1).split(',')
                if len(parts)==3:
                    for key,p in zip(ABS_LABELS[:3],parts):
                        put(key,_safe_float(p),fallback=True)
            put('pRel',vals.get('pRel'),fallback=True)
            put('kRel',vals.get('kLinRel'),fallback=True)
            put('omegaRel',vals.get('omegaLinRel'),fallback=True)
            put('kUpdateRel',vals.get('kRelUpdate'),fallback=True)
            put('omegaUpdateRel',vals.get('omegaRelUpdate'),fallback=True)
            put('nutUpdateRel',vals.get('nutMeanRelUpdate'),fallback=True)
            for key in PHYS_LABELS:
                put(key,vals.get(key),fallback=True)
        elif prefix.startswith('NODALS_SST_COMPACT_ABS'):
            # Legacy compact output is 'Ux=initial->final' rather than UxAbs=.
            for name,key in (('Ux','UxAbs'),('Uy','UyAbs'),('Uz','UzAbs'),
                             ('p','pAbs'),('k','kAbs'),('omega','omegaAbs')):
                m=re.search(r'\b'+name+r'=\S+?->([-+\d.eE]+)',line)
                if m:put(key,_safe_float(m.group(1)),fallback=True)
        elif prefix.startswith('NODALS_SST_COMPACT_REL'):
            for old,new in (('Ux','UxRel'),('Uy','UyRel'),('Uz','UzRel'),
                            ('k','kRel'),('omega','omegaRel'),('continuity','contRel')):
                put(new,vals.get(old),fallback=True)
        elif prefix.startswith(('NODALS_SST_PLOT_METRICS', 'NODALS_SST_COMPACT_PHYS', 'NODALS_SST_G2_FINAL')):
            for key in PHYS_LABELS:
                put(key,vals.get(key))

        if len(self.samples)>self.max_points:
            n_remove=len(self.samples)-self.max_points
            for old in sorted(self.samples)[:n_remove]:del self.samples[old]
            updated=True
        if updated:self.version+=1
        return updated

    def data(self, key, positive=False, scale=1.0):
        xs=[];ys=[]
        for it,row in sorted(self.samples.items()):
            v=row.get(key)
            if v is not None and (not positive or v>0):
                xs.append(it);ys.append(v*scale)
        return xs,ys

class LogFollower:
    """Byte-offset tailer; survives missing file, truncation and rotation."""
    def __init__(self, path, store):
        self.path=Path(path)
        self.store=store
        self.handle=None
        self.ident=None
        self.pos=0
        self.pending=b''

    def poll(self):
        try:st=self.path.stat()
        except (FileNotFoundError,PermissionError):return False
        ident=(st.st_dev,st.st_ino)
        if self.handle is None or ident!=self.ident or st.st_size<self.pos:
            if self.handle is not None:self.handle.close()
            self.handle=self.path.open('rb')
            self.ident=ident
            self.pos=0
            self.pending=b''
            # Files reused with the same path are a new plotted run.
            self.store.samples.clear();self.store.version+=1
        self.handle.seek(self.pos)
        chunk=self.handle.read()
        self.pos=self.handle.tell()
        if not chunk:return False
        content=self.pending+chunk
        parts=content.split(b'\n')
        self.pending=parts.pop()
        changed=False
        for b in parts:
            try:line=b.decode('utf-8','replace')
            except Exception:continue
            changed=self.store.consume(line) or changed
        return changed


def draw(fig, axs, store, pressure_scale=1.0):
    ax_abs,ax_rel,ax_phys,ax_wall=axs
    for ax in (ax_abs,ax_rel,ax_phys,ax_wall):
        ax.cla()
    for key in ABS_LABELS:
        x,y=store.data(key,positive=True)
        if x:ax_abs.semilogy(x,y,label=key,linewidth=1.0)
    for key in REL_LABELS:
        x,y=store.data(key,positive=True)
        if x:ax_rel.semilogy(x,y,label=key,linewidth=1.0)
    x,y=store.data('pressureDrop',scale=pressure_scale)
    if x:ax_phys.plot(x,y,label='pressureDrop',linewidth=1.2)
    x,y=store.data('wallDarcy')
    if x:ax_wall.plot(x,y,label='wallDarcy',linewidth=1.0,linestyle='--')
    ax_abs.set_title('Absolute residuals (available fields only)')
    ax_rel.set_title('Relative residuals and SST updates (available fields only)')
    ax_phys.set_title('Pressure drop and normalized wall traction')
    ax_abs.set_ylabel('Absolute (log)')
    ax_rel.set_ylabel('Relative (log)')
    ax_phys.set_ylabel('Pressure drop'+(' (scaled)' if pressure_scale!=1 else ' (native)'))
    ax_wall.set_ylabel('wallDarcy (dimensionless)')
    for ax in (ax_abs,ax_rel,ax_phys):
        ax.set_xlabel('SIMPLE iteration')
        ax.grid(True,alpha=.25)
    for ax in (ax_abs,ax_rel,ax_phys,ax_wall):
        h,l=ax.get_legend_handles_labels()
        if h:ax.legend(loc='best',fontsize=7,ncol=2)
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.965))
    fig.canvas.draw_idle()


def selftest():
    store=SeriesStore()
    lines=[
        'NODALS_SST_RESIDUALS it=10 UxAbs=1e-3 UyAbs=2e-3 UzAbs=3e-3 pAbs=4e-5 contAbs=5e-6 kAbs=6e-7 omegaAbs=7e-6 UxRel=0.1 UyRel=0.2 UzRel=0.3 pRel=0.4 contRel=0.5 kRel=0.6 omegaRel=0.7 kUpdateRel=0.01 omegaUpdateRel=0.02',
        'NODALS_SST_G2 it=11 absCont=1e-7 absMom=[1e-3,2e-3,3e-3] pRel=0.03 kRelUpdate=0.005 omegaRelUpdate=0.006 wallDarcy=0.0375 pressureDrop=1101.2',
        'NODALS_SST_PLOT_METRICS it=10 pressureDrop=1100.1 wallDarcy=0.038',
        'NODALS_SST_COMPACT_PHYS it=12 wallDarcy=0.040 pressureDrop=1112.4',
        'NODALS_SST_RESIDUALS it=13 UxAbs=NaN UyAbs=-inf UzAbs=0.0',
        'unrelated debug line',
    ]
    for line in lines:store.consume(line)
    assert len(store.samples)==4
    assert store.samples[10]['UxAbs']==1e-3
    assert store.samples[11]['pressureDrop']==1101.2
    assert store.samples[12]['wallDarcy']==.04
    assert 'UxAbs' not in store.samples[13]
    assert store.data('UxAbs',positive=True)[0]==[10,11]
    x=SeriesStore()
    x.consume('NODALS_SST_RESIDUALS it=1 kAbs=1e-3')
    assert x.data('pressureDrop')==([],[])
    print('NODALS_MONITOR_SELFTEST=PASS parser=RESIDUALS_G2_METRICS_COMPACT missingData=BLANK noPhysicsLocks=1')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--log',type=Path,help='Existing or future terminal.log')
    ap.add_argument('--interval',type=float,default=3.0,help='GUI refresh in seconds (default 3)')
    ap.add_argument('--pressure-scale',type=float,default=1.0,help='Only display scaling, e.g. 1000 for kinematic to Pa')
    ap.add_argument('--max-points',type=int,default=5000)
    ap.add_argument('--snapshot',type=Path,help='Save one PNG (headless) and exit; optional')
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:
        selftest();return 0
    if not args.log:
        ap.error('--log is required unless --self-test')
    if not (0.2<=args.interval<=3600 and math.isfinite(args.interval)):
        ap.error('--interval must be finite and in [0.2, 3600]')
    if not (math.isfinite(args.pressure_scale) and args.pressure_scale>0):
        ap.error('--pressure-scale must be finite and positive')
    if args.max_points<50:ap.error('--max-points must be >=50')
    try:
        if args.snapshot:
            import matplotlib
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        if not args.snapshot and plt.get_backend().lower() in ('agg','pdf','svg','ps','cairo','template'):
            raise RuntimeError('Noninteractive matplotlib backend: '+plt.get_backend())
    except Exception as e:
        print('NODALS_MONITOR_ERROR plotting unavailable: '+str(e),file=sys.stderr,flush=True)
        return 0  # The plotting utility is never a solver failure condition.
    store=SeriesStore(args.max_points)
    follow=LogFollower(args.log,store)
    try:
        follow.poll()
        fig,axes=plt.subplots(3,1,figsize=(11,9))
        wall=axes[2].twinx()
        fig.suptitle('NodalS live monitor — '+str(args.log.parent.name),fontsize=11)
        view=(axes[0],axes[1],axes[2],wall)
        draw(fig,view,store,args.pressure_scale)
        if args.snapshot:
            args.snapshot.parent.mkdir(parents=True,exist_ok=True)
            fig.savefig(args.snapshot,dpi=90)
            print('NODALS_MONITOR_SNAPSHOT='+str(args.snapshot),flush=True)
            plt.close(fig)
            return 0
        plt.ion()
        plt.show(block=False)
        while plt.fignum_exists(fig.number):
            if follow.poll():
                draw(fig,view,store,args.pressure_scale)
            plt.pause(args.interval)
        return 0
    except Exception as exc:
        print('NODALS_MONITOR_ERROR monitor stopped: '+str(exc),file=sys.stderr,flush=True)
        return 0

if __name__=='__main__':
    sys.exit(main())
