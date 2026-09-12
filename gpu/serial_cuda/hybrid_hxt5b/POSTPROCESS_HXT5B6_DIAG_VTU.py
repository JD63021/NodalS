#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,math,subprocess,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

def load_pyplot(mode):
    """Return matplotlib.pyplot or None without letting a broken binary import abort analysis."""
    if mode == "off":
        return None
    probe = subprocess.run(
        [sys.executable, "-c", "import matplotlib.pyplot as plt"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    if probe.returncode != 0:
        tail = (probe.stderr or "").strip().splitlines()
        reason = tail[-1] if tail else "matplotlib import failed"
        if mode == "on":
            raise RuntimeError("plotting requested but matplotlib is unusable: " + reason)
        print("HXT5B6_PLOT_STATUS=SKIPPED reason=matplotlib_import_failed")
        return None
    import matplotlib.pyplot as plt
    return plt

def ln(t):return t.split('}',1)[-1]
def child(p,n):
    for c in p:
        if ln(c.tag)==n:return c
def da(p,n):
    if p is None:return None
    for a in p.iter():
        if ln(a.tag)=='DataArray' and a.attrib.get('Name')==n:return a
def ar(a,dtype=float):
    if a is None:return None
    if a.attrib.get('format','ascii').lower()!='ascii':raise RuntimeError('non-ASCII VTU unsupported')
    return np.fromstring(a.text or '',sep=' ',dtype=dtype)
def wm(x,w):
    s=np.sum(w);return float(np.sum(x*w)/s) if s>0 else float('nan')
def fit(x,y,w):
    g=np.isfinite(x)&np.isfinite(y)&np.isfinite(w)&(w>0);x,y,w=x[g],y[g],w[g]
    if len(x)<3:return None
    xb,yb=wm(x,w),wm(y,w);xx=x-xb;den=float(np.sum(w*xx*xx))
    if den<=0:return None
    m=float(np.sum(w*xx*(y-yb))/den);b=yb-m*xb;yp=m*x+b
    ssr=float(np.sum(w*(y-yp)**2));sst=float(np.sum(w*(y-yb)**2));r2=1-ssr/sst if sst>0 else float('nan')
    return m,b,r2,len(x)
def csvw(p,rows):
    if not rows:return
    with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def prel(a,b):
    qa,qb=a['u']/a['Ub'],b['u']/b['Ub'];e=a['edges'];aw=e[1:]**2-e[:-1]**2;g=np.isfinite(qa)&np.isfinite(qb)&(aw>0)
    return float(np.sqrt(np.sum(aw[g]*(qa[g]-qb[g])**2)/np.sum(aw[g]*qb[g]**2)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--vtu',required=True,type=Path);ap.add_argument('--outdir',required=True,type=Path);ap.add_argument('--D',type=float,default=.004);ap.add_argument('--ub',type=float,default=50);ap.add_argument('--moody',type=float,default=.0284587039);ap.add_argument('--window-D',type=float,default=2);ap.add_argument('--step-D',type=float,default=.5);ap.add_argument('--zone-D',type=float,default=1);ap.add_argument('--radial-bins',type=int,default=24);ap.add_argument('--profile-halfwidth-D',type=float,default=.5);ap.add_argument('--reference-D',type=float,default=18);ap.add_argument('--plots',choices=('auto','on','off'),default='auto',help='auto: plot when matplotlib is usable; on: require plots; off: numerical postprocess only');args=ap.parse_args();args.outdir.mkdir(parents=True,exist_ok=True)
    root=ET.parse(args.vtu).getroot();pc=next(e for e in root.iter() if ln(e.tag)=='Piece');n=int(pc.attrib['NumberOfCells']);cd=child(pc,'CellData')
    def get(name,nc=1,dtype=float):
        q=ar(da(cd,name),dtype)
        if q is None:raise RuntimeError('missing '+name)
        if nc>1:q=q.reshape((-1,nc))
        if len(q)!=n:raise RuntimeError(name+' size mismatch')
        return q
    p=get('pressure_Q0P0');U=get('velocity_cell_average_full',3);V=get('cell_volume');C=get('cell_centroid',3);wall=get('wall_spalding_mask',dtype=np.int64)==1;A=get('wall_area');yp=get('wall_yplus_spalding');ypm=get('wall_yplus_max_spalding');ut=get('wall_utau_spalding');ut2=get('wall_utau2_mean_spalding');sl=get('wall_slip_spalding');sy=get('wall_sample_y')
    zD=C[:,2]/args.D;r=np.hypot(C[:,0],C[:,1]);R=args.D/2
    # pressure sliding fits
    nb=200;edges=np.linspace(0,20,nb+1);mid=.5*(edges[:-1]+edges[1:]);idx=np.clip(np.digitize(zD,edges)-1,0,nb-1);pb=np.full(nb,np.nan);vb=np.zeros(nb)
    for j in range(nb):
        m=idx==j
        if np.any(m):pb[j]=wm(p[m],V[m]);vb[j]=np.sum(V[m])
    srows=[]
    for c in np.arange(args.window_D/2,20-args.window_D/2+1e-9,args.step_D):
        m=np.isfinite(pb)&(mid>=c-args.window_D/2)&(mid<=c+args.window_D/2);q=fit(mid[m]*args.D,pb[m],vb[m])
        if q:
            m0,b0,r2,nn=q;fd=-2*args.D*m0/(args.ub**2);srows.append(dict(zcenter_D=c,dpdz_kinematic=m0,f_D=fd,error_vs_moody_percent=100*(fd/args.moody-1),R2=r2,bins=nn))
    csvw(args.outdir/'pressure_sliding_friction.csv',srows)
    # profiles
    stations=[.5,1]+list(np.arange(2,20,1))+[19.5];pro=[]
    for s in stations:
        m=np.abs(zD-s)<=args.profile_halfwidth_D;w=V[m];uz=U[m,2];rr=r[m]/R;Ub=wm(uz,w);core=rr<=.1;uc=wm(uz[core],w[core]) if np.any(core) else np.nan;tr=np.sqrt(max(0,wm(U[m,0]**2+U[m,1]**2,w)));re=np.linspace(0,1,args.radial_bins+1);rc=.5*(re[:-1]+re[1:]);ri=np.clip(np.digitize(rr,re)-1,0,args.radial_bins-1);q=np.full(args.radial_bins,np.nan)
        for j in range(args.radial_bins):
            g=ri==j
            if np.any(g):q[j]=wm(uz[g],w[g])
        pro.append(dict(z=s,Ub=Ub,core=uc/Ub,tr=tr/abs(Ub),u=q,r=rc,edges=re))
    ref=min(pro,key=lambda q:abs(q['z']-args.reference_D));pm=[]
    for q in pro:pm.append(dict(z_D=q['z'],Ub=q['Ub'],UcoreOverUb=q['core'],transverseRmsOverUb=q['tr'],profileRelL2VsReference=prel(q,ref)))
    csvw(args.outdir/'velocity_profile_metrics.csv',pm)
    # yplus zones
    wr=[];z=0.0
    while z<20-1e-12:
        b=min(20,z+args.zone_D);m=wall&(zD>=z)&(zD<b)&(A>0)
        if np.any(m):
            aa=A[m];fd=8*wm(ut2[m],aa)/(args.ub**2);wr.append(dict(zstart_D=z,zend_D=b,zmid_D=.5*(z+b),yPlusMean=wm(yp[m],aa),yPlusMax=float(np.max(ypm[m])),uTauMean=wm(ut[m],aa),wallDarcy=fd,errorWallDarcyVsMoodyPct=100*(fd/args.moody-1),slipMean=wm(sl[m],aa),sampleY=wm(sy[m],aa),wallArea=float(np.sum(aa))))
        z=b
    csvw(args.outdir/'wall_yplus_axial.csv',wr)
    # plots are optional so numerical analysis is independent of matplotlib ABI health.
    plt=load_pyplot(args.plots)
    if plt is not None:
        fig,ax=plt.subplots(figsize=(10,5.7));ax.plot([x['zcenter_D'] for x in srows],[x['f_D'] for x in srows],marker='o',ms=3,label='pressure');ax.plot([x['zmid_D'] for x in wr],[x['wallDarcy'] for x in wr],marker='s',ms=3,label='Spalding wall');ax.axhline(args.moody,label='Moody');ax.set_xlabel('z/D');ax.set_ylabel('Darcy f');ax.grid(True,alpha=.25);ax.legend();fig.tight_layout();fig.savefig(args.outdir/'01_pressure_vs_wall_friction.png',dpi=220);plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,5.7));ax.plot([x['z_D'] for x in pm],[x['UcoreOverUb'] for x in pm],marker='o');ax.set_xlabel('z/D');ax.set_ylabel('Ucore/Ub');ax.grid(True,alpha=.25);fig.tight_layout();fig.savefig(args.outdir/'02_velocity_profile_development.png',dpi=220);plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,5.7));ax.plot([x['zmid_D'] for x in wr],[x['yPlusMean'] for x in wr],marker='o',label='mean y+');ax.plot([x['zmid_D'] for x in wr],[x['yPlusMax'] for x in wr],marker='s',label='max y+');ax.axhline(30,ls='--',label='y+=30');ax.set_xlabel('z/D');ax.set_ylabel('y+');ax.grid(True,alpha=.25);ax.legend();fig.tight_layout();fig.savefig(args.outdir/'03_yplus_vs_z.png',dpi=220);plt.close(fig)
        fig,ax=plt.subplots(figsize=(10.5,6))
        for q in pro:ax.plot(q['r'],q['u']/q['Ub'],lw=1,label=f"{q['z']:g}D")
        ax.set_xlabel('r/R');ax.set_ylabel('Uz/Ub');ax.grid(True,alpha=.25);ax.legend(ncol=2,fontsize=7);fig.tight_layout();fig.savefig(args.outdir/'04_radial_profiles.png',dpi=220);plt.close(fig)
        print('HXT5B6_PLOT_STATUS=PASS')
    print('VELOCITY_PROFILE_BEGIN')
    for x in pm:print('PROFILE_METRIC',f"zD={x['z_D']:.6g}",f"Ub={x['Ub']:.12e}",f"UcoreOverUb={x['UcoreOverUb']:.12e}",f"transverseRmsOverUb={x['transverseRmsOverUb']:.12e}",f"profileRelL2VsReference={x['profileRelL2VsReference']:.12e}")
    print('VELOCITY_PROFILE_END');print('YPLUS_AXIAL_BEGIN')
    for x in wr:print('YPLUS_ZONE',f"zD=[{x['zstart_D']:.6g},{x['zend_D']:.6g}]",f"mean={x['yPlusMean']:.9e}",f"max={x['yPlusMax']:.9e}",f"uTauMean={x['uTauMean']:.9e}",f"wallDarcy={x['wallDarcy']:.9e}",f"errMoodyPct={x['errorWallDarcyVsMoodyPct']:.6f}",f"slipMean={x['slipMean']:.9e}",f"sampleY={x['sampleY']:.9e}")
    print('YPLUS_AXIAL_END');print('HXT5B6_POSTPROCESS=PASS')
if __name__=='__main__':main()
