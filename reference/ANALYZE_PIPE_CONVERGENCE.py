#!/usr/bin/env python3
import sys,re,math,glob,os

pat = {
'cells': re.compile(r'P1BF3_MESH .* cells=(\d+)'),
'geom': re.compile(r'P1BF3_PIPE_GEOMETRY .* L=([0-9eE+\-.]+) R=([0-9eE+\-.]+) D=([0-9eE+\-.]+) Uchar=([0-9eE+\-.]+) Re=([0-9eE+\-.]+) nu=([0-9eE+\-.]+)'),
'err': re.compile(r'P1BF3_PIPE_ERROR U_L2=([0-9eE+\-.]+) U_relL2=([0-9eE+\-.]+) U_meshNormalized_L2=([0-9eE+\-.]+) U_meshNormalized_relL2=([0-9eE+\-.]+) transverse_L2=([0-9eE+\-.]+) P_abs_L2=([0-9eE+\-.]+) P_abs_relL2=([0-9eE+\-.]+) P_shifted_L2=([0-9eE+\-.]+)'),
'flow': re.compile(r'P1BF3_PIPE_FLOW .* massRelative=([0-9eE+\-.]+)'),
'press': re.compile(r'P1BF3_PIPE_PRESSURE .* dropFaceOwner=([0-9eE+\-.]+) dropAxialFit=([0-9eE+\-.]+) hpDrop=([0-9eE+\-.]+) faceRelError=([0-9eE+\-.]+) fitRelError=([0-9eE+\-.]+)'),
'work': re.compile(r'P1BF3_WORK outerIts=(\d+) .* avgPCG=([0-9eE+\-.]+)'),
'time': re.compile(r'P1BF3_TIMING .* simpleSolveSeconds=([0-9eE+\-.]+) totalSeconds=([0-9eE+\-.]+)'),
'result': re.compile(r'P1BF3_RESULT status=(PASS|FAIL)')}

def expand(args):
    out=[]
    for a in args:
        if os.path.isdir(a): out += glob.glob(os.path.join(a,'**','*.log'),recursive=True)
        else:
            g=glob.glob(a)
            out += g if g else [a]
    return sorted(set(out))

def parse(path):
    try: s=open(path,errors='replace').read()
    except OSError: return None
    d={'path':path}
    m=pat['cells'].search(s)
    if m: d['cells']=int(m.group(1))
    m=pat['geom'].search(s)
    if m:
        d.update(L=float(m.group(1)),R=float(m.group(2)),D=float(m.group(3)),U=float(m.group(4)),Re=float(m.group(5)),nu=float(m.group(6)))
    m=pat['err'].search(s)
    if m:
        ks=['U_L2','U_rel','U_mesh_L2','U_mesh_rel','transverse','P_abs','P_abs_rel','P_shifted']
        d.update({k:float(v) for k,v in zip(ks,m.groups())})
    m=pat['flow'].search(s)
    if m: d['mass']=float(m.group(1))
    m=pat['press'].search(s)
    if m:
        ks=['dropFace','dropFit','hpDrop','faceRel','fitRel']; d.update({k:float(v) for k,v in zip(ks,m.groups())})
    m=pat['work'].search(s)
    if m: d['outer']=int(m.group(1)); d['avgP']=float(m.group(2))
    m=pat['time'].search(s)
    if m: d['solve']=float(m.group(1)); d['total']=float(m.group(2))
    m=pat['result'].search(s)
    if m: d['status']=m.group(1)
    if 'cells' in d and 'L' in d and 'R' in d:
        vol=math.pi*d['R']**2*d['L']; d['h']=(vol/d['cells'])**(1/3)
    return d if 'cells' in d and 'U_L2' in d else None

def order(a,b,key):
    ea,eb=abs(a[key]),abs(b[key])
    if ea<=0 or eb<=0 or a['h']<=0 or b['h']<=0: return float('nan')
    return math.log(ea/eb)/math.log(a['h']/b['h'])

files=expand(sys.argv[1:])
rows=[r for r in (parse(x) for x in files) if r]
if not rows:
    print('No completed P1BF3 pipe logs found.',file=sys.stderr); sys.exit(2)
# group by Re because convergence orders must compare identical physics
groups={}
for r in rows: groups.setdefault(round(r.get('Re',-1),10),[]).append(r)
for Re,rs in sorted(groups.items()):
    rs.sort(key=lambda r:r['cells'])
    print(f'\n=== Re={Re:g} ===')
    print('cells\th\tU_L2\tU_order\tP_shifted_L2\tP_order\t|fitRelDrop|\tdrop_order\tmassRel\touterIts\tavgPressureWork\tsimple_s\tstatus\tlog')
    prev=None
    for r in rs:
        uo=po=do=float('nan')
        if prev:
            uo=order(prev,r,'U_L2'); po=order(prev,r,'P_shifted'); do=order(prev,r,'fitRel')
        print(f"{r['cells']}\t{r.get('h',float('nan')):.9e}\t{r['U_L2']:.9e}\t{uo:.4f}\t{r['P_shifted']:.9e}\t{po:.4f}\t{abs(r.get('fitRel',float('nan'))):.9e}\t{do:.4f}\t{r.get('mass',float('nan')):.3e}\t{r.get('outer',-1)}\t{r.get('avgP',float('nan')):.3f}\t{r.get('solve',float('nan')):.6f}\t{r.get('status','?')}\t{r['path']}")
        prev=r
