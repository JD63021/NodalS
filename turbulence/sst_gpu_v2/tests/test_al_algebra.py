#!/usr/bin/env python3
"""Small discrete AL identity test. Does not verify CUDA assembly or SIMPLE."""
import numpy as np

def check(seed,pin):
    rng=np.random.default_rng(seed)
    n,m=19,8
    H=rng.standard_normal((n,n))
    A=np.eye(n)*3+0.01*(H+H.T)
    T,_=np.linalg.qr(rng.standard_normal((n,n-3)))
    B=rng.standard_normal((m,n))
    active=np.ones(m,dtype=bool)
    if pin is not None:active[pin]=False
    Bb=(B@T)[active];Ab=T.T@A@T
    mp=(0.3+rng.random(m))[active]
    b=rng.standard_normal(T.shape[1])
    c=rng.standard_normal(m)[active]
    N=len(c);v=len(b)
    baseline=np.block([[Ab,Bb.T],[Bb,np.zeros((N,N))]])
    solution=np.linalg.solve(baseline,np.r_[b,c])
    for gamma in (0.,0.1,1.,100.,1e4,1e6):
        G=(Bb.T/mp)@Bb
        rhs=b+gamma*Bb.T@(c/mp)
        aug=np.block([[Ab+gamma*G,Bb.T],[Bb,np.zeros((N,N))]])
        out=np.linalg.solve(aug,np.r_[rhs,c])
        rel=np.linalg.norm(out[:v]-solution[:v])/max(1.,np.linalg.norm(solution[:v]))
        assert rel<2e-5,(gamma,rel)
        assert np.linalg.norm(Bb@out[:v]-c)<5e-7
        assert np.linalg.eigvalsh(G).min()>-1e-10
        assert np.linalg.norm(gamma*Bb.T@((Bb@out[:v]-c)/mp))<2e-3
    print(f'AL_SADDLE_INVARIANCE pin={pin} status=PASS')

check(9,None)
check(13,0)
