#!/usr/bin/env python3
"""Host-side algebra check for generalized Robin-Nitsche 3-D tangential tensor.
No CUDA dependencies. Mirrors genrobin_entry() as a scalar reference.
This checks coefficient limits, symmetry, and Robin boundary consistency;
it does NOT replace GPU assembled/matrix-free parity or CFD validation.
"""
import math

def inner(a,b): return sum(x*y for x,y in zip(a,b))
def tangent(a,n):
    z=inner(a,n)
    return [a[i]-z*n[i] for i in range(3)]

def traction(i,g,n,mu):
    e=[float(k==i) for k in range(3)]
    dn=inner(g,n)
    t1=tangent(e,n)
    t2=tangent(g,n)
    return [mu*(t1[k]*dn+n[i]*t2[k]) for k in range(3)]

def entry(i,j,n,pa,pb,ga,gb,mu,beta,beta_n):
    ea=[float(k==i) for k in range(3)]
    eb=[float(k==j) for k in range(3)]
    wa=[pa*z for z in tangent(ea,n)]
    ub=[pb*z for z in tangent(eb,n)]
    ta=traction(i,ga,n,mu)
    tb=traction(j,gb,n,mu)
    alpha=beta/(beta+beta_n)
    # Generalized term minus the ORIGINAL +beta * wall-trace Robin term
    return (-alpha*(inner(tb,wa)+inner(ta,ub)) +
            (beta*beta_n/(beta+beta_n)-beta)*inner(wa,ub) -
            inner(ta,tb)/(beta+beta_n))

def check():
    inv=1/math.sqrt(14)
    n=[1*inv,2*inv,3*inv]
    ga=[.2,-1.2,.7];gb=[1.1,.3,-.4]
    mu=.014
    for beta in [1e-5,.2,17.]:
        for bn in [.01,1.,90.]:
            for i in range(3):
                for j in range(3):
                    x=entry(i,j,n,.7,-.3,ga,gb,mu,beta,bn)
                    y=entry(j,i,n,-.3,.7,gb,ga,mu,beta,bn)
                    assert abs(x-y)<5e-13*max(1,abs(x),abs(y)),('symmetry',x,y)
    # Dirichlet limit: beta -> infinity, extra wall correction should be
    # ordinary symmetric Nitsche minus old beta, so check TOTAL term here.
    beta=1e8;bn=mu*15/.025
    i=1;j=2;pa=.3;pb=.8
    combined=entry(i,j,n,pa,pb,ga,gb,mu,beta,bn)
    pij=(float(i==j)-n[i]*n[j])
    actual=combined+beta*pa*pb*pij
    ref=-(inner(traction(j,gb,n,mu),[pa*z for z in tangent([float(k==i) for k in range(3)],n)])+
          inner(traction(i,ga,n,mu),[pb*z for z in tangent([float(k==j) for k in range(3)],n)])) + bn*pa*pb*pij
    assert abs(actual-ref)<5e-6*max(1,abs(ref)),('Dirichlet limit',actual,ref)
    # Robin limit: beta_n -> infinity => exact natural beta Robin, correction vanishes.
    val=entry(1,1,n,pa,pb,ga,gb,mu,.21,1e14)
    assert abs(val)<1e-12,('Robin limit',val)
    print('GENROBIN_ALGEBRA=PASS symmetry=27x3x3 Robin_limit=PASS Dirichlet_limit=PASS')
if __name__=='__main__':check()
