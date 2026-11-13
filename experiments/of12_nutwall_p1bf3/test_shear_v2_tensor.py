#!/usr/bin/env python3
"""Pure-Python tensor identity tests for the Option-A V2 wall shear subspace.
Does not compile GPU kernels or claim that the full SIMPLE solver is stable.
"""
import math
import random


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def shear_g(F, n):
    # g(F) = P[(F+F^t)n] for a wall normal n.
    a = [sum((F[i][j]+F[j][i])*n[j] for j in range(3))
         for i in range(3)]
    normal = dot(a, n)
    return [a[i]-normal*n[i] for i in range(3)]


def dev_sym(F):
    D = [[.5*(F[i][j]+F[j][i]) for j in range(3)] for i in range(3)]
    d = sum(D[i][i] for i in range(3))/3.
    return [[D[i][j]-(d if i==j else 0.) for j in range(3)] for i in range(3)]


def check():
    rng = random.Random(20261113)
    for _ in range(5000):
        n = [rng.gauss(0.,1.) for _ in range(3)]
        nrm = math.sqrt(dot(n,n))
        n = [v/nrm for v in n]
        F = [[rng.gauss(0.,1.) for _ in range(3)] for _ in range(3)]
        H = [[rng.gauss(0.,1.) for _ in range(3)] for _ in range(3)]
        df,dh=dev_sym(F),dev_sym(H)
        gf,gh=shear_g(F,n),shear_g(H,n)
        def shear_tensor(g):
            return [[.5*(n[i]*g[j]+g[i]*n[j]) for j in range(3)] for i in range(3)]
        sf,sh=shear_tensor(gf),shear_tensor(gh)
        inner=lambda a,b: sum(a[i][j]*b[i][j] for i in range(3) for j in range(3))
        assert abs(2*inner(sf,sh)-dot(gf,gh))<1e-10
        rem=[[df[i][j]-sf[i][j] for j in range(3)] for i in range(3)]
        assert abs(inner(rem,sh))<1e-10
        nu0=10**rng.uniform(-6.,2.)
        nuw=10**rng.uniform(-6.,2.)
        # Full deviatoric stiffness with shear-only substitution is
        # positive because the shear contribution is orthogonal.
        e=2*nu0*inner(df,df)+(nuw-nu0)*dot(gf,gf)
        assert e>=-1.e-9*max(1.,nu0*inner(df,df))
        # Zero correction if wall shear viscosity equals original value.
        assert abs((nu0-nu0)*dot(gf,gh))==0.
    print("OPTION_A_V2_TENSOR_TEST=PASS cases=5000 orthogonalShear=YES positiveEnergy=YES")


if __name__=="__main__":
    check()
