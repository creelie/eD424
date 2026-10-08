import os as _os
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA_DIR = _os.path.join(_PKG_ROOT, "data")
def _data(_name):
    """Resolve a bundled or cached data file, wherever the script is run from."""
    _os.makedirs(_DATA_DIR, exist_ok=True)
    return _os.path.join(_DATA_DIR, _name)

import pickle, time
import sympy as sp
from math import comb

w, v2 = sp.symbols('w v2', real=True, positive=True)
sqrt2 = sp.sqrt(2)

data = pickle.load(open(_data('region_top_ND.pkl'),'rb'))
N = sp.sympify(data['N'], locals={'w':w,'v2':v2})
D = sp.sympify(data['D'], locals={'w':w,'v2':v2})

def poly_to_coeff_dict(expr):
    coeffs = {}
    for term in sp.Add.make_args(sp.expand(expr)):
        pd = term.as_powers_dict()
        i, j = int(pd.get(w,0)), int(pd.get(v2,0))
        c = term
        if i: c = c/w**i
        if j: c = c/v2**j
        c = sp.expand(sp.radsimp(c))
        coeffs[(i,j)] = coeffs.get((i,j), sp.Integer(0)) + c
    return coeffs

t0 = time.time()
Nc = poly_to_coeff_dict(N)
Dc = poly_to_coeff_dict(D)
nw = max(i for i,j in Nc); nvN = max(j for i,j in Nc)
dw = max(i for i,j in Dc); dvD = max(j for i,j in Dc)
print(f"N: {len(Nc)} terms, degree (w={nw},v2={nvN})")
print(f"D: {len(Dc)} terms, degree (w={dw},v2={dvD})")

def bernstein_coeffs(coeffs, nu, nv):
    A = [[sp.Integer(0)]*(nv+1) for _ in range(nu+1)]
    for (i,j),c in coeffs.items():
        A[i][j] += c
    def binom(n,k): return sp.Integer(comb(n,k))
    def transform_1d(vec,n):
        b=[sp.Integer(0)]*(n+1)
        for j in range(n+1):
            s=sp.Integer(0)
            for k in range(j+1):
                s += binom(j,k)*vec[k]/binom(n,k)
            b[j]=s
        return b
    B1=[transform_1d(A[i], nv) for i in range(nu+1)]
    B2=[[sp.Integer(0)]*(nv+1) for _ in range(nu+1)]
    for jv in range(nv+1):
        col=[B1[i][jv] for i in range(nu+1)]
        newcol=transform_1d(col,nu)
        for i in range(nu+1):
            B2[i][jv]=newcol[i]
    return B2

def exact_sign(expr):
    # expr = a + b*sqrt2 exactly, a,b rational; radsimp clears sqrt2 out of
    # any denominator first (multiplying by the conjugate), then plain
    # coeff-extraction on the fully-expanded numerator gives a,b exactly.
    r = sp.expand(sp.radsimp(expr))
    a = sp.Rational(r.coeff(sqrt2, 0))
    b = sp.Rational(r.coeff(sqrt2, 1))
    if b == 0:
        return 1 if a > 0 else (-1 if a < 0 else 0)
    if a == 0:
        return 1 if b > 0 else -1
    if (a > 0 and b > 0):
        return 1
    if (a < 0 and b < 0):
        return -1
    disc = a**2 - 2*b**2
    if disc == 0:
        return 0
    s = 1 if disc > 0 else -1
    return (1 if a > 0 else -1) * s

t1 = time.time()
BN = bernstein_coeffs(Nc, nw, nvN)
BD = bernstein_coeffs(Dc, dw, dvD)
print(f"Bernstein transforms done, {time.time()-t1:.1f}s")

flatN = [BN[i][j] for i in range(nw+1) for j in range(nvN+1)]
flatD = [BD[i][j] for i in range(dw+1) for j in range(dvD+1)]

signsN = [exact_sign(c) for c in flatN]
signsD = [exact_sign(c) for c in flatD]
print(f"N: {sum(1 for s in signsN if s>0)} positive, {sum(1 for s in signsN if s==0)} zero, {sum(1 for s in signsN if s<0)} negative  (of {len(signsN)})")
print(f"D: {sum(1 for s in signsD if s>0)} positive, {sum(1 for s in signsD if s==0)} zero, {sum(1 for s in signsD if s<0)} negative  (of {len(signsD)})")
print(f"total time: {time.time()-t0:.1f}s")

zeroN = [(i,j) for idx,(i,j) in enumerate([(i,j) for i in range(nw+1) for j in range(nvN+1)]) if signsN[idx]==0]
zeroD = [(i,j) for idx,(i,j) in enumerate([(i,j) for i in range(dw+1) for j in range(dvD+1)]) if signsD[idx]==0]
print("zero coords N:", zeroN[:20], "..." if len(zeroN)>20 else "")
print("zero coords D:", zeroD[:20], "..." if len(zeroD)>20 else "")
negN = [(i,j) for idx,(i,j) in enumerate([(i,j) for i in range(nw+1) for j in range(nvN+1)]) if signsN[idx]<0]
negD = [(i,j) for idx,(i,j) in enumerate([(i,j) for i in range(dw+1) for j in range(dvD+1)]) if signsD[idx]<0]
print("neg coords N:", negN[:20])
print("neg coords D:", negD[:20])
