import os as _os
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA_DIR = _os.path.join(_PKG_ROOT, "data")
def _data(_name):
    """Resolve a bundled or cached data file, wherever the script is run from."""
    _os.makedirs(_DATA_DIR, exist_ok=True)
    return _os.path.join(_DATA_DIR, _name)

import pickle, time
from math import comb
import numpy as np
import sympy as sp

u, v = sp.symbols('u v', real=True, positive=True)
x, y = sp.symbols('x y', real=True, positive=True)

data = pickle.load(open(_data('region6_ND.pkl'),'rb'))
Nc = {k: sp.sympify(vv) for k,vv in data['Nc'].items()}
Dc = {k: sp.sympify(vv) for k,vv in data['Dc'].items()}
NU, NVn, DU, DVn = data['NU'], data['NVn'], data['DU'], data['DVn']

def substitute_box(coeffs, nu, nv, u_lo, u_hi, v_lo, v_hi):
    ux = u_lo + (u_hi-u_lo)*x
    vy = v_lo + (v_hi-v_lo)*y
    upows = [sp.Integer(1)]
    for _ in range(nu): upows.append(sp.expand(upows[-1]*ux))
    vpows = [sp.Integer(1)]
    for _ in range(nv): vpows.append(sp.expand(vpows[-1]*vy))
    expr = sp.Integer(0)
    for (i,j),c in coeffs.items():
        expr += c*upows[i]*vpows[j]
    expr = sp.expand(expr)
    out = {}
    for term in expr.as_ordered_terms():
        pd = term.as_powers_dict()
        i,j = int(pd.get(x,0)), int(pd.get(y,0))
        c = term
        if i: c = c/x**i
        if j: c = c/y**j
        out[(i,j)] = out.get((i,j), sp.Integer(0)) + sp.nsimplify(sp.expand(c))
    return out

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

def check_box(coeffs, nu, nv, u_lo, u_hi, v_lo, v_hi, expect_sign):
    bc = substitute_box(coeffs, nu, nv, u_lo, u_hi, v_lo, v_hi)
    B = bernstein_coeffs(bc, nu, nv)
    flat = [B[i][j] for i in range(nu+1) for j in range(nv+1)]
    bad = [z for z in flat if (expect_sign>0 and z<0) or (expect_sign<0 and z>0)]
    nz = sum(1 for z in flat if z==0)
    return len(bad)==0, nz, len(flat)

u_lo_true = 1/np.sqrt(3)
v3 = np.sqrt(3)-np.sqrt(2)
v_hi_true = np.tan(np.pi/8)
MARGIN = 0.0008
v_lo_g = sp.nsimplify(round(v3+MARGIN, 6))
v_hi_g = sp.nsimplify(round(v_hi_true-MARGIN, 6))
u_lo_g = sp.nsimplify(round(u_lo_true+MARGIN, 6))

def u_max_num(vv): return 2*vv/(1+vv**2)

NBOX = 14
t0 = time.time()
all_ok = True
print(f"domain: v in ({float(v_lo_g):.6f},{float(v_hi_g):.6f}), u in ({float(u_lo_g):.6f}, u_max(v))")
for k in range(NBOX):
    v_lo = v_lo_g + (v_hi_g-v_lo_g)*k/NBOX
    v_hi = v_lo_g + (v_hi_g-v_lo_g)*(k+1)/NBOX
    v_hi_f = float(v_hi)
    u_hi_f = u_max_num(v_hi_f)*0.9985
    u_hi = sp.nsimplify(round(u_hi_f, 6))
    if u_hi <= u_lo_g:
        u_lo_box, u_hi_box = u_hi, u_lo_g
    else:
        u_lo_box, u_hi_box = u_lo_g, u_hi
    okN, nzN, totN = check_box(Nc, NU, NVn, u_lo_box, u_hi_box, v_lo, v_hi, expect_sign=1)
    okD, nzD, totD = check_box(Dc, DU, DVn, u_lo_box, u_hi_box, v_lo, v_hi, expect_sign=1)
    print(f"  box {k:2d}: v=[{float(v_lo):.5f},{float(v_hi):.5f}] u=[{float(u_lo_box):.5f},{float(u_hi_box):.5f}] "
          f"N:{'ok' if okN else 'FAIL'}({nzN}/{totN} zero) D:{'ok' if okD else 'FAIL'}({nzD}/{totD} zero) "
          f"{time.time()-t0:.1f}s")
    all_ok = all_ok and okN and okD

print()
print("ALL BOXES CERTIFIED:", all_ok)
