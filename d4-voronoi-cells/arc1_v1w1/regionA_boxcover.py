import os as _os, sys as _sys
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA_DIR = _os.path.join(_PKG_ROOT, "data")
for _p in (_PKG_ROOT, _os.path.join(_PKG_ROOT, "core")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
def _data(_name):
    """Resolve a bundled data file, wherever the script is run from."""
    _c = _os.path.join(_DATA_DIR, _name)
    if _os.path.exists(_c):
        return _c
    if _os.path.exists(_name):
        return _name
    raise SystemExit(
        "required data file %r not found; expected it in %s. "
        "Run the matching *_derive.py step first, or fetch the bundled "
        "copy from the package's data/ directory." % (_name, _DATA_DIR))

import pickle, time
from math import comb
import numpy as np
import sympy as sp

u, v = sp.symbols('u v', real=True, positive=True)
x, y = sp.symbols('x y', real=True, positive=True)

data = pickle.load(open(_data('regionA_ND.pkl'),'rb'))
Nc = {k: sp.sympify(vv) for k,vv in data['Nc'].items()}
Dc = {k: sp.sympify(vv) for k,vv in data['Dc'].items()}
NU, NVn = max(i for i,j in Nc), max(j for i,j in Nc)
DU, DVn = max(i for i,j in Dc), max(j for i,j in Dc)

def substitute_box(coeffs, nu, nv, u_lo, u_hi, v_lo, v_hi):
    ux = u_lo+(u_hi-u_lo)*x
    vy = v_lo+(v_hi-v_lo)*y
    upows=[sp.Integer(1)]
    for _ in range(nu): upows.append(sp.expand(upows[-1]*ux))
    vpows=[sp.Integer(1)]
    for _ in range(nv): vpows.append(sp.expand(vpows[-1]*vy))
    expr = sp.Integer(0)
    for (i,j),c in coeffs.items():
        expr += c*upows[i]*vpows[j]
    expr = sp.expand(expr)
    out={}
    for term in expr.as_ordered_terms():
        pd = term.as_powers_dict()
        i,j = int(pd.get(x,0)), int(pd.get(y,0))
        c = term
        if i: c=c/x**i
        if j: c=c/y**j
        out[(i,j)] = out.get((i,j), sp.Integer(0)) + sp.nsimplify(sp.expand(c))
    return out

def bernstein_coeffs(coeffs, nu, nv):
    A=[[sp.Integer(0)]*(nv+1) for _ in range(nu+1)]
    for (i,j),c in coeffs.items(): A[i][j]+=c
    def binom(n,k): return sp.Integer(comb(n,k))
    def tf(vec,n):
        b=[sp.Integer(0)]*(n+1)
        for j in range(n+1):
            s=sp.Integer(0)
            for k in range(j+1): s+=binom(j,k)*vec[k]/binom(n,k)
            b[j]=s
        return b
    B1=[tf(A[i],nv) for i in range(nu+1)]
    B2=[[sp.Integer(0)]*(nv+1) for _ in range(nu+1)]
    for jv in range(nv+1):
        col=[B1[i][jv] for i in range(nu+1)]
        nc=tf(col,nu)
        for i in range(nu+1): B2[i][jv]=nc[i]
    return B2

def check_box(coeffs, nu, nv, u_lo,u_hi,v_lo,v_hi, expect_sign):
    bc = substitute_box(coeffs, nu, nv, u_lo,u_hi,v_lo,v_hi)
    B = bernstein_coeffs(bc, nu, nv)
    flat=[B[i][j] for i in range(nu+1) for j in range(nv+1)]
    bad=[z for z in flat if (expect_sign>0 and z<0) or (expect_sign<0 and z>0)]
    nz=sum(1 for z in flat if z==0)
    return len(bad)==0, nz, len(flat)

t1 = 0.169918
t2 = float(np.arctan(0.5))
v1 = np.tan(t1/2)
v2 = np.tan(t2/2)
MARGIN = 0.0008
v_lo_g = sp.nsimplify(round(0+MARGIN,6))
v_hi_g = sp.nsimplify(round(v2-MARGIN,6))

def u_lo_num(vv): return 2*vv/(1+vv**2)  # W-curve floor (safe: use as-is, floor grows INTO region so need +margin)
def theta_max_Y(tt):
    val = 1.0/(np.sin(tt)+np.cos(tt)); return np.arcsin(min(val,1.0))
def u_hi_num(vv):
    tt = 2*np.arctan(vv)
    if tt < t1:
        return 1/np.sqrt(3)
    else:
        return np.tan(theta_max_Y(tt)/2)

NBOX = 16
t0=time.time()
all_ok=True
print(f"domain: v in ({float(v_lo_g):.6f},{float(v_hi_g):.6f})")
for k in range(NBOX):
    v_lo = v_lo_g + (v_hi_g-v_lo_g)*k/NBOX
    v_hi = v_lo_g + (v_hi_g-v_lo_g)*(k+1)/NBOX
    v_lo_f, v_hi_f = float(v_lo), float(v_hi)
    u_lo_f = u_lo_num(v_lo_f)*1.0015  # floor rises with v; use v_lo's floor value nudged up slightly for safety margin
    u_hi_f = min(u_hi_num(v_lo_f), u_hi_num(v_hi_f))*0.9985  # ceiling: conservative min over box, shrunk
    u_lo = sp.nsimplify(round(u_lo_f,6))
    u_hi = sp.nsimplify(round(u_hi_f,6))
    if u_hi <= u_lo:
        u_lo, u_hi = u_hi, u_lo  # tiny sliver swap safeguard
    okN,nzN,totN = check_box(Nc,NU,NVn,u_lo,u_hi,v_lo,v_hi, expect_sign=1)
    okD,nzD,totD = check_box(Dc,DU,DVn,u_lo,u_hi,v_lo,v_hi, expect_sign=1)
    print(f"  box {k:2d}: v=[{v_lo_f:.5f},{v_hi_f:.5f}] u=[{float(u_lo):.5f},{float(u_hi):.5f}] "
          f"N:{'ok' if okN else 'FAIL'}({nzN}/{totN}) D:{'ok' if okD else 'FAIL'}({nzD}/{totD}) {time.time()-t0:.1f}s")
    all_ok = all_ok and okN and okD

print()
print("ALL BOXES CERTIFIED:", all_ok)
