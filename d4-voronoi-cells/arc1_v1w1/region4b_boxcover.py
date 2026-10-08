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

import pickle, sympy as sp, numpy as np, time
from math import comb

u, v = sp.symbols('u v', real=True, positive=True)
data = pickle.load(open(_data('region_uv_ND.pkl'),'rb'))
N = sp.expand(sp.sympify(data['N'], locals={'u':u,'v':v}))
D = sp.expand(sp.sympify(data['D'], locals={'u':u,'v':v}))

def poly_to_coeff_dict(expr):
    coeffs = {}
    for term in expr.as_ordered_terms():
        pd = term.as_powers_dict()
        i = int(pd.get(u,0)); j = int(pd.get(v,0))
        c = term
        if i: c = c/u**i
        if j: c = c/v**j
        coeffs[(i,j)] = coeffs.get((i,j), sp.Integer(0)) + sp.nsimplify(sp.expand(c))
    return coeffs

Nc = poly_to_coeff_dict(N)
Dc = poly_to_coeff_dict(D)
NU, NV = 10, 10

x, y = sp.symbols('x y')

def substitute_box(coeffs, u_lo, u_hi, v_lo, v_hi):
    expr = sp.Integer(0)
    ux = u_lo+(u_hi-u_lo)*x
    vy = v_lo+(v_hi-v_lo)*y
    # precompute powers
    upows = [sp.Integer(1)]
    for _ in range(NU): upows.append(sp.expand(upows[-1]*ux))
    vpows = [sp.Integer(1)]
    for _ in range(NV): vpows.append(sp.expand(vpows[-1]*vy))
    for (i,j), c in coeffs.items():
        expr += c*upows[i]*vpows[j]
    expr = sp.expand(expr)
    out = {}
    for term in expr.as_ordered_terms():
        pd = term.as_powers_dict()
        i = int(pd.get(x,0)); j = int(pd.get(y,0))
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

def check_box(coeffs, u_lo,u_hi,v_lo,v_hi, expect_sign):
    bc = substitute_box(coeffs, u_lo,u_hi,v_lo,v_hi)
    B = bernstein_coeffs(bc, NU, NV)
    flat = [B[i][j] for i in range(NU+1) for j in range(NV+1)]
    bad = [x for x in flat if (expect_sign>0 and x<0) or (expect_sign<0 and x>0)]
    nz = sum(1 for x in flat if x==0)
    return len(bad)==0, nz, len(flat)

# ---- numeric boundary function u_max(v) ----
def theta_max_of_t(t):
    val = 1.0/(np.sin(t)+np.cos(t)); return np.arcsin(val)
def t_of_v(vv): return 2*np.arctan(vv)
def u_of_theta(th): return np.tan(th/2)
def u_max_num(vv): return u_of_theta(theta_max_of_t(t_of_v(vv)))

vstar = float(sp.sqrt(5)) - 2
vmax = float(sp.tan(sp.pi/8))
print("vstar=",vstar," vmax=",vmax)

MARGIN_V = 0.0007
v_lo_global = sp.nsimplify(round(vstar+MARGIN_V, 6))
v_hi_global = sp.nsimplify(round(vmax-MARGIN_V, 6))
print("v_lo_global,v_hi_global:", float(v_lo_global), float(v_hi_global))

NBOX = 16
results = []
t_start = time.time()
for k in range(NBOX):
    v_lo = v_lo_global + (v_hi_global-v_lo_global)*k/NBOX
    v_hi = v_lo_global + (v_hi_global-v_lo_global)*(k+1)/NBOX
    v_hi_f = float(v_hi)
    umax_f = u_max_num(v_hi_f)
    u_hi_f = umax_f*0.995
    u_hi = sp.nsimplify(round(u_hi_f, 6))
    u_lo = sp.Integer(0)
    okN, nzN, totN = check_box(Nc, u_lo,u_hi,v_lo,v_hi, expect_sign=-1)
    okD, nzD, totD = check_box(Dc, u_lo,u_hi,v_lo,v_hi, expect_sign=-1)
    results.append((k, float(v_lo), float(v_hi), float(u_hi), okN, okD, nzN, nzD))
    print(f"box {k}: v=[{float(v_lo):.5f},{float(v_hi):.5f}] u=[0,{float(u_hi):.5f}] "
          f"N_ok={okN}(nz={nzN}/{totN}) D_ok={okD}(nz={nzD}/{totD})  t={time.time()-t_start:.1f}s")

all_ok = all(r[4] and r[5] for r in results)
print()
print("ALL BOXES CERTIFIED:", all_ok)

with open(_os.path.join(_DATA_DIR, 'region4b_boxcover_results.pkl'), 'wb') as f:
    pickle.dump({'results': results, 'v_lo_global': str(v_lo_global), 'v_hi_global': str(v_hi_global),
                 'vstar': vstar, 'vmax': vmax, 'MARGIN_V': MARGIN_V, 'NBOX': NBOX}, f)
