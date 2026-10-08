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
from scipy.optimize import brentq

theta, t = sp.symbols('theta t', real=True, positive=True)
u, v = sp.symbols('u v', real=True, positive=True)
x, y = sp.symbols('x y', real=True, positive=True)

data = pickle.load(open(_data('region6_data.pkl'),'rb'))
total = sp.sympify(data['total'], locals={'theta':theta,'t':t})
vol_fixed = sp.Rational(16,3)
expr = total - (8 - vol_fixed)  # F_Omega/D_Omega = vol_moving - (8-vol_fixed)
print("threshold (8-vol_fixed):", 8-vol_fixed)

def to_uv_ND(expr):
    subs_map = {
        sp.sin(theta): 2*u/(1+u**2),
        sp.cos(theta): (1-u**2)/(1+u**2),
        sp.tan(theta): 2*u/(1-u**2),
        sp.sin(t): 2*v/(1+v**2),
        sp.cos(t): (1-v**2)/(1+v**2),
        sp.tan(t): 2*v/(1-v**2),
    }
    expr_uv = sp.cancel(sp.together(expr.subs(subs_map)))
    N, D = sp.fraction(expr_uv)
    return sp.expand(N), sp.expand(D)

t0 = time.time()
N, D = to_uv_ND(expr)
print(f"mapped to (u,v): N {len(str(N))} chars, D {len(str(D))} chars, {time.time()-t0:.1f}s")

def poly_to_coeff_dict(expr):
    coeffs = {}
    for term in sp.Add.make_args(sp.expand(expr)):
        pd = term.as_powers_dict()
        i, j = int(pd.get(u,0)), int(pd.get(v,0))
        c = term
        if i: c = c/u**i
        if j: c = c/v**j
        coeffs[(i,j)] = coeffs.get((i,j), sp.Integer(0)) + sp.nsimplify(sp.expand(c))
    return coeffs

Nc = poly_to_coeff_dict(N)
Dc = poly_to_coeff_dict(D)
NU = max(i for i,j in Nc); NVn = max(j for i,j in Nc)
DU = max(i for i,j in Dc); DVn = max(j for i,j in Dc)
print(f"N degree (u={NU},v={NVn})  D degree (u={DU},v={DVn})")

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

# numeric sign check first
f = sp.lambdify((u,v), N/D, 'numpy')
u_lo_true = 1/np.sqrt(3)
v3 = np.sqrt(3)-np.sqrt(2)
v_hi_true = np.tan(np.pi/8)
print("u_lo_true:", u_lo_true, " v3:", v3, " v_hi_true:", v_hi_true)
for vv in [v3+0.001, 0.35, 0.4, v_hi_true-0.001]:
    def u_max_of_v(vv):
        # W-curve: u = 2v/(1+v^2)
        return 2*vv/(1+vv**2)
    uu = 0.5*(u_lo_true + u_max_of_v(vv))
    print(f"v={vv:.4f} u={uu:.4f}: N/D={f(uu,vv):.6f}")

with open(_data('region6_ND.pkl'),'wb') as fp:
    pickle.dump({'Nc': {k:str(vv) for k,vv in Nc.items()}, 'Dc': {k:str(vv) for k,vv in Dc.items()},
                 'NU':NU,'NVn':NVn,'DU':DU,'DVn':DVn}, fp)
print("saved.")
