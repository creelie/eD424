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
import numpy as np
import sympy as sp

theta, t = sp.symbols('theta t', real=True, positive=True)
u, v = sp.symbols('u v', real=True, positive=True)

data = pickle.load(open(_data('regionA_data.pkl'),'rb'))
total = sp.sympify(data['total'], locals={'theta':theta,'t':t})
expr = total - sp.Rational(31,12)

subs_map = {
    sp.sin(theta): 2*u/(1+u**2),
    sp.cos(theta): (1-u**2)/(1+u**2),
    sp.tan(theta): 2*u/(1-u**2),
    sp.sin(t): 2*v/(1+v**2),
    sp.cos(t): (1-v**2)/(1+v**2),
    sp.tan(t): 2*v/(1-v**2),
}
t0=time.time()
expr_uv = sp.cancel(sp.together(expr.subs(subs_map)))
N, D = sp.fraction(expr_uv)
N = sp.expand(N); D = sp.expand(D)
print(f"N {len(str(N))} chars, D {len(str(D))} chars, {time.time()-t0:.1f}s")

def poly_to_coeff_dict(expr, u, v):
    coeffs = {}
    for term in sp.Add.make_args(sp.expand(expr)):
        pd = term.as_powers_dict()
        i, j = int(pd.get(u,0)), int(pd.get(v,0))
        c = term
        if i: c = c/u**i
        if j: c = c/v**j
        coeffs[(i,j)] = coeffs.get((i,j), sp.Integer(0)) + sp.nsimplify(sp.expand(c))
    return coeffs

Nc = poly_to_coeff_dict(N,u,v)
Dc = poly_to_coeff_dict(D,u,v)
print("N degree:", max(i for i,j in Nc), max(j for i,j in Nc))
print("D degree:", max(i for i,j in Dc), max(j for i,j in Dc))

# numeric sign check
f = sp.lambdify((u,v), N, 'numpy')
g = sp.lambdify((u,v), D, 'numpy')
t1 = 0.169918
def theta_max_Y(tt):
    val = 1.0/(np.sin(tt)+np.cos(tt)); return np.arcsin(min(val,1.0))
for vv,label in [(np.tan(0.02/2),'t=0.02'), (np.tan(0.1/2),'t=0.1'), (np.tan(0.169918/2),'t=t1'),
                  (np.tan(0.25/2),'t=0.25'), (np.tan(0.45/2),'t=0.45')]:
    tt = 2*np.arctan(vv)
    u_lo = 2*vv/(1+vv**2)
    if tt < t1:
        u_hi = 1/np.sqrt(3)
    else:
        u_hi = np.tan(theta_max_Y(tt)/2)
    uu = 0.5*(u_lo+u_hi)
    print(f"{label}: v={vv:.4f} u_lo={u_lo:.4f} u_hi={u_hi:.4f} u={uu:.4f}  N={f(uu,vv):.6f} D={g(uu,vv):.6f} N/D={f(uu,vv)/g(uu,vv):.6f}")

with open(_os.path.join(_DATA_DIR, 'regionA_ND.pkl'), 'wb') as fp:
    pickle.dump({'Nc': {k:str(vv) for k,vv in Nc.items()}, 'Dc': {k:str(vv) for k,vv in Dc.items()}}, fp)
print("saved")
