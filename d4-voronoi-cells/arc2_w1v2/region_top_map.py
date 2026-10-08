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

theta, t = sp.symbols('theta t', real=True, positive=True)
u, v = sp.symbols('u v', real=True, positive=True)
w, v2 = sp.symbols('w v2', real=True, positive=True)

data = pickle.load(open(_data('region_top_data.pkl'),'rb'))
total = sp.sympify(data['total'], locals={'theta':theta,'t':t})
expr = total - sp.Rational(5,3)

t0 = time.time()
# Weierstrass: theta -> u=tan(theta/2), t -> v=tan(t/2)
subs_map = {
    sp.sin(theta): 2*u/(1+u**2),
    sp.cos(theta): (1-u**2)/(1+u**2),
    sp.tan(theta): 2*u/(1-u**2),
    sp.sin(t): 2*v/(1+v**2),
    sp.cos(t): (1-v**2)/(1+v**2),
    sp.tan(t): 2*v/(1-v**2),
}
expr_uv = sp.cancel(sp.together(expr.subs(subs_map)))
print(f"mapped to (u,v): {len(str(expr_uv))} chars, {time.time()-t0:.1f}s")

# region: u in (u_lo(v), 1), v in (0, tan(pi/8)), u_lo(v)=(1-v^2)/(1+v^2)
# exact rational remap: u = [(1-v^2) + 2*w*v^2] / (1+v^2), w in (0,1)
u_of_w_v = ((1-v**2) + 2*w*v**2)/(1+v**2)
expr_wv = sp.cancel(sp.together(expr_uv.subs({u: u_of_w_v})))
print(f"substituted u(w,v): {len(str(expr_wv))} chars, {time.time()-t0:.1f}s")

# rescale v = (sqrt2-1)*v2 to map v in (0, tan(pi/8)) to v2 in (0,1)
sqrt2 = sp.sqrt(2)
expr_final = sp.cancel(sp.together(expr_wv.subs({v: (sqrt2-1)*v2})))
print(f"substituted v=(sqrt2-1)v2: {len(str(expr_final))} chars, {time.time()-t0:.1f}s")

N, D = sp.fraction(expr_final)
N = sp.expand(N); D = sp.expand(D)
print(f"N: {len(str(N))} chars, D: {len(str(D))} chars, {time.time()-t0:.1f}s")

with open(_data('region_top_ND.pkl'),'wb') as f:
    pickle.dump({'N': str(N), 'D': str(D)}, f)
print("saved ND.")
