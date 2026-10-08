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

import sympy as sp
import numpy as np
import pickle, time

sqrt2 = sp.sqrt(2)
t = sp.symbols('t', real=True, positive=True)  # t = tan(theta/2)
cos_th = (1-t**2)/(1+t**2)
sin_th = 2*t/(1+t**2)

roots_num = []
roots_exact = []
for i in range(4):
    for j in range(i+1,4):
        for si in (1,-1):
            for sj in (1,-1):
                v = [0,0,0,0]; v[i]=si; v[j]=sj
                roots_num.append(np.array(v,dtype=float)/np.sqrt(2))
                ev = sp.Matrix([sp.Rational(si) if k==i else (sp.Rational(sj) if k==j else 0) for k in range(4)])/sqrt2
                roots_exact.append(ev)
roots_num = np.array(roots_num)

def find(vec):
    v = np.array(vec,dtype=float)/np.linalg.norm(vec)
    d = roots_num @ v
    return int(np.argmax(d))

ia = find([1,1,0,0]); ib = find([1,0,1,0])
alpha_e, beta_e = roots_exact[ia], roots_exact[ib]
s = sp.Rational(1,2)
beta_perp_raw = beta_e - s*alpha_e
beta_perp = sp.simplify(beta_perp_raw / sp.sqrt((beta_perp_raw.T*beta_perp_raw)[0]))
alpha_perp_raw = alpha_e - s*beta_e
alpha_perp = sp.simplify(alpha_perp_raw / sp.sqrt((alpha_perp_raw.T*alpha_perp_raw)[0]))

u1 = sp.simplify(cos_th*alpha_e + sin_th*beta_perp)
u2 = sp.simplify(cos_th*beta_e + sin_th*alpha_perp)
print("u1(t) =", u1.T)
print("u2(t) =", u2.T)

fixed_idx = [k for k in range(24) if k not in (ia,ib)]
active_exact = [roots_exact[k] for k in fixed_idx] + [u1, u2]

with open(_data('swap_quads_theta03.pkl'),'rb') as f:
    data = pickle.load(f)
quad_for_vertex = data['quad_for_vertex']

t0 = time.time()
exact_verts = []
for k, quad in enumerate(quad_for_vertex):
    pts = [active_exact[idx] for idx in quad]
    G = sp.Matrix(4,4, lambda a,b: sp.together((pts[a].T*pts[b])[0,0]))
    alpha_c = G.solve(sp.ones(4,1))
    zI = sp.zeros(4,1)
    for a in range(4):
        zI += alpha_c[a]*pts[a]
    zI = sp.Matrix([sp.cancel(sp.radsimp(zI[i])) for i in range(4)])
    exact_verts.append(zI)

print(f"ALL 33 exact vertices (rational-t form) computed in {time.time()-t0:.1f}s")
with open(_os.path.join(_DATA_DIR, 'swap_exact_verts_rational.pkl'), 'wb') as f:
    pickle.dump(exact_verts, f)

# numeric cross-check at theta=0.3 -> t = tan(0.15)
t0val = np.tan(0.15)
verts_num = np.load(_data('swap_verts_theta03.npy'))
maxerr = 0
for k in range(33):
    zval = np.array([complex(exact_verts[k][i].subs(t, t0val)).real for i in range(4)])
    err = np.linalg.norm(zval - verts_num[k])
    maxerr = max(maxerr, err)
print("max numeric cross-check error:", maxerr)
