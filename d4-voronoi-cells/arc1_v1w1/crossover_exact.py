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

import numpy as np
import sympy as sp
import itertools, time
from scipy.spatial import HalfspaceIntersection, ConvexHull

sqrt2 = sp.sqrt(2)
t = sp.symbols('t', real=True, positive=True)
cos_th = (1 - t**2) / (1 + t**2)
sin_th = 2*t / (1 + t**2)

roots_num = []
roots_exact = []
for i in range(4):
    for j in range(i+1, 4):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0,0,0,0]; v[i]=si; v[j]=sj
                roots_num.append(np.array(v, dtype=float)/np.sqrt(2))
                ev = sp.Matrix([sp.Rational(si) if k==i else (sp.Rational(sj) if k==j else 0) for k in range(4)])/sqrt2
                roots_exact.append(ev)
roots_num = np.array(roots_num)

def find(vec):
    v = np.array(vec, dtype=float)/np.linalg.norm(vec)
    d = roots_num @ v
    return int(np.argmax(d))

ia = find([1,1,0,0]); ib = find([1,0,1,0])
alpha_e, beta_e = roots_exact[ia], roots_exact[ib]
alpha_n, beta_n = roots_num[ia], roots_num[ib]
s_e = sp.Rational(1,2)
s_n = alpha_n @ beta_n

beta_perp_raw = beta_e - s_e*alpha_e
beta_perp_e = sp.simplify(beta_perp_raw / sp.sqrt((beta_perp_raw.T*beta_perp_raw)[0]))
alpha_perp_raw = alpha_e - s_e*beta_e
alpha_perp_e = sp.simplify(alpha_perp_raw / sp.sqrt((alpha_perp_raw.T*alpha_perp_raw)[0]))

beta_perp_n = beta_n - s_n*alpha_n; beta_perp_n /= np.linalg.norm(beta_perp_n)
alpha_perp_n = alpha_n - s_n*beta_n; alpha_perp_n /= np.linalg.norm(alpha_perp_n)

fixed_idx = [k for k in range(24) if k not in (ia, ib)]
fixed_exact = [roots_exact[k] for k in fixed_idx]
fixed_num = roots_num[fixed_idx]

def build_path(u1_e, u2_e, u1_n_fn, u2_n_fn, theta_ref, label):
    t0 = time.time()
    active_exact = fixed_exact + [u1_e, u2_e]

    def dirs_num(theta):
        return np.vstack([fixed_num, u1_n_fn(theta), u2_n_fn(theta)])

    dirs0 = dirs_num(theta_ref)
    A = dirs0; b = -np.ones(len(dirs0))
    hs = np.hstack([A, b.reshape(-1,1)])
    hi = HalfspaceIntersection(hs, np.zeros(4))
    verts0 = hi.intersections
    hull0 = ConvexHull(verts0, qhull_options='QJ')
    vals = dirs0 @ verts0.T
    tight = vals > 1 - 1e-6
    quad_for_vertex = []
    for k in range(verts0.shape[0]):
        supp = sorted(np.where(tight[:,k])[0].tolist())
        found = None
        for quad in itertools.combinations(supp, 4):
            pts = dirs0[list(quad)]
            G = pts @ pts.T
            if abs(np.linalg.det(G)) > 1e-6:
                found = quad
                break
        quad_for_vertex.append(found)
    assert all(q is not None for q in quad_for_vertex)

    exact_verts = []
    for quad in quad_for_vertex:
        pts = [active_exact[idx] for idx in quad]
        G = sp.Matrix(4,4, lambda a,b: sp.together((pts[a].T*pts[b])[0,0]))
        alpha_c = G.solve(sp.ones(4,1))
        zI = sp.zeros(4,1)
        for a in range(4):
            zI += alpha_c[a]*pts[a]
        zI = sp.Matrix([sp.cancel(sp.radsimp(zI[i])) for i in range(4)])
        exact_verts.append(zI)

    vol_terms = []
    for simp in hull0.simplices:
        M = sp.Matrix.hstack(*[exact_verts[idx] for idx in simp])
        vol_terms.append(sp.cancel(M.det()))

    t0val = np.tan(theta_ref/2)
    total = sp.Integer(0)
    for d in vol_terms:
        dnum = float(d.subs(t, t0val))
        sign = 1 if dnum > 0 else -1
        total += sign*d

    print(f"[{label}] {len(exact_verts)} verts, {len(hull0.simplices)} simplices, "
          f"built in {time.time()-t0:.1f}s")
    return total

# swap path (reference in the SECOND piece, theta_ref>pi/6, valid near the crossover)
u1_swap_e = sp.simplify(cos_th*alpha_e + sin_th*beta_perp_e)
u2_swap_e = sp.simplify(cos_th*beta_e + sin_th*alpha_perp_e)
u1_swap_n = lambda th: np.cos(th)*alpha_n + np.sin(th)*beta_perp_n
u2_swap_n = lambda th: np.cos(th)*beta_n + np.sin(th)*alpha_perp_n
total_swap = build_path(u1_swap_e, u2_swap_e, u1_swap_n, u2_swap_n, 0.65, "swap")

# mixed path (single piece throughout (0,theta*), per the 100-sample scan)
u1_mixed_e = sp.simplify(cos_th*alpha_e - sin_th*beta_perp_e)
u2_mixed_e = sp.simplify(cos_th*beta_e + sin_th*alpha_perp_e)
u1_mixed_n = lambda th: np.cos(th)*alpha_n - np.sin(th)*beta_perp_n
u2_mixed_n = lambda th: np.cos(th)*beta_n + np.sin(th)*alpha_perp_n
total_mixed = build_path(u1_mixed_e, u2_mixed_e, u1_mixed_n, u2_mixed_n, 0.65, "mixed")

import pickle
with open(_os.path.join(_DATA_DIR, 'crossover_totals.pkl'), 'wb') as f:
    pickle.dump({'total_swap': total_swap, 'total_mixed': total_mixed}, f)
print("saved.")
