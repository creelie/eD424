import os as _os
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA_DIR = _os.path.join(_PKG_ROOT, "data")
def _data(_name):
    """Resolve a bundled or cached data file, wherever the script is run from."""
    _os.makedirs(_DATA_DIR, exist_ok=True)
    return _os.path.join(_DATA_DIR, _name)

import re, time, pickle
import numpy as np
import sympy as sp
from scipy.spatial import HalfspaceIntersection, ConvexHull

theta, t = sp.symbols('theta t', real=True, positive=True)

def _build_root_system():
    roots, labels = [], []
    for i in range(4):
        for j in range(i + 1, 4):
            for si in (1, -1):
                for sj in (1, -1):
                    vv = np.zeros(4)
                    vv[i], vv[j] = si, sj
                    roots.append(vv / np.sqrt(2))
                    sgn = lambda s: '+' if s == 1 else '-'
                    labels.append(f"{sgn(si)}e{i+1}{sgn(sj)}e{j+1}")
    roots = np.array(roots)
    def find(vec):
        v_ = np.array(vec, dtype=float); v_ /= np.linalg.norm(v_)
        idx = np.argmax(roots @ v_)
        assert (roots @ v_)[idx] > 1 - 1e-9
        return idx
    idx0 = find([1, 1, 0, 0])
    return roots, labels, idx0, roots[idx0], roots[find([1, -1, 0, 0])], roots[find([0, 0, 1, 1])]

roots, labels, idx0, u0, v1, w1 = _build_root_system()

def u1_vec(th, tt):
    return np.cos(th)*u0 + np.sin(th)*(np.cos(tt)*v1 + np.sin(tt)*w1)

def polytope_full(th, tt):
    u1 = u1_vec(th, tt)
    dirs = roots.copy(); dirs[idx0] = u1
    hs = np.hstack([dirs, (-np.ones(len(dirs))).reshape(-1,1)])
    hi = HalfspaceIntersection(hs, np.zeros(4))
    out = []
    for vpt in hi.intersections:
        vals = dirs @ vpt
        touching = frozenset(labels[k] if k!=idx0 else 'u1' for k,val in enumerate(vals) if abs(val-1)<1e-6)
        out.append((vpt, touching))
    return out

def theta_A(tt): return 2*np.arctan(np.cos(tt))

t0 = 0.15
theta0 = 0.5*(theta_A(t0) + np.pi/2)
verts_info = polytope_full(theta0, t0)
verts = np.array([v_ for v_,_ in verts_info])
patterns = [p for _,p in verts_info]
on_cap_idx = [i for i,p in enumerate(patterns) if 'u1' in p]
off_cap_idx = [i for i,p in enumerate(patterns) if 'u1' not in p]

hull = ConvexHull(verts, qhull_options='QJ')
on_cap_set = set(on_cap_idx)
moving_simplices = [s for s in hull.simplices if any(vv in on_cap_set for vv in s)]
fixed_simplices = [s for s in hull.simplices if all(vv not in on_cap_set for vv in s)]

import itertools

def _extract_in_order(sol, zlist, simplify_fn=None):
    """Read a solve() result back in the supplied unknown order.

    sympy returns a dict keyed by symbol; its iteration order is not
    guaranteed to match the order the unknowns were passed in, so the
    components are looked up by key rather than taken from .values().
    Returns None when the system leaves a coordinate undetermined.
    """
    if simplify_fn is None:
        simplify_fn = sp.simplify
    if not sol:
        return None
    d = sol[0]
    out = []
    for zi in zlist:
        if zi not in d:
            return None
        out.append(simplify_fn(d[zi]))
    return sp.Matrix(out)

sqrt2 = sp.sqrt(2)
candidates_num, candidates_sym = [], []
for i in range(4):
    for s in (1,-1):
        vv = np.zeros(4); vv[i]=s*np.sqrt(2)
        candidates_num.append(vv)
        vs_ = sp.zeros(4,1); vs_[i]=s*sqrt2
        candidates_sym.append(vs_)
for eps in itertools.product([1,-1], repeat=4):
    candidates_num.append(np.array(eps)/np.sqrt(2))
    candidates_sym.append(sp.Matrix(eps)/sqrt2)
candidates_num.append(np.array([np.sqrt(2), np.sqrt(2), 0, 0]))
candidates_sym.append(sp.Matrix([sqrt2, sqrt2, 0, 0]))
candidates_num = np.array(candidates_num)

fixed_sym = {}
for i in off_cap_idx:
    dists = np.linalg.norm(candidates_num - verts[i], axis=1)
    j = np.argmin(dists)
    assert dists[j] < 1e-6
    fixed_sym[i] = candidates_sym[j]

vol_fixed_exact = sp.Integer(0)
for s in fixed_simplices:
    M = sp.Matrix.hstack(*[fixed_sym[i] for i in s]).T
    Mnum = verts[list(s)]
    sign = 1 if np.linalg.det(Mnum) > 0 else -1
    vol_fixed_exact += sign*M.det()/24
vol_fixed_exact = sp.nsimplify(sp.simplify(vol_fixed_exact))
print("vol_fixed_exact:", vol_fixed_exact, "(expect 19/3)")
assert vol_fixed_exact == sp.Rational(19,3)

u0s = sp.Matrix([1,1,0,0])/sqrt2
v1s = sp.Matrix([1,-1,0,0])/sqrt2
w1s = sp.Matrix([0,0,1,1])/sqrt2
u1s = sp.cos(theta)*u0s + sp.sin(theta)*(sp.cos(t)*v1s + sp.sin(t)*w1s)

def root_sym(spec):
    r = sp.zeros(4,1)
    for m in re.finditer(r'([+-])e(\d)', spec):
        r[int(m.group(2))-1] = 1 if m.group(1)=='+' else -1
    return r/sqrt2

def solve_vertex_moving(facet_labels):
    z = sp.Matrix(sp.symbols('z1 z2 z3 z4', real=True))
    eqs = [sp.Eq((u1s.T*z)[0,0], 1)]
    for lab in facet_labels[:3]:
        eqs.append(sp.Eq((root_sym(lab).T*z)[0,0], 1))
    sol = sp.solve(eqs, list(z), dict=True)
    return _extract_in_order(sol, list(z))

t_start = time.time()
moving_sym, moving_cache = {}, {}
for i in on_cap_idx:
    pat = patterns[i]
    if pat in moving_cache:
        moving_sym[i] = moving_cache[pat]; continue
    zv = solve_vertex_moving([p for p in pat if p != 'u1'])
    assert zv is not None
    numeric = np.array([float(z.subs({theta: theta0, t: t0})) for z in zv])
    assert np.allclose(numeric, verts[i], atol=1e-5)
    moving_sym[i] = zv
    moving_cache[pat] = zv
print(f"moving vertices solved: {len(moving_cache)} distinct patterns, {time.time()-t_start:.1f}s")

sym_vertex = dict(fixed_sym); sym_vertex.update(moving_sym)

terms = []
for s in moving_simplices:
    M = sp.Matrix.hstack(*[sym_vertex[i] for i in s]).T
    Mnum = verts[list(s)]
    sign = 1 if np.linalg.det(Mnum) > 0 else -1
    terms.append(sp.cancel(sign*M.det()/24))
print(f"summing {len(terms)} moving simplex terms, {time.time()-t_start:.1f}s")
total = sp.cancel(sp.together(sum(terms)))
num_check = float(total.subs({theta: theta0, t: t0}))
print(f"vol_moving_top closed form: {len(str(total))} chars, {time.time()-t_start:.1f}s")
print(f"numeric check: {num_check}")

# independent numeric check via direct polytope volume computation
def vol_direct(th, tt):
    verts_i = polytope_full(th, tt)
    vv = np.array([v_ for v_,_ in verts_i])
    hull = ConvexHull(vv)
    return hull.volume

vd = vol_direct(theta0, t0)
print(f"direct hull volume: {vd}  (expect 19/3 + vol_moving = {19/3+num_check})")

with open(_data('region_top_data.pkl'),'wb') as f:
    pickle.dump({'total': str(total), 'theta0':theta0, 't0':t0, 'num_check':num_check, 'vol_direct': vd}, f)
print("saved.")
