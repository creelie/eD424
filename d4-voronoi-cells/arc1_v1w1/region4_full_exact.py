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

import numpy as np, sympy as sp, re, itertools
from scipy.spatial import HalfspaceIntersection, ConvexHull

def _extract_in_order(sol, zlist, simplify_fn):
    """Read a solve() result back in the supplied unknown order.

    sympy returns a dict keyed by symbol; its iteration order is not
    guaranteed to match the order the unknowns were passed in, so the
    components are looked up by key rather than taken from .values().
    """
    if not sol:
        return None
    d = sol[0]
    out = []
    for zi in zlist:
        if zi not in d:
            return None
        out.append(simplify_fn(d[zi]))
    return sp.Matrix(out)


# ---------- numeric root system ----------
roots, labels = [], []
for i in range(4):
    for j in range(i+1,4):
        for si in (1,-1):
            for sj in (1,-1):
                v = np.zeros(4)
                v[i]=si; v[j]=sj
                roots.append(v/np.sqrt(2))
                sgn=lambda s: '+' if s==1 else '-'
                labels.append(f"{sgn(si)}e{i+1}{sgn(sj)}e{j+1}")
roots=np.array(roots)

def find(vec):
    v=np.array(vec,dtype=float); v/=np.linalg.norm(v)
    idx=np.argmax(roots@v)
    assert (roots@v)[idx]>1-1e-9
    return idx

idx0=find([1,1,0,0]); u0=roots[idx0]
v1=roots[find([1,-1,0,0])]; w1=roots[find([0,0,1,1])]

def u1_vec(theta,t):
    return np.cos(theta)*u0+np.sin(theta)*(np.cos(t)*v1+np.sin(t)*w1)

def polytope_full(theta,t):
    u1=u1_vec(theta,t)
    dirs=roots.copy(); dirs[idx0]=u1
    hs=np.hstack([dirs,(-np.ones(len(dirs))).reshape(-1,1)])
    hi = HalfspaceIntersection(hs, np.zeros(4))
    out = []
    for v in hi.intersections:
        vals = dirs @ v
        touching = frozenset(labels[k] if k!=idx0 else 'u1'
                              for k,val in enumerate(vals) if abs(val-1)<1e-6)
        out.append((v, touching))
    return out

theta0, t0 = 0.05, 0.2
verts_info = polytope_full(theta0, t0)
verts = np.array([v for v,_ in verts_info])
patterns = [p for _,p in verts_info]
n = len(verts)
on_cap_idx = [i for i,p in enumerate(patterns) if 'u1' in p]
off_cap_idx = [i for i,p in enumerate(patterns) if 'u1' not in p]

hull = ConvexHull(verts, qhull_options='QJ')
on_cap_set = set(on_cap_idx)
moving_simplices = [s for s in hull.simplices if any(v in on_cap_set for v in s)]
fixed_simplices  = [s for s in hull.simplices if all(v not in on_cap_set for v in s)]

# ---------- match off-cap (fixed) vertices to known axis/sign families ----------
candidates_num = []
candidates_sym = []
sqrt2 = sp.sqrt(2)
for i in range(4):
    for s in (1,-1):
        v = np.zeros(4); v[i] = s*np.sqrt(2)
        candidates_num.append(v)
        vs = sp.zeros(4,1); vs[i] = s*sqrt2
        candidates_sym.append(vs)
for eps in itertools.product([1,-1], repeat=4):
    candidates_num.append(np.array(eps)/np.sqrt(2))
    candidates_sym.append(sp.Matrix(eps)/sqrt2)
candidates_num = np.array(candidates_num)

fixed_sym = {}
for i in off_cap_idx:
    dists = np.linalg.norm(candidates_num - verts[i], axis=1)
    j = np.argmin(dists)
    assert dists[j] < 1e-6
    fixed_sym[i] = candidates_sym[j]

# ---------- exact symbolic moving vertices ----------
u0s = sp.Matrix([1,1,0,0])/sqrt2
v1s = sp.Matrix([1,-1,0,0])/sqrt2
w1s = sp.Matrix([0,0,1,1])/sqrt2
theta, t = sp.symbols('theta t', real=True, positive=True)
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
    return _extract_in_order(sol, list(z), sp.simplify)

moving_sym = {}
moving_pattern_cache = {}
for i in on_cap_idx:
    pat = patterns[i]
    if pat in moving_pattern_cache:
        moving_sym[i] = moving_pattern_cache[pat]
        continue
    labs = [p for p in pat if p != 'u1']
    zv = solve_vertex_moving(labs)
    assert zv is not None
    numeric = np.array([float(x.subs({theta: theta0, t: t0})) for x in zv])
    assert np.allclose(numeric, verts[i], atol=1e-5), (i, pat, numeric, verts[i])
    moving_sym[i] = zv
    moving_pattern_cache[pat] = zv

print(f"Solved {len(moving_pattern_cache)} distinct moving-vertex patterns exactly "
      f"(of {len(on_cap_idx)} on-cap vertex instances).")

sym_vertex = dict(fixed_sym)
sym_vertex.update(moving_sym)

# ---------- exact determinant sum over the 60 fixed-only simplices (sanity: should be 5) ----------
print("Computing exact fixed-only volume contribution (expect exactly 5)...")
vol_fixed_exact = sp.Integer(0)
for s in fixed_simplices:
    M = sp.Matrix.hstack(*[sym_vertex[i] for i in s]).T
    d = M.det()
    # fix sign using the numeric determinant at (theta0,t0)
    Mnum = verts[list(s)]
    sign = 1 if np.linalg.det(Mnum) > 0 else -1
    vol_fixed_exact += sign*d/24
vol_fixed_exact = sp.nsimplify(sp.simplify(vol_fixed_exact))
print("  exact vol_fixed =", vol_fixed_exact)

import pickle
with open(_os.path.join(_DATA_DIR, 'region4_sym_data.pkl'), 'wb') as f:
    pickle.dump({
        'sym_vertex_keys': list(sym_vertex.keys()),
        'moving_simplices': moving_simplices,
        'fixed_simplices': fixed_simplices,
        'on_cap_idx': on_cap_idx,
        'off_cap_idx': off_cap_idx,
        'patterns': patterns,
        'verts': verts,
    }, f)
# save sympy data separately via srepr since sympy objects don't pickle as cleanly with lambdas
with open(_os.path.join(_DATA_DIR, 'region4_sym_vertex.pkl'), 'wb') as f:
    pickle.dump({i: sp.srepr(v) for i,v in sym_vertex.items()}, f)

print("done part 1")
