import itertools
import numpy as np
import sympy as sp
import re
from scipy.spatial import HalfspaceIntersection, ConvexHull

# ---------- numeric root system (for combinatorics / hull triangulation) ----------
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
print("total vertices:", len(verts))

on_cap_idx = [i for i,p in enumerate(patterns) if 'u1' in p]
off_cap_idx = [i for i,p in enumerate(patterns) if 'u1' not in p]
print("on-cap (moving):", len(on_cap_idx), " off-cap (fixed):", len(off_cap_idx))

hull = ConvexHull(verts, qhull_options='QJ')
print("scipy hull volume:", hull.volume, " #simplices:", len(hull.simplices))

on_cap_set = set(on_cap_idx)
moving_simplices = [s for s in hull.simplices if any(v in on_cap_set for v in s)]
fixed_simplices  = [s for s in hull.simplices if all(v not in on_cap_set for v in s)]
print("moving simplices:", len(moving_simplices), " fixed-only simplices:", len(fixed_simplices))

vol_fixed = sum(abs(np.linalg.det(verts[list(s)]))/24.0 for s in fixed_simplices)
vol_moving_num = sum(abs(np.linalg.det(verts[list(s)]))/24.0 for s in moving_simplices)
print("numeric vol_fixed:", vol_fixed, " numeric vol_moving:", vol_moving_num,
      " sum:", vol_fixed+vol_moving_num)

# ---------- exact symbolic vertex coordinates ----------
sqrt2 = sp.sqrt(2)
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

def _extract(sol, z, simplify_fn):
    """Read a solve() result back in the order z1,z2,z3,z4.

    sympy returns a dict keyed by symbol, and its iteration order is not
    guaranteed to match the order the unknowns were supplied in, so the
    components must be looked up by key rather than taken from .values().
    A key that is absent means the system did not determine that
    coordinate, which we report rather than silently accept.
    """
    if not sol:
        return None
    d = sol[0]
    out = []
    for zi in z:
        if zi not in d:
            return None
        out.append(simplify_fn(d[zi]))
    return sp.Matrix(out)

def solve_vertex_fixed(facet_labels):
    """Exact coordinates of a vertex cut out by fixed facets only.

    A vertex of this cell can lie on more than four facets, in which case
    an arbitrary four of them may be linearly dependent and determine
    nothing.  We therefore try every four-subset, in the order the labels
    come, and keep the first that pins the point down.
    """
    z = list(sp.symbols('z1 z2 z3 z4', real=True))
    for combo in itertools.combinations(facet_labels, 4):
        M = sp.Matrix([list(root_sym(lab).T) for lab in combo])
        if M.det() == 0:
            continue
        eqs = [sp.Eq((root_sym(lab).T*sp.Matrix(z))[0, 0], 1) for lab in combo]
        got = _extract(sp.solve(eqs, z, dict=True), z,
                       lambda v: sp.nsimplify(sp.simplify(v)))
        if got is not None:
            return got
    return None

def solve_vertex_moving(facet_labels):
    """Exact coordinates of a vertex on the moving cap and three fixed facets.

    Same degeneracy caveat as above, applied to the three fixed facets.
    """
    z = list(sp.symbols('z1 z2 z3 z4', real=True))
    for combo in itertools.combinations(facet_labels, 3):
        M = sp.Matrix([list(u1s.T)] + [list(root_sym(lab).T) for lab in combo])
        if sp.simplify(M.det()) == 0:
            continue
        eqs = [sp.Eq((u1s.T*sp.Matrix(z))[0, 0], 1)]
        for lab in combo:
            eqs.append(sp.Eq((root_sym(lab).T*sp.Matrix(z))[0, 0], 1))
        got = _extract(sp.solve(eqs, z, dict=True), z, sp.simplify)
        if got is not None:
            return got
    return None

# cache: pattern (frozenset) -> symbolic vertex vector
sym_cache = {}
for i, pat in enumerate(patterns):
    if pat in sym_cache:
        continue
    labs = [p for p in pat if p != 'u1']
    if 'u1' in pat:
        zv = solve_vertex_moving(labs)
    else:
        zv = solve_vertex_fixed(labs)
    assert zv is not None, (
        "no independent facet subset found for pattern %s" % (sorted(pat),))
    sym_cache[pat] = zv
    # sanity numeric check
    if 'u1' not in pat:
        numeric = np.array([float(x) for x in zv])
    else:
        numeric = np.array([float(x.subs({theta: theta0, t: t0})) for x in zv])
    assert np.allclose(numeric, verts[i], atol=1e-5), (i, pat, numeric, verts[i])

print("all", len(sym_cache), "distinct vertex patterns solved exactly and numerically verified")
