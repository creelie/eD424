import re, time
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
print(f"rep point theta={theta0:.4f} t={t0:.4f}: {len(verts)} verts, {len(on_cap_idx)} moving, {len(off_cap_idx)} fixed")

hull = ConvexHull(verts, qhull_options='QJ')
on_cap_set = set(on_cap_idx)
moving_simplices = [s for s in hull.simplices if any(vv in on_cap_set for vv in s)]
fixed_simplices = [s for s in hull.simplices if all(vv not in on_cap_set for vv in s)]
print(f"{len(hull.simplices)} boundary simplices: {len(fixed_simplices)} fixed-only, {len(moving_simplices)} moving")

import itertools
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
unmatched = []
for i in off_cap_idx:
    dists = np.linalg.norm(candidates_num - verts[i], axis=1)
    j = np.argmin(dists)
    if dists[j] < 1e-6:
        fixed_sym[i] = candidates_sym[j]
    else:
        unmatched.append(i)
print(f"fixed vertices matched exactly: {len(fixed_sym)}/{len(off_cap_idx)}  unmatched: {len(unmatched)}")
if unmatched:
    for i in unmatched:
        print("  unmatched vertex:", verts[i], patterns[i])

if not unmatched:
    vol_fixed_exact = sp.Integer(0)
    for s in fixed_simplices:
        M = sp.Matrix.hstack(*[fixed_sym[i] for i in s]).T
        Mnum = verts[list(s)]
        sign = 1 if np.linalg.det(Mnum) > 0 else -1
        vol_fixed_exact += sign*M.det()/24
    vol_fixed_exact = sp.nsimplify(sp.simplify(vol_fixed_exact))
    print(f"EXACT fixed-only volume contribution: {vol_fixed_exact}")

on_cap_patterns = sorted({tuple(sorted(v[1])) for v in [(vv,p) for vv,p in zip(verts,patterns)] if 'u1' in v[1]})
print(f"\non-cap patterns ({len(on_cap_patterns)}):")
for p in on_cap_patterns: print(" ", p)
