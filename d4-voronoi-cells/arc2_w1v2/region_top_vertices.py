import numpy as np, re
import sympy as sp
from scipy.spatial import HalfspaceIntersection

def _build_root_system():
    roots, labels = [], []
    for i in range(4):
        for j in range(i + 1, 4):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(4)
                    v[i], v[j] = si, sj
                    roots.append(v / np.sqrt(2))
                    sgn = lambda s: '+' if s == 1 else '-'
                    labels.append(f"{sgn(si)}e{i+1}{sgn(sj)}e{j+1}")
    roots = np.array(roots)
    def find(vec):
        v = np.array(vec, dtype=float); v /= np.linalg.norm(v)
        idx = np.argmax(roots @ v)
        assert (roots @ v)[idx] > 1 - 1e-9
        return idx
    idx0 = find([1, 1, 0, 0])
    return roots, labels, idx0, roots[idx0], roots[find([1, -1, 0, 0])], roots[find([0, 0, 1, 1])]

roots, labels, idx0, u0, v1, w1 = _build_root_system()

def u1_vec(theta, t):
    return np.cos(theta)*u0 + np.sin(theta)*(np.cos(t)*v1 + np.sin(t)*w1)

def full_vertex_set(theta, t):
    u1 = u1_vec(theta, t)
    dirs = roots.copy(); dirs[idx0] = u1
    hs = np.hstack([dirs, (-np.ones(len(dirs))).reshape(-1,1)])
    hi = HalfspaceIntersection(hs, np.zeros(4))
    out = []
    for v in hi.intersections:
        vals = dirs @ v
        touching = frozenset(labels[k] if k!=idx0 else 'u1' for k,val in enumerate(vals) if abs(val-1)<1e-6)
        out.append((tuple(np.round(v,6)), touching))
    return out

def theta_A(t): return 2*np.arctan(np.cos(t))

# representative points in region top: theta between A(t) and pi/2
t0 = 0.15
theta0 = 0.5*(theta_A(t0) + np.pi/2)
verts = full_vertex_set(theta0, t0)
on_cap = [v for v in verts if 'u1' in v[1]]
off_cap = [v for v in verts if 'u1' not in v[1]]
print(f"theta0={theta0:.4f} t0={t0:.4f}  total={len(verts)} on_cap(moving)={len(on_cap)} off_cap(fixed)={len(off_cap)}")

# check fixed across multiple sample points across full t range
samples = [(0.05, 0.5*(theta_A(0.05)+np.pi/2)),
           (0.15, 0.5*(theta_A(0.15)+np.pi/2)),
           (0.35, 0.5*(theta_A(0.35)+np.pi/2)),
           (0.6, 0.5*(theta_A(0.6)+np.pi/2)),
           (0.75, 0.5*(theta_A(0.75)+np.pi/2))]
vsets = [{v[1]: v[0] for v in full_vertex_set(th,tt) if 'u1' not in v[1]} for tt,th in samples]
common = set(vsets[0])
for vs in vsets[1:]: common &= set(vs)
print(f"common off-cap patterns across samples: {len(common)} (of first sample's {len(vsets[0])})")
n_diff = sum(1 for k in common if not all(np.allclose(vs[k], vsets[0][k]) for vs in vsets))
print(f"off-cap coordinate mismatch count: {n_diff}/{len(common)} (0 = all confirmed fixed)")

on_cap_patterns = sorted({tuple(sorted(v[1])) for v in on_cap})
print(f"\non-cap (moving) vertex patterns: {len(on_cap_patterns)}")
for p in on_cap_patterns:
    print(" ", p)

print(f"\noff-cap (fixed) vertex patterns: {len(vsets[0])}")
