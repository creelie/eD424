import os as _os, sys as _sys
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA_DIR = _os.path.join(_PKG_ROOT, "data")
for _p in (_PKG_ROOT, _os.path.join(_PKG_ROOT, "core")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull

roots = []
for i in range(4):
    for j in range(i + 1, 4):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(4)
                v[i] = si
                v[j] = sj
                roots.append(v / np.sqrt(2))
roots = np.array(roots)

def find(vec):
    vec = np.array(vec, dtype=float)
    vec /= np.linalg.norm(vec)
    d = roots @ vec
    idx = np.argmax(d)
    assert d[idx] > 1 - 1e-9
    return idx

idx0 = find([1, 1, 0, 0])
u0 = roots[idx0]
v1 = roots[find([1, -1, 0, 0])]
w1 = roots[find([0, 0, 1, 1])]
v2 = roots[find([0, 0, 1, -1])]

def F_direct(theta, e_perp):
    e_perp = e_perp / np.linalg.norm(e_perp)
    u1 = np.cos(theta) * u0 + np.sin(theta) * e_perp
    dirs = roots.copy()
    dirs[idx0] = u1
    Am = dirs
    bm = -np.ones(len(dirs))
    hs = np.hstack([Am, bm.reshape(-1, 1)])
    hi = HalfspaceIntersection(hs, np.zeros(4))
    hull = ConvexHull(hi.intersections, qhull_options='QJ')
    return hull.volume - 8.0

def arc_point(t, A, B):
    a, b = np.cos(t), np.sin(t)
    return a * A + b * B

# Full, unbiased, fine grid over the ENTIRE reduced fundamental domain,
# theta in (0.15, pi/2) -- excluding only the already-proven near-0 region --
# both reduced arcs, full t range each. 150x150 = 22500 volume computations
# per arc.
thetas = np.linspace(0.15, np.pi/2 - 0.005, 150)

def full_scan(A, B, tmax, name):
    ts = np.linspace(0.0, tmax, 150)
    Fmat = np.zeros((len(thetas), len(ts)))
    worst = (1e9, None, None)
    for i, theta in enumerate(thetas):
        for j, t in enumerate(ts):
            e = arc_point(t, A, B)
            F = F_direct(theta, e)
            Fmat[i, j] = F
            if F < worst[0]:
                worst = (F, theta, t)
    print(f"{name}: global min on grid F={worst[0]:.6f} at theta={worst[1]:.4f}, t={worst[2]:.4f}")
    return worst, Fmat

r1, M1 = full_scan(v1, w1, np.pi/2, "v1-w1 arc")
r2, M2 = full_scan(w1, v2, np.pi/4, "w1-v2 arc")

print()
print(f"OVERALL MIN over the reduced fundamental domain (theta>0.15):")
best = r1 if r1[0] < r2[0] else r2
print(f"  F = {best[0]:.6f}  at theta={best[1]:.4f}, t={best[2]:.4f}")
np.save(_os.path.join(_DATA_DIR, 'Fmat_v1w1.npy'), M1)
np.save(_os.path.join(_DATA_DIR, 'Fmat_w1v2.npy'), M2)
np.save(_os.path.join(_DATA_DIR, 'thetas.npy'), thetas)
