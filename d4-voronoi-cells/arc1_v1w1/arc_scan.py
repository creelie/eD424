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

# Reduced fundamental domain (by this session's new symmetry lemmas):
# only two combinatorially-independent arcs: v1-w1 (=v1-v2 pointwise) and
# w1-v2 (symmetric about its own midpoint, so only t in [0, pi/4] needed).
# Scan a fine (theta, t) grid over BOTH arcs looking for any sign violation.

def arc_point(t, A, B):
    a, b = np.cos(t), np.sin(t)
    return a * A + b * B

results = []
worst = (1e9, None, None, None)
thetas = np.linspace(0.02, np.pi/2 - 0.02, 60)

print("Scanning v1-w1 arc (t in [0, pi/2], full range needed -- not")
print("self-symmetric under any single generator found so far):")
ts_vw = np.linspace(0.0, np.pi/2, 60)
for theta in thetas:
    for t in ts_vw:
        e = arc_point(t, v1, w1)
        F = F_direct(theta, e)
        if F < worst[0]:
            worst = (F, theta, t, 'v1-w1')

print(f"  min so far: F={worst[0]:.6f} at theta={worst[1]:.4f}, t={worst[2]:.4f}")
print()
print("Scanning w1-v2 arc (t in [0, pi/4] suffices by midpoint symmetry):")
ts_wv = np.linspace(0.0, np.pi/4, 40)
for theta in thetas:
    for t in ts_wv:
        e = arc_point(t, w1, v2)
        F = F_direct(theta, e)
        if F < worst[0]:
            worst = (F, theta, t, 'w1-v2')

print(f"  min so far: F={worst[0]:.6f} at theta={worst[1]:.4f}, t={worst[2]:.4f}, arc={worst[3]}")
print()
print(f"GLOBAL MIN over both reduced arcs (coarse grid): F={worst[0]:.6f}")
print(f"  at theta={worst[1]:.6f}, t={worst[2]:.6f}, arc={worst[3]}")
