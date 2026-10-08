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

t = 0.79
e = np.cos(t)*v1 + np.sin(t)*w1
print(f"Profile of F(theta) along the v1-w1 arc at fixed t={t} (near the")
print(f"grid-observed minimum-location), theta from 0.02 to pi/2-0.01:")
prev = None
increasing = True
for theta in np.linspace(0.02, np.pi/2 - 0.01, 40):
    F = F_direct(theta, e)
    marker = ""
    if prev is not None and F < prev - 1e-9:
        increasing = False
        marker = "  <-- DECREASE"
    print(f"  theta={theta:.4f}  F={F:.6f}{marker}")
    prev = F
print()
print(f"Strictly non-decreasing throughout: {increasing}")
