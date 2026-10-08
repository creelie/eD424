#!/usr/bin/env python3
"""
Richardson-style convergence test: is the small positive second-difference
seen near theta~pi/2 a genuine curvature (real, if small, non-concavity),
a genuine kink/breakpoint (derivative jump), or pure grid/qhull-precision
noise? Fix a point (theta, s0) on the v1->edge(w1,v2) geodesic and shrink
the step h geometrically, tracking d2(h) = f(s0-h)+f(s0+h)-2f(s0).
  - f'' != 0 (smooth, real curvature):      d2(h)/h^2 -> nonzero constant
  - f affine there (no real curvature):     d2(h) -> ~0, no clean h^2 scaling,
                                             looks like noise at hull-precision
  - genuine kink (breakpoint, derivative jump): d2(h) -> nonzero CONSTANT
                                             (does not shrink as h->0)
"""
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
    v = np.array(vec, dtype=float)
    v /= np.linalg.norm(v)
    d = roots @ v
    idx = np.argmax(d)
    assert d[idx] > 1 - 1e-9
    return idx

idx0 = find([1, 1, 0, 0])
u0 = roots[idx0]
v1 = roots[find([1, -1, 0, 0])]
w1 = roots[find([0, 0, 1, 1])]
v2 = roots[find([0, 0, 1, -1])]

def defect(theta, e_perp):
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

def geodesic_point(A, B, s, ang):
    return (np.sin(ang - s) * A + np.sin(s) * B) / np.sin(ang)

rng = np.random.default_rng(12345)
cs = [rng.uniform(0,1) for _ in range(3)]
c2 = cs[2]
P = np.cos(c2 * np.pi/4) * w1 + np.sin(c2 * np.pi/4) * v2
A, B = v1/np.linalg.norm(v1), P/np.linalg.norm(P)
ang = np.arccos(np.clip(A @ B, -1, 1))

theta = 1.5707
s0 = 0.222

print(f"theta={theta}, s0={s0}, ang={ang:.6f}")
print(f"{'h':>12} {'d2(h)':>14} {'d2(h)/h^2':>14}")
for h in [0.08, 0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125, 0.000625, 0.0003125]:
    fm = defect(theta, geodesic_point(A, B, s0-h, ang))
    f0 = defect(theta, geodesic_point(A, B, s0, ang))
    fp = defect(theta, geodesic_point(A, B, s0+h, ang))
    d2 = fm + fp - 2*f0
    print(f"{h:12.7f} {d2:14.6e} {d2/h**2:14.6e}")

print()
print("Also checking combinatorial (vertex-count) stability across this window,")
print("to see if a breakpoint (region change) sits right at/near s0:")
def n_vertices(theta, e_perp):
    e_perp = e_perp / np.linalg.norm(e_perp)
    u1 = np.cos(theta) * u0 + np.sin(theta) * e_perp
    dirs = roots.copy()
    dirs[idx0] = u1
    Am = dirs
    bm = -np.ones(len(dirs))
    hs = np.hstack([Am, bm.reshape(-1, 1)])
    hi = HalfspaceIntersection(hs, np.zeros(4))
    hull = ConvexHull(hi.intersections, qhull_options='QJ')
    return len(hull.vertices)

for s in np.linspace(s0-0.05, s0+0.05, 21):
    nv = n_vertices(theta, geodesic_point(A, B, s, ang))
    print(f"  s={s:.5f}  n_vertices={nv}")
