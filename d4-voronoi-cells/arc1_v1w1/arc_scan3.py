import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.optimize import minimize

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

# Target the previously-flagged "unresolved transition window" theta~1.09-1.26,
# and t strictly interior (away from root-aligned endpoints t=0, pi/2 or pi/4).
thetas = np.linspace(1.00, 1.35, 70)

def scan_arc(A, B, tlo, thi, name):
    ts = np.linspace(tlo, thi, 70)
    worst = (1e9, None, None)
    for theta in thetas:
        for t in ts:
            e = arc_point(t, A, B)
            F = F_direct(theta, e)
            if F < worst[0]:
                worst = (F, theta, t)
    print(f"{name}: min F={worst[0]:.6f} at theta={worst[1]:.4f}, t={worst[2]:.4f}")
    return worst

r1 = scan_arc(v1, w1, 0.15, np.pi/2 - 0.15, "v1-w1 arc (interior)")
r2 = scan_arc(w1, v2, 0.10, np.pi/4, "w1-v2 arc (interior, t<=pi/4 by symmetry)")

best = r1 if r1[0] < r2[0] else r2
A, B = (v1, w1) if r1[0] < r2[0] else (w1, v2)
name = "v1-w1" if r1[0] < r2[0] else "w1-v2"
print(f"\nOverall worst: {name} arc, F={best[0]:.6f} at theta={best[1]:.4f}, t={best[2]:.4f}")
print("Refining locally...")

def obj(x):
    theta, t = x
    e = arc_point(t, A, B)
    return F_direct(theta, e)

res = minimize(obj, x0=[best[1], best[2]], method='Nelder-Mead',
                options={'xatol':1e-8, 'fatol':1e-12, 'maxiter':800})
print(f"Refined minimum: F={res.fun:.8f} at theta={res.x[0]:.6f}, t={res.x[1]:.6f}")
